//! An HTTP endpoint on 127.0.0.1, at a random path, that passes each request
//! (with its Range) on to one URL with headers added: ffmpeg reads the media
//! through it, and the media server's credentials never reach its command
//! line or a process list. `vergence worker`'s `LoopbackForwarder`, in tokio.

use std::time::Duration;

use futures_util::StreamExt;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinHandle;

pub struct Forwarder {
    port: u16,
    secret: String,
    task: JoinHandle<()>,
}

impl Forwarder {
    pub async fn start(target: String, headers: Vec<(String, String)>) -> std::io::Result<Forwarder> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let port = listener.local_addr()?.port();
        let secret = format!("{:032x}", rand::random::<u128>());
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(20))
            .read_timeout(Duration::from_secs(60))
            .build()
            .map_err(std::io::Error::other)?;
        let path = format!("/{secret}");
        let task = tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                let (client, target, headers, path) = (client.clone(), target.clone(), headers.clone(), path.clone());
                tokio::spawn(async move {
                    let _ = serve(stream, &client, &target, &headers, &path).await;
                });
            }
        });
        Ok(Forwarder { port, secret, task })
    }

    pub fn url(&self) -> String {
        format!("http://127.0.0.1:{}/{}", self.port, self.secret)
    }
}

impl Drop for Forwarder {
    fn drop(&mut self) {
        self.task.abort();
    }
}

/// The request line and headers, up to the blank line.
async fn read_head(stream: &mut TcpStream) -> std::io::Result<Option<String>> {
    let mut buffer = Vec::new();
    let mut chunk = [0u8; 4096];
    loop {
        if let Some(end) = buffer.windows(4).position(|w| w == b"\r\n\r\n") {
            return Ok(Some(String::from_utf8_lossy(&buffer[..end]).into_owned()));
        }
        if buffer.len() > 65536 {
            return Ok(None);
        }
        let n = stream.read(&mut chunk).await?;
        if n == 0 {
            return Ok(None);
        }
        buffer.extend_from_slice(&chunk[..n]);
    }
}

async fn serve(mut stream: TcpStream, client: &reqwest::Client, target: &str, headers: &[(String, String)],
               path: &str) -> std::io::Result<()> {
    let Some(head) = read_head(&mut stream).await? else { return Ok(()) };
    let mut lines = head.split("\r\n");
    let request: Vec<&str> = lines.next().unwrap_or("").split(' ').collect();
    let method = request.first().copied().unwrap_or("");
    if request.len() < 2 || !matches!(method, "GET" | "HEAD") || request[1] != path {
        stream.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").await?;
        return stream.shutdown().await;
    }
    let is_head = method == "HEAD";
    let mut upstream = if is_head { client.head(target) } else { client.get(target) };
    for (name, value) in headers {
        upstream = upstream.header(name, value);
    }
    for line in lines {
        if let Some((name, value)) = line.split_once(':')
            && name.trim().eq_ignore_ascii_case("range")
        {
            upstream = upstream.header("Range", value.trim());
        }
    }
    let response = match upstream.send().await {
        Ok(r) => r,
        Err(_) => {
            stream.write_all(b"HTTP/1.1 502 Relay\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").await?;
            return stream.shutdown().await;
        }
    };
    let mut out = format!("HTTP/1.1 {} Relay\r\nConnection: close\r\nAccept-Ranges: bytes\r\n", response.status().as_u16());
    for name in ["Content-Length", "Content-Range", "Content-Type"] {
        if let Some(value) = response.headers().get(name).and_then(|v| v.to_str().ok()) {
            out += &format!("{name}: {value}\r\n");
        }
    }
    out += "\r\n";
    stream.write_all(out.as_bytes()).await?;
    if !is_head {
        let mut body = response.bytes_stream();
        while let Some(chunk) = body.next().await {
            match chunk {
                Ok(bytes) => stream.write_all(&bytes).await?,
                Err(_) => break,
            }
        }
    }
    stream.shutdown().await
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A one-shot upstream that records the request it got and answers with a body.
    async fn upstream() -> (String, JoinHandle<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/media", listener.local_addr().unwrap());
        let task = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let head = read_head(&mut stream).await.unwrap().unwrap();
            stream
                .write_all(b"HTTP/1.1 206 Partial Content\r\nContent-Length: 5\r\nContent-Range: bytes 10-14/100\r\n\r\nhello")
                .await
                .unwrap();
            head
        });
        (url, task)
    }

    async fn raw_get(url: &str, extra: &str) -> String {
        let rest = url.strip_prefix("http://").unwrap();
        let (host, path) = rest.split_once('/').unwrap();
        let mut stream = TcpStream::connect(host).await.unwrap();
        stream.write_all(format!("GET /{path} HTTP/1.1\r\nHost: x\r\n{extra}\r\n").as_bytes()).await.unwrap();
        let mut out = String::new();
        stream.read_to_string(&mut out).await.unwrap();
        out
    }

    #[tokio::test]
    async fn forwards_range_and_adds_credentials() {
        let (target, seen) = upstream().await;
        let forwarder = Forwarder::start(target, vec![("Authorization".into(), "MediaBrowser Token=\"k\"".into())])
            .await
            .unwrap();
        let reply = raw_get(&forwarder.url(), "Range: bytes=10-14\r\n").await;
        assert!(reply.starts_with("HTTP/1.1 206 Relay"), "{reply}");
        assert!(reply.contains("Content-Range: bytes 10-14/100"));
        assert!(reply.ends_with("hello"));
        let head = seen.await.unwrap().to_ascii_lowercase();
        assert!(head.contains("range: bytes=10-14"));
        assert!(head.contains("authorization: mediabrowser token=\"k\""));
    }

    #[tokio::test]
    async fn refuses_other_paths() {
        let forwarder = Forwarder::start("http://127.0.0.1:9/".into(), vec![]).await.unwrap();
        let wrong = forwarder.url().rsplit_once('/').unwrap().0.to_string() + "/guess";
        assert!(raw_get(&wrong, "").await.starts_with("HTTP/1.1 404"));
    }
}
