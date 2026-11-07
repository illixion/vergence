//! The paired channel end to end: a device's TLS-PSK client (vergence-companion's OpenSSL side,
//! what the Quest's C client and the Vision Pro's Network.framework one also speak) against the
//! coordinator's listener, raw HTTP/1.1 inside, as the transport spec in `paired.rs` describes.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::Path;
use std::time::Duration;

use axum::body::Body;
use axum::http::Request;
use axum::routing::get;
use axum::{Json, Router};
use openssl::ssl::SslStream;
use serde_json::{Value, json};
use tower::ServiceExt;
use vergence_companion::tls::PskClient;
use vergence_companion::{Channel, Credential, CredentialStore, DeviceRecord, JsonFileStore};
use vergence_coordinator::api::{AppState, router};
use vergence_coordinator::auth::{Grant, Tokens};
use vergence_coordinator::paired::{self, PairedDevices};
use vergence_coordinator::queue::Workload;
use vergence_coordinator::source::{Jellyfin, ServerConfig, Sources};
use vergence_coordinator::store::Store;

const ITEM: &str = "289477b1bdb872265402628924353285";

/// A Jellyfin that knows one half side-by-side title, 6 s at 24 fps.
async fn mock_jellyfin() -> String {
    let app = Router::new().route(
        "/Items",
        get(|| async {
            Json(json!({"Items": [{
                "Id": ITEM, "RunTimeTicks": 60_000_000i64, "Video3DFormat": "HalfSideBySide",
                "MediaSources": [{"Id": ITEM, "ETag": "e1", "Path": "/m/t.mkv", "MediaStreams": [
                    {"Type": "Video", "RealFrameRate": 24.0, "Width": 1920, "Height": 1080, "Codec": "hevc"}]}]
            }]}))
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    format!("http://{address}")
}

fn record(id: &str, token: &str, grants: &[&str]) -> DeviceRecord {
    DeviceRecord {
        device_id: id.into(),
        name: format!("Device {id}"),
        token: token.into(),
        grants: grants.iter().map(|g| g.to_string()).collect(),
        ssh_public_key: None,
        paired_at: 0,
    }
}

struct Harness {
    port: u16,
    devices_path: std::path::PathBuf,
    plain: Router,
    admin_token: String,
    _dir: tempfile::TempDir,
}

async fn harness() -> Harness {
    let dir = tempfile::tempdir().unwrap();
    let config = ServerConfig { url: mock_jellyfin().await, api_key: Some("k".into()), api_key_file: None };
    let sources = Sources { jellyfin: Some(Jellyfin::new(&config, reqwest::Client::new()).unwrap()), stash: None };
    let mut tokens = Tokens::load(&dir.path().join("tokens.json")).unwrap();
    let admin_token = tokens.add("desk", vec![Grant::Admin]).unwrap();
    let state = AppState::new(Store::open(dir.path()).unwrap(), sources, Workload::defaults(), tokens).unwrap();
    let devices_path = dir.path().join("devices.json");
    let store = JsonFileStore::new(&devices_path);
    store.put(record("HS-1", "headset-token", &["play"])).unwrap();
    store.put(record("DESK-1", "desk-token", &["play", "work", "admin"])).unwrap();
    let devices = PairedDevices::open(&devices_path);
    state.set_paired(devices.clone());
    let app = router(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let paired_app = app.clone();
    tokio::spawn(async move { paired::serve(listener, paired_app, devices, std::future::pending()).await.unwrap() });
    Harness { port, devices_path, plain: app, admin_token, _dir: dir }
}

/// Connects as `device_id` holding `token`, on the API channel.
fn connect(port: u16, channel: &Channel, device_id: Option<&str>, token: &str) -> Result<SslStream<TcpStream>, String> {
    let tcp = TcpStream::connect(("127.0.0.1", port)).map_err(|e| e.to_string())?;
    tcp.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    let credential = Credential::new(token, device_id.map(str::to_string));
    let client = PskClient::new(channel, &credential).map_err(|e| e.to_string())?;
    client.connect(tcp).map_err(|e| e.to_string())
}

struct Response {
    status: u16,
    headers: HashMap<String, String>,
    body: Vec<u8>,
}

impl Response {
    fn json(&self) -> Value {
        serde_json::from_slice(&self.body).unwrap_or(Value::Null)
    }
}

/// One HTTP/1.1 exchange on an open connection, as a minimal client would make it.
fn exchange(stream: &mut SslStream<TcpStream>, method: &str, path: &str, extra: &[(&str, &str)], body: &[u8]) -> std::io::Result<Response> {
    let mut head = format!("{method} {path} HTTP/1.1\r\nHost: coordinator\r\nContent-Length: {}\r\n", body.len());
    if !body.is_empty() {
        head.push_str("Content-Type: application/json\r\n");
    }
    for (k, v) in extra {
        head.push_str(&format!("{k}: {v}\r\n"));
    }
    head.push_str("\r\n");
    stream.write_all(head.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()?;
    let mut buf = Vec::new();
    let mut chunk = [0u8; 16384];
    let end = loop {
        if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break i + 4;
        }
        let n = stream.read(&mut chunk)?;
        if n == 0 {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }
        buf.extend_from_slice(&chunk[..n]);
    };
    let text = String::from_utf8_lossy(&buf[..end]).to_string();
    let mut lines = text.split("\r\n");
    let status: u16 = lines.next().unwrap().split(' ').nth(1).unwrap().parse().unwrap();
    let headers: HashMap<String, String> = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_string()))
        .collect();
    let mut rest = buf[end..].to_vec();
    // Every response the coordinator sends has a length.
    let length: usize = headers["content-length"].parse().unwrap();
    while rest.len() < length {
        let n = stream.read(&mut chunk)?;
        if n == 0 {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }
        rest.extend_from_slice(&chunk[..n]);
    }
    assert_eq!(rest.len(), length, "nothing after the body (one request at a time)");
    Ok(Response { status, headers, body: rest })
}

async fn blocking<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    tokio::task::spawn_blocking(f).await.unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn a_paired_device_uses_the_api_over_tls_psk() {
    let h = harness().await;
    let port = h.port;
    let channel = paired::channel();

    let results = blocking(move || {
        let mut headset = connect(port, &channel, Some("HS-1"), "headset-token").unwrap();
        // Several requests in turn on one connection.
        let health = exchange(&mut headset, "GET", "/v1/health", &[], b"").unwrap();
        assert_eq!(health.status, 200);
        assert_eq!(health.json()["service"], "vergenced");
        let unknown = exchange(&mut headset, "GET", &format!("/v1/titles/jellyfin/{ITEM}/smooth-motion-3d"), &[], b"").unwrap();
        assert_eq!((unknown.status, unknown.json()["state"].clone()), (404, json!("none")));
        let body = json!({"server": "jellyfin", "id": ITEM, "seconds": 2.0, "player": "vp"}).to_string();
        let playing = exchange(&mut headset, "POST", "/v1/playing", &[], body.as_bytes()).unwrap();
        assert_eq!(playing.status, 200, "{}", String::from_utf8_lossy(&playing.body));
        assert_eq!(playing.json()["pieces"], 3);
        let chunk = exchange(&mut headset, "GET", &format!("/v1/titles/jellyfin/{ITEM}/smooth-motion-3d/chunks/0"), &[], b"").unwrap();
        assert_eq!(chunk.status, 404, "nothing analysed yet");
        // The play grant isn't admin, and a bearer token sent along doesn't change who it is.
        let queue = exchange(&mut headset, "GET", "/v1/queue", &[], b"").unwrap();
        assert_eq!(queue.status, 403);
        let claim = exchange(&mut headset, "POST", "/v1/work/claim", &[], br#"{"worker":"x","models":{}}"#).unwrap();
        assert_eq!(claim.status, 403);

        // An admin device sees the queue, with the headset's playhead under its device name.
        let mut desk = connect(port, &paired::channel(), Some("DESK-1"), "desk-token").unwrap();
        let queue = exchange(&mut desk, "GET", "/v1/queue", &[], b"").unwrap();
        assert_eq!(queue.status, 200);
        let text = String::from_utf8_lossy(&queue.body).to_string();
        assert!(text.contains("device-HS-1/vp"), "{text}");
        assert!(queue.headers.contains_key("content-type"));
        (headset, desk)
    })
    .await;
    let (mut headset, _desk) = results;

    // Forgetting the device locks it out: on its open connection at once, and in new handshakes.
    JsonFileStore::new(&h.devices_path).remove("HS-1").unwrap();
    let port = h.port;
    blocking(move || {
        let health = exchange(&mut headset, "GET", &format!("/v1/titles/jellyfin/{ITEM}/smooth-motion-3d"), &[], b"").unwrap();
        assert_eq!(health.status, 403);
        assert!(connect(port, &paired::channel(), Some("HS-1"), "headset-token").is_err());
    })
    .await;
}

#[tokio::test(flavor = "multi_thread")]
async fn strangers_are_refused_at_the_handshake() {
    let h = harness().await;
    let port = h.port;
    blocking(move || {
        let ch = paired::channel();
        // An unknown device, the wrong token for a known one, the shared identity, another channel.
        assert!(connect(port, &ch, Some("NOPE"), "headset-token").is_err());
        assert!(connect(port, &ch, Some("HS-1"), "wrong").is_err());
        assert!(connect(port, &ch, None, "headset-token").is_err());
        assert!(connect(port, &Channel::new("VergenceAPI/v1", "Vergence-Work-PSK-v1"), Some("HS-1"), "headset-token").is_err());
        // Plain HTTP on the TLS port gets nowhere either.
        let mut tcp = TcpStream::connect(("127.0.0.1", port)).unwrap();
        tcp.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        tcp.write_all(b"GET /v1/health HTTP/1.1\r\nHost: x\r\n\r\n").unwrap();
        let mut out = Vec::new();
        let _ = tcp.read_to_end(&mut out);
        assert!(!String::from_utf8_lossy(&out).contains("vergenced"));
        // And the right key still works afterwards.
        let mut ok = connect(port, &ch, Some("HS-1"), "headset-token").unwrap();
        assert_eq!(exchange(&mut ok, "GET", "/v1/health", &[], b"").unwrap().status, 200);
    })
    .await;
}

#[tokio::test(flavor = "multi_thread")]
async fn the_derived_token_works_over_plain_http() {
    let h = harness().await;
    let send = |token: String, uri: &str| {
        let request = Request::builder().uri(uri).header("authorization", format!("Bearer {token}")).body(Body::empty()).unwrap();
        h.plain.clone().oneshot(request)
    };
    let derived = paired::http_token("headset-token", "HS-1");
    let status = |r: axum::response::Response| r.status().as_u16();
    let title = format!("/v1/titles/jellyfin/{ITEM}/smooth-motion-3d");
    assert_eq!(status(send(derived.clone(), &title).await.unwrap()), 404);
    assert_eq!(status(send(derived.clone(), "/v1/queue").await.unwrap()), 403);
    // The pairing token itself is no bearer token.
    assert_eq!(status(send("headset-token".into(), &title).await.unwrap()), 403);
    assert_eq!(status(send(h.admin_token.clone(), "/v1/queue").await.unwrap()), 200);
    JsonFileStore::new(&h.devices_path).remove("HS-1").unwrap();
    assert_eq!(status(send(derived, &title).await.unwrap()), 403);
    assert!(Path::new(&h.devices_path).exists());
}
