//! The paired channel: the coordinator's HTTP API over TLS 1.2 with an external PSK, for devices
//! paired with RAVECompanion's six-digit comparison (`vergence-companion`). Plain HTTP with a
//! bearer token stays for tools on this computer, the LAN's own scripts and the Jellyfin plugin.
//!
//! The wire, exactly (Hypnos `docs/plans/analysis-workers.md`, "The paired channel"):
//!
//! - **Port:** [`DEFAULT_PAIRED_PORT`] (8421) on every interface, beside the HTTP port (8420).
//! - **TLS:** 1.2 only, `TLS_PSK_WITH_AES_128_GCM_SHA256` (0x00A8, OpenSSL `PSK-AES128-GCM-SHA256`),
//!   no certificate. No identity hint is sent.
//! - **PSK identity:** `VergenceAPI/v1/<device id>` (UTF-8), the device id the pairing carried.
//!   The bare `VergenceAPI/v1` (a shared token) is refused.
//! - **PSK:** `HKDF-SHA256(ikm: the device's pairing token (UTF-8), salt: "Vergence-API-PSK-v1",
//!   info: "psk")`, 32 bytes: RAVECompanion's channel key schedule
//!   (`RAVECompanionChannel(identityBase: "VergenceAPI/v1", salt: "Vergence-API-PSK-v1")`).
//! - **Then HTTP/1.1,** straight away: the same routes as the HTTP port, keep-alive, any number of
//!   requests in turn on one connection. No `Authorization` header is needed (one sent is ignored):
//!   the device is the PSK identity the handshake used, and its grants are the ones the person
//!   allowed when pairing, read again for every request, so forgetting a device locks it out at
//!   once. A device's player and worker names are `device-<device id>/<name>`, as over HTTP.
//! - **No identify message.** RAVECompanion's identify proof exists because a Network.framework
//!   server can't tell which PSK a connection used; OpenSSL can, so this channel needs none.
//!
//! The derived HTTP token ([`http_token`]) lets the same device use the plain port instead.

use std::future::Future;
use std::net::UdpSocket;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use axum::Router;
use axum::body::Body;
use hkdf::Hkdf;
use hyper::body::Incoming;
use hyper_util::rt::{TokioIo, TokioTimer};
use sha2::Sha256;
use tokio::net::TcpListener;
use tokio::sync::watch;
use tower::ServiceExt;
use vergence_companion::tls::PskServer;
use vergence_companion::{Channel, CredentialStore, DeviceRecord, JsonFileStore, Namespace};

use crate::auth::Grant;

/// The paired channel's port.
pub const DEFAULT_PAIRED_PORT: u16 = 8421;
/// The PSK identity base: a device presents `VergenceAPI/v1/<device id>`.
pub const IDENTITY_BASE: &str = "VergenceAPI/v1";
/// The channel's HKDF salt.
pub const PSK_SALT: &str = "Vergence-API-PSK-v1";
/// The salt of the derived HTTP token (the plain port's bearer token for a paired device).
pub const HTTP_TOKEN_SALT: &str = "Vergence-HTTP-Token-v1";
/// How long a handshake may take.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);
/// An idle keep-alive connection is closed after this long without a request.
const IDLE_TIMEOUT: Duration = Duration::from_secs(60);

/// Vergence's RAVECompanion namespace (`_vergence-*._tcp`, the `vergence://pair` link, direct
/// pairing port 4869), so Longwave's companion and Vergence never answer each other.
pub fn namespace() -> Namespace {
    Namespace::new("vergence", "Vergence", "vergence", 4869, "Vergence")
}

/// The API channel's key schedule.
pub fn channel() -> Channel {
    Channel::new(IDENTITY_BASE, PSK_SALT)
}

/// The bearer token a paired device uses on the plain HTTP port:
/// `hex(HKDF-SHA256(ikm: pairing token, salt: "Vergence-HTTP-Token-v1", info: device id))`, 32
/// bytes, lowercase. Derived, so the pairing token (which keys the PSK) never crosses the network.
pub fn http_token(pairing_token: &str, device_id: &str) -> String {
    let hk = Hkdf::<Sha256>::new(Some(HTTP_TOKEN_SALT.as_bytes()), pairing_token.as_bytes());
    let mut out = [0u8; 32];
    hk.expand(device_id.as_bytes(), &mut out).expect("32 bytes is a valid length");
    out.iter().map(|b| format!("{b:02x}")).collect()
}

/// The name a paired device goes by among tokens, players and workers.
pub fn device_name(device_id: &str) -> String {
    format!("device-{device_id}")
}

/// Equal without returning early.
fn same(a: &str, b: &str) -> bool {
    a.len() == b.len() && a.bytes().zip(b.bytes()).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// The paired devices, read from a `devices.json` (vergence-companion's `JsonFileStore`) and read
/// again whenever the file changes, so a pairing made by another process (`vergenced pair`, the
/// desktop app's pairing window) or a device forgotten there counts at once. Clones share.
#[derive(Clone)]
pub struct PairedDevices {
    backing: Backing,
    loaded: Arc<Mutex<Loaded>>,
}

/// Where the records are: a `devices.json` another process may rewrite (`vergenced pair`), read
/// again when it changes; or a store this process shares (the desktop app's Keychain on a Mac).
#[derive(Clone)]
enum Backing {
    File(PathBuf),
    Store(Arc<dyn vergence_companion::CredentialStore>),
}

#[derive(Default)]
struct Loaded {
    stamp: Option<(SystemTime, u64)>,
    records: Vec<DeviceRecord>,
}

impl std::fmt::Debug for PairedDevices {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.backing {
            Backing::File(path) => f.debug_struct("PairedDevices").field("path", path).finish_non_exhaustive(),
            Backing::Store(_) => f.debug_struct("PairedDevices").field("store", &"shared").finish_non_exhaustive(),
        }
    }
}

impl PairedDevices {
    pub fn open(path: impl Into<PathBuf>) -> PairedDevices {
        PairedDevices { backing: Backing::File(path.into()), loaded: Default::default() }
    }

    /// The devices of a store this process shares with its pairing (the desktop app's Keychain
    /// store on a Mac). It is asked on every lookup and keeps its own cache.
    pub fn from_store(store: Arc<dyn vergence_companion::CredentialStore>) -> PairedDevices {
        PairedDevices { backing: Backing::Store(store), loaded: Default::default() }
    }

    /// The file, for a file-backed list.
    pub fn path(&self) -> Option<&Path> {
        match &self.backing {
            Backing::File(path) => Some(path),
            Backing::Store(_) => None,
        }
    }

    /// The records, read again if the file changed. A missing or unreadable file (or store) is
    /// no devices.
    pub fn records(&self) -> Vec<DeviceRecord> {
        let path = match &self.backing {
            Backing::File(path) => path,
            Backing::Store(store) => {
                return store.devices().unwrap_or_else(|e| {
                    tracing::warn!(error = %e, "couldn't read the paired devices");
                    Vec::new()
                });
            }
        };
        let stamp = std::fs::metadata(path).ok().and_then(|m| Some((m.modified().ok()?, m.len())));
        let mut loaded = self.loaded.lock().unwrap_or_else(|e| e.into_inner());
        if loaded.stamp != stamp || stamp.is_none() {
            loaded.records = match JsonFileStore::new(path).devices() {
                Ok(records) => records,
                Err(e) => {
                    tracing::warn!(path = %path.display(), error = %e, "couldn't read the paired devices");
                    Vec::new()
                }
            };
            loaded.stamp = stamp;
        }
        loaded.records.clone()
    }

    fn record(&self, device_id: &str) -> Option<DeviceRecord> {
        self.records().into_iter().find(|r| r.device_id == device_id)
    }

    /// A device's pairing token, for the TLS listener.
    pub fn token_for(&self, device_id: &str) -> Option<String> {
        self.record(device_id).map(|r| r.token)
    }

    /// The grants a device was allowed (unknown names left out), or `None` when it isn't paired.
    pub fn grants(&self, device_id: &str) -> Option<Vec<Grant>> {
        self.record(device_id).map(|r| r.grants.iter().filter_map(|g| Grant::parse(g)).collect())
    }

    /// The device whose derived HTTP token this is, with its grants.
    pub fn by_http_token(&self, token: &str) -> Option<(String, Vec<Grant>)> {
        self.records().into_iter().find(|r| same(&http_token(&r.token, &r.device_id), token)).map(|r| {
            let grants = r.grants.iter().filter_map(|g| Grant::parse(g)).collect();
            (r.device_id, grants)
        })
    }

    /// The TLS side: every paired device's key, picked by the identity a connection presents.
    pub fn psk_server(&self) -> Result<PskServer, openssl::error::ErrorStack> {
        let devices = self.clone();
        PskServer::new(channel(), move |device| device.and_then(|id| devices.token_for(id)))
    }
}

/// The device a request on the paired channel came from, put in the request's extensions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PairedPeer {
    pub device_id: String,
}

/// Serves `router` on the paired channel until `shutdown` resolves; open connections are then
/// closed once their current request is answered.
pub async fn serve(
    listener: TcpListener,
    router: Router,
    devices: PairedDevices,
    shutdown: impl Future<Output = ()> + Send,
) -> std::io::Result<()> {
    let server = devices.psk_server().map_err(std::io::Error::other)?;
    let (stop_tx, stop_rx) = watch::channel(false);
    let mut shutdown = std::pin::pin!(shutdown);
    loop {
        let accepted = tokio::select! {
            _ = &mut shutdown => break,
            accepted = listener.accept() => accepted,
        };
        let (tcp, address) = match accepted {
            Ok(pair) => pair,
            Err(e) => {
                // Out of descriptors, or a connection reset before it was taken: keep listening.
                tracing::warn!(error = %e, "paired accept failed");
                tokio::time::sleep(Duration::from_millis(100)).await;
                continue;
            }
        };
        let server = server.clone();
        let router = router.clone();
        let stop = stop_rx.clone();
        tokio::spawn(async move {
            if let Err(reason) = connection(tcp, server, router, stop).await {
                tracing::debug!(%address, %reason, "paired connection ended");
            }
        });
    }
    let _ = stop_tx.send(true);
    Ok(())
}

async fn connection(
    tcp: tokio::net::TcpStream,
    server: PskServer,
    router: Router,
    mut stop: watch::Receiver<bool>,
) -> Result<(), String> {
    tcp.set_nodelay(true).ok();
    let ssl = server.ssl().map_err(|e| e.to_string())?;
    let mut stream = tokio_openssl::SslStream::new(ssl, tcp).map_err(|e| e.to_string())?;
    match tokio::time::timeout(HANDSHAKE_TIMEOUT, Pin::new(&mut stream).accept()).await {
        Ok(Ok(())) => {}
        Ok(Err(e)) => return Err(format!("handshake: {e}")),
        Err(_) => return Err("handshake timed out".into()),
    }
    // Only a device's own identity: the bare base (a shared token) names no device.
    let Some(Some(device_id)) = PskServer::matched(stream.ssl()) else {
        return Err("no device identity".into());
    };
    tracing::debug!(device = %device_id, "paired connection");
    let peer = PairedPeer { device_id };
    let service = hyper::service::service_fn(move |mut request: hyper::Request<Incoming>| {
        request.extensions_mut().insert(peer.clone());
        // A header can't stand in for the channel's identity, so none is read here.
        request.headers_mut().remove(hyper::header::AUTHORIZATION);
        router.clone().oneshot(request.map(Body::new))
    });
    let mut builder = hyper::server::conn::http1::Builder::new();
    builder.timer(TokioTimer::new()).header_read_timeout(IDLE_TIMEOUT).keep_alive(true);
    let conn = builder.serve_connection(TokioIo::new(stream), service);
    let mut conn = std::pin::pin!(conn);
    let stopped = async move {
        let _ = stop.wait_for(|stopped| *stopped).await;
    };
    tokio::select! {
        result = conn.as_mut() => return result.map_err(|e| e.to_string()),
        () = stopped => {}
    }
    conn.as_mut().graceful_shutdown();
    conn.await.map_err(|e| e.to_string())
}

/// The LAN address the default route leaves from (no packet is sent), for a pairing link.
pub fn lan_addresses() -> Vec<String> {
    let socket = UdpSocket::bind("0.0.0.0:0").and_then(|s| s.connect("192.0.2.1:9").map(|()| s));
    match socket.and_then(|s| s.local_addr()) {
        Ok(addr) if !addr.ip().is_unspecified() && !addr.ip().is_loopback() => vec![addr.ip().to_string()],
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_http_token_is_derived_per_device() {
        let token = http_token("pairing-token", "HS-1");
        assert_eq!(token.len(), 64);
        assert_eq!(token, http_token("pairing-token", "HS-1"));
        assert_ne!(token, http_token("pairing-token", "HS-2"));
        assert_ne!(token, http_token("other", "HS-1"));
    }

    #[test]
    fn devices_are_read_again_when_the_file_changes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("devices.json");
        let devices = PairedDevices::open(&path);
        assert!(devices.records().is_empty());
        let store = JsonFileStore::new(&path);
        let record = |id: &str, grants: &[&str]| DeviceRecord {
            device_id: id.into(),
            name: "Vision Pro".into(),
            token: format!("token-{id}"),
            grants: grants.iter().map(|g| g.to_string()).collect(),
            ssh_public_key: None,
            paired_at: 0,
        };
        store.put(record("HS-1", &["play", "bogus"])).unwrap();
        assert_eq!(devices.grants("HS-1"), Some(vec![Grant::Play]));
        assert_eq!(devices.token_for("HS-1").as_deref(), Some("token-HS-1"));
        assert_eq!(devices.by_http_token(&http_token("token-HS-1", "HS-1")), Some(("HS-1".into(), vec![Grant::Play])));
        assert_eq!(devices.by_http_token("token-HS-1"), None);
        store.put(record("HS-2", &["play", "work"])).unwrap();
        store.remove("HS-1").unwrap();
        assert_eq!(devices.grants("HS-1"), None);
        assert_eq!(devices.grants("HS-2"), Some(vec![Grant::Play, Grant::Work]));
    }
}
