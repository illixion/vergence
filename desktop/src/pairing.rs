//! Pairing a headset: vergence-companion's host side (RAVECompanion's six-digit comparison) on
//! the pairing port, open only while the person has asked to pair. The window shows the code
//! and the device's name, with Pair and Decline.
//!
//! A paired device keeps its record (its token and grants) in `devices.json`, and the
//! coordinator gets a `play` token for it. Until the chunk API runs over the paired TLS-PSK
//! channel, that token is the interim the plan allows: a bearer token over HTTP on the LAN. It
//! is *derived* from the device's pairing token, never the pairing token itself (which keys
//! the PSK channel and mustn't cross the network):
//!
//! ```text
//! http token = hex(HKDF-SHA256(ikm: pairing token (UTF-8), salt: "Vergence-HTTP-Token-v1",
//!                              info: device id (UTF-8), 32 bytes))
//! ```
//! so a device computes it itself from its grant, and nothing else has to reach it.

use std::net::{TcpListener, UdpSocket};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use hkdf::Hkdf;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use vergence_companion::net::{Decision, serve_pairing};
use vergence_companion::{CredentialStore, DeviceRecord, HostConfig, HostOutcome, HostPairing, JsonFileStore, Namespace, PairLink};
use vergence_coordinator::auth::Grant;

use crate::coordinator::CoordinatorHost;
use crate::settings::{Paths, write_atomic};

/// How long the pairing port stays open without a pairing.
const OPEN_FOR: Duration = Duration::from_secs(300);

/// Vergence's own namespace, so a Longwave companion and Vergence never answer each other.
pub fn namespace() -> Namespace {
    Namespace::new("vergence", "Vergence", "vergence", 4869, "Vergence")
}

/// The bearer token a paired device uses with the coordinator over HTTP (see the module docs).
pub fn http_token(record: &DeviceRecord) -> String {
    let hk = Hkdf::<Sha256>::new(Some(b"Vergence-HTTP-Token-v1"), record.token.as_bytes());
    let mut out = [0u8; 32];
    hk.expand(record.device_id.as_bytes(), &mut out).expect("32 bytes is a valid length");
    out.iter().map(|b| format!("{b:02x}")).collect()
}

fn sha256_hex(text: &str) -> String {
    Sha256::digest(text.as_bytes()).iter().map(|b| format!("{b:02x}")).collect()
}

/// The coordinator token's name for a device.
fn token_name(device_id: &str) -> String {
    format!("device-{device_id}")
}

/// Which coordinator token each device was given (`device-tokens.json`).
#[derive(Default, Serialize, Deserialize)]
struct DeviceTokens(std::collections::BTreeMap<String, String>);

impl DeviceTokens {
    fn load(paths: &Paths) -> DeviceTokens {
        std::fs::read(paths.device_tokens()).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default()
    }

    fn save(&self, paths: &Paths) -> std::io::Result<()> {
        write_atomic(&paths.device_tokens(), &serde_json::to_vec_pretty(&self.0)?, true)
    }
}

struct Waiting {
    id: u64,
    code: String,
    device_name: String,
    device_id: String,
    requested: Vec<String>,
    decide: mpsc::Sender<bool>,
}

struct Open {
    port: u16,
    link: String,
    addresses: Vec<String>,
    stop: Arc<AtomicBool>,
}

#[derive(Default)]
struct Inner {
    open: Option<Open>,
    waiting: Option<Waiting>,
    last: Option<String>,
}

pub struct Pairing {
    paths: Paths,
    coordinator: Arc<CoordinatorHost>,
    inner: Mutex<Inner>,
    next_id: AtomicU64,
    /// Called when something the window shows changes (a code to show, an outcome).
    on_change: Mutex<Option<Arc<dyn Fn() + Send + Sync>>>,
}

impl Pairing {
    pub fn new(paths: Paths, coordinator: Arc<CoordinatorHost>) -> Pairing {
        Pairing {
            paths,
            coordinator,
            inner: Default::default(),
            next_id: AtomicU64::new(1),
            on_change: Default::default(),
        }
    }

    pub fn on_change(&self, f: Arc<dyn Fn() + Send + Sync>) {
        *self.on_change.lock().unwrap() = Some(f);
    }

    fn changed(&self) {
        let f = self.on_change.lock().unwrap().clone();
        if let Some(f) = f {
            f();
        }
    }

    fn store(&self) -> JsonFileStore {
        JsonFileStore::new(self.paths.devices())
    }

    /// Opens the pairing port until a device pairs, the person stops it, or five minutes pass.
    pub fn start(self: &Arc<Self>, port: u16) -> Result<Value, String> {
        if self.inner.lock().unwrap().open.is_some() {
            return Ok(self.status());
        }
        let listener = TcpListener::bind(("0.0.0.0", port))
            .map_err(|e| format!("Couldn't open the pairing port {port}: {e}. Is another Vergence pairing?"))?;
        listener.set_nonblocking(true).map_err(|e| e.to_string())?;
        let port = listener.local_addr().map_err(|e| e.to_string())?.port();
        let store = self.store();
        let host_id = store.host_id().map_err(|e| e.to_string())?;
        let name = computer_name();
        let addresses = lan_addresses();
        let ns = namespace();
        let link = PairLink::new(&ns, host_id.clone(), name.clone(), Some(port), addresses.clone()).url();
        let mut config = HostConfig::new(ns, host_id, name);
        config.addresses = addresses.clone();
        config.default_grants = vec!["play".into()];
        let stop = Arc::new(AtomicBool::new(false));
        {
            let mut inner = self.inner.lock().unwrap();
            inner.open = Some(Open { port, link, addresses, stop: stop.clone() });
            inner.last = None;
        }
        let this = self.clone();
        std::thread::Builder::new()
            .name("pairing".into())
            .spawn(move || this.listen(listener, config, stop))
            .map_err(|e| e.to_string())?;
        self.changed();
        Ok(self.status())
    }

    pub fn stop(&self) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(open) = inner.open.take() {
            open.stop.store(true, Ordering::SeqCst);
        }
        // An unanswered prompt is abandoned with the port.
        inner.waiting = None;
    }

    fn listen(self: Arc<Self>, listener: TcpListener, config: HostConfig, stop: Arc<AtomicBool>) {
        let until = Instant::now() + OPEN_FOR;
        while !stop.load(Ordering::SeqCst) && Instant::now() < until {
            let stream = match listener.accept() {
                Ok((stream, _)) => stream,
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(200));
                    continue;
                }
                Err(e) => {
                    tracing::warn!(error = %e, "pairing accept failed");
                    std::thread::sleep(Duration::from_millis(500));
                    continue;
                }
            };
            let _ = stream.set_nonblocking(false);
            let outcome = serve_pairing(
                stream,
                HostPairing::new(config.clone()),
                |request, peer| self.ask(request, peer, &stop),
                |record| self.enroll(record),
            );
            let (message, paired) = match outcome {
                Ok(HostOutcome::Paired(record)) => (format!("Paired with {}.", record.name), true),
                Ok(HostOutcome::Declined) => ("Declined.".to_string(), false),
                Ok(HostOutcome::Abandoned) => ("The pairing didn't finish.".to_string(), false),
                Err(e) => (format!("The pairing failed: {e}"), false),
            };
            {
                let mut inner = self.inner.lock().unwrap();
                inner.waiting = None;
                inner.last = Some(message);
            }
            self.changed();
            if paired {
                break;
            }
        }
        let mut inner = self.inner.lock().unwrap();
        if inner.open.as_ref().is_some_and(|open| Arc::ptr_eq(&open.stop, &stop)) {
            inner.open = None;
        }
        drop(inner);
        self.changed();
    }

    /// Shows the code and waits for Pair or Decline (or the device giving up).
    fn ask(&self, request: &vergence_companion::ApprovalRequest, peer: &vergence_companion::net::PeerWatch, stop: &AtomicBool) -> Decision {
        let (tx, rx) = mpsc::channel();
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        self.inner.lock().unwrap().waiting = Some(Waiting {
            id,
            code: request.code.clone(),
            device_name: request.device_name.clone(),
            device_id: request.device_id.clone(),
            requested: request.requested.clone(),
            decide: tx,
        });
        self.changed();
        let decision = loop {
            match rx.recv_timeout(Duration::from_millis(250)) {
                // Only `play` is offered: what a headset needs, and what its coordinator token holds.
                Ok(true) => break Decision::Pair(vec!["play".into()]),
                Ok(false) => break Decision::Decline,
                Err(mpsc::RecvTimeoutError::Timeout) if !peer.is_lost() && !stop.load(Ordering::SeqCst) => continue,
                Err(_) => break Decision::Abandon,
            }
        };
        let mut inner = self.inner.lock().unwrap();
        if inner.waiting.as_ref().is_some_and(|w| w.id == id) {
            inner.waiting = None;
        }
        decision
    }

    /// Stores the device, then gives it a coordinator token, before the accept goes out.
    fn enroll(&self, record: &DeviceRecord) -> std::io::Result<()> {
        let store = self.store();
        store.put(record.clone())?;
        let name = token_name(&record.device_id);
        let result = self
            .coordinator
            .put_token_hash(&name, &sha256_hex(&http_token(record)), vec![Grant::Play])
            .map_err(std::io::Error::other)
            .and_then(|()| {
                let mut tokens = DeviceTokens::load(&self.paths);
                tokens.0.insert(record.device_id.clone(), name);
                tokens.save(&self.paths)
            });
        if result.is_err() {
            let _ = store.remove(&record.device_id);
        }
        result
    }

    /// The person's answer to the code on screen.
    pub fn decide(&self, id: u64, pair: bool) -> Result<(), String> {
        let inner = self.inner.lock().unwrap();
        match &inner.waiting {
            Some(waiting) if waiting.id == id => {
                let _ = waiting.decide.send(pair);
                Ok(())
            }
            _ => Err("That pairing request is gone.".into()),
        }
    }

    pub fn devices(&self) -> Vec<Value> {
        let tokens = DeviceTokens::load(&self.paths);
        self.store()
            .devices()
            .unwrap_or_default()
            .into_iter()
            .map(|d| {
                json!({
                    "id": d.device_id, "name": d.name, "grants": d.grants, "pairedAt": d.paired_at,
                    "token": tokens.0.get(&d.device_id),
                })
            })
            .collect()
    }

    /// Unpairs a device: its record and its coordinator token go.
    pub fn forget(&self, device_id: &str) -> Result<(), String> {
        let mut tokens = DeviceTokens::load(&self.paths);
        if let Some(name) = tokens.0.remove(device_id) {
            self.coordinator.remove_token(&name)?;
            tokens.save(&self.paths).map_err(|e| e.to_string())?;
        }
        self.store().remove(device_id).map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn status(&self) -> Value {
        let inner = self.inner.lock().unwrap();
        json!({
            "open": inner.open.as_ref().map(|o| json!({ "port": o.port, "link": o.link, "addresses": o.addresses })),
            "waiting": inner.waiting.as_ref().map(|w| json!({
                "id": w.id, "code": format!("{} {}", &w.code[..3.min(w.code.len())], &w.code[3.min(w.code.len())..]),
                "deviceName": w.device_name, "deviceId": w.device_id, "requested": w.requested,
            })),
            "last": inner.last,
            "devices": self.devices(),
        })
    }
}

/// This computer's name, as devices are shown it.
pub fn computer_name() -> String {
    #[cfg(target_os = "macos")]
    if let Ok(out) = std::process::Command::new("/usr/sbin/scutil").args(["--get", "ComputerName"]).output() {
        let name = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if out.status.success() && !name.is_empty() {
            return name;
        }
    }
    #[cfg(windows)]
    if let Ok(name) = std::env::var("COMPUTERNAME") {
        return name;
    }
    #[cfg(unix)]
    {
        let mut buf = [0u8; 256];
        if unsafe { libc::gethostname(buf.as_mut_ptr().cast(), buf.len()) } == 0 {
            let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
            let name = String::from_utf8_lossy(&buf[..end]).trim_end_matches(".local").to_string();
            if !name.is_empty() {
                return name;
            }
        }
    }
    "Vergence".into()
}

/// The LAN address the default route leaves from (no packet is sent).
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
    use vergence_companion::net::pair_with_host;
    use vergence_companion::{DeviceConfig, DevicePairing};

    fn free_port() -> u16 {
        TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
    }

    #[test]
    fn the_http_token_is_derived_not_the_pairing_token() {
        let record = DeviceRecord {
            device_id: "HS-1".into(),
            name: "Vision Pro".into(),
            token: "pairing-token".into(),
            grants: vec!["play".into()],
            ssh_public_key: None,
            paired_at: 0,
        };
        let token = http_token(&record);
        assert_eq!(token.len(), 64);
        assert_ne!(token, record.token);
        assert_eq!(token, http_token(&record));
        let other = DeviceRecord { device_id: "HS-2".into(), ..record.clone() };
        assert_ne!(token, http_token(&other));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_headset_pairs_and_gets_a_play_token() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::new(dir.path());
        let coordinator = Arc::new(CoordinatorHost::new(paths.clone()));
        let settings = crate::settings::Settings { port: free_port(), ..Default::default() };
        coordinator.start(&settings).await.unwrap();
        let pairing = Arc::new(Pairing::new(paths.clone(), coordinator.clone()));
        let port = free_port();
        let status = pairing.start(port).unwrap();
        assert_eq!(status["open"]["port"], port);
        assert!(status["open"]["link"].as_str().unwrap().starts_with("vergence://pair?"));

        // The person presses Pair once the code shows.
        let approver = {
            let pairing = pairing.clone();
            std::thread::spawn(move || {
                for _ in 0..200 {
                    if let Some(id) = pairing.status()["waiting"]["id"].as_u64() {
                        pairing.decide(id, true).unwrap();
                        return;
                    }
                    std::thread::sleep(Duration::from_millis(25));
                }
                panic!("no request to approve");
            })
        };
        let grant = tokio::task::spawn_blocking(move || {
            let stream = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
            let device = DevicePairing::new(DeviceConfig {
                namespace: namespace(),
                device_id: "HS-1".into(),
                device_name: "Vision Pro".into(),
                host_id: None,
                requested: vec!["play".into()],
                ssh_public_key: None,
            });
            pair_with_host(stream, device, Duration::from_secs(10), |_| {}).unwrap()
        })
        .await
        .unwrap();
        approver.join().unwrap();
        assert_eq!(grant.grants.as_deref(), Some(&["play".to_string()][..]));

        // The device's derived token works with the coordinator, for play only.
        let record = DeviceRecord {
            device_id: "HS-1".into(),
            name: "Vision Pro".into(),
            token: grant.token.clone(),
            grants: vec!["play".into()],
            ssh_public_key: None,
            paired_at: 0,
        };
        let state = coordinator.state().unwrap();
        assert_eq!(state.tokens.read().unwrap().check(&http_token(&record), Grant::Play), Some("device-HS-1"));
        assert_eq!(state.tokens.read().unwrap().check(&http_token(&record), Grant::Work), None);
        assert_eq!(state.tokens.read().unwrap().check(&grant.token, Grant::Play), None);
        let devices = pairing.devices();
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0]["token"], "device-HS-1");
        // A successful pairing closes the port.
        for _ in 0..50 {
            if pairing.status()["open"].is_null() {
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(pairing.status()["open"].is_null());
        assert_eq!(pairing.status()["last"], "Paired with Vision Pro.");

        pairing.forget("HS-1").unwrap();
        assert!(pairing.devices().is_empty());
        assert_eq!(state.tokens.read().unwrap().check(&http_token(&record), Grant::Play), None);
        coordinator.shutdown().await;
    }
}
