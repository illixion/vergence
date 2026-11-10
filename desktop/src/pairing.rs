//! Pairing a headset: vergence-companion's host side (RAVECompanion's six-digit comparison) on
//! the pairing port, open only while the person has asked to pair. The window shows the code
//! and the device's name, with Pair and Decline.
//!
//! A paired device's record (its token and grants) is kept in the login Keychain on a Mac and in
//! `devices.json` (0600) on Windows and Linux (`crate::coordinator::device_store`). From then on
//! it reaches the coordinator over the paired TLS-PSK channel, keyed from that token
//! (`vergence_coordinator::paired`, port 8421), which reads the same store. It may also
//! use the plain HTTP port with a bearer token *derived* from its pairing token, never the
//! pairing token itself (which keys the PSK channel and mustn't cross the network):
//!
//! ```text
//! http token = hex(HKDF-SHA256(ikm: pairing token (UTF-8), salt: "Vergence-HTTP-Token-v1",
//!                              info: device id (UTF-8), 32 bytes))
//! ```
//! so a device computes it itself from its grant, and nothing else has to reach it. The
//! coordinator also keeps that token's hash as `device-<id>`, for copies of the coordinator that
//! predate reading `devices.json`.

use std::net::TcpListener;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use vergence_companion::host::ExpectedDevice;
use vergence_companion::net::{Decision, serve_pairing};
use vergence_companion::{DeviceRecord, HostConfig, HostOutcome, HostPairing, Namespace, PairLink};
use vergence_coordinator::auth::Grant;

use crate::coordinator::CoordinatorHost;
use crate::presence::{PairRequest, Presence};
use crate::settings::{Paths, write_atomic};

/// How long the pairing port stays open without a pairing.
const OPEN_FOR: Duration = Duration::from_secs(300);
/// How long the listener opened for one headset's request waits for it (RAVEPairingServer's two minutes).
const REQUEST_OPEN_FOR: Duration = Duration::from_secs(120);

/// What a listener is for.
#[derive(Clone)]
enum Kind {
    /// The fixed port, for a headset that has this computer's address or link: any device.
    Direct,
    /// An ephemeral port, found by the one headset whose request opened it.
    Request(PairRequest),
}

/// Vergence's own namespace, so a Longwave companion and Vergence never answer each other.
pub fn namespace() -> Namespace {
    vergence_coordinator::paired::namespace()
}

/// The bearer token a paired device uses with the coordinator over HTTP (see the module docs).
pub fn http_token(record: &DeviceRecord) -> String {
    vergence_coordinator::paired::http_token(&record.token, &record.device_id)
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

/// The listener a headset's request opened.
struct Incoming {
    request: PairRequest,
    port: u16,
    stop: Arc<AtomicBool>,
    /// The headset has connected: its exchange is running and is left to finish.
    connected: Arc<AtomicBool>,
}

#[derive(Default)]
struct Inner {
    open: Option<Open>,
    incoming: Option<Incoming>,
    /// Requests that came while another headset was pairing, in order.
    queued: Vec<PairRequest>,
    waiting: Option<Waiting>,
    last: Option<String>,
}

pub struct Pairing {
    paths: Paths,
    coordinator: Arc<CoordinatorHost>,
    inner: Mutex<Inner>,
    next_id: AtomicU64,
    /// Announcing this computer and watching for headsets asking to pair, while discoverable.
    presence: Mutex<Option<Presence>>,
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
            presence: Default::default(),
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

    /// The coordinator's device store: the login Keychain on a Mac, `devices.json` elsewhere.
    fn store(&self) -> crate::coordinator::DeviceStore {
        self.coordinator.devices()
    }

    /// Opens the pairing port until a device pairs, the person stops it, or five minutes pass.
    pub fn start(self: &Arc<Self>, port: u16) -> Result<Value, String> {
        if self.inner.lock().unwrap().open.is_some() {
            return Ok(self.status());
        }
        let listener = TcpListener::bind(("0.0.0.0", port))
            .map_err(|e| format!("Couldn't open the pairing port {port}: {e}. Is another Vergence pairing?"))?;
        let port = listener.local_addr().map_err(|e| e.to_string())?.port();
        let (config, addresses) = self.host_config(None)?;
        let link = PairLink::new(&namespace(), config.host_id.clone(), config.host_name.clone(), Some(port), addresses.clone()).url();
        let stop = Arc::new(AtomicBool::new(false));
        {
            let mut inner = self.inner.lock().unwrap();
            inner.open = Some(Open { port, link, addresses, stop: stop.clone() });
            inner.last = None;
        }
        self.spawn_listener(listener, config, stop, Kind::Direct, Arc::new(AtomicBool::new(false)))?;
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

    /// What a listener tells the device: this computer, its addresses, and for a request the one
    /// device it is for. Returns the addresses too.
    fn host_config(&self, expected: Option<&PairRequest>) -> Result<(HostConfig, Vec<String>), String> {
        let host_id = self.store().host_id().map_err(|e| e.to_string())?;
        let name = computer_name();
        let addresses = lan_addresses();
        let mut config = HostConfig::new(namespace(), host_id, name);
        config.addresses = addresses.clone();
        config.default_grants = vec!["play".into()];
        config.expected_device = expected.map(|r| ExpectedDevice { id: r.device_id.clone(), name: r.name.clone() });
        Ok((config, addresses))
    }

    fn spawn_listener(
        self: &Arc<Self>,
        listener: TcpListener,
        config: HostConfig,
        stop: Arc<AtomicBool>,
        kind: Kind,
        connected: Arc<AtomicBool>,
    ) -> Result<(), String> {
        listener.set_nonblocking(true).map_err(|e| e.to_string())?;
        let this = self.clone();
        let spawned = std::thread::Builder::new()
            .name("pairing".into())
            .spawn(move || this.listen(listener, config, stop, kind, connected))
            .map(|_| ())
            .map_err(|e| e.to_string());
        if spawned.is_err() {
            let mut inner = self.inner.lock().unwrap();
            inner.open = None;
            inner.incoming = None;
        }
        spawned
    }

    // MARK: Discovery

    /// Announces this computer and answers headsets asking to pair, until `stop_discovery`.
    /// Runs apart from the caller, which may be waiting on the Keychain for the host's id.
    pub fn start_discovery(self: &Arc<Self>) {
        let this = self.clone();
        let spawned = std::thread::Builder::new().name("presence-start".into()).spawn(move || {
            let host_id = match this.store().host_id() {
                Ok(id) => id,
                Err(e) => return tracing::warn!(error = %e, "discovery needs this computer's id"),
            };
            let (asked, withdrawn) = (Arc::downgrade(&this), Arc::downgrade(&this));
            let presence = Presence::start(
                host_id,
                computer_name(),
                Arc::new(move |request| {
                    if let Some(this) = asked.upgrade() {
                        this.requested(request);
                    }
                }),
                Arc::new(move |device_id| {
                    if let Some(this) = withdrawn.upgrade() {
                        this.withdrawn(device_id);
                    }
                }),
            );
            // Replaces any earlier one, withdrawn once the lock is let go of.
            let earlier = this.presence.lock().unwrap().replace(presence);
            drop(earlier);
            this.changed();
        });
        if let Err(e) = spawned {
            tracing::warn!(error = %e, "couldn't start discovery");
        }
    }

    pub fn stop_discovery(&self) {
        let earlier = self.presence.lock().unwrap().take();
        drop(earlier);
        let mut inner = self.inner.lock().unwrap();
        inner.queued.clear();
        if let Some(incoming) = inner.incoming.take() {
            incoming.stop.store(true, Ordering::SeqCst);
        }
        drop(inner);
        self.changed();
    }

    /// A headset asked to pair: opens a listener for it alone, or queues it behind the one
    /// pairing now. A headset's request outranks the fixed port left open by hand: the person is
    /// at the headset, and Pair here is still what lets it in.
    fn requested(self: &Arc<Self>, request: PairRequest) {
        {
            let mut inner = self.inner.lock().unwrap();
            if inner.incoming.is_some() {
                if !inner.queued.iter().any(|r| r.device_id == request.device_id) {
                    inner.queued.push(request);
                }
                return;
            }
        }
        self.stop();
        if let Err(e) = self.open_for(request.clone()) {
            tracing::warn!(error = %e, "couldn't open a listener for a pairing request");
            self.inner.lock().unwrap().last = Some(format!("{} asked to pair, but: {e}", request.name));
            self.changed();
        }
    }

    /// The headset took its request back (it gave up, or finished): forget it, and close its
    /// listener unless the exchange is already running.
    fn withdrawn(&self, device_id: &str) {
        let mut inner = self.inner.lock().unwrap();
        inner.queued.retain(|r| r.device_id != device_id);
        if let Some(incoming) = &inner.incoming
            && incoming.request.device_id == device_id
            && !incoming.connected.load(Ordering::SeqCst)
        {
            incoming.stop.store(true, Ordering::SeqCst);
        }
    }

    fn open_for(self: &Arc<Self>, request: PairRequest) -> Result<(), String> {
        let listener = TcpListener::bind(("0.0.0.0", 0)).map_err(|e| format!("no port to listen on ({e})"))?;
        let port = listener.local_addr().map_err(|e| e.to_string())?.port();
        let (config, _) = self.host_config(Some(&request))?;
        let stop = Arc::new(AtomicBool::new(false));
        let connected = Arc::new(AtomicBool::new(false));
        {
            let mut inner = self.inner.lock().unwrap();
            inner.incoming = Some(Incoming { request: request.clone(), port, stop: stop.clone(), connected: connected.clone() });
            inner.last = None;
        }
        self.spawn_listener(listener, config, stop, Kind::Request(request), connected)?;
        self.changed();
        Ok(())
    }

    fn listen(self: Arc<Self>, listener: TcpListener, config: HostConfig, stop: Arc<AtomicBool>, kind: Kind, connected: Arc<AtomicBool>) {
        let (until, device) = match &kind {
            Kind::Direct => (Instant::now() + OPEN_FOR, None),
            Kind::Request(request) => (Instant::now() + REQUEST_OPEN_FOR, Some(request.clone())),
        };
        // Found by headsets browsing for `_vergence-pair._tcp` while the port is open: a request's
        // listener names the one device it is for, and only that device matches it.
        let port = listener.local_addr().map(|a| a.port()).unwrap_or_default();
        let mut announcement =
            crate::announce::Announcement::pairing(&config.host_name, port, &config.host_id, device.as_ref().map(|r| r.device_id.as_str()));
        tracing::info!(port, announced = announcement.is_announced(), request = device.is_some(), "pairing open");
        let mut exchanged = false;
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
            connected.store(true, Ordering::SeqCst);
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
            exchanged = true;
            // A request's listener is for one exchange, whatever came of it; the fixed port
            // stays open for another try until someone pairs.
            if paired || device.is_some() {
                break;
            }
            connected.store(false, Ordering::SeqCst);
        }
        drop(announcement);
        let mut inner = self.inner.lock().unwrap();
        match &kind {
            Kind::Direct => {
                if inner.open.as_ref().is_some_and(|open| Arc::ptr_eq(&open.stop, &stop)) {
                    inner.open = None;
                }
            }
            Kind::Request(request) => {
                if inner.incoming.as_ref().is_some_and(|incoming| Arc::ptr_eq(&incoming.stop, &stop)) {
                    inner.incoming = None;
                    if !exchanged && !stop.load(Ordering::SeqCst) {
                        inner.last = Some(format!("{} asked to pair but never connected.", request.name));
                    }
                }
            }
        }
        let next = match kind {
            Kind::Request(_) if inner.incoming.is_none() && !inner.queued.is_empty() => Some(inner.queued.remove(0)),
            _ => None,
        };
        drop(inner);
        self.changed();
        if let Some(next) = next {
            self.requested(next);
        }
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
            "incoming": inner.incoming.as_ref().map(|i| json!({ "deviceName": i.request.name, "port": i.port })),
            "queued": inner.queued.iter().map(|r| &r.name).collect::<Vec<_>>(),
            "visible": self.presence.lock().unwrap().as_ref().map(|p| {
                let v = p.visibility();
                json!({ "announced": v.announced, "watching": v.watching })
            }),
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
    vergence_coordinator::paired::lan_addresses()
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

    /// A coordinator and pairing over a temp dir, with free ports.
    async fn host() -> (tempfile::TempDir, Arc<CoordinatorHost>, Arc<Pairing>) {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::new(dir.path());
        let coordinator = Arc::new(CoordinatorHost::new(paths.clone()));
        let settings = crate::settings::Settings { port: free_port(), paired_port: free_port(), ..Default::default() };
        coordinator.start(&settings).await.unwrap();
        let pairing = Arc::new(Pairing::new(paths, coordinator.clone()));
        (dir, coordinator, pairing)
    }

    fn wait_for(what: &str, mut ready: impl FnMut() -> bool) {
        for _ in 0..200 {
            if ready() {
                return;
            }
            std::thread::sleep(Duration::from_millis(25));
        }
        panic!("timed out waiting for {what}");
    }

    fn request(id: &str) -> PairRequest {
        PairRequest { device_id: id.into(), name: format!("Headset {id}") }
    }

    /// A headset dialling `port` with its own id, as the Swift client does once it finds the record.
    fn dial(port: u16, id: &str) -> Result<vergence_companion::PairingGrant, String> {
        let stream = std::net::TcpStream::connect(("127.0.0.1", port)).map_err(|e| e.to_string())?;
        let device = DevicePairing::new(DeviceConfig {
            namespace: namespace(),
            device_id: id.into(),
            device_name: format!("Headset {id}"),
            host_id: None,
            requested: vec!["play".into()],
            ssh_public_key: None,
        });
        pair_with_host(stream, device, Duration::from_secs(10), |_| {}).map_err(|e| e.to_string())
    }

    fn approve_when_asked(pairing: &Arc<Pairing>, pair: bool) -> std::thread::JoinHandle<()> {
        let pairing = pairing.clone();
        std::thread::spawn(move || {
            wait_for("a code to approve", || pairing.status()["waiting"]["id"].is_u64());
            let id = pairing.status()["waiting"]["id"].as_u64().unwrap();
            pairing.decide(id, pair).unwrap();
        })
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_pairing_request_opens_a_listener_for_that_headset_alone() {
        let (_dir, coordinator, pairing) = host().await;
        pairing.requested(request("HS-1"));
        let status = pairing.status();
        assert_eq!(status["incoming"]["deviceName"], "Headset HS-1");
        let port = status["incoming"]["port"].as_u64().unwrap() as u16;

        // Another device dialling it first is turned away, enrolled with nothing, and uses up the listener.
        let stranger = tokio::task::spawn_blocking(move || dial(port, "HS-2")).await.unwrap();
        assert!(stranger.is_err(), "{stranger:?}");
        wait_for("the listener to close", || pairing.status()["incoming"].is_null());
        assert!(pairing.devices().is_empty());

        // The headset asks again (a request is made once per appearance) and pairs.
        pairing.requested(request("HS-1"));
        let port = pairing.status()["incoming"]["port"].as_u64().unwrap() as u16;
        let approver = approve_when_asked(&pairing, true);
        let grant = tokio::task::spawn_blocking(move || dial(port, "HS-1")).await.unwrap().unwrap();
        approver.join().unwrap();
        assert_eq!(grant.grants.as_deref(), Some(&["play".to_string()][..]));
        wait_for("the listener to close", || pairing.status()["incoming"].is_null());
        assert_eq!(pairing.status()["last"], "Paired with Headset HS-1.");
        assert_eq!(pairing.devices().len(), 1);
        coordinator.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn requests_wait_their_turn_and_a_withdrawn_one_is_dropped() {
        let (_dir, coordinator, pairing) = host().await;
        pairing.requested(request("HS-1"));
        pairing.requested(request("HS-2"));
        pairing.requested(request("HS-3"));
        pairing.requested(request("HS-2"));
        assert_eq!(pairing.status()["incoming"]["deviceName"], "Headset HS-1");
        assert_eq!(pairing.status()["queued"], json!(["Headset HS-2", "Headset HS-3"]));

        // HS-2 gives up while it waits; HS-1 gives up too, which opens HS-3's.
        pairing.withdrawn("HS-2");
        assert_eq!(pairing.status()["queued"], json!(["Headset HS-3"]));
        pairing.withdrawn("HS-1");
        wait_for("HS-3's turn", || pairing.status()["incoming"]["deviceName"] == "Headset HS-3");
        assert_eq!(pairing.status()["queued"], json!([]));

        // A headset's request closes the fixed port left open by hand, which it would otherwise wait behind.
        pairing.withdrawn("HS-3");
        wait_for("the listener to close", || pairing.status()["incoming"].is_null());
        pairing.start(free_port()).unwrap();
        assert!(!pairing.status()["open"].is_null());
        pairing.requested(request("HS-4"));
        assert!(pairing.status()["open"].is_null());
        assert_eq!(pairing.status()["incoming"]["deviceName"], "Headset HS-4");
        pairing.stop_discovery();
        wait_for("the listener to close", || pairing.status()["incoming"].is_null());
        coordinator.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_headset_pairs_and_gets_a_play_token() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::new(dir.path());
        let coordinator = Arc::new(CoordinatorHost::new(paths.clone()));
        let settings = crate::settings::Settings { port: free_port(), paired_port: free_port(), ..Default::default() };
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
        // And it reaches the coordinator over the paired TLS-PSK channel, with its play grant.
        let paired_port = settings.paired_port;
        let token = grant.token.clone();
        let (health, queue) = tokio::task::spawn_blocking(move || {
            use std::io::{Read, Write};
            let credential = vergence_companion::Credential::new(token, Some("HS-1".into()));
            let client = vergence_companion::tls::PskClient::new(&vergence_coordinator::paired::channel(), &credential).unwrap();
            let get = |path: &str| {
                let tcp = std::net::TcpStream::connect(("127.0.0.1", paired_port)).unwrap();
                tcp.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
                let mut tls = client.connect(tcp).unwrap();
                write!(tls, "GET {path} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n").unwrap();
                let mut out = Vec::new();
                let _ = tls.read_to_end(&mut out);
                String::from_utf8_lossy(&out).to_string()
            };
            let health = get("/v1/titles/jellyfin/x/smooth-motion-3d");
            let queue = get("/v1/queue");
            (health, queue)
        })
        .await
        .unwrap();
        assert!(health.starts_with("HTTP/1.1 404"), "{health}");
        assert!(health.contains("\"state\":\"none\""), "{health}");
        assert!(queue.starts_with("HTTP/1.1 403"), "{queue}");

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
