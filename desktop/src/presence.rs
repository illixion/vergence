//! This computer's side of RAVECompanion discovery (`RAVECompanionPresence`), for as long as the
//! app runs: it announces itself, and watches for a headset asking to pair.
//!
//! - It announces `_vergence-mac._tcp`, a bare record with this computer's id, its LAN addresses
//!   and `open=1` (the paired channel listens whenever the app does, so there is nothing for a
//!   paired headset to knock open). A headset browses for it to list the computers it can pair
//!   with.
//! - It browses for `_vergence-knock._tcp`, the record a headset advertises while it wants a
//!   computer. One naming this computer's id with `pair=1` is a request to pair: the owner opens a
//!   one-off pairing listener for that device (`Pairing`), the headset finds it by its own id and
//!   dials it, and the six-digit comparison follows. A request is passed on once per appearance;
//!   when the headset withdraws it, `on_withdrawn` says so.
//!
//! A knock can be copied or forged by anything on the network, and that is fine: it only decides
//! whether a port is listening for a while, bound to one device id. Getting a token still takes
//! the person pressing Pair on a matching code. (A paired headset's knock, tagged from its token,
//! is not watched for: nothing here is closed until it knocks.)
//!
//! The announcement goes through the system's responder (`announce`); the knock records are read
//! with a browser of our own (`mdns-sd`), which resolves their TXT on every platform alike.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use mdns_sd::{ServiceDaemon, ServiceEvent};
use vergence_companion::discovery::key;

use crate::announce::Announcement;

/// A headset asking to pair, by its id and the name it gave.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PairRequest {
    pub device_id: String,
    pub name: String,
}

/// What the window can say about being found.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Visibility {
    /// The host record is registered with the system's responder.
    pub announced: bool,
    /// The browser for headsets' knocks is running.
    pub watching: bool,
}

type Handler = Arc<dyn Fn(PairRequest) + Send + Sync>;
type Withdrawn = Arc<dyn Fn(&str) + Send + Sync>;

/// Running while this is alive; dropping it withdraws the announcement and stops watching.
pub struct Presence {
    stop: Arc<AtomicBool>,
    visibility: Arc<Mutex<Visibility>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Presence {
    pub fn start(host_id: String, name: String, on_request: Handler, on_withdrawn: Withdrawn) -> Presence {
        let stop = Arc::new(AtomicBool::new(false));
        let visibility = Arc::new(Mutex::new(Visibility::default()));
        let (flag, shown) = (stop.clone(), visibility.clone());
        let thread = std::thread::Builder::new()
            .name("presence".into())
            .spawn(move || run(&host_id, &name, &flag, &shown, &on_request, &on_withdrawn))
            .inspect_err(|e| tracing::warn!(error = %e, "couldn't start discovery"))
            .ok();
        Presence { stop, visibility, thread }
    }

    pub fn visibility(&self) -> Visibility {
        *self.visibility.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl Drop for Presence {
    /// Returns once the announcement is withdrawn, so a quitting app leaves no record behind.
    /// Never drop one while holding a lock its callbacks take: they may be running.
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(thread) = self.thread.take()
            && thread.thread().id() != std::thread::current().id()
        {
            let _ = thread.join();
        }
    }
}

/// How often the LAN addresses are looked at again (the record carries them).
const ADDRESS_CHECK: Duration = Duration::from_secs(30);

fn run(host_id: &str, name: &str, stop: &AtomicBool, visibility: &Mutex<Visibility>, on_request: &Handler, on_withdrawn: &Withdrawn) {
    let set = |f: &dyn Fn(&mut Visibility)| f(&mut visibility.lock().unwrap_or_else(|e| e.into_inner()));
    let mut addresses = crate::pairing::lan_addresses();
    let mut announcement = Announcement::host(name, host_id, &addresses);
    let announced = announcement.is_announced();
    set(&|v| v.announced = announced);
    tracing::info!(announced, "visible to headsets as {name}");

    let knock_type = format!("{}.local.", vergence_coordinator::paired::namespace().knock_service_type());
    let daemon = ServiceDaemon::new().inspect_err(|e| tracing::info!(error = %e, "can't watch for headsets")).ok();
    let events = daemon.as_ref().and_then(|d| d.browse(&knock_type).inspect_err(|e| tracing::info!(error = %e, "can't browse for knocks")).ok());
    set(&|v| v.watching = events.is_some());

    // Each live request record, by the record's name, to the device it is for.
    let mut live: HashMap<String, String> = HashMap::new();
    let mut checked = Instant::now();
    while !stop.load(Ordering::SeqCst) {
        if let Some(events) = &events {
            match events.recv_timeout(Duration::from_millis(500)) {
                Ok(ServiceEvent::ServiceResolved(info)) => {
                    let fullname = info.get_fullname().to_string();
                    match request_of(host_id, |k| info.get_property_val_str(k).map(str::to_string)) {
                        Some(request) => {
                            let known = live.values().any(|d| *d == request.device_id);
                            live.insert(fullname, request.device_id.clone());
                            if !known {
                                tracing::info!("a headset is asking to pair");
                                on_request(request);
                            }
                        }
                        // The record is still there but no longer a request.
                        None => withdraw(&mut live, &fullname, on_withdrawn),
                    }
                }
                Ok(ServiceEvent::ServiceRemoved(_, fullname)) => withdraw(&mut live, &fullname, on_withdrawn),
                Ok(_) => {}
                Err(mdns_sd::RecvTimeoutError::Timeout) => {}
                Err(mdns_sd::RecvTimeoutError::Disconnected) => break,
            }
        } else {
            std::thread::sleep(Duration::from_millis(500));
        }
        if checked.elapsed() >= ADDRESS_CHECK {
            checked = Instant::now();
            let now = crate::pairing::lan_addresses();
            // Another network: say so in the record, which can only be replaced.
            if now != addresses || !announcement.is_announced() {
                addresses = now;
                announcement = Announcement::host(name, host_id, &addresses);
                let announced = announcement.is_announced();
                set(&|v| v.announced = announced);
            }
        }
    }
    if let Some(daemon) = daemon {
        let _ = daemon.shutdown();
    }
}

fn withdraw(live: &mut HashMap<String, String>, fullname: &str, on_withdrawn: &Withdrawn) {
    if let Some(device) = live.remove(fullname)
        && !live.values().any(|d| *d == device)
    {
        on_withdrawn(&device);
    }
}

/// The request a knock record makes of this computer, if it makes one: it names this computer's
/// id, says which device it is from and asks to pair (`pair=1`). A paired device's knock carries
/// a tag instead and is not one.
pub fn request_of(host_id: &str, txt: impl Fn(&str) -> Option<String>) -> Option<PairRequest> {
    if txt(key::HOST_ID).as_deref() != Some(host_id) || txt(key::PAIR_REQUEST).as_deref() != Some("1") {
        return None;
    }
    let device_id = txt(key::DEVICE_ID).filter(|id| !id.is_empty() && id.len() <= 64)?;
    // What the headset calls itself is shown to the person, so it is bounded and has no controls.
    let name: String = txt(key::NAME).unwrap_or_default().chars().filter(|c| !c.is_control()).take(80).collect();
    let name = if name.trim().is_empty() { "Headset".to_string() } else { name };
    Some(PairRequest { device_id, name })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn txt<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |k| pairs.iter().find(|(key, _)| *key == k).map(|(_, v)| v.to_string())
    }

    #[test]
    fn a_pairing_knock_for_this_computer_is_a_request() {
        let request = request_of("HOST", txt(&[("v", "1"), ("id", "HOST"), ("hs", "DEV-1"), ("n", "Vision Pro"), ("pair", "1")]));
        assert_eq!(request, Some(PairRequest { device_id: "DEV-1".into(), name: "Vision Pro".into() }));
    }

    #[test]
    fn other_knocks_are_not() {
        // For another computer.
        assert_eq!(request_of("HOST", txt(&[("id", "OTHER"), ("hs", "D"), ("pair", "1")])), None);
        // A paired device's knock: tagged, not asking to pair.
        assert_eq!(request_of("HOST", txt(&[("id", "HOST"), ("hs", "D"), ("t", "tag")])), None);
        // No device named, or an absurd one.
        assert_eq!(request_of("HOST", txt(&[("id", "HOST"), ("pair", "1")])), None);
        assert_eq!(request_of("HOST", txt(&[("id", "HOST"), ("hs", &"x".repeat(65)), ("pair", "1")])), None);
    }

    #[test]
    fn a_name_is_bounded_and_never_empty() {
        let long = "n".repeat(200);
        let request = request_of("H", txt(&[("id", "H"), ("hs", "D"), ("pair", "1"), ("n", &long)])).unwrap();
        assert_eq!(request.name.chars().count(), 80);
        let request = request_of("H", txt(&[("id", "H"), ("hs", "D"), ("pair", "1"), ("n", "a\u{7}b\nc")])).unwrap();
        assert_eq!(request.name, "abc");
        let request = request_of("H", txt(&[("id", "H"), ("hs", "D"), ("pair", "1"), ("n", " \n ")])).unwrap();
        assert_eq!(request.name, "Headset");
    }

    #[test]
    fn a_request_is_passed_on_once_and_withdrawn_when_its_last_record_goes() {
        let gone = Arc::new(Mutex::new(Vec::<String>::new()));
        let record = gone.clone();
        let on_withdrawn: Withdrawn = Arc::new(move |d| record.lock().unwrap().push(d.to_string()));
        let mut live: HashMap<String, String> = HashMap::new();
        live.insert("a".into(), "DEV".into());
        live.insert("b".into(), "DEV".into());
        withdraw(&mut live, "a", &on_withdrawn);
        assert!(gone.lock().unwrap().is_empty());
        withdraw(&mut live, "b", &on_withdrawn);
        assert_eq!(*gone.lock().unwrap(), ["DEV"]);
        withdraw(&mut live, "b", &on_withdrawn);
        assert_eq!(gone.lock().unwrap().len(), 1);
    }
}
