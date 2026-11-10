//! Announces this computer to headsets on the local network, as RAVECompanion's namespace names
//! the services (`_vergence-*._tcp`):
//!
//! - the host record, `_vergence-mac._tcp` (TXT `v`, `id`, `open`, `addr`): a bare record with
//!   nothing listening behind it, up as long as the app is. A headset browses for it to list the
//!   computers it can pair with;
//! - a pairing listener, `_vergence-pair._tcp` (TXT `id`, and `hs` for the one device it was
//!   opened for): up only while a pairing is open. A headset that asked to pair finds the one
//!   with its own id and dials it, and the six-digit comparison does the rest. Nothing a record
//!   says grants anything.
//!
//! Through the system's own responder: `dns-sd` on macOS (mDNSResponder), `avahi-publish-service`
//! on Linux, and on Windows the DNS Client service's own mDNS responder through
//! `DnsServiceRegister` (dnsapi.dll, Windows 10 1809 and later), so nothing extra is installed.

use std::process::{Child, Command, Stdio};

use vergence_companion::discovery::{VERSION, key};

/// A record, withdrawn when this is dropped.
pub struct Announcement {
    child: Option<Child>,
    #[cfg(windows)]
    registration: Option<windows_dnssd::Registration>,
}

impl Announcement {
    /// The host record: `name` is how headsets list this computer. No socket stands behind it;
    /// SRV needs a port, so the discard port stands in (as RAVECompanionPresence does).
    pub fn host(name: &str, host_id: &str, addresses: &[String]) -> Announcement {
        let service = vergence_coordinator::paired::namespace().host_service_type();
        let mut txt = vec![
            (key::VERSION.to_string(), VERSION.to_string()),
            (key::HOST_ID.to_string(), host_id.to_string()),
            // The paired channel listens whenever the app does, so there is nothing to knock open.
            (key::OPEN.to_string(), "1".to_string()),
        ];
        if !addresses.is_empty() {
            txt.push((key::ADDRESSES.to_string(), addresses.join(",")));
        }
        Self::service(name, &service, 9, &txt)
    }

    /// Announces `name`'s pairing listener on `port`. Never fails: without a responder the
    /// headset pairs by address instead. `device_id` is the one device a listener opened for a
    /// pairing request is for; the one on the fixed port is for any.
    pub fn pairing(name: &str, port: u16, host_id: &str, device_id: Option<&str>) -> Announcement {
        let service = vergence_coordinator::paired::namespace().pair_service_type();
        let mut txt = vec![(key::HOST_ID.to_string(), host_id.to_string())];
        if let Some(device_id) = device_id {
            txt.push((key::DEVICE_ID.to_string(), device_id.to_string()));
        }
        Self::service(name, &service, port, &txt)
    }

    fn service(name: &str, service: &str, port: u16, txt: &[(String, String)]) -> Announcement {
        let pairs: Vec<String> = txt.iter().map(|(k, v)| format!("{k}={v}")).collect();
        let command = if cfg!(target_os = "macos") {
            let mut args = vec!["-R".to_string(), name.into(), service.into(), "local".into(), port.to_string()];
            args.extend(pairs);
            supervised("/usr/bin/dns-sd", args)
        } else if cfg!(target_os = "linux") {
            let mut args = vec![name.to_string(), service.to_string(), port.to_string()];
            args.extend(pairs);
            supervised("avahi-publish-service", args)
        } else {
            None
        };
        #[cfg(windows)]
        let registration = {
            let borrowed: Vec<(&str, &str)> = txt.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
            windows_dnssd::Registration::new(name, service, port, &borrowed)
        };
        let child = command.and_then(|mut c| {
            match c.stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).spawn() {
                Ok(child) => Some(child),
                Err(e) => {
                    tracing::info!(error = %e, "no mDNS responder to announce on");
                    None
                }
            }
        });
        Announcement {
            child,
            #[cfg(windows)]
            registration,
        }
    }

    /// Whether a responder is carrying the record: its process still runs, or the Windows
    /// registration was confirmed.
    pub fn is_announced(&mut self) -> bool {
        #[cfg(windows)]
        if self.registration.is_some() {
            return true;
        }
        match self.child.as_mut().map(Child::try_wait) {
            Some(Ok(None)) => true,
            // Gone already: the responder isn't installed, or refused the record.
            _ => false,
        }
    }
}

impl Drop for Announcement {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            stop(&mut child);
        }
    }
}

/// The responder, run under a small shell that ends it when this app does by any means (a crash,
/// a kill): a record that stays up after the app is gone would list a computer nobody answers
/// for, until a restart. The responder's own process is the shell's child, so dropping the
/// announcement asks the shell to stop (it ends the responder) rather than killing it outright.
#[cfg(unix)]
fn supervised(program: &str, args: Vec<String>) -> Option<Command> {
    const SUPERVISOR: &str = r#"parent=$PPID; "$@" & child=$!; trap 'kill $child 2>/dev/null; exit 0' TERM INT; (while kill -0 $parent 2>/dev/null && kill -0 $child 2>/dev/null; do sleep 2; done; kill $child 2>/dev/null) & wait $child"#;
    let mut c = Command::new("/bin/sh");
    c.arg("-c").arg(SUPERVISOR).arg("sh").arg(program).args(args);
    Some(c)
}

#[cfg(not(unix))]
fn supervised(_program: &str, _args: Vec<String>) -> Option<Command> {
    None
}

#[cfg(unix)]
fn stop(child: &mut Child) {
    // SAFETY: a signal to our own child, which is still ours to wait for.
    unsafe { libc::kill(child.id() as libc::pid_t, libc::SIGTERM) };
    for _ in 0..20 {
        if matches!(child.try_wait(), Ok(Some(_))) {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    let _ = child.kill();
    let _ = child.wait();
}

#[cfg(not(unix))]
fn stop(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

/// Windows: a DNS-SD service instance registered with the DNS Client service's mDNS responder.
#[cfg(windows)]
mod windows_dnssd {
    use std::ffi::c_void;
    use std::sync::{Arc, Condvar, Mutex};
    use std::time::Duration;

    use windows_sys::Win32::NetworkManagement::Dns::{
        DNS_QUERY_REQUEST_VERSION1, DNS_SERVICE_INSTANCE, DNS_SERVICE_REGISTER_REQUEST, DnsServiceConstructInstance,
        DnsServiceDeRegister, DnsServiceFreeInstance, DnsServiceRegister,
    };

    /// `DNS_REQUEST_PENDING`: the call was accepted and its callback will follow.
    const PENDING: u32 = 9506;
    /// How long to wait for the responder to confirm a registration or its withdrawal.
    const WAIT: Duration = Duration::from_secs(3);

    /// What the completion callback reports; shared with it through the request's context.
    struct Signal {
        status: Mutex<Option<u32>>,
        changed: Condvar,
        /// Our own instance, which the callback must not free.
        ours: usize,
    }

    impl Signal {
        fn wait(&self) -> Option<u32> {
            let guard = self.status.lock().unwrap_or_else(|e| e.into_inner());
            let (guard, _) = self.changed.wait_timeout_while(guard, WAIT, |s| s.is_none()).unwrap_or_else(|e| e.into_inner());
            *guard
        }
    }

    unsafe extern "system" fn completed(status: u32, context: *const c_void, instance: *const DNS_SERVICE_INSTANCE) {
        // SAFETY: the context is the `Signal` the registration keeps alive until both of its
        // callbacks have run (or leaks it if one never comes).
        let signal = unsafe { &*(context as *const Signal) };
        if !instance.is_null() && instance as usize != signal.ours {
            // The callback's copy of the instance is ours to free.
            unsafe { DnsServiceFreeInstance(instance) };
        }
        *signal.status.lock().unwrap_or_else(|e| e.into_inner()) = Some(status);
        signal.changed.notify_all();
    }

    fn wide(text: &str) -> Vec<u16> {
        text.encode_utf16().chain(std::iter::once(0)).collect()
    }

    pub struct Registration {
        request: Box<DNS_SERVICE_REGISTER_REQUEST>,
        instance: *mut DNS_SERVICE_INSTANCE,
        signal: Arc<Signal>,
    }

    // SAFETY: the raw pointers are owned here and used only to deregister and free, once.
    unsafe impl Send for Registration {}
    unsafe impl Sync for Registration {}

    impl Registration {
        /// Registers `<name>.<service>.local` on `port` with a TXT record of `txt`, answered
        /// for `<computer name>.local` at this computer's LAN address.
        pub fn new(name: &str, service: &str, port: u16, txt: &[(&str, &str)]) -> Option<Registration> {
            // A DNS label is at most 63 bytes, and a dot would start another one.
            let label: String = name.replace('.', " ").chars().take(60).collect();
            let instance_name = wide(&format!("{label}.{service}.local"));
            let computer = std::env::var("COMPUTERNAME").unwrap_or_else(|_| "vergence".into());
            let host = wide(&format!("{computer}.local"));
            let ip4: Option<u32> = vergence_coordinator::paired::lan_addresses()
                .iter()
                .find_map(|a| a.parse::<std::net::Ipv4Addr>().ok())
                .map(|ip| u32::from_ne_bytes(ip.octets()));
            let keys: Vec<Vec<u16>> = txt.iter().map(|(k, _)| wide(k)).collect();
            let values: Vec<Vec<u16>> = txt.iter().map(|(_, v)| wide(v)).collect();
            let key_ptrs: Vec<*const u16> = keys.iter().map(|k| k.as_ptr()).collect();
            let value_ptrs: Vec<*const u16> = values.iter().map(|v| v.as_ptr()).collect();
            // SAFETY: every pointer is valid for the call; the function copies what it keeps.
            let instance = unsafe {
                DnsServiceConstructInstance(
                    instance_name.as_ptr(),
                    host.as_ptr(),
                    ip4.as_ref().map_or(std::ptr::null(), |ip| ip as *const u32),
                    std::ptr::null(),
                    port,
                    0,
                    0,
                    key_ptrs.len() as u32,
                    key_ptrs.as_ptr(),
                    value_ptrs.as_ptr(),
                )
            };
            if instance.is_null() {
                tracing::info!("couldn't make the DNS-SD instance to announce pairing");
                return None;
            }
            let signal = Arc::new(Signal { status: Mutex::new(None), changed: Condvar::new(), ours: instance as usize });
            let request = Box::new(DNS_SERVICE_REGISTER_REQUEST {
                Version: DNS_QUERY_REQUEST_VERSION1,
                InterfaceIndex: 0,
                pServiceInstance: instance,
                pRegisterCompletionCallback: Some(completed),
                pQueryContext: Arc::as_ptr(&signal) as *mut c_void,
                hCredentials: std::ptr::null_mut(),
                unicastEnabled: 0,
            });
            // SAFETY: the request, its instance and its context outlive the registration.
            let status = unsafe { DnsServiceRegister(&*request, std::ptr::null_mut()) };
            if status != PENDING {
                tracing::info!(status, "the mDNS responder refused the pairing announcement");
                unsafe { DnsServiceFreeInstance(instance) };
                return None;
            }
            match signal.wait() {
                Some(0) => tracing::info!("pairing announced over mDNS"),
                other => tracing::info!(status = ?other, "the pairing announcement wasn't confirmed"),
            }
            Some(Registration { request, instance, signal })
        }
    }

    impl Drop for Registration {
        fn drop(&mut self) {
            *self.signal.status.lock().unwrap_or_else(|e| e.into_inner()) = None;
            // SAFETY: the same request the registration was made with.
            let status = unsafe { DnsServiceDeRegister(&*self.request, std::ptr::null_mut()) };
            let confirmed = status == PENDING && self.signal.wait().is_some();
            if confirmed || status != PENDING {
                unsafe { DnsServiceFreeInstance(self.instance) };
            } else {
                // A callback may still come and read the context: keep it (and the instance) alive.
                std::mem::forget(self.signal.clone());
                tracing::info!("the pairing announcement's withdrawal wasn't confirmed");
            }
        }
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use std::io::Read;
    use std::time::Duration;

    #[test]
    fn the_pairing_listener_can_be_browsed_for() {
        let name = format!("Vergence test {}", std::process::id());
        let mut announcement = Announcement::pairing(&name, 4869, "TEST-HOST", None);
        assert!(announcement.is_announced());
        let mut browse = Command::new("/usr/bin/dns-sd")
            .args(["-B", "_vergence-pair._tcp", "local"])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        std::thread::sleep(Duration::from_secs(3));
        let _ = browse.kill();
        let mut out = String::new();
        browse.stdout.take().unwrap().read_to_string(&mut out).unwrap();
        let _ = browse.wait();
        assert!(out.contains(&name), "browse saw: {out}");
        drop(announcement);
        // Nothing of it is left running: not the responder, nor the shell that watches it.
        let running = || Command::new("/usr/bin/pgrep").args(["-f", &name]).output().unwrap().status.success();
        for _ in 0..80 {
            if !running() {
                return;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        panic!("the announcement's processes are still running");
    }
}
