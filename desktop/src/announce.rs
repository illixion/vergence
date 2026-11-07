//! Announces the pairing listener on the local network while it is open, as RAVECompanion's
//! namespace names it: `_vergence-pair._tcp` on the pairing port, TXT `id=<host id>`. A headset
//! (the Quest's Settings → Analysis computer → Find computers) browses for it and dials the
//! port, and the six-digit comparison does the rest; the record grants nothing.
//!
//! Through the system's own responder: `dns-sd` on macOS (mDNSResponder), `avahi-publish-service`
//! on Linux, and on Windows the DNS Client service's own mDNS responder through
//! `DnsServiceRegister` (dnsapi.dll, Windows 10 1809 and later), so nothing extra is installed.

use std::process::{Child, Command, Stdio};

/// The record, withdrawn when this is dropped.
pub struct Announcement {
    child: Option<Child>,
    #[cfg(windows)]
    registration: Option<windows_dnssd::Registration>,
}

impl Announcement {
    /// Announces `name`'s pairing listener on `port`. Never fails: without a responder the
    /// headset pairs by address instead.
    pub fn pairing(name: &str, port: u16, host_id: &str) -> Announcement {
        let service = vergence_coordinator::paired::namespace().pair_service_type();
        let txt = format!("id={host_id}");
        let command = if cfg!(target_os = "macos") {
            let mut c = Command::new("/usr/bin/dns-sd");
            c.args(["-R", name, &service, "local", &port.to_string(), &txt]);
            Some(c)
        } else if cfg!(target_os = "linux") {
            let mut c = Command::new("avahi-publish-service");
            c.args([name, &service, &port.to_string(), &txt]);
            Some(c)
        } else {
            None
        };
        #[cfg(windows)]
        let registration = windows_dnssd::Registration::new(name, &service, port, &[("id", host_id)]);
        let child = command.and_then(|mut c| {
            match c.stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).spawn() {
                Ok(child) => Some(child),
                Err(e) => {
                    tracing::info!(error = %e, "no mDNS responder to announce pairing on");
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

    pub fn is_announced(&self) -> bool {
        #[cfg(windows)]
        if self.registration.is_some() {
            return true;
        }
        self.child.is_some()
    }
}

impl Drop for Announcement {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
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
        let announcement = Announcement::pairing(&name, 4869, "TEST-HOST");
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
    }
}
