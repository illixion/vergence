//! Announces the pairing listener on the local network while it is open, as RAVECompanion's
//! namespace names it: `_vergence-pair._tcp` on the pairing port, TXT `id=<host id>`. A headset
//! (the Quest's Settings → Analysis computer → Find computers) browses for it and dials the
//! port, and the six-digit comparison does the rest; the record grants nothing.
//!
//! Through the system's own responder: `dns-sd` on macOS (mDNSResponder), `avahi-publish-service`
//! on Linux. Windows has no such command; there the headset types the address (or the
//! `vergence://pair` link) the window shows, until Windows' DnsServiceRegister is wired in.

use std::process::{Child, Command, Stdio};

/// The record, withdrawn when this is dropped.
pub struct Announcement {
    child: Option<Child>,
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
        let child = command.and_then(|mut c| {
            match c.stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).spawn() {
                Ok(child) => Some(child),
                Err(e) => {
                    tracing::info!(error = %e, "no mDNS responder to announce pairing on");
                    None
                }
            }
        });
        Announcement { child }
    }

    pub fn is_announced(&self) -> bool {
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
