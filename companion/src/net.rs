//! Blocking TCP drivers for the two state machines. Run them on their own
//! thread (or `spawn_blocking`); the pairing port itself is a plain
//! `TcpListener` the caller binds (`0.0.0.0:<Namespace::default_pair_port>`)
//! and keeps open only while pairing is allowed.

use std::io::{self, ErrorKind, Read, Write};
use std::net::{Shutdown, SocketAddr, TcpStream, ToSocketAddrs};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use crate::device::{DeviceEvent, DevicePairing};
use crate::error::PairingError;
use crate::host::{ApprovalRequest, HostEvent, HostOutcome, HostPairing};
use crate::link::PairLink;
use crate::message::PairingGrant;

/// What the user decided.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Decision {
    /// Pair, with the grants they ticked.
    Pair(Vec<String>),
    Decline,
    /// Closed the prompt, or it was cancelled because the device went away.
    Abandon,
}

/// Set once the device hangs up (or the pairing times out) while the user is
/// still deciding, so the prompt can close itself. Safe to poll from a UI timer.
#[derive(Clone, Debug, Default)]
pub struct PeerWatch(Arc<AtomicBool>);

impl PeerWatch {
    pub fn is_lost(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

fn remaining(deadline: Instant) -> Result<Duration, PairingError> {
    let left = deadline.saturating_duration_since(Instant::now());
    if left.is_zero() {
        Err(PairingError::TimedOut)
    } else {
        Ok(left)
    }
}

fn read_some(
    stream: &mut TcpStream,
    deadline: Instant,
    buf: &mut [u8],
) -> Result<usize, PairingError> {
    stream.set_read_timeout(Some(remaining(deadline)?))?;
    match stream.read(buf) {
        Ok(0) => Err(PairingError::Closed),
        Ok(n) => Ok(n),
        Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {
            Err(PairingError::TimedOut)
        }
        Err(e)
            if matches!(
                e.kind(),
                ErrorKind::ConnectionReset | ErrorKind::ConnectionAborted
            ) =>
        {
            Err(PairingError::Closed)
        }
        Err(e) => Err(e.into()),
    }
}

/// Runs one pairing as the host over an accepted connection.
///
/// `approve` is called once the exchange checks out, with the code to show
/// and a [`PeerWatch`]; it may block until the user answers. On approval
/// `enroll` must store the record (and refresh any keyring) before the accept
/// goes out; if it fails, the device is denied and the outcome is
/// `Abandoned`. Protocol failures and timeouts come back as `Abandoned` too,
/// as in RAVECompanion; only a socket error that isn't the peer's doing is an
/// `Err`.
pub fn serve_pairing(
    mut stream: TcpStream,
    mut pairing: HostPairing,
    approve: impl FnOnce(&ApprovalRequest, &PeerWatch) -> Decision,
    enroll: impl FnOnce(&crate::store::DeviceRecord) -> io::Result<()>,
) -> Result<HostOutcome, PairingError> {
    stream.set_nodelay(true).ok();
    let deadline = pairing.deadline();
    let mut buf = [0u8; 8192];
    let request = loop {
        let n = match read_some(&mut stream, deadline, &mut buf) {
            Ok(n) => n,
            Err(PairingError::Io(e)) => return Err(PairingError::Io(e)),
            Err(_) => {
                pairing.peer_closed();
                let _ = stream.shutdown(Shutdown::Both);
                return Ok(HostOutcome::Abandoned);
            }
        };
        let event = pairing.receive(&buf[..n]);
        let out = pairing.take_outgoing();
        if !out.is_empty() {
            stream.write_all(&out)?;
        }
        match event {
            Ok(Some(HostEvent::ApprovalNeeded(request))) => break request,
            Ok(None) => {}
            Err(_) => {
                let _ = stream.shutdown(Shutdown::Both);
                return Ok(HostOutcome::Abandoned);
            }
        }
    };

    // Watch the connection while the user decides: the device says nothing
    // more until the answer, so a read returning is a hang-up (or the deadline).
    let watch = PeerWatch::default();
    let watcher = {
        let mut reader = stream.try_clone()?;
        let flag = watch.0.clone();
        reader.set_read_timeout(Some(
            remaining(deadline).unwrap_or(Duration::from_millis(1)),
        ))?;
        thread::spawn(move || {
            let mut sink = [0u8; 256];
            loop {
                match reader.read(&mut sink) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => continue,
                }
            }
            flag.store(true, Ordering::SeqCst);
        })
    };

    let decision = approve(&request, &watch);
    let lost = watch.is_lost() || Instant::now() >= deadline;
    let outcome = match decision {
        Decision::Pair(_) if lost => {
            pairing.abandon();
            HostOutcome::Abandoned
        }
        Decision::Pair(grants) => {
            let record = pairing.approve(grants)?;
            match enroll(&record) {
                Ok(()) => HostOutcome::Paired(record),
                Err(_) => {
                    pairing.retract();
                    HostOutcome::Abandoned
                }
            }
        }
        Decision::Decline => {
            pairing.decline();
            HostOutcome::Declined
        }
        Decision::Abandon => {
            pairing.abandon();
            HostOutcome::Abandoned
        }
    };
    let out = pairing.take_outgoing();
    let sent = if out.is_empty() {
        Ok(())
    } else {
        stream.write_all(&out).and_then(|_| stream.flush())
    };
    let _ = stream.shutdown(Shutdown::Both);
    let _ = watcher.join();
    match (sent, &outcome) {
        // The device can't have received its grant: it isn't paired, whatever
        // the store says. The caller should forget the record.
        (Err(e), HostOutcome::Paired(_)) => Err(PairingError::Io(e)),
        _ => Ok(outcome),
    }
}

/// Runs one pairing as a device over a connected stream. `on_code` gets the
/// six digits to show while the user approves on the host.
pub fn pair_with_host(
    mut stream: TcpStream,
    mut pairing: DevicePairing,
    timeout: Duration,
    mut on_code: impl FnMut(&str),
) -> Result<PairingGrant, PairingError> {
    stream.set_nodelay(true).ok();
    let deadline = Instant::now() + timeout;
    stream.write_all(&pairing.take_outgoing())?;
    let mut buf = [0u8; 8192];
    loop {
        let n = read_some(&mut stream, deadline, &mut buf)?;
        let events = pairing.receive(&buf[..n])?;
        let out = pairing.take_outgoing();
        if !out.is_empty() {
            stream.write_all(&out)?;
        }
        for event in events {
            match event {
                DeviceEvent::Confirm { code } => on_code(&code),
                DeviceEvent::Paired(grant) => {
                    let _ = stream.shutdown(Shutdown::Both);
                    return Ok(grant);
                }
            }
        }
    }
}

/// Dials a link's addresses in turn (3 s each); the first that answers wins.
pub fn connect_link(link: &PairLink) -> io::Result<(TcpStream, String)> {
    let mut last = io::Error::new(ErrorKind::NotFound, "the link has no addresses");
    for address in &link.addresses {
        let target: Vec<SocketAddr> = match (address.as_str(), link.port).to_socket_addrs() {
            Ok(addrs) => addrs.collect(),
            Err(e) => {
                last = e;
                continue;
            }
        };
        for addr in target {
            match TcpStream::connect_timeout(&addr, Duration::from_secs(3)) {
                Ok(stream) => return Ok((stream, address.clone())),
                Err(e) => last = e,
            }
        }
    }
    Err(last)
}
