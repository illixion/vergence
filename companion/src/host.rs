//! Pairing, the host's side (`RAVEPairingServer`), without I/O.
//!
//! The coordinator plays the host: headsets and workers pair *to* it. Feed it
//! the bytes a device sends ([`HostPairing::receive`]), send what
//! [`HostPairing::take_outgoing`] returns, and when it asks for approval show
//! the six-digit code and call [`HostPairing::approve`] or
//! [`HostPairing::decline`]. [`crate::net::serve_pairing`] does the socket
//! part for a blocking `TcpStream`.

use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::credential::Credential;
use crate::error::PairingError;
use crate::exchange::{Agreement, PairingExchange};
use crate::message::{PairingGrant, PairingMessage};
use crate::namespace::Namespace;
use crate::store::{DeviceRecord, now_secs};

/// How long one pairing may take, whatever happens.
pub const PAIRING_TIMEOUT: Duration = Duration::from_secs(120);

/// The device a Bonjour-found pairing listener was opened for (from its knock).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpectedDevice {
    pub id: String,
    pub name: String,
}

/// Decides whether an offered public key line may be shown to the user.
pub type PublicKeyFilter = Arc<dyn Fn(&str) -> bool + Send + Sync>;

#[derive(Clone)]
pub struct HostConfig {
    pub namespace: Namespace,
    /// The id devices remember this host by ([`crate::store::CredentialStore::host_id`]).
    pub host_id: String,
    pub host_name: String,
    /// `None` when the device dialled the fixed pairing port (from an address
    /// link): its commit says who it is.
    pub expected_device: Option<ExpectedDevice>,
    /// What the device asks for when its commit names nothing.
    pub default_grants: Vec<String>,
    /// The host's addresses for the grant, wired first.
    pub addresses: Vec<String>,
    /// Whether an offered public key line is one the app would install; one
    /// that isn't is dropped before the user sees it. `None` drops every one.
    pub accepts_public_key: Option<PublicKeyFilter>,
}

impl fmt::Debug for HostConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HostConfig")
            .field("namespace", &self.namespace)
            .field("host_id", &self.host_id)
            .field("host_name", &self.host_name)
            .field("expected_device", &self.expected_device)
            .field("default_grants", &self.default_grants)
            .field("addresses", &self.addresses)
            .finish_non_exhaustive()
    }
}

impl HostConfig {
    pub fn new(
        namespace: Namespace,
        host_id: impl Into<String>,
        host_name: impl Into<String>,
    ) -> Self {
        Self {
            namespace,
            host_id: host_id.into(),
            host_name: host_name.into(),
            expected_device: None,
            default_grants: Vec::new(),
            addresses: Vec::new(),
            accepts_public_key: None,
        }
    }
}

/// What the app shows the user: compare `code` with the device's screen.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApprovalRequest {
    /// Six digits; [`PairingExchange::display`] spaces them.
    pub code: String,
    pub device_id: String,
    pub device_name: String,
    /// What the device asks for (sorted); the dialog starts from these and the user decides.
    pub requested: Vec<String>,
    /// The public key line the device offered, already vetted.
    pub ssh_public_key: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HostEvent {
    /// The exchange checked out: ask the user.
    ApprovalNeeded(ApprovalRequest),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HostOutcome {
    /// Store the record (before sending the accept, see [`HostPairing::approve`]).
    Paired(DeviceRecord),
    /// The user chose not to pair.
    Declined,
    /// Timed out, abandoned or malformed: no answer from the user.
    Abandoned,
}

enum State {
    AwaitingCommit,
    AwaitingReveal {
        commitment: Vec<u8>,
        device: ExpectedDevice,
        requested: Vec<String>,
        ssh: Option<String>,
    },
    AwaitingApproval {
        request: ApprovalRequest,
        agreement: Agreement,
    },
    Done(HostOutcome),
}

pub struct HostPairing {
    config: HostConfig,
    exchange: PairingExchange,
    inbox: Vec<u8>,
    outbox: Vec<u8>,
    state: State,
    started: Instant,
}

impl fmt::Debug for HostPairing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HostPairing")
            .field("config", &self.config)
            .field("outcome", &self.outcome())
            .finish()
    }
}

impl HostPairing {
    pub fn new(config: HostConfig) -> Self {
        Self::with_exchange(config, PairingExchange::random())
    }

    /// With a given key and nonce, for tests.
    pub fn with_exchange(config: HostConfig, exchange: PairingExchange) -> Self {
        Self {
            config,
            exchange,
            inbox: Vec::new(),
            outbox: Vec::new(),
            state: State::AwaitingCommit,
            started: Instant::now(),
        }
    }

    pub fn config(&self) -> &HostConfig {
        &self.config
    }

    pub fn deadline(&self) -> Instant {
        self.started + PAIRING_TIMEOUT
    }

    pub fn is_expired(&self, now: Instant) -> bool {
        now >= self.deadline()
    }

    /// The question waiting for the user, if any.
    pub fn approval(&self) -> Option<&ApprovalRequest> {
        match &self.state {
            State::AwaitingApproval { request, .. } => Some(request),
            _ => None,
        }
    }

    pub fn outcome(&self) -> Option<&HostOutcome> {
        match &self.state {
            State::Done(outcome) => Some(outcome),
            _ => None,
        }
    }

    pub fn is_finished(&self) -> bool {
        matches!(self.state, State::Done(_))
    }

    /// Bytes to send to the device, in order. Empty when there is nothing.
    pub fn take_outgoing(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.outbox)
    }

    /// Feeds bytes read from the device. An error ends the pairing (abandoned).
    pub fn receive(&mut self, bytes: &[u8]) -> Result<Option<HostEvent>, PairingError> {
        if self.is_finished() {
            return Ok(None);
        }
        self.inbox.extend_from_slice(bytes);
        let mut event = None;
        // While the user decides, anything more just waits (the device has
        // nothing to say until the answer).
        while matches!(
            self.state,
            State::AwaitingCommit | State::AwaitingReveal { .. }
        ) {
            let message = match PairingMessage::take(&mut self.inbox) {
                Ok(Some(message)) => message,
                Ok(None) => break,
                Err(e) => return Err(self.fail(e.into())),
            };
            match self.handle(message) {
                Ok(Some(e)) => event = Some(e),
                Ok(None) => {}
                Err(e) => return Err(self.fail(e)),
            }
        }
        Ok(event)
    }

    fn fail(&mut self, error: PairingError) -> PairingError {
        self.state = State::Done(HostOutcome::Abandoned);
        error
    }

    fn handle(&mut self, message: PairingMessage) -> Result<Option<HostEvent>, PairingError> {
        match (
            std::mem::replace(&mut self.state, State::Done(HostOutcome::Abandoned)),
            message,
        ) {
            (
                State::AwaitingCommit,
                PairingMessage::Commit {
                    headset_id,
                    headset_name,
                    commitment,
                    requested,
                    ssh_public_key,
                },
            ) => {
                if headset_id.is_empty() || headset_id.chars().count() > 64 {
                    return Err(PairingError::Protocol(
                        "a commit without a usable device id",
                    ));
                }
                let device = match &self.config.expected_device {
                    Some(expected) if expected.id != headset_id => {
                        return Err(PairingError::Protocol(
                            "a commit from another device than the one asked for",
                        ));
                    }
                    Some(expected) => expected.clone(),
                    None => ExpectedDevice {
                        id: headset_id,
                        name: headset_name.chars().take(80).collect(),
                    },
                };
                let mut requested = requested.unwrap_or_else(|| self.config.default_grants.clone());
                requested.sort();
                requested.dedup();
                let ssh = ssh_public_key.filter(|key| {
                    self.config
                        .accepts_public_key
                        .as_ref()
                        .is_some_and(|accepts| accepts(key))
                });
                self.outbox.extend(
                    PairingMessage::MacHello {
                        mac_id: self.config.host_id.clone(),
                        mac_name: self.config.host_name.clone(),
                        public_key: self.exchange.public_key().to_vec(),
                        nonce: self.exchange.nonce().to_vec(),
                    }
                    .framed(),
                );
                self.state = State::AwaitingReveal {
                    commitment,
                    device,
                    requested,
                    ssh,
                };
                Ok(None)
            }
            (
                State::AwaitingReveal {
                    commitment,
                    device,
                    requested,
                    ssh,
                },
                PairingMessage::Reveal { public_key, nonce },
            ) => {
                let agreement = PairingExchange::agree(
                    &self.config.namespace,
                    &self.exchange,
                    &public_key,
                    &commitment,
                    &public_key,
                    &nonce,
                    &self.exchange.public_key(),
                    &self.exchange.nonce(),
                )
                .ok_or(PairingError::Protocol(
                    "the reveal didn't match its commitment",
                ))?;
                let request = ApprovalRequest {
                    code: agreement.code.clone(),
                    device_id: device.id,
                    device_name: device.name,
                    requested,
                    ssh_public_key: ssh,
                };
                self.state = State::AwaitingApproval {
                    request: request.clone(),
                    agreement,
                };
                Ok(Some(HostEvent::ApprovalNeeded(request)))
            }
            (_, message) => Err(PairingError::Protocol(match message {
                PairingMessage::Commit { .. } => "a second commit",
                PairingMessage::Reveal { .. } => "a reveal before the commit",
                _ => "a host's message from the device",
            })),
        }
    }

    /// The user approved with these grants: mints the device's token, seals
    /// the grant and queues the accept. Store the returned record **before**
    /// sending the queued bytes; if storing fails, call [`Self::retract`] so the
    /// device is told no instead.
    pub fn approve(
        &mut self,
        grants: impl IntoIterator<Item = String>,
    ) -> Result<DeviceRecord, PairingError> {
        self.approve_with_token(grants, Credential::generate_token())
    }

    pub fn approve_with_token(
        &mut self,
        grants: impl IntoIterator<Item = String>,
        token: String,
    ) -> Result<DeviceRecord, PairingError> {
        let State::AwaitingApproval { request, agreement } = &self.state else {
            return Err(PairingError::State("nothing is waiting for approval"));
        };
        let mut grants: Vec<String> = grants.into_iter().collect();
        grants.sort();
        grants.dedup();
        let grant = PairingGrant {
            token: token.clone(),
            mac_id: self.config.host_id.clone(),
            mac_name: self.config.host_name.clone(),
            addresses: self.config.addresses.clone(),
            grants: Some(grants.clone()),
        };
        let sealed = PairingExchange::seal(&grant, &agreement.key);
        let record = DeviceRecord {
            device_id: request.device_id.clone(),
            name: request.device_name.clone(),
            token,
            grants,
            ssh_public_key: request.ssh_public_key.clone(),
            paired_at: now_secs(),
        };
        self.outbox.extend(
            PairingMessage::Accept {
                sealed_grant: sealed,
            }
            .framed(),
        );
        self.state = State::Done(HostOutcome::Paired(record.clone()));
        Ok(record)
    }

    /// After [`Self::approve`], when the record couldn't be stored: the accept
    /// is withdrawn (if not yet taken) and a deny sent instead.
    pub fn retract(&mut self) {
        if matches!(self.state, State::Done(HostOutcome::Paired(_))) {
            self.outbox.clear();
            self.outbox.extend(PairingMessage::Deny {}.framed());
            self.state = State::Done(HostOutcome::Abandoned);
        }
    }

    /// The user said no: a deny is queued.
    pub fn decline(&mut self) {
        self.end_with_deny(HostOutcome::Declined);
    }

    /// Gives up: a deny is queued if the device was waiting for an answer.
    pub fn abandon(&mut self) {
        self.end_with_deny(HostOutcome::Abandoned);
    }

    fn end_with_deny(&mut self, outcome: HostOutcome) {
        match self.state {
            State::Done(_) => {}
            State::AwaitingApproval { .. } => {
                self.outbox.extend(PairingMessage::Deny {}.framed());
                self.state = State::Done(outcome);
            }
            _ => self.state = State::Done(HostOutcome::Abandoned),
        }
    }

    /// The device hung up (or the connection failed).
    pub fn peer_closed(&mut self) {
        if !self.is_finished() {
            self.state = State::Done(HostOutcome::Abandoned);
        }
    }
}
