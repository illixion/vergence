//! Pairing, the device's side (`RAVEPairingClient`), without I/O: for a Rust
//! worker pairing with a host, and for tests of the host.
//!
//! The commit is queued on creation; send [`DevicePairing::take_outgoing`]
//! once connected, then feed what the host sends to [`DevicePairing::receive`].

use std::fmt;

use crate::error::PairingError;
use crate::exchange::{Agreement, PairingExchange};
use crate::message::{PairingGrant, PairingMessage};
use crate::namespace::Namespace;

#[derive(Clone, Debug)]
pub struct DeviceConfig {
    pub namespace: Namespace,
    /// This device's stable id towards hosts.
    pub device_id: String,
    pub device_name: String,
    /// The host this pairing is for; `None` takes the first that answers (a
    /// typed address names no host id up front).
    pub host_id: Option<String>,
    /// What to ask the host for. A Swift host rejects a commit naming a grant
    /// it doesn't know, so ask only for what it offers.
    pub requested: Vec<String>,
    /// A public key line offered for the host to install.
    pub ssh_public_key: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeviceEvent {
    /// Show these six digits for the user to compare with the host's screen.
    Confirm { code: String },
    /// The user approved on the host.
    Paired(PairingGrant),
}

enum State {
    AwaitingHello,
    AwaitingAnswer(Agreement),
    Done,
}

pub struct DevicePairing {
    config: DeviceConfig,
    exchange: PairingExchange,
    commitment: [u8; 32],
    inbox: Vec<u8>,
    outbox: Vec<u8>,
    state: State,
}

impl fmt::Debug for DevicePairing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DevicePairing")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl DevicePairing {
    pub fn new(config: DeviceConfig) -> Self {
        Self::with_exchange(config, PairingExchange::random())
    }

    pub fn with_exchange(config: DeviceConfig, exchange: PairingExchange) -> Self {
        let commitment = PairingExchange::commitment(&exchange.public_key(), &exchange.nonce());
        let mut requested = config.requested.clone();
        requested.sort();
        requested.dedup();
        let commit = PairingMessage::Commit {
            headset_id: config.device_id.clone(),
            headset_name: config.device_name.clone(),
            commitment: commitment.to_vec(),
            requested: Some(requested),
            ssh_public_key: config.ssh_public_key.clone(),
        };
        Self {
            config,
            exchange,
            commitment,
            inbox: Vec::new(),
            outbox: commit.framed(),
            state: State::AwaitingHello,
        }
    }

    pub fn take_outgoing(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.outbox)
    }

    pub fn is_finished(&self) -> bool {
        matches!(self.state, State::Done)
    }

    /// Feeds bytes read from the host. An error (including [`PairingError::Denied`]) ends the pairing.
    pub fn receive(&mut self, bytes: &[u8]) -> Result<Vec<DeviceEvent>, PairingError> {
        let mut events = Vec::new();
        if self.is_finished() {
            return Ok(events);
        }
        self.inbox.extend_from_slice(bytes);
        while !self.is_finished() {
            let message = match PairingMessage::take(&mut self.inbox) {
                Ok(Some(message)) => message,
                Ok(None) => break,
                Err(e) => {
                    self.state = State::Done;
                    return Err(e.into());
                }
            };
            match self.handle(message) {
                Ok(event) => events.push(event),
                Err(e) => {
                    self.state = State::Done;
                    return Err(e);
                }
            }
        }
        Ok(events)
    }

    fn handle(&mut self, message: PairingMessage) -> Result<DeviceEvent, PairingError> {
        match (std::mem::replace(&mut self.state, State::Done), message) {
            (
                State::AwaitingHello,
                PairingMessage::MacHello {
                    mac_id,
                    public_key,
                    nonce,
                    ..
                },
            ) => {
                if self
                    .config
                    .host_id
                    .as_ref()
                    .is_some_and(|id| !id.is_empty() && *id != mac_id)
                {
                    return Err(PairingError::Protocol("another host answered"));
                }
                let agreement = PairingExchange::agree(
                    &self.config.namespace,
                    &self.exchange,
                    &public_key,
                    &self.commitment,
                    &self.exchange.public_key(),
                    &self.exchange.nonce(),
                    &public_key,
                    &nonce,
                )
                .ok_or(PairingError::Protocol("the host's key is unusable"))?;
                self.outbox.extend(
                    PairingMessage::Reveal {
                        public_key: self.exchange.public_key().to_vec(),
                        nonce: self.exchange.nonce().to_vec(),
                    }
                    .framed(),
                );
                let code = agreement.code.clone();
                self.state = State::AwaitingAnswer(agreement);
                Ok(DeviceEvent::Confirm { code })
            }
            (State::AwaitingAnswer(agreement), PairingMessage::Accept { sealed_grant }) => Ok(
                DeviceEvent::Paired(PairingExchange::open(&sealed_grant, &agreement.key)?),
            ),
            (_, PairingMessage::Deny {}) => Err(PairingError::Denied),
            _ => Err(PairingError::Protocol("a message out of order")),
        }
    }
}
