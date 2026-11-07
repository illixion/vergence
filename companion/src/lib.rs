//! RAVECompanion pairing, ported from RAVESDK (Swift) against `longwave-golden.json`.
//!
//! A device (a headset, a worker) pairs once with a host (Longwave's
//! Companion; here Vergence's coordinator) by numeric comparison, gets a token
//! minted for it alone, and from then on reaches the host over TLS-PSK channels
//! keyed from the token, each connection proving which device it is. The wire
//! format is Longwave's, byte for byte.
//!
//! - [`namespace`]: service types and key-derivation labels per app.
//! - [`exchange`], [`message`]: the six-digit exchange and its framed JSON.
//! - [`host`], [`device`]: sans-IO state machines for each side; [`net`]
//!   drives them over a blocking `TcpStream`.
//! - [`credential`], [`keyring`], [`store`]: tokens, PSKs, proofs, grants.
//! - [`discovery`]: knock tags; [`link`]: the address link.
//! - `tls` (feature `tls`, on by default): the TLS 1.2 PSK channel on OpenSSL.

pub mod credential;
pub mod device;
pub mod discovery;
pub mod error;
pub mod exchange;
pub mod host;
pub mod json;
pub mod keyring;
pub mod link;
pub mod message;
pub mod namespace;
pub mod net;
pub mod store;
#[cfg(feature = "tls")]
pub mod tls;

pub use credential::{Channel, Credential, Identify, exporter_label};
pub use device::{DeviceConfig, DeviceEvent, DevicePairing};
pub use error::PairingError;
pub use exchange::{Agreement, PairingExchange};
pub use host::{ApprovalRequest, ExpectedDevice, HostConfig, HostEvent, HostOutcome, HostPairing};
pub use keyring::{Keyring, KeyringEntry};
pub use link::PairLink;
pub use message::{PairingGrant, PairingMessage};
pub use namespace::Namespace;
pub use store::{CredentialStore, DeviceRecord, JsonFileStore};
