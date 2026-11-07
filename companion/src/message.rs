//! The pairing messages and their framing (`RAVEPairingMessage`,
//! `RAVEPairingGrant`).
//!
//! Each message is a 4-byte little-endian length and a JSON body, on a plain
//! TCP connection: everything that needs protecting is protected by the
//! exchange. Field names are the wire format (`macID`, `headsetID`…) and stay
//! as Longwave named them.

use serde::{Deserialize, Serialize};

use crate::json::{data, to_swift_json};

/// What the host hands a device once the user has approved it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PairingGrant {
    pub token: String,
    /// The host's id.
    #[serde(rename = "macID")]
    pub mac_id: String,
    #[serde(rename = "macName")]
    pub mac_name: String,
    /// LAN addresses, wired first, as a starting host for the connection.
    pub addresses: Vec<String>,
    /// Raw values of the grants the user allowed this device.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grants: Option<Vec<String>>,
}

/// The messages of a pairing, in order: device `commit`, host `macHello`,
/// device `reveal`, host `accept` or `deny`. Grants travel as raw strings; a
/// Swift peer decodes them into its own enum and rejects a message naming one
/// it doesn't know, so a device should request only what the host offers.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PairingMessage {
    #[serde(rename = "commit")]
    Commit {
        #[serde(rename = "headsetID")]
        headset_id: String,
        #[serde(rename = "headsetName")]
        headset_name: String,
        #[serde(with = "data")]
        commitment: Vec<u8>,
        /// The features the device asks for; the host's dialog starts from them.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        requested: Option<Vec<String>>,
        /// A public key line the host may install if the user allows it.
        #[serde(
            rename = "sshPublicKey",
            default,
            skip_serializing_if = "Option::is_none"
        )]
        ssh_public_key: Option<String>,
    },
    #[serde(rename = "macHello")]
    MacHello {
        #[serde(rename = "macID")]
        mac_id: String,
        #[serde(rename = "macName")]
        mac_name: String,
        #[serde(rename = "publicKey", with = "data")]
        public_key: Vec<u8>,
        #[serde(with = "data")]
        nonce: Vec<u8>,
    },
    #[serde(rename = "reveal")]
    Reveal {
        #[serde(rename = "publicKey", with = "data")]
        public_key: Vec<u8>,
        #[serde(with = "data")]
        nonce: Vec<u8>,
    },
    #[serde(rename = "accept")]
    Accept {
        #[serde(rename = "sealedGrant", with = "data")]
        sealed_grant: Vec<u8>,
    },
    /// `{"deny":{}}` on the wire, as Swift's synthesised `Codable` writes a case
    /// without values.
    #[serde(rename = "deny")]
    Deny {},
}

#[derive(Debug, thiserror::Error)]
pub enum FrameError {
    #[error("a frame of {0} bytes is over the {max} byte limit", max = MAX_FRAME_BYTES)]
    Oversized(usize),
    #[error("a frame isn't a valid message: {0}")]
    Json(#[from] serde_json::Error),
}

/// The largest body a frame may carry.
pub const MAX_FRAME_BYTES: usize = 16 * 1024;

/// A body behind its 4-byte little-endian length.
pub fn frame(body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + body.len());
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(body);
    out
}

/// Pulls one complete frame's body off the front of `buffer`, if it holds one.
pub fn take_frame(buffer: &mut Vec<u8>) -> Result<Option<Vec<u8>>, FrameError> {
    if buffer.len() < 4 {
        return Ok(None);
    }
    let length = u32::from_le_bytes(buffer[..4].try_into().unwrap()) as usize;
    if length > MAX_FRAME_BYTES {
        return Err(FrameError::Oversized(length));
    }
    if buffer.len() < 4 + length {
        return Ok(None);
    }
    let body = buffer[4..4 + length].to_vec();
    buffer.drain(..4 + length);
    Ok(Some(body))
}

impl PairingMessage {
    pub fn to_json(&self) -> Vec<u8> {
        to_swift_json(self)
    }

    pub fn framed(&self) -> Vec<u8> {
        frame(&self.to_json())
    }

    /// Pulls one complete message off the front of `buffer`, if it holds one.
    pub fn take(buffer: &mut Vec<u8>) -> Result<Option<Self>, FrameError> {
        match take_frame(buffer)? {
            Some(body) => Ok(Some(serde_json::from_slice(&body)?)),
            None => Ok(None),
        }
    }

    /// The message's name on the wire, for logs.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Commit { .. } => "commit",
            Self::MacHello { .. } => "macHello",
            Self::Reveal { .. } => "reveal",
            Self::Accept { .. } => "accept",
            Self::Deny {} => "deny",
        }
    }
}

/// Writes one frame (length and body) to a stream: the pairing framing, also
/// a natural carrier for a channel's identify message.
pub fn write_frame<W: std::io::Write>(writer: &mut W, body: &[u8]) -> std::io::Result<()> {
    writer.write_all(&frame(body))?;
    writer.flush()
}

/// Reads one frame's body from a stream.
pub fn read_frame<R: std::io::Read>(reader: &mut R) -> std::io::Result<Vec<u8>> {
    let mut length = [0u8; 4];
    reader.read_exact(&mut length)?;
    let length = u32::from_le_bytes(length) as usize;
    if length > MAX_FRAME_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            FrameError::Oversized(length),
        ));
    }
    let mut body = vec![0u8; length];
    reader.read_exact(&mut body)?;
    Ok(body)
}
