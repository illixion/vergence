//! Per-device credentials, channel key schedules and connection proofs
//! (`RAVECompanionCredential`, `RAVECompanionChannel`, `RAVECompanionIdentify`).
//!
//! Every pairing mints a token for that one device. Each channel derives its
//! own TLS-PSK from the token with its own salt; the PSK identity names the
//! device, so a listener can hold every enrolled device's key at once. Each
//! connection then identifies itself: the device id and an HMAC under its token
//! over a TLS exporter value, so a device can neither claim another device's
//! grants nor replay the proof on another connection.
//!
//! The channel is TLS 1.2 with `TLS_PSK_WITH_AES_128_GCM_SHA256` (0x00A8):
//! Apple's Network.framework silently discards an external PSK under TLS 1.3,
//! so every peer pins 1.2 and this suite.

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use hkdf::Hkdf;
use hmac::{Hmac, Mac};
use rand_core::{OsRng, RngCore};
use serde::{Deserialize, Serialize};
use sha2::Sha256;

use crate::json::{data, to_swift_json};
use crate::namespace::Namespace;

/// The secret a device presents to a host. `device_id` is `None` for a host's
/// older shared token (Longwave's, before per-device pairing), which used a
/// fixed PSK identity.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Credential {
    pub token: String,
    pub device_id: Option<String>,
}

impl std::fmt::Debug for Credential {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Credential")
            .field("device_id", &self.device_id)
            .finish_non_exhaustive()
    }
}

impl Credential {
    pub fn new(token: impl Into<String>, device_id: Option<String>) -> Self {
        Self {
            token: token.into(),
            device_id,
        }
    }

    pub fn is_legacy(&self) -> bool {
        self.device_id.is_none()
    }

    /// `<base>` for a shared token, `<base>/<device id>` for a device token.
    pub fn psk_identity(&self, base: &str) -> Vec<u8> {
        match &self.device_id {
            Some(id) => format!("{base}/{id}").into_bytes(),
            None => base.as_bytes().to_vec(),
        }
    }

    /// A fresh 256-bit URL-safe token.
    pub fn generate_token() -> String {
        let mut bytes = [0u8; 32];
        OsRng.fill_bytes(&mut bytes);
        URL_SAFE_NO_PAD.encode(bytes)
    }
}

/// One channel's TLS-PSK key schedule: its PSK identity base and HKDF salt,
/// which keep each channel's keys independent though all come from the token.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Channel {
    pub identity_base: String,
    pub salt: Vec<u8>,
    pub info: Vec<u8>,
}

/// `TLS_PSK_WITH_AES_128_GCM_SHA256`.
pub const PSK_CIPHERSUITE: u16 = 0x00A8;
/// The same suite by its OpenSSL name.
pub const PSK_CIPHER_NAME: &str = "PSK-AES128-GCM-SHA256";

impl Channel {
    /// `info` is `"psk"`, as every Longwave channel uses.
    pub fn new(identity_base: impl Into<String>, salt: &str) -> Self {
        Self::with_info(identity_base, salt, "psk")
    }

    pub fn with_info(identity_base: impl Into<String>, salt: &str, info: &str) -> Self {
        Self {
            identity_base: identity_base.into(),
            salt: salt.as_bytes().to_vec(),
            info: info.as_bytes().to_vec(),
        }
    }

    /// HKDF-SHA256(token, salt, info), 32 bytes.
    pub fn derive(&self, token: &str) -> [u8; 32] {
        let mut key = [0u8; 32];
        Hkdf::<Sha256>::new(Some(&self.salt), token.as_bytes())
            .expand(&self.info, &mut key)
            .expect("32 bytes is a valid HKDF-SHA256 length");
        key
    }

    pub fn identity(&self, credential: &Credential) -> Vec<u8> {
        credential.psk_identity(&self.identity_base)
    }

    /// Which credential a presented PSK identity names: `Some(None)` for the
    /// bare base (the shared token), `Some(Some(id))` for a device, `None` for
    /// an identity that names nothing on this channel.
    pub fn parse_identity(&self, identity: &[u8]) -> Option<Option<String>> {
        let text = std::str::from_utf8(identity).ok()?;
        if text == self.identity_base {
            return Some(None);
        }
        let id = text.strip_prefix(&self.identity_base)?.strip_prefix('/')?;
        (!id.is_empty() && id.chars().count() <= 64).then(|| Some(id.to_string()))
    }
}

/// A connection's first message: who it is, bound to this TLS session.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Identify {
    /// `None`: a shared token. Swift writes it out only when present.
    #[serde(rename = "deviceID", default, skip_serializing_if = "Option::is_none")]
    pub device_id: Option<String>,
    #[serde(with = "data")]
    pub proof: Vec<u8>,
}

impl Identify {
    /// The proof for `credential` over a session's exporter value.
    pub fn make(namespace: &Namespace, credential: &Credential, binding: &[u8]) -> Self {
        Self {
            device_id: credential.device_id.clone(),
            proof: proof(namespace, &credential.token, binding).to_vec(),
        }
    }

    pub fn encoded(&self) -> Vec<u8> {
        to_swift_json(self)
    }

    pub fn decode(bytes: &[u8]) -> Option<Self> {
        serde_json::from_slice(bytes).ok()
    }
}

/// The exporter label of a channel's identify proof:
/// `EXPORTER-<App>-<Channel>-Device-v1`, so a proof made for one channel is
/// useless on another. The exporter context is **empty** (not absent), which
/// is what Apple's `sec_protocol_metadata_create_secret` exports.
pub fn exporter_label(app: &str, channel: &str) -> String {
    format!("EXPORTER-{app}-{channel}-Device-v1")
}

fn proof_mac(namespace: &Namespace, token: &str) -> Hmac<Sha256> {
    let mut key = [0u8; 32];
    Hkdf::<Sha256>::new(Some(&namespace.proof_salt()), token.as_bytes())
        .expand(&[], &mut key)
        .expect("32 bytes is a valid HKDF-SHA256 length");
    <Hmac<Sha256> as Mac>::new_from_slice(&key).expect("HMAC takes any key length")
}

/// HMAC-SHA256 of the binding under HKDF(token, `<prefix>-Device-Proof-v1`).
pub fn proof(namespace: &Namespace, token: &str, binding: &[u8]) -> [u8; 32] {
    let mut mac = proof_mac(namespace, token);
    mac.update(binding);
    mac.finalize().into_bytes().into()
}

/// Constant-time check of a proof.
pub fn validates(namespace: &Namespace, proof: &[u8], token: &str, binding: &[u8]) -> bool {
    let mut mac = proof_mac(namespace, token);
    mac.update(binding);
    mac.verify_slice(proof).is_ok()
}
