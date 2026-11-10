//! The pairing exchange (`RAVEPairingExchange`): numeric comparison.
//!
//! The device commits to its key and nonce before it sees the host's and
//! reveals them only after; both ends then show a six-digit code drawn from the
//! whole transcript, and the user approves on the host only if the codes match.
//! Something sitting between the two would have had to fix its keys on each
//! side before learning what the other would contribute, so its two codes match
//! by chance one time in a million. The agreed key then seals the grant on its
//! way to the device.

use chacha20poly1305::aead::{Aead, KeyInit};
use chacha20poly1305::{ChaCha20Poly1305, Key, Nonce};
use hkdf::Hkdf;
use rand_core::{OsRng, RngCore};
use sha2::{Digest, Sha256};
use x25519_dalek::{PublicKey, StaticSecret};

use crate::json::to_swift_json;
use crate::message::PairingGrant;
use crate::namespace::Namespace;

/// One side's ephemeral contribution: an X25519 key and a 32-byte nonce.
#[derive(Clone)]
pub struct PairingExchange {
    secret: StaticSecret,
    nonce: [u8; 32],
}

impl std::fmt::Debug for PairingExchange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PairingExchange")
            .field("public_key", &self.public_key())
            .finish_non_exhaustive()
    }
}

impl Default for PairingExchange {
    fn default() -> Self {
        Self::random()
    }
}

/// What both ends know once the four values are in.
#[derive(Clone, PartialEq, Eq)]
pub struct Agreement {
    /// Six digits, for both screens.
    pub code: String,
    /// The ChaCha20-Poly1305 key the grant is sealed under.
    pub key: [u8; 32],
}

impl std::fmt::Debug for Agreement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Agreement")
            .field("code", &self.code)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SealError {
    #[error("the sealed box is too short")]
    TooShort,
    #[error("the sealed box doesn't open with this key")]
    Authentication,
    #[error("the grant isn't valid JSON: {0}")]
    Json(#[from] serde_json::Error),
}

impl PairingExchange {
    /// A fresh key and nonce from the OS's random source.
    pub fn random() -> Self {
        Self::new(
            StaticSecret::random_from_rng(OsRng).to_bytes(),
            random_nonce(),
        )
    }

    /// From a raw private key (CryptoKit's `rawRepresentation`) and a nonce.
    pub fn new(private_key: [u8; 32], nonce: [u8; 32]) -> Self {
        Self {
            secret: StaticSecret::from(private_key),
            nonce,
        }
    }

    pub fn public_key(&self) -> [u8; 32] {
        PublicKey::from(&self.secret).to_bytes()
    }

    pub fn nonce(&self) -> [u8; 32] {
        self.nonce
    }

    /// The device's opening move: bind all enrollment metadata the host will
    /// use before approval, as well as the ephemeral key and nonce.
    pub fn commitment(
        namespace: &Namespace,
        public_key: &[u8], nonce: &[u8],
        device_id: &str, device_name: &str,
        requested: Option<&[String]>, ssh_public_key: Option<&str>,
    ) -> [u8; 32] {
        let mut transcript = b"RAVECompanionPairCommit-v2".to_vec();
        append_field(&namespace.pair_info(), &mut transcript);
        append_field(public_key, &mut transcript);
        append_field(nonce, &mut transcript);
        append_field(device_id.as_bytes(), &mut transcript);
        append_field(device_name.as_bytes(), &mut transcript);
        let mut grants = requested.unwrap_or_default().to_vec();
        grants.sort();
        grants.dedup();
        append_length(grants.len() as u32, &mut transcript);
        for grant in grants { append_field(grant.as_bytes(), &mut transcript); }
        match ssh_public_key {
            Some(key) => { transcript.push(1); append_field(key.as_bytes(), &mut transcript); }
            None => transcript.push(0),
        }
        Sha256::digest(transcript).into()
    }

    /// Runs on both ends once all four values are known. `None` when the
    /// revealed device values don't match the commitment, or a key is bad.
    #[allow(clippy::too_many_arguments)]
    pub fn agree(
        namespace: &Namespace,
        mine: &PairingExchange,
        their_public_key: &[u8],
        commitment: &[u8],
        device_public_key: &[u8],
        device_nonce: &[u8],
        host_public_key: &[u8],
        host_nonce: &[u8],
        host_id: &str, host_name: &str,
        device_id: &str, device_name: &str,
        requested: Option<&[String]>, ssh_public_key: Option<&str>,
    ) -> Option<Agreement> {
        if Self::commitment(namespace, device_public_key, device_nonce, device_id, device_name,
                            requested, ssh_public_key).as_slice() != commitment {
            return None;
        }
        let theirs: [u8; 32] = their_public_key.try_into().ok()?;
        let shared = mine.secret.diffie_hellman(&PublicKey::from(theirs));
        // CryptoKit refuses a low-order point; an all-zero secret is the sign of one.
        if !shared.was_contributory() {
            return None;
        }
        let mut transcript_input = b"RAVECompanionPairTranscript-v2".to_vec();
        append_field(commitment, &mut transcript_input);
        append_field(host_public_key, &mut transcript_input);
        append_field(host_nonce, &mut transcript_input);
        append_field(device_public_key, &mut transcript_input);
        append_field(device_nonce, &mut transcript_input);
        append_field(host_id.as_bytes(), &mut transcript_input);
        append_field(host_name.as_bytes(), &mut transcript_input);
        let transcript: [u8; 32] = Sha256::digest(transcript_input).into();
        let mut key = [0u8; 32];
        Hkdf::<Sha256>::new(Some(&transcript), shared.as_bytes())
            .expand(&namespace.pair_info(), &mut key)
            .expect("32 bytes is a valid HKDF-SHA256 length");
        let number = u32::from_be_bytes(transcript[..4].try_into().unwrap()) % 1_000_000;
        Some(Agreement {
            code: format!("{number:06}"),
            key,
        })
    }

    /// ChaChaPoly's combined form, as CryptoKit writes it: nonce, ciphertext, tag.
    pub fn seal(grant: &PairingGrant, key: &[u8; 32]) -> Vec<u8> {
        let mut nonce = [0u8; 12];
        OsRng.fill_bytes(&mut nonce);
        Self::seal_with_nonce(&to_swift_json(grant), key, nonce)
    }

    /// Seals raw bytes under a given nonce. Only for tests and vectors: a nonce
    /// must never repeat under one key.
    pub fn seal_with_nonce(plaintext: &[u8], key: &[u8; 32], nonce: [u8; 12]) -> Vec<u8> {
        let cipher = ChaCha20Poly1305::new(Key::from_slice(key));
        let body = cipher
            .encrypt(Nonce::from_slice(&nonce), plaintext)
            .expect("ChaCha20-Poly1305 encryption of a small message cannot fail");
        let mut combined = nonce.to_vec();
        combined.extend_from_slice(&body);
        combined
    }

    /// Opens a combined box to its plaintext bytes.
    pub fn open_bytes(sealed: &[u8], key: &[u8; 32]) -> Result<Vec<u8>, SealError> {
        if sealed.len() < 12 + 16 {
            return Err(SealError::TooShort);
        }
        let cipher = ChaCha20Poly1305::new(Key::from_slice(key));
        cipher
            .decrypt(Nonce::from_slice(&sealed[..12]), &sealed[12..])
            .map_err(|_| SealError::Authentication)
    }

    pub fn open(sealed: &[u8], key: &[u8; 32]) -> Result<PairingGrant, SealError> {
        Ok(serde_json::from_slice(&Self::open_bytes(sealed, key)?)?)
    }

    /// "123456" → "123 456".
    pub fn display(code: &str) -> String {
        if code.chars().count() != 6 {
            return code.to_string();
        }
        let (a, b) = code.split_at(3);
        format!("{a} {b}")
    }
}

fn append_field(value: &[u8], transcript: &mut Vec<u8>) {
    append_length(value.len() as u32, transcript);
    transcript.extend_from_slice(value);
}

fn append_length(length: u32, transcript: &mut Vec<u8>) {
    transcript.extend_from_slice(&length.to_be_bytes());
}

/// 32 random bytes.
pub fn random_nonce() -> [u8; 32] {
    let mut nonce = [0u8; 32];
    OsRng.fill_bytes(&mut nonce);
    nonce
}
