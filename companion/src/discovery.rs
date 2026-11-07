//! Knock tags and the TXT keys of the three Bonjour services
//! (`RAVECompanionDiscovery`).
//!
//! A paired device advertises a knock carrying a tag only a holder of its token
//! can make, rolling every minute; seeing a valid one, the host opens its
//! ports. A knock can be copied or forged by anything on the LAN, and that is
//! fine: it only decides whether a port is listening. Getting in still takes
//! the token.

use std::time::{SystemTime, UNIX_EPOCH};

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use hkdf::Hkdf;
use hmac::{Hmac, Mac};
use sha2::Sha256;
use subtle::ConstantTimeEq;

use crate::namespace::Namespace;

/// TXT keys. Short, because a record travels in every mDNS answer.
pub mod key {
    pub const VERSION: &str = "v";
    pub const HOST_ID: &str = "id";
    pub const ADDRESSES: &str = "addr";
    pub const OPEN: &str = "open";
    pub const DEVICE_ID: &str = "hs";
    pub const TAG: &str = "t";
    pub const PAIR_REQUEST: &str = "pair";
    pub const NAME: &str = "n";
    /// "windows" on a Windows host's record; absent on a Mac's.
    pub const OS: &str = "os";
    /// The host's fixed pairing port. A host that lists it is paired by
    /// dialling it rather than knocking for a listener.
    pub const PAIR_PORT: &str = "pp";
}

pub const VERSION: &str = "1";

/// A tag is valid in its own minute and the one either side.
pub const TAG_WINDOW_MINUTES: i64 = 1;

pub fn minute_of(time: SystemTime) -> i64 {
    match time.duration_since(UNIX_EPOCH) {
        Ok(after) => (after.as_secs() / 60) as i64,
        Err(before) => -(before.duration().as_secs().div_ceil(60) as i64),
    }
}

/// HMAC of the host, the device and the minute under a key derived from the
/// token, truncated to 12 bytes and base64url'd: it gates a listener, it
/// isn't the authentication.
pub fn knock_tag(
    namespace: &Namespace,
    token: &str,
    host_id: &str,
    device_id: &str,
    minute: i64,
) -> String {
    let mut key = [0u8; 32];
    Hkdf::<Sha256>::new(Some(&namespace.knock_salt()), token.as_bytes())
        .expand(b"knock", &mut key)
        .expect("32 bytes is a valid HKDF-SHA256 length");
    let mut mac = <Hmac<Sha256> as Mac>::new_from_slice(&key).expect("HMAC takes any key length");
    mac.update(format!("{host_id}|{device_id}|{minute}").as_bytes());
    URL_SAFE_NO_PAD.encode(&mac.finalize().into_bytes()[..12])
}

pub fn is_valid_knock(
    namespace: &Namespace,
    tag: &str,
    token: &str,
    host_id: &str,
    device_id: &str,
    now: SystemTime,
) -> bool {
    let current = minute_of(now);
    (-TAG_WINDOW_MINUTES..=TAG_WINDOW_MINUTES).any(|offset| {
        let expected = knock_tag(namespace, token, host_id, device_id, current + offset);
        bool::from(expected.as_bytes().ct_eq(tag.as_bytes()))
    })
}
