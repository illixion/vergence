//! JSON the way Swift's `JSONEncoder` writes it: `Data` as padded standard
//! base64, nil optionals left out, and `/` escaped as `\/`. Key order isn't
//! part of the format (Swift's own encoder doesn't fix it); ours follows the
//! struct's field order.

use serde::Serialize;

/// Serialises like `JSONEncoder` with its default options.
pub fn to_swift_json<T: Serialize + ?Sized>(value: &T) -> Vec<u8> {
    let plain = serde_json::to_vec(value).expect("serialising plain data cannot fail");
    // A `/` can only occur inside a JSON string, so escaping every one of them
    // is exactly what Swift does.
    let mut out = Vec::with_capacity(plain.len() + 8);
    for byte in plain {
        if byte == b'/' {
            out.extend_from_slice(b"\\/");
        } else {
            out.push(byte);
        }
    }
    out
}

/// `#[serde(with = "data")]` for a `Vec<u8>` Swift encodes as `Data`.
pub(crate) mod data {
    use base64::Engine;
    use base64::engine::general_purpose::STANDARD;
    use serde::{Deserialize, Deserializer, Serializer, de::Error};

    pub fn serialize<S: Serializer>(bytes: &[u8], serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&STANDARD.encode(bytes))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<u8>, D::Error> {
        let text = String::deserialize(deserializer)?;
        STANDARD.decode(text.as_bytes()).map_err(D::Error::custom)
    }
}
