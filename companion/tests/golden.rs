//! Every value in `longwave-golden.json` (generated 2026-11-01 from Longwave's
//! `Shared/CompanionPairing.swift` and `Shared/CompanionDevices.swift`, the same
//! file RAVESDK's `RAVECompanionTests` pins), reproduced by the Rust port.

use serde_json::Value;
use vergence_companion::credential::{Channel, Credential, Identify, proof, validates};
use vergence_companion::discovery::knock_tag;
use vergence_companion::exchange::PairingExchange;
use vergence_companion::link::PairLink;
use vergence_companion::message::{PairingGrant, PairingMessage};
use vergence_companion::namespace::Namespace;

struct Golden(Value);

impl Golden {
    fn load() -> Self {
        let text = include_str!("fixtures/longwave-golden.json");
        Self(serde_json::from_str(text).unwrap())
    }
    fn s(&self, key: &str) -> &str {
        self.0[key].as_str().unwrap()
    }
}

fn to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn device() -> PairingExchange {
    PairingExchange::new([0x11; 32], [0x33; 32])
}

fn host() -> PairingExchange {
    PairingExchange::new([0x22; 32], [0x44; 32])
}

fn expected_grant() -> PairingGrant {
    PairingGrant {
        token: "tok-123".into(),
        mac_id: "MAC-1".into(),
        mac_name: "Studio".into(),
        addresses: vec!["192.0.2.10".into()],
        grants: Some(vec!["audio".into(), "screen".into()]),
    }
}

#[test]
fn knock_tag_matches() {
    let g = Golden::load();
    assert_eq!(
        knock_tag(
            &Namespace::longwave(),
            "tok-123",
            "MAC-1",
            "HS-1",
            29_000_000
        ),
        g.s("knockTag")
    );
}

#[test]
fn exchange_keys_commitment_code_key_and_sealed_grant() {
    let g = Golden::load();
    let ns = Namespace::longwave();
    let (device, host) = (device(), host());
    assert_eq!(to_hex(&device.public_key()), g.s("hsPublicKey"));
    assert_eq!(to_hex(&host.public_key()), g.s("macPublicKey"));
    let requested = vec!["audio".to_string(), "screen".to_string()];
    let commitment = PairingExchange::commitment(
        &ns, &device.public_key(), &device.nonce(), "device-1", "Vision Pro", Some(&requested), None,
    );
    assert_eq!(to_hex(&commitment), g.s("commitment"));

    for (mine, theirs) in [(&device, host.public_key()), (&host, device.public_key())] {
        let agreement = PairingExchange::agree(
            &ns,
            mine,
            &theirs,
            &commitment,
            &device.public_key(),
            &device.nonce(),
            &host.public_key(),
            &host.nonce(),
            "MAC-1", "Studio", "device-1", "Vision Pro", Some(&requested), None,
        )
        .expect("both ends agree");
        assert_eq!(agreement.code, g.s("code"));
        assert_eq!(to_hex(&agreement.key), g.s("key"));

        let sealed = PairingExchange::seal(&expected_grant(), &agreement.key);
        assert_eq!(PairingExchange::open(&sealed, &agreement.key).unwrap(), expected_grant());
        let plaintext = PairingExchange::open_bytes(&sealed, &agreement.key).unwrap();
        let ours: Value =
            serde_json::from_slice(&vergence_companion::json::to_swift_json(&expected_grant()))
                .unwrap();
        assert_eq!(ours, serde_json::from_slice::<Value>(&plaintext).unwrap());
    }
}

#[test]
fn a_wrong_reveal_or_key_is_refused() {
    let (device, host) = (device(), host());
    let ns = Namespace::longwave();
    let requested = vec!["audio".to_string(), "screen".to_string()];
    let commitment = PairingExchange::commitment(
        &ns, &device.public_key(), &device.nonce(), "device-1", "Vision Pro", Some(&requested), None,
    );
    let wrong_nonce = PairingExchange::agree(
        &ns,
        &host,
        &device.public_key(),
        &commitment,
        &device.public_key(),
        &[0x34; 32],
        &host.public_key(),
        &host.nonce(),
        "MAC-1", "Studio", "device-1", "Vision Pro", Some(&requested), None,
    );
    assert!(wrong_nonce.is_none());
    // A low-order point (all zeros) gives no contributory secret.
    let zero = [0u8; 32];
    let low = PairingExchange::agree(
        &ns,
        &host,
        &zero,
        &PairingExchange::commitment(&ns, &zero, &device.nonce(), "device-1", "Vision Pro", Some(&requested), None),
        &zero,
        &device.nonce(),
        &host.public_key(),
        &host.nonce(),
        "MAC-1", "Studio", "device-1", "Vision Pro", Some(&requested), None,
    );
    assert!(low.is_none());
}

#[test]
fn pairing_v2_rejects_tampered_enrollment_and_host_metadata() {
    let ns = Namespace::longwave();
    let (device, host) = (device(), host());
    let requested = vec!["audio".to_string(), "screen".to_string()];
    let commitment = PairingExchange::commitment(
        &ns, &device.public_key(), &device.nonce(), "device-1", "Vision Pro", Some(&requested), None,
    );
    let agree = |device_id: &str, grants: &[String], host_id: &str| PairingExchange::agree(
        &ns, &device, &host.public_key(), &commitment,
        &device.public_key(), &device.nonce(), &host.public_key(), &host.nonce(),
        host_id, "Studio", device_id, "Vision Pro", Some(grants), None,
    );
    assert!(agree("other-device", &requested, "MAC-1").is_none());
    assert!(agree("device-1", &["audio".into(), "projects".into(), "screen".into()], "MAC-1").is_none());
    assert!(agree("device-1", &requested, "spoofed-MAC").is_some_and(|a| a.code != agree("device-1", &requested, "MAC-1").unwrap().code));
    let with_ssh = PairingExchange::commitment(
        &ns, &device.public_key(), &device.nonce(), "device-1", "Vision Pro", Some(&requested), Some("ssh-ed25519 AAAA attacker"),
    );
    assert_ne!(with_ssh, commitment);
}

#[test]
fn proof_psk_and_identities() {
    let g = Golden::load();
    let ns = Namespace::longwave();
    let p = proof(&ns, "tok-123", &[7; 32]);
    assert_eq!(to_hex(&p), g.s("proof"));
    assert!(validates(&ns, &p, "tok-123", &[7; 32]));
    assert!(!validates(&ns, &p, "tok-124", &[7; 32]));
    assert!(!validates(&ns, &p, "tok-123", &[8; 32]));

    let audio = Channel::new("LongwaveAudio/v5", "Longwave-Audio-PSK-v5");
    assert_eq!(to_hex(&audio.derive("tok-123")), g.s("pskAudio"));
    let device = Credential::new("t", Some("HS-1".into()));
    let legacy = Credential::new("t", None);
    assert_eq!(
        device.psk_identity("LongwaveAudio/v5"),
        g.s("pskIdentityDevice").as_bytes()
    );
    assert_eq!(
        legacy.psk_identity("LongwaveAudio/v5"),
        g.s("pskIdentityLegacy").as_bytes()
    );
    assert_eq!(
        audio.parse_identity(g.s("pskIdentityDevice").as_bytes()),
        Some(Some("HS-1".into()))
    );
    assert_eq!(
        audio.parse_identity(g.s("pskIdentityLegacy").as_bytes()),
        Some(None)
    );
    assert_eq!(audio.parse_identity(b"LongwaveAudio/v5/"), None);
    assert_eq!(audio.parse_identity(b"LongwaveAudio/v55/HS-1"), None);
    assert_eq!(audio.parse_identity(b"LongwaveVideo/v5/HS-1"), None);

    let identify = Identify {
        device_id: Some("HS-1".into()),
        proof: vec![1, 2, 3],
    };
    assert_eq!(identify.encoded(), g.s("identifyJSON").as_bytes());
    assert_eq!(
        Identify::decode(g.s("identifyJSON").as_bytes()),
        Some(identify)
    );
    // A shared token's identify leaves the id out, as Swift does.
    assert_eq!(
        Identify {
            device_id: None,
            proof: vec![1, 2, 3]
        }
        .encoded(),
        br#"{"proof":"AQID"}"#
    );
}

#[test]
fn pair_link() {
    let g = Golden::load();
    let ns = Namespace::longwave();
    let link = PairLink::new(
        &ns,
        "MAC-1",
        "Studio Mac",
        Some(4859),
        vec!["192.0.2.10".into(), "100.64.0.3".into()],
    );
    // The golden file is Swift-escaped JSON; the URL itself has plain slashes.
    assert_eq!(link.url(), g.s("pairLink"));
    assert_eq!(PairLink::parse(&ns, g.s("pairLink")), Some(link));
}

#[test]
fn pair_link_vetting() {
    let ns = Namespace::longwave();
    let ok = "longwave://pair?mac=M&addr=10.0.0.1";
    assert_eq!(PairLink::parse(&ns, ok).unwrap().name, "Mac");
    assert!(PairLink::parse(&ns, "LONGWAVE://PAIR?mac=M&addr=10.0.0.1").is_some());
    assert!(PairLink::parse(&ns, "longwave://pair?mac=M&addr=10.0.0.1&port=22").is_none());
    assert!(PairLink::parse(&ns, "longwave://pair?mac=M&addr=example.com").is_none());
    assert!(PairLink::parse(&ns, "longwave://pair?addr=10.0.0.1").is_none());
    assert!(PairLink::parse(&ns, "other://pair?mac=M&addr=10.0.0.1").is_none());
    assert!(PairLink::parse(&ns, "longwave://open?mac=M&addr=10.0.0.1").is_none());
    let v6 = PairLink::parse(&ns, "longwave://pair?mac=M&addr=fe80::1%25en0&addr=::1").unwrap();
    assert_eq!(v6.addresses, vec!["fe80::1%en0", "::1"]);
    // Characters URLComponents escapes in a value, and the round trip.
    let odd = PairLink::new(&ns, "a&b=c", "Mac #1 é/?", None, vec!["10.0.0.1".into()]);
    assert_eq!(
        odd.url(),
        "longwave://pair?mac=a%26b%3Dc&name=Mac%20%231%20%C3%A9/?&port=4859&addr=10.0.0.1"
    );
    assert_eq!(PairLink::parse(&ns, &odd.url()), Some(odd));
}

#[test]
fn every_longwave_message_decodes_to_the_known_values_and_reencodes() {
    let (device, host) = (device(), host());
    let namespace = Namespace::longwave();
    let requested = vec!["audio".to_string(), "screen".to_string()];
    let commitment = PairingExchange::commitment(
        &namespace, &device.public_key(), &device.nonce(), "device-1", "Vision Pro", Some(&requested), None,
    );
    let agreement = PairingExchange::agree(
        &namespace, &device, &host.public_key(), &commitment,
        &device.public_key(), &device.nonce(), &host.public_key(), &host.nonce(),
        "MAC-1", "Studio", "device-1", "Vision Pro", Some(&requested), None,
    ).unwrap();
    let expected = vec![
        PairingMessage::Commit {
            headset_id: "device-1".into(),
            headset_name: "Vision Pro".into(),
            commitment: commitment.to_vec(),
            requested: Some(requested.clone()),
            ssh_public_key: None,
        },
        PairingMessage::MacHello {
            mac_id: "MAC-1".into(),
            mac_name: "Studio".into(),
            public_key: host.public_key().to_vec(),
            nonce: host.nonce().to_vec(),
        },
        PairingMessage::Reveal {
            public_key: device.public_key().to_vec(),
            nonce: device.nonce().to_vec(),
        },
        PairingMessage::Accept {
            sealed_grant: PairingExchange::seal(&expected_grant(), &agreement.key),
        },
        PairingMessage::Deny {},
    ];
    for (index, want) in expected.iter().enumerate() {
        let frame = want.framed();
        let mut buffer = frame.clone();
        let message = PairingMessage::take(&mut buffer).unwrap().unwrap();
        assert!(buffer.is_empty(), "message {index}");
        assert_eq!(&message, want, "message {index}");
        assert_eq!(message.framed(), frame, "message {index}");
    }
}

#[test]
fn framing_splits_across_reads_and_refuses_oversized_frames() {
    let messages = vec![
        PairingMessage::Commit {
            headset_id: "hs".into(),
            headset_name: "Vision Pro".into(),
            commitment: vec![1, 2, 3],
            requested: Some(vec!["projects".into(), "screen".into()]),
            ssh_public_key: None,
        },
        PairingMessage::MacHello {
            mac_id: "mac".into(),
            mac_name: "Studio Mac".into(),
            public_key: vec![0; 32],
            nonce: vec![0; 32],
        },
        PairingMessage::Deny {},
    ];
    let stream: Vec<u8> = messages.iter().flat_map(|m| m.framed()).collect();
    let mut buffer = Vec::new();
    let mut received = Vec::new();
    for byte in stream {
        buffer.push(byte);
        while let Some(m) = PairingMessage::take(&mut buffer).unwrap() {
            received.push(m);
        }
    }
    assert_eq!(received, messages);
    let mut oversized = vec![0xFF, 0xFF, 0x00, 0x00];
    assert!(PairingMessage::take(&mut oversized).is_err());
}

#[test]
fn display_code() {
    assert_eq!(PairingExchange::display("123456"), "123 456");
    assert_eq!(PairingExchange::display("12345"), "12345");
}
