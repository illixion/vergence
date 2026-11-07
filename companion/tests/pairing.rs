//! The two state machines against each other, in memory and over TCP, plus the
//! store, the keyring and knock windows.

use std::net::TcpListener;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use vergence_companion::credential::proof;
use vergence_companion::discovery::{is_valid_knock, knock_tag, minute_of};
use vergence_companion::net::{Decision, pair_with_host, serve_pairing};
use vergence_companion::{
    CredentialStore, DeviceConfig, DeviceEvent, DevicePairing, ExpectedDevice, HostConfig,
    HostEvent, HostOutcome, HostPairing, Identify, JsonFileStore, Keyring, Namespace, PairingError,
    PairingMessage,
};

fn ns() -> Namespace {
    Namespace::new("vergence", "Vergence", "vergence", 4869, "Vergence")
}

fn host_config() -> HostConfig {
    let mut config = HostConfig::new(ns(), "HOST-1", "Studio");
    config.addresses = vec!["192.0.2.10".into()];
    config.default_grants = vec!["play".into()];
    config
}

fn device_config() -> DeviceConfig {
    DeviceConfig {
        namespace: ns(),
        device_id: "HS-1".into(),
        device_name: "Vision Pro".into(),
        host_id: Some("HOST-1".into()),
        requested: vec!["work".into(), "play".into()],
        ssh_public_key: Some("ssh-ed25519 AAAA x".into()),
    }
}

/// Runs the exchange up to the approval prompt; returns both sides and the codes.
fn to_approval(host: &mut HostPairing, device: &mut DevicePairing) -> (String, String) {
    assert_eq!(host.receive(&device.take_outgoing()).unwrap(), None);
    let events = device.receive(&host.take_outgoing()).unwrap();
    let [DeviceEvent::Confirm { code: device_code }] = events.as_slice() else {
        panic!("{events:?}")
    };
    let Some(HostEvent::ApprovalNeeded(request)) = host.receive(&device.take_outgoing()).unwrap()
    else {
        panic!("no approval")
    };
    assert_eq!(request.device_id, "HS-1");
    assert_eq!(request.device_name, "Vision Pro");
    assert_eq!(request.requested, vec!["play", "work"]);
    // No acceptance predicate: the offered key is dropped before the user sees it.
    assert_eq!(request.ssh_public_key, None);
    (device_code.clone(), request.code)
}

#[test]
fn in_memory_pairing_hands_the_device_its_grant() {
    let mut host = HostPairing::new(host_config());
    let mut device = DevicePairing::new(device_config());
    let (device_code, host_code) = to_approval(&mut host, &mut device);
    assert_eq!(device_code, host_code);
    assert_eq!(host_code.len(), 6);

    let record = host
        .approve(vec!["work".into(), "play".into(), "work".into()])
        .unwrap();
    assert_eq!(record.device_id, "HS-1");
    assert_eq!(record.grants, vec!["play", "work"]);
    assert!(host.approve(vec![]).is_err(), "approving twice");
    assert_eq!(host.outcome(), Some(&HostOutcome::Paired(record.clone())));

    let events = device.receive(&host.take_outgoing()).unwrap();
    let [DeviceEvent::Paired(grant)] = events.as_slice() else {
        panic!("{events:?}")
    };
    assert_eq!(grant.token, record.token);
    assert_eq!(grant.mac_id, "HOST-1");
    assert_eq!(grant.mac_name, "Studio");
    assert_eq!(grant.addresses, vec!["192.0.2.10"]);
    assert_eq!(
        grant.grants.as_deref(),
        Some(&["play".to_string(), "work".to_string()][..])
    );
}

#[test]
fn an_accepted_public_key_reaches_the_prompt() {
    let mut config = host_config();
    config.accepts_public_key = Some(Arc::new(|key: &str| key.starts_with("ssh-ed25519 ")));
    let mut host = HostPairing::new(config);
    let mut device = DevicePairing::new(device_config());
    host.receive(&device.take_outgoing()).unwrap();
    device.receive(&host.take_outgoing()).unwrap();
    let Some(HostEvent::ApprovalNeeded(request)) = host.receive(&device.take_outgoing()).unwrap()
    else {
        panic!()
    };
    assert_eq!(
        request.ssh_public_key.as_deref(),
        Some("ssh-ed25519 AAAA x")
    );
}

#[test]
fn declining_denies_the_device() {
    let mut host = HostPairing::new(host_config());
    let mut device = DevicePairing::new(device_config());
    to_approval(&mut host, &mut device);
    host.decline();
    assert_eq!(host.outcome(), Some(&HostOutcome::Declined));
    assert!(matches!(
        device.receive(&host.take_outgoing()),
        Err(PairingError::Denied)
    ));
}

#[test]
fn a_failed_enrolment_is_retracted_into_a_deny() {
    let mut host = HostPairing::new(host_config());
    let mut device = DevicePairing::new(device_config());
    to_approval(&mut host, &mut device);
    host.approve(vec!["play".into()]).unwrap();
    host.retract();
    assert_eq!(host.outcome(), Some(&HostOutcome::Abandoned));
    assert!(matches!(
        device.receive(&host.take_outgoing()),
        Err(PairingError::Denied)
    ));
}

#[test]
fn a_listener_for_one_device_refuses_another_and_names_it_from_the_knock() {
    let mut config = host_config();
    config.expected_device = Some(ExpectedDevice {
        id: "HS-2".into(),
        name: "Quest".into(),
    });
    let mut host = HostPairing::new(config.clone());
    let mut device = DevicePairing::new(device_config());
    assert!(matches!(
        host.receive(&device.take_outgoing()),
        Err(PairingError::Protocol(_))
    ));
    assert_eq!(host.outcome(), Some(&HostOutcome::Abandoned));

    config.expected_device = Some(ExpectedDevice {
        id: "HS-1".into(),
        name: "From the knock".into(),
    });
    let mut host = HostPairing::new(config);
    let mut device = DevicePairing::new(device_config());
    host.receive(&device.take_outgoing()).unwrap();
    device.receive(&host.take_outgoing()).unwrap();
    let Some(HostEvent::ApprovalNeeded(request)) = host.receive(&device.take_outgoing()).unwrap()
    else {
        panic!()
    };
    assert_eq!(request.device_name, "From the knock");
}

#[test]
fn a_reveal_that_breaks_its_commitment_is_abandoned() {
    let mut host = HostPairing::new(host_config());
    let mut device = DevicePairing::new(device_config());
    host.receive(&device.take_outgoing()).unwrap();
    host.take_outgoing();
    let forged = PairingMessage::Reveal {
        public_key: vec![9; 32],
        nonce: vec![9; 32],
    }
    .framed();
    assert!(matches!(
        host.receive(&forged),
        Err(PairingError::Protocol(_))
    ));
    assert_eq!(host.outcome(), Some(&HostOutcome::Abandoned));
    assert!(host.take_outgoing().is_empty());
}

#[test]
fn a_device_rejects_another_host() {
    let mut host = HostPairing::new(HostConfig::new(ns(), "HOST-2", "Other"));
    let mut device = DevicePairing::new(device_config());
    host.receive(&device.take_outgoing()).unwrap();
    assert!(matches!(
        device.receive(&host.take_outgoing()),
        Err(PairingError::Protocol(_))
    ));
}

#[test]
fn byte_at_a_time_delivery() {
    let mut host = HostPairing::new(host_config());
    let mut device = DevicePairing::new(device_config());
    let mut event = None;
    for byte in device.take_outgoing() {
        assert_eq!(host.receive(&[byte]).unwrap(), None);
    }
    let mut events = Vec::new();
    for byte in host.take_outgoing() {
        events.extend(device.receive(&[byte]).unwrap());
    }
    assert_eq!(events.len(), 1);
    for byte in device.take_outgoing() {
        if let Some(e) = host.receive(&[byte]).unwrap() {
            event = Some(e);
        }
    }
    assert!(matches!(event, Some(HostEvent::ApprovalNeeded(_))));
}

#[test]
fn over_tcp_with_the_store_and_keyring() {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(JsonFileStore::new(
        dir.path().join("pairing").join("devices.json"),
    ));
    let host_id = store.host_id().unwrap();
    assert_eq!(store.host_id().unwrap(), host_id, "stable host id");
    let keyring = Keyring::from_records(ns(), &store.devices().unwrap());
    assert!(keyring.credentials().is_empty());

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let mut config = HostConfig::new(ns(), host_id.clone(), "Studio");
    config.addresses = vec!["127.0.0.1".into()];

    let host_store = store.clone();
    let host_keyring = keyring.clone();
    let host = thread::spawn(move || {
        let (stream, _) = listener.accept().unwrap();
        serve_pairing(
            stream,
            HostPairing::new(config),
            |request, watch| {
                assert!(!watch.is_lost());
                Decision::Pair(request.requested.clone())
            },
            |record| {
                host_store.put(record.clone())?;
                host_keyring.replace_from_records(&host_store.devices()?);
                Ok(())
            },
        )
    });

    let mut config = device_config();
    config.host_id = None;
    let stream = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
    let mut seen = None;
    let grant = pair_with_host(
        stream,
        DevicePairing::new(config),
        Duration::from_secs(10),
        |code| seen = Some(code.to_string()),
    )
    .unwrap();
    let outcome = host.join().unwrap().unwrap();
    let HostOutcome::Paired(record) = outcome else {
        panic!("{outcome:?}")
    };
    assert_eq!(seen.unwrap().len(), 6);
    assert_eq!(grant.token, record.token);
    assert_eq!(grant.mac_id, host_id);

    // Stored at 0600, and the keyring now knows the device.
    let stored = store.device("HS-1").unwrap().unwrap();
    assert_eq!(stored, record);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(store.path())
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o600);
    }
    assert!(keyring.allows(Some("HS-1"), "play") && keyring.allows(Some("HS-1"), "work"));
    assert!(!keyring.allows(Some("HS-1"), "admin"));
    let binding = [5u8; 32];
    let identify = Identify::make(&ns(), &stored.credential(), &binding);
    assert_eq!(
        keyring.verify(&identify, &binding).unwrap().credential,
        stored.credential()
    );
    assert!(keyring.verify(&identify, &[6u8; 32]).is_none());

    // Re-pairing replaces; removing forgets.
    let mut again = stored.clone();
    again.token = "new".into();
    store.put(again).unwrap();
    assert_eq!(store.devices().unwrap().len(), 1);
    assert!(store.remove("HS-1").unwrap());
    assert!(!store.remove("HS-1").unwrap());
}

#[test]
fn a_device_that_hangs_up_during_approval_is_noticed() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let host = thread::spawn(move || {
        let (stream, _) = listener.accept().unwrap();
        serve_pairing(
            stream,
            HostPairing::new(host_config()),
            |_, watch| {
                for _ in 0..200 {
                    if watch.is_lost() {
                        return Decision::Abandon;
                    }
                    thread::sleep(Duration::from_millis(10));
                }
                Decision::Pair(vec!["play".into()])
            },
            |_| panic!("must not enroll"),
        )
    });
    use std::io::{Read, Write};
    let mut stream = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
    let mut device = DevicePairing::new(device_config());
    stream.write_all(&device.take_outgoing()).unwrap();
    let mut buf = [0u8; 1024];
    loop {
        let n = stream.read(&mut buf).unwrap();
        if !device.receive(&buf[..n]).unwrap().is_empty() {
            break;
        }
    }
    stream.write_all(&device.take_outgoing()).unwrap();
    drop(stream);
    assert_eq!(host.join().unwrap().unwrap(), HostOutcome::Abandoned);
}

#[test]
fn knock_tags_hold_for_a_minute_either_side_and_bind_everything() {
    let ns = ns();
    let now = UNIX_EPOCH + Duration::from_secs(1_800_000_000);
    let minute = minute_of(now);
    for offset in [-1, 0, 1] {
        let tag = knock_tag(&ns, "tok", "mac", "hs", minute + offset);
        assert!(is_valid_knock(&ns, &tag, "tok", "mac", "hs", now));
    }
    let stale = knock_tag(&ns, "tok", "mac", "hs", minute - 2);
    assert!(!is_valid_knock(&ns, &stale, "tok", "mac", "hs", now));
    let tag = knock_tag(&ns, "tok", "mac", "hs", minute_of(SystemTime::now()));
    let now = SystemTime::now();
    assert!(!is_valid_knock(&ns, &tag, "other", "mac", "hs", now));
    assert!(!is_valid_knock(&ns, &tag, "tok", "mac2", "hs", now));
    assert!(!is_valid_knock(&ns, &tag, "tok", "mac", "hs2", now));
    assert!(!is_valid_knock(
        &Namespace::longwave(),
        &tag,
        "tok",
        "mac",
        "hs",
        now
    ));
    assert!(tag.len() < 32);
    assert_eq!(ns.knock_service_type(), "_vergence-knock._tcp");
}

#[test]
fn keyring_checks_the_claimed_device_only() {
    use vergence_companion::{Credential, KeyringEntry};
    let a = Credential::new("token-a", Some("A".into()));
    let b = Credential::new("token-b", Some("B".into()));
    let keyring = Keyring::new(
        ns(),
        vec![
            KeyringEntry {
                credential: a.clone(),
                grants: ["play".to_string()].into(),
            },
            KeyringEntry {
                credential: b.clone(),
                grants: ["work".to_string()].into(),
            },
        ],
    );
    let binding = [9u8; 32];
    let proof_a = proof(&ns(), "token-a", &binding).to_vec();
    let claim = |id: &str| Identify {
        device_id: Some(id.into()),
        proof: proof_a.clone(),
    };
    assert_eq!(keyring.verify(&claim("A"), &binding).unwrap().credential, a);
    assert!(keyring.verify(&claim("B"), &binding).is_none());
    assert!(keyring.verify(&claim("A"), &[8u8; 32]).is_none());
    assert!(keyring.allows(Some("A"), "play") && !keyring.allows(Some("A"), "work"));
    assert_eq!(
        keyring
            .entries_with("work")
            .iter()
            .map(|e| &e.credential)
            .collect::<Vec<_>>(),
        vec![&b]
    );
}

#[test]
fn tokens_are_url_safe_and_fresh() {
    let t = vergence_companion::Credential::generate_token();
    assert_eq!(t.len(), 43);
    assert!(
        t.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    );
    assert_ne!(t, vergence_companion::Credential::generate_token());
}
