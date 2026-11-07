//! The TLS 1.2 PSK channel between the Rust server and client.
#![cfg(feature = "tls")]

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread;

use vergence_companion::credential::exporter_label;
use vergence_companion::message::{read_frame, write_frame};
use vergence_companion::tls::{PskClient, PskServer, identify};
use vergence_companion::{Channel, Credential, Identify, Keyring, KeyringEntry, Namespace};

fn ns() -> Namespace {
    Namespace::new("vergence", "Vergence", "vergence", 4869, "Vergence")
}

fn channel() -> Channel {
    Channel::new("VergenceWork/v1", "Vergence-Work-PSK-v1")
}

fn keyring() -> Keyring {
    Keyring::new(
        ns(),
        vec![
            KeyringEntry {
                credential: Credential::new("token-a", Some("A".into())),
                grants: ["work".to_string()].into(),
            },
            KeyringEntry {
                credential: Credential::new("token-b", Some("B".into())),
                grants: ["play".to_string()].into(),
            },
            KeyringEntry {
                credential: Credential::new("shared", None),
                grants: ["play".to_string()].into(),
            },
        ],
    )
}

/// One connection: the server accepts, reads an identify frame, verifies it
/// and answers with the device it proved (or "refused").
fn serve_once(
    server: PskServer,
    keyring: Keyring,
) -> (u16, thread::JoinHandle<Result<String, String>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let handle = thread::spawn(move || {
        let (stream, _) = listener.accept().unwrap();
        let mut session = server.accept(stream).map_err(|e| e.to_string())?;
        let label = exporter_label("Vergence", "Work");
        let body = read_frame(&mut session.stream).map_err(|e| e.to_string())?;
        let claim = Identify::decode(&body).ok_or("bad identify")?;
        let answer = match session.verify(&claim, &label, &keyring) {
            Some(entry) => entry
                .credential
                .device_id
                .unwrap_or_else(|| "shared".into()),
            None => "refused".into(),
        };
        write_frame(&mut session.stream, answer.as_bytes()).map_err(|e| e.to_string())?;
        Ok(answer)
    });
    (port, handle)
}

fn connect(
    port: u16,
    credential: &Credential,
    claim_as: Option<Option<&str>>,
) -> Result<String, String> {
    let client = PskClient::new(&channel(), credential).unwrap();
    let mut stream = client
        .connect(TcpStream::connect(("127.0.0.1", port)).unwrap())
        .map_err(|e| e.to_string())?;
    assert_eq!(stream.ssl().version_str(), "TLSv1.2");
    assert_eq!(
        stream.ssl().current_cipher().unwrap().name(),
        "PSK-AES128-GCM-SHA256"
    );
    let label = exporter_label("Vergence", "Work");
    let mut message = identify(&ns(), credential, &stream, &label).unwrap();
    if let Some(claim) = claim_as {
        message.device_id = claim.map(String::from);
    }
    write_frame(&mut stream, &message.encoded()).unwrap();
    let answer = read_frame(&mut stream).map_err(|e| e.to_string())?;
    let mut rest = Vec::new();
    let _ = stream.read_to_end(&mut rest);
    let _ = stream.flush();
    Ok(String::from_utf8(answer).unwrap())
}

#[test]
fn a_paired_device_connects_and_proves_who_it_is() {
    let keyring = keyring();
    let (port, server) = serve_once(
        PskServer::with_keyring(channel(), keyring.clone()).unwrap(),
        keyring,
    );
    let answer = connect(port, &Credential::new("token-a", Some("A".into())), None).unwrap();
    assert_eq!(answer, "A");
    assert_eq!(server.join().unwrap().unwrap(), "A");
}

#[test]
fn the_shared_token_uses_the_bare_identity() {
    let keyring = keyring();
    let (port, server) = serve_once(
        PskServer::with_keyring(channel(), keyring.clone()).unwrap(),
        keyring,
    );
    assert_eq!(
        connect(port, &Credential::new("shared", None), None).unwrap(),
        "shared"
    );
    assert_eq!(server.join().unwrap().unwrap(), "shared");
}

#[test]
fn a_device_cannot_claim_another_devices_grants() {
    let keyring = keyring();
    let (port, server) = serve_once(
        PskServer::with_keyring(channel(), keyring.clone()).unwrap(),
        keyring,
    );
    let answer = connect(
        port,
        &Credential::new("token-a", Some("A".into())),
        Some(Some("B")),
    )
    .unwrap();
    assert_eq!(answer, "refused");
    server.join().unwrap().unwrap();
}

#[test]
fn an_unknown_or_wrong_key_fails_the_handshake() {
    for credential in [
        Credential::new("token-a", Some("Z".into())),
        Credential::new("wrong", Some("A".into())),
    ] {
        let keyring = keyring();
        let (port, server) = serve_once(
            PskServer::with_keyring(channel(), keyring.clone()).unwrap(),
            keyring,
        );
        assert!(connect(port, &credential, None).is_err());
        assert!(server.join().unwrap().is_err());
    }
}

#[test]
fn a_revoked_device_is_refused_at_once() {
    let keyring = keyring();
    let server = PskServer::with_keyring(channel(), keyring.clone()).unwrap();
    keyring.replace(vec![]);
    let (port, handle) = serve_once(server, keyring);
    assert!(connect(port, &Credential::new("token-a", Some("A".into())), None).is_err());
    assert!(handle.join().unwrap().is_err());
}

#[test]
fn another_channels_key_does_not_open_this_one() {
    let keyring = keyring();
    let other = Channel::new("VergenceWork/v1", "Vergence-Play-PSK-v1");
    let (port, server) = serve_once(
        PskServer::with_keyring(other, keyring.clone()).unwrap(),
        keyring,
    );
    assert!(connect(port, &Credential::new("token-a", Some("A".into())), None).is_err());
    assert!(server.join().unwrap().is_err());
}
