//! The Rust host for the interop check (`companion/interop/run.sh`): accepts
//! one pairing on `127.0.0.1:<pair-port>`, approves what the device asks for,
//! then accepts one TLS-PSK connection on `<tls-port>` and verifies its
//! identify message.
//!
//!     cargo run -p vergence-companion --example interop_host -- <pair-port> <tls-port> <store.json>

use std::net::TcpListener;
use std::process::exit;

use vergence_companion::message::{read_frame, write_frame};
use vergence_companion::net::{Decision, serve_pairing};
use vergence_companion::tls::PskServer;
use vergence_companion::{
    Channel, CredentialStore, HostConfig, HostOutcome, HostPairing, Identify, JsonFileStore,
    Keyring, Namespace,
};

fn fail(reason: impl std::fmt::Display) -> ! {
    println!("FAIL {reason}");
    exit(1)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let [_, pair_port, tls_port, store_path] = args.as_slice() else {
        fail("usage: interop_host <pair-port> <tls-port> <store.json>")
    };
    let pair_port: u16 = pair_port.parse().unwrap();
    let namespace = Namespace::new(
        "vergence-interop",
        "VergenceInterop",
        "vergence-interop",
        pair_port,
        "Rust interop host",
    );
    let channel = Channel::new("VergenceInterop/v1", "Vergence-Interop-PSK-v1");
    let label = "EXPORTER-VergenceInterop-Work-Device-v1";

    let store = JsonFileStore::new(store_path);
    let keyring = Keyring::from_records(namespace.clone(), &store.devices().unwrap());
    let mut config = HostConfig::new(namespace.clone(), store.host_id().unwrap(), "Rust Host");
    config.addresses = vec!["127.0.0.1".into()];

    let pairing = TcpListener::bind(("127.0.0.1", pair_port)).unwrap();
    let tls = TcpListener::bind(("127.0.0.1", tls_port.parse::<u16>().unwrap())).unwrap();
    println!("LISTENING");
    let (stream, _) = pairing.accept().unwrap();
    drop(pairing);
    let outcome = serve_pairing(
        stream,
        HostPairing::new(config),
        |request, _| {
            println!("CODE {}", request.code);
            // A person takes a moment; the Swift harness polls the client's state.
            std::thread::sleep(std::time::Duration::from_millis(500));
            Decision::Pair(request.requested.clone())
        },
        |record| {
            store.put(record.clone())?;
            keyring.replace_from_records(&store.devices()?);
            Ok(())
        },
    )
    .unwrap_or_else(|e| fail(e));
    let HostOutcome::Paired(record) = outcome else {
        fail(format!("{outcome:?}"))
    };
    println!("PAIRED {} grants={}", record.name, record.grants.join(","));

    let server = PskServer::with_keyring(channel, keyring.clone()).unwrap();
    let (stream, _) = tls.accept().unwrap();
    let mut session = server.accept(stream).unwrap_or_else(|e| fail(e));
    let body = read_frame(&mut session.stream).unwrap_or_else(|e| fail(e));
    let identify = Identify::decode(&body).unwrap_or_else(|| fail("bad identify"));
    let entry = session.verify(&identify, label, &keyring);
    let reply = match &entry {
        Some(e) => format!(
            "ok {}",
            e.credential.device_id.as_deref().unwrap_or("shared")
        ),
        None => "refused".into(),
    };
    write_frame(&mut session.stream, reply.as_bytes()).unwrap();
    let _ = session.stream.shutdown();
    match entry {
        Some(e) => println!(
            "OK verified {} on {}",
            e.credential.device_id.unwrap_or_default(),
            session
                .stream
                .ssl()
                .current_cipher()
                .map(|c| c.name())
                .unwrap_or("?")
        ),
        None => fail("identify refused"),
    }
}
