//! The Rust device for the interop check (`companion/interop/run.sh`): pairs
//! with a host on `127.0.0.1:<pair-port>` through an address link, then opens
//! the TLS-PSK channel on `<tls-port>` and identifies itself.
//!
//!     cargo run -p vergence-companion --example interop_device -- <pair-port> <tls-port>

use std::net::TcpStream;
use std::process::exit;
use std::time::Duration;

use vergence_companion::message::{read_frame, write_frame};
use vergence_companion::net::{connect_link, pair_with_host};
use vergence_companion::tls::{PskClient, identify};
use vergence_companion::{Channel, Credential, DeviceConfig, DevicePairing, Namespace, PairLink};

fn fail(reason: impl std::fmt::Display) -> ! {
    println!("FAIL {reason}");
    exit(1)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let [_, pair_port, tls_port] = args.as_slice() else {
        fail("usage: interop_device <pair-port> <tls-port>")
    };
    let pair_port: u16 = pair_port.parse().unwrap();
    let namespace = Namespace::new(
        "vergence-interop",
        "VergenceInterop",
        "vergence-interop",
        pair_port,
        "Swift interop host",
    );
    let channel = Channel::new("VergenceInterop/v1", "Vergence-Interop-PSK-v1");
    let label = "EXPORTER-VergenceInterop-Work-Device-v1";

    let link = PairLink::new(&namespace, "", "Swift", None, vec!["127.0.0.1".into()]);
    let (stream, _) = connect_link(&link).unwrap_or_else(|e| fail(e));
    let pairing = DevicePairing::new(DeviceConfig {
        namespace: namespace.clone(),
        device_id: "rust-device".into(),
        device_name: "Rust harness".into(),
        host_id: None,
        requested: vec!["play".into(), "work".into()],
        ssh_public_key: None,
    });
    let grant = pair_with_host(stream, pairing, Duration::from_secs(30), |code| {
        println!("CODE {code}")
    })
    .unwrap_or_else(|e| fail(e));
    println!(
        "GRANT host={} grants={}",
        grant.mac_id,
        grant.grants.clone().unwrap_or_default().join(",")
    );

    let credential = Credential::new(grant.token, Some("rust-device".into()));
    let client = PskClient::new(&channel, &credential).unwrap();
    // The Swift host opens its channel listener once pairing has finished.
    let tcp = (0..50)
        .find_map(|_| {
            TcpStream::connect(("127.0.0.1", tls_port.parse::<u16>().unwrap()))
                .ok()
                .or_else(|| {
                    std::thread::sleep(Duration::from_millis(100));
                    None
                })
        })
        .unwrap_or_else(|| fail("the channel port never opened"));
    let mut stream = client.connect(tcp).unwrap_or_else(|e| fail(e));
    let message = identify(&namespace, &credential, &stream, label).unwrap();
    write_frame(&mut stream, &message.encoded()).unwrap();
    let reply = read_frame(&mut stream).unwrap_or_else(|e| fail(e));
    let reply = String::from_utf8_lossy(&reply).into_owned();
    if reply.starts_with("ok ") {
        println!("OK channel reply: {reply}");
    } else {
        fail(format!("channel reply: {reply}"))
    }
}
