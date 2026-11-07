# vergence-companion

RAVECompanion's pairing protocol in Rust, for Vergence's analysis coordinator. Headsets and
workers pair *to* the coordinator once, by six-digit numeric comparison, get a token minted for
them alone, and from then on reach it over TLS-PSK channels keyed from that token, each
connection proving which device it is. Ported from RAVESDK's `Sources/RAVECompanion` (Swift);
the wire format is Longwave's, byte for byte, and `tests/golden.rs` reproduces every value in
`tests/fixtures/longwave-golden.json` (the vectors RAVESDK's own tests pin; test keys only).

## Using it as the host (the coordinator's side)

```rust
use vergence_companion::{*, net::{serve_pairing, Decision}, tls::PskServer};

let ns = Namespace::new("vergence", "Vergence", "vergence", 4869, "Vergence");
let store = JsonFileStore::new(dir.join("devices.json"));      // 0600, atomic rewrite
let keyring = Keyring::from_records(ns.clone(), &store.devices()?);

// Pairing: a plain TCP listener on ns.default_pair_port, open only while pairing is allowed.
let mut config = HostConfig::new(ns.clone(), store.host_id()?, "Studio");
config.addresses = lan_addresses();                             // wired first
let (stream, _) = pair_listener.accept()?;
let outcome = serve_pairing(stream, HostPairing::new(config),
    |request, peer| {                                           // show PairingExchange::display(&request.code)
        ask_user(request, peer)                                 // poll peer.is_lost() to close the prompt
    },                                                          // -> Decision::Pair(grants) / Decline / Abandon
    |record| { store.put(record.clone())?; keyring.replace_from_records(&store.devices()?); Ok(()) },
)?;                                                             // HostOutcome::Paired(DeviceRecord) / Declined / Abandoned

// Channel: TLS 1.2, TLS_PSK_WITH_AES_128_GCM_SHA256, the PSK picked by the identity.
let channel = Channel::new("VergenceWork/v1", "Vergence-Work-PSK-v1");
let server = PskServer::with_keyring(channel, keyring.clone())?;
let mut session = server.accept(tcp_stream)?;                   // session.device_id: who the PSK named
let identify = Identify::decode(&message::read_frame(&mut session.stream)?).unwrap();
let entry = session.verify(&identify, &exporter_label("Vergence", "Work"), &keyring);
// entry.grants: the raw grant names the user allowed ("play", "work", "admin").
```

`HostPairing` is a sans-IO state machine (`receive` bytes, `take_outgoing`, `approve` /
`decline` / `retract` / `peer_closed`), so the same exchange runs on any socket, async or not;
`net::serve_pairing` is its blocking driver. `DevicePairing` and `net::pair_with_host` are the
device's side (a Rust worker), and `tls::PskClient` plus `tls::identify` its channel.

Grants travel as raw strings and the crate keeps them so; the caller maps them to its own
(the coordinator: `play`, `work`, `admin`). A Swift peer decodes them into its own enum and
rejects a commit naming one it doesn't know, so a device should ask only for what the host offers.

## Modules

| Module | Swift original | What |
| --- | --- | --- |
| `namespace` | `RAVECompanionNamespace` | service types, key-derivation labels, link scheme, pairing port |
| `exchange` | `RAVEPairingExchange` | X25519, commitment, transcript, six-digit code, HKDF key, ChaChaPoly sealed grant |
| `message` | `RAVEPairingMessage`, `RAVEPairingGrant` | the framed JSON messages (4-byte LE length, 16 KiB cap) |
| `host`, `device` | `RAVEPairingServer`, `RAVEPairingClient` | sans-IO state machines |
| `net` | `RAVEDirectPairingListener` (in part) | blocking drivers, dialling an address link |
| `credential` | `RAVECompanionCredential`, `RAVECompanionChannel`, `RAVECompanionIdentify` | tokens, PSK and identity derivation, identify proofs |
| `keyring` | `RAVECompanionKeyring` | enrolled devices, proof verification, grant checks |
| `store` | (the host app's own) | `CredentialStore` trait, `JsonFileStore` |
| `discovery` | `RAVECompanionDiscovery` | knock tags and TXT keys |
| `link` | `RAVECompanionPairLink` | the `<scheme>://pair?…` address link |
| `tls` | `RAVECompanionChannel.configure` | OpenSSL TLS 1.2 PSK server and client, exporter binding |

JSON is written the way Swift's `JSONEncoder` writes it (`Data` as padded base64, nil optionals
left out, `/` as `\/`), so frames match Longwave's apart from key order, which isn't part of the
format.

## OpenSSL

rustls has no external-PSK support, so the `tls` feature (on by default) uses the `openssl`
crate. Network.framework drops an external PSK under TLS 1.3, so both versions are pinned to 1.2
and the only suite is `PSK-AES128-GCM-SHA256` (0x00A8), as the Swift and BouncyCastle peers do.
The identify binding is RFC 5705 keying material with an **empty** context (not none), which is
what Apple's `sec_protocol_metadata_create_secret` exports.

- **macOS:** links Homebrew's `openssl@3`, which `openssl-sys` finds by itself
  (`brew install openssl@3`). Set `OPENSSL_DIR=$(brew --prefix openssl@3)` if it doesn't.
- **Linux:** the distribution's `libssl-dev` / `openssl-devel`.
- **Anywhere, without a system OpenSSL:** `--features vendored-openssl` builds it from source
  (needs a C compiler and Perl). The Windows desktop app will likely want this.
- `--no-default-features` leaves TLS out: the protocol, store and drivers need no OpenSSL.

## Tests and the interop check

`cargo test -p vergence-companion` runs the golden vectors, both state machines against each
other (in memory, byte at a time and over TCP with the store and keyring), and the TLS channel
(Rust server and client: the right device connects, a device claiming another's id is refused,
an unknown, revoked or wrong-channel key fails the handshake).

`interop/run.sh` (macOS, RAVESDK beside vergence) runs RAVESDK's own Swift code against the
Rust port on loopback, both ways:

1. Swift `RAVEPairingClient` (address link) pairs with the Rust `serve_pairing`; the Swift
   client then opens `RAVECompanionChannel.tcpParameters` to the Rust `PskServer` and sends
   `RAVECompanionIdentify.make`, which the Rust keyring verifies.
2. The Rust device pairs with Swift's `RAVEPairingServer` behind `RAVEDirectPairingListener`,
   then opens a Rust `PskClient` channel to a Network.framework PSK listener, which verifies the
   Rust identify with `RAVECompanionKeyring`.

Each direction passes when both sides showed the same code and both ended OK.

## Not done yet

- **Bonjour.** The knock-tag maths and the TXT keys are here; advertising the host's record,
  browsing for knocks and the per-device `_<stem>-pair._tcp` listener are not. Until then devices
  pair through the fixed pairing port, from an address link (`PairLink`) the desktop app shows or
  sends. `mdns-sd` is the likely crate.
- **Async pairing.** The pairing drivers are blocking; the coordinator runs them on a thread. The
  TLS channel is async there already: `PskServer::ssl` and `PskServer::matched` let it run the
  handshake through `tokio-openssl` (`coordinator/src/paired.rs`).
- **DTLS** (Longwave's UDP channels) isn't ported; the coordinator only needs TCP.
- **Knock-gated ports** (`RAVECompanionPresence`, `RAVECompanionPortListener`'s rebind retry):
  host-app behaviour, left to the coordinator.
