//! `vergenced`: the analysis coordinator, headless.
//!
//! ```text
//! vergenced serve --config FILE
//! vergenced token add NAME --grants play,work,admin --config FILE   prints the new token once
//! vergenced token list|remove NAME --config FILE
//! vergenced pair [--grants play,work] [--port 4869] --config FILE   pair a headset or worker here
//! vergenced devices list|remove ID --config FILE
//! ```
//!
//! `serve` answers plain HTTP on `listen` (bearer tokens) and, for paired devices, the same API
//! over TLS-PSK on `pairedListen` (`vergence_coordinator::paired`). `pair` runs beside it: it
//! opens the pairing port until one device pairs, shows the six digits to compare, and asks.

use std::io::{BufRead, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use vergence_companion::net::{Decision, serve_pairing};
use vergence_companion::{CredentialStore, HostConfig, HostOutcome, HostPairing, JsonFileStore, PairLink};

use vergence_coordinator::api::{self, AppState};
use vergence_coordinator::auth::{Grant, Tokens};
use vergence_coordinator::config::Config;
use vergence_coordinator::paired::{self, PairedDevices};
use vergence_coordinator::store::Store;

fn usage() -> ExitCode {
    eprintln!(
        "usage:\n  vergenced serve --config FILE\n  vergenced token add NAME --grants play,work,admin --config FILE\n  vergenced token list --config FILE\n  vergenced token remove NAME --config FILE\n  vergenced pair [--grants play,work] [--port 4869] --config FILE\n  vergenced devices list --config FILE\n  vergenced devices remove ID --config FILE"
    );
    ExitCode::from(2)
}

fn option(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(config_path) = option(&args, "--config").map(PathBuf::from) else { return usage() };
    let config = match Config::load(&config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("vergenced: {e}");
            return ExitCode::FAILURE;
        }
    };
    match args.first().map(String::as_str) {
        Some("serve") => serve(config),
        Some("token") => token(&args[1..], &config),
        Some("pair") => pair(&args[1..], &config),
        Some("devices") => devices(&args[1..], &config),
        _ => usage(),
    }
}

fn token(args: &[String], config: &Config) -> ExitCode {
    let mut tokens = match Tokens::load(&config.tokens_path()) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("vergenced: {e}");
            return ExitCode::FAILURE;
        }
    };
    match (args.first().map(String::as_str), args.get(1)) {
        (Some("add"), Some(name)) if !name.starts_with("--") => {
            let grants: Option<Vec<Grant>> =
                option(args, "--grants").unwrap_or_else(|| "play".into()).split(',').map(Grant::parse).collect();
            let Some(grants) = grants else {
                eprintln!("vergenced: grants are play, work and admin");
                return ExitCode::from(2);
            };
            match tokens.add(name, grants) {
                Ok(token) => {
                    println!("{token}");
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("vergenced: {e}");
                    ExitCode::FAILURE
                }
            }
        }
        (Some("list"), _) => {
            for entry in tokens.entries() {
                println!("{}\t{:?}", entry.name, entry.grants);
            }
            ExitCode::SUCCESS
        }
        (Some("remove"), Some(name)) => match tokens.remove(name) {
            Ok(true) => ExitCode::SUCCESS,
            Ok(false) => {
                eprintln!("vergenced: no token named {name}");
                ExitCode::FAILURE
            }
            Err(e) => {
                eprintln!("vergenced: {e}");
                ExitCode::FAILURE
            }
        },
        _ => usage(),
    }
}

fn serve(config: Config) -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "vergenced=info,vergence_coordinator=info".into()),
        )
        .init();
    let runtime = tokio::runtime::Runtime::new().expect("a tokio runtime");
    match runtime.block_on(run(config)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            tracing::error!(error = %e, "vergenced stopped");
            ExitCode::FAILURE
        }
    }
}

async fn run(config: Config) -> std::io::Result<()> {
    let store = Store::open(&config.store)?;
    let tokens = Tokens::load(&config.tokens_path())?;
    if tokens.entries().is_empty() {
        tracing::warn!("no tokens yet: add one with `vergenced token add NAME --grants play,work`");
    }
    let state = AppState::new(store, config.sources()?, config.workloads.clone(), tokens)?;
    {
        let titles = state.queue.lock().unwrap().titles().count();
        tracing::info!(titles, "results read back");
    }
    if let Some(minutes) = config.scan_every_minutes {
        let state = state.clone();
        tokio::spawn(async move {
            loop {
                let summary = api::scan_now(&state).await;
                tracing::info!(%summary, "library scanned");
                tokio::time::sleep(Duration::from_secs(minutes.max(1) * 60)).await;
            }
        });
    }
    {
        let state = state.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(5)).await;
                state.queue.lock().unwrap().expire(std::time::Instant::now());
            }
        });
    }
    let router = api::router(state.clone());
    if let Some(paired_listen) = &config.paired_listen {
        let devices = PairedDevices::open(config.devices_path());
        state.set_paired(devices.clone());
        match tokio::net::TcpListener::bind(paired_listen).await {
            Ok(listener) => {
                tracing::info!(listen = %paired_listen, devices = devices.records().len(), "serving paired devices (TLS-PSK)");
                let router = router.clone();
                tokio::spawn(async move {
                    if let Err(e) = paired::serve(listener, router, devices, std::future::pending()).await {
                        tracing::error!(error = %e, "the paired listener stopped");
                    }
                });
            }
            Err(e) => tracing::error!(listen = %paired_listen, error = %e, "couldn't open the paired port"),
        }
    }
    let listener = tokio::net::TcpListener::bind(&config.listen).await?;
    tracing::info!(listen = %config.listen, "serving");
    // With each connection's address, so loopback callers can be told apart (`api::Caller::secure`).
    axum::serve(listener, router.into_make_service_with_connect_info::<std::net::SocketAddr>())
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
}

/// How long `pair` keeps the pairing port open.
const PAIR_OPEN_FOR: Duration = Duration::from_secs(300);

fn computer_name() -> String {
    for var in ["VERGENCE_HOST_NAME", "COMPUTERNAME", "HOSTNAME"] {
        if let Ok(name) = std::env::var(var)
            && !name.trim().is_empty()
        {
            return name.trim().to_string();
        }
    }
    std::process::Command::new("hostname")
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().split('.').next().unwrap_or_default().to_string())
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| "vergenced".into())
}

/// Opens the pairing port until one device pairs (or five minutes pass), asking on the terminal.
fn pair(args: &[String], config: &Config) -> ExitCode {
    let grants: Vec<String> = option(args, "--grants").unwrap_or_else(|| "play".into()).split(',').map(|g| g.trim().to_string()).collect();
    if grants.iter().any(|g| Grant::parse(g).is_none()) {
        eprintln!("vergenced: grants are play, work and admin");
        return ExitCode::from(2);
    }
    let namespace = paired::namespace();
    let port = match option(args, "--port").map(|p| p.parse::<u16>()) {
        None => namespace.default_pair_port,
        Some(Ok(p)) => p,
        Some(Err(_)) => return usage(),
    };
    let store = JsonFileStore::new(config.devices_path());
    let host_id = match store.host_id() {
        Ok(id) => id,
        Err(e) => {
            eprintln!("vergenced: {}: {e}", config.devices_path().display());
            return ExitCode::FAILURE;
        }
    };
    let listener = match TcpListener::bind(("0.0.0.0", port)).and_then(|l| l.set_nonblocking(true).map(|()| l)) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("vergenced: couldn't open the pairing port {port}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let name = computer_name();
    let addresses = paired::lan_addresses();
    let mut host = HostConfig::new(namespace.clone(), host_id.clone(), name.clone());
    host.addresses = addresses.clone();
    host.default_grants = vec!["play".into()];
    println!("Pairing is open on port {port} for five minutes.");
    if addresses.is_empty() {
        println!("Pair from the device with this computer's address.");
    } else {
        println!("Pair from the device with this computer's address: {}", addresses.join(", "));
        println!("Link: {}", PairLink::new(&namespace, host_id, name, Some(port), addresses).url());
    }
    let _ = std::io::stdout().flush();
    let until = Instant::now() + PAIR_OPEN_FOR;
    while Instant::now() < until {
        let stream = match listener.accept() {
            Ok((stream, _)) => stream,
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(200));
                continue;
            }
            Err(e) => {
                eprintln!("vergenced: pairing accept failed: {e}");
                std::thread::sleep(Duration::from_millis(500));
                continue;
            }
        };
        let _ = stream.set_nonblocking(false);
        let enrolled = std::cell::RefCell::new(None::<String>);
        let outcome = serve_pairing(
            stream,
            HostPairing::new(host.clone()),
            |request, peer| {
                println!(
                    "\n{} ({}) asks to pair. Check that it shows {} {}.",
                    request.device_name,
                    request.device_id,
                    &request.code[..3.min(request.code.len())],
                    &request.code[3.min(request.code.len())..]
                );
                print!("Pair it, allowing {}? [y/N] ", grants.join(", "));
                let _ = std::io::stdout().flush();
                let mut answer = String::new();
                if std::io::stdin().lock().read_line(&mut answer).is_err() || peer.is_lost() {
                    return Decision::Abandon;
                }
                if answer.trim().eq_ignore_ascii_case("y") || answer.trim().eq_ignore_ascii_case("yes") {
                    Decision::Pair(grants.clone())
                } else {
                    Decision::Decline
                }
            },
            |record| {
                store.put(record.clone())?;
                *enrolled.borrow_mut() = Some(record.device_id.clone());
                Ok(())
            },
        );
        match outcome {
            Ok(HostOutcome::Paired(record)) => {
                println!("Paired with {} ({}), allowed {}.", record.name, record.device_id, record.grants.join(", "));
                return ExitCode::SUCCESS;
            }
            Ok(HostOutcome::Declined) => println!("Declined."),
            Ok(HostOutcome::Abandoned) => println!("The pairing didn't finish."),
            Err(e) => {
                // The device can't have received its grant: forget what was stored for it.
                if let Some(id) = enrolled.borrow().as_deref() {
                    let _ = store.remove(id);
                }
                eprintln!("vergenced: the pairing failed: {e}");
            }
        }
    }
    eprintln!("vergenced: nothing paired; the pairing port is closed");
    ExitCode::FAILURE
}

fn devices(args: &[String], config: &Config) -> ExitCode {
    let store = JsonFileStore::new(config.devices_path());
    match (args.first().map(String::as_str), args.get(1)) {
        (Some("list"), _) => match store.devices() {
            Ok(devices) => {
                for d in devices {
                    println!("{}\t{}\t{}", d.device_id, d.name, d.grants.join(","));
                }
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("vergenced: {e}");
                ExitCode::FAILURE
            }
        },
        (Some("remove"), Some(id)) if !id.starts_with("--") => match store.remove(id) {
            Ok(true) => ExitCode::SUCCESS,
            Ok(false) => {
                eprintln!("vergenced: no device {id}");
                ExitCode::FAILURE
            }
            Err(e) => {
                eprintln!("vergenced: {e}");
                ExitCode::FAILURE
            }
        },
        _ => usage(),
    }
}
