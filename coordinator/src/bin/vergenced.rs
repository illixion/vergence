//! `vergenced`: the analysis coordinator, headless.
//!
//! ```text
//! vergenced serve --config FILE
//! vergenced token add NAME --grants play,work,admin --config FILE   prints the new token once
//! vergenced token list|remove NAME --config FILE
//! ```

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use vergence_coordinator::api::{self, AppState};
use vergence_coordinator::auth::{Grant, Tokens};
use vergence_coordinator::config::Config;
use vergence_coordinator::store::Store;

fn usage() -> ExitCode {
    eprintln!(
        "usage:\n  vergenced serve --config FILE\n  vergenced token add NAME --grants play,work,admin --config FILE\n  vergenced token list --config FILE\n  vergenced token remove NAME --config FILE"
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
    let listener = tokio::net::TcpListener::bind(&config.listen).await?;
    tracing::info!(listen = %config.listen, "serving");
    axum::serve(listener, api::router(state))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
}
