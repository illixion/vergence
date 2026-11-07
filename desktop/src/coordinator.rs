//! The analysis coordinator (`vergence-coordinator`), run in this process: the same axum server
//! `vergenced` runs headless, on the app's tokio runtime, with its store under the app-data
//! folder. Changing a server or the port restarts it; titles and their finished pieces are kept
//! on disk, so a restart only drops leases, which workers claim again.
//!
//! The app's own UI is its admin: it reads the queue and manages tokens through the shared
//! state directly, so it needs no admin token of its own.
//!
//! Paired headsets reach the same API over TLS-PSK on the paired port (8421 by default, every
//! interface, `vergence_coordinator::paired`), keyed from the pairings in `devices.json`; plain
//! HTTP with tokens stays on the coordinator's port.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use vergence_coordinator::api::{self, AppState};
use vergence_coordinator::auth::{Grant, TokenEntry, Tokens};
use vergence_coordinator::config::Config;
use vergence_coordinator::paired::{self, PairedDevices};
use vergence_coordinator::queue::Workload;
use vergence_coordinator::store::Store;

use crate::settings::{Paths, Server, Settings, server_config, write_atomic};

struct Running {
    state: Arc<AppState>,
    listen: String,
    /// None when the port couldn't be opened (the state is still there for tokens and the queue).
    shutdown: Option<oneshot::Sender<()>>,
    server: Option<JoinHandle<()>>,
    /// The paired listener; None when its port couldn't be opened.
    paired: Option<(oneshot::Sender<()>, JoinHandle<()>)>,
    paired_error: Option<String>,
    chores: Vec<JoinHandle<()>>,
}

pub struct CoordinatorHost {
    paths: Paths,
    running: tokio::sync::Mutex<Option<Running>>,
    /// The state, for synchronous readers (commands, the tray).
    state: Mutex<Option<(Arc<AppState>, String)>>,
    error: Mutex<Option<String>>,
    paired: Mutex<(Option<String>, Option<String>)>,
    last_scan: Mutex<Option<Value>>,
}

impl CoordinatorHost {
    pub fn new(paths: Paths) -> CoordinatorHost {
        CoordinatorHost {
            paths,
            running: Default::default(),
            state: Default::default(),
            error: Default::default(),
            paired: Default::default(),
            last_scan: Default::default(),
        }
    }

    fn config(&self, settings: &Settings) -> Config {
        Config {
            listen: settings.listen(),
            store: self.paths.store(),
            tokens_file: Some(self.paths.tokens()),
            jellyfin: server_config(&self.paths, settings, Server::Jellyfin),
            stash: server_config(&self.paths, settings, Server::Stash),
            workloads: Workload::defaults(),
            scan_every_minutes: settings.scan.then(|| u64::from(settings.scan_every_hours.max(1)) * 60),
            paired_listen: Some(settings.paired_listen()),
            devices_file: Some(self.paths.devices()),
        }
    }

    /// Starts the coordinator for these settings, stopping one already running first.
    pub async fn start(self: &Arc<Self>, settings: &Settings) -> Result<(), String> {
        let mut running = self.running.lock().await;
        if let Some(old) = running.take() {
            stop(old).await;
        }
        *self.state.lock().unwrap() = None;
        let config = self.config(settings);
        let result = self.open(&config).await;
        match result {
            Ok(now) => {
                *self.state.lock().unwrap() = Some((now.state.clone(), now.listen.clone()));
                *self.paired.lock().unwrap() = (
                    now.paired.as_ref().and(config.paired_listen.clone()),
                    now.paired_error.clone(),
                );
                *self.error.lock().unwrap() = if now.server.is_some() {
                    None
                } else {
                    Some(format!("Port {} is taken: is vergenced or another copy of Vergence running?", settings.port))
                };
                let serving = now.server.is_some();
                *running = Some(now);
                if serving { Ok(()) } else { Err(self.error.lock().unwrap().clone().unwrap_or_default()) }
            }
            Err(e) => {
                *self.error.lock().unwrap() = Some(e.clone());
                Err(e)
            }
        }
    }

    async fn open(self: &Arc<Self>, config: &Config) -> Result<Running, String> {
        let store = Store::open(&config.store).map_err(|e| format!("The coordinator's store: {e}"))?;
        let tokens = Tokens::load(&config.tokens_path()).map_err(|e| format!("The coordinator's tokens: {e}"))?;
        let sources = config.sources().map_err(|e| format!("A media server's key couldn't be read: {e}"))?;
        let state = AppState::new(store, sources, config.workloads.clone(), tokens).map_err(|e| format!("The coordinator's store: {e}"))?;
        let devices = PairedDevices::open(config.devices_path());
        state.set_paired(devices.clone());
        let mut chores = Vec::new();
        {
            let state = state.clone();
            chores.push(tokio::spawn(async move {
                loop {
                    tokio::time::sleep(Duration::from_secs(5)).await;
                    state.queue.lock().unwrap().expire(Instant::now());
                }
            }));
        }
        if let Some(minutes) = config.scan_every_minutes {
            let state = state.clone();
            let host = self.clone();
            chores.push(tokio::spawn(async move {
                loop {
                    let summary = api::scan_now(&state).await;
                    tracing::info!(%summary, "library scanned");
                    *host.last_scan.lock().unwrap() = Some(summary);
                    tokio::time::sleep(Duration::from_secs(minutes.max(1) * 60)).await;
                }
            }));
        }
        let (server, shutdown) = match tokio::net::TcpListener::bind(&config.listen).await {
            Ok(listener) => {
                let (tx, rx) = oneshot::channel::<()>();
                let router = api::router(state.clone());
                let listen = config.listen.clone();
                let server = tokio::spawn(async move {
                    tracing::info!(%listen, "coordinator serving");
                    let result = axum::serve(listener, router)
                        .with_graceful_shutdown(async {
                            let _ = rx.await;
                        })
                        .await;
                    if let Err(e) = result {
                        tracing::error!(error = %e, "the coordinator stopped");
                    }
                });
                (Some(server), Some(tx))
            }
            Err(e) => {
                tracing::warn!(listen = %config.listen, error = %e, "the coordinator couldn't listen");
                (None, None)
            }
        };
        let (paired, paired_error) = match &config.paired_listen {
            None => (None, None),
            Some(listen) => match tokio::net::TcpListener::bind(listen).await {
                Ok(listener) => {
                    let (tx, rx) = oneshot::channel::<()>();
                    let router = api::router(state.clone());
                    let listen = listen.clone();
                    let task = tokio::spawn(async move {
                        tracing::info!(%listen, "paired channel serving");
                        let stopped = async {
                            let _ = rx.await;
                        };
                        if let Err(e) = paired::serve(listener, router, devices, stopped).await {
                            tracing::error!(error = %e, "the paired channel stopped");
                        }
                    });
                    (Some((tx, task)), None)
                }
                Err(e) => {
                    tracing::warn!(%listen, error = %e, "the paired channel couldn't listen");
                    (None, Some(format!("Paired headsets can't connect: port {} is taken ({e}).", listen.rsplit(':').next().unwrap_or(listen))))
                }
            },
        };
        Ok(Running { state, listen: config.listen.clone(), shutdown, server, paired, paired_error, chores })
    }

    pub async fn shutdown(&self) {
        if let Some(old) = self.running.lock().await.take() {
            stop(old).await;
        }
        *self.state.lock().unwrap() = None;
        *self.paired.lock().unwrap() = (None, None);
    }

    pub fn state(&self) -> Option<Arc<AppState>> {
        self.state.lock().unwrap().as_ref().map(|(state, _)| state.clone())
    }

    fn require(&self) -> Result<Arc<AppState>, String> {
        self.state().ok_or_else(|| self.error.lock().unwrap().clone().unwrap_or_else(|| "The coordinator isn't running.".into()))
    }

    pub fn status(&self) -> Value {
        let state = self.state.lock().unwrap().clone();
        let serving = state.is_some() && self.error.lock().unwrap().is_none();
        json!({
            "running": serving,
            "listen": state.as_ref().map(|(_, listen)| listen.clone()),
            "error": *self.error.lock().unwrap(),
            "titles": state.as_ref().map(|(s, _)| s.queue.lock().unwrap().titles().count()),
            "pairedListen": self.paired.lock().unwrap().0.clone(),
            "pairedError": self.paired.lock().unwrap().1.clone(),
            "lastScan": *self.last_scan.lock().unwrap(),
        })
    }

    /// The queue as the admin view sees it: titles with progress, leases, workers, playheads.
    pub fn snapshot(&self) -> Result<Value, String> {
        let state = self.require()?;
        let mut snapshot = state.queue.lock().unwrap().snapshot(Instant::now());
        if let Some(titles) = snapshot.get_mut("titles").and_then(Value::as_array_mut) {
            for title in titles {
                let bytes = title.get("key").and_then(Value::as_str).map(|key| state.store.bytes(key));
                title["bytes"] = json!(bytes);
            }
        }
        Ok(snapshot)
    }

    pub async fn scan_now(&self) -> Result<Value, String> {
        let state = self.require()?;
        let summary = api::scan_now(&state).await;
        *self.last_scan.lock().unwrap() = Some(summary.clone());
        Ok(summary)
    }

    /// Removes a title and its results.
    pub fn remove_title(&self, key: &str) -> Result<bool, String> {
        let state = self.require()?;
        let removed = state.queue.lock().unwrap().remove(key);
        state.store.remove_title(key).map_err(|e| e.to_string())?;
        Ok(removed)
    }

    pub fn tokens(&self) -> Result<Vec<TokenEntry>, String> {
        match self.state() {
            Some(state) => Ok(state.tokens.read().unwrap().entries().to_vec()),
            None => Tokens::load(&self.paths.tokens()).map(|t| t.entries().to_vec()).map_err(|e| e.to_string()),
        }
    }

    /// Makes a token (replacing one of the same name) and returns it, to be shown once.
    pub fn add_token(&self, name: &str, grants: Vec<Grant>) -> Result<String, String> {
        let name = name.trim();
        if name.is_empty() || name.len() > 64 || name.contains('/') {
            return Err("Give the token a short name, without a slash.".into());
        }
        match self.state() {
            Some(state) => state.tokens.write().unwrap().add(name, grants),
            None => Tokens::load(&self.paths.tokens()).and_then(|mut t| t.add(name, grants)),
        }
        .map_err(|e| e.to_string())
    }

    pub fn remove_token(&self, name: &str) -> Result<bool, String> {
        match self.state() {
            Some(state) => state.tokens.write().unwrap().remove(name),
            None => Tokens::load(&self.paths.tokens()).and_then(|mut t| t.remove(name)),
        }
        .map_err(|e| e.to_string())
    }

    /// Adds a token the coordinator didn't make (a paired device's), by its SHA-256: rewrites
    /// the tokens file in the coordinator's own format and reloads it, holding the tokens lock
    /// so no other change can land in between.
    pub fn put_token_hash(&self, name: &str, sha256_hex: &str, grants: Vec<Grant>) -> Result<(), String> {
        let state = self.state();
        let mut guard = state.as_ref().map(|s| s.tokens.write().unwrap());
        let path = self.paths.tokens();
        let mut entries = Tokens::load(&path).map_err(|e| e.to_string())?.entries().to_vec();
        entries.retain(|e| e.name != name);
        entries.push(TokenEntry { name: name.to_string(), token_sha256: sha256_hex.to_string(), grants });
        let bytes = serde_json::to_vec_pretty(&entries).map_err(|e| e.to_string())?;
        write_atomic(&path, &bytes, true).map_err(|e| e.to_string())?;
        if let Some(tokens) = guard.as_deref_mut() {
            *tokens = Tokens::load(&path).map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}

async fn stop(running: Running) {
    for chore in &running.chores {
        chore.abort();
    }
    if let Some((tx, mut task)) = running.paired {
        let _ = tx.send(());
        if tokio::time::timeout(Duration::from_secs(3), &mut task).await.is_err() {
            task.abort();
        }
    }
    if let Some(tx) = running.shutdown {
        let _ = tx.send(());
    }
    if let Some(mut server) = running.server {
        // A player holding a connection open shouldn't keep the old server alive.
        if tokio::time::timeout(Duration::from_secs(3), &mut server).await.is_err() {
            server.abort();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn free_port() -> u16 {
        std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
    }

    #[tokio::test]
    async fn serves_health_and_keeps_tokens_across_restarts() {
        let dir = tempfile::tempdir().unwrap();
        let host = Arc::new(CoordinatorHost::new(Paths::new(dir.path())));
        let settings = Settings { port: free_port(), paired_port: free_port(), ..Settings::default() };
        host.start(&settings).await.unwrap();
        assert_eq!(host.status()["pairedListen"], settings.paired_listen());
        let health: Value = reqwest::get(format!("{}/v1/health", settings.local_url())).await.unwrap().json().await.unwrap();
        assert_eq!(health["service"], "vergenced");

        let token = host.add_token("phone", vec![Grant::Play]).unwrap();
        // The device's token is known by its hash only.
        host.put_token_hash("headset", &"ab".repeat(32), vec![Grant::Play]).unwrap();
        let state = host.state().unwrap();
        assert_eq!(state.tokens.read().unwrap().check(&token, Grant::Play), Some("phone"));
        let names = |h: &CoordinatorHost| h.tokens().unwrap().into_iter().map(|e| e.name).collect::<Vec<_>>();
        assert_eq!(names(&host), ["phone", "headset"]);

        // The queue answers a play token, and refuses one without a grant for it.
        let client = reqwest::Client::new();
        let queue = client.get(format!("{}/v1/queue", settings.local_url())).bearer_auth(&token).send().await.unwrap();
        assert_eq!(queue.status(), 403);
        assert!(host.snapshot().unwrap()["titles"].as_array().unwrap().is_empty());

        // A restart on another port reads the same tokens back.
        let moved = Settings { port: free_port(), paired_port: free_port(), ..settings.clone() };
        host.start(&moved).await.unwrap();
        assert!(reqwest::get(format!("{}/v1/health", settings.local_url())).await.is_err());
        assert_eq!(names(&host), ["phone", "headset"]);
        assert!(host.remove_token("phone").unwrap());
        assert_eq!(names(&host), ["headset"]);
        host.shutdown().await;
        assert_eq!(names(&host), ["headset"]);
    }

    #[tokio::test]
    async fn a_taken_port_is_reported_and_tokens_still_work() {
        let dir = tempfile::tempdir().unwrap();
        let taken = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let host = Arc::new(CoordinatorHost::new(Paths::new(dir.path())));
        let settings = Settings { port: taken.local_addr().unwrap().port(), paired_port: free_port(), ..Settings::default() };
        let error = host.start(&settings).await.unwrap_err();
        assert!(error.contains("taken"), "{error}");
        assert_eq!(host.status()["running"], false);
        host.add_token("worker", vec![Grant::Work]).unwrap();
        assert_eq!(host.tokens().unwrap().len(), 1);
    }
}
