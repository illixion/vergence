//! The worker sidecar (Mac): `vergence worker --config FILE`, run as a child process against
//! this app's own coordinator, with a `work` token the app mints for it. Its config holds paths
//! only (the token's file, the servers' key files), never a secret.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde::Serialize;
use serde_json::{Value, json};
use vergence_coordinator::auth::Grant;

use crate::coordinator::CoordinatorHost;
use crate::settings::{Paths, Server, Settings, has_secret, write_atomic};
use crate::sidecar;

/// Lines of the worker's output kept for the window.
const LOG_LINES: usize = 400;
/// The coordinator token the app gives its own worker.
pub const TOKEN_NAME: &str = "this-computer-worker";

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkerServer {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key_file: Option<String>,
}

/// `vergence worker`'s config file (see `Sources/VergenceCLI/Worker.swift`).
#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkerConfig {
    pub coordinator: String,
    pub token_file: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub remote: bool,
    pub models: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ffmpeg: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub jellyfin: Option<WorkerServer>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stash: Option<WorkerServer>,
}

impl WorkerConfig {
    pub fn new(paths: &Paths, settings: &Settings, ffmpeg: Option<&Path>) -> Result<WorkerConfig, String> {
        let models = settings.models_folder.clone().ok_or("Choose the folder of interpolation models first (Settings).")?;
        let server = |server: Server| {
            settings.url(server).map(|url| WorkerServer {
                url: url.to_string(),
                api_key_file: has_secret(paths, server).then(|| paths.secret(server).to_string_lossy().into_owned()),
            })
        };
        let (jellyfin, stash) = (server(Server::Jellyfin), server(Server::Stash));
        if jellyfin.is_none() && stash.is_none() {
            return Err("Add a Jellyfin or Stash server first (Settings): the worker reads the media from it.".into());
        }
        Ok(WorkerConfig {
            coordinator: settings.local_url(),
            token_file: token_path(paths).to_string_lossy().into_owned(),
            name: settings.worker_name.clone(),
            remote: false,
            models,
            ffmpeg: ffmpeg.map(|p| p.to_string_lossy().into_owned()),
            jellyfin,
            stash,
        })
    }

    pub fn write(&self, paths: &Paths) -> std::io::Result<PathBuf> {
        let path = paths.worker_dir().join("worker.json");
        write_atomic(&path, &serde_json::to_vec_pretty(self)?, true)?;
        Ok(path)
    }
}

fn token_path(paths: &Paths) -> PathBuf {
    paths.worker_dir().join("worker.token")
}

struct Running {
    pid: Option<u32>,
    stop: Option<tokio::sync::oneshot::Sender<()>>,
}

pub struct WorkerHost {
    paths: Paths,
    coordinator: Arc<CoordinatorHost>,
    running: Mutex<Option<Running>>,
    log: Arc<Mutex<VecDeque<String>>>,
    last_exit: Arc<Mutex<Option<String>>>,
}

impl WorkerHost {
    pub fn new(paths: Paths, coordinator: Arc<CoordinatorHost>) -> WorkerHost {
        WorkerHost {
            paths,
            coordinator,
            running: Default::default(),
            log: Default::default(),
            last_exit: Default::default(),
        }
    }

    pub fn is_running(&self) -> bool {
        self.running.lock().unwrap().is_some()
    }

    pub fn start(self: &Arc<Self>, settings: &Settings) -> Result<(), String> {
        if !cfg!(target_os = "macos") {
            return Err("This computer's worker arrives with the ONNX Runtime engine; the Core ML one runs on a Mac.".into());
        }
        if self.is_running() {
            return Ok(());
        }
        let cli = sidecar::find_cli(settings).ok_or("No vergence command line found: build it (swift build -c release) or set its path in Settings.")?;
        if self.coordinator.state().is_none() {
            return Err("The coordinator isn't running.".into());
        }
        let config = WorkerConfig::new(&self.paths, settings, sidecar::find_ffmpeg().as_deref())?;
        let token = self.coordinator.add_token(TOKEN_NAME, vec![Grant::Work]).map_err(|e| format!("Couldn't make the worker's token: {e}"))?;
        write_atomic(&token_path(&self.paths), token.as_bytes(), true).map_err(|e| e.to_string())?;
        let config_path = config.write(&self.paths).map_err(|e| e.to_string())?;
        let mut child = sidecar::spawn(&cli, &["worker".into(), "--config".into(), config_path.into()], false)
            .map_err(|e| format!("Couldn't start {}: {e}", cli.display()))?;
        self.push_log(format!("started {} (pid {})", cli.display(), child.pid.unwrap_or(0)));
        *self.last_exit.lock().unwrap() = None;
        *self.running.lock().unwrap() = Some(Running { pid: child.pid, stop: child.take_stopper() });
        let this = self.clone();
        tokio::spawn(async move {
            while let Some(line) = child.lines.recv().await {
                this.push_log(line);
            }
            let status = child.exit.await;
            let message = match status {
                Ok(Ok(status)) => format!("the worker stopped ({status})"),
                Ok(Err(e)) => format!("the worker stopped: {e}"),
                Err(e) => format!("the worker stopped: {e}"),
            };
            this.push_log(message.clone());
            *this.last_exit.lock().unwrap() = Some(message);
            *this.running.lock().unwrap() = None;
        });
        Ok(())
    }

    /// Stops the worker (its PID only); its leases expire and go back to the queue.
    pub fn stop(&self) {
        if let Some(running) = self.running.lock().unwrap().as_mut()
            && let Some(stop) = running.stop.take()
        {
            let _ = stop.send(());
        }
    }

    /// Stops it and waits (up to 5 s) for it to be gone, at quit.
    pub async fn stop_and_wait(&self) {
        self.stop();
        for _ in 0..50 {
            if !self.is_running() {
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
    }

    fn push_log(&self, line: String) {
        let mut log = self.log.lock().unwrap();
        if log.len() == LOG_LINES {
            log.pop_front();
        }
        log.push_back(line);
    }

    pub fn status(&self, settings: &Settings) -> Value {
        let running = self.running.lock().unwrap();
        json!({
            "available": cfg!(target_os = "macos"),
            "running": running.is_some(),
            "pid": running.as_ref().and_then(|r| r.pid),
            "cli": sidecar::find_cli(settings).map(|p| p.to_string_lossy().into_owned()),
            "modelsFolder": settings.models_folder,
            "lastExit": *self.last_exit.lock().unwrap(),
            "log": self.log.lock().unwrap().iter().collect::<Vec<_>>(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::set_secret;

    #[test]
    fn the_config_names_files_not_secrets() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::new(dir.path());
        let mut settings = Settings { port: 8421, ..Settings::default() };
        assert!(WorkerConfig::new(&paths, &settings, None).unwrap_err().contains("models"));
        settings.models_folder = Some("/Models/interp".into());
        assert!(WorkerConfig::new(&paths, &settings, None).unwrap_err().contains("Jellyfin or Stash"));
        settings.jellyfin_url = Some("http://127.0.0.1:8097".into());
        settings.stash_url = Some("http://127.0.0.1:9998".into());
        set_secret(&paths, Server::Jellyfin, "the-jellyfin-key-0123").unwrap();

        let config = WorkerConfig::new(&paths, &settings, Some(Path::new("/opt/homebrew/bin/ffmpeg"))).unwrap();
        let written = config.write(&paths).unwrap();
        let text = std::fs::read_to_string(&written).unwrap();
        assert!(!text.contains("the-jellyfin-key-0123"));
        let value: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(value["coordinator"], "http://127.0.0.1:8421");
        assert_eq!(value["models"], "/Models/interp");
        assert_eq!(value["remote"], false);
        assert_eq!(value["ffmpeg"], "/opt/homebrew/bin/ffmpeg");
        assert_eq!(value["jellyfin"]["apiKeyFile"], paths.secret(Server::Jellyfin).to_string_lossy().as_ref());
        // Stash has no key here: none is named, and no `null` reaches the Swift decoder.
        assert_eq!(value["stash"], json!({ "url": "http://127.0.0.1:9998" }));
        assert!(value.get("name").is_none());
        assert_eq!(value["tokenFile"], token_path(&paths).to_string_lossy().as_ref());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(&written).unwrap().permissions().mode() & 0o777, 0o600);
        }
    }
}
