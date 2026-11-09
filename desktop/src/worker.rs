//! The worker sidecar: `vergence worker --config -` on a Mac (Core ML), `vergence-worker.exe
//! --config -` on Windows (ONNX Runtime; the same config format), run as a child process
//! against this app's own coordinator. Its config goes through the child's standard input, with
//! a `work` token made for this run only (never written, its hash included), so nothing secret
//! reaches a file. It names no media server: the worker is on loopback, so each lease carries the
//! credential a headset handed over for that piece's server (`vergence_coordinator::credentials`).

use std::collections::VecDeque;
use std::path::Path;
use std::sync::{Arc, Mutex};

use serde::Serialize;
use serde_json::{Value, json};
use vergence_coordinator::auth::Grant;

use crate::coordinator::CoordinatorHost;
use crate::settings::{Paths, Settings};
use crate::sidecar::{self, ToolKind};

/// Whether this platform has a worker to run (Core ML on a Mac, ONNX Runtime on Windows).
pub const AVAILABLE: bool = cfg!(any(target_os = "macos", windows));

/// Lines of the worker's output kept for the window.
const LOG_LINES: usize = 400;
/// The coordinator token the app gives its own worker.
pub const TOKEN_NAME: &str = "this-computer-worker";

/// `vergence worker`'s config (see `Sources/VergenceCLI/Worker.swift`), as its standard input.
#[derive(PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkerConfig {
    pub coordinator: String,
    pub token: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub remote: bool,
    pub models: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ffmpeg: Option<String>,
}

impl std::fmt::Debug for WorkerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerConfig")
            .field("coordinator", &self.coordinator)
            .field("token", &"<redacted>")
            .field("models", &self.models)
            .finish_non_exhaustive()
    }
}

impl WorkerConfig {
    pub fn new(settings: &Settings, token: String, ffmpeg: Option<&Path>) -> Result<WorkerConfig, String> {
        let models = settings.models_folder.clone().ok_or("Choose the folder of interpolation models first (Settings).")?;
        Ok(WorkerConfig {
            coordinator: settings.local_url(),
            token,
            name: settings.worker_name.clone(),
            remote: false,
            models,
            ffmpeg: ffmpeg.map(|p| p.to_string_lossy().into_owned()),
        })
    }

    /// The bytes for the worker's standard input.
    pub fn to_stdin(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("the config encodes")
    }
}

/// What older versions wrote: the worker's config and its token as files. Removed at start.
pub fn remove_old_files(paths: &Paths) {
    for name in ["worker.token", "worker.json"] {
        let path = paths.worker_dir().join(name);
        match std::fs::remove_file(&path) {
            Ok(()) => tracing::info!(file = %path.display(), "removed a file older versions kept"),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => tracing::warn!(file = %path.display(), error = %e, "couldn't remove an old file"),
        }
    }
}

struct Running {
    pid: Option<u32>,
    stop: Option<tokio::sync::oneshot::Sender<()>>,
}

pub struct WorkerHost {
    coordinator: Arc<CoordinatorHost>,
    running: Mutex<Option<Running>>,
    log: Arc<Mutex<VecDeque<String>>>,
    last_exit: Arc<Mutex<Option<String>>>,
}

impl WorkerHost {
    pub fn new(coordinator: Arc<CoordinatorHost>) -> WorkerHost {
        WorkerHost {
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
        if !AVAILABLE {
            return Err("This computer's worker runs on a Mac or Windows for now.".into());
        }
        if self.is_running() {
            return Ok(());
        }
        let tool = sidecar::find_tool(settings, ToolKind::Worker).ok_or(sidecar::missing_tool(ToolKind::Worker))?;
        if self.coordinator.state().is_none() {
            return Err("The coordinator isn't running.".into());
        }
        if settings.models_folder.is_none() {
            return Err("Choose the folder of interpolation models first (Settings).".into());
        }
        let token = self.coordinator.add_unsaved_token(TOKEN_NAME, vec![Grant::Work])?;
        let config = WorkerConfig::new(settings, token, sidecar::find_ffmpeg().as_deref())?;
        let cli = tool.program.clone();
        let mut child = sidecar::spawn_with_input(&cli, &tool.arguments(["--config".into(), "-".into()]), false, Some(config.to_stdin()))
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
            "available": AVAILABLE,
            "running": running.is_some(),
            "pid": running.as_ref().and_then(|r| r.pid),
            "cli": sidecar::find_tool(settings, ToolKind::Worker).map(|t| t.program.to_string_lossy().into_owned()),
            "modelsFolder": settings.models_folder,
            "lastExit": *self.last_exit.lock().unwrap(),
            "log": self.log.lock().unwrap().iter().collect::<Vec<_>>(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_config_goes_by_standard_input_and_names_no_server() {
        let mut settings = Settings { port: 8421, ..Settings::default() };
        assert!(WorkerConfig::new(&settings, "t".into(), None).unwrap_err().contains("models"));
        settings.models_folder = Some("/Models/interp".into());
        settings.jellyfin_url = Some("http://127.0.0.1:8097".into());
        let config = WorkerConfig::new(&settings, "the-run-token".into(), Some(Path::new("/opt/homebrew/bin/ffmpeg"))).unwrap();
        let value: Value = serde_json::from_slice(&config.to_stdin()).unwrap();
        assert_eq!(value["coordinator"], "http://127.0.0.1:8421");
        assert_eq!(value["token"], "the-run-token");
        assert_eq!(value["models"], "/Models/interp");
        assert_eq!(value["remote"], false);
        assert_eq!(value["ffmpeg"], "/opt/homebrew/bin/ffmpeg");
        assert!(value.get("jellyfin").is_none() && value.get("stash").is_none() && value.get("tokenFile").is_none());
        assert!(value.get("name").is_none());
        assert!(!format!("{config:?}").contains("the-run-token"));
    }

    #[test]
    fn older_files_are_removed() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::new(dir.path());
        std::fs::create_dir_all(paths.worker_dir()).unwrap();
        std::fs::write(paths.worker_dir().join("worker.token"), "old").unwrap();
        std::fs::write(paths.worker_dir().join("worker.json"), "{}").unwrap();
        remove_old_files(&paths);
        assert!(!paths.worker_dir().join("worker.token").exists());
        assert!(!paths.worker_dir().join("worker.json").exists());
    }
}
