//! Where the app keeps things and its settings (`settings.json`), under the platform's app-data
//! folder. No media server key is kept here, or anywhere on disk: a headset hands its sign-in
//! over when it starts a stream, and one typed into the window is held the same way, in this
//! process's memory only (`crate::coordinator::CoordinatorHost::hold_credential`). The page is
//! told only that one is held, by whom and for how long, never the key.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// The files under the app-data folder.
#[derive(Clone, Debug)]
pub struct Paths {
    pub root: PathBuf,
}

impl Paths {
    pub fn new(root: impl Into<PathBuf>) -> Paths {
        Paths { root: root.into() }
    }

    pub fn settings(&self) -> PathBuf {
        self.root.join("settings.json")
    }

    /// Where older versions kept the media servers' API keys; removed at start.
    pub fn old_secrets(&self) -> PathBuf {
        self.root.join("secrets")
    }

    /// The coordinator's store: titles, pieces and `tokens.json`.
    pub fn store(&self) -> PathBuf {
        self.root.join("coordinator")
    }

    pub fn tokens(&self) -> PathBuf {
        self.store().join("tokens.json")
    }

    /// Paired headsets (vergence-companion's `JsonFileStore`).
    pub fn devices(&self) -> PathBuf {
        self.root.join("devices.json")
    }

    /// Which coordinator token each paired device was given.
    pub fn device_tokens(&self) -> PathBuf {
        self.root.join("device-tokens.json")
    }

    pub fn worker_dir(&self) -> PathBuf {
        self.root.join("worker")
    }

    pub fn convert_queue(&self) -> PathBuf {
        self.root.join("convert-queue.json")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Server {
    Jellyfin,
    Stash,
}

impl Server {
    pub fn name(self) -> &'static str {
        match self {
            Server::Jellyfin => "jellyfin",
            Server::Stash => "stash",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Settings {
    /// The coordinator's port (plain HTTP, tokens).
    pub port: u16,
    /// The port paired headsets reach the coordinator on (TLS-PSK, every interface).
    pub paired_port: u16,
    /// Listen on every interface (headsets and other workers on the LAN), not just this computer.
    pub lan: bool,
    pub jellyfin_url: Option<String>,
    pub stash_url: Option<String>,
    /// Queue every stereo title the servers list, at start and then every `scan_every_hours`.
    pub scan: bool,
    pub scan_every_hours: u32,
    /// The folder of interpolation models the worker loads (RIFE*480x288, NeuFlow*512x288).
    pub models_folder: Option<String>,
    /// The `vergence` executable, if not found by itself.
    pub vergence_cli: Option<String>,
    /// The Core ML depth model for conversions (empty: the one `vergence convert` finds itself).
    pub depth_model: Option<String>,
    /// The model a conversion downloads when no `depth_model` is given (`models`): Small unless
    /// a build with `noncommercial-models` offers more and its licence was accepted.
    #[serde(default)]
    pub depth_variant: Option<String>,
    /// The restricted models whose licence the person accepted.
    #[serde(default)]
    pub licences_accepted: Vec<String>,
    /// The worker's name towards the coordinator (empty: this computer's name).
    pub worker_name: Option<String>,
    /// Start the worker with the app.
    pub worker_at_start: bool,
    /// Check for updates (off until the person turns it on).
    pub check_updates: bool,
    /// Reach apps.illixion.com for updates only over DNS-over-HTTPS and Encrypted Client Hello, so a
    /// network doesn't see the name; where that is blocked the check goes to GitHub directly (never
    /// to apps.illixion.com in the clear). On by default.
    pub private_updates: bool,
    /// Announce this computer to headsets on the network, and answer one asking to pair
    /// (`presence`). Off, a headset pairs only by this computer's address or link.
    pub discoverable: bool,
}

pub const DEFAULT_PORT: u16 = 8420;
pub const DEFAULT_PAIRED_PORT: u16 = vergence_coordinator::paired::DEFAULT_PAIRED_PORT;

impl Default for Settings {
    fn default() -> Self {
        Settings {
            port: DEFAULT_PORT,
            paired_port: DEFAULT_PAIRED_PORT,
            lan: false,
            jellyfin_url: None,
            stash_url: None,
            scan: false,
            scan_every_hours: 6,
            models_folder: None,
            vergence_cli: None,
            depth_model: None,
            depth_variant: None,
            licences_accepted: Vec::new(),
            worker_name: None,
            worker_at_start: false,
            check_updates: false,
            private_updates: true,
            discoverable: true,
        }
    }
}

impl Settings {
    pub fn load(paths: &Paths) -> Settings {
        std::fs::read(paths.settings()).ok().and_then(|bytes| serde_json::from_slice(&bytes).ok()).unwrap_or_default()
    }

    pub fn save(&self, paths: &Paths) -> std::io::Result<()> {
        write_atomic(&paths.settings(), &serde_json::to_vec_pretty(self)?, false)
    }

    /// Where the coordinator listens.
    pub fn listen(&self) -> String {
        format!("{}:{}", if self.lan { "0.0.0.0" } else { "127.0.0.1" }, self.port)
    }

    /// Where paired headsets reach the coordinator (TLS-PSK only, so on every interface).
    pub fn paired_listen(&self) -> String {
        format!("0.0.0.0:{}", self.paired_port)
    }

    /// How this computer reaches its own coordinator.
    pub fn local_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    pub fn url(&self, server: Server) -> Option<&str> {
        match server {
            Server::Jellyfin => self.jellyfin_url.as_deref(),
            Server::Stash => self.stash_url.as_deref(),
        }
    }

    /// Trims every text field, empties to `None`, and checks what can be checked.
    pub fn normalized(mut self) -> Result<Settings, String> {
        fn clean(value: &mut Option<String>) {
            *value = value.take().map(|v| v.trim().to_string()).filter(|v| !v.is_empty());
        }
        for field in [
            &mut self.jellyfin_url,
            &mut self.stash_url,
            &mut self.models_folder,
            &mut self.vergence_cli,
            &mut self.depth_model,
            &mut self.worker_name,
        ] {
            clean(field);
        }
        for (name, url) in [("Jellyfin", &mut self.jellyfin_url), ("Stash", &mut self.stash_url)] {
            if let Some(value) = url {
                if !(value.starts_with("http://") || value.starts_with("https://")) {
                    return Err(format!("The {name} address must start with http:// or https://."));
                }
                while value.ends_with('/') {
                    value.pop();
                }
            }
        }
        if self.port < 1024 || self.paired_port < 1024 {
            return Err("Pick a port from 1024 up.".into());
        }
        if self.port == self.paired_port {
            return Err("The paired port must differ from the coordinator's port.".into());
        }
        self.scan_every_hours = self.scan_every_hours.clamp(1, 24 * 7);
        Ok(self)
    }
}

/// Writes a file through a temporary one and a rename; `private` makes it 0600 on Unix.
pub fn write_atomic(path: &Path, bytes: &[u8], private: bool) -> std::io::Result<()> {
    let parent = path.parent().ok_or_else(|| std::io::Error::other("no parent folder"))?;
    std::fs::create_dir_all(parent)?;
    let mut file = tempfile::NamedTempFile::new_in(parent)?;
    std::io::Write::write_all(&mut file, bytes)?;
    #[cfg(unix)]
    if private {
        use std::os::unix::fs::PermissionsExt;
        file.as_file().set_permissions(std::fs::Permissions::from_mode(0o600))?;
    }
    #[cfg(not(unix))]
    let _ = private;
    file.as_file().sync_all()?;
    file.persist(path).map_err(|e| e.error)?;
    Ok(())
}

/// Deletes the key files older versions kept (`secrets/<server>.key`).
pub fn remove_old_secrets(paths: &Paths) {
    let dir = paths.old_secrets();
    match std::fs::remove_dir_all(&dir) {
        Ok(()) => tracing::info!(dir = %dir.display(), "removed the media keys older versions kept on disk"),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => tracing::warn!(dir = %dir.display(), error = %e, "couldn't remove the old media keys"),
    }
}

/// What the page may know about a held credential: never the key.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct KeyHint {
    pub set: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// Who handed it over: a headset's name, or "this computer".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from: Option<String>,
    /// Seconds until it is forgotten.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seconds_left: Option<u64>,
}

pub fn key_hint(held: &[vergence_coordinator::credentials::HeldSummary], server: Server) -> KeyHint {
    let name = server.name();
    match held.iter().find(|h| h.server.name() == name) {
        Some(h) => KeyHint { set: true, url: Some(h.url.clone()), from: Some(h.from.clone()), seconds_left: Some(h.seconds_left) },
        None => KeyHint { set: false, url: None, from: None, seconds_left: None },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_round_trip_and_fill_in_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::new(dir.path());
        assert_eq!(Settings::load(&paths), Settings::default());
        let settings = Settings {
            port: 9000,
            lan: true,
            jellyfin_url: Some("http://127.0.0.1:8097".into()),
            scan: true,
            scan_every_hours: 3,
            models_folder: Some("/models".into()),
            ..Settings::default()
        };
        settings.save(&paths).unwrap();
        assert_eq!(Settings::load(&paths), settings);
        assert_eq!(Settings::load(&paths).listen(), "0.0.0.0:9000");
        // A file from an older version, with fields missing, keeps every other default.
        std::fs::write(paths.settings(), br#"{"lan":true}"#).unwrap();
        let old = Settings::load(&paths);
        assert!(old.lan);
        assert_eq!(old.port, DEFAULT_PORT);
        assert!(!old.check_updates);
        assert!(old.private_updates, "a file from before the setting keeps it on");
        assert_eq!(old.listen(), "0.0.0.0:8420");
    }

    #[test]
    fn normalizing_trims_and_checks() {
        let settings = Settings {
            jellyfin_url: Some(" http://jf.local:8096/ ".into()),
            stash_url: Some("  ".into()),
            ..Settings::default()
        }
        .normalized()
        .unwrap();
        assert_eq!(settings.jellyfin_url.as_deref(), Some("http://jf.local:8096"));
        assert_eq!(settings.stash_url, None);
        assert!(Settings { jellyfin_url: Some("jf.local".into()), ..Settings::default() }.normalized().is_err());
        assert!(Settings { port: 80, ..Settings::default() }.normalized().is_err());
    }

    #[test]
    fn old_key_files_are_removed() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::new(dir.path());
        std::fs::create_dir_all(paths.old_secrets()).unwrap();
        std::fs::write(paths.old_secrets().join("jellyfin.key"), "0123456789abcdef").unwrap();
        remove_old_secrets(&paths);
        assert!(!paths.old_secrets().exists());
        remove_old_secrets(&paths);
    }
}
