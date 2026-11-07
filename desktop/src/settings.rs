//! Where the app keeps things, its settings (`settings.json`), and the media servers' API keys
//! (`secrets/<server>.key`, readable by this user only). Everything lives under the platform's
//! app-data folder. Keys stay in this process: the page is told only whether one is set and its
//! last four characters, never the key.

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

    /// A media server's API key (`jellyfin` or `stash`).
    pub fn secret(&self, server: Server) -> PathBuf {
        self.root.join("secrets").join(format!("{}.key", server.name()))
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
    /// Check for updates. Off until Vergence has a release site.
    pub check_updates: bool,
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

/// Keeps a server's key (or removes it, for an empty one).
pub fn set_secret(paths: &Paths, server: Server, key: &str) -> std::io::Result<()> {
    let key = key.trim();
    let path = paths.secret(server);
    if key.is_empty() {
        return match std::fs::remove_file(&path) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e),
            _ => Ok(()),
        };
    }
    #[cfg(unix)]
    if let Some(dir) = path.parent() {
        use std::os::unix::fs::PermissionsExt;
        std::fs::create_dir_all(dir)?;
        std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))?;
    }
    write_atomic(&path, key.as_bytes(), true)
}

pub fn has_secret(paths: &Paths, server: Server) -> bool {
    paths.secret(server).is_file()
}

/// What the page may know about a key: that it is set, and its last four characters.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct KeyHint {
    pub set: bool,
    pub ends_with: Option<String>,
}

pub fn key_hint(paths: &Paths, server: Server) -> KeyHint {
    match std::fs::read_to_string(paths.secret(server)) {
        Ok(key) => {
            let key = key.trim();
            let tail: String = key.chars().rev().take(4).collect::<Vec<_>>().into_iter().rev().collect();
            KeyHint { set: true, ends_with: (key.chars().count() >= 12).then_some(tail) }
        }
        Err(_) => KeyHint { set: false, ends_with: None },
    }
}

/// The coordinator's (and the worker's) configuration for one server: its address and the
/// key's file, never the key itself.
pub fn server_config(paths: &Paths, settings: &Settings, server: Server) -> Option<vergence_coordinator::source::ServerConfig> {
    let url = settings.url(server)?;
    Some(vergence_coordinator::source::ServerConfig {
        url: url.to_string(),
        api_key: None,
        api_key_file: has_secret(paths, server).then(|| paths.secret(server).to_string_lossy().into_owned()),
    })
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
    fn keys_are_private_and_only_hinted() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::new(dir.path());
        assert!(!key_hint(&paths, Server::Jellyfin).set);
        set_secret(&paths, Server::Jellyfin, " 0123456789abcdef \n").unwrap();
        assert_eq!(std::fs::read_to_string(paths.secret(Server::Jellyfin)).unwrap(), "0123456789abcdef");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(paths.secret(Server::Jellyfin)).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600);
        }
        let hint = key_hint(&paths, Server::Jellyfin);
        assert_eq!(hint, KeyHint { set: true, ends_with: Some("cdef".into()) });
        // A short key isn't hinted at all: four characters would be too much of it.
        set_secret(&paths, Server::Stash, "short").unwrap();
        assert_eq!(key_hint(&paths, Server::Stash).ends_with, None);

        let settings = Settings { jellyfin_url: Some("http://jf".into()), ..Settings::default() };
        let config = server_config(&paths, &settings, Server::Jellyfin).unwrap();
        assert_eq!(config.api_key, None);
        assert_eq!(config.key().unwrap().as_deref(), Some("0123456789abcdef"));
        assert!(server_config(&paths, &settings, Server::Stash).is_none());

        set_secret(&paths, Server::Jellyfin, "").unwrap();
        assert!(!key_hint(&paths, Server::Jellyfin).set);
        set_secret(&paths, Server::Jellyfin, "").unwrap();
    }
}
