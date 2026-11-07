//! `vergenced`'s configuration file (JSON).
//!
//! ```json
//! {
//!   "listen": "127.0.0.1:8420",
//!   "store": "/var/lib/vergenced",
//!   "jellyfin": { "url": "http://127.0.0.1:8096", "apiKeyFile": "/var/lib/vergenced/jellyfin.key" },
//!   "stash": { "url": "http://127.0.0.1:9999", "apiKeyFile": "/var/lib/vergenced/stash.key" },
//!   "scanEveryMinutes": 360
//! }
//! ```
//! Keys can be given inline as `apiKey` too; a file keeps them out of the config.
//! Tokens for players and workers live in `<store>/tokens.json` (`vergenced token add`).

use std::path::PathBuf;

use serde::Deserialize;

use crate::queue::Workload;
use crate::source::{Jellyfin, ServerConfig, Sources, Stash};

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Config {
    #[serde(default = "default_listen")]
    pub listen: String,
    pub store: PathBuf,
    #[serde(default)]
    pub tokens_file: Option<PathBuf>,
    #[serde(default)]
    pub jellyfin: Option<ServerConfig>,
    #[serde(default)]
    pub stash: Option<ServerConfig>,
    #[serde(default = "Workload::defaults")]
    pub workloads: Vec<Workload>,
    /// Queue every stereo title the servers list, at start and then this often.
    #[serde(default)]
    pub scan_every_minutes: Option<u64>,
}

fn default_listen() -> String {
    "127.0.0.1:8420".into()
}

impl Config {
    pub fn load(path: &std::path::Path) -> Result<Config, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn tokens_path(&self) -> PathBuf {
        self.tokens_file.clone().unwrap_or_else(|| self.store.join("tokens.json"))
    }

    pub fn sources(&self) -> std::io::Result<Sources> {
        let http = reqwest::Client::builder()
            .user_agent(concat!("vergenced/", env!("CARGO_PKG_VERSION")))
            .timeout(std::time::Duration::from_secs(20))
            .build()
            .map_err(std::io::Error::other)?;
        Ok(Sources {
            jellyfin: self.jellyfin.as_ref().map(|c| Jellyfin::new(c, http.clone())).transpose()?,
            stash: self.stash.as_ref().map(|c| Stash::new(c, http.clone())).transpose()?,
        })
    }
}
