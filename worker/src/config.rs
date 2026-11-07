//! The worker's configuration file: the same JSON as `vergence worker`'s
//! (so one file can serve either), plus keys only this worker reads.
//!
//! ```json
//! {
//!   "coordinator": "http://home-server:8420",
//!   "tokenFile": "~/.config/vergence/worker.token",
//!   "name": "gpu-pc",
//!   "remote": false,
//!   "models": "~/Models/onnx",
//!   "computeUnits": "all",
//!   "ffmpeg": "C:/ffmpeg/bin/ffmpeg.exe",
//!   "jellyfin": { "url": "http://home-server:8096", "apiKeyFile": "~/.config/vergence/jellyfin.key" },
//!   "stash": { "url": "http://home-server:9999", "apiKeyFile": "~/.config/vergence/stash.key" },
//!
//!   "executionProvider": "auto",
//!   "deviceId": 0,
//!   "shareCoreMLProfile": false,
//!   "yieldProcesses": ["obs64.exe"],
//!
//!   "workloads": ["smooth-motion-3d", "smooth-motion-2d", "ai-3d-depth", "ai-hdr"],
//!   "depthModel": "~/Models/DepthAnythingV2Small.onnx",
//!   "depthSize": "770x574",
//!   "gainModel": "~/Models/GainMapGMNet.onnx"
//! }
//! ```
//!
//! - `models` holds `RIFE*480x288*.onnx` and `NeuFlow*512x288*.onnx`
//!   (`scripts/export-onnx-models.py`), and `RIFE*960x576*.onnx` for 2D titles.
//! - `workloads`: the ones it takes (the coordinator's names); by default every
//!   one its models allow: 3D always, 2D when the larger RIFE is there, depth
//!   with `depthModel`, AI HDR with `gainModel`, as `vergence worker` decides.
//! - `depthModel`: Depth Anything V2 Small as ONNX (`scripts/export-depth-onnx.py`
//!   or `vergence-convert --download-model`; Apache-2.0 weights only), run at
//!   `depthSize` when its input size is free (770×574 by default, the Mac
//!   worker's).
//! - `gainModel`: GMNet as ONNX (`scripts/export-gmnet-onnx.py`).
//! - `executionProvider`: `auto` (the default: CUDA, then DirectML, then Core
//!   ML, then CPU, whichever loads first), `cuda`, `tensorrt`, `directml`,
//!   `coreml` or `cpu`. `--ep` overrides it.
//! - `computeUnits` (`ane`, `gpu`, `cpu`, `all`) applies to the Core ML provider.
//! - `shareCoreMLProfile`: name the models exactly as the Core ML workers do,
//!   so this worker may fill titles they started. Only once a parity check
//!   (`scripts/check-onnx-worker.sh`) has shown the two agree; off by default.
//! - `yieldProcesses`: more executable names that make the worker stand
//!   aside while they run (see `yielding`).
//!
//! Unknown keys are ignored, as Swift's decoder ignores this worker's keys.

use std::path::{Path, PathBuf};

use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Server {
    pub url: String,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub api_key_file: Option<String>,
}

impl Server {
    pub fn key(&self) -> Result<Option<String>, String> {
        if let Some(key) = &self.api_key {
            return Ok(Some(key.clone()));
        }
        let Some(file) = &self.api_key_file else { return Ok(None) };
        read_secret(file).map(Some)
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Config {
    pub coordinator: String,
    #[serde(default)]
    pub token: Option<String>,
    #[serde(default)]
    pub token_file: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub remote: Option<bool>,
    pub models: String,
    #[serde(default)]
    pub compute_units: Option<String>,
    #[serde(default)]
    pub ffmpeg: Option<String>,
    #[serde(default)]
    pub jellyfin: Option<Server>,
    #[serde(default)]
    pub stash: Option<Server>,
    #[serde(default)]
    pub execution_provider: Option<String>,
    #[serde(default)]
    pub device_id: Option<i32>,
    #[serde(default, rename = "shareCoreMLProfile")]
    pub share_core_ml_profile: Option<bool>,
    #[serde(default)]
    pub yield_processes: Option<Vec<String>>,
    #[serde(default)]
    pub workloads: Option<Vec<String>>,
    #[serde(default)]
    pub depth_model: Option<String>,
    #[serde(default)]
    pub depth_size: Option<String>,
    #[serde(default)]
    pub gain_model: Option<String>,
}

impl Config {
    pub fn load(path: &Path) -> Result<Config, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// `depthSize` as (width, height): 770×574 unless set.
    pub fn depth_size(&self) -> Result<(usize, usize), String> {
        let Some(text) = &self.depth_size else { return Ok((770, 574)) };
        let parsed = text.split_once('x').and_then(|(w, h)| Some((w.trim().parse().ok()?, h.trim().parse().ok()?)));
        match parsed {
            Some((w, h)) if w % 14 == 0 && h % 14 == 0 && w > 0 && h > 0 => Ok((w, h)),
            _ => Err(format!("depthSize {text} isn't WIDTHxHEIGHT in multiples of 14")),
        }
    }

    pub fn token(&self) -> Result<String, String> {
        match (&self.token, &self.token_file) {
            (Some(token), _) => Ok(token.clone()),
            (None, Some(file)) => read_secret(file),
            (None, None) => Err("the config needs token or tokenFile".into()),
        }
    }
}

fn read_secret(file: &str) -> Result<String, String> {
    let path = expand_tilde(file);
    std::fs::read_to_string(&path).map(|s| s.trim().to_string()).map_err(|e| format!("{}: {e}", path.display()))
}

/// `~/x` against the home folder (`HOME`, or `USERPROFILE` on Windows).
pub fn expand_tilde(path: &str) -> PathBuf {
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"));
    match (path.strip_prefix("~/").or_else(|| path.strip_prefix("~\\")), home) {
        (Some(rest), Some(home)) => PathBuf::from(home).join(rest),
        _ if path == "~" => std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).map(PathBuf::from).unwrap_or_default(),
        _ => PathBuf::from(path),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_swift_workers_config() {
        let json = br#"{"coordinator": "http://127.0.0.1:8420", "tokenFile": "/tmp/t", "name": "check",
            "models": "/m", "computeUnits": "gpu", "jellyfin": {"url": "http://127.0.0.1:8097", "apiKeyFile": "/k"},
            "shareCoreMLProfile": true, "somethingNew": 1}"#;
        let config: Config = serde_json::from_slice(json).unwrap();
        assert_eq!(config.name.as_deref(), Some("check"));
        assert_eq!(config.share_core_ml_profile, Some(true));
        assert_eq!(config.jellyfin.unwrap().api_key_file.as_deref(), Some("/k"));
    }

    #[test]
    fn expands_the_home_folder() {
        let home = std::env::var("HOME").unwrap();
        assert_eq!(expand_tilde("~/a/b"), PathBuf::from(home).join("a/b"));
        assert_eq!(expand_tilde("/a/b"), PathBuf::from("/a/b"));
    }
}
