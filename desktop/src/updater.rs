//! Self-updates, verified before anything downloaded runs. The verification, the sources and the
//! download are the `signed-update` crate's (`~/Projects/signed-update`, `docs/PROTOCOL.md`),
//! shared with Apparition: a release is one `SHA256SUMS` signed with SSHSIG by a key pinned in
//! `release-signers`, fetched from apps.illixion.com (which only redirects, so hosting can move)
//! and from GitHub Releases directly when that is down. This file keeps what is Vergence's: the
//! status the window shows, the setting that turns checking on, and when to install.
//!
//! **Signers:** `release-signers` pins the same two keys as Apparition's releases (a hardware key
//! and an offline backup) under the namespace [`NAMESPACE`]; `desktop/scripts/bless-release.sh`
//! signs a release with them. **Windows:** installing an update there isn't ported yet (Apparition's
//! `update-windows.ps1` helper, which swaps a running executable); a check says where to download.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use serde::Serialize;
use signed_update::{Config, Source, Update, is_newer, signer_keys};

const VERSION: &str = env!("CARGO_PKG_VERSION");
/// The SSHSIG namespace releases are signed under (`ssh-keygen -Y sign -n vergence-release`).
pub const NAMESPACE: &str = "vergence-release";
/// The pinned release keys (ssh-keygen allowed_signers format).
const SIGNERS: &str = include_str!("../release-signers");
/// The GitHub repository releases are published to.
const REPO: &str = "illixion/vergence";
pub const CHECK_INTERVAL: Duration = Duration::from_secs(6 * 60 * 60);

#[derive(Clone, Serialize, Default, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Status {
    pub version: Option<String>,
    /// "", "off", "unconfigured", "available", "downloading", "installing", "error".
    pub phase: &'static str,
    pub message: String,
    pub can_install: bool,
}

pub struct Updates {
    pub status: Mutex<Status>,
    config: Config,
    candidate: Mutex<Option<Update>>,
    operation: tokio::sync::Mutex<()>,
    installing: AtomicBool,
}

impl Default for Updates {
    fn default() -> Self {
        Self::with_config(config())
    }
}

/// Where Vergence looks for releases: apps.illixion.com first (redirects only; it points at
/// GitHub, or at a self-hosted copy while GitHub is down), then GitHub itself.
pub fn config() -> Config {
    Config::new("Vergence", NAMESPACE, SIGNERS)
        .source(Source::apps("vergence"))
        .source(Source::github(REPO))
        .user_agent(concat!("Vergence/", env!("CARGO_PKG_VERSION")))
}

/// This platform's installer, as a signed file name with the version left open (the signed name
/// binds the version, so a host can't relabel an old release as new).
pub fn installer_pattern() -> Option<&'static str> {
    if cfg!(target_os = "macos") {
        Some("Vergence-{version}-macos-universal.zip")
    } else if cfg!(windows) {
        Some("Vergence-{version}-windows-x86_64.zip")
    } else {
        None
    }
}

pub fn signers_pinned() -> bool {
    !signer_keys(SIGNERS).is_empty()
}

impl Updates {
    pub fn with_config(config: Config) -> Self {
        Self {
            status: Mutex::default(),
            config,
            candidate: Mutex::default(),
            operation: tokio::sync::Mutex::default(),
            installing: AtomicBool::new(false),
        }
    }

    /// Looks for a newer verified release. `enabled` is the person's setting.
    pub async fn check(&self, enabled: bool) -> Result<Status, String> {
        if self.installing.load(Ordering::SeqCst) {
            return Err("An update is being installed.".into());
        }
        let _guard = self.operation.lock().await;
        let status = match (enabled, installer_pattern()) {
            (false, _) => Status { phase: "off", message: "Checking for updates is off.".into(), ..Status::default() },
            (true, _) if !signers_pinned() => Status {
                phase: "unconfigured",
                message: "No release signing key is pinned in this build.".into(),
                ..Status::default()
            },
            (true, None) => Status {
                phase: "unconfigured",
                message: "There is no Vergence release for this platform.".into(),
                ..Status::default()
            },
            (true, Some(pattern)) => match signed_update::check(&self.config, VERSION, pattern).await {
                Ok(Some(update)) => {
                    let version = update.version.clone();
                    let can_install = cfg!(target_os = "macos") && !cfg!(debug_assertions);
                    let message = if can_install {
                        format!("Signed by {}. Ready to download, verify and restart.", update.signer)
                    } else if cfg!(windows) {
                        format!("Signed by {}. Download Vergence {version} from github.com/{REPO}/releases and replace Vergence.exe.", update.signer)
                    } else {
                        format!("Signed by {}. Development builds are not replaced.", update.signer)
                    };
                    *self.candidate.lock().unwrap() = Some(update);
                    Status { version: Some(version), phase: "available", message, can_install }
                }
                Ok(None) => {
                    *self.candidate.lock().unwrap() = None;
                    Status::default()
                }
                Err(e) => Status { phase: "error", message: format!("Couldn't check for updates: {e}"), ..Status::default() },
            },
        };
        *self.status.lock().unwrap() = status.clone();
        Ok(status)
    }

    /// Downloads, verifies and installs the announced update; true means restart now.
    pub async fn install(&self) -> Result<bool, String> {
        if self.installing.swap(true, Ordering::SeqCst) {
            return Err("The update is already being installed.".into());
        }
        let result = self.install_inner().await;
        self.installing.store(false, Ordering::SeqCst);
        if let Err(e) = &result {
            let mut status = self.status.lock().unwrap();
            status.phase = "error";
            status.message = format!("Update failed: {e}. You can retry; your settings are unchanged.");
        }
        result
    }

    async fn install_inner(&self) -> Result<bool, String> {
        let _guard = self.operation.lock().await;
        if cfg!(debug_assertions) {
            return Err("Development builds are not replaced.".into());
        }
        let update = self.candidate.lock().unwrap().clone().ok_or("Check for updates first.")?;
        if !is_newer(&update.version, VERSION) {
            return Err("Refusing to install an older version.".into());
        }
        self.status.lock().unwrap().phase = "downloading";
        let scratch = tempfile::Builder::new().prefix("vergence-update-").tempdir().map_err(|e| e.to_string())?;
        let package = update.download(scratch.path(), |_| {}).await.map_err(|e| e.to_string())?;
        self.status.lock().unwrap().phase = "installing";
        tokio::task::spawn_blocking(move || install_package(&package)).await.map_err(|e| e.to_string())??;
        Ok(true)
    }
}

/// Replaces the running app with the verified package.
fn install_package(package: &std::path::Path) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        let bundle = signed_update::install::current_bundle().ok_or("Only an installed Vergence.app updates itself.")?;
        signed_update::install::install_bundle(package, &bundle).map_err(|e| e.to_string())
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = package;
        Err("Installing updates on this platform isn't ported yet: download the new version instead.".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_release_keys_are_pinned_and_sources_are_ordered() {
        assert!(signers_pinned());
        let pinned = signer_keys(SIGNERS);
        assert_eq!(pinned.len(), 2);
        assert!(pinned.iter().all(|(blob, principal)| blob.len() == 51 && principal == "releases@vergence"));
        let config = config();
        assert_eq!(config.namespace, NAMESPACE);
        assert_eq!(
            config.sources,
            [Source::apps("vergence"), Source::github("illixion/vergence")],
            "apps.illixion.com asks first; GitHub is the last resort"
        );
        if let Some(pattern) = installer_pattern() {
            assert!(pattern.starts_with("Vergence-{version}-"));
        }
    }

    #[tokio::test]
    async fn checking_is_off_by_default_and_reports_a_source_that_is_down() {
        let updates = Updates::default();
        assert_eq!(updates.check(false).await.unwrap().phase, "off");
        // Nothing listens on port 1: both sources fail, and the window is told why.
        let down = Updates::with_config(Config::new("Vergence", NAMESPACE, SIGNERS).source(Source::mirror("http://127.0.0.1:1", "vergence")));
        if installer_pattern().is_some() {
            let status = down.check(true).await.unwrap();
            assert_eq!(status.phase, "error");
            assert!(status.message.contains("Couldn't check for updates"), "{}", status.message);
        }
        assert!(down.install().await.is_err());
    }
}
