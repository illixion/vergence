//! Self-updates, verified before anything downloaded runs: a release site lists releases in
//! `release.json`; only a release's `SHA256SUMS` is trusted, and only once its SSHSIG
//! signature checks out against a key pinned in `release-signers`. A package is hashed against
//! that signed list before it is unpacked. URLs are hints; a version is bound to the signed file
//! names, so a site can't relabel an old release as new.
//!
//! Adapted from Apparition's desktop updater and release verification (`desktop/src/updater.rs`,
//! `core/src/release.rs`; MIT, same author).
//!
//! **Signers:** `release-signers` pins the same two keys as Apparition's releases (a hardware key
//! and an offline backup) under the namespace [`NAMESPACE`]; `desktop/scripts/bless-release.sh`
//! signs a release's `SHA256SUMS` with them. **Placeholder:** there is no release site yet, so
//! [`SITE`] is `None`, a check says updates aren't set up, and checking is off by default
//! (Settings). The Windows install step (Apparition's `update-windows.ps1` helper, which swaps a
//! running executable) isn't ported yet.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use base64::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256, Sha512};

/// PLACEHOLDER: the site serving `release.json` (e.g. `Some("https://…/vergence/")`).
pub const SITE: Option<&str> = None;
/// The SSHSIG namespace releases are signed under (`ssh-keygen -Y sign -n vergence-release`).
pub const NAMESPACE: &str = "vergence-release";
/// The pinned release keys (ssh-keygen allowed_signers format).
const SIGNERS: &str = include_str!("../release-signers");
const VERSION: &str = env!("CARGO_PKG_VERSION");
const MAX_ASSET_BYTES: u64 = 2 << 30;
pub const CHECK_INTERVAL: Duration = Duration::from_secs(6 * 60 * 60);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Index {
    pub format: String,
    pub version: u32,
    pub releases: Vec<Release>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Release {
    pub version: String,
    #[serde(default)]
    pub notes: Option<String>,
    /// Paths of the signed manifest and its signature, relative to the site.
    pub sums: String,
    pub signature: String,
    pub assets: Vec<Asset>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Asset {
    pub name: String,
    pub size: u64,
    pub sha256: String,
    pub urls: Vec<String>,
}

/// A release whose manifest is signed by a pinned key, and whose listed assets all match it.
#[derive(Debug, Clone)]
pub struct Verified {
    pub release: Release,
    pub signer: String,
    sums: BTreeMap<String, String>,
}

#[derive(Clone, Serialize, Default, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Status {
    pub version: Option<String>,
    /// "", "off", "unconfigured", "available", "downloading", "installing", "error".
    pub phase: &'static str,
    pub message: String,
    pub can_install: bool,
}

#[derive(Default)]
pub struct Updates {
    pub status: Mutex<Status>,
    candidate: Mutex<Option<(Verified, Asset)>>,
    operation: tokio::sync::Mutex<()>,
    installing: AtomicBool,
}

/// This platform's installer file name in release `version` (signed, so it binds the version).
pub fn installer_name(version: &str) -> Option<String> {
    let platform = if cfg!(target_os = "macos") {
        "macos-universal.zip"
    } else if cfg!(windows) {
        "windows-x86_64.zip"
    } else {
        return None;
    };
    Some(format!("Vergence-{version}-{platform}"))
}

pub fn signers_pinned() -> bool {
    !signer_keys(SIGNERS).is_empty()
}

impl Updates {
    /// Looks for a newer verified release. `enabled` is the person's setting.
    pub async fn check(&self, enabled: bool) -> Result<Status, String> {
        if self.installing.load(Ordering::SeqCst) {
            return Err("An update is being installed.".into());
        }
        let _guard = self.operation.lock().await;
        let status = match (enabled, SITE) {
            (false, _) => Status { phase: "off", message: "Checking for updates is off.".into(), ..Status::default() },
            (true, None) => Status {
                phase: "unconfigured",
                message: "Vergence has no release site yet; updates arrive once it does.".into(),
                ..Status::default()
            },
            (true, Some(_)) if !signers_pinned() => Status {
                phase: "unconfigured",
                message: "No release signing key is pinned in this build.".into(),
                ..Status::default()
            },
            (true, Some(site)) => match latest_installer(site, VERSION, SIGNERS).await {
                Ok(Some((verified, asset))) => {
                    let version = verified.release.version.clone();
                    let message = format!("Signed by {}. Ready to download, verify and restart.", verified.signer);
                    *self.candidate.lock().unwrap() = Some((verified, asset));
                    Status {
                        version: Some(version),
                        phase: "available",
                        message,
                        can_install: cfg!(target_os = "macos") && !cfg!(debug_assertions),
                    }
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
        let (verified, asset) = self.candidate.lock().unwrap().clone().ok_or("Check for updates first.")?;
        if !is_newer(&verified.release.version, VERSION) {
            return Err("Refusing to install an older version.".into());
        }
        self.status.lock().unwrap().phase = "downloading";
        let scratch = tempfile::Builder::new().prefix("vergence-update-").tempdir().map_err(|e| e.to_string())?;
        let package = download(&verified, &asset, scratch.path()).await?;
        self.status.lock().unwrap().phase = "installing";
        tokio::task::spawn_blocking(move || install_package(&package)).await.map_err(|e| e.to_string())??;
        Ok(true)
    }
}

fn http() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .user_agent(concat!("Vergence/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(600))
        .build()
        .map_err(|e| e.to_string())
}

async fn fetch(client: &reqwest::Client, url: &str) -> Result<Vec<u8>, String> {
    let mut response = client.get(url).timeout(Duration::from_secs(30)).send().await.map_err(|e| format!("Can't reach {url}: {e}"))?;
    if !response.status().is_success() {
        return Err(format!("{url}: {}", response.status()));
    }
    let mut bytes = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(|e| e.to_string())? {
        if bytes.len() + chunk.len() > 8 * 1024 * 1024 {
            return Err("The release metadata is too large.".into());
        }
        bytes.extend(chunk);
    }
    Ok(bytes)
}

/// The newest release with this platform's installer, if newer than `current`, verified.
pub async fn latest_installer(site: &str, current: &str, signers: &str) -> Result<Option<(Verified, Asset)>, String> {
    let site = reqwest::Url::parse(site).map_err(|e| format!("bad release site: {e}"))?;
    let client = http()?;
    let index: Index = serde_json::from_slice(&fetch(&client, site.join("release.json").map_err(|e| e.to_string())?.as_str()).await?)
        .map_err(|e| format!("release.json isn't a release list: {e}"))?;
    if index.format != "vergence-releases" || index.version != 1 {
        return Err(format!("release.json is format {} {}, which this version can't read", index.format, index.version));
    }
    let Some(release) = newest_with_installer(&index, current) else { return Ok(None) };
    let sums = fetch(&client, site.join(&release.sums).map_err(|e| e.to_string())?.as_str()).await?;
    let signature = fetch(&client, site.join(&release.signature).map_err(|e| e.to_string())?.as_str()).await?;
    let verified = verify_release(release.clone(), &sums, &String::from_utf8_lossy(&signature), signers)?;
    let name = installer_name(&verified.release.version).ok_or("no installer for this platform")?;
    let asset = verified.release.assets.iter().find(|a| a.name == name).cloned().ok_or("the installer vanished")?;
    Ok(Some((verified, asset)))
}

fn newest_with_installer<'a>(index: &'a Index, current: &str) -> Option<&'a Release> {
    index
        .releases
        .iter()
        .filter(|r| is_newer(&r.version, current) && installer_name(&r.version).is_some_and(|name| r.assets.iter().any(|a| a.name == name)))
        .reduce(|a, b| if is_newer(&b.version, &a.version) { b } else { a })
}

/// Checks `sums` against `signature` and `signers`, then `release`'s assets against `sums`.
pub fn verify_release(release: Release, sums: &[u8], signature: &str, signers: &str) -> Result<Verified, String> {
    let signer = verify_sshsig(sums, signature, NAMESPACE, signers)?;
    let sums = parse_sums(&String::from_utf8_lossy(sums))?;
    let tag = format!("-{}-", release.version);
    for name in sums.keys() {
        if !name.starts_with("Vergence-") || !name.contains(&tag) {
            return Err(format!("The signed manifest lists {name}, which isn't part of Vergence {}", release.version));
        }
    }
    for asset in &release.assets {
        match sums.get(&asset.name) {
            Some(hash) if hash.eq_ignore_ascii_case(&asset.sha256) => {}
            Some(_) => return Err(format!("release.json's hash for {} doesn't match the signed manifest", asset.name)),
            None => return Err(format!("{} isn't in the signed manifest", asset.name)),
        }
    }
    Ok(Verified { release, signer, sums })
}

/// Downloads `asset` into `dir`, trying its URLs in turn; the file only when its SHA-256 is the signed one.
pub async fn download(verified: &Verified, asset: &Asset, dir: &Path) -> Result<PathBuf, String> {
    let expected = verified.sums.get(&asset.name).ok_or_else(|| format!("{} isn't in the signed manifest", asset.name))?.to_ascii_lowercase();
    if asset.name.contains(['/', '\\']) || asset.name.starts_with('.') {
        return Err(format!("bad asset name {}", asset.name));
    }
    let client = http()?;
    let target = dir.join(&asset.name);
    let mut failures = Vec::new();
    for url in &asset.urls {
        match download_one(&client, url, &target, &expected, asset.size).await {
            Ok(()) => return Ok(target),
            Err(e) => {
                let _ = std::fs::remove_file(&target);
                failures.push(format!("{url}: {e}"));
            }
        }
    }
    Err(format!("Couldn't download {}: {}", asset.name, failures.join("; ")))
}

async fn download_one(client: &reqwest::Client, url: &str, path: &Path, expected: &str, size: u64) -> Result<(), String> {
    use std::io::Write;
    let mut response = client.get(url).send().await.map_err(|e| e.to_string())?;
    if !response.status().is_success() {
        return Err(response.status().to_string());
    }
    let mut file = std::fs::File::create(path).map_err(|e| e.to_string())?;
    let mut hash = Sha256::new();
    let mut received = 0u64;
    while let Some(chunk) = response.chunk().await.map_err(|e| e.to_string())? {
        received += chunk.len() as u64;
        if received > MAX_ASSET_BYTES || (size > 0 && received > size) {
            return Err(format!("sent more than the release's {size} bytes"));
        }
        hash.update(&chunk);
        file.write_all(&chunk).map_err(|e| e.to_string())?;
    }
    file.sync_all().map_err(|e| e.to_string())?;
    let got = hex(&hash.finalize());
    if got != expected {
        return Err(format!("its SHA-256 is {got}, not the signed {expected}"));
    }
    Ok(())
}

/// Replaces the running app with the verified package.
fn install_package(package: &Path) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        let executable = std::env::current_exe().map_err(|e| e.to_string())?;
        let bundle = executable
            .parent()
            .and_then(Path::parent)
            .and_then(Path::parent)
            .filter(|p| p.extension().is_some_and(|e| e == "app"))
            .ok_or("Only an installed Vergence.app updates itself.")?;
        install_mac(package, bundle)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = package;
        Err("Installing updates on this platform isn't ported yet: download the new version instead.".into())
    }
}

/// Unpacks the bundle beside the installed one, checks its code-signing seal, and swaps the
/// two atomically (`renamex_np` with `RENAME_SWAP`).
#[cfg(target_os = "macos")]
fn install_mac(package: &Path, bundle: &Path) -> Result<(), String> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    use std::process::Command;
    let parent = bundle.parent().ok_or("no app folder")?;
    let staged = tempfile::Builder::new()
        .prefix(".vergence-update-")
        .tempdir_in(parent)
        .map_err(|_| "Move Vergence to a writable Applications folder before updating".to_string())?;
    let unpacked = Command::new("/usr/bin/ditto").args(["-x", "-k"]).arg(package).arg(staged.path()).status().map_err(|e| e.to_string())?;
    if !unpacked.success() {
        return Err("Couldn't unpack the signed update".into());
    }
    let name = bundle.file_name().ok_or("no bundle name")?;
    let replacement = staged.path().join(name);
    if !replacement.join("Contents/Info.plist").is_file() {
        return Err(format!("The update has no {}", name.to_string_lossy()));
    }
    let sealed = Command::new("/usr/bin/codesign").args(["--verify", "--deep", "--strict"]).arg(&replacement).status().map_err(|e| e.to_string())?;
    if !sealed.success() {
        return Err("The updated app's code-signing seal is invalid".into());
    }
    let from = CString::new(replacement.as_os_str().as_bytes()).map_err(|e| e.to_string())?;
    let to = CString::new(bundle.as_os_str().as_bytes()).map_err(|e| e.to_string())?;
    if unsafe { libc::renamex_np(from.as_ptr(), to.as_ptr(), libc::RENAME_SWAP) } != 0 {
        return Err(format!("Couldn't exchange the app bundles: {}", std::io::Error::last_os_error()));
    }
    Ok(())
}

/// Whether dotted version `candidate` is newer than `current` ("0.10.0" > "0.9.2"); anything
/// after '-' or '+' is ignored.
pub fn is_newer(candidate: &str, current: &str) -> bool {
    let parts = |v: &str| -> Vec<u64> { v.split(['-', '+']).next().unwrap_or("").split('.').map(|p| p.parse().unwrap_or(0)).collect() };
    let (a, b) = (parts(candidate), parts(current));
    for i in 0..a.len().max(b.len()) {
        let (x, y) = (a.get(i).copied().unwrap_or(0), b.get(i).copied().unwrap_or(0));
        if x != y {
            return x > y;
        }
    }
    false
}

/// `sha256sum` output: `<hex>  <name>` per line. Name → lowercase hex.
pub fn parse_sums(text: &str) -> Result<BTreeMap<String, String>, String> {
    let mut sums = BTreeMap::new();
    for line in text.lines().map(str::trim_end).filter(|l| !l.is_empty()) {
        let (hash, name) = line.split_once(char::is_whitespace).ok_or_else(|| format!("bad SHA256SUMS line: {line}"))?;
        let name = name.trim_start().trim_start_matches('*');
        if hash.len() != 64 || !hash.bytes().all(|b| b.is_ascii_hexdigit()) || name.is_empty() {
            return Err(format!("bad SHA256SUMS line: {line}"));
        }
        if sums.insert(name.to_string(), hash.to_ascii_lowercase()).is_some() {
            return Err(format!("SHA256SUMS lists {name} twice"));
        }
    }
    if sums.is_empty() {
        return Err("SHA256SUMS is empty".into());
    }
    Ok(sums)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

// MARK: SSHSIG
//
// OpenSSH's PROTOCOL.sshsig. The armored signature is
//   "SSHSIG" | u32 version | str publickey | str namespace | str reserved | str hash | str signature
// and what the key signs is
//   "SSHSIG" | str namespace | str reserved | str hash | str H(message)
// with str a big-endian u32 length and the bytes. Only ed25519 keys are accepted.

struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.0.len() < n {
            return Err("truncated SSH signature".into());
        }
        let (head, rest) = self.0.split_at(n);
        self.0 = rest;
        Ok(head)
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_be_bytes(self.take(4)?.try_into().expect("four bytes")))
    }
    fn string(&mut self) -> Result<&'a [u8], String> {
        let n = self.u32()? as usize;
        self.take(n)
    }
}

fn ssh_string(bytes: &[u8]) -> Vec<u8> {
    let mut out = (bytes.len() as u32).to_be_bytes().to_vec();
    out.extend_from_slice(bytes);
    out
}

/// The ed25519 keys in an allowed_signers text: (key blob, principal or comment).
fn signer_keys(signers: &str) -> Vec<(Vec<u8>, String)> {
    signers
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .filter_map(|line| {
            let parts: Vec<&str> = line.split_whitespace().collect();
            let i = parts.iter().position(|p| *p == "ssh-ed25519")?;
            let blob = BASE64_STANDARD.decode(parts.get(i + 1)?).ok()?;
            Some((blob, parts[..i].join(" ")))
        })
        .collect()
}

/// Verifies a detached, armored SSHSIG over `data` made under `namespace` by one of `signers`,
/// and returns that signer's principal.
pub fn verify_sshsig(data: &[u8], armored: &str, namespace: &str, signers: &str) -> Result<String, String> {
    let body = armored
        .split("-----BEGIN SSH SIGNATURE-----")
        .nth(1)
        .and_then(|rest| rest.split("-----END SSH SIGNATURE-----").next())
        .ok_or("not an SSH signature")?;
    let blob = BASE64_STANDARD.decode(body.split_whitespace().collect::<String>()).map_err(|_| "the SSH signature is damaged")?;
    let mut r = Reader(&blob);
    if r.take(6)? != b"SSHSIG" {
        return Err("not an SSHSIG signature".into());
    }
    if r.u32()? != 1 {
        return Err("unsupported SSHSIG version".into());
    }
    let public_key = r.string()?;
    let signed_namespace = r.string()?;
    let reserved = r.string()?;
    let hash_name = r.string()?;
    let signature = r.string()?;
    if signed_namespace != namespace.as_bytes() {
        return Err(format!("signed for \"{}\", not \"{namespace}\"", String::from_utf8_lossy(signed_namespace)));
    }
    let digest = match hash_name {
        b"sha512" => Sha512::digest(data).to_vec(),
        b"sha256" => Sha256::digest(data).to_vec(),
        other => return Err(format!("unsupported SSHSIG hash {}", String::from_utf8_lossy(other))),
    };
    let signers = signer_keys(signers);
    let (_, signer) = signers.iter().find(|(key, _)| key == public_key).ok_or("signed by a key that isn't a pinned release signer")?;
    let mut key = Reader(public_key);
    if key.string()? != b"ssh-ed25519" {
        return Err("the signing key isn't ed25519".into());
    }
    let key: [u8; 32] = key.string()?.try_into().map_err(|_| "bad ed25519 key")?;
    let mut sig = Reader(signature);
    if sig.string()? != b"ssh-ed25519" {
        return Err("the signature isn't ed25519".into());
    }
    let sig: [u8; 64] = sig.string()?.try_into().map_err(|_| "bad ed25519 signature")?;
    let mut message = b"SSHSIG".to_vec();
    message.extend(ssh_string(signed_namespace));
    message.extend(ssh_string(reserved));
    message.extend(ssh_string(hash_name));
    message.extend(ssh_string(&digest));
    ed25519_dalek::VerifyingKey::from_bytes(&key)
        .map_err(|_| "bad ed25519 key")?
        .verify_strict(&message, &ed25519_dalek::Signature::from_bytes(&sig))
        .map_err(|_| "the release signature doesn't verify")?;
    Ok(signer.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    /// A throwaway key and ssh-keygen's signature over `data`; None without ssh-keygen.
    fn sign(dir: &Path, data: &[u8], namespace: &str) -> Option<(String, String)> {
        let key = dir.join("key");
        if !key.exists() {
            let made = Command::new("ssh-keygen").args(["-q", "-t", "ed25519", "-N", "", "-C", "test", "-f"]).arg(&key).status().ok()?;
            assert!(made.success());
        }
        let file = dir.join("SHA256SUMS");
        std::fs::write(&file, data).unwrap();
        let _ = std::fs::remove_file(dir.join("SHA256SUMS.sig"));
        let signed = Command::new("ssh-keygen").args(["-Y", "sign", "-q", "-n", namespace, "-f"]).arg(&key).arg(&file).status().ok()?;
        assert!(signed.success());
        let public = std::fs::read_to_string(dir.join("key.pub")).unwrap();
        Some((std::fs::read_to_string(dir.join("SHA256SUMS.sig")).unwrap(), format!("releases@test {public}")))
    }

    fn release(version: &str, assets: &[(&str, &str)]) -> Release {
        Release {
            version: version.into(),
            notes: None,
            sums: "SHA256SUMS".into(),
            signature: "SHA256SUMS.sig".into(),
            assets: assets.iter().map(|(n, h)| Asset { name: n.to_string(), size: 0, sha256: h.to_string(), urls: vec![] }).collect(),
        }
    }

    #[test]
    fn verifies_signed_manifests_only() {
        let dir = tempfile::tempdir().unwrap();
        let hash = "ab".repeat(32);
        let sums = format!("{hash}  Vergence-1.2.0-macos-universal.zip\n");
        let Some((signature, signers)) = sign(dir.path(), sums.as_bytes(), NAMESPACE) else {
            eprintln!("no ssh-keygen; skipped");
            return;
        };
        assert_eq!(verify_sshsig(sums.as_bytes(), &signature, NAMESPACE, &signers).unwrap(), "releases@test");
        assert!(verify_sshsig(b"other", &signature, NAMESPACE, &signers).is_err());
        assert!(verify_sshsig(sums.as_bytes(), &signature, "file", &signers).is_err());
        // A key that isn't pinned verifies nothing against the real signers.
        assert!(verify_sshsig(sums.as_bytes(), &signature, NAMESPACE, SIGNERS).is_err());

        let good = release("1.2.0", &[("Vergence-1.2.0-macos-universal.zip", &hash)]);
        assert!(verify_release(good, sums.as_bytes(), &signature, &signers).is_ok());
        let wrong = release("1.2.0", &[("Vergence-1.2.0-macos-universal.zip", &"cd".repeat(32))]);
        assert!(verify_release(wrong, sums.as_bytes(), &signature, &signers).is_err());
        // A genuine older manifest relabelled as newer.
        assert!(verify_release(release("1.3.0", &[]), sums.as_bytes(), &signature, &signers).is_err());
        let (apparition, _) = sign(dir.path(), sums.as_bytes(), "apparition-release").unwrap();
        assert!(verify_release(release("1.2.0", &[]), sums.as_bytes(), &apparition, &signers).is_err());
    }

    #[test]
    fn versions_sums_and_placeholders() {
        assert!(is_newer("0.10.0", "0.9.2"));
        assert!(!is_newer("0.1.0", "0.1.0"));
        assert!(!is_newer("0.1.0-beta", "0.1.0"));
        assert!(parse_sums("").is_err());
        assert!(parse_sums("xyz  a").is_err());
        let a = "0".repeat(64);
        assert!(parse_sums(&format!("{a}  x\n{a}  x\n")).is_err());
        assert_eq!(parse_sums(&format!("{a} *x\n")).unwrap()["x"], a);
        // The release keys are pinned (the hardware key and the offline backup); no site yet.
        assert!(SITE.is_none());
        assert!(signers_pinned());
        let pinned = signer_keys(SIGNERS);
        assert_eq!(pinned.len(), 2);
        assert!(pinned.iter().all(|(blob, principal)| blob.len() == 51 && principal == "releases@vergence"));
        let index = Index {
            format: "vergence-releases".into(),
            version: 1,
            releases: vec![release("9.0.0", &[]), release("9.1.0", &[(&installer_name("9.1.0").unwrap_or_default(), &a)])],
        };
        if installer_name("1").is_some() {
            assert_eq!(newest_with_installer(&index, "0.1.0").unwrap().version, "9.1.0");
            assert!(newest_with_installer(&index, "9.1.0").is_none());
        }
    }

    #[tokio::test]
    async fn checking_is_off_by_default_and_unconfigured_when_on() {
        let updates = Updates::default();
        assert_eq!(updates.check(false).await.unwrap().phase, "off");
        assert_eq!(updates.check(true).await.unwrap().phase, "unconfigured");
        assert!(updates.install().await.is_err());
    }
}
