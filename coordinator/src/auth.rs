//! Bearer tokens with grants, until pairing (RAVECompanion over TLS-PSK)
//! replaces them: `play` (fetch chunks, report the playhead), `work` (claim
//! pieces, upload results), `admin` (everything, the desktop app's own UI).
//! Only a token's SHA-256 is kept, in a JSON file readable by its owner only.

use std::path::{Path, PathBuf};

use rand::RngCore;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Grant {
    Play,
    Work,
    Admin,
}

impl Grant {
    pub fn parse(name: &str) -> Option<Grant> {
        match name {
            "play" => Some(Grant::Play),
            "work" => Some(Grant::Work),
            "admin" => Some(Grant::Admin),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TokenEntry {
    pub name: String,
    pub token_sha256: String,
    pub grants: Vec<Grant>,
}

#[derive(Clone, Debug, Default)]
pub struct Tokens {
    entries: Vec<TokenEntry>,
    /// Tokens made for this run only ([`Tokens::add_unsaved`]): never written, gone at a restart.
    unsaved: Vec<TokenEntry>,
    path: Option<PathBuf>,
}

fn hash(token: &str) -> String {
    Sha256::digest(token.as_bytes()).iter().map(|b| format!("{b:02x}")).collect()
}

/// Equal without returning early, so timing says nothing about the token.
fn same(a: &str, b: &str) -> bool {
    a.len() == b.len() && a.bytes().zip(b.bytes()).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

impl Tokens {
    pub fn load(path: &Path) -> std::io::Result<Tokens> {
        let entries = match std::fs::read(path) {
            Ok(bytes) => serde_json::from_slice(&bytes)?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(e) => return Err(e),
        };
        Ok(Tokens { entries, unsaved: Vec::new(), path: Some(path.to_path_buf()) })
    }

    pub fn in_memory(entries: Vec<TokenEntry>) -> Tokens {
        Tokens { entries, unsaved: Vec::new(), path: None }
    }

    pub fn entries(&self) -> &[TokenEntry] {
        &self.entries
    }

    /// Makes a token, keeps its hash and returns it (shown once).
    pub fn add(&mut self, name: &str, grants: Vec<Grant>) -> std::io::Result<String> {
        let mut bytes = [0u8; 24];
        rand::rng().fill_bytes(&mut bytes);
        let token: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
        self.entries.retain(|e| e.name != name);
        self.entries.push(TokenEntry { name: name.to_string(), token_sha256: hash(&token), grants });
        self.save()?;
        Ok(token)
    }

    /// Makes a token kept for this run only, its hash never written: what the desktop app gives
    /// its own worker, through the worker's standard input, at every start.
    pub fn add_unsaved(&mut self, name: &str, grants: Vec<Grant>) -> String {
        let mut bytes = [0u8; 24];
        rand::rng().fill_bytes(&mut bytes);
        let token: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
        self.unsaved.retain(|e| e.name != name);
        self.unsaved.push(TokenEntry { name: name.to_string(), token_sha256: hash(&token), grants });
        token
    }

    pub fn remove(&mut self, name: &str) -> std::io::Result<bool> {
        let before = self.entries.len() + self.unsaved.len();
        self.entries.retain(|e| e.name != name);
        self.unsaved.retain(|e| e.name != name);
        self.save()?;
        Ok(self.entries.len() + self.unsaved.len() != before)
    }

    fn save(&self) -> std::io::Result<()> {
        let Some(path) = &self.path else { return Ok(()) };
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(&self.entries)?)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600))?;
        }
        std::fs::rename(tmp, path)
    }

    /// The token's name, if it holds `grant` (admin holds every grant).
    pub fn check(&self, token: &str, grant: Grant) -> Option<&str> {
        let digest = hash(token);
        self.entries
            .iter()
            .chain(&self.unsaved)
            .find(|e| same(&e.token_sha256, &digest))
            .filter(|e| e.grants.contains(&grant) || e.grants.contains(&Grant::Admin))
            .map(|e| e.name.as_str())
    }
}

/// The token from `Authorization: Bearer …`, or from Jellyfin's own
/// `Authorization: MediaBrowser …, Token="…"` (what a player already sends to
/// the plugin's routes), or `X-Emby-Token`.
pub fn token_from_headers(headers: &axum::http::HeaderMap) -> Option<String> {
    if let Some(value) = headers.get(axum::http::header::AUTHORIZATION).and_then(|v| v.to_str().ok()) {
        if let Some(token) = value.strip_prefix("Bearer ") {
            return Some(token.trim().to_string());
        }
        if let Some(start) = value.find("Token=\"") {
            let rest = &value[start + 7..];
            if let Some(end) = rest.find('"') {
                return Some(rest[..end].to_string());
            }
        }
    }
    headers.get("X-Emby-Token").and_then(|v| v.to_str().ok()).map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_hold_their_grants() {
        let dir = tempfile::tempdir().unwrap();
        let mut tokens = Tokens::load(&dir.path().join("tokens.json")).unwrap();
        let play = tokens.add("vision-pro", vec![Grant::Play]).unwrap();
        let admin = tokens.add("desktop", vec![Grant::Admin]).unwrap();
        let reloaded = Tokens::load(&dir.path().join("tokens.json")).unwrap();
        assert_eq!(reloaded.check(&play, Grant::Play), Some("vision-pro"));
        assert_eq!(reloaded.check(&play, Grant::Work), None);
        assert_eq!(reloaded.check(&admin, Grant::Work), Some("desktop"));
        assert_eq!(reloaded.check("nope", Grant::Play), None);
        assert!(!std::fs::read_to_string(dir.path().join("tokens.json")).unwrap().contains(&play));
    }

    #[test]
    fn unsaved_tokens_work_and_never_reach_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokens.json");
        let mut tokens = Tokens::load(&path).unwrap();
        tokens.add("script", vec![Grant::Play]).unwrap();
        let worker = tokens.add_unsaved("worker", vec![Grant::Work]);
        assert_eq!(tokens.check(&worker, Grant::Work), Some("worker"));
        let written = std::fs::read_to_string(&path).unwrap();
        assert!(!written.contains("worker") && !written.contains(&worker));
        assert_eq!(Tokens::load(&path).unwrap().check(&worker, Grant::Work), None);
        assert!(tokens.remove("worker").unwrap());
        assert_eq!(tokens.check(&worker, Grant::Work), None);
    }

    #[test]
    fn reads_jellyfin_style_headers() {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert("authorization", "MediaBrowser Client=\"x\", Token=\"abc\"".parse().unwrap());
        assert_eq!(token_from_headers(&headers).as_deref(), Some("abc"));
        headers.insert("authorization", "Bearer def".parse().unwrap());
        assert_eq!(token_from_headers(&headers).as_deref(), Some("def"));
    }
}
