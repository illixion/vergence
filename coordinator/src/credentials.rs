//! Media-server credentials a player hands over, held in memory only.
//!
//! A headset holds its own Jellyfin and Stash sign-ins (shared between the
//! person's devices with Device Setup), and gives the coordinator the one it
//! plays with when it starts a stream (`credentials` on `POST /v1/playing`).
//! The coordinator keeps it in this process only, never on disk, refreshes it
//! with every stream start, and forgets it [`HELD_FOR`] after the last one, so
//! a quit, a restart or a quiet day leaves no token behind. A worker gets the
//! one for its piece's server in its lease, over the same kinds of channel the
//! credentials may arrive on (the paired TLS-PSK channel or loopback).

use std::collections::HashMap;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::source::Server;

/// How long a handed-over credential is kept after the stream that brought it.
pub const HELD_FOR: Duration = Duration::from_secs(12 * 3600);

/// One server's address and key, as it travels: on a player's report, and to a
/// worker in a lease. The key is a Jellyfin access token or API key, or a Stash
/// API key.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServerCredential {
    pub url: String,
    pub key: String,
}

impl std::fmt::Debug for ServerCredential {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServerCredential").field("url", &self.url).field("key", &"<redacted>").finish()
    }
}

impl ServerCredential {
    /// The checks a credential passes before it is kept.
    pub fn check(&self) -> Result<(), &'static str> {
        let url = self.url.trim();
        if !(url.starts_with("http://") || url.starts_with("https://")) || url.len() > 2048 {
            return Err("credentials: the server's url must be http(s), at most 2048 bytes");
        }
        if self.key.trim().is_empty() || self.key.len() > 4096 || self.key.chars().any(char::is_control) {
            return Err("credentials: the key must be 1 to 4096 bytes, without control characters");
        }
        Ok(())
    }
}

/// What a player may send: one credential per server it plays from.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HandedOver {
    #[serde(default)]
    pub jellyfin: Option<ServerCredential>,
    #[serde(default)]
    pub stash: Option<ServerCredential>,
}

impl HandedOver {
    pub fn entries(self) -> impl Iterator<Item = (Server, ServerCredential)> {
        [(Server::Jellyfin, self.jellyfin), (Server::Stash, self.stash)].into_iter().filter_map(|(s, c)| Some((s, c?)))
    }
}

struct Held {
    credential: ServerCredential,
    from: String,
    at: Instant,
}

/// The credentials held now, one per server.
#[derive(Default)]
pub struct MediaCredentials {
    held: HashMap<Server, Held>,
}

/// What the app's window may show of a held credential: never the key.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HeldSummary {
    pub server: Server,
    pub url: String,
    /// Who handed it over: a paired device's name, or "this computer".
    pub from: String,
    pub seconds_ago: u64,
    pub seconds_left: u64,
}

impl MediaCredentials {
    /// Keeps (or refreshes) `server`'s credential.
    pub fn put(&mut self, server: Server, credential: ServerCredential, from: &str, now: Instant) {
        self.held.insert(server, Held { credential, from: from.to_string(), at: now });
    }

    /// `server`'s credential while it is still held.
    pub fn get(&mut self, server: Server, now: Instant) -> Option<ServerCredential> {
        self.expire(now);
        self.held.get(&server).map(|h| h.credential.clone())
    }

    pub fn has(&mut self, server: Server, now: Instant) -> bool {
        self.get(server, now).is_some()
    }

    pub fn forget(&mut self, server: Option<Server>) {
        match server {
            Some(server) => {
                self.held.remove(&server);
            }
            None => self.held.clear(),
        }
    }

    pub fn summary(&mut self, now: Instant) -> Vec<HeldSummary> {
        self.expire(now);
        let mut list: Vec<HeldSummary> = self
            .held
            .iter()
            .map(|(server, h)| {
                let ago = now.saturating_duration_since(h.at);
                HeldSummary {
                    server: *server,
                    url: h.credential.url.clone(),
                    from: h.from.clone(),
                    seconds_ago: ago.as_secs(),
                    seconds_left: HELD_FOR.saturating_sub(ago).as_secs(),
                }
            })
            .collect();
        list.sort_by_key(|s| s.server);
        list
    }

    fn expire(&mut self, now: Instant) {
        self.held.retain(|_, h| now.saturating_duration_since(h.at) < HELD_FOR);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cred(key: &str) -> ServerCredential {
        ServerCredential { url: "https://media.example/jellyfin".into(), key: key.into() }
    }

    #[test]
    fn held_until_they_expire_and_refreshed_by_each_stream() {
        let mut held = MediaCredentials::default();
        let start = Instant::now();
        held.put(Server::Jellyfin, cred("a"), "vp", start);
        assert_eq!(held.get(Server::Jellyfin, start + HELD_FOR / 2).unwrap().key, "a");
        held.put(Server::Jellyfin, cred("b"), "vp", start + HELD_FOR / 2);
        assert_eq!(held.get(Server::Jellyfin, start + HELD_FOR).unwrap().key, "b");
        assert!(held.get(Server::Jellyfin, start + HELD_FOR * 2).is_none());
        assert!(held.get(Server::Stash, start).is_none());
    }

    #[test]
    fn checks_and_never_prints_the_key() {
        assert!(cred("k").check().is_ok());
        assert!(ServerCredential { url: "file:///etc".into(), key: "k".into() }.check().is_err());
        assert!(cred("").check().is_err());
        assert!(cred("a\nb").check().is_err());
        assert!(!format!("{:?}", cred("secret-key")).contains("secret-key"));
    }
}
