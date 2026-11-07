//! The coordinator's worker API: claim a piece, keep its lease, give it back,
//! upload the result. What it hands out is its `Lease`, read here as far as
//! the worker needs it (as `vergence worker`'s `WorkLease` does).
//!
//! ```text
//! POST /v1/work/claim                          {worker, remote, models} → a lease, or 204
//! POST /v1/work/leases/{lease}/heartbeat?worker=NAME
//! POST /v1/work/leases/{lease}/release?worker=NAME
//! PUT  /v1/work/leases/{lease}/result?worker=NAME   the track
//! ```

use std::collections::BTreeMap;
use std::time::Duration;

use serde::Deserialize;
use serde_json::json;

#[derive(Clone, Debug, Deserialize)]
pub struct Source {
    pub server: String,
    pub id: String,
}

#[derive(Clone, Copy, Debug, Deserialize)]
pub struct Rate {
    pub num: i64,
    pub den: i64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Info {
    pub frame_rate: Rate,
    pub width: usize,
    pub height: usize,
    pub codec: String,
    pub stereo: String,
    #[serde(default)]
    pub media_source_id: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Lease {
    pub lease: u64,
    pub title: String,
    pub source: Source,
    /// An older coordinator names none: Smooth Motion for 3D titles was all it had.
    #[serde(default = "default_workload")]
    pub workload: String,
    pub kinds: Vec<String>,
    pub piece: usize,
    pub first_frame: i64,
    pub end_frame: i64,
    pub frame_count: i64,
    pub info: Info,
    pub lease_seconds: f64,
    pub live: bool,
}

fn default_workload() -> String {
    "smooth-motion-3d".into()
}

#[derive(Clone)]
pub struct Coordinator {
    base: String,
    token: String,
    name: String,
    http: reqwest::Client,
}

impl Coordinator {
    pub fn new(base: &str, token: String, name: String) -> Result<Coordinator, String> {
        let http = reqwest::Client::builder()
            .user_agent(concat!("vergence-worker/", env!("CARGO_PKG_VERSION")))
            .timeout(Duration::from_secs(60))
            .build()
            .map_err(|e| e.to_string())?;
        Ok(Coordinator { base: base.trim_end_matches('/').to_string(), token, name, http })
    }

    fn lease_url(&self, lease: u64, action: &str) -> String {
        format!("{}/v1/work/leases/{lease}/{action}", self.base)
    }

    /// `profiles`: each workload this worker takes, with its models per kind.
    /// `continues`: the title and frame this worker's last piece ended at,
    /// when it keeps state for the next one (a sticky run).
    pub async fn claim(&self, remote: bool, profiles: &BTreeMap<String, BTreeMap<String, String>>,
                       continues: Option<&(String, i64)>) -> Result<Option<Lease>, String> {
        // `models` for a coordinator that knows only one workload.
        let models = profiles.get("smooth-motion-3d").or_else(|| profiles.values().next()).cloned().unwrap_or_default();
        let workloads: Vec<&String> = profiles.keys().collect();
        let response = self
            .http
            .post(format!("{}/v1/work/claim", self.base))
            .bearer_auth(&self.token)
            .json(&json!({ "worker": self.name, "remote": remote, "models": models, "workloads": workloads,
                           "workloadModels": profiles,
                           "continues": continues.map(|(title, frame)| json!({ "title": title, "frame": frame })) }))
            .send()
            .await
            .map_err(|e| e.to_string())?;
        match response.status().as_u16() {
            200 => response.json::<Lease>().await.map(Some).map_err(|e| format!("the lease didn't decode: {e}")),
            204 => Ok(None),
            status => Err(format!("the coordinator answered {status}: {}", response.text().await.unwrap_or_default())),
        }
    }

    async fn post(&self, lease: u64, action: &str) -> Result<u16, String> {
        let response = self
            .http
            .post(self.lease_url(lease, action))
            .query(&[("worker", &self.name)])
            .bearer_auth(&self.token)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        Ok(response.status().as_u16())
    }

    pub async fn heartbeat(&self, lease: u64) -> Result<u16, String> {
        self.post(lease, "heartbeat").await
    }

    pub async fn release(&self, lease: u64) -> Result<u16, String> {
        self.post(lease, "release").await
    }

    pub async fn upload(&self, lease: u64, track: Vec<u8>) -> Result<(), String> {
        let response = self
            .http
            .put(self.lease_url(lease, "result"))
            .query(&[("worker", &self.name)])
            .bearer_auth(&self.token)
            .header("Content-Type", "application/octet-stream")
            .body(track)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        let status = response.status().as_u16();
        if status != 200 {
            return Err(format!("the result was refused ({status}): {}", response.text().await.unwrap_or_default()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_the_coordinators_lease() {
        use vergence_coordinator::queue::Lease as Theirs;
        use vergence_coordinator::source::{MediaInfo, Rate, SourceRef, StereoLayout};
        use vergence_coordinator::track::Kind;
        let theirs = Theirs {
            lease: 7,
            title: "jellyfin:abc:smooth-motion-3d".into(),
            source: SourceRef::jellyfin("abc"),
            workload: "smooth-motion-3d".into(),
            kinds: vec![Kind::FlowPair, Kind::Disparity],
            piece: 3,
            start_seconds: 6.0,
            end_seconds: 8.0,
            first_frame: 144,
            end_frame: 192,
            frame_count: 1440,
            info: MediaInfo {
                duration_seconds: 60.06,
                frame_rate: Rate { num: 24000, den: 1001 },
                width: 1920,
                height: 1080,
                codec: "h264".into(),
                stereo: StereoLayout::HalfSideBySide,
                identity: "e".into(),
                media_source_id: Some("abc".into()),
            },
            lease_seconds: 120.0,
            live: true,
        };
        let lease: Lease = serde_json::from_str(&serde_json::to_string(&theirs).unwrap()).unwrap();
        assert_eq!((lease.lease, lease.piece, lease.first_frame, lease.end_frame), (7, 3, 144, 192));
        assert_eq!((lease.source.server.as_str(), lease.source.id.as_str()), ("jellyfin", "abc"));
        assert_eq!(lease.kinds, ["flowPair", "disparity"]);
        assert_eq!((lease.info.frame_rate.num, lease.info.frame_rate.den), (24000, 1001));
        assert_eq!(lease.info.stereo, "halfSideBySide");
        assert_eq!(lease.info.media_source_id.as_deref(), Some("abc"));
        assert!(lease.live);
    }
}
