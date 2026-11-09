//! Source adapters: what the coordinator needs to know about a title on a
//! media server, read with the coordinator's own credentials. Workers read
//! the media themselves, with theirs; nothing here hands out stream URLs.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
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

    pub fn from_name(name: &str) -> Option<Server> {
        match name {
            "jellyfin" => Some(Server::Jellyfin),
            "stash" => Some(Server::Stash),
            _ => None,
        }
    }
}

/// A title on a media server: a Jellyfin item id, or a Stash scene id.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SourceRef {
    pub server: Server,
    pub id: String,
}

impl SourceRef {
    pub fn jellyfin(id: impl Into<String>) -> Self {
        SourceRef { server: Server::Jellyfin, id: normalize_jellyfin_id(&id.into()) }
    }

    pub fn stash(id: impl Into<String>) -> Self {
        SourceRef { server: Server::Stash, id: id.into() }
    }
}

/// Jellyfin writes item ids with and without dashes; the coordinator keeps
/// them without, lower case, as the server's JSON does.
pub fn normalize_jellyfin_id(id: &str) -> String {
    id.chars().filter(|c| *c != '-').flat_map(char::to_lowercase).collect()
}

/// How the two eyes are stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum StereoLayout {
    Mono,
    HalfSideBySide,
    FullSideBySide,
    HalfTopAndBottom,
    FullTopAndBottom,
    /// A Blu-ray's MVC: decoded from the disc by vergence itself.
    Mvc,
}

impl StereoLayout {
    pub fn is_stereo(self) -> bool {
        self != StereoLayout::Mono
    }

    /// One eye's stored size, given the coded picture's.
    pub fn eye_size(self, width: u32, height: u32) -> (u32, u32) {
        match self {
            StereoLayout::HalfSideBySide | StereoLayout::FullSideBySide => (width / 2, height),
            StereoLayout::HalfTopAndBottom | StereoLayout::FullTopAndBottom => (width, height / 2),
            StereoLayout::Mono | StereoLayout::Mvc => (width, height),
        }
    }

    /// Jellyfin's `Video3DFormat`.
    pub fn from_jellyfin(format: &str) -> StereoLayout {
        match format {
            "HalfSideBySide" => StereoLayout::HalfSideBySide,
            "FullSideBySide" => StereoLayout::FullSideBySide,
            "HalfTopAndBottom" => StereoLayout::HalfTopAndBottom,
            "FullTopAndBottom" => StereoLayout::FullTopAndBottom,
            "MVC" => StereoLayout::Mvc,
            _ => StereoLayout::Mono,
        }
    }

    /// The layout a file name or tag says, following the naming Jellyfin and
    /// players read: a `3D` token, then `HSBS`, `FSBS`, `SBS` (half), `HTAB`,
    /// `FTAB`, `TAB` or `OU` (half), e.g. `Film (2009).3D.HSBS.mkv`.
    pub fn from_name(name: &str) -> StereoLayout {
        let tokens: Vec<String> = name
            .split(|c: char| !c.is_ascii_alphanumeric())
            .filter(|t| !t.is_empty())
            .map(|t| t.to_ascii_lowercase())
            .collect();
        for (i, token) in tokens.iter().enumerate() {
            if token != "3d" {
                continue;
            }
            let next = tokens.get(i + 1).map(String::as_str);
            let layout = match next {
                Some("hsbs") | Some("sbs") => StereoLayout::HalfSideBySide,
                Some("fsbs") => StereoLayout::FullSideBySide,
                Some("htab") | Some("tab") | Some("ou") | Some("hou") => StereoLayout::HalfTopAndBottom,
                Some("ftab") | Some("fou") => StereoLayout::FullTopAndBottom,
                _ => continue,
            };
            return layout;
        }
        StereoLayout::Mono
    }
}

/// A frame rate as a rational, snapped to the broadcast rates a float like
/// Jellyfin's 23.976 stands for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rate {
    pub num: u32,
    pub den: u32,
}

impl Rate {
    pub fn from_float(rate: f64) -> Rate {
        for (num, den) in [(24000, 1001), (30000, 1001), (60000, 1001), (48000, 1001), (120000, 1001)] {
            if (rate - num as f64 / den as f64).abs() < 0.005 {
                return Rate { num, den };
            }
        }
        if (rate - rate.round()).abs() < 0.005 && rate >= 1.0 {
            return Rate { num: rate.round() as u32, den: 1 };
        }
        Rate { num: (rate * 1000.0).round().max(1.0) as u32, den: 1000 }
    }

    pub fn fps(self) -> f64 {
        self.num as f64 / self.den as f64
    }
}

/// What the coordinator knows about a title.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MediaInfo {
    pub duration_seconds: f64,
    pub frame_rate: Rate,
    /// The coded picture's size.
    #[serde(default)]
    pub width: u32,
    #[serde(default)]
    pub height: u32,
    #[serde(default)]
    pub codec: String,
    pub stereo: StereoLayout,
    /// Changes when the file behind the title is replaced (Jellyfin's
    /// ETag, Stash's oshash): every result made from the old file is dropped.
    pub identity: String,
    /// Jellyfin's media source id, for a worker's stream request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_source_id: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum SourceError {
    #[error("no {0} server is configured")]
    NotConfigured(&'static str),
    #[error("the server has no such title")]
    NotFound,
    #[error("the title has no video stream")]
    NoVideo,
    #[error("the server answered {0}")]
    Status(u16),
    #[error("the server couldn't be reached: {0}")]
    Http(#[from] reqwest::Error),
    #[error("the server's answer was unexpected: {0}")]
    Shape(String),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ServerConfig {
    pub url: String,
    #[serde(default)]
    pub api_key: Option<String>,
    /// A file holding the key, read at start (kept out of the config file).
    #[serde(default)]
    pub api_key_file: Option<String>,
}

impl ServerConfig {
    pub fn key(&self) -> std::io::Result<Option<String>> {
        if let Some(key) = &self.api_key {
            return Ok(Some(key.clone()));
        }
        match &self.api_key_file {
            Some(path) => Ok(Some(std::fs::read_to_string(path)?.trim().to_string())),
            None => Ok(None),
        }
    }
}

#[derive(Clone)]
pub struct Jellyfin {
    base: String,
    key: String,
    http: reqwest::Client,
}

impl Jellyfin {
    pub fn new(config: &ServerConfig, http: reqwest::Client) -> std::io::Result<Jellyfin> {
        let key = config.key()?.ok_or_else(|| std::io::Error::other("Jellyfin needs an API key"))?;
        Ok(Jellyfin { base: config.url.trim_end_matches('/').to_string(), key, http })
    }

    fn get(&self, path_and_query: &str) -> reqwest::RequestBuilder {
        self.http
            .get(format!("{}/{}", self.base, path_and_query))
            .header("Authorization", format!("MediaBrowser Client=\"vergenced\", Token=\"{}\"", self.key))
    }

    async fn items(&self, query: &str) -> Result<Vec<Value>, SourceError> {
        let response = self.get(&format!("Items?{query}")).send().await?;
        if !response.status().is_success() {
            return Err(SourceError::Status(response.status().as_u16()));
        }
        let body: Value = response.json().await?;
        Ok(body["Items"].as_array().cloned().unwrap_or_default())
    }

    pub async fn describe(&self, id: &str) -> Result<MediaInfo, SourceError> {
        let items = self.items(&format!("Ids={id}&Fields=MediaSources,MediaStreams")).await?;
        let item = items.first().ok_or(SourceError::NotFound)?;
        Self::info(item)
    }

    /// The media info from an item's JSON (`Fields=MediaSources`).
    pub fn info(item: &Value) -> Result<MediaInfo, SourceError> {
        let source = item["MediaSources"].get(0).ok_or(SourceError::NoVideo)?;
        let video = source["MediaStreams"]
            .as_array()
            .and_then(|s| s.iter().find(|m| m["Type"] == "Video"))
            .ok_or(SourceError::NoVideo)?;
        let ticks = item["RunTimeTicks"].as_i64().or_else(|| source["RunTimeTicks"].as_i64()).unwrap_or(0);
        let rate = video["RealFrameRate"].as_f64().or_else(|| video["AverageFrameRate"].as_f64()).unwrap_or(24000.0 / 1001.0);
        let stereo = item["Video3DFormat"]
            .as_str()
            .map(StereoLayout::from_jellyfin)
            .unwrap_or_else(|| StereoLayout::from_name(source["Path"].as_str().unwrap_or("")));
        Ok(MediaInfo {
            duration_seconds: ticks as f64 / 10_000_000.0,
            frame_rate: Rate::from_float(rate),
            width: video["Width"].as_u64().unwrap_or(0) as u32,
            height: video["Height"].as_u64().unwrap_or(0) as u32,
            codec: video["Codec"].as_str().unwrap_or("").to_string(),
            stereo,
            identity: source["ETag"]
                .as_str()
                .map(str::to_string)
                .unwrap_or_else(|| format!("{}:{}", source["Size"], ticks)),
            media_source_id: source["Id"].as_str().map(str::to_string),
        })
    }

    /// Every 3D film and episode, for backfill.
    pub async fn list_3d(&self) -> Result<Vec<String>, SourceError> {
        let items = self
            .items("Recursive=true&IncludeItemTypes=Movie,Episode&Is3D=true&Fields=MediaSources&IsVirtualItem=false")
            .await?;
        Ok(items.iter().filter_map(|i| i["Id"].as_str().map(normalize_jellyfin_id)).collect())
    }
}

#[derive(Clone)]
pub struct Stash {
    base: String,
    key: Option<String>,
    http: reqwest::Client,
}

const STASH_SCENE_FIELDS: &str =
    "id files { basename fingerprints { type value } frame_rate duration width height video_codec } tags { name }";

impl Stash {
    pub fn new(config: &ServerConfig, http: reqwest::Client) -> std::io::Result<Stash> {
        Ok(Stash { base: config.url.trim_end_matches('/').to_string(), key: config.key()?, http })
    }

    async fn query(&self, query: &str, variables: Value) -> Result<Value, SourceError> {
        let mut request = self.http.post(format!("{}/graphql", self.base)).json(&json!({"query": query, "variables": variables}));
        if let Some(key) = &self.key {
            request = request.header("ApiKey", key);
        }
        let response = request.send().await?;
        if !response.status().is_success() {
            return Err(SourceError::Status(response.status().as_u16()));
        }
        let body: Value = response.json().await?;
        if let Some(errors) = body.get("errors") {
            return Err(SourceError::Shape(errors.to_string()));
        }
        Ok(body["data"].clone())
    }

    pub async fn describe(&self, id: &str) -> Result<MediaInfo, SourceError> {
        let data = self
            .query(&format!("query($id: ID!) {{ findScene(id: $id) {{ {STASH_SCENE_FIELDS} }} }}"), json!({"id": id}))
            .await?;
        let scene = &data["findScene"];
        if scene.is_null() {
            return Err(SourceError::NotFound);
        }
        Self::info(scene)
    }

    /// The media info from a scene's JSON. A scene can have several files;
    /// the first is the one Stash streams.
    pub fn info(scene: &Value) -> Result<MediaInfo, SourceError> {
        let file = scene["files"].get(0).ok_or(SourceError::NoVideo)?;
        let oshash = file["fingerprints"]
            .as_array()
            .and_then(|f| f.iter().find(|p| p["type"] == "oshash"))
            .and_then(|p| p["value"].as_str())
            .ok_or_else(|| SourceError::Shape("the file has no oshash".into()))?;
        // Stash has no stereo field: a tag ("3D HSBS") or the file's name says it.
        let mut stereo = StereoLayout::from_name(file["basename"].as_str().unwrap_or(""));
        if stereo == StereoLayout::Mono
            && let Some(tags) = scene["tags"].as_array()
        {
            stereo = tags
                .iter()
                .filter_map(|t| t["name"].as_str())
                .map(StereoLayout::from_name)
                .find(|l| l.is_stereo())
                .unwrap_or(StereoLayout::Mono);
        }
        Ok(MediaInfo {
            duration_seconds: file["duration"].as_f64().unwrap_or(0.0),
            frame_rate: Rate::from_float(file["frame_rate"].as_f64().unwrap_or(30.0)),
            width: file["width"].as_u64().unwrap_or(0) as u32,
            height: file["height"].as_u64().unwrap_or(0) as u32,
            codec: file["video_codec"].as_str().unwrap_or("").to_string(),
            stereo,
            identity: oshash.to_string(),
            media_source_id: None,
        })
    }

    /// Every scene that is stereo by its name or tags, for backfill.
    pub async fn list_3d(&self) -> Result<Vec<String>, SourceError> {
        let mut ids = Vec::new();
        let mut page = 1;
        loop {
            let data = self
                .query(
                    &format!(
                        "query($page: Int!) {{ findScenes(filter: {{ page: $page, per_page: 200, sort: \"id\" }}) {{ count scenes {{ {STASH_SCENE_FIELDS} }} }} }}"
                    ),
                    json!({"page": page}),
                )
                .await?;
            let scenes = data["findScenes"]["scenes"].as_array().cloned().unwrap_or_default();
            if scenes.is_empty() {
                break;
            }
            for scene in &scenes {
                if let (Some(id), Ok(info)) = (scene["id"].as_str(), Self::info(scene))
                    && info.stereo.is_stereo()
                {
                    ids.push(id.to_string());
                }
            }
            page += 1;
        }
        Ok(ids)
    }
}

/// The configured servers.
#[derive(Clone, Default)]
pub struct Sources {
    pub jellyfin: Option<Jellyfin>,
    pub stash: Option<Stash>,
}

/// The HTTP client every source adapter uses.
pub fn http_client() -> std::io::Result<reqwest::Client> {
    reqwest::Client::builder()
        .user_agent(concat!("vergenced/", env!("CARGO_PKG_VERSION")))
        .timeout(std::time::Duration::from_secs(20))
        .build()
        .map_err(std::io::Error::other)
}

impl Sources {
    /// Describes a title with a credential a player handed over
    /// ([`crate::credentials`]), for a server the configuration names no key for.
    pub async fn describe_with(source: &SourceRef, credential: &crate::credentials::ServerCredential) -> Result<MediaInfo, SourceError> {
        let config = ServerConfig { url: credential.url.clone(), api_key: Some(credential.key.clone()), api_key_file: None };
        let http = http_client().map_err(|e| SourceError::Shape(e.to_string()))?;
        match source.server {
            Server::Jellyfin => Jellyfin::new(&config, http).map_err(|e| SourceError::Shape(e.to_string()))?.describe(&source.id).await,
            Server::Stash => Stash::new(&config, http).map_err(|e| SourceError::Shape(e.to_string()))?.describe(&source.id).await,
        }
    }

    /// Whether the coordinator holds a key for `server` and so describes its titles itself.
    pub fn has(&self, server: Server) -> bool {
        match server {
            Server::Jellyfin => self.jellyfin.is_some(),
            Server::Stash => self.stash.is_some(),
        }
    }

    pub async fn describe(&self, source: &SourceRef) -> Result<MediaInfo, SourceError> {
        match source.server {
            Server::Jellyfin => self.jellyfin.as_ref().ok_or(SourceError::NotConfigured("Jellyfin"))?.describe(&source.id).await,
            Server::Stash => self.stash.as_ref().ok_or(SourceError::NotConfigured("Stash"))?.describe(&source.id).await,
        }
    }

    /// These sources, with a handed-over credential standing in for each server the
    /// configuration names no key for (so a keyless coordinator can still scan while it holds one).
    pub fn or_held(
        &self,
        jellyfin: Option<crate::credentials::ServerCredential>,
        stash: Option<crate::credentials::ServerCredential>,
    ) -> std::io::Result<Sources> {
        let config = |c: crate::credentials::ServerCredential| ServerConfig { url: c.url, api_key: Some(c.key), api_key_file: None };
        let jellyfin = match (&self.jellyfin, jellyfin) {
            (Some(j), _) => Some(j.clone()),
            (None, Some(c)) => Some(Jellyfin::new(&config(c), http_client()?)?),
            (None, None) => None,
        };
        let stash = match (&self.stash, stash) {
            (Some(s), _) => Some(s.clone()),
            (None, Some(c)) => Some(Stash::new(&config(c), http_client()?)?),
            (None, None) => None,
        };
        Ok(Sources { jellyfin, stash })
    }

    pub async fn list_3d(&self) -> Vec<(SourceRef, Result<(), SourceError>)> {
        let mut out = Vec::new();
        if let Some(j) = &self.jellyfin {
            match j.list_3d().await {
                Ok(ids) => out.extend(ids.into_iter().map(|id| (SourceRef::jellyfin(id), Ok(())))),
                Err(e) => out.push((SourceRef::jellyfin(""), Err(e))),
            }
        }
        if let Some(s) = &self.stash {
            match s.list_3d().await {
                Ok(ids) => out.extend(ids.into_iter().map(|id| (SourceRef::stash(id), Ok(())))),
                Err(e) => out.push((SourceRef::stash(""), Err(e))),
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_say_the_layout() {
        assert_eq!(StereoLayout::from_name("Film (2009).3D.HSBS.mkv"), StereoLayout::HalfSideBySide);
        assert_eq!(StereoLayout::from_name("film.3d.fsbs.mp4"), StereoLayout::FullSideBySide);
        assert_eq!(StereoLayout::from_name("Film 3D-TAB"), StereoLayout::HalfTopAndBottom);
        assert_eq!(StereoLayout::from_name("3D FTAB"), StereoLayout::FullTopAndBottom);
        assert_eq!(StereoLayout::from_name("Film (2009).mkv"), StereoLayout::Mono);
        assert_eq!(StereoLayout::from_name("SBS without the 3D token"), StereoLayout::Mono);
    }

    #[test]
    fn rates_snap_to_broadcast_ones() {
        assert_eq!(Rate::from_float(23.976), Rate { num: 24000, den: 1001 });
        assert_eq!(Rate::from_float(29.97003), Rate { num: 30000, den: 1001 });
        assert_eq!(Rate::from_float(25.0), Rate { num: 25, den: 1 });
        assert_eq!(Rate::from_float(12.5), Rate { num: 12500, den: 1000 });
    }

    #[test]
    fn jellyfin_items_read() {
        let item = json!({
            "Id": "289477b1bdb872265402628924353285", "RunTimeTicks": 6000230000i64, "Video3DFormat": "HalfSideBySide",
            "MediaSources": [{"Id": "289477b1bdb872265402628924353285", "ETag": "697d", "Path": "/m/a.mkv",
                "MediaStreams": [{"Type": "Audio"}, {"Type": "Video", "RealFrameRate": 23.976, "Width": 1920, "Height": 1080, "Codec": "h264"}]}]
        });
        let info = Jellyfin::info(&item).unwrap();
        assert_eq!(info.stereo, StereoLayout::HalfSideBySide);
        assert_eq!(info.frame_rate, Rate { num: 24000, den: 1001 });
        assert_eq!(info.identity, "697d");
        assert!((info.duration_seconds - 600.023).abs() < 1e-6);
        assert_eq!(info.stereo.eye_size(info.width, info.height), (960, 1080));
    }

    #[test]
    fn stash_scenes_read() {
        let scene = json!({"id": "6", "tags": [{"name": "3D HSBS"}], "files": [{"basename": "clip.mkv",
            "fingerprints": [{"type": "md5", "value": "x"}, {"type": "oshash", "value": "f058"}],
            "frame_rate": 24, "duration": 300.2, "width": 1920, "height": 1080, "video_codec": "h264"}]});
        let info = Stash::info(&scene).unwrap();
        assert_eq!(info.identity, "f058");
        assert_eq!(info.stereo, StereoLayout::HalfSideBySide);
        assert_eq!(info.frame_rate, Rate { num: 24, den: 1 });
    }

    #[test]
    fn jellyfin_ids_lose_their_dashes() {
        assert_eq!(SourceRef::jellyfin("289477B1-BDB8-7226-5402-628924353285").id, "289477b1bdb872265402628924353285");
    }
}
