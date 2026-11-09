//! The coordinator end to end over HTTP: a player's report queues a title
//! described by a (mock) Jellyfin, a worker claims and uploads its pieces,
//! and the player reads chunks as they fill. The last test talks to the
//! disposable dev servers (Hypnos `scripts/dev-jellyfin.sh`, `dev-stash.sh`)
//! and is ignored unless asked for.

use std::sync::Arc;

use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode};
use axum::routing::get;
use axum::{Json, Router};
use serde_json::{Value, json};
use tower::ServiceExt;
use vergence_coordinator::api::{AppState, router};
use vergence_coordinator::auth::{Grant, Tokens};
use vergence_coordinator::queue::Workload;
use vergence_coordinator::source::{Jellyfin, ServerConfig, Sources, Stash};
use vergence_coordinator::store::Store;
use vergence_coordinator::track;

const ITEM: &str = "289477b1bdb872265402628924353285";

/// A Jellyfin that knows one half side-by-side title, 6 s at 24 fps.
async fn mock_jellyfin(etag: &'static str) -> String {
    let app = Router::new().route(
        "/Items",
        get(move || async move {
            Json(json!({"Items": [{
                "Id": ITEM, "RunTimeTicks": 60_000_000i64, "Video3DFormat": "HalfSideBySide",
                "MediaSources": [{"Id": ITEM, "ETag": etag, "Path": "/m/t.mkv", "MediaStreams": [
                    {"Type": "Video", "RealFrameRate": 24.0, "Width": 1920, "Height": 1080, "Codec": "hevc"}]}]
            }]}))
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    format!("http://{address}")
}

struct Harness {
    app: Router,
    play: String,
    work: String,
    admin: String,
    _dir: tempfile::TempDir,
}

async fn harness(etag: &'static str, dir: Option<tempfile::TempDir>) -> Harness {
    let dir = dir.unwrap_or_else(|| tempfile::tempdir().unwrap());
    let url = mock_jellyfin(etag).await;
    let config = ServerConfig { url, api_key: Some("k".into()), api_key_file: None };
    let sources = Sources { jellyfin: Some(Jellyfin::new(&config, reqwest::Client::new()).unwrap()), stash: None };
    let mut tokens = Tokens::load(&dir.path().join("tokens.json")).unwrap();
    let play = tokens.add("vp", vec![Grant::Play]).unwrap();
    let work = tokens.add("mac", vec![Grant::Work]).unwrap();
    let admin = tokens.add("desk", vec![Grant::Admin]).unwrap();
    let state = AppState::new(Store::open(dir.path()).unwrap(), sources, Workload::defaults(), tokens).unwrap();
    Harness { app: router(state), play, work, admin, _dir: dir }
}

/// A coordinator with no key for any server: the player describes the title.
async fn keyless_harness() -> Harness {
    let dir = tempfile::tempdir().unwrap();
    let sources = Sources { jellyfin: None, stash: None };
    let mut tokens = Tokens::load(&dir.path().join("tokens.json")).unwrap();
    let play = tokens.add("vp", vec![Grant::Play]).unwrap();
    let work = tokens.add("mac", vec![Grant::Work]).unwrap();
    let admin = tokens.add("desk", vec![Grant::Admin]).unwrap();
    let state = AppState::new(Store::open(dir.path()).unwrap(), sources, Workload::defaults(), tokens).unwrap();
    Harness { app: router(state), play, work, admin, _dir: dir }
}

fn media(identity: &str) -> Value {
    json!({"durationSeconds": 6.0, "frameRate": {"num": 24, "den": 1}, "stereo": "halfSideBySide", "identity": identity})
}

#[tokio::test]
async fn a_handed_over_credential_describes_the_title_and_reaches_only_local_workers() {
    let h = keyless_harness().await;
    let url = mock_jellyfin("etag-h").await;
    let report = json!({"server": "jellyfin", "id": ITEM, "seconds": 1.0,
                        "credentials": {"jellyfin": {"url": url, "key": "the-players-token"}}});
    // Over plain HTTP from the network, credentials are refused.
    let (status, _) = h.json_from("192.168.1.20:50000", "POST", "/v1/playing", &h.play, report.clone()).await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    let (status, _) = h.json("POST", "/v1/playing", &h.play, report.clone()).await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    // From this computer they are kept, and describe the title without `media`.
    let (status, body) = h.json_from("127.0.0.1:50000", "POST", "/v1/playing", &h.play, report).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["pieces"], 3);
    // A worker on the network gets its lease without the credential; one on loopback gets it.
    let claim = json!({"worker": "gpu0", "models": {"flowPair": "RIFE 480x288", "disparity": "NeuFlow 512x288"}});
    let (status, remote) = h.json_from("192.168.1.30:50000", "POST", "/v1/work/claim", &h.work, claim.clone()).await;
    assert_eq!(status, StatusCode::OK);
    assert!(remote.get("credential").is_none(), "{remote}");
    let (status, local) = h.json_from("127.0.0.1:50001", "POST", "/v1/work/claim", &h.work, claim).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(local["credential"]["key"], "the-players-token");
    assert_eq!(local["credential"]["url"], url);
}

#[tokio::test]
async fn web_pages_are_turned_away() {
    let h = keyless_harness().await;
    let request = |method: &str, origin: Option<&str>| {
        let mut builder = Request::builder().method(method).uri("/v1/health");
        if let Some(origin) = origin {
            builder = builder.header("origin", origin);
        }
        builder.body(Body::empty()).unwrap()
    };
    let call = |r| async { h.app.clone().oneshot(r).await.unwrap() };
    assert_eq!(call(request("GET", None)).await.status(), StatusCode::OK);
    let page = call(request("GET", Some("https://evil.example"))).await;
    assert_eq!(page.status(), StatusCode::FORBIDDEN);
    assert!(page.headers().keys().all(|k| !k.as_str().starts_with("access-control-")));
    let preflight = call(request("OPTIONS", Some("https://evil.example"))).await;
    assert_eq!(preflight.status(), StatusCode::FORBIDDEN);
    assert!(preflight.headers().keys().all(|k| !k.as_str().starts_with("access-control-")));
}

#[tokio::test]
async fn without_a_key_the_player_describes_the_title() {
    let h = keyless_harness().await;
    let report = |media: Option<Value>| {
        let mut body = json!({"server": "stash", "id": "42", "seconds": 1.0});
        if let Some(m) = media {
            body["media"] = m;
        }
        body
    };
    // Nothing known and nothing sent: asked for the description, not a bad gateway.
    let (status, _) = h.json("POST", "/v1/playing", &h.play, report(None)).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    // Nonsense is refused.
    let mut bad = media("etag");
    bad["durationSeconds"] = json!(1.0e9);
    let (status, _) = h.json("POST", "/v1/playing", &h.play, report(Some(bad))).await;
    assert_eq!(status, StatusCode::BAD_GATEWAY);
    let (status, _) = h.json("POST", "/v1/playing", &h.play, report(Some(json!({"durationSeconds": 6.0, "frameRate": {"num": 24, "den": 1}, "stereo": "halfSideBySide", "identity": ""})))).await;
    assert_eq!(status, StatusCode::BAD_GATEWAY);
    // A sensible description queues the title and a worker can claim it.
    let (status, body) = h.json("POST", "/v1/playing", &h.play, report(Some(media("etag-1")))).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let lease = h.claim().await.expect("a piece to claim");
    assert_eq!(lease["source"]["server"], "stash");
    // Later reports may leave `media` out, and enqueue takes it in a body.
    let (status, _) = h.json("POST", "/v1/playing", &h.play, report(None)).await;
    assert_eq!(status, StatusCode::OK);
    let (status, _) = h.json("POST", "/v1/titles/stash/43/smooth-motion-3d", &h.play, json!({"media": media("e2")})).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    let (status, _) = h.json("POST", "/v1/titles/stash/44/smooth-motion-3d", &h.play, json!({})).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    let _ = &h.admin;
}

impl Harness {
    async fn send(&self, method: &str, uri: &str, token: &str, body: Body) -> (StatusCode, axum::http::HeaderMap, Vec<u8>) {
        self.send_from(None, method, uri, token, body).await
    }

    /// As a connection from `peer` would arrive (what `into_make_service_with_connect_info` adds).
    async fn send_from(
        &self,
        peer: Option<&str>,
        method: &str,
        uri: &str,
        token: &str,
        body: Body,
    ) -> (StatusCode, axum::http::HeaderMap, Vec<u8>) {
        let mut request = Request::builder()
            .method(method)
            .uri(uri)
            .header("authorization", format!("Bearer {token}"))
            .header("content-type", "application/json")
            .body(body)
            .unwrap();
        if let Some(peer) = peer {
            let address: std::net::SocketAddr = peer.parse().unwrap();
            request.extensions_mut().insert(axum::extract::ConnectInfo(address));
        }
        let response = self.app.clone().oneshot(request).await.unwrap();
        let status = response.status();
        let headers = response.headers().clone();
        (status, headers, to_bytes(response.into_body(), usize::MAX).await.unwrap().to_vec())
    }

    async fn json_from(&self, peer: &str, method: &str, uri: &str, token: &str, body: Value) -> (StatusCode, Value) {
        let (status, _, bytes) = self.send_from(Some(peer), method, uri, token, Body::from(body.to_string())).await;
        (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
    }

    async fn json(&self, method: &str, uri: &str, token: &str, body: Value) -> (StatusCode, Value) {
        let (status, _, bytes) = self.send(method, uri, token, Body::from(body.to_string())).await;
        (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
    }

    async fn claim(&self) -> Option<Value> {
        let (status, lease) = self
            .json(
                "POST",
                "/v1/work/claim",
                &self.work,
                json!({"worker": "gpu0", "models": {"flowPair": "RIFE 480x288", "disparity": "NeuFlow 512x288"}}),
            )
            .await;
        (status == StatusCode::OK).then_some(lease)
    }

    /// Uploads what `vergence worker` would make for the lease.
    async fn finish(&self, lease: &Value) -> StatusCode {
        let body = piece(lease, 960);
        let uri = format!("/v1/work/leases/{}/result?worker=gpu0", lease["lease"]);
        self.send("PUT", &uri, &self.work, Body::from(body)).await.0
    }
}

fn record(kind: u8, eye: u8, time: i64, end: i64) -> Vec<u8> {
    let mut body = vec![kind, eye, 3, 1];
    body.extend_from_slice(&2u16.to_le_bytes());
    body.extend_from_slice(&2u16.to_le_bytes());
    body.extend_from_slice(&time.to_le_bytes());
    body.extend_from_slice(&end.to_le_bytes());
    body.extend_from_slice(&24u32.to_le_bytes());
    body.extend_from_slice(&1f32.to_le_bytes());
    body.extend_from_slice(&0f32.to_le_bytes());
    body.extend_from_slice(&[1, 2, 3, 4]);
    let mut frame = (body.len() as u32).to_le_bytes().to_vec();
    frame.extend(body);
    frame
}

fn piece(lease: &Value, width: u32) -> Vec<u8> {
    let mut data = track::write_header(&track::Header {
        media_id: "worker's own".into(),
        producer: "test".into(),
        models: [("flowPair".into(), "RIFE 480x288".into()), ("disparity".into(), "NeuFlow 512x288".into())].into(),
        source_width: width,
        source_height: 1080,
        video_end_frame: None,
    });
    let (first, end, count) =
        (lease["firstFrame"].as_i64().unwrap(), lease["endFrame"].as_i64().unwrap(), lease["frameCount"].as_i64().unwrap());
    for k in first..end {
        if k + 1 < count {
            data.extend(record(1, 0, k, k + 1));
        }
        data.extend(record(2, 2, k, k));
    }
    data
}

#[tokio::test]
async fn a_title_flows_from_player_to_worker_and_back() {
    let h = harness("v1", None).await;
    // Nothing yet.
    let (status, body) = h.json("GET", &format!("/SmoothMotion/{ITEM}"), &h.play, Value::Null).await;
    assert_eq!((status, body["state"].as_str()), (StatusCode::OK, Some("none")));
    // Tokens keep to their grants.
    assert_eq!(h.json("POST", "/v1/work/claim", &h.play, json!({"worker": "x", "models": {}})).await.0, StatusCode::FORBIDDEN);
    assert_eq!(h.json("GET", "/v1/queue", &h.work, Value::Null).await.0, StatusCode::FORBIDDEN);

    // A player opens the title at 3 s: it is described and queued, and the piece at its playhead goes first.
    let (status, body) =
        h.json("POST", "/v1/playing", &h.play, json!({"server": "jellyfin", "id": ITEM, "seconds": 3.0})).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!((body["pieces"].as_u64(), body["chunks"].as_u64(), body["state"].as_str()), (Some(3), Some(1), Some("queued")));
    let first = h.claim().await.unwrap();
    assert_eq!((first["piece"].as_u64(), first["live"].as_bool()), (Some(1), Some(true)));
    assert_eq!(first["info"]["stereo"], "halfSideBySide");

    // A result for the wrong picture size is refused once the size is fixed; first the good one.
    assert_eq!(h.finish(&first).await, StatusCode::OK);
    let uri = format!("/v1/titles/jellyfin/{ITEM}/smooth-motion-3d/chunks/0");
    let (status, headers, bytes) = h.send("GET", &uri, &h.play, Body::empty()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(headers["vergence-chunk-complete"], "false");
    assert_eq!(headers["vergence-pieces"], "1/3");
    let chunk = track::parse(&bytes).unwrap();
    assert_eq!(chunk.records.len(), 48 * 2);
    assert_eq!(chunk.header.models["flowPair"], "RIFE 480x288");
    assert_eq!((chunk.header.source_width, chunk.header.producer.starts_with("vergenced")), (960, true));
    // The plugin's routes only serve whole chunks.
    let (status, _, _) = h.send("GET", &format!("/SmoothMotion/{ITEM}/Chunks/0"), &h.play, Body::empty()).await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    let second = h.claim().await.unwrap();
    let wrong = piece(&second, 1920);
    let uri2 = format!("/v1/work/leases/{}/result?worker=gpu0", second["lease"]);
    assert_eq!(h.send("PUT", &uri2, &h.work, Body::from(wrong)).await.0, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(h.finish(&second).await, StatusCode::OK);
    let third = h.claim().await.unwrap();
    assert_eq!(h.finish(&third).await, StatusCode::OK);
    assert!(h.claim().await.is_none());

    let (_, body) = h.json("GET", &format!("/SmoothMotion/{ITEM}"), &h.play, Value::Null).await;
    assert_eq!((body["state"].as_str(), body["chunks"].as_u64(), body["chunkSeconds"].as_f64()), (Some("ready"), Some(1), Some(10.0)));
    let (status, headers, bytes) = h.send("GET", &format!("/SmoothMotion/{ITEM}/Chunks/0"), &h.play, Body::empty()).await;
    assert_eq!((status, &headers["vergence-chunk-complete"]), (StatusCode::OK, &"true".parse::<axum::http::HeaderValue>().unwrap()));
    let whole = track::parse(&bytes).unwrap();
    // 144 frames: 143 flow pairs and 144 disparities, in time order.
    assert_eq!(whole.records.len(), 143 + 144);
    assert!(whole.records.windows(2).all(|w| w[0].time.value <= w[1].time.value));

    let (status, view) = h.json("GET", "/v1/queue", &h.admin, Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(view["workers"][0]["pieces"], 3);
}

#[tokio::test]
async fn results_survive_a_restart_and_a_replaced_file_drops_them() {
    let h = harness("v1", None).await;
    h.json("POST", "/v1/playing", &h.play, json!({"server": "jellyfin", "id": ITEM, "seconds": 0.0})).await;
    let lease = h.claim().await.unwrap();
    assert_eq!(h.finish(&lease).await, StatusCode::OK);
    let Harness { _dir: dir, .. } = h;

    // Same store, same file: the piece is still done and the profile still fixed.
    let h = harness("v1", Some(dir)).await;
    let (_, body) = h.json("GET", &format!("/v1/titles/jellyfin/{ITEM}/smooth-motion-3d"), &h.play, Value::Null).await;
    assert_eq!(body["piecesDone"], 1);
    let Harness { _dir: dir, .. } = h;

    // The file was replaced: describing it again drops what was made from the old one.
    let h = harness("v2", Some(dir)).await;
    let (_, body) = h.json("POST", "/v1/playing", &h.play, json!({"server": "jellyfin", "id": ITEM, "seconds": 0.0})).await;
    assert_eq!(body["piecesDone"], 0);
}

#[tokio::test]
async fn lease_ops_are_the_holders_alone() {
    let h = harness("v1", None).await;
    h.json("POST", &format!("/v1/titles/jellyfin/{ITEM}/smooth-motion-3d"), &h.play, Value::Null).await;
    let lease = h.claim().await.unwrap();
    let id = &lease["lease"];
    let (status, body) = h.json("POST", &format!("/v1/work/leases/{id}/heartbeat?worker=gpu0"), &h.work, Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body["leaseSeconds"].as_f64().unwrap() >= 20.0);
    let (status, _) = h.json("POST", &format!("/v1/work/leases/{id}/release?worker=other"), &h.work, Value::Null).await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    let (status, _) = h.json("POST", &format!("/v1/work/leases/{id}/release?worker=gpu0"), &h.work, Value::Null).await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    assert_eq!(h.claim().await.unwrap()["piece"], lease["piece"]);
}

/// Reads the dev servers: `cargo test -p vergence-coordinator -- --ignored dev_servers`.
#[tokio::test]
#[ignore]
async fn dev_servers() {
    let home = std::env::var("HOME").unwrap();
    let jellyfin = ServerConfig {
        url: "http://127.0.0.1:8097".into(),
        api_key: None,
        api_key_file: Some(format!("{home}/.local/share/hypnos-dev-jellyfin/config/dev-api-key.txt")),
    };
    let stash = ServerConfig { url: "http://127.0.0.1:9998".into(), api_key: None, api_key_file: None };
    let http = reqwest::Client::new();
    let j = Jellyfin::new(&jellyfin, http.clone()).unwrap();
    let s = Stash::new(&stash, http).unwrap();
    let sources = Arc::new(Sources { jellyfin: Some(j.clone()), stash: Some(s.clone()) });
    let listed = sources.list_3d().await;
    println!("stereo titles: {listed:?}");
    let stash_scene = s.describe("6").await.unwrap();
    println!("stash 6: {stash_scene:?}");
    assert!(stash_scene.duration_seconds > 0.0 && !stash_scene.identity.is_empty());
    let movies = reqwest::Client::new()
        .get("http://127.0.0.1:8097/Items?Recursive=true&IncludeItemTypes=Movie")
        .header(
            "Authorization",
            format!("MediaBrowser Token=\"{}\"", jellyfin.key().unwrap().unwrap()),
        )
        .send()
        .await
        .unwrap()
        .json::<Value>()
        .await
        .unwrap();
    let id = movies["Items"][0]["Id"].as_str().unwrap().to_string();
    let info = j.describe(&id).await.unwrap();
    println!("jellyfin {id}: {info:?}");
    assert!(info.duration_seconds > 0.0 && info.width > 0);
}
