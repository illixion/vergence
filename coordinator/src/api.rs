//! The coordinator's HTTP API.
//!
//! Players (grant `play`):
//! ```text
//! POST /v1/playing                                    {server, id, workload?, seconds, rate?, player?, stopped?}
//! GET  /v1/titles/{server}/{id}/{workload}            status (404 {"state":"none"} when unknown)
//! POST /v1/titles/{server}/{id}/{workload}            queue it for backfill (202)
//! GET  /v1/titles/{server}/{id}/{workload}/chunks/{n} the pieces ready so far, as a track; headers
//!                                                     Vergence-Chunk-Complete: true|false, Vergence-Pieces: done/total
//! ```
//! The Jellyfin plugin's routes, for Smooth Motion on 3D titles, serve only whole chunks:
//! ```text
//! GET  /SmoothMotion/{itemId}   POST /SmoothMotion/{itemId}/Analyze   GET /SmoothMotion/{itemId}/Chunks/{n}
//! ```
//! Workers (grant `work`; `?worker=` names one of several behind a token):
//! ```text
//! POST /v1/work/claim                          {worker, remote, models, workloads?} → a lease, or 204
//! POST /v1/work/leases/{lease}/heartbeat       → {leaseSeconds}
//! POST /v1/work/leases/{lease}/release
//! PUT  /v1/work/leases/{lease}/result          the piece as a track (VGAT); 422 with the reason when it fails the checks
//! ```
//! Admin: `GET /v1/queue`, `POST /v1/scan` (queue every stereo title the servers list),
//! `DELETE /v1/titles/{server}/{id}/{workload}`.
//!
//! Who is calling: a bearer token over plain HTTP (`vergenced token add`, or a paired device's
//! derived token), or a paired device on the TLS-PSK channel ([`crate::paired`]), whose grants are
//! the ones it was paired with. Both reach the same routes.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::{Duration, Instant};

use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, FromRequestParts, Path, Query, State};
use axum::http::{HeaderValue, StatusCode, header, request::Parts};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post, put};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::json;

use crate::auth::{Grant, Tokens, token_from_headers};
use crate::paired::{PairedDevices, PairedPeer, device_name};
use crate::queue::{AddError, AddOutcome, Claim, Queue, Title, Workload};
use crate::source::{Server, SourceRef, Sources};
use crate::store::Store;
use crate::track;

/// How long a title's description is trusted before a player's report reads it again.
const DESCRIBE_EVERY: Duration = Duration::from_secs(300);

pub struct AppState {
    pub queue: Mutex<Queue>,
    pub store: Store,
    pub sources: Sources,
    pub workloads: Vec<Workload>,
    pub tokens: RwLock<Tokens>,
    /// Paired devices, when this coordinator has any (set once, by [`AppState::set_paired`]).
    paired: OnceLock<PairedDevices>,
    described: Mutex<HashMap<String, Instant>>,
}

impl AppState {
    /// The state, with every stored title read back into the queue.
    pub fn new(store: Store, sources: Sources, workloads: Vec<Workload>, tokens: Tokens) -> std::io::Result<Arc<AppState>> {
        let mut queue = Queue::new();
        for (meta, done) in store.load()? {
            queue.restore(meta, &done);
        }
        Ok(Arc::new(AppState {
            queue: Mutex::new(queue),
            store,
            sources,
            workloads,
            tokens: RwLock::new(tokens),
            paired: OnceLock::new(),
            described: Mutex::new(HashMap::new()),
        }))
    }

    /// Lets paired devices in: on the TLS-PSK channel, and with their derived token over HTTP.
    /// Set once; a second call is ignored.
    pub fn set_paired(&self, devices: PairedDevices) {
        let _ = self.paired.set(devices);
    }

    pub fn paired(&self) -> Option<&PairedDevices> {
        self.paired.get()
    }
}

pub type Shared = Arc<AppState>;

#[derive(Debug)]
pub struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({ "reason": self.1 }))).into_response()
    }
}

fn err(status: StatusCode, reason: impl Into<String>) -> ApiError {
    ApiError(status, reason.into())
}

type ApiResult<T> = Result<T, ApiError>;

/// Who a request comes from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Caller {
    /// Plain HTTP, with the token it sent (if any).
    Token(Option<String>),
    /// A paired device on the TLS-PSK channel.
    Device(String),
}

impl<S: Send + Sync> FromRequestParts<S> for Caller {
    type Rejection = std::convert::Infallible;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        Ok(match parts.extensions.get::<PairedPeer>() {
            Some(peer) => Caller::Device(peer.device_id.clone()),
            None => Caller::Token(token_from_headers(&parts.headers)),
        })
    }
}

fn holds(grants: &[Grant], grant: Grant) -> bool {
    grants.contains(&grant) || grants.contains(&Grant::Admin)
}

/// The caller's name if it holds `grant` (admin holds every grant).
fn authorize(state: &AppState, caller: &Caller, grant: Grant) -> ApiResult<String> {
    let refused = || err(StatusCode::FORBIDDEN, "the token doesn't allow this");
    match caller {
        Caller::Device(id) => {
            let grants = state.paired().and_then(|d| d.grants(id)).ok_or_else(|| err(StatusCode::FORBIDDEN, "this device isn't paired"))?;
            holds(&grants, grant).then(|| device_name(id)).ok_or_else(refused)
        }
        Caller::Token(None) => Err(err(StatusCode::UNAUTHORIZED, "a token is needed")),
        Caller::Token(Some(token)) => {
            if let Some(name) = state.tokens.read().expect("tokens lock").check(token, grant) {
                return Ok(name.to_string());
            }
            match state.paired().and_then(|d| d.by_http_token(token)) {
                Some((id, grants)) if holds(&grants, grant) => Ok(device_name(&id)),
                _ => Err(refused()),
            }
        }
    }
}

fn source_ref(server: &str, id: &str) -> ApiResult<SourceRef> {
    match Server::from_name(server) {
        Some(Server::Jellyfin) => Ok(SourceRef::jellyfin(id)),
        Some(Server::Stash) => Ok(SourceRef::stash(id)),
        None => Err(err(StatusCode::NOT_FOUND, "unknown server")),
    }
}

impl AppState {
    fn workload(&self, name: &str) -> ApiResult<Workload> {
        self.workloads.iter().find(|w| w.name == name).cloned().ok_or_else(|| err(StatusCode::NOT_FOUND, "unknown workload"))
    }

    /// The title's key, describing it from its server first if it is new or
    /// its description is old (a replaced file resets its results).
    pub async fn ensure_title(&self, source: &SourceRef, workload: &Workload) -> ApiResult<String> {
        let key = Title::key_for(source, &workload.name);
        let fresh = self.described.lock().expect("described lock").get(&key).is_some_and(|t| t.elapsed() < DESCRIBE_EVERY);
        let known = self.queue.lock().expect("queue lock").title(&key).is_some();
        if known && fresh {
            return Ok(key);
        }
        let info = match self.sources.describe(source).await {
            Ok(info) => info,
            // A server that is away doesn't stop a known title's results being served.
            Err(e) if known => {
                tracing::warn!(title = %key, error = %e, "couldn't describe a known title again");
                return Ok(key);
            }
            Err(e) => return Err(err(StatusCode::BAD_GATEWAY, e.to_string())),
        };
        let outcome = self.queue.lock().expect("queue lock").add(source.clone(), workload, info);
        let outcome = outcome.map_err(|e: AddError| err(StatusCode::UNPROCESSABLE_ENTITY, e.to_string()))?;
        self.described.lock().expect("described lock").insert(key.clone(), Instant::now());
        if outcome == AddOutcome::Reset {
            tracing::info!(title = %key, "the file changed, its results were dropped");
            self.store.clear_pieces(&key).map_err(internal)?;
        }
        if outcome != AddOutcome::Unchanged {
            self.save_meta(&key)?;
        }
        Ok(key)
    }

    fn save_meta(&self, key: &str) -> ApiResult<()> {
        let title = self.queue.lock().expect("queue lock").title(key).cloned();
        if let Some(title) = title {
            self.store.save_meta(&title).map_err(internal)?;
        }
        Ok(())
    }

    fn status_json(&self, key: &str) -> Option<serde_json::Value> {
        let status = self.queue.lock().expect("queue lock").status(key)?;
        let mut value = serde_json::to_value(status).expect("status encodes");
        value["bytes"] = json!(self.store.bytes(key));
        Some(value)
    }

    /// Chunk `n`'s bytes and whether it is whole, or nothing when none of it is ready.
    fn chunk(&self, key: &str, n: usize) -> ApiResult<Option<ChunkBytes>> {
        let (title, pieces) = {
            let queue = self.queue.lock().expect("queue lock");
            let Some(title) = queue.title(key) else { return Ok(None) };
            let Some(pieces) = queue.chunk_pieces(key, n) else { return Ok(None) };
            (title.clone(), pieces)
        };
        let (done, whole) = pieces;
        if done.is_empty() {
            return Ok(None);
        }
        let total = ((n + 1) * crate::queue::PIECES_PER_CHUNK).min(title.pieces.len()) - n * crate::queue::PIECES_PER_CHUNK;
        let bytes = self.store.chunk(&title, &done).map_err(internal)?;
        Ok(Some(ChunkBytes { bytes, whole, done: done.len(), total }))
    }
}

struct ChunkBytes {
    bytes: Vec<u8>,
    whole: bool,
    done: usize,
    total: usize,
}

fn internal(e: impl std::fmt::Display) -> ApiError {
    tracing::error!(error = %e, "internal error");
    err(StatusCode::INTERNAL_SERVER_ERROR, "internal error")
}

pub fn router(state: Shared) -> Router {
    Router::new()
        .route("/v1/health", get(|| async { Json(json!({ "service": "vergenced", "version": env!("CARGO_PKG_VERSION") })) }))
        .route("/v1/playing", post(playing))
        .route("/v1/titles/{server}/{id}/{workload}", get(title_status).post(enqueue).delete(remove))
        .route("/v1/titles/{server}/{id}/{workload}/chunks/{n}", get(chunk))
        .route("/v1/queue", get(queue_view))
        .route("/v1/scan", post(scan))
        .route("/v1/work/claim", post(claim))
        .route("/v1/work/leases/{lease}/heartbeat", post(heartbeat))
        .route("/v1/work/leases/{lease}/release", post(release))
        .route("/v1/work/leases/{lease}/result", put(result))
        .route("/SmoothMotion/{item}", get(compat_status))
        .route("/SmoothMotion/{item}/Analyze", post(compat_analyze))
        .route("/SmoothMotion/{item}/Chunks/{n}", get(compat_chunk))
        .layer(DefaultBodyLimit::max(track::MAX_RECORD_BYTES))
        .with_state(state)
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Playing {
    server: String,
    id: String,
    #[serde(default)]
    workload: Option<String>,
    seconds: f64,
    #[serde(default = "one")]
    rate: f64,
    /// Several players behind one token tell themselves apart with this.
    #[serde(default)]
    player: Option<String>,
    #[serde(default)]
    stopped: bool,
}

fn one() -> f64 {
    1.0
}

async fn playing(State(state): State<Shared>, caller: Caller, Json(body): Json<Playing>) -> ApiResult<Response> {
    let name = authorize(&state, &caller, Grant::Play)?;
    let player = format!("{name}/{}", body.player.as_deref().unwrap_or("default"));
    if body.stopped {
        state.queue.lock().expect("queue lock").stop_playhead(&player);
        return Ok(StatusCode::NO_CONTENT.into_response());
    }
    let source = source_ref(&body.server, &body.id)?;
    let workload = state.workload(body.workload.as_deref().unwrap_or(Workload::SMOOTH_MOTION_3D))?;
    let key = state.ensure_title(&source, &workload).await?;
    state.queue.lock().expect("queue lock").report_playhead(&player, &key, body.seconds, body.rate, Instant::now());
    Ok(Json(state.status_json(&key)).into_response())
}

async fn title_status(
    State(state): State<Shared>,
    caller: Caller,
    Path((server, id, workload)): Path<(String, String, String)>,
) -> ApiResult<Response> {
    authorize(&state, &caller, Grant::Play)?;
    let key = Title::key_for(&source_ref(&server, &id)?, &workload);
    Ok(match state.status_json(&key) {
        Some(status) => Json(status).into_response(),
        None => (StatusCode::NOT_FOUND, Json(json!({ "state": "none" }))).into_response(),
    })
}

async fn enqueue(
    State(state): State<Shared>,
    caller: Caller,
    Path((server, id, workload)): Path<(String, String, String)>,
) -> ApiResult<Response> {
    authorize(&state, &caller, Grant::Play)?;
    let workload = state.workload(&workload)?;
    let key = state.ensure_title(&source_ref(&server, &id)?, &workload).await?;
    Ok((StatusCode::ACCEPTED, Json(state.status_json(&key))).into_response())
}

async fn remove(
    State(state): State<Shared>,
    caller: Caller,
    Path((server, id, workload)): Path<(String, String, String)>,
) -> ApiResult<StatusCode> {
    authorize(&state, &caller, Grant::Admin)?;
    let key = Title::key_for(&source_ref(&server, &id)?, &workload);
    let removed = state.queue.lock().expect("queue lock").remove(&key);
    state.store.remove_title(&key).map_err(internal)?;
    Ok(if removed { StatusCode::NO_CONTENT } else { StatusCode::NOT_FOUND })
}

fn chunk_response(ChunkBytes { bytes, whole, done, total }: ChunkBytes) -> Response {
    let mut response = bytes.into_response();
    let headers = response.headers_mut();
    headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("application/octet-stream"));
    headers.insert("Vergence-Chunk-Complete", HeaderValue::from_static(if whole { "true" } else { "false" }));
    headers.insert("Vergence-Pieces", HeaderValue::from_str(&format!("{done}/{total}")).expect("ascii"));
    headers.insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static(if whole { "private, max-age=86400" } else { "no-store" }),
    );
    response
}

async fn chunk(
    State(state): State<Shared>,
    caller: Caller,
    Path((server, id, workload, n)): Path<(String, String, String, usize)>,
) -> ApiResult<Response> {
    authorize(&state, &caller, Grant::Play)?;
    let key = Title::key_for(&source_ref(&server, &id)?, &workload);
    match state.chunk(&key, n)? {
        Some(chunk) => Ok(chunk_response(chunk)),
        None => Err(err(StatusCode::NOT_FOUND, "nothing of that chunk is ready")),
    }
}

async fn queue_view(State(state): State<Shared>, caller: Caller) -> ApiResult<Response> {
    authorize(&state, &caller, Grant::Admin)?;
    let snapshot = state.queue.lock().expect("queue lock").snapshot(Instant::now());
    Ok(Json(snapshot).into_response())
}

/// Queues every stereo title the servers list, for each workload that needs both eyes.
pub async fn scan_now(state: &AppState) -> serde_json::Value {
    let mut added = 0;
    let mut problems = Vec::new();
    for (source, listed) in state.sources.list_3d().await {
        if let Err(e) = listed {
            problems.push(format!("{}: {e}", source.server.name()));
            continue;
        }
        for workload in state.workloads.iter().filter(|w| w.needs_stereo) {
            match state.ensure_title(&source, workload).await {
                Ok(_) => added += 1,
                Err(ApiError(_, reason)) => problems.push(format!("{}: {reason}", Title::key_for(&source, &workload.name))),
            }
        }
    }
    json!({ "titles": added, "problems": problems })
}

async fn scan(State(state): State<Shared>, caller: Caller) -> ApiResult<Response> {
    authorize(&state, &caller, Grant::Admin)?;
    Ok(Json(scan_now(&state).await).into_response())
}

#[derive(Deserialize)]
struct WorkerQuery {
    #[serde(default)]
    worker: Option<String>,
}

fn worker_name(token: &str, worker: Option<&str>) -> String {
    format!("{token}/{}", worker.unwrap_or("default"))
}

async fn claim(State(state): State<Shared>, caller: Caller, Json(mut body): Json<Claim>) -> ApiResult<Response> {
    let token = authorize(&state, &caller, Grant::Work)?;
    body.worker = worker_name(&token, Some(&body.worker));
    let lease = state.queue.lock().expect("queue lock").claim(&body, Instant::now());
    let Some(lease) = lease else { return Ok(StatusCode::NO_CONTENT.into_response()) };
    // The first lease of a title fixes its profile.
    state.save_meta(&lease.title)?;
    tracing::debug!(title = %lease.title, piece = lease.piece, worker = %body.worker, live = lease.live, "leased");
    Ok(Json(lease).into_response())
}

fn lease_error(e: crate::queue::UploadError) -> ApiError {
    use crate::queue::UploadError::*;
    match e {
        NoLease => err(StatusCode::GONE, e.to_string()),
        NotYours => err(StatusCode::FORBIDDEN, e.to_string()),
        _ => err(StatusCode::UNPROCESSABLE_ENTITY, e.to_string()),
    }
}

async fn heartbeat(
    State(state): State<Shared>,
    caller: Caller,
    Path(lease): Path<u64>,
    Query(q): Query<WorkerQuery>,
) -> ApiResult<Response> {
    let token = authorize(&state, &caller, Grant::Work)?;
    let worker = worker_name(&token, q.worker.as_deref());
    let seconds = state.queue.lock().expect("queue lock").heartbeat(lease, &worker, Instant::now()).map_err(lease_error)?;
    Ok(Json(json!({ "leaseSeconds": seconds })).into_response())
}

async fn release(
    State(state): State<Shared>,
    caller: Caller,
    Path(lease): Path<u64>,
    Query(q): Query<WorkerQuery>,
) -> ApiResult<StatusCode> {
    let token = authorize(&state, &caller, Grant::Work)?;
    let worker = worker_name(&token, q.worker.as_deref());
    state.queue.lock().expect("queue lock").release(lease, &worker).map_err(lease_error)?;
    Ok(StatusCode::NO_CONTENT)
}

async fn result(
    State(state): State<Shared>,
    caller: Caller,
    Path(lease): Path<u64>,
    Query(q): Query<WorkerQuery>,
    body: Bytes,
) -> ApiResult<Response> {
    let token = authorize(&state, &caller, Grant::Work)?;
    let worker = worker_name(&token, q.worker.as_deref());
    let parsed = track::parse(&body).map_err(|e| err(StatusCode::UNPROCESSABLE_ENTITY, e.to_string()))?;
    let (key, piece) = state.queue.lock().expect("queue lock").check(lease, &worker, &parsed).map_err(|e| {
        tracing::warn!(lease, worker = %worker, error = %e, "a result failed its checks");
        lease_error(e)
    })?;
    let bytes = state.store.save_piece(&key, piece, &body, &parsed).map_err(internal)?;
    let size = (parsed.header.source_width, parsed.header.source_height);
    let completed = state.queue.lock().expect("queue lock").complete(lease, size, parsed.header.video_end_frame, Instant::now());
    // Pieces past where the video ends hold nothing; they are stored empty so their chunks serve.
    if let Some((_, _, emptied)) = completed {
        let empty = track::write_header(&parsed.header);
        let none = track::parse(&empty).map_err(internal)?;
        for i in emptied {
            state.store.save_piece(&key, i, &empty, &none).map_err(internal)?;
        }
        if let Some(end) = parsed.header.video_end_frame {
            tracing::info!(title = %key, end, "the video stream ends before the container: the title ends there");
        }
    }
    state.save_meta(&key)?;
    tracing::debug!(title = %key, piece, bytes, "piece done");
    Ok(Json(state.status_json(&key)).into_response())
}

// MARK: The Jellyfin plugin's routes

async fn compat_status(State(state): State<Shared>, caller: Caller, Path(item): Path<String>) -> ApiResult<Response> {
    authorize(&state, &caller, Grant::Play)?;
    let key = Title::key_for(&SourceRef::jellyfin(&item), Workload::SMOOTH_MOTION_3D);
    let Some(status) = state.queue.lock().expect("queue lock").status(&key) else {
        return Ok(Json(json!({ "state": "none" })).into_response());
    };
    // Players that know only these routes fetch a chunk once and keep it, so
    // they are told "ready" only when every chunk is whole.
    let mut body = json!({ "state": status.state, "progress": status.progress });
    if status.state == "ready" {
        body = json!({
            "state": "ready", "chunkSeconds": status.chunk_seconds, "chunks": status.chunks,
            "frameRate": status.frame_rate, "bytes": state.store.bytes(&key),
        });
    }
    Ok(Json(body).into_response())
}

async fn compat_analyze(State(state): State<Shared>, caller: Caller, Path(item): Path<String>) -> ApiResult<Response> {
    authorize(&state, &caller, Grant::Play)?;
    let workload = state.workload(Workload::SMOOTH_MOTION_3D)?;
    let key = state
        .ensure_title(&SourceRef::jellyfin(&item), &workload)
        .await
        .map_err(|ApiError(_, reason)| err(StatusCode::CONFLICT, reason))?;
    let status = state.queue.lock().expect("queue lock").status(&key);
    Ok((StatusCode::ACCEPTED, Json(json!({ "state": status.map(|s| s.state).unwrap_or("queued") }))).into_response())
}

async fn compat_chunk(
    State(state): State<Shared>,
    caller: Caller,
    Path((item, n)): Path<(String, usize)>,
) -> ApiResult<Response> {
    authorize(&state, &caller, Grant::Play)?;
    let key = Title::key_for(&SourceRef::jellyfin(&item), Workload::SMOOTH_MOTION_3D);
    match state.chunk(&key, n)? {
        Some(chunk) if chunk.whole => Ok(chunk_response(chunk)),
        _ => Err(err(StatusCode::NOT_FOUND, "that chunk isn't whole yet")),
    }
}
