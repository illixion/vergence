//! The work queue, without I/O: titles cut into 2 s pieces, leases for the
//! workers that claim them, and priorities from where the players are.
//!
//! - **Unit of work:** a piece, 2 s of one title for one workload (a set of
//!   record kinds). Chunks stay 10 s for players and storage, five pieces each.
//! - **Priority:** live (within 30 s ahead of a playhead), then ahead (the rest
//!   of a playing title), then backfill (everything else, oldest title first).
//!   Remote workers never take live pieces.
//! - **Leases** last about twice a worker's measured piece time; an expired
//!   one returns its piece to the queue. A seek doesn't cancel anything: the
//!   pieces behind the new playhead just fall to backfill.
//! - **Profiles:** one set of models per title. The first worker granted a
//!   piece fixes it; workers running other models skip that title, so
//!   neighbouring chunks never come from numerically different networks.

use std::collections::{BTreeMap, HashMap};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::source::{MediaInfo, SourceRef};
use crate::track::{Kind, ParsedTrack};

pub const PIECE_SECONDS: f64 = 2.0;
pub const CHUNK_SECONDS: f64 = 10.0;
pub const PIECES_PER_CHUNK: usize = 5;
/// Pieces this close ahead of a playhead are live.
pub const LIVE_SECONDS: f64 = 30.0;
/// A playhead not reported again for this long is gone.
pub const PLAYHEAD_TTL: Duration = Duration::from_secs(20);
const DEFAULT_LEASE: Duration = Duration::from_secs(120);
const MIN_LEASE: Duration = Duration::from_secs(20);
const MAX_LEASE: Duration = Duration::from_secs(900);

/// A set of record kinds made together, e.g. Smooth Motion for 3D titles
/// (flow pairs on the left eye and right→left disparity).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Workload {
    pub name: String,
    pub kinds: Vec<Kind>,
    /// Only stereo titles (both eyes are read).
    pub needs_stereo: bool,
    /// Only flat (2D) titles: the whole picture is one eye.
    #[serde(default)]
    pub mono_only: bool,
}

impl Workload {
    pub const SMOOTH_MOTION_3D: &'static str = "smooth-motion-3d";
    /// Smooth Motion for 2D titles, flat or played as AI 3D: the whole
    /// picture's flow between frames, one `flowPair` record per pair (eye
    /// mono). Larger than the 3D workload's (RIFE at 960×576 against
    /// 480×288), since there is no disparity to make.
    pub const SMOOTH_MOTION_2D: &'static str = "smooth-motion-2d";
    /// AI 3D's depth for a 2D title: the offline pipeline's refined,
    /// shot-anchored depth with a larger model than a headset runs live, one
    /// `depth` record per frame (eye mono), which the player warps from.
    pub const AI_3D_DEPTH: &'static str = "ai-3d-depth";
    /// AI HDR for a 2D title: one `gainMap` record per frame, a 16×9×8
    /// bilateral grid of log2 gain (VergenceCore's `GainGrid` pins it).
    pub const AI_HDR: &'static str = "ai-hdr";

    pub fn defaults() -> Vec<Workload> {
        vec![
            Workload {
                name: Self::SMOOTH_MOTION_3D.into(),
                kinds: vec![Kind::FlowPair, Kind::Disparity],
                needs_stereo: true,
                mono_only: false,
            },
            Workload {
                name: Self::SMOOTH_MOTION_2D.into(),
                kinds: vec![Kind::FlowPair],
                needs_stereo: false,
                mono_only: true,
            },
            Workload { name: Self::AI_3D_DEPTH.into(), kinds: vec![Kind::Depth], needs_stereo: false, mono_only: true },
            Workload { name: Self::AI_HDR.into(), kinds: vec![Kind::GainMap], needs_stereo: false, mono_only: true },
        ]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PieceState {
    Pending,
    Leased,
    Done,
}

/// The models a title's records come from, per kind name.
pub type Profile = BTreeMap<String, String>;

#[derive(Clone, Debug)]
pub struct Title {
    pub key: String,
    pub source: SourceRef,
    pub workload: Workload,
    pub info: MediaInfo,
    pub profile: Option<Profile>,
    /// One eye's size as the workers decoded it, fixed by the first result.
    pub source_size: Option<(u32, u32)>,
    /// Added order, for backfill.
    pub ordinal: u64,
    pub pieces: Vec<PieceState>,
    /// Where the video stream really ends, when a worker found it shorter
    /// than the container (whose duration the servers report).
    pub video_frames: Option<i64>,
}

/// What a title's results are stored and served as.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TitleMeta {
    pub key: String,
    pub source: SourceRef,
    pub workload: Workload,
    pub info: MediaInfo,
    pub profile: Option<Profile>,
    pub source_size: Option<(u32, u32)>,
    pub ordinal: u64,
    #[serde(default)]
    pub video_frames: Option<i64>,
}

impl Title {
    pub fn key_for(source: &SourceRef, workload: &str) -> String {
        format!("{}/{}/{}", source.server.name(), source.id, workload)
    }

    pub fn meta(&self) -> TitleMeta {
        TitleMeta {
            key: self.key.clone(),
            source: self.source.clone(),
            workload: self.workload.clone(),
            info: self.info.clone(),
            profile: self.profile.clone(),
            source_size: self.source_size,
            ordinal: self.ordinal,
            video_frames: self.video_frames,
        }
    }

    pub fn piece_count(info: &MediaInfo) -> usize {
        (info.duration_seconds / PIECE_SECONDS).ceil().max(0.0) as usize
    }

    pub fn chunk_count(&self) -> usize {
        self.pieces.len().div_ceil(PIECES_PER_CHUNK)
    }

    /// Frames in the whole title: the container's duration at the frame
    /// rate, or where the video stream turned out to end.
    pub fn frame_count(&self) -> i64 {
        let nominal = (self.info.duration_seconds * self.info.frame_rate.fps()).round() as i64;
        self.video_frames.map_or(nominal, |end| end.min(nominal))
    }

    /// How far before the container's end a worker may say the video ended:
    /// a picture shorter than its audio by a few seconds is common, a claim
    /// further back is a decode that failed and is refused.
    pub fn video_end_window(&self) -> i64 {
        let nominal = (self.info.duration_seconds * self.info.frame_rate.fps()).round() as i64;
        ((10.0 * self.info.frame_rate.fps()).round() as i64).max(nominal / 50)
    }

    /// The frame indices piece `i` holds: frames whose time is in [2i, 2i+2).
    pub fn piece_frames(&self, i: usize) -> std::ops::Range<i64> {
        let (num, den) = (self.info.frame_rate.num as i64, self.info.frame_rate.den as i64);
        let first = |seconds: i64| (seconds * num + den - 1) / den;
        let start = first(i as i64 * PIECE_SECONDS as i64);
        let end = first((i as i64 + 1) * PIECE_SECONDS as i64).min(self.frame_count());
        start..end.max(start)
    }

    pub fn done(&self) -> usize {
        self.pieces.iter().filter(|p| **p == PieceState::Done).count()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TitleStatus {
    /// ready (every piece done), running (some done or leased), queued.
    pub state: &'static str,
    pub progress: f64,
    pub pieces_done: usize,
    pub pieces_leased: usize,
    pub pieces: usize,
    pub piece_seconds: f64,
    pub chunk_seconds: f64,
    pub chunks: usize,
    pub frame_rate: f64,
}

/// A worker asking for a piece.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Claim {
    /// The worker's own stable name.
    pub worker: String,
    /// Behind a slow link: never given live pieces.
    #[serde(default)]
    pub remote: bool,
    /// The models it runs, per kind name (what its tracks' headers will say).
    pub models: Profile,
    /// Workloads it takes; all it can do when absent.
    #[serde(default)]
    pub workloads: Option<Vec<String>>,
    /// Models for particular workloads, where they differ from `models`
    /// (the 2D workload's RIFE runs at another size than the 3D one's).
    #[serde(default)]
    pub workload_models: BTreeMap<String, Profile>,
    /// The worker holds a title's state up to a frame (its last piece's end:
    /// depth anchoring, gain keyframes) and would rather carry on from there.
    #[serde(default)]
    pub continues: Option<Continues>,
}

/// A worker's run: the title it is in and the first frame it would take next.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Continues {
    pub title: String,
    pub frame: i64,
}

impl Claim {
    /// The models this worker runs for `workload`.
    pub fn models_for(&self, workload: &str) -> &Profile {
        self.workload_models.get(workload).unwrap_or(&self.models)
    }
}

/// A piece handed to a worker.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Lease {
    pub lease: u64,
    pub title: String,
    pub source: SourceRef,
    pub workload: String,
    pub kinds: Vec<Kind>,
    pub piece: usize,
    pub start_seconds: f64,
    pub end_seconds: f64,
    /// Frame indices [first, end) whose records the piece holds. A flow pair
    /// from the last of them reaches the next frame, so decode one past `end`
    /// when there is one (`frameCount` says).
    pub first_frame: i64,
    pub end_frame: i64,
    pub frame_count: i64,
    pub info: MediaInfo,
    pub lease_seconds: f64,
    /// Live work: a player is waiting for it.
    pub live: bool,
    /// The piece's server credential a player handed over, for a worker on the
    /// paired channel or loopback ([`crate::credentials`]); a worker without one
    /// reads with its own.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub credential: Option<crate::credentials::ServerCredential>,
}

#[derive(Clone, Debug)]
struct LeaseState {
    title: String,
    piece: usize,
    worker: String,
    granted: Instant,
    until: Instant,
}

#[derive(Clone, Debug)]
struct Playhead {
    title: String,
    seconds: f64,
    rate: f64,
    at: Instant,
}

impl Playhead {
    fn now(&self, now: Instant) -> f64 {
        self.seconds + self.rate * now.saturating_duration_since(self.at).as_secs_f64()
    }
}

#[derive(Clone, Debug, Default)]
struct WorkerStats {
    piece_seconds: Option<f64>,
    pieces: u64,
    last_seen: Option<Instant>,
    remote: bool,
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum UploadError {
    #[error("no such lease (it expired or was released)")]
    NoLease,
    #[error("the lease belongs to another worker")]
    NotYours,
    #[error("the track's models don't match the title's profile")]
    Profile,
    #[error("the track's picture size doesn't match the title's")]
    SourceSize,
    #[error("a record of kind {0} isn't part of this workload")]
    Kind(&'static str),
    #[error("a record at {0:.3} s is outside the piece or off the title's frame grid")]
    Time(f64),
    #[error("{kind} has {got} records where {expected} were expected")]
    Count { kind: &'static str, got: usize, expected: i64 },
    #[error("a {0} record is on the wrong eye")]
    Eye(&'static str),
    #[error("{0} records differ in size or channels within the piece")]
    Shape(&'static str),
    #[error("the video can't end at frame {0}: outside the piece, or too far before the title's end")]
    VideoEnd(i64),
}

#[derive(Debug, PartialEq)]
pub enum AddOutcome {
    Added,
    Unchanged,
    /// The file behind the title changed: its results were dropped.
    Reset,
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum AddError {
    #[error("the title isn't stereo, and this workload needs both eyes")]
    NotStereo,
    #[error("the title is stereo, and this workload is for 2D titles")]
    Stereo,
    #[error("the title's length or frame rate is unknown")]
    NoTiming,
    #[error("Blu-ray discs aren't read by workers yet")]
    Disc,
}

#[derive(Default)]
pub struct Queue {
    titles: BTreeMap<String, Title>,
    leases: HashMap<u64, LeaseState>,
    playheads: HashMap<String, Playhead>,
    workers: HashMap<String, WorkerStats>,
    next_lease: u64,
    next_ordinal: u64,
}

#[derive(Clone, Copy, PartialEq, PartialOrd)]
struct Rank {
    class: u8,
    distance: f64,
    ordinal: u64,
    piece: usize,
}

impl Queue {
    pub fn new() -> Queue {
        Queue { next_lease: 1, ..Default::default() }
    }

    /// A title read back from storage, with the pieces whose results are kept.
    pub fn restore(&mut self, meta: TitleMeta, done: &[usize]) {
        let mut pieces = vec![PieceState::Pending; Title::piece_count(&meta.info)];
        for &i in done {
            if let Some(p) = pieces.get_mut(i) {
                *p = PieceState::Done;
            }
        }
        self.next_ordinal = self.next_ordinal.max(meta.ordinal + 1);
        self.titles.insert(
            meta.key.clone(),
            Title {
                key: meta.key,
                source: meta.source,
                workload: meta.workload,
                info: meta.info,
                profile: meta.profile,
                source_size: meta.source_size,
                ordinal: meta.ordinal,
                pieces,
                video_frames: meta.video_frames,
            },
        );
    }

    pub fn title(&self, key: &str) -> Option<&Title> {
        self.titles.get(key)
    }

    pub fn titles(&self) -> impl Iterator<Item = &Title> {
        self.titles.values()
    }

    /// Adds a title, or refreshes what is known about it.
    pub fn add(&mut self, source: SourceRef, workload: &Workload, info: MediaInfo) -> Result<AddOutcome, AddError> {
        if workload.needs_stereo && !info.stereo.is_stereo() {
            return Err(AddError::NotStereo);
        }
        if workload.mono_only && info.stereo.is_stereo() {
            return Err(AddError::Stereo);
        }
        if info.stereo == crate::source::StereoLayout::Mvc {
            return Err(AddError::Disc);
        }
        if info.duration_seconds <= 0.0 || info.frame_rate.num == 0 {
            return Err(AddError::NoTiming);
        }
        let key = Title::key_for(&source, &workload.name);
        if let Some(title) = self.titles.get_mut(&key) {
            if title.info.identity == info.identity {
                return Ok(AddOutcome::Unchanged);
            }
            title.pieces = vec![PieceState::Pending; Title::piece_count(&info)];
            title.info = info;
            title.profile = None;
            title.source_size = None;
            title.video_frames = None;
            self.leases.retain(|_, l| l.title != key);
            return Ok(AddOutcome::Reset);
        }
        let pieces = vec![PieceState::Pending; Title::piece_count(&info)];
        let ordinal = self.next_ordinal;
        self.next_ordinal += 1;
        self.titles.insert(
            key.clone(),
            Title {
                key,
                source,
                workload: workload.clone(),
                info,
                profile: None,
                source_size: None,
                ordinal,
                pieces,
                video_frames: None,
            },
        );
        Ok(AddOutcome::Added)
    }

    pub fn remove(&mut self, key: &str) -> bool {
        self.leases.retain(|_, l| l.title != key);
        self.playheads.retain(|_, p| p.title != key);
        self.titles.remove(key).is_some()
    }

    /// A player says where it is playing.
    pub fn report_playhead(&mut self, player: &str, title: &str, seconds: f64, rate: f64, now: Instant) {
        self.playheads.insert(
            player.to_string(),
            Playhead { title: title.to_string(), seconds: seconds.max(0.0), rate: rate.clamp(0.0, 4.0), at: now },
        );
    }

    /// A player stopped.
    pub fn stop_playhead(&mut self, player: &str) {
        self.playheads.remove(player);
    }

    /// Returns expired leases' pieces to the queue and forgets old playheads.
    pub fn expire(&mut self, now: Instant) {
        let expired: Vec<u64> = self.leases.iter().filter(|(_, l)| l.until <= now).map(|(id, _)| *id).collect();
        for id in expired {
            if let Some(lease) = self.leases.remove(&id)
                && let Some(title) = self.titles.get_mut(&lease.title)
                && title.pieces[lease.piece] == PieceState::Leased
            {
                title.pieces[lease.piece] = PieceState::Pending;
            }
        }
        self.playheads.retain(|_, p| now.saturating_duration_since(p.at) < PLAYHEAD_TTL);
    }

    fn lease_length(&self, worker: &str) -> Duration {
        self.workers
            .get(worker)
            .and_then(|w| w.piece_seconds)
            .map(|s| Duration::from_secs_f64(s * 2.0 + 5.0).clamp(MIN_LEASE, MAX_LEASE))
            .unwrap_or(DEFAULT_LEASE)
    }

    /// The best piece for this worker, leased to it.
    pub fn claim(&mut self, claim: &Claim, now: Instant) -> Option<Lease> {
        self.expire(now);
        let stats = self.workers.entry(claim.worker.clone()).or_default();
        stats.last_seen = Some(now);
        stats.remote = claim.remote;

        let mut best: Option<(Rank, String)> = None;
        for title in self.titles.values() {
            if !Self::eligible(title, claim) {
                continue;
            }
            let heads: Vec<f64> = self.playheads.values().filter(|p| p.title == title.key).map(|p| p.now(now)).collect();
            let others: Vec<usize> = self
                .leases
                .values()
                .filter(|l| l.title == title.key && l.worker != claim.worker)
                .map(|l| l.piece)
                .collect();
            if let Some(rank) = Self::best_piece(title, &heads, claim.remote, &others)
                && best.as_ref().is_none_or(|(b, _)| rank < *b)
            {
                best = Some((rank, title.key.clone()));
            }
        }
        // Sticky runs: the piece after the worker's last one, when it is free and no more urgent
        // than what the queue would hand out otherwise (live pieces near a playhead still come first).
        if let Some(run) = &claim.continues
            && let Some(title) = self.titles.get(&run.title)
            && Self::eligible(title, claim)
            && let Some(piece) = (0..title.pieces.len()).find(|&i| title.piece_frames(i).start == run.frame)
            && title.pieces[piece] == PieceState::Pending
        {
            let heads: Vec<f64> = self.playheads.values().filter(|p| p.title == title.key).map(|p| p.now(now)).collect();
            let at = piece as f64 * PIECE_SECONDS;
            let class = heads
                .iter()
                .filter(|&&head| at + PIECE_SECONDS > head)
                .map(|&head| if at - head < LIVE_SECONDS && !claim.remote { 0 } else { 1 })
                .min()
                .unwrap_or(2);
            if best.as_ref().is_none_or(|(b, _)| class <= b.class) {
                best = Some((Rank { class, distance: 0.0, ordinal: title.ordinal, piece }, title.key.clone()));
            }
        }
        let (rank, key) = best?;
        let length = self.lease_length(&claim.worker);
        let id = self.next_lease;
        self.next_lease += 1;
        let title = self.titles.get_mut(&key).expect("the title was just found");
        title.pieces[rank.piece] = PieceState::Leased;
        if title.profile.is_none() {
            let models = claim.models_for(&title.workload.name);
            title.profile =
                Some(title.workload.kinds.iter().map(|k| (k.name().to_string(), models[k.name()].clone())).collect());
        }
        self.leases.insert(
            id,
            LeaseState { title: key.clone(), piece: rank.piece, worker: claim.worker.clone(), granted: now, until: now + length },
        );
        let title = &self.titles[&key];
        let frames = title.piece_frames(rank.piece);
        Some(Lease {
            lease: id,
            title: key,
            source: title.source.clone(),
            workload: title.workload.name.clone(),
            kinds: title.workload.kinds.clone(),
            piece: rank.piece,
            start_seconds: rank.piece as f64 * PIECE_SECONDS,
            end_seconds: ((rank.piece + 1) as f64 * PIECE_SECONDS).min(title.info.duration_seconds),
            first_frame: frames.start,
            end_frame: frames.end,
            frame_count: title.frame_count(),
            info: title.info.clone(),
            lease_seconds: length.as_secs_f64(),
            live: rank.class == 0,
            credential: None,
        })
    }

    /// The worker takes this title's workload, with the title's profile.
    fn eligible(title: &Title, claim: &Claim) -> bool {
        if let Some(names) = &claim.workloads
            && !names.contains(&title.workload.name)
        {
            return false;
        }
        let models = claim.models_for(&title.workload.name);
        if !title.workload.kinds.iter().all(|k| models.contains_key(k.name())) {
            return false;
        }
        !title.profile.as_ref().is_some_and(|profile| profile.iter().any(|(k, m)| models.get(k) != Some(m)))
    }

    /// The pending piece of `title` this worker should take, if any.
    /// `others`: pieces other workers hold now. In backfill, a piece right
    /// behind one of theirs is the one that worker carries on into next, so a
    /// new run starts halfway along the longest pending stretch instead.
    fn best_piece(title: &Title, heads: &[f64], remote: bool, others: &[usize]) -> Option<Rank> {
        let first_pending = |from: usize| (from..title.pieces.len()).find(|&i| title.pieces[i] == PieceState::Pending);
        let mut best: Option<Rank> = None;
        let mut consider = |rank: Rank| {
            if best.is_none_or(|b| rank < b) {
                best = Some(rank);
            }
        };
        for &head in heads {
            // Remote workers start past the live window.
            let from = if remote {
                ((head + LIVE_SECONDS) / PIECE_SECONDS).ceil().max(0.0) as usize
            } else {
                (head / PIECE_SECONDS).floor().max(0.0) as usize
            };
            if let Some(i) = first_pending(from) {
                let distance = i as f64 * PIECE_SECONDS - head;
                let class = if distance < LIVE_SECONDS { 0 } else { 1 };
                if remote && class == 0 {
                    continue;
                }
                consider(Rank { class, distance, ordinal: title.ordinal, piece: i });
            }
        }
        if let Some(i) = first_pending(0) {
            let behind_another = |i: usize| i > 0 && others.contains(&(i - 1));
            let piece = if behind_another(i) { Self::split_longest_stretch(title, others).unwrap_or(i) } else { i };
            consider(Rank { class: 2, distance: 0.0, ordinal: title.ordinal, piece });
        }
        best
    }

    /// Where a new run should start: the longest stretch of pending pieces,
    /// at its start when nobody is about to carry on into it, else halfway.
    fn split_longest_stretch(title: &Title, others: &[usize]) -> Option<usize> {
        let mut longest: Option<(usize, usize)> = None;
        let mut i = 0;
        while i < title.pieces.len() {
            if title.pieces[i] != PieceState::Pending {
                i += 1;
                continue;
            }
            let start = i;
            while i < title.pieces.len() && title.pieces[i] == PieceState::Pending {
                i += 1;
            }
            if longest.is_none_or(|(s, e)| i - start > e - s) {
                longest = Some((start, i));
            }
        }
        let (start, end) = longest?;
        Some(if start > 0 && others.contains(&(start - 1)) { start + (end - start) / 2 } else { start })
    }

    /// Keeps a lease alive; returns its new length in seconds.
    pub fn heartbeat(&mut self, lease: u64, worker: &str, now: Instant) -> Result<f64, UploadError> {
        let length = self.lease_length(worker);
        let state = self.leases.get_mut(&lease).ok_or(UploadError::NoLease)?;
        if state.worker != worker {
            return Err(UploadError::NotYours);
        }
        state.until = now + length;
        Ok(length.as_secs_f64())
    }

    /// The worker gives the piece back (it has to yield, or failed).
    pub fn release(&mut self, lease: u64, worker: &str) -> Result<(), UploadError> {
        let state = self.leases.get(&lease).ok_or(UploadError::NoLease)?;
        if state.worker != worker {
            return Err(UploadError::NotYours);
        }
        let state = self.leases.remove(&lease).expect("just found");
        if let Some(title) = self.titles.get_mut(&state.title)
            && title.pieces[state.piece] == PieceState::Leased
        {
            title.pieces[state.piece] = PieceState::Pending;
        }
        Ok(())
    }

    /// Checks a worker's result for its lease without accepting it yet:
    /// the right models, kinds and picture size, every record inside the
    /// piece on the title's frame grid, and about as many as there are frames.
    /// Returns the title key and piece.
    pub fn check(&self, lease: u64, worker: &str, track: &ParsedTrack) -> Result<(String, usize), UploadError> {
        let state = self.leases.get(&lease).ok_or(UploadError::NoLease)?;
        if state.worker != worker {
            return Err(UploadError::NotYours);
        }
        let title = &self.titles[&state.title];
        if let Some(profile) = &title.profile
            && profile.iter().any(|(k, m)| track.header.models.get(k) != Some(m))
        {
            return Err(UploadError::Profile);
        }
        if let Some(size) = title.source_size
            && size != (track.header.source_width, track.header.source_height)
        {
            return Err(UploadError::SourceSize);
        }
        let mut frames = title.piece_frames(state.piece);
        // A worker whose decode ended inside the piece says where the video stream ends; near the
        // container's end that is a picture shorter than its audio, and the piece holds what there is.
        let mut title_end = title.frame_count();
        if let Some(end) = track.header.video_end_frame {
            if end < frames.start || end >= frames.end || end < title_end - title.video_end_window() {
                return Err(UploadError::VideoEnd(end));
            }
            frames.end = end;
            title_end = end;
        }
        let fps = title.info.frame_rate.fps();
        let mut counts: BTreeMap<Kind, usize> = BTreeMap::new();
        let mut shapes: BTreeMap<Kind, (u8, u16, u16)> = BTreeMap::new();
        for record in &track.records {
            if !title.workload.kinds.contains(&record.kind) {
                return Err(UploadError::Kind(record.kind.name()));
            }
            if record.eye != record.kind.eye() {
                return Err(UploadError::Eye(record.kind.name()));
            }
            let shape = (record.channels, record.width, record.height);
            if *shapes.entry(record.kind).or_insert(shape) != shape {
                return Err(UploadError::Shape(record.kind.name()));
            }
            let seconds = record.time.seconds();
            let position = seconds * fps;
            let index = position.round();
            if (position - index).abs() > 0.05 || !frames.contains(&(index as i64)) {
                return Err(UploadError::Time(seconds));
            }
            *counts.entry(record.kind).or_default() += 1;
        }
        let frames_here = frames.end - frames.start;
        for kind in &title.workload.kinds {
            // A flow pair needs the next frame: the title's last frame has none.
            let expected = if kind.is_pair() && frames.end >= title_end { frames_here - 1 } else { frames_here };
            let got = counts.get(kind).copied().unwrap_or(0);
            // Decoders can drop a frame or two at a cut or a seek point.
            if (got as i64) < (expected as f64 * 0.9).floor() as i64 - 1 || got as i64 > expected + 1 {
                return Err(UploadError::Count { kind: kind.name(), got, expected: expected.max(0) });
            }
        }
        Ok((state.title.clone(), state.piece))
    }

    /// Accepts a checked result: the piece is done (whoever else holds it).
    /// With `video_end`, the title ends there: later pieces hold no frames,
    /// are done empty and are returned, for the caller to store as such.
    pub fn complete(&mut self, lease: u64, source_size: (u32, u32), video_end: Option<i64>, now: Instant)
                    -> Option<(String, usize, Vec<usize>)> {
        let state = self.leases.remove(&lease)?;
        self.leases.retain(|_, l| !(l.title == state.title && l.piece == state.piece));
        let title = self.titles.get_mut(&state.title)?;
        title.pieces[state.piece] = PieceState::Done;
        title.source_size.get_or_insert(source_size);
        let mut emptied = Vec::new();
        if let Some(end) = video_end {
            title.video_frames = Some(title.video_frames.map_or(end, |known| known.min(end)));
            for i in state.piece + 1..title.pieces.len() {
                if title.piece_frames(i).is_empty() && title.pieces[i] != PieceState::Done {
                    title.pieces[i] = PieceState::Done;
                    emptied.push(i);
                }
            }
            let key = state.title.clone();
            self.leases.retain(|_, l| !(l.title == key && emptied.contains(&l.piece)));
        }
        let stats = self.workers.entry(state.worker).or_default();
        let took = now.saturating_duration_since(state.granted).as_secs_f64();
        stats.piece_seconds = Some(stats.piece_seconds.map_or(took, |s| s * 0.7 + took * 0.3));
        stats.pieces += 1;
        stats.last_seen = Some(now);
        Some((state.title, state.piece, emptied))
    }

    pub fn status(&self, key: &str) -> Option<TitleStatus> {
        let title = self.titles.get(key)?;
        let done = title.done();
        let leased = title.pieces.iter().filter(|p| **p == PieceState::Leased).count();
        let total = title.pieces.len();
        Some(TitleStatus {
            state: if done == total {
                "ready"
            } else if done + leased > 0 {
                "running"
            } else {
                "queued"
            },
            progress: if total == 0 { 1.0 } else { done as f64 / total as f64 },
            pieces_done: done,
            pieces_leased: leased,
            pieces: total,
            piece_seconds: PIECE_SECONDS,
            chunk_seconds: CHUNK_SECONDS,
            chunks: title.chunk_count(),
            frame_rate: title.info.frame_rate.fps(),
        })
    }

    /// Chunk `n`'s done pieces, and whether it is whole.
    pub fn chunk_pieces(&self, key: &str, n: usize) -> Option<(Vec<usize>, bool)> {
        let title = self.titles.get(key)?;
        let range = n * PIECES_PER_CHUNK..((n + 1) * PIECES_PER_CHUNK).min(title.pieces.len());
        if range.is_empty() {
            return None;
        }
        let done: Vec<usize> = range.clone().filter(|&i| title.pieces[i] == PieceState::Done).collect();
        let whole = done.len() == range.len();
        Some((done, whole))
    }

    /// For the admin view.
    pub fn snapshot(&self, now: Instant) -> serde_json::Value {
        serde_json::json!({
            "titles": self.titles.values().map(|t| serde_json::json!({
                "key": t.key, "status": self.status(&t.key), "profile": t.profile,
                "stereo": t.info.stereo, "durationSeconds": t.info.duration_seconds,
            })).collect::<Vec<_>>(),
            "leases": self.leases.iter().map(|(id, l)| serde_json::json!({
                "lease": id, "title": l.title, "piece": l.piece, "worker": l.worker,
                "secondsLeft": l.until.saturating_duration_since(now).as_secs_f64(),
            })).collect::<Vec<_>>(),
            "workers": self.workers.iter().map(|(name, w)| serde_json::json!({
                "worker": name, "pieces": w.pieces, "pieceSeconds": w.piece_seconds, "remote": w.remote,
                "idleSeconds": w.last_seen.map(|t| now.saturating_duration_since(t).as_secs_f64()),
            })).collect::<Vec<_>>(),
            "playheads": self.playheads.iter().map(|(player, p)| serde_json::json!({
                "player": player, "title": p.title, "seconds": p.now(now), "rate": p.rate,
            })).collect::<Vec<_>>(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::{Rate, StereoLayout};
    use crate::track::{self, Header, tests::record};

    fn info(seconds: f64) -> MediaInfo {
        MediaInfo {
            duration_seconds: seconds,
            frame_rate: Rate { num: 24, den: 1 },
            width: 1920,
            height: 1080,
            codec: "h264".into(),
            stereo: StereoLayout::HalfSideBySide,
            identity: "a".into(),
            media_source_id: None,
        }
    }

    fn models(flow: &str) -> Profile {
        [("flowPair".to_string(), flow.to_string()), ("disparity".to_string(), "NeuFlow 512x288".to_string())].into()
    }

    fn claim(worker: &str, remote: bool) -> Claim {
        Claim {
            worker: worker.into(),
            remote,
            models: models("RIFE 480x288"),
            workloads: None,
            workload_models: BTreeMap::new(),
            continues: None,
        }
    }

    fn queue_with(seconds: f64) -> (Queue, String) {
        let mut q = Queue::new();
        let source = SourceRef::jellyfin("abc");
        q.add(source.clone(), &Workload::defaults()[0], info(seconds)).unwrap();
        (q, Title::key_for(&source, Workload::SMOOTH_MOTION_3D))
    }

    /// A worker's result for a lease, as `vergence worker` would make it.
    fn result(lease: &Lease, flow: &str) -> ParsedTrack {
        let mut data = track::write_header(&Header {
            media_id: "x".into(),
            producer: "test".into(),
            models: models(flow),
            source_width: 960,
            source_height: 1080,
            video_end_frame: None,
        });
        for k in lease.first_frame..lease.end_frame {
            if k + 1 < lease.frame_count {
                data.extend(record(Kind::FlowPair, 0, k, k + 1, 24, 2, 2));
            }
            data.extend(record(Kind::Disparity, 2, k, k, 24, 2, 2));
        }
        track::parse(&data).unwrap()
    }

    #[test]
    fn rejects_mono_titles_for_3d_work() {
        let mut q = Queue::new();
        let mut mono = info(10.0);
        mono.stereo = StereoLayout::Mono;
        assert_eq!(q.add(SourceRef::stash("1"), &Workload::defaults()[0], mono), Err(AddError::NotStereo));
    }

    #[test]
    fn backfill_goes_in_order() {
        let (mut q, _) = queue_with(10.0);
        let now = Instant::now();
        let pieces: Vec<usize> = (0..5).map(|_| q.claim(&claim("w", false), now).unwrap().piece).collect();
        assert_eq!(pieces, vec![0, 1, 2, 3, 4]);
        assert!(q.claim(&claim("w", false), now).is_none());
    }

    #[test]
    fn playheads_come_first_and_remote_workers_stay_out_of_the_live_window() {
        let (mut q, key) = queue_with(600.0);
        let now = Instant::now();
        q.report_playhead("vp", &key, 101.0, 1.0, now);
        let live = q.claim(&claim("mac", false), now).unwrap();
        assert_eq!((live.piece, live.live), (50, true));
        let remote = q.claim(&claim("far", true), now).unwrap();
        assert!(!remote.live);
        assert!(remote.start_seconds >= 101.0 + LIVE_SECONDS - PIECE_SECONDS, "{}", remote.start_seconds);
        // After a seek the old playhead's pieces stay leased, the new one's come first.
        q.report_playhead("vp", &key, 300.0, 1.0, now);
        assert_eq!(q.claim(&claim("mac", false), now).unwrap().piece, 150);
    }

    #[test]
    fn the_playhead_moves_with_time() {
        let (mut q, key) = queue_with(600.0);
        let now = Instant::now();
        q.report_playhead("vp", &key, 0.0, 1.0, now);
        assert_eq!(q.claim(&claim("mac", false), now + Duration::from_secs(10)).unwrap().piece, 5);
    }

    #[test]
    fn a_title_keeps_one_profile() {
        let (mut q, _) = queue_with(10.0);
        let now = Instant::now();
        q.claim(&claim("a", false), now).unwrap();
        let other = Claim { models: models("RIFE 960x576"), ..claim("b", false) };
        assert!(q.claim(&other, now).is_none(), "a worker with other models skips the title");
    }

    #[test]
    fn expired_leases_go_back() {
        let (mut q, _) = queue_with(2.0);
        let now = Instant::now();
        let lease = q.claim(&claim("a", false), now).unwrap();
        assert!(q.claim(&claim("b", false), now).is_none());
        let later = now + Duration::from_secs_f64(lease.lease_seconds + 1.0);
        assert_eq!(q.claim(&claim("b", false), later).unwrap().piece, 0);
    }

    #[test]
    fn checks_uploads() {
        let (mut q, key) = queue_with(5.0);
        let now = Instant::now();
        let lease = q.claim(&claim("a", false), now).unwrap();
        assert_eq!((lease.first_frame, lease.end_frame), (0, 48));
        assert_eq!(q.check(lease.lease, "b", &result(&lease, "RIFE 480x288")), Err(UploadError::NotYours));
        assert_eq!(q.check(lease.lease, "a", &result(&lease, "RIFE 960x576")), Err(UploadError::Profile));
        let good = result(&lease, "RIFE 480x288");
        assert_eq!(q.check(lease.lease, "a", &good), Ok((key.clone(), 0)));
        q.complete(lease.lease, (960, 1080), None, now);
        assert_eq!(q.status(&key).unwrap().pieces_done, 1);
        assert_eq!(q.chunk_pieces(&key, 0), Some((vec![0], false)));

        // A transcode whose timestamps restarted at zero is caught by the frame grid.
        let second = q.claim(&claim("a", false), now).unwrap();
        assert_eq!(second.piece, 1);
        let mut shifted = second.clone();
        shifted.first_frame = 0;
        shifted.end_frame = 48;
        assert!(matches!(q.check(second.lease, "a", &result(&shifted, "RIFE 480x288")), Err(UploadError::Time(_))));
        // The last piece has one flow pair fewer than frames.
        let last = q.claim(&claim("a", false), now).unwrap();
        assert_eq!((last.piece, last.end_frame, last.frame_count), (2, 120, 120));
        assert!(q.check(last.lease, "a", &result(&last, "RIFE 480x288")).is_ok());
    }

    #[test]
    fn a_replaced_file_resets_the_title() {
        let (mut q, key) = queue_with(4.0);
        let now = Instant::now();
        let lease = q.claim(&claim("a", false), now).unwrap();
        q.complete(lease.lease, (960, 1080), None, now);
        let mut replaced = info(4.0);
        replaced.identity = "b".into();
        assert_eq!(q.add(SourceRef::jellyfin("abc"), &Workload::defaults()[0], replaced), Ok(AddOutcome::Reset));
        assert_eq!(q.status(&key).unwrap().pieces_done, 0);
        assert!(q.title(&key).unwrap().profile.is_none());
    }

    fn mono_info(seconds: f64) -> MediaInfo {
        MediaInfo { stereo: StereoLayout::Mono, ..info(seconds) }
    }

    fn workload(name: &str) -> Workload {
        Workload::defaults().into_iter().find(|w| w.name == name).unwrap()
    }

    /// A 2D worker's result: one mono flow pair per frame.
    fn mono_result(lease: &Lease, flow: &str, eye: u8) -> ParsedTrack {
        let mut data = track::write_header(&Header {
            media_id: "x".into(),
            producer: "test".into(),
            models: [("flowPair".to_string(), flow.to_string())].into(),
            source_width: 1920,
            source_height: 1080,
            video_end_frame: None,
        });
        for k in lease.first_frame..lease.end_frame {
            if k + 1 < lease.frame_count {
                data.extend(record(Kind::FlowPair, eye, k, k + 1, 24, 4, 2));
            }
        }
        track::parse(&data).unwrap()
    }

    #[test]
    fn the_2d_workload_takes_only_flat_titles() {
        let mut q = Queue::new();
        let flat = workload(Workload::SMOOTH_MOTION_2D);
        assert_eq!(q.add(SourceRef::jellyfin("sbs"), &flat, info(10.0)), Err(AddError::Stereo));
        assert_eq!(q.add(SourceRef::jellyfin("flat"), &flat, mono_info(10.0)), Ok(AddOutcome::Added));
    }

    #[test]
    fn a_worker_runs_other_models_for_the_2d_workload() {
        let mut q = Queue::new();
        let source = SourceRef::jellyfin("flat");
        q.add(source.clone(), &workload(Workload::SMOOTH_MOTION_2D), mono_info(4.0)).unwrap();
        let key = Title::key_for(&source, Workload::SMOOTH_MOTION_2D);
        let now = Instant::now();
        // One worker, two sizes of RIFE: 480×288 for 3D titles, 960×576 for 2D ones.
        let mut both = claim("mac", false);
        both.workload_models.insert(
            Workload::SMOOTH_MOTION_2D.into(),
            [("flowPair".to_string(), "RIFE 960x576".to_string())].into(),
        );
        let lease = q.claim(&both, now).unwrap();
        assert_eq!((lease.workload.as_str(), lease.kinds.clone()), (Workload::SMOOTH_MOTION_2D, vec![Kind::FlowPair]));
        assert_eq!(q.title(&key).unwrap().profile.as_ref().unwrap()["flowPair"], "RIFE 960x576");
        // The title's profile now holds: a worker offering only the 3D size skips it.
        assert!(q.claim(&claim("other", false), now).is_none());
    }

    #[test]
    fn checks_2d_uploads() {
        let mut q = Queue::new();
        let source = SourceRef::jellyfin("flat");
        q.add(source.clone(), &workload(Workload::SMOOTH_MOTION_2D), mono_info(5.0)).unwrap();
        let key = Title::key_for(&source, Workload::SMOOTH_MOTION_2D);
        let now = Instant::now();
        let worker = Claim { models: [("flowPair".to_string(), "RIFE 960x576".to_string())].into(), ..claim("a", false) };
        let lease = q.claim(&worker, now).unwrap();
        assert_eq!(q.check(lease.lease, "a", &mono_result(&lease, "RIFE 960x576", 0)), Ok((key.clone(), 0)));
        // Flow on the left eye isn't what a 2D player looks up.
        assert_eq!(q.check(lease.lease, "a", &mono_result(&lease, "RIFE 960x576", 1)), Err(UploadError::Eye("flowPair")));
        // Nor are disparity records part of it.
        assert_eq!(q.check(lease.lease, "a", &result(&lease, "RIFE 960x576")), Err(UploadError::Kind("disparity")));
        // Records of one kind must share a shape.
        let good = mono_result(&lease, "RIFE 960x576", 0);
        let mut data = track::write_header(&good.header);
        for k in lease.first_frame..lease.end_frame {
            let w = if k == 10 { 8 } else { 4 };
            data.extend(record(Kind::FlowPair, 0, k, k + 1, 24, w, 2));
        }
        assert_eq!(q.check(lease.lease, "a", &track::parse(&data).unwrap()), Err(UploadError::Shape("flowPair")));
        // Too few records (a decode that stopped early).
        let mut short = track::write_header(&good.header);
        for k in lease.first_frame..lease.first_frame + 20 {
            short.extend(record(Kind::FlowPair, 0, k, k + 1, 24, 4, 2));
        }
        assert!(matches!(q.check(lease.lease, "a", &track::parse(&short).unwrap()), Err(UploadError::Count { .. })));
        q.complete(lease.lease, (1920, 1080), None, now);
        assert_eq!(q.status(&key).unwrap().pieces_done, 1);
    }

    #[test]
    fn checks_depth_uploads() {
        let mut q = Queue::new();
        let source = SourceRef::jellyfin("flat");
        let depth = workload(Workload::AI_3D_DEPTH);
        assert_eq!(q.add(SourceRef::jellyfin("sbs"), &depth, info(4.0)), Err(AddError::Stereo));
        q.add(source.clone(), &depth, mono_info(4.0)).unwrap();
        let now = Instant::now();
        let profile: Profile = [("depth".to_string(), "DA2 Small 770x574".to_string())].into();
        // A Smooth Motion worker doesn't get depth work.
        assert!(q.claim(&claim("flow", false), now).is_none());
        let worker = Claim { models: profile.clone(), ..claim("a", false) };
        let lease = q.claim(&worker, now).unwrap();
        assert_eq!(lease.kinds, vec![Kind::Depth]);
        let upload = |eye: u8, frames: std::ops::Range<i64>| {
            let mut data = track::write_header(&Header {
                media_id: "x".into(),
                producer: "test".into(),
                models: profile.clone(),
                source_width: 1280,
                source_height: 720,
                video_end_frame: None,
            });
            for k in frames {
                data.extend(record(Kind::Depth, eye, k, k, 24, 8, 4));
            }
            track::parse(&data).unwrap()
        };
        // One depth record a frame, every frame of the piece (none needs a next frame).
        assert!(q.check(lease.lease, "a", &upload(0, 0..48)).is_ok());
        assert_eq!(q.check(lease.lease, "a", &upload(2, 0..48)), Err(UploadError::Eye("depth")));
        // The context frames before and after a piece stay with the worker.
        assert!(matches!(q.check(lease.lease, "a", &upload(0, 0..50)), Err(UploadError::Time(_))));
    }

    #[test]
    fn a_mac_and_a_pc_split_a_depth_title_by_its_shared_profile() {
        // Depth Anything V2 Small has one depth profile on Core ML and ONNX
        // Runtime; Smooth Motion's flow keeps each family's own.
        let shared = "Depth Anything V2 Small 770x574 offline-v5 context-1s";
        let worker = |name: &str, flow: &str, depth: &str| Claim {
            worker: name.into(),
            remote: false,
            models: models(flow),
            workloads: Some(vec![Workload::SMOOTH_MOTION_3D.into(), Workload::AI_3D_DEPTH.into()]),
            workload_models: BTreeMap::from([
                (Workload::SMOOTH_MOTION_3D.to_string(), models(flow)),
                (Workload::AI_3D_DEPTH.to_string(), [("depth".to_string(), depth.to_string())].into()),
            ]),
            continues: None,
        };
        let mac = worker("mac", "RIFE425lite_480x288 480x288", shared);
        let pc = worker("pc", "RIFE425lite_480x288 480x288 onnx", shared);
        let mut q = Queue::new();
        let source = SourceRef::jellyfin("film");
        q.add(source.clone(), &workload(Workload::AI_3D_DEPTH), mono_info(6.0)).unwrap();
        let now = Instant::now();
        let first = q.claim(&mac, now).unwrap();
        let second = q.claim(&pc, now).unwrap();
        assert_eq!((first.title.clone(), first.workload.as_str()), (second.title.clone(), Workload::AI_3D_DEPTH));
        assert_ne!(first.piece, second.piece);
        // A worker naming its own engine's depth doesn't join the title.
        let other = worker("other", "RIFE 480x288", "DepthAnythingV2Small 770x574 offline-v5 context-1s onnx");
        assert!(q.claim(&other, now).is_none());
        // The 3D flow profiles stay apart: a title one family started is the other's no longer.
        q.add(SourceRef::jellyfin("sbs"), &workload(Workload::SMOOTH_MOTION_3D), info(4.0)).unwrap();
        let mac_flow = Claim { workloads: Some(vec![Workload::SMOOTH_MOTION_3D.into()]), ..mac.clone() };
        let pc_flow = Claim { workloads: Some(vec![Workload::SMOOTH_MOTION_3D.into()]), ..pc.clone() };
        assert!(q.claim(&mac_flow, now).is_some());
        assert!(q.claim(&pc_flow, now).is_none());
    }

    #[test]
    fn checks_gain_grid_uploads() {
        let mut q = Queue::new();
        let source = SourceRef::stash("7");
        q.add(source.clone(), &workload(Workload::AI_HDR), mono_info(2.0)).unwrap();
        let now = Instant::now();
        let profile: Profile = [("gainMap".to_string(), "GainMapGMNet grid16x9x8 key500ms".to_string())].into();
        let lease = q.claim(&Claim { models: profile.clone(), ..claim("a", false) }, now).unwrap();
        assert_eq!((lease.kinds.clone(), lease.first_frame, lease.end_frame), (vec![Kind::GainMap], 0, 48));
        let upload = |w: u16, h: u16| {
            let mut data = track::write_header(&Header {
                media_id: "x".into(),
                producer: "test".into(),
                models: profile.clone(),
                source_width: 1280,
                source_height: 720,
                video_end_frame: None,
            });
            for k in 0..48 {
                data.extend(record(Kind::GainMap, 0, k, k, 24, if k == 47 { w } else { 16 }, h));
            }
            track::parse(&data).unwrap()
        };
        // Every frame's grid, all the same shape (the record helper writes one channel).
        assert!(q.check(lease.lease, "a", &upload(16, 9)).is_ok());
        assert_eq!(q.check(lease.lease, "a", &upload(8, 9)), Err(UploadError::Shape("gainMap")));
    }

    #[test]
    fn a_video_shorter_than_its_container_ends_the_title() {
        // The 4K demo: 109.8 s by the container (2635 frames at 24), its picture 2548 frames.
        let mut q = Queue::new();
        let source = SourceRef::jellyfin("demo");
        q.add(source.clone(), &workload(Workload::AI_3D_DEPTH), mono_info(109.8)).unwrap();
        let key = Title::key_for(&source, Workload::AI_3D_DEPTH);
        let now = Instant::now();
        let profile: Profile = [("depth".to_string(), "DA2 Small".to_string())].into();
        let worker = Claim { models: profile.clone(), ..claim("a", false) };
        let leases: Vec<Lease> = std::iter::from_fn(|| q.claim(&worker, now)).collect();
        assert_eq!(leases.len(), 55);
        let upload = |frames: std::ops::Range<i64>, end: Option<i64>| {
            let mut data = track::write_header(&Header {
                media_id: "x".into(),
                producer: "test".into(),
                models: profile.clone(),
                source_width: 1280,
                source_height: 720,
                video_end_frame: end,
            });
            for k in frames {
                data.extend(record(Kind::Depth, 0, k, k, 24, 8, 4));
            }
            track::parse(&data).unwrap()
        };
        // A decode that stopped mid-film can't claim the end.
        let middle = &leases[41];
        assert_eq!(q.check(middle.lease, "a", &upload(1968..2000, Some(2000))), Err(UploadError::VideoEnd(2000)));
        // Without the claim, the short last piece fails its count, as before.
        let last = &leases[53];
        assert_eq!((last.first_frame, last.end_frame), (2544, 2592));
        assert!(matches!(q.check(last.lease, "a", &upload(2544..2548, None)), Err(UploadError::Count { .. })));
        // With it, the piece holds the four frames there are, and the title ends at 2548.
        assert!(q.check(last.lease, "a", &upload(2544..2548, Some(2548))).is_ok());
        let (_, _, emptied) = q.complete(last.lease, (1280, 720), Some(2548), now).unwrap();
        assert_eq!(emptied, vec![54]);
        assert_eq!(q.title(&key).unwrap().frame_count(), 2548);
        assert_eq!(q.title(&key).unwrap().meta().video_frames, Some(2548));
        for lease in &leases[..53] {
            let frames = lease.first_frame..lease.end_frame;
            q.check(lease.lease, "a", &upload(frames, None)).unwrap();
            q.complete(lease.lease, (1280, 720), None, now);
        }
        assert_eq!(q.status(&key).unwrap().state, "ready");
    }

    #[test]
    fn a_worker_carries_on_into_the_next_piece_of_its_run() {
        let mut q = Queue::new();
        let older = SourceRef::jellyfin("older");
        let source = SourceRef::jellyfin("run");
        q.add(older.clone(), &workload(Workload::AI_3D_DEPTH), mono_info(20.0)).unwrap();
        q.add(source.clone(), &workload(Workload::AI_3D_DEPTH), mono_info(20.0)).unwrap();
        let key = Title::key_for(&source, Workload::AI_3D_DEPTH);
        let now = Instant::now();
        let profile: Profile = [("depth".to_string(), "DA2 Small".to_string())].into();
        let worker = |name: &str, continues: Option<Continues>| Claim {
            models: profile.clone(),
            continues,
            ..claim(name, false)
        };
        // Backfill would take the older title first; a worker in a run of the newer one carries on.
        let run = Continues { title: key.clone(), frame: 48 };
        let next = q.claim(&worker("a", Some(run.clone())), now).unwrap();
        assert_eq!((next.title.as_str(), next.piece), (key.as_str(), 1));
        // Not when that piece is taken: then the queue's own choice.
        let taken = q.claim(&worker("b", Some(run)), now).unwrap();
        assert_ne!((taken.title.as_str(), taken.piece), (key.as_str(), 1));
        // A live piece near someone's playhead still comes first.
        q.report_playhead("vp", &Title::key_for(&older, Workload::AI_3D_DEPTH), 10.0, 1.0, now);
        let live = q.claim(&worker("a", Some(Continues { title: key.clone(), frame: 96 })), now).unwrap();
        assert!(live.live && live.title != key, "{live:?}");
    }

    #[test]
    fn a_second_worker_starts_its_run_halfway_along_the_rest() {
        let mut q = Queue::new();
        let source = SourceRef::jellyfin("long");
        q.add(source.clone(), &workload(Workload::AI_3D_DEPTH), mono_info(200.0)).unwrap();
        let now = Instant::now();
        let profile: Profile = [("depth".to_string(), "DA2 Small".to_string())].into();
        let a = q.claim(&Claim { models: profile.clone(), ..claim("a", false) }, now).unwrap();
        assert_eq!(a.piece, 0);
        // Worker b would sit right behind a's run; it starts in the middle of the 99 pieces left.
        let b = q.claim(&Claim { models: profile.clone(), ..claim("b", false) }, now).unwrap();
        assert_eq!(b.piece, 1 + 99 / 2);
        // One worker claiming on its own still goes in order.
        let a2 = q.claim(&Claim { models: profile, ..claim("a", false) }, now).unwrap();
        assert_eq!(a2.piece, 1);
    }
}
