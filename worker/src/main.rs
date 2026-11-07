//! `vergence-worker`: analysis for a coordinator (`vergenced`) with ONNX
//! Runtime, for the PCs `vergence worker` (Core ML) can't run on.
//!
//! ```text
//! vergence-worker --config FILE [--once] [--ep auto|cuda|tensorrt|directml|coreml|cpu]
//! vergence-worker bench --models DIR [--ep NAME] [--compute-units all|gpu|ane|cpu] [--runs N]
//! ```
//!
//! Claims 2 s pieces from the coordinator, reads each from the media server
//! itself through a loopback forwarder (the credentials never reach ffmpeg's
//! command line), runs RIFE 4.25 lite and NeuFlow v2 exported to ONNX
//! (`scripts/export-onnx-models.py`), and uploads the records as a small
//! track: the same claim, heartbeat, release and upload loop, config file,
//! decoding and records as the Swift worker. Its models are named
//! `"<model> <size> onnx"` in track headers, so a title never mixes Core ML
//! and ONNX Runtime results, unless the config sets `shareCoreMLProfile`.
//!
//! It runs at low priority and stands aside (no claims, its lease released,
//! its models unloaded so the GPU's memory goes back too) while the machine
//! is wanted: see `yielding`. `--once` takes one piece and
//! exits (for tests); otherwise it polls until stopped, backing off while
//! there is nothing to do. `bench` times the models alone. Design: Hypnos
//! `docs/plans/analysis-workers.md`.

mod analyze;
mod client;
mod config;
mod engine;
mod forwarder;
mod gain;
mod preprocess;
mod record;
mod workloads;
mod yielding;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::analyze::{Failure, Piece};
use crate::client::{Coordinator, Lease};
use crate::config::{Config, expand_tilde};
use crate::engine::{Engine, EngineOptions, Provider, Wanted};
use crate::forwarder::Forwarder;
use crate::preprocess::{DISPARITY_SIZE, FLOW_SIZE};
use crate::yielding::Yield;

fn log(message: impl AsRef<str>) {
    eprintln!("vergence-worker: {}", message.as_ref());
}

fn usage() -> ExitCode {
    eprintln!(
        "usage:\n  vergence-worker --config FILE [--once] [--ep auto|cuda|tensorrt|directml|coreml|cpu]\n  \
         vergence-worker bench --models DIR [--ep NAME] [--compute-units all|gpu|ane|cpu] [--runs N]"
    );
    ExitCode::from(2)
}

fn option(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let provider = match option(&args, "--ep").map(|n| Provider::parse(&n)) {
        Some(None) => return usage(),
        Some(Some(p)) => Some(p),
        None => None,
    };
    if args.first().map(String::as_str) == Some("bench") {
        let Some(models) = option(&args, "--models") else { return usage() };
        let runs = option(&args, "--runs").and_then(|r| r.parse().ok()).unwrap_or(20);
        let options = EngineOptions { compute_units: option(&args, "--compute-units"), ..Default::default() };
        return match bench(&expand_tilde(&models), provider.unwrap_or(Provider::Auto), &options, runs) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                log(e);
                ExitCode::FAILURE
            }
        };
    }
    let Some(config_path) = option(&args, "--config").map(|p| expand_tilde(&p)) else { return usage() };
    let once = args.iter().any(|a| a == "--once");
    let runtime = tokio::runtime::Builder::new_multi_thread().enable_all().build().expect("a tokio runtime");
    match runtime.block_on(run(&config_path, provider, once)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            log(e);
            ExitCode::FAILURE
        }
    }
}

/// Below the user's own work; ffmpeg inherits it.
fn lower_priority() {
    #[cfg(unix)]
    unsafe {
        libc::setpriority(libc::PRIO_PROCESS as _, 0, 10);
    }
    #[cfg(windows)]
    unsafe {
        use windows_sys::Win32::System::Threading::{BELOW_NORMAL_PRIORITY_CLASS, GetCurrentProcess, SetPriorityClass};
        SetPriorityClass(GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS);
    }
}

fn host_name() -> String {
    std::env::var("COMPUTERNAME")
        .or_else(|_| std::env::var("HOSTNAME"))
        .ok()
        .or_else(|| {
            std::process::Command::new("hostname").output().ok().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        })
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| "worker".into())
}

/// What the track headers name each kind's model, per workload: the
/// coordinator's profiles. The Mac worker's names, with " onnx" after them
/// unless `shareCoreMLProfile` (a title never mixes the two families).
fn profiles(engine: &Engine, wanted: &Wanted, share_core_ml: bool) -> BTreeMap<String, BTreeMap<String, String>> {
    let suffix = if share_core_ml { "" } else { " onnx" };
    let one = |kind: &str, name: String| BTreeMap::from([(kind.to_string(), name)]);
    let mut out = BTreeMap::new();
    if let Some(stereo) = &engine.stereo {
        let (fw, fh) = FLOW_SIZE;
        let (dw, dh) = DISPARITY_SIZE;
        out.insert(workloads::SMOOTH_MOTION_3D.to_string(), BTreeMap::from([
            ("flowPair".to_string(), format!("{} {fw}x{fh}{suffix}", stereo.flow.name)),
            ("disparity".to_string(), format!("{} {dw}x{dh}{suffix}", stereo.disparity.name)),
        ]));
    }
    if let Some(mono) = &engine.mono {
        out.insert(workloads::SMOOTH_MOTION_2D.to_string(),
                   one("flowPair", format!("{} {}x{}{suffix}", mono.name, mono.width, mono.height)));
    }
    if let (Some(depth), Some((path, _))) = (&engine.depth, &wanted.depth) {
        let model = depth.model.lock().unwrap_or_else(|p| p.into_inner());
        out.insert(workloads::AI_3D_DEPTH.to_string(),
                   one("depth", workloads::depth_profile(path, (model.width, model.height), suffix)));
    }
    if let Some(gain) = &engine.gain {
        let key = (gain::KEYFRAME_SECONDS * 1000.0).round() as i64;
        out.insert(workloads::AI_HDR.to_string(), one("gainMap", format!("{} grid16x9x8 key{key}ms hl2{suffix}", gain.name)));
    }
    out
}

/// The workloads the config asks for, or every one its models allow (3D
/// always, 2D when the larger RIFE is there, depth and AI HDR when their
/// models are named), as `vergence worker` decides.
fn wanted(config: &Config) -> Result<Wanted, String> {
    let named = config.workloads.as_ref();
    let asked = |w: &str| named.map(|n| n.iter().any(|x| x == w));
    let known = [workloads::SMOOTH_MOTION_3D, workloads::SMOOTH_MOTION_2D, workloads::AI_3D_DEPTH, workloads::AI_HDR];
    if let Some(unknown) = named.and_then(|n| n.iter().find(|w| !known.contains(&w.as_str()))) {
        return Err(format!("this worker doesn't make {unknown}"));
    }
    let depth = match (asked(workloads::AI_3D_DEPTH).unwrap_or(config.depth_model.is_some()), &config.depth_model) {
        (false, _) => None,
        (true, None) => return Err("ai-3d-depth needs depthModel in the config".into()),
        (true, Some(path)) => {
            let path = expand_tilde(path);
            // Refused without the noncommercial-models feature; its licence said with it.
            if let Some(line) = vergence_convert::licence::check(&path)? {
                log(line);
            }
            Some((path, config.depth_size()?))
        }
    };
    let gain = match (asked(workloads::AI_HDR).unwrap_or(config.gain_model.is_some()), &config.gain_model) {
        (false, _) => None,
        (true, None) => return Err("ai-hdr needs gainModel in the config".into()),
        (true, Some(path)) => Some(expand_tilde(path)),
    };
    let wanted = Wanted {
        stereo: asked(workloads::SMOOTH_MOTION_3D).unwrap_or(true),
        mono: match asked(workloads::SMOOTH_MOTION_2D) {
            Some(false) => None,
            Some(true) => Some(true),
            None => Some(false),
        },
        depth,
        gain,
    };
    if !wanted.stereo && wanted.mono.is_none() && wanted.depth.is_none() && wanted.gain.is_none() {
        return Err("the config asks for no workload this worker makes".into());
    }
    Ok(wanted)
}

/// How long the machine must have been free before the worker goes back to work.
const BACK_TO_WORK_AFTER: Duration = Duration::from_secs(30);

/// Loads the models again after standing aside, on the provider that loaded first.
type Loader = Arc<dyn Fn() -> Result<Engine, String> + Send + Sync>;

struct Worker {
    config: Config,
    coordinator: Coordinator,
    /// None while standing aside: the models are unloaded then.
    engine: Arc<Mutex<Option<Engine>>>,
    loader: Loader,
    /// Per workload: what its track headers name each kind's model.
    profiles: BTreeMap<String, BTreeMap<String, String>>,
    producer: String,
    yielder: Arc<dyn Yield>,
    /// A sticky run: the title and frame the last piece ended at, when its
    /// workload keeps state for the next (AI HDR and AI 3D depth).
    last_run: Mutex<Option<(String, i64)>>,
}

async fn run(config_path: &Path, provider: Option<Provider>, once: bool) -> Result<(), String> {
    let config = Config::load(config_path)?;
    let config_dir = config_path.parent().map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from("."));
    let provider = provider
        .or_else(|| config.execution_provider.as_deref().and_then(Provider::parse))
        .unwrap_or(Provider::Auto);
    let options = EngineOptions {
        compute_units: config.compute_units.clone(),
        device_id: config.device_id,
        cache_dir: Some(config_dir.join("model-cache")),
    };
    if let Some(dir) = &options.cache_dir {
        let _ = std::fs::create_dir_all(dir);
    }
    let models_dir = expand_tilde(&config.models);
    let wanted = wanted(&config)?;
    let engine = {
        let (models_dir, options, wanted) = (models_dir.clone(), options.clone(), wanted.clone());
        tokio::task::spawn_blocking(move || Engine::load(&models_dir, &wanted, provider, &options))
            .await
            .map_err(|e| e.to_string())??
    };
    let loaded = engine.provider;
    let profiles = profiles(&engine, &wanted, config.share_core_ml_profile.unwrap_or(false));
    // Loads again exactly what loaded: a 2D model found once is required after.
    let again = Wanted { mono: engine.mono.as_ref().map(|_| true), ..wanted };
    let loader: Loader = Arc::new(move || Engine::load(&models_dir, &again, loaded, &options));
    let name = config.name.clone().unwrap_or_else(host_name);
    let producer = format!("vergence-worker {} (ONNX Runtime, {})", env!("CARGO_PKG_VERSION"), engine.provider.name());
    lower_priority();
    let worker = Worker {
        coordinator: Coordinator::new(&config.coordinator, config.token()?, name.clone())?,
        yielder: Arc::new(yielding::for_this_machine(&config_dir, config.yield_processes.clone().unwrap_or_default())),
        config,
        engine: Arc::new(Mutex::new(Some(engine))),
        loader,
        profiles,
        producer,
        last_run: Mutex::new(None),
    };
    let list: Vec<String> = worker
        .profiles
        .iter()
        .map(|(workload, models)| format!("{workload} ({})", models.values().cloned().collect::<Vec<_>>().join(", ")))
        .collect();
    log(format!("{name} on {}: {}", worker.producer, list.join("; ")));
    worker.run(once).await
}

impl Worker {
    async fn yield_reason(&self) -> Option<String> {
        let yielder = self.yielder.clone();
        tokio::task::spawn_blocking(move || yielder.reason()).await.ok().flatten()
    }

    /// Gives the GPU's memory back while standing aside.
    async fn unload(&self) {
        let engine = self.engine.clone();
        let dropped = tokio::task::spawn_blocking(move || engine.lock().unwrap_or_else(|p| p.into_inner()).take().is_some())
            .await
            .unwrap_or(false);
        if dropped {
            log("unloaded the models (their GPU memory is free again)");
        }
    }

    async fn ensure_loaded(&self) -> Result<(), String> {
        let (engine, loader) = (self.engine.clone(), self.loader.clone());
        tokio::task::spawn_blocking(move || {
            let mut slot = engine.lock().unwrap_or_else(|p| p.into_inner());
            if slot.is_none() {
                let started = Instant::now();
                *slot = Some(loader()?);
                log(format!("loaded the models again in {:.1} s", started.elapsed().as_secs_f64()));
            }
            Ok(())
        })
        .await
        .map_err(|e| e.to_string())?
    }

    async fn run(&self, once: bool) -> Result<(), String> {
        let mut idle = 2u64;
        let mut standing_aside: Option<String> = None;
        loop {
            if let Some(reason) = self.yield_reason().await {
                if standing_aside.as_ref() != Some(&reason) {
                    log(format!("standing aside: {reason}"));
                }
                self.unload().await;
                if once {
                    return Ok(());
                }
                standing_aside = Some(reason);
                tokio::time::sleep(Duration::from_secs(5)).await;
                continue;
            } else if standing_aside.is_some() {
                // Only once the machine has been free for a while: a stream
                // showing a still desktop encodes little, and a game between
                // levels is quiet for a moment.
                let mut quiet = Duration::ZERO;
                while quiet < BACK_TO_WORK_AFTER && self.yield_reason().await.is_none() {
                    tokio::time::sleep(Duration::from_secs(5)).await;
                    quiet += Duration::from_secs(5);
                }
                if quiet < BACK_TO_WORK_AFTER {
                    continue;
                }
                standing_aside = None;
                log("back to work");
            }
            if let Err(e) = self.ensure_loaded().await {
                log(format!("the models didn't load again: {e}"));
                if once {
                    return Err(e);
                }
                tokio::time::sleep(Duration::from_secs(30)).await;
                continue;
            }
            let run = self.last_run.lock().unwrap_or_else(|p| p.into_inner()).clone();
            let lease = match self.coordinator.claim(self.config.remote.unwrap_or(false), &self.profiles, run.as_ref()).await {
                Ok(lease) => lease,
                Err(e) => {
                    log(format!("claim failed: {e}"));
                    if once {
                        return Err(e);
                    }
                    tokio::time::sleep(Duration::from_secs(10)).await;
                    continue;
                }
            };
            let Some(lease) = lease else {
                if once {
                    log("nothing to do");
                    return Ok(());
                }
                tokio::time::sleep(Duration::from_secs(idle)).await;
                idle = (idle * 2).min(30);
                continue;
            };
            idle = 2;
            let result = self.work(&lease).await;
            if let Err(e) = &result {
                if once {
                    return Err(e.clone());
                }
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
            if once {
                return Ok(());
            }
        }
    }

    /// One lease: analyse, upload, or give it back.
    async fn work(&self, lease: &Lease) -> Result<(), String> {
        let started = Instant::now();
        let stop = Arc::new(AtomicBool::new(false));
        let heartbeat = {
            let coordinator = self.coordinator.clone();
            let (id, every) = (lease.lease, Duration::from_secs_f64((lease.lease_seconds / 3.0).max(5.0)));
            tokio::spawn(async move {
                loop {
                    tokio::time::sleep(every).await;
                    let _ = coordinator.heartbeat(id).await;
                }
            })
        };
        let watcher = {
            let (yielder, stop) = (self.yielder.clone(), stop.clone());
            tokio::spawn(async move {
                loop {
                    tokio::time::sleep(Duration::from_secs(3)).await;
                    let y = yielder.clone();
                    if let Ok(Some(reason)) = tokio::task::spawn_blocking(move || y.reason()).await {
                        log(format!("standing aside mid-piece: {reason}"));
                        stop.store(true, Ordering::Relaxed);
                        break;
                    }
                }
            })
        };
        let outcome = self.analyze(lease, stop).await;
        heartbeat.abort();
        watcher.abort();
        let outcome = match outcome {
            Ok(track) => {
                let size = track.len();
                self.coordinator.upload(lease.lease, track).await.map(|()| size).map_err(Failure::Error)
            }
            Err(e) => Err(e),
        };
        let carried = self.last_run.lock().unwrap_or_else(|p| p.into_inner()).take()
            .is_some_and(|(title, frame)| title == lease.title && frame == lease.first_frame);
        match outcome {
            Ok(size) => {
                log(format!(
                    "piece {} of {} ({}{}) in {:.1} s, {:.0} KB",
                    lease.piece,
                    lease.title,
                    if lease.live { "live" } else { "ahead/backfill" },
                    if carried { ", carried on" } else { "" },
                    started.elapsed().as_secs_f64(),
                    size as f64 / 1024.0
                ));
                // AI HDR keeps its keyframes across pieces, AI 3D depth its anchoring.
                if self.config.sticky_runs != Some(false)
                    && (lease.workload == workloads::AI_HDR || lease.workload == workloads::AI_3D_DEPTH)
                    && lease.end_frame < lease.frame_count
                {
                    *self.last_run.lock().unwrap_or_else(|p| p.into_inner()) = Some((lease.title.clone(), lease.end_frame));
                }
                Ok(())
            }
            Err(failure) => {
                let _ = self.coordinator.release(lease.lease).await;
                match failure {
                    Failure::Yielded(reason) => {
                        log(format!("gave back piece {} of {}: {reason}", lease.piece, lease.title));
                        Ok(())
                    }
                    Failure::Error(e) => {
                        log(format!("piece {} of {} failed: {e}", lease.piece, lease.title));
                        Err(e)
                    }
                }
            }
        }
    }

    async fn analyze(&self, lease: &Lease, stop: Arc<AtomicBool>) -> Result<Vec<u8>, Failure> {
        let (url, headers) = self.media_request(lease)?;
        let forwarder = Forwarder::start(url, headers).await.map_err(|e| format!("the loopback forwarder: {e}"))?;
        let (lease, url) = (lease.clone(), forwarder.url());
        let models = self.profiles.get(&lease.workload).cloned().ok_or(format!("a lease for {}, which this worker didn't ask for", lease.workload))?;
        let (engine, producer) = (self.engine.clone(), self.producer.clone());
        let ffmpeg = self.config.ffmpeg.as_deref().map(|p| expand_tilde(p).to_string_lossy().into_owned());
        let sticky = self.config.sticky_runs != Some(false);
        let result = tokio::task::spawn_blocking(move || {
            let piece = Piece { lease: &lease, url: &url, ffmpeg: ffmpeg.as_deref(), models: &models, producer: &producer, sticky };
            let mut slot = engine.lock().unwrap_or_else(|p| p.into_inner());
            let engine = slot.as_mut().ok_or_else(|| Failure::Error("the models aren't loaded".into()))?;
            analyze::run(&piece, engine, &stop)
        })
        .await
        .map_err(|e| Failure::Error(format!("the analysis panicked: {e}")))?;
        drop(forwarder);
        result
    }

    /// The piece's URL on its server and the header that authorises it.
    fn media_request(&self, lease: &Lease) -> Result<(String, Vec<(String, String)>), String> {
        match lease.source.server.as_str() {
            "jellyfin" => {
                let server = self.config.jellyfin.as_ref().ok_or("no Jellyfin in the worker's config")?;
                let mut url = format!("{}/Videos/{}/stream?static=true", server.url.trim_end_matches('/'), lease.source.id);
                if let Some(id) = &lease.info.media_source_id {
                    url += &format!("&mediaSourceId={id}");
                }
                let headers = server
                    .key()?
                    .map(|k| vec![("Authorization".to_string(), format!("MediaBrowser Client=\"vergence\", Token=\"{k}\""))])
                    .unwrap_or_default();
                Ok((url, headers))
            }
            "stash" => {
                let server = self.config.stash.as_ref().ok_or("no Stash in the worker's config")?;
                let url = format!("{}/scene/{}/stream", server.url.trim_end_matches('/'), lease.source.id);
                Ok((url, server.key()?.map(|k| vec![("ApiKey".to_string(), k)]).unwrap_or_default()))
            }
            other => Err(format!("unknown server {other}")),
        }
    }
}

/// Times each model alone and both at once, as a piece runs them.
fn bench(models: &Path, provider: Provider, options: &EngineOptions, runs: usize) -> Result<(), String> {
    let loading = Instant::now();
    let wanted = Wanted { stereo: true, mono: Some(false), ..Default::default() };
    let mut engine = Engine::load(models, &wanted, provider, options)?;
    let stereo = engine.stereo.as_mut().expect("asked for");
    log(format!(
        "{} and {}{} on {} (loaded and warmed in {:.1} s)",
        stereo.flow.name,
        stereo.disparity.name,
        engine.mono.as_ref().map(|m| format!(" and {}", m.name)).unwrap_or_default(),
        engine.provider.name(),
        loading.elapsed().as_secs_f64()
    ));
    let (fw, fh) = FLOW_SIZE;
    let (dw, dh) = DISPARITY_SIZE;
    let noise = |n: usize, seed: u32| -> Vec<f32> {
        (0..n).map(|i| ((i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed) >> 8) as f32 / 16_777_216.0).collect()
    };
    let (a, b) = (noise(3 * fw * fh, 1), noise(3 * fw * fh, 2));
    let (l, r) = (noise(3 * dw * dh, 3), noise(3 * dw * dh, 4));
    let time = |f: &mut dyn FnMut() -> Result<(), String>| -> Result<f64, String> {
        let start = Instant::now();
        for _ in 0..runs {
            f()?;
        }
        Ok(start.elapsed().as_secs_f64() * 1000.0 / runs as f64)
    };
    let flow_ms = time(&mut || stereo.flow.run(&a, &b).map(drop))?;
    let disparity_ms = time(&mut || stereo.disparity.run(&l, &r).map(drop))?;
    let engine::StereoModels { flow, disparity } = stereo;
    let both_ms = time(&mut || {
        std::thread::scope(|s| {
            let job = s.spawn(|| flow.run(&a, &b).map(drop));
            let d = disparity.run(&l, &r).map(drop);
            job.join().expect("flow panicked").and(d)
        })
    })?;
    log(format!(
        "RIFE {fw}x{fh} {flow_ms:.1} ms, NeuFlow {dw}x{dh} {disparity_ms:.1} ms, both at once {both_ms:.1} ms \
         a frame ({:.1} fps) over {runs} runs",
        1000.0 / both_ms
    ));
    if let Some(mono) = engine.mono.as_mut() {
        let (w, h) = (mono.width, mono.height);
        let (a, b) = (noise(3 * w * h, 5), noise(3 * w * h, 6));
        let ms = time(&mut || mono.run(&a, &b).map(drop))?;
        log(format!("RIFE {w}x{h} (2D) {ms:.1} ms a frame ({:.1} fps)", 1000.0 / ms));
    }
    Ok(())
}
