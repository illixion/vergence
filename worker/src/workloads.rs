//! The workloads beyond Smooth Motion for 3D titles, as `vergence worker`'s
//! producers make them (`Sources/VergenceCLI/WorkerProducers.swift`):
//!
//! - `smooth-motion-2d`: the whole picture's flow between frames, RIFE at
//!   960×576, stored at 240×144 with the finer 2D step (`MonoFlowProducer`).
//! - `ai-hdr`: GMNet on keyframes (every half second on the title's grid and
//!   at every cut), each gain map fitted into a 16×9×8 grid, one `gainMap`
//!   record a frame eased between them; a piece decodes one keyframe interval
//!   either side, so it sees the keyframes a whole-file run sees
//!   (`GainGridProducer`).
//! - `ai-3d-depth`: the offline depth pipeline (`vergence_convert::depth`, the
//!   Rust port of `DepthConverter` the Windows converter runs) over the piece
//!   with a second of context before it and two frames after, decoded first to
//!   a ProRes intermediate of at most 1036 pixels as the Mac worker decodes it
//!   (`DepthProducer`).

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Command, Stdio};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::analyze::{Failure, Piece, ffmpeg_input_args, frames_wanted};
use crate::client::Lease;
use crate::engine::{Engine, GAIN_SIZES};
use crate::gain::{self, GainGrid, Keyframe};
use crate::preprocess::{rgb_planar, rgb_planar_parallel, whole};
use crate::record::{EYE_MONO, KIND_GAIN_MAP, Record, Time, flow_pair_record_with_step, track_header};

pub const SMOOTH_MOTION_3D: &str = "smooth-motion-3d";
pub const SMOOTH_MOTION_2D: &str = "smooth-motion-2d";
pub const AI_3D_DEPTH: &str = "ai-3d-depth";
pub const AI_HDR: &str = "ai-hdr";

/// `SharedFieldAnalysis.monoFlowStep`: 2D flow is stored at a finer step.
const MONO_FLOW_STEP: f32 = 1.0 / 16.0;
/// Seconds of context the depth pipeline settles over before a piece.
pub const DEPTH_CONTEXT_SECONDS: f64 = 1.0;
/// `DepthCacheStore.pipelineVersion`, which the Mac's profile names.
pub const DEPTH_PIPELINE_VERSION: u32 = 5;

fn time(lease: &Lease, index: i64) -> Time {
    let rate = lease.info.frame_rate;
    Time { value: index * rate.den, timescale: rate.num as i32 }
}

fn mono_size(lease: &Lease) -> Result<(usize, usize), Failure> {
    if lease.info.stereo != "mono" {
        return Err(format!("the title is stereo ({}); {} is for 2D", lease.info.stereo, lease.workload).into());
    }
    let (w, h) = (lease.info.width, lease.info.height);
    if w == 0 || h == 0 || w % 4 != 0 || h % 4 != 0 {
        return Err(format!("the picture's size ({w}x{h}) is unusable").into());
    }
    Ok((w, h))
}

/// ffmpeg decoding `count` whole frames from `first`, as I420, and its
/// messages read on the side (so a chatty decode can't stall).
struct Decode {
    child: Child,
    output: Option<ChildStdout>,
    messages: Option<std::thread::JoinHandle<String>>,
}

impl Decode {
    fn start(piece: &Piece, first: i64, count: i64) -> Result<Decode, Failure> {
        let mut args = ffmpeg_input_args(piece.lease, piece.url, first);
        args.extend(
            ["-map", "0:v:0", "-fps_mode", "passthrough", "-frames:v", &count.to_string(), "-f", "rawvideo", "-pix_fmt",
             "yuv420p", "-"]
                .map(String::from),
        );
        let mut child = command(piece.ffmpeg.unwrap_or("ffmpeg"))
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("ffmpeg didn't start: {e}"))?;
        let mut stderr = child.stderr.take().expect("piped");
        let messages = std::thread::spawn(move || {
            let mut text = String::new();
            let _ = stderr.read_to_string(&mut text);
            text
        });
        Ok(Decode { output: child.stdout.take(), child, messages: Some(messages) })
    }

    /// Stops ffmpeg; its messages, for an error.
    fn finish(mut self) -> String {
        let _ = self.child.kill();
        let _ = self.child.wait();
        self.messages.take().and_then(|m| m.join().ok()).unwrap_or_default().chars().take(300).collect()
    }
}

fn command(program: &str) -> Command {
    #[allow(unused_mut)]
    let mut command = Command::new(program);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000);
    }
    command
}

/// Reads frames on a thread of its own, turning each into whatever `prepare`
/// makes of it, while the caller runs the models.
fn frames<T: Send + 'static>(output: ChildStdout, frame_bytes: usize, first: i64, count: i64,
                             prepare: impl Fn(i64, &[u8]) -> T + Send + 'static) -> std::sync::mpsc::Receiver<(i64, T)> {
    let (send, receive) = std::sync::mpsc::sync_channel(3);
    std::thread::spawn(move || {
        let mut output = output;
        let mut frame = vec![0u8; frame_bytes];
        for index in first..first + count {
            if output.read_exact(&mut frame).is_err() || send.send((index, prepare(index, &frame))).is_err() {
                break;
            }
        }
    });
    receive
}

fn stopped(stop: &AtomicBool) -> Result<(), Failure> {
    if stop.load(Ordering::Relaxed) { Err(Failure::Yielded("asked to stop".into())) } else { Ok(()) }
}

// MARK: - smooth-motion-2d

pub fn mono_flow(piece: &Piece, engine: &mut Engine, stop: &AtomicBool) -> Result<Vec<u8>, Failure> {
    let lease = piece.lease;
    let (w, h) = mono_size(lease)?;
    let model = engine.mono.as_mut().ok_or_else(|| Failure::Error("the 2D flow model isn't loaded".into()))?;
    let (mw, mh) = (model.width, model.height);
    let mut track = track_header(&lease.title, piece.producer, piece.models, w as u32, h as u32);
    let wanted = frames_wanted(lease);
    let mut decode = Decode::start(piece, lease.first_frame, wanted)?;
    let inputs = frames(decode.output.take().expect("piped"), w * h * 3 / 2, lease.first_frame, wanted, move |_, frame| {
        let [y, cb, cr] = whole(frame, w, h);
        rgb_planar_parallel(y, cb, cr, mw, mh, 1.0)
    });
    // Records are packed on a thread of their own, while the next pair runs.
    let (records, packing) = std::sync::mpsc::sync_channel::<(Vec<f32>, Vec<f32>, Time, Time)>(3);
    let packer = std::thread::spawn(move || {
        for (flow, mask, from, to) in packing {
            track.extend(flow_pair_record_with_step(&flow, &mask, mw, mh, from, to, MONO_FLOW_STEP).deflated().framed());
        }
        track
    });
    let mut previous: Option<(Vec<f32>, i64)> = None;
    let mut decoded = 0;
    let result = (|| -> Result<(), Failure> {
        for (index, picture) in inputs.iter() {
            stopped(stop)?;
            if let Some((before, at)) = previous.take().filter(|(_, at)| *at < lease.end_frame) {
                let (flow, mask) = model.run(&before, &picture)?;
                records
                    .send((flow, mask, time(lease, at), time(lease, index)))
                    .map_err(|_| Failure::Error("the record packer stopped".into()))?;
            }
            previous = Some((picture, index));
            decoded = index + 1 - lease.first_frame;
        }
        Ok(())
    })();
    drop(inputs);
    drop(records);
    let track = packer.join().map_err(|_| Failure::Error("the record packer panicked".into()))?;
    let messages = decode.finish();
    result?;
    // Frames ran out inside the piece: the video stream ends there (the coordinator takes that
    // only near the title's end).
    if decoded > 0 && decoded < lease.end_frame - lease.first_frame {
        return Ok(crate::record::with_video_end(track, lease.first_frame + decoded));
    }
    if decoded < lease.end_frame - lease.first_frame {
        return Err(format!("ffmpeg gave {decoded} of {wanted} frames: {messages}").into());
    }
    Ok(track)
}

// MARK: - ai-hdr

/// What the decoding thread passes on for a frame.
struct Seen {
    cut: bool,
    /// A keyframe's picture, for GMNet.
    frame: Option<Vec<u8>>,
}

pub fn gain_grids(piece: &Piece, engine: &mut Engine, stop: &AtomicBool) -> Result<Vec<u8>, Failure> {
    let lease = piece.lease;
    let (w, h) = mono_size(lease)?;
    let rate = lease.info.frame_rate;
    let interval = gain::keyframe_interval(rate.num as f64 / rate.den as f64);
    // A sticky run carries on after the last frame the previous piece decoded, with its keyframes
    // and that frame's signature, so this piece's context isn't decoded again.
    let run = engine.gain_run.take().filter(|run| {
        run.title == lease.title && run.next == lease.first_frame && run.last_decoded + 1 >= lease.first_frame
            && run.last_decoded + 1 < lease.end_frame
    });
    let model = engine.gain.as_mut().ok_or_else(|| Failure::Error("GMNet isn't loaded".into()))?;
    let first = run.as_ref().map_or((lease.first_frame - interval).max(0), |run| run.last_decoded + 1);
    let last = lease.frame_count.min(lease.end_frame + interval);
    let mut decode = Decode::start(piece, first, last - first)?;
    // The cut detector runs in order, on the decoding thread.
    let previous = std::sync::Mutex::new(run.as_ref().map(|run| run.signature.clone()));
    let last_signature = std::sync::Arc::new(std::sync::Mutex::new(None::<Vec<f32>>));
    let keep = last_signature.clone();
    let seen = frames(decode.output.take().expect("piped"), w * h * 3 / 2, first, last - first, move |index, frame| {
        let [y, _, _] = whole(frame, w, h);
        let signature = gain::signature(&y);
        let mut previous = previous.lock().unwrap_or_else(|p| p.into_inner());
        let cut = previous.as_ref().is_some_and(|p| gain::is_cut(p, &signature));
        *keep.lock().unwrap_or_else(|p| p.into_inner()) = Some(signature.clone());
        *previous = Some(signature);
        Seen { cut, frame: gain::is_keyframe(index, interval, cut).then(|| frame.to_vec()) }
    });
    let mut keyframes: Vec<Keyframe> = run
        .map(|run| run.keyframes.into_iter().filter(|k| k.index >= lease.first_frame - 2 * interval).collect())
        .unwrap_or_default();
    let mut decoded_to = first;
    let result = (|| -> Result<(), Failure> {
        for (index, Seen { cut, frame }) in seen.iter() {
            stopped(stop)?;
            decoded_to = index + 1;
            let Some(frame) = frame else { continue };
            let [y, cb, cr] = whole(&frame, w, h);
            let aspect = w as f64 / h.max(1) as f64;
            let &(iw, ih) = GAIN_SIZES
                .iter()
                .min_by(|a, b| {
                    let d = |s: &(usize, usize)| (s.0 as f64 / s.1 as f64 / aspect).ln().abs();
                    d(a).partial_cmp(&d(b)).unwrap_or(std::cmp::Ordering::Equal)
                })
                .expect("sizes");
            let image = rgb_planar(y, cb, cr, iw, ih, 1.0);
            let thumbnail = rgb_planar(y, cb, cr, 256, 256, 1.0);
            let (gain, gw, gh) = model.run(&image, iw, ih, &thumbnail)?;
            let rgb = rgb_planar(y, cb, cr, gw, gh, 1.0);
            let n = gw * gh;
            let luma: Vec<f32> = (0..n).map(|i| gain::luma(rgb[i], rgb[n + i], rgb[2 * n + i])).collect();
            keyframes.push(Keyframe { index, grid: GainGrid::fit(&gain, &luma, gw, gh).expanding_highlights(), cut });
        }
        Ok(())
    })();
    drop(seen);
    let messages = decode.finish();
    result?;
    let ended_early = decoded_to > lease.first_frame && decoded_to < lease.end_frame;
    if decoded_to < lease.end_frame && !ended_early {
        return Err(format!("ffmpeg gave frames up to {decoded_to} of {first}..{last}: {messages}").into());
    }
    let mut track = track_header(&lease.title, piece.producer, piece.models, w as u32, h as u32);
    for index in lease.first_frame..lease.end_frame.min(decoded_to) {
        if let Some(grid) = gain::grid_at(index, &keyframes) {
            let at = time(lease, index);
            let record = Record::float16(&grid.0, KIND_GAIN_MAP, EYE_MONO, gain::LUMA_BINS, gain::WIDTH, gain::HEIGHT, at, at);
            track.extend(record.deflated().framed());
        }
    }
    if !ended_early && let Some(signature) = last_signature.lock().unwrap_or_else(|p| p.into_inner()).take() {
        engine.gain_run = Some(gain::Run {
            title: lease.title.clone(),
            next: lease.end_frame,
            last_decoded: decoded_to - 1,
            signature,
            keyframes,
        });
    }
    Ok(if ended_early { crate::record::with_video_end(track, decoded_to) } else { track })
}

// MARK: - ai-3d-depth

/// The ffmpeg and ffprobe the depth pipeline runs: beside the configured
/// ffmpeg, else found as the converter finds them.
fn tools(ffmpeg: Option<&str>) -> Result<vergence_convert::probe::Tools, String> {
    if let Some(ffmpeg) = ffmpeg {
        let ffmpeg = PathBuf::from(ffmpeg);
        let ffprobe = ffmpeg.with_file_name(if cfg!(windows) { "ffprobe.exe" } else { "ffprobe" });
        if ffprobe.is_file() {
            return Ok(vergence_convert::probe::Tools { ffmpeg, ffprobe });
        }
    }
    vergence_convert::probe::Tools::locate()
}

/// The HEVC encoder the depth maps are stored with between the passes, found once.
fn encoder(tools: &vergence_convert::probe::Tools) -> Result<String, String> {
    static ENCODER: OnceLock<String> = OnceLock::new();
    if let Some(name) = ENCODER.get() {
        return Ok(name.clone());
    }
    let name = vergence_convert::decode::hevc_encoder(tools)?;
    Ok(ENCODER.get_or_init(|| name).clone())
}

/// Removes the piece's temporary folder however the piece ends.
struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A sticky depth run (`DepthProducer.run` on the Mac): the title, where its
/// next piece starts, and the shot-space ranges the anchoring locked for the
/// last piece's final frames, which the next piece's overlap starts from.
#[derive(Clone, Debug)]
pub struct DepthRun {
    pub title: String,
    pub next: i64,
    pub locks: BTreeMap<i64, vergence_convert::maths::ShotRange>,
}

/// Frames a carried-on piece decodes before its own: the temporal window's
/// reach. The anchoring's state comes from the last piece, so its second of
/// context isn't needed (`DepthProducer.runOverlap`).
pub const DEPTH_RUN_OVERLAP: i64 = 2;

pub fn depth(piece: &Piece, engine: &mut Engine, stop: &AtomicBool) -> Result<Vec<u8>, Failure> {
    use vergence_convert::{decode::Span, depth};
    let lease = piece.lease;
    let (w, h) = mono_size(lease)?;
    // Carrying on a run: the anchoring goes on from the lock the last piece gave this
    // piece's first overlap frame, as one decode of the whole title would have it.
    let previous = engine.depth_run.take();
    let seed = previous
        .as_ref()
        .filter(|run| piece.sticky && run.title == lease.title && run.next == lease.first_frame)
        .and_then(|run| run.locks.get(&(lease.first_frame - DEPTH_RUN_OVERLAP)).copied());
    let depth_engine = engine.depth.as_ref().ok_or_else(|| Failure::Error("the depth model isn't loaded".into()))?;
    let rate = lease.info.frame_rate;
    let before = (DEPTH_CONTEXT_SECONDS * rate.num as f64 / rate.den as f64).round() as i64;
    let first = if seed.is_some() { lease.first_frame - DEPTH_RUN_OVERLAP } else { (lease.first_frame - before).max(0) };
    let last = lease.frame_count.min(lease.end_frame + 2);
    let wanted = last - first;
    let scratch = Scratch(std::env::temp_dir().join(format!("vergence-depth-{}-{}", std::process::id(), lease.lease)));
    std::fs::create_dir_all(&scratch.0).map_err(|e| format!("{}: {e}", scratch.0.display()))?;
    let intermediate = scratch.0.join("piece.mov");

    let mut args = ffmpeg_input_args(lease, piece.url, first);
    args.extend(
        ["-map", "0:v:0", "-fps_mode", "passthrough", "-frames:v", &wanted.to_string(), "-vf",
         "scale='min(iw,1036)':'min(ih,1036)':force_original_aspect_ratio=decrease:force_divisible_by=2", "-c:v",
         "prores_ks", "-profile:v", "2", "-pix_fmt", "yuv422p10le", "-an", "-sn", "-dn", "-y"]
            .map(String::from),
    );
    args.push(intermediate.to_string_lossy().into_owned());
    let made = command(piece.ffmpeg.unwrap_or("ffmpeg"))
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("ffmpeg didn't start: {e}"))?;
    if !made.status.success() {
        let message: String = String::from_utf8_lossy(&made.stderr).chars().take(300).collect();
        return Err(format!("ffmpeg couldn't make the piece ({}): {message}", made.status).into());
    }
    stopped(stop)?;

    let tools = tools(piece.ffmpeg)?;
    let probe = tools.probe(&intermediate)?;
    let encoder = encoder(&tools)?;
    let directory = scratch.0.join("depth");
    let (entry, _) = depth::convert(
        depth::Job {
            tools: &tools,
            input: &intermediate,
            probe: &probe,
            span: Span { start: 0.0, length: None, frames: None },
            first_frame: 0,
            expected_frames: wanted as usize,
            encoder: &encoder,
            directory: &directory,
            model: depth_engine.model.clone(),
            progress: &mut |_| {},
            seed,
            worker_run: true,
        },
        &depth_engine.gpu,
    )?;
    let lock = entry.meta.shot_lock.clone();
    stopped(stop)?;
    let maps = depth::maps(&tools, &entry)?;
    let decoded_to = first + maps.len() as i64;
    // ffmpeg made the intermediate cleanly, short of the piece: the video stream ends there.
    let ended_early = decoded_to > lease.first_frame && decoded_to < lease.end_frame.min(last);
    if decoded_to < lease.end_frame.min(last) && !ended_early {
        return Err(format!("the depth pass gave {} of {wanted} frames", maps.len()).into());
    }
    let mut track = track_header(&lease.title, piece.producer, piece.models, w as u32, h as u32);
    // The intermediate's k-th frame is the title's frame first + k.
    for (k, map) in maps.into_iter().enumerate() {
        let index = first + k as i64;
        if index < lease.first_frame || index >= lease.end_frame {
            continue;
        }
        let at = time(lease, index);
        let record = vergence_convert::track::Record::depth(map.payload, map.width, map.height, at.value, at.timescale,
                                                            map.scale, map.offset);
        track.extend(record.deflated_framed());
    }
    // Keep the anchoring's state at the end of this piece for the next one.
    if piece.sticky
        && !ended_early
        && let Some(lock) = lock.filter(|l| l.len() == 2 * (decoded_to - first) as usize)
    {
        let locks = (0..(decoded_to - first) as usize)
            .map(|k| (first + k as i64, vergence_convert::maths::ShotRange { lo: lock[2 * k], hi: lock[2 * k + 1] }))
            .filter(|(index, _)| *index >= lease.end_frame - DEPTH_RUN_OVERLAP)
            .collect();
        engine.depth_run = Some(DepthRun { title: lease.title.clone(), next: lease.end_frame, locks });
    }
    Ok(if ended_early { crate::record::with_video_end(track, decoded_to) } else { track })
}

/// The depth profile's name. Depth Anything V2 Small and Base get one name
/// whatever runs them (family, input size, licence, pipeline, context), so a
/// Mac (`vergence worker`, Core ML) and a PC can split a title: their records
/// agree within the tolerance `scripts/check-onnx-workload.sh` measured (Hypnos
/// `docs/plans/analysis-workers.md`, question 3); the Swift worker's
/// `DepthProducer.profile` says the same. Any other model keeps its file's name
/// and the engine family's suffix. A restrictive licence is in the name either
/// way, so a non-commercial model's titles never mix with Small's.
pub fn depth_profile(model: &Path, (w, h): (usize, usize), suffix: &str) -> String {
    let stem = model.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let licence = vergence_convert::licence::restriction_of(model).map(|r| format!(" {}", r.name)).unwrap_or_default();
    let tail = format!("{licence} offline-v{DEPTH_PIPELINE_VERSION} context-1s");
    match shared_depth_family(&stem) {
        Some(family) => format!("{family} {w}x{h}{tail}"),
        None => format!("{stem} {w}x{h}{tail}{suffix}"),
    }
}

/// The model families whose depth is shared across engines.
pub fn shared_depth_family(model_name: &str) -> Option<&'static str> {
    let name = model_name.to_lowercase().replace(['-', '_'], "");
    if !name.contains("depthanything") || !name.contains("v2") || name.contains("video") {
        return None;
    }
    if name.contains("small") {
        Some("Depth Anything V2 Small")
    } else if name.contains("base") {
        Some("Depth Anything V2 Base")
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_the_depth_profile() {
        // Shared with the Mac worker's Core ML names (`DepthProducer.profile`).
        assert_eq!(depth_profile(Path::new("/m/DepthAnythingV2Small.onnx"), (770, 574), " onnx"),
                   "Depth Anything V2 Small 770x574 offline-v5 context-1s");
        assert_eq!(depth_profile(Path::new("/m/DepthAnythingV2Base.onnx"), (518, 392), " onnx"),
                   "Depth Anything V2 Base 518x392 CC-BY-NC-4.0 offline-v5 context-1s");
        // Not checked across engines: its own name and the family's suffix.
        assert_eq!(depth_profile(Path::new("/m/DepthAnythingV2Large.onnx"), (518, 392), " onnx"),
                   "DepthAnythingV2Large 518x392 CC-BY-NC-4.0 offline-v5 context-1s onnx");
        assert_eq!(shared_depth_family("VideoDepthAnythingV2Small"), None);
    }
}
