//! Pass 1: depth for every frame, as `DepthConverter` makes it for a single
//! sweep. Each frame is decoded at up to 1036 pixels a side, inferred (the
//! model on one thread, everything after it on another), its robust range and
//! histogram taken, joint-bilaterally refined on its own edges at the model's
//! size and upsampled ×2 guided by the picture, anchored on the previous
//! frame's static pixels, checked for a cut, and emitted two frames later as
//! the ±2-frame window (cut-aware, depth-gated, each neighbour brought into
//! the centre's units) normalised into its own range, 8-bit. The emitted maps
//! are an HEVC video in the work folder; the post-pass turns the per-frame
//! statistics into each frame's display mapping (locked across the shot,
//! smoothed), which goes into `meta.json` beside it.
//!
//! Differences from the Mac, on purpose: one sweep at every frame rate (the
//! Mac sweeps sources of 48 fps and up twice, every other frame each time, so
//! a cache becomes playable sooner; a file conversion gains nothing from it),
//! and the anchor sums are folded in f64 directly.

use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Instant;

use half::f16;
use serde::{Deserialize, Serialize};

use crate::decode::{FrameReader, FrameWriter, Span, encoder_args, input_args, scale_matrix};
use crate::gpu::{DepthRefiner, Gpu, Refined};
use crate::maths::{
    AlignmentSums, AnchorCeiling, BINS, CutDetector, FrameChangeDetector, STATIC_LUMA_THRESHOLD, ScaleShiftTracker,
    ShotFrame, TEMPORAL_DEPTH_SIGMA, letterbox, percentile, relative_mapping, shot_mapping, temporal_window,
};
use crate::model::{self, DepthModel};
use crate::probe::{Probe, Rate, Tools};

/// Frames are decoded to fit inside this before depth (`DepthConverter.maxDecodeDimension`).
const MAX_DECODE_DIMENSION: usize = 1036;
pub const META_VERSION: u32 = 1;
const DEPTH_BITRATE: u64 = 4_000_000;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Meta {
    pub version: u32,
    pub model_name: String,
    pub completed: bool,
    pub frame_count: usize,
    /// The source's first frame index this depth starts at.
    pub first_frame: i64,
    pub rate: (u64, u64),
    /// What was asked for: the part's length in seconds.
    pub duration: f64,
    pub decoded_width: usize,
    pub decoded_height: usize,
    /// The stored maps' size (twice the model's).
    pub depth_width: usize,
    pub depth_height: usize,
    pub uv_scale: [f32; 2],
    pub uv_offset: [f32; 2],
    /// Stored byte / 255 → display depth: `v · scale + bias`, per frame.
    pub display_scale: Vec<f32>,
    pub display_bias: Vec<f32>,
}

pub struct Entry {
    pub directory: PathBuf,
    pub meta: Meta,
}

impl Entry {
    pub fn video(&self) -> PathBuf {
        self.directory.join("depth.mkv")
    }

    /// A finished entry in `directory`, if there is one.
    pub fn at(directory: &Path) -> Option<Entry> {
        let bytes = std::fs::read(directory.join("meta.json")).ok()?;
        let meta: Meta = serde_json::from_slice(&bytes).ok()?;
        let entry = Entry { directory: directory.to_path_buf(), meta };
        (entry.meta.completed && entry.meta.version == META_VERSION && entry.video().is_file()).then_some(entry)
    }
}

/// The decode size: within 1036 a side, never larger than the frame, even.
pub fn decode_size(width: usize, height: usize) -> (usize, usize) {
    display_decode_size(width as f64, height as f64)
}

/// The same from the picture's display size (non-square pixels applied to its
/// width, as AVFoundation's `naturalSize` gives it the Mac's converter).
pub fn display_decode_size(width: f64, height: f64) -> (usize, usize) {
    let scale = (MAX_DECODE_DIMENSION as f64 / width.max(height)).min(1.0);
    (((width * scale) as usize) & !1, ((height * scale) as usize) & !1)
}

/// The picture's width with its pixel aspect applied.
pub fn display_width(width: usize, sample_aspect_ratio: Option<&str>) -> f64 {
    let ratio = sample_aspect_ratio
        .and_then(|r| r.split_once(':'))
        .and_then(|(n, d)| Some((n.parse::<f64>().ok()?, d.parse::<f64>().ok()?)))
        .filter(|&(n, d)| n > 0.0 && d > 0.0)
        .map(|(n, d)| n / d)
        .unwrap_or(1.0);
    width as f64 * ratio
}

struct Stat {
    lo: f32,
    hi: f32,
    cut_before: bool,
    relative: Option<crate::maths::Affine>,
}

struct Inferred {
    /// RGB 0…1 interleaved at decode size, as the guide sees it.
    video: Vec<f16>,
    /// None: static against the last inferred frame, its depth repeats.
    raw: Option<Vec<f16>>,
}

pub struct Job<'a> {
    pub tools: &'a Tools,
    pub input: &'a Path,
    pub probe: &'a Probe,
    pub span: Span,
    pub first_frame: i64,
    pub expected_frames: usize,
    pub encoder: &'a str,
    pub directory: &'a Path,
    /// Loaded once and kept: a converter's files and a worker's pieces share it.
    pub model: std::sync::Arc<std::sync::Mutex<DepthModel>>,
    pub progress: &'a mut dyn FnMut(f64),
}

/// Makes the depth for `job` into its directory (replacing what is there).
pub fn convert(job: Job, gpu: &Gpu) -> Result<(Entry, Option<String>), String> {
    let video = job.probe.video().ok_or("no video stream")?;
    let (width, height) = (video.width.unwrap_or(0), video.height.unwrap_or(0));
    if width < 2 || height < 2 {
        return Err("the video has no picture size".into());
    }
    let _ = std::fs::remove_dir_all(job.directory);
    std::fs::create_dir_all(job.directory).map_err(|e| format!("{}: {e}", job.directory.display()))?;
    let (dw, dh) = display_decode_size(display_width(width, video.sample_aspect_ratio.as_deref()), height as f64);
    let hdr = job.probe.is_hdr();
    let hlg = video.color_transfer.as_deref() == Some("arib-std-b67");
    let colour = job.probe.colour();
    let mut args = input_args(job.input, job.probe, job.span);
    args.extend([
        "-vf".to_string(),
        format!("scale={dw}:{dh}:flags=area:in_color_matrix={}:in_range={}", scale_matrix(&colour.matrix),
                if colour.full_range { "pc" } else { "tv" }),
        "-f".into(),
        "rawvideo".into(),
        "-pix_fmt".into(),
        if hdr { "rgb48le" } else { "rgb24" }.into(),
        "-".into(),
    ]);
    let bytes_per = if hdr { 6 } else { 3 };
    let reader = FrameReader::start(job.tools, &args, dw * dh * bytes_per)?;
    let shared = job.model.clone();
    let (mw, mh, model_name, provider_name) = {
        let m = shared.lock().unwrap_or_else(|p| p.into_inner());
        (m.width, m.height, m.name.clone(), m.provider.name().to_string())
    };

    // Stage A: decode and prepare on one thread, infer on another.
    let (prepared_send, prepared_receive) = mpsc::sync_channel::<(Vec<f16>, Option<Vec<f32>>)>(3);
    let preparing_thread = std::thread::spawn(move || {
        let mut reader = reader;
        let mut gate = FrameChangeDetector::default();
        let mut frame = Vec::new();
        let mut preparing = 0f64;
        while reader.next(&mut frame) {
            let started = Instant::now();
            let (video, input) = prepare(&frame, hdr.then_some(hlg), (dw, dh), (mw, mh), &mut gate);
            preparing += started.elapsed().as_secs_f64();
            if prepared_send.send((video, input)).is_err() {
                break;
            }
        }
        if std::env::var_os("VERGENCE_CONVERT_DEBUG").is_some() {
            eprintln!("  depth timing: preparing {preparing:.2} s");
        }
        reader.finish()
    });
    let (send, receive) = mpsc::sync_channel::<Result<Inferred, String>>(2);
    let stage_a = std::thread::spawn(move || -> Result<(), String> {
        let mut model = shared.lock().unwrap_or_else(|p| p.into_inner());
        let mut inferring = 0f64;
        for (video, input) in prepared_receive.iter() {
            let raw = match input {
                None => None,
                Some(input) => {
                    let started = Instant::now();
                    let inferred = model.infer(&input);
                    inferring += started.elapsed().as_secs_f64();
                    match inferred {
                        Ok(map) => Some(map.into_iter().map(f16::from_f32).collect()),
                        Err(e) => {
                            let _ = send.send(Err(e));
                            break;
                        }
                    }
                }
            };
            if send.send(Ok(Inferred { video, raw })).is_err() {
                break;
            }
        }
        drop(prepared_receive);
        if std::env::var_os("VERGENCE_CONVERT_DEBUG").is_some() {
            eprintln!("  depth timing: inferring {inferring:.2} s");
        }
        preparing_thread.join().map_err(|_| "the decoder's thread panicked".to_string())?
    });

    let mut refiner = DepthRefiner::new(gpu, (dw, dh), (mw, mh));
    let (hw, hh) = refiner.hi_size;
    let rate = job.probe.rate();
    let gop = (rate.fps() * 2.0).round().max(1.0) as usize;
    let mut writer_args: Vec<String> = ["-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p"].map(String::from).to_vec();
    writer_args.extend(["-s".into(), format!("{hw}x{hh}"), "-r".into(), format!("{}/{}", rate.num, rate.den), "-i".into(), "-".into()]);
    writer_args.extend(encoder_args(job.encoder, DEPTH_BITRATE, gop, false));
    writer_args.extend(["-pix_fmt".into(), "yuv420p".into()]);
    let partial = job.directory.join("depth.partial.mkv");
    writer_args.push(partial.to_string_lossy().into_owned());
    let mut writer = FrameWriter::start(job.tools, &writer_args)?;
    let neutral = vec![128u8; (hw / 2) * (hh / 2) * 2];

    let mut stats: Vec<Stat> = Vec::new();
    let mut cuts = CutDetector::default();
    let mut tracker = ScaleShiftTracker::default();
    let mut tracker_seconds: Option<f64> = None;
    let mut background = AnchorCeiling::NONE;
    let mut previous: Option<Refined> = None;
    let mut pair = 0;
    let mut next_emit = 0;
    let fps = rate.fps();
    let total = job.expected_frames.max(1);
    let emit = |stats: &[Stat], c: usize, last: usize, refiner: &mut DepthRefiner, writer: &mut FrameWriter| -> Result<(), String> {
        let cut_flags: Vec<bool> = stats.iter().map(|s| s.cut_before).collect();
        let relatives: Vec<Option<crate::maths::Affine>> = stats.iter().map(|s| s.relative).collect();
        let (weights, frames) = temporal_window(&cut_flags, c, last);
        let mut slots = [0usize; 5];
        let mut affines = [(1f32, 0f32); 5];
        for i in 0..5 {
            slots[i] = frames[i] % 5;
            if weights[i] > 0.0 {
                let m = relative_mapping(&relatives, frames[i], c);
                affines[i] = (m.a, m.b);
            }
        }
        let stat = &stats[c];
        let span = (stat.hi - stat.lo).max(1e-6);
        let luma = refiner.encode(slots, weights, affines, stat.lo, span, TEMPORAL_DEPTH_SIGMA * span)?;
        writer.write(&luma)?;
        writer.write(&neutral)
    };
    let (mut waited, mut refining, mut emitting) = (0f64, 0f64, 0f64);
    let mut mark = Instant::now();
    for item in receive.iter() {
        waited += mark.elapsed().as_secs_f64();
        let Inferred { video, raw } = item?;
        let started = Instant::now();
        let index = stats.len();
        let seconds = index as f64 / fps;
        match raw {
            Some(raw) => {
                let values: Vec<f32> = raw.iter().map(|v| v.to_f32()).collect();
                let (lo, hi) = values.iter().filter(|v| v.is_finite()).fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
                let (raw_lo, raw_hi) = if lo.is_finite() && hi.is_finite() { (lo, hi) } else { (0.0, 1.0) };
                let mut bins = vec![0u32; BINS];
                let inv = 1.0 / (raw_hi - raw_lo).max(1e-6);
                for &v in values.iter().filter(|v| v.is_finite()) {
                    bins[(((v - raw_lo) * inv).clamp(0.0, 0.999999) * 256.0) as usize] += 1;
                }
                let count: usize = bins.iter().map(|&b| b as usize).sum();
                let p02 = percentile(&bins, count, 0.02, raw_lo, raw_hi);
                let mut p98 = percentile(&bins, count, 0.98, raw_lo, raw_hi);
                if p98 - p02 < 1e-6 {
                    p98 = p02 + 1.0;
                }
                let refined = refiner.refine(&raw, &video, pair, index % 5)?;
                let anchor = previous.as_ref().map(|prev| anchor_sums(&refined, prev, background));
                let shape = CutDetector::shape(&bins);
                let cut_before = index > 0 && cuts.is_cut(shape.clone(), p02, p98, anchor.as_ref().map(|a| a.1));
                if index == 0 {
                    cuts.is_cut(shape, p02, p98, None);
                }
                let relative = if cut_before { None } else { anchor.as_ref().and_then(|(sums, _)| sums.fit(mw * mh, p98 - p02)) };
                let dt = tracker_seconds.map(|s| (seconds - s).max(0.0)).unwrap_or(0.0);
                let live = tracker.update(p02, p98, relative, cut_before || index == 0, dt);
                tracker_seconds = Some(seconds);
                background = AnchorCeiling::new(live);
                previous = Some(refined);
                pair = 1 - pair;
                if std::env::var_os("VERGENCE_CONVERT_DEBUG").is_some() {
                    eprintln!("frame {index}: lo {p02:.4} hi {p98:.4} cut {cut_before} relative {relative:?} luma {:?} anchor w {:?} live {live:?}",
                              anchor.as_ref().map(|a| a.1), anchor.as_ref().map(|a| a.0.w / (mw * mh) as f64));
                }
                stats.push(Stat { lo: p02, hi: p98, cut_before, relative });
            }
            None => {
                if std::env::var_os("VERGENCE_CONVERT_DEBUG").is_some() {
                    eprintln!("frame {index}: static");
                }
                if index == 0 {
                    return Err("depth inference failed on the first frame".into());
                }
                refiner.repeat((index - 1) % 5, index % 5);
                let repeated = &stats[index - 1];
                stats.push(Stat { lo: repeated.lo, hi: repeated.hi, cut_before: false, relative: Some(crate::maths::Affine::IDENTITY) });
            }
        }
        refining += started.elapsed().as_secs_f64();
        let started = Instant::now();
        while next_emit + crate::maths::TEMPORAL_RADIUS < stats.len() {
            emit(&stats, next_emit, stats.len() - 1, &mut refiner, &mut writer)?;
            next_emit += 1;
        }
        emitting += started.elapsed().as_secs_f64();
        mark = Instant::now();
        if stats.len().is_multiple_of(10) {
            (job.progress)((stats.len() as f64 / total as f64).min(1.0));
        }
    }
    if std::env::var_os("VERGENCE_CONVERT_DEBUG").is_some() {
        eprintln!("  depth timing: waiting for inference {waited:.2} s, refining {refining:.2} s, emitting {emitting:.2} s");
    }
    stage_a.join().map_err(|_| "the depth model's thread panicked".to_string())??;
    if stats.is_empty() {
        return Err("no video frames in the range".into());
    }
    while next_emit < stats.len() {
        emit(&stats, next_emit, stats.len() - 1, &mut refiner, &mut writer)?;
        next_emit += 1;
    }
    writer.finish()?;
    std::fs::rename(&partial, job.directory.join("depth.mkv")).map_err(|e| e.to_string())?;

    // Post-pass: the display mapping.
    let frames: Vec<ShotFrame> = stats
        .iter()
        .enumerate()
        .map(|(i, s)| ShotFrame { seconds: i as f64 / fps, lo: s.lo, hi: s.hi, cut_before: s.cut_before, relative: s.relative })
        .collect();
    let mapping = shot_mapping(&frames);
    let (uv_scale, uv_offset) = letterbox(dw, dh, hw, hh);
    let meta = Meta {
        version: META_VERSION,
        model_name,
        completed: true,
        frame_count: stats.len(),
        first_frame: job.first_frame,
        rate: (rate.num, rate.den),
        duration: job.span.length.unwrap_or(0.0),
        decoded_width: dw,
        decoded_height: dh,
        depth_width: hw,
        depth_height: hh,
        uv_scale,
        uv_offset,
        display_scale: stats.iter().zip(&mapping).map(|(s, m)| m.a * (s.hi - s.lo)).collect(),
        display_bias: stats.iter().zip(&mapping).map(|(s, m)| m.apply(s.lo)).collect(),
    };
    std::fs::write(job.directory.join("meta.json"), serde_json::to_vec(&meta).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    (job.progress)(1.0);
    Ok((Entry { directory: job.directory.to_path_buf(), meta }, Some(provider_name)))
}

/// One decoded frame (RGB at decode size, 8 or 16 bits) as the guide sees it
/// (RGBA half floats) and as the model sees it, unless it is static against
/// the last inferred frame (None: its depth repeats). The conversions run on
/// a few threads: at 1036×582 they are as long as the inference.
fn prepare(frame: &[u8], hdr: Option<bool>, (dw, dh): (usize, usize), model_size: (usize, usize), gate: &mut FrameChangeDetector)
    -> (Vec<f16>, Option<Vec<f32>>) {
    let sixteen = hdr.is_some();
    let pixels = dw * dh;
    let mut rgb = vec![0f32; pixels * 3];
    let mut video = vec![f16::ZERO; pixels * 4];
    let bands = 4;
    let rows = dh.div_ceil(bands);
    std::thread::scope(|scope| {
        for (band, (rgb, video)) in rgb.chunks_mut(rows * dw * 3).zip(video.chunks_mut(rows * dw * 4)).enumerate() {
            let first = band * rows * dw * 3;
            scope.spawn(move || {
                for (i, (c, v)) in rgb.as_chunks_mut::<3>().0.iter_mut().zip(video.as_chunks_mut::<4>().0.iter_mut()).enumerate() {
                    for k in 0..3 {
                        let at = first + i * 3 + k;
                        c[k] = if sixteen {
                            u16::from_le_bytes([frame[2 * at], frame[2 * at + 1]]) as f32 / 65535.0
                        } else {
                            frame[at] as f32 / 255.0
                        };
                        v[k] = f16::from_f32(c[k]);
                    }
                    v[3] = f16::ONE;
                }
            });
        }
    });
    let signature = FrameChangeDetector::signature(dw, dh, |x, y| {
        let i = (y * dw + x) * 3;
        [rgb[i] * 255.0, rgb[i + 1] * 255.0, rgb[i + 2] * 255.0]
    });
    let input = (!gate.should_reuse(signature)).then(|| model::input(&rgb, dw, dh, hdr, model_size));
    (video, input)
}

/// The anchor fit's sums of this frame's refined map against the previous
/// inferred frame's, over pixels whose luma held still and that both put in
/// the background (`depthAnchorReduce`); and the mean luma change.
fn anchor_sums(current: &Refined, previous: &Refined, background: AnchorCeiling) -> (AlignmentSums, f32) {
    let mut sums = AlignmentSums::default();
    let mut luma = 0f64;
    let softness = background.softness.max(1e-6);
    for i in 0..current.map.len() {
        let x = current.map[i];
        if !x.is_finite() {
            continue;
        }
        let y = previous.map[i];
        let dl = (current.guide[i] - previous.guide[i]).abs();
        let w = if y.is_finite() {
            (1.0 - dl / STATIC_LUMA_THRESHOLD).clamp(0.0, 1.0) * ((background.ceiling - x.max(y)) / softness).clamp(0.0, 1.0)
        } else {
            0.0
        };
        if w > 0.0 {
            sums.add(x, y, w);
        }
        luma += dl as f64;
    }
    (sums, (luma / current.map.len().max(1) as f64) as f32)
}

/// Decodes the stored depth maps back (luma only), frame by frame.
pub fn reader(tools: &Tools, entry: &Entry) -> Result<FrameReader, String> {
    let args: Vec<String> = ["-nostdin", "-v", "error", "-i"]
        .map(String::from)
        .into_iter()
        .chain([entry.video().to_string_lossy().into_owned()])
        .chain(["-map", "0:v:0", "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"].map(String::from))
        .collect();
    let (w, h) = (entry.meta.depth_width, entry.meta.depth_height);
    FrameReader::start(tools, &args, w * h + (w / 2) * (h / 2) * 2)
}

/// One stored map as a `depth` record's samples: the picture's part, halved
/// to the model's resolution, with the frame's display mapping.
pub struct DepthMap {
    pub payload: Vec<u8>,
    pub width: usize,
    pub height: usize,
    /// value = byte · scale + offset.
    pub scale: f32,
    pub offset: f32,
}

/// The entry's maps as `depth` records' samples, frame by frame
/// (`OfflineDepthConversion.records`).
pub fn maps(tools: &Tools, entry: &Entry) -> Result<Vec<DepthMap>, String> {
    let meta = &entry.meta;
    let (w, h) = (meta.depth_width, meta.depth_height);
    let x0 = (meta.uv_offset[0] * w as f32).round() as usize;
    let y0 = (meta.uv_offset[1] * h as f32).round() as usize;
    let cw = (w - x0).min((meta.uv_scale[0] * w as f32).round() as usize);
    let ch = (h - y0).min((meta.uv_scale[1] * h as f32).round() as usize);
    let (ow, oh) = (cw / 2, ch / 2);
    let mut reader = reader(tools, entry)?;
    let mut frame = Vec::new();
    let mut out = Vec::with_capacity(meta.frame_count);
    while out.len() < meta.frame_count && reader.next(&mut frame) {
        let index = out.len();
        let mut payload = vec![0u8; ow * oh];
        for y in 0..oh {
            let r0 = (y0 + 2 * y) * w + x0;
            let r1 = r0 + w;
            for x in 0..ow {
                let sum = frame[r0 + 2 * x] as u32 + frame[r0 + 2 * x + 1] as u32 + frame[r1 + 2 * x] as u32 + frame[r1 + 2 * x + 1] as u32;
                payload[y * ow + x] = ((sum + 2) / 4) as u8;
            }
        }
        out.push(DepthMap { payload, width: ow, height: oh, scale: meta.display_scale[index] / 255.0, offset: meta.display_bias[index] });
    }
    reader.finish()?;
    Ok(out)
}

/// Writes the depth as analysis-track `depth` records (`OfflineDepthConversion.writeAnalysisTrack`).
pub fn write_track(tools: &Tools, entry: &Entry, path: &Path, media_id: &str, source: (usize, usize)) -> Result<usize, String> {
    let meta = &entry.meta;
    let models = std::collections::BTreeMap::from([(
        "depth".to_string(),
        format!("{} (Vergence offline depth v5, shot space before the depth budget)", meta.model_name),
    )]);
    let mut out = crate::track::header(media_id, "vergence convert", &models, source.0 as u32, source.1 as u32);
    let rate = Rate { num: meta.rate.0, den: meta.rate.1 };
    let maps = maps(tools, entry)?;
    for (index, map) in maps.iter().enumerate() {
        let value = (meta.first_frame + index as i64) * rate.den as i64;
        let record = crate::track::Record::depth(map.payload.clone(), map.width, map.height, value, rate.num as i32, map.scale, map.offset);
        out.extend(record.deflated_framed());
    }
    let partial = path.with_extension("partial.vgat");
    std::fs::write(&partial, &out).map_err(|e| format!("{}: {e}", partial.display()))?;
    std::fs::rename(&partial, path).map_err(|e| e.to_string())?;
    Ok(maps.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_at_the_display_size() {
        // ffmpeg's even-rounded 1036×582 intermediate of a 16:9 picture has
        // pixels of 776:777: AVFoundation, and so the Mac, decodes it at 1034×582.
        assert_eq!(display_decode_size(display_width(1036, Some("776:777")), 582.0), (1034, 582));
        assert_eq!(decode_size(1920, 1080), (1036, 582));
        assert_eq!(display_width(720, Some("0:1")), 720.0);
        assert_eq!(display_width(720, None), 720.0);
    }
}
