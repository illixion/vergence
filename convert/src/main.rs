//! `vergence-convert`: any 2D video to an AI 3D side-by-side copy, on Windows
//! and Linux (and on a Mac, for tests): the Mac's `vergence convert`
//! (`Sources/VergenceCLI/Convert.swift`) with ONNX Runtime, wgpu and an ffmpeg
//! executable in place of Core ML, Metal and AVFoundation.
//!
//! # The contract (the same as `vergence convert`'s, so the desktop app's convert queue runs either)
//!
//! ```text
//! vergence-convert [convert] VIDEO|FOLDER... [--output FILE|DIR] [--layout fsbs|hsbs|tab|htab]
//!                  [--strength S] [--convergence C] [--sample SECONDS] [--start S] [--frames N]
//!                  [--model PATH.onnx] [--download-model] [--variant small|base|large] [--depth-track FILE.vgat]
//!                  [--quality low|standard|high|max] [--bitrate MBPS] [--max-eye PIXELS] [--no-audio]
//!                  [--dry-run] [--force]
//! ```
//!
//! - The same flags, values, defaults and errors (usage errors exit 64, failed
//!   files exit 1). A leading `convert` is accepted and ignored, so the app's
//!   `<cli> convert FILE ...` works unchanged. `--layout spatial` fails: MV-HEVC
//!   is made on a Mac only (no MV-HEVC encoder exists on Windows yet).
//! - The same outputs: `VIDEO.3D.FSBS.mp4`, `.3D.HSBS.mp4`, `.3D.FTAB.mp4`,
//!   `.3D.HTAB.mp4` (`VIDEO.sample.3D.…` for `--sample`), beside the input or
//!   in/at `--output` (a trailing `/` or `\` makes it a folder); an existing
//!   output is skipped without `--force`; inputs whose names say they are 3D
//!   already, and files that declare a stereo layout, are skipped. Work goes in
//!   a hidden `.NAME.work` folder beside the output; its finished depth is kept
//!   when a run stops, and the next run starts from it.
//! - The same lines on stderr, which the app reads (`desktop/src/convert.rs`):
//!   `vergence: NAME -> OUTPUT`, the settings line, `  depth  42%`,
//!   `  depth: N frames in S s (F fps)` or `  depth: N frames from an earlier run`,
//!   `  depth track: N records -> PATH`, `  picture  42%`, `  picture: N frames in …`,
//!   `  done: N frames in S s, F fps overall`; `vergence: NAME: …already made (--force to redo)`
//!   and `…, skipped` for skips; `vergence: downloading …` while fetching the model;
//!   `vergence: NAME: REASON` when a file fails.
//! - Progress: redrawn in place with `\r` on a terminal; with `VERGENCE_PROGRESS=1`
//!   (a caller reading a pipe: Windows has no pseudo-terminal to hand a child)
//!   as whole lines, `  depth  42%`, each time the whole percentage changes;
//!   otherwise not at all. `vergence convert` honours `VERGENCE_PROGRESS=1` the same way.
//! - ffmpeg and ffprobe: `VERGENCE_FFMPEG` (ffmpeg's path, ffprobe beside it),
//!   else PATH. The model: `--model`, `VERGENCE_DEPTH_MODEL`, else
//!   `DepthAnythingV2Small.onnx` in the models folder (`%LOCALAPPDATA%\Vergence\DepthModels`,
//!   `~/Library/Application Support/Vergence/DepthModels`, `~/.local/share/vergence/DepthModels`),
//!   which `--download-model` fills from huggingface.co/onnx-community/depth-anything-v2-small
//!   (Apache-2.0, about 100 MB). `--variant base|large` (an addition of this
//!   tool's, only in a build with the `noncommercial-models` feature) names
//!   Depth Anything V2 Base or Large instead, CC-BY-NC-4.0: refused without
//!   the feature, its licence said with it (`vergence_convert::licence`).
//!   `VERGENCE_DEPTH_EP` picks the execution provider
//!   (auto: CUDA, DirectML, Core ML, CPU, the first that loads), `VERGENCE_HEVC_ENCODER`
//!   the encoder (auto: NVENC, VideoToolbox, AMF, Quick Sync, x265).
//!
//! # What it does
//!
//! Two passes, as the Mac's: depth (`depth.rs`) then the picture: each frame
//! decoded again as YCbCr, both eyes drawn by the depth-displaced mesh with
//! the soft-knee budget (`gpu.rs`, `shaders/warp.wgsl`), packed and encoded as
//! HEVC (10-bit with the source's colour tags for 10-bit and HDR sources).
//! Audio MP4 holds (AAC, AC-3, E-AC-3, ALAC, MP3) is copied; lossless tracks
//! (TrueHD, DTS-HD MA, FLAC, PCM) become ALAC and other lossy ones AAC, as the
//! Mac's remux makes them. Subtitles are left out. Then the `st3d` box.

mod mp4;
mod options;

use vergence_convert::{decode, depth, gpu, maths, model};

use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::mpsc;
use std::time::Instant;

use vergence_convert::decode::{FrameReader, FrameWriter, Span, encoder_args, input_args, scale_matrix};
use crate::options::{Layout, Options};
use vergence_convert::probe::{Probe, Tools};

const USAGE: &str = "usage: vergence-convert [convert] VIDEO|FOLDER... [--output FILE|DIR] [--layout fsbs|hsbs|tab|htab]
         [--strength S] [--convergence C] [--sample SECONDS] [--start S] [--frames N]
         [--model PATH.onnx] [--download-model] [--variant small|base|large] [--depth-track FILE.vgat]
         [--quality low|standard|high|max] [--bitrate MBPS] [--max-eye PIXELS] [--no-audio] [--dry-run] [--force]";

/// `Pseudo3DSettings`' defaults: Convolution's Subtle.
const DEFAULT_STRENGTH: f64 = 0.008;
const DEFAULT_CONVERGENCE: f64 = 1.0;

fn log(line: impl AsRef<str>) {
    let mut err = std::io::stderr().lock();
    let _ = writeln!(err, "{}", line.as_ref());
}

fn clock(seconds: f64) -> String {
    let s = seconds.round() as i64;
    format!("{}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60)
}

/// `  LABEL  42%`: redrawn in place on a terminal; with VERGENCE_PROGRESS=1, a
/// line each time the whole percentage changes.
struct Progress {
    label: &'static str,
    interactive: bool,
    last: i64,
}

impl Progress {
    fn new(label: &'static str) -> Progress {
        Progress { label, interactive: std::io::stderr().is_terminal(), last: -1 }
    }

    /// Off a terminal, only when the caller asks (VERGENCE_PROGRESS=1).
    fn wanted(&self) -> bool {
        self.interactive || std::env::var("VERGENCE_PROGRESS").is_ok_and(|v| v == "1")
    }

    fn report(&mut self, fraction: f64) {
        if !self.wanted() {
            return;
        }
        let percent = (fraction.clamp(0.0, 1.0) * 100.0).round() as i64;
        if percent == self.last {
            return;
        }
        self.last = percent;
        let mut err = std::io::stderr().lock();
        if self.interactive {
            let _ = write!(err, "\r  {} {:3}%", self.label, percent);
        } else {
            let _ = writeln!(err, "  {} {:3}%", self.label, percent);
        }
        let _ = err.flush();
    }

    fn end(&self) {
        if self.interactive {
            let _ = write!(std::io::stderr().lock(), "\r");
        }
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let o = match options::parse(&args) {
        Ok(o) if !o.positional.is_empty() => o,
        Ok(_) => {
            log("vergence: convert needs a video file or a folder of them");
            log(USAGE);
            return ExitCode::from(64);
        }
        Err(e) => {
            log(format!("vergence: {}", e.0));
            log(USAGE);
            return ExitCode::from(64);
        }
    };
    lower_priority();
    match convert(&o) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            log(format!("vergence: {e}"));
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

/// True when every file converted or was skipped.
fn convert(o: &Options) -> Result<bool, String> {
    if o.layout == Layout::Spatial {
        return Err("spatial video (MV-HEVC) is made on a Mac only (vergence convert --layout spatial): \
                    there is no MV-HEVC encoder here; use fsbs, hsbs, tab or htab"
            .into());
    }
    let inputs = options::video_inputs(&o.positional, &mut |note| log(note));
    if inputs.is_empty() {
        return Err("convert needs a video file or a folder of them".into());
    }
    if o.depth_track.is_some() && inputs.len() > 1 {
        return Err("--depth-track takes one video".into());
    }
    if inputs.len() > 1 && o.output != "-" {
        std::fs::create_dir_all(&o.output).map_err(|e| format!("{}: {e}", o.output))?;
    }
    let tools = Tools::locate()?;
    let model = if o.dry_run { None } else { Some(depth_model(o)?) };
    let gpu = if o.dry_run { None } else { Some(gpu::Gpu::new()?) };
    let encoder = if o.dry_run { None } else { Some(decode::hevc_encoder(&tools)?) };
    let (mut made, mut skipped, mut failed) = (0, 0, 0);
    // Loaded the first time a file needs depth, then kept for the rest.
    let loaded = std::cell::OnceCell::new();
    for input in &inputs {
        let output = options::converted(input, o, inputs.len() == 1);
        let context =
            Context { tools: &tools, model: model.as_deref(), loaded: &loaded, gpu: gpu.as_ref(), encoder: encoder.as_deref() };
        match convert_one(input, &output, o, &context) {
            Ok(true) => made += 1,
            Ok(false) => skipped += 1,
            Err(e) => {
                log(format!("vergence: {}: {e}", name(input)));
                failed += 1;
            }
        }
    }
    if inputs.len() > 1 {
        log(format!("vergence: {made} converted, {skipped} skipped, {failed} failed"));
    }
    Ok(failed == 0)
}

fn name(path: &Path) -> String {
    path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

/// --model, VERGENCE_DEPTH_MODEL, else the installed model of `--variant`
/// (Depth Anything V2 Small; downloaded first with --download-model). A model
/// whose weights restrict their use is refused unless this build takes them,
/// and its licence is said when it does (`licence`).
fn depth_model(o: &Options) -> Result<PathBuf, String> {
    let path = resolved_depth_model(o)?;
    if let Some(line) = vergence_convert::licence::check(&path)? {
        log(format!("vergence: {line}"));
    }
    Ok(path)
}

fn resolved_depth_model(o: &Options) -> Result<PathBuf, String> {
    if let Some(path) = o.model.clone().or_else(|| std::env::var("VERGENCE_DEPTH_MODEL").ok()) {
        let path = PathBuf::from(path);
        if !path.is_file() {
            return Err(format!("no depth model at {}", path.display()));
        }
        if path.extension().is_none_or(|e| !e.eq_ignore_ascii_case("onnx")) {
            return Err(format!("{} isn't an ONNX model: this converter runs Depth Anything V2 Small as .onnx \
                                (--download-model fetches it)", path.display()));
        }
        return Ok(path);
    }
    let wanted = model::download_named(o.variant.as_deref().unwrap_or("small"))?;
    if let Some(path) = model::installed(&wanted) {
        return Ok(path);
    }
    if !o.download_model {
        return Err(format!("no depth model: pass --model PATH (Depth Anything V2 Small as .onnx), set VERGENCE_DEPTH_MODEL, \
                            or add --download-model to fetch {} from {} ({}, about {} MB)",
                           wanted.display, wanted.repo, wanted.licence, wanted.megabytes));
    }
    log(format!("vergence: downloading {} from {} ({})", wanted.display, wanted.repo, wanted.licence));
    model::download(&wanted)
}

type SharedModel = std::sync::Arc<std::sync::Mutex<model::DepthModel>>;

struct Context<'a> {
    tools: &'a Tools,
    model: Option<&'a Path>,
    loaded: &'a std::cell::OnceCell<SharedModel>,
    gpu: Option<&'a gpu::Gpu>,
    encoder: Option<&'a str>,
}

/// An opaque id for the media in an analysis track: its name and size hashed
/// (FNV-1a, as `mediaID(of:)`), never the name itself.
fn media_id(path: &Path) -> String {
    let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let mut hash: u64 = 0xCBF2_9CE4_8422_2325;
    for byte in format!("{}\0{size}", name(path)).bytes() {
        hash = (hash ^ byte as u64).wrapping_mul(0x100_0000_01B3);
    }
    format!("{hash:x}")
}

/// One eye: the frame's size, capped by --max-eye, even.
fn eye_size(width: usize, height: usize, max_eye: Option<usize>) -> (usize, usize) {
    let (mut w, mut h) = (width as f64, height as f64);
    if let Some(cap) = max_eye
        && w.max(h) > cap as f64
    {
        let scale = cap as f64 / w.max(h);
        w *= scale;
        h *= scale;
    }
    ((w.round() as usize) & !1, (h.round() as usize) & !1)
}

/// Converts one file; false when there was nothing to do.
fn convert_one(input: &Path, output: &Path, o: &Options, cx: &Context) -> Result<bool, String> {
    let file = name(input);
    if output.exists() && !o.force {
        log(format!("vergence: {file}: {} is already made (--force to redo)", name(output)));
        return Ok(false);
    }
    let probe = cx.tools.probe(input)?;
    let video = probe.video().ok_or("no video stream")?;
    if let Some(kind) = probe.already_3d() {
        log(format!("vergence: {file}: already 3D ({kind}), skipped"));
        return Ok(false);
    }
    let (width, height) = (video.width.unwrap_or(0), video.height.unwrap_or(0));
    if width < 64 || height < 64 {
        return Err(format!("the picture is too small ({width}×{height})"));
    }
    let rate = probe.rate();
    let fps = rate.fps();
    let duration = probe.duration();
    let start = o.start.or_else(|| o.sample.map(|s| ((duration - s) / 2.0).max(0.0))).unwrap_or(0.0);
    if start >= duration {
        return Err(format!("--start {start} is past the end ({})", clock(duration)));
    }
    let mut length = o.sample.unwrap_or(f64::INFINITY).min(duration - start);
    // The depth window looks two frames ahead; give it those.
    if let Some(frames) = o.frames {
        length = length.min((frames as f64 + 2.5) / fps);
    }
    let whole = start == 0.0 && length >= duration;

    let layout = o.layout;
    let eye = eye_size(width, height, o.max_eye);
    let packed = if layout.is_side_by_side() { (eye.0 * 2, eye.1) } else { (eye.0, eye.1 * 2) };
    let picture = if layout.is_half() { eye } else { packed };
    let hdr = probe.is_hdr();
    let ten_bit = probe.bit_depth() > 8 || hdr;
    let strength = o.strength.unwrap_or(DEFAULT_STRENGTH) as f32;
    let convergence = o.convergence.unwrap_or(DEFAULT_CONVERGENCE) as f32;
    let bits = o.bitrate.unwrap_or_else(|| {
        maths::bits_per_second(picture.0, picture.1, fps, o.quality,
                               probe.video_bits_per_second().map(|b| b * if layout.is_half() { 1.2 } else { 2.0 }))
    });
    let model_name = cx.model.and_then(|m| m.file_stem()).map(|s| s.to_string_lossy().into_owned());

    log(format!("vergence: {file} -> {}", output.display()));
    log(format!(
        "  {}, {}×{} (eyes {}×{}{}), HEVC {:.1} Mbps{}, strength {strength:.4}, convergence {convergence:.2}, model {}",
        layout.description(), picture.0, picture.1, eye.0, eye.1, if layout.is_half() { ", squeezed to half" } else { "" },
        bits as f64 / 1e6, if ten_bit { if hdr { ", 10-bit HDR" } else { ", 10-bit" } } else { "" },
        model_name.as_deref().unwrap_or("not looked up")
    ));
    if !whole {
        log(format!("  from {} for {length:.1} s{}", clock(start), o.frames.map(|f| format!(", {f} frames")).unwrap_or_default()));
    }
    let (Some(model), Some(gpu), Some(encoder)) = (cx.model, cx.gpu, cx.encoder) else { return Ok(false) };
    if o.dry_run {
        return Ok(false);
    }

    let parent = output.parent().map(Path::to_path_buf).unwrap_or_default();
    if !parent.as_os_str().is_empty() {
        std::fs::create_dir_all(&parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    let work = parent.join(format!(".{}.work", name(output)));
    std::fs::create_dir_all(&work).map_err(|e| format!("{}: {e}", work.display()))?;
    // Kept only while it holds finished depth worth starting again from.
    struct Cleanup(PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            if depth::Entry::at(&self.0.join("depth")).is_none() {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
    }
    let _cleanup = Cleanup(work.clone());

    let began = Instant::now();
    let first_frame = if whole { 0 } else { (start * fps - 1e-6).ceil() as i64 };
    let expected_duration = if whole { duration } else { (start + length).min(duration) - start };
    let expected_frames = ((expected_duration * fps).round() as usize).max(1);
    let depth_dir = work.join("depth");
    let model_stem = model_name.unwrap_or_default();
    let entry = match depth::Entry::at(&depth_dir) {
        Some(earlier) if earlier.meta.model_name == model_stem
            && (earlier.meta.duration - expected_duration).abs() < 0.05
            && (whole || earlier.meta.first_frame == first_frame) => {
            log(format!("  depth: {} frames from an earlier run", earlier.meta.frame_count));
            earlier
        }
        _ => {
            let depth_began = Instant::now();
            let mut progress = Progress::new("depth");
            let span = Span { start, length: (!whole).then_some(length), frames: None };
            if cx.loaded.get().is_none() {
                let loaded = model::DepthModel::load(model, model::Provider::from_env()?, None)?;
                let _ = cx.loaded.set(std::sync::Arc::new(std::sync::Mutex::new(loaded)));
            }
            let model = cx.loaded.get().expect("just loaded").clone();
            let (entry, provider) = depth::convert(depth::Job {
                tools: cx.tools, input, probe: &probe, span, first_frame, expected_frames, encoder, directory: &depth_dir, model,
                progress: &mut |f| progress.report(f),
            }, gpu)?;
            progress.end();
            log(format!("  engines: depth model on {}, warp on {}, encoder {encoder}", provider.unwrap_or_default(), gpu.adapter_name));
            let seconds = depth_began.elapsed().as_secs_f64();
            log(format!("  depth: {} frames in {seconds:.1} s ({:.1} fps)", entry.meta.frame_count,
                        entry.meta.frame_count as f64 / seconds.max(0.001)));
            entry
        }
    };
    if let Some(path) = &o.depth_track {
        let path = PathBuf::from(path);
        let records = depth::write_track(cx.tools, &entry, &path, &media_id(input), (width, height))?;
        log(format!("  depth track: {records} records -> {}", path.display()));
    }

    // Pass 2: both eyes, encoded.
    let picture_began = Instant::now();
    let mut progress = Progress::new("picture");
    let result = encode_picture(PictureJob {
        tools: cx.tools, input, probe: &probe, output, entry: &entry, gpu, encoder,
        span: Span { start, length: (!whole).then_some(length), frames: o.frames },
        eye, picture, side_by_side: layout.is_side_by_side(), ten_bit, strength, convergence, bits,
        include_audio: o.include_audio, expected_frames: o.frames.unwrap_or(expected_frames),
        progress: &mut |f| progress.report(f),
    })?;
    progress.end();
    let seconds = picture_began.elapsed().as_secs_f64();
    log(format!(
        "  picture: {} frames in {seconds:.1} s ({:.1} fps){}; {} audio track(s) copied",
        result.frames, result.frames as f64 / seconds.max(0.001),
        if result.flat > 0 { format!(", {} without depth (flat)", result.flat) } else { String::new() }, result.audio
    ));
    for note in &result.audio_notes {
        log(format!("  audio: {note}"));
    }
    mp4::write_st3d(output, if layout.is_side_by_side() { mp4::StereoMode::LeftRight } else { mp4::StereoMode::TopBottom })?;
    let _ = std::fs::remove_dir_all(&work);
    let seconds = began.elapsed().as_secs_f64();
    log(format!("  done: {} frames in {seconds:.1} s, {:.1} fps overall", result.frames, result.frames as f64 / seconds.max(0.001)));
    Ok(true)
}

struct PictureJob<'a> {
    tools: &'a Tools,
    input: &'a Path,
    probe: &'a Probe,
    output: &'a Path,
    entry: &'a depth::Entry,
    gpu: &'a gpu::Gpu,
    encoder: &'a str,
    span: Span,
    eye: (usize, usize),
    picture: (usize, usize),
    side_by_side: bool,
    ten_bit: bool,
    strength: f32,
    convergence: f32,
    bits: u64,
    include_audio: bool,
    expected_frames: usize,
    progress: &'a mut dyn FnMut(f64),
}

struct PictureResult {
    frames: usize,
    flat: usize,
    audio: usize,
    audio_notes: Vec<String>,
}

/// Codecs an MP4 takes as they are.
const COPIED_AUDIO: [&str; 5] = ["aac", "ac3", "eac3", "alac", "mp3"];

/// The audio half of the encoder's command line: what each track becomes, as
/// the Mac's ffmpeg remux plan decides it.
fn audio_args(probe: &Probe) -> (Vec<String>, Vec<String>) {
    let mut args = Vec::new();
    let mut notes = Vec::new();
    for (n, audio) in probe.audio().into_iter().enumerate() {
        args.extend(["-map".to_string(), format!("1:{}", audio.index)]);
        let codec = audio.codec_name.as_deref().unwrap_or("?");
        let lossless = matches!(codec, "truehd" | "flac" | "mlp") || codec.starts_with("pcm_")
            || (codec == "dts" && audio.profile.as_deref().is_some_and(|p| p.contains("MA")));
        if COPIED_AUDIO.contains(&codec) {
            args.extend([format!("-c:a:{n}"), "copy".into()]);
        } else if lossless {
            args.extend([format!("-c:a:{n}"), "alac".into()]);
            notes.push(format!("{} {codec} to ALAC (lossless)", n + 1));
        } else {
            let channels = audio.channels.unwrap_or(2).clamp(1, 8);
            args.extend([format!("-c:a:{n}"), "aac".into(), format!("-b:a:{n}"), format!("{}k", channels * 96)]);
            notes.push(format!("{} {codec} to AAC", n + 1));
        }
    }
    (args, notes)
}

fn encode_picture(job: PictureJob) -> Result<PictureResult, String> {
    let colour = job.probe.colour();
    let (ew, eh) = job.eye;
    let rate = job.probe.rate();
    let pixel_format = if job.ten_bit { "yuv420p10le" } else { "yuv420p" };
    let mut args = input_args(job.input, job.probe, job.span);
    args.extend([
        "-vf".to_string(),
        format!("scale={ew}:{eh}:flags=bicubic:in_color_matrix={m}:out_color_matrix={m}:in_range={}:out_range=tv",
                if colour.full_range { "pc" } else { "tv" }, m = scale_matrix(&colour.matrix)),
        "-f".into(), "rawvideo".into(), "-pix_fmt".into(), pixel_format.into(), "-".into(),
    ]);
    let sample = if job.ten_bit { 2 } else { 1 };
    let frame_bytes = ew * eh * sample * 3 / 2;
    let source = FrameReader::start(job.tools, &args, frame_bytes)?;
    let depth = depth::reader(job.tools, job.entry)?;

    // The encoder: raw packed frames in, the source's audio beside them.
    let (pw, ph) = job.picture;
    let partial = job.output.with_extension("partial.mp4");
    let _ = std::fs::remove_file(&partial);
    let mut enc: Vec<String> = ["-nostdin", "-v", "error", "-y", "-f", "rawvideo"].map(String::from).to_vec();
    enc.extend(["-pix_fmt".into(), if job.ten_bit { "p010le" } else { "nv12" }.into(), "-s".into(), format!("{pw}x{ph}"),
                "-r".into(), format!("{}/{}", rate.num, rate.den)]);
    // The colour goes on the frames as they come in: ffmpeg takes the frames'
    // own over the output options.
    let tags = ["-color_primaries".to_string(), colour.primaries.clone(), "-color_trc".into(), colour.transfer.clone(),
                "-colorspace".into(), colour.matrix.clone(), "-color_range".into(), "tv".into()];
    enc.extend(tags.iter().cloned());
    enc.extend(["-i".into(), "-".into()]);
    let mut audio_notes = Vec::new();
    let mut audio_count = 0;
    if job.include_audio && !job.probe.audio().is_empty() {
        // Audio from the first frame on, as long as the picture.
        let first = if job.span.start > 0.0 { job.span.start } else { job.probe.video_start().max(0.0) };
        if first > 0.0 {
            enc.extend(["-ss".into(), format!("{first:.6}")]);
        }
        let frames = job.expected_frames as f64;
        enc.extend(["-t".into(), format!("{:.6}", frames / rate.fps()), "-i".into(), job.input.to_string_lossy().into_owned()]);
        let (a, notes) = audio_args(job.probe);
        audio_count = job.probe.audio().len();
        audio_notes = notes;
        enc.extend(["-map".into(), "0:v".into()]);
        enc.extend(a);
    }
    let gop = (rate.fps() * 2.0).round().max(1.0) as usize;
    enc.extend(encoder_args(job.encoder, job.bits, gop, job.ten_bit));
    if job.encoder == "libx265"
        && let Some((master, cll)) = job.probe.hdr10_metadata().filter(|_| job.probe.is_hdr())
    {
        let mut params = format!("log-level=error:hdr10=1:master-display={master}");
        if let Some(cll) = cll {
            params += &format!(":max-cll={cll}");
        }
        enc.extend(["-x265-params".into(), params]);
    }
    enc.extend(tags);
    enc.extend(["-tag:v".into(), "hvc1".into(), "-f".into(), "mp4".into(), partial.to_string_lossy().into_owned()]);
    let writer = FrameWriter::start(job.tools, &enc)?;

    // Decoding on one thread, writing on another; the GPU on this one.
    let (planes_send, planes_receive) = mpsc::sync_channel::<(Vec<u8>, Option<Vec<u8>>)>(2);
    let (depth_w, depth_h) = (job.entry.meta.depth_width, job.entry.meta.depth_height);
    let reading = std::thread::spawn(move || -> Result<(), String> {
        let (mut source, mut depth) = (source, depth);
        let mut depth_open = true;
        loop {
            let mut frame = Vec::new();
            if !source.next(&mut frame) {
                break;
            }
            let mut map = Vec::new();
            let map = (depth_open && depth.next(&mut map)).then(|| {
                map.truncate(depth_w * depth_h);
                map
            });
            depth_open = map.is_some();
            if planes_send.send((frame, map)).is_err() {
                break;
            }
        }
        drop(depth);
        source.finish()
    });
    let (out_send, out_receive) = mpsc::sync_channel::<Vec<u8>>(2);
    let writing = std::thread::spawn(move || -> Result<(), String> {
        let mut writer = writer;
        for frame in out_receive {
            writer.write(&frame)?;
        }
        writer.finish()
    });

    let mut warper = gpu::Warper::new(job.gpu, gpu::PictureFormat {
        eye: job.eye,
        picture: job.picture,
        side_by_side: job.side_by_side,
        bits: if job.ten_bit { 10 } else { 8 },
        kr_kb: gpu::matrix_coefficients(&colour.matrix),
        full_range: false,
    });
    let meta = &job.entry.meta;
    let (mut frames, mut flat) = (0usize, 0usize);
    let mut failure = None;
    let debug = std::env::var_os("VERGENCE_CONVERT_DEBUG").is_some();
    let (mut waited, mut rendering, mut sending) = (0f64, 0f64, 0f64);
    let mut mark = Instant::now();
    for (frame, map) in planes_receive.iter() {
        waited += mark.elapsed().as_secs_f64();
        let luma = ew * eh * sample;
        let chroma = luma / 4;
        let planes = [&frame[..luma], &frame[luma..luma + chroma], &frame[luma + chroma..luma + 2 * chroma]];
        let depth = map.as_ref().filter(|_| frames < meta.frame_count).map(|values| gpu::WarpDepth {
            values,
            size: (depth_w, depth_h),
            value_scale: meta.display_scale[frames],
            value_bias: meta.display_bias[frames],
            uv: (meta.uv_scale, meta.uv_offset),
        });
        if depth.is_none() {
            flat += 1;
        }
        let started = Instant::now();
        let out = match warper.render(planes, depth.as_ref(), job.strength, job.convergence) {
            Ok(out) => out,
            Err(e) => {
                failure = Some(e);
                break;
            }
        };
        rendering += started.elapsed().as_secs_f64();
        let started = Instant::now();
        if let Some(out) = out
            && out_send.send(out).is_err()
        {
            break;
        }
        sending += started.elapsed().as_secs_f64();
        mark = Instant::now();
        frames += 1;
        if frames % 24 == 0 {
            (job.progress)((frames as f64 / job.expected_frames.max(1) as f64).min(1.0));
        }
    }
    if failure.is_none() {
        match warper.finish() {
            Ok(Some(out)) => {
                let _ = out_send.send(out);
            }
            Ok(None) => {}
            Err(e) => failure = Some(e),
        }
    }
    if debug {
        log(format!("  picture timing: waiting for frames {waited:.2} s, GPU {rendering:.2} s, waiting for the encoder {sending:.2} s"));
    }
    drop(planes_receive);
    drop(out_send);
    let read = reading.join().map_err(|_| "the decoder's thread panicked".to_string())?;
    let written = writing.join().map_err(|_| "the encoder's thread panicked".to_string())?;
    if let Some(e) = failure {
        let _ = std::fs::remove_file(&partial);
        return Err(e);
    }
    if let Err(e) = written.and(read) {
        let _ = std::fs::remove_file(&partial);
        return Err(e);
    }
    if frames == 0 {
        let _ = std::fs::remove_file(&partial);
        return Err("no video frames in the range".into());
    }
    let _ = std::fs::remove_file(job.output);
    std::fs::rename(&partial, job.output).map_err(|e| e.to_string())?;
    (job.progress)(1.0);
    Ok(PictureResult { frames, flat, audio: audio_count, audio_notes })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eye_sizes_like_the_mac() {
        assert_eq!(eye_size(1920, 1080, None), (1920, 1080));
        assert_eq!(eye_size(3840, 2160, Some(1920)), (1920, 1080));
        assert_eq!(eye_size(1279, 721, None), (1278, 720));
    }

    #[test]
    fn hashes_media_like_the_mac() {
        // FNV-1a over "name\0size".
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.mp4");
        std::fs::write(&path, b"xyz").unwrap();
        let mut hash: u64 = 0xCBF2_9CE4_8422_2325;
        for byte in b"a.mp4\x003" {
            hash = (hash ^ *byte as u64).wrapping_mul(0x100_0000_01B3);
        }
        assert_eq!(media_id(&path), format!("{hash:x}"));
    }
}
