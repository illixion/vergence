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
//!                  [--upscale 2x] [--upscale-model PATH.pth] [--input-layout auto|2d|sbs|hsbs|tab|htab]
//!                  [--keep-repeated-frames] [--layout ...|2d] [--dry-run] [--force]
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
//! `--upscale 2x` (`upscale.rs`) runs Real-ESRGAN's compact network through
//! ONNX Runtime first, from the authors' own `.pth` (`--download-model`
//! fetches it beside the depth models, `--upscale-model` or
//! VERGENCE_UPSCALE_MODEL names one): a 2D source is upscaled and then made
//! 3D from depth (depth still at the source's size), or with `--layout 2d`
//! only upscaled (`VIDEO.2x.mp4`); a side-by-side or top-and-bottom source
//! (declared by the file, or seen in its picture and name as the Mac sees
//! them, `layout.rs`; `--input-layout` says it otherwise) has each eye
//! upscaled with their fine detail brought into agreement, no depth, packed
//! as `--layout` says (`VIDEO.2x.3D.FSBS.mp4`). Unlike the Mac it runs an
//! eye in one piece (no tiles).
//!
//! A video that repeats one frame in five (`repeats.rs`) has the repeats
//! dropped by ffmpeg's decimate in every pass, at four fifths of the rate,
//! unless `--keep-repeated-frames`.
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

use vergence_convert::{decode, depth, gpu, layout, maths, model, repeats, upscale};

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
         [--quality low|standard|high|max] [--bitrate MBPS] [--max-eye PIXELS] [--no-audio] [--dry-run] [--force]
         [--upscale 2x] [--upscale-model PATH.pth] [--input-layout auto|2d|sbs|hsbs|tab|htab] [--keep-repeated-frames]";

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
    // The depth model is looked up now unless upscaling, when a file may need
    // none (a 3D source, --layout 2d); then when a file first needs it.
    let needs_depth = o.upscale.is_none() || (o.layout != Layout::Flat && o.input_layout == "2d");
    let model = std::cell::OnceCell::new();
    if !o.dry_run && needs_depth {
        let _ = model.set(depth_model(o)?);
    }
    let network = if o.dry_run || o.upscale.is_none() { None } else { Some(upscale_network(o)?) };
    let gpu = if o.dry_run { None } else { Some(gpu::Gpu::new()?) };
    let encoder = if o.dry_run { None } else { Some(decode::hevc_encoder(&tools)?) };
    let (mut made, mut skipped, mut failed) = (0, 0, 0);
    // Loaded the first time a file needs depth, then kept for the rest.
    let loaded = std::cell::OnceCell::new();
    let upscaler = std::cell::OnceCell::new();
    for input in &inputs {
        let output = options::converted(input, o, inputs.len() == 1);
        let context = Context {
            tools: &tools, model: &model, loaded: &loaded, gpu: gpu.as_ref(), encoder: encoder.as_deref(),
            network: network.as_ref(), upscaler: &upscaler, options: o,
        };
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

/// --upscale-model, VERGENCE_UPSCALE_MODEL, else the installed general
/// network (fetched first with --download-model).
fn upscale_network(o: &Options) -> Result<upscale::Network, String> {
    if let Some(path) = o.upscale_model.clone().or_else(|| std::env::var("VERGENCE_UPSCALE_MODEL").ok()) {
        let path = PathBuf::from(path);
        if !path.is_file() {
            return Err(format!("no upscaling model at {}", path.display()));
        }
        return upscale::Network::load(&path);
    }
    let weights = &upscale::GENERAL;
    if let Some(path) = upscale::installed(weights) {
        return upscale::Network::load(&path);
    }
    if !o.download_model {
        return Err(format!("no upscaling model: pass --upscale-model PATH (Real-ESRGAN's {}.pth), set VERGENCE_UPSCALE_MODEL, \
                            or add --download-model to fetch it from github.com/xinntao/Real-ESRGAN ({}, about 5 MB)",
                           weights.name, weights.licence));
    }
    log(format!("vergence: downloading {} from github.com/xinntao/Real-ESRGAN ({})", weights.name, weights.licence));
    upscale::Network::load(&upscale::download(weights)?)
}

type SharedModel = std::sync::Arc<std::sync::Mutex<model::DepthModel>>;

struct Context<'a> {
    tools: &'a Tools,
    model: &'a std::cell::OnceCell<PathBuf>,
    loaded: &'a std::cell::OnceCell<SharedModel>,
    gpu: Option<&'a gpu::Gpu>,
    encoder: Option<&'a str>,
    network: Option<&'a upscale::Network>,
    upscaler: &'a std::cell::OnceCell<std::cell::RefCell<upscale::Upscaler>>,
    options: &'a Options,
}

impl Context<'_> {
    fn depth_model(&self) -> Result<&Path, String> {
        if self.model.get().is_none() {
            let _ = self.model.set(depth_model(self.options)?);
        }
        Ok(self.model.get().expect("set"))
    }
}

/// How a frame-packed source holds its eyes.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Packing {
    side_by_side: bool,
    half: bool,
}

impl Packing {
    /// Each eye's shape as it shows, for a `width`×`height` frame.
    fn eye(self, width: usize, height: usize) -> (f64, f64) {
        let (w, h) = (width as f64, height as f64);
        match (self.side_by_side, self.half) {
            (true, false) => (w / 2.0, h),
            (false, false) => (w, h / 2.0),
            _ => (w, h),
        }
    }

    /// Half or full, by which gives the frame a picture-shaped eye (4:3 to
    /// 2.8:1, or square), as `PackedStereoLayout.correctingResolution` decides.
    fn corrected(self, width: usize, height: usize) -> Packing {
        let plausible = |(w, h): (f64, f64)| {
            let a = w / h.max(1.0);
            (1.25..=2.8).contains(&a) || (0.97..=1.03).contains(&a)
        };
        let half = plausible(Packing { half: true, ..self }.eye(width, height));
        let full = plausible(Packing { half: false, ..self }.eye(width, height));
        if half != full { Packing { half, ..self } } else { self }
    }

    fn describe(self) -> String {
        format!("{}{}", if self.side_by_side { "side by side" } else { "top and bottom" }, if self.half { ", half" } else { "" })
    }
}

/// The layout a file name says, as `StereoLayoutDetector.layout(fromFileName:)`
/// reads it; the bool says whether full or half was named.
fn named_packing(name: &str) -> Option<(Packing, bool)> {
    let lower = name.to_lowercase();
    let words: Vec<&str> = lower.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).collect();
    let has = |any: &[&str]| any.iter().any(|a| words.contains(a));
    let phrase = |any: &[&str]| any.iter().any(|a| lower.contains(a));
    let sbs = |half| Packing { side_by_side: true, half };
    let ou = |half| Packing { side_by_side: false, half };
    if has(&["hsbs"]) || phrase(&["half-sbs", "half sbs", "half_sbs", "half.sbs"]) {
        return Some((sbs(true), true));
    }
    if has(&["hou", "htab", "htb"]) || phrase(&["half-ou", "half ou", "half_ou", "half.ou", "half-tab", "half_tab"]) {
        return Some((ou(true), true));
    }
    if has(&["fsbs"]) || phrase(&["full-sbs", "full sbs", "full_sbs", "full.sbs"]) {
        return Some((sbs(false), true));
    }
    if has(&["fou", "ftab"]) || phrase(&["full-ou", "full ou", "full_ou", "full.ou"]) {
        return Some((ou(false), true));
    }
    if has(&["sbs", "lr"]) || phrase(&["side-by-side", "side by side", "sidebyside", "side_by_side"]) {
        return Some((sbs(false), false));
    }
    if has(&["ou", "tab", "tb", "overunder"]) || phrase(&["over-under", "over under", "top-bottom", "top-and-bottom", "top_bottom"]) {
        return Some((ou(false), false));
    }
    None
}

/// What the picture pass makes.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Mode {
    /// Both eyes warped from depth (each frame upscaled first with --upscale).
    Depth,
    /// A frame-packed source, each eye upscaled; no depth.
    Stereo(Packing),
    /// --layout 2d: upscaled, still 2D.
    Flat,
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
    let mut probe = cx.tools.probe(input)?;
    let video = probe.video().ok_or("no video stream")?;
    let (width, height) = (video.width.unwrap_or(0), video.height.unwrap_or(0));
    let upscaling = o.upscale.is_some();
    // 2D or 3D: --input-layout, else what the file declares, and when
    // upscaling (where a 3D source is worth taking) what its name says.
    let declared = probe.already_3d();
    if declared == Some("MV-HEVC spatial video") {
        log(format!("vergence: {file}: already 3D (MV-HEVC spatial video){}, skipped",
                    if upscaling { "; upscaling spatial video isn't done yet" } else { "" }));
        return Ok(false);
    }
    let stereo: Option<(Packing, String)> = match o.input_layout.as_str() {
        "auto" => match declared {
            Some(kind) => Some((Packing { side_by_side: kind == "side by side", half: false }.corrected(width, height),
                                "declared by the file".to_string())),
            // The picture, as the Mac reads it, then the name; the picture wins when they disagree.
            None if upscaling => {
                let named = named_packing(&file);
                let picture = layout::detect(cx.tools, input, &probe);
                match (picture, named) {
                    (Some(sbs), Some((packing, explicit))) if packing.side_by_side == sbs => Some((
                        if explicit { packing } else { packing.corrected(width, height) },
                        "the file name says so, and the picture agrees".to_string(),
                    )),
                    (Some(sbs), _) => Some((Packing { side_by_side: sbs, half: false }.corrected(width, height),
                                            format!("the picture's halves match {}", if sbs { "side by side" } else { "over-under" }))),
                    (None, Some((packing, explicit))) => Some((
                        if explicit { packing } else { packing.corrected(width, height) },
                        "the file name says so".to_string(),
                    )),
                    (None, None) => None,
                }
            }
            None => None,
        },
        "2d" => None,
        given => Some((Packing { side_by_side: given.contains("sbs"), half: given.starts_with('h') }, format!("--input-layout {given}"))),
    };
    if let Some((packing, _)) = &stereo
        && !upscaling
    {
        let kind = if packing.side_by_side { "side by side" } else { "over-under" };
        log(format!("vergence: {file}: already 3D ({kind}), skipped (--upscale 2x upscales its eyes)"));
        return Ok(false);
    }
    if width < 64 || height < 64 {
        return Err(format!("the picture is too small ({width}×{height})"));
    }
    let mode = match &stereo {
        Some((packing, note)) => {
            if o.layout == Layout::Flat {
                return Err(format!("it is 3D ({note}); --layout 2d is for 2D sources (--input-layout 2d if it isn't 3D)"));
            }
            Mode::Stereo(*packing)
        }
        None if o.layout == Layout::Flat => Mode::Flat,
        None => Mode::Depth,
    };
    // Frames a pulldown repeats, judged on stretches across the file.
    let verdict = if o.dry_run { None } else { Some(repeats::detect(cx.tools, input, &probe)?) };
    let source_rate = probe.rate();
    probe.drop_repeats = verdict.is_some_and(|v| v.is_pulldown()) && !o.keep_repeated_frames;
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
    // Each eye as it shows (a squeezed eye stretched back out), then as it is
    // made: twice that with --upscale, capped by --max-eye.
    let (source_eye, regions) = match mode {
        Mode::Stereo(packing) => {
            let half = if packing.side_by_side { (width as f64 / 2.0, height as f64) } else { (width as f64, height as f64 / 2.0) };
            let first = upscale::Region { x: 0.0, y: 0.0, width: half.0, height: half.1 };
            let second = upscale::Region {
                x: if packing.side_by_side { half.0 } else { 0.0 }, y: if packing.side_by_side { 0.0 } else { half.1 }, ..first
            };
            (packing.eye(width, height), vec![first, second])
        }
        _ => ((width as f64, height as f64), vec![upscale::Region { x: 0.0, y: 0.0, width: width as f64, height: height as f64 }]),
    };
    let factor = if upscaling { 2.0 } else { 1.0 };
    let eye = eye_size((source_eye.0 * factor).round() as usize, (source_eye.1 * factor).round() as usize, o.max_eye);
    let model_eye = ((eye.0 / 2) & !1, (eye.1 / 2) & !1);
    let made = if upscaling { (model_eye.0 * 2, model_eye.1 * 2) } else { eye };
    let packed = if layout == Layout::Flat { made } else if layout.is_side_by_side() { (made.0 * 2, made.1) } else { (made.0, made.1 * 2) };
    let picture = if layout.is_half() { made } else { packed };
    let hdr = probe.is_hdr();
    let ten_bit = probe.bit_depth() > 8 || hdr;
    let strength = o.strength.unwrap_or(DEFAULT_STRENGTH) as f32;
    let convergence = o.convergence.unwrap_or(DEFAULT_CONVERGENCE) as f32;
    let share = match mode {
        Mode::Depth => (if layout.is_half() { 1.2 } else { 2.0 }) * if upscaling { 2.5 } else { 1.0 },
        Mode::Stereo(_) => 2.5 * if layout.is_half() { 0.6 } else { 1.0 },
        Mode::Flat => 2.5,
    };
    let bits = o.bitrate.unwrap_or_else(|| {
        maths::bits_per_second(picture.0, picture.1, fps, o.quality, probe.video_bits_per_second().map(|b| b * share))
    });
    let model = if mode == Mode::Depth && !o.dry_run { Some(cx.depth_model()?) } else { None };
    let model_name = model.and_then(|m| m.file_stem()).map(|s| s.to_string_lossy().into_owned());

    log(format!("vergence: {file} -> {}", output.display()));
    let settings = match mode {
        Mode::Depth => format!("strength {strength:.4}, convergence {convergence:.2}, model {}", model_name.as_deref().unwrap_or("not looked up")),
        Mode::Stereo(packing) => format!("3D source ({}; {}), no depth", packing.describe(), stereo.as_ref().map(|s| s.1.as_str()).unwrap_or("")),
        Mode::Flat => "2D".into(),
    };
    log(format!(
        "  {}, {}×{} ({} {}×{}{}), HEVC {:.1} Mbps{}, {settings}",
        layout.description(), picture.0, picture.1, if layout == Layout::Flat { "picture" } else { "eyes" }, made.0, made.1,
        if layout.is_half() { ", squeezed to half" } else { "" },
        bits as f64 / 1e6, if ten_bit { if hdr { ", 10-bit HDR" } else { ", 10-bit" } } else { "" },
    ));
    if let Some(network) = cx.network {
        log(format!("  upscaled 2× from {}×{} by {} (Real-ESRGAN, {} convolutions)", model_eye.0, model_eye.1, network.name,
                    network.convolutions()));
    }
    if let Some(v) = verdict.filter(|v| v.is_pulldown()) {
        log(format!("  frames: one in five repeats the one before ({} of {} moving stretches): {}", v.repeating, v.moving,
                    if probe.drop_repeats { format!("dropped, {:.3} fps becomes {:.3}", source_rate.fps(), fps) }
                    else { "kept (--keep-repeated-frames)".into() }));
    }
    if !whole {
        log(format!("  from {} for {length:.1} s{}", clock(start), o.frames.map(|f| format!(", {f} frames")).unwrap_or_default()));
    }
    let (Some(gpu), Some(encoder)) = (cx.gpu, cx.encoder) else { return Ok(false) };
    if o.dry_run {
        return Ok(false);
    }
    let upscaler = match cx.network {
        Some(network) => {
            if cx.upscaler.get().is_none() {
                let _ = cx.upscaler.set(std::cell::RefCell::new(upscale::Upscaler::load(network, model::Provider::from_env()?)?));
            }
            cx.upscaler.get()
        }
        None => None,
    };

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
    let entry = if let Some(model) = model {
    Some(match depth::Entry::at(&depth_dir) {
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
    })
    } else {
        None
    };
    if let (Some(path), Some(entry)) = (&o.depth_track, &entry) {
        let path = PathBuf::from(path);
        let records = depth::write_track(cx.tools, entry, &path, &media_id(input), (width, height))?;
        log(format!("  depth track: {records} records -> {}", path.display()));
    }

    // Pass 2: both eyes, encoded.
    let picture_began = Instant::now();
    let mut progress = Progress::new("picture");
    let result = encode_picture(PictureJob {
        tools: cx.tools, input, probe: &probe, output, entry: entry.as_ref(), gpu, encoder,
        span: Span { start, length: (!whole).then_some(length), frames: o.frames },
        eye: made, picture, side_by_side: layout.is_side_by_side(), ten_bit, strength, convergence, bits,
        include_audio: o.include_audio, expected_frames: o.frames.unwrap_or(expected_frames),
        progress: &mut |f| progress.report(f),
        upscale: upscaler.map(|upscaler| UpscaleStage {
            upscaler, regions: regions.clone(), input_eye: model_eye, source: (width, height), mode, flat: layout == Layout::Flat,
        }),
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
    if layout != Layout::Flat {
        mp4::write_st3d(output, if layout.is_side_by_side() { mp4::StereoMode::LeftRight } else { mp4::StereoMode::TopBottom })?;
    }
    let _ = std::fs::remove_dir_all(&work);
    let seconds = began.elapsed().as_secs_f64();
    log(format!("  done: {} frames in {seconds:.1} s, {:.1} fps overall", result.frames, result.frames as f64 / seconds.max(0.001)));
    Ok(true)
}

/// The upscaler in front of the picture pass.
struct UpscaleStage<'a> {
    upscaler: &'a std::cell::RefCell<upscale::Upscaler>,
    /// Where each eye is in the source frame.
    regions: Vec<upscale::Region>,
    /// Each eye going in; it comes out twice that (the job's `eye`).
    input_eye: (usize, usize),
    /// The source frame's size.
    source: (usize, usize),
    mode: Mode,
    /// No 3D out (`--layout 2d`).
    flat: bool,
}

struct PictureJob<'a> {
    tools: &'a Tools,
    input: &'a Path,
    probe: &'a Probe,
    output: &'a Path,
    entry: Option<&'a depth::Entry>,
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
    upscale: Option<UpscaleStage<'a>>,
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
    let in_range = if colour.full_range { "pc" } else { "tv" };
    let m = scale_matrix(&colour.matrix);
    let mut args = input_args(job.input, job.probe, job.span);
    // Upscaling reads the frame as it is, as planar float RGB; the warp alone
    // takes YCbCr at the eye's size.
    let frame_bytes = if let Some(stage) = &job.upscale {
        args.extend(["-vf".to_string(), job.probe.video_filter(&format!("scale=w=iw:h=ih:in_color_matrix={m}:in_range={in_range}")),
                     "-f".into(), "rawvideo".into(), "-pix_fmt".into(), "gbrpf32le".into(), "-".into()]);
        stage.source.0 * stage.source.1 * 3 * 4
    } else {
        args.extend([
            "-vf".to_string(),
            job.probe.video_filter(&format!("scale={ew}:{eh}:flags=bicubic:in_color_matrix={m}:out_color_matrix={m}:in_range={in_range}:out_range=tv")),
            "-f".into(), "rawvideo".into(), "-pix_fmt".into(), pixel_format.into(), "-".into(),
        ]);
        ew * eh * if job.ten_bit { 2 } else { 1 } * 3 / 2
    };
    let sample = if job.ten_bit { 2 } else { 1 };
    let source = FrameReader::start(job.tools, &args, frame_bytes)?;
    let depth = job.entry.map(|entry| depth::reader(job.tools, entry)).transpose()?;
    // An upscaled 3D or 2D picture goes to the encoder as RGB, packed at full
    // size; its scaler squeezes a half layout and makes it YCbCr.
    let rgb_out = job.upscale.as_ref().is_some_and(|s| s.mode != Mode::Depth);
    let packed = match &job.upscale {
        Some(stage) if stage.flat => job.eye,
        _ if job.side_by_side => (ew * 2, eh),
        _ => (ew, eh * 2),
    };

    // The encoder: raw packed frames in, the source's audio beside them.
    let (pw, ph) = job.picture;
    let partial = job.output.with_extension("partial.mp4");
    let _ = std::fs::remove_file(&partial);
    let mut enc: Vec<String> = ["-nostdin", "-v", "error", "-y", "-f", "rawvideo"].map(String::from).to_vec();
    if rgb_out {
        enc.extend(["-pix_fmt".into(), "gbrp16le".into(), "-s".into(), format!("{}x{}", packed.0, packed.1)]);
    } else {
        enc.extend(["-pix_fmt".into(), if job.ten_bit { "p010le" } else { "nv12" }.into(), "-s".into(), format!("{pw}x{ph}")]);
    }
    enc.extend(["-r".into(), format!("{}/{}", rate.num, rate.den)]);
    // The colour goes on the frames as they come in: ffmpeg takes the frames'
    // own over the output options.
    let tags = ["-color_primaries".to_string(), colour.primaries.clone(), "-color_trc".into(), colour.transfer.clone(),
                "-colorspace".into(), colour.matrix.clone(), "-color_range".into(), "tv".into()];
    if !rgb_out {
        enc.extend(tags.iter().cloned());
    }
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
    if rgb_out {
        // The RGB frames carry no colour of their own: say it on them, as the
        // YCbCr ones say it coming in.
        enc.extend(["-vf".into(), format!(
            "scale={pw}:{ph}:flags=bicubic:out_color_matrix={m}:out_range=tv,format={pixel_format},\
             setparams=color_primaries={}:color_trc={}:colorspace={}:range=tv",
            colour.primaries, colour.transfer, colour.matrix)]);
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
    let (depth_w, depth_h) = job.entry.map(|e| (e.meta.depth_width, e.meta.depth_height)).unwrap_or((0, 0));
    let reading = std::thread::spawn(move || -> Result<(), String> {
        let (mut source, mut depth) = (source, depth);
        let mut depth_open = depth.is_some();
        loop {
            let mut frame = Vec::new();
            if !source.next(&mut frame) {
                break;
            }
            let mut map = Vec::new();
            let map = (depth_open && depth.as_mut().is_some_and(|d| d.next(&mut map))).then(|| {
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
    let (mut frames, mut flat) = (0usize, 0usize);
    let mut failure = None;
    let debug = std::env::var_os("VERGENCE_CONVERT_DEBUG").is_some();
    let (mut waited, mut rendering, mut sending) = (0f64, 0f64, 0f64);
    let mut mark = Instant::now();
    let (mut eyes_in, mut batch, mut yuv) = (Vec::new(), Vec::new(), Vec::new());
    let mut network = 0f64;
    // An upscaled pair's fusion and packing run on a thread of their own,
    // beside the network working on the next frame.
    let finishing = job.upscale.as_ref().filter(|_| rgb_out).map(|stage| {
        let (send, receive) = mpsc::sync_channel::<(Vec<Vec<f32>>, Vec<f32>)>(2);
        let (out_send, (iw, ih), side_by_side) = (out_send.clone(), stage.input_eye, job.side_by_side);
        let thread = std::thread::spawn(move || -> f64 {
            let mut fusing = 0.0;
            let each = 3 * ew * eh;
            let places = [(0, 0), if side_by_side { (ew, 0) } else { (0, eh) }];
            for (eyes_in, mut out) in receive {
                let began = Instant::now();
                if eyes_in.len() == 2 {
                    let (a, b) = out.split_at_mut(each);
                    upscale::fuse_eyes([&eyes_in[0], &eyes_in[1]], [a, b], iw, ih);
                }
                let eyes: Vec<&[f32]> = out.chunks_exact(each).collect();
                let packed_frame = upscale::pack_gbrp16(&eyes, ew, eh, packed, &places[..eyes.len()]);
                fusing += began.elapsed().as_secs_f64();
                if out_send.send(packed_frame).is_err() {
                    break;
                }
            }
            fusing
        });
        (send, thread)
    });
    let (finish_send, finish_thread) = finishing.map(|(s, t)| (Some(s), Some(t))).unwrap_or((None, None));
    for (decoded, map) in planes_receive.iter() {
        waited += mark.elapsed().as_secs_f64();
        let mut frame = decoded;
        if let Some(stage) = &job.upscale {
            let started = Instant::now();
            let (fw, fh) = stage.source;
            let n = fw * fh;
            let values: Vec<f32> = frame.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
            // gbrp: G, B, R planes.
            let planes = [&values[2 * n..3 * n], &values[..n], &values[n..2 * n]];
            let (iw, ih) = stage.input_eye;
            batch.clear();
            eyes_in.clear();
            for region in &stage.regions {
                let mut eye = Vec::new();
                upscale::sample_eye(planes, fw, fh, *region, iw, ih, &mut eye);
                batch.extend_from_slice(&eye);
                eyes_in.push(eye);
            }
            let network_began = Instant::now();
            let out = match stage.upscaler.borrow_mut().run(&batch, stage.regions.len(), iw, ih) {
                Ok(out) => out,
                Err(e) => {
                    failure = Some(e);
                    break;
                }
            };
            network += network_began.elapsed().as_secs_f64();
            rendering += started.elapsed().as_secs_f64();
            if stage.mode == Mode::Depth {
                upscale::rgb_to_yuv420(&out[..3 * ew * eh], ew, eh, gpu::matrix_coefficients(&colour.matrix),
                                       if job.ten_bit { 10 } else { 8 }, &mut yuv);
                frame = std::mem::take(&mut yuv);
            } else {
                // Fused and packed as gbrp16le on the finishing thread.
                if finish_send.as_ref().is_none_or(|s| s.send((std::mem::take(&mut eyes_in), out)).is_err()) {
                    break;
                }
                mark = Instant::now();
                frames += 1;
                if frames % 24 == 0 {
                    (job.progress)((frames as f64 / job.expected_frames.max(1) as f64).min(1.0));
                }
                continue;
            }
        }
        let luma = ew * eh * sample;
        let chroma = luma / 4;
        let planes = [&frame[..luma], &frame[luma..luma + chroma], &frame[luma + chroma..luma + 2 * chroma]];
        let meta = job.entry.map(|e| &e.meta);
        let depth = map.as_ref().zip(meta).filter(|(_, meta)| frames < meta.frame_count).map(|(values, meta)| gpu::WarpDepth {
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
    drop(finish_send);
    let fusing = finish_thread.map(|t| t.join().unwrap_or(0.0)).unwrap_or(0.0);
    if debug {
        log(format!("  picture timing: waiting for frames {waited:.2} s, GPU {rendering:.2} s, waiting for the encoder {sending:.2} s"));
        if let Some(stage) = &job.upscale {
            log(format!("  upscaling: network {network:.2} s on {}, fusing and packing {fusing:.2} s (beside it)",
                        stage.upscaler.borrow().provider.name()));
        }
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
