//! ffmpeg as a decoder (raw frames on its stdout) and as an encoder (raw
//! frames on its stdin), run as separate processes so nothing of ffmpeg is
//! linked into this MIT program.

use std::io::{Read, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Stdio};
use std::thread::JoinHandle;

use crate::probe::{Probe, Tools};

/// What part of the source a pass reads: from `start` seconds, at most
/// `frames` frames (None: to the end).
#[derive(Clone, Copy, Debug)]
pub struct Span {
    pub start: f64,
    pub length: Option<f64>,
    pub frames: Option<usize>,
}

/// ffmpeg's input half for a pass over the video: frame-exact seeking, every
/// frame passed through as decoded (rawvideo would otherwise repeat the first
/// after a seek), H.264's SEI dropped (its frame-packing messages trip ffmpeg
/// 8 and 9).
pub fn input_args(input: &Path, probe: &Probe, span: Span) -> Vec<String> {
    let mut args: Vec<String> = ["-nostdin", "-v", "error"].map(String::from).to_vec();
    if probe.video().and_then(|v| v.codec_name.as_deref()) == Some("h264") {
        args.extend(["-bsf:v", "filter_units=remove_types=6"].map(String::from));
    }
    if span.start > 0.0 {
        args.extend(["-ss".to_string(), format!("{:.6}", span.start)]);
    }
    if let Some(length) = span.length {
        args.extend(["-t".to_string(), format!("{length:.6}")]);
    }
    args.extend(["-i".to_string(), input.to_string_lossy().into_owned()]);
    let index = probe.video().map(|v| v.index).unwrap_or(0);
    args.extend(["-map".to_string(), format!("0:{index}"), "-fps_mode".into(), "passthrough".into()]);
    if let Some(frames) = span.frames {
        args.extend(["-frames:v".to_string(), frames.to_string()]);
    }
    args
}

/// ffmpeg's name for a matrix in the scale filter's `in_color_matrix`.
pub fn scale_matrix(matrix: &str) -> &'static str {
    match matrix {
        "bt2020nc" | "bt2020c" => "bt2020",
        "smpte170m" => "smpte170m",
        "bt470bg" => "bt470",
        "fcc" => "fcc",
        "smpte240m" => "smpte240m",
        _ => "bt709",
    }
}

/// A running decoder: fixed-size frames off its stdout.
pub struct FrameReader {
    child: Child,
    stdout: ChildStdout,
    errors: Option<JoinHandle<String>>,
    pub frame_bytes: usize,
}

impl FrameReader {
    pub fn start(tools: &Tools, args: &[String], frame_bytes: usize) -> Result<FrameReader, String> {
        let mut child = Tools::command(&tools.ffmpeg)
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("ffmpeg didn't start: {e}"))?;
        let mut stderr = child.stderr.take().expect("piped");
        let errors = std::thread::spawn(move || {
            let mut text = String::new();
            let _ = stderr.read_to_string(&mut text);
            text
        });
        let stdout = child.stdout.take().expect("piped");
        Ok(FrameReader { child, stdout, errors: Some(errors), frame_bytes })
    }

    /// The next frame into `frame` (resized to fit); false at the end.
    pub fn next(&mut self, frame: &mut Vec<u8>) -> bool {
        frame.resize(self.frame_bytes, 0);
        self.stdout.read_exact(frame).is_ok()
    }

    /// Stops it; what it said on stderr, if it failed.
    pub fn finish(mut self) -> Result<(), String> {
        let _ = self.child.kill();
        let status = self.child.wait().ok();
        let message = self.errors.take().and_then(|h| h.join().ok()).unwrap_or_default();
        if !message.trim().is_empty() && status.is_some_and(|s| !s.success()) {
            return Err(format!("ffmpeg: {}", message.trim().chars().take(300).collect::<String>()));
        }
        Ok(())
    }
}

impl Drop for FrameReader {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// A running encoder taking raw frames on its stdin.
pub struct FrameWriter {
    child: Child,
    stdin: Option<ChildStdin>,
    errors: Option<JoinHandle<String>>,
}

impl FrameWriter {
    pub fn start(tools: &Tools, args: &[String]) -> Result<FrameWriter, String> {
        let mut child = Tools::command(&tools.ffmpeg)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("ffmpeg didn't start: {e}"))?;
        let mut stderr = child.stderr.take().expect("piped");
        let errors = std::thread::spawn(move || {
            let mut text = String::new();
            let _ = stderr.read_to_string(&mut text);
            text
        });
        Ok(FrameWriter { stdin: child.stdin.take(), child, errors: Some(errors) })
    }

    pub fn write(&mut self, frame: &[u8]) -> Result<(), String> {
        let stdin = self.stdin.as_mut().ok_or("the encoder is closed")?;
        stdin.write_all(frame).map_err(|_| "the encoder stopped taking frames".to_string())
    }

    /// Closes its input and waits for it to finish the file.
    pub fn finish(mut self) -> Result<(), String> {
        drop(self.stdin.take());
        let status = self.child.wait().map_err(|e| e.to_string())?;
        let message = self.errors.take().and_then(|h| h.join().ok()).unwrap_or_default();
        if !status.success() {
            return Err(format!("the encoder failed ({status}): {}", message.trim().chars().take(400).collect::<String>()));
        }
        Ok(())
    }
}

impl Drop for FrameWriter {
    fn drop(&mut self) {
        if self.stdin.is_some() {
            drop(self.stdin.take());
            let _ = self.child.kill();
        }
        let _ = self.child.wait();
    }
}

/// An HEVC encoder ffmpeg can run here, best first: NVENC, VideoToolbox,
/// AMF, Quick Sync, then x265. VERGENCE_HEVC_ENCODER picks one outright.
pub fn hevc_encoder(tools: &Tools) -> Result<String, String> {
    if let Ok(name) = std::env::var("VERGENCE_HEVC_ENCODER")
        && !name.is_empty()
    {
        return Ok(name);
    }
    for name in ["hevc_nvenc", "hevc_videotoolbox", "hevc_amf", "hevc_qsv", "libx265"] {
        // A one-frame encode: being listed doesn't mean the hardware is there.
        let ok = Tools::command(&tools.ffmpeg)
            .args(["-nostdin", "-v", "error", "-f", "lavfi", "-i", "color=black:s=256x256:d=0.1", "-frames:v", "1",
                   "-pix_fmt", "yuv420p", "-c:v", name, "-f", "null", "-"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success());
        if ok {
            return Ok(name.to_string());
        }
    }
    Err("no HEVC encoder works in this ffmpeg (NVENC, VideoToolbox, AMF, Quick Sync or x265)".into())
}

/// Encoder options for `encoder` at `bits_per_second`, a keyframe every `gop` frames.
pub fn encoder_args(encoder: &str, bits_per_second: u64, gop: usize, ten_bit: bool) -> Vec<String> {
    let rate = bits_per_second.to_string();
    let mut args: Vec<String> = vec!["-c:v".into(), encoder.into(), "-g".into(), gop.to_string()];
    match encoder {
        "hevc_nvenc" => args.extend(
            ["-preset", "p5", "-tune", "hq", "-rc", "vbr", "-b:v", &rate, "-maxrate", &(bits_per_second * 3 / 2).to_string(),
             "-bufsize", &(bits_per_second * 2).to_string(), "-bf", "3", "-profile:v", if ten_bit { "main10" } else { "main" }]
                .map(String::from),
        ),
        "hevc_videotoolbox" => args.extend(["-b:v", &rate, "-profile:v", if ten_bit { "main10" } else { "main" }].map(String::from)),
        "libx265" => args.extend(["-b:v", &rate, "-preset", "medium"].map(String::from)),
        _ => args.extend(["-b:v", &rate].map(String::from)),
    }
    args
}
