//! Frames a pulldown repeats, found as VergenceCore's `RepeatedFrames` finds
//! them: in most five-frame stretches where something moves, one frame
//! barely differs from the one before. Such a video (a film remuxed to an odd
//! rate such as 31.24 fps) has the repeats dropped with ffmpeg's
//! `decimate=cycle=5`, one in every five, so the rest keep their place in
//! time at four fifths of the rate.

use std::path::Path;

use crate::probe::{Probe, Tools};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Verdict {
    /// Five-frame stretches with movement in them.
    pub moving: usize,
    /// Those with one frame that repeats the one before.
    pub repeating: usize,
}

impl Verdict {
    pub fn is_pulldown(self) -> bool {
        self.moving >= 8 && self.repeating as f64 >= 0.8 * self.moving as f64
    }
}

/// Movement below this (mean absolute difference, 0-255) says nothing.
const STILL: f32 = 0.4;
/// A repeat differs by less than this share of the stretch's next smallest difference.
const REPEAT_SHARE: f32 = 0.3;

/// `runs`: stretches of consecutive frames, each value a frame's difference to the one before.
pub fn judge(runs: &[Vec<f32>]) -> Verdict {
    let mut verdict = Verdict { moving: 0, repeating: 0 };
    for run in runs {
        for cycle in run.chunks_exact(5) {
            let mut sorted = cycle.to_vec();
            sorted.sort_by(f32::total_cmp);
            if sorted[1] >= STILL {
                verdict.moving += 1;
                if sorted[0] < REPEAT_SHARE * sorted[1] {
                    verdict.repeating += 1;
                }
            }
        }
    }
    verdict
}

/// Luma on a 48×48 grid, as the Mac's `LumaSignature` samples a frame.
fn signature(gray: &[u8], w: usize, h: usize) -> Vec<f32> {
    const GRID: usize = 48;
    let mut out = Vec::with_capacity(GRID * GRID);
    for gy in 0..GRID {
        let y = gy * (h - 1) / (GRID - 1);
        for gx in 0..GRID {
            out.push(gray[y * w + gx * (w - 1) / (GRID - 1)] as f32);
        }
    }
    out
}

/// Whether the video repeats one frame in five: up to six stretches of 60
/// frames across it, decoded small.
pub fn detect(tools: &Tools, input: &Path, probe: &Probe) -> Result<Verdict, String> {
    let video = probe.video().ok_or("no video stream")?;
    let (fw, fh) = (video.width.unwrap_or(0), video.height.unwrap_or(0));
    if fw < 48 || fh < 48 {
        return Ok(Verdict { moving: 0, repeating: 0 });
    }
    let w = fw.min(384) & !1;
    let h = ((fh as f64 * w as f64 / fw as f64).round() as usize & !1).max(48);
    let duration = probe.duration();
    let fps = probe.rate().fps().max(1.0);
    let seconds = (61.0 / fps).min(duration);
    let stretches = if duration > 30.0 { 6 } else { 1 };
    let mut runs = Vec::new();
    for i in 0..stretches {
        let at = if stretches == 1 { 0.0 } else { (duration - seconds) * (0.1 + 0.8 * i as f64 / (stretches - 1) as f64) };
        let mut args: Vec<String> = ["-nostdin", "-v", "error"].map(String::from).to_vec();
        if at > 0.0 {
            args.extend(["-ss".into(), format!("{at:.3}")]);
        }
        args.extend(["-i".into(), input.to_string_lossy().into_owned(), "-map".into(), format!("0:{}", video.index),
                     "-frames:v".into(), "61".into(), "-fps_mode".into(), "passthrough".into(),
                     "-vf".into(), format!("scale={w}:{h},format=gray"), "-f".into(), "rawvideo".into(), "-".into()]);
        let output = Tools::command(&tools.ffmpeg).args(&args).output().map_err(|e| format!("ffmpeg didn't start: {e}"))?;
        if !output.status.success() {
            continue;
        }
        let frames: Vec<Vec<f32>> = output.stdout.chunks_exact(w * h).map(|f| signature(f, w, h)).collect();
        runs.push(frames.windows(2).map(|p| p[0].iter().zip(&p[1]).map(|(a, b)| (a - b).abs()).sum::<f32>() / p[0].len() as f32).collect());
    }
    Ok(judge(&runs))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Differences measured on a film remuxed at 31.24 fps (the Mac's test has the same).
    const PULLDOWN: [f32; 40] = [14.83, 15.27, 0.61, 15.45, 15.10, 14.56, 14.44, 0.16, 14.53, 14.38, 14.38, 14.24, 0.22,
                                 14.23, 14.89, 14.63, 14.61, 0.58, 14.77, 14.97, 15.09, 15.93, 0.19, 16.33, 16.53, 15.57,
                                 15.33, 0.16, 15.94, 13.39, 19.18, 22.70, 0.69, 25.96, 29.29, 24.77, 20.01, 0.09, 24.03, 24.02];

    #[test]
    fn finds_one_repeat_in_five() {
        assert!(judge(&[PULLDOWN.to_vec(), PULLDOWN[3..].to_vec()]).is_pulldown());
        let moving: Vec<f32> = (0..60).map(|i| 10.0 + (i % 7) as f32).collect();
        assert!(!judge(std::slice::from_ref(&moving)).is_pulldown());
        assert_eq!(judge(&[vec![0.3; 60]]).moving, 0);
        let mut some = moving;
        for i in (0..60).step_by(15) {
            some[i] = 0.1;
        }
        assert!(!judge(&[some]).is_pulldown());
    }
}
