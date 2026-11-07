//! Telling a frame-packed 3D video from its picture, as VergenceCore's
//! `StereoLayoutDetector.scores` does: the best normalised cross-correlation
//! of the left half against the right (and the top against the bottom),
//! trying horizontal shifts up to 6 % of the width, over eight frames from
//! across the video. Scores near 1 mean the halves show the same scene.

use std::path::Path;

use crate::probe::{Probe, Tools};

/// Above this the halves show the same scene; the winner must beat the other split by `MARGIN`.
pub const THRESHOLD: f64 = 0.6;
pub const MARGIN: f64 = 0.15;

/// The best correlation of two regions of a grey picture (`w` wide), nil when one is flat.
fn best_shift_correlation(p: &[u8], w: usize, a: (usize, usize), b: (usize, usize), width: usize, height: usize) -> Option<f64> {
    let max_shift = (width * 6 / 100).max(1) as i64;
    let inset = max_shift as usize + width / 20;
    let top = height / 10;
    if width < 2 * inset + 8 || height < 2 * top + 8 {
        return None;
    }
    let mut best: Option<f64> = None;
    let step = (max_shift / 12).max(1);
    let mut shift = -max_shift;
    while shift <= max_shift {
        let (mut sa, mut sb, mut saa, mut sbb, mut sab, mut n) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        for y in top..height - top {
            let row_a = (a.1 + y) * w + a.0;
            let row_b = ((b.1 + y) * w + b.0) as i64 + shift;
            for x in inset..width - inset {
                let va = p[row_a + x] as f64;
                let vb = p[(row_b + x as i64) as usize] as f64;
                sa += va;
                sb += vb;
                saa += va * va;
                sbb += vb * vb;
                sab += va * vb;
                n += 1.0;
            }
        }
        let cov = sab - sa * sb / n;
        let (var_a, var_b) = (saa - sa * sa / n, sbb - sb * sb / n);
        if var_a / n > 4.0 && var_b / n > 4.0 {
            let r = cov / (var_a * var_b).sqrt();
            if best.is_none_or(|b| r > b) {
                best = Some(r);
            }
        }
        shift += step;
    }
    best
}

/// (side by side, over-under) scores of one grey picture.
pub fn scores(p: &[u8], w: usize, h: usize) -> Option<(f64, f64)> {
    let sbs = best_shift_correlation(p, w, (0, 0), (w / 2, 0), w / 2, h)?;
    let ou = best_shift_correlation(p, w, (0, 0), (0, h / 2), w, h / 2)?;
    Some((sbs, ou))
}

/// What eight pictures from across the video say: Some(true) side by side,
/// Some(false) over-under, None neither (2D, or unsure).
pub fn detect(tools: &Tools, input: &Path, probe: &Probe) -> Option<bool> {
    let video = probe.video()?;
    let (fw, fh) = (video.width?, video.height?);
    let w = fw.min(480) & !1;
    let h = ((fh as f64 * w as f64 / fw as f64).round() as usize & !1).max(2);
    let (mut sbs, mut ou, mut counted) = (0.0, 0.0, 0usize);
    for i in 0..8 {
        let at = probe.duration() * (0.1 + 0.8 * i as f64 / 7.0);
        let output = Tools::command(&tools.ffmpeg)
            .args(["-nostdin", "-v", "error", "-ss", &format!("{at:.3}"), "-i"])
            .arg(input)
            .args(["-frames:v", "1", "-vf", &format!("scale={w}:{h},format=gray"), "-f", "rawvideo", "-"])
            .output()
            .ok()?;
        if output.stdout.len() != w * h {
            continue;
        }
        if let Some((a, b)) = scores(&output.stdout, w, h) {
            sbs += a;
            ou += b;
            counted += 1;
        }
    }
    if counted == 0 {
        return None;
    }
    let (sbs, ou) = (sbs / counted as f64, ou / counted as f64);
    (sbs.max(ou) >= THRESHOLD && (sbs - ou).abs() >= MARGIN).then_some(sbs > ou)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tells_side_by_side_from_2d() {
        let (w, h) = (256, 72);
        let texture = |x: usize, y: usize| ((x.wrapping_mul(73_856_093) ^ y.wrapping_mul(19_349_663)) % 200) as u8 + 20;
        // The same scene in both halves, shifted by a few pixels of disparity.
        let sbs: Vec<u8> = (0..w * h).map(|i| { let (x, y) = (i % w, i / w); texture(x % 128 + if x >= 128 { 3 } else { 0 }, y) }).collect();
        let (a, b) = scores(&sbs, w, h).unwrap();
        assert!(a > 0.9 && a - b > MARGIN, "{a} {b}");
        let flat: Vec<u8> = (0..w * h).map(|i| texture(i % w, i / w)).collect();
        let (a, b) = scores(&flat, w, h).unwrap();
        assert!(a < THRESHOLD && b < THRESHOLD, "{a} {b}");
    }
}
