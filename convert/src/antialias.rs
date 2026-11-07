//! `--antialias`: stair-stepped edges made lines, the Mac's `EdgeAntialias`
//! (VergenceApple) on the CPU, step for step: the structure tensor of luma
//! at half size (luma blurred to σ 2 pixels, Sobel, products blurred to σ 8)
//! says which way each edge runs and how straight it is; along strong,
//! straight edges the picture is averaged along the edge (Gaussian σ 12 over
//! ±24 pixels), never across it, so a staircase becomes the line it stands
//! for and texture stays.
//!
//!   out = picture + α · (picture averaged along the edge − picture)
//!   α   = smoothstep(0.45, 0.8, coherence) · smoothstep(0.01, 0.035, edge strength)

const ALONG_SIGMA: f32 = 12.0;
const ALONG_RADIUS: i32 = 24;
const COHERENCE: (f32, f32) = (0.45, 0.8);
const STRENGTH: (f32, f32) = (0.01, 0.035);

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(16)
}

/// A separable Gaussian over `channels` interleaved values per pixel, clamped at the edges.
fn gaussian(data: &mut [f32], w: usize, h: usize, channels: usize, sigma: f32) {
    let radius = (3.0 * sigma).ceil() as i64;
    let weights: Vec<f32> = (-radius..=radius).map(|k| (-0.5 * (k * k) as f32 / (sigma * sigma)).exp()).collect();
    let total: f32 = weights.iter().sum();
    let pass = |src: &[f32], dst: &mut [f32], horizontal: bool| {
        let rows = h.div_ceil(threads());
        std::thread::scope(|scope| {
            for (band, chunk) in dst.chunks_mut(rows * w * channels).enumerate() {
                let weights = &weights;
                scope.spawn(move || {
                    for (r, row) in chunk.chunks_mut(w * channels).enumerate() {
                        let y = band * rows + r;
                        for x in 0..w {
                            for c in 0..channels {
                                let mut sum = 0f32;
                                for (i, k) in (-radius..=radius).enumerate() {
                                    let (sx, sy) = if horizontal {
                                        ((x as i64 + k).clamp(0, w as i64 - 1) as usize, y)
                                    } else {
                                        (x, (y as i64 + k).clamp(0, h as i64 - 1) as usize)
                                    };
                                    sum += weights[i] * src[(sy * w + sx) * channels + c];
                                }
                                row[x * channels + c] = sum / total;
                            }
                        }
                    }
                });
            }
        });
    };
    let mut tmp = vec![0f32; data.len()];
    pass(data, &mut tmp, true);
    pass(&tmp, data, false);
}

/// The tensor (gx², gx·gy, gy²) at half size from full-size luma.
fn tensor(luma: &[f32], w: usize, h: usize) -> (Vec<f32>, usize, usize) {
    let (hw, hh) = ((w / 2).max(1), (h / 2).max(1));
    let mut half = vec![0f32; hw * hh];
    for y in 0..hh {
        for x in 0..hw {
            let mut sum = 0f32;
            for dy in 0..2 {
                for dx in 0..2 {
                    let (sx, sy) = ((2 * x + dx).min(w - 1), (2 * y + dy).min(h - 1));
                    sum += luma[sy * w + sx];
                }
            }
            half[y * hw + x] = sum / 4.0;
        }
    }
    gaussian(&mut half, hw, hh, 1, 1.0);
    let at = |x: i64, y: i64| half[(y.clamp(0, hh as i64 - 1) as usize) * hw + x.clamp(0, hw as i64 - 1) as usize];
    let mut t = vec![0f32; hw * hh * 3];
    for y in 0..hh as i64 {
        for x in 0..hw as i64 {
            let (a, b, c) = (at(x - 1, y - 1), at(x, y - 1), at(x + 1, y - 1));
            let (d, f) = (at(x - 1, y), at(x + 1, y));
            let (g, hh_, i) = (at(x - 1, y + 1), at(x, y + 1), at(x + 1, y + 1));
            let gx = ((c + 2.0 * f + i) - (a + 2.0 * d + g)) / 16.0;
            let gy = ((g + 2.0 * hh_ + i) - (a + 2.0 * b + c)) / 16.0;
            let o = (y as usize * hw + x as usize) * 3;
            t[o] = gx * gx;
            t[o + 1] = gx * gy;
            t[o + 2] = gy * gy;
        }
    }
    gaussian(&mut t, hw, hh, 3, 4.0);
    (t, hw, hh)
}

/// Bilinear, pixel-centred coordinates, clamped to the plane.
fn bilinear(p: &[f32], w: usize, h: usize, x: f32, y: f32) -> f32 {
    let x = (x - 0.5).clamp(0.0, (w - 1) as f32);
    let y = (y - 0.5).clamp(0.0, (h - 1) as f32);
    let (x0, y0) = (x as usize, y as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let top = p[y0 * w + x0] + (p[y0 * w + x1] - p[y0 * w + x0]) * fx;
    let bottom = p[y1 * w + x0] + (p[y1 * w + x1] - p[y1 * w + x0]) * fx;
    top + (bottom - top) * fy
}

/// At full-size position (x, y) (pixel centres at +0.5): α and the edge's direction.
fn edge_at(t: &[f32], hw: usize, hh: usize, x: f32, y: f32) -> (f32, f32, f32) {
    let hx = (x / 2.0).clamp(0.5, hw as f32 - 0.5);
    let hy = (y / 2.0).clamp(0.5, hh as f32 - 0.5);
    let mut j = [0f32; 3];
    let (fx, fy) = (hx - 0.5, hy - 0.5);
    let (x0, y0) = (fx as usize, fy as usize);
    let (x1, y1) = ((x0 + 1).min(hw - 1), (y0 + 1).min(hh - 1));
    let (ax, ay) = (fx - x0 as f32, fy - y0 as f32);
    for (c, out) in j.iter_mut().enumerate() {
        let v = |xx: usize, yy: usize| t[(yy * hw + xx) * 3 + c];
        let top = v(x0, y0) + (v(x1, y0) - v(x0, y0)) * ax;
        let bottom = v(x0, y1) + (v(x1, y1) - v(x0, y1)) * ax;
        *out = top + (bottom - top) * ay;
    }
    let (tr, d) = (j[0] + j[2], ((j[0] - j[2]) * (j[0] - j[2]) + 4.0 * j[1] * j[1]).sqrt());
    let (l1, l2) = ((tr + d) / 2.0, (tr - d) / 2.0);
    let coherence = ((l1 - l2) / (l1 + l2 + 1e-12)).powi(2);
    let alpha = smoothstep(COHERENCE.0, COHERENCE.1, coherence) * smoothstep(STRENGTH.0, STRENGTH.1, l1.max(0.0).sqrt());
    let theta = 0.5 * (2.0 * j[1]).atan2(j[0] - j[2]);
    (alpha, -theta.sin(), theta.cos())
}

/// Smooths `planes` (each `pw`×`ph`, `scale` full-size pixels to one of
/// theirs) along the edges of the tensor made at full size.
fn smooth(planes: &mut [&mut [f32]], pw: usize, ph: usize, scale: f32, t: &[f32], hw: usize, hh: usize) {
    let sources: Vec<Vec<f32>> = planes.iter().map(|p| p.to_vec()).collect();
    let weights: Vec<f32> = (1..=ALONG_RADIUS).map(|k| (-0.5 * (k * k) as f32 / (ALONG_SIGMA * ALONG_SIGMA)).exp()).collect();
    // Each pixel's α and direction once, then every plane.
    let mut edges = vec![(0f32, 0f32, 0f32); pw * ph];
    let rows = ph.div_ceil(threads());
    std::thread::scope(|scope| {
        for (band, chunk) in edges.chunks_mut(rows * pw).enumerate() {
            scope.spawn(move || {
                for (i, e) in chunk.iter_mut().enumerate() {
                    let (x, y) = (i % pw, band * rows + i / pw);
                    *e = edge_at(t, hw, hh, (x as f32 + 0.5) * scale, (y as f32 + 0.5) * scale);
                }
            });
        }
    });
    for (plane, source) in planes.iter_mut().zip(&sources) {
        std::thread::scope(|scope| {
            for (band, chunk) in plane.chunks_mut(rows * pw).enumerate() {
                let (source, weights, edges) = (source, &weights, &edges);
                scope.spawn(move || {
                    for (i, out) in chunk.iter_mut().enumerate() {
                        let (x, y) = (i % pw, band * rows + i / pw);
                        let (alpha, tx, ty) = edges[y * pw + x];
                        if alpha < 1.0 / 512.0 {
                            continue;
                        }
                        let (cx, cy) = (x as f32 + 0.5, y as f32 + 0.5);
                        let here = source[y * pw + x];
                        let (mut sum, mut total) = (here, 1f32);
                        for (k, w) in weights.iter().enumerate() {
                            let k = (k + 1) as f32 / scale;
                            sum += w * (bilinear(source, pw, ph, cx + tx * k, cy + ty * k)
                                + bilinear(source, pw, ph, cx - tx * k, cy - ty * k));
                            total += 2.0 * w;
                        }
                        *out = here + alpha * (sum / total - here);
                    }
                });
            }
        });
    }
}

/// One eye as planar float RGB (R, G, B planes, 0…1), in place.
pub fn rgb(eye: &mut [f32], w: usize, h: usize) {
    let n = w * h;
    let luma: Vec<f32> = (0..n).map(|i| 0.2126 * eye[i] + 0.7152 * eye[n + i] + 0.0722 * eye[2 * n + i]).collect();
    let (t, hw, hh) = tensor(&luma, w, h);
    let mut planes: Vec<&mut [f32]> = eye.chunks_exact_mut(n).take(3).collect();
    smooth(&mut planes, w, h, 1.0, &t, hw, hh);
}

/// A 4:2:0 picture as the decoder gives it (8-bit, or 10-bit in 16-bit
/// little-endian samples; limited range), in place.
pub fn yuv420(frame: &mut [u8], w: usize, h: usize, ten_bit: bool) {
    let max = if ten_bit { 1023.0 } else { 255.0 };
    let read = |frame: &[u8], i: usize| -> f32 {
        if ten_bit { u16::from_le_bytes([frame[2 * i], frame[2 * i + 1]]) as f32 / max } else { frame[i] as f32 / max }
    };
    let (cw, ch) = (w / 2, h / 2);
    let mut planes: Vec<Vec<f32>> = [(0, w * h), (w * h, cw * ch), (w * h + cw * ch, cw * ch)]
        .iter()
        .map(|&(start, len)| (start..start + len).map(|i| read(frame, i)).collect())
        .collect();
    // Limited-range luma to 0…1, as the RGB the Mac measures would be.
    let luma: Vec<f32> = planes[0].iter().map(|y| ((y * 255.0 - 16.0) / 219.0).clamp(0.0, 1.0)).collect();
    let (t, hw, hh) = tensor(&luma, w, h);
    {
        let (luma_plane, chroma) = planes.split_at_mut(1);
        smooth(&mut [&mut luma_plane[0][..]], w, h, 1.0, &t, hw, hh);
        let (cb, cr) = chroma.split_at_mut(1);
        smooth(&mut [&mut cb[0][..], &mut cr[0][..]], cw, ch, 2.0, &t, hw, hh);
    }
    let mut i = 0;
    for plane in &planes {
        for v in plane {
            let q = (v * max).round().clamp(0.0, max) as u16;
            if ten_bit {
                frame[2 * i..2 * i + 2].copy_from_slice(&q.to_le_bytes());
            } else {
                frame[i] = q as u8;
            }
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Where the edge crosses half brightness in column `x`.
    fn edge(img: &[f32], w: usize, h: usize, x: usize) -> f32 {
        for y in 1..h {
            let (a, b) = (img[(y - 1) * w + x], img[y * w + x]);
            if (a - 0.5) * (b - 0.5) <= 0.0 && a != b {
                return (y - 1) as f32 + (0.5 - a) / (b - a) + 0.5;
            }
        }
        f32::NAN
    }

    #[test]
    fn stairs_become_the_line_they_stand_for() {
        // The Mac's test: y = 40 + 0.15 x drawn in steps 4 pixels tall.
        let (w, h) = (480, 160);
        let line = |x: f32| 40.0 + 0.15 * x;
        let mut eye = vec![0f32; 3 * w * h];
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    let stepped = (line(x as f32 + 0.5) / 4.0).floor() * 4.0 + 2.0;
                    eye[c * w * h + y * w + x] = if (y as f32 + 0.5) < stepped { 0.2 } else { 0.8 };
                }
            }
        }
        let wobble = |img: &[f32]| {
            let xs: Vec<usize> = (60..w - 60).collect();
            (xs.iter().map(|&x| (edge(img, w, h, x) - line(x as f32 + 0.5)).powi(2)).sum::<f32>() / xs.len() as f32).sqrt()
        };
        let before = wobble(&eye[w * h..2 * w * h]);
        rgb(&mut eye, w, h);
        let after = wobble(&eye[w * h..2 * w * h]);
        assert!(before > 1.0 && after < before * 0.25, "edge RMS off the line {before} -> {after}");
    }

    #[test]
    fn a_flat_picture_stays() {
        let (w, h) = (64, 48);
        let mut frame: Vec<u8> = vec![100; w * h * 3 / 2];
        let before = frame.clone();
        yuv420(&mut frame, w, h, false);
        assert_eq!(frame, before);
    }
}
