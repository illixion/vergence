//! Pictures into model inputs and model outputs into stored fields: a port of
//! VergenceCore's `SharedFieldAnalysis` (`rgbPlanar`, `areaDownsample`),
//! operation for operation in f32, so the same frame gives the same tensors
//! as the Swift worker before either model sees it.

/// One borrowed 8-bit plane.
#[derive(Clone, Copy)]
pub struct Plane<'a> {
    pub data: &'a [u8],
    pub width: usize,
    pub height: usize,
    pub stride: usize,
}

impl Plane<'_> {
    /// Bilinear sample at (x, y), edges clamped (`AnalysisPlane.sample`).
    #[inline(always)]
    fn sample(&self, x: f32, y: f32) -> f32 {
        let cx = x.max(0.0).min((self.width - 1) as f32);
        let cy = y.max(0.0).min((self.height - 1) as f32);
        let (x0, y0) = (cx as usize, cy as usize);
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let (fx, fy) = (cx - x0 as f32, cy - y0 as f32);
        let r0 = &self.data[y0 * self.stride..];
        let r1 = &self.data[y1 * self.stride..];
        let (a, b) = (r0[x0] as f32, r0[x1] as f32);
        let (c, d) = (r1[x0] as f32, r1[x1] as f32);
        let top = a + (b - a) * fx;
        let bottom = c + (d - c) * fx;
        top + (bottom - top) * fy
    }
}

/// RIFE's working size.
pub const FLOW_SIZE: (usize, usize) = (480, 288);
/// NeuFlow's working size.
pub const DISPARITY_SIZE: (usize, usize) = (512, 288);
/// How much smaller than the model's output the fields are stored.
pub const STORAGE_FACTOR: usize = 4;
/// Stored fields are int16 in steps of 1/32 of a stored pixel.
pub const STEP: f32 = 1.0 / 32.0;

/// A 4:2:0 BT.709 limited-range picture as planar RGB, 0…1 (times `scale`),
/// resized to `width`×`height` bilinearly the way PyTorch's
/// `F.interpolate(align_corners=False)` does (no prefilter). Chroma is
/// centre-sited.
pub fn rgb_planar(y: Plane, cb: Plane, cr: Plane, width: usize, height: usize, scale: f32) -> Vec<f32> {
    let plane = width * height;
    let mut out = vec![0f32; 3 * plane];
    let sx = y.width as f32 / width as f32;
    let sy = y.height as f32 / height as f32;
    let csx = cb.width as f32 / y.width as f32;
    let csy = cb.height as f32 / y.height as f32;
    let (r, rest) = out.split_at_mut(plane);
    let (g, b) = rest.split_at_mut(plane);
    for oy in 0..height {
        let fy = (oy as f32 + 0.5) * sy - 0.5;
        let cfy = (fy + 0.5) * csy - 0.5;
        for ox in 0..width {
            let fx = (ox as f32 + 0.5) * sx - 0.5;
            let cfx = (fx + 0.5) * csx - 0.5;
            let luma = (y.sample(fx, fy) - 16.0) / 219.0;
            let u = (cb.sample(cfx, cfy) - 128.0) / 224.0;
            let v = (cr.sample(cfx, cfy) - 128.0) / 224.0;
            let i = oy * width + ox;
            r[i] = (luma + 1.5748 * v).clamp(0.0, 1.0) * scale;
            g[i] = (luma - 0.18733 * u - 0.46812 * v).clamp(0.0, 1.0) * scale;
            b[i] = (luma + 1.8556 * u).clamp(0.0, 1.0) * scale;
        }
    }
    out
}

/// Box average over `factor`×`factor` blocks, planar. Dimensions must divide.
pub fn area_downsample(planar: &[f32], channels: usize, width: usize, height: usize, factor: usize) -> Vec<f32> {
    assert!(width.is_multiple_of(factor) && height.is_multiple_of(factor) && planar.len() == channels * width * height);
    let (w, h) = (width / factor, height / factor);
    let mut out = vec![0f32; channels * w * h];
    let norm = 1.0 / (factor * factor) as f32;
    for c in 0..channels {
        for oy in 0..h {
            for ox in 0..w {
                let mut sum = 0f32;
                for dy in 0..factor {
                    let row = c * width * height + (oy * factor + dy) * width + ox * factor;
                    for dx in 0..factor {
                        sum += planar[row + dx];
                    }
                }
                out[c * w * h + oy * w + ox] = sum * norm;
            }
        }
    }
    out
}

/// The two eyes of a frame-packed I420 picture, read in place: (Y, Cb, Cr) each.
pub fn eyes<'a>(frame: &'a [u8], w: usize, h: usize, side_by_side: bool) -> ([Plane<'a>; 3], [Plane<'a>; 3]) {
    let (luma, chroma) = frame.split_at(w * h);
    let (cb, cr) = chroma.split_at((w / 2) * (h / 2));
    fn p(data: &[u8], width: usize, height: usize, stride: usize) -> Plane<'_> {
        Plane { data, width, height, stride }
    }
    if side_by_side {
        let eye = |e: usize| {
            [
                p(&luma[e * w / 2..], w / 2, h, w),
                p(&cb[e * w / 4..], w / 4, h / 2, w / 2),
                p(&cr[e * w / 4..], w / 4, h / 2, w / 2),
            ]
        };
        (eye(0), eye(1))
    } else {
        let eye = |e: usize| {
            [
                p(&luma[e * w * h / 2..], w, h / 2, w),
                p(&cb[e * (w / 2) * (h / 4)..], w / 2, h / 4, w / 2),
                p(&cr[e * (w / 2) * (h / 4)..], w / 2, h / 4, w / 2),
            ]
        };
        (eye(0), eye(1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 16×8 I420 picture with a known pattern (as in the Swift test that
    /// produced `SWIFT_RGB`).
    pub(crate) fn pattern() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let (w, h) = (16usize, 8usize);
        let y = (0..w * h).map(|i| (16 + (i * 37 + (i / w) * 11) % 220) as u8).collect();
        let cb = (0..(w / 2) * (h / 2)).map(|i| (40 + (i * 53) % 180) as u8).collect();
        let cr = (0..(w / 2) * (h / 2)).map(|i| (200 - (i * 29) % 170) as u8).collect();
        (y, cb, cr)
    }

    /// `SharedFieldAnalysis.rgbPlanar` on `pattern()` resized to 6×3, as the
    /// Swift code printed it (`Float.bitPattern`), so equality is exact.
    const SWIFT_RGB: [u32; 54] = include!("../tests/fixtures/swift_rgb_planar.txt");

    #[test]
    fn matches_swift_rgb_planar_bit_for_bit() {
        let (y, cb, cr) = pattern();
        let rgb = rgb_planar(
            Plane { data: &y, width: 16, height: 8, stride: 16 },
            Plane { data: &cb, width: 8, height: 4, stride: 8 },
            Plane { data: &cr, width: 8, height: 4, stride: 8 },
            6,
            3,
            1.0,
        );
        let bits: Vec<u32> = rgb.iter().map(|v| v.to_bits()).collect();
        assert_eq!(bits, SWIFT_RGB);
    }

    #[test]
    fn area_downsample_averages_blocks() {
        let planar: Vec<f32> = (0..2 * 4 * 4).map(|v| v as f32).collect();
        let out = area_downsample(&planar, 2, 4, 4, 2);
        assert_eq!(out, vec![2.5, 4.5, 10.5, 12.5, 18.5, 20.5, 26.5, 28.5]);
    }

    #[test]
    fn eyes_split_side_by_side_and_top_and_bottom() {
        let (w, h) = (8, 4);
        let frame: Vec<u8> = (0..w * h * 3 / 2).map(|i| i as u8).collect();
        let (l, r) = eyes(&frame, w, h, true);
        assert_eq!((l[0].data[0], r[0].data[0], r[0].width, r[0].stride), (0, 4, 4, 8));
        assert_eq!((l[1].data[0], r[1].data[0]), (32, 34));
        let (l, r) = eyes(&frame, w, h, false);
        assert_eq!((l[0].data[0], r[0].data[0], r[0].height), (0, 16, 2));
        assert_eq!((l[2].data[0], r[2].data[0]), (40, 44));
    }
}
