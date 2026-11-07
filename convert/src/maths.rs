//! VergenceCore's depth maths, ported: `DepthRefinement.swift` (the affine, the
//! depth budget, anchor fits, the locked scale/shift tracker, cut detection,
//! the converter's whole-timeline mapping), `DepthAnchor.swift` and
//! `DepthHistogram.swift`, plus the converter's own small policies
//! (`DepthConverter`'s temporal window, the letterbox transform, the
//! static-frame gate). Tested against `tests/fixtures/core.json`, which
//! `scripts/make-convert-fixtures.sh` makes from the Swift itself.
//!
//! Floats are `f32` where Swift's are `Float`, `f64` where they are `Double`,
//! so the two agree to the last bit or close to it.

/// `y = a·x + b`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Affine {
    pub a: f32,
    pub b: f32,
}

impl Affine {
    pub const IDENTITY: Affine = Affine { a: 1.0, b: 0.0 };

    /// The mapping that sends `lo` to 0 and `hi` to 1.
    pub fn mapping(lo: f32, hi: f32) -> Affine {
        let span = (hi - lo).max(1e-6);
        Affine { a: 1.0 / span, b: -lo / span }
    }

    pub fn apply(self, x: f32) -> f32 {
        self.a * x + self.b
    }

    /// The mapping that sends a frame's raw `lo`…`hi` onto `range` (shot
    /// space): `DepthAffine(mappingLo:hi:onto:)`.
    pub fn onto(lo: f32, hi: f32, range: ShotRange) -> Affine {
        let a = (range.hi - range.lo) / (hi - lo).max(1e-6);
        Affine { a, b: range.lo - a * lo }
    }

    /// `self(inner(x))`.
    pub fn after(self, inner: Affine) -> Affine {
        Affine { a: self.a * inner.a, b: self.a * inner.b + self.b }
    }

    pub fn inverse(self) -> Option<Affine> {
        (self.a.abs() > 1e-9 && self.a.is_finite() && self.b.is_finite()).then(|| Affine { a: 1.0 / self.a, b: -self.b / self.a })
    }

    pub fn lerp(self, other: Affine, t: f32) -> Affine {
        Affine { a: self.a + (other.a - self.a) * t, b: self.b + (other.b - self.b) * t }
    }

    pub fn is_usable(self) -> bool {
        self.a.is_finite() && self.b.is_finite() && self.a > 0.0
    }
}

/// The soft-knee depth budget (`DepthBudget`): identity between the knees, tanh
/// toe and shoulder outside.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Budget {
    pub knee_lo: f32,
    pub knee_hi: f32,
}

impl Budget {
    pub const STUDIO: Budget = Budget { knee_lo: 0.08, knee_hi: 0.7 };

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn apply(self, y: f32) -> f32 {
        if !y.is_finite() {
            return if y > 0.0 { 1.0 } else { 0.0 };
        }
        if y > self.knee_hi {
            let w = 1.0 - self.knee_hi;
            return if w > 0.0 { self.knee_hi + w * ((y - self.knee_hi) / w).tanh() } else { y.min(1.0) };
        }
        if y < self.knee_lo {
            let w = self.knee_lo;
            return if w > 0.0 { self.knee_lo - w * ((self.knee_lo - y) / w).tanh() } else { y.max(0.0) };
        }
        y
    }
}

/// Luma change (0–1) below which a pixel anchors the fit (`DepthRefinement.staticLumaThreshold`).
pub const STATIC_LUMA_THRESHOLD: f32 = 0.04;
/// Mean luma change that reads as a cut (`DepthRefinement.cutMeanLumaDelta`).
pub const CUT_MEAN_LUMA_DELTA: f32 = 0.12;

/// Weighted second-moment sums over the anchor pixels (`DepthAlignmentSums`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AlignmentSums {
    pub w: f64,
    pub wx: f64,
    pub wy: f64,
    pub wxx: f64,
    pub wyy: f64,
    pub wxy: f64,
}

impl AlignmentSums {
    pub const ANCHOR_CEILING: f32 = 0.9;
    const MIN_ANCHOR_FRACTION: f64 = 0.15;
    const MIN_CORRELATION: f64 = 0.6;
    const MIN_RELATIVE_SPREAD: f64 = 0.04;
    const SLOPE_BOUNDS: (f64, f64) = (0.33, 3.0);

    pub fn add(&mut self, x: f32, y: f32, weight: f32) {
        let (w, x, y) = (weight as f64, x as f64, y as f64);
        self.w += w;
        self.wx += w * x;
        self.wy += w * y;
        self.wxx += w * x * x;
        self.wyy += w * y * y;
        self.wxy += w * x * y;
    }

    /// The reduced-major-axis affine with `y ≈ r(x)`, or None when the anchor
    /// can't be trusted.
    pub fn fit(&self, pixel_count: usize, frame_span: f32) -> Option<Affine> {
        if pixel_count == 0 || self.w < Self::MIN_ANCHOR_FRACTION * pixel_count as f64 {
            return None;
        }
        let w = self.w;
        let (mx, my) = (self.wx / w, self.wy / w);
        let vx = self.wxx / w - mx * mx;
        let vy = self.wyy / w - my * my;
        let cov = self.wxy / w - mx * my;
        let min_spread = Self::MIN_RELATIVE_SPREAD * frame_span.max(1e-6) as f64;
        if vx <= min_spread * min_spread || vy <= 0.0 || cov <= 0.0 {
            return None;
        }
        if cov / (vx.sqrt() * vy.sqrt()) < Self::MIN_CORRELATION {
            return None;
        }
        let a = (vy / vx).sqrt();
        if !(Self::SLOPE_BOUNDS.0..=Self::SLOPE_BOUNDS.1).contains(&a) {
            return None;
        }
        let b = my - a * mx;
        let affine = Affine { a: a as f32, b: b as f32 };
        affine.is_usable().then_some(affine)
    }
}

/// The previous frame's background layer (`DepthAnchorCeiling`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnchorCeiling {
    pub ceiling: f32,
    pub softness: f32,
}

impl AnchorCeiling {
    pub const NONE: AnchorCeiling = AnchorCeiling { ceiling: f32::MAX, softness: 1.0 };

    pub fn new(mapping: Affine) -> AnchorCeiling {
        match mapping.inverse() {
            Some(inverse) => {
                AnchorCeiling { ceiling: inverse.apply(AlignmentSums::ANCHOR_CEILING), softness: 0.05 / mapping.a }
            }
            None => AnchorCeiling::NONE,
        }
    }
}

/// The temporally locked scale/shift (`DepthScaleShiftTracker`).
#[derive(Clone, Debug, Default)]
pub struct ScaleShiftTracker {
    pub state: Option<Affine>,
}

/// A frame's robust depth range in shot space, where an earlier run's
/// anchoring put it (`DepthShotRange`): what carries a worker's run on into a
/// title's next piece.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ShotRange {
    pub lo: f32,
    pub hi: f32,
}

impl ScaleShiftTracker {
    /// A tracker carrying on from `state`, the mapping an earlier run locked.
    pub fn carrying_on(state: Affine) -> ScaleShiftTracker {
        ScaleShiftTracker { state: Some(state) }
    }

    const HOLD_SECONDS: f64 = 3.0;
    const RECOVER_SECONDS: f64 = 0.6;
    const UNANCHORED_SECONDS: f64 = 0.5;
    const SPAN_BOUNDS: (f32, f32) = (0.35, 8.0);

    pub fn reset(&mut self) {
        self.state = None;
    }

    pub fn update(&mut self, lo: f32, hi: f32, relative: Option<Affine>, cut: bool, dt: f64) -> Affine {
        let natural = Affine::mapping(lo, hi);
        let previous = match self.state {
            Some(previous) if !cut && previous.is_usable() => previous,
            _ => {
                self.state = Some(natural);
                return natural;
            }
        };
        let carried = relative.map(|r| previous.after(r)).unwrap_or(previous);
        let span = carried.apply(hi) - carried.apply(lo);
        let tau = match relative {
            None => Self::UNANCHORED_SECONDS,
            Some(_) if span > 1.0 => Self::HOLD_SECONDS,
            Some(_) => Self::RECOVER_SECONDS,
        };
        let step = (1.0 - (-dt.max(0.0) / tau.max(1e-3)).exp()) as f32;
        let mut next = carried.lerp(natural, step);
        let next_span = next.apply(hi) - next.apply(lo);
        let (lower, upper) = Self::SPAN_BOUNDS;
        if !next.is_usable() || !next_span.is_finite() {
            next = natural;
        } else if !(lower..=upper).contains(&next_span) {
            let target = next_span.clamp(lower, upper);
            let t = (target - next_span) / (1.0 - next_span);
            next = next.lerp(natural, t.clamp(0.0, 1.0));
        }
        self.state = Some(next);
        next
    }
}

/// Scene-cut detection on per-frame depth statistics (`DepthCutDetector`).
#[derive(Clone, Debug)]
pub struct CutDetector {
    prev_shape: Option<Vec<f32>>,
    prev_lo: f32,
    prev_hi: f32,
}

impl Default for CutDetector {
    fn default() -> Self {
        CutDetector { prev_shape: None, prev_lo: 0.0, prev_hi: 1.0 }
    }
}

impl CutDetector {
    const SHAPE_DISTANCE: f32 = 0.5;
    const RANGE_RATIO_BOUNDS: (f32, f32) = (0.6, 1.667);
    const RANGE_LO_JUMP: f32 = 0.35;

    pub fn is_cut(&mut self, shape: Vec<f32>, lo: f32, hi: f32, luma_jump: Option<f32>) -> bool {
        let result = match &self.prev_shape {
            Some(prev) if prev.len() == shape.len() => {
                if luma_jump.is_some_and(|j| j > CUT_MEAN_LUMA_DELTA) {
                    true
                } else {
                    let distance = 0.5 * shape.iter().zip(prev).fold(0f32, |acc, (a, b)| acc + (a - b).abs());
                    let prev_span = (self.prev_hi - self.prev_lo).max(1e-6);
                    let ratio = (hi - lo) / prev_span;
                    let lo_jump = (lo - self.prev_lo).abs() / prev_span;
                    distance > Self::SHAPE_DISTANCE
                        || !(Self::RANGE_RATIO_BOUNDS.0..=Self::RANGE_RATIO_BOUNDS.1).contains(&ratio)
                        || lo_jump > Self::RANGE_LO_JUMP
                }
            }
            _ => false,
        };
        self.prev_shape = Some(shape);
        self.prev_lo = lo;
        self.prev_hi = hi;
        result
    }

    pub fn shape(bins: &[u32]) -> Vec<f32> {
        let total: u64 = bins.iter().map(|&b| b as u64).sum();
        if total == 0 {
            return vec![0.0; bins.len()];
        }
        bins.iter().map(|&b| b as f32 / total as f32).collect()
    }
}

/// Number of histogram bins (`DepthHistogram.binCount`).
pub const BINS: usize = 256;

/// Value at quantile `q`, interpolated within its bin (`DepthHistogram.percentile`).
pub fn percentile(bins: &[u32], total: usize, q: f32, lo: f32, hi: f32) -> f32 {
    if total == 0 {
        return lo;
    }
    let target = total as f32 * q;
    let mut cumulative = 0f32;
    for (i, &count) in bins.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let next = cumulative + count as f32;
        if next >= target {
            let frac = (target - cumulative) / count as f32;
            return lo + (i as f32 + frac) / BINS as f32 * (hi - lo);
        }
        cumulative = next;
    }
    hi
}

/// One frame of the converter's timeline (`DepthShotMapping.Frame`; one lattice).
#[derive(Clone, Copy, Debug)]
pub struct ShotFrame {
    pub seconds: f64,
    pub lo: f32,
    pub hi: f32,
    pub cut_before: bool,
    pub relative: Option<Affine>,
}

/// Raw units → shot space for every frame (`DepthShotMapping.map`, single
/// lattice: `vergence-convert` makes depth in one sweep).
pub fn shot_mapping(frames: &[ShotFrame]) -> Vec<Affine> {
    shot_mapping_seeded(frames, None).0
}

/// The same, carrying on from an earlier run (`DepthShotMapping.map(_:seed:)`):
/// `seed` is where that run's anchoring put the first frame here (a frame
/// both decoded) in shot space, so the chain continues as if the timeline had
/// never been cut. Also returns each frame's locked mapping, what a later run
/// seeds from.
pub fn shot_mapping_seeded(frames: &[ShotFrame], seed: Option<ShotRange>) -> (Vec<Affine>, Vec<Affine>) {
    let n = frames.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut tracker = ScaleShiftTracker::default();
    let mut last_seconds: Option<f64> = None;
    let mut locked = vec![Affine::IDENTITY; n];
    for (i, f) in frames.iter().enumerate() {
        if i == 0
            && let Some(seed) = seed
        {
            let state = Affine::onto(f.lo, f.hi, seed);
            tracker = ScaleShiftTracker::carrying_on(state);
            last_seconds = Some(f.seconds);
            locked[0] = state;
            continue;
        }
        if f.cut_before {
            tracker.reset();
        }
        let fresh = tracker.state.is_none();
        let dt = last_seconds.map(|s| f.seconds - s).unwrap_or(0.0);
        locked[i] = tracker.update(f.lo, f.hi, if fresh { None } else { f.relative }, fresh,
                                   if dt.is_finite() { dt.max(0.0) } else { 0.0 });
        last_seconds = Some(f.seconds);
    }
    let weights = [1f32, 2.0, 1.0];
    let radius = weights.len() / 2;
    let mut result = locked.clone();
    let mut start = 0;
    for i in 0..=n {
        if !(i == n || (i > 0 && frames[i].cut_before)) {
            continue;
        }
        for k in start..i {
            let (mut lo, mut hi, mut wsum) = (0f32, 0f32, 0f32);
            for (o, &w) in weights.iter().enumerate() {
                let j = k as isize - radius as isize + o as isize;
                if j < start as isize || j >= i as isize {
                    continue;
                }
                let j = j as usize;
                lo += w * locked[j].apply(frames[j].lo);
                hi += w * locked[j].apply(frames[j].hi);
                wsum += w;
            }
            lo /= wsum;
            hi /= wsum;
            let span = (frames[k].hi - frames[k].lo).max(1e-6);
            let a = (hi - lo) / span;
            let candidate = Affine { a, b: lo - a * frames[k].lo };
            result[k] = if candidate.is_usable() { candidate } else { locked[k] };
        }
        start = i;
    }
    (result, locked)
}

/// UV transform from frame UV into the content region of a letterboxed depth
/// map (`CoreMLDepthProvider.letterboxUVTransform`): sample at `uv·scale + offset`.
pub fn letterbox(video_w: usize, video_h: usize, depth_w: usize, depth_h: usize) -> ([f32; 2], [f32; 2]) {
    if video_w == 0 || video_h == 0 || depth_w == 0 || depth_h == 0 {
        return ([1.0, 1.0], [0.0, 0.0]);
    }
    let (vw, vh, dw, dh) = (video_w as f32, video_h as f32, depth_w as f32, depth_h as f32);
    let s = (dw / vw).min(dh / vh);
    let content = [vw * s / dw, vh * s / dh];
    (content, [(1.0 - content[0]) * 0.5, (1.0 - content[1]) * 0.5])
}

/// The ±2 temporal window's binomial weights (`DepthConverter.temporalBaseWeights`).
pub const TEMPORAL_WEIGHTS: [f32; 5] = [1.0, 4.0, 6.0, 4.0, 1.0];
pub const TEMPORAL_RADIUS: usize = 2;
/// The depth-similarity gate, a share of the centre frame's range.
pub const TEMPORAL_DEPTH_SIGMA: f32 = 0.15;

/// Frame `j`'s raw units → frame `c`'s through the chain of anchor fits
/// (`relative[k]` maps k → k-1); identity when implausible (`relativeMapping`).
pub fn relative_mapping(relative: &[Option<Affine>], j: usize, c: usize) -> Affine {
    if j == c {
        return Affine::IDENTITY;
    }
    let mut m = Affine::IDENTITY;
    if j > c {
        for k in (c + 1..=j).rev() {
            m = relative[k].unwrap_or(Affine::IDENTITY).after(m);
        }
    } else {
        for r in &relative[j + 1..=c] {
            m = r.and_then(Affine::inverse).unwrap_or(Affine::IDENTITY).after(m);
        }
    }
    if (0.5..=2.0).contains(&m.a) && m.b.is_finite() { m } else { Affine::IDENTITY }
}

/// The ±2 window around `c` truncated at cuts, normalised: weights per slot
/// (0 for a slot left out) and each slot's frame index (the centre where left out).
pub fn temporal_window(cut_before: &[bool], c: usize, last_available: usize) -> ([f32; 5], [usize; 5]) {
    let mut weights = [0f32; 5];
    let mut frames = [c; 5];
    for i in 0..5 {
        let j = c as isize - TEMPORAL_RADIUS as isize + i as isize;
        let mut included = j >= 0 && j as usize <= last_available;
        if included && j as usize != c {
            let j = j as usize;
            let (from, to) = (j.min(c) + 1, j.max(c));
            included = !cut_before[from..=to].iter().any(|&cut| cut);
        }
        if included {
            weights[i] = TEMPORAL_WEIGHTS[i];
            frames[i] = j as usize;
        }
    }
    let sum: f32 = weights.iter().sum();
    if sum > 0.0 {
        for w in &mut weights {
            *w /= sum;
        }
    }
    (weights, frames)
}

/// The converter's static-frame gate (`FrameChangeDetector` over
/// `LumaSignature`): a 48×48 luma grid, compared with the last inferred frame.
#[derive(Default)]
pub struct FrameChangeDetector {
    reference: Vec<f32>,
    consecutive_skips: usize,
}

pub const SIGNATURE_GRID: usize = 48;

impl FrameChangeDetector {
    const THRESHOLD: f32 = 1.0;
    const MAX_CONSECUTIVE_SKIPS: usize = 60;

    /// `(B + 2G + R) / 4` on the 0–255 scale at the grid points of an RGB
    /// frame whose samples `sample(x, y)` returns on the 0–255 scale.
    pub fn signature(width: usize, height: usize, sample: impl Fn(usize, usize) -> [f32; 3]) -> Option<Vec<f32>> {
        if width < SIGNATURE_GRID || height < SIGNATURE_GRID {
            return None;
        }
        let mut out = Vec::with_capacity(SIGNATURE_GRID * SIGNATURE_GRID);
        for gy in 0..SIGNATURE_GRID {
            let y = gy * (height - 1) / (SIGNATURE_GRID - 1);
            for gx in 0..SIGNATURE_GRID {
                let x = gx * (width - 1) / (SIGNATURE_GRID - 1);
                let [r, g, b] = sample(x, y);
                out.push((b + 2.0 * g + r) * 0.25);
            }
        }
        Some(out)
    }

    /// True when the frame is static against the last inferred one (its depth
    /// can be reused); false makes it the new reference.
    pub fn should_reuse(&mut self, signature: Option<Vec<f32>>) -> bool {
        let Some(signature) = signature else {
            self.reference.clear();
            return false;
        };
        if self.consecutive_skips < Self::MAX_CONSECUTIVE_SKIPS
            && signature.len() == self.reference.len()
            && !signature.is_empty()
        {
            let delta = signature.iter().zip(&self.reference).map(|(a, b)| (a - b).abs()).sum::<f32>() / signature.len() as f32;
            if delta < Self::THRESHOLD {
                self.consecutive_skips += 1;
                return true;
            }
        }
        self.reference = signature;
        self.consecutive_skips = 0;
        false
    }
}

/// Bits per second for the packed picture (`SpatialBitrate.bitsPerSecond` with
/// `secondEye: false`).
pub fn bits_per_second(width: usize, height: usize, frame_rate: f64, quality: Quality, source: Option<f64>) -> u64 {
    let rate = if frame_rate > 0.0 { frame_rate } else { 30.0 };
    let mut bits = (width * height) as f64 * rate * quality.bits_per_pixel();
    if let Some(source) = source
        && source > 0.0
        && quality != Quality::Max
    {
        bits = bits.min(source * 1.25);
    }
    bits.max(1_000_000.0).round() as u64
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quality {
    Low,
    Standard,
    High,
    Max,
}

impl Quality {
    pub fn parse(text: &str) -> Option<Quality> {
        Some(match text {
            "low" => Quality::Low,
            "standard" => Quality::Standard,
            "high" => Quality::High,
            "max" => Quality::Max,
            _ => return None,
        })
    }

    fn bits_per_pixel(self) -> f64 {
        match self {
            Quality::Low => 0.07,
            Quality::Standard => 0.12,
            Quality::High => 0.18,
            Quality::Max => 0.3,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn fixtures() -> Value {
        serde_json::from_str(include_str!("../tests/fixtures/core.json")).unwrap()
    }

    fn affine(v: &Value) -> Option<Affine> {
        v.as_array().map(|a| Affine { a: a[0].as_f64().unwrap() as f32, b: a[1].as_f64().unwrap() as f32 })
    }

    fn f(v: &Value) -> f32 {
        v.as_f64().unwrap() as f32
    }

    fn close(a: f32, b: f32, tolerance: f32) -> bool {
        (a - b).abs() <= tolerance * (1.0 + b.abs())
    }

    fn same(a: Option<Affine>, b: Option<Affine>, tolerance: f32) -> bool {
        match (a, b) {
            (Some(a), Some(b)) => close(a.a, b.a, tolerance) && close(a.b, b.b, tolerance),
            (None, None) => true,
            _ => false,
        }
    }

    #[test]
    fn budget_matches_swift() {
        let d = fixtures();
        let ys = d["budget"]["y"].as_array().unwrap();
        let expected = d["budget"]["studio"].as_array().unwrap();
        for (y, e) in ys.iter().zip(expected) {
            assert!(close(Budget::STUDIO.apply(f(y)), f(e), 1e-6), "{y} → {e}");
        }
    }

    #[test]
    fn anchor_fits_match_swift() {
        for case in fixtures()["fits"].as_array().unwrap() {
            let s: Vec<f64> = case["sums"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
            let sums = AlignmentSums { w: s[0], wx: s[1], wy: s[2], wxx: s[3], wyy: s[4], wxy: s[5] };
            let fit = sums.fit(case["pixels"].as_u64().unwrap() as usize, f(&case["span"]));
            assert!(same(fit, affine(&case["fit"]), 1e-6), "{case} gave {fit:?}");
        }
    }

    #[test]
    fn tracker_matches_swift() {
        let mut tracker = ScaleShiftTracker::default();
        for case in fixtures()["tracker"].as_array().unwrap() {
            let mapping = tracker.update(f(&case["lo"]), f(&case["hi"]), affine(&case["relative"]),
                                         case["cut"].as_bool().unwrap(), case["dt"].as_f64().unwrap());
            assert!(same(Some(mapping), affine(&case["mapping"]), 1e-5), "{case} gave {mapping:?}");
        }
    }

    #[test]
    fn cuts_match_swift() {
        let mut detector = CutDetector::default();
        for case in fixtures()["cuts"].as_array().unwrap() {
            let bins: Vec<u32> = case["bins"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
            let jump = case["jump"].as_f64().map(|j| j as f32);
            let cut = detector.is_cut(CutDetector::shape(&bins), f(&case["lo"]), f(&case["hi"]), jump);
            assert_eq!(cut, case["cut"].as_bool().unwrap());
        }
    }

    #[test]
    fn percentiles_match_swift() {
        for case in fixtures()["percentiles"].as_array().unwrap() {
            let bins: Vec<u32> = case["bins"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
            let value = percentile(&bins, case["total"].as_u64().unwrap() as usize, f(&case["q"]), f(&case["lo"]), f(&case["hi"]));
            assert!(close(value, f(&case["value"]), 1e-6), "{value} vs {}", case["value"]);
        }
    }

    #[test]
    fn shot_mapping_matches_swift() {
        let d = fixtures();
        let cases = d["shotMapping"].as_array().unwrap();
        let frames: Vec<ShotFrame> = cases
            .iter()
            .map(|c| ShotFrame {
                seconds: c["seconds"].as_f64().unwrap(),
                lo: f(&c["lo"]),
                hi: f(&c["hi"]),
                cut_before: c["cut"].as_bool().unwrap(),
                relative: affine(&c["relative"]),
            })
            .collect();
        for (got, case) in shot_mapping(&frames).into_iter().zip(cases) {
            assert!(same(Some(got), affine(&case["mapping"]), 1e-5), "{case} gave {got:?}");
        }
    }

    #[test]
    fn ceilings_match_swift() {
        for case in fixtures()["ceilings"].as_array().unwrap() {
            let c = AnchorCeiling::new(affine(&case["mapping"]).unwrap());
            assert!(close(c.ceiling, f(&case["ceiling"]), 1e-6) && close(c.softness, f(&case["softness"]), 1e-6));
        }
    }

    #[test]
    fn window_stops_at_cuts() {
        let cuts = [false, false, false, true, false, false];
        let (w, frames) = temporal_window(&cuts, 2, 5);
        // Frame 3 starts a new shot: slots 3 and 4 (frames 3, 4) are left out.
        assert_eq!(frames, [0, 1, 2, 2, 2]);
        assert!((w[0] - 1.0 / 11.0).abs() < 1e-6 && w[3] == 0.0 && w[4] == 0.0);
        let (w, _) = temporal_window(&cuts, 0, 0);
        assert_eq!(w, [0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn relative_mapping_chains_fits() {
        let r = [None, Some(Affine { a: 2.0, b: 0.0 }), Some(Affine { a: 1.0, b: 0.1 })];
        // 2 → 1 → 0: x·1+0.1 then ·2.
        let m = relative_mapping(&r, 2, 0);
        assert!((m.a - 2.0).abs() < 1e-6 && (m.b - 0.2).abs() < 1e-6);
        let back = relative_mapping(&r, 0, 2);
        assert!((back.a - 0.5).abs() < 1e-6 && (back.b + 0.1).abs() < 1e-6);
        // Implausible products fall back to identity.
        assert_eq!(relative_mapping(&[None, Some(Affine { a: 3.0, b: 0.0 })], 1, 0), Affine::IDENTITY);
    }

    #[test]
    fn letterbox_matches_the_provider() {
        // 1036×582 into 1036×784: full width, 582/784 of the height, centred.
        let (scale, offset) = letterbox(1036, 582, 1036, 784);
        assert!((scale[0] - 1.0).abs() < 1e-6 && (scale[1] - 582.0 / 784.0).abs() < 1e-6);
        assert!((offset[1] - (1.0 - 582.0 / 784.0) / 2.0).abs() < 1e-6);
    }

    #[test]
    fn bitrate_like_spatial_bitrate() {
        // 3840×1080 at 23.976, standard: 11.9 Mbps, as `vergence convert` says for a 1080p film.
        let bits = bits_per_second(3840, 1080, 24000.0 / 1001.0, Quality::Standard, None);
        assert_eq!((bits as f64 / 1e5).round() / 10.0, 11.9);
        assert_eq!(bits_per_second(64, 64, 24.0, Quality::Low, Some(10.0)), 1_000_000);
    }

    #[test]
    fn static_frames_reuse_depth() {
        let mut gate = FrameChangeDetector::default();
        let flat = |v: f32| FrameChangeDetector::signature(64, 64, move |_, _| [v, v, v]);
        assert!(!gate.should_reuse(flat(100.0)));
        assert!(gate.should_reuse(flat(100.5)));
        assert!(!gate.should_reuse(flat(110.0)));
        assert!(!gate.should_reuse(FrameChangeDetector::signature(16, 16, |_, _| [0.0; 3])));
    }

    /// As VergenceCore's `aSeededRunCarriesOnTheChainAsIfNeverCut`: a run from
    /// frame 30 seeded with the whole run's lock there maps every later frame
    /// as the uncut run does; a fresh start doesn't.
    #[test]
    fn a_seeded_run_carries_on_the_chain_as_if_never_cut() {
        let frames: Vec<ShotFrame> = (0..80)
            .map(|i| ShotFrame {
                seconds: i as f64 / 24.0,
                lo: 0.1 + 0.01 * i as f32,
                hi: 0.9 + 0.03 * i as f32,
                cut_before: false,
                relative: (i > 0).then_some(Affine { a: 0.97, b: 0.01 }),
            })
            .collect();
        let (whole, locked) = shot_mapping_seeded(&frames, None);
        let seed = ShotRange { lo: locked[30].apply(frames[30].lo), hi: locked[30].apply(frames[30].hi) };
        let (piece, _) = shot_mapping_seeded(&frames[30..], Some(seed));
        let fresh = shot_mapping(&frames[30..]);
        let (mut seeded, mut unseeded) = (0f32, 0f32);
        for i in 32..79 {
            seeded = seeded.max((piece[i - 30].apply(0.5) - whole[i].apply(0.5)).abs());
            unseeded = unseeded.max((fresh[i - 30].apply(0.5) - whole[i].apply(0.5)).abs());
        }
        assert!(seeded < 1e-5, "{seeded}");
        assert!(unseeded > 0.01, "{unseeded}");
    }
}
