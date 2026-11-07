//! AI HDR gain grids: VergenceCore's `GainGrid` and `GainGridSchedule`
//! (`GainGrid.swift`, where the record layout is pinned), ported operation for
//! operation. A keyframe's gain map is fitted into a 16×9×8 grid over (x, y,
//! luma) as log2 gains; frames between keyframes of one shot ease from one
//! grid to the next, and a frame before a cut keeps its shot's grid.

use crate::preprocess::Plane;

pub const WIDTH: usize = 16;
pub const HEIGHT: usize = 9;
pub const LUMA_BINS: usize = 8;
pub const COUNT: usize = WIDTH * HEIGHT * LUMA_BINS;
/// GMNet runs this often, in seconds, and at every cut.
pub const KEYFRAME_SECONDS: f64 = 0.5;
/// `DepthRefinement.cutMeanLumaDelta`.
const CUT_MEAN_LUMA_DELTA: f32 = 0.12;
/// `GainGrid.pivotLuma`, `highlightExpansion`, `peakLog2Gain`: a grid keeps
/// only the gain above each cell's mid-tone gain, doubled, up to GMNet's 5×.
pub const PIVOT_LUMA: f32 = 0.5;
pub const HIGHLIGHT_EXPANSION: f32 = 2.0;
pub fn peak_log2_gain() -> f32 {
    5f32.log2()
}

/// log2 of the linear gain, planar: bin by bin, rows top to bottom.
#[derive(Clone, Debug, PartialEq)]
pub struct GainGrid(pub Vec<f32>);

#[inline(always)]
fn index(x: usize, y: usize, l: usize) -> usize {
    (l * HEIGHT + y) * WIDTH + x
}

/// The lookup's axis: the two cells around `t` (0…1) and the weight of the second.
#[inline(always)]
fn axis(t: f32, n: usize) -> (usize, usize, f32) {
    let f = (t * n as f32 - 0.5).max(0.0).min((n - 1) as f32);
    let i0 = f as usize;
    (i0, (i0 + 1).min(n - 1), f - i0 as f32)
}

/// Non-linear luma of full-range R′G′B′ (BT.709 weights).
#[inline(always)]
pub fn luma(r: f32, g: f32, b: f32) -> f32 {
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

impl GainGrid {
    /// The grid that best stands for a gain map (`GainGrid.fit`): each pixel
    /// splatted into the cells around its (u, v, luma) with the lookup's
    /// trilinear weights, as a log gain; empty cells take the nearest cell's
    /// value at the same level, or the nearest level's in their own column.
    pub fn fit(gain: &[f32], luma_plane: &[f32], width: usize, height: usize) -> GainGrid {
        assert!(gain.len() == width * height && luma_plane.len() == gain.len(), "map and luma sizes");
        let mut sum = vec![0f64; COUNT];
        let mut weight = vec![0f64; COUNT];
        let mut total = 0f64;
        for py in 0..height {
            let (y0, y1, fy) = axis((py as f32 + 0.5) / height as f32, HEIGHT);
            for px in 0..width {
                let i = py * width + px;
                let g = gain[i].max(1e-3).log2() as f64;
                total += g;
                let (x0, x1, fx) = axis((px as f32 + 0.5) / width as f32, WIDTH);
                let (l0, l1, fl) = axis(luma_plane[i].clamp(0.0, 1.0), LUMA_BINS);
                for (l, wl) in [(l0, 1.0 - fl), (l1, fl)] {
                    if wl <= 0.0 {
                        continue;
                    }
                    for (y, wy) in [(y0, 1.0 - fy), (y1, fy)] {
                        if wy <= 0.0 {
                            continue;
                        }
                        for (x, wx) in [(x0, 1.0 - fx), (x1, fx)] {
                            if wx <= 0.0 {
                                continue;
                            }
                            // The weight is a float product, as in Swift, then widened.
                            let w = (wl * wy * wx) as f64;
                            let c = index(x, y, l);
                            sum[c] += w * g;
                            weight[c] += w;
                        }
                    }
                }
            }
        }
        let mean = (total / gain.len().max(1) as f64) as f32;
        let mut grid = vec![mean; COUNT];
        let mut known = vec![false; COUNT];
        for c in 0..COUNT {
            if weight[c] > 1e-3 {
                grid[c] = (sum[c] / weight[c]) as f32;
                known[c] = true;
            }
        }
        let mut filled = grid.clone();
        for l in 0..LUMA_BINS {
            let cells: Vec<(usize, usize)> = (0..HEIGHT)
                .flat_map(|y| (0..WIDTH).map(move |x| (x, y)))
                .filter(|&(x, y)| known[index(x, y, l)])
                .collect();
            if cells.is_empty() {
                continue;
            }
            for y in 0..HEIGHT {
                for x in 0..WIDTH {
                    if known[index(x, y, l)] {
                        continue;
                    }
                    let d = |&(cx, cy): &(usize, usize)| {
                        let (dx, dy) = (cx as i64 - x as i64, cy as i64 - y as i64);
                        dx * dx + dy * dy
                    };
                    // Swift's `min(by:)`: the first of the nearest.
                    let nearest = cells.iter().fold(None::<&(usize, usize)>, |best, c| match best {
                        Some(b) if d(b) <= d(c) => Some(b),
                        _ => Some(c),
                    });
                    let &(nx, ny) = nearest.expect("cells isn't empty");
                    filled[index(x, y, l)] = grid[index(nx, ny, l)];
                    known[index(x, y, l)] = true;
                }
            }
        }
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let levels: Vec<usize> = (0..LUMA_BINS).filter(|&l| known[index(x, y, l)]).collect();
                if levels.is_empty() || levels.len() == LUMA_BINS {
                    continue;
                }
                for l in 0..LUMA_BINS {
                    if known[index(x, y, l)] {
                        continue;
                    }
                    let nearest = levels.iter().fold(None::<usize>, |best, &k| match best {
                        Some(b) if b.abs_diff(l) <= k.abs_diff(l) => Some(b),
                        _ => Some(k),
                    });
                    filled[index(x, y, l)] = filled[index(x, y, nearest.expect("levels isn't empty"))];
                }
            }
        }
        GainGrid(filled)
    }

    /// `GainGrid.expandingHighlights`: the gain measured from each cell's
    /// mid-tone gain (at `PIVOT_LUMA`) and expanded, within 0 and the peak.
    pub fn expanding_highlights(&self) -> GainGrid {
        let f = (PIVOT_LUMA * LUMA_BINS as f32 - 0.5).max(0.0).min((LUMA_BINS - 1) as f32);
        let l0 = f as usize;
        let (l1, t) = ((l0 + 1).min(LUMA_BINS - 1), f - l0 as f32);
        let peak = peak_log2_gain();
        let mut out = self.0.clone();
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let (a, b) = (self.0[index(x, y, l0)], self.0[index(x, y, l1)]);
                let mid_tone = a + (b - a) * t;
                for l in 0..LUMA_BINS {
                    let i = index(x, y, l);
                    out[i] = (HIGHLIGHT_EXPANSION * (self.0[i] - mid_tone)).max(0.0).min(peak);
                }
            }
        }
        GainGrid(out)
    }

    /// Linear in log gain, `t` from self (0) to `next` (1).
    pub fn eased(&self, next: &GainGrid, t: f32) -> GainGrid {
        GainGrid(self.0.iter().zip(&next.0).map(|(a, b)| a + (b - a) * t).collect())
    }
}

/// Frames between keyframes at `frame_rate` (12 at 24 fps).
pub fn keyframe_interval(frame_rate: f64) -> i64 {
    ((KEYFRAME_SECONDS * frame_rate).round() as i64).max(1)
}

/// Whether frame `index` is a keyframe: on the title's grid (so a piece and a
/// whole-file run agree), or the first frame after a cut.
pub fn is_keyframe(index: i64, interval: i64, cut: bool) -> bool {
    cut || index % interval == 0
}

#[derive(Clone, Debug)]
pub struct Keyframe {
    pub index: i64,
    pub grid: GainGrid,
    /// The frame starts a shot.
    pub cut: bool,
}

/// A sticky run's state, kept between a title's pieces: the keyframes so far,
/// the last frame decoded and its signature (the cut detector's), and where
/// the next piece starts. The next piece then decodes on from there instead
/// of its keyframe interval of context, and comes out the same.
#[derive(Clone, Debug)]
pub struct Run {
    pub title: String,
    pub next: i64,
    pub last_decoded: i64,
    pub signature: Vec<f32>,
    pub keyframes: Vec<Keyframe>,
}

/// Frame `index`'s grid from the keyframes around it (sorted by index); None
/// before the first.
pub fn grid_at(index: i64, keyframes: &[Keyframe]) -> Option<GainGrid> {
    let a = keyframes.iter().rposition(|k| k.index <= index)?;
    let key = &keyframes[a];
    if key.index == index || a + 1 >= keyframes.len() || keyframes[a + 1].cut {
        return Some(key.grid.clone());
    }
    let next = &keyframes[a + 1];
    let t = (index - key.index) as f32 / (next.index - key.index) as f32;
    Some(key.grid.eased(&next.grid, t))
}

/// A frame's luma on a sparse 48×48 grid, 0…255: what tells a cut.
pub fn signature(plane: &Plane) -> Vec<f32> {
    let n = 48;
    let mut out = vec![0f32; n * n];
    for j in 0..n {
        for i in 0..n {
            out[j * n + i] = plane.sample((i as f32 + 0.5) * plane.width as f32 / n as f32 - 0.5,
                                          (j as f32 + 0.5) * plane.height as f32 / n as f32 - 0.5);
        }
    }
    out
}

/// A cut lies between the frames these signatures came from.
pub fn is_cut(previous: &[f32], current: &[f32]) -> bool {
    if previous.len() != current.len() || current.is_empty() {
        return false;
    }
    let mean = previous.iter().zip(current).fold(0f32, |s, (a, b)| s + (b - a).abs()) / current.len() as f32;
    mean > CUT_MEAN_LUMA_DELTA * 255.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fits_a_flat_gain_and_fills_empty_cells() {
        // Half the frame dark, half bright; the bright half gains 4× (2 stops).
        let (w, h) = (64, 36);
        let mut gain = vec![1.0f32; w * h];
        let mut l = vec![0.1f32; w * h];
        for y in 0..h {
            for x in w / 2..w {
                gain[y * w + x] = 4.0;
                l[y * w + x] = 0.9;
            }
        }
        let grid = GainGrid::fit(&gain, &l, w, h);
        // A bright cell on the right, a dark one on the left.
        assert!((grid.0[index(15, 4, 7)] - 2.0).abs() < 1e-4);
        assert!(grid.0[index(0, 4, 0)].abs() < 1e-4);
        // Bright luma on the left was never seen there: it takes the nearest bright cell's.
        assert!((grid.0[index(0, 4, 7)] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn spreads_highlights_from_the_mid_tones() {
        let mut v = vec![0f32; COUNT];
        for l in 0..LUMA_BINS {
            for y in 0..HEIGHT {
                for x in 0..WIDTH {
                    v[index(x, y, l)] = 0.1 * l as f32 + if x == 0 { 1.0 } else { 0.0 };
                }
            }
        }
        let g = GainGrid(v).expanding_highlights();
        // Bins 0–3 sit at or below the pivot (luma 0.5, halfway between bins 3 and 4).
        assert!((0..4).all(|l| g.0[index(5, 4, l)] == 0.0));
        // Bin 7 was 0.35 over the pivot: now 0.7, in a cell a stop up too.
        assert!((g.0[index(5, 4, 7)] - 0.7).abs() < 1e-5 && (g.0[index(0, 4, 7)] - 0.7).abs() < 1e-5);
        let steep = GainGrid((0..COUNT).map(|i| (i / (WIDTH * HEIGHT)) as f32).collect()).expanding_highlights();
        assert_eq!(steep.0.iter().cloned().fold(0.0, f32::max), peak_log2_gain());
    }

    #[test]
    fn eases_within_a_shot_and_holds_before_a_cut() {
        let k = |index, value: f32, cut| Keyframe { index, grid: GainGrid(vec![value; COUNT]), cut };
        let keys = vec![k(0, 0.0, false), k(12, 1.2, false), k(18, 2.0, true)];
        assert_eq!(grid_at(6, &keys).unwrap().0[0], 0.6);
        assert_eq!(grid_at(15, &keys).unwrap().0[0], 1.2);
        assert_eq!(grid_at(20, &keys).unwrap().0[0], 2.0);
        assert!(grid_at(-1, &keys).is_none());
        assert_eq!(keyframe_interval(24000.0 / 1001.0), 12);
        assert!(is_keyframe(24, 12, false) && is_keyframe(5, 12, true) && !is_keyframe(5, 12, false));
    }

    #[test]
    fn tells_a_cut() {
        assert!(is_cut(&[10.0; 4], &[60.0; 4]));
        assert!(!is_cut(&[10.0; 4], &[30.0; 4]));
    }
}
