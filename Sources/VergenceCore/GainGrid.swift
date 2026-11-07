/*
 Vergence - AI HDR for video: a frame's gain as a small bilateral grid

 An SDR video gets HDR highlights from GMNet's gain map (Liao et al.,
 "Learning Gain Map for Inverse Tone Mapping", ICLR 2025, MIT), the network
 Convolution already runs on photos. A gain *image* per frame would cost a
 network run per frame and swim with motion, so the gain is kept as a grid
 over (x, y, luma) instead (Hypnos `docs/plans/analysis-workers.md`, "AI HDR
 for video, in short"):

 - A worker (`vergence worker`, workload `ai-hdr`) runs GMNet on keyframes,
   every cut and every half second, and fits each map into a grid.
 - Frames between two keyframes of one shot ease linearly from one grid to
   the next; a frame before a cut keeps its shot's grid (snap, never blend
   across a cut). Every frame gets its own `gainMap` record.
 - The player looks every pixel's gain up with its own luma, so a moving
   highlight keeps its boost, and in-between frames and both stereo eyes get
   consistent gains without a network of their own.

 The record (`AnalysisKind.gainMap`), pinned here:

   eye mono, encoding float16, channels 8, width 16, height 9
   channel l, row y, column x  =  log2 of the linear gain for pixels in
   column cell x, row cell y, whose luma falls in bin l

 Cells and bins are sampled at their centres: column x at (x + ½) / 16 of the
 picture's width, row y at (y + ½) / 9 of its height, bin l at luma
 (l + ½) / 8. A pixel's gain is the trilinear interpolation at (u, v, Y′),
 clamped to the edge cells: exactly a 16×9×8 3D texture sampled with
 normalised coordinates, linear filtering and clamp-to-edge. Y′ is the SDR
 picture's own non-linear luma, 0.2126 R′ + 0.7152 G′ + 0.0722 B′ on full
 range 0…1. The gain multiplies linear light: HDR = SDR linear × 2^g. 0 is
 SDR as it is; GMNet's real-world checkpoint peaks at 5× (2.32).

 The player caps the frame's average gain below the window's brightness
 limiter (2.5–3× on a Vision Pro) by scaling every log gain by the same
 factor (`cappedScale`).

 Foundation only: a Quest's GLES pass reads the same records.
 */

import Foundation

public struct GainGrid: Sendable, Equatable {
    public static let width = 16
    public static let height = 9
    public static let lumaBins = 8
    public static let count = width * height * lumaBins

    /// log2 of the linear gain, planar: bin by bin, rows top to bottom.
    public var log2Gain: [Float]

    public init(log2Gain: [Float]) {
        precondition(log2Gain.count == Self.count, "a gain grid has 16×9×8 cells")
        self.log2Gain = log2Gain
    }

    /// No gain anywhere: the SDR picture as it is.
    public static let identity = GainGrid(log2Gain: [Float](repeating: 0, count: count))

    @inline(__always) static func index(_ x: Int, _ y: Int, _ l: Int) -> Int {
        (l * height + y) * width + x
    }

    // MARK: Lookup (what the player's pass does)

    /// The log2 gain for a pixel at normalised (u, v) whose luma is `luma`.
    public func lookup(u: Float, v: Float, luma: Float) -> Float {
        func axis(_ t: Float, _ n: Int) -> (Int, Int, Float) {
            let f = min(max(t * Float(n) - 0.5, 0), Float(n - 1))
            let i0 = Int(f)
            return (i0, min(i0 + 1, n - 1), f - Float(i0))
        }
        let (x0, x1, fx) = axis(u, Self.width)
        let (y0, y1, fy) = axis(v, Self.height)
        let (l0, l1, fl) = axis(luma, Self.lumaBins)
        func g(_ x: Int, _ y: Int, _ l: Int) -> Float { log2Gain[Self.index(x, y, l)] }
        func plane(_ l: Int) -> Float {
            let top = g(x0, y0, l) + (g(x1, y0, l) - g(x0, y0, l)) * fx
            let bottom = g(x0, y1, l) + (g(x1, y1, l) - g(x0, y1, l)) * fx
            return top + (bottom - top) * fy
        }
        let a = plane(l0), b = plane(l1)
        return a + (b - a) * fl
    }

    /// Non-linear luma of full-range R′G′B′ (BT.709 weights).
    @inline(__always) public static func luma(r: Float, g: Float, b: Float) -> Float {
        0.2126 * r + 0.7152 * g + 0.0722 * b
    }

    // MARK: Fitting (what a worker does with a keyframe's gain map)

    /// The grid that best stands for a gain map: each of the map's pixels is
    /// splatted into the cells around its (u, v, luma) with the lookup's own
    /// trilinear weights, as a log gain. A cell no pixel reached takes the
    /// nearest cell's value at the same luma level, or, for a level the frame
    /// has nowhere, the nearest level's in its own column.
    /// - Parameters:
    ///   - gain: linear multipliers (≥ 1), row-major, `width`×`height`.
    ///   - lumaPlane: the frame's luma (`luma(r:g:b:)`) at the same size.
    public static func fit(gain: [Float], luma lumaPlane: [Float], width: Int, height: Int) -> GainGrid {
        precondition(gain.count == width * height && lumaPlane.count == gain.count, "map and luma sizes")
        var sum = [Double](repeating: 0, count: count)
        var weight = [Double](repeating: 0, count: count)
        var total = 0.0
        func axis(_ t: Float, _ n: Int) -> (Int, Int, Float) {
            let f = min(max(t * Float(n) - 0.5, 0), Float(n - 1))
            let i0 = Int(f)
            return (i0, min(i0 + 1, n - 1), f - Float(i0))
        }
        for py in 0 ..< height {
            let (y0, y1, fy) = axis((Float(py) + 0.5) / Float(height), Self.height)
            for px in 0 ..< width {
                let i = py * width + px
                let g = Double(log2(max(gain[i], 1e-3)))
                total += g
                let (x0, x1, fx) = axis((Float(px) + 0.5) / Float(width), Self.width)
                let (l0, l1, fl) = axis(min(max(lumaPlane[i], 0), 1), Self.lumaBins)
                for (l, wl) in [(l0, 1 - fl), (l1, fl)] where wl > 0 {
                    for (y, wy) in [(y0, 1 - fy), (y1, fy)] where wy > 0 {
                        for (x, wx) in [(x0, 1 - fx), (x1, fx)] where wx > 0 {
                            let w = Double(wl * wy * wx)
                            let c = index(x, y, l)
                            sum[c] += w * g
                            weight[c] += w
                        }
                    }
                }
            }
        }
        let mean = Float(total / Double(max(gain.count, 1)))
        var grid = [Float](repeating: mean, count: count)
        let enough = 1e-3
        var known = [Bool](repeating: false, count: count)
        for c in 0 ..< count where weight[c] > enough {
            grid[c] = Float(sum[c] / weight[c])
            known[c] = true
        }
        // An empty cell is a luma level the frame doesn't have there. It takes
        // what the nearest cell with that level got, so a highlight moving into
        // a dark area before the next keyframe keeps its boost; a level the
        // whole frame lacks takes the nearest level in its own column.
        var filled = grid
        for l in 0 ..< lumaBins {
            let cells = (0 ..< Self.height).flatMap { y in (0 ..< Self.width).map { x in (x, y) } }
                .filter { known[index($0.0, $0.1, l)] }
            guard !cells.isEmpty else { continue }
            for y in 0 ..< Self.height {
                for x in 0 ..< Self.width where !known[index(x, y, l)] {
                    let nearest = cells.min { a, b in
                        (a.0 - x) * (a.0 - x) + (a.1 - y) * (a.1 - y) < (b.0 - x) * (b.0 - x) + (b.1 - y) * (b.1 - y)
                    }!
                    filled[index(x, y, l)] = grid[index(nearest.0, nearest.1, l)]
                    known[index(x, y, l)] = true
                }
            }
        }
        for y in 0 ..< Self.height {
            for x in 0 ..< Self.width {
                let levels = (0 ..< lumaBins).filter { known[index(x, y, $0)] }
                guard !levels.isEmpty, levels.count < lumaBins else { continue }
                for l in 0 ..< lumaBins where !known[index(x, y, l)] {
                    let nearest = levels.min { abs($0 - l) < abs($1 - l) }!
                    filled[index(x, y, l)] = filled[index(x, y, nearest)]
                }
            }
        }
        return GainGrid(log2Gain: filled)
    }

    // MARK: Highlights apart from the diffuse tones

    /// On film, GMNet's gain is close to one smooth curve over luma (the
    /// grid fits its map within 1 % on average): mid-tones at code 0.4–0.6
    /// get 1.3–1.8×, the top of the range 2–3×. Trained on phone photos,
    /// whose highlights clip, it keeps its 4–5× for clipped areas, which a
    /// graded film seldom has (a 3D film's 20:00–30:00 sat at p99 code
    /// 0.64–0.98). So the picture got brighter as a whole while a light only
    /// rose about 1.5× over a lit wall: no highlights to see, as if no HDR.
    /// A worker's grid therefore keeps only the gain above each cell's
    /// mid-tone gain (at `pivotLuma`), and doubles it in log2: mid-tones and
    /// shadows stay as SDR, the top of the range spreads to GMNet's 5× peak.
    public static let pivotLuma: Float = 0.5
    public static let highlightExpansion: Float = 2
    /// GMNet's real-world checkpoint's peak, log2 (what MaxCLL declares).
    public static let peakLog2Gain: Float = log2(5)

    /// This grid with its gain measured from each cell's mid-tone gain and
    /// expanded (see `pivotLuma`), never below 0 or above `peakLog2Gain`.
    public func expandingHighlights(by expansion: Float = highlightExpansion,
                                    pivot: Float = pivotLuma) -> GainGrid {
        let f = min(max(pivot * Float(Self.lumaBins) - 0.5, 0), Float(Self.lumaBins - 1))
        let l0 = Int(f), l1 = min(l0 + 1, Self.lumaBins - 1), t = f - Float(l0)
        var out = log2Gain
        for y in 0 ..< Self.height {
            for x in 0 ..< Self.width {
                let a = log2Gain[Self.index(x, y, l0)], b = log2Gain[Self.index(x, y, l1)]
                let midTone = a + (b - a) * t
                for l in 0 ..< Self.lumaBins {
                    let i = Self.index(x, y, l)
                    out[i] = min(max(expansion * (log2Gain[i] - midTone), 0), Self.peakLog2Gain)
                }
            }
        }
        return GainGrid(log2Gain: out)
    }

    // MARK: Over time

    /// Linear in log gain, `t` from self (0) to `next` (1).
    public func eased(to next: GainGrid, _ t: Float) -> GainGrid {
        GainGrid(log2Gain: zip(log2Gain, next.log2Gain).map { $0 + ($1 - $0) * t })
    }

    /// The factor every log gain is scaled by so that `meanGain`, the frame's
    /// average linear gain, comes out at about `cap`: 1 when it is below.
    public static func cappedScale(meanGain: Float, cap: Float) -> Float {
        guard cap > 1 else { return 0 }
        guard meanGain > cap else { return 1 }
        return log2(cap) / log2(meanGain)
    }

    // MARK: Records

    public func record(time: AnalysisTime) -> AnalysisRecord {
        AnalysisRecord.encoding(log2Gain, kind: .gainMap, as: .float16, channels: Self.lumaBins, width: Self.width,
                                height: Self.height, time: time)
    }

    /// A record's grid, when it has the pinned layout (inflate it first).
    public init?(record: AnalysisRecord) {
        guard record.kind == .gainMap, record.eye == .mono, !record.isDeflated, record.channels == Self.lumaBins,
              record.width == Self.width, record.height == Self.height else { return nil }
        self.init(log2Gain: record.values())
    }
}

/// Keyframes for a run of frames and the grid each frame gets between them:
/// eased within a shot, held before a cut.
public enum GainGridSchedule {
    /// Keyframe spacing, in seconds: GMNet runs this often and at every cut.
    public static let keyframeSeconds = 0.5

    /// Frames between keyframes at `frameRate` (12 at 24 fps).
    public static func keyframeInterval(frameRate: Double) -> Int64 {
        max(1, Int64((keyframeSeconds * frameRate).rounded()))
    }

    /// Whether frame `index` is a keyframe: on the title's grid (so a piece and
    /// a whole-file run agree), or the first frame after a cut.
    public static func isKeyframe(_ index: Int64, interval: Int64, cut: Bool) -> Bool {
        cut || index % interval == 0
    }

    public struct Keyframe: Sendable {
        public var index: Int64
        public var grid: GainGrid
        /// The frame starts a shot.
        public var cut: Bool
        public init(index: Int64, grid: GainGrid, cut: Bool) {
            self.index = index
            self.grid = grid
            self.cut = cut
        }
    }

    /// Frame `index`'s grid from the keyframes around it (sorted by index):
    /// the one at or before it, eased toward the next unless the next starts
    /// a new shot. Nil before the first keyframe.
    public static func grid(at index: Int64, keyframes: [Keyframe]) -> GainGrid? {
        guard let a = keyframes.lastIndex(where: { $0.index <= index }) else { return nil }
        let key = keyframes[a]
        guard key.index != index, a + 1 < keyframes.count, !keyframes[a + 1].cut else { return key.grid }
        let next = keyframes[a + 1]
        let t = Float(index - key.index) / Float(next.index - key.index)
        return key.grid.eased(to: next.grid, t)
    }

    /// A frame's luma on a sparse 48×48 grid, 0…255: what tells a cut, as the
    /// depth pipeline's held-depth policy does (`DepthRefinement.cutMeanLumaDelta`).
    public static func signature(_ plane: AnalysisPlane) -> [Float] {
        let n = 48
        var out = [Float](repeating: 0, count: n * n)
        for j in 0 ..< n {
            for i in 0 ..< n {
                out[j * n + i] = plane.sample((Float(i) + 0.5) * Float(plane.width) / Float(n) - 0.5,
                                              (Float(j) + 0.5) * Float(plane.height) / Float(n) - 0.5)
            }
        }
        return out
    }

    /// A cut lies between the frames these signatures came from.
    public static func isCut(_ previous: [Float], _ current: [Float]) -> Bool {
        guard previous.count == current.count, !current.isEmpty else { return false }
        let mean = zip(previous, current).reduce(Float(0)) { $0 + abs($1.0 - $1.1) } / Float(current.count)
        return mean > DepthRefinement.cutMeanLumaDelta * 255
    }
}
