/*
 Vergence - how a frame's depth is spread, and the strength that would suit it

 The composition and strength policy are measurement only; nothing there
 changes the picture. `DepthGapCompressor` (at the end) does.

 A fixed strength (a share of the picture's width) can't suit every shot.
 Depth is relative, so every shot is spread over the whole budget: the nearest
 content on the glass, the farthest at the full shift. On the headset
 (2026-10-10) a landscape with a near subject (Nope's horse scene) fused
 only at 0.35× the shipped strength, while an interior (House) read flat
 at 0.35× and right at 1×. In the landscape nearly every pixel sits at the far
 end, so nearly the whole picture carries the maximum shift.

 `DepthComposition` says how the warp's 0…1 depth (near = 1) is spread over a
 frame's pixels, from the realtime refiner's histogram. `StereoStrengthPolicy`
 turns it into the share of the strength it would use: the mean shift across
 the picture is held under a target, so a shot that is mostly far is
 attenuated and one with depth spread through it keeps it all.
 `DepthShotTracker` averages both over a shot (between cuts), which is the
 unit the policy will act on: a strength that moved within a shot would
 visibly breathe.

 The target is a guess to be fitted from those device findings; the
 debug server's per-shot readout exists to fit it.
 */

import Foundation

/// The spread of one frame's warp depth (0 far … 1 near) over its pixels.
public struct DepthComposition: Sendable, Equatable, Codable {
    public var mean: Float
    public var p05: Float
    public var p20: Float
    public var p50: Float
    public var p80: Float
    public var p95: Float
    /// Share of pixels in the far quarter (depth < 0.25): with convergence 1
    /// they carry more than three quarters of the full shift.
    public var farShare: Float
    /// Share of depth texels on a step larger than a quarter, a half and three
    /// quarters of the shown range, against a neighbour a few texels away (the
    /// GPU's `depthStepCensus`; zero when not measured). A near subject against
    /// a far background puts its whole outline on a large step.
    public var stepShare25: Float = 0
    public var stepShare50: Float = 0
    public var stepShare75: Float = 0

    public init(mean: Float, p05: Float, p20: Float, p50: Float, p80: Float, p95: Float, farShare: Float,
                stepShare25: Float = 0, stepShare50: Float = 0, stepShare75: Float = 0) {
        self.mean = mean
        self.p05 = p05
        self.p20 = p20
        self.p50 = p50
        self.p80 = p80
        self.p95 = p95
        self.farShare = farShare
        self.stepShare25 = stepShare25
        self.stepShare50 = stepShare50
        self.stepShare75 = stepShare75
    }

    /// Mean shift over the picture, as a share of the full strength, for a
    /// convergence (1 = nearest on the glass). Pixels nearer than the
    /// convergence plane cross in front and count by their distance too.
    public func meanShift(convergence: Float = 1) -> Float { abs(convergence - mean) }

    /// Depth spread across the bulk of the picture.
    public var spread: Float { p95 - p05 }

    /// From a histogram of `bins` over `[lo, hi]` in the depth map's own units,
    /// shown through `mapping` (shot space) and `budget` (the warp's 0…1).
    /// Nil for an empty histogram.
    public static func measure(
        bins: [UInt32], lo: Float, hi: Float, mapping: DepthAffine, budget: DepthBudget
    ) -> DepthComposition? {
        let total = bins.reduce(0) { $0 + Int($1) }
        guard total > 0, lo.isFinite, hi.isFinite, hi >= lo else { return nil }
        func shown(_ x: Float) -> Float { min(1, max(0, budget.apply(mapping(x)))) }
        // Mapping and budget are monotone, so a quantile of the shown depth is
        // the shown value of the map's quantile.
        func quantile(_ q: Float) -> Float {
            shown(DepthHistogram.percentile(bins, total: total, q: q, lo: lo, hi: hi))
        }
        let width = (hi - lo) / Float(DepthHistogram.binCount)
        var sum: Double = 0
        var far = 0
        for (i, count) in bins.enumerated() where count > 0 {
            let d = shown(lo + (Float(i) + 0.5) * width)
            sum += Double(d) * Double(count)
            if d < 0.25 { far += Int(count) }
        }
        return DepthComposition(
            mean: Float(sum / Double(total)),
            p05: quantile(0.05), p20: quantile(0.2), p50: quantile(0.5),
            p80: quantile(0.8), p95: quantile(0.95),
            farShare: Float(far) / Float(total)
        )
    }
}

/// The share of the strength a shot would get. Proposed only: logged and
/// reported, not yet applied.
public enum StereoStrengthPolicy {
    /// Mean shift across the picture, as a share of the full strength, that a
    /// shot may use. Uncalibrated: Nope's horse scene wanted about 0.35× and its
    /// mean shift is near 0.9, which puts this near 0.3.
    public static let targetMeanShift: Float = 0.3
    /// Never flatter than this: a shot that is all background still keeps
    /// some roundness.
    public static let minimumScale: Float = 0.25

    public static func proposedScale(for composition: DepthComposition, convergence: Float = 1) -> Float {
        let shift = composition.meanShift(convergence: convergence)
        guard shift > 1e-3 else { return 1 }
        return min(1, max(minimumScale, targetMeanShift / shift))
    }
}

/// Averages per-frame compositions over a shot.
public struct DepthShotTracker: Sendable {
    public struct Shot: Sendable, Equatable, Codable {
        /// Media time of the shot's first and last measured frames.
        public var start: Double
        public var end: Double
        public var frames: Int
        /// The shot's frames averaged field by field.
        public var composition: DepthComposition
        public var proposedScale: Float
    }

    private var start = 0.0
    private var end = 0.0
    private var frames = 0
    private var sums = [Double](repeating: 0, count: 10)

    public init() {}

    /// The shot so far, nil before its first frame.
    public var current: Shot? { frames > 0 ? shot() : nil }

    /// Adds a frame. A cut (or a seek) closes the shot so far, which is
    /// returned, and starts a new one with this frame.
    public mutating func add(_ composition: DepthComposition, seconds: Double, cut: Bool) -> Shot? {
        var finished: Shot?
        if cut, frames > 0 {
            finished = shot()
            reset()
        }
        if frames == 0 { start = seconds }
        end = seconds
        frames += 1
        let c = composition
        for (i, v) in [c.mean, c.p05, c.p20, c.p50, c.p80, c.p95, c.farShare, c.stepShare25, c.stepShare50, c.stepShare75].enumerated() {
            sums[i] += Double(v)
        }
        return finished
    }

    public mutating func reset() {
        frames = 0
        sums = [Double](repeating: 0, count: 10)
    }

    private func shot() -> Shot {
        let n = Double(max(frames, 1))
        let a = sums.map { Float($0 / n) }
        let composition = DepthComposition(mean: a[0], p05: a[1], p20: a[2], p50: a[3], p80: a[4], p95: a[5], farShare: a[6],
                                           stepShare25: a[7], stepShare50: a[8], stepShare75: a[9])
        return Shot(start: start, end: end, frames: frames, composition: composition,
                    proposedScale: StereoStrengthPolicy.proposedScale(for: composition))
    }
}

/// What the realtime depth path measured lately, for the host's debug readout.
/// Written on the pump queue, read anywhere.
public enum StereoDepthMetrics {
    public struct Snapshot: Sendable, Codable {
        public var frame: DepthComposition?
        public var frameProposedScale: Float?
        public var currentShot: DepthShotTracker.Shot?
        /// How much of the depth range the shot's gap curve left (1 = none
        /// squeezed; `DepthGapCompressor`).
        public var gapSpan: Float?
        /// Finished shots, newest last.
        public var recentShots: [DepthShotTracker.Shot] = []
    }

    static let keptShots = 40
    private static let state = VergenceLock(Snapshot())

    public static var snapshot: Snapshot { state.withLock { $0 } }

    public static func reset() {
        state.withLock { $0 = Snapshot() }
    }

    /// Records a frame's composition and the tracker's state after it.
    public static func record(_ composition: DepthComposition, current: DepthShotTracker.Shot?, finished: DepthShotTracker.Shot?,
                              gapSpan: Float? = nil) {
        state.withLock {
            $0.frame = composition
            $0.gapSpan = gapSpan
            $0.frameProposedScale = StereoStrengthPolicy.proposedScale(for: composition)
            $0.currentShot = current
            if let finished {
                $0.recentShots.append(finished)
                if $0.recentShots.count > keptShots { $0.recentShots.removeFirst($0.recentShots.count - keptShots) }
            }
        }
    }
}

/// Squeezes the depth no pixel occupies out of a shot.
///
/// A near subject before a far background (Nope's horse scene, Alpha's
/// cliffs) spends the whole depth budget on the empty gap between them, so
/// the step at the subject's outline is the full shift. That step is the
/// strip each eye uncovers beside the subject, which no warp can fill with
/// what was really behind it: on the headset it read as smeared shirt and
/// stretched hills (2026-10-11). A conversion artist compresses the space
/// between layers and keeps each layer's own roundness; this does the same
/// from the shot's depth histogram. A depth range that holds hardly any
/// pixels gets a slope of `floorSlope`, an occupied one keeps slope 1, and
/// the near end stays where it was (on the glass with convergence 1), so the
/// background comes forward by what the gap gave up. A room whose depth runs
/// continuously through the picture (House) has no empty range and is left
/// as it was, which a single strength can't do for both.
///
/// The tails are squeezed too: in Nope's wide shot by the corral, a thin
/// rail near the camera was the only thing near and 97 % of the picture sat
/// at the far end, so everything but the rail carried the full shift and the
/// fence posts between doubled (headset 2026-10-11). Squeezed, the rail stays
/// on the glass and the scene comes forward to meet it. The density is averaged over the shot
/// (an EMA restarted at a cut), so the curve settles within about a second
/// and then holds instead of breathing frame to frame.
public struct DepthGapCompressor: Sendable {
    /// Ranges of shot-space depth (0 far … 1 near) the curve is built over.
    public static let binCount = 32
    public struct Tuning: Sendable, Equatable {
        /// Slope given to an empty range: the gap keeps this share of its
        /// depth, so what's behind a subject still reads as behind it. Nope
        /// fused at about 0.35× strength, nearly all of it gap.
        public var floorSlope: Float = 0.25
        /// A range holding at most `emptyDensity` of an even spread counts as
        /// empty, and at least `fullDensity` as occupied; between, the slope
        /// eases. In Nope's horse scene the gap still held 7 % of the pixels
        /// (the horse's body, outline ramps), about 0.13 of an even spread, so
        /// "empty" has to reach past that.
        public var emptyDensity: Float = 0.2
        public var fullDensity: Float = 0.6

        public init() {}
    }

    /// The tuning every compressor uses; the host's debug server sets it live.
    nonisolated(unsafe) public static var tuning = Tuning()
    /// How fast the density follows the shot, seconds.
    public var timeConstant: Double = 0.8

    private var density: [Float]?

    public init() {}

    public mutating func reset() { density = nil }

    /// The curve for this frame. `bins` is the refiner's histogram over
    /// `[lo, hi]` in the depth map's own units, shown through `mapping`.
    public mutating func update(
        bins: [UInt32], lo: Float, hi: Float, mapping: DepthAffine, dt: Double, cut: Bool
    ) -> DepthGapCurve {
        let n = Self.binCount
        let total = bins.reduce(0) { $0 + Int($1) }
        guard total > 0, lo.isFinite, hi.isFinite, hi > lo else { return density.map(curve) ?? .identity }
        var frame = [Float](repeating: 0, count: n)
        let width = (hi - lo) / Float(bins.count)
        for (i, count) in bins.enumerated() where count > 0 {
            let y = mapping(lo + (Float(i) + 0.5) * width)
            guard y.isFinite else { continue }
            let k = min(n - 1, max(0, Int(y * Float(n))))
            frame[k] += Float(count)
        }
        // Relative to an even spread: 1 = this range holds 1/n of the picture.
        let even = Float(total) / Float(n)
        frame = frame.map { $0 / even }
        if cut || density == nil {
            density = frame
        } else if var d = density {
            let alpha = Float(1 - exp(-max(dt, 0) / timeConstant))
            for k in 0..<n { d[k] += (frame[k] - d[k]) * alpha }
            density = d
        }
        return curve(density ?? frame)
    }

    func curve(_ density: [Float]) -> DepthGapCurve {
        let n = Self.binCount
        // Spread each range's pixels over its neighbours a little, so a
        // surface's own depth isn't cut into by a quiet range inside it.
        var smooth = [Float](repeating: 0, count: n)
        for k in 0..<n {
            let a = density[max(k - 1, 0)], b = density[k], c = density[min(k + 1, n - 1)]
            smooth[k] = (a + 2 * b + c) / 4
        }
        let tuning = Self.tuning
        let fullDensity = max(tuning.fullDensity, tuning.emptyDensity + 1e-3)
        guard smooth.contains(where: { $0 >= fullDensity }) else { return .identity }
        var slopes = [Float](repeating: 1, count: n)
        for k in 0..<n {
            let t = min(1, max(0, (smooth[k] - tuning.emptyDensity) / (fullDensity - tuning.emptyDensity)))
            let eased = t * t * (3 - 2 * t)
            let floor = min(1, max(0, tuning.floorSlope))
            slopes[k] = floor + (1 - floor) * eased
        }
        // Knots at the range edges, the near end (1) fixed.
        var knots = [Float](repeating: 1, count: n + 1)
        for k in stride(from: n - 1, through: 0, by: -1) {
            knots[k] = knots[k + 1] - slopes[k] / Float(n)
        }
        return DepthGapCurve(knots: knots)
    }
}

/// A monotone remap of shot-space depth: piecewise linear through `knots`
/// at 0, 1/n, … 1, slope 1 outside. Mirrored by `depthGapCurve` in the shaders.
public struct DepthGapCurve: Sendable, Equatable {
    public var knots: [Float]

    public static let identity = DepthGapCurve(
        knots: (0...DepthGapCompressor.binCount).map { Float($0) / Float(DepthGapCompressor.binCount) }
    )

    public init(knots: [Float]) { self.knots = knots }

    /// How much of the 0…1 range is left (1 = untouched).
    public var span: Float { (knots.last ?? 1) - (knots.first ?? 0) }

    public func apply(_ y: Float) -> Float {
        let n = knots.count - 1
        guard n > 0, y.isFinite else { return y }
        if y <= 0 { return knots[0] + y }
        if y >= 1 { return knots[n] + (y - 1) }
        let x = y * Float(n)
        let k = min(n - 1, Int(x))
        return knots[k] + (knots[k + 1] - knots[k]) * (x - Float(k))
    }
}
