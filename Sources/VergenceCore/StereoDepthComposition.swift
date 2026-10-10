/*
 Vergence - how a frame's depth is spread, and the strength that would suit it

 Measurement only, for now: nothing here changes the picture yet.

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

    public init(mean: Float, p05: Float, p20: Float, p50: Float, p80: Float, p95: Float, farShare: Float) {
        self.mean = mean
        self.p05 = p05
        self.p20 = p20
        self.p50 = p50
        self.p80 = p80
        self.p95 = p95
        self.farShare = farShare
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
    private var sums = [Double](repeating: 0, count: 7)

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
        for (i, v) in [c.mean, c.p05, c.p20, c.p50, c.p80, c.p95, c.farShare].enumerated() {
            sums[i] += Double(v)
        }
        return finished
    }

    public mutating func reset() {
        frames = 0
        sums = [Double](repeating: 0, count: 7)
    }

    private func shot() -> Shot {
        let n = Double(max(frames, 1))
        let a = sums.map { Float($0 / n) }
        let composition = DepthComposition(mean: a[0], p05: a[1], p20: a[2], p50: a[3], p80: a[4], p95: a[5], farShare: a[6])
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
    public static func record(_ composition: DepthComposition, current: DepthShotTracker.Shot?, finished: DepthShotTracker.Shot?) {
        state.withLock {
            $0.frame = composition
            $0.frameProposedScale = StereoStrengthPolicy.proposedScale(for: composition)
            $0.currentShot = current
            if let finished {
                $0.recentShots.append(finished)
                if $0.recentShots.count > keptShots { $0.recentShots.removeFirst($0.recentShots.count - keptShots) }
            }
        }
    }
}
