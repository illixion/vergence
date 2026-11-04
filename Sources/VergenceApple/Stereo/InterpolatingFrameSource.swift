#if canImport(Metal) && canImport(VideoToolbox)
import CoreMedia
import CoreVideo
import Foundation
import QuartzCore
import VergenceCore

/// Makes the frame halfway between two decoded frames. Called on the owning
/// `InterpolatingFrameSource`'s work queue, one request at a time.
public protocol MidFrameInterpolator: AnyObject, Sendable {
    /// Get ready for frames of this size, interpolated at `scale` of it
    /// (1: full size; below 1 the in-between frame is made smaller and scaled
    /// back up, for less work): true when they can be interpolated. May
    /// block (a model load: ~60 ms warm, seconds cold).
    func prepare(width: Int, height: Int, scale: Double) -> Bool
    /// The BGRA frame halfway between `previous` and `current` (both BGRA, of
    /// the prepared size), handed to `completion` on any queue; nil on failure.
    /// `processMs` is the interpolation itself, without the format conversions.
    func midFrame(between previous: CVPixelBuffer, and current: CVPixelBuffer,
                  completion: @escaping @Sendable (_ frame: CVPixelBuffer?, _ processMs: Double) -> Void)
    /// The same, for an interpolator that needs to know which frames these
    /// are (one reading fields computed ahead, by presentation time in
    /// seconds). Defaults to the untimed call.
    func midFrame(between previous: CVPixelBuffer, at previousTime: Double, and current: CVPixelBuffer,
                  at currentTime: Double,
                  completion: @escaping @Sendable (_ frame: CVPixelBuffer?, _ processMs: Double) -> Void)
}

extension MidFrameInterpolator {
    public func midFrame(between previous: CVPixelBuffer, at previousTime: Double, and current: CVPixelBuffer,
                         at currentTime: Double,
                         completion: @escaping @Sendable (_ frame: CVPixelBuffer?, _ processMs: Double) -> Void) {
        midFrame(between: previous, and: current, completion: completion)
    }
}

/// Doubles a low-frame-rate source (24, 25, 30 fps) for the stereo pump by
/// putting an interpolated frame between each two decoded ones, so slow pans
/// move in half the steps. It wraps the source *before* depth and warp: both
/// eyes are made from the one interpolated picture, so they can't disagree the
/// way two separately interpolated eyes could.
///
/// What it never interpolates across: a cut (the held-depth policy's luma
/// grid), a seek or a dropped frame (a gap unlike the source's spacing), a
/// speed above 1.1× (the frames are already coming faster), or a source above
/// every tier's `maxSourceFrameRate`. Those frames pass through as they came.
///
/// **Tiers** pick the interpolator by the source's rate: the first whose
/// `maxSourceFrameRate` the source doesn't exceed. A flat player doubles 24-30
/// fps with VideoToolbox (the better guesser) and 50-60 fps with
/// `FlowMidFrameInterpolator` (the one fast enough for an in-between frame
/// every 16.7 ms); see `Engine`.
///
/// **It reads one real frame ahead.** The pump asks for the frame 1.5 of its
/// spacings past the last one it took; once in-between frames halve that
/// spacing, the next real frame always lies past the ask, and an ask answered
/// only "at or before" would never reach it. So once the source is eligible,
/// the inner source is asked for the next real frame (halfway to the one after
/// it), and a frame up to one real spacing past the ask is handed out.
///
/// The interpolation runs off the pump queue; until it finishes, asks answer
/// nil and the pump tries again on its next tick.
///
/// **It steps down before the picture falls behind** (`InterpolationGovernor`).
/// The interpolator and the depth model share the Neural Engine: on a Vision
/// Pro a map took ~28 ms alone and ~38-45 ms beside it, and once a warm device
/// made both slower every frame came late and the picture fell behind the
/// sound (device, 2026-10-31). So it measures itself: interpolation that takes
/// most of a real frame's time, or a reserve running low, first makes the
/// in-between frames at a smaller size, then pauses them; headroom steps it
/// back up. It reacts to the slowdown, not to the thermal state, which a
/// headset reported as serious the moment a 2D video opened (2026-11-01).
public final class InterpolatingFrameSource: LookaheadFrameSource, @unchecked Sendable {
    /// VideoToolbox's tier: sources above this rate are too fast for it (an
    /// in-between frame every 16.7 ms at 60 fps, where it takes 5-18) and
    /// pass through unless a faster tier takes them.
    public static let maxSourceFrameRate = 40.0
    /// The GPU flow tier: 50 and 60 fps (59.94 included), doubled to 100 and
    /// 120 for a display that has them.
    public static let maxFastSourceFrameRate = 61.0

    /// An interpolator and the fastest source it takes.
    public struct Tier: Sendable {
        public let interpolator: MidFrameInterpolator
        public let maxSourceFrameRate: Double
        public init(_ interpolator: MidFrameInterpolator, upTo maxSourceFrameRate: Double) {
            self.interpolator = interpolator
            self.maxSourceFrameRate = maxSourceFrameRate
        }
    }

    /// Which interpolator makes the in-between frames.
    public enum Engine: Sendable {
        /// VideoToolbox up to 40 fps; faster sources pass through (AI 3D,
        /// whose depth model couldn't keep up with 120 eyes a second anyway).
        case videoToolbox
        /// VideoToolbox up to 40 fps and the GPU flow interpolator for 50-60 fps.
        case automatic
        /// The GPU flow interpolator for every source up to 60 fps (to compare
        /// it with VideoToolbox on 24-30 fps).
        case gpu
    }
    /// Interpolations whose median the governor judges speed by.
    static let timingWindow = 12
    static let maxPlaybackRate = 1.1

    private let inner: LookaheadFrameSource
    /// Slowest first.
    private let tiers: [Tier]
    private let work = DispatchQueue(label: "vergence.interpolation", qos: .userInteractive)
    private let lock = NSLock()

    // Guarded by `lock`.
    private var ready: [PumpFrame] = []
    private var inFlight = false
    private var generation = 0
    private var previous: PumpFrame?
    private var previousSignature: [Float]?
    private var realSpacing = SourceFrameSpacing()
    private var lastSequence: Int
    private var preparedSize: (width: Int, height: Int)?
    private var preparedOK = false
    private var preparedScale = 1.0
    private var preparedFor: ObjectIdentifier?
    /// Recent interpolation times with conversions, at the current scale.
    private var recentMs: [Double] = []
    private var preparing = false
    private var stats = Stats()
    private var governor: InterpolationGovernor

    public init(wrapping inner: LookaheadFrameSource, tiers: [Tier],
                thresholds: InterpolationGovernor.Thresholds = .depthPump) {
        precondition(!tiers.isEmpty, "an interpolating source needs an interpolator")
        self.inner = inner
        self.tiers = tiers.sorted { $0.maxSourceFrameRate < $1.maxSourceFrameRate }
        self.lastSequence = inner.sequence
        self.governor = InterpolationGovernor(thresholds: thresholds)
    }

    /// One interpolator for sources up to `maxSourceFrameRate`.
    public convenience init(wrapping inner: LookaheadFrameSource, interpolator: MidFrameInterpolator,
                            maxSourceFrameRate: Double = InterpolatingFrameSource.maxSourceFrameRate,
                            thresholds: InterpolationGovernor.Thresholds = .depthPump) {
        self.init(wrapping: inner, tiers: [Tier(interpolator, upTo: maxSourceFrameRate)], thresholds: thresholds)
    }

    /// The fastest source any tier interpolates.
    public var highestSourceFrameRate: Double { tiers.last?.maxSourceFrameRate ?? 0 }

    /// `inner` wrapped with the interpolators `engine` names that exist here,
    /// or `inner` itself where none does. VideoToolbox's exists only on a
    /// device or a Mac (not the simulator); the GPU one wherever Metal does.
    public static func wrappingIfSupported(_ inner: LookaheadFrameSource,
                                           thresholds: InterpolationGovernor.Thresholds = .depthPump,
                                           engine: Engine = .videoToolbox) -> LookaheadFrameSource {
        var tiers: [Tier] = []
        if engine != .gpu, let vt = makeVideoToolboxInterpolator() {
            tiers.append(Tier(vt, upTo: maxSourceFrameRate))
        }
        if engine != .videoToolbox {
            if let flow = FlowMidFrameInterpolator() {
                tiers.append(Tier(flow, upTo: maxFastSourceFrameRate))
            } else {
                VergenceLog.pipeline.error("frame interpolation: GPU interpolator unavailable (no Metal, or its kernels failed)")
            }
        }
        guard !tiers.isEmpty else {
            VergenceLog.pipeline.notice("frame interpolation: not supported here, playing source frames")
            return inner
        }
        return InterpolatingFrameSource(wrapping: inner, tiers: tiers, thresholds: thresholds)
    }

    private static func makeVideoToolboxInterpolator() -> MidFrameInterpolator? {
        #if targetEnvironment(simulator)
        return nil
        #else
        if #available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *), VTMidFrameInterpolator.isSupported {
            return VTMidFrameInterpolator()
        }
        return nil
        #endif
    }

    // MARK: LookaheadFrameSource, forwarded

    public var isFinished: Bool { inner.isFinished }
    public var endTime: Double? { inner.endTime }
    public var sequence: Int { inner.sequence }
    public var decodesAheadWhilePaused: Bool { inner.decodesAheadWhilePaused }
    public func itemTime(forHostTime hostTime: CFTimeInterval) -> CMTime { inner.itemTime(forHostTime: hostTime) }
    public func dueHostTime(forItemTime itemTime: CMTime, at hostTime: CFTimeInterval) -> CFTimeInterval? {
        inner.dueHostTime(forItemTime: itemTime, at: hostTime)
    }
    public func playbackRate(hostTime: CFTimeInterval) -> Double? { inner.playbackRate(hostTime: hostTime) }

    public func nextFrame(hostTime: CFTimeInterval) -> PumpFrame? {
        frame(atItemTime: inner.itemTime(forHostTime: hostTime))
    }

    public func invalidate() {
        lock.withLock { resetLocked() }
        inner.invalidate()
    }

    /// Forget the frames held and in flight (a seek the clock can't show,
    /// such as a loop back to the same spot). The source stays open.
    public func reset() {
        lock.withLock { resetLocked() }
    }

    // MARK: Frames

    public func frame(atItemTime itemTime: CMTime) -> PumpFrame? {
        let t = itemTime.seconds
        guard t.isFinite else { return inner.frame(atItemTime: itemTime) }
        lock.lock()
        defer { lock.unlock() }
        logStatsIfDue()

        if inner.sequence != lastSequence {
            lastSequence = inner.sequence
            resetLocked()
        }
        let slack = eligibleSpacing ?? 0
        if let first = ready.first {
            let firstSeconds = first.itemTime.seconds
            if firstSeconds <= t + slack {
                // Behind the ask by more than a jump: a seek forward.
                if firstSeconds >= t - LookaheadPull.maxStep {
                    ready.removeFirst()
                    return first
                }
                resetLocked()
            } else if firstSeconds > t + slack + LookaheadPull.maxStep {
                resetLocked() // a seek back
            } else {
                return nil
            }
        }
        if inFlight { return nil }

        var innerTarget = t
        if let spacing = eligibleSpacing, let previous {
            innerTarget = max(t, previous.itemTime.seconds + 1.5 * spacing)
        }
        guard let real = inner.frame(atItemTime: CMTime(seconds: innerTarget, preferredTimescale: 600_000)) else {
            return nil
        }
        let seconds = real.itemTime.seconds
        let signature = LumaSignature.sample(real.pixelBuffer)
        let before = previous
        let beforeSignature = previousSignature
        if let before, seconds.isFinite { realSpacing.note(gap: seconds - before.itemTime.seconds) }
        previous = real
        previousSignature = signature

        guard let before else {
            stats.passed += 1
            return real
        }
        let now = CACurrentMediaTime()
        // Only on a clock that says how fast it runs: nil while paused.
        var reserve: Double?
        if let rate = inner.playbackRate(hostTime: now), rate > 0.01 {
            let playhead = inner.itemTime(forHostTime: now).seconds
            if playhead.isFinite, seconds.isFinite { reserve = (seconds - playhead) / rate }
        }
        if let reserve { stats.minReserve = min(stats.minReserve, reserve) }
        let medianMs = recentMs.count >= Self.timingWindow / 2 ? recentMs.sorted()[recentMs.count / 2] : nil
        let budgetMs = eligibleSpacing.map { $0 * 1000 }
        let scaleBefore = governor.scale
        if let change = governor.update(reserve: reserve, medianMs: medianMs, budgetMs: budgetMs, now: now) {
            VergenceLog.pipeline.notice("frame interpolation \(change, privacy: .public)")
        }
        if governor.scale != scaleBefore { recentMs.removeAll() }
        if let reason = governor.isPaused ? .backedOff
            : skipReason(before: before, beforeSignature: beforeSignature, real: real, signature: signature) {
            stats.skipped[reason, default: 0] += 1
            stats.passed += 1
            return reason.leavesOutMidFrame ? real.standingIn : real
        }
        guard let interpolator = tier?.interpolator else {
            stats.passed += 1
            return real
        }
        inFlight = true
        let requestGeneration = generation
        let midSeconds = (before.itemTime.seconds + seconds) / 2
        let midTime = CMTime(seconds: midSeconds, preferredTimescale: 600_000)
        let submitted = CACurrentMediaTime()
        let beforeSeconds = before.itemTime.seconds
        work.async { [weak self] in
            interpolator.midFrame(between: before.pixelBuffer, at: beforeSeconds, and: real.pixelBuffer,
                                  at: seconds) { [weak self] mid, processMs in
                guard let self else { return }
                let totalMs = (CACurrentMediaTime() - submitted) * 1000
                self.lock.withLock {
                    guard requestGeneration == self.generation else { return }
                    self.inFlight = false
                    if let mid {
                        self.ready = [PumpFrame(pixelBuffer: mid, itemTime: midTime), real]
                        self.stats.noteInterpolated(processMs: processMs, totalMs: totalMs)
                        self.recentMs.append(totalMs)
                        if self.recentMs.count > Self.timingWindow { self.recentMs.removeFirst() }
                    } else {
                        self.ready = [real.standingIn]
                        self.stats.failed += 1
                    }
                }
            }
        }
        return nil
    }

    /// The source's spacing when some tier is slow enough to interpolate it.
    private var eligibleSpacing: Double? {
        let spacing = realSpacing.seconds
        guard spacing.isFinite, spacing >= 1 / highestSourceFrameRate else { return nil }
        return spacing
    }

    /// The tier for the source's rate, nil while it's unknown or too fast.
    private var tier: Tier? {
        guard let spacing = eligibleSpacing else { return nil }
        return tiers.first { spacing >= 1 / $0.maxSourceFrameRate }
    }

    enum SkipReason: String {
        case learning = "learning rate", notReady = "not ready", fast = "fast source", gap, cut, speed, size,
             backedOff = "backed off"

        /// A pair that would have had an in-between frame. A source whose
        /// rate isn't known yet, or a fast one, has none to leave out, and a
        /// gap is frames lost before here.
        var leavesOutMidFrame: Bool { self != .learning && self != .fast && self != .gap }
    }

    private func skipReason(before: PumpFrame, beforeSignature: [Float]?, real: PumpFrame,
                            signature: [Float]?) -> SkipReason? {
        guard let spacing = eligibleSpacing else {
            return realSpacing.seconds.isFinite ? .fast : .learning
        }
        let gap = real.itemTime.seconds - before.itemTime.seconds
        guard gap > 0, gap < 1.5 * spacing else { return .gap }
        guard let signature, let beforeSignature,
              let delta = LumaSignature.meanDelta(signature, beforeSignature),
              delta < HeldDepthPolicy.maxLumaDelta else { return .cut }
        if let rate = inner.playbackRate(hostTime: CACurrentMediaTime()), rate > Self.maxPlaybackRate { return .speed }
        let width = CVPixelBufferGetWidth(real.pixelBuffer)
        let height = CVPixelBufferGetHeight(real.pixelBuffer)
        guard CVPixelBufferGetWidth(before.pixelBuffer) == width,
              CVPixelBufferGetHeight(before.pixelBuffer) == height else { return .size }
        let scale = governor.scale
        guard let interpolator = tier?.interpolator else { return .fast }
        let id = ObjectIdentifier(interpolator)
        if let size = preparedSize, size.width == width, size.height == height, preparedScale == scale,
           preparedFor == id {
            return preparedOK ? nil : .size
        }
        if !preparing {
            preparing = true
            preparedSize = nil
            work.async { [weak self] in
                let started = CACurrentMediaTime()
                let ok = interpolator.prepare(width: width, height: height, scale: scale)
                let ms = (CACurrentMediaTime() - started) * 1000
                VergenceLog.pipeline.notice(
                    "frame interpolation: \(String(describing: type(of: interpolator)), privacy: .public) \(width)×\(height) at \(scale * 100, format: .fixed(precision: 0), privacy: .public)% \(ok ? "ready" : "unavailable", privacy: .public) in \(ms, format: .fixed(precision: 0), privacy: .public) ms"
                )
                guard let self else { return }
                self.lock.withLock {
                    self.preparing = false
                    self.preparedSize = (width, height)
                    self.preparedScale = scale
                    self.preparedFor = id
                    self.preparedOK = ok
                }
            }
        }
        return .notReady
    }

    private func resetLocked() {
        ready.removeAll()
        inFlight = false
        generation += 1
        previous = nil
        previousSignature = nil
        recentMs.removeAll()
        governor.noteDiscontinuity()
    }

    // MARK: Measurement

    struct Stats {
        var interpolated = 0
        var passed = 0
        var failed = 0
        var skipped: [SkipReason: Int] = [:]
        var processMs: [Double] = []
        var totalMs: [Double] = []
        var windowStart = CACurrentMediaTime()
        var minReserve = Double.infinity

        mutating func noteInterpolated(processMs: Double, totalMs: Double) {
            interpolated += 1
            self.processMs.append(processMs)
            self.totalMs.append(totalMs)
        }
    }

    private func logStatsIfDue() {
        let now = CACurrentMediaTime()
        let window = now - stats.windowStart
        guard window >= 5 else { return }
        defer { stats = Stats(windowStart: now) }
        guard stats.interpolated + stats.passed + stats.failed > 0 else { return }
        func median(_ values: [Double]) -> Double { values.isEmpty ? .nan : values.sorted()[values.count / 2] }
        let skipped = stats.skipped.sorted { $0.key.rawValue < $1.key.rawValue }
            .map { "\($0.key.rawValue) \($0.value)" }.joined(separator: ", ")
        let spacing = realSpacing.seconds
        let sourceFps = spacing.isFinite ? 1 / spacing : 0
        VergenceLog.pipeline.info(
            "interp \(sourceFps, format: .fixed(precision: 2), privacy: .public) fps source, reserve ≥ \(self.stats.minReserve * 1000, format: .fixed(precision: 0), privacy: .public) ms, thermal \(ProcessInfo.processInfo.thermalState.rawValue, privacy: .public), \(self.governor.state, privacy: .public): \(self.stats.interpolated, privacy: .public) made (\(Double(self.stats.interpolated) / window, format: .fixed(precision: 1), privacy: .public)/s), \(self.stats.passed, privacy: .public) passed [\(skipped, privacy: .public)], \(self.stats.failed, privacy: .public) failed; process median \(median(self.stats.processMs), format: .fixed(precision: 1), privacy: .public) max \(self.stats.processMs.max() ?? .nan, format: .fixed(precision: 1), privacy: .public) ms, with conversions median \(median(self.stats.totalMs), format: .fixed(precision: 1), privacy: .public) max \(self.stats.totalMs.max() ?? .nan, format: .fixed(precision: 1), privacy: .public) ms"
        )
    }
}

/// How much interpolation runs. Interpolating while the depth model can't
/// keep up costs more than it gives: every frame comes late and the picture
/// drifts behind the sound. So, the way AI 3D steps its picture down when
/// frames stop reaching the eyes, it judges by what it measures:
/// - **Too slow** (the median of recent interpolations over `slowFraction`
///   of a real frame's spacing) or **a low reserve** (a real frame taken less
///   than `lowReserve` ahead of the playhead): one step down, from full size
///   to `scales`' smaller sizes, then to paused. One step per `cooldown`, so
///   each step's effect shows before the next; frames already late pause it
///   at once.
/// - **Headroom** (median under `fastFraction` of the spacing, the reserve
///   fine) for `stepUpAfter`: one step back up.
/// - **Paused:** resumes at the smallest size once the reserve is back above
///   `resumeReserve` and the cooldown has passed. A third pause within a
///   minute keeps it off until the source is rebuilt.
/// The thermal state is not an input: a warm device that still keeps up
/// should keep interpolating, and one that can't shows it in the timings.
public struct InterpolationGovernor: Sendable {
    /// Seconds a real frame is taken ahead of the playhead, in wall time,
    /// below which interpolation steps down and above which it may resume.
    public struct Thresholds: Sendable {
        public var lowReserve: Double
        public var resumeReserve: Double
        public init(lowReserve: Double, resumeReserve: Double) {
            self.lowReserve = lowReserve
            self.resumeReserve = resumeReserve
        }
        /// The stereo pump, which takes frames well ahead for depth (about
        /// 0.46 s of 1080p at 48 fps fits its reserve).
        public static let depthPump = Thresholds(lowReserve: 0.2, resumeReserve: 0.45)
        /// A flat player asking `InterpolatedFramePresenter.lead` ahead.
        public static let flat = Thresholds(lowReserve: 0.04, resumeReserve: 0.1)
    }
    let thresholds: Thresholds

    /// Interpolation sizes, as a share of the picture's: full, then two
    /// thirds (1080p made at 720p: 5.6 ms instead of 16 on an M2 Max). The
    /// real frames keep their full size either way.
    static let scales: [Double] = [1, 2.0 / 3]
    static let slowFraction = 0.8
    static let fastFraction = 0.45
    static let stepUpAfter = 20.0
    static let cooldown = 8.0
    static let maxPauses = 3
    static let pauseWindow = 60.0
    /// After a start, a seek or a resume the reserve is empty because the
    /// pump is starting over, not because it is behind, so for this long the
    /// reserve doesn't count.
    static let grace = 3.0

    private(set) var level = 0
    private(set) var isPaused = false
    private(set) var isOff = false
    private var changedAt = -Double.infinity
    private var pauses: [Double] = []
    private var graceUntil: Double?
    private var wasRunning = false

    init(thresholds: Thresholds = .depthPump) { self.thresholds = thresholds }

    /// The size to interpolate at, as a share of the picture's.
    var scale: Double { Self.scales[level] }

    var state: String {
        if isOff { return "off" }
        if isPaused { return "paused" }
        return level == 0 ? "on" : String(format: "on at %.0f%%", scale * 100)
    }

    mutating func noteDiscontinuity() { graceUntil = nil }

    /// - Parameters:
    ///   - reserve: how far ahead the real frame was taken (nil: clock stopped or unknown).
    ///   - medianMs: recent interpolation time, nil until measured at this size.
    ///   - budgetMs: a real frame's spacing, nil while the source isn't eligible.
    /// - Returns: a description of a change, for the log.
    mutating func update(reserve: Double?, medianMs: Double?, budgetMs: Double?, now: Double) -> String? {
        if isOff { return nil }
        let running = reserve != nil
        if running, !wasRunning || graceUntil == nil { graceUntil = now + Self.grace }
        wasRunning = running
        let inGrace = graceUntil.map { now < $0 } ?? false
        let settled = now - changedAt >= Self.cooldown
        let reserveText = reserve.map { String(format: "reserve %.0f ms", $0 * 1000) } ?? "clock stopped"

        if isPaused {
            guard settled, !inGrace, let reserve, reserve >= thresholds.resumeReserve else { return nil }
            isPaused = false
            level = Self.scales.count - 1
            changedAt = now
            return "resumed at \(Int(scale * 100))% (\(reserveText))"
        }

        let late = !inGrace && (reserve.map { $0 < 0 } ?? false)
        let low = !inGrace && (reserve.map { $0 < thresholds.lowReserve } ?? false)
        var slow = false
        var fast = false
        var timing = ""
        if let medianMs, let budgetMs, budgetMs > 0 {
            slow = medianMs > Self.slowFraction * budgetMs
            fast = medianMs < Self.fastFraction * budgetMs
            timing = String(format: ", %.0f of %.0f ms", medianMs, budgetMs)
        }
        if late || ((low || slow) && settled) {
            let why = (late ? "frames late" : low ? "reserve low" : "too slow") + " (\(reserveText)\(timing))"
            changedAt = now
            if !late, level + 1 < Self.scales.count {
                level += 1
                return "stepped down to \(Int(scale * 100))%: \(why)"
            }
            isPaused = true
            pauses = pauses.filter { now - $0 < Self.pauseWindow } + [now]
            if pauses.count >= Self.maxPauses {
                isOff = true
                return "off for this picture: \(why), \(pauses.count) pauses in a minute"
            }
            return "paused: \(why)"
        }
        if level > 0, fast, !low, now - changedAt >= Self.stepUpAfter {
            level -= 1
            changedAt = now
            return "stepped up to \(Int(scale * 100))% (\(reserveText)\(timing))"
        }
        return nil
    }
}

#if !targetEnvironment(simulator)
import VideoToolbox

/// VideoToolbox's low-latency frame interpolation (the only interpolator
/// visionOS has; the higher-quality frame-rate conversion is macOS/iOS only).
/// It takes 4:2:0 frames (`420v`, measured on an M2 Max), so BGRA frames are
/// converted in and the result converted back. At most 1920 on a side and
/// 1920×1080 pixels (same Mac). All of it runs on the caller's work queue and
/// the processor's completion queue, one request at a time.
///
/// Its result goes through `InterpolationGuard`, which redoes the tiles where it
/// matched evenly spaced stripes to the wrong stripe (blinds, still or panning,
/// flickered at 48 Hz without it).
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *)
public final class VTMidFrameInterpolator: MidFrameInterpolator, @unchecked Sendable {
    public static var isSupported: Bool { VTLowLatencyFrameInterpolationConfiguration.isSupported }

    private var processor: VTFrameProcessor?
    private var width = 0
    private var height = 0
    private var toYUV: VTPixelTransferSession?
    private var toBGRA: VTPixelTransferSession?
    private var yuvPool: CVPixelBufferPool?
    private var midPool: CVPixelBufferPool?
    private var bgraPool: CVPixelBufferPool?
    /// The last `current` converted, reused as the next request's `previous`.
    private var lastConverted: (source: CVPixelBuffer, yuv: CVPixelBuffer)?
    /// nil when the guard is off or couldn't be made (then frames pass as they are).
    private let resultGuard: InterpolationGuard?

    /// - Parameter guardsResults: run `InterpolationGuard` on each result.
    public init(guardsResults: Bool = true) {
        resultGuard = guardsResults ? InterpolationGuard() : nil
    }

    /// GPU time of the guard's last pass (0 without the guard).
    public var guardGPUMilliseconds: Double { resultGuard?.lastGPUMilliseconds ?? 0 }

    deinit { processor?.endSession() }

    public func prepare(width: Int, height: Int, scale: Double) -> Bool {
        processor?.endSession()
        processor = nil
        lastConverted = nil
        // The size the model works at; the transfers scale in and out.
        let fullWidth = width
        let fullHeight = height
        let width = max(2, Int((Double(fullWidth) * scale / 2).rounded()) * 2)
        let height = max(2, Int((Double(fullHeight) * scale / 2).rounded()) * 2)
        if #available(macOS 27.0, iOS 27.0, tvOS 27.0, visionOS 27.0, *) {
            let maxDimension = VTLowLatencyFrameInterpolationConfiguration.maximumDimension(forSpatialScaleFactor: 1) ?? 0
            let maxPixels = VTLowLatencyFrameInterpolationConfiguration.maximumPixelCount(forSpatialScaleFactor: 1) ?? 0
            VergenceLog.pipeline.notice(
                "frame interpolation limits: \(maxDimension, privacy: .public) per side, \(maxPixels, privacy: .public) pixels"
            )
            if maxDimension > 0, width > maxDimension || height > maxDimension || width * height > maxPixels {
                return false
            }
        }
        guard let configuration = VTLowLatencyFrameInterpolationConfiguration(
            frameWidth: width, frameHeight: height, numberOfInterpolatedFrames: 1
        ) else { return false }
        let formats = configuration.supportedPixelFormats
        let format = formats.contains(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange)
            ? kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange : (formats.first ?? 0)
        VergenceLog.pipeline.notice(
            "frame interpolation formats: \(formats.map(fourCC).joined(separator: " "), privacy: .public)"
        )
        let processor = VTFrameProcessor()
        do {
            try processor.startSession(configuration: configuration)
        } catch {
            VergenceLog.pipeline.error("frame interpolation: session failed: \(error, privacy: .public)")
            return false
        }
        func pool(_ attributes: [String: Any], format: OSType, width: Int = width, height: Int = height) -> CVPixelBufferPool? {
            var attributes = attributes
            attributes[kCVPixelBufferPixelFormatTypeKey as String] = format
            attributes[kCVPixelBufferWidthKey as String] = width
            attributes[kCVPixelBufferHeightKey as String] = height
            if attributes[kCVPixelBufferIOSurfacePropertiesKey as String] == nil {
                attributes[kCVPixelBufferIOSurfacePropertiesKey as String] = [String: Any]()
            }
            var pool: CVPixelBufferPool?
            CVPixelBufferPoolCreate(nil, nil, attributes as CFDictionary, &pool)
            return pool
        }
        var toYUV: VTPixelTransferSession?
        var toBGRA: VTPixelTransferSession?
        VTPixelTransferSessionCreate(allocator: nil, pixelTransferSessionOut: &toYUV)
        VTPixelTransferSessionCreate(allocator: nil, pixelTransferSessionOut: &toBGRA)
        let yuvPool = pool(configuration.sourcePixelBufferAttributes as [String: Any], format: format)
        let midPool = pool(configuration.destinationPixelBufferAttributes as [String: Any], format: format)
        let bgraPool = pool([kCVPixelBufferMetalCompatibilityKey as String: true], format: kCVPixelFormatType_32BGRA,
                            width: fullWidth, height: fullHeight)
        guard let toYUV, let toBGRA, yuvPool != nil, midPool != nil, bgraPool != nil else {
            processor.endSession()
            return false
        }
        if let resultGuard, !resultGuard.prepare(width: fullWidth, height: fullHeight) {
            VergenceLog.pipeline.error("frame interpolation: static-region guard unavailable")
        }
        self.processor = processor
        self.toYUV = toYUV
        self.toBGRA = toBGRA
        self.yuvPool = yuvPool
        self.midPool = midPool
        self.bgraPool = bgraPool
        self.width = width
        self.height = height
        return true
    }

    public func midFrame(between previous: CVPixelBuffer, and current: CVPixelBuffer,
                         completion: @escaping @Sendable (CVPixelBuffer?, Double) -> Void) {
        guard let processor, let toBGRA,
              let previousYUV = lastConverted?.source === previous ? lastConverted?.yuv : convert(previous),
              let currentYUV = convert(current),
              let midYUV = makeBuffer(midPool) else {
            completion(nil, .nan)
            return
        }
        lastConverted = (current, currentYUV)
        copyColorTags(from: current, to: midYUV)
        // The times only order the two inputs and place the result between them.
        guard let previousFrame = VTFrameProcessorFrame(buffer: previousYUV, presentationTimeStamp: CMTime(value: 0, timescale: 2)),
              let currentFrame = VTFrameProcessorFrame(buffer: currentYUV, presentationTimeStamp: CMTime(value: 2, timescale: 2)),
              let midFrame = VTFrameProcessorFrame(buffer: midYUV, presentationTimeStamp: CMTime(value: 1, timescale: 2)),
              let parameters = VTLowLatencyFrameInterpolationParameters(
                  sourceFrame: currentFrame, previousFrame: previousFrame,
                  interpolationPhase: [0.5], destinationFrames: [midFrame]) else {
            completion(nil, .nan)
            return
        }
        // The completion runs on the processor's queue; one request is in
        // flight at a time, so nothing else touches these meanwhile.
        let job = Job(previous: previous, current: current, midYUV: midYUV, toBGRA: toBGRA, bgraPool: bgraPool,
                      resultGuard: resultGuard)
        let started = CACurrentMediaTime()
        processor.process(parameters: parameters) { _, error in
            let processMs = (CACurrentMediaTime() - started) * 1000
            guard error == nil, let bgraPool = job.bgraPool else {
                if let error { VergenceLog.pipeline.error("frame interpolation failed: \(error, privacy: .public)") }
                completion(nil, processMs)
                return
            }
            var out: CVPixelBuffer?
            CVPixelBufferPoolCreatePixelBuffer(nil, bgraPool, &out)
            guard let out, VTPixelTransferSessionTransferImage(job.toBGRA, from: job.midYUV, to: out) == noErr else {
                completion(nil, processMs)
                return
            }
            CVBufferPropagateAttachments(job.current, out)
            guard let resultGuard = job.resultGuard else {
                completion(out, processMs)
                return
            }
            // The governor's time is the processor's plus the guard's (about
            // 0.4 ms at 1080p on an M2 Max), not the conversion between them.
            let guardStarted = CACurrentMediaTime()
            resultGuard.apply(previous: job.previous, current: job.current, mid: out) { guarded in
                completion(guarded, processMs + (CACurrentMediaTime() - guardStarted) * 1000)
            }
        }
    }

    private struct Job: @unchecked Sendable {
        let previous: CVPixelBuffer
        let current: CVPixelBuffer
        let midYUV: CVPixelBuffer
        let toBGRA: VTPixelTransferSession
        let bgraPool: CVPixelBufferPool?
        let resultGuard: InterpolationGuard?
    }

    private func convert(_ bgra: CVPixelBuffer) -> CVPixelBuffer? {
        guard let toYUV, let yuv = makeBuffer(yuvPool) else { return nil }
        copyColorTags(from: bgra, to: yuv)
        return VTPixelTransferSessionTransferImage(toYUV, from: bgra, to: yuv) == noErr ? yuv : nil
    }

    private func makeBuffer(_ pool: CVPixelBufferPool?) -> CVPixelBuffer? {
        guard let pool else { return nil }
        var buffer: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &buffer)
        return buffer
    }

    /// The same primaries and transfer function as the frame (so neither
    /// conversion changes its colour), and one fixed matrix both ways.
    private func copyColorTags(from source: CVPixelBuffer, to dest: CVPixelBuffer) {
        for key in [kCVImageBufferColorPrimariesKey, kCVImageBufferTransferFunctionKey] {
            if let value = CVBufferCopyAttachment(source, key, nil) {
                CVBufferSetAttachment(dest, key, value, .shouldPropagate)
            }
        }
        let matrix = CVBufferCopyAttachment(source, kCVImageBufferYCbCrMatrixKey, nil)
            ?? kCVImageBufferYCbCrMatrix_ITU_R_709_2
        CVBufferSetAttachment(dest, kCVImageBufferYCbCrMatrixKey, matrix, .shouldPropagate)
    }
}

private func fourCC(_ value: OSType) -> String {
    let bytes = [24, 16, 8, 0].map { UInt8((value >> UInt32($0)) & 0xff) }
    return String(bytes: bytes, encoding: .ascii) ?? "\(value)"
}
#endif
private extension PumpFrame {
    var standingIn: PumpFrame {
        var frame = self
        frame.standsInForMidFrame = true
        return frame
    }
}
#endif
