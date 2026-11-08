#if canImport(Metal) && canImport(VideoToolbox)
import CoreMedia
import CoreVideo
import Foundation
import QuartzCore
import VergenceCore

/// Makes a frame bigger as it is played. Called on the owning
/// `UpscalingFrameSource`'s work queue, one request at a time.
public protocol RealtimeFrameUpscaler: AnyObject, Sendable {
    /// The size a `width`×`height` picture comes out at, or nil when it
    /// can't be upscaled (already big enough, or a size the upscaler doesn't
    /// take).
    func outputSize(width: Int, height: Int) -> (width: Int, height: Int)?
    /// Get ready for frames of this size: true when they can be upscaled.
    /// May block (a model load, seconds the first time).
    func prepare(width: Int, height: Int) -> Bool
    /// `frame` (BGRA, of the prepared size) upscaled to `outputSize`, BGRA,
    /// handed to `completion` on any queue with the milliseconds it took;
    /// nil on failure.
    func upscale(_ frame: CVPixelBuffer, completion: @escaping @Sendable (_ frame: CVPixelBuffer?, _ ms: Double) -> Void)
}

/// Plays a small picture sharper: each decoded frame goes through a learned
/// upscaler (`VTRealtimeUpscaler`, VideoToolbox's low-latency super
/// resolution) before it is shown, up to 1080p. It wraps the frame output
/// *below* Smooth Motion, so an interpolator makes its in-between frames from
/// the sharper frames, and once per real frame.
///
/// **It works a few frames ahead.** Its frames go through a pipeline of up
/// to `depth` frames, in order, one upscaling at a time: asked for a frame
/// with the pipeline empty, it takes the newest one at or before that time
/// from its source, starts upscaling it and answers nil; an ask finds the
/// oldest frame done. Each frame handed out tops the pipeline up with the
/// frames after the last one taken, each read up to 1.5 frame spacings past
/// it (never further, so no frame is skipped). So a fast upscaler hands out
/// more than one frame per draw tick and gets ahead of the playhead by the
/// player's lead (`InterpolatedFramePresenter.lead`): one frame per tick
/// never would where the ticks come as often as the frames (a Mac window drew
/// a 24 fps picture at 24 ticks a second, every frame handed out at its time).
///
/// **What passes through as decoded:** HDR (half-float) frames, since the
/// upscaler is 8-bit 4:2:0 and would band them; a picture it doesn't take
/// (720p's 1.5× is the last that fits 1080p); frames while its model loads
/// (the first second or two of a picture, seconds after an install); frames
/// while `isEnabled` is off; and every frame after it **turns itself off for
/// the picture**, when the median of 12 upscales takes over
/// `maxShareOfFrameTime` of a frame's time, so it can't keep the player behind
/// its sound. Frames that pass through after upscaled ones change size, which
/// an interpolator above handles as any size change.
public final class UpscalingFrameSource: LookaheadFrameSource, ResettableFrameSource, @unchecked Sendable {
    /// The share of a frame's time an upscale may take (median of
    /// `timingWindow`) before it stops for the picture: what is left goes to
    /// Smooth Motion and drawing.
    public static let maxShareOfFrameTime = 0.6
    static let timingWindow = 12
    /// Frames in the pipeline at once (upscaling, waiting, or done and not
    /// asked for yet): three 1080p results, 25 MB.
    static let depth = 3

    /// A frame taken from the source, in order: waiting to be upscaled,
    /// being upscaled, or done (`result`).
    private struct Slot {
        let id: Int
        let frame: PumpFrame
        var started = false
        var result: PumpFrame?
        var seconds: Double { frame.itemTime.seconds }
    }

    private let inner: LookaheadFrameSource
    private let upscaler: RealtimeFrameUpscaler
    private let work = DispatchQueue(label: "vergence.upscale", qos: .userInteractive)
    private let lock = NSLock()

    // Guarded by `lock`.
    private var enabled = true
    private var slots: [Slot] = []
    private var nextSlotID = 0
    private var upscaling = false
    private var generation = 0
    private var lastSequence: Int
    /// The newest frame taken from the source, and the last handed out.
    private var lastFetched: Double?
    private var lastTaken: Double?
    private var spacing = SourceFrameSpacing()
    private var prepared: (width: Int, height: Int, ok: Bool)?
    private var preparing = false
    private var recentMs: [Double] = []
    private var offForPicture = false
    private var stats = Stats()

    public init(wrapping inner: LookaheadFrameSource, upscaler: RealtimeFrameUpscaler) {
        self.inner = inner
        self.upscaler = upscaler
        lastSequence = inner.sequence
    }

    /// `inner` wrapped with VideoToolbox's upscaler where this device has it
    /// (a Mac, an iPhone, iPad or Vision Pro on 26 or later; not the
    /// simulator), or `inner` itself.
    public static func wrappingIfSupported(_ inner: LookaheadFrameSource) -> LookaheadFrameSource {
        #if targetEnvironment(simulator)
        VergenceLog.pipeline.notice("upscaling: not supported in the simulator, playing source frames")
        return inner
        #else
        if #available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *), VTRealtimeUpscaler.isSupported {
            return UpscalingFrameSource(wrapping: inner, upscaler: VTRealtimeUpscaler())
        }
        VergenceLog.pipeline.notice("upscaling: not supported here, playing source frames")
        return inner
        #endif
    }

    /// Whether a `width`×`height` picture would be upscaled here, and to what.
    public static func upscaledSize(width: Int, height: Int) -> CGSize? {
        #if targetEnvironment(simulator)
        return nil
        #else
        if #available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *),
           let factor = VTRealtimeUpscaler.factor(width: width, height: height) {
            return CGSize(width: (Double(width) * Double(factor)).rounded(),
                          height: (Double(height) * Double(factor)).rounded())
        }
        return nil
        #endif
    }

    /// Off: frames pass through as decoded (the player shows the picture no
    /// bigger than it decoded, so there is nothing to gain). Takes effect
    /// from the next frame taken.
    public var isEnabled: Bool {
        get { lock.withLock { enabled } }
        set { lock.withLock { enabled = newValue } }
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

    public func reset() {
        lock.withLock { resetLocked() }
        (inner as? ResettableFrameSource)?.reset()
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
        // An ask a jump away from what the pipeline holds: a seek either way.
        if let head = slots.first, head.seconds < t - LookaheadPull.maxStep || head.seconds > t + LookaheadPull.maxStep {
            resetLocked()
        }
        if slots.isEmpty {
            guard let frame = inner.frame(atItemTime: itemTime) else { return nil }
            enqueue(frame)
        }
        // The oldest frame, once it's done and asked for; a frame read ahead
        // waits for an ask at or past its time.
        guard let head = slots.first, head.seconds <= t + 1e-6, let result = head.result else { return nil }
        slots.removeFirst()
        hand(result)
        fill()
        return result
    }

    /// Notes a frame handed out: its spacing, and whether it came after its
    /// time (more than half a frame behind a running playhead).
    private func hand(_ frame: PumpFrame) {
        let seconds = frame.itemTime.seconds
        if let lastTaken, seconds.isFinite { spacing.note(gap: seconds - lastTaken) }
        lastTaken = seconds
        stats.handed += 1
        if frame.sourceSize != nil, spacing.seconds.isFinite, let playhead = playheadSeconds(),
           seconds < playhead - spacing.seconds / 2 {
            stats.late += 1
        }
    }

    /// Tops the pipeline up to `depth` with the frames after the last one
    /// taken, each read 1.5 spacings past it so none is skipped.
    private func fill() {
        guard spacing.seconds.isFinite else { return }
        while slots.count < Self.depth, let last = lastFetched,
              let next = inner.frame(atItemTime: CMTime(seconds: last + 1.5 * spacing.seconds,
                                                        preferredTimescale: 600_000)) {
            enqueue(next)
        }
    }

    /// Adds `frame` to the pipeline: done at once when it passes through,
    /// else waiting its turn to be upscaled.
    private func enqueue(_ frame: PumpFrame) {
        lastFetched = frame.itemTime.seconds
        let buffer = frame.pixelBuffer
        var slot = Slot(id: nextSlotID, frame: frame)
        nextSlotID += 1
        if let reason = passReason(buffer, width: CVPixelBufferGetWidth(buffer),
                                   height: CVPixelBufferGetHeight(buffer)) {
            stats.passed[reason, default: 0] += 1
            slot.result = frame
        }
        slots.append(slot)
        startNext()
    }

    /// Starts the first waiting frame when nothing is being upscaled: one
    /// request at a time, as the upscaler wants.
    private func startNext() {
        // Turned off meanwhile: what waits passes through.
        if offForPicture || !enabled {
            for index in slots.indices where !slots[index].started && slots[index].result == nil {
                slots[index].result = slots[index].frame
            }
        }
        guard !upscaling, let index = slots.firstIndex(where: { !$0.started && $0.result == nil }) else { return }
        slots[index].started = true
        upscaling = true
        let slot = slots[index]
        let requestGeneration = generation
        let upscaler = upscaler
        let buffer = slot.frame.pixelBuffer
        let size = CGSize(width: CVPixelBufferGetWidth(buffer), height: CVPixelBufferGetHeight(buffer))
        work.async { [weak self] in
            upscaler.upscale(buffer) { [weak self] output, ms in
                guard let self else { return }
                self.lock.withLock {
                    guard requestGeneration == self.generation else { return }
                    self.upscaling = false
                    var result = slot.frame
                    if let output {
                        result = PumpFrame(pixelBuffer: output, itemTime: slot.frame.itemTime,
                                           standsInForMidFrame: slot.frame.standsInForMidFrame)
                        result.sourceSize = size
                        self.stats.noteUpscaled(ms: ms)
                        self.noteTiming(ms)
                    } else {
                        self.stats.failed += 1
                    }
                    if let index = self.slots.firstIndex(where: { $0.id == slot.id }) {
                        self.slots[index].result = result
                    }
                    self.startNext()
                }
            }
        }
    }

    enum PassReason: String {
        case disabled, hdr = "not 8-bit", size, notReady = "not ready", off = "off for picture"
    }

    private func passReason(_ buffer: CVPixelBuffer, width: Int, height: Int) -> PassReason? {
        if offForPicture { return .off }
        if !enabled { return .disabled }
        guard CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32BGRA else { return .hdr }
        if let prepared, prepared.width == width, prepared.height == height {
            return prepared.ok ? nil : .size
        }
        // A size change while the model loads is picked up once it's done.
        if preparing { return .notReady }
        guard upscaler.outputSize(width: width, height: height) != nil else {
            prepared = (width, height, false)
            return .size
        }
        do {
            preparing = true
            prepared = nil
            let upscaler = upscaler
            work.async { [weak self] in
                let started = CACurrentMediaTime()
                let ok = upscaler.prepare(width: width, height: height)
                let ms = (CACurrentMediaTime() - started) * 1000
                VergenceLog.pipeline.notice(
                    "upscaling: \(width)×\(height) \(ok ? "ready" : "unavailable", privacy: .public) in \(ms, format: .fixed(precision: 0), privacy: .public) ms"
                )
                guard let self else { return }
                self.lock.withLock {
                    self.preparing = false
                    self.prepared = (width, height, ok)
                }
            }
        }
        return .notReady
    }

    /// Stops for the picture once upscaling takes too much of a frame's time.
    private func noteTiming(_ ms: Double) {
        guard ms.isFinite else { return }
        recentMs.append(ms)
        if recentMs.count > Self.timingWindow { recentMs.removeFirst() }
        guard recentMs.count == Self.timingWindow, spacing.seconds.isFinite, !offForPicture else { return }
        let median = recentMs.sorted()[recentMs.count / 2]
        let budget = Self.maxShareOfFrameTime * spacing.seconds * 1000
        if median > budget {
            offForPicture = true
            VergenceLog.pipeline.notice(
                "upscaling off for this picture: median \(median, format: .fixed(precision: 1), privacy: .public) ms over \(budget, format: .fixed(precision: 1), privacy: .public) ms of a frame's time"
            )
        }
    }

    private func playheadSeconds() -> Double? {
        let now = CACurrentMediaTime()
        guard let rate = inner.playbackRate(hostTime: now), rate > 0.01 else { return nil }
        let playhead = inner.itemTime(forHostTime: now).seconds
        return playhead.isFinite ? playhead : nil
    }

    private func resetLocked() {
        slots.removeAll()
        upscaling = false
        generation += 1
        lastFetched = nil
        lastTaken = nil
    }

    // MARK: Measurement

    struct Stats {
        var handed = 0
        var upscaled = 0
        var failed = 0
        var late = 0
        var passed: [PassReason: Int] = [:]
        var ms: [Double] = []
        var windowStart = CACurrentMediaTime()

        mutating func noteUpscaled(ms: Double) {
            upscaled += 1
            self.ms.append(ms)
        }
    }

    /// `upscale …` every 5 s while frames come: how many were upscaled and
    /// passed (why), the median and p90 time a frame, and how many were
    /// handed out after their time.
    private func logStatsIfDue() {
        let now = CACurrentMediaTime()
        guard now - stats.windowStart >= 5 else { return }
        defer { stats = Stats(windowStart: now) }
        guard stats.handed > 0 else { return }
        let sorted = stats.ms.sorted()
        let median = sorted.isEmpty ? Double.nan : sorted[sorted.count / 2]
        let p90 = sorted.isEmpty ? Double.nan : sorted[min(sorted.count - 1, sorted.count * 9 / 10)]
        let passed = stats.passed.sorted { $0.key.rawValue < $1.key.rawValue }
            .map { "\($0.key.rawValue) \($0.value)" }.joined(separator: ", ")
        VergenceLog.pipeline.info(
            "upscale \(self.stats.upscaled) upscaled, \(self.stats.failed) failed, passed [\(passed, privacy: .public)], \(self.stats.late) late; \(median, format: .fixed(precision: 1), privacy: .public) ms median, \(p90, format: .fixed(precision: 1), privacy: .public) p90"
        )
    }
}
#endif
