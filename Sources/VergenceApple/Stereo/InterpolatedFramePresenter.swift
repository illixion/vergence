#if canImport(Metal) && canImport(VideoToolbox)
import CoreMedia
import Foundation
import QuartzCore

/// Shows an `InterpolatingFrameSource` (or an `UpscalingFrameSource`, or one
/// over the other) in a flat player that draws whatever frame is due at each
/// display tick (no stereo pump, no depth).
///
/// The pump takes frames well ahead and schedules each for its due time; a
/// flat player instead asks "what is on screen now?". The interpolating
/// source hands frames out up to a real frame's spacing *early* (it reads one
/// real frame ahead to place the in-between frame), so asked at the playhead
/// it would show both halves of a pair at once. This holds what it is handed
/// until each is due, and asks `lead` ahead of the playhead so the
/// interpolation has that long to finish before the frame is needed; the
/// interpolating source's governor sees that lead as its reserve
/// (`InterpolationGovernor.Thresholds.flat`).
///
/// It reaches the lead a step at a time, so no decoded frame between the
/// playhead and the lead is skipped (an output asked past a frame drops it).
///
/// **A frame counts as due half a draw tick early.** When frames come as
/// often as the draw loop ticks (60 fps doubled to 120 on a 120 Hz loop), a
/// tick that lands a hair before a frame's time would leave it for the next
/// tick, which then finds two due, shows the newer and drops the older: a
/// hitch every few hundred milliseconds wherever the ticks and the frames
/// line up. Due within half a tick (and half a frame), each tick takes the
/// one frame nearest it. The tick is measured from the calls.
/// Not thread-safe: call it from the player's draw loop only.
public final class InterpolatedFramePresenter {
    /// How far ahead of the playhead frames are asked for, in item seconds at
    /// normal speed. Three 24 fps frames: enough for one ~16-33 ms
    /// interpolation to finish with time to spare, small enough to hold.
    public static let lead = 0.15
    /// The furthest an ask moves past the last frame taken: 1.5 spacings of
    /// the frames handed out, so each ask yields at most the next frame (an
    /// output asked past a frame drops it). Until the spacing is known, under
    /// one spacing of anything up to 80 fps: skipping frames there would make
    /// a 60 fps source look like a 20 fps one and get it interpolated. It
    /// only means the first few frames come in later than `lead`.
    static let defaultStep = 1.0 / 80
    /// Asks per tick: catches up after a stall without draining the source.
    static let maxPullsPerTick = 3
    /// A playhead this far behind the last frame shown is a seek back.
    static let backwardJump = 0.1

    public let source: LookaheadFrameSource
    private var pending: [PumpFrame] = []
    private var lastPulled: Double?
    private var lastShown: Double?
    private var spacing = SourceFrameSpacing()
    private var lastTick: CFTimeInterval?
    /// Smoothed time between calls, once measured.
    private(set) var tickInterval: Double?
    /// For the 5 s log line: what reached the screen.
    private var logStart: CFTimeInterval?
    private var ticksLogged = 0
    private var shownLogged = 0

    public init(source: LookaheadFrameSource) {
        self.source = source
    }

    /// The newest frame due at `hostTime` that hasn't been shown yet, or nil
    /// (keep showing the last one).
    public func frame(forHostTime hostTime: CFTimeInterval) -> PumpFrame? {
        let playhead = source.itemTime(forHostTime: hostTime).seconds
        guard playhead.isFinite else { return nil }
        noteTick(hostTime)
        if let lastShown, playhead < lastShown - Self.backwardJump { reset() }
        if let first = pending.first, first.itemTime.seconds > playhead + Self.lead + 1 { reset() }

        // Where the clock will be `lead` from now: the playhead itself while
        // paused, further at a higher speed.
        let ahead = source.itemTime(forHostTime: hostTime + Self.lead).seconds
        let horizon = ahead.isFinite ? max(ahead, playhead) : playhead
        let step = spacing.seconds.isFinite ? 1.5 * spacing.seconds : Self.defaultStep
        for _ in 0 ..< Self.maxPullsPerTick {
            let target = min(horizon, max(lastPulled ?? playhead, playhead) + step)
            // Already holding frames out to the horizon. The interpolating
            // source reads a real frame ahead of any ask, so asking anyway
            // would run ahead as far as the output decodes (5 s on device
            // while interpolation was paused: ~130 held 1080p frames).
            if let lastPulled, target <= lastPulled { break }
            guard let frame = source.frame(atItemTime: CMTime(seconds: target, preferredTimescale: 600_000)) else { break }
            let seconds = frame.itemTime.seconds
            if let lastPulled {
                if seconds <= lastPulled { continue } // a repeat; nothing newer
                spacing.note(gap: seconds - lastPulled)
            }
            lastPulled = seconds
            pending.append(frame)
            if seconds >= horizon { break }
        }

        var due: PumpFrame?
        let early = earlyAllowance
        while let first = pending.first, first.itemTime.seconds <= playhead + early {
            due = pending.removeFirst()
        }
        if let due { lastShown = due.itemTime.seconds }
        logIfDue(hostTime, shown: due != nil)
        return due
    }

    /// Half a tick, or half the frames' spacing where that is shorter; none
    /// until the tick is measured.
    var earlyAllowance: Double {
        guard let tickInterval else { return 0 }
        let frames = spacing.seconds
        return 0.5 * (frames.isFinite ? min(tickInterval, frames) : tickInterval)
    }

    /// `presenter …` every 5 s of drawing: frames shown and draw ticks a
    /// second, the measured tick, so a device run says whether 120 frames a
    /// second reached the screen.
    private func logIfDue(_ hostTime: CFTimeInterval, shown: Bool) {
        ticksLogged += 1
        if shown { shownLogged += 1 }
        guard let start = logStart else {
            logStart = hostTime
            return
        }
        let window = hostTime - start
        guard window >= 5 else { return }
        VergenceLog.pipeline.info(
            "presenter \(Double(self.shownLogged) / window, format: .fixed(precision: 1), privacy: .public) frames/s shown over \(Double(self.ticksLogged) / window, format: .fixed(precision: 1), privacy: .public) draws/s, tick \((self.tickInterval ?? .nan) * 1000, format: .fixed(precision: 2), privacy: .public) ms"
        )
        logStart = hostTime
        ticksLogged = 0
        shownLogged = 0
    }

    private func noteTick(_ hostTime: CFTimeInterval) {
        defer { lastTick = hostTime }
        guard let lastTick else { return }
        let gap = hostTime - lastTick
        // A stall or a paused view isn't the loop's rate; nor is a repeat call.
        guard gap > 1.0 / 400, gap < 1.0 / 20 else { return }
        tickInterval = tickInterval.map { 0.9 * $0 + 0.1 * gap } ?? gap
    }

    /// Drop what is held, after a seek or a loop.
    public func reset() {
        pending.removeAll()
        lastPulled = nil
        lastShown = nil
        (source as? ResettableFrameSource)?.reset()
    }
}
#endif
