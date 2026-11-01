#if canImport(Metal) && canImport(VideoToolbox)
import CoreMedia
import Foundation
import QuartzCore

/// Shows an `InterpolatingFrameSource` in a flat player that draws whatever
/// frame is due at each display tick (no stereo pump, no depth).
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

    public init(source: LookaheadFrameSource) {
        self.source = source
    }

    /// The newest frame due at `hostTime` that hasn't been shown yet, or nil
    /// (keep showing the last one).
    public func frame(forHostTime hostTime: CFTimeInterval) -> PumpFrame? {
        let playhead = source.itemTime(forHostTime: hostTime).seconds
        guard playhead.isFinite else { return nil }
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
        while let first = pending.first, first.itemTime.seconds <= playhead {
            due = pending.removeFirst()
        }
        if let due { lastShown = due.itemTime.seconds }
        return due
    }

    /// Drop what is held, after a seek or a loop.
    public func reset() {
        pending.removeAll()
        lastPulled = nil
        lastShown = nil
        (source as? InterpolatingFrameSource)?.reset()
    }
}
#endif
