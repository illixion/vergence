/*
 Vergence - how many frames the stereo pump shows, and whether that keeps up.

 Kept apart from `StereoPump` (visionOS only) so the arithmetic is testable on
 any platform. Every time here is either on the source's own timeline (frame
 presentation times) or wall time (`CACurrentMediaTime`), and the names say
 which.

 - `SourceFrameSpacing`: the source's own frame rate, read from the gaps
   between the presentation times of frames shown.
 - `FrameRateCap`: at a faster speed, a pump whose every depth map is an
   inference shows no more frames a second than the video has at normal
   speed, evenly spaced, instead of as many as the inference manages (it
   would take every other frame, then every third, unevenly). It counts on
   the video's timeline: a wall-clock allowance can't be even, since a tick
   comes either when the last one's work ends or on the next 60 Hz beat.
 - `StereoPerformance.measured`: what share of the frames it could have shown
   it did show. "Could have" is the frames the video has at normal speed in
   that wall time, so a 2× video showing all of its 1× frame rate keeps up.
 */

import Foundation

/// The source's frame spacing on its own timeline: the shortest gap between
/// two shown frames' presentation times, once it has been seen twice. One
/// short gap alone (a timestamp glitch at a segment boundary) can't set it,
/// and later frames lost to an overload can't widen it.
struct SourceFrameSpacing {
    /// Seconds between frames; infinite until known.
    private(set) var seconds = Double.infinity
    /// Gaps seen once, in `binWidth` bins, until one repeats.
    private var seen: Set<Int> = []
    private static let binWidth = 0.0005
    /// Gaps outside this range are a seek, a pause or a loop, not a frame.
    static let range = (1.0 / 240.0) ..< 0.25

    mutating func note(gap: Double) {
        guard Self.range.contains(gap), gap < seconds else { return }
        let bin = Int((gap / Self.binWidth).rounded())
        // A neighbouring bin is the same gap rounded the other way.
        if seen.contains(bin) || seen.contains(bin - 1) || seen.contains(bin + 1) {
            seconds = gap
            seen = seen.filter { Double($0) * Self.binWidth < gap }
        } else {
            seen.insert(bin)
        }
    }
}

/// Which frames a pump shows at a faster speed, so that it shows no more
/// frames a second than the video has at normal speed, evenly spaced: a
/// frame is shown once the video has moved on by what normal speed plays
/// between two frames (two frames on at 2×, 2, 1, 2, 1 at 1.5×), on the
/// video's own timeline. At normal speed or slower every frame is shown.
struct FrameRateCap {
    /// Where on the video's timeline the next frame to show is due.
    private var due: Double?

    /// Whether the frame at presentation time `pts` is shown, for a source
    /// `spacing` seconds between frames playing at `rate` (2 at double
    /// speed).
    mutating func admits(frameAt pts: Double, spacing: Double, rate: Double) -> Bool {
        guard spacing.isFinite, spacing > 0, rate > 1.01 else {
            due = nil
            return true
        }
        let step = rate * spacing
        // Too soon; well before that is a seek back, shown at once.
        if let due, pts < due - 0.4 * spacing, pts > due - step - spacing { return false }
        // On time, the next is due a step on from this one's due time, which
        // keeps the pattern; late, or after a seek, a step on from this frame.
        if let previous = due, abs(pts - previous) <= spacing {
            due = previous + step
        } else {
            due = pts + step
        }
        return true
    }
}

extension StereoPerformance {
    /// One window of delivery: `frames` shown over `window` wall seconds,
    /// covering `itemSpan` seconds of the source, whose frames are `spacing`
    /// apart, by a pump that can show at most `maxShownPerSecond`. Nil when
    /// too little was shown to judge.
    static func measured(frames: Int, itemSpan: Double, window: Double, spacing: Double,
                         maxShownPerSecond: Double, sourceHeight: Int) -> StereoPerformance? {
        guard frames >= 4, spacing.isFinite, spacing > 0, itemSpan > 0, window > 0 else { return nil }
        // Every frame the playhead passed; at a faster speed no more than the
        // video has at normal speed in the same time (a 60 fps video at 2×
        // passes 120 a second, and showing 60 of them is keeping up); never
        // more than the pump's ticks.
        let expected = min(itemSpan / spacing, window / spacing, window * maxShownPerSecond)
        return StereoPerformance(
            shownFramesPerSecond: Double(frames) / window,
            sourceFramesPerSecond: 1 / spacing,
            deliveredFraction: min(1, Double(frames) / expected),
            playbackRate: itemSpan / window,
            sourceHeight: sourceHeight
        )
    }
}
