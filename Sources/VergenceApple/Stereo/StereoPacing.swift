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
 - `LookaheadPull` / `LookaheadClock`: a pump fed by a clocked source takes
   each frame a little before it is due, so the frame's depth and warp are
   done by then and the renderer shows it on time. These say which frame to
   ask for next and when frames already taken stop belonging to the clock.
 - `HeldDepthPolicy`: which frames real-time depth infers and which reuse
   the map of the frame before.
 */

import Foundation
import VergenceCore

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
///
/// A pump that takes every frame in turn (rather than the newest at each
/// tick) also can't show frames faster than it ticks, so `minimumInterval`
/// spaces them that far apart in wall time: a 120 fps video shows every
/// other frame at 60.
struct FrameRateCap {
    /// Where on the video's timeline the next frame to show is due.
    private var due: Double?

    /// Whether the frame at presentation time `pts` is shown, for a source
    /// `spacing` seconds between frames playing at `rate` (2 at double
    /// speed). `limitsToNormalSpeed` false shows every frame at any speed
    /// that `minimumInterval` (wall seconds) allows.
    mutating func admits(frameAt pts: Double, spacing: Double, rate: Double,
                         limitsToNormalSpeed: Bool = true, minimumInterval: Double = 0) -> Bool {
        guard spacing.isFinite, spacing > 0 else {
            due = nil
            return true
        }
        let perFrame = limitsToNormalSpeed ? max(rate, 1) * spacing : spacing
        let step = max(perFrame, rate * minimumInterval)
        guard step > spacing * 1.01 else {
            due = nil
            return true
        }
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

/// Which frame a pump that works ahead of the display asks its source for.
///
/// Each frame in turn, not the newest one due by the end of the window: a
/// frame whose depth is still being inferred holds the pump up for longer
/// than a frame lasts, and asking for the newest frame afterwards would leave
/// out the frames in between (the old synchronous pump showed half of a
/// 60 fps video that way). Times are on the source's own timeline.
enum LookaheadPull {
    /// How far past the frame before it a frame taken may be and still be
    /// the next one; further is a jump (a seek, a loop, or a pump so far
    /// behind that it starts over).
    static let maxStep = 0.25

    /// The item time to ask for after the frame at `last`, with the frame
    /// showing now at `playhead` and the frame due `lead` from now at
    /// `windowEnd`. Halfway between the next frame and the one after, so a
    /// slightly uneven timestamp still finds it; never past `windowEnd`; the
    /// playhead itself to start (or after a seek) and when the next frame
    /// would be late already. Until the frame spacing is known, the newest
    /// frame in the window.
    static func target(last: Double?, spacing: Double, playhead: Double, windowEnd: Double) -> Double {
        guard let last else { return min(playhead, windowEnd) }
        guard spacing.isFinite, spacing > 0 else { return windowEnd }
        if last + spacing < playhead - 0.5 * spacing { return min(playhead, windowEnd) }
        return min(last + 1.5 * spacing, windowEnd)
    }
}

/// When frames a pump took ahead stop matching the source's clock.
enum LookaheadClock {
    /// How far (host seconds) a queued frame's due time may move before it is
    /// queued again: under half a 60 fps frame, so a clock nudge stays put
    /// and a speed change or a stall doesn't.
    static let retimeTolerance = 0.006

    /// Playhead movement past what the clock's rate explains, in seconds of
    /// item time: about a frame, so stepping a paused video one frame counts.
    static let jumpTolerance = 0.01

    /// Frames taken up to `last` no longer belong to a clock now at
    /// `playhead` with its window ending at `windowEnd`: a seek, or a loop.
    static func isDiscontinuous(last: Double, playhead: Double, windowEnd: Double) -> Bool {
        last < playhead - LookaheadPull.maxStep || last > windowEnd + LookaheadPull.maxStep
    }

    /// The playhead jumped (a seek, a loop): it read `previous` at `rate`
    /// `elapsed` host seconds ago and reads `playhead` at `rate` now.
    /// Running, it can't have moved back or further than the faster of the
    /// two rates carries it; a pause or a stall in between stops it short,
    /// which isn't a jump. Paused, it can't move at all.
    static func isJump(previous: Double, previousRate: Double, elapsed: Double,
                       playhead: Double, rate: Double) -> Bool {
        let furthest = previous + max(previousRate, rate, 0) * max(elapsed, 0)
        return playhead < previous - jumpTolerance || playhead > furthest + jumpTolerance
    }

    /// A frame queued to show at `queuedFor` is now due at `due` (nil when
    /// the clock has stopped: a pause or a stall), so it has to be queued
    /// again.
    static func needsRetime(queuedFor: Double, due: Double?) -> Bool {
        guard let due else { return true }
        return abs(due - queuedFor) > retimeTolerance
    }
}

/// Which frames real-time depth infers, and which reuse the map of the frame
/// before: half-rate depth against full-rate video, never more than one frame
/// old (the user's bound, 2026-07-06). Times are when each frame is shown, on
/// the host clock, so the inference rate doesn't follow the playback speed.
///
/// A frame reuses the map only when it is shown within `maxHeldAge` of the
/// frame the map was inferred for (48-60 fps alternate, 24-30 fps never
/// hold), follows it closely on the video's timeline (a seek always infers),
/// is the first to reuse that map, and isn't past a cut (mean luma within
/// `maxLumaDelta` on a sparse grid).
struct HeldDepthPolicy {
    static let maxHeldAge = 1.0 / 35.0
    static let maxHeldStep = 0.1
    /// Mean absolute luma change (0-255) on `LumaSignature`'s grid: the
    /// refiner's own cut threshold (`DepthRefinement.cutMeanLumaDelta`, 0.12
    /// of the range). Half that also caught fast motion inside one shot, and
    /// the extra maps (35 a second at 26 ms each) left frames late for ten
    /// seconds (device, 2026-10-22); the depth is at most a frame old either
    /// way, so only a cut needs its own map.
    static let maxLumaDelta: Float = DepthRefinement.cutMeanLumaDelta * 255
    static let maxHeldFrames = 1

    enum Decision: Equatable { case infer, hold }

    private var key: (seconds: Double, shownAt: Double, signature: [Float]?, held: Int)?

    /// Whether the frame at `seconds` (item time), shown at host time
    /// `shownAt`, reuses the last inferred map. On `.infer` it becomes the
    /// frame later ones are judged against.
    mutating func decide(seconds: Double, shownAt: Double, signature: [Float]?) -> Decision {
        if var key, seconds.isFinite,
           seconds >= key.seconds, seconds - key.seconds < Self.maxHeldStep,
           shownAt >= key.shownAt, shownAt - key.shownAt < Self.maxHeldAge,
           key.held < Self.maxHeldFrames,
           let signature, let reference = key.signature,
           LumaSignature.meanDelta(signature, reference).map({ $0 < Self.maxLumaDelta }) == true {
            key.held += 1
            self.key = key
            return .hold
        }
        key = seconds.isFinite ? (seconds, shownAt, signature, 0) : nil
        return .infer
    }

    /// The next frame infers, whatever it is.
    mutating func forget() { key = nil }
}

/// A frame's luma on a sparse grid: enough to tell a cut, or a static frame,
/// in a few microseconds.
enum LumaSignature {
    /// Samples per side (48×48 = 2304 points).
    static let gridSize = 48

    /// Mean absolute difference, on the 0-255 scale; nil when the two weren't
    /// sampled alike.
    static func meanDelta(_ a: [Float], _ b: [Float]) -> Float? {
        guard a.count == b.count, !a.isEmpty else { return nil }
        var sum: Float = 0
        for i in 0..<a.count { sum += abs(a[i] - b[i]) }
        return sum / Float(a.count)
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
