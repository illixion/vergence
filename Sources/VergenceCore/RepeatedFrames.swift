/*
 Vergence - frames a pulldown repeats

 Film is 24 (or 25) pictures a second. A copy made for 30 fps shows one
 picture in every four twice; remuxed to fit audio of another speed, it can
 end up at any rate at all (a 3D film at 31.24 fps, say: NTSC's 29.97 sped up
 by 25/23.976 to match PAL audio). Its frames then come four new, one
 repeat, which judders on a headset and costs a fifth more work for nothing.

 `judge` tells from each frame's difference to the one before it whether a
 video does that: in most five-frame stretches where something moves, one
 frame barely differs from the one before. `RepeatDropper` then drops one
 frame in every five, the one most like its predecessor, as ffmpeg's
 `decimate` does: exactly one in five, so the frames that stay keep their
 place in time (kept frame k shows at k × 5/4 of the old frame time) and the
 audio stays in sync, even across a cut where the cadence breaks.
 */

import Foundation

public enum RepeatedFrames {
    public static let cycle = 5

    public struct Verdict: Sendable, Equatable {
        /// Five-frame stretches with movement in them.
        public var moving: Int
        /// Those with one frame that repeats the one before.
        public var repeating: Int
        public var isPulldown: Bool { moving >= 8 && Double(repeating) >= 0.8 * Double(moving) }
    }

    /// Movement below this (mean absolute difference, on a 0-255 scale) says
    /// nothing either way.
    static let still: Float = 0.4
    /// A repeat differs from its frame by less than this share of the
    /// stretch's next smallest difference (compression noise, not movement).
    static let repeatShare: Float = 0.3

    /// `runs`: stretches of consecutive frames, each value a frame's
    /// difference to the frame before it.
    public static func judge(_ runs: [[Float]]) -> Verdict {
        var verdict = Verdict(moving: 0, repeating: 0)
        for run in runs {
            var start = 0
            while start + cycle <= run.count {
                let sorted = run[start ..< start + cycle].sorted()
                if sorted[1] >= still {
                    verdict.moving += 1
                    if sorted[0] < repeatShare * sorted[1] { verdict.repeating += 1 }
                }
                start += cycle
            }
        }
        return verdict
    }
}

/// Decides, five frames at a time, which to keep.
public struct RepeatDropper: Sendable {
    private var pending: [(index: Int, difference: Float)] = []
    private var next = 0

    public init() {}

    /// Takes the next frame's difference to the one before (nil for the
    /// first, never dropped) and returns the frames now decided to keep, by
    /// their index in the order they came.
    public mutating func add(_ difference: Float?) -> [Int] {
        pending.append((next, difference ?? .infinity))
        next += 1
        guard pending.count == RepeatedFrames.cycle else { return [] }
        let drop = pending.indices.min { pending[$0].difference < pending[$1].difference }!
        let kept = pending.enumerated().filter { $0.offset != drop }.map(\.element.index)
        pending.removeAll(keepingCapacity: true)
        return kept
    }

    /// The frames left at the end, all kept.
    public mutating func finish() -> [Int] {
        defer { pending.removeAll() }
        return pending.map(\.index)
    }
}
