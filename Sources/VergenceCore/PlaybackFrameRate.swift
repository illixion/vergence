/*
 Vergence - frames a second at a playback speed

 Playback speed changes the frame rate, not which frames are shown: a
 60 fps video at 2× shows all 120 of its frames a second, and at 0.5× 30.
 The players (Hypnos's 2D Metal player, the stereo pump, the film player)
 ask the display for that rate and draw that often, up to what the display
 can do; past it they leave frames out evenly.

 Foundation only, like the rest of VergenceCore.
 */

import Foundation

public enum PlaybackFrameRate {
    /// The rates a video's measured rate is snapped to: a stream measures
    /// 23.976 as 23.975 or 24, and the display wants the real one.
    public static let standard: [Double] = [
        24000.0 / 1001, 24, 25, 30000.0 / 1001, 30, 48, 50, 60000.0 / 1001, 60, 100, 120000.0 / 1001, 120,
    ]
    /// The most frames a second a player draws: Vision Pro's fastest display
    /// mode (an M2 Vision Pro tops out at 100 Hz and shows what it can).
    public static let maximum = 120.0
    /// What a player draws at normal speed when nothing says otherwise.
    public static let normalDrawRate = 60.0

    /// The standard rate nearest `measured`.
    public static func snapped(_ measured: Double) -> Double {
        guard measured.isFinite, measured > 0 else { return measured }
        return standard.min { abs($0 - measured) < abs($1 - measured) } ?? measured
    }

    /// Frames a second shown for a video of `source` frames a second played at
    /// `speed`, with twice its frames when `doubled` (Smooth Motion), up to
    /// `maximum`.
    public static func shown(source: Double, doubled: Bool = false, speed: Double = 1) -> Double {
        guard source.isFinite, source > 0 else { return source }
        return min(maximum, source * (doubled ? 2 : 1) * max(speed, 0))
    }

    /// How often a draw loop that drew `base` times a second at normal speed
    /// draws at `speed`: as many times more faster than normal, up to
    /// `maximum`; slower than normal it keeps `base` (it shows each frame for
    /// longer, not fewer draws).
    public static func drawRate(base: Double = normalDrawRate, speed: Double) -> Double {
        min(maximum, max(base, base * speed))
    }
}
