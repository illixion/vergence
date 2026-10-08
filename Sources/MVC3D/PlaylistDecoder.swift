/*
 mvc3d - a playlist decoded to stereo frames, with seeking

 Joins `PlaylistStereoReader` and `MVCDecoder`: frames come out in display
 order on the playlist's timeline, and only the ones a player would show.
 Pictures decoded only as references (before a PlayItem's IN time, or
 between a seek's entry point and its target) are dropped.
 */

import Foundation

public final class PlaylistDecoder {
    public let playlist: BlurayPlaylist
    private let reader: PlaylistStereoReader
    private let decoder: MVCDecoder
    /// Frames before this playlist time (90 kHz) are not shown.
    private var showFrom: Int64 = 0
    private var lastShown: Int64 = .min
    private var stoppedEarly = false

    public var unpairedBaseCount: Int { reader.unpairedBaseCount }
    public var resyncCount: Int { reader.resyncCount }
    public var corruptNALCount: Int { decoder.corruptNALCount }
    public var unsupportedNALCount: Int { decoder.unsupportedNALCount }

    /// - Parameter stereo: false decodes the base view alone.
    public init(disc: BlurayDisc, playlist: BlurayPlaylist, stereo: Bool = true, threads: Int = 0) throws {
        self.playlist = playlist
        reader = disc.stereoReader(for: playlist, stereo: stereo)
        decoder = try MVCDecoder(threads: threads)
    }

    /// Makes the next frame the first one at or after `seconds`. Decoding
    /// restarts at the entry point before it, so the cost is up to a GOP
    /// (about a second) of pictures decoded and dropped.
    public func seek(to seconds: Double) throws {
        try seek(toPTS: Int64((max(seconds, 0) * 90_000).rounded()))
    }

    private func seek(toPTS pts: Int64) throws {
        decoder.reset()
        try reader.seek(to: Double(pts) / 90_000)
        showFrom = pts
        lastShown = .min
        stoppedEarly = false
    }

    /// Decodes until `output` returns false or the playlist ends. A frame's
    /// planes are only valid during the call. Calling again continues where
    /// it stopped (by seeking just past the last frame shown, since stopping
    /// abandons the rest of an access unit).
    public func decode(output: (StereoFrame) throws -> Bool) throws {
        struct Stop: Error {}
        if stoppedEarly { try seek(toPTS: lastShown + 1) }
        let show: (StereoFrame) throws -> Void = { [self] frame in
            if let pts = frame.pts {
                // A PlayItem's pictures before its IN time land on the
                // previous item's span, which has already been shown.
                guard pts >= showFrom, pts > lastShown else { return }
                lastShown = pts
            }
            if try !output(frame) { throw Stop() }
        }
        do {
            while let unit = try reader.next() {
                try decoder.decode(unit, output: show)
            }
            try decoder.finish(output: show)
        } catch is Stop {
            stoppedEarly = true
        }
    }
}
