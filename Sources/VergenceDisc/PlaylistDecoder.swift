/*
 Vergence - stereo access units decoded to frames, with seeking

 `StereoDecoder` joins any `StereoAccessUnitReader` (a disc playlist, a
 Matroska MVC track) and `MVCDecoder`: frames come out in display order on
 the reader's timeline, and only the ones a player would show. Pictures
 decoded only as references (before a PlayItem's IN time, or between a
 seek's entry point and its target) are dropped.

 `PlaylistDecoder` is the disc's: a playlist's reader plus its side PIDs
 (audio and subtitles read from the same bytes as the picture).
 */

import Foundation

/// Where stereo access units come from, in decode order, with seeking.
public protocol StereoAccessUnitReader: AnyObject {
    /// The next access unit, or nil at the end.
    func next() throws -> StereoAccessUnit?
    /// Moves to the random access point at or before `seconds` on the
    /// reader's timeline and returns its time, 90 kHz.
    @discardableResult func seek(to seconds: Double) throws -> Int64
    /// Base-view pictures that came without a dependent view.
    var unpairedBaseCount: Int { get }
    /// Places where damaged bytes were skipped.
    var resyncCount: Int { get }
}

extension PlaylistStereoReader: StereoAccessUnitReader {}

public final class StereoDecoder {
    public let reader: any StereoAccessUnitReader
    private let decoder: MVCDecoder
    /// Frames before this time (90 kHz) are not shown.
    private var showFrom: Int64 = 0
    private var lastShown: Int64 = .min
    private var stoppedEarly = false
    /// Called after each access unit is read, before it is decoded.
    public var afterEachUnit: (() -> Void)?

    public var unpairedBaseCount: Int { reader.unpairedBaseCount }
    public var resyncCount: Int { reader.resyncCount }
    public var corruptNALCount: Int { decoder.corruptNALCount }
    public var unsupportedNALCount: Int { decoder.unsupportedNALCount }

    public init(reader: any StereoAccessUnitReader, threads: Int = 0) throws {
        self.reader = reader
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

    /// Decodes until `output` returns false or the stream ends. A frame's
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
                afterEachUnit?()
                try decoder.decode(unit, output: show)
            }
            afterEachUnit?()
            try decoder.finish(output: show)
        } catch is Stop {
            stoppedEarly = true
        }
    }
}

public final class PlaylistDecoder {
    public let playlist: BlurayPlaylist
    private let reader: PlaylistStereoReader
    private let decoder: StereoDecoder

    /// PIDs of the base clips to hand to `onSidePackets` as the video is
    /// read: audio and subtitles, from the same bytes. Takes effect at the
    /// next seek.
    public var sidePIDs: Set<UInt16> {
        get { reader.sidePIDs }
        set { reader.sidePIDs = newValue }
    }
    /// Called during `decode` with the side PIDs' PES (playlist timestamps)
    /// as they are read, a little ahead of the pictures they go with.
    public var onSidePackets: (([PESPacket]) -> Void)?

    public var unpairedBaseCount: Int { decoder.unpairedBaseCount }
    public var resyncCount: Int { decoder.resyncCount }
    public var corruptNALCount: Int { decoder.corruptNALCount }
    public var unsupportedNALCount: Int { decoder.unsupportedNALCount }

    /// - Parameter stereo: false decodes the base view alone.
    public init(disc: any ClipSource, playlist: BlurayPlaylist, stereo: Bool = true, threads: Int = 0) throws {
        self.playlist = playlist
        let reader = disc.stereoReader(for: playlist, stereo: stereo)
        self.reader = reader
        decoder = try StereoDecoder(reader: reader, threads: threads)
        decoder.afterEachUnit = { [weak self] in
            guard let self, let onSidePackets = self.onSidePackets else { return }
            let side = reader.takeSidePackets()
            if !side.isEmpty { onSidePackets(side) }
        }
    }

    /// See `StereoDecoder.seek(to:)`.
    public func seek(to seconds: Double) throws {
        try decoder.seek(to: seconds)
    }

    /// See `StereoDecoder.decode(output:)`.
    public func decode(output: (StereoFrame) throws -> Bool) throws {
        try decoder.decode(output: output)
    }
}
