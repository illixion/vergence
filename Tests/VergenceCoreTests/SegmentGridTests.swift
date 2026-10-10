import Foundation
import Testing
@testable import VergenceCore

@Suite struct SegmentGridTests {
    @Test func countsSegmentsUpToTheLastFrame() {
        // 7.03 s at 30 fps in 4 s segments: 0-4, 4-7.03.
        let grid = SegmentGrid(segmentSeconds: 4, durationSeconds: 7.0331, frameRate: 30)
        #expect(grid.count == 2)
        #expect(grid.duration(0) == 4)
        #expect(abs(grid.duration(1) - 3.0331) < 1e-9)
        #expect(grid.segment(at: 5) == 1 && grid.segment(at: 99) == 1 && grid.segment(at: -1) == 0)
        // Ending exactly on a boundary, or a frame after it: no empty segment.
        #expect(SegmentGrid(segmentSeconds: 2, durationSeconds: 8, frameRate: 24).count == 4)
        #expect(SegmentGrid(segmentSeconds: 1, durationSeconds: 7.0331, frameRate: 30).count == 7)
        // The sample table says where the last frame is.
        #expect(SegmentGrid(segmentSeconds: 1, durationSeconds: 7.0331, frameRate: 30, lastFrameSeconds: 7.01).count == 8)
        #expect(SegmentGrid(segmentSeconds: 1, durationSeconds: 7.0331, frameRate: 30, lastFrameSeconds: 6.9998).count == 7)
        #expect(SegmentGrid(segmentSeconds: 2, durationSeconds: 0, frameRate: 24).count == 0)
    }

    @Test func writesAPlaylistOfTheWholeGrid() {
        let grid = SegmentGrid(segmentSeconds: 4, durationSeconds: 9, frameRate: 30)
        let playlist = grid.mediaPlaylist(initURI: "init.mp4") { "\($0).m4s" }
        #expect(playlist.contains("#EXT-X-MAP:URI=\"init.mp4\""))
        #expect(playlist.contains("#EXT-X-TARGETDURATION:4"))
        #expect(playlist.components(separatedBy: "#EXTINF:").count - 1 == 3)
        #expect(playlist.contains("#EXTINF:1.00000,\n2.m4s"))
        #expect(playlist.hasSuffix("#EXT-X-ENDLIST\n"))
    }
}

@Suite struct FragmentTimesTests {
    typealias M = SphericalMetadataTests

    static func traf(track: UInt32, decodeTime: UInt64, version: UInt8, compositionOffset: Int32?) -> [UInt8] {
        let tfhd = M.box("tfhd", [0, 0, 0, 0] + M.be32(track))
        let time: [UInt8] = version == 1 ? M.be32(UInt32(decodeTime >> 32)) + M.be32(UInt32(decodeTime & 0xFFFF_FFFF)) : M.be32(UInt32(decodeTime))
        let tfdt = M.box("tfdt", [version, 0, 0, 0] + time)
        // One sample: duration, plus a composition offset when given.
        let flags: UInt32 = 0x100 | (compositionOffset == nil ? 0 : 0x800)
        var trun = M.be32(flags) + M.be32(1) + M.be32(512)
        if let compositionOffset { trun += M.be32(UInt32(bitPattern: compositionOffset)) }
        return M.box("traf", tfhd + tfdt + M.box("trun", trun))
    }

    @Test func movesOneTrackToItsTimeAndTheOthersByTheSameAmount() throws {
        let initSegment = Data(M.box("moov",
            M.box("trak", M.box("tkhd", [0, 0, 0, 0] + M.be32(0) + M.be32(0) + M.be32(1) + [UInt8](repeating: 0, count: 68))
                + M.box("mdia", M.box("mdhd", [0, 0, 0, 0] + M.be32(0) + M.be32(0) + M.be32(96_000) + M.be32(0))))
            + M.box("trak", M.box("tkhd", [0, 0, 0, 0] + M.be32(0) + M.be32(0) + M.be32(2) + [UInt8](repeating: 0, count: 68))
                + M.box("mdia", M.box("mdhd", [0, 0, 0, 0] + M.be32(0) + M.be32(0) + M.be32(48_000) + M.be32(0))))))
        let scales = FragmentTimes.timescales(initSegment: initSegment)
        #expect(scales == [1: 96_000, 2: 48_000])

        var segment = Data(M.box("moof", Self.traf(track: 1, decodeTime: 3982, version: 1, compositionOffset: 100)
                                     + Self.traf(track: 2, decodeTime: 0, version: 0, compositionOffset: nil)) + M.box("mdat", [1, 2, 3]))
        #expect(FragmentTimes.firstPresentation(segment) == [1: 4082, 2: 0])
        // Video to 2 s; the audio follows it (the writer had it 1991 samples
        // before the video), then keeps that shift when told to.
        let shift = try FragmentTimes.retime(&segment, track: 1, to: 192_000, timescales: scales)
        #expect(FragmentTimes.firstPresentation(segment) == [1: 192_000, 2: 93_959])
        #expect(abs(shift - (192_000 - 4082) / 96_000.0) < 1e-12)
        try FragmentTimes.retime(&segment, track: 1, to: 192_001, timescales: scales, othersBy: 0)
        #expect(FragmentTimes.firstPresentation(segment) == [1: 192_001, 2: 93_959])
        // Before zero can't be written.
        #expect(throws: FragmentTimes.Failure.self) { try FragmentTimes.retime(&segment, track: 1, to: 0, timescales: scales) }
    }

    static let scales: [UInt32: UInt32] = [1: 90_000, 2: 48_000]

    static func moof(sequence: UInt32, _ trafs: [UInt8]) -> [UInt8] {
        M.box("moof", M.box("mfhd", [0, 0, 0, 0] + M.be32(sequence)) + trafs) + M.box("mdat", [0, 0])
    }

    /// The sequence numbers of a segment's mfhds, in order.
    static func sequences(_ data: Data) -> [UInt32] {
        FragmentTimes.boxes(data, data.startIndex ..< data.endIndex).filter { $0.0 == "moof" }.flatMap { moof in
            FragmentTimes.boxes(data, moof.1).filter { $0.0 == "mfhd" }.map { FragmentTimes.u32(data, $0.1.lowerBound + 4) }
        }
    }

    /// Two moofs of two trafs each, tracks interleaved: the first sample of
    /// each track is its first traf's, and every traf moves, not just the
    /// first one of each track.
    @Test func retimesEveryFragmentOfASegmentWithSeveralMoofs() throws {
        var segment = Data(Self.moof(sequence: 1, Self.traf(track: 1, decodeTime: 9_000, version: 1, compositionOffset: 3_000)
                                         + Self.traf(track: 2, decodeTime: 4_000, version: 0, compositionOffset: nil))
                           + Self.moof(sequence: 2, Self.traf(track: 2, decodeTime: 28_000, version: 0, compositionOffset: nil)
                                         + Self.traf(track: 1, decodeTime: 54_000, version: 1, compositionOffset: -1_500)))
        #expect(FragmentTimes.firstPresentation(segment) == [1: 12_000, 2: 4_000])
        // Video to 1 s + 12 000 ticks: 1 s later, and the audio a second too.
        let shift = try FragmentTimes.retime(&segment, track: 1, to: 102_000, timescales: Self.scales)
        #expect(shift == 1)
        #expect(FragmentTimes.fragments(segment).map { [$0.track, UInt32($0.decodeTime)] }
                == [[1, 99_000], [2, 52_000], [2, 76_000], [1, 144_000]])
        // Each track to its own time, the other left alone (as the segmenter does).
        try FragmentTimes.retime(&segment, track: 2, to: 52_010, timescales: Self.scales, othersBy: 0)
        #expect(FragmentTimes.fragments(segment).map { [$0.track, UInt32($0.decodeTime)] }
                == [[1, 99_000], [2, 52_010], [2, 76_010], [1, 144_000]])
        #expect(FragmentTimes.fragments(segment).map(\.tfdtVersion) == [1, 0, 0, 1])
        // A track the timescales don't name is left as it is.
        var unnamed = segment
        try FragmentTimes.retime(&unnamed, track: 1, to: 90_000, timescales: [1: 90_000], othersBy: 99)
        #expect(FragmentTimes.fragments(unnamed).map { UInt32($0.decodeTime) } == [87_000, 52_010, 76_010, 132_000])
        // No samples of the track asked for: an error, not a guess.
        #expect(throws: FragmentTimes.Failure.self) { try FragmentTimes.retime(&segment, track: 3, to: 0, timescales: Self.scales) }
    }

    /// A 32-bit tfdt holds 24.8 h at 48 kHz but 13.3 h at 90 kHz: past it,
    /// the segment is refused rather than wrapped to a time near zero, and a
    /// 64-bit one takes the same time.
    @Test func refusesATimeA32BitTfdtCantHold() throws {
        var segment = Data(Self.moof(sequence: 1, Self.traf(track: 1, decodeTime: 0, version: 1, compositionOffset: nil)
                                         + Self.traf(track: 2, decodeTime: 0, version: 0, compositionOffset: nil)))
        let late = Int64(UInt32.max) + 1
        #expect(throws: FragmentTimes.Failure.self) {
            try FragmentTimes.retime(&segment, track: 1, to: late * 2, timescales: Self.scales)
        }
        // Right at the limit still fits.
        var fits = Data(Self.moof(sequence: 1, Self.traf(track: 2, decodeTime: 0, version: 0, compositionOffset: nil)))
        try FragmentTimes.retime(&fits, track: 2, to: Int64(UInt32.max), timescales: Self.scales)
        #expect(FragmentTimes.firstPresentation(fits) == [2: Int64(UInt32.max)])
        var wide = Data(Self.moof(sequence: 1, Self.traf(track: 1, decodeTime: 0, version: 1, compositionOffset: nil)))
        try FragmentTimes.retime(&wide, track: 1, to: late * 2, timescales: Self.scales)
        #expect(FragmentTimes.firstPresentation(wide) == [1: late * 2])
    }

    /// Segment n's fragments count on from n + 1, whatever the run numbered
    /// them, so segments from different runs read as one stream.
    @Test func renumbersFragmentsAfterTheSegment() {
        var one = Data(Self.moof(sequence: 1, Self.traf(track: 1, decodeTime: 0, version: 1, compositionOffset: nil)))
        FragmentTimes.renumber(&one, segment: 41)
        #expect(Self.sequences(one) == [42])
        var two = Data(Self.moof(sequence: 1, Self.traf(track: 1, decodeTime: 0, version: 1, compositionOffset: nil))
                       + Self.moof(sequence: 2, Self.traf(track: 2, decodeTime: 0, version: 0, compositionOffset: nil)))
        let before = FragmentTimes.fragments(two).map(\.decodeTime)
        FragmentTimes.renumber(&two, segment: 0)
        #expect(Self.sequences(two) == [1, 2])
        FragmentTimes.renumber(&two, segment: 7)
        #expect(Self.sequences(two) == [8, 9])
        #expect(FragmentTimes.fragments(two).map(\.decodeTime) == before)
    }
}
