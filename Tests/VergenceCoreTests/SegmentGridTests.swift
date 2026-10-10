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
}
