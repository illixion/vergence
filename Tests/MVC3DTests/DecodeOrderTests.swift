import CryptoKit
import Foundation
import Testing
@testable import MVC3D

/// `Fixtures/x264-ref4-bframes2.264`: 40 frames of ffmpeg's testsrc2,
/// 160x96, from
///
///     ffmpeg -f lavfi -i testsrc2=size=160x96:rate=24 -frames:v 40 -c:v libx264 \
///       -x264-params ref=4:bframes=2:b-pyramid=0:b-adapt=0:scenecut=0:keyint=40:aud=1 \
///       -pix_fmt yuv420p -video_track_timescale 24 small.mp4
///     ffmpeg -i small.mp4 -c copy -bsf:v h264_mp4toannexb -f h264 x264-ref4-bframes2.264
///
/// Four reference frames in a DPB of four (`max_dec_frame_buffering` 4) with
/// B-frames: upstream edge264 counted the reference the sliding window was
/// about to drop and output each P-frame ahead of its B-frames.
private let fixture = Bundle.module.url(forResource: "x264-ref4-bframes2", withExtension: "264", subdirectory: "Fixtures")!

/// Display index of each access unit, in decode order (the MP4's PTS).
private let displayOrder: [Int64] = [0, 3, 1, 2, 6, 4, 5, 9, 7, 8, 12, 10, 11, 15, 13, 14, 18, 16, 17, 21, 19, 20,
                                     24, 22, 23, 27, 25, 26, 30, 28, 29, 33, 31, 32, 36, 34, 35, 39, 37, 38]

/// MD5 of `ffmpeg -i x264-ref4-bframes2.264 -f rawvideo -`.
private let ffmpegOutputMD5 = "3d6ed4ba3af3747a248702c41a5a4983"

/// The stream's access units, split at their delimiters.
private func accessUnits() throws -> [Data] {
    let data = try Data(contentsOf: fixture)
    let starts = AnnexB.nalUnits(in: data).filter { $0.type == 9 }.map { $0.range.lowerBound - 3 }
    return zip(starts, starts.dropFirst() + [data.endIndex]).map { data[$0 ..< $1] }
}

@Test(arguments: [0, -1])
func framesComeOutInDisplayOrderWithTheirTimestamps(threads: Int) throws {
    let units = try accessUnits()
    #expect(units.count == displayOrder.count)
    let decoder = try MVCDecoder(threads: threads)
    var timestamps: [Int64?] = []
    var md5 = Insecure.MD5()
    let collect: (StereoFrame) -> Void = { frame in
        timestamps.append(frame.pts)
        for plane in frame.left {
            for y in 0 ..< plane.height { md5.update(bufferPointer: UnsafeRawBufferPointer(plane.row(y))) }
        }
    }
    for (unit, pts) in zip(units, displayOrder) {
        try decoder.decode(StereoAccessUnit(pts: pts, dts: nil, data: unit, hasDependentView: false), output: collect)
    }
    try decoder.finish(output: collect)
    #expect(timestamps == (0 ..< 40).map { $0 })
    #expect(md5.finalize().map { String(format: "%02x", $0) }.joined() == ffmpegOutputMD5)
}
