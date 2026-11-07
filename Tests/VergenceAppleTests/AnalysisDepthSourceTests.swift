#if canImport(Metal)
import CoreMedia
import CoreVideo
import Foundation
import Metal
import Testing
@testable import VergenceApple
@testable import VergenceCore

/// AI 3D "warp only": a worker's depth track is the pump's depth for every
/// frame it holds, mapped to the values the warp reads; frames it doesn't
/// hold go to the device's own source.
@Suite("Analysis depth source", .enabled(if: MTLCreateSystemDefaultDevice() != nil))
struct AnalysisDepthSourceTests {
    final class CountingFallback: PumpDepthSource, @unchecked Sendable {
        var asked: [Double] = []
        let flattensWhenUnavailable = true
        let prefersDenseWarpGrid = false
        func frameDepth(itemTime: CMTime, frame: CVPixelBuffer) -> PumpFrameDepth? {
            asked.append(itemTime.seconds)
            return nil
        }
        func invalidate() {}
    }

    static func track() throws -> AnalysisTrackIndex {
        var data = try AnalysisTrack.header(AnalysisTrackHeader(mediaID: "t", producer: "test", models: ["depth": "m"],
                                                                sourceWidth: 64, sourceHeight: 36))
        // Frames 0 and 1 of a 24 fps title: a left-to-right ramp, shot space 0.2…1.2.
        for k in 0 ..< 2 {
            let values = (0 ..< 8 * 4).map { i in 0.2 + Float(i % 8) / 7 }
            let record = AnalysisRecord.encoding(values, kind: .depth, as: .uint8, channels: 1, width: 8, height: 4,
                                                 time: AnalysisTime(value: Int64(k), timescale: 24), range: 0.2 ... 1.2)
            data.append(AnalysisTrack.record(try record.deflated()))
        }
        return try AnalysisTrackIndex(data)
    }

    @Test func framesTheTrackHoldsComeFromIt() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let fallback = CountingFallback()
        let source = AnalysisDepthSource(lookup: try Self.track(), frameRate: 24, device: device, fallback: fallback,
                                         inflate: AnalysisDeflate.inflate)
        var buffer: CVPixelBuffer?
        CVPixelBufferCreate(nil, 64, 36, kCVPixelFormatType_32BGRA, nil, &buffer)
        let frame = try #require(buffer)

        // Frame 1, a millisecond off: the track's map, identity UV, values back in shot space.
        let depth = try #require(source.frameDepth(itemTime: CMTime(seconds: 1.0 / 24 + 0.001, preferredTimescale: 90_000),
                                                   frame: frame))
        #expect(depth.texture.width == 8 && depth.texture.height == 4 && depth.budget == .studio)
        #expect(depth.uvScale == SIMD2(1, 1) && depth.uvOffset == SIMD2(0, 0))
        var bytes = [UInt8](repeating: 0, count: 8)
        depth.texture.getBytes(&bytes, bytesPerRow: 8, from: MTLRegionMake2D(0, 0, 8, 1), mipmapLevel: 0)
        let first = Float(bytes[0]) / 255 * depth.valueScale + depth.valueBias
        let last = Float(bytes[7]) / 255 * depth.valueScale + depth.valueBias
        #expect(abs(first - 0.2) < 0.01 && abs(last - 1.2) < 0.01, "\(first) \(last)")
        #expect(fallback.asked.isEmpty)

        // Frame 5 isn't in the track yet: the device's own source has it.
        _ = source.frameDepth(itemTime: CMTime(value: 5, timescale: 24), frame: frame)
        #expect(fallback.asked.count == 1 && source.trackFrames == 1 && source.fallbackFrames == 1)
    }
}
#endif
