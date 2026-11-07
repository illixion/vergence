#if os(visionOS)
import AVFoundation
import CoreMedia
import CoreVideo
import Foundation
import Metal
import Testing
@testable import VergenceApple
@testable import VergenceCore

/// StereoPump is visionOS-only and this test target doesn't build for the
/// visionOS simulator as a whole, so this suite was run from a scratch
/// package that depends on Vergence by path (`xcodebuild test` on a visionOS
/// simulator, 2026-11-07: both pass).
///
/// AI HDR through the stereo pump: an SDR picture with gain grids leaves as
/// BT.2020 PQ eyes, the grid's gain applied (here ×2 everywhere, past its
/// ramp), and without grids it stays SDR.
@Suite("Stereo pump AI HDR", .serialized, .enabled(if: MTLCreateSystemDefaultDevice() != nil && StereoWarpResources.shared != nil))
struct StereoPumpAIHDRTests {
    /// Hands out a grey SDR frame every call, a frame later each time (24 fps).
    final class GreySource: PumpFrameSource, @unchecked Sendable {
        let frame: CVPixelBuffer
        var index: Int64 = 0
        init(grey: UInt8) {
            var buffer: CVPixelBuffer?
            let attributes: [String: Any] = [kCVPixelBufferMetalCompatibilityKey as String: true,
                                             kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any]()]
            CVPixelBufferCreate(nil, 320, 180, kCVPixelFormatType_32BGRA, attributes as CFDictionary, &buffer)
            CVPixelBufferLockBaseAddress(buffer!, [])
            let base = CVPixelBufferGetBaseAddress(buffer!)!.assumingMemoryBound(to: UInt8.self)
            for i in 0 ..< CVPixelBufferGetBytesPerRow(buffer!) * 180 { base[i] = i % 4 == 3 ? 255 : grey }
            CVPixelBufferUnlockBaseAddress(buffer!, [])
            for (key, value) in [(kCVImageBufferColorPrimariesKey, kCVImageBufferColorPrimaries_ITU_R_709_2),
                                 (kCVImageBufferTransferFunctionKey, kCVImageBufferTransferFunction_ITU_R_709_2),
                                 (kCVImageBufferYCbCrMatrixKey, kCVImageBufferYCbCrMatrix_ITU_R_709_2)] {
                CVBufferSetAttachment(buffer!, key, value, .shouldPropagate)
            }
            frame = buffer!
        }
        func nextFrame(hostTime: CFTimeInterval) -> PumpFrame? {
            defer { index += 1 }
            return PumpFrame(pixelBuffer: frame, itemTime: CMTime(value: index, timescale: 24))
        }
        func invalidate() {}
    }

    static func gains(log2: Float) throws -> GainGridPlayback {
        var data = try AnalysisTrack.header(AnalysisTrackHeader(mediaID: "t", producer: "test", models: [:],
                                                                sourceWidth: 320, sourceHeight: 180))
        for k in 0 ..< 2000 {
            data.append(AnalysisTrack.record(GainGrid(log2Gain: [Float](repeating: log2, count: GainGrid.count))
                .record(time: AnalysisTime(value: Int64(k), timescale: 24))))
        }
        return GainGridPlayback(lookup: try AnalysisTrackIndex(data), frameRate: 24, cap: 3,
                                inflate: { data, _ in data })
    }

    /// Runs a pump until a captured frame at least `afterSeconds` in arrives.
    static func capture(grey: UInt8, gains: GainGridPlayback?, afterSeconds: Double) async throws -> StereoEyeCapture.Frame {
        let warp = try #require(StereoWarpResources.shared)
        let renderer = AVSampleBufferVideoRenderer()
        let made = StereoPump(videoRenderer: renderer, frameSource: GreySource(grey: grey), warp: warp,
                              startHostTime: CACurrentMediaTime(), depthSource: nil,
                              frameInterval: 1.0 / 240, onVideoSizeKnown: { _ in })
        let pump = try #require(made)
        pump.gainGrids = gains
        pump.start()
        defer { pump.stop() }
        let deadline = Date().addingTimeInterval(20)
        while Date() < deadline {
            let generation = StereoEyeCapture.request()
            for _ in 0 ..< 100 {
                if let frame = StereoEyeCapture.latest(after: generation) {
                    if frame.itemSeconds >= afterSeconds { return frame }
                    break
                }
                try await Task.sleep(for: .milliseconds(10))
            }
        }
        throw CancellationError()
    }

    static func centre(_ buffer: CVPixelBuffer) -> (Float, Float, Float) {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let p = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
            + (CVPixelBufferGetHeight(buffer) / 2) * CVPixelBufferGetBytesPerRow(buffer) + 4 * (CVPixelBufferGetWidth(buffer) / 2)
        return (Float(p[2]) / 255, Float(p[1]) / 255, Float(p[0]) / 255)
    }

    static func pq(_ nits: Float) -> Float {
        let m1: Float = 0.1593017578125, m2: Float = 78.84375, c1: Float = 0.8359375, c2: Float = 18.8515625, c3: Float = 18.6875
        let y = pow(nits / 10000, m1)
        return pow((c1 + c2 * y) / (1 + c3 * y), m2)
    }

    @Test func anSDRPictureWithGridsLeavesAsBT2020PQ() async throws {
        let frame = try await Self.capture(grey: 128, gains: try Self.gains(log2: 1), afterSeconds: 1.2)
        // The capture is the eyes' code values, quantised to 8 bits; grey stays grey in BT.2020.
        let expected = Self.pq(pow(Float(128) / 255, 2.4) * 2 * GainGridPass.sdrWhiteNits)
        for eye in [frame.left, frame.right] {
            let (r, g, b) = Self.centre(eye)
            #expect(abs(r - expected) < 0.02 && abs(g - expected) < 0.02 && abs(b - expected) < 0.02,
                    "\(r) \(g) \(b) vs \(expected)")
            let transfer = CVBufferCopyAttachment(eye, kCVImageBufferTransferFunctionKey, nil) as? String
            #expect(transfer == kCVImageBufferTransferFunction_SMPTE_ST_2084_PQ as String)
        }
    }

    @Test func withoutGridsItStaysSDR() async throws {
        let frame = try await Self.capture(grey: 128, gains: nil, afterSeconds: 0.2)
        let (r, _, _) = Self.centre(frame.left)
        #expect(abs(r - Float(128) / 255) < 0.02, "\(r)")
    }
}
#endif
