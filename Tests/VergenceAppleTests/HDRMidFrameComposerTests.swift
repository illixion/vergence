#if canImport(Metal) && canImport(VideoToolbox)
import CoreVideo
import Foundation
import Metal
import Testing
import VideoToolbox
@testable import VergenceApple

/// HDR in-between frames: the 8-bit interpolator's motion over the real
/// frames' full-precision blend (`HDRMidFrameComposer`).
struct HDRMidFrameComposerTests {
    static let width = 256
    static let height = 64

    /// A half-float RGBA frame tagged PQ / BT.2020, its code value per pixel.
    static func frame(_ value: (Int, Int) -> Float) -> CVPixelBuffer {
        var made: CVPixelBuffer?
        CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_64RGBAHalf, [
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
        ] as CFDictionary, &made)
        let buffer = made!
        CVBufferSetAttachment(buffer, kCVImageBufferColorPrimariesKey, kCVImageBufferColorPrimaries_ITU_R_2020, .shouldPropagate)
        CVBufferSetAttachment(buffer, kCVImageBufferTransferFunctionKey, kCVImageBufferTransferFunction_SMPTE_ST_2084_PQ, .shouldPropagate)
        CVBufferSetAttachment(buffer, kCVImageBufferYCbCrMatrixKey, kCVImageBufferYCbCrMatrix_ITU_R_2020, .shouldPropagate)
        CVPixelBufferLockBaseAddress(buffer, [])
        let base = CVPixelBufferGetBaseAddress(buffer)!
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        for y in 0..<height {
            let row = (base + y * stride).assumingMemoryBound(to: Float16.self)
            for x in 0..<width {
                let v = Float16(value(x, y))
                row[x * 4] = v; row[x * 4 + 1] = v; row[x * 4 + 2] = v; row[x * 4 + 3] = 1
            }
        }
        CVPixelBufferUnlockBaseAddress(buffer, [])
        return buffer
    }

    static func value(_ buffer: CVPixelBuffer, _ x: Int, _ y: Int, channel: Int = 1) -> Float {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let row = (CVPixelBufferGetBaseAddress(buffer)! + y * CVPixelBufferGetBytesPerRow(buffer))
            .assumingMemoryBound(to: Float16.self)
        return Float(row[x * 4 + channel])
    }

    /// The 420v frame VideoToolbox's interpolator would see.
    static func yuv(_ rgb: CVPixelBuffer) -> CVPixelBuffer {
        var made: CVPixelBuffer?
        CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange, [
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
        ] as CFDictionary, &made)
        let out = made!
        for key in [kCVImageBufferColorPrimariesKey, kCVImageBufferTransferFunctionKey, kCVImageBufferYCbCrMatrixKey] {
            CVBufferSetAttachment(out, key, CVBufferCopyAttachment(rgb, key, nil)!, .shouldPropagate)
        }
        var session: VTPixelTransferSession?
        VTPixelTransferSessionCreate(allocator: nil, pixelTransferSessionOut: &session)
        #expect(VTPixelTransferSessionTransferImage(session!, from: rgb, to: out) == noErr)
        return out
    }

    static func compose(_ previous: CVPixelBuffer, _ current: CVPixelBuffer, mid: CVPixelBuffer) throws -> CVPixelBuffer {
        let composer = try #require(HDRMidFrameComposer())
        let done = DispatchSemaphore(value: 0)
        final class Box: @unchecked Sendable { var buffer: CVPixelBuffer? }
        let box = Box()
        composer.compose(previous: previous, current: current, previousYUV: yuv(previous), currentYUV: yuv(current),
                         midYUV: yuv(mid)) { out in
            box.buffer = out
            done.signal()
        }
        done.wait()
        return try #require(box.buffer)
    }

    /// A gradient far finer than an 8-bit step (0.2 over 256 pixels: 0.0008 a
    /// pixel, where 8 bits step 0.0039).
    static func gradient(_ x: Int, _ y: Int) -> Float { 0.35 + 0.2 * Float(x) / Float(width) }

    @Test func aStillPictureKeepsEveryLevel() throws {
        let picture = Self.frame(Self.gradient)
        let mid = try Self.compose(picture, picture, mid: picture)
        var levels = Set<Float>()
        var worst: Float = 0
        for x in 0..<Self.width {
            let v = Self.value(mid, x, 30)
            levels.insert(v)
            worst = max(worst, abs(v - Self.value(picture, x, 30)))
        }
        // Float16 near 0.5 steps 0.00024 (about 4 per 8-bit step): every
        // pixel keeps its own level, as the real frame has.
        #expect(worst < 0.0005)
        #expect(levels.count > 150)
        #expect(CVPixelBufferGetPixelFormatType(mid) == kCVPixelFormatType_64RGBAHalf)
        let transfer = CVBufferCopyAttachment(mid, kCVImageBufferTransferFunctionKey, nil) as? String
        #expect(transfer == (kCVImageBufferTransferFunction_SMPTE_ST_2084_PQ as String))
    }

    @Test func whatTheInterpolatorChangedIsAddedToTheBlend() throws {
        // A fade: the interpolator's frame is halfway, so the composed frame
        // is too, within an 8-bit step, and still has its fine levels.
        let previous = Self.frame(Self.gradient)
        let current = Self.frame { Self.gradient($0, $1) + 0.1 }
        let halfway = Self.frame { Self.gradient($0, $1) + 0.05 }
        let mid = try Self.compose(previous, current, mid: halfway)
        var worst: Float = 0
        for x in 0..<Self.width { worst = max(worst, abs(Self.value(mid, x, 30) - Self.value(halfway, x, 30))) }
        #expect(worst < 2.0 / 255)
    }

    @Test func motionComesFromTheInterpolator() throws {
        // A bright bar moving 16 pixels: the interpolator put it halfway, so
        // the composed frame shows it there, not two half-bright bars.
        let bar: (Int) -> (Int, Int) -> Float = { at in { x, y in (at..<(at + 16)).contains(x) ? 0.75 : Self.gradient(x, y) } }
        let previous = Self.frame(bar(96))
        let current = Self.frame(bar(112))
        let halfway = Self.frame(bar(104))
        let mid = try Self.compose(previous, current, mid: halfway)
        #expect(abs(Self.value(mid, 106, 30) - 0.75) < 0.03)     // where only the moved bar is
        #expect(abs(Self.value(mid, 98, 30) - Self.gradient(98, 30)) < 0.03) // uncovered, background again
    }

    @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *)
    @Test func videoToolboxInBetweenFramesOfHDRStayHalfFloatAndFine() throws {
        guard VTMidFrameInterpolator.isSupported else { return }
        let interpolator = VTMidFrameInterpolator(guardsResults: true)
        guard interpolator.prepare(width: Self.width, height: Self.height, scale: 1) else { return }
        let picture = Self.frame(Self.gradient)
        let done = DispatchSemaphore(value: 0)
        final class Box: @unchecked Sendable { var buffer: CVPixelBuffer? }
        let box = Box()
        interpolator.midFrame(between: picture, and: picture) { out, _ in
            box.buffer = out
            done.signal()
        }
        done.wait()
        let mid = try #require(box.buffer)
        #expect(CVPixelBufferGetPixelFormatType(mid) == kCVPixelFormatType_64RGBAHalf)
        var levels = Set<Float>()
        var worst: Float = 0
        // Inside the picture: VideoToolbox's frame strays by a couple of
        // levels at its last few columns (measured: 1.7 at x = 252 of 256).
        for x in 8..<(Self.width - 8) {
            levels.insert(Self.value(mid, x, 30))
            worst = max(worst, abs(Self.value(mid, x, 30) - Self.value(picture, x, 30)))
        }
        #expect(levels.count > 140, "an 8-bit in-between frame would have about 50 levels here")
        #expect(worst < 1.0 / 255, "worst \(worst * 255) of 255")
    }
}
#endif
