#if canImport(Metal) && canImport(VideoToolbox)
import CoreVideo
import Foundation
import Metal
import Testing
@testable import VergenceApple
@testable import VergenceCore

/// The player's Metal pass: a packed top/bottom pair in, the in-between pair
/// out, from stored fields. What has to hold: motion is applied at its stored
/// scale in x and y (a 16:9 eye's field grid is not 16:9), halfway, and the
/// right eye reads the left eye's field across by its disparity, toward the
/// left eye's position of the same point.
@Suite("Shared-field interpolator", .enabled(if: MTLCreateSystemDefaultDevice() != nil))
struct SharedFieldInterpolatorTests {
    static let eyeWidth = 64, eyeHeight = 32

    final class NoProvider: SharedFieldProvider, @unchecked Sendable {
        func fields(from start: Double, to end: Double) -> SharedFields? { nil }
    }

    /// A packed frame whose pixel (x, y) in either eye is `pixel(eye, x, y)` as (B, G, R).
    static func frame(_ pixel: (Int, Int, Int) -> (UInt8, UInt8, UInt8)) -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        let attributes: [String: Any] = [kCVPixelBufferMetalCompatibilityKey as String: true,
                                         kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any]()]
        CVPixelBufferCreate(nil, eyeWidth, 2 * eyeHeight, kCVPixelFormatType_32BGRA, attributes as CFDictionary, &buffer)
        CVPixelBufferLockBaseAddress(buffer!, [])
        let base = CVPixelBufferGetBaseAddress(buffer!)!.assumingMemoryBound(to: UInt8.self)
        let stride = CVPixelBufferGetBytesPerRow(buffer!)
        for row in 0 ..< 2 * eyeHeight {
            for x in 0 ..< eyeWidth {
                let (b, g, r) = pixel(row / eyeHeight, x, row % eyeHeight)
                let p = base + row * stride + 4 * x
                p[0] = b; p[1] = g; p[2] = r; p[3] = 255
            }
        }
        CVPixelBufferUnlockBaseAddress(buffer!, [])
        return buffer!
    }

    static func read(_ buffer: CVPixelBuffer, eye: Int, x: Int, y: Int) -> (Int, Int, Int) {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let p = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
            + (eye * eyeHeight + y) * CVPixelBufferGetBytesPerRow(buffer) + 4 * x
        return (Int(p[0]), Int(p[1]), Int(p[2]))
    }

    static func interpolator() throws -> SharedFieldInterpolator {
        let interpolator = try #require(SharedFieldInterpolator(provider: NoProvider()))
        #expect(interpolator.prepare(width: eyeWidth, height: 2 * eyeHeight, scale: 1))
        return interpolator
    }

    @Test func motionIsHalvedAndScaledPerAxis() throws {
        // A ramp in x (blue) and y (green) moving (+8, +4) eye pixels; the
        // field grid is 16×16, so a stored pixel is 4 eye pixels across and 2 down.
        func ramp(_ x: Int, _ y: Int) -> (UInt8, UInt8, UInt8) {
            (UInt8(clamping: 4 * x), UInt8(clamping: 8 * y), 0)
        }
        let a = Self.frame { _, x, y in ramp(x, y) }
        let b = Self.frame { _, x, y in ramp(x - 8, y - 4) }
        let n = 16 * 16
        let flow = [Float](repeating: 2, count: n) + [Float](repeating: 2, count: n)       // F0→1
            + [Float](repeating: -2, count: n) + [Float](repeating: -2, count: n)          // F1→0
            + [Float](repeating: 0.5, count: n)
        let fields = SharedFields(flow: flow, flowWidth: 16, flowHeight: 16,
                                  disparity0: [Float](repeating: 0, count: 64), disparity1: [Float](repeating: 0, count: 64),
                                  disparityWidth: 16, disparityHeight: 4)
        let mid = try #require(try Self.interpolator().midFrame(between: a, and: b, fields: fields))
        for eye in 0 ..< 2 {
            for (x, y) in [(12, 6), (30, 16), (50, 25)] {
                let (blue, green, _) = Self.read(mid, eye: eye, x: x, y: y)
                #expect(abs(blue - 4 * (x - 4)) <= 1 && abs(green - 8 * (y - 2)) <= 1,
                        "eye \(eye) (\(x), \(y)): \(blue), \(green)")
            }
        }
    }

    @Test func theRightEyeReadsTheLeftEyesFieldAcrossItsDisparity() throws {
        // No motion; frame 0 is all 200, frame 1 all 40, and the mask takes
        // frame 0 in the left half of the left eye's field only. A right
        // pixel x is the left eye's x + 16 (disparity 4 stored pixels).
        let a = Self.frame { _, _, _ in (200, 200, 200) }
        let b = Self.frame { _, _, _ in (40, 40, 40) }
        let n = 16 * 8
        let mask = (0 ..< n).map { Float($0 % 16 < 8 ? 1 : 0) }
        let fields = SharedFields(flow: [Float](repeating: 0, count: 4 * n) + mask, flowWidth: 16, flowHeight: 8,
                                  disparity0: [Float](repeating: 4, count: n), disparity1: [Float](repeating: 4, count: n),
                                  disparityWidth: 16, disparityHeight: 8)
        let mid = try #require(try Self.interpolator().midFrame(between: a, and: b, fields: fields))
        #expect(Self.read(mid, eye: 0, x: 20, y: 10).0 == 200)
        #expect(Self.read(mid, eye: 1, x: 20, y: 10).0 == 40)    // reads the mask at 36
        #expect(Self.read(mid, eye: 1, x: 8, y: 10).0 == 200)    // reads it at 24
        #expect(Self.read(mid, eye: 0, x: 50, y: 10).0 == 40)
    }

    @Test func withoutFieldsOrTimesItMakesNothing() async throws {
        let interpolator = try Self.interpolator()
        let a = Self.frame { _, _, _ in (0, 0, 0) }
        let made = await withCheckedContinuation { done in
            interpolator.midFrame(between: a, at: 0, and: a, at: 1.0 / 24) { frame, _ in done.resume(returning: frame != nil) }
        }
        #expect(!made)
        let untimed = await withCheckedContinuation { done in
            interpolator.midFrame(between: a, and: a) { frame, _ in done.resume(returning: frame != nil) }
        }
        #expect(!untimed)
    }
}
#endif
