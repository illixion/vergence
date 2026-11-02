#if canImport(Metal)
import CoreVideo
import Foundation
import Metal
import Testing
@testable import VergenceApple

/// What the guard must do: where the two real frames agree and the in-between
/// frame strays (an interpolator shifting still stripes by half their
/// spacing), that part becomes the blend of the real frames; where things
/// move, or where the in-between frame is right, it is left alone.
@Suite("Static-region guard", .enabled(if: MTLCreateSystemDefaultDevice() != nil))
struct StaticRegionGuardTests {
    static let width = 128, height = 96

    /// A BGRA frame whose pixel (x, y) is the grey `level(x, y)`.
    static func frame(width: Int = width, height: Int = height, _ level: (Int, Int) -> UInt8) -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        let attributes: [String: Any] = [kCVPixelBufferMetalCompatibilityKey as String: true,
                                         kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any]()]
        CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA, attributes as CFDictionary, &buffer)
        CVPixelBufferLockBaseAddress(buffer!, [])
        let base = CVPixelBufferGetBaseAddress(buffer!)!.assumingMemoryBound(to: UInt8.self)
        let stride = CVPixelBufferGetBytesPerRow(buffer!)
        for y in 0 ..< height {
            for x in 0 ..< width {
                let v = level(x, y)
                let p = base + y * stride + 4 * x
                p[0] = v; p[1] = v; p[2] = v; p[3] = 255
            }
        }
        CVPixelBufferUnlockBaseAddress(buffer!, [])
        return buffer!
    }

    static func level(_ buffer: CVPixelBuffer, _ x: Int, _ y: Int) -> Int {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let p = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
        return Int(p[y * CVPixelBufferGetBytesPerRow(buffer) + 4 * x + 1])
    }

    /// Mean |a - b| over the box, in levels.
    static func difference(_ a: CVPixelBuffer, _ b: CVPixelBuffer, x: Range<Int>, y: Range<Int>) -> Double {
        var total = 0
        for yy in y { for xx in x { total += abs(level(a, xx, yy) - level(b, xx, yy)) } }
        return Double(total) / Double(x.count * y.count)
    }

    /// Vertical blinds: 8 px slats, light and dark, shifted right by `shift`.
    static func blinds(shift: Int) -> (Int, Int) -> UInt8 {
        { x, _ in ((x - shift + 64) / 4) % 2 == 0 ? 200 : 60 }
    }

    static func guarded() throws -> StaticRegionGuard {
        let staticGuard = try #require(StaticRegionGuard())
        #expect(staticGuard.prepare(width: width, height: height))
        return staticGuard
    }

    @Test func stillStripesShiftedOnlyInTheInBetweenFrameArePutBack() throws {
        let staticGuard = try Self.guarded()
        let still = Self.frame(Self.blinds(shift: 0))
        // The interpolator's mistake: the slats shifted by half their spacing
        // inside a patch (x 32-96, y 16-80), right elsewhere.
        let mid = Self.frame { x, y in
            (32 ..< 96).contains(x) && (16 ..< 80).contains(y) ? Self.blinds(shift: 4)(x, y) : Self.blinds(shift: 0)(x, y)
        }
        #expect(Self.difference(mid, still, x: 32 ..< 96, y: 16 ..< 80) > 100)
        let out = staticGuard.apply(previous: still, current: still, mid: mid)
        #expect(Self.difference(out, still, x: 32 ..< 96, y: 16 ..< 80) < 1)
        #expect(Self.difference(out, still, x: 0 ..< Self.width, y: 0 ..< Self.height) < 1)
    }

    @Test func movingStripesKeepTheInterpolatorsFrame() throws {
        let staticGuard = try Self.guarded()
        // The slats move 4 px between the real frames; the in-between frame
        // has them halfway, which the blend of the two would smear.
        let previous = Self.frame(Self.blinds(shift: 0))
        let current = Self.frame(Self.blinds(shift: 4))
        let mid = Self.frame(Self.blinds(shift: 2))
        let out = staticGuard.apply(previous: previous, current: current, mid: mid)
        #expect(Self.difference(out, mid, x: 0 ..< Self.width, y: 0 ..< Self.height) == 0)
    }

    @Test func aRightInBetweenFrameOfAStillPictureIsUnchanged() throws {
        let staticGuard = try Self.guarded()
        let still = Self.frame(Self.blinds(shift: 0))
        // Within a few levels of the blend (noise, compression): not wrong.
        let mid = Self.frame { x, y in Self.blinds(shift: 0)(x, y) &+ UInt8((x + y) % 3) }
        let out = staticGuard.apply(previous: still, current: still, mid: mid)
        #expect(Self.difference(out, mid, x: 0 ..< Self.width, y: 0 ..< Self.height) == 0)
    }

    @Test func aFrameOfAnotherSizePassesAsItIs() throws {
        let staticGuard = try Self.guarded()
        let small = Self.frame(width: 64, height: 48, Self.blinds(shift: 0))
        let mid = Self.frame(width: 64, height: 48, Self.blinds(shift: 4))
        #expect(staticGuard.apply(previous: small, current: small, mid: mid) === mid)
    }

    /// Wall time of one guard pass on a 1080p frame (printed, not asserted).
    @Test func costAt1080p() throws {
        let staticGuard = try #require(StaticRegionGuard())
        #expect(staticGuard.prepare(width: 1920, height: 1080))
        let a = Self.frame(width: 1920, height: 1080, Self.blinds(shift: 0))
        let m = Self.frame(width: 1920, height: 1080, Self.blinds(shift: 4))
        _ = staticGuard.apply(previous: a, current: a, mid: m)
        var times: [Double] = []
        for _ in 0 ..< 30 {
            let started = Date()
            _ = staticGuard.apply(previous: a, current: a, mid: m)
            times.append(Date().timeIntervalSince(started) * 1000)
        }
        print(String(format: "static-region guard at 1080p: %.2f ms median", times.sorted()[times.count / 2]))
    }
}
#endif
