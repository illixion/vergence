#if canImport(Metal)
import CoreVideo
import Foundation
import Metal
import Testing
@testable import VergenceApple

/// `--antialias`'s pass: a line drawn as stairs four pixels tall becomes
/// straight again, a flat or noisy area is left alone, and the two eyes of a
/// packed frame don't bleed into each other.
@Suite("Edge-tangent anti-aliasing", .enabled(if: MTLCreateSystemDefaultDevice() != nil))
struct EdgeAntialiasTests {
    static let width = 480, height = 160

    static func frame(_ fill: (Int, Int) -> Float) -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        let attributes: [String: Any] = [kCVPixelBufferMetalCompatibilityKey as String: true,
                                         kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any]()]
        CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA, attributes as CFDictionary, &buffer)
        CVPixelBufferLockBaseAddress(buffer!, [])
        let base = CVPixelBufferGetBaseAddress(buffer!)!.assumingMemoryBound(to: UInt8.self)
        for y in 0 ..< height {
            for x in 0 ..< width {
                let p = base + y * CVPixelBufferGetBytesPerRow(buffer!) + 4 * x
                let v = UInt8((min(max(fill(x, y), 0), 1) * 255).rounded())
                p[0] = v; p[1] = v; p[2] = v; p[3] = 255
            }
        }
        CVPixelBufferUnlockBaseAddress(buffer!, [])
        return buffer!
    }

    static func grey(_ buffer: CVPixelBuffer, _ x: Int, _ y: Int) -> Float {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
        return Float(base[y * CVPixelBufferGetBytesPerRow(buffer) + 4 * x + 1]) / 255
    }

    /// Where the edge crosses half brightness in column `x`, sub-pixel.
    static func edge(_ buffer: CVPixelBuffer, _ x: Int) -> Float {
        for y in 1 ..< height {
            let a = grey(buffer, x, y - 1), b = grey(buffer, x, y)
            if (a - 0.5) * (b - 0.5) <= 0, a != b { return Float(y - 1) + (0.5 - a) / (b - a) + 0.5 }
        }
        return .nan
    }

    /// RMS distance of the edge from the straight line it stands for, over the middle columns.
    static func wobble(_ buffer: CVPixelBuffer, line: (Float) -> Float) -> Float {
        let xs = Array(60 ..< width - 60)
        let sq = xs.map { x -> Float in let d = edge(buffer, x) - line(Float(x) + 0.5); return d * d }
        return (sq.reduce(0, +) / Float(xs.count)).squareRoot()
    }

    @Test func stairsBecomeTheLineTheyStandFor() throws {
        // y = 40 + 0.15 x, drawn the way a line-doubled field draws it: in steps 4 pixels tall.
        let line: (Float) -> Float = { 40 + 0.15 * $0 }
        let picture = Self.frame { x, y in
            let stepped = (line(Float(x) + 0.5) / 4).rounded(.down) * 4 + 2
            return Float(y) + 0.5 < stepped ? 0.2 : 0.8
        }
        let before = Self.wobble(picture, line: line)
        let aa = try EdgeAntialias()
        try aa.apply(picture, eyes: [.init(x: 0, y: 0, width: Self.width, height: Self.height)])
        let after = Self.wobble(picture, line: line)
        #expect(before > 1)
        #expect(after < before * 0.25, "edge RMS off the line \(before) → \(after) px")
    }

    @Test func textureAndTheOtherEyeAreLeftAlone() throws {
        // Left eye: hash noise (no straight edge). Right eye: flat black.
        func noise(_ x: Int, _ y: Int) -> Float {
            var h = UInt32(truncatingIfNeeded: x &* 374_761_393 &+ y &* 668_265_263)
            h = (h ^ (h >> 13)) &* 1_274_126_177
            return Float(h & 255) / 255
        }
        let half = Self.width / 2
        let picture = Self.frame { x, y in x < half ? noise(x, y) : 0 }
        try EdgeAntialias().apply(picture, eyes: [.init(x: 0, y: 0, width: half, height: Self.height),
                                                   .init(x: half, y: 0, width: half, height: Self.height)])
        var changed = 0
        for y in 0 ..< Self.height {
            for x in 0 ..< half where abs(Self.grey(picture, x, y) - noise(x, y)) > 2.0 / 255 { changed += 1 }
            for x in half ..< Self.width { #expect(Self.grey(picture, x, y) == 0) }
        }
        #expect(changed < half * Self.height / 50, "\(changed) noise pixels changed")
    }
}
#endif
