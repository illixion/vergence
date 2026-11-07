#if canImport(Metal)
import CoreVideo
import Foundation
import Metal
import Testing
@testable import VergenceApple
@testable import VergenceCore

/// The player's AI HDR pass: a pixel's gain is the grid read at its own
/// (u, v, luma), as `GainGrid.lookup` says, on linear light; the frame's
/// average gain is held to the cap.
@Suite("Gain grid pass", .enabled(if: MTLCreateSystemDefaultDevice() != nil))
struct GainGridPassTests {
    static let width = 64, height = 36

    static func buffer(_ format: OSType, _ fill: ((Int, Int) -> UInt8)? = nil) -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        let attributes: [String: Any] = [kCVPixelBufferMetalCompatibilityKey as String: true,
                                         kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any]()]
        CVPixelBufferCreate(nil, width, height, format, attributes as CFDictionary, &buffer)
        if let fill {
            CVPixelBufferLockBaseAddress(buffer!, [])
            let base = CVPixelBufferGetBaseAddress(buffer!)!.assumingMemoryBound(to: UInt8.self)
            for y in 0 ..< height {
                for x in 0 ..< width {
                    let p = base + y * CVPixelBufferGetBytesPerRow(buffer!) + 4 * x
                    let v = fill(x, y)
                    p[0] = v; p[1] = v; p[2] = v; p[3] = 255
                }
            }
            CVPixelBufferUnlockBaseAddress(buffer!, [])
        }
        return buffer!
    }

    static func red(_ buffer: CVPixelBuffer, _ x: Int, _ y: Int) -> Float {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let p = (CVPixelBufferGetBaseAddress(buffer)! + y * CVPixelBufferGetBytesPerRow(buffer))
            .assumingMemoryBound(to: UInt16.self)
        return HalfFloat.value(from: p[4 * x])
    }

    @Test func eachPixelGetsTheGridsGainForItsOwnLuma() throws {
        // Gain rises with luma and with x: g = l / 4 + x / 30.
        var values = [Float](repeating: 0, count: GainGrid.count)
        for l in 0 ..< 8 { for y in 0 ..< 9 { for x in 0 ..< 16 {
            values[GainGrid.index(x, y, l)] = Float(l) / 4 + Float(x) / 30
        } } }
        let grid = GainGrid(log2Gain: values)
        let pass = try #require(GainGridPass())
        // A grey ramp across the frame.
        let sdr = Self.buffer(kCVPixelFormatType_32BGRA) { x, _ in UInt8(40 + 3 * x) }
        let out = Self.buffer(kCVPixelFormatType_64RGBAHalf)
        #expect(pass.apply(grid, to: sdr, into: out, cap: 100))
        #expect(pass.lastScale == 1)
        for (x, y) in [(3, 5), (31, 18), (60, 30)] {
            let c = Float(40 + 3 * x) / 255
            let expected = pow(c, 2.4) * exp2(grid.lookup(u: (Float(x) + 0.5) / Float(Self.width),
                                                          v: (Float(y) + 0.5) / Float(Self.height), luma: c))
            let got = Self.red(out, x, y)
            #expect(abs(got - expected) / expected < 0.01, "(\(x), \(y)): \(got) vs \(expected)")
        }
    }

    @Test func theFramesAverageGainIsHeldToTheCap() throws {
        // ×4 everywhere, capped at ×2: every log gain is halved.
        let grid = GainGrid(log2Gain: [Float](repeating: 2, count: GainGrid.count))
        let pass = try #require(GainGridPass())
        let sdr = Self.buffer(kCVPixelFormatType_32BGRA) { _, _ in 128 }
        let out = Self.buffer(kCVPixelFormatType_64RGBAHalf)
        #expect(pass.apply(grid, to: sdr, into: out, cap: 2))
        #expect(abs(pass.lastMeanGain - 4) < 0.01 && abs(pass.lastScale - 0.5) < 0.001)
        let expected = pow(Float(128) / 255, 2.4) * 2
        #expect(abs(Self.red(out, 10, 10) - expected) / expected < 0.01)
    }

    @Test func aiThreeDEyesBecomeBT2020PQWithTheRampAndCap() throws {
        // ×2 everywhere; a mid-grey SDR eye (0.5) → 0.5^2.4 × 2 × 203 nits, in BT.2020 (grey stays grey), PQ.
        let grid = GainGrid(log2Gain: [Float](repeating: 1, count: GainGrid.count))
        let pass = try #require(GainGridPass())
        let sdr = Self.buffer(kCVPixelFormatType_32BGRA) { _, _ in 128 }
        func pq(_ nits: Float) -> Float {
            let m1: Float = 0.1593017578125, m2: Float = 78.84375, c1: Float = 0.8359375, c2: Float = 18.8515625, c3: Float = 18.6875
            let y = pow(nits / 10000, m1)
            return pow((c1 + c2 * y) / (1 + c3 * y), m2)
        }
        let grey = Float(128) / 255
        for (strength, gain) in [(Float(1), Float(2)), (0.5, sqrt(2)), (0, 1)] {
            let eye = Self.buffer(kCVPixelFormatType_64RGBAHalf)
            Self.fillHalf(eye, grey)
            #expect(pass.applyPQ(grid, source: sdr, eye: eye, cap: 3, strength: strength))
            let expected = pq(pow(grey, 2.4) * gain * GainGridPass.sdrWhiteNits)
            #expect(abs(Self.red(eye, 20, 20) - expected) < 0.003, "strength \(strength)")
        }
    }

    static func fillHalf(_ buffer: CVPixelBuffer, _ value: Float) {
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        for y in 0 ..< height {
            let row = (CVPixelBufferGetBaseAddress(buffer)! + y * CVPixelBufferGetBytesPerRow(buffer))
                .assumingMemoryBound(to: UInt16.self)
            for x in 0 ..< 4 * width { row[x] = HalfFloat.bits(from: x % 4 == 3 ? 1 : value) }
        }
    }
}
#endif
