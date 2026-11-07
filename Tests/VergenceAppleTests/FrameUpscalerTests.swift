#if canImport(CoreML) && canImport(Metal)
import CoreML
import CoreVideo
import Foundation
import Testing
@testable import VergenceApple

/// A tiny network of the compact Real-ESRGAN shape (3→4→4→48 channels, PReLUs,
/// pixel shuffle by 4) saved by torch.save, with PyTorch's own 2× output for a
/// 24×16 picture (`tiny-srvgg-*`): the checkpoint read, the written Core ML
/// specification and the kernels around it have to reproduce it.
@Suite struct FrameUpscalerTests {
    static func fixture(_ name: String) throws -> URL {
        try #require(Bundle.module.url(forResource: "Fixtures/\(name)", withExtension: nil))
    }

    static func frame(width: Int, height: Int, rgb: [UInt8]?) throws -> CVPixelBuffer {
        var out: CVPixelBuffer?
        CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA,
                            [kCVPixelBufferIOSurfacePropertiesKey: [String: Any](), kCVPixelBufferMetalCompatibilityKey: true] as CFDictionary, &out)
        let buffer = try #require(out)
        CVPixelBufferLockBaseAddress(buffer, [])
        let row = CVPixelBufferGetBytesPerRow(buffer)
        let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
        for y in 0 ..< height {
            for x in 0 ..< width {
                let p = base + y * row + x * 4
                let i = (y * width + x) * 3
                p[0] = rgb?[i + 2] ?? 0; p[1] = rgb?[i + 1] ?? 0; p[2] = rgb?[i] ?? 0; p[3] = 255
            }
        }
        CVPixelBufferUnlockBaseAddress(buffer, [])
        return buffer
    }

    static func pixels(_ buffer: CVPixelBuffer, x0: Int, y0: Int, width: Int, height: Int) -> [Float] {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let row = CVPixelBufferGetBytesPerRow(buffer)
        let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
        var out: [Float] = []
        for y in y0 ..< y0 + height {
            for x in x0 ..< x0 + width {
                let p = base + y * row + x * 4
                out += [Float(p[2]), Float(p[1]), Float(p[0])].map { $0 / 255 }
            }
        }
        return out
    }

    @Test func readsTheNetworkFromItsCheckpoint() throws {
        let network = try UpscaleNetwork(checkpoint: Self.fixture("tiny-srvgg.pth"))
        #expect(network.scale == 4)
        #expect(network.convolutions == 3)
        let spec = network.specification(batch: 2, width: 24, height: 16)
        #expect(spec.count > 2000)
    }

    @Test(arguments: [[MLComputeUnits.cpuAndGPU], [.cpuAndNeuralEngine], [.cpuAndNeuralEngine, .cpuAndGPU]])
    func matchesPyTorch(units: [MLComputeUnits]) async throws {
        let network = try UpscaleNetwork(checkpoint: Self.fixture("tiny-srvgg.pth"))
        let input = [UInt8](try Data(contentsOf: Self.fixture("tiny-srvgg-input.rgb")))
        let expected = try Data(contentsOf: Self.fixture("tiny-srvgg-output.f32")).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        // Both eyes side by side in one frame, out into a frame twice the size,
        // once in one piece and once cut into tiles.
        var packed = [UInt8](repeating: 0, count: 48 * 16 * 3)
        for y in 0 ..< 16 {
            for x in 0 ..< 48 { for c in 0 ..< 3 { packed[(y * 48 + x) * 3 + c] = input[(y * 24 + x % 24) * 3 + c] } }
        }
        let source = try Self.frame(width: 48, height: 16, rgb: packed)
        for maxTile in [FrameUpscaler.tilePixels, 100] {
            let upscaler = try await FrameUpscaler(network: network, width: 24, height: 16, eyes: 2, units: units, fuseEyes: false,
                                                  maxTilePixels: maxTile)
            #expect((upscaler.grid.x * upscaler.grid.y > 1) == (maxTile == 100))
            let destination = try Self.frame(width: 96, height: 32, rgb: nil)
            try upscaler.upscale(source, regions: [.init(x: 0, y: 0, width: 24, height: 16), .init(x: 24, y: 0, width: 24, height: 16)],
                                 into: destination, at: [(0, 0), (48, 0)])
            for x0 in [0, 48] {
                let got = Self.pixels(destination, x0: x0, y0: 0, width: 48, height: 32)
                let worst = zip(got, expected).map { abs($0 - $1) }.max() ?? 1
                #expect(worst < 3.0 / 255, "eye at \(x0), tiles \(upscaler.grid): \(worst * 255) levels off")
            }
        }
    }

    /// A 2D picture is cut in two to share between the Neural Engine and the
    /// GPU; the seam mustn't show.
    @Test func splitsA2DFrameBetweenTwoUnits() async throws {
        let network = try UpscaleNetwork(checkpoint: Self.fixture("tiny-srvgg.pth"))
        let (w, h) = (40, 72)
        var rgb = [UInt8](repeating: 0, count: w * h * 3)
        for i in rgb.indices { rgb[i] = UInt8((i * 37 + (i / (w * 3)) * 11) % 251) }
        let source = try Self.frame(width: w, height: h, rgb: rgb)
        var results: [[Float]] = []
        for units: [MLComputeUnits] in [[.cpuAndGPU], [.cpuAndNeuralEngine, .cpuAndGPU]] {
            let upscaler = try await FrameUpscaler(network: network, width: w, height: h, eyes: 1, units: units)
            #expect(upscaler.grid.y == units.count)
            let destination = try Self.frame(width: w * 2, height: h * 2, rgb: nil)
            try upscaler.upscale(source, regions: [.init(x: 0, y: 0, width: Double(w), height: Double(h))], into: destination, at: [(0, 0)])
            results.append(Self.pixels(destination, x0: 0, y0: 0, width: w * 2, height: h * 2))
        }
        let worst = zip(results[0], results[1]).map { abs($0 - $1) }.max() ?? 1
        #expect(worst < 3.0 / 255, "\(worst * 255) levels apart")
    }
}
#endif
