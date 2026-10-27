#if canImport(Metal)
import CoreGraphics
import Foundation
import Metal
import Testing
@testable import VergenceApple

@Suite("Still stereo GPU", .serialized)
struct StillStereoRendererTests {
    static let width = 256, height = 128
    func renderer() throws -> StillStereoRenderer {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let path = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Sources/VergenceApple/Stereo/StereoShaders.metal")
        let library = try device.makeLibrary(source: String(contentsOf: path, encoding: .utf8), options: nil)
        return try #require(StillStereoRenderer(device: device, library: library))
    }
    func source(redRange: Range<Int> = 110..<146) throws -> CGImage {
        var pixels = [UInt8](repeating: 0, count: Self.width * Self.height * 4)
        for y in 0..<Self.height { for x in 0..<Self.width {
            let i = (y * Self.width + x) * 4
            pixels[i] = redRange.contains(x) ? 255 : 0
            pixels[i + 1] = y < Self.height / 2 ? 180 : 40
            pixels[i + 2] = redRange.contains(x) ? 0 : 255
            pixels[i + 3] = 255
        } }
        let provider = try #require(CGDataProvider(data: Data(pixels) as CFData))
        return try #require(CGImage(width: Self.width, height: Self.height, bitsPerComponent: 8,
            bitsPerPixel: 32, bytesPerRow: Self.width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue), provider: provider,
            decode: nil, shouldInterpolate: false, intent: .defaultIntent))
    }
    func pixels(_ t: MTLTexture) throws -> [Float16] {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float,
            width: t.width, height: t.height, mipmapped: false)
        descriptor.storageMode = .shared
        let readable = try #require(t.device.makeTexture(descriptor: descriptor))
        let command = try #require(t.device.makeCommandQueue()?.makeCommandBuffer())
        let blit = try #require(command.makeBlitCommandEncoder())
        blit.copy(from: t, to: readable); blit.endEncoding(); command.commit(); command.waitUntilCompleted()
        #expect(command.status == .completed)
        var out = [Float16](repeating: 0, count: t.width * t.height * 4)
        readable.getBytes(&out, bytesPerRow: t.width * 8, from: MTLRegionMake2D(0, 0, t.width, t.height), mipmapLevel: 0)
        return out
    }
    @Test func nearestSubjectStaysOnGlassInBothEyes() throws {
        let map = StillDepthMap(width: Self.width, height: Self.height, normalizedValues:
            (0..<(Self.width * Self.height)).map { (110..<146).contains($0 % Self.width) ? 1 : 0 })
        let pair = try renderer().render(image: source(), depth: map, strength: 0.02, refine: false)
        for eye in [pair.left, pair.right] {
            let values = try pixels(eye)
            let row = Self.height / 2
            let red = (0..<Self.width).filter { values[(row * Self.width + $0) * 4] > 0.8 }
            #expect(abs((red.first ?? 0) - 110) <= 2)
            #expect(abs((red.last ?? 0) - 145) <= 2)
        }
    }
    @Test func zeroDisparityPreservesOrientationAndEyeEquality() throws {
        let map = StillDepthMap(width: 64, height: 32, values: .init(repeating: 128, count: 64 * 32))
        let pair = try renderer().render(image: source(), depth: map, strength: 0)
        let left = try pixels(pair.left), right = try pixels(pair.right)
        #expect(left == right)
        #expect(left[(10 * Self.width + 30) * 4 + 1] > left[(110 * Self.width + 30) * 4 + 1])
        #expect(left[(64 * Self.width + 128) * 4] > 0.8)
    }
    @Test func mirrorHappensBeforeDisparitySoDepthDoesNotInvert() throws {
        // An off-centre red marker at the far plane. In the left eye it must
        // still shift left after mirroring (uncrossed/behind), never right.
        let original = try source(redRange: 70..<106)
        let map = StillDepthMap(width: 64, height: 32, values: .init(repeating: 0, count: 64 * 32))
        let engine = try renderer()
        let pair = try engine.render(image: original, depth: map, strength: 0.04, refine: false, mirror: true)
        let left = try pixels(pair.left), right = try pixels(pair.right)
        func redCenter(_ values: [Float16]) -> Double {
            let xs = (20..<(Self.width - 20)).filter { values[(64 * Self.width + $0) * 4] > 0.8 }
            return Double(xs.reduce(0, +)) / Double(max(xs.count, 1))
        }
        #expect(redCenter(left) < redCenter(right) - 10)
        #expect(redCenter(left) > 150 && redCenter(left) < 164)
        #expect(redCenter(right) > 170 && redCenter(right) < 184)
    }

    @Test func extendedRangeHighlightsSurviveWarp() throws {
        let width = 16, height = 8
        let floats = [Float16](repeating: 2, count: width * height * 4)
        let bytes = floats.withUnsafeBytes { Data($0) }
        let provider = try #require(CGDataProvider(data: bytes as CFData))
        let image = try #require(CGImage(width: width, height: height, bitsPerComponent: 16,
            bitsPerPixel: 64, bytesPerRow: width * 8,
            space: CGColorSpace(name: CGColorSpace.extendedLinearDisplayP3)!,
            bitmapInfo: [.floatComponents, .byteOrder16Little, CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue)],
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        let map = StillDepthMap(width: 8, height: 4, values: .init(repeating: 128, count: 32))
        let pair = try renderer().render(image: image, depth: map, strength: 0)
        let values = try pixels(pair.left)
        #expect(values[(4 * width + 8) * 4] > 1.5)
    }

    @Test func floatingDepthRetainsMoreThanEightBitDetail() throws {
        let map = try #require(StillDepthExtraction.normalizedMap(
            (values: (0..<4096).map { Float($0) }, width: 64, height: 64),
            imageWidth: 64, imageHeight: 64, preservePrecision: true))
        let values = try #require(map.preciseValues)
        #expect(Set(values).count > 3000)
        #expect(Set(map.values).count <= 256)
        #expect(values.allSatisfy { $0.isFinite && $0 >= 0 && $0 <= 1 })
    }

    @Test func malformedDepthIsRejected() throws {
        let engine = try renderer(), image = try source()
        #expect(throws: StillStereoRenderer.Failure.self) {
            try engine.render(image: image, depth: StillDepthMap(width: 5, height: 5, values: [0]))
        }
    }
}
#endif
