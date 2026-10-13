#if canImport(Metal)
import Foundation
import Metal
import Testing
@testable import VergenceApple

private let width = 768, height = 320

/// These compile and execute the real shaders even under `swift test`, where
/// SwiftPM normally omits Metal resources and older GPU tests silently return.
@Suite("Disconnected stereo boundaries", .serialized)
struct StereoBoundaryWarpTests {

    struct Fixture {
        let device: MTLDevice
        let queue: MTLCommandQueue
        let warp: StereoBoundaryWarp
        let source: MTLTexture
        let depth: MTLTexture
        let output: MTLTexture
        let positions: MTLBuffer
        let indices: MTLBuffer
        let count: Int

        init(depthValue: (Int, Int) -> Float, foreground: (Int, Int) -> Bool) throws {
            let device = try #require(MTLCreateSystemDefaultDevice())
            self.device = device
            queue = try #require(device.makeCommandQueue())
            let path = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
                .deletingLastPathComponent().appendingPathComponent("Sources/VergenceApple/Stereo/RAVEStereoShaders.metal")
            let library = try device.makeLibrary(source: String(contentsOf: path, encoding: .utf8), options: nil)
            warp = try #require(StereoBoundaryWarp(device: device, library: library))
            func texture(_ format: MTLPixelFormat) throws -> MTLTexture {
                let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format,
                    width: width, height: height, mipmapped: false)
                d.storageMode = .shared
                d.usage = [.shaderRead, .shaderWrite, .renderTarget]
                return try #require(device.makeTexture(descriptor: d))
            }
            source = try texture(.rgba8Unorm)
            depth = try texture(.r32Float)
            output = try texture(.bgra8Unorm)
            var colors = [UInt8](repeating: 255, count: width * height * 4)
            var values = [Float](repeating: 0, count: width * height)
            for y in 0..<height {
                for x in 0..<width {
                    let i = (y * width + x) * 4
                    // Red subject, visibly textured green/blue background.
                    colors[i] = foreground(x, y) ? 255 : 0
                    colors[i + 1] = foreground(x, y) ? 0 : UInt8(80 + (x % 16) * 8)
                    colors[i + 2] = foreground(x, y) ? 0 : 180
                    values[y * width + x] = depthValue(x, y)
                }
            }
            source.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                           withBytes: colors, bytesPerRow: width * 4)
            depth.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                          withBytes: values, bytesPerRow: width * 4)
            let nx = 385, ny = 161
            var vertices: [SIMD2<Float>] = []
            var triangles: [UInt32] = []
            for y in 0..<ny {
                for x in 0..<nx { vertices.append(SIMD2(Float(x) / Float(nx - 1), Float(y) / Float(ny - 1))) }
            }
            for y in 0..<(ny - 1) {
                for x in 0..<(nx - 1) {
                    let i = UInt32(y * nx + x), row = UInt32(nx)
                    triangles.append(contentsOf: [i, i + row, i + 1, i + 1, i + row, i + row + 1])
                }
            }
            positions = try #require(device.makeBuffer(bytes: vertices, length: vertices.count * MemoryLayout<SIMD2<Float>>.stride))
            indices = try #require(device.makeBuffer(bytes: triangles, length: triangles.count * MemoryLayout<UInt32>.stride))
            count = triangles.count
        }

        func render(eye: Float, strength: Float = 0.02, mirror: Bool = false,
                    holes: Bool = false, uv: SIMD4<Float> = SIMD4(1, 1, 0, 0)) throws -> [UInt8] {
            let command = try #require(queue.makeCommandBuffer())
            let u = StereoBoundaryWarp.Uniforms(stereo: SIMD4(strength, 1, eye, mirror ? 1 : 0),
                uv: uv, mapping: SIMD4(1, 0, 0, 1), adjustments: SIMD4(0, 1, 1, 0),
                boundary: SIMD4(0.08, Float(width), 1 / 384, holes ? 1 : 0))
            #expect(warp.encode(source: source, depth: depth, destination: output,
                positions: positions, indices: indices, indexCount: count, uniforms: u, commandBuffer: command))
            command.commit()
            command.waitUntilCompleted()
            #expect(command.status == .completed, "\(String(describing: command.error))")
            var result = [UInt8](repeating: 0, count: width * height * 4)
            output.getBytes(&result, bytesPerRow: width * 4,
                            from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)
            for i in stride(from: 0, to: result.count, by: 4) { result.swapAt(i, i + 2) }
            return result
        }
    }

    @Test(arguments: [Float(-1), Float(1)])
    func subjectIsNotStretchedAndHolesContainOnlyBackground(eye: Float) throws {
        let f = try Fixture(depthValue: { x, _ in (300..<460).contains(x) ? 1 : 0 },
                            foreground: { x, _ in (300..<460).contains(x) })
        let out = try f.render(eye: eye)
        let y = height / 2
        let red = (0..<width).filter { out[(y * width + $0) * 4] > 200 }
        #expect(abs((red.first ?? 0) - 300) <= 2)
        #expect(abs((red.last ?? 0) - 459) <= 2)
        // Every non-subject pixel has background blue; there are no black
        // holes and no foreground-coloured streak across the uncovered strip.
        for x in 0..<width where x < 298 || x > 461 {
            #expect(out[(y * width + x) * 4] < 10)
            #expect(out[(y * width + x) * 4 + 2] > 150)
        }
        let mask = try f.render(eye: eye, holes: true)
        let holeCount = (0..<width).filter {
            let i = (y * width + $0) * 4
            return mask[i] == 255 && mask[i + 2] == 255
        }.count
        #expect(holeCount > 10 && holeCount < 45)
    }

    @Test(arguments: [Float(-1), Float(1)])
    func thinRailKeepsItsWidth(eye: Float) throws {
        let f = try Fixture(depthValue: { x, _ in (380..<388).contains(x) ? 1 : 0 },
                            foreground: { x, _ in (380..<388).contains(x) })
        let out = try f.render(eye: eye)
        let reds = (370..<400).filter { out[(height / 2 * width + $0) * 4] > 200 }
        #expect(reds.count >= 4 && reds.count <= 8)
    }

    @Test func zeroStrengthPreservesPictureAcrossDepthEdges() throws {
        let f = try Fixture(depthValue: { x, _ in (300..<460).contains(x) ? 1 : 0 },
                            foreground: { x, _ in (300..<460).contains(x) })
        let out = try f.render(eye: 1, strength: 0)
        for x in 0..<width {
            let i = (height / 2 * width + x) * 4
            #expect(out[i] == ((300..<460).contains(x) ? 255 : 0))
            #expect(out[i + 3] == 255)
        }
    }

    @Test(arguments: [Float(-1), Float(1)])
    func horizontalRailHasNoBlackRow(eye: Float) throws {
        let f = try Fixture(depthValue: { _, y in (140..<160).contains(y) ? 1 : 0 },
                            foreground: { _, y in (140..<160).contains(y) })
        let out = try f.render(eye: eye)
        for y in 130..<170 {
            for x in stride(from: 10, to: width - 10, by: 7) {
                let i = (y * width + x) * 4
                #expect(out[i] > 200 || out[i + 2] > 150)
            }
        }
    }

    @Test func smoothSlopeHasNoInteriorHoles() throws {
        let f = try Fixture(depthValue: { x, _ in Float(x) / Float(width) }, foreground: { _, _ in false })
        let mask = try f.render(eye: 1, holes: true)
        for x in 30..<(width - 30) {
            #expect(mask[(height / 2 * width + x) * 4] == 0)
        }
    }

    @Test func mirrorPreservesSubjectAndLetterboxRemap() throws {
        let f = try Fixture(depthValue: { x, _ in (100..<260).contains(x) ? 1 : 0 },
                            foreground: { x, _ in (100..<260).contains(x) })
        let out = try f.render(eye: -1, mirror: true)
        let reds = (0..<width).filter { out[(height / 2 * width + $0) * 4] > 200 }
        #expect(abs((reds.first ?? 0) - (width - 260)) <= 2)
        #expect(abs((reds.last ?? 0) - (width - 101)) <= 2)
        // Constant depth across rows: a scale/offset remap must not change
        // horizontal geometry, even when the model contains letterbox bars.
        let remapped = try f.render(eye: -1, mirror: true, uv: SIMD4(1, 0.6, 0, 0.2))
        #expect(out == remapped)
    }
}
#endif
