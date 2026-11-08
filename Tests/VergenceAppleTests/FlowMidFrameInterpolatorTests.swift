#if canImport(Metal)
import CoreVideo
import Foundation
import Metal
import Testing
@testable import VergenceApple

/// What the GPU interpolator must do: a picture that moves between two frames
/// is shown halfway in the in-between frame (not as the two blended), a still
/// one comes out unchanged, and an object moving over a still background
/// lands halfway with the background around it intact.
@Suite("Flow mid-frame interpolator", .enabled(if: MTLCreateSystemDefaultDevice() != nil))
struct FlowMidFrameInterpolatorTests {
    static let width = 320, height = 192

    /// Smooth random texture (a fixed seed), so block matching has detail to
    /// lock on to, as real pictures do. Sampled at any (x, y).
    static let texture: [UInt8] = {
        var state: UInt64 = 0x9E37_79B9_7F4A_7C15
        func next() -> UInt8 {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return UInt8(truncatingIfNeeded: state >> 56)
        }
        // 64×64 random, box-blurred 3×3 so neighbouring pixels correlate.
        let side = 64
        let raw = (0 ..< side * side).map { _ in Int(next()) }
        var blurred = [UInt8](repeating: 0, count: side * side)
        for y in 0 ..< side {
            for x in 0 ..< side {
                var sum = 0
                for dy in -1 ... 1 { for dx in -1 ... 1 { sum += raw[((y + dy + side) % side) * side + (x + dx + side) % side] } }
                blurred[y * side + x] = UInt8(sum / 9)
            }
        }
        return blurred
    }()

    /// The texture at (x, y), tiled, each texel 3 px wide.
    static func pattern(_ x: Int, _ y: Int) -> UInt8 {
        let tx = ((x / 3) % 64 + 64) % 64, ty = ((y / 3) % 64 + 64) % 64
        return texture[ty * 64 + tx]
    }

    static func frame(_ level: (Int, Int) -> UInt8) -> CVPixelBuffer {
        InterpolationGuardTests.frame(width: width, height: height, level)
    }

    static func interpolator() throws -> FlowMidFrameInterpolator {
        let flow = try #require(FlowMidFrameInterpolator())
        #expect(flow.prepare(width: width, height: height, scale: 1))
        return flow
    }

    /// Mean |a - b| in levels over the frame inside a margin (the edges see
    /// texture come in from outside the frame, which no interpolator knows).
    static func difference(_ a: CVPixelBuffer, _ b: CVPixelBuffer, margin: Int = 40) -> Double {
        InterpolationGuardTests.difference(a, b, x: margin ..< width - margin, y: margin ..< height - margin)
    }

    @Test("a pan is shown halfway, far closer than the blend",
          arguments: [(8, 0), (6, 4), (24, -10), (50, 16)])
    func panLandsHalfway(motion: (Int, Int)) throws {
        let flow = try Self.interpolator()
        let (mx, my) = motion
        let previous = Self.frame { x, y in Self.pattern(x, y) }
        let current = Self.frame { x, y in Self.pattern(x - mx, y - my) }
        let halfway = Self.frame { x, y in Self.pattern(x - mx / 2, y - my / 2) }
        let blend = Self.frame { x, y in UInt8((Int(Self.pattern(x, y)) + Int(Self.pattern(x - mx, y - my))) / 2) }
        let mid = try #require(flow.midFrame(between: previous, and: current))
        let error = Self.difference(mid, halfway)
        let blendError = Self.difference(blend, halfway)
        #expect(error < 2.5, "motion \(motion): \(error) levels off halfway")
        #expect(error < blendError / 4, "motion \(motion): \(error) vs the blend's \(blendError)")
    }

    @Test func aStillPictureIsUnchanged() throws {
        let flow = try Self.interpolator()
        let still = Self.frame { x, y in Self.pattern(x, y) }
        let mid = try #require(flow.midFrame(between: still, and: still))
        #expect(Self.difference(mid, still, margin: 0) < 0.5)
    }

    @Test("an object moving over a still background lands halfway, the background intact")
    func objectOverStillBackground() throws {
        let flow = try Self.interpolator()
        // A 64 px square of another texture moving 20 px right.
        func scene(_ shift: Int) -> (Int, Int) -> UInt8 {
            { x, y in
                let inside = (100 + shift ..< 164 + shift).contains(x) && (64 ..< 128).contains(y)
                return inside ? Self.pattern(x - shift + 97, y + 41) : Self.pattern(x, y) / 2
            }
        }
        let previous = Self.frame(scene(0))
        let current = Self.frame(scene(20))
        let halfway = Self.frame(scene(10))
        let mid = try #require(flow.midFrame(between: previous, and: current))
        // The square itself (inside its halfway place, away from its edges).
        let square = InterpolationGuardTests.difference(mid, halfway, x: 118 ..< 166, y: 72 ..< 120)
        // The background well away from the square.
        let background = InterpolationGuardTests.difference(mid, halfway, x: 200 ..< 300, y: 20 ..< 172)
        #expect(square < 3, "square \(square)")
        #expect(background < 1, "background \(background)")
        // Everything, the uncovered and covered strips included: still well
        // under the blend there.
        let blend = Self.frame { x, y in UInt8((Int(scene(0)(x, y)) + Int(scene(20)(x, y))) / 2) }
        let around = 90 ..< 190
        let all = InterpolationGuardTests.difference(mid, halfway, x: around, y: 56 ..< 136)
        let blendAll = InterpolationGuardTests.difference(blend, halfway, x: around, y: 56 ..< 136)
        #expect(all < blendAll / 2, "around the square \(all) vs the blend's \(blendAll)")
    }

    @Test func aFrameOfAnotherSizeFails() throws {
        let flow = try Self.interpolator()
        let small = InterpolationGuardTests.frame(width: 64, height: 64) { _, _ in 100 }
        #expect(flow.midFrame(between: small, and: small) == nil)
    }

    @Test func costAt1080p() throws {
        let flow = try #require(FlowMidFrameInterpolator())
        #expect(flow.prepare(width: 1920, height: 1080, scale: 1))
        let previous = InterpolationGuardTests.frame(width: 1920, height: 1080) { x, y in Self.pattern(x, y) }
        let current = InterpolationGuardTests.frame(width: 1920, height: 1080) { x, y in Self.pattern(x - 12, y - 4) }
        var times: [Double] = []
        for _ in 0 ..< 12 {
            #expect(flow.midFrame(between: previous, and: current) != nil)
            times.append(flow.lastGPUMilliseconds)
        }
        let median = times.sorted()[times.count / 2]
        print("flow interpolation at 1080p: GPU \(String(format: "%.2f", median)) ms median")
        #expect(median < 8)
    }

    /// A private texture's channels as floats (a blit to a shared copy).
    static func read(_ texture: MTLTexture, channels: Int) throws -> [Float] {
        let device = texture.device
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: texture.pixelFormat, width: texture.width, height: texture.height, mipmapped: false)
        descriptor.storageMode = .shared
        let shared = try #require(device.makeTexture(descriptor: descriptor))
        let queue = try #require(device.makeCommandQueue())
        let commands = try #require(queue.makeCommandBuffer())
        let blit = try #require(commands.makeBlitCommandEncoder())
        blit.copy(from: texture, to: shared)
        blit.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        let count = texture.width * texture.height * channels
        if texture.pixelFormat == .r32Float {
            var values = [Float](repeating: 0, count: count)
            shared.getBytes(&values, bytesPerRow: texture.width * 4,
                            from: MTLRegionMake2D(0, 0, texture.width, texture.height), mipmapLevel: 0)
            return values
        }
        var halves = [Float16](repeating: 0, count: count)
        shared.getBytes(&halves, bytesPerRow: texture.width * 2 * channels,
                        from: MTLRegionMake2D(0, 0, texture.width, texture.height), mipmapLevel: 0)
        return halves.map(Float.init)
    }

    @Test("the motion it writes is half the pan, at half size")
    func motionIsHalfThePan() throws {
        let flow = try Self.interpolator()
        let previous = Self.frame { x, y in Self.pattern(x, y) }
        let current = Self.frame { x, y in Self.pattern(x - 12, y - 6) }
        let done = DispatchSemaphore(value: 0)
        nonisolated(unsafe) var motion: MidFrameMotion?
        flow.midFrame(between: previous, and: current, wantsMotion: true) { _, found, _ in
            motion = found
            done.signal()
        }
        done.wait()
        let found = try #require(motion)
        #expect(found.texture.width == Self.width / 2 && found.texture.height == Self.height / 2)
        let values = try Self.read(found.texture, channels: 2)
        var sum = SIMD2<Float>(0, 0), count: Float = 0
        for y in 20 ..< found.texture.height - 20 {
            for x in 20 ..< found.texture.width - 20 {
                let i = (y * found.texture.width + x) * 2
                sum += SIMD2(values[i], values[i + 1])
                count += 1
            }
        }
        let mean = sum / count
        // mid(x) = previous(x - d): the pan moves content by +(12, 6), so d is +(6, 3).
        #expect(abs(mean.x - 6) < 0.5 && abs(mean.y - 3) < 0.5, "mean motion \(mean)")
    }

    @Test("a held map moved along the motion lands with the picture")
    func depthFollowsMotion() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let mover = try #require(MidFrameDepth(device: device))
        // A 64×32 map over a 320×192 frame: near (1) left of x = 32, far (0) right of it.
        let mapWidth = 64, mapHeight = 32
        var values = [Float](repeating: 0, count: mapWidth * mapHeight)
        for y in 0 ..< mapHeight { for x in 0 ..< 32 { values[y * mapWidth + x] = 1 } }
        let mapDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r32Float, width: mapWidth, height: mapHeight, mipmapped: false)
        mapDescriptor.storageMode = .shared
        let map = try #require(device.makeTexture(descriptor: mapDescriptor))
        map.replace(region: MTLRegionMake2D(0, 0, mapWidth, mapHeight), mipmapLevel: 0,
                    withBytes: values, bytesPerRow: mapWidth * 4)
        // Everything moved right 50 frame pixels (10 map texels) by the in-between frame.
        let motionDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rg16Float, width: Self.width / 2, height: Self.height / 2, mipmapped: false)
        motionDescriptor.storageMode = .shared
        let motionTexture = try #require(device.makeTexture(descriptor: motionDescriptor))
        let shift = [Float16](repeating: 0, count: Self.width / 2 * Self.height / 2 * 2).enumerated()
            .map { $0.offset % 2 == 0 ? Float16(50) : 0 }
        motionTexture.replace(region: MTLRegionMake2D(0, 0, Self.width / 2, Self.height / 2), mipmapLevel: 0,
                              withBytes: shift, bytesPerRow: Self.width / 2 * 4)
        let motion = MidFrameMotion(texture: motionTexture, frameWidth: Self.width, frameHeight: Self.height,
                                    pool: nil)
        let depth = PumpFrameDepth(texture: map, uvScale: SIMD2(1, 1), uvOffset: .zero, valueScale: 1, valueBias: 0)
        let queue = try #require(device.makeCommandQueue())
        let commands = try #require(queue.makeCommandBuffer())
        let moved = try #require(mover.encode(depth, along: motion, into: commands))
        commands.commit()
        commands.waitUntilCompleted()
        let out = try Self.read(moved.texture, channels: 1)
        let row = Array(out[(mapHeight / 2) * mapWidth ..< (mapHeight / 2 + 1) * mapWidth])
        // The edge moved from texel 32 to 42.
        #expect(row[38] > 0.9 && row[40] > 0.9, "near side: \(row[36 ..< 46])")
        #expect(row[44] < 0.1 && row[50] < 0.1, "far side: \(row[36 ..< 46])")
        #expect(row[10] > 0.9 && row[60] < 0.1)
    }
}
#endif
