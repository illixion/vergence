#if canImport(Metal)
import CoreVideo
import Foundation
import Metal
import VergenceCore

/// Puts back what an optical-flow interpolator invents where nothing moved.
///
/// VideoToolbox's low-latency interpolator misreads evenly spaced stripes
/// (window blinds, railings, a striped shirt) that stand still: two stripe
/// positions look alike, so the flow matches the wrong one and the in-between
/// frame shows the stripes shifted by half their spacing, inside an oval patch.
/// Only the in-between frames carry it, so at 48 Hz the patch flickers.
/// Measured on a still shot of out-of-focus blinds: the in-between frames
/// differed from the blend of their neighbours by 5.8 levels (up to 11) while
/// the two real frames differed from each other by 0.4.
///
/// So, per 16×16 tile: where the two real frames agree (nothing moved there)
/// but the in-between frame strays far from them, the tile takes the blend of
/// the two real frames instead, which is the right in-between picture for
/// something that stands still. Tiles with motion keep the interpolator's
/// frame. Weights are widened by a tile (the patch has soft edges) and blended
/// smoothly between tile centres, so no tile edge shows.
///
/// On that shot it took the error from 5.8 to 0.5, replacing about 4.5% of the
/// frame; on a moving shot it replaced 0.7% and left the error unchanged.
public final class StaticRegionGuard: @unchecked Sendable {
    /// Tile side in pixels.
    public static let tile = 16
    /// Below this mean change between the real frames (0-255) a tile counts as
    /// still; it fades out over the next half of it again.
    public static let stillLevel: Float = 3
    /// Above this mean distance from the blend (0-255) a still tile's
    /// in-between frame counts as wrong; fully replaced at twice it.
    public static let errorLevel: Float = 4

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let statsPipeline: MTLComputePipelineState
    private let weightPipeline: MTLComputePipelineState
    private let widenPipeline: MTLComputePipelineState
    private let applyPipeline: MTLComputePipelineState
    private var textureCache: CVMetalTextureCache?
    private var outputPool: CVPixelBufferPool?
    private var width = 0
    private var height = 0
    private var stats: MTLTexture?
    private var weights: MTLTexture?
    private var widened: MTLTexture?

    /// nil without Metal or if the kernels don't compile.
    public init?(device: MTLDevice? = MTLCreateSystemDefaultDevice()) {
        guard let device, let queue = device.makeCommandQueue() else { return nil }
        do {
            let library = try device.makeLibrary(source: Self.source, options: nil)
            func pipeline(_ name: String) throws -> MTLComputePipelineState {
                guard let function = library.makeFunction(name: name) else { throw GuardError.missingKernel }
                return try device.makeComputePipelineState(function: function)
            }
            statsPipeline = try pipeline("guardTileStats")
            weightPipeline = try pipeline("guardTileWeight")
            widenPipeline = try pipeline("guardTileWiden")
            applyPipeline = try pipeline("guardApply")
        } catch {
            VergenceLog.pipeline.error("static-region guard: kernels failed: \(error, privacy: .public)")
            return nil
        }
        self.device = device
        self.queue = queue
    }

    private enum GuardError: Error { case missingKernel }

    /// Sets the frame size (BGRA, all three frames alike).
    public func prepare(width: Int, height: Int) -> Bool {
        var cache: CVMetalTextureCache?
        CVMetalTextureCacheCreate(nil, nil, device, nil, &cache)
        var pool: CVPixelBufferPool?
        let attributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
        ]
        CVPixelBufferPoolCreate(nil, nil, attributes as CFDictionary, &pool)
        let tilesWide = (width + Self.tile - 1) / Self.tile
        let tilesHigh = (height + Self.tile - 1) / Self.tile
        func make(_ format: MTLPixelFormat) -> MTLTexture? {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: format, width: tilesWide, height: tilesHigh, mipmapped: false)
            descriptor.usage = [.shaderRead, .shaderWrite]
            descriptor.storageMode = .private
            return device.makeTexture(descriptor: descriptor)
        }
        guard let cache, let pool, let stats = make(.rg16Float), let weights = make(.r16Float),
              let widened = make(.r16Float) else { return false }
        textureCache = cache
        outputPool = pool
        self.stats = stats
        self.weights = weights
        self.widened = widened
        self.width = width
        self.height = height
        return true
    }

    struct Params {
        var tile: UInt32
        var stillLevel: Float
        var errorLevel: Float
    }

    /// The in-between frame with its wrong still tiles put back, or `mid`
    /// itself when the guard can't run (wrong size or format). `completion`
    /// runs on a Metal completion thread.
    public func apply(previous: CVPixelBuffer, current: CVPixelBuffer, mid: CVPixelBuffer,
                      completion: @escaping @Sendable (CVPixelBuffer) -> Void) {
        let midBox = Box(buffer: mid)
        guard let commands = encode(previous: previous, current: current, mid: mid, completion: completion) else {
            completion(midBox.buffer)
            return
        }
        commands.commit()
    }

    /// Synchronous, for tests and the CLI.
    public func apply(previous: CVPixelBuffer, current: CVPixelBuffer, mid: CVPixelBuffer) -> CVPixelBuffer {
        let done = DispatchSemaphore(value: 0)
        let result = ResultBox()
        apply(previous: previous, current: current, mid: mid) { out in
            result.buffer = out
            done.signal()
        }
        done.wait()
        return result.buffer ?? mid
    }

    private final class ResultBox: @unchecked Sendable { var buffer: CVPixelBuffer? }
    private struct Box: @unchecked Sendable { let buffer: CVPixelBuffer }

    private func encode(previous: CVPixelBuffer, current: CVPixelBuffer, mid: CVPixelBuffer,
                        completion: @escaping @Sendable (CVPixelBuffer) -> Void) -> MTLCommandBuffer? {
        guard width > 0,
              [previous, current, mid].allSatisfy({
                  CVPixelBufferGetWidth($0) == width && CVPixelBufferGetHeight($0) == height
                      && CVPixelBufferGetPixelFormatType($0) == kCVPixelFormatType_32BGRA
              }),
              let pool = outputPool, let stats, let weights, let widened else { return nil }
        var output: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &output)
        guard let output,
              let a = texture(previous), let b = texture(current), let m = texture(mid), let out = texture(output),
              let commands = queue.makeCommandBuffer(),
              let encoder = commands.makeComputeCommandEncoder() else { return nil }
        CVBufferPropagateAttachments(mid, output)

        var params = Params(tile: UInt32(Self.tile), stillLevel: Self.stillLevel, errorLevel: Self.errorLevel)
        let tiles = MTLSize(width: stats.width, height: stats.height, depth: 1)
        let group = MTLSize(width: 8, height: 8, depth: 1)
        func dispatch(_ size: MTLSize) {
            encoder.dispatchThreadgroups(
                MTLSize(width: (size.width + 7) / 8, height: (size.height + 7) / 8, depth: 1),
                threadsPerThreadgroup: group)
        }
        encoder.setBytes(&params, length: MemoryLayout<Params>.stride, index: 0)

        encoder.setComputePipelineState(statsPipeline)
        encoder.setTexture(a.texture, index: 0)
        encoder.setTexture(b.texture, index: 1)
        encoder.setTexture(m.texture, index: 2)
        encoder.setTexture(stats, index: 3)
        encoder.dispatchThreadgroups(tiles, threadsPerThreadgroup: MTLSize(width: Self.tile, height: Self.tile, depth: 1))

        encoder.setComputePipelineState(weightPipeline)
        encoder.setTexture(stats, index: 0)
        encoder.setTexture(weights, index: 1)
        dispatch(tiles)

        encoder.setComputePipelineState(widenPipeline)
        encoder.setTexture(weights, index: 0)
        encoder.setTexture(widened, index: 1)
        dispatch(tiles)

        encoder.setComputePipelineState(applyPipeline)
        encoder.setTexture(a.texture, index: 0)
        encoder.setTexture(b.texture, index: 1)
        encoder.setTexture(m.texture, index: 2)
        encoder.setTexture(widened, index: 3)
        encoder.setTexture(out.texture, index: 4)
        dispatch(MTLSize(width: width, height: height, depth: 1))
        encoder.endEncoding()

        // Hold the CVMetalTextures (and so their IOSurfaces) until the pass is done.
        let held = HeldTextures([a.held, b.held, m.held, out.held])
        let result = Box(buffer: output)
        let fallback = Box(buffer: mid)
        commands.addCompletedHandler { buffer in
            held.release()
            completion(buffer.status == .completed ? result.buffer : fallback.buffer)
        }
        return commands
    }

    private final class HeldTextures: @unchecked Sendable {
        private var textures: [CVMetalTexture]
        init(_ textures: [CVMetalTexture]) { self.textures = textures }
        func release() { textures.removeAll() }
    }

    private func texture(_ buffer: CVPixelBuffer) -> (texture: MTLTexture, held: CVMetalTexture)? {
        guard let textureCache else { return nil }
        var cvTexture: CVMetalTexture?
        guard CVMetalTextureCacheCreateTextureFromImage(
            nil, textureCache, buffer, nil, .bgra8Unorm,
            CVPixelBufferGetWidth(buffer), CVPixelBufferGetHeight(buffer), 0, &cvTexture) == kCVReturnSuccess,
            let cvTexture, let texture = CVMetalTextureGetTexture(cvTexture) else { return nil }
        return (texture, cvTexture)
    }

    static let source = """
    #include <metal_stdlib>
    using namespace metal;

    struct Params {
        uint tile;
        float stillLevel;
        float errorLevel;
    };

    // Per tile, one 16x16 threadgroup: mean change between the real frames,
    // and mean distance of the in-between frame from their blend, both over
    // RGB in 0-255.
    kernel void guardTileStats(texture2d<float> previous [[texture(0)]],
                               texture2d<float> current [[texture(1)]],
                               texture2d<float> mid [[texture(2)]],
                               texture2d<float, access::write> stats [[texture(3)]],
                               uint2 gid [[thread_position_in_grid]],
                               uint2 tile [[threadgroup_position_in_grid]],
                               uint index [[thread_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint simdgroup [[simdgroup_index_in_threadgroup]],
                               uint simdgroups [[simdgroups_per_threadgroup]]) {
        threadgroup float2 partial[32];
        bool inside = gid.x < mid.get_width() && gid.y < mid.get_height();
        float2 sums = 0;
        if (inside) {
            float3 a = previous.read(gid).rgb, b = current.read(gid).rgb, m = mid.read(gid).rgb;
            float3 d = abs(b - a), e = abs(m - (a + b) * 0.5);
            sums = float2(d.r + d.g + d.b, e.r + e.g + e.b);
        }
        float2 count = float2(inside ? 1.0 : 0.0, 0);
        sums = simd_sum(sums);
        count.x = simd_sum(count.x);
        if (lane == 0) { partial[simdgroup] = sums; partial[16 + simdgroup] = count; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (index == 0) {
            float2 total = 0;
            float n = 0;
            for (uint i = 0; i < simdgroups; i++) { total += partial[i]; n += partial[16 + i].x; }
            n *= 3.0 / 255.0;
            if (tile.x < stats.get_width() && tile.y < stats.get_height())
                stats.write(float4(total / max(n, 1e-6), 0, 0), tile);
        }
    }

    // A tile is wrong when its neighbourhood stood still (the most any of the
    // 3x3 tiles moved) but its in-between frame strays.
    kernel void guardTileWeight(texture2d<float> stats [[texture(0)]],
                                texture2d<float, access::write> weights [[texture(1)]],
                                constant Params &p [[buffer(0)]],
                                uint2 gid [[thread_position_in_grid]]) {
        int w = stats.get_width(), h = stats.get_height();
        if (int(gid.x) >= w || int(gid.y) >= h) return;
        float moved = 0;
        for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++)
                moved = max(moved, stats.read(uint2(clamp(int(gid.x) + dx, 0, w - 1),
                                                    clamp(int(gid.y) + dy, 0, h - 1))).r);
        float strays = stats.read(gid).g;
        float still = clamp((p.stillLevel * 1.5 - moved) / (p.stillLevel * 0.5), 0.0, 1.0);
        float wrong = clamp((strays - p.errorLevel) / p.errorLevel, 0.0, 1.0);
        weights.write(float4(still * wrong), gid);
    }

    // Widened by a tile: the patch's soft edge falls in tiles that stray less.
    kernel void guardTileWiden(texture2d<float> weights [[texture(0)]],
                               texture2d<float, access::write> widened [[texture(1)]],
                               uint2 gid [[thread_position_in_grid]]) {
        int w = weights.get_width(), h = weights.get_height();
        if (int(gid.x) >= w || int(gid.y) >= h) return;
        float most = 0;
        for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++)
                most = max(most, weights.read(uint2(clamp(int(gid.x) + dx, 0, w - 1),
                                                    clamp(int(gid.y) + dy, 0, h - 1))).r);
        widened.write(float4(most), gid);
    }

    // Each pixel: the weight blended between tile centres, then the
    // interpolator's pixel or the blend of the real frames.
    kernel void guardApply(texture2d<float> previous [[texture(0)]],
                           texture2d<float> current [[texture(1)]],
                           texture2d<float> mid [[texture(2)]],
                           texture2d<float> weights [[texture(3)]],
                           texture2d<float, access::write> out [[texture(4)]],
                           constant Params &p [[buffer(0)]],
                           uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= out.get_width() || gid.y >= out.get_height()) return;
        constexpr sampler tiles(coord::pixel, address::clamp_to_edge, filter::linear);
        float2 at = (float2(gid) + 0.5) / float(p.tile);
        float w = weights.sample(tiles, at).r;
        float4 m = mid.read(gid);
        if (w <= 0.0) { out.write(m, gid); return; }
        float4 blend = (previous.read(gid) + current.read(gid)) * 0.5;
        out.write(mix(m, blend, w), gid);
    }
    """
}
#endif
