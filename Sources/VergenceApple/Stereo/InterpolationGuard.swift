#if canImport(Metal)
import CoreVideo
import Foundation
import Metal
import VergenceCore

/// Redoes the tiles of an in-between frame that an optical-flow interpolator
/// got wrong, from the two real frames.
///
/// VideoToolbox's low-latency interpolator (the only one visionOS has) misreads
/// evenly spaced stripes (window blinds, railings, a striped shirt): two stripe
/// positions look alike, so the flow matches the wrong one and the in-between
/// frame shows the stripes shifted by about half their spacing, in a patch with
/// bent edges. Only the in-between frames carry it, so at 48 Hz the patch
/// flickers. It happens on still stripes and, worse, on stripes the camera pans
/// across (House S03E15, 37:29: a fifth of the frame wrong, 10-15 levels).
///
/// So, per 16×16 tile:
/// 1. Find how the tile moved between the real frames by block matching:
///    the symmetric shift d (whole pixels, up to 12 across and 3 down) for
///    which `previous(x - d)` and `current(x + d)` agree best, with a small
///    preference for slow motion, which picks the right stripe since stripes
///    are much further apart than a frame's motion. Matched on half-size luma:
///    every other shift sparsely over a 36-pixel block around the tile, then
///    the shifts next to the best densely.
/// 2. Their blend at that shift, `(previous(x - d) + current(x + d)) / 2`, is a
///    good in-between picture where they matched well. Where the
///    interpolator's frame strays far from it, the tile is wrong and takes the
///    reconstruction instead. Where nothing matches (something appears or is
///    uncovered) the interpolator's frame stays, being the better guess.
/// 3. Weights are widened by a tile (the patch has soft edges); each pixel
///    mixes the four nearest tiles' reconstructions, so no tile edge shows.
///
/// A still tile is the case d = 0, where the reconstruction is the plain blend.
/// Tiles whose in-between frame is already close to the plain blend (flat or
/// still areas: most of a frame) skip the search.
public final class InterpolationGuard: @unchecked Sendable {
    /// Tile side in pixels.
    public static let tile = 16
    /// Largest symmetric shift searched, in pixels (motion between the real
    /// frames is twice it: 24 across, 6 down).
    public static let searchX = 12, searchY = 3
    /// Mean distance (0-255) from the reconstruction above which the
    /// interpolator's tile counts as wrong; fully replaced 4 levels above it.
    /// Raised by how well the real frames matched (`2.5 × match + 2`).
    public static let errorLevel: Float = 4
    /// Mean mismatch (0-255) of the real frames at the found shift above which
    /// the reconstruction is not trusted; none at all at twice it.
    public static let matchLevel: Float = 4

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let lumaPipeline: MTLComputePipelineState
    private let statsPipeline: MTLComputePipelineState
    private let searchPipeline: MTLComputePipelineState
    private let checkPipeline: MTLComputePipelineState
    private let widenPipeline: MTLComputePipelineState
    private let applyPipeline: MTLComputePipelineState
    private var textureCache: CVMetalTextureCache?
    /// One per frame format (BGRA, half-float HDR); made as first needed.
    private var outputPools: [OSType: CVPixelBufferPool] = [:]
    private var width = 0
    private var height = 0
    private var luma: MTLTexture?
    private var stats: MTLTexture?
    private var motion: MTLTexture?
    private var weights: MTLTexture?
    private var widened: MTLTexture?
    /// GPU time of the last pass, for measuring (set on a completion thread).
    public private(set) var lastGPUMilliseconds: Double = 0

    /// nil without Metal or if the kernels don't compile.
    public init?(device: MTLDevice? = MTLCreateSystemDefaultDevice()) {
        guard let device, let queue = device.makeCommandQueue() else { return nil }
        do {
            let library = try device.makeLibrary(source: Self.source, options: nil)
            func pipeline(_ name: String) throws -> MTLComputePipelineState {
                guard let function = library.makeFunction(name: name) else { throw GuardError.missingKernel }
                return try device.makeComputePipelineState(function: function)
            }
            lumaPipeline = try pipeline("guardLuma")
            statsPipeline = try pipeline("guardTileStats")
            searchPipeline = try pipeline("guardTileSearch")
            checkPipeline = try pipeline("guardTileCheck")
            widenPipeline = try pipeline("guardTileWiden")
            applyPipeline = try pipeline("guardApply")
        } catch {
            VergenceLog.pipeline.error("interpolation guard: kernels failed: \(error, privacy: .public)")
            return nil
        }
        guard searchPipeline.maxTotalThreadsPerThreadgroup >= Self.searchThreads.width,
              (Self.searchX / 2 * 2 + 1) * (Self.searchY / 2 * 2 + 1) <= Self.searchThreads.width
        else { return nil }
        self.device = device
        self.queue = queue
    }

    private enum GuardError: Error { case missingKernel }

    /// Threads per tile in the search: one per coarse shift, then seven per
    /// refined one.
    private static let searchThreads = MTLSize(width: 64, height: 1, depth: 1)

    /// Sets the frame size (BGRA or half-float RGBA, all three frames alike;
    /// the guard's levels are on the 0-255 scale of either's code values).
    public func prepare(width: Int, height: Int) -> Bool {
        var cache: CVMetalTextureCache?
        CVMetalTextureCacheCreate(nil, nil, device, nil, &cache)
        let tilesWide = (width + Self.tile - 1) / Self.tile
        let tilesHigh = (height + Self.tile - 1) / Self.tile
        func make(_ format: MTLPixelFormat, _ w: Int = tilesWide, _ h: Int = tilesHigh) -> MTLTexture? {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: format, width: w, height: h, mipmapped: false)
            descriptor.usage = [.shaderRead, .shaderWrite]
            descriptor.storageMode = .private
            return device.makeTexture(descriptor: descriptor)
        }
        guard let cache,
              let luma = make(.rg16Float, (width + 1) / 2, (height + 1) / 2),
              let stats = make(.rg16Float), let motion = make(.rg16Float),
              let weights = make(.rg16Float), let widened = make(.r16Float) else { return false }
        textureCache = cache
        outputPools = [:]
        self.luma = luma
        self.stats = stats
        self.motion = motion
        self.weights = weights
        self.widened = widened
        self.width = width
        self.height = height
        return true
    }

    struct Params {
        var tile: UInt32
        var searchX: Int32
        var searchY: Int32
        var errorLevel: Float
        var matchLevel: Float
    }

    /// The in-between frame with its wrong tiles redone, or `mid` itself when
    /// the guard can't run (wrong size or format). `completion` runs on a Metal
    /// completion thread.
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
        let format = CVPixelBufferGetPixelFormatType(mid)
        guard width > 0, FramePixelFormat.isSupported(format),
              [previous, current, mid].allSatisfy({
                  CVPixelBufferGetWidth($0) == width && CVPixelBufferGetHeight($0) == height
                      && CVPixelBufferGetPixelFormatType($0) == format
              }),
              let pool = outputPool(format), let luma, let stats, let motion, let weights, let widened else { return nil }
        var output: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &output)
        guard let output,
              let a = texture(previous), let b = texture(current), let m = texture(mid), let out = texture(output),
              let commands = queue.makeCommandBuffer(),
              let encoder = commands.makeComputeCommandEncoder() else { return nil }
        CVBufferPropagateAttachments(mid, output)

        var params = Params(tile: UInt32(Self.tile), searchX: Int32(Self.searchX), searchY: Int32(Self.searchY),
                            errorLevel: Self.errorLevel, matchLevel: Self.matchLevel)
        let tiles = MTLSize(width: stats.width, height: stats.height, depth: 1)
        let tileThreads = MTLSize(width: Self.tile, height: Self.tile, depth: 1)
        func dispatch(_ size: MTLSize) {
            encoder.dispatchThreadgroups(
                MTLSize(width: (size.width + 7) / 8, height: (size.height + 7) / 8, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        }
        encoder.setBytes(&params, length: MemoryLayout<Params>.stride, index: 0)

        encoder.setComputePipelineState(lumaPipeline)
        encoder.setTexture(a.texture, index: 0)
        encoder.setTexture(b.texture, index: 1)
        encoder.setTexture(luma, index: 2)
        dispatch(MTLSize(width: luma.width, height: luma.height, depth: 1))

        encoder.setComputePipelineState(statsPipeline)
        encoder.setTexture(a.texture, index: 0)
        encoder.setTexture(b.texture, index: 1)
        encoder.setTexture(m.texture, index: 2)
        encoder.setTexture(stats, index: 3)
        encoder.dispatchThreadgroups(tiles, threadsPerThreadgroup: tileThreads)

        encoder.setComputePipelineState(searchPipeline)
        encoder.setTexture(luma, index: 0)
        encoder.setTexture(stats, index: 1)
        encoder.setTexture(motion, index: 2)
        encoder.dispatchThreadgroups(tiles, threadsPerThreadgroup: Self.searchThreads)

        encoder.setComputePipelineState(checkPipeline)
        encoder.setTexture(a.texture, index: 0)
        encoder.setTexture(b.texture, index: 1)
        encoder.setTexture(m.texture, index: 2)
        encoder.setTexture(motion, index: 3)
        encoder.setTexture(weights, index: 4)
        encoder.setTexture(stats, index: 5)
        encoder.dispatchThreadgroups(tiles, threadsPerThreadgroup: tileThreads)

        encoder.setComputePipelineState(widenPipeline)
        encoder.setTexture(weights, index: 0)
        encoder.setTexture(widened, index: 1)
        dispatch(tiles)

        encoder.setComputePipelineState(applyPipeline)
        encoder.setTexture(a.texture, index: 0)
        encoder.setTexture(b.texture, index: 1)
        encoder.setTexture(m.texture, index: 2)
        encoder.setTexture(widened, index: 3)
        encoder.setTexture(motion, index: 4)
        encoder.setTexture(out.texture, index: 5)
        dispatch(MTLSize(width: width, height: height, depth: 1))
        encoder.endEncoding()

        // Hold the CVMetalTextures (and so their IOSurfaces) until the pass is done.
        let held = HeldTextures([a.held, b.held, m.held, out.held])
        let result = Box(buffer: output)
        let fallback = Box(buffer: mid)
        commands.addCompletedHandler { [weak self] buffer in
            held.release()
            self?.lastGPUMilliseconds = (buffer.gpuEndTime - buffer.gpuStartTime) * 1000
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
        return FramePixelFormat.texture(buffer, cache: textureCache)
    }

    /// Called on one completion or encode thread at a time (one request in
    /// flight per interpolator).
    private func outputPool(_ format: OSType) -> CVPixelBufferPool? {
        if let pool = outputPools[format] { return pool }
        let pool = FramePixelFormat.pool(format: format, width: width, height: height)
        outputPools[format] = pool
        return pool
    }

    static let source = """
    #include <metal_stdlib>
    using namespace metal;

    struct Params {
        uint tile;
        int searchX;
        int searchY;
        float errorLevel;
        float matchLevel;
    };

    constant float3 lumaWeights = float3(0.299, 0.587, 0.114);

    static float4 readClamped(texture2d<float> t, int2 at) {
        return t.read(uint2(clamp(at, int2(0), int2(t.get_width() - 1, t.get_height() - 1))));
    }

    // Half-size luma of the two real frames (previous in r, current in g),
    // for the motion search.
    kernel void guardLuma(texture2d<float> previous [[texture(0)]],
                          texture2d<float> current [[texture(1)]],
                          texture2d<float, access::write> luma [[texture(2)]],
                          uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= luma.get_width() || gid.y >= luma.get_height()) return;
        float2 sum = 0;
        for (int dy = 0; dy < 2; dy++)
            for (int dx = 0; dx < 2; dx++) {
                int2 at = int2(gid) * 2 + int2(dx, dy);
                sum += float2(dot(readClamped(previous, at).rgb, lumaWeights),
                              dot(readClamped(current, at).rgb, lumaWeights));
            }
        luma.write(float4(sum * 0.25, 0, 0), gid);
    }

    // Per tile, one 16x16 threadgroup: mean change between the real frames,
    // and mean distance of the in-between frame from their plain blend, both
    // over RGB in 0-255.
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

    // Per tile, 64 threads. First a coarse search over every other shift d:
    // how well previous(x - d) and current(x + d) agree over a 36-pixel block
    // around the tile (half-size luma, every 6 pixels), plus a little for
    // speed so the slowest of equally good matches wins (the right stripe, not
    // one a spacing away). Then the 3x3 shifts around the best, densely (every
    // half-size pixel of a 32-pixel block: sparse samples can't place a sharp
    // edge to the pixel), each shift's 16 rows split over seven threads. The
    // best d goes to motion. A tile whose in-between frame is already close
    // to the plain blend has nothing to put right and keeps d = 0.
    kernel void guardTileSearch(texture2d<float> luma [[texture(0)]],
                                texture2d<float> stats [[texture(1)]],
                                texture2d<float, access::write> motion [[texture(2)]],
                                constant Params &p [[buffer(0)]],
                                uint2 tile [[threadgroup_position_in_grid]],
                                uint index [[thread_index_in_threadgroup]]) {
        threadgroup float cost[64];
        threadgroup float2 partial[64];
        threadgroup float2 coarse;
        if (tile.x >= motion.get_width() || tile.y >= motion.get_height()) return;
        if (stats.read(tile).g < p.errorLevel) {
            if (index == 0) motion.write(float4(0), tile);
            return;
        }
        constexpr sampler halfSize(coord::pixel, address::clamp_to_edge, filter::linear);
        float2 frame = float2(luma.get_width(), luma.get_height()) * 2.0;
        float2 centre = (float2(tile) + 0.5) * float(p.tile);
        int2 reach = int2(p.searchX, p.searchY) / 2;
        uint across = uint(2 * reach.x + 1), count = across * uint(2 * reach.y + 1);
        cost[index] = INFINITY;
        if (index < count) {
            float2 d = float2(2 * (int2(index % across, index / across) - reach));
            float sum = 0, used = 0;
            for (int oy = -15; oy <= 15; oy += 6)
                for (int ox = -15; ox <= 15; ox += 6) {
                    float2 q = centre + float2(ox, oy);
                    if (any(q - abs(d) < 0.0) || any(q + abs(d) > frame)) continue;
                    sum += abs(luma.sample(halfSize, (q - d) * 0.5).r - luma.sample(halfSize, (q + d) * 0.5).g);
                    used += 1;
                }
            if (used > 0) cost[index] = sum / used * 255.0 + 0.08 * length(d);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (index == 0) {
            uint best = 0;
            for (uint i = 1; i < count; i++) if (cost[i] < cost[best]) best = i;
            coarse = float2(2 * (int2(best % across, best / across) - reach));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        constexpr uint parts = 7;
        if (index < 9 * parts) {
            uint candidate = index % 9, part = index / 9;
            float2 r = coarse + float2(int(candidate % 3) - 1, int(candidate / 3) - 1);
            float dense = 0, used = 0;
            for (int row = int(part); row < 16; row += int(parts))
                for (int ox = -15; ox <= 15; ox += 2) {
                    float2 q = centre + float2(ox, -15 + 2 * row);
                    if (any(q - abs(r) < 0.0) || any(q + abs(r) > frame)) continue;
                    dense += abs(luma.sample(halfSize, (q - r) * 0.5).r - luma.sample(halfSize, (q + r) * 0.5).g);
                    used += 1;
                }
            partial[index] = float2(dense, used);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (index == 0) {
            uint best = 0;
            float bestCost = INFINITY;
            for (uint candidate = 0; candidate < 9; candidate++) {
                float2 r = coarse + float2(int(candidate % 3) - 1, int(candidate / 3) - 1);
                float2 total = 0;
                for (uint part = 0; part < parts; part++) total += partial[candidate + 9 * part];
                if (abs(r.x) > float(p.searchX) || abs(r.y) > float(p.searchY) || total.y <= 0) continue;
                float c = total.x / total.y * 255.0 + 0.08 * length(r);
                if (c < bestCost) { bestCost = c; best = candidate; }
            }
            motion.write(float4(coarse + float2(int(best % 3) - 1, int(best / 3) - 1), 0, 0), tile);
        }
    }

    // Per tile, one 16x16 threadgroup, at the tile's shift d: how far the
    // in-between frame strays from the reconstruction over the tile, and how
    // well the real frames match over the 48-pixel block around it (each
    // thread three by three samples a tile apart). Writes the tile's weight
    // (wrong, and the reconstruction trusted) and the trust alone.
    kernel void guardTileCheck(texture2d<float> previous [[texture(0)]],
                               texture2d<float> current [[texture(1)]],
                               texture2d<float> mid [[texture(2)]],
                               texture2d<float> motion [[texture(3)]],
                               texture2d<float, access::write> weights [[texture(4)]],
                               texture2d<float> stats [[texture(5)]],
                               constant Params &p [[buffer(0)]],
                               uint2 gid [[thread_position_in_grid]],
                               uint2 tile [[threadgroup_position_in_grid]],
                               uint index [[thread_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint simdgroup [[simdgroup_index_in_threadgroup]],
                               uint simdgroups [[simdgroups_per_threadgroup]]) {
        threadgroup float4 partial[32];
        bool tileInside = tile.x < motion.get_width() && tile.y < motion.get_height();
        if (!tileInside) return;
        // Skipped by the search (already close to the plain blend): nothing to
        // put right, and the trust at d = 0 is how much the tile changed.
        float2 blendStats = stats.read(tile).rg;
        if (blendStats.g < p.errorLevel) {
            if (index == 0)
                weights.write(float4(0, clamp((2.0 * p.matchLevel - blendStats.r) / p.matchLevel, 0.0, 1.0), 0, 0), tile);
            return;
        }
        int2 d = int2(motion.read(tile).xy);
        int2 at = int2(gid);
        bool inside = gid.x < mid.get_width() && gid.y < mid.get_height();
        float4 sums = 0;   // x: stray, y: its count, z: match, w: its count
        int2 size = int2(mid.get_width(), mid.get_height());
        // Pixels whose reconstruction would read outside the frame keep the
        // interpolator's pixel, so they don't count either.
        if (inside && all(at - abs(d) >= 0) && all(at + abs(d) < size)) {
            float3 recon = (readClamped(previous, at - d).rgb + readClamped(current, at + d).rgb) * 0.5;
            sums.x = abs(dot(mid.read(gid).rgb - recon, lumaWeights));
            sums.y = 1;
        }
        for (int ty = -1; ty <= 1; ty++)
            for (int tx = -1; tx <= 1; tx++) {
                // Only where both shifted reads fall inside the frame: clamped
                // reads at the edge would look like a mismatch.
                int2 q = at + int2(tx, ty) * int(p.tile);
                int2 lo = q - abs(d), hi = q + abs(d);
                if (any(lo < 0) || any(hi >= size)) continue;
                sums.z += abs(dot(readClamped(previous, q - d).rgb - readClamped(current, q + d).rgb, lumaWeights));
                sums.w += 1;
            }
        sums = simd_sum(sums);
        if (lane == 0) partial[simdgroup] = sums;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (index == 0) {
            float4 total = 0;
            for (uint i = 0; i < simdgroups; i++) total += partial[i];
            float stray = total.x / max(total.y, 1.0) * 255.0;
            float match = total.z / max(total.w, 1.0) * 255.0;
            float trust = clamp((2.0 * p.matchLevel - match) / p.matchLevel, 0.0, 1.0);
            float wrong = clamp((stray - max(p.errorLevel, 2.5 * match + 2.0)) / 4.0, 0.0, 1.0);
            weights.write(float4(wrong * trust, trust, 0, 0), tile);
        }
    }

    // Widened by a tile (the patch's soft edge falls in tiles that stray
    // less), but only as far as the tile's own reconstruction is trusted.
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
        widened.write(float4(most * weights.read(gid).g), gid);
    }

    // Each pixel mixes the interpolator's pixel with the four nearest tiles'
    // reconstructions, each by its weight times its bilinear share.
    kernel void guardApply(texture2d<float> previous [[texture(0)]],
                           texture2d<float> current [[texture(1)]],
                           texture2d<float> mid [[texture(2)]],
                           texture2d<float> weights [[texture(3)]],
                           texture2d<float> motion [[texture(4)]],
                           texture2d<float, access::write> out [[texture(5)]],
                           constant Params &p [[buffer(0)]],
                           uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= out.get_width() || gid.y >= out.get_height()) return;
        float2 at = (float2(gid) + 0.5) / float(p.tile) - 0.5;
        int2 base = int2(floor(at));
        float2 f = at - float2(base);
        int2 last = int2(weights.get_width() - 1, weights.get_height() - 1);
        int2 size = int2(out.get_width(), out.get_height());
        float4 m = mid.read(gid);
        float total = 0;
        float4 mixed = 0;
        for (int j = 0; j < 2; j++)
            for (int i = 0; i < 2; i++) {
                uint2 t = uint2(clamp(base + int2(i, j), int2(0), last));
                float share = (i == 0 ? 1.0 - f.x : f.x) * (j == 0 ? 1.0 - f.y : f.y);
                float w = weights.read(t).r * share;
                if (w <= 0.0) continue;
                int2 d = int2(motion.read(t).xy);
                if (any(int2(gid) - abs(d) < 0) || any(int2(gid) + abs(d) >= size)) continue;
                float4 recon = (readClamped(previous, int2(gid) - d) + readClamped(current, int2(gid) + d)) * 0.5;
                mixed += recon * w;
                total += w;
            }
        if (total <= 0.0) { out.write(m, gid); return; }
        out.write(m * (1.0 - total) + mixed, gid);
    }
    """
}
#endif
