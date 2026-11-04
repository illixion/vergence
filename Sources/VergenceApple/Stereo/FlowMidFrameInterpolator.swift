#if canImport(Metal)
import CoreVideo
import Foundation
import Metal
import QuartzCore
import VergenceCore

/// Makes the in-between frame on the GPU alone, the way a game's frame
/// generation does (AMD's AFMF/FSR 3): it estimates how each block moved
/// between the two real frames, moves both halfway along that motion, and
/// blends them. No Neural Engine and no model, so it costs a few milliseconds
/// where VideoToolbox's low-latency interpolator costs 5-18 at 720p-1080p
/// (M2 Max), which is what doubling a 60 fps video to 120 needs (an
/// in-between frame every 16.7 ms). It is the weaker guesser on hard motion
/// (fast, thin, things appearing), which matters less at 60 fps: the frames
/// are 16.7 ms apart, so motion between them is small and a wrong vector
/// shows as a slight blur, not a smear.
///
/// The motion is *symmetric*: for each block it looks for the shift d at which
/// `previous(x - d)` and `current(x + d)` agree, so the result is already the
/// motion to the midpoint at every output pixel, with no forward splatting and
/// no holes.
/// 1. Luma of both frames at half and quarter size.
/// 2. Quarter size, 8×8 blocks (32 px of the frame): every shift up to
///    `searchRadius` (±32 × ±20 px, motion up to 64 × 40 px a frame), with a
///    small preference for slow motion so flat areas stay still.
/// 3. Half size, 8×8 blocks (16 px of the frame): each block tries the
///    vectors of the four nearest coarser blocks and their neighbours one
///    pixel either way, so an edge between two motions takes the right one,
///    and a parabola through the costs places it to a fraction of a pixel.
///    (A further level of 8 px blocks at full size scored worse on real
///    footage, as such small blocks match grain, and cost 2.5× as much.)
/// 4. A 3×3 vector median drops the odd wrong block.
/// 5. Each output pixel tries the vectors of the blocks around it (the four
///    nearest by their bilinear share, the ring around those a little) and
///    weights each by how well the two frames agree along it, so a pixel at an
///    object's edge takes the object's or the background's motion, whichever
///    fits. Where none fits (something uncovered, or motion past the search),
///    it falls back to the plain blend of the two frames.
///
/// It has one size: about 1.1 ms of GPU at 1080p on an M2 Max, less than
/// anything a smaller one would save, so the governor's `scale` changes
/// nothing (and a governor that finds it too slow pauses it). BGRA, or half-float RGBA for HDR (`FramePixelFormat`).
public final class FlowMidFrameInterpolator: MidFrameInterpolator, @unchecked Sendable {
    /// Largest symmetric shift searched at quarter size, in quarter pixels
    /// (×4 for the frame's): 8 × 5 → 32 × 20 px, motion 64 × 40 px a frame.
    public static let searchRadius = (x: 8, y: 5)
    /// Block side in pixels at each level.
    static let block = 8
    /// Pixels around a block that also count toward its match.
    static let margin = 2
    /// Added to a match's mean difference (0-255) per pixel of shift: the
    /// slowest of equally good matches wins, so flat areas stay still.
    static let motionPenalty: Float = 0.03
    /// How sharply a pixel's candidate vectors are told apart by how well
    /// the two frames agree along them (0-255). This, the fallback levels and
    /// the motion penalty each moved held-out PSNR by under 0.1 dB across
    /// 5-20, 12-60 and 0.01-0.1 (Bumblebee clips, 2026-11-04).
    static let sigma: Float = 10
    /// Where the best candidate still disagrees by more than `fallbackLow`
    /// (0-255), the pixel starts leaning to the plain blend; fully at `fallbackHigh`.
    static let fallbackLow: Float = 28
    static let fallbackHigh: Float = 72

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let lumaPipeline: MTLComputePipelineState
    private let downPipeline: MTLComputePipelineState
    private let searchPipeline: MTLComputePipelineState
    private let refinePipeline: MTLComputePipelineState
    private let smoothPipeline: MTLComputePipelineState
    private let composePipeline: MTLComputePipelineState
    private var textureCache: CVMetalTextureCache?
    /// One per frame format (BGRA, half-float HDR); made as first needed.
    private var outputPools: [OSType: CVPixelBufferPool] = [:]
    private var width = 0
    private var height = 0
    private var lumaHalf: MTLTexture?
    private var lumaQuarter: MTLTexture?
    private var fieldQuarter: MTLTexture?
    private var fieldHalf: MTLTexture?
    private var fieldSmooth: MTLTexture?
    /// GPU time of the last frame, for measuring (set on a completion thread).
    public private(set) var lastGPUMilliseconds: Double = 0

    /// nil without Metal or if the kernels don't compile.
    public init?(device: MTLDevice? = MTLCreateSystemDefaultDevice()) {
        guard let device, let queue = device.makeCommandQueue() else { return nil }
        do {
            let library = try device.makeLibrary(source: Self.source, options: nil)
            func pipeline(_ name: String) throws -> MTLComputePipelineState {
                guard let function = library.makeFunction(name: name) else { throw FlowError.missingKernel }
                return try device.makeComputePipelineState(function: function)
            }
            lumaPipeline = try pipeline("flowLuma")
            downPipeline = try pipeline("flowDown")
            searchPipeline = try pipeline("flowSearch")
            refinePipeline = try pipeline("flowRefine")
            smoothPipeline = try pipeline("flowSmooth")
            composePipeline = try pipeline("flowCompose")
        } catch {
            VergenceLog.pipeline.error("flow interpolation: kernels failed: \(error, privacy: .public)")
            return nil
        }
        let candidates = (2 * Self.searchRadius.x + 1) * (2 * Self.searchRadius.y + 1)
        guard candidates <= Self.searchThreads.width,
              searchPipeline.maxTotalThreadsPerThreadgroup >= Self.searchThreads.width,
              refinePipeline.maxTotalThreadsPerThreadgroup >= Self.refineThreads.width else { return nil }
        self.device = device
        self.queue = queue
    }

    private enum FlowError: Error { case missingKernel }

    /// One thread per shift in the quarter-size search.
    private static let searchThreads = MTLSize(width: 256, height: 1, depth: 1)
    /// 37 candidates, then the four costs for the sub-pixel fit.
    private static let refineThreads = MTLSize(width: 64, height: 1, depth: 1)

    public func prepare(width: Int, height: Int, scale: Double) -> Bool {
        var cache: CVMetalTextureCache?
        CVMetalTextureCacheCreate(nil, nil, device, nil, &cache)
        func make(_ format: MTLPixelFormat, _ w: Int, _ h: Int) -> MTLTexture? {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: format, width: max(1, w), height: max(1, h), mipmapped: false)
            descriptor.usage = [.shaderRead, .shaderWrite]
            descriptor.storageMode = .private
            return device.makeTexture(descriptor: descriptor)
        }
        func blocks(_ side: Int, _ levelScale: Int) -> Int {
            let levelSide = (side + levelScale - 1) / levelScale
            return (levelSide + Self.block - 1) / Self.block
        }
        guard width >= 32, height >= 32, let cache,
              let lumaHalf = make(.rg16Float, (width + 1) / 2, (height + 1) / 2),
              let lumaQuarter = make(.rg16Float, (width + 3) / 4, (height + 3) / 4),
              let fieldQuarter = make(.rgba16Float, blocks(width, 4), blocks(height, 4)),
              let fieldHalf = make(.rgba16Float, blocks(width, 2), blocks(height, 2)),
              let fieldSmooth = make(.rgba16Float, blocks(width, 2), blocks(height, 2))
        else { return false }
        textureCache = cache
        outputPools = [:]
        self.lumaHalf = lumaHalf
        self.lumaQuarter = lumaQuarter
        self.fieldQuarter = fieldQuarter
        self.fieldHalf = fieldHalf
        self.fieldSmooth = fieldSmooth
        self.width = width
        self.height = height
        return true
    }

    struct Params {
        var searchX: Int32
        var searchY: Int32
        var levelScale: Int32
        var block: Int32
        var margin: Int32
        var subpixel: Int32
        var motionPenalty: Float
        var sigma: Float
        var fallbackLow: Float
        var fallbackHigh: Float
    }

    public func midFrame(between previous: CVPixelBuffer, and current: CVPixelBuffer,
                         completion: @escaping @Sendable (CVPixelBuffer?, Double) -> Void) {
        let started = CACurrentMediaTime()
        guard let commands = encode(previous: previous, current: current, completion: { out in
            completion(out, (CACurrentMediaTime() - started) * 1000)
        }) else {
            completion(nil, .nan)
            return
        }
        commands.commit()
    }

    /// Synchronous, for tests and the CLI.
    public func midFrame(between previous: CVPixelBuffer, and current: CVPixelBuffer) -> CVPixelBuffer? {
        let done = DispatchSemaphore(value: 0)
        let result = ResultBox()
        midFrame(between: previous, and: current) { out, _ in
            result.buffer = out
            done.signal()
        }
        done.wait()
        return result.buffer
    }

    private final class ResultBox: @unchecked Sendable { var buffer: CVPixelBuffer? }
    private struct Box: @unchecked Sendable { let buffer: CVPixelBuffer }

    private func encode(previous: CVPixelBuffer, current: CVPixelBuffer,
                        completion: @escaping @Sendable (CVPixelBuffer?) -> Void) -> MTLCommandBuffer? {
        let format = CVPixelBufferGetPixelFormatType(previous)
        guard width > 0, FramePixelFormat.isSupported(format),
              [previous, current].allSatisfy({
                  CVPixelBufferGetWidth($0) == width && CVPixelBufferGetHeight($0) == height
                      && CVPixelBufferGetPixelFormatType($0) == format
              }),
              let pool = outputPool(format), let lumaHalf, let lumaQuarter,
              let fieldQuarter, let fieldHalf, let fieldSmooth else { return nil }
        var output: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &output)
        guard let output,
              let a = texture(previous), let b = texture(current), let out = texture(output),
              let commands = queue.makeCommandBuffer(),
              let encoder = commands.makeComputeCommandEncoder() else { return nil }
        CVBufferPropagateAttachments(current, output)

        func params(levelScale: Int, subpixel: Bool = false) -> Params {
            Params(searchX: Int32(Self.searchRadius.x), searchY: Int32(Self.searchRadius.y),
                   levelScale: Int32(levelScale), block: Int32(Self.block), margin: Int32(Self.margin),
                   subpixel: subpixel ? 1 : 0, motionPenalty: Self.motionPenalty, sigma: Self.sigma,
                   fallbackLow: Self.fallbackLow, fallbackHigh: Self.fallbackHigh)
        }
        func perPixel(_ texture: MTLTexture) {
            encoder.dispatchThreadgroups(
                MTLSize(width: (texture.width + 7) / 8, height: (texture.height + 7) / 8, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        }
        func perBlock(_ field: MTLTexture, _ threads: MTLSize) {
            encoder.dispatchThreadgroups(MTLSize(width: field.width, height: field.height, depth: 1),
                                         threadsPerThreadgroup: threads)
        }

        encoder.setComputePipelineState(lumaPipeline)
        encoder.setTexture(a.texture, index: 0)
        encoder.setTexture(b.texture, index: 1)
        encoder.setTexture(lumaHalf, index: 2)
        perPixel(lumaHalf)

        encoder.setComputePipelineState(downPipeline)
        encoder.setTexture(lumaHalf, index: 0)
        encoder.setTexture(lumaQuarter, index: 1)
        perPixel(lumaQuarter)

        var p = params(levelScale: 4)
        encoder.setComputePipelineState(searchPipeline)
        encoder.setBytes(&p, length: MemoryLayout<Params>.stride, index: 0)
        encoder.setTexture(lumaQuarter, index: 0)
        encoder.setTexture(fieldQuarter, index: 1)
        perBlock(fieldQuarter, Self.searchThreads)

        encoder.setComputePipelineState(refinePipeline)
        p = params(levelScale: 2, subpixel: true)
        encoder.setBytes(&p, length: MemoryLayout<Params>.stride, index: 0)
        encoder.setTexture(lumaHalf, index: 0)
        encoder.setTexture(fieldQuarter, index: 1)
        encoder.setTexture(fieldHalf, index: 2)
        perBlock(fieldHalf, Self.refineThreads)

        encoder.setComputePipelineState(smoothPipeline)
        encoder.setTexture(fieldHalf, index: 0)
        encoder.setTexture(fieldSmooth, index: 1)
        perPixel(fieldSmooth)

        p = params(levelScale: 2)
        encoder.setComputePipelineState(composePipeline)
        encoder.setBytes(&p, length: MemoryLayout<Params>.stride, index: 0)
        encoder.setTexture(a.texture, index: 0)
        encoder.setTexture(b.texture, index: 1)
        encoder.setTexture(fieldSmooth, index: 2)
        encoder.setTexture(out.texture, index: 3)
        perPixel(out.texture)
        encoder.endEncoding()

        // Hold the CVMetalTextures (and so their IOSurfaces) until the pass is done.
        let held = HeldTextures([a.held, b.held, out.held])
        let result = Box(buffer: output)
        commands.addCompletedHandler { [weak self] buffer in
            held.release()
            self?.lastGPUMilliseconds = (buffer.gpuEndTime - buffer.gpuStartTime) * 1000
            completion(buffer.status == .completed ? result.buffer : nil)
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

    /// One request is in flight at a time, so one caller at a time.
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
        int searchX;
        int searchY;
        int levelScale;   // frame pixels per pixel of this level
        int block;        // block side, in this level's pixels
        int margin;
        int subpixel;
        float motionPenalty;
        float sigma;
        float fallbackLow;
        float fallbackHigh;
    };

    constant float3 lumaWeights = float3(0.2126, 0.7152, 0.0722);

    // Half-size luma of both frames, previous in r and current in g.
    kernel void flowLuma(texture2d<float> previous [[texture(0)]],
                         texture2d<float> current [[texture(1)]],
                         texture2d<float, access::write> luma [[texture(2)]],
                         uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= luma.get_width() || gid.y >= luma.get_height()) return;
        uint2 last = uint2(previous.get_width() - 1, previous.get_height() - 1);
        float2 sum = 0;
        for (uint dy = 0; dy < 2; dy++)
            for (uint dx = 0; dx < 2; dx++) {
                uint2 at = min(gid * 2 + uint2(dx, dy), last);
                sum += float2(dot(previous.read(at).rgb, lumaWeights), dot(current.read(at).rgb, lumaWeights));
            }
        luma.write(float4(sum * 0.25, 0, 0), gid);
    }

    kernel void flowDown(texture2d<float> source [[texture(0)]],
                         texture2d<float, access::write> half_ [[texture(1)]],
                         uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= half_.get_width() || gid.y >= half_.get_height()) return;
        uint2 last = uint2(source.get_width() - 1, source.get_height() - 1);
        float2 sum = 0;
        for (uint dy = 0; dy < 2; dy++)
            for (uint dx = 0; dx < 2; dx++) sum += source.read(min(gid * 2 + uint2(dx, dy), last)).rg;
        half_.write(float4(sum * 0.25, 0, 0), gid);
    }

    static float2 readLuma(texture2d<float> luma, int2 at) {
        return luma.read(uint2(clamp(at, int2(0), int2(luma.get_width() - 1, luma.get_height() - 1)))).rg;
    }

    // Mean |previous(q - d) - current(q + d)| (0-255) over the block and its
    // margin, d in this level's pixels.
    static float blockCost(texture2d<float> luma, int2 block, int2 d, constant Params &p) {
        int2 origin = block * p.block - p.margin;
        int side = p.block + 2 * p.margin;
        float sum = 0;
        for (int y = 0; y < side; y++)
            for (int x = 0; x < side; x++) {
                int2 q = origin + int2(x, y);
                sum += abs(readLuma(luma, q - d).r - readLuma(luma, q + d).g);
            }
        return sum / float(side * side) * 255.0;
    }

    // Quarter size, one threadgroup per block, one thread per shift: every
    // shift in the search window. Writes (d in frame pixels, cost, cost at 0).
    kernel void flowSearch(texture2d<float> luma [[texture(0)]],
                           texture2d<float, access::write> field [[texture(1)]],
                           constant Params &p [[buffer(0)]],
                           uint2 block [[threadgroup_position_in_grid]],
                           uint index [[thread_index_in_threadgroup]]) {
        threadgroup float cost[256];
        if (block.x >= field.get_width() || block.y >= field.get_height()) return;
        int across = 2 * p.searchX + 1;
        int count = across * (2 * p.searchY + 1);
        cost[index] = INFINITY;
        if (int(index) < count) {
            int2 d = int2(int(index) % across - p.searchX, int(index) / across - p.searchY);
            cost[index] = blockCost(luma, int2(block), d, p) + p.motionPenalty * length(float2(d * p.levelScale));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (index == 0) {
            int best = 0;
            for (int i = 1; i < count; i++) if (cost[i] < cost[best]) best = i;
            int2 d = int2(best % across - p.searchX, best / across - p.searchY);
            int zero = p.searchY * across + p.searchX;
            field.write(float4(float2(d * p.levelScale), cost[best], cost[zero]), block);
        }
    }

    // Half size, one threadgroup per block: the vectors of the four
    // nearest coarser blocks, each as is and one pixel of this level either
    // way, and no motion; the cheapest wins. With subpixel, a parabola
    // through the costs either side of it places it to a fraction of a pixel.
    kernel void flowRefine(texture2d<float> luma [[texture(0)]],
                           texture2d<float> parent [[texture(1)]],
                           texture2d<float, access::write> field [[texture(2)]],
                           constant Params &p [[buffer(0)]],
                           uint2 block [[threadgroup_position_in_grid]],
                           uint index [[thread_index_in_threadgroup]]) {
        threadgroup float cost[64];
        threadgroup int2 candidate[64];
        threadgroup int bestIndex;
        threadgroup float side[4];
        if (block.x >= field.get_width() || block.y >= field.get_height()) return;
        float s = float(p.levelScale);
        float2 at = (float2(block) + 0.5) * 0.5 - 0.5;
        int2 base = int2(floor(at));
        int2 last = int2(parent.get_width() - 1, parent.get_height() - 1);
        cost[index] = INFINITY;
        if (index < 37) {
            int2 d = int2(0);
            if (index < 36) {
                uint which = index / 9, offset = index % 9;
                int2 coarser = clamp(base + int2(which % 2, which / 2), int2(0), last);
                d = int2(round(parent.read(uint2(coarser)).xy / s)) + int2(int(offset % 3) - 1, int(offset / 3) - 1);
            }
            candidate[index] = d;
            cost[index] = blockCost(luma, int2(block), d, p) + p.motionPenalty * length(float2(d) * s);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (index == 0) {
            int best = 0;
            for (int i = 1; i < 37; i++) if (cost[i] < cost[best]) best = i;
            bestIndex = best;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int2 d = candidate[bestIndex];
        float c0 = cost[bestIndex];
        if (p.subpixel != 0) {
            if (index < 4) {
                int2 step = index == 0 ? int2(-1, 0) : index == 1 ? int2(1, 0) : index == 2 ? int2(0, -1) : int2(0, 1);
                side[index] = blockCost(luma, int2(block), d + step, p) + p.motionPenalty * length(float2(d + step) * s);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (index == 0) {
                float2 fraction = 0;
                float cx = side[0] - 2 * c0 + side[1], cy = side[2] - 2 * c0 + side[3];
                if (cx > 1e-3) fraction.x = clamp((side[0] - side[1]) / (2 * cx), -0.5, 0.5);
                if (cy > 1e-3) fraction.y = clamp((side[2] - side[3]) / (2 * cy), -0.5, 0.5);
                field.write(float4((float2(d) + fraction) * s, c0, cost[36]), block);
            }
        } else if (index == 0) {
            field.write(float4(float2(d) * s, c0, cost[36]), block);
        }
    }

    // 3x3 vector median: the vector closest (L1) to its neighbours' overall,
    // so a lone wrong block takes its neighbours' motion. Costs stay the block's own.
    kernel void flowSmooth(texture2d<float> field [[texture(0)]],
                           texture2d<float, access::write> smooth [[texture(1)]],
                           uint2 gid [[thread_position_in_grid]]) {
        int w = field.get_width(), h = field.get_height();
        if (int(gid.x) >= w || int(gid.y) >= h) return;
        float2 v[9];
        for (int i = 0; i < 9; i++)
            v[i] = field.read(uint2(clamp(int(gid.x) + i % 3 - 1, 0, w - 1), clamp(int(gid.y) + i / 3 - 1, 0, h - 1))).xy;
        int best = 4;
        float bestSum = INFINITY;
        for (int i = 0; i < 9; i++) {
            float sum = 0;
            for (int j = 0; j < 9; j++) sum += abs(v[i].x - v[j].x) + abs(v[i].y - v[j].y);
            if (sum < bestSum - 1e-4 || (i == 4 && sum <= bestSum + 1e-4)) { bestSum = sum; best = i; }
        }
        float4 own = field.read(gid);
        smooth.write(float4(v[best], own.z, own.w), gid);
    }

    // Each pixel: the nearby blocks' vectors, each weighted by its
    // share and by how well previous(x - d) and current(x + d)
    // agree; their average at the best is the in-between pixel. Where even
    // the best disagrees, the plain blend of the two frames.
    kernel void flowCompose(texture2d<float> previous [[texture(0)]],
                            texture2d<float> current [[texture(1)]],
                            texture2d<float> field [[texture(2)]],
                            texture2d<float, access::write> out [[texture(3)]],
                            constant Params &p [[buffer(0)]],
                            uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= out.get_width() || gid.y >= out.get_height()) return;
        constexpr sampler linear(coord::pixel, address::clamp_to_edge, filter::linear);
        float2 x = float2(gid) + 0.5;
        float blockSide = float(p.block * p.levelScale);
        float2 at = x / blockSide - 0.5;
        int2 base = int2(floor(at));
        float2 f = at - float2(base);
        int2 last = int2(field.get_width() - 1, field.get_height() - 1);
        float3 sum = 0;
        float total = 0;
        float bestError = INFINITY;
        // The 2x2 nearest blocks carry their bilinear share; the ring
        // around them a small one, so an edge can take a vector from one
        // block further out.
        for (int j = -1; j < 3; j++)
            for (int i = -1; i < 3; i++) {
                float sx = i == 0 ? 1.0 - f.x : i == 1 ? f.x : 0.0;
                float sy = j == 0 ? 1.0 - f.y : j == 1 ? f.y : 0.0;
                float share = sx * sy + 0.02;
                float2 d = field.read(uint2(clamp(base + int2(i, j), int2(0), last))).xy;
                float3 a = previous.sample(linear, x - d).rgb;
                float3 b = current.sample(linear, x + d).rgb;
                float e = abs(dot(a - b, lumaWeights)) * 255.0;
                bestError = min(bestError, e);
                float r = e / p.sigma;
                float w = share * (1.0 / (1.0 + r * r) + 1e-3);
                sum += (a + b) * 0.5 * w;
                total += w;
            }
        float3 warped = sum / max(total, 1e-6);
        float3 plain = (previous.read(gid).rgb + current.read(gid).rgb) * 0.5;
        float fallback = smoothstep(p.fallbackLow, p.fallbackHigh, bestError);
        out.write(float4(mix(warped, plain, fallback), 1.0), gid);
    }
    """
}
#endif
