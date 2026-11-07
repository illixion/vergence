/*
 Vergence - AI HDR for video on the player: one gain-grid lookup per pixel

 Turns an SDR frame into HDR with a worker's gain grid (`GainGrid`, the
 `ai-hdr` workload): each pixel's gain is the grid's 3D texture read at its
 own (u, v, Y′), and it multiplies the pixel's linear light. Everything
 before it (interpolation, depth, the warp) stays SDR; this is the last step
 (Hypnos `docs/plans/analysis-workers.md`, "AI HDR for video, in short").

 Three dispatches in one command buffer:
 1. Every fourth pixel each way: the linear gain it would get, summed per
    threadgroup.
 2. The frame's mean gain, and the scale every log gain is multiplied by so
    the mean stays at `cap` or below (`GainGrid.cappedScale`): the window
    brightness limiter dims the whole window past about 2.5–3× on a Vision
    Pro, which would undo the highlights it was meant to show.
 3. Out = linear(SDR) × 2^(g × scale), as extended linear light in half
    floats (the pump's HDR format), BT.709 primaries, the SDR signal
    decoded with BT.1886's 2.4 power.

 This is the reference pass, checked against `GainGrid.lookup` in the tests.
 In Convolution the same lookup is meant to go into shaders that already
 run (`imageFragmentShader` for flat video, the stereo warp for AI 3D)
 rather than as a pass of its own; see the plan.
 */

#if canImport(Metal)
import CoreVideo
import Foundation
import Metal
import VergenceCore

public final class GainGridPass: @unchecked Sendable {
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let sumPipeline: MTLComputePipelineState
    private let meanPipeline: MTLComputePipelineState
    private let applyPipeline: MTLComputePipelineState
    private let gridTexture: MTLTexture
    private let partials: MTLBuffer
    private let result: MTLBuffer
    private var textureCache: CVMetalTextureCache?
    private static let maxGroups = 4096
    /// The last frame's mean linear gain before the cap, and the scale applied.
    public private(set) var lastMeanGain: Float = 1
    public private(set) var lastScale: Float = 1

    public init?(device: MTLDevice? = MTLCreateSystemDefaultDevice()) {
        guard let device, let queue = device.makeCommandQueue() else { return nil }
        do {
            let library = try device.makeLibrary(source: Self.source, options: nil)
            func pipeline(_ name: String) throws -> MTLComputePipelineState {
                try device.makeComputePipelineState(function: library.makeFunction(name: name)!)
            }
            sumPipeline = try pipeline("gainGridSum")
            meanPipeline = try pipeline("gainGridMean")
            applyPipeline = try pipeline("gainGridApply")
        } catch {
            VergenceLog.pipeline.error("gain grid: kernels failed: \(error, privacy: .public)")
            return nil
        }
        let descriptor = MTLTextureDescriptor()
        descriptor.textureType = .type3D
        descriptor.pixelFormat = .r32Float
        descriptor.width = GainGrid.width
        descriptor.height = GainGrid.height
        descriptor.depth = GainGrid.lumaBins
        descriptor.usage = .shaderRead
        descriptor.storageMode = .shared
        guard let grid = device.makeTexture(descriptor: descriptor),
              let partials = device.makeBuffer(length: 8 * Self.maxGroups, options: .storageModeShared),
              let result = device.makeBuffer(length: 8, options: .storageModeShared) else { return nil }
        self.device = device
        self.queue = queue
        gridTexture = grid
        self.partials = partials
        self.result = result
        CVMetalTextureCacheCreate(nil, nil, device, nil, &textureCache)
    }

    struct Params {
        var cap: Float
        var groups: UInt32
    }

    /// `sdr` (BGRA, BT.709 SDR) with `grid` applied, into `output`
    /// (`kCVPixelFormatType_64RGBAHalf`, the same size). Synchronous; false if
    /// the GPU couldn't run it.
    /// - Parameter cap: the most the frame's average gain may reach (linear).
    public func apply(_ grid: GainGrid, to sdr: CVPixelBuffer, into output: CVPixelBuffer, cap: Float) -> Bool {
        guard CVPixelBufferGetPixelFormatType(sdr) == kCVPixelFormatType_32BGRA,
              CVPixelBufferGetPixelFormatType(output) == kCVPixelFormatType_64RGBAHalf,
              CVPixelBufferGetWidth(sdr) == CVPixelBufferGetWidth(output),
              CVPixelBufferGetHeight(sdr) == CVPixelBufferGetHeight(output),
              let input = texture(sdr, .bgra8Unorm), let out = texture(output, .rgba16Float),
              let commands = queue.makeCommandBuffer(), let encoder = commands.makeComputeCommandEncoder() else { return false }
        // The grid's planar layout is the 3D texture's: x, then y, then luma.
        grid.log2Gain.withUnsafeBytes { raw in
            gridTexture.replace(region: MTLRegionMake3D(0, 0, 0, GainGrid.width, GainGrid.height, GainGrid.lumaBins),
                                mipmapLevel: 0, slice: 0, withBytes: raw.baseAddress!,
                                bytesPerRow: 4 * GainGrid.width, bytesPerImage: 4 * GainGrid.width * GainGrid.height)
        }
        let width = CVPixelBufferGetWidth(sdr), height = CVPixelBufferGetHeight(sdr)
        // Pass 1 covers every fourth pixel: 16×16 threads a group.
        let sampledW = (width + 3) / 4, sampledH = (height + 3) / 4
        var groupsW = (sampledW + 15) / 16, groupsH = (sampledH + 15) / 16
        while groupsW * groupsH > Self.maxGroups { groupsW = (groupsW + 1) / 2; groupsH = (groupsH + 1) / 2 }
        var params = Params(cap: cap, groups: UInt32(groupsW * groupsH))
        encoder.setTexture(input.texture, index: 0)
        encoder.setTexture(gridTexture, index: 1)
        encoder.setTexture(out.texture, index: 2)
        encoder.setBuffer(partials, offset: 0, index: 0)
        encoder.setBuffer(result, offset: 0, index: 1)
        encoder.setBytes(&params, length: MemoryLayout<Params>.stride, index: 2)
        encoder.setComputePipelineState(sumPipeline)
        encoder.dispatchThreadgroups(MTLSize(width: groupsW, height: groupsH, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        encoder.setComputePipelineState(meanPipeline)
        encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.setComputePipelineState(applyPipeline)
        encoder.dispatchThreadgroups(MTLSize(width: (width + 15) / 16, height: (height + 15) / 16, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        encoder.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        withExtendedLifetime((input.held, out.held)) {}
        guard commands.status == .completed else { return false }
        let values = result.contents().assumingMemoryBound(to: Float.self)
        lastMeanGain = values[0]
        lastScale = values[1]
        return true
    }

    private func texture(_ buffer: CVPixelBuffer, _ format: MTLPixelFormat) -> (texture: MTLTexture, held: CVMetalTexture)? {
        guard let textureCache else { return nil }
        var cvTexture: CVMetalTexture?
        guard CVMetalTextureCacheCreateTextureFromImage(nil, textureCache, buffer, nil, format,
                                                        CVPixelBufferGetWidth(buffer), CVPixelBufferGetHeight(buffer),
                                                        0, &cvTexture) == kCVReturnSuccess,
              let cvTexture, let texture = CVMetalTextureGetTexture(cvTexture) else { return nil }
        return (texture, cvTexture)
    }

    static let source = """
    #include <metal_stdlib>
    using namespace metal;

    struct Params { float cap; uint groups; };

    // Normalised (u, v, Y'), linear filtering, clamp to edge: GainGrid.lookup.
    static float log2Gain(texture3d<float> grid, float2 uv, float3 rgb) {
        constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);
        float luma = dot(saturate(rgb), float3(0.2126, 0.7152, 0.0722));
        return grid.sample(s, float3(uv, luma)).r;
    }

    kernel void gainGridSum(texture2d<float> sdr [[texture(0)]],
                            texture3d<float> grid [[texture(1)]],
                            device float *partials [[buffer(0)]],
                            constant Params &p [[buffer(2)]],
                            uint2 gid [[thread_position_in_grid]],
                            uint2 group [[threadgroup_position_in_grid]],
                            uint2 groups [[threadgroups_per_grid]],
                            uint index [[thread_index_in_threadgroup]]) {
        threadgroup float sums[256];
        threadgroup float counts[256];
        float sum = 0, count = 0;
        uint w = sdr.get_width(), h = sdr.get_height();
        // A group may stand for more than its 16×16 when the frame is large: stride over the rest.
        uint stepX = groups.x * 16 * 4, stepY = groups.y * 16 * 4;
        for (uint y = gid.y * 4; y < h; y += stepY) {
            for (uint x = gid.x * 4; x < w; x += stepX) {
                float2 uv = (float2(x, y) + 0.5) / float2(w, h);
                sum += exp2(log2Gain(grid, uv, sdr.read(uint2(x, y)).rgb));
                count += 1;
            }
        }
        sums[index] = sum;
        counts[index] = count;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 128; stride > 0; stride >>= 1) {
            if (index < stride) { sums[index] += sums[index + stride]; counts[index] += counts[index + stride]; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (index == 0) {
            uint g = group.y * groups.x + group.x;
            partials[2 * g] = sums[0];
            partials[2 * g + 1] = counts[0];
        }
    }

    kernel void gainGridMean(device float *partials [[buffer(0)]],
                             device float *result [[buffer(1)]],
                             constant Params &p [[buffer(2)]],
                             uint index [[thread_index_in_threadgroup]]) {
        threadgroup float sums[256];
        threadgroup float counts[256];
        float sum = 0, count = 0;
        for (uint g = index; g < p.groups; g += 256) { sum += partials[2 * g]; count += partials[2 * g + 1]; }
        sums[index] = sum;
        counts[index] = count;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 128; stride > 0; stride >>= 1) {
            if (index < stride) { sums[index] += sums[index + stride]; counts[index] += counts[index + stride]; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (index == 0) {
            float mean = counts[0] > 0 ? sums[0] / counts[0] : 1;
            // GainGrid.cappedScale.
            float scale = p.cap <= 1 ? 0 : (mean <= p.cap ? 1 : log2(p.cap) / log2(mean));
            result[0] = mean;
            result[1] = scale;
        }
    }

    kernel void gainGridApply(texture2d<float> sdr [[texture(0)]],
                              texture3d<float> grid [[texture(1)]],
                              texture2d<float, access::write> out [[texture(2)]],
                              device float *result [[buffer(1)]],
                              uint2 gid [[thread_position_in_grid]]) {
        uint w = out.get_width(), h = out.get_height();
        if (gid.x >= w || gid.y >= h) return;
        float4 pixel = sdr.read(gid);
        float2 uv = (float2(gid) + 0.5) / float2(w, h);
        float gain = exp2(log2Gain(grid, uv, pixel.rgb) * result[1]);
        float3 linear = pow(saturate(pixel.rgb), 2.4);
        out.write(float4(linear * gain, pixel.a), gid);
    }
    """
}
#endif
