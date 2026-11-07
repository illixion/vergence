/*
 Vergence - AI HDR for video on the player: the gain-grid lookup on the GPU

 Turns an SDR frame into HDR with a worker's gain grid (`GainGrid`, the
 `ai-hdr` workload): each pixel's gain is the grid's 3D texture read at its
 own (u, v, Y′), and it multiplies the pixel's linear light. Everything
 before it (interpolation, depth, the warp) stays SDR; this is the last step
 (Hypnos `docs/plans/analysis-workers.md`, "AI HDR for video, in short").

 The pieces, each used where a player already draws:
 - `encodeMean`: every fourth pixel of the SDR source each way, the linear
   gain it would get, reduced to the frame's mean; then the scale every log
   gain is multiplied by so that mean stays at `cap` or below
   (`GainGrid.cappedScale`: the window brightness limiter dims the whole
   window past about 2.5–3× on a Vision Pro), times the ramp's strength. The
   result buffer holds (mean, scale) for the shader that applies it.
 - `encodePQEyes`: for AI 3D, the pump's warped half-float eyes, still SDR
   code values, rewritten in place as BT.2020 PQ (each eye copied aside and
   drawn back; framebuffer fetch isn't there on every GPU, the simulator's
   included): SDR
   white at 203 nits (BT.2408), so the eyes go on through the pump's 10-bit
   `xf20` path like any HDR picture. One pass after the warp covers every
   warp variant (mesh, layered, boundary) without touching their shaders.
 - Flat video applies the lookup in Convolution's own `imageFragmentShader`
   variant, reading the same grid texture and result buffer.
 - `apply`: the whole thing on one frame into a half-float buffer, the
   reference the tests check against `GainGrid.lookup`.

 SDR is decoded with BT.1886's 2.4 power; the gain multiplies linear light
 in BT.709 primaries, 1.0 = SDR white.
 */

#if canImport(Metal)
import CoreVideo
import Foundation
import Metal
import VergenceCore

public final class GainGridPass: @unchecked Sendable {
    public let device: MTLDevice
    private let queue: MTLCommandQueue
    private let sumPipeline: MTLComputePipelineState
    private let meanPipeline: MTLComputePipelineState
    private let applyPipeline: MTLComputePipelineState
    private let pqEyePipeline: MTLRenderPipelineState
    private var textureCache: CVMetalTextureCache?
    private static let maxGroups = 4096
    /// Reference white of an SDR picture made HDR, in nits (BT.2408).
    public static let sdrWhiteNits: Float = 203

    /// Per-frame resources, cycled: a frame's may still be read while the next is encoded.
    public final class Slot: @unchecked Sendable {
        public let grid: MTLTexture
        let partials: MTLBuffer
        /// (mean linear gain before the cap, the scale applied to log gains).
        public let result: MTLBuffer
        init(grid: MTLTexture, partials: MTLBuffer, result: MTLBuffer) {
            self.grid = grid
            self.partials = partials
            self.result = result
        }
    }
    private var slots: [Slot] = []
    private var nextSlot = 0
    private let lock = NSLock()

    /// The last frame's mean linear gain before the cap, and the scale applied (after it completed).
    public var lastMeanGain: Float { lock.withLock { lastResult.0 } }
    public var lastScale: Float { lock.withLock { lastResult.1 } }
    private var lastResult: (Float, Float) = (1, 1)

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
            let render = MTLRenderPipelineDescriptor()
            render.vertexFunction = library.makeFunction(name: "gainGridQuad")
            render.fragmentFunction = library.makeFunction(name: "gainGridPQEye")
            render.colorAttachments[0].pixelFormat = .rgba16Float
            pqEyePipeline = try device.makeRenderPipelineState(descriptor: render)
        } catch {
            VergenceLog.pipeline.error("gain grid: kernels failed: \(error, privacy: .public)")
            return nil
        }
        self.device = device
        self.queue = queue
        for _ in 0 ..< 4 {
            guard let slot = Self.makeSlot(device) else { return nil }
            slots.append(slot)
        }
        CVMetalTextureCacheCreate(nil, nil, device, nil, &textureCache)
    }

    private static func makeSlot(_ device: MTLDevice) -> Slot? {
        let descriptor = MTLTextureDescriptor()
        descriptor.textureType = .type3D
        descriptor.pixelFormat = .r32Float
        descriptor.width = GainGrid.width
        descriptor.height = GainGrid.height
        descriptor.depth = GainGrid.lumaBins
        descriptor.usage = .shaderRead
        descriptor.storageMode = .shared
        guard let grid = device.makeTexture(descriptor: descriptor),
              let partials = device.makeBuffer(length: 8 * maxGroups, options: .storageModeShared),
              let result = device.makeBuffer(length: 8, options: .storageModeShared) else { return nil }
        return Slot(grid: grid, partials: partials, result: result)
    }

    struct EyeParams {
        var size: SIMD2<Float>
        var whiteNits: Float
        var padding: Float
    }

    struct Params {
        var cap: Float
        var groups: UInt32
        var strength: Float
        var whiteNits: Float
    }

    /// Uploads `grid` into the next slot and encodes the frame's mean and
    /// scale for it (from `sdr`, the picture as decoded). Returns the slot,
    /// whose `grid` and `result` the applying shader reads.
    /// - Parameters:
    ///   - cap: the most the frame's average gain may reach (linear).
    ///   - strength: the ramp, 0…1, folded into the scale.
    public func encodeMean(of sdr: MTLTexture, grid: GainGrid, cap: Float, strength: Float,
                           into commands: MTLCommandBuffer) -> Slot? {
        let slot = lock.withLock { () -> Slot in
            defer { nextSlot = (nextSlot + 1) % slots.count }
            return slots[nextSlot]
        }
        grid.log2Gain.withUnsafeBytes { raw in
            slot.grid.replace(region: MTLRegionMake3D(0, 0, 0, GainGrid.width, GainGrid.height, GainGrid.lumaBins),
                              mipmapLevel: 0, slice: 0, withBytes: raw.baseAddress!,
                              bytesPerRow: 4 * GainGrid.width, bytesPerImage: 4 * GainGrid.width * GainGrid.height)
        }
        guard let encoder = commands.makeComputeCommandEncoder() else { return nil }
        let width = sdr.width, height = sdr.height
        let sampledW = (width + 3) / 4, sampledH = (height + 3) / 4
        var groupsW = (sampledW + 15) / 16, groupsH = (sampledH + 15) / 16
        while groupsW * groupsH > Self.maxGroups { groupsW = (groupsW + 1) / 2; groupsH = (groupsH + 1) / 2 }
        var params = Params(cap: cap, groups: UInt32(groupsW * groupsH), strength: min(max(strength, 0), 1),
                            whiteNits: Self.sdrWhiteNits)
        encoder.setTexture(sdr, index: 0)
        encoder.setTexture(slot.grid, index: 1)
        encoder.setBuffer(slot.partials, offset: 0, index: 0)
        encoder.setBuffer(slot.result, offset: 0, index: 1)
        encoder.setBytes(&params, length: MemoryLayout<Params>.stride, index: 2)
        encoder.setComputePipelineState(sumPipeline)
        encoder.dispatchThreadgroups(MTLSize(width: groupsW, height: groupsH, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        encoder.setComputePipelineState(meanPipeline)
        encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        commands.addCompletedHandler { [weak self] _ in
            let values = slot.result.contents().assumingMemoryBound(to: Float.self)
            let read = (values[0], values[1])
            self?.lock.withLock { self?.lastResult = read }
        }
        return slot
    }

    /// Rewrites warped SDR eyes (half float, SDR code values) as BT.2020 PQ with
    /// `slot`'s grid and scale, in place.
    public func encodePQEyes(_ eyes: [MTLTexture], slot: Slot, into commands: MTLCommandBuffer) -> Bool {
        for eye in eyes {
            guard let copy = scratch(like: eye), let blit = commands.makeBlitCommandEncoder() else { return false }
            blit.copy(from: eye, to: copy)
            blit.endEncoding()
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = eye
            pass.colorAttachments[0].loadAction = .dontCare
            pass.colorAttachments[0].storeAction = .store
            guard let encoder = commands.makeRenderCommandEncoder(descriptor: pass) else { return false }
            var params = EyeParams(size: SIMD2(Float(eye.width), Float(eye.height)), whiteNits: Self.sdrWhiteNits,
                                   padding: 0)
            encoder.setRenderPipelineState(pqEyePipeline)
            encoder.setFragmentTexture(slot.grid, index: 0)
            encoder.setFragmentTexture(copy, index: 1)
            encoder.setFragmentBuffer(slot.result, offset: 0, index: 1)
            encoder.setFragmentBytes(&params, length: MemoryLayout<EyeParams>.stride, index: 2)
            encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            encoder.endEncoding()
        }
        return true
    }

    /// A private texture the eye is copied into before it is drawn back
    /// (Metal orders the copy, the draw and the next eye's copy).
    private var scratchTexture: MTLTexture?
    private func scratch(like eye: MTLTexture) -> MTLTexture? {
        lock.withLock {
            if let scratchTexture, scratchTexture.width == eye.width, scratchTexture.height == eye.height,
               scratchTexture.pixelFormat == eye.pixelFormat { return scratchTexture }
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: eye.pixelFormat, width: eye.width,
                                                                      height: eye.height, mipmapped: false)
            descriptor.usage = .shaderRead
            descriptor.storageMode = .private
            scratchTexture = device.makeTexture(descriptor: descriptor)
            return scratchTexture
        }
    }

    /// `sdr` (BGRA, BT.709 SDR) with `grid` applied, into `output`
    /// (`kCVPixelFormatType_64RGBAHalf`, the same size) as extended linear
    /// light, 1.0 = SDR white. Synchronous, for checks; false if the GPU
    /// couldn't run it.
    public func apply(_ grid: GainGrid, to sdr: CVPixelBuffer, into output: CVPixelBuffer, cap: Float,
                      strength: Float = 1) -> Bool {
        guard CVPixelBufferGetPixelFormatType(sdr) == kCVPixelFormatType_32BGRA,
              CVPixelBufferGetPixelFormatType(output) == kCVPixelFormatType_64RGBAHalf,
              CVPixelBufferGetWidth(sdr) == CVPixelBufferGetWidth(output),
              CVPixelBufferGetHeight(sdr) == CVPixelBufferGetHeight(output),
              let input = texture(sdr, .bgra8Unorm), let out = texture(output, .rgba16Float),
              let commands = queue.makeCommandBuffer(),
              let slot = encodeMean(of: input.texture, grid: grid, cap: cap, strength: strength, into: commands),
              let encoder = commands.makeComputeCommandEncoder() else { return false }
        let width = out.texture.width, height = out.texture.height
        encoder.setComputePipelineState(applyPipeline)
        encoder.setTexture(input.texture, index: 0)
        encoder.setTexture(slot.grid, index: 1)
        encoder.setTexture(out.texture, index: 2)
        encoder.setBuffer(slot.result, offset: 0, index: 1)
        encoder.dispatchThreadgroups(MTLSize(width: (width + 15) / 16, height: (height + 15) / 16, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        encoder.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        withExtendedLifetime((input.held, out.held)) {}
        record(slot)
        return commands.status == .completed
    }

    private func record(_ slot: Slot) {
        let values = slot.result.contents().assumingMemoryBound(to: Float.self)
        let read = (values[0], values[1])
        lock.withLock { lastResult = read }
    }

    /// Synchronous: the PQ eye pass over one half-float buffer holding SDR
    /// code values, for checks.
    public func applyPQ(_ grid: GainGrid, source sdr: CVPixelBuffer, eye: CVPixelBuffer, cap: Float,
                        strength: Float = 1) -> Bool {
        guard let input = texture(sdr, .bgra8Unorm), let out = texture(eye, .rgba16Float),
              let commands = queue.makeCommandBuffer(),
              let slot = encodeMean(of: input.texture, grid: grid, cap: cap, strength: strength, into: commands),
              encodePQEyes([out.texture], slot: slot, into: commands) else { return false }
        commands.commit()
        commands.waitUntilCompleted()
        withExtendedLifetime((input.held, out.held)) {}
        record(slot)
        return commands.status == .completed
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

    /// HDR10 static metadata for an SDR picture made HDR: a P3 D65 mastering
    /// display of 1,000 nits in a BT.2020 container, MaxCLL at GMNet's peak
    /// (5× SDR white) and MaxFALL at the default cap. Big-endian, as
    /// CoreVideo's attachments carry them (SMPTE ST 2086, CTA-861.3).
    public static let staticMetadata: (masteringDisplay: Data, contentLightLevel: Data) = {
        var mastering = Data()
        func u16(_ v: Double, _ unit: Double) { mastering.append(contentsOf: withUnsafeBytes(of: UInt16(v / unit).bigEndian, Array.init)) }
        func u32(_ v: Double, _ unit: Double) { mastering.append(contentsOf: withUnsafeBytes(of: UInt32(v / unit).bigEndian, Array.init)) }
        // Green, blue, red, then the white point, in 0.00002 steps.
        for (x, y) in [(0.265, 0.690), (0.150, 0.060), (0.680, 0.320), (0.3127, 0.3290)] {
            u16(x, 0.00002)
            u16(y, 0.00002)
        }
        u32(1000, 0.0001)
        u32(0.005, 0.0001)
        var light = Data()
        light.append(contentsOf: withUnsafeBytes(of: UInt16(sdrWhiteNits * 5).bigEndian, Array.init))
        light.append(contentsOf: withUnsafeBytes(of: UInt16(sdrWhiteNits * GainGridPlayback.defaultCap).bigEndian, Array.init))
        return (mastering, light)
    }()

    /// The Metal for the lookup, for a host shader to include: `gainGridLog2`
    /// reads a pixel's log gain from the grid texture, `gainGridLinear`
    /// decodes an SDR value. Convolution's flat-video shader carries the same.
    public static let lookupSource = """
    static float gainGridLog2(texture3d<float> grid, float2 uv, float3 sdr) {
        constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);
        float luma = dot(saturate(sdr), float3(0.2126, 0.7152, 0.0722));
        return grid.sample(s, float3(uv, luma)).r;
    }
    static float3 gainGridLinear(float3 sdr) { return pow(saturate(sdr), 2.4); }
    """

    static let source = """
    #include <metal_stdlib>
    using namespace metal;

    struct Params { float cap; uint groups; float strength; float whiteNits; };

    \(lookupSource)

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
                sum += exp2(gainGridLog2(grid, uv, sdr.read(uint2(x, y)).rgb));
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
            // GainGrid.cappedScale, then the ramp.
            float scale = p.cap <= 1 ? 0 : (mean <= p.cap ? 1 : log2(p.cap) / log2(mean));
            result[0] = mean;
            result[1] = scale * p.strength;
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
        float gain = exp2(gainGridLog2(grid, uv, pixel.rgb) * result[1]);
        out.write(float4(gainGridLinear(pixel.rgb) * gain, pixel.a), gid);
    }

    struct QuadOut { float4 position [[position]]; };

    vertex QuadOut gainGridQuad(uint id [[vertex_id]]) {
        float2 p = float2((id << 1) & 2, id & 2);
        QuadOut out;
        out.position = float4(p * 2.0 - 1.0, 0, 1);
        return out;
    }

    // SMPTE ST 2084 inverse EOTF.
    static float3 gainGridPQ(float3 nits) {
        const float m1 = 0.1593017578125, m2 = 78.84375;
        const float c1 = 0.8359375, c2 = 18.8515625, c3 = 18.6875;
        float3 y = pow(clamp(nits / 10000.0, 0.0, 1.0), m1);
        return pow((c1 + c2 * y) / (1.0 + c3 * y), m2);
    }

    struct EyeParams { float2 size; float whiteNits; float padding; };

    fragment float4 gainGridPQEye(QuadOut in [[stage_in]],
                                  texture3d<float> grid [[texture(0)]],
                                  texture2d<float> eyeCopy [[texture(1)]],
                                  device float *result [[buffer(1)]],
                                  constant EyeParams &p [[buffer(2)]]) {
        // The eye is its picture's size: its UV is the source's, give or take the warp's few pixels,
        // which a 16×9 grid doesn't see.
        float4 eye = eyeCopy.read(uint2(in.position.xy));
        float2 uv = in.position.xy / p.size;
        float gain = exp2(gainGridLog2(grid, uv, eye.rgb) * result[1]);
        float3 linear709 = gainGridLinear(eye.rgb) * gain;
        // BT.709 to BT.2020 primaries (ITU-R BT.2087), columns.
        const float3x3 toBT2020 = float3x3(float3(0.6274, 0.0691, 0.0164),
                                           float3(0.3293, 0.9195, 0.0880),
                                           float3(0.0433, 0.0114, 0.8956));
        return float4(gainGridPQ(max(toBT2020 * linear709, 0.0) * p.whiteNits), eye.a);
    }
    """
}
#endif
