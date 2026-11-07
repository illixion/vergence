/*
 Vergence - keeping an upscaled stereo pair's detail in agreement

 An upscaling network run on each eye alone sharpens and invents fine detail
 in each eye on its own, and where the two disagree the eyes see different
 textures at the same place: shimmer that no depth explains. Measured on a
 3D film (720×400 eyes, 2×), the general network added a third more detail
 than Lanczos and a quarter more left/right disagreement in it.

 This pulls the detail of each eye halfway towards the other's, where both
 eyes see the same point:
 1. Disparity at the source's size, both ways: SAD block matching on luma
    (7×7, ±4 % of the width, sub-pixel), no model; `vergence-convert` does
    the same on the CPU (`convert/src/upscale.rs`). Pixels without texture,
    without one clear best match, or whose match doesn't match back (a
    left-right check, so occlusions drop out) count as unknown.
 2. Each upscaled eye's detail band: the picture less a Gaussian blur of it
    (σ 1.5 output pixels), blurred within its own eye.
 3. out = eye + ½ · confidence · (other eye's detail, warped by the
    disparity − this eye's detail), the difference held to ±10 code values
    (a match that is wrong but passed the checks would otherwise leave a
    speck of the wrong texture).
 The coarse picture of each eye is untouched, so nothing moves in depth;
 where the disparity is unknown the eye stays as the network made it. On
 the same film this took the disagreement down by a quarter in a dialogue
 scene and 8 % in a dark, fast one, against an independent flow measurement
 (DIS), for 2-4 % less detail.
 */

#if canImport(Metal) && canImport(CoreVideo)
import Metal

final class StereoDetailFusion {
    struct Params {
        var eyeOrigin0 = SIMD2<Float>(), eyeSpan0 = SIMD2<Float>(), eyeOrigin1 = SIMD2<Float>(), eyeSpan1 = SIMD2<Float>()
        var eyeSize = SIMD2<UInt32>(), placement0 = SIMD2<UInt32>(), placement1 = SIMD2<UInt32>(), destination = SIMD2<UInt32>()
        var radius: Int32 = 16
        var weight: Float = 0.5
        var textureFloor: Float = 0.008
        var direction: UInt32 = 0
        var limit: Float = 10.0 / 255
    }

    private let device: MTLDevice
    private let luma, match, check, blur, fuse: MTLComputePipelineState
    private var lumaTexture, flowTexture, confidenceTexture: MTLTexture?
    private var stage, blurredA, blurredB: MTLTexture?

    init(device: MTLDevice) async throws {
        self.device = device
        let library = try await device.makeLibrary(source: Self.shaders, options: nil)
        func pipeline(_ name: String) async throws -> MTLComputePipelineState {
            guard let f = library.makeFunction(name: name) else { throw FrameUpscaler.Failure("no kernel \(name)") }
            return try await device.makeComputePipelineState(function: f)
        }
        luma = try await pipeline("stereo_luma")
        match = try await pipeline("stereo_match")
        check = try await pipeline("stereo_check")
        blur = try await pipeline("stereo_blur")
        fuse = try await pipeline("stereo_fuse")
    }

    /// Where the network's output should go first: a stage the size of the
    /// destination, fused into the destination by `encode`.
    func stage(width: Int, height: Int) throws -> MTLTexture {
        if let stage, stage.width == width, stage.height == height { return stage }
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false)
        d.usage = [.shaderRead, .shaderWrite]
        d.storageMode = .private
        guard let s = device.makeTexture(descriptor: d), let a = device.makeTexture(descriptor: d),
              let b = device.makeTexture(descriptor: d) else { throw FrameUpscaler.Failure("out of GPU memory") }
        stage = s; blurredA = a; blurredB = b
        return s
    }

    private func eyeTextures(width: Int, height: Int) throws {
        if let lumaTexture, lumaTexture.width == width, lumaTexture.height == height * 2 { return }
        func make(_ format: MTLPixelFormat) throws -> MTLTexture {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: width, height: height * 2, mipmapped: false)
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .private
            guard let t = device.makeTexture(descriptor: d) else { throw FrameUpscaler.Failure("out of GPU memory") }
            return t
        }
        lumaTexture = try make(.r16Float)
        flowTexture = try make(.rg16Float)
        confidenceTexture = try make(.rg16Float)
    }

    /// Disparity from `source` (both eyes at `params.eyeSize`), then the
    /// stage's detail fused into `destination`.
    func encode(_ commands: MTLCommandBuffer, source: MTLTexture, destination: MTLTexture, params: Params) throws {
        let w = Int(params.eyeSize.x), h = Int(params.eyeSize.y)
        try eyeTextures(width: w, height: h)
        guard let stage, let blurredA, let blurredB, let lumaTexture, let flowTexture, let confidenceTexture,
              let encoder = commands.makeComputeCommandEncoder() else { throw FrameUpscaler.Failure("no stage") }
        var p = params
        func dispatch(_ pipeline: MTLComputePipelineState, _ textures: [MTLTexture], _ size: MTLSize) {
            encoder.setComputePipelineState(pipeline)
            encoder.setTextures(textures, range: 0 ..< textures.count)
            encoder.setBytes(&p, length: MemoryLayout<Params>.stride, index: 0)
            let tw = pipeline.threadExecutionWidth
            encoder.dispatchThreads(size, threadsPerThreadgroup: MTLSize(width: tw, height: max(1, pipeline.maxTotalThreadsPerThreadgroup / tw / 2), depth: 1))
        }
        let eyes = MTLSize(width: w, height: h, depth: 2)
        let picture = MTLSize(width: Int(params.destination.x), height: Int(params.destination.y), depth: 1)
        dispatch(luma, [source, lumaTexture], eyes)
        dispatch(match, [lumaTexture, flowTexture], eyes)
        dispatch(check, [flowTexture, confidenceTexture], eyes)
        p.direction = 0
        dispatch(blur, [stage, blurredA], picture)
        p.direction = 1
        dispatch(blur, [blurredA, blurredB], picture)
        dispatch(fuse, [stage, blurredB, confidenceTexture, destination], MTLSize(width: w * 2, height: h * 2, depth: 2))
        encoder.endEncoding()
    }

    static let shaders = """
    #include <metal_stdlib>
    using namespace metal;

    struct Params {
        float2 eyeOrigin0, eyeSpan0, eyeOrigin1, eyeSpan1;
        uint2 eyeSize, placement0, placement1, destination;
        int radius;
        float weight;
        float textureFloor;
        uint direction;
        float limit;
    };

    // Both eyes' luma at the source's eye size, stacked: eye e at rows e·H.
    kernel void stereo_luma(texture2d<float, access::sample> source [[texture(0)]],
                            texture2d<half, access::write> luma [[texture(1)]],
                            constant Params& p [[buffer(0)]], uint3 gid [[thread_position_in_grid]]) {
        if (gid.x >= p.eyeSize.x || gid.y >= p.eyeSize.y) return;
        float2 origin = gid.z == 0 ? p.eyeOrigin0 : p.eyeOrigin1;
        float2 span = gid.z == 0 ? p.eyeSpan0 : p.eyeSpan1;
        constexpr sampler linear(filter::linear, address::clamp_to_edge, coord::normalized);
        float3 c = source.sample(linear, origin + (float2(gid.xy) + 0.5) / float2(p.eyeSize) * span).rgb;
        luma.write(half4(half(dot(saturate(c), float3(0.2126, 0.7152, 0.0722)))), uint2(gid.x, gid.y + gid.z * p.eyeSize.y));
    }

    static float at(texture2d<half, access::read> luma, int2 q, uint eye, uint2 size) {
        q = clamp(q, int2(0), int2(size) - 1);
        return float(luma.read(uint2(q.x, q.y + int(eye * size.y))).r);
    }

    // Eye e's disparity to the other: e(x) matches other(x + d), d in ±radius;
    // (d, 1) where one match is clearly best, (0, 0) where none is.
    kernel void stereo_match(texture2d<half, access::read> luma [[texture(0)]],
                             texture2d<half, access::write> flow [[texture(1)]],
                             constant Params& p [[buffer(0)]], uint3 gid [[thread_position_in_grid]]) {
        if (gid.x >= p.eyeSize.x || gid.y >= p.eyeSize.y) return;
        uint e = gid.z, o = 1 - e;
        int2 x = int2(gid.xy);
        float own[49];
        float sum = 0, squares = 0;
        for (int j = 0; j < 7; j++) for (int i = 0; i < 7; i++) {
            float v = at(luma, x + int2(i - 3, j - 3), e, p.eyeSize);
            own[j * 7 + i] = v;
            sum += v;
            squares += v * v;
        }
        float mean = sum / 49;
        uint2 out = uint2(gid.x, gid.y + e * p.eyeSize.y);
        // Too flat to match: the window's standard deviation.
        if (sqrt(max(squares / 49 - mean * mean, 0.0)) < p.textureFloor) { flow.write(half4(0), out); return; }
        int radius = min(p.radius, 48);
        float costs[97];
        float best = INFINITY;
        int bestD = 0;
        for (int d = -radius; d <= radius; d++) {
            float cost = 0;
            for (int j = 0; j < 7; j++) for (int i = 0; i < 7; i++)
                cost += fabs(own[j * 7 + i] - at(luma, x + int2(i - 3 + d, j - 3), o, p.eyeSize));
            costs[d + radius] = cost;
            if (cost < best) { best = cost; bestD = d; }
        }
        float second = INFINITY;
        for (int d = -radius; d <= radius; d++) if (abs(d - bestD) > 1) second = min(second, costs[d + radius]);
        float before = bestD > -radius ? costs[bestD - 1 + radius] : INFINITY;
        float after = bestD < radius ? costs[bestD + 1 + radius] : INFINITY;
        if (!(best < second * 0.95)) { flow.write(half4(0), out); return; }
        float offset = 0;
        if (isfinite(before) && isfinite(after)) {
            float curve = before - 2 * best + after;
            if (curve > 0) offset = clamp(0.5 * (before - after) / curve, -0.5, 0.5);
        }
        flow.write(half4(half(float(bestD) + offset), 1, 0, 0), out);
    }

    // Kept where the other eye's match points back.
    kernel void stereo_check(texture2d<half, access::read> flow [[texture(0)]],
                             texture2d<half, access::write> confidence [[texture(1)]],
                             constant Params& p [[buffer(0)]], uint3 gid [[thread_position_in_grid]]) {
        if (gid.x >= p.eyeSize.x || gid.y >= p.eyeSize.y) return;
        uint e = gid.z, o = 1 - e;
        uint2 out = uint2(gid.x, gid.y + e * p.eyeSize.y);
        half2 f = flow.read(out).rg;
        int target = int(round(float(gid.x) + float(f.x)));
        if (f.y < 0.5 || target < 0 || target >= int(p.eyeSize.x)) { confidence.write(half4(0), out); return; }
        half2 g = flow.read(uint2(target, gid.y + o * p.eyeSize.y)).rg;
        bool agrees = g.y > 0.5 && fabs(float(f.x) + float(g.x)) <= 1.0;
        confidence.write(half4(f.x, agrees ? 1 : 0, 0, 0), out);
    }

    static bool inEye(uint2 q, uint2 origin, uint2 size) {
        return q.x >= origin.x && q.y >= origin.y && q.x < origin.x + size.x && q.y < origin.y + size.y;
    }

    // Gaussian, σ 1.5, one direction, kept within each eye.
    kernel void stereo_blur(texture2d<half, access::read> input [[texture(0)]],
                            texture2d<half, access::write> output [[texture(1)]],
                            constant Params& p [[buffer(0)]], uint3 gid [[thread_position_in_grid]]) {
        if (gid.x >= p.destination.x || gid.y >= p.destination.y) return;
        uint2 size = p.eyeSize * 2;
        uint2 origin = inEye(gid.xy, p.placement1, size) ? p.placement1 : p.placement0;
        int2 low = int2(origin), high = int2(origin + size) - 1;
        const float w[5] = {0.2666, 0.2134, 0.1096, 0.0361, 0.0076};
        int2 step = p.direction == 0 ? int2(1, 0) : int2(0, 1);
        half4 sum = input.read(gid.xy) * half(w[0]);
        for (int k = 1; k < 5; k++) {
            sum += input.read(uint2(clamp(int2(gid.xy) + step * k, low, high))) * half(w[k]);
            sum += input.read(uint2(clamp(int2(gid.xy) - step * k, low, high))) * half(w[k]);
        }
        output.write(sum, gid.xy);
    }

    kernel void stereo_fuse(texture2d<half, access::sample> picture [[texture(0)]],
                            texture2d<half, access::sample> blurred [[texture(1)]],
                            texture2d<half, access::sample> confidence [[texture(2)]],
                            texture2d<float, access::write> destination [[texture(3)]],
                            constant Params& p [[buffer(0)]], uint3 gid [[thread_position_in_grid]]) {
        uint2 size = p.eyeSize * 2;
        if (gid.x >= size.x || gid.y >= size.y) return;
        uint e = gid.z;
        uint2 own = e == 0 ? p.placement0 : p.placement1, other = e == 0 ? p.placement1 : p.placement0;
        uint2 at = own + gid.xy;
        if (at.x >= p.destination.x || at.y >= p.destination.y) return;
        float3 mine = float3(picture.read(at).rgb);
        float3 detail = mine - float3(blurred.read(at).rgb);
        constexpr sampler linear(filter::linear, address::clamp_to_edge, coord::normalized);
        float2 low = (float2(gid.xy) + 0.5) / 2 - 0.5;
        low = clamp(low, float2(0), float2(p.eyeSize) - 1);
        float2 c = float2(confidence.sample(linear, (low + 0.5 + float2(0, e * p.eyeSize.y))
                                            / float2(p.eyeSize.x, p.eyeSize.y * 2)).rg);
        float x = float(gid.x) + 2 * c.x;
        float3 out = mine;
        if (c.y > 0.01 && x >= 0 && x <= float(size.x) - 1) {
            float2 there = (float2(other) + float2(x, float(gid.y)) + 0.5) / float2(p.destination);
            float3 theirs = float3(picture.sample(linear, there).rgb) - float3(blurred.sample(linear, there).rgb);
            // Held to ±limit: a wrong match moves a little detail, never a speck.
            out = mine + p.weight * c.y * clamp(theirs - detail, -p.limit, p.limit);
        }
        destination.write(float4(saturate(out), 1.0), at);
    }
    """
}
#endif
