/*
 Vergence - edge-tangent anti-aliasing: stair-stepped edges made lines

 An SD film deinterlaced by doubling each field's lines has its diagonal
 edges in stairs a line or two tall; an upscaler then sharpens those stairs
 into crisp sawtooth edges. FXAA-style passes look for steps a pixel tall and
 barely touch these (3–5 % off the staircase measure below, on a 3D film
 upscaled 2×), and re-making the fields by edge-slope tracing before
 upscaling (ffmpeg `estdif`, the old `--antialias`) took off at most 30 % for
 4–15 % of the texture.

 This pass works on the finished picture. The structure tensor of its luma
 says at every pixel which way the edge there runs and how straight it is
 over a stair's length: luma at half size blurred to σ 2 pixels (so a
 stair's tread and riser read as the slope they stand for), Sobel gradients,
 their products blurred to σ 8 pixels. Along strong, straight edges the
 picture is averaged along the edge, never across it (a Gaussian of σ 12
 pixels over ±24), which turns a staircase into the line it stands for;
 texture is not straight over that length, so it stays:

   out = picture + α · (picture averaged along the edge − picture)
   α   = smoothstep(0.45, 0.8, coherence) · smoothstep(0.01, 0.035, edge strength)

 A line drawn in stairs 4 pixels tall ends up 0.12 pixels RMS off the line
 it stands for (1.14 before). On five spots of that film after `convert
 --upscale 2x --antialias` (staircase energy: luma's second difference along
 the edge over its contrast, through the encoder) it takes 22–41 % off,
 the visible sawtooth gone, for 1–9 % of the texture elsewhere; about 5 ms a
 2880×800 frame on an M2 Max. Each eye of a packed frame is done on its own,
 so nothing crosses the seam. BGRA or half-float RGBA, in place.
 */

#if canImport(Metal) && canImport(CoreVideo)
import CoreVideo
import Foundation
import Metal

public final class EdgeAntialias: @unchecked Sendable {
    public struct Failure: Error, CustomStringConvertible {
        public let description: String
        init(_ description: String) { self.description = description }
    }

    /// One eye's rectangle in the frame, in pixels.
    public struct Eye: Sendable, Equatable {
        public var x, y, width, height: Int
        public init(x: Int, y: Int, width: Int, height: Int) {
            self.x = x
            self.y = y
            self.width = width
            self.height = height
        }
    }

    struct Params {
        var origin = SIMD2<Int32>()
        var size = SIMD2<Int32>()
        var halfOrigin = SIMD2<Int32>()
        var halfSize = SIMD2<Int32>()
        /// The Gaussian `aa_blur` runs, in half-size pixels.
        var blurSigma: Float = 1
        var blurRadius: Int32 = 3
        var alongSigma: Float = 12
        var alongRadius: Int32 = 24
        var coherence = SIMD2<Float>(0.45, 0.8)
        var strength = SIMD2<Float>(0.01, 0.035)
        var direction: UInt32 = 0
    }

    public let device: MTLDevice
    private let queue: MTLCommandQueue
    private let luma, tensor, blur, apply: MTLComputePipelineState
    private var textureCache: CVMetalTextureCache?
    private var copy: MTLTexture?
    private var tensorA, tensorB: MTLTexture?

    public init(device: MTLDevice? = MTLCreateSystemDefaultDevice()) throws {
        guard let device, let queue = device.makeCommandQueue() else { throw Failure("no Metal device") }
        self.device = device
        self.queue = queue
        let library = try device.makeLibrary(source: Self.shaders, options: nil)
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let f = library.makeFunction(name: name) else { throw Failure("no kernel \(name)") }
            return try device.makeComputePipelineState(function: f)
        }
        luma = try pipeline("aa_luma")
        tensor = try pipeline("aa_tensor")
        blur = try pipeline("aa_blur")
        apply = try pipeline("aa_apply")
        CVMetalTextureCacheCreate(nil, nil, device, nil, &textureCache)
    }

    /// `frame` (BGRA or half-float RGBA) with each of `eyes` anti-aliased, in place.
    public func apply(_ frame: CVPixelBuffer, eyes: [Eye]) throws {
        let format: MTLPixelFormat
        switch CVPixelBufferGetPixelFormatType(frame) {
        case kCVPixelFormatType_32BGRA: format = .bgra8Unorm
        case kCVPixelFormatType_64RGBAHalf: format = .rgba16Float
        default: throw Failure("frames must be BGRA or half-float RGBA")
        }
        let width = CVPixelBufferGetWidth(frame), height = CVPixelBufferGetHeight(frame)
        guard let textureCache else { throw Failure("no texture cache") }
        var cv: CVMetalTexture?
        CVMetalTextureCacheCreateTextureFromImage(nil, textureCache, frame, nil, format, width, height, 0, &cv)
        guard let cv, let picture = CVMetalTextureGetTexture(cv) else { throw Failure("can't see the frame as a texture") }
        try textures(width: width, height: height, format: format)
        guard let copy, var tensorA = self.tensorA, var tensorB = self.tensorB, let commands = queue.makeCommandBuffer(),
              let blit = commands.makeBlitCommandEncoder() else { throw Failure("no command buffer") }
        blit.copy(from: picture, to: copy)
        blit.endEncoding()
        guard let encoder = commands.makeComputeCommandEncoder() else { throw Failure("no encoder") }
        for eye in eyes {
            var p = Params()
            p.origin = SIMD2(Int32(eye.x), Int32(eye.y))
            p.size = SIMD2(Int32(eye.width), Int32(eye.height))
            p.halfOrigin = SIMD2(Int32(eye.x / 2), Int32(eye.y / 2))
            p.halfSize = SIMD2(Int32(max(1, eye.width / 2)), Int32(max(1, eye.height / 2)))
            func run(_ pipeline: MTLComputePipelineState, _ textures: [MTLTexture], _ w: Int, _ h: Int) {
                encoder.setComputePipelineState(pipeline)
                encoder.setTextures(textures, range: 0 ..< textures.count)
                encoder.setBytes(&p, length: MemoryLayout<Params>.stride, index: 0)
                let tw = pipeline.threadExecutionWidth
                encoder.dispatchThreads(MTLSize(width: w, height: h, depth: 1),
                                        threadsPerThreadgroup: MTLSize(width: tw, height: max(1, pipeline.maxTotalThreadsPerThreadgroup / tw / 2), depth: 1))
            }
            let hw = Int(p.halfSize.x), hh = Int(p.halfSize.y)
            func gaussian(_ sigma: Float) {
                p.blurSigma = sigma
                p.blurRadius = Int32((3 * sigma).rounded(.up))
                p.direction = 0
                run(blur, [tensorA, tensorB], hw, hh)
                p.direction = 1
                run(blur, [tensorB, tensorA], hw, hh)
            }
            // Luma at half size, blurred to σ 2 full-size pixels, so a stair's
            // tread and riser read as the slope they stand for.
            run(luma, [copy, tensorA], hw, hh)
            gaussian(1)
            run(tensor, [tensorA, tensorB], hw, hh)
            swap(&tensorA, &tensorB)
            // The tensor integrated over σ 8 full-size pixels: a stair's length.
            gaussian(4)
            swap(&tensorA, &tensorB)
            run(apply, [copy, tensorA, picture], eye.width, eye.height)
        }
        encoder.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        withExtendedLifetime(cv) {}
        if let error = commands.error { throw Failure("the GPU failed: \(error.localizedDescription)") }
    }

    private func textures(width: Int, height: Int, format: MTLPixelFormat) throws {
        if let copy, copy.width == width, copy.height == height, copy.pixelFormat == format { return }
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: width, height: height, mipmapped: false)
        d.usage = [.shaderRead]
        d.storageMode = .private
        let t = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: max(1, (width + 1) / 2),
                                                         height: max(1, (height + 1) / 2), mipmapped: false)
        t.usage = [.shaderRead, .shaderWrite]
        t.storageMode = .private
        guard let c = device.makeTexture(descriptor: d), let a = device.makeTexture(descriptor: t),
              let b = device.makeTexture(descriptor: t) else { throw Failure("out of GPU memory") }
        copy = c
        tensorA = a
        tensorB = b
    }

    static let shaders = """
    #include <metal_stdlib>
    using namespace metal;

    struct Params {
        int2 origin, size, halfOrigin, halfSize;
        float blurSigma; int blurRadius;
        float alongSigma; int alongRadius;
        float2 coherence, strength;
        uint direction;
    };

    // Each half-size pixel: the luma of its 2×2 full-size pixels, averaged.
    kernel void aa_luma(texture2d<float, access::read> picture [[texture(0)]],
                        texture2d<float, access::write> half_ [[texture(1)]],
                        constant Params& p [[buffer(0)]],
                        uint2 gid [[thread_position_in_grid]]) {
        if (int(gid.x) >= p.halfSize.x || int(gid.y) >= p.halfSize.y) return;
        float sum = 0;
        for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++) {
            int2 at = clamp(int2(gid) * 2 + int2(dx, dy), int2(0), p.size - 1) + p.origin;
            sum += dot(picture.read(uint2(at)).rgb, float3(0.2126, 0.7152, 0.0722));
        }
        half_.write(float4(sum / 4, 0, 0, 0), uint2(int2(gid) + p.halfOrigin));
    }

    static float aaAt(texture2d<float, access::read> t, constant Params& p, int2 at) {
        return t.read(uint2(clamp(at, int2(0), p.halfSize - 1) + p.halfOrigin)).x;
    }

    // The Sobel gradient of the blurred half-size luma, per full-size pixel,
    // as its products (gx², gx·gy, gy²).
    kernel void aa_tensor(texture2d<float, access::read> lum [[texture(0)]],
                          texture2d<float, access::write> tensor [[texture(1)]],
                          constant Params& p [[buffer(0)]],
                          uint2 gid [[thread_position_in_grid]]) {
        if (int(gid.x) >= p.halfSize.x || int(gid.y) >= p.halfSize.y) return;
        int2 c = int2(gid);
        float a = aaAt(lum, p, c + int2(-1, -1)), b = aaAt(lum, p, c + int2(0, -1)), cc = aaAt(lum, p, c + int2(1, -1));
        float d = aaAt(lum, p, c + int2(-1, 0)), f = aaAt(lum, p, c + int2(1, 0));
        float g = aaAt(lum, p, c + int2(-1, 1)), h = aaAt(lum, p, c + int2(0, 1)), i = aaAt(lum, p, c + int2(1, 1));
        float gx = ((cc + 2 * f + i) - (a + 2 * d + g)) / 16;
        float gy = ((g + 2 * h + i) - (a + 2 * b + cc)) / 16;
        tensor.write(float4(gx * gx, gx * gy, gy * gy, 0), uint2(c + p.halfOrigin));
    }

    // A separable Gaussian over one eye's half-size texture, clamped to the eye.
    kernel void aa_blur(texture2d<float, access::read> source [[texture(0)]],
                        texture2d<float, access::write> target [[texture(1)]],
                        constant Params& p [[buffer(0)]],
                        uint2 gid [[thread_position_in_grid]]) {
        if (int(gid.x) >= p.halfSize.x || int(gid.y) >= p.halfSize.y) return;
        int2 step = p.direction == 0 ? int2(1, 0) : int2(0, 1);
        float4 sum = 0; float weights = 0;
        for (int k = -p.blurRadius; k <= p.blurRadius; k++) {
            float w = exp(-0.5 * float(k * k) / (p.blurSigma * p.blurSigma));
            int2 at = clamp(int2(gid) + k * step, int2(0), p.halfSize - 1) + p.halfOrigin;
            sum += w * source.read(uint2(at));
            weights += w;
        }
        target.write(sum / weights, uint2(int2(gid) + p.halfOrigin));
    }

    kernel void aa_apply(texture2d<float, access::sample> picture [[texture(0)]],
                         texture2d<float, access::sample> tensor [[texture(1)]],
                         texture2d<float, access::write> target [[texture(2)]],
                         constant Params& p [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
        if (int(gid.x) >= p.size.x || int(gid.y) >= p.size.y) return;
        constexpr sampler s(coord::pixel, filter::linear, address::clamp_to_edge);
        int2 at = int2(gid) + p.origin;
        float4 here = picture.sample(s, float2(at) + 0.5);
        // The tensor at this pixel, from the half-size one, inside the eye.
        float2 hp = clamp((float2(gid) + 0.5) / 2, float2(0.5), float2(p.halfSize) - 0.5) + float2(p.halfOrigin);
        float3 j = tensor.sample(s, hp).xyz;
        float tr = j.x + j.z, d = sqrt((j.x - j.z) * (j.x - j.z) + 4 * j.y * j.y);
        float l1 = (tr + d) / 2, l2 = (tr - d) / 2;
        float coherence = (l1 - l2) / (l1 + l2 + 1e-12); coherence *= coherence;
        float alpha = smoothstep(p.coherence.x, p.coherence.y, coherence) * smoothstep(p.strength.x, p.strength.y, sqrt(max(l1, 0.0)));
        if (alpha < 1.0 / 512) { target.write(here, uint2(at)); return; }
        float theta = 0.5 * atan2(2 * j.y, j.x - j.z);
        float2 along = float2(-sin(theta), cos(theta));
        float2 lo = float2(p.origin) + 0.5, hi = float2(p.origin + p.size) - 0.5;
        float4 sum = here; float weights = 1;
        for (int k = 1; k <= p.alongRadius; k++) {
            float w = exp(-0.5 * float(k * k) / (p.alongSigma * p.alongSigma));
            float2 o = along * float(k);
            sum += w * (picture.sample(s, clamp(float2(at) + 0.5 + o, lo, hi)) + picture.sample(s, clamp(float2(at) + 0.5 - o, lo, hi)));
            weights += 2 * w;
        }
        target.write(mix(here, sum / weights, alpha), uint2(at));
    }
    """
}
#endif
