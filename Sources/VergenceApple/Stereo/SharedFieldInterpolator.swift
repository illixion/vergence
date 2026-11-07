/*
 Vergence - Smooth Motion for true stereo on the player: both in-between eyes
 from one motion field (and a 2D picture's in-between frame from its own)

 Makes the frame halfway between two packed stereo frames (a 3D Blu-ray's
 top/bottom pair) from fields computed ahead, off the headset
 (`SharedFields`: the left eye's RIFE flow and mask, each real frame's
 right→left disparity). No network runs here; one compute pass warps and
 blends. Hypnos `docs/plans/blu-ray-3d-smooth-motion.md`, option E:

 - Left eye: RIFE's own result at ½. With F½→0 = F1→0 / 2 and F½→1 =
   F0→1 / 2 (what the stored pair means), frame 0 is read at x + F½→0, frame
   1 at x + F½→1, blended by the mask.
 - Right eye: the left eye's field read across. A right pixel x sits at
   x + D½(x) in the left eye (D½ the in-place mix of the two real pairs'
   disparities) and moves as the left eye does there:
   F_R(x) = F_L(x + D½(x)), the mask likewise. The disparity change over the
   half frame is left out on purpose (a second disparity read at the moved
   position cost 7 dB on silhouettes, E2).

 Matches the PyTorch reference (`scripts/stereo-shared-field.py`'s
 `transfer` and `warp_blend`): fields upsampled bilinearly with pixel
 centres at +½ (`F.interpolate(align_corners=False)`), frames read
 bilinearly and clamped to each eye's edge (RIFE's warp, border padding).
 Blending is in the frames' own encoding, as RIFE's was.

 A 2D picture (`Packing.mono`, the `smooth-motion-2d` workload: RIFE at
 960×576 on a worker) is the left eye alone, the whole frame. Its fields come
 and go with the coordinator's pieces, so a mono interpolator can hold the
 player's own on-device one as a `fallback` and use it for any pair without
 fields: the picture keeps its in-between frames while pieces arrive.

 The kernel is compiled from source at first use rather than taken from the
 package's metallib: the SwiftPM command line never builds `.metal` files,
 and the `vergence interpolate` check runs this same code on a Mac.
 */

#if canImport(Metal) && canImport(VideoToolbox)
import CoreVideo
import Foundation
import Metal
import QuartzCore
import VergenceCore

public final class SharedFieldInterpolator: MidFrameInterpolator, @unchecked Sendable {
    /// How the two eyes share the frame.
    public enum Packing: Sendable {
        /// Left eye on top (a 3D Blu-ray as `MVCPictureSource` packs it).
        case topAndBottom
        case sideBySide
        /// A 2D picture: the whole frame, flow only.
        case mono
    }

    public let provider: SharedFieldProvider
    public let packing: Packing
    /// Makes the in-between frame of a pair without fields (a mono picture's
    /// on-device interpolator), or nil to pass such pairs through.
    public let fallback: MidFrameInterpolator?

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    private var textureCache: CVMetalTextureCache?
    private var outputPools: [OSType: CVPixelBufferPool] = [:]
    private var fallbackReady = false
    /// Which made the last in-between frame, for a line when that changes.
    private var lastSource: String?
    private var width = 0
    private var height = 0
    /// Field textures, remade when a record's size changes. One request at a
    /// time (the owner's contract), and each is filled only after the last
    /// pass that read it has completed.
    private var flowTexture: MTLTexture?
    private var maskTexture: MTLTexture?
    private var disparityTextures: [MTLTexture] = []

    /// nil without Metal or if the kernel doesn't compile.
    /// - Parameter fallback: for `.mono`, the interpolator a pair without fields goes to.
    public init?(provider: SharedFieldProvider, packing: Packing = .topAndBottom,
                 fallback: MidFrameInterpolator? = nil) {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { return nil }
        do {
            let library = try device.makeLibrary(source: Self.source, options: nil)
            guard let function = library.makeFunction(name: "sharedFieldMidFrame") else { return nil }
            pipeline = try device.makeComputePipelineState(function: function)
        } catch {
            VergenceLog.pipeline.error("shared-field interpolation: kernel failed: \(error, privacy: .public)")
            return nil
        }
        self.device = device
        self.queue = queue
        self.provider = provider
        self.packing = packing
        self.fallback = fallback
    }

    /// The governor's `scale` is ignored here (a warp pass costs the same at
    /// any field size and the real frames are full size); the fallback gets it.
    public func prepare(width: Int, height: Int, scale: Double) -> Bool {
        switch packing {
        case .topAndBottom: guard height % 2 == 0 else { return false }
        case .sideBySide: guard width % 2 == 0 else { return false }
        case .mono: break
        }
        var cache: CVMetalTextureCache?
        CVMetalTextureCacheCreate(nil, nil, device, nil, &cache)
        guard let cache else { return false }
        textureCache = cache
        outputPools = [:]
        self.width = width
        self.height = height
        fallbackReady = fallback?.prepare(width: width, height: height, scale: scale) ?? false
        return true
    }

    /// An output pool for frames of `format` (8-bit BGRA, or half-float RGBA for HDR).
    private func outputPool(_ format: OSType) -> CVPixelBufferPool? {
        if let pool = outputPools[format] { return pool }
        var pool: CVPixelBufferPool?
        let attributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: format,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
        ]
        CVPixelBufferPoolCreate(nil, nil, attributes as CFDictionary, &pool)
        outputPools[format] = pool
        return pool
    }

    public func midFrame(between previous: CVPixelBuffer, and current: CVPixelBuffer,
                         completion: @escaping @Sendable (CVPixelBuffer?, Double) -> Void) {
        // Without times there is nothing to look fields up by.
        guard let fallback, fallbackReady else { completion(nil, .nan); return }
        fallback.midFrame(between: previous, and: current, completion: completion)
    }

    public func midFrame(between previous: CVPixelBuffer, at previousTime: Double, and current: CVPixelBuffer,
                         at currentTime: Double,
                         completion: @escaping @Sendable (CVPixelBuffer?, Double) -> Void) {
        let started = CACurrentMediaTime()
        guard let fields = provider.fields(from: previousTime, to: currentTime),
              packing != .mono || !fields.hasDisparity else {
            noteSource(fallback != nil && fallbackReady ? "on-device" : "none")
            guard let fallback, fallbackReady else { completion(nil, .nan); return }
            fallback.midFrame(between: previous, at: previousTime, and: current, at: currentTime, completion: completion)
            return
        }
        noteSource("fields")
        guard packing == .mono || fields.hasDisparity else { completion(nil, .nan); return }
        if encode(previous: previous, current: current, fields: fields, completion: { out in
            completion(out, (CACurrentMediaTime() - started) * 1000)
        }) == nil {
            completion(nil, (CACurrentMediaTime() - started) * 1000)
        }
    }

    /// Synchronous, for the CLI check and tests: the in-between frame or nil.
    public func midFrame(between previous: CVPixelBuffer, and current: CVPixelBuffer,
                         fields: SharedFields) -> CVPixelBuffer? {
        let done = DispatchSemaphore(value: 0)
        let result = ResultBox()
        guard encode(previous: previous, current: current, fields: fields, completion: { out in
            result.buffer = out
            done.signal()
        }) != nil else { return nil }
        done.wait()
        return result.buffer
    }

    private final class ResultBox: @unchecked Sendable { var buffer: CVPixelBuffer? }

    /// A line when the in-between frames start coming from somewhere else.
    private func noteSource(_ source: String) {
        guard source != lastSource, packing == .mono else { return }
        lastSource = source
        VergenceLog.pipeline.notice("Smooth Motion 2D: in-between frames from \(source, privacy: .public)")
    }

    // MARK: The pass

    struct Params {
        var eyeSize: SIMD2<Float>
        var eyeOffset: SIMD2<Float>
        /// Stored flow pixels → eye pixels, x and y.
        var flowScale: SIMD2<Float>
        /// Stored disparity pixels → eye pixels (x only).
        var disparityScale: Float
        var t: Float
    }

    /// Encodes and commits the pass; `completion` gets the frame (or nil if
    /// the GPU failed). Returns nil, without calling `completion`, when it
    /// couldn't start.
    private func encode(previous: CVPixelBuffer, current: CVPixelBuffer, fields: SharedFields,
                        completion: @escaping @Sendable (CVPixelBuffer?) -> Void) -> CVPixelBuffer? {
        guard width > 0,
              CVPixelBufferGetWidth(previous) == width, CVPixelBufferGetHeight(previous) == height,
              CVPixelBufferGetWidth(current) == width, CVPixelBufferGetHeight(current) == height,
              CVPixelBufferGetPixelFormatType(previous) == CVPixelBufferGetPixelFormatType(current),
              Self.textureFormat(CVPixelBufferGetPixelFormatType(current)) != nil,
              let pool = outputPool(CVPixelBufferGetPixelFormatType(current)) else { return nil }
        var output: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &output)
        guard let output,
              let a = texture(previous), let b = texture(current), let out = texture(output),
              uploadFields(fields),
              let flowTexture, let maskTexture, disparityTextures.count == 2,
              let commands = queue.makeCommandBuffer(),
              let encoder = commands.makeComputeCommandEncoder() else { return nil }
        CVBufferPropagateAttachments(current, output)

        let eye: SIMD2<Float> = switch packing {
        case .topAndBottom: SIMD2(Float(width), Float(height / 2))
        case .sideBySide: SIMD2(Float(width / 2), Float(height))
        case .mono: SIMD2(Float(width), Float(height))
        }
        var params = Params(
            eyeSize: eye,
            // A mono picture's "right eye" starts past the frame, so no pixel is one.
            eyeOffset: packing == .topAndBottom ? SIMD2(0, eye.y) : SIMD2(eye.x, 0),
            flowScale: SIMD2(eye.x / Float(fields.flowWidth), eye.y / Float(fields.flowHeight)),
            disparityScale: fields.hasDisparity ? eye.x / Float(fields.disparityWidth) : 0,
            t: 0.5)
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(a.texture, index: 0)
        encoder.setTexture(b.texture, index: 1)
        encoder.setTexture(flowTexture, index: 2)
        encoder.setTexture(maskTexture, index: 3)
        encoder.setTexture(disparityTextures[0], index: 4)
        encoder.setTexture(disparityTextures[1], index: 5)
        encoder.setTexture(out.texture, index: 6)
        encoder.setBytes(&params, length: MemoryLayout<Params>.stride, index: 0)
        let group = MTLSize(width: 16, height: 16, depth: 1)
        encoder.dispatchThreadgroups(
            MTLSize(width: (width + 15) / 16, height: (height + 15) / 16, depth: 1), threadsPerThreadgroup: group)
        encoder.endEncoding()
        // The CVMetalTextures keep their IOSurfaces bound until the pass is
        // done, and must go back before their cache does: the owner may be
        // gone by the time the pass completes.
        let held = HeldTextures(textures: [a.held, b.held, out.held], cache: textureCache)
        let result = OutputBox(buffer: output)
        commands.addCompletedHandler { buffer in
            held.release()
            completion(buffer.status == .completed ? result.buffer : nil)
        }
        commands.commit()
        return output
    }

    private final class HeldTextures: @unchecked Sendable {
        private var textures: [CVMetalTexture]
        private let cache: CVMetalTextureCache?
        init(textures: [CVMetalTexture], cache: CVMetalTextureCache?) {
            self.textures = textures
            self.cache = cache
        }
        func release() { textures.removeAll() }
    }
    private struct OutputBox: @unchecked Sendable { let buffer: CVPixelBuffer }

    static func textureFormat(_ pixelFormat: OSType) -> MTLPixelFormat? {
        switch pixelFormat {
        case kCVPixelFormatType_32BGRA: .bgra8Unorm
        case kCVPixelFormatType_64RGBAHalf: .rgba16Float
        default: nil
        }
    }

    private func texture(_ buffer: CVPixelBuffer) -> (texture: MTLTexture, held: CVMetalTexture)? {
        guard let textureCache, let format = Self.textureFormat(CVPixelBufferGetPixelFormatType(buffer)) else { return nil }
        var cvTexture: CVMetalTexture?
        guard CVMetalTextureCacheCreateTextureFromImage(
            nil, textureCache, buffer, nil, format,
            CVPixelBufferGetWidth(buffer), CVPixelBufferGetHeight(buffer), 0, &cvTexture) == kCVReturnSuccess,
            let cvTexture, let texture = CVMetalTextureGetTexture(cvTexture) else { return nil }
        return (texture, cvTexture)
    }

    /// Half floats: filterable everywhere, and finer than the stored step
    /// for any motion under 64 stored pixels (1,000 at 1080p).
    private func uploadFields(_ fields: SharedFields) -> Bool {
        let fw = fields.flowWidth, fh = fields.flowHeight, n = fw * fh
        // A 2D picture's fields have no disparity: the kernel still binds two (unread) textures.
        let dw = max(fields.disparityWidth, 1), dh = max(fields.disparityHeight, 1)
        func make(_ format: MTLPixelFormat, _ w: Int, _ h: Int, _ existing: MTLTexture?) -> MTLTexture? {
            if let existing, existing.width == w, existing.height == h { return existing }
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: w, height: h,
                                                                      mipmapped: false)
            descriptor.usage = .shaderRead
            descriptor.storageMode = .shared
            return device.makeTexture(descriptor: descriptor)
        }
        flowTexture = make(.rgba16Float, fw, fh, flowTexture)
        maskTexture = make(.r16Float, fw, fh, maskTexture)
        disparityTextures = (0 ..< 2).map { i in
            make(.r16Float, dw, dh, disparityTextures.indices.contains(i) ? disparityTextures[i] : nil)
        }.compactMap { $0 }
        guard let flowTexture, let maskTexture, disparityTextures.count == 2 else { return false }

        var rgba = [UInt16](repeating: 0, count: 4 * n)
        var mask = [UInt16](repeating: 0, count: n)
        fields.flow.withUnsafeBufferPointer { f in
            for i in 0 ..< n {
                for c in 0 ..< 4 { rgba[4 * i + c] = HalfFloat.bits(from: f[c * n + i]) }
                mask[i] = HalfFloat.bits(from: f[4 * n + i])
            }
        }
        flowTexture.replace(region: MTLRegionMake2D(0, 0, fw, fh), mipmapLevel: 0, withBytes: rgba,
                            bytesPerRow: 8 * fw)
        maskTexture.replace(region: MTLRegionMake2D(0, 0, fw, fh), mipmapLevel: 0, withBytes: mask,
                            bytesPerRow: 2 * fw)
        for (texture, values) in zip(disparityTextures, [fields.disparity0, fields.disparity1]) where fields.hasDisparity {
            let bits = values.map { HalfFloat.bits(from: $0) }
            texture.replace(region: MTLRegionMake2D(0, 0, dw, dh), mipmapLevel: 0, withBytes: bits,
                            bytesPerRow: 2 * dw)
        }
        return true
    }

    static let source = """
    #include <metal_stdlib>
    using namespace metal;

    struct Params {
        float2 eyeSize;
        float2 eyeOffset;
        float2 flowScale;
        float disparityScale;
        float t;
    };

    kernel void sharedFieldMidFrame(texture2d<float> previous [[texture(0)]],
                                    texture2d<float> current [[texture(1)]],
                                    texture2d<float> flow [[texture(2)]],
                                    texture2d<float> mask [[texture(3)]],
                                    texture2d<float> disparity0 [[texture(4)]],
                                    texture2d<float> disparity1 [[texture(5)]],
                                    texture2d<float, access::write> out [[texture(6)]],
                                    constant Params &p [[buffer(0)]],
                                    uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= out.get_width() || gid.y >= out.get_height()) return;
        // Fields: normalised, edge-clamped, bilinear = F.interpolate(align_corners=False).
        constexpr sampler field(coord::normalized, address::clamp_to_edge, filter::linear);
        constexpr sampler pixels(coord::pixel, address::clamp_to_edge, filter::linear);

        float2 position = float2(gid);
        bool right = p.eyeOffset.y > 0 ? position.y >= p.eyeOffset.y : position.x >= p.eyeOffset.x;
        float2 origin = right ? p.eyeOffset : float2(0);
        float2 x = position - origin;            // pixel index within the eye

        // Where this pixel's motion is read in the left eye's field.
        float2 at = x;
        if (right) {
            float2 uv = (x + 0.5) / p.eyeSize;
            float d = mix(disparity0.sample(field, uv).r, disparity1.sample(field, uv).r, p.t) * p.disparityScale;
            at.x = clamp(x.x + d, 0.0f, p.eyeSize.x - 1.0f);
        }
        float2 uv = (at + 0.5) / p.eyeSize;
        float4 f = flow.sample(field, uv);
        float w0 = saturate(mask.sample(field, uv).r);
        float2 f01 = f.xy * p.flowScale;
        float2 f10 = f.zw * p.flowScale;
        // RIFE ran at ½: F½→0 = F1→0 / 2, F½→1 = F0→1 / 2.
        float2 ft0 = f10 * 0.5;
        float2 ft1 = f01 * 0.5;

        // Read each frame at the moved position, clamped to this eye's pixel centres.
        float2 lo = origin + 0.5, hi = origin + p.eyeSize - 0.5;
        float4 a = previous.sample(pixels, clamp(origin + x + ft0 + 0.5, lo, hi));
        float4 b = current.sample(pixels, clamp(origin + x + ft1 + 0.5, lo, hi));
        out.write(a * w0 + b * (1.0f - w0), gid);
    }
    """
}
#endif
