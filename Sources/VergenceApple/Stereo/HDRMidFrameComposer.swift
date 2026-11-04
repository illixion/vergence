#if canImport(Metal) && canImport(CoreVideo)
import CoreVideo
import Foundation
import Metal
import VergenceCore

/// Makes a half-float (HDR) in-between frame from an interpolator that only
/// works in 8 bits, without the 8 bits showing.
///
/// VideoToolbox's low-latency interpolator takes `420v` alone. Interpolating
/// an HDR pair there and converting back would make every other frame an
/// 8-bit PQ picture: a sky would band on the in-between frames and not on the
/// real ones, flickering at 48 Hz. So only what the interpolator *changed* is
/// taken from it:
///
///     mid = (previous₁₆ + current₁₆) / 2 + M · (mid₈ − (previous₈ + current₈) / 2)
///
/// where ₈ are the 8-bit Y'CbCr frames the interpolator saw (its input and its
/// output) and M is their matrix to R'G'B'. Where nothing moves (the
/// interpolator returns the blend) the frame is the full-precision blend, so
/// flat areas and gradients keep every bit; where something moves, the 8-bit
/// motion is added to it, and motion hides the quantisation. A change under
/// about 3 levels is faded out: it is the interpolator's own noise (it moves
/// a still 8-bit picture by up to 1.7 levels). Each term is in
/// code values (PQ or HLG, as the frames are tagged), so no transfer function
/// is applied or inverted. The 8-bit frames may be smaller (the governor's
/// reduced size): they are sampled, the real frames read.
final class HDRMidFrameComposer: @unchecked Sendable {
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    private var textureCache: CVMetalTextureCache?
    private var pool: CVPixelBufferPool?
    private var poolSize = (width: 0, height: 0)

    init?(device: MTLDevice? = MTLCreateSystemDefaultDevice()) {
        guard let device, let queue = device.makeCommandQueue() else { return nil }
        do {
            let library = try device.makeLibrary(source: Self.source, options: nil)
            guard let function = library.makeFunction(name: "hdrMidCompose") else { return nil }
            pipeline = try device.makeComputePipelineState(function: function)
        } catch {
            VergenceLog.pipeline.error("HDR in-between frames: kernel failed: \(error, privacy: .public)")
            return nil
        }
        self.device = device
        self.queue = queue
        CVMetalTextureCacheCreate(nil, nil, device, nil, &textureCache)
    }

    /// The R'G'B' change for a Y'CbCr change, by the frames' matrix:
    /// (R per Cr, G per Cb, G per Cr, B per Cb).
    static func coefficients(matrix: CFTypeRef?) -> SIMD4<Float> {
        let name = matrix as? String
        if name == (kCVImageBufferYCbCrMatrix_ITU_R_2020 as String) {
            return SIMD4(1.4746, 0.16455, 0.57135, 1.8814)
        }
        if name == (kCVImageBufferYCbCrMatrix_ITU_R_601_4 as String)
            || name == (kCVImageBufferYCbCrMatrix_SMPTE_240M_1995 as String) {
            return SIMD4(1.402, 0.344136, 0.714136, 1.772)
        }
        return SIMD4(1.5748, 0.1873, 0.4681, 1.8556) // BT.709
    }

    /// `previous` and `current` are half-float RGBA; the three Y'CbCr frames
    /// are the interpolator's `420v` input and output. `completion` runs on a
    /// Metal completion thread with the composed frame (tagged as `current`),
    /// or nil.
    func compose(previous: CVPixelBuffer, current: CVPixelBuffer,
                 previousYUV: CVPixelBuffer, currentYUV: CVPixelBuffer, midYUV: CVPixelBuffer,
                 completion: @escaping @Sendable (CVPixelBuffer?) -> Void) {
        let width = CVPixelBufferGetWidth(current)
        let height = CVPixelBufferGetHeight(current)
        if pool == nil || poolSize != (width, height) {
            pool = FramePixelFormat.pool(format: FramePixelFormat.hdr, width: width, height: height)
            poolSize = (width, height)
        }
        var output: CVPixelBuffer?
        if let pool { CVPixelBufferPoolCreatePixelBuffer(nil, pool, &output) }
        guard let output, let textureCache,
              let a = FramePixelFormat.texture(previous, cache: textureCache),
              let b = FramePixelFormat.texture(current, cache: textureCache),
              let out = FramePixelFormat.texture(output, cache: textureCache),
              let pY = plane(previousYUV, 0, .r8Unorm), let pC = plane(previousYUV, 1, .rg8Unorm),
              let cY = plane(currentYUV, 0, .r8Unorm), let cC = plane(currentYUV, 1, .rg8Unorm),
              let mY = plane(midYUV, 0, .r8Unorm), let mC = plane(midYUV, 1, .rg8Unorm),
              let commands = queue.makeCommandBuffer(),
              let encoder = commands.makeComputeCommandEncoder() else {
            completion(nil)
            return
        }
        CVBufferPropagateAttachments(current, output)
        var coefficients = Self.coefficients(
            matrix: CVBufferCopyAttachment(midYUV, kCVImageBufferYCbCrMatrixKey, nil))
        encoder.setComputePipelineState(pipeline)
        for (index, texture) in [a.texture, b.texture, pY.texture, pC.texture, cY.texture, cC.texture,
                                 mY.texture, mC.texture, out.texture].enumerated() {
            encoder.setTexture(texture, index: index)
        }
        encoder.setBytes(&coefficients, length: MemoryLayout<SIMD4<Float>>.stride, index: 0)
        encoder.dispatchThreadgroups(
            MTLSize(width: (width + 15) / 16, height: (height + 15) / 16, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        encoder.endEncoding()
        let held = Held([a.held, b.held, out.held, pY.held, pC.held, cY.held, cC.held, mY.held, mC.held])
        let result = Box(buffer: output)
        commands.addCompletedHandler { buffer in
            held.release()
            completion(buffer.status == .completed ? result.buffer : nil)
        }
        commands.commit()
    }

    private func plane(_ buffer: CVPixelBuffer, _ index: Int, _ format: MTLPixelFormat)
        -> (texture: MTLTexture, held: CVMetalTexture)? {
        guard let textureCache else { return nil }
        var cvTexture: CVMetalTexture?
        guard CVMetalTextureCacheCreateTextureFromImage(
            nil, textureCache, buffer, nil, format,
            CVPixelBufferGetWidthOfPlane(buffer, index), CVPixelBufferGetHeightOfPlane(buffer, index), index,
            &cvTexture) == kCVReturnSuccess,
            let cvTexture, let texture = CVMetalTextureGetTexture(cvTexture) else { return nil }
        return (texture, cvTexture)
    }

    private final class Held: @unchecked Sendable {
        private var textures: [CVMetalTexture]
        init(_ textures: [CVMetalTexture]) { self.textures = textures }
        func release() { textures.removeAll() }
    }
    private struct Box: @unchecked Sendable { let buffer: CVPixelBuffer }

    static let source = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void hdrMidCompose(texture2d<float> previous [[texture(0)]],
                              texture2d<float> current [[texture(1)]],
                              texture2d<float> previousY [[texture(2)]],
                              texture2d<float> previousC [[texture(3)]],
                              texture2d<float> currentY [[texture(4)]],
                              texture2d<float> currentC [[texture(5)]],
                              texture2d<float> midY [[texture(6)]],
                              texture2d<float> midC [[texture(7)]],
                              texture2d<float, access::write> output [[texture(8)]],
                              constant float4 &k [[buffer(0)]],
                              uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= output.get_width() || gid.y >= output.get_height()) return;
        constexpr sampler s(filter::linear, address::clamp_to_edge, coord::normalized);
        float2 uv = (float2(gid) + 0.5) / float2(output.get_width(), output.get_height());
        // Video range: luma 16-235, chroma 16-240 of 255.
        float dy = (midY.sample(s, uv).r - 0.5 * (previousY.sample(s, uv).r + currentY.sample(s, uv).r))
                   * (255.0 / 219.0);
        float2 dc = (midC.sample(s, uv).rg - 0.5 * (previousC.sample(s, uv).rg + currentC.sample(s, uv).rg))
                    * (255.0 / 224.0);
        float3 d = float3(dy + k.x * dc.y, dy - k.y * dc.x - k.z * dc.y, dy + k.w * dc.x);
        // VideoToolbox moves a still 8-bit picture by up to ~1.7 levels on its
        // own (measured on an M2 Max); a change that small is its noise, not
        // motion, and would put 8-bit grain back. Fade it out below 3 levels.
        float size = max(abs(d.r), max(abs(d.g), abs(d.b)));
        d *= smoothstep(1.0 / 255.0, 3.0 / 255.0, size);
        float4 a = previous.read(gid);
        float4 b = current.read(gid);
        output.write(float4(clamp(0.5 * (a.rgb + b.rgb) + d, 0.0, 1.0), 1.0), gid);
    }
    """
}
#endif
