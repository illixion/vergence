#if canImport(Metal)
import Foundation
import Metal
import VergenceCore

/// Depth for an in-between frame, moved there from the real frame before it.
///
/// Real-time AI 3D infers one map per real frame; an in-between frame reuses
/// the map of the frame before (`HeldDepthPolicy`), so on anything moving the
/// depth edge sits where the subject was half a frame ago while the picture
/// shows it where it is now. The interpolator knows where every pixel of the
/// in-between frame came from (`MidFrameMotion`: `mid(x) ≈ previous(x − d)`),
/// so the map is read there instead: `depth_mid(x) = depth_previous(x − d)`.
/// One small pass over the map (a few hundred thousand texels), no inference.
///
/// It needs only the map before, which is ready by the time the in-between
/// frame is drawn; the map after would also fill what the motion uncovers,
/// but would make the in-between frame wait for the next inference.
final class MidFrameDepth: @unchecked Sendable {
    private let device: MTLDevice
    private let pipeline: MTLComputePipelineState
    private let lock = NSLock()
    private var free: [MTLTexture] = []
    /// Maps a half-second lookahead holds at 48 fps, and the one on screen.
    private static let maxFree = 16

    init?(device: MTLDevice) {
        guard let library = try? device.makeLibrary(source: Self.source, options: nil),
              let function = library.makeFunction(name: "depthAlongMotion"),
              let pipeline = try? device.makeComputePipelineState(function: function) else {
            VergenceLog.pipeline.error("mid-frame depth: kernel failed to compile")
            return nil
        }
        self.device = device
        self.pipeline = pipeline
    }

    struct Params {
        var uvScale: SIMD2<Float>
        var uvOffset: SIMD2<Float>
        var frameSize: SIMD2<Float>
    }

    /// `depth` moved along `motion`, encoded into `commandBuffer`; nil when
    /// it can't be (the map then stays as it is).
    func encode(_ depth: PumpFrameDepth, along motion: MidFrameMotion,
                into commandBuffer: MTLCommandBuffer) -> PumpFrameDepth? {
        let source = depth.texture
        guard let output = take(matching: source),
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }
        var params = Params(uvScale: depth.uvScale, uvOffset: depth.uvOffset,
                            frameSize: SIMD2(Float(motion.frameWidth), Float(motion.frameHeight)))
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(source, index: 0)
        encoder.setTexture(motion.texture, index: 1)
        encoder.setTexture(output, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<Params>.stride, index: 0)
        encoder.dispatchThreadgroups(
            MTLSize(width: (output.width + 15) / 16, height: (output.height + 15) / 16, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        encoder.endEncoding()
        let held = Returned(texture: output, owner: self)
        // The motion stays alive until the pass has read it.
        commandBuffer.addCompletedHandler { _ in withExtendedLifetime(motion) {} }
        return PumpFrameDepth(texture: output, uvScale: depth.uvScale, uvOffset: depth.uvOffset,
                              valueScale: depth.valueScale, valueBias: depth.valueBias, budget: depth.budget,
                              keepAlive: [held])
    }

    private func take(matching source: MTLTexture) -> MTLTexture? {
        let reused: MTLTexture? = lock.withLock {
            guard let index = free.firstIndex(where: {
                $0.width == source.width && $0.height == source.height && $0.pixelFormat == source.pixelFormat
            }) else { return nil }
            return free.remove(at: index)
        }
        if let reused { return reused }
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: source.pixelFormat, width: source.width, height: source.height, mipmapped: false)
        descriptor.usage = [.shaderRead, .shaderWrite]
        descriptor.storageMode = .private
        return device.makeTexture(descriptor: descriptor)
    }

    fileprivate func giveBack(_ texture: MTLTexture) {
        lock.withLock { if free.count < Self.maxFree { free.append(texture) } }
    }

    /// Hands the map's texture back when the frame's depth is let go.
    private final class Returned: @unchecked Sendable {
        let texture: MTLTexture
        weak var owner: MidFrameDepth?
        init(texture: MTLTexture, owner: MidFrameDepth) {
            self.texture = texture
            self.owner = owner
        }
        deinit { owner?.giveBack(texture) }
    }

    /// Bilinear by hand: a 32-bit float map isn't filterable on every GPU.
    static let source = """
    #include <metal_stdlib>
    using namespace metal;

    struct Params {
        float2 uvScale;    // frame UV -> the map's content region
        float2 uvOffset;
        float2 frameSize;  // the frame's pixels, which the motion is in
    };

    static float bilinear(texture2d<float> map, float2 uv) {
        float2 size = float2(map.get_width(), map.get_height());
        float2 at = clamp(uv * size - 0.5, float2(0), size - 1);
        uint2 a = uint2(floor(at));
        uint2 b = min(a + 1, uint2(size - 1));
        float2 f = at - float2(a);
        float top = mix(map.read(a).r, map.read(uint2(b.x, a.y)).r, f.x);
        float bottom = mix(map.read(uint2(a.x, b.y)).r, map.read(b).r, f.x);
        return mix(top, bottom, f.y);
    }

    kernel void depthAlongMotion(texture2d<float> depth [[texture(0)]],
                                 texture2d<float> motion [[texture(1)]],
                                 texture2d<float, access::write> out [[texture(2)]],
                                 constant Params &p [[buffer(0)]],
                                 uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= out.get_width() || gid.y >= out.get_height()) return;
        float2 depthUV = (float2(gid) + 0.5) / float2(out.get_width(), out.get_height());
        float2 frameUV = (depthUV - p.uvOffset) / p.uvScale;
        // Letterbox bars and anything outside the picture keep their value.
        if (any(frameUV < 0.0) || any(frameUV > 1.0)) {
            out.write(depth.read(gid), gid);
            return;
        }
        constexpr sampler linear(coord::normalized, address::clamp_to_edge, filter::linear);
        float2 d = motion.sample(linear, frameUV).xy;
        float2 from = clamp(frameUV - d / p.frameSize, 0.0, 1.0);
        out.write(float4(bilinear(depth, from * p.uvScale + p.uvOffset), 0, 0, 1), gid);
    }
    """
}
#endif
