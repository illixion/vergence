#if canImport(Metal)
import Metal

struct VideoStereoUniforms {
    var brightness: Float
    var contrast: Float
    var saturation: Float
    var depthStrength: Float
    var convergence: Float
    var eyeSign: Float
    var mirror: Float
    var useDepth: Float
    var depthUVScale: SIMD2<Float>
    var depthUVOffset: SIMD2<Float>
    var depthValueScale: Float
    var depthValueBias: Float
    var depthBudget: SIMD2<Float>
    var edgeFill: Float = 0
    var extendedRange: Float = 0
    /// Share of the width, at each side, over which depth fades to nothing
    /// (`pseudo3DEdgeFeather`); 0 = no fade (the pass is drawn enlarged instead).
    var edgeFeather: Float = StereoEdge.featherWidth
    var reserved: Float = 0
}

/// How the picture's sides are kept clean: each eye's image is shifted inward
/// at one side by as much as the farthest content moves, leaving a strip with
/// nothing to show.
enum StereoEdge {
    /// The fade's width, as the shaders always had it.
    static let featherWidth: Float = 0.05

    /// How much larger the mesh pass draws so that strip falls outside the
    /// eye: the largest shift is `strength × max(convergence, 1 − convergence)`
    /// of the width (the depth budget keeps depth in 0…1), times how far
    /// beyond an eye's own position the viewer's offset can take it.
    static func zoom(strength: Float, convergence: Float, viewerReach: Float) -> Float {
        let shift = abs(strength) * max(convergence, 1 - convergence) * (1 + max(0, viewerReach))
        let crop = min(max(shift, 0), 0.08)
        return 1 / (1 - 2 * crop)
    }
}

/// The occlusion-tested mesh pass shared by video and stills. Caller owns its
/// serial render queue and command buffer; no clock, decoding or presentation.
final class StereoMeshWarp {
    private let resources: StereoWarpResources
    private var depths: [MTLTexture?] = [nil, nil]
    private var colors: [MTLTexture?] = [nil, nil]

    init(resources: StereoWarpResources) { self.resources = resources }

    func reset() {
        depths = [nil, nil]
        colors = [nil, nil]
    }

    /// `overlay` draws more into the same pass, after the mesh, so a second renderer can blend
    /// over its multisampled result without a round trip through memory.
    func encode(source: MTLTexture, depth: MTLTexture, destination: MTLTexture,
                uniforms: VideoStereoUniforms, dense: Bool, eye: Int,
                commandBuffer: MTLCommandBuffer, zoom: Float = 1,
                overlay: ((MTLRenderCommandEncoder) -> Void)? = nil) -> Bool {
        let samples = resources.msaaSampleCount
        func attachment(_ format: MTLPixelFormat, existing: MTLTexture?) -> MTLTexture? {
            if let existing, existing.width == destination.width, existing.height == destination.height,
               existing.pixelFormat == format { return existing }
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format,
                width: destination.width, height: destination.height, mipmapped: false)
            d.usage = .renderTarget
            d.storageMode = .private
            if samples > 1 {
                d.textureType = .type2DMultisample
                d.sampleCount = samples
                #if os(iOS) || os(tvOS) || os(visionOS)
                d.storageMode = .memoryless
                #endif
            }
            return resources.device.makeTexture(descriptor: d)
        }
        depths[eye] = attachment(.depth32Float, existing: depths[eye])
        guard let z = depths[eye] else { return false }
        let pass = MTLRenderPassDescriptor()
        if samples > 1 {
            colors[eye] = attachment(destination.pixelFormat, existing: colors[eye])
            guard let color = colors[eye] else { return false }
            pass.colorAttachments[0].texture = color
            pass.colorAttachments[0].resolveTexture = destination
            pass.colorAttachments[0].storeAction = .multisampleResolve
        } else {
            pass.colorAttachments[0].texture = destination
            pass.colorAttachments[0].storeAction = .store
        }
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1)
        pass.depthAttachment.texture = z
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.clearDepth = 1
        pass.depthAttachment.storeAction = .dontCare
        guard let enc = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else { return false }
        var uniforms = uniforms
        if zoom > 1 {
            // The whole pass (the mesh and whatever `overlay` draws over it)
            // enlarged about the centre; what falls outside the eye is cropped.
            let w = Double(destination.width) * Double(zoom), h = Double(destination.height) * Double(zoom)
            enc.setViewport(MTLViewport(originX: (Double(destination.width) - w) / 2,
                                        originY: (Double(destination.height) - h) / 2,
                                        width: w, height: h, znear: 0, zfar: 1))
        }
        enc.setRenderPipelineState(resources.meshPipelineState)
        enc.setDepthStencilState(resources.meshDepthState)
        enc.setVertexBuffer(dense ? resources.denseGridPositions : resources.gridPositions, offset: 0, index: 0)
        enc.setVertexBytes(&uniforms, length: MemoryLayout<VideoStereoUniforms>.stride, index: 1)
        enc.setVertexTexture(depth, index: 0)
        enc.setFragmentTexture(source, index: 0)
        enc.setFragmentTexture(depth, index: 1)
        enc.setFragmentBytes(&uniforms, length: MemoryLayout<VideoStereoUniforms>.stride, index: 0)
        enc.drawIndexedPrimitives(type: .triangle,
            indexCount: dense ? resources.denseGridIndexCount : resources.gridIndexCount,
            indexType: .uint32,
            indexBuffer: dense ? resources.denseGridIndices : resources.gridIndices, indexBufferOffset: 0)
        overlay?(enc)
        enc.endEncoding()
        return true
    }
}
#endif
