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

    func encode(source: MTLTexture, depth: MTLTexture, destination: MTLTexture,
                uniforms: VideoStereoUniforms, dense: Bool, eye: Int,
                commandBuffer: MTLCommandBuffer) -> Bool {
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
        enc.endEncoding()
        return true
    }
}
#endif
