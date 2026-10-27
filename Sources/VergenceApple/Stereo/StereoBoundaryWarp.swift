#if canImport(Metal)
import Metal
import VergenceCore

/// A/B prototype: disconnected depth surfaces, followed by an explicit,
/// background-only fill. Owned and called on the pump queue, never the main
/// actor. No history: a seek or cut cannot reuse another frame's background.
final class StereoBoundaryWarp {
    struct Uniforms {
        var stereo: SIMD4<Float> // strength, convergence, eye sign, mirror
        var uv: SIMD4<Float> // depth scale.xy, offset.xy
        var mapping: SIMD4<Float> // value scale, bias, budget knees
        var adjustments: SIMD4<Float> // brightness, contrast, saturation, index count
        var boundary: SIMD4<Float> // depth jump, output width, grid pitch, show holes
    }

    private let device: MTLDevice
    private let renderPipeline: MTLRenderPipelineState
    private let fillPipeline: MTLComputePipelineState
    private let depthState: MTLDepthStencilState
    private(set) var surfaces: MTLTexture?
    private var zBuffer: MTLTexture?
    private var coverage: MTLTexture?
    private var multisampleColor: MTLTexture?
    private var multisampleCoverage: MTLTexture?
    private let samples: Int

    init?(device: MTLDevice, library: MTLLibrary) {
        self.device = device
        samples = device.supportsTextureSampleCount(4) ? 4 : 1
        guard let vertex = library.makeFunction(name: "boundaryMeshVertex"),
              let fragment = library.makeFunction(name: "boundaryMeshFragment"),
              let fill = library.makeFunction(name: "boundaryHoleFill") else { return nil }
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = vertex
        descriptor.fragmentFunction = fragment
        // RGB + (1 + normalized depth). A zero alpha is an uncovered pixel.
        // Resolve coverage independently: packed depth / coverage recovers
        // the surface depth without mistaking a partially covered edge for a hole.
        descriptor.colorAttachments[0].pixelFormat = .rgba16Float
        descriptor.colorAttachments[1].pixelFormat = .r16Float
        descriptor.rasterSampleCount = samples
        descriptor.depthAttachmentPixelFormat = .depth32Float
        let z = MTLDepthStencilDescriptor()
        z.depthCompareFunction = .lessEqual
        z.isDepthWriteEnabled = true
        guard let render = try? device.makeRenderPipelineState(descriptor: descriptor),
              let compute = try? device.makeComputePipelineState(function: fill),
              let state = device.makeDepthStencilState(descriptor: z) else { return nil }
        renderPipeline = render
        fillPipeline = compute
        depthState = state
    }

    /// Encodes both passes in the caller's command buffer. Scratch is reused
    /// between eyes only after the preceding eye's fill has read it.
    func encode(source: MTLTexture, depth: MTLTexture, destination: MTLTexture,
                positions: MTLBuffer, indices: MTLBuffer, indexCount: Int,
                uniforms: Uniforms, commandBuffer: MTLCommandBuffer) -> Bool {
        if surfaces?.width != destination.width || surfaces?.height != destination.height {
            let color = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float,
                width: destination.width, height: destination.height, mipmapped: false)
            color.storageMode = .private
            color.usage = [.renderTarget, .shaderRead]
            let z = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .depth32Float,
                width: destination.width, height: destination.height, mipmapped: false)
            z.storageMode = .private
            if samples > 1 {
                z.textureType = .type2DMultisample
                z.sampleCount = samples
                #if os(iOS) || os(tvOS) || os(visionOS)
                z.storageMode = .memoryless
                #endif
            }
            z.usage = [.renderTarget]
            surfaces = device.makeTexture(descriptor: color)
            zBuffer = device.makeTexture(descriptor: z)
            let c = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r16Float,
                width: destination.width, height: destination.height, mipmapped: false)
            c.storageMode = .private
            c.usage = [.renderTarget, .shaderRead]
            coverage = device.makeTexture(descriptor: c)
            if samples > 1 {
                z.pixelFormat = .rgba16Float
                multisampleColor = device.makeTexture(descriptor: z)
                z.pixelFormat = .r16Float
                multisampleCoverage = device.makeTexture(descriptor: z)
            }
        }
        guard let surfaces, let zBuffer, let coverage else { return false }
        var uniforms = uniforms
        uniforms.adjustments.w = Float(indexCount)
        let pass = MTLRenderPassDescriptor()
        if samples > 1 {
            guard let multisampleColor, let multisampleCoverage else { return false }
            pass.colorAttachments[0].texture = multisampleColor
            pass.colorAttachments[0].resolveTexture = surfaces
            pass.colorAttachments[0].storeAction = .multisampleResolve
            pass.colorAttachments[1].texture = multisampleCoverage
            pass.colorAttachments[1].resolveTexture = coverage
            pass.colorAttachments[1].storeAction = .multisampleResolve
        } else {
            pass.colorAttachments[0].texture = surfaces
            pass.colorAttachments[0].storeAction = .store
            pass.colorAttachments[1].texture = coverage
            pass.colorAttachments[1].storeAction = .store
        }
        pass.colorAttachments[1].loadAction = .clear
        pass.colorAttachments[1].clearColor = MTLClearColorMake(0, 0, 0, 0)
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0)
        pass.depthAttachment.texture = zBuffer
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.storeAction = .dontCare
        pass.depthAttachment.clearDepth = 1
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else { return false }
        encoder.setRenderPipelineState(renderPipeline)
        encoder.setDepthStencilState(depthState)
        encoder.setVertexBuffer(positions, offset: 0, index: 0)
        encoder.setVertexBuffer(indices, offset: 0, index: 1)
        encoder.setVertexBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 2)
        encoder.setVertexTexture(depth, index: 0)
        encoder.setFragmentTexture(source, index: 0)
        encoder.setFragmentTexture(depth, index: 1)
        encoder.setFragmentBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 0)
        // Non-indexed draw gives all three vertices the same triangle's
        // split decision. An indexed vertex_id cannot identify its cell.
        // Clip crossing triangles into near/far source portions rather than
        // deleting the foreground part together with the stretched bridge.
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: indexCount * 2)
        encoder.endEncoding()
        guard let fill = commandBuffer.makeComputeCommandEncoder() else { return false }
        fill.setComputePipelineState(fillPipeline)
        fill.setTexture(surfaces, index: 0)
        fill.setTexture(destination, index: 1)
        fill.setTexture(coverage, index: 2)
        fill.setBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 0)
        fill.dispatchThreads(MTLSize(width: destination.width, height: destination.height, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 16, height: 8, depth: 1))
        fill.endEncoding()
        return true
    }
}
#endif
