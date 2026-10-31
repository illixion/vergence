#if canImport(Metal)
import Metal
import VergenceCore

/// Opt-in renderer for the silhouette of a near subject against a far background, drawn over the
/// connected mesh in the same pass (`DepthRefinement.layersSilhouettes`).
///
/// The mesh folds the picture back over itself at such an outline in one eye and stretches it
/// across the strip it uncovers in the other, because there one surface is asked to be two. This
/// splits the picture into a near layer and a far layer once per frame (`prepare`), and each eye
/// then ray-casts both across its own disparity (`encodePatch`). The result is blended over the
/// mesh with a trust weight that is zero wherever the split isn't well founded, so the mesh stays
/// the exact fallback there. The shaders' comment (`StereoShaders.metal`, "Layered silhouettes")
/// has the method; `docs/investigations/ai-3d-coat-stairstepping-2026-10-29.md` has the evidence.
///
/// Owned and called on the pump queue, never the main actor.
final class StereoLayeredWarp {
    /// What selects seeds and trusts the split. The defaults are the candidate values; the replay
    /// lab sweeps them. Pairs are the low and high end of a smoothstep.
    struct Tuning {
        /// Depth range of the wide window, in pixels of disparity, below which there is no
        /// discontinuity worth separating.
        var seedSignificance = SIMD2<Float>(1.0, 2.5)
        /// Gaussian mass of the weaker side's seeds at a texel. A side's mass falls away from its
        /// seeds until the kernel's cutoff, so this places the edge of the trusted band, which has
        /// to reach as far as the far content an eye uncovers: the disparity between the layers
        /// plus half of the soft edge, about 14 px in the recorded footage. Ten times these values
        /// end the band short of that, so the uncovered strip comes out too wide and a pixel or two
        /// out of place; half of them or less lets it run into far strips narrower than the margin,
        /// where the next surface's colour shows as a notch. The band doesn't follow the strength:
        /// at twice the shipped one it ends short in the same way, and the mesh draws the rest.
        var support = SIMD2<Float>(2e-6, 2e-5)
        /// Distance between the two sides' colours (0…1 per channel).
        var contrast = SIMD2<Float>(0.06, 0.15)
        /// Variance of depth within either side.
        var spread = SIMD2<Float>(0.0016, 0.01)
        /// Disparity between the two sides' depths, in pixels.
        var separation = SIMD2<Float>(0.5, 1.5)
        /// How far from an outline a side's seeds start, in table texels (four composition pixels
        /// each): the margin that keeps the outline's own soft edge out of the colours taken for
        /// either side. Two texels is about the width of an edge in the recorded footage. It
        /// doesn't follow the strength or the convergence, which change how far a layer moves but
        /// not how soft an edge is. A margin wider than a narrow far strip finds no clean far
        /// pixels in it and takes the next surface's colour for the strip instead; a narrower one
        /// lets the soft edge into both sides.
        var seedRadius = 2
        /// The Gaussian's sigma as a share of that distance.
        var sigmaRatio: Float = 0.625
        /// Where the Gaussian is cut off, in sigmas.
        var kernelSigmas: Float = 4
    }

    struct Params {
        var depthMap: SIMD4<Float>
        var mapping: SIMD4<Float>
        var stereo: SIMD4<Float>
        var sizes: SIMD4<Int32>
        var radii: SIMD4<Int32>
        var shape: SIMD4<Float>
        var gates: SIMD4<Float>
        var gates2: SIMD4<Float>
    }

    struct PatchUniforms {
        var stereo: SIMD4<Float>
        var sizes: SIMD4<Int32>
        var tone: SIMD4<Float>
    }

    /// The tables one frame's split is built in, and the layers it leaves.
    struct Layers {
        let width: Int
        let height: Int
        let tableWidth: Int
        let tableHeight: Int
        let table: MTLTexture
        let rangeRows: MTLTexture
        let ranges: MTLTexture
        let seeds: MTLTexture
        let nearRows: MTLTexture
        let farRows: MTLTexture
        let depthRows: MTLTexture
        let nearSums: MTLTexture
        let farSums: MTLTexture
        let depthSums: MTLTexture
        let trust: MTLTexture
        let reachRows: MTLTexture
        let reach: MTLTexture
        /// Near layer premultiplied (rgb) and its alpha.
        let nearLayer: MTLTexture
        /// Far layer (rgb) and the trust weight.
        let farLayer: MTLTexture
        /// Depth of the near and far layers.
        let depth: MTLTexture
    }

    private struct Frame {
        var convergence: Float
        var pixelsPerDepth: Float
        var scanRadius: Int
    }

    private struct Pipelines {
        let downsample: MTLComputePipelineState
        let rangeH: MTLComputePipelineState
        let rangeV: MTLComputePipelineState
        let seedBlurH: MTLComputePipelineState
        let seedBlurV: MTLComputePipelineState
        let trust: MTLComputePipelineState
        let reachH: MTLComputePipelineState
        let reachV: MTLComputePipelineState
        let compose: MTLComputePipelineState
    }

    /// The decomposition runs at no more than this width; a larger picture is composed at this size
    /// and sampled up. Fine source detail can be lost inside the trusted patch above this cap.
    static let maxWidth = 2048
    /// Below this the farthest a pixel moves is a pixel or two and the mesh has nothing to fold.
    static let minimumShift: Float = 2

    var tuning = Tuning()
    private(set) var layers: Layers?
    private let device: MTLDevice
    private let pipelines: Pipelines
    private let patchPipeline: MTLRenderPipelineState
    private let patchDepthState: MTLDepthStencilState
    private var frame: Frame?

    init?(device: MTLDevice, library: MTLLibrary, pixelFormat: MTLPixelFormat, sampleCount: Int) {
        func compute(_ name: String) -> MTLComputePipelineState? {
            library.makeFunction(name: name).flatMap { try? device.makeComputePipelineState(function: $0) }
        }
        guard let downsample = compute("layerDownsample"),
              let rangeH = compute("layerRangeH"),
              let rangeV = compute("layerRangeV"),
              let seedBlurH = compute("layerSeedBlurH"),
              let seedBlurV = compute("layerSeedBlurV"),
              let trust = compute("layerTrust"),
              let reachH = compute("layerReachH"),
              let reachV = compute("layerReachV"),
              let compose = compute("layerCompose"),
              let vertex = library.makeFunction(name: "layeredPatchVertex"),
              let fragment = library.makeFunction(name: "layeredPatchFragment") else { return nil }
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = vertex
        descriptor.fragmentFunction = fragment
        // The patch draws inside the mesh's pass, so it matches that pass's attachments.
        descriptor.rasterSampleCount = sampleCount
        descriptor.depthAttachmentPixelFormat = .depth32Float
        guard let color = descriptor.colorAttachments[0] else { return nil }
        color.pixelFormat = pixelFormat
        color.isBlendingEnabled = true
        color.rgbBlendOperation = .add
        color.alphaBlendOperation = .add
        color.sourceRGBBlendFactor = .sourceAlpha
        color.destinationRGBBlendFactor = .oneMinusSourceAlpha
        color.sourceAlphaBlendFactor = .zero
        color.destinationAlphaBlendFactor = .one
        let depthDescriptor = MTLDepthStencilDescriptor()
        depthDescriptor.depthCompareFunction = .always
        depthDescriptor.isDepthWriteEnabled = false
        guard let render = try? device.makeRenderPipelineState(descriptor: descriptor),
              let depthState = device.makeDepthStencilState(descriptor: depthDescriptor) else { return nil }
        self.device = device
        pipelines = Pipelines(downsample: downsample, rangeH: rangeH, rangeV: rangeV,
                              seedBlurH: seedBlurH, seedBlurV: seedBlurV, trust: trust,
                              reachH: reachH, reachV: reachV, compose: compose)
        patchPipeline = render
        patchDepthState = depthState
    }

    /// Frees the tables. The next `prepare` rebuilds them.
    func reset() {
        layers = nil
        frame = nil
    }

    /// Splits the frame into its layers, once for both eyes. False when there is nothing to do
    /// (the strongest disparity is a pixel or two) or a table couldn't be made; the mesh then
    /// draws alone.
    func prepare(source: MTLTexture, depth: PumpFrameDepth, strength: Float, convergence: Float,
                 continuesBackground: Bool = false, commandBuffer: MTLCommandBuffer) -> Bool {
        frame = nil
        guard source.width > 0, source.height > 0 else { return false }
        let width = min(source.width, Self.maxWidth)
        let height = max(1, Int((Double(source.height) * Double(width) / Double(source.width)).rounded()))
        let pixelsPerDepth = strength * Float(width)
        let maxShift = pixelsPerDepth * max(convergence, 1 - convergence)
        guard maxShift.isFinite, maxShift >= Self.minimumShift,
              let layers = makeLayers(width: width, height: height) else { return false }

        // Seeds start `seedRadius` texels inside each side of an outline. How far the trust is
        // extended past it does follow the largest shift, since that is how far a pixel can move.
        let seedRadius = max(tuning.seedRadius, 1)
        let sigma = tuning.sigmaRatio * Float(seedRadius)
        var params = Params(
            depthMap: SIMD4(depth.uvScale.x, depth.uvScale.y, depth.uvOffset.x, depth.uvOffset.y),
            mapping: SIMD4(depth.valueScale, depth.valueBias, depth.budget.knees.x, depth.budget.knees.y),
            stereo: SIMD4(strength, convergence, pixelsPerDepth, maxShift),
            sizes: SIMD4(Int32(width), Int32(height), Int32(layers.tableWidth), Int32(layers.tableHeight)),
            radii: SIMD4(Int32(seedRadius), Int32(3 * seedRadius), Int32((tuning.kernelSigmas * sigma).rounded(.up)),
                         Int32(((maxShift + 8) / 4).rounded(.up))),
            shape: SIMD4(sigma, tuning.support.x, tuning.support.y, continuesBackground ? 1 : 0),
            gates: SIMD4(tuning.seedSignificance.x, tuning.seedSignificance.y,
                         tuning.contrast.x, tuning.contrast.y),
            gates2: SIMD4(tuning.spread.x, tuning.spread.y, tuning.separation.x, tuning.separation.y))

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return false }
        func dispatch(_ pipeline: MTLComputePipelineState, _ textures: [MTLTexture], over size: MTLSize) {
            encoder.setComputePipelineState(pipeline)
            for (index, texture) in textures.enumerated() { encoder.setTexture(texture, index: index) }
            encoder.setBytes(&params, length: MemoryLayout<Params>.stride, index: 0)
            encoder.dispatchThreads(size, threadsPerThreadgroup: MTLSize(width: 16, height: 8, depth: 1))
        }
        let tableSize = MTLSize(width: layers.tableWidth, height: layers.tableHeight, depth: 1)
        dispatch(pipelines.downsample, [source, depth.texture, layers.table], over: tableSize)
        dispatch(pipelines.rangeH, [layers.table, layers.rangeRows], over: tableSize)
        dispatch(pipelines.rangeV, [layers.rangeRows, layers.ranges, layers.seeds], over: tableSize)
        dispatch(pipelines.seedBlurH, [layers.table, layers.seeds, layers.nearRows, layers.farRows,
                                       layers.depthRows], over: tableSize)
        dispatch(pipelines.seedBlurV, [layers.nearRows, layers.farRows, layers.depthRows,
                                       layers.nearSums, layers.farSums, layers.depthSums], over: tableSize)
        dispatch(pipelines.trust, [layers.nearSums, layers.farSums, layers.depthSums, layers.trust],
                 over: tableSize)
        dispatch(pipelines.reachH, [layers.trust, layers.reachRows], over: tableSize)
        dispatch(pipelines.reachV, [layers.reachRows, layers.reach], over: tableSize)
        dispatch(pipelines.compose, [source, depth.texture, layers.nearSums, layers.farSums, layers.depthSums,
                                     layers.ranges, layers.trust, layers.nearLayer, layers.farLayer,
                                     layers.depth], over: MTLSize(width: width, height: height, depth: 1))
        encoder.endEncoding()

        frame = Frame(convergence: convergence, pixelsPerDepth: pixelsPerDepth,
                      scanRadius: Int(maxShift.rounded(.up)) + 2)
        return true
    }

    /// Draws one eye's patch in the mesh pass's own encoder, after the mesh. Does nothing unless
    /// `prepare` has succeeded for this frame. `eyeSign` is the mesh's (+1 left, -1 right).
    func encodePatch(into encoder: MTLRenderCommandEncoder, eyeSign: Float, mirror: Bool,
                     brightness: Float, contrast: Float, saturation: Float, extendedRange: Bool) {
        guard let layers, let frame else { return }
        // The layers are stored as the source is, and a mirrored picture reads it flipped, so the
        // same displacement runs the other way round in them.
        var uniforms = PatchUniforms(
            stereo: SIMD4(mirror ? -eyeSign : eyeSign, frame.convergence, frame.pixelsPerDepth, mirror ? 1 : 0),
            sizes: SIMD4(Int32(layers.width), Int32(layers.height), Int32(frame.scanRadius), 0),
            tone: SIMD4(brightness, contrast, saturation, extendedRange ? 1 : 0))
        encoder.setRenderPipelineState(patchPipeline)
        encoder.setDepthStencilState(patchDepthState)
        encoder.setFragmentTexture(layers.depth, index: 0)
        encoder.setFragmentTexture(layers.nearLayer, index: 1)
        encoder.setFragmentTexture(layers.farLayer, index: 2)
        encoder.setFragmentTexture(layers.reach, index: 3)
        encoder.setFragmentBytes(&uniforms, length: MemoryLayout<PatchUniforms>.stride, index: 0)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
    }

    private func makeLayers(width: Int, height: Int) -> Layers? {
        if let layers, layers.width == width, layers.height == height { return layers }
        layers = nil
        let tableWidth = (width + 3) / 4
        let tableHeight = (height + 3) / 4
        func texture(_ format: MTLPixelFormat, _ w: Int, _ h: Int) -> MTLTexture? {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: w, height: h,
                                                                      mipmapped: false)
            descriptor.storageMode = .private
            descriptor.usage = [.shaderRead, .shaderWrite]
            return device.makeTexture(descriptor: descriptor)
        }
        func table() -> MTLTexture? { texture(.rgba32Float, tableWidth, tableHeight) }
        guard let a = table(), let b = table(), let c = table(), let d = table(), let e = table(),
              let f = table(), let g = table(), let h = table(), let i = table(), let j = table(),
              let k = table(), let l = table(), let m = table(),
              let nearLayer = texture(.rgba16Float, width, height),
              let farLayer = texture(.rgba16Float, width, height),
              let depth = texture(.rg16Float, width, height) else { return nil }
        let made = Layers(width: width, height: height, tableWidth: tableWidth, tableHeight: tableHeight,
                          table: a, rangeRows: b, ranges: c, seeds: d, nearRows: e, farRows: f,
                          depthRows: g, nearSums: h, farSums: i, depthSums: j, trust: k,
                          reachRows: l, reach: m, nearLayer: nearLayer, farLayer: farLayer, depth: depth)
        layers = made
        return made
    }
}
#endif
