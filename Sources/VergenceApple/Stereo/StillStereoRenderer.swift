#if canImport(Metal)
import CoreGraphics
import CoreImage
import Metal
import VergenceCore

/// One-shot, full-colour stereo. Invoke on a worker queue; this waits for the
/// GPU and never owns a display clock. The caller bounds the decoded source.
public final class StillStereoRenderer: @unchecked Sendable {
    public struct Result: @unchecked Sendable {
        public let left: MTLTexture
        public let right: MTLTexture
        public let depth: MTLTexture
        public let colorSpace: CGColorSpace
    }
    public enum Failure: Error { case unavailable, invalidDepth, allocation, gpu }
    private let device: MTLDevice
    private let library: MTLLibrary
    private let queue: MTLCommandQueue
    private let context: CIContext
    private let resources: StereoWarpResources
    private let warp: StereoMeshWarp
    private let guide: MTLComputePipelineState
    private let horizontal: MTLComputePipelineState
    private let vertical: MTLComputePipelineState
    private let colorSpace = CGColorSpace(name: CGColorSpace.extendedLinearDisplayP3)!

    public convenience init?() {
        guard let metal = VergenceMetal.shared else { return nil }
        self.init(device: metal.device, library: metal.library)
    }

    /// Injection is for the host diagnostic harness, which compiles the real
    /// Metal source when SwiftPM does not produce a default.metallib.
    public init?(device: MTLDevice, library: MTLLibrary) {
        guard let queue = device.makeCommandQueue(),
              let resources = StereoWarpResources(device: device, library: library, pixelFormat: .rgba16Float),
              let g = library.makeFunction(name: "depthGuideLuma"),
              let h = library.makeFunction(name: "depthJointBilateralH"),
              let v = library.makeFunction(name: "depthJointBilateralV"),
              let gp = try? device.makeComputePipelineState(function: g),
              let hp = try? device.makeComputePipelineState(function: h),
              let vp = try? device.makeComputePipelineState(function: v) else { return nil }
        self.device = device
        self.library = library
        self.queue = queue
        self.context = CIContext(mtlDevice: device, options: [.workingColorSpace: colorSpace, .cacheIntermediates: false])
        self.resources = resources
        self.warp = StereoMeshWarp(resources: resources)
        self.guide = gp
        self.horizontal = hp
        self.vertical = vp
    }

    /// `strength` is the same per-eye UV shift used by video. Closest depth is
    /// on the glass (convergence 1); farther content has uncrossed disparity.
    /// Colour stays linear extended P3/half-float through the warp.
    public func render(image: CGImage, depth map: StillDepthMap,
                       strength: Float = Float(Pseudo3DSettings.default.depthStrength),
                       refine: Bool = true, separatesBoundaries: Bool = false, mirror: Bool = false) throws -> Result {
        guard map.width > 0, map.height > 0, map.values.count == map.width * map.height,
              map.preciseValues == nil || map.preciseValues?.count == map.values.count else { throw Failure.invalidDepth }
        func texture(_ format: MTLPixelFormat, _ width: Int, _ height: Int,
                     shared: Bool = false) throws -> MTLTexture {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format,
                width: width, height: height, mipmapped: false)
            d.storageMode = shared ? .shared : .private
            d.usage = [.shaderRead, .shaderWrite, .renderTarget]
            guard let t = device.makeTexture(descriptor: d) else { throw Failure.allocation }
            return t
        }
        let source = try texture(.rgba16Float, image.width, image.height)
        let input = CIImage(cgImage: image).transformed(by: CGAffineTransform(translationX: 0, y: CGFloat(image.height)).scaledBy(x: 1, y: -1))
        context.render(input, to: source, commandBuffer: nil,
            bounds: CGRect(x: 0, y: 0, width: image.width, height: image.height), colorSpace: colorSpace)
        let raw = try texture(.r32Float, map.width, map.height, shared: true)
        let values = map.preciseValues ?? map.values.map { Float($0) / 255 }
        raw.replace(region: MTLRegionMake2D(0, 0, map.width, map.height), mipmapLevel: 0,
                    withBytes: values, bytesPerRow: map.width * MemoryLayout<Float>.stride)
        guard let command = queue.makeCommandBuffer() else { throw Failure.allocation }
        var finalDepth = raw
        if refine {
            // Shared video kernels: model-resolution smoothing, then 2x guided
            // upsample. No luma snap, median erosion or temporal state.
            for factor in [1, 2] {
                let w = min(map.width * factor, image.width), h = min(map.height * factor, image.height)
                let luma = try texture(.r16Float, w, h)
                let scratch = try texture(.r32Float, w, h)
                let output = try texture(.r32Float, w, h)
                guard let enc = command.makeComputeCommandEncoder() else { throw Failure.allocation }
                let threads = MTLSize(width: w, height: h, depth: 1)
                let group = MTLSize(width: 16, height: 8, depth: 1)
                var uv = DepthGuideParams(uvScale: SIMD2(1, 1), uvOffset: .zero)
                var params = DepthStabilizeParams(blurRadius: factor == 1 ? 4 : 6,
                    blurSigma: factor == 1 ? 2 : 3, baseAlpha: 1, motionGain: 0, hasPrev: 0, sigmaLuma: 0.06)
                enc.setComputePipelineState(guide)
                enc.setTexture(source, index: 0)
                enc.setTexture(luma, index: 1)
                enc.setBytes(&uv, length: MemoryLayout<DepthGuideParams>.stride, index: 0)
                enc.dispatchThreads(threads, threadsPerThreadgroup: group)
                enc.setComputePipelineState(horizontal)
                enc.setTexture(finalDepth, index: 0)
                enc.setTexture(luma, index: 1)
                enc.setTexture(scratch, index: 2)
                enc.setBytes(&params, length: MemoryLayout<DepthStabilizeParams>.stride, index: 0)
                enc.dispatchThreads(threads, threadsPerThreadgroup: group)
                enc.setComputePipelineState(vertical)
                enc.setTexture(scratch, index: 0)
                enc.setTexture(output, index: 2)
                enc.dispatchThreads(threads, threadsPerThreadgroup: group)
                enc.endEncoding()
                finalDepth = output
            }
        }
        let left = try texture(.rgba16Float, image.width, image.height)
        let right = try texture(.rgba16Float, image.width, image.height)
        let boundary = separatesBoundaries ? StereoBoundaryWarp(device: device, library: library) : nil
        if separatesBoundaries && boundary == nil { throw Failure.unavailable }
        for (eye, target) in [left, right].enumerated() {
            if let boundary {
                let uniforms = StereoBoundaryWarp.Uniforms(
                    stereo: SIMD4(strength, 1, eye == 0 ? 1 : -1, mirror ? 1 : 0),
                    uv: SIMD4(1, 1, 0, 0), mapping: SIMD4(1, 0, 0, 1),
                    adjustments: SIMD4(0, 1, 1, 0),
                    boundary: SIMD4(0.08, Float(target.width), 1 / Float(StereoWarpResources.denseGridColumns - 1), 2))
                guard boundary.encode(source: source, depth: finalDepth, destination: target,
                    positions: resources.denseGridPositions, indices: resources.denseGridIndices,
                    indexCount: resources.denseGridIndexCount, uniforms: uniforms, commandBuffer: command) else {
                    throw Failure.allocation
                }
                continue
            }
            let u = VideoStereoUniforms(brightness: 0, contrast: 1, saturation: 1,
                depthStrength: strength, convergence: 1, eyeSign: eye == 0 ? 1 : -1,
                mirror: mirror ? 1 : 0, useDepth: 1, depthUVScale: SIMD2(1, 1), depthUVOffset: .zero,
                depthValueScale: 1, depthValueBias: 0, depthBudget: SIMD2(0, 1), extendedRange: 1)
            guard warp.encode(source: source, depth: finalDepth, destination: target,
                uniforms: u, dense: true, eye: eye, commandBuffer: command) else { throw Failure.allocation }
        }
        command.commit()
        command.waitUntilCompleted()
        guard command.status == .completed else { throw Failure.gpu }
        return Result(left: left, right: right, depth: finalDepth, colorSpace: colorSpace)
    }
}
#endif
