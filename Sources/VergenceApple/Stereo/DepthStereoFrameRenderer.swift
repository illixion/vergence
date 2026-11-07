/*
 Vergence - one frame's two eyes from its depth, packed into one picture

 The stereo pump's warp without the pump: no clock, renderer or presentation.
 A decoded frame (BGRA, or half-float RGBA holding an HDR picture's PQ/HLG
 code values; `FramePixelFormat`) and its depth go in; both eyes come out,
 drawn by the occlusion-tested mesh on the dense grid exactly as cached depth
 plays, and are copied side by side or top and bottom into a frame of the
 same format for an encoder. `vergence convert` writes a file from these.

 Calls must be serialised and made off the main thread: each waits for the GPU.
 */

#if canImport(Metal) && canImport(CoreVideo)
import CoreVideo
import Metal
import VergenceCore

public final class DepthStereoFrameRenderer: @unchecked Sendable {
    public enum Arrangement: Sendable {
        /// Left eye on the left: a picture twice the frame's width.
        case sideBySide
        /// Left eye on top: twice the frame's height.
        case topAndBottom

        /// The packed picture's size for a frame of `width`×`height`.
        public func packedSize(width: Int, height: Int) -> (width: Int, height: Int) {
            self == .sideBySide ? (width * 2, height) : (width, height * 2)
        }
    }

    public enum Failure: Error, LocalizedError {
        case unsupportedFormat(OSType)
        case sizeMismatch
        case allocation
        case gpu(String)

        public var errorDescription: String? {
            switch self {
            case .unsupportedFormat(let format): return "frames must be 32BGRA or 64RGBAHalf, not \(format)"
            case .sizeMismatch: return "the packed picture isn't twice the frame"
            case .allocation: return "out of GPU memory for the eyes"
            case .gpu(let detail): return "the warp failed on the GPU (\(detail))"
            }
        }
    }

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let cache: CVMetalTextureCache
    private let sdr: StereoMeshWarp
    private let hdr: StereoMeshWarp?
    private var eyes: [MTLTexture] = []

    /// Nil without Metal or the package's shaders (see `VergenceMetal.useShaders`).
    public init?() {
        guard let resources = StereoWarpResources.shared,
              let queue = resources.device.makeCommandQueue() else { return nil }
        var cache: CVMetalTextureCache?
        CVMetalTextureCacheCreate(nil, nil, resources.device, nil, &cache)
        guard let cache else { return nil }
        self.device = resources.device
        self.queue = queue
        self.cache = cache
        self.sdr = StereoMeshWarp(resources: resources)
        self.hdr = StereoWarpResources.sharedHDR.map { StereoMeshWarp(resources: $0) }
    }

    /// Draws both eyes of `frame` into `packed` (the same pixel format, packed
    /// size). `strength` and `convergence` are `Pseudo3DSettings`'; without
    /// `depth` both eyes show the frame as it is (flat). The packed picture
    /// gets the frame's colour tags.
    public func render(frame: CVPixelBuffer, depth: PumpFrameDepth?, strength: Float, convergence: Float,
                       arrangement: Arrangement, into packed: CVPixelBuffer) throws {
        let format = CVPixelBufferGetPixelFormatType(frame)
        guard let metalFormat = FramePixelFormat.metal(format),
              CVPixelBufferGetPixelFormatType(packed) == format else { throw Failure.unsupportedFormat(format) }
        let width = CVPixelBufferGetWidth(frame), height = CVPixelBufferGetHeight(frame)
        let size = arrangement.packedSize(width: width, height: height)
        guard CVPixelBufferGetWidth(packed) == size.width, CVPixelBufferGetHeight(packed) == size.height else {
            throw Failure.sizeMismatch
        }
        let isHDR = format == FramePixelFormat.hdr
        let warp = isHDR ? (hdr ?? sdr) : sdr
        guard let source = FramePixelFormat.texture(frame, cache: cache),
              let target = FramePixelFormat.texture(packed, cache: cache),
              let command = queue.makeCommandBuffer() else { throw Failure.allocation }
        if eyes.first.map({ $0.width != width || $0.height != height || $0.pixelFormat != metalFormat }) ?? true {
            eyes = try (0 ..< 2).map { _ in
                let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: metalFormat, width: width, height: height,
                                                                 mipmapped: false)
                d.usage = [.renderTarget, .shaderRead]
                d.storageMode = .private
                guard let texture = device.makeTexture(descriptor: d) else { throw Failure.allocation }
                return texture
            }
        }
        if let depth {
            for (eye, destination) in eyes.enumerated() {
                // As the pump draws cached depth (`StereoPump.encodeEye`).
                let uniforms = VideoStereoUniforms(
                    brightness: 0, contrast: 1, saturation: 1,
                    depthStrength: strength, convergence: convergence, eyeSign: eye == 0 ? 1 : -1,
                    mirror: 0, useDepth: 1, depthUVScale: depth.uvScale, depthUVOffset: depth.uvOffset,
                    depthValueScale: depth.valueScale, depthValueBias: depth.valueBias,
                    depthBudget: depth.budget.knees, edgeFill: DepthRefinement.fillsUncoveredEdges ? 1 : 0)
                guard warp.encode(source: source.texture, depth: depth.texture, destination: destination,
                                  uniforms: uniforms, dense: true, eye: eye, commandBuffer: command) else {
                    throw Failure.allocation
                }
            }
        }
        guard let blit = command.makeBlitCommandEncoder() else { throw Failure.allocation }
        for eye in 0 ..< 2 {
            let origin = arrangement == .sideBySide ? MTLOrigin(x: eye * width, y: 0, z: 0)
                                                    : MTLOrigin(x: 0, y: eye * height, z: 0)
            blit.copy(from: depth == nil ? source.texture : eyes[eye], sourceSlice: 0, sourceLevel: 0,
                      sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0), sourceSize: MTLSize(width: width, height: height, depth: 1),
                      to: target.texture, destinationSlice: 0, destinationLevel: 0, destinationOrigin: origin)
        }
        blit.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        withExtendedLifetime((source.held, target.held, depth)) {}
        guard command.status == .completed else {
            throw Failure.gpu(command.error?.localizedDescription ?? "status \(command.status.rawValue)")
        }
        CVBufferPropagateAttachments(frame, packed)
    }
}
#endif
