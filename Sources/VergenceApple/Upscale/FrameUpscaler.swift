/*
 Vergence - upscaling frames 2× with the compact Real-ESRGAN network

 One or two eyes of a frame (a 2D picture, or both halves of a side-by-side
 or top-and-bottom one) go through `UpscaleNetwork`'s Core ML model in one
 batch and come out at twice the size, placed into a destination frame:

   frame ──prepare (Metal)──▶ planar float16 RGB tiles ──Core ML──▶ 2× tiles
         ──finish (Metal)──▶ destination pixels

 The model has a fixed input size (the Neural Engine runs nothing else well),
 compiled once per size and cached. An eye bigger than about a megapixel is
 cut into tiles that overlap by 36 pixels, which keeps the network's view of
 every pixel whole (its 34 convolutions see 34 pixels around each one); the
 overlap is computed twice and thrown away. A pair of eyes then has its fine
 detail brought into agreement (`StereoDetailFusion`). Pixel buffers back both ends, so
 nothing is copied between the GPU, the Neural Engine and the CPU.

 Colour passes straight through: the network sees the frame's own code
 values (BT.709 gamma for SDR, PQ or HLG for HDR) in 0…1, as it was trained
 on gamma-coded pictures.
 */

#if canImport(CoreML) && canImport(Metal) && canImport(CoreVideo)
import CoreML
import CoreVideo
import Foundation
import Metal
import VergenceCore

public final class FrameUpscaler: @unchecked Sendable {
    public struct Failure: Error, CustomStringConvertible {
        public let description: String
        init(_ text: String) { description = text }
    }

    /// Where an eye comes from: a rectangle of the source frame in its
    /// pixels (a squeezed half-resolution eye is stretched back out).
    public struct Region: Sendable, Equatable {
        public var x, y, width, height: Double
        public init(x: Double, y: Double, width: Double, height: Double) {
            self.x = x; self.y = y; self.width = width; self.height = height
        }
    }

    /// Each eye's size going in; it comes out twice that.
    public let inputSize: (width: Int, height: Int)
    public var outputSize: (width: Int, height: Int) { (inputSize.width * 2, inputSize.height * 2) }
    public let eyes: Int
    let grid: (x: Int, y: Int)
    let core: (width: Int, height: Int)
    let tile: (width: Int, height: Int)
    let halo: Int
    var batch: Int { eyes * grid.x * grid.y }

    /// One model per compute unit, each taking its share of the batch.
    private struct Part {
        let model: MLModel
        let input: CVPixelBuffer
        let output: CVPixelBuffer
        let inputArray: MLMultiArray
        let options: MLPredictionOptions
    }
    private let parts: [Part]
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let prepare: MTLComputePipelineState
    private let finish: MTLComputePipelineState
    private var textureCache: CVMetalTextureCache
    /// Both eyes' detail brought into agreement (`StereoDetailFusion`).
    private let fusion: StereoDetailFusion?
    public var fusesEyes: Bool { fusion != nil }

    /// The largest tile, in input pixels.
    public static let tilePixels = 1024 * 640
    static let overlap = 36

    /// `units`: where the network runs. Two (the default: the Neural Engine
    /// and the GPU) split each batch between them and run at once, a 2D
    /// frame cut in two for it; one runs it all.
    /// `fuseEyes`: with two eyes, bring their fine detail into agreement.
    public init(network: UpscaleNetwork, width: Int, height: Int, eyes: Int,
                units: [MLComputeUnits] = [.cpuAndNeuralEngine, .cpuAndGPU],
                fuseEyes: Bool = true, maxTilePixels: Int = FrameUpscaler.tilePixels) async throws {
        guard width >= 16, height >= 16, (1 ... 2).contains(eyes), (1 ... 2).contains(units.count) else {
            throw Failure("can't upscale \(eyes) eyes of \(width)×\(height) on \(units.count) units")
        }
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { throw Failure("no Metal device") }
        inputSize = (width, height)
        self.eyes = eyes
        // Tiles: as few as keep each under tilePixels, and an even batch to
        // share between two units.
        var gx = 1, gy = 1
        while (width + gx - 1) / gx * ((height + gy - 1) / gy) > maxTilePixels
            || (units.count == 2 && eyes * gx * gy % 2 == 1 && height >= 64) {
            if (width + gx - 1) / gx >= (height + gy - 1) / gy * 2 { gx += 1 } else { gy += 1 }
        }
        grid = (gx, gy)
        core = ((width + gx - 1) / gx, (height + gy - 1) / gy)
        halo = Self.overlap
        // A tile stays inside the eye (so the network's zero padding falls on
        // the picture's real edges, as in one piece): the overlap is on the
        // inner side of an edge tile.
        tile = (min(width, core.width + (gx > 1 ? 2 * halo : 0)), min(height, core.height + (gy > 1 ? 2 * halo : 0)))
        let batch = eyes * gx * gy
        let shares = batch % units.count == 0 ? units : [units[0]]
        let share = batch / shares.count

        let compiled = try await network.compiledModel(batch: share, width: tile.width, height: tile.height)
        func planes(_ w: Int, _ h: Int) throws -> (CVPixelBuffer, MLMultiArray) {
            var buffer: CVPixelBuffer?
            let attributes: [String: Any] = [kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
                                             kCVPixelBufferMetalCompatibilityKey as String: true]
            CVPixelBufferCreate(nil, w, share * 3 * h, kCVPixelFormatType_OneComponent16Half, attributes as CFDictionary, &buffer)
            guard let buffer else { throw Failure("can't make \(w)×\(share * 3 * h) planes") }
            let array = MLMultiArray(pixelBuffer: buffer, shape: [NSNumber(value: share), 3, NSNumber(value: h), NSNumber(value: w)])
            return (buffer, array)
        }
        var parts: [Part] = []
        for unit in shares {
            let configuration = MLModelConfiguration()
            configuration.computeUnits = unit
            let model = try await MLModel.load(contentsOf: compiled, configuration: configuration)
            let (input, inputArray) = try planes(tile.width, tile.height)
            let (output, outputArray) = try planes(tile.width * 2, tile.height * 2)
            let options = MLPredictionOptions()
            options.outputBackings = ["upscaled": outputArray]
            parts.append(Part(model: model, input: input, output: output, inputArray: inputArray, options: options))
        }
        self.parts = parts

        self.device = device
        self.queue = queue
        let library = try await device.makeLibrary(source: Self.shaders, options: nil)
        guard let p = library.makeFunction(name: "upscale_prepare"), let f = library.makeFunction(name: "upscale_finish") else {
            throw Failure("the upscaler's kernels didn't build")
        }
        prepare = try await device.makeComputePipelineState(function: p)
        finish = try await device.makeComputePipelineState(function: f)
        var cache: CVMetalTextureCache?
        CVMetalTextureCacheCreate(nil, nil, device, nil, &cache)
        guard let cache else { throw Failure("no Metal texture cache") }
        textureCache = cache
        fusion = eyes == 2 && fuseEyes ? try await StereoDetailFusion(device: device) : nil
    }

    /// Upscales `regions` of `frame` (one per eye) into `destination`, the
    /// eye `i` with its top left corner at `placements[i]`. Both frames are
    /// `FramePixelFormat.sdr` or `.hdr`.
    public func upscale(_ frame: CVPixelBuffer, regions: [Region], into destination: CVPixelBuffer,
                        at placements: [(x: Int, y: Int)]) throws {
        guard regions.count == eyes, placements.count == eyes else { throw Failure("\(eyes) eyes, \(regions.count) regions") }
        let source = try texture(frame)
        let target = try texture(destination)
        let ins = try parts.map { try texture($0.input, format: .r16Float) }
        let outs = try parts.map { try texture($0.output, format: .r16Float) }
        var params = Params()
        params.share = UInt32(batch / parts.count)
        let fw = Double(CVPixelBufferGetWidth(frame)), fh = Double(CVPixelBufferGetHeight(frame))
        params.eyeOrigin0 = SIMD2(Float(regions[0].x / fw), Float(regions[0].y / fh))
        params.eyeSpan0 = SIMD2(Float(regions[0].width / fw), Float(regions[0].height / fh))
        let second = regions.count > 1 ? regions[1] : regions[0]
        params.eyeOrigin1 = SIMD2(Float(second.x / fw), Float(second.y / fh))
        params.eyeSpan1 = SIMD2(Float(second.width / fw), Float(second.height / fh))
        params.eyeSize = SIMD2(UInt32(inputSize.width), UInt32(inputSize.height))
        params.tileSize = SIMD2(UInt32(tile.width), UInt32(tile.height))
        params.grid = SIMD2(UInt32(grid.x), UInt32(grid.y))
        params.core = SIMD2(UInt32(core.width), UInt32(core.height))
        params.halo = SIMD2(UInt32(grid.x > 1 ? halo : 0), UInt32(grid.y > 1 ? halo : 0))
        let p1 = placements.count > 1 ? placements[1] : placements[0]
        params.placement0 = SIMD2(UInt32(placements[0].x), UInt32(placements[0].y))
        params.placement1 = SIMD2(UInt32(p1.x), UInt32(p1.y))
        params.destination = SIMD2(UInt32(CVPixelBufferGetWidth(destination)), UInt32(CVPixelBufferGetHeight(destination)))

        // Prepare, then the model, then finish (into a stage, with the eyes'
        // detail fused from there into the destination).
        try run(prepare, params, [source, ins[0], ins.last!], MTLSize(width: tile.width, height: tile.height, depth: batch))
        if parts.count == 1 {
            _ = try parts[0].model.prediction(from: Inputs(image: parts[0].inputArray), options: parts[0].options)
        } else {
            let failures = Failures()
            DispatchQueue.concurrentPerform(iterations: parts.count) { i in
                do { _ = try parts[i].model.prediction(from: Inputs(image: parts[i].inputArray), options: parts[i].options) }
                catch { failures.add(error) }
            }
            if let error = failures.first { throw error }
        }
        let size = MTLSize(width: outputSize.width, height: outputSize.height, depth: eyes)
        guard let fusion else { return try run(finish, params, [outs[0], outs.last!, target], size) }
        let stage = try fusion.stage(width: target.width, height: target.height)
        var fused = StereoDetailFusion.Params()
        fused.eyeOrigin0 = params.eyeOrigin0; fused.eyeSpan0 = params.eyeSpan0
        fused.eyeOrigin1 = params.eyeOrigin1; fused.eyeSpan1 = params.eyeSpan1
        fused.eyeSize = params.eyeSize
        fused.placement0 = params.placement0; fused.placement1 = params.placement1
        fused.destination = params.destination
        fused.radius = Int32(min(48, max(4, Int((Double(inputSize.width) * 0.04).rounded()))))
        try run(finish, params, [outs[0], outs.last!, stage], size) { commands in
            try fusion.encode(commands, source: source, destination: target, params: fused)
        }
    }

    private func run(_ pipeline: MTLComputePipelineState, _ params: Params, _ textures: [MTLTexture], _ size: MTLSize,
                     then more: ((MTLCommandBuffer) throws -> Void)? = nil) throws {
        guard let commands = queue.makeCommandBuffer(), let encoder = commands.makeComputeCommandEncoder() else {
            throw Failure("no command buffer")
        }
        var p = params
        encoder.setComputePipelineState(pipeline)
        encoder.setTextures(textures, range: 0 ..< textures.count)
        encoder.setBytes(&p, length: MemoryLayout<Params>.stride, index: 0)
        let w = pipeline.threadExecutionWidth
        encoder.dispatchThreads(size, threadsPerThreadgroup: MTLSize(width: w, height: max(1, pipeline.maxTotalThreadsPerThreadgroup / w / 2), depth: 1))
        encoder.endEncoding()
        try more?(commands)
        commands.commit()
        commands.waitUntilCompleted()
        if let error = commands.error { throw Failure("the GPU failed: \(error.localizedDescription)") }
    }

    private func texture(_ buffer: CVPixelBuffer, format: MTLPixelFormat? = nil) throws -> MTLTexture {
        let pixelFormat = format ?? FramePixelFormat.metal(CVPixelBufferGetPixelFormatType(buffer))
        guard let pixelFormat else { throw Failure("frames must be BGRA or half-float RGBA") }
        var out: CVMetalTexture?
        CVMetalTextureCacheCreateTextureFromImage(nil, textureCache, buffer, nil, pixelFormat,
                                                  CVPixelBufferGetWidth(buffer), CVPixelBufferGetHeight(buffer), 0, &out)
        guard let out, let texture = CVMetalTextureGetTexture(out) else { throw Failure("can't see a frame as a texture") }
        return texture
    }

    private struct Params {
        var eyeOrigin0 = SIMD2<Float>(), eyeSpan0 = SIMD2<Float>(), eyeOrigin1 = SIMD2<Float>(), eyeSpan1 = SIMD2<Float>()
        var eyeSize = SIMD2<UInt32>(), tileSize = SIMD2<UInt32>(), grid = SIMD2<UInt32>(), core = SIMD2<UInt32>()
        var halo = SIMD2<UInt32>(), placement0 = SIMD2<UInt32>(), placement1 = SIMD2<UInt32>(), destination = SIMD2<UInt32>()
        var share: UInt32 = 1
    }

    private final class Failures: @unchecked Sendable {
        private let lock = NSLock()
        private var errors: [Error] = []
        func add(_ error: Error) { lock.withLock { errors.append(error) } }
        var first: Error? { lock.withLock { errors.first } }
    }

    private final class Inputs: MLFeatureProvider {
        let image: MLMultiArray
        init(image: MLMultiArray) { self.image = image }
        var featureNames: Set<String> { ["image"] }
        func featureValue(for name: String) -> MLFeatureValue? { name == "image" ? MLFeatureValue(multiArray: image) : nil }
    }

    static let shaders = """
    #include <metal_stdlib>
    using namespace metal;

    struct Params {
        float2 eyeOrigin0, eyeSpan0, eyeOrigin1, eyeSpan1;
        uint2 eyeSize, tileSize, grid, core, halo, placement0, placement1, destination;
        uint share;
    };

    // Each batch item b is eye (b / tiles), tile (b % tiles), row-major. A
    // tile starts `halo` before its own part, kept inside the eye.
    static int2 tileOrigin(constant Params& p, uint t) {
        uint2 at = uint2(t % p.grid.x, t / p.grid.x);
        return clamp(int2(at * p.core) - int2(p.halo), int2(0), int2(p.eyeSize) - int2(p.tileSize));
    }

    kernel void upscale_prepare(texture2d<float, access::sample> source [[texture(0)]],
                                texture2d<half, access::write> first [[texture(1)]],
                                texture2d<half, access::write> second [[texture(2)]],
                                constant Params& p [[buffer(0)]],
                                uint3 gid [[thread_position_in_grid]]) {
        if (gid.x >= p.tileSize.x || gid.y >= p.tileSize.y) return;
        uint tiles = p.grid.x * p.grid.y;
        uint eye = gid.z / tiles;
        int2 px = clamp(tileOrigin(p, gid.z % tiles) + int2(gid.xy), int2(0), int2(p.eyeSize) - 1);
        float2 origin = eye == 0 ? p.eyeOrigin0 : p.eyeOrigin1;
        float2 span = eye == 0 ? p.eyeSpan0 : p.eyeSpan1;
        constexpr sampler linear(filter::linear, address::clamp_to_edge, coord::normalized);
        float4 c = clamp(source.sample(linear, origin + (float2(px) + 0.5) / float2(p.eyeSize) * span), 0.0, 1.0);
        // Batch item b goes to unit b / share, as its item b % share.
        uint row = (gid.z % p.share) * 3 * p.tileSize.y + gid.y;
        texture2d<half, access::write> planes = gid.z < p.share ? first : second;
        planes.write(half4(half(c.r)), uint2(gid.x, row));
        planes.write(half4(half(c.g)), uint2(gid.x, row + p.tileSize.y));
        planes.write(half4(half(c.b)), uint2(gid.x, row + 2 * p.tileSize.y));
    }

    kernel void upscale_finish(texture2d<half, access::read> first [[texture(0)]],
                               texture2d<half, access::read> second [[texture(1)]],
                               texture2d<float, access::write> destination [[texture(2)]],
                               constant Params& p [[buffer(0)]],
                               uint3 gid [[thread_position_in_grid]]) {
        uint2 size = p.eyeSize * 2;
        if (gid.x >= size.x || gid.y >= size.y) return;
        uint2 at = min(gid.xy / 2 / p.core, p.grid - 1);
        uint t = at.y * p.grid.x + at.x;
        uint b = gid.z * p.grid.x * p.grid.y + t;
        int2 local = int2(gid.xy) - 2 * tileOrigin(p, t);
        uint height = p.tileSize.y * 2;
        uint row = (b % p.share) * 3 * height + uint(local.y);
        texture2d<half, access::read> planes = b < p.share ? first : second;
        float3 c = float3(planes.read(uint2(local.x, row)).r,
                          planes.read(uint2(local.x, row + height)).r,
                          planes.read(uint2(local.x, row + 2 * height)).r);
        uint2 out = (gid.z == 0 ? p.placement0 : p.placement1) + gid.xy;
        if (out.x < p.destination.x && out.y < p.destination.y) destination.write(float4(saturate(c), 1.0), out);
    }
    """
}
#endif
