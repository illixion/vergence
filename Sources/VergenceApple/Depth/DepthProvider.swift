/*
 Spatial Stash - Monocular Depth Provider

 Runs a Core ML monocular-depth model (Depth Anything V2) on decoded video
 frames and exposes the latest depth as a single-channel `MTLTexture` for the
 fake-3D warp (videoPseudo3DEyeFragmentShader samples it when `useDepth` is set).

 Design:
 - `depth(for:)` is fully synchronous and runs on the pump queue (never main):
   the frame isn't shown until its depth is ready, so image and depth always
   match exactly — slow inference yields fewer frames, never ghosting.
 - The returned texture is provider-owned (a stabilize ping-pong buffer, or a
   blitted copy when stabilization is unavailable), so it stays valid while the
   warp reads it even as the next inference overwrites Vision's output buffer.

 The model is loaded by name from the managed store (Application Support, via
 DepthModelStore) — any Depth Anything V2 variant, since the I/O is identical
 across sizes. Models get there via the in-app download (DepthModelManager) or by
 being dropped into Documents and imported on launch. If no model is present,
 `init?` returns nil and the warp falls back to the heuristic depth, so the app
 still builds and runs without the (large) model asset.
 */

#if canImport(Metal)
import Accelerate
import CoreMedia
import CoreML
import CoreVideo
import Metal
import os
import QuartzCore
import Vision
import VideoToolbox
import VergenceCore

public protocol DepthProvider: AnyObject {
    /// Synchronously infer + stabilize depth for this exact frame and return the
    /// matched depth map. Called on the pump queue so the frame isn't shown until
    /// its depth is ready — no lag/ghosting from reusing a stale map. Under slow
    /// inference the pump simply renders fewer frames rather than mismatching.
    func depth(for pixelBuffer: CVPixelBuffer) -> MTLTexture?
}

/// Everything the warp needs to use a depth map for one frame.
public struct PumpFrameDepth {
    public let texture: MTLTexture
    /// Letterbox UV transform (frame UV → depth content region).
    public let uvScale: SIMD2<Float>
    public let uvOffset: SIMD2<Float>
    /// Stored sample → display depth affine (identity for realtime inference;
    /// cached depth bakes its per-frame encode + display ranges here).
    public let valueScale: Float
    public let valueBias: Float
    /// How the mapped value becomes the warp's 0…1: `.legacy` clamps, a
    /// refined source (anchored, unbounded shot-space depth) uses the studio
    /// S-curve so near content plateaus instead of clipping.
    public let budget: DepthBudget
    /// What keeps `texture` valid while this value is held: a decoded
    /// cache frame's texture-cache view, or the pooled copy of an inferred
    /// map. The pump keeps a frame's depth for drawing it again while
    /// paused, after its source has moved on.
    let keepAlive: [Any]

    public init(
        texture: MTLTexture, uvScale: SIMD2<Float>, uvOffset: SIMD2<Float>,
        valueScale: Float, valueBias: Float, budget: DepthBudget = .legacy
    ) {
        self.init(texture: texture, uvScale: uvScale, uvOffset: uvOffset,
                  valueScale: valueScale, valueBias: valueBias, budget: budget, keepAlive: [])
    }

    init(
        texture: MTLTexture, uvScale: SIMD2<Float>, uvOffset: SIMD2<Float>,
        valueScale: Float, valueBias: Float, budget: DepthBudget, keepAlive: [Any]
    ) {
        self.texture = texture
        self.uvScale = uvScale
        self.uvOffset = uvOffset
        self.valueScale = valueScale
        self.valueBias = valueBias
        self.budget = budget
        self.keepAlive = keepAlive
    }
}

/// Depth for one frame that may still be being worked out on another thread
/// (`PumpDepthSource.requestDepth`). Frames that reuse a map share its
/// request. Thread-safe.
public final class PumpDepthRequest: @unchecked Sendable {
    private let lock = NSLock()
    private var resolved = false
    private var result: PumpFrameDepth?
    private var cancelled = false
    private var seconds = 0.0
    private var whenReady: (@Sendable () -> Void)?
    private var onCancel: (@Sendable () -> Void)?

    /// - Parameter whenReady: called once, on whatever thread resolves it.
    public init(whenReady: (@Sendable () -> Void)? = nil, onCancel: (@Sendable () -> Void)? = nil) {
        self.whenReady = whenReady
        self.onCancel = onCancel
    }

    /// Already resolved: a source that looks depth up on the spot.
    public convenience init(depth: PumpFrameDepth?) {
        self.init()
        resolve(depth, computeSeconds: 0)
    }

    public var isReady: Bool {
        lock.lock(); defer { lock.unlock() }
        return resolved
    }

    /// Nil until ready, and when there is none (an inference failure).
    public var depth: PumpFrameDepth? {
        lock.lock(); defer { lock.unlock() }
        return result
    }

    /// How long the map took to work out, once ready.
    public var computeSeconds: Double {
        lock.lock(); defer { lock.unlock() }
        return seconds
    }

    public var isCancelled: Bool {
        lock.lock(); defer { lock.unlock() }
        return cancelled
    }

    /// Nothing is waiting for it any more (the frames were thrown away after
    /// a seek); work not yet started is skipped.
    public func cancel() {
        lock.lock()
        cancelled = true
        whenReady = nil
        let release = onCancel
        onCancel = nil
        lock.unlock()
        release?()
    }

    public func resolve(_ depth: PumpFrameDepth?, computeSeconds: Double) {
        lock.lock()
        guard !resolved else { lock.unlock(); return }
        resolved = true
        result = cancelled ? nil : depth
        seconds = computeSeconds
        let notify = whenReady
        whenReady = nil
        lock.unlock()
        notify?()
    }
}

/// StereoPump's per-frame depth supplier. Implementations are called only on
/// the pump queue (including `invalidate`, from the stop-drain block).
public protocol PumpDepthSource: AnyObject {
    /// Depth for the frame with presentation time `itemTime`, or nil when none
    /// is available right now (inference failure / cache seek gap).
    func frameDepth(itemTime: CMTime, frame: CVPixelBuffer) -> PumpFrameDepth?
    /// True when depth is worked out on another thread, ahead of the frame
    /// being drawn (`requestDepth`): the pump asks as it takes each frame and
    /// keeps drawing earlier frames meanwhile. False: `frameDepth` is asked
    /// as each frame is drawn (a lookup, such as cached depth).
    var computesDepthAhead: Bool { get }
    /// Start working out `frame`'s depth, in the order frames are shown.
    /// Differences in `shownAt` represent wall seconds at the intended
    /// playback speed. A clocked reserve uses a media-derived reference so
    /// pauses do not alter depth reuse. `whenReady` runs on any thread.
    func requestDepth(for frame: PumpFrame, shownAt: CFTimeInterval,
                      whenReady: @escaping @Sendable () -> Void) -> PumpDepthRequest
    /// True when missing depth should render FLAT (with the pump's strength
    /// ramp back). Both real sources set this: falling back to the
    /// different-looking heuristic warp for a cache seek gap or an isolated
    /// realtime inference hiccup is a visible one-frame depth pop. False only
    /// makes sense for a source that is heuristic-adjacent to begin with.
    var flattensWhenUnavailable: Bool { get }
    /// True to warp with the dense mesh grid. Cached depth is edge-aware
    /// (joint-bilateral) + lookahead-smoothed, so it both tolerates and needs
    /// the finer grid (sharp depth edges alias the moderate grid into
    /// silhouette stairsteps). Realtime depth stays on the moderate grid — its
    /// residual high-frequency detail rendered as per-vertex wobble when the
    /// grid was densified.
    var prefersDenseWarpGrid: Bool { get }
    /// True when a new depth map costs an inference. Played faster than
    /// normal speed, the pump then shows no more frames a second than the
    /// video has at normal speed, evenly spaced, rather than as many as the
    /// inference can manage (`FrameRateCap`).
    var limitsToSourceFrameRate: Bool { get }
    /// Release decode/GPU resources; the pump is shutting down.
    func invalidate()
    /// Drop any held map so the next request computes it afresh, as if the
    /// frame were the first of a shot (a paused picture captured for the debug
    /// server, so a changed setting shows on it).
    func recomputeNextDepth()
}

extension PumpDepthSource {
    public var limitsToSourceFrameRate: Bool { false }
    public func recomputeNextDepth() {}
    public var computesDepthAhead: Bool { false }
    public func requestDepth(for frame: PumpFrame, shownAt: CFTimeInterval,
                             whenReady: @escaping @Sendable () -> Void) -> PumpDepthRequest {
        PumpDepthRequest(depth: frameDepth(itemTime: frame.itemTime, frame: frame.pixelBuffer))
    }
}

/// Which pipeline a depth model is being loaded for. The preference is split:
/// real-time inference gates every frame (fast model), while offline
/// pre-processing can afford a slower, higher-quality one.
public enum DepthModelRole: String, Sendable {
    case realtime
    case preprocess

    public var defaultsKey: String {
        switch self {
        case .realtime: return "realtimeDepthModelName"
        case .preprocess: return "preprocessDepthModelName"
        }
    }
}

/// Realtime source: Core ML inference on the frames themselves, on a queue of
/// its own, so the pump keeps drawing the frames before while the model runs
/// (it used to wait for each map, so inference, warp and transfer took turns
/// and a 60 fps video showed about 30 frames a second).
///
/// High-frame-rate video shows at full rate with depth inferred at about half
/// rate: a frame shown within 1/35 s of the frame the last map was inferred
/// for reuses that map (`HeldDepthPolicy`). That's a *bounded* ≤1-frame depth
/// age (same as cached playback of a two-pass conversion's first sweep), not
/// the unbounded staleness of an async-decoupled pump, and never across a cut:
/// a frame that doesn't look like the one the map is for gets its own.
/// ≤30fps content never holds: every frame spacing exceeds the interval, so
/// it stays fully lock-step.
///
/// The interval is in presentation (wall) time, so the inference rate doesn't
/// follow the playback speed: at 2× a 60 fps video shows 60 frames a second
/// (the pump's cap, `limitsToSourceFrameRate`), each two source frames on,
/// and still infers every other one, as at normal speed.
public final class RealtimeDepthSource: PumpDepthSource, @unchecked Sendable {
    private let provider: CoreMLDepthProvider
    /// Studio refinement (`DepthRefinement`), decided at creation: anchored
    /// shot-space depth, the depth budget, edge-aware filtering.
    private let refined: Bool
    /// The model and the refiner run here, one frame at a time in the order
    /// frames are shown (the refiner's temporal state depends on it). Nothing
    /// else touches `provider`.
    private let inferenceQueue = DispatchQueue(label: "com.illixion.vergence.depth-inference", qos: .userInteractive)
    /// The provider's output is a two-deep ring that the next inference but
    /// one overwrites, and with frames taken ahead a map can still be waiting
    /// to be drawn by then; each map is copied out.
    private let pool: DepthTexturePool?
    /// Pump queue only.
    private var policy = HeldDepthPolicy()
    /// Pump queue only: the request for the map later frames may reuse.
    private var keyRequest: PumpDepthRequest?
    private let stateLock = NSLock()
    private var invalidated = false

    public init?(device: MTLDevice) {
        guard let provider = CoreMLDepthProvider(device: device, role: .realtime) else { return nil }
        self.provider = provider
        self.refined = DepthRefinement.isEnabled && provider.supportsRefinement
        self.pool = DepthTexturePool(device: device)
    }

    /// An inference hiccup renders flat (+ ramp back) rather than swapping in
    /// the heuristic warp for one frame — a visible depth pop.
    public let flattensWhenUnavailable = true
    /// Refined depth is edge-snapped by the joint bilateral, the same reason
    /// cached depth takes the dense grid: on the moderate grid its sharp
    /// silhouettes would stairstep. Unrefined depth keeps the moderate grid
    /// (its residual detail rendered as per-vertex wobble on the dense one).
    public var prefersDenseWarpGrid: Bool { refined }
    public let limitsToSourceFrameRate = true
    public let computesDepthAhead = true

    public func requestDepth(for frame: PumpFrame, shownAt: CFTimeInterval,
                             whenReady: @escaping @Sendable () -> Void) -> PumpDepthRequest {
        let seconds = frame.itemTime.seconds
        // A map that failed or was thrown away has nothing to lend.
        if let key = keyRequest, key.isCancelled || (key.isReady && key.depth == nil) {
            keyRequest = nil
            policy.forget()
        }
        let signature = LumaSignature.sample(frame.pixelBuffer)
        if policy.decide(seconds: seconds, shownAt: shownAt, signature: signature) == .hold {
            if let keyRequest { return keyRequest }
            // The policy outlived its map; start over from this frame.
            policy.forget()
            _ = policy.decide(seconds: seconds, shownAt: shownAt, signature: signature)
        }
        // A canceled reserve must release its large source frames at once,
        // even while an earlier inference is still blocking the serial queue.
        let input = PendingDepthInput(frame)
        let request = PumpDepthRequest(whenReady: whenReady, onCancel: { input.clear() })
        keyRequest = request
        inferenceQueue.async { [self] in
            guard !request.isCancelled, !isInvalidated, let frame = input.take() else {
                input.clear()
                request.resolve(nil, computeSeconds: 0)
                return
            }
            let start = CACurrentMediaTime()
            let depth = computeDepth(for: frame.pixelBuffer, seconds: seconds)
            request.resolve(depth, computeSeconds: CACurrentMediaTime() - start)
        }
        return request
    }

    /// Synchronous, for a caller that wants one map now; still on the
    /// inference queue, which owns the provider.
    public func frameDepth(itemTime: CMTime, frame: CVPixelBuffer) -> PumpFrameDepth? {
        inferenceQueue.sync { computeDepth(for: frame, seconds: itemTime.seconds) }
    }

    /// Inference queue only.
    private func computeDepth(for frame: CVPixelBuffer, seconds: Double) -> PumpFrameDepth? {
        let inferred = refined
            ? provider.refinedDepth(for: frame, frameSeconds: seconds)
            : provider.depth(for: frame)
        guard let inferred, let owned = pool?.copy(inferred) else { return nil }
        let uv = CoreMLDepthProvider.letterboxUVTransform(
            videoWidth: CVPixelBufferGetWidth(frame), videoHeight: CVPixelBufferGetHeight(frame),
            depthWidth: owned.texture.width, depthHeight: owned.texture.height
        )
        return PumpFrameDepth(
            texture: owned.texture, uvScale: uv.scale, uvOffset: uv.offset,
            valueScale: 1, valueBias: 0, budget: refined ? .studio : .legacy, keepAlive: [owned]
        )
    }

    private var isInvalidated: Bool {
        stateLock.lock(); defer { stateLock.unlock() }
        return invalidated
    }

    public func invalidate() {
        stateLock.lock()
        invalidated = true
        stateLock.unlock()
        keyRequest?.cancel()
        keyRequest = nil
    }

    public func recomputeNextDepth() {
        keyRequest = nil
        policy.forget()
        inferenceQueue.async { [provider] in provider.resetRefinement() }
    }
}

/// The queued inference closure retains this cell, not the full picture.
/// Canceling a seek's queued work clears the picture before that closure runs.
final class PendingDepthInput: @unchecked Sendable {
    private let lock = NSLock()
    private var frame: PumpFrame?
    init(_ frame: PumpFrame) { self.frame = frame }
    func take() -> PumpFrame? {
        lock.withLock {
            defer { frame = nil }
            return frame
        }
    }
    func clear() { lock.withLock { frame = nil } }
}

/// Textures inferred maps are copied into, handed back when the last frame
/// drawn with one lets go of it (`PooledDepthTexture`). The pump only lets go
/// after the warp that read it has completed.
final class DepthTexturePool: @unchecked Sendable {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let lock = NSLock()
    private var free: [MTLTexture] = []
    /// More than a pump at 60 fps has in flight; extras are let go.
    private static let maxFree = 8

    init?(device: MTLDevice) {
        guard let commandQueue = device.makeCommandQueue() else { return nil }
        self.device = device
        self.commandQueue = commandQueue
    }

    /// A copy of `source` that lives as long as the returned object.
    func copy(_ source: MTLTexture) -> PooledDepthTexture? {
        guard let texture = take(matching: source),
              let cmd = commandQueue.makeCommandBuffer(),
              let blit = cmd.makeBlitCommandEncoder() else { return nil }
        blit.copy(from: source, to: texture)
        blit.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        guard cmd.status == .completed else { giveBack(texture); return nil }
        return PooledDepthTexture(texture: texture, pool: self)
    }

    private func take(matching source: MTLTexture) -> MTLTexture? {
        lock.lock()
        if let index = free.firstIndex(where: {
            $0.width == source.width && $0.height == source.height && $0.pixelFormat == source.pixelFormat
        }) {
            let texture = free.remove(at: index)
            lock.unlock()
            return texture
        }
        lock.unlock()
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: source.pixelFormat, width: source.width, height: source.height, mipmapped: false
        )
        desc.usage = [.shaderRead]
        desc.storageMode = .private
        return device.makeTexture(descriptor: desc)
    }

    fileprivate func giveBack(_ texture: MTLTexture) {
        lock.lock(); defer { lock.unlock() }
        if free.count < Self.maxFree { free.append(texture) }
    }
}

final class PooledDepthTexture: @unchecked Sendable {
    let texture: MTLTexture
    private let pool: DepthTexturePool

    init(texture: MTLTexture, pool: DepthTexturePool) {
        self.texture = texture
        self.pool = pool
    }

    deinit { pool.giveBack(texture) }
}

/// CPU mirror of the Metal `DepthStabilizeParams` struct (int, 3 floats, uint,
/// float). Internal (not private): the offline `DepthConverter` reuses the
/// blur/bilateral kernels.
struct DepthStabilizeParams {
    var blurRadius: Int32
    var blurSigma: Float
    var baseAlpha: Float
    var motionGain: Float
    var hasPrev: UInt32
    /// Joint-bilateral luma sigma; unused by the plain gaussian kernels.
    var sigmaLuma: Float = 0
}

public final class CoreMLDepthProvider: DepthProvider, @unchecked Sendable {
    private let request: VNCoreMLRequest
    /// Keeps the shared model alive for as long as this provider runs.
    private let loadedModel: LoadedDepthModel
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private var textureCache: CVMetalTextureCache?
    /// Base name of the loaded model (no extension) — identifies which model
    /// produced a depth cache entry.
    public let modelName: String

    // Depth stabilization: a spatial gaussian low-pass band-limits the depth so
    // the coarser warp grid doesn't alias its detail into rippling waves, plus a
    // motion-adaptive temporal EMA that damps frame-to-frame jitter without
    // ghosting on motion. Runs at the model's native depth resolution as two
    // separable 1D passes (H → scratch → V+EMA) in one command buffer.
    private let blurHPipeline: MTLComputePipelineState?
    private let blurVEMAPipeline: MTLComputePipelineState?
    private var stabilizeAvailable: Bool { blurHPipeline != nil && blurVEMAPipeline != nil }
    private let stabilizeEnabled = true
    /// Spatial low-pass radius/sigma (depth texels). Band-limits depth for the
    /// 193-wide warp grid (which samples ~every 2.7 texels of the 518-wide map).
    private let blurRadius: Int32 = 6
    private let blurSigma: Float = 3.0
    /// Temporal: blend toward new map where stable (lower = smoother) and how
    /// fast it trusts the new map as depth changes (avoids ghosting on motion).
    private let emaBaseAlpha: Float = 0.35
    private let emaMotionGain: Float = 4.0
    private var emaTexA: MTLTexture?
    private var emaTexB: MTLTexture?
    /// Intermediate target for the horizontal blur pass.
    private var blurScratchTex: MTLTexture?
    private var emaWidth = 0
    private var emaHeight = 0
    private var writeA = true
    private var hasPrevDepth = false

    private let signposter = VergenceLog.signposter

    // The model's input, prepared on the GPU. Vision would otherwise scale
    // every full-size frame down itself and, for an HDR frame (PQ or HLG,
    // BT.2020), colour-convert it for the model on the CPU first: on a
    // Vision Pro that put depth at 40-46 ms a frame for a 1080p HDR film
    // against 32-34 ms for SDR, past the 41.7 ms a 24 fps frame has (device,
    // 2026-10-10). A pixel transfer scales the frame to fit the model's input
    // (same aspect, so Vision's letterbox and `letterboxUVTransform` are
    // unchanged) and converts it to BT.709, so Vision gets a small SDR image.
    /// The model's input image size, if it declares one.
    private let modelInputSize: (width: Int, height: Int)?
    private var inputTransfer: VTPixelTransferSession?
    private var inputPool: CVPixelBufferPool?
    private var inputPoolSize = (width: 0, height: 0)
    private var loggedInputPreparation = false

    /// Studio refinement chain (anchoring, locked scale/shift, edge-aware
    /// filtering), built on first use. See `RealtimeDepthRefiner`.
    private lazy var refiner: RealtimeDepthRefiner? = textureCache.flatMap {
        RealtimeDepthRefiner(device: device, commandQueue: commandQueue, textureCache: $0)
    }

    /// Whether `refinedDepth` can run (its kernels loaded).
    public var supportsRefinement: Bool { refiner != nil }

    /// The last refined frame's mapping, whether it was anchored, whether it
    /// was a cut (tests and the measurement harness).
    var refinementDiagnostics: (mapping: DepthAffine, anchored: Bool, cut: Bool)? {
        refiner.map { ($0.lastMapping, $0.lastAnchored, $0.lastCut) }
    }

    public convenience init?(device: MTLDevice, role: DepthModelRole = .realtime) {
        guard let modelURL = Self.findModelURL(role: role) else { return nil }
        self.init(device: device, modelURL: modelURL)
    }

    /// Load a specific model file, bypassing the role preference. Internal:
    /// `DepthPipelineBenchmark` benchmarks every installed model in turn, which the
    /// preference-driven initializer cannot express.
    init?(device: MTLDevice, modelURL: URL) {
        guard let compiledURL = Self.compiledModelURL(for: modelURL),
              let queue = device.makeCommandQueue(),
              // Shared with the warmer and other providers: already loaded
              // (and warm) when something expressed intent to use it.
              let loaded = DepthModelCache.shared.model(for: compiledURL) else { return nil }
        self.loadedModel = loaded
        modelInputSize = loaded.inputSize
        let request = VNCoreMLRequest(model: loaded.vnModel)
        // The model fixes its own (square) input size; Vision letterboxes the
        // frame into it, preserving aspect. `.scaleFill` stretched non-square
        // video, distorting depth — the warp remaps UVs into the content region
        // via VideoStereoUniforms.depthUVScale/Offset (see depthUVTransform).
        request.imageCropAndScaleOption = .scaleFit

        self.request = request
        self.device = device
        self.commandQueue = queue
        self.modelName = modelURL.deletingPathExtension().lastPathComponent
        self.blurHPipeline = Self.makeComputePipeline("depthBlurH")
        self.blurVEMAPipeline = Self.makeComputePipeline("depthBlurVEMA")
        CVMetalTextureCacheCreate(nil, nil, device, nil, &textureCache)
        VergenceLog.pipeline.info("CoreMLDepthProvider loaded model: \(modelURL.lastPathComponent, privacy: .public), stabilize=\(self.stabilizeAvailable)")
    }

    /// The kernels live in the package's own library, not the app's — a bare
    /// `device.makeDefaultLibrary()` here would search the host bundle and find
    /// nothing, silently disabling stabilization.
    private static func makeComputePipeline(_ function: String) -> MTLComputePipelineState? {
        VergenceMetal.shared?.computePipeline(function)
    }

    /// Resolve the depth model to load for a role. Fake-3D requires a real
    /// model (there is no heuristic fallback): use the role's explicit
    /// preference if present (falling back to the legacy single-preference key
    /// for pre-split installs), else the first installed model, else nil (no
    /// model → the engine declines to run fake-3D rather than showing a
    /// heuristic warp). `.mlmodelc` loads directly; `.mlpackage` is compiled +
    /// cached by `compiledModelURL`.
    private static func findModelURL(role: DepthModelRole) -> URL? {
        let preferred = UserDefaults.standard.string(forKey: role.defaultsKey)
            ?? UserDefaults.standard.string(forKey: "preferredDepthModelName")
            ?? ""
        if !preferred.isEmpty {
            let fm = FileManager.default
            for dir in modelSearchDirectories() {
                for ext in ["mlmodelc", "mlpackage"] {
                    let url = dir.appendingPathComponent(preferred).appendingPathExtension(ext)
                    if fm.fileExists(atPath: url.path) { return url }
                }
            }
        }
        return DepthModelStore.installedModelURLs().first
    }

    /// The model file `init?` would load for `role` (before compiling).
    static func modelURL(role: DepthModelRole) -> URL? { findModelURL(role: role) }

    /// Whether a depth model is available to load for `role`. Fake-3D needs
    /// one — the engine uses this to decline (fall back to 2D) rather than
    /// warp heuristically.
    public static func hasAvailableModel(role: DepthModelRole) -> Bool { findModelURL(role: role) != nil }

    /// Base name of the model `init?` would load for `role`, without loading
    /// it. Used to key depth cache lookups before spinning up a provider.
    public static func resolvedModelName(role: DepthModelRole) -> String? {
        findModelURL(role: role)?.deletingPathExtension().lastPathComponent
    }

    /// Where `.scaleFit` anchors the frame inside the model's input. Vision
    /// doesn't document the padding placement; 0.5 = centered. If on-device
    /// parallax reads vertically offset on non-square video, flip y to 0 (top)
    /// or 1 (bottom).
    private static let letterboxAnchor = SIMD2<Float>(0.5, 0.5)

    /// UV transform mapping frame UV → the content region of a `.scaleFit`
    /// letterboxed depth map: sample depth at `uv * scale + offset`. Identity
    /// when the aspects already match. Shared by the realtime warp and the
    /// offline depth converter so cached and live depth agree.
    public static func letterboxUVTransform(
        videoWidth: Int, videoHeight: Int, depthWidth: Int, depthHeight: Int
    ) -> (scale: SIMD2<Float>, offset: SIMD2<Float>) {
        guard videoWidth > 0, videoHeight > 0, depthWidth > 0, depthHeight > 0 else {
            return (SIMD2(1, 1), SIMD2(0, 0))
        }
        let vw = Float(videoWidth), vh = Float(videoHeight)
        let dw = Float(depthWidth), dh = Float(depthHeight)
        let s = min(dw / vw, dh / vh)
        let content = SIMD2(vw * s / dw, vh * s / dh)
        return (content, (SIMD2(1, 1) - content) * letterboxAnchor)
    }

    /// The managed store, then the app bundle. Documents is intentionally NOT
    /// searched: it's only a drop-off inbox, drained into the store on launch
    /// (`DepthModelStore.importInboxModels`) and when Settings opens. Keeping the
    /// store authoritative means a model can never load yet be invisible to the
    /// Settings picker / undeletable — a model pushed to Documents loads after
    /// the next launch, once imported into the store.
    private static func modelSearchDirectories() -> [URL] {
        [DepthModelStore.modelsDirectory, Bundle.main.bundleURL]
    }

    /// Returns a loadable compiled-model URL. `.mlmodelc` is used directly;
    /// `.mlpackage` is compiled and cached under Application Support (recompiled
    /// only when the source is newer), so dropping an `.mlpackage` into Documents
    /// to swap models works without a rebuild.
    static func compiledModelURL(for url: URL) -> URL? {
        if url.pathExtension == "mlmodelc" { return url }

        let fm = FileManager.default
        let cacheDir = DepthModelStore.compiledCacheDirectory
        let cached = cacheDir
            .appendingPathComponent(url.deletingPathExtension().lastPathComponent)
            .appendingPathExtension("mlmodelc")

        if fm.fileExists(atPath: cached.path),
           let cachedDate = (try? fm.attributesOfItem(atPath: cached.path))?[.modificationDate] as? Date,
           let sourceDate = (try? fm.attributesOfItem(atPath: url.path))?[.modificationDate] as? Date,
           cachedDate >= sourceDate {
            return cached
        }

        guard let compiled = try? MLModel.compileModel(at: url) else { return nil }
        try? fm.removeItem(at: cached)
        if (try? fm.copyItem(at: compiled, to: cached)) != nil {
            return cached
        }
        return compiled
    }

    /// Run the Vision request and return the first observation. One request
    /// instance per provider — callers must not run concurrently (the realtime
    /// pump and the offline converter each own their own provider).
    private func performInference(on pixelBuffer: CVPixelBuffer) -> VNObservation? {
        let input = preparedInput(pixelBuffer) ?? pixelBuffer
        let handler = VNImageRequestHandler(cvPixelBuffer: input, orientation: .up, options: [:])
        try? handler.perform([request])
        return request.results?.first
    }

    /// `pixelBuffer` scaled on the GPU to fit the model's input and
    /// converted to BT.709; nil (use the frame as is) when it is already no
    /// larger than the input or the transfer isn't available.
    private func preparedInput(_ pixelBuffer: CVPixelBuffer) -> CVPixelBuffer? {
        guard let modelInputSize else { return nil }
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        guard width > 0, height > 0 else { return nil }
        let scale = min(Double(modelInputSize.width) / Double(width), Double(modelInputSize.height) / Double(height))
        let transfer = CVBufferCopyAttachment(pixelBuffer, kCVImageBufferTransferFunctionKey, nil) as? String
        let isHDR = transfer == (kCVImageBufferTransferFunction_SMPTE_ST_2084_PQ as String)
            || transfer == (kCVImageBufferTransferFunction_ITU_R_2100_HLG as String)
        guard scale < 1 || isHDR else { return nil }
        // Even sizes, the frame's aspect.
        let target = (width: max(2, Int((Double(width) * min(scale, 1) / 2).rounded()) * 2),
                      height: max(2, Int((Double(height) * min(scale, 1) / 2).rounded()) * 2))
        if inputTransfer == nil {
            var session: VTPixelTransferSession?
            guard VTPixelTransferSessionCreate(allocator: nil, pixelTransferSessionOut: &session) == noErr,
                  let session else { return nil }
            VTSessionSetProperty(session, key: kVTPixelTransferPropertyKey_DestinationColorPrimaries,
                                 value: kCVImageBufferColorPrimaries_ITU_R_709_2)
            VTSessionSetProperty(session, key: kVTPixelTransferPropertyKey_DestinationTransferFunction,
                                 value: kCVImageBufferTransferFunction_ITU_R_709_2)
            VTSessionSetProperty(session, key: kVTPixelTransferPropertyKey_DestinationYCbCrMatrix,
                                 value: kCVImageBufferYCbCrMatrix_ITU_R_709_2)
            inputTransfer = session
        }
        if inputPool == nil || inputPoolSize != target {
            let attributes: [String: Any] = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: target.width,
                kCVPixelBufferHeightKey as String: target.height,
                kCVPixelBufferIOSurfacePropertiesKey as String: [:],
                kCVPixelBufferMetalCompatibilityKey as String: true,
            ]
            var pool: CVPixelBufferPool?
            CVPixelBufferPoolCreate(nil, nil, attributes as CFDictionary, &pool)
            inputPool = pool
            inputPoolSize = target
        }
        guard let inputTransfer, let inputPool else { return nil }
        var output: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(nil, inputPool, &output) == kCVReturnSuccess, let output,
              VTPixelTransferSessionTransferImage(inputTransfer, from: pixelBuffer, to: output) == noErr else { return nil }
        // Said outright, so Vision converts nothing.
        CVBufferSetAttachment(output, kCVImageBufferColorPrimariesKey, kCVImageBufferColorPrimaries_ITU_R_709_2, .shouldPropagate)
        CVBufferSetAttachment(output, kCVImageBufferTransferFunctionKey, kCVImageBufferTransferFunction_ITU_R_709_2, .shouldPropagate)
        CVBufferSetAttachment(output, kCVImageBufferYCbCrMatrixKey, kCVImageBufferYCbCrMatrix_ITU_R_709_2, .shouldPropagate)
        if !loggedInputPreparation {
            loggedInputPreparation = true
            VergenceLog.pipeline.info("Depth input prepared on the GPU: \(width)×\(height)\(isHDR ? " HDR" : "", privacy: .public) → \(target.width)×\(target.height) BT.709 for a \(modelInputSize.width)×\(modelInputSize.height) model")
        }
        return output
    }

    /// Raw model output for one frame: unstabilized and unnormalized (values in
    /// the model's own arbitrary per-frame scale). For the offline converter,
    /// which does its own normalization and temporal smoothing. The texture may
    /// be a transient view — valid only while `keepAlive` is retained.
    func inferRawDepth(from pixelBuffer: CVPixelBuffer) -> (texture: MTLTexture, keepAlive: [Any])? {
        let inferState = signposter.beginInterval("depth-inference")
        defer { signposter.endInterval("depth-inference", inferState) }
        switch performInference(on: pixelBuffer) {
        case let obs as VNPixelBufferObservation:
            return transientTexture(from: obs.pixelBuffer)
        case let obs as VNCoreMLFeatureValueObservation:
            guard let array = obs.featureValue.multiArrayValue,
                  let texture = rawTexture(from: array) else { return nil }
            return (texture, [])
        default:
            return nil
        }
    }

    /// Depth for one still image, read back to the CPU (see `StillDepthMap`).
    /// Leaves the video stabilization state alone: that EMA would blend one
    /// image's depth into the next. Same threading rule as every other entry
    /// point — one caller at a time per provider.
    public func stillDepth(for image: CGImage, preservePrecision: Bool = false) -> StillDepthMap? {
        let inferState = signposter.beginInterval("depth-inference-still")
        defer { signposter.endInterval("depth-inference-still", inferState) }
        let handler = VNImageRequestHandler(cgImage: image, orientation: .up, options: [:])
        try? handler.perform([request])
        let plane: (values: [Float], width: Int, height: Int)?
        switch request.results?.first {
        case let obs as VNPixelBufferObservation:
            plane = StillDepthExtraction.floats(from: obs.pixelBuffer)
        case let obs as VNCoreMLFeatureValueObservation:
            plane = obs.featureValue.multiArrayValue.flatMap { StillDepthExtraction.floats(from: $0) }
        default:
            plane = nil
        }
        guard let plane else { return nil }
        return StillDepthExtraction.normalizedMap(plane, imageWidth: image.width, imageHeight: image.height, preservePrecision: preservePrecision)
    }

    public func depth(for pixelBuffer: CVPixelBuffer) -> MTLTexture? {
        let inferState = signposter.beginInterval("depth-inference")

        // When stabilization will run, the raw texture only needs to survive the
        // stabilize pass's synchronous wait (same call stack), so the pixel-buffer
        // path can use a zero-copy texture-cache view with keep-alives instead of
        // an owned blit copy. Without stabilization the raw texture IS the result
        // and outlives this call (the warp reads it later), so it must be copied.
        let raw: MTLTexture?
        var keepAlive: [Any] = []
        switch performInference(on: pixelBuffer) {
        case let obs as VNPixelBufferObservation:
            if stabilizeEnabled, stabilizeAvailable,
               let transient = transientTexture(from: obs.pixelBuffer) {
                raw = transient.texture
                keepAlive = transient.keepAlive
            } else {
                raw = copiedTexture(from: obs.pixelBuffer)
            }
        case let obs as VNCoreMLFeatureValueObservation:
            raw = obs.featureValue.multiArrayValue.flatMap { makeTexture(from: $0) }
        default:
            raw = nil
        }
        signposter.endInterval("depth-inference", inferState)
        guard let raw else { return nil }

        // Band-limit + temporally smooth to remove ripple (spatial aliasing) and
        // wobble (temporal jitter). Falls back to the raw map if unavailable.
        guard stabilizeEnabled else { return raw }
        let stabilizeState = signposter.beginInterval("depth-stabilize")
        defer { signposter.endInterval("depth-stabilize", stabilizeState) }
        let stabilized = stabilize(rawDepth: raw)
        withExtendedLifetime(keepAlive) {}
        // If stabilize failed, only the owned-copy path may be returned raw; a
        // transient view would dangle once Vision recycles its buffer.
        if stabilized == nil, !keepAlive.isEmpty { return nil }
        return stabilized ?? raw
    }

    /// Refined realtime depth for one frame: unbounded shot-space depth (near
    /// ≈ 1 on the frame's own range while the lock holds), to be shown through
    /// `DepthBudget.studio`. `frameSeconds` is the frame's media time; a jump
    /// back or of more than half a second restarts the lock. Nil on failure
    /// (the pump renders that frame flat). Same threading rule as `depth(for:)`.
    /// The next refined frame starts a new shot (no temporal blend, mapping
    /// and gap curve from that frame alone).
    public func resetRefinement() {
        refiner?.reset()
    }

    public func refinedDepth(for pixelBuffer: CVPixelBuffer, frameSeconds: Double) -> MTLTexture? {
        guard let refiner else { return depth(for: pixelBuffer) }
        let inferState = signposter.beginInterval("depth-inference")
        let raw: MTLTexture?
        var keepAlive: [Any] = []
        switch performInference(on: pixelBuffer) {
        case let obs as VNPixelBufferObservation:
            if let transient = transientTexture(from: obs.pixelBuffer) {
                raw = transient.texture
                keepAlive = transient.keepAlive
            } else {
                raw = nil
            }
        case let obs as VNCoreMLFeatureValueObservation:
            raw = obs.featureValue.multiArrayValue.flatMap { makeTexture(from: $0) }
        default:
            raw = nil
        }
        signposter.endInterval("depth-inference", inferState)
        guard let raw else { return nil }
        StackHeadroom.check("CoreMLDepthProvider.refinedDepth")
        let refined = refiner.refine(raw: raw, frame: pixelBuffer, frameSeconds: frameSeconds)
        withExtendedLifetime(keepAlive) {}
        return refined
    }

    // MARK: - Depth stabilization

    /// Spatially band-limit + motion-adaptively temporally smooth `rawDepth`
    /// (same resolution). Returns a private r16Float texture from the ping-pong
    /// pair; safe to publish because the next call writes the *other* buffer,
    /// never the one the warp may currently be reading.
    private func stabilize(rawDepth: MTLTexture) -> MTLTexture? {
        guard let blurHPipeline, let blurVEMAPipeline else { return nil }
        let w = rawDepth.width, h = rawDepth.height
        guard w > 0, h > 0,
              ensureEMATextures(width: w, height: h),
              let scratch = blurScratchTex,
              let prev = writeA ? emaTexB : emaTexA,
              let next = writeA ? emaTexA : emaTexB,
              let cmd = commandQueue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder() else { return nil }

        var params = DepthStabilizeParams(
            blurRadius: blurRadius,
            blurSigma: blurSigma,
            baseAlpha: emaBaseAlpha,
            motionGain: emaMotionGain,
            hasPrev: hasPrevDepth ? 1 : 0
        )
        let tg = MTLSize(width: 8, height: 8, depth: 1)
        let groups = MTLSize(width: (w + 7) / 8, height: (h + 7) / 8, depth: 1)

        // Separable gaussian: H pass into the scratch, then V pass + EMA. Both
        // textures are hazard-tracked, so Metal orders the two dispatches.
        enc.setComputePipelineState(blurHPipeline)
        enc.setTexture(rawDepth, index: 0)
        enc.setTexture(scratch, index: 1)
        enc.setBytes(&params, length: MemoryLayout<DepthStabilizeParams>.stride, index: 0)
        enc.dispatchThreadgroups(groups, threadsPerThreadgroup: tg)

        enc.setComputePipelineState(blurVEMAPipeline)
        enc.setTexture(scratch, index: 0)
        enc.setTexture(prev, index: 1)
        enc.setTexture(next, index: 2)
        enc.setBytes(&params, length: MemoryLayout<DepthStabilizeParams>.stride, index: 0)
        enc.dispatchThreadgroups(groups, threadsPerThreadgroup: tg)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        writeA.toggle()
        hasPrevDepth = true
        return next
    }

    private func ensureEMATextures(width: Int, height: Int) -> Bool {
        if emaTexA != nil, emaTexB != nil, blurScratchTex != nil, emaWidth == width, emaHeight == height {
            return true
        }
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r16Float, width: width, height: height, mipmapped: false
        )
        desc.usage = [.shaderRead, .shaderWrite]
        desc.storageMode = .private
        guard let a = device.makeTexture(descriptor: desc),
              let b = device.makeTexture(descriptor: desc),
              let scratch = device.makeTexture(descriptor: desc) else { return false }
        emaTexA = a
        emaTexB = b
        blurScratchTex = scratch
        emaWidth = width
        emaHeight = height
        writeA = true
        hasPrevDepth = false
        return true
    }

    // MARK: - Output → texture

    private static func depthPixelFormat(for pixelBuffer: CVPixelBuffer) -> MTLPixelFormat {
        switch CVPixelBufferGetPixelFormatType(pixelBuffer) {
        case kCVPixelFormatType_OneComponent8:
            return .r8Unorm
        case kCVPixelFormatType_OneComponent32Float, kCVPixelFormatType_DepthFloat32:
            return .r32Float
        default: // OneComponent16Half / DepthFloat16 and friends
            return .r16Float
        }
    }

    /// Image output, zero-copy: a texture-cache view of Vision's buffer, valid
    /// only while `keepAlive` is retained (until the stabilize pass's synchronous
    /// wait returns, on the same call stack). Never return this texture itself.
    private func transientTexture(from pixelBuffer: CVPixelBuffer) -> (texture: MTLTexture, keepAlive: [Any])? {
        guard let cache = textureCache else { return nil }
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        var cvTexture: CVMetalTexture?
        guard CVMetalTextureCacheCreateTextureFromImage(
            nil, cache, pixelBuffer, nil, Self.depthPixelFormat(for: pixelBuffer), width, height, 0, &cvTexture
        ) == kCVReturnSuccess, let cvTexture, let texture = CVMetalTextureGetTexture(cvTexture) else { return nil }
        return (texture, [cvTexture, pixelBuffer])
    }

    /// Image output, owned copy: import via the texture cache, then blit into a
    /// private texture the provider owns so it stays valid after Vision recycles
    /// its buffer. Fallback for when stabilization can't run (the raw texture is
    /// then returned to the warp directly and must outlive this call).
    private func copiedTexture(from pixelBuffer: CVPixelBuffer) -> MTLTexture? {
        guard let (transient, keepAlive) = transientTexture(from: pixelBuffer) else { return nil }
        let width = transient.width
        let height = transient.height
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: transient.pixelFormat, width: width, height: height, mipmapped: false
        )
        desc.usage = [.shaderRead]
        desc.storageMode = .private
        guard let dest = device.makeTexture(descriptor: desc),
              let cmdBuf = commandQueue.makeCommandBuffer(),
              let blit = cmdBuf.makeBlitCommandEncoder() else { return nil }
        blit.copy(from: transient, sourceSlice: 0, sourceLevel: 0,
                  sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                  sourceSize: MTLSize(width: width, height: height, depth: 1),
                  to: dest, destinationSlice: 0, destinationLevel: 0,
                  destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
        blit.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        withExtendedLifetime(keepAlive) {}
        return dest
    }

    /// MLMultiArray output: normalize to 0-1 and upload to an r16Float texture.
    /// Assumes the last two dimensions are height × width.
    private func makeTexture(from array: MLMultiArray) -> MTLTexture? {
        guard let plane = StillDepthExtraction.floats(from: array) else { return nil }
        let width = plane.width, height = plane.height
        let count = width * height
        var values = plane.values

        var minV: Float = 0
        var maxV: Float = 0
        values.withUnsafeBufferPointer { buffer in
            vDSP_minv(buffer.baseAddress!, 1, &minV, vDSP_Length(count))
            vDSP_maxv(buffer.baseAddress!, 1, &maxV, vDSP_Length(count))
        }
        let range = max(maxV - minV, 1e-5)
        var scale = 1 / range
        var bias = -minV / range
        values.withUnsafeMutableBufferPointer { buffer in
            vDSP_vsmsa(
                buffer.baseAddress!, 1, &scale, &bias,
                buffer.baseAddress!, 1, vDSP_Length(count)
            )
        }

        var halfValues = [UInt16](repeating: 0, count: count)
        values.withUnsafeMutableBytes { sourceBytes in
            halfValues.withUnsafeMutableBytes { destBytes in
                var source = vImage_Buffer(
                    data: sourceBytes.baseAddress,
                    height: 1,
                    width: vImagePixelCount(count),
                    rowBytes: count * MemoryLayout<Float>.stride
                )
                var dest = vImage_Buffer(
                    data: destBytes.baseAddress,
                    height: 1,
                    width: vImagePixelCount(count),
                    rowBytes: count * MemoryLayout<UInt16>.stride
                )
                vImageConvert_PlanarFtoPlanar16F(&source, &dest, 0)
            }
        }

        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r16Float, width: width, height: height, mipmapped: false
        )
        desc.usage = [.shaderRead]
        guard let texture = device.makeTexture(descriptor: desc) else { return nil }
        halfValues.withUnsafeBytes { bytes in
            texture.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                            withBytes: bytes.baseAddress!, bytesPerRow: width * 2)
        }
        return texture
    }

    /// MLMultiArray output, raw: upload the model's values verbatim (no min/max
    /// normalization — the offline converter normalizes with its own temporally
    /// smoothed range). float16 → r16Float memcpy; float32/double → r32Float.
    private func rawTexture(from array: MLMultiArray) -> MTLTexture? {
        let shape = array.shape.map(\.intValue)
        guard shape.count >= 2 else { return nil }
        let height = shape[shape.count - 2]
        let width = shape[shape.count - 1]
        guard width > 0, height > 0 else { return nil }

        func makeTexture(_ format: MTLPixelFormat, bytes: UnsafeRawPointer, bytesPerRow: Int) -> MTLTexture? {
            let desc = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: format, width: width, height: height, mipmapped: false
            )
            desc.usage = [.shaderRead]
            guard let texture = device.makeTexture(descriptor: desc) else { return nil }
            texture.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                            withBytes: bytes, bytesPerRow: bytesPerRow)
            return texture
        }

        let strides = array.strides.map(\.intValue)
        guard strides.count == shape.count else { return nil }
        let rowStride = strides[strides.count - 2], columnStride = strides[strides.count - 1]
        guard rowStride > 0, columnStride > 0 else { return nil }
        if columnStride == 1 {
            switch array.dataType {
            case .float16:
                return makeTexture(.r16Float, bytes: array.dataPointer, bytesPerRow: rowStride * 2)
            case .float32:
                return makeTexture(.r32Float, bytes: array.dataPointer, bytesPerRow: rowStride * 4)
            default: break
            }
        }
        guard let plane = StillDepthExtraction.floats(from: array) else { return nil }
        return plane.values.withUnsafeBytes { bytes in
            makeTexture(.r32Float, bytes: bytes.baseAddress!, bytesPerRow: width * 4)
        }
    }
}
#endif
