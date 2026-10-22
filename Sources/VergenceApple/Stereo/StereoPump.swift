/*
 Vergence - StereoPump.

 Owns all per-frame GPU work for windowed fake-3D: pull the newest decoded
 frame, get depth for *that exact frame*, warp it into left/right eye buffers,
 convert to the compositor's native 420f, tag, and enqueue.

 CRITICAL: everything here runs on a dedicated background queue with its own
 MTLCommandQueue — NEVER on the main thread. An earlier version pumped on the
 main actor and blocked it with waitUntilCompleted ~90x/s, which starved Core
 Animation commits (backboardd render-watchdog SIGKILL) and the main-queue
 AVPlayer prepare callbacks (mediaplaybackd async-prepare watchdog), taking
 down the whole compositor.

 A packed stereo video (side by side, over/under) takes the same road with the
 warp swapped for a crop: each eye is cut out of the decoded frame
 (`PackedStereoLayout`), so real 3D plays straight from the decoder with
 no depth model and no MV-HEVC conversion. Everything after the eye targets is
 shared, which is the point: the format, tags and pacing the compositor accepts
 were found on device once, for both.

 Frames arrive through `PumpFrameSource`, not from an AVPlayer directly. That
 seam is the point of this file living in a package: `AVPlayerFrameSource`
 (below) pulls from an `AVPlayerItemVideoOutput` exactly as Spatial Stash
 always has, while a browser can push decoded frames in from a WKWebView with
 no AVPlayer anywhere in the picture.

 Working ahead. A clocked source (`LookaheadFrameSource`) hands out frames
 before they are due, so the pump takes each one `lead` early, in turn, and
 queues the finished eyes for the host time the frame is due: the renderer
 shows it on time, in step with the audio. It used to take the frame due *now*
 and show it once its depth and warp were done, a tick (~33 ms in real time)
 behind the sound. Real-time depth is inferred on its own queue
 (`RealtimeDepthSource`) while the pump draws the frames before, so the model,
 the warp and the transfer overlap instead of taking turns; at 48-60 fps every
 other frame reuses the map before it (≤1 frame of depth age, never across a
 cut). Pauses, stalls, speed changes and seeks are read off the clock itself:
 frames queued for a time the clock no longer keeps are flushed and queued
 again, and frames taken for a position the playhead left are dropped. A push
 source has no clock: each frame shows as soon as it is drawn, or with depth
 worked out ahead a fixed `pushDelay` after it came, so they stay evenly
 spaced.

 visionOS 26 APIs: CMReadySampleBuffer / CMTaggedDynamicBuffer / CVReadOnlyPixelBuffer.
 */

#if os(visionOS)

import AVFoundation
import CoreMedia
import CoreVideo
import Metal
import os
import QuartzCore
import VideoToolbox
import VergenceCore

/// CPU mirror of the Metal `VideoStereoUniforms` struct (8 floats, 2 float2s at
/// offsets 32/40 matching Metal's float2 alignment, then 2 more floats).
/// CPU mirror of the Metal `PackedEyeUniforms` struct.
struct PackedEyeUniforms {
    var brightness: Float
    var contrast: Float
    var saturation: Float
    var mirror: Float
    var regionScale: SIMD2<Float>
    var regionOffset: SIMD2<Float>
}

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
    var pad0: Float = 0
}

// MARK: - StereoPump (off-main: decode -> warp -> enqueue)

/// Owns all per-frame GPU work on a dedicated background queue with its own
/// command queue. `@unchecked Sendable`: the AVFoundation / CoreVideo / Metal
/// objects it holds are documented thread-safe, config is lock-guarded, and all
/// mutable state is touched only on `queue`.
public final class StereoPump: @unchecked Sendable {
    public struct Config: Equatable {
        public var brightness: Float = 0
        public var contrast: Float = 1
        public var saturation: Float = 1
        /// Matches Pseudo3DSettings' Subtle default — always overwritten by
        /// makePumpConfig, but a missed config path should fail conservative.
        public var depthStrength: Float = 0.008
        /// Matches Pseudo3DSettings' default. Also the conservative fallback if a
        /// config path is ever missed: at 1.0 nothing crosses the window frame.
        public var convergence: Float = 1.0
        public var mirror: Bool = false
        /// Scales the picture's depth toward flat (a menu is opening over it;
        /// see `Pseudo3DStereoEngine.setChromeOpen`). A packed pair can't be
        /// part-flat, so it shows one eye twice below a half.
        public var depthScale: Float = 1

        var isFlat: Bool { depthScale < 0.5 }

        public init(
            brightness: Float = 0,
            contrast: Float = 1,
            saturation: Float = 1,
            depthStrength: Float = 0.008,
            convergence: Float = 1.0,
            mirror: Bool = false,
            depthScale: Float = 1
        ) {
            self.brightness = brightness
            self.contrast = contrast
            self.saturation = saturation
            self.depthStrength = depthStrength
            self.convergence = convergence
            self.mirror = mirror
            self.depthScale = depthScale
        }
    }

    private let videoRenderer: AVSampleBufferVideoRenderer
    private let frameSource: PumpFrameSource
    private let warp: StereoWarpResources
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private lazy var boundaryWarp = VergenceMetal.shared.flatMap {
        StereoBoundaryWarp(device: device, library: $0.library)
    }
    private let startHostTime: CFTimeInterval
    private let onVideoSizeKnown: @Sendable (CGSize) -> Void

    private let queue = DispatchQueue(label: "com.illixion.vergence.stereo-pump", qos: .userInteractive)
    private var timer: DispatchSourceTimer?
    /// Backgrounded: **visionOS refuses GPU submission from a background app**,
    /// so every command buffer a tick builds is aborted with
    /// `kIOGPUCommandBufferCallbackErrorBackgroundExecutionNotPermitted`. An
    /// `AVPlayer`-driven pump never noticed, because backgrounding pauses the
    /// player and a paused player yields no new frames — but a pump fed by a
    /// source that keeps producing (a web page's `<video>`, which keeps decoding
    /// and playing audio) ticks straight through it: ~6,800 aborted command
    /// buffers in 20 seconds on device, plus depth inference for every one of
    /// them. Read and written under `lifecycleLock`; only ever set from the
    /// notification observers below.
    private var isBackgrounded = false
    private let lifecycleLock = NSLock()
    private var lifecycleObservers: [NSObjectProtocol] = []

    private var textureCache: CVMetalTextureCache?
    /// BGRA Metal render targets for the warp (intermediate, not handed to the
    /// compositor).
    private var bgraPool: CVPixelBufferPool?
    /// 420v buffers handed to the renderer — built exactly like the validated
    /// test pattern (merged with `recommendedPixelBufferAttributes`).
    private var outPool: CVMutablePixelBuffer.Pool?
    private var poolWidth = 0
    private var poolHeight = 0
    /// Per-eye depth attachments for the occlusion-correct mesh warp ([0]=left,
    /// [1]=right). Separate textures because both eyes render in one command
    /// buffer; sharing one depth attachment let the right eye (2nd pass) test
    /// against the left eye's geometry, warping it more than the left.
    /// Multisampled + memoryless when MSAA is on (tile memory only, no backing).
    private var eyeDepthTextures: [MTLTexture?] = [nil, nil]
    /// Per-eye multisampled color targets for the MSAA mesh pass, resolved into
    /// the eye buffer by the render pass. Memoryless — only the resolve output
    /// (the eye buffer itself) is ever stored.
    private var eyeMSAAColorTextures: [MTLTexture?] = [nil, nil]
    /// Converts the BGRA warp output into the compositor's native 420v format.
    private let transferSession: VTPixelTransferSession
    /// Per-frame depth: realtime inference or PTS-matched cached depth (nil →
    /// heuristic warp only, e.g. a restored window whose model vanished).
    private let depthSource: PumpDepthSource?
    /// Set for a video that already carries both eyes: crop instead of warp.
    private let packedLayout: PackedStereoLayout?
    /// Pump-queue only: the packed frame size last checked against the tag.
    private var checkedPackedSize: CGSize?
    private var correctedLayout: PackedStereoLayout?
    private var lastReportedSize: CGSize?
    /// Pump-queue only. One line per pump, not one per frame at 60Hz.
    private var didWarnAboutPixelFormat = false
    /// When the depth source last transitioned unavailable → available; drives
    /// the flat→3D strength ramp in flatten mode (cached seek gaps, startup).
    private var depthResumeTime: CFTimeInterval?
    /// Set for a source that can hand frames out early; the pump then works
    /// `lead` seconds ahead of the display.
    private let clock: LookaheadFrameSource?
    private let lead: Double
    /// Pump-queue only: frames taken and not yet on screen, in order.
    private var schedule: [ScheduledFrame] = []
    /// Pump-queue only: the frame on screen (the last one queued whose time
    /// came), drawn again when a paused picture changes.
    private var shown: ScheduledFrame?
    /// Pump-queue only: the item time of the last frame taken from the source.
    private var lastTakenSeconds: Double?
    /// Pump-queue only: the clock as the last tick read it, to tell a seek.
    private var lastClockReading: (host: CFTimeInterval, item: Double, rate: Double)?
    /// Pump-queue only: the renderer was flushed under the frame on screen,
    /// so it is queued again unless a newer one shows at once.
    private var needsRedisplay = false
    /// Pump-queue only: the frame on screen with its depth being worked out
    /// again (a debug capture asking for it), shown once ready.
    private var pendingReshow: ScheduledFrame?
    /// Pump-queue only. Set by the stop drain; a depth completion arriving
    /// after it does nothing.
    private var stopped = false
    /// Pump-queue only: the request of the last frame drawn, to count maps
    /// reused against maps worked out.
    private var lastDrawnRequest: ObjectIdentifier?
    /// Flushes the renderer when visionOS took its decoding away; see
    /// `DecoderRecovery`. Made in `start()`, as it captures self.
    private var decoderRecovery: DecoderRecovery?
    /// Draws subtitles into both eyes (see StereoSubtitles.swift). Nil only if
    /// its shaders failed to build, which leaves the picture without them.
    private let subtitles: StereoSubtitleCompositor?

    private let configLock = NSLock()
    private var config = Config()

    private let signposter = VergenceLog.signposter

    /// Tick rate: 60fps in both modes — the warp is only a few ms, so the pump
    /// follows the source up to 60 (and no faster: a 120 fps video shows every
    /// other frame). Cached mode looks depth up by PTS; realtime mode infers
    /// at ~30Hz and holds the map for the in-between frame (see
    /// RealtimeDepthSource); a frame whose map isn't ready waits for it, so a
    /// slow model shows fewer frames, never older depth.
    private let frameInterval: Double
    /// How far ahead of the display the pump works, for a clocked source:
    /// long enough for a frame's inference, warp and transfer with room for a
    /// slow one; a cached or packed picture only needs the warp.
    private static let realtimeLead = 0.075
    private static let drawLead = 0.04
    /// A push source can't hand frames out early, so with depth worked out
    /// ahead each frame shows this long after it came: a map and a draw fit
    /// in it, and frames then show as evenly as they came (a frame reusing
    /// the map before it is drawn right after that one, and shown at once
    /// they would come in pairs).
    private static let pushDelay = 0.04
    /// Frames from a push source waiting for depth, at most; past that the
    /// model can't keep up, and a newer frame is let go rather than queued
    /// behind them.
    private static let maxPushBacklog = 2
    /// Frames taken and not yet on screen, at most: 75 ms at 60 fps is about
    /// five, and each holds a decoded frame and two eye buffers.
    private static let maxScheduled = 8
    /// Pump-queue only. For a depth source that infers per frame
    /// (`PumpDepthSource.limitsToSourceFrameRate`): faster than normal speed,
    /// no more frames a second than the video has at normal speed. A frame
    /// it leaves out is taken from the source and dropped, so the next tick
    /// gets a newer one.
    private var frameRateCap = FrameRateCap()
    /// Pump-queue only: the last frame taken from the source, shown or not.
    private var lastSourceSeconds: Double?
    /// Seconds to ramp depth strength back after a flat gap (avoids a 3D "pop").
    private let depthRampDuration: Double = 0.15
    /// Rolling per-stage wall time for the tick, logged once per window so the
    /// serialized cost (depth → warp → transfer, each waited on in turn) is
    /// readable in the in-app console on real content. Pump-queue only.
    private var stageTiming = StageTiming()
    private let stageTimingLogInterval: Double = 5

    /// Pump-queue only: how many of the source's frames reach the eyes,
    /// reported through `onPerformance` every `performanceInterval`.
    private var delivery = DeliveryWindow()
    private let performanceInterval: Double = 2
    /// Any thread may read; set before `start()`. Called on the pump queue.
    public var onPerformance: (@Sendable (StereoPerformance) -> Void)?
    /// Any thread may read; set before `start()`. Called once, on the pump
    /// queue, when the first frame is one no warp can show right (Dolby
    /// Vision profile 5, `DolbyVisionFrames`); nothing is rendered from then
    /// on, and the host should go back to 2D.
    public var onUnshowablePicture: (@Sendable () -> Void)?
    /// Pump-queue only.
    private var checkedPicture = false
    /// `StackHeadroom.baseline` has run for this pump.
    private var loggedStackBaseline = false
    private var pictureUnshowable = false

    private struct DeliveryWindow {
        var frames = 0
        var itemSpan = 0.0
        var lastItemSeconds: Double?
        var sourceHeight = 0
        /// The source's own frame spacing. Lifetime, so a window of drops
        /// can't redefine it.
        var sourceSpacing = SourceFrameSpacing()
        var windowStart = CACurrentMediaTime()
    }

    private struct StageTiming {
        /// Looking depth up as a frame is drawn (a source that doesn't work
        /// it out ahead).
        var lookup = 0.0
        /// Maps worked out ahead, their total time, and the frames drawn
        /// with the map of the frame before.
        var maps = 0
        var mapSeconds = 0.0
        var reused = 0
        var warp = 0.0
        var transfer = 0.0
        var drawn = 0
        var shown = 0
        /// Queued after the time they were due (shown as soon as possible).
        var late = 0
        /// The last frame drawn's size.
        var width = 0
        var height = 0
        var windowStart = CACurrentMediaTime()
    }

    /// The two finished eye buffers of one frame. Kept, so a flush (a pause,
    /// a speed change) can queue them again without drawing them again.
    private struct Eyes {
        let left: CVReadOnlyPixelBuffer
        let right: CVReadOnlyPixelBuffer
    }

    /// A frame from the moment it is taken until it leaves the screen.
    /// Pump-queue only.
    private final class ScheduledFrame {
        let frame: PumpFrame
        /// Depth being worked out ahead (`PumpDepthSource.computesDepthAhead`).
        let request: PumpDepthRequest?
        /// The depth it was drawn with, once looked up: a frame is drawn
        /// again (a paused picture's setting changed) with the same depth,
        /// since the source may have moved past it.
        var depth: PumpFrameDepth?
        var hasDepth = false
        var eyes: Eyes?
        /// The host time it is queued in the renderer for; nil while not.
        var queuedFor: CFTimeInterval?
        /// Counted in the delivery report (queued again after a flush, it
        /// isn't another frame shown).
        var delivered = false
        /// When the pump took it (host clock).
        let takenAt: CFTimeInterval

        init(frame: PumpFrame, request: PumpDepthRequest?, takenAt: CFTimeInterval) {
            self.frame = frame
            self.request = request
            self.takenAt = takenAt
        }

        var itemSeconds: Double { frame.itemTime.seconds }
    }
    /// Bounds in-flight eye buffers so a stalled compositor can never make the
    /// pool allocate IOSurfaces without limit.
    private let maxInFlightBuffers = 8

    /// - Parameter warp: shared pipelines and grids; `StereoWarpResources.shared`
    ///   unless a caller has reason to build its own.
    /// - Returns: nil when Metal can't give the pump its own command queue or
    ///   the pixel-transfer session can't be created — both mean "no fake-3D",
    ///   which callers already handle by falling back to flat playback.
    /// - Parameter packedLayout: the frame already holds both eyes; each is
    ///   cropped out and `depthSource` is ignored. The reported video size is
    ///   then one eye's display size, which is what the plane is fitted to.
    public init?(
        videoRenderer: AVSampleBufferVideoRenderer,
        frameSource: PumpFrameSource,
        warp: StereoWarpResources,
        startHostTime: CFTimeInterval,
        depthSource: PumpDepthSource?,
        frameInterval: Double,
        packedLayout: PackedStereoLayout? = nil,
        onVideoSizeKnown: @escaping @Sendable (CGSize) -> Void
    ) {
        var session: VTPixelTransferSession?
        VTPixelTransferSessionCreate(allocator: kCFAllocatorDefault, pixelTransferSessionOut: &session)
        guard let commandQueue = warp.device.makeCommandQueue(), let session else { return nil }
        self.videoRenderer = videoRenderer
        self.frameSource = frameSource
        self.warp = warp
        self.device = warp.device
        self.commandQueue = commandQueue
        self.startHostTime = startHostTime
        let usedDepthSource = packedLayout == nil ? depthSource : nil
        let clock = frameSource as? LookaheadFrameSource
        self.depthSource = usedDepthSource
        self.packedLayout = packedLayout
        self.frameInterval = frameInterval
        self.clock = clock
        self.lead = clock == nil ? 0
            : usedDepthSource?.computesDepthAhead == true ? Self.realtimeLead : Self.drawLead
        self.onVideoSizeKnown = onVideoSizeKnown
        self.transferSession = session
        self.subtitles = VergenceMetal.shared.flatMap {
            StereoSubtitleCompositor(device: $0.device, library: $0.library)
        }
        CVMetalTextureCacheCreate(nil, nil, warp.device, nil, &textureCache)
    }

    /// Any thread. Nil turns subtitles off. The last frame is drawn again with
    /// the new text, since a paused picture has no new frame coming.
    public func setSubtitles(_ overlay: StereoSubtitleOverlay?) {
        subtitles?.setOverlay(overlay)
        queue.async { [weak self] in self?.reshowLastFrame() }
    }

    /// A change redraws the last frame, so it shows on a paused picture too.
    public func updateConfig(_ newConfig: Config) {
        configLock.lock()
        let changed = config != newConfig
        config = newConfig
        configLock.unlock()
        if changed { queue.async { [weak self] in self?.reshowLastFrame() } }
    }

    private func currentConfig() -> Config {
        configLock.lock(); defer { configLock.unlock() }
        return config
    }

    public func start() {
        observeAppLifecycle()
        decoderRecovery = DecoderRecovery(renderer: videoRenderer) { [weak self] _ in
            self?.queue.async { self?.flushAndReshow() }
        }
        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now(), repeating: frameInterval, leeway: .milliseconds(2))
        timer.setEventHandler { [weak self] in self?.tick() }
        self.timer = timer
        timer.resume()
    }

    public func stop() {
        decoderRecovery?.cancel()
        for observer in lifecycleObservers { NotificationCenter.default.removeObserver(observer) }
        lifecycleObservers = []
        timer?.cancel()
        timer = nil
        // Drain on the pump queue so no tick races teardown.
        queue.async { [weak self] in
            guard let self else { return }
            stopped = true
            for entry in schedule { entry.request?.cancel() }
            schedule = []
            shown = nil
            pendingReshow?.request?.cancel()
            pendingReshow = nil
            depthSource?.invalidate()
            frameSource.invalidate()
            if let textureCache { CVMetalTextureCacheFlush(textureCache, 0) }
            self.bgraPool = nil
            self.outPool = nil
            self.eyeDepthTextures = [nil, nil]
            self.eyeMSAAColorTextures = [nil, nil]
            self.boundaryWarp = nil
        }
    }

    /// The timer keeps running while backgrounded rather than being cancelled:
    /// the flag is one lock away on a path that already takes locks, and
    /// tearing the timer down and back up would have to be ordered against
    /// `stop()` racing a notification. Ticks become a lock and a return.
    private func observeAppLifecycle() {
        guard lifecycleObservers.isEmpty else { return }
        let center = NotificationCenter.default
        let setBackgrounded: @Sendable (Bool) -> Void = { [weak self] backgrounded in
            guard let self else { return }
            lifecycleLock.lock()
            isBackgrounded = backgrounded
            lifecycleLock.unlock()
        }
        lifecycleObservers = [
            center.addObserver(
                forName: VergenceAppLifecycle.didEnterBackground, object: nil, queue: nil
            ) { _ in setBackgrounded(true) },
            center.addObserver(
                forName: VergenceAppLifecycle.willEnterForeground, object: nil, queue: nil
            ) { _ in setBackgrounded(false) }
        ]
    }

    // MARK: Per-frame work (always on `queue`)

    private var backgroundedNow: Bool {
        lifecycleLock.lock(); defer { lifecycleLock.unlock() }
        return isBackgrounded
    }

    /// The renderer lost its decoding (see `DecoderRecovery`) and was
    /// flushed: queue the frames again, the one on screen first.
    private func flushAndReshow() {
        videoRenderer.flush()
        unqueueAll()
        service(now: CACurrentMediaTime())
    }

    /// A setting or the subtitles changed. A paused picture has no new frame
    /// coming, so the one on screen is drawn again, and the frames taken
    /// ahead of it are drawn afresh before play resumes. While the clock
    /// runs, the frames drawn next carry the change (within `lead`); drawing
    /// the one on screen again would put it between frames already queued.
    private func reshowLastFrame() {
        guard !stopped, !backgroundedNow, !pictureUnshowable, let textureCache, let shown else { return }
        let now = CACurrentMediaTime()
        if let clock, (clock.playbackRate(hostTime: now) ?? 0) > 0.01 { return }
        for entry in schedule where entry.queuedFor == nil { entry.eyes = nil }
        if videoRenderer.isReadyForMoreMediaData, let eyes = draw(shown, cache: textureCache) {
            shown.eyes = eyes
            enqueue(eyes, at: now)
            shown.queuedFor = now
        }
        service(now: now)
    }

    /// A debug capture wants the paused frame's depth worked out again: ask
    /// for it as for the first frame of a shot, and show it once ready.
    private func recomputeShownDepth() {
        guard let shown, let depthSource, depthSource.computesDepthAhead else {
            depthSource?.recomputeNextDepth()
            reshowLastFrame()
            return
        }
        guard pendingReshow == nil else { return }
        depthSource.recomputeNextDepth()
        let now = CACurrentMediaTime()
        let request = depthSource.requestDepth(for: shown.frame, shownAt: now, whenReady: depthReadyHandler)
        pendingReshow = ScheduledFrame(frame: shown.frame, request: request, takenAt: now)
    }

    private func tick() {
        if !loggedStackBaseline {
            loggedStackBaseline = true
            StackHeadroom.baseline("StereoPump.tick")
        }
        StackHeadroom.check("StereoPump.tick")
        // A disconnected window can retain its SwiftUI/RealityKit graph.
        // A host-side close must end the timer and drop the frames rather
        // than masquerading as a pause and answering captures indefinitely.
        if frameSource.isFinished {
            stop()
            return
        }
        // No GPU work at all while backgrounded — not the warp, and not the
        // depth inference that feeds it. Frames taken ahead are let go (their
        // inference called off); the frame source keeps taking frames and
        // discarding all but the newest, so foregrounding resumes on current
        // content rather than replaying a backlog.
        guard !backgroundedNow else {
            if !schedule.isEmpty { discardSchedule(now: CACurrentMediaTime()) }
            return
        }

        // A renderer whose decoding visionOS took away (another app's
        // immersive space) drops everything until flushed; the recovery
        // flushes on this queue and queues the frames again.
        if decoderRecovery?.checkRenderer() == true { return }

        let now = CACurrentMediaTime()
        if let clock { reconcile(with: clock, now: now) }
        let took = takeFrames(now: now)
        service(now: now)
        // A paused picture draws nothing new; a capture (debug server) gets
        // the frame on screen drawn again instead.
        if !took, StereoEyeCapture.isRequested, !pictureUnshowable,
           !schedule.contains(where: { $0.eyes == nil }) {
            if StereoEyeCapture.recomputesDepth { recomputeShownDepth() } else { reshowLastFrame() }
        }
    }

    /// The clock moved under the frames taken ahead. Past a seek or a loop
    /// they belong to the old position and are dropped; after a pause, a
    /// stall or a speed change the ones already queued are due at other
    /// times (or not at all), so the renderer is flushed and they are queued
    /// again.
    private func reconcile(with clock: LookaheadFrameSource, now: CFTimeInterval) {
        let playhead = clock.itemTime(forHostTime: now).seconds
        let windowEnd = clock.itemTime(forHostTime: now + lead).seconds
        guard playhead.isFinite, windowEnd.isFinite else { return }
        let rate = clock.playbackRate(hostTime: now) ?? 0
        let previous = lastClockReading
        lastClockReading = (now, playhead, rate)
        // First, so the frame on screen is the last one whose time came.
        prune(now: now)
        let jumped = previous.map {
            LookaheadClock.isJump(previous: $0.item, previousRate: $0.rate, elapsed: now - $0.host,
                                  playhead: playhead, rate: rate)
        } ?? false
        if jumped || lastTakenSeconds.map({
            LookaheadClock.isDiscontinuous(last: $0, playhead: playhead, windowEnd: windowEnd)
        }) == true {
            // Taken afresh from the playhead: the source has new frames there.
            discardSchedule(now: now)
            lastTakenSeconds = nil
            return
        }
        for entry in schedule {
            guard let queuedFor = entry.queuedFor, queuedFor > now else { continue }
            let due = clock.dueHostTime(forItemTime: entry.frame.itemTime, at: now)
            if LookaheadClock.needsRetime(queuedFor: queuedFor, due: due) {
                videoRenderer.flush()
                unqueueAll()
                return
            }
        }
    }

    /// Take the frames due within `lead`, each in turn (a push source: its
    /// one new frame). True when any frame came.
    private func takeFrames(now: CFTimeInterval) -> Bool {
        guard let clock else {
            guard let frame = frameSource.nextFrame(hostTime: now) else { return false }
            if schedule.filter({ $0.eyes == nil }).count >= Self.maxPushBacklog {
                noteSourceFrame(itemSeconds: frame.itemTime.seconds)
                return true
            }
            accept(frame, now: now)
            return true
        }
        let playhead = clock.itemTime(forHostTime: now).seconds
        let windowEnd = clock.itemTime(forHostTime: now + lead).seconds
        guard playhead.isFinite, windowEnd.isFinite else { return false }
        var took = false
        for _ in 0 ..< Self.maxScheduled * 2 {
            guard schedule.count < Self.maxScheduled else { break }
            let target = LookaheadPull.target(last: lastTakenSeconds, spacing: delivery.sourceSpacing.seconds,
                                              playhead: playhead, windowEnd: windowEnd)
            // Nothing new can come at or before the last frame taken, and
            // asking there would be answered with a frame already taken (the
            // output hands out whatever differs from the last one it gave).
            // A seek back starts over from the playhead (`reconcile`).
            if let last = lastTakenSeconds, target <= last + 1e-6 { break }
            guard let frame = clock.frame(atItemTime: CMTime(seconds: target, preferredTimescale: 600_000)) else {
                break
            }
            took = true
            accept(frame, now: now)
        }
        return took
    }

    /// One frame from the source: scheduled, with its depth asked for, or
    /// left out (the frame-rate cap, a picture no warp can show).
    private func accept(_ frame: PumpFrame, now: CFTimeInterval) {
        if !checkedPicture {
            checkedPicture = true
            if DolbyVisionFrames.needsSystemRenderer(frame.pixelBuffer) {
                pictureUnshowable = true
                onUnshowablePicture?()
            }
        }
        if pictureUnshowable { return }
        let seconds = frame.itemTime.seconds
        if let last = lastTakenSeconds, seconds.isFinite {
            if abs(seconds - last) < 1e-6 { return }
            // Back in time, or a jump forward: a seek or a loop. The frames
            // taken for the old position would show before this one.
            if seconds < last || seconds - last > LookaheadPull.maxStep { discardSchedule(now: now) }
        }
        noteSourceFrame(itemSeconds: seconds)
        lastTakenSeconds = seconds.isFinite ? seconds : nil
        guard frameRateCap.admits(
            frameAt: seconds, spacing: delivery.sourceSpacing.seconds,
            rate: frameSource.playbackRate(hostTime: now) ?? 1,
            limitsToNormalSpeed: depthSource?.limitsToSourceFrameRate == true,
            minimumInterval: clock == nil ? 0 : frameInterval
        ) else { return }
        reportSizeIfNeeded(frame.pixelBuffer)
        var request: PumpDepthRequest?
        if let depthSource, depthSource.computesDepthAhead {
            let shownAt = clock?.dueHostTime(forItemTime: frame.itemTime, at: now) ?? now
            request = depthSource.requestDepth(for: frame, shownAt: shownAt, whenReady: depthReadyHandler)
        }
        schedule.append(ScheduledFrame(frame: frame, request: request, takenAt: now))
    }

    /// Runs on the inference queue when a map is ready.
    private var depthReadyHandler: @Sendable () -> Void {
        { [weak self] in self?.depthReady() }
    }

    private func depthReady() {
        queue.async { [weak self] in
            guard let self, !backgroundedNow else { return }
            service(now: CACurrentMediaTime())
        }
    }

    /// Draw what can be drawn, then queue what is drawn, in order.
    private func service(now: CFTimeInterval) {
        guard !stopped, !pictureUnshowable, let textureCache else { return }
        if let reshow = pendingReshow, reshow.request?.isReady ?? true {
            pendingReshow = nil
            if videoRenderer.isReadyForMoreMediaData, let eyes = draw(reshow, cache: textureCache) {
                reshow.eyes = eyes
                reshow.delivered = true
                enqueue(eyes, at: now)
                reshow.queuedFor = now
                shown = reshow
            }
        }
        var index = 0
        while index < schedule.count {
            let entry = schedule[index]
            if entry.eyes != nil { index += 1; continue }
            // Maps come in order, so the first frame still waiting for one
            // holds up the frames after it.
            if let request = entry.request, !request.isReady { break }
            guard let eyes = draw(entry, cache: textureCache) else {
                schedule.remove(at: index)
                continue
            }
            entry.eyes = eyes
            index += 1
        }
        let drawn = CACurrentMediaTime()
        queueDrawn(now: drawn)
        prune(now: drawn)
        logStageTimingIfDue(now: drawn)
    }

    /// Whether the frame's time has come: the clock's playhead has reached
    /// it. A push source (no playhead) shows each frame once drawn.
    private func isDue(_ entry: ScheduledFrame, playhead: Double?) -> Bool {
        guard let playhead else { return false }
        return entry.itemSeconds <= playhead + 0.001
    }

    /// Queue drawn frames in order, each for the host time it is due (now,
    /// for a push source or a stopped clock). While paused only the frame at
    /// the playhead goes in. A frame whose time has passed while a newer one
    /// is drawn and due too is left out, rather than shown for an instant.
    private func queueDrawn(now: CFTimeInterval) {
        let playhead = clock.map { $0.itemTime(forHostTime: now).seconds }.flatMap { $0.isFinite ? $0 : nil }
        let running = clock.map { ($0.playbackRate(hostTime: now) ?? 0) > 0.01 } ?? true
        if needsRedisplay {
            needsRedisplay = false
            // Unless the next frame shows right away, the one that was on
            // screen goes back in first (a flush may have cleared it).
            let next = schedule.first { $0.queuedFor == nil }
            let nextShowsNow = next.map { $0.eyes != nil && (clock == nil || isDue($0, playhead: playhead)) } ?? false
            if !nextShowsNow, let shown, let eyes = shown.eyes, videoRenderer.isReadyForMoreMediaData {
                enqueue(eyes, at: now)
                shown.queuedFor = now
            }
        }
        var index = 0
        while index < schedule.count {
            let entry = schedule[index]
            if entry.queuedFor != nil { index += 1; continue }
            guard let eyes = entry.eyes else { break }
            let isDueNow = isDue(entry, playhead: playhead)
            if isDueNow, index + 1 < schedule.count, schedule[index + 1].eyes != nil,
               isDue(schedule[index + 1], playhead: playhead) {
                schedule.remove(at: index)
                continue
            }
            if clock != nil, !running, !isDueNow { break }
            guard videoRenderer.isReadyForMoreMediaData else { break }
            let dueAt = clock == nil
                ? (entry.request != nil ? entry.takenAt + Self.pushDelay : nil)
                : running ? clock?.dueHostTime(forItemTime: entry.frame.itemTime, at: now) : nil
            let at = max(dueAt ?? now, now)
            enqueue(eyes, at: at)
            entry.queuedFor = at
            if !entry.delivered {
                entry.delivered = true
                stageTiming.shown += 1
                if let dueAt, dueAt < now - 0.002 { stageTiming.late += 1 }
                noteDelivered(itemSeconds: entry.itemSeconds,
                              height: CVPixelBufferGetHeight(entry.frame.pixelBuffer), at: now)
            }
            index += 1
        }
    }

    /// Frames whose time on screen has come leave the schedule; the newest of
    /// them is the frame on screen.
    private func prune(now: CFTimeInterval) {
        while let first = schedule.first, let queuedFor = first.queuedFor, queuedFor <= now {
            shown = schedule.removeFirst()
        }
    }

    /// The renderer was flushed: nothing is queued in it any more.
    private func unqueueAll() {
        for entry in schedule { entry.queuedFor = nil }
        needsRedisplay = true
    }

    /// Drop every frame taken ahead (a seek, the app backgrounded), calling
    /// off their depth, and flush what is queued for later. The frame on
    /// screen stays until a new one replaces it.
    private func discardSchedule(now: CFTimeInterval) {
        if schedule.contains(where: { ($0.queuedFor ?? 0) > now }) { videoRenderer.flush() }
        for entry in schedule { entry.request?.cancel() }
        schedule = []
        lastDrawnRequest = nil
    }

    /// Every frame taken from the source, shown or left out by the cap,
    /// tells its frame spacing.
    private func noteSourceFrame(itemSeconds: Double) {
        guard itemSeconds.isFinite else { return }
        if let last = lastSourceSeconds { delivery.sourceSpacing.note(gap: itemSeconds - last) }
        lastSourceSeconds = itemSeconds
    }

    /// Counts a frame that reached the eyes against the source time it
    /// covers. A pump that can't keep up takes only the newest decoded
    /// frame, so the source frames between are never shown: the gaps between
    /// shown frames grow past the source's own spacing. `itemSeconds` must be
    /// the frame's own presentation time: the time the tick asked for would
    /// make the gaps the ticks' spacing instead (a 24 fps film read as 59 or
    /// 120 fps, so every report said it was overloaded).
    private func noteDelivered(itemSeconds: Double, height: Int, at now: CFTimeInterval) {
        guard itemSeconds.isFinite else { return }
        delivery.sourceHeight = height
        if let last = delivery.lastItemSeconds {
            let gap = itemSeconds - last
            // A seek, a loop or a pause/resume isn't a run of dropped frames.
            if SourceFrameSpacing.range.contains(gap) {
                delivery.itemSpan += gap
                delivery.frames += 1
            }
        }
        delivery.lastItemSeconds = itemSeconds
        let window = now - delivery.windowStart
        guard window >= performanceInterval else { return }
        defer {
            delivery.frames = 0
            delivery.itemSpan = 0
            delivery.windowStart = now
        }
        guard let onPerformance,
              let report = StereoPerformance.measured(
                frames: delivery.frames, itemSpan: delivery.itemSpan, window: window,
                spacing: delivery.sourceSpacing.seconds, maxShownPerSecond: 1 / frameInterval,
                sourceHeight: delivery.sourceHeight) else { return }
        onPerformance(report)
    }

    /// Draw a scheduled frame, looking its depth up first if that hasn't
    /// happened: the map it was given (worked out ahead, or shared with the
    /// frame before), or for a looked-up source the frame's own by
    /// presentation time. Either way image and depth match within the bound
    /// `HeldDepthPolicy` keeps, never across a cut.
    private func draw(_ entry: ScheduledFrame, cache: CVMetalTextureCache) -> Eyes? {
        if !entry.hasDepth {
            entry.hasDepth = true
            if let request = entry.request {
                entry.depth = request.depth
                let id = ObjectIdentifier(request)
                if id == lastDrawnRequest {
                    stageTiming.reused += 1
                } else {
                    stageTiming.maps += 1
                    stageTiming.mapSeconds += request.computeSeconds
                }
                lastDrawnRequest = id
            } else if let depthSource {
                StackHeadroom.check("StereoPump.draw")
                let start = CACurrentMediaTime()
                let depthState = signposter.beginInterval("pump-depth")
                entry.depth = depthSource.frameDepth(itemTime: entry.frame.itemTime, frame: entry.frame.pixelBuffer)
                signposter.endInterval("pump-depth", depthState)
                stageTiming.lookup += CACurrentMediaTime() - start
            }
        }
        return drawEyes(from: entry.frame.pixelBuffer, itemTime: entry.frame.itemTime, depth: entry.depth, cache: cache)
    }

    /// Warp the mono frame into two BGRA eye render targets and convert each
    /// to a 420f buffer (the compositor's native format) via
    /// VTPixelTransferSession, ready to queue. The 420f buffers come from a
    /// pool merged with `recommendedPixelBufferAttributes` — the construction
    /// validated on-device. `CVMutablePixelBuffer` is noncopyable, so the two
    /// eye buffers stay local: borrowed by `transfer`, then consumed by
    /// `CVReadOnlyPixelBuffer`.
    private func drawEyes(from src: CVPixelBuffer, itemTime: CMTime, depth frameDepth: PumpFrameDepth?,
                          cache: CVMetalTextureCache) -> Eyes? {
        let drawStart = CACurrentMediaTime()
        let width = CVPixelBufferGetWidth(src)
        let height = CVPixelBufferGetHeight(src)
        // Eye buffers are one eye's display size: the frame itself when warping
        // a mono picture, a half (stretched back out if squeezed) when cropping.
        let layout = packedLayout(forFrame: CGSize(width: width, height: height))
        let eyeSize = layout?.eyeSize(packed: CGSize(width: width, height: height))
            ?? CGSize(width: width, height: height)
        let eyeWidth = Int(eyeSize.width)
        let eyeHeight = Int(eyeSize.height)
        guard width > 0, height > 0, eyeWidth > 0, eyeHeight > 0,
              ensurePools(width: eyeWidth, height: eyeHeight),
              let bgraPool, let outPool,
              let srcTexture = makeTexture(from: src, cache: cache) else { return nil }

        // 1. Warp into two BGRA eye render targets.
        var leftBGRA: CVPixelBuffer?
        var rightBGRA: CVPixelBuffer?
        let aux: [String: Any] = [kCVPixelBufferPoolAllocationThresholdKey as String: maxInFlightBuffers]
        guard CVPixelBufferPoolCreatePixelBufferWithAuxAttributes(nil, bgraPool, aux as CFDictionary, &leftBGRA) == kCVReturnSuccess,
              CVPixelBufferPoolCreatePixelBufferWithAuxAttributes(nil, bgraPool, aux as CFDictionary, &rightBGRA) == kCVReturnSuccess,
              let leftBGRA, let rightBGRA,
              let leftTex = makeTexture(from: leftBGRA, cache: cache),
              let rightTex = makeTexture(from: rightBGRA, cache: cache),
              let cmdBuf = commandQueue.makeCommandBuffer() else { return nil }
        StackHeadroom.check("StereoPump.drawEyes")

        var cfg = currentConfig()
        cfg.depthStrength *= max(0, min(1, cfg.depthScale))
        let fullDepthStrength = cfg.depthStrength
        if depthSource?.flattensWhenUnavailable == true {
            // Cached mode: a seek/startup gap renders FLAT (never the heuristic
            // warp, never stale depth), then depth ramps back in to avoid a pop.
            if frameDepth == nil {
                depthResumeTime = nil
                cfg.depthStrength = 0
            } else {
                let now = CACurrentMediaTime()
                let since = depthResumeTime ?? { depthResumeTime = now; return now }()
                cfg.depthStrength *= Float(min(1, (now - since) / depthRampDuration))
            }
        }

        let warpState = signposter.beginInterval("pump-warp")
        if let layout {
            encodePackedEye(.left, layout: layout, into: leftTex, source: srcTexture, config: cfg, commandBuffer: cmdBuf)
            encodePackedEye(cfg.isFlat ? .left : .right, layout: layout, into: rightTex, source: srcTexture,
                            config: cfg, commandBuffer: cmdBuf)
        } else {
            encodeEye(into: leftTex, source: srcTexture, depth: frameDepth, eyeSign: 1.0, config: cfg, commandBuffer: cmdBuf)
            encodeEye(into: rightTex, source: srcTexture, depth: frameDepth, eyeSign: -1.0, config: cfg, commandBuffer: cmdBuf)
        }
        // The depth budget keeps depth inside [0, 1], so no warped pixel shifts further than
        // (1 - convergence) × strength; the margin in front of that keeps the
        // strength the ramp is heading for, so text doesn't sink mid-ramp.
        let capture = StereoEyeCapture.takeRequest()
            ? StereoEyeCapture.encodeDepthReadback(frameDepth, device: device, commandBuffer: cmdBuf) : nil
        subtitles?.encode(
            left: leftTex, right: rightTex, packed: layout != nil && !cfg.isFlat,
            maxContentShift: (1 - cfg.convergence) * cfg.depthStrength,
            depthStrength: fullDepthStrength,
            transfer: StereoSubtitleCompositor.transferMode(of: src), commandBuffer: cmdBuf
        )
        cmdBuf.commit()
        // Safe here: this runs on the background pump queue, never main. The warp
        // must complete before VTPixelTransferSession reads the BGRA surfaces.
        cmdBuf.waitUntilCompleted()
        subtitles?.didComplete()
        if let capture {
            StereoEyeCapture.deliver(.init(
                left: leftBGRA, right: rightBGRA, source: src, depth: capture.read(),
                itemSeconds: itemTime.seconds, depthStrength: cfg.depthStrength, convergence: cfg.convergence
            ))
        }
        signposter.endInterval("pump-warp", warpState)
        let warpDone = CACurrentMediaTime()

        // 2. Convert each BGRA eye → 420f from the recommended-attributes pool,
        // carrying the source's color tags so the compositor reads gamma/range
        // correctly (untagged YCbCr was being misinterpreted = washed out).
        let transferState = signposter.beginInterval("pump-transfer")
        defer { signposter.endInterval("pump-transfer", transferState) }
        guard let left = try? outPool.makeMutablePixelBuffer() else { return nil }
        guard transfer(from: leftBGRA, to: left) else { return nil }
        tagColor(left, from: src)
        guard let right = try? outPool.makeMutablePixelBuffer() else { return nil }
        guard transfer(from: rightBGRA, to: right) else { return nil }
        tagColor(right, from: src)
        let eyes = Eyes(left: CVReadOnlyPixelBuffer(left), right: CVReadOnlyPixelBuffer(right))

        stageTiming.warp += warpDone - drawStart
        stageTiming.transfer += CACurrentMediaTime() - warpDone
        stageTiming.drawn += 1
        stageTiming.width = width
        stageTiming.height = height
        return eyes
    }

    /// Tag, describe and queue a frame's eyes for host time `hostTime`, on
    /// the renderer's timeline: its synchronizer free-runs at rate 1 from
    /// `startHostTime`, so the source's clock stays the only one governing
    /// audio, pausing and seeking, and the pump translates it into host time
    /// (`LookaheadFrameSource.dueHostTime`). The format description is built
    /// from the tagged group, exactly as Apple's sample does. (Omitting the
    /// format description / using BGRA was a cause of the earlier GPU hang.)
    private func enqueue(_ eyes: Eyes, at hostTime: CFTimeInterval) {
        let presentation = CMTime(seconds: hostTime - startHostTime, preferredTimescale: 90_000)
        let leftTags: [CMTag] = [.videoLayerID(0), .stereoView(.leftEye), .mediaType(.video)]
        let rightTags: [CMTag] = [.videoLayerID(1), .stereoView(.rightEye), .mediaType(.video)]
        let tagged: [CMTaggedDynamicBuffer] = [
            CMTaggedDynamicBuffer(tags: leftTags, content: .pixelBuffer(eyes.left)),
            CMTaggedDynamicBuffer(tags: rightTags, content: .pixelBuffer(eyes.right))
        ]
        let sample = CMReadySampleBuffer(
            taggedBuffers: tagged,
            formatDescription: CMTaggedBufferGroupFormatDescription(taggedBuffers: tagged),
            presentationTimeStamp: presentation.isValid ? presentation : .zero,
            duration: CMTime(value: 1, timescale: 90)
        )
        sample.withUnsafeSampleBuffer { videoRenderer.enqueue($0) }
    }

    /// One line per window, so the stages' cost on real content is readable
    /// in the in-app console.
    private func logStageTimingIfDue(now: CFTimeInterval) {
        let window = now - stageTiming.windowStart
        guard window >= stageTimingLogInterval else { return }
        let timing = stageTiming
        stageTiming = StageTiming()
        // A paused picture drawn again (a menu fading it) shows no new frame.
        guard timing.shown > 0, timing.drawn > 0 else { return }
        // os_log interpolations are closures; pull the values into locals so
        // they don't capture self.
        let drawn = Double(timing.drawn)
        let fps = Double(timing.shown) / window
        let warpMs = timing.warp / drawn * 1000
        let transferMs = timing.transfer / drawn * 1000
        let width = timing.width, height = timing.height
        let packed = packedLayout == nil ? "" : " packed"
        let depth: String
        if depthSource?.computesDepthAhead == true {
            let mapMs = timing.maps > 0 ? timing.mapSeconds / Double(timing.maps) * 1000 : 0
            let reusedShare = Double(timing.reused) / Double(max(timing.maps + timing.reused, 1)) * 100
            depth = String(format: "depth %.1f ms a map, %.1f maps/s, %.0f%% of frames reuse the one before",
                           mapMs, Double(timing.maps) / window, reusedShare)
        } else if depthSource != nil {
            depth = String(format: "depth %.1f ms", timing.lookup / drawn * 1000)
        } else {
            depth = "no depth"
        }
        let schedule = clock == nil ? "shown as drawn"
            : String(format: "%.0f ms ahead, %.0f%% late", lead * 1000,
                     timing.shown > 0 ? Double(timing.late) / Double(timing.shown) * 100 : 0)
        VergenceLog.pipeline.info(
            "pump \(width)×\(height)\(packed, privacy: .public): \(fps, format: .fixed(precision: 1), privacy: .public) fps over \(window, format: .fixed(precision: 1), privacy: .public) s — \(depth, privacy: .public), warp \(warpMs, format: .fixed(precision: 1), privacy: .public) ms, transfer \(transferMs, format: .fixed(precision: 1), privacy: .public) ms, \(schedule, privacy: .public)"
        )
    }

    private func transfer(from source: CVPixelBuffer, to dest: borrowing CVMutablePixelBuffer) -> Bool {
        var ok = false
        dest.withUnsafeBuffer { destBuffer in
            ok = VTPixelTransferSessionTransferImage(transferSession, from: source, to: destBuffer) == noErr
        }
        return ok
    }

    /// Copy the source frame's color primaries / transfer function / YCbCr matrix
    /// onto the eye buffer (defaulting to Rec.709). Without these the compositor
    /// guesses the color space and renders the eyes washed out.
    private func tagColor(_ dest: borrowing CVMutablePixelBuffer, from src: CVPixelBuffer) {
        dest.withUnsafeBuffer { d in
            let primaries = CVBufferGetAttachment(src, kCVImageBufferColorPrimariesKey, nil)?.takeUnretainedValue()
                ?? kCVImageBufferColorPrimaries_ITU_R_709_2
            CVBufferSetAttachment(d, kCVImageBufferColorPrimariesKey, primaries, .shouldPropagate)

            let transfer = CVBufferGetAttachment(src, kCVImageBufferTransferFunctionKey, nil)?.takeUnretainedValue()
                ?? kCVImageBufferTransferFunction_ITU_R_709_2
            CVBufferSetAttachment(d, kCVImageBufferTransferFunctionKey, transfer, .shouldPropagate)

            let matrix = CVBufferGetAttachment(src, kCVImageBufferYCbCrMatrixKey, nil)?.takeUnretainedValue()
                ?? kCVImageBufferYCbCrMatrix_ITU_R_709_2
            CVBufferSetAttachment(d, kCVImageBufferYCbCrMatrixKey, matrix, .shouldPropagate)
        }
    }

    private func encodeEye(
        into dest: MTLTexture,
        source: MTLTexture,
        depth: PumpFrameDepth?,
        eyeSign: Float,
        config: Config,
        commandBuffer: MTLCommandBuffer
    ) {
        var uniforms = VideoStereoUniforms(
            brightness: config.brightness,
            contrast: config.contrast,
            saturation: config.saturation,
            depthStrength: config.depthStrength,
            convergence: config.convergence,
            eyeSign: eyeSign,
            mirror: config.mirror ? 1.0 : 0.0,
            useDepth: depth != nil ? 1.0 : 0.0,
            depthUVScale: depth?.uvScale ?? SIMD2(1, 1),
            depthUVOffset: depth?.uvOffset ?? SIMD2(0, 0),
            depthValueScale: depth?.valueScale ?? 1,
            depthValueBias: depth?.valueBias ?? 0,
            depthBudget: (depth?.budget ?? .legacy).knees,
            edgeFill: DepthRefinement.fillsUncoveredEdges ? 1 : 0
        )

        // With a real depth map: occlusion-correct depth-displaced mesh. Without
        // one: the per-pixel heuristic warp. Each eye gets its own depth
        // attachment (0=left, 1=right) — the two eyes render in one command
        // buffer, so a shared attachment cross-contaminated the depth test.
        let eyeIndex = eyeSign > 0 ? 0 : 1
        if let depth, DepthRefinement.separatesBoundaries, let boundaryWarp {
            let dense = depthSource?.prefersDenseWarpGrid == true
            let columns = dense ? StereoWarpResources.denseGridColumns : StereoWarpResources.gridColumns
            let u = StereoBoundaryWarp.Uniforms(
                stereo: SIMD4(config.depthStrength, config.convergence, eyeSign, config.mirror ? 1 : 0),
                uv: SIMD4(depth.uvScale.x, depth.uvScale.y, depth.uvOffset.x, depth.uvOffset.y),
                mapping: SIMD4(depth.valueScale, depth.valueBias, uniforms.depthBudget.x, uniforms.depthBudget.y),
                adjustments: SIMD4(config.brightness, config.contrast, config.saturation, 0),
                boundary: SIMD4(0.08, Float(dest.width), 1 / Float(columns - 1), DepthRefinement.showsBoundaryHoles ? 1 : 0))
            if boundaryWarp.encode(source: source, depth: depth.texture, destination: dest,
                positions: dense ? warp.denseGridPositions : warp.gridPositions,
                indices: dense ? warp.denseGridIndices : warp.gridIndices,
                indexCount: dense ? warp.denseGridIndexCount : warp.gridIndexCount,
                uniforms: u, commandBuffer: commandBuffer) { return }
        }
        if let depth, let depthAttachment = ensureDepthTexture(width: dest.width, height: dest.height, eye: eyeIndex) {
            let desc = MTLRenderPassDescriptor()
            // MSAA: rasterize into a memoryless multisampled target and resolve
            // into the eye buffer — the mesh's occlusion boundaries are raster
            // edges with no texture-side AA and stairstep without it. The
            // pipeline is built at pseudo3DMSAASampleCount, so the attachments
            // must match; if the transient MSAA target can't be made (never
            // observed — memoryless has no memory backing), skip the frame.
            if warp.msaaSampleCount > 1 {
                guard let msaaColor = ensureMSAAColorTexture(width: dest.width, height: dest.height, eye: eyeIndex) else { return }
                desc.colorAttachments[0].texture = msaaColor
                desc.colorAttachments[0].resolveTexture = dest
                desc.colorAttachments[0].storeAction = .multisampleResolve
            } else {
                desc.colorAttachments[0].texture = dest
                desc.colorAttachments[0].storeAction = .store
            }
            desc.colorAttachments[0].loadAction = .clear
            desc.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
            desc.depthAttachment.texture = depthAttachment
            desc.depthAttachment.loadAction = .clear
            desc.depthAttachment.clearDepth = 1.0
            desc.depthAttachment.storeAction = .dontCare
            guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: desc) else { return }
            // Grid density per depth source: dense for cached (edge-aware) depth,
            // moderate for realtime — see PumpDepthSource.prefersDenseWarpGrid.
            let dense = depthSource?.prefersDenseWarpGrid == true
            encoder.setRenderPipelineState(warp.meshPipelineState)
            encoder.setDepthStencilState(warp.meshDepthState)
            encoder.setVertexBuffer(dense ? warp.denseGridPositions : warp.gridPositions, offset: 0, index: 0)
            encoder.setVertexBytes(&uniforms, length: MemoryLayout<VideoStereoUniforms>.size, index: 1)
            encoder.setVertexTexture(depth.texture, index: 0)
            encoder.setFragmentTexture(source, index: 0)
            encoder.setFragmentTexture(depth.texture, index: 1)
            encoder.setFragmentBytes(&uniforms, length: MemoryLayout<VideoStereoUniforms>.size, index: 0)
            encoder.drawIndexedPrimitives(
                type: .triangle, indexCount: dense ? warp.denseGridIndexCount : warp.gridIndexCount,
                indexType: .uint32, indexBuffer: dense ? warp.denseGridIndices : warp.gridIndices, indexBufferOffset: 0
            )
            encoder.endEncoding()
            return
        }

        let desc = MTLRenderPassDescriptor()
        desc.colorAttachments[0].texture = dest
        desc.colorAttachments[0].loadAction = .dontCare
        desc.colorAttachments[0].storeAction = .store
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: desc) else { return }
        encoder.setRenderPipelineState(warp.eyePipelineState)
        encoder.setFragmentTexture(source, index: 0)
        encoder.setFragmentTexture(source, index: 1) // placeholder; unused when useDepth=0
        encoder.setFragmentBytes(&uniforms, length: MemoryLayout<VideoStereoUniforms>.size, index: 0)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
        encoder.endEncoding()
    }

    /// The packed layout with its half/full tag checked against the frame's
    /// shape (see `correctingResolution`); re-checked only when the size
    /// changes.
    private func packedLayout(forFrame size: CGSize) -> PackedStereoLayout? {
        guard let packedLayout else { return nil }
        if checkedPackedSize != size {
            checkedPackedSize = size
            let corrected = packedLayout.correctingResolution(forPacked: size)
            correctedLayout = corrected
            if corrected != packedLayout {
                let tagged = packedLayout.isHalfResolution ? "half" : "full"
                let used = corrected.isHalfResolution ? "half" : "full"
                VergenceLog.pipeline.notice(
                    "packed \(Int(size.width), privacy: .public)×\(Int(size.height), privacy: .public) frame tagged \(tagged, privacy: .public) resolution reads as \(used, privacy: .public); using \(used, privacy: .public)"
                )
            }
        }
        return correctedLayout
    }

    /// One eye of a packed stereo frame: a crop of the source, stretched to
    /// the eye buffer.
    private func encodePackedEye(
        _ eye: PackedStereoLayout.Eye,
        layout: PackedStereoLayout,
        into dest: MTLTexture,
        source: MTLTexture,
        config: Config,
        commandBuffer: MTLCommandBuffer
    ) {
        let region = layout.eyeRegion(eye, mirrored: config.mirror)
        var uniforms = PackedEyeUniforms(
            brightness: config.brightness,
            contrast: config.contrast,
            saturation: config.saturation,
            mirror: config.mirror ? 1 : 0,
            regionScale: region.scale,
            regionOffset: region.offset
        )
        let desc = MTLRenderPassDescriptor()
        desc.colorAttachments[0].texture = dest
        desc.colorAttachments[0].loadAction = .dontCare
        desc.colorAttachments[0].storeAction = .store
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: desc) else { return }
        encoder.setRenderPipelineState(warp.packedEyePipelineState)
        encoder.setFragmentTexture(source, index: 0)
        encoder.setFragmentBytes(&uniforms, length: MemoryLayout<PackedEyeUniforms>.size, index: 0)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
        encoder.endEncoding()
    }

    private func ensureDepthTexture(width: Int, height: Int, eye: Int) -> MTLTexture? {
        if let existing = eyeDepthTextures[eye], existing.width == width, existing.height == height {
            return existing
        }
        let texture = device.makeTexture(descriptor: transientAttachmentDescriptor(
            pixelFormat: .depth32Float, width: width, height: height
        ))
        eyeDepthTextures[eye] = texture
        return texture
    }

    /// Multisampled color target for the MSAA mesh pass (resolved into the eye
    /// buffer). Only called when pseudo3DMSAASampleCount > 1.
    private func ensureMSAAColorTexture(width: Int, height: Int, eye: Int) -> MTLTexture? {
        if let existing = eyeMSAAColorTextures[eye], existing.width == width, existing.height == height {
            return existing
        }
        let texture = device.makeTexture(descriptor: transientAttachmentDescriptor(
            pixelFormat: .bgra8Unorm, width: width, height: height
        ))
        eyeMSAAColorTextures[eye] = texture
        return texture
    }

    /// Descriptor for a load-clear/store-discard render attachment: multisampled
    /// at the mesh pipeline's sample count and memoryless (tile memory only —
    /// contents never survive the pass, which is all the mesh pass needs from
    /// its depth buffer and its pre-resolve color).
    private func transientAttachmentDescriptor(pixelFormat: MTLPixelFormat, width: Int, height: Int) -> MTLTextureDescriptor {
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: pixelFormat, width: width, height: height, mipmapped: false
        )
        desc.usage = [.renderTarget]
        let samples = warp.msaaSampleCount
        if samples > 1 {
            desc.textureType = .type2DMultisample
            desc.sampleCount = samples
            desc.storageMode = .memoryless
        } else {
            desc.storageMode = .private
        }
        return desc
    }

    // MARK: GPU helpers

    /// (Re)creates the BGRA warp pool and the 420v output pool for the current
    /// source dimensions. Returns false if either pool can't be made.
    private func ensurePools(width: Int, height: Int) -> Bool {
        if bgraPool != nil, outPool != nil, poolWidth == width, poolHeight == height { return true }

        let bgraAttrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any]()
        ]
        var bgra: CVPixelBufferPool?
        guard CVPixelBufferPoolCreate(nil, nil, bgraAttrs as CFDictionary, &bgra) == kCVReturnSuccess else { return false }

        // 420 output pool, merged with the renderer's recommended attributes —
        // the construction that validated on-device. Full-range (420f, luma
        // 0-255) rather than video-range (420v, 16-235): our warp output is
        // full-range RGB and the compositor reads the tagged buffer as full
        // range, so video-range here lifts blacks / lowers whites (washed out).
        let eyeSize = CVImageSize(width: width, height: height)
        let defaultAttributes = CVPixelBufferCreationAttributes(
            pixelFormatType: CVPixelFormatType(rawValue: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange),
            size: eyeSize
        )
        let recommended = videoRenderer.recommendedPixelBufferAttributes
        guard let merged = CVPixelBufferAttributes(merging: [CVPixelBufferAttributes(defaultAttributes), recommended]),
              let creation = CVPixelBufferCreationAttributes(merged),
              let out = try? CVMutablePixelBuffer.Pool(pixelBufferAttributes: creation) else { return false }

        bgraPool = bgra
        outPool = out
        poolWidth = width
        poolHeight = height
        return true
    }

    private func makeTexture(from pixelBuffer: CVPixelBuffer, cache: CVMetalTextureCache) -> MTLTexture? {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        // A non-BGRA buffer binds successfully and renders garbage (see the
        // `PumpFrameSource` note), so nothing downstream can report it. Say it
        // here, once, rather than leaving a new frame source to be debugged
        // from a screenshot.
        let format = CVPixelBufferGetPixelFormatType(pixelBuffer)
        if format != kCVPixelFormatType_32BGRA, !didWarnAboutPixelFormat {
            didWarnAboutPixelFormat = true
            VergenceLog.pipeline.error(
                "frame source is not 32BGRA (\(fourCC(format), privacy: .public)) — the warp will render garbage"
            )
        }
        var cvTexture: CVMetalTexture?
        guard CVMetalTextureCacheCreateTextureFromImage(
            nil, cache, pixelBuffer, nil, .bgra8Unorm, width, height, 0, &cvTexture
        ) == kCVReturnSuccess, let cvTexture else { return nil }
        return CVMetalTextureGetTexture(cvTexture)
    }

    private func reportSizeIfNeeded(_ pixelBuffer: CVPixelBuffer) {
        let frame = CGSize(width: CVPixelBufferGetWidth(pixelBuffer), height: CVPixelBufferGetHeight(pixelBuffer))
        let size = packedLayout(forFrame: frame)?.eyeSize(packed: frame) ?? frame
        guard size.width > 0, size.height > 0, lastReportedSize != size else { return }
        lastReportedSize = size
        onVideoSizeKnown(size)
    }
}

/// An OSType reads as gibberish in decimal; these are four-character codes
/// (`BGRA`, `420f`, `420v`) and naming them is the whole point of the log.
/// File scope, not a method: an os_log interpolation is a closure, and a method
/// call inside one is an implicit `self` capture the compiler rejects.
private func fourCC(_ value: OSType) -> String {
    let bytes = [24, 16, 8, 0].map { UInt8((value >> $0) & 0xFF) }
    guard bytes.allSatisfy({ $0 >= 0x20 && $0 < 0x7F }) else { return String(value) }
    return String(bytes.map { Character(UnicodeScalar($0)) })
}

#endif

/// How well the stereo pump keeps up with its source, over the last couple
/// of seconds of playback. Reported only while frames are arriving, so a
/// paused or stalled player says nothing (a stall is the player's to report).
public struct StereoPerformance: Sendable, Equatable {
    /// Frames that reached the eyes, per second of wall time.
    public var shownFramesPerSecond: Double
    /// The source's own frame rate, from the shortest gap between frames.
    public var sourceFramesPerSecond: Double
    /// Of the source frames the playhead passed, the share that was shown
    /// (1 = every frame); faster than normal speed, of the frames the video
    /// has at normal speed in that time, as no more can be asked of the
    /// conversion (`StereoPerformance.measured`). Below ~0.75 the picture
    /// visibly stutters.
    public var deliveredFraction: Double
    /// Source seconds per wall second; well under 1 means the player itself
    /// is starving (decode or network), not the conversion.
    public var playbackRate: Double
    /// The decoded picture's height in pixels (a packed stereo frame's
    /// whole height), so a host can tell what a lower stream would save.
    /// 0 when unknown.
    public var sourceHeight: Int

    public init(shownFramesPerSecond: Double, sourceFramesPerSecond: Double,
                deliveredFraction: Double, playbackRate: Double, sourceHeight: Int = 0) {
        self.shownFramesPerSecond = shownFramesPerSecond
        self.sourceFramesPerSecond = sourceFramesPerSecond
        self.deliveredFraction = deliveredFraction
        self.playbackRate = playbackRate
        self.sourceHeight = sourceHeight
    }

    /// The conversion, not the source, is what's falling behind.
    public var isOverloaded: Bool {
        deliveredFraction < 0.75 && playbackRate > 0.8
    }
}
