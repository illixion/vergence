#if canImport(Metal)
import AVFoundation
import CoreMedia
import CoreVideo
import Foundation
import VergenceCore

/// One decoded frame on its way to the warp. `@unchecked Sendable`: a
/// decoded pixel buffer is only read once made, and the frame goes from the
/// pump queue to the depth model's.
public struct PumpFrame: @unchecked Sendable {
    public let pixelBuffer: CVPixelBuffer
    /// Presentation time on the *source's* own timeline — what a cached depth
    /// source matches its baked depth against, and what a realtime source uses
    /// to decide whether a held map may be reused. Not the renderer timeline.
    public let itemTime: CMTime
    /// A real frame from a doubling source (`InterpolatingFrameSource`) that
    /// came without the in-between frame before it, by design: a cut, a pair
    /// the interpolator backed off from or had no fields for. The pump
    /// counts it as both frames when it reports how many it showed, so
    /// passing frames through doesn't read as the pump falling behind.
    public var standsInForMidFrame = false

    public init(pixelBuffer: CVPixelBuffer, itemTime: CMTime, standsInForMidFrame: Bool = false) {
        self.pixelBuffer = pixelBuffer
        self.itemTime = itemTime
        self.standsInForMidFrame = standsInForMidFrame
    }
}

/// Dolby Vision frames only the system's renderer can show. A profile 5
/// picture is IPT-PQ-c2: without its RPU applied it decodes to purple and
/// green, and AVFoundation hands decoded frames over with the RPU attached,
/// not applied. Its frames carry `DolbyVisionRPUData` and no colour primaries
/// (8.1 states BT.2020, and shows correctly as its HDR10 base). Nothing here
/// applies an RPU; a host that wants such a picture in 3D gets it converted
/// to HDR10 elsewhere (a server).
public enum DolbyVisionFrames {
    public static func needsSystemRenderer(_ pixelBuffer: CVPixelBuffer) -> Bool {
        CVBufferCopyAttachment(pixelBuffer, "DolbyVisionRPUData" as CFString, nil) != nil
            && CVBufferCopyAttachment(pixelBuffer, kCVImageBufferColorPrimariesKey, nil) == nil
    }
}

/// What a source frame says about its picture: its transfer function
/// (`kCVImageBufferTransferFunctionKey`, e.g. "SMPTE_ST_2084_PQ") and whether
/// a Dolby Vision RPU rides on it. A host reads this to say what plays (HDR10,
/// HLG, SDR); the warp keeps the transfer in both eyes but never applies the
/// RPU.
public struct FrameColorSignal: Sendable, Equatable {
    public var transferFunction: String?
    public var hasDolbyVisionRPU: Bool

    public init(transferFunction: String?, hasDolbyVisionRPU: Bool) {
        self.transferFunction = transferFunction
        self.hasDolbyVisionRPU = hasDolbyVisionRPU
    }

    public init(_ pixelBuffer: CVPixelBuffer) {
        transferFunction = CVBufferCopyAttachment(pixelBuffer, kCVImageBufferTransferFunctionKey, nil) as? String
        hasDolbyVisionRPU = CVBufferCopyAttachment(pixelBuffer, "DolbyVisionRPUData" as CFString, nil) != nil
    }
}

/// Where `StereoPump` gets frames. Called only on the pump queue (including
/// `invalidate`, from the stop-drain block), at the pump's tick rate.
///
/// The two implementations differ in kind, which is why this is a protocol and
/// not a parameter: a pull source is asked for the frame matching *now* and
/// answers nil when nothing new has decoded, while a push source hands over
/// whatever its decoder most recently deposited. Both must return nil rather
/// than repeat a frame the pump has already seen — a tick that re-warps the
/// same frame costs a full GPU pass and enqueues a duplicate.
///
/// **Frames must be `kCVPixelFormatType_32BGRA` (or, for an HDR picture,
/// `kCVPixelFormatType_64RGBAHalf`, `FramePixelFormat`) and Metal-compatible.**
/// The pump binds plane 0 as one texture, so a 4:2:0 buffer does not fail —
/// it renders, with each texel eating four luma bytes, which comes out as four
/// side-by-side greyscale copies of the frame at a quarter width each. Configure
/// `AVPlayerItemVideoOutput` or the `VTDecompressionSession` accordingly;
/// `makeTexture` logs the format once if it sees anything else.
public protocol PumpFrameSource: AnyObject, Sendable {
    /// The host permanently closed this source. A paused source stays open.
    var isFinished: Bool { get }
    /// The newest not-yet-delivered frame, or nil if none is ready.
    /// - Parameter hostTime: the pump's `CACurrentMediaTime()` for this tick, so
    ///   a source driven by a host-synchronised clock (AVPlayer) can map it onto
    ///   its own timeline. Push sources ignore it.
    func nextFrame(hostTime: CFTimeInterval) -> PumpFrame?
    /// Source seconds per wall second at `hostTime`: 2 at double speed, 0
    /// paused. Nil when the source can't tell (a push source fed at its own
    /// pace), which the pump takes as normal speed.
    func playbackRate(hostTime: CFTimeInterval) -> Double?
    /// Release decode resources; the pump is shutting down.
    func invalidate()
}

public extension PumpFrameSource {
    var isFinished: Bool { false }
    func playbackRate(hostTime: CFTimeInterval) -> Double? { nil }
}

/// A pull source on a host-synchronised clock that can hand out frames before
/// they are due. The pump takes each frame a little ahead (time for its depth
/// and warp), tags it with the host time it is due, and the renderer shows it
/// then, in step with the audio. A source that can't answer these (a push
/// source) gets each frame shown once drawn, or with real-time depth a fixed
/// delay after it came (`StereoPump.pushDelay`).
public protocol LookaheadFrameSource: PumpFrameSource {
    /// End of the finite media timeline, to avoid refilling past EOF.
    var endTime: Double? { get }
    /// Changes after a flush, including a seek to the same position.
    var sequence: Int { get }
    /// Whether frames ahead of the playhead keep coming while the clock is
    /// stopped, so a paused clock can refill the pump's reserve. False means
    /// the pump never holds the clock for depth (see
    /// `StereoPump.updateProcessingHold`).
    var decodesAheadWhilePaused: Bool { get }
    /// The source's item time at `hostTime`; at a future host time, where the
    /// clock will be if it keeps its current rate.
    func itemTime(forHostTime hostTime: CFTimeInterval) -> CMTime
    /// The newest not-yet-taken frame at or before `itemTime`, or nil.
    func frame(atItemTime itemTime: CMTime) -> PumpFrame?
    /// The host time the frame at `itemTime` is due on screen, on the clock
    /// as it runs at `hostTime`; nil while the clock is stopped (paused,
    /// stalled).
    func dueHostTime(forItemTime itemTime: CMTime, at hostTime: CFTimeInterval) -> CFTimeInterval?
}

public extension LookaheadFrameSource {
    var endTime: Double? { nil }
    var sequence: Int { 0 }
    var decodesAheadWhilePaused: Bool { true }
}

/// Pulls from an `AVPlayerItemVideoOutput` — the original Spatial Stash path,
/// where AVPlayer stays the clock, the audio source and the transport, and only
/// pixels are diverted. The output decodes well ahead of the playhead (0.5-1 s
/// measured on device, `DepthPipelineBenchmark`), so frames can be taken early.
public final class AVPlayerFrameSource: LookaheadFrameSource, @unchecked Sendable {
    /// A paused player decodes nothing new: the output then has the frame at
    /// the playhead and whatever it decoded before the pause (about 0.4 s;
    /// nothing ahead before the player first plays), measured on a Mac
    /// 2026-10-27 with the same output settings.
    public let decodesAheadWhilePaused = false
    private let lock = NSLock()
    private var currentOutput: AVPlayerItemVideoOutput
    private var output: AVPlayerItemVideoOutput { lock.withLock { currentOutput } }
    private weak var item: AVPlayerItem?
    /// Swap a BGRA output for a half-float one once a frame says PQ/HLG.
    private let upgradesForHDR: Bool
    private var upgradeStarted = false
    public var endTime: Double? {
        guard let value = item?.duration.seconds, value.isFinite, value > 0 else { return nil }
        return value
    }

    /// - Parameter upgradesForHDR: once a frame says PQ or HLG, the BGRA output
    ///   is replaced by a half-float one on the item (`FramePixelFormat.hdr`):
    ///   AVFoundation hands HDR frames over still encoded, and 8 bits of PQ
    ///   band. Nothing says whether a stream is HDR before its first frame
    ///   (an HLS item has no track format until it plays), so the first few
    ///   frames come as BGRA. Needs `item`.
    public init(output: AVPlayerItemVideoOutput, item: AVPlayerItem? = nil, upgradesForHDR: Bool = false) {
        self.item = item
        self.currentOutput = output
        self.upgradesForHDR = upgradesForHDR
    }

    /// The output now in use (it changes once for HDR, `upgradesForHDR`).
    public var videoOutput: AVPlayerItemVideoOutput { output }

    private func upgradeIfHDR(_ buffer: CVPixelBuffer) {
        guard upgradesForHDR, CVPixelBufferGetPixelFormatType(buffer) == FramePixelFormat.sdr,
              FramePixelFormat.forTransfer(CVBufferCopyAttachment(buffer, kCVImageBufferTransferFunctionKey, nil),
                                           width: CVPixelBufferGetWidth(buffer),
                                           height: CVPixelBufferGetHeight(buffer)) == FramePixelFormat.hdr
        else { return }
        let start = lock.withLock { () -> Bool in
            guard !upgradeStarted else { return false }
            upgradeStarted = true
            return true
        }
        guard start else { return }
        DispatchQueue.main.async { [weak self] in
            guard let self, let item = self.item else { return }
            let output = AVPlayerItemVideoOutput(pixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: FramePixelFormat.hdr,
                kCVPixelBufferMetalCompatibilityKey as String: true,
            ])
            item.add(output)
            let old = self.lock.withLock { () -> AVPlayerItemVideoOutput in
                let old = self.currentOutput
                self.currentOutput = output
                return old
            }
            item.remove(old)
            VergenceLog.pipeline.notice("frame source: HDR picture, frames now half float")
        }
    }

    public func nextFrame(hostTime: CFTimeInterval) -> PumpFrame? {
        frame(atItemTime: output.itemTime(forHostTime: hostTime))
    }

    public func itemTime(forHostTime hostTime: CFTimeInterval) -> CMTime {
        output.itemTime(forHostTime: hostTime)
    }

    public func frame(atItemTime itemTime: CMTime) -> PumpFrame? {
        var presentationTime = CMTime.invalid
        guard itemTime.isNumeric, output.hasNewPixelBuffer(forItemTime: itemTime),
              let buffer = output.copyPixelBuffer(forItemTime: itemTime, itemTimeForDisplay: &presentationTime) else {
            return nil
        }
        upgradeIfHDR(buffer)
        // The frame's own time, not the one asked for: that runs up to a
        // frame past it, so cached depth could match the next frame's, and
        // frame gaps would be the asking's spacing.
        return PumpFrame(pixelBuffer: buffer, itemTime: presentationTime.isNumeric ? presentationTime : itemTime)
    }

    /// The output maps host time to item time only, along a line at the
    /// clock's current rate; this reads the line backwards.
    public func dueHostTime(forItemTime itemTime: CMTime, at hostTime: CFTimeInterval) -> CFTimeInterval? {
        let now = output.itemTime(forHostTime: hostTime)
        let later = output.itemTime(forHostTime: hostTime + 1)
        guard itemTime.isNumeric, now.isNumeric, later.isNumeric else { return nil }
        let rate = (later - now).seconds
        guard rate > 0.01 else { return nil }
        return hostTime + (itemTime - now).seconds / rate
    }

    /// From the item's clock: how far it moves over the last tenth of a second.
    public func playbackRate(hostTime: CFTimeInterval) -> Double? {
        let now = output.itemTime(forHostTime: hostTime)
        let before = output.itemTime(forHostTime: hostTime - 0.1)
        guard now.isNumeric, before.isNumeric else { return nil }
        return (now - before).seconds / 0.1
    }

    public func invalidate() {}
}
#endif
