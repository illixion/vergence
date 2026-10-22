#if canImport(Metal)
import AVFoundation
import CoreMedia
import CoreVideo
import Foundation
import VergenceCore

/// One decoded frame on its way to the warp.
public struct PumpFrame {
    public let pixelBuffer: CVPixelBuffer
    /// Presentation time on the *source's* own timeline — what a cached depth
    /// source matches its baked depth against, and what a realtime source uses
    /// to decide whether a held map may be reused. Not the renderer timeline.
    public let itemTime: CMTime

    public init(pixelBuffer: CVPixelBuffer, itemTime: CMTime) {
        self.pixelBuffer = pixelBuffer
        self.itemTime = itemTime
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
/// **Frames must be `kCVPixelFormatType_32BGRA` and Metal-compatible.** The pump
/// binds plane 0 as one `.bgra8Unorm` texture, so a 4:2:0 buffer does not fail —
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

/// Pulls from an `AVPlayerItemVideoOutput` — the original Spatial Stash path,
/// where AVPlayer stays the clock, the audio source and the transport, and only
/// pixels are diverted.
public final class AVPlayerFrameSource: PumpFrameSource, @unchecked Sendable {
    private let output: AVPlayerItemVideoOutput

    public init(output: AVPlayerItemVideoOutput) {
        self.output = output
    }

    public func nextFrame(hostTime: CFTimeInterval) -> PumpFrame? {
        let itemTime = output.itemTime(forHostTime: hostTime)
        var presentationTime = CMTime.invalid
        guard output.hasNewPixelBuffer(forItemTime: itemTime),
              let buffer = output.copyPixelBuffer(forItemTime: itemTime, itemTimeForDisplay: &presentationTime) else {
            return nil
        }
        // The frame's own time, not the tick's: the tick runs up to a tick
        // behind the frame, so cached depth could match the next frame's,
        // and frame gaps would be the ticks' spacing.
        return PumpFrame(pixelBuffer: buffer, itemTime: presentationTime.isNumeric ? presentationTime : itemTime)
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
