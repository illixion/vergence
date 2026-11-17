/*
 Vergence - the frame source of a looping item

 `AVPlayerLooper` plays a file as copies of its item queued back to back, so
 a loop has no seek and no gap however short the clip. Each copy has its own
 frame output, so whatever reads frames has to change source at every pass.
 This holds the current copy's source and swaps it under its readers: an
 interpolator keeps its session and a pump its timing, and a new `sequence`
 makes them drop what they held, as after a seek. Rebuilding them instead
 restarted the interpolator every pass (Hypnos, 2026-11-16: 50 ms of each
 0.6 s pass undoubled, and the display criteria stated again).
 */

#if canImport(Metal)
import AVFoundation
import CoreMedia
import Foundation

public final class LoopCopyFrameSource: LookaheadFrameSource, @unchecked Sendable {
    private let lock = NSLock()
    private var inner: any LookaheadFrameSource
    private var passes = 0

    public init(_ inner: any LookaheadFrameSource) { self.inner = inner }

    /// The next pass's copy.
    public func follow(_ next: any LookaheadFrameSource) {
        lock.withLock {
            inner = next
            passes += 1
        }
    }

    private var current: any LookaheadFrameSource { lock.withLock { inner } }

    public var decodesAheadWhilePaused: Bool { current.decodesAheadWhilePaused }
    public var endTime: Double? { current.endTime }
    public var sequence: Int { lock.withLock { passes &* 1_000_003 &+ inner.sequence } }
    public var isFinished: Bool { current.isFinished }
    public func nextFrame(hostTime: CFTimeInterval) -> PumpFrame? { current.nextFrame(hostTime: hostTime) }
    public func playbackRate(hostTime: CFTimeInterval) -> Double? { current.playbackRate(hostTime: hostTime) }
    public func itemTime(forHostTime hostTime: CFTimeInterval) -> CMTime { current.itemTime(forHostTime: hostTime) }
    public func frame(atItemTime itemTime: CMTime) -> PumpFrame? { current.frame(atItemTime: itemTime) }
    public func dueHostTime(forItemTime itemTime: CMTime, at hostTime: CFTimeInterval) -> CFTimeInterval? {
        current.dueHostTime(forItemTime: itemTime, at: hostTime)
    }
    public func invalidate() { current.invalidate() }
}
#endif
