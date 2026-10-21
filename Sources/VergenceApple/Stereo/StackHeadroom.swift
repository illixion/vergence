/*
 Vergence - StackHeadroom.

 Diagnostic for a stack overflow on the stereo-pump queue (Hypnos on Vision
 Pro, 2026-10-21): the pump's worker thread ran out of its 512 KB stack inside
 an os_log call from `RealtimeDepthRefiner`, yet every frame on the reported
 call path is under ~5 KB. `check(_:)` measures how much of the current
 thread's stack is left at a named point and logs it once it drops below
 `threshold`, again only at each new low, so a run shows where the stack goes
 without flooding the log. The first check on each pump also logs a baseline.
 */

#if canImport(Metal)
import Foundation
import VergenceCore

enum StackHeadroom {
    /// Free stack below which a point is reported.
    static let threshold = 128 * 1024
    /// A new report needs a low this much below the last one.
    private static let step = 16 * 1024

    private static let lowest = VergenceLock<Int>(.max)

    /// Bytes left between the caller's frame and the bottom of this thread's
    /// stack, with the stack's total size. Nil when the bounds are unknown.
    static func current() -> (free: Int, size: Int)? {
        let thread = pthread_self()
        let top = UInt(bitPattern: pthread_get_stackaddr_np(thread))
        let size = pthread_get_stacksize_np(thread)
        guard top > 0, size > 0 else { return nil }
        var marker: UInt8 = 0
        let here = withUnsafeMutablePointer(to: &marker) { UInt(bitPattern: $0) }
        guard here <= top, here > top - UInt(size) else { return nil }
        return (Int(here - (top - UInt(size))), size)
    }

    /// Logs the stack in use at `point` the first time a pump calls it.
    static func baseline(_ point: StaticString) {
        guard let stack = current() else { return }
        VergenceLog.pipeline.notice("Stack at \(String(describing: point), privacy: .public): \((stack.size - stack.free) / 1024, privacy: .public) KB used of \(stack.size / 1024, privacy: .public) KB")
    }

    /// Logs `point` when the free stack there is below `threshold` and a new low.
    @inline(__always)
    static func check(_ point: StaticString) {
        guard let stack = current(), stack.free < threshold else { return }
        let report = lowest.withLock { low in
            guard stack.free + step <= low || low == .max else { return false }
            low = stack.free
            return true
        }
        guard report else { return }
        VergenceLog.pipeline.error("Stack low at \(String(describing: point), privacy: .public): \(stack.free / 1024, privacy: .public) KB free of \(stack.size / 1024, privacy: .public) KB, thread \(Thread.current.name ?? "", privacy: .public) queue \(String(cString: __dispatch_queue_get_label(nil)), privacy: .public)")
    }
}
#endif
