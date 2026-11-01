#if canImport(CoreMedia)
import CoreMedia
import CoreVideo
import Foundation
import Testing
@testable import VergenceApple

/// The interpolating wrapper driven the way `StereoPump.takeFrames` drives a
/// lookahead source: 60 Hz ticks, each asking `LookaheadPull.target` past the
/// last frame taken until the source answers nil, with the spacing learned
/// from the frames it takes. The fake interpolator answers on the work queue
/// with a frame whose luma is the average of its inputs'.
@Suite("Interpolating frame source")
struct InterpolatingFrameSourceTests {
    /// Frame k at k / fps, its luma `luma(k)`. Hands out the frame showing at
    /// the asked time when it differs from the last one handed out, as
    /// `AVPlayerItemVideoOutput` does.
    final class FakeSource: LookaheadFrameSource, @unchecked Sendable {
        let fps: Double
        let luma: (Int) -> UInt8
        var lastGiven: Int?
        var asked: [Double] = []
        init(fps: Double, luma: @escaping (Int) -> UInt8 = { _ in 128 }) {
            self.fps = fps
            self.luma = luma
        }
        func itemTime(forHostTime hostTime: CFTimeInterval) -> CMTime { CMTime(seconds: hostTime, preferredTimescale: 600_000) }
        func dueHostTime(forItemTime itemTime: CMTime, at hostTime: CFTimeInterval) -> CFTimeInterval? { itemTime.seconds }
        func nextFrame(hostTime: CFTimeInterval) -> PumpFrame? { nil }
        func invalidate() {}
        func frame(atItemTime itemTime: CMTime) -> PumpFrame? {
            asked.append(itemTime.seconds)
            let k = Int((itemTime.seconds * fps + 1e-6).rounded(.down))
            guard k >= 0, k != lastGiven else { return nil }
            lastGiven = k
            return PumpFrame(pixelBuffer: makeFrame(luma: luma(k)), itemTime: CMTime(value: Int64(k) * 1000, timescale: Int32(fps * 1000)))
        }
    }

    final class FakeInterpolator: MidFrameInterpolator, @unchecked Sendable {
        var made = 0
        func prepare(width: Int, height: Int, scale: Double) -> Bool { true }
        func midFrame(between previous: CVPixelBuffer, and current: CVPixelBuffer,
                      completion: @escaping @Sendable (CVPixelBuffer?, Double) -> Void) {
            made += 1
            let luma = (Int(firstByte(previous)) + Int(firstByte(current))) / 2
            completion(makeFrame(luma: UInt8(luma)), 1)
        }
    }

    static func makeFrame(luma: UInt8) -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        CVPixelBufferCreate(nil, 64, 64, kCVPixelFormatType_32BGRA, nil, &buffer)
        CVPixelBufferLockBaseAddress(buffer!, [])
        memset(CVPixelBufferGetBaseAddress(buffer!), Int32(luma), CVPixelBufferGetDataSize(buffer!))
        CVPixelBufferUnlockBaseAddress(buffer!, [])
        return buffer!
    }

    static func firstByte(_ buffer: CVPixelBuffer) -> UInt8 {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        return CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self).pointee
    }

    struct Taken {
        let seconds: Double
        let luma: UInt8
    }

    /// Plays `from`..<`to` seconds of item time, one 60 Hz tick at a time,
    /// with a `lead` of frames taken ahead. Each tick waits a moment so the
    /// work queue can answer.
    static func play(_ source: LookaheadFrameSource, from start: Double = 0, to end: Double,
                     lead: Double = 0.2, taken: inout [Taken], spacing: inout SourceFrameSpacing,
                     last: inout Double?) {
        var playhead = start
        while playhead < end {
            let windowEnd = playhead + lead
            for _ in 0 ..< 40 {
                let s = spacing.seconds
                let target = LookaheadPull.target(last: last, spacing: s.isFinite ? s : 1.0 / 60,
                                                  playhead: playhead, windowEnd: windowEnd)
                if let last, target <= last + 1e-6 { break }
                guard let frame = source.frame(atItemTime: CMTime(seconds: target, preferredTimescale: 600_000)) else { break }
                let seconds = frame.itemTime.seconds
                if let last { spacing.note(gap: seconds - last) }
                last = seconds
                taken.append(Taken(seconds: seconds, luma: firstByte(frame.pixelBuffer)))
            }
            Thread.sleep(forTimeInterval: 0.002)
            playhead += 1.0 / 60
        }
    }

    static func play(_ source: LookaheadFrameSource, to end: Double) -> [Taken] {
        var taken: [Taken] = []
        var spacing = SourceFrameSpacing()
        var last: Double?
        play(source, to: end, taken: &taken, spacing: &spacing, last: &last)
        return taken
    }

    @Test("a 24 fps source plays at 48 fps, each made frame halfway between its neighbours")
    func doublesTwentyFour() {
        let inner = FakeSource(fps: 24, luma: { UInt8(10 + 4 * $0) })
        let source = InterpolatingFrameSource(wrapping: inner, interpolator: FakeInterpolator())
        let taken = Self.play(source, to: 2)
        // Past the first few frames (spacing learned, interpolator prepared),
        // every step is half a source frame, in order, with no gap.
        let settled = taken.drop { $0.seconds < 0.5 }
        #expect(settled.count > 50)
        let steps = zip(settled, settled.dropFirst()).map { $1.seconds - $0.seconds }
        #expect(steps.allSatisfy { abs($0 - 1.0 / 48) < 0.001 })
        for frame in settled {
            let k = frame.seconds * 24
            let expected = 10 + 4 * k
            #expect(abs(Double(frame.luma) - expected) <= 1, "luma at \(frame.seconds)")
        }
    }

    @Test("a cut is never interpolated across")
    func cutPassesThrough() {
        let inner = FakeSource(fps: 24, luma: { $0 < 30 ? 60 : 200 })
        let source = InterpolatingFrameSource(wrapping: inner, interpolator: FakeInterpolator())
        let taken = Self.play(source, to: 2)
        let cut = 30.0 / 24
        let before = taken.last { $0.seconds < cut - 1e-6 }
        let at = taken.first { $0.seconds >= cut - 1e-6 }
        #expect(before.map { abs($0.seconds - 29.0 / 24) < 1e-6 } == true)
        #expect(at.map { abs($0.seconds - cut) < 1e-6 && $0.luma == 200 } == true)
        #expect(!taken.contains { $0.luma != 60 && $0.luma != 200 })
    }

    @Test("a 60 fps source passes through untouched")
    func fastSourcePassesThrough() {
        let inner = FakeSource(fps: 60)
        let interpolator = FakeInterpolator()
        let source = InterpolatingFrameSource(wrapping: inner, interpolator: interpolator)
        let taken = Self.play(source, to: 1)
        #expect(interpolator.made == 0)
        let steps = zip(taken, taken.dropFirst()).map { $1.seconds - $0.seconds }
        #expect(steps.allSatisfy { abs($0 - 1.0 / 60) < 0.001 })
    }

    @Test("after a seek no frame from before it comes out")
    func seekDropsOldFrames() {
        let inner = FakeSource(fps: 24, luma: { UInt8(($0 * 3) % 250) })
        let source = InterpolatingFrameSource(wrapping: inner, interpolator: FakeInterpolator())
        var taken: [Taken] = []
        var spacing = SourceFrameSpacing()
        var last: Double?
        Self.play(source, to: 1, taken: &taken, spacing: &spacing, last: &last)
        // The pump starts over from the new playhead.
        last = nil
        inner.lastGiven = nil
        var after: [Taken] = []
        Self.play(source, from: 5, to: 6, taken: &after, spacing: &spacing, last: &last)
        #expect(after.first.map { abs($0.seconds - 5) < 1e-6 } == true)
        #expect(after.allSatisfy { $0.seconds >= 5 - 1e-6 })
        let steps = zip(after.dropFirst(4), after.dropFirst(5)).map { $1.seconds - $0.seconds }
        #expect(steps.allSatisfy { abs($0 - 1.0 / 48) < 0.001 })
    }
}

#if !targetEnvironment(simulator)
/// The real VideoToolbox interpolator, with its format conversions, on this
/// Mac: a frame comes back, BGRA, the source's size and colour tags, its luma
/// between the two inputs'.
@Suite("VideoToolbox mid-frame interpolator")
struct VTMidFrameInterpolatorTests {
    @Test("makes a 1080p frame between two others, at full size and made at two thirds",
          arguments: [1.0, 2.0 / 3])
    func makesMidFrame(scale: Double) async throws {
        guard #available(macOS 26.0, *), VTMidFrameInterpolator.isSupported else { return }
        let interpolator = VTMidFrameInterpolator()
        #expect(interpolator.prepare(width: 1920, height: 1080, scale: scale))
        func frame(_ luma: UInt8) -> CVPixelBuffer {
            var buffer: CVPixelBuffer?
            let attributes = [kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any]()] as CFDictionary
            CVPixelBufferCreate(nil, 1920, 1080, kCVPixelFormatType_32BGRA, attributes, &buffer)
            CVPixelBufferLockBaseAddress(buffer!, [])
            memset(CVPixelBufferGetBaseAddress(buffer!), Int32(luma), CVPixelBufferGetDataSize(buffer!))
            CVPixelBufferUnlockBaseAddress(buffer!, [])
            CVBufferSetAttachment(buffer!, kCVImageBufferColorPrimariesKey, kCVImageBufferColorPrimaries_ITU_R_709_2, .shouldPropagate)
            CVBufferSetAttachment(buffer!, kCVImageBufferTransferFunctionKey, kCVImageBufferTransferFunction_ITU_R_709_2, .shouldPropagate)
            return buffer!
        }
        let previous = frame(60), current = frame(200)
        struct Result: @unchecked Sendable { let frame: CVPixelBuffer?; let ms: Double }
        let result: Result = await withCheckedContinuation { continuation in
            interpolator.midFrame(between: previous, and: current) { mid, ms in
                continuation.resume(returning: Result(frame: mid, ms: ms))
            }
        }
        let mid = try #require(result.frame)
        #expect(CVPixelBufferGetPixelFormatType(mid) == kCVPixelFormatType_32BGRA)
        #expect(CVPixelBufferGetWidth(mid) == 1920 && CVPixelBufferGetHeight(mid) == 1080)
        #expect(CVBufferCopyAttachment(mid, kCVImageBufferTransferFunctionKey, nil) != nil)
        CVPixelBufferLockBaseAddress(mid, .readOnly)
        let row = CVPixelBufferGetBaseAddress(mid)!.assumingMemoryBound(to: UInt8.self) + 540 * CVPixelBufferGetBytesPerRow(mid)
        let green = row[960 * 4 + 1]
        CVPixelBufferUnlockBaseAddress(mid, .readOnly)
        #expect(green > 60 && green < 200, "mid green \(green)")
        print("VT mid-frame at \(Int(scale * 100))%: \(String(format: "%.1f", result.ms)) ms, green \(green)")
    }
}
#endif

@Suite("Interpolation governor")
struct InterpolationGovernorTests {
    /// A 24 fps source's spacing.
    static let budget = 1000.0 / 24

    @Test("a low reserve steps down to a smaller size first, then pauses; a refilled one resumes small")
    func stepsDownThenPauses() {
        var governor = InterpolationGovernor()
        func update(_ reserve: Double?, _ now: Double) -> String? {
            governor.update(reserve: reserve, medianMs: nil, budgetMs: Self.budget, now: now)
        }
        #expect(update(0.05, 0) == nil) // starting: grace
        #expect(update(0.05, 2) == nil)
        #expect(update(0.6, 4) == nil)
        #expect(update(0.1, 5)?.hasPrefix("stepped down to 66%") == true)
        #expect(!governor.isPaused && governor.scale < 1)
        #expect(update(0.1, 8) == nil) // one step per cooldown
        #expect(update(0.1, 13.1)?.hasPrefix("paused") == true)
        #expect(governor.isPaused)
        #expect(update(0.3, 21.2) == nil) // not refilled
        #expect(update(0.6, 21.3)?.hasPrefix("resumed at 66%") == true)
        #expect(!governor.isPaused && governor.scale < 1)
    }

    @Test("interpolation taking most of a frame's time steps down; headroom later steps back up")
    func timingsDriveTheSize() {
        var governor = InterpolationGovernor()
        #expect(governor.update(reserve: 0.6, medianMs: 36, budgetMs: Self.budget, now: 0)?
            .hasPrefix("stepped down") == true)
        #expect(governor.update(reserve: 0.6, medianMs: 12, budgetMs: Self.budget, now: 10) == nil)
        #expect(governor.update(reserve: 0.6, medianMs: 12, budgetMs: Self.budget, now: 20.1)?
            .hasPrefix("stepped up to 100%") == true)
        #expect(governor.scale == 1)
    }

    @Test("a device that keeps up keeps interpolating at full size, warm or not")
    func keepsUp() {
        var governor = InterpolationGovernor()
        for t in stride(from: 0.0, to: 120, by: 0.5) {
            #expect(governor.update(reserve: 0.4, medianMs: 25, budgetMs: Self.budget, now: t) == nil)
        }
        #expect(governor.state == "on")
    }

    @Test("pausing again after a pause and resume starts a new grace, not a step down")
    func resumeFromPauseHasGrace() {
        var governor = InterpolationGovernor()
        _ = governor.update(reserve: 0.6, medianMs: nil, budgetMs: Self.budget, now: 0)
        _ = governor.update(reserve: nil, medianMs: nil, budgetMs: Self.budget, now: 10) // player paused
        #expect(governor.update(reserve: 0.05, medianMs: nil, budgetMs: Self.budget, now: 20) == nil)
        #expect(governor.state == "on")
    }

    @Test("frames already late pause it at once; three pauses in a minute turn it off")
    func lateAndOff() {
        var governor = InterpolationGovernor()
        func update(_ reserve: Double, _ now: Double) -> String? {
            governor.update(reserve: reserve, medianMs: nil, budgetMs: Self.budget, now: now)
        }
        _ = update(0.6, 0)
        var t = 4.0
        #expect(update(-0.01, t)?.hasPrefix("paused") == true)
        for _ in 0 ..< 2 {
            t += 9
            #expect(update(0.6, t)?.hasPrefix("resumed") == true)
            t += 1
            _ = update(-0.01, t)
        }
        #expect(governor.state == "off")
        #expect(update(0.9, t + 30) == nil)
        #expect(governor.isPaused)
    }
}
#endif
