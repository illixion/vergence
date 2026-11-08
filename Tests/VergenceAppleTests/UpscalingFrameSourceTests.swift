#if canImport(CoreMedia)
import CoreMedia
import CoreVideo
import Foundation
import Testing
@testable import VergenceApple

/// The upscaling wrapper driven the way `InterpolatedFramePresenter` and the
/// stereo pump drive a lookahead source (60 Hz ticks, a lead of frames taken
/// ahead), over a fake output whose frames are 64×64 and a fake upscaler that
/// doubles them on its work queue.
@Suite("Upscaling frame source")
struct UpscalingFrameSourceTests {
    /// Frame k at k / fps, `format` (BGRA or half-float RGBA), its first byte
    /// `luma(k)`; hands each frame out once, as `AVPlayerItemVideoOutput` does.
    final class FakeSource: LookaheadFrameSource, @unchecked Sendable {
        let fps: Double
        let format: OSType
        let luma: (Int) -> UInt8
        var lastGiven: Int?
        init(fps: Double, format: OSType = kCVPixelFormatType_32BGRA, luma: @escaping (Int) -> UInt8 = { UInt8($0 % 250) }) {
            self.fps = fps
            self.format = format
            self.luma = luma
        }
        func itemTime(forHostTime hostTime: CFTimeInterval) -> CMTime { CMTime(seconds: hostTime, preferredTimescale: 600_000) }
        func dueHostTime(forItemTime itemTime: CMTime, at hostTime: CFTimeInterval) -> CFTimeInterval? { itemTime.seconds }
        func nextFrame(hostTime: CFTimeInterval) -> PumpFrame? { nil }
        func invalidate() {}
        func frame(atItemTime itemTime: CMTime) -> PumpFrame? {
            let k = Int((itemTime.seconds * fps + 1e-6).rounded(.down))
            guard k >= 0, k != lastGiven, k > (lastGiven ?? -1) else { return nil }
            lastGiven = k
            return PumpFrame(pixelBuffer: makeFrame(64, 64, format: format, luma: luma(k)),
                             itemTime: CMTime(value: Int64(k) * 1000, timescale: Int32(fps * 1000)))
        }
    }

    final class FakeUpscaler: RealtimeFrameUpscaler, @unchecked Sendable {
        let ms: Double
        var made = 0
        var prepares = 0
        init(ms: Double = 1) { self.ms = ms }
        func outputSize(width: Int, height: Int) -> (width: Int, height: Int)? {
            width <= 64 ? (width * 2, height * 2) : nil
        }
        func prepare(width: Int, height: Int) -> Bool {
            prepares += 1
            return true
        }
        func upscale(_ frame: CVPixelBuffer, completion: @escaping @Sendable (CVPixelBuffer?, Double) -> Void) {
            made += 1
            completion(makeFrame(CVPixelBufferGetWidth(frame) * 2, CVPixelBufferGetHeight(frame) * 2,
                                 luma: firstByte(frame)), ms)
        }
    }

    static func makeFrame(_ width: Int, _ height: Int, format: OSType = kCVPixelFormatType_32BGRA,
                          luma: UInt8) -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        CVPixelBufferCreate(nil, width, height, format, nil, &buffer)
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

    /// Plays `0..<end` seconds at 60 Hz ticks with `lead` taken ahead, each
    /// tick asking past the last frame taken until the source answers nil.
    static func play(_ source: LookaheadFrameSource, to end: Double, lead: Double = 0.15) -> [PumpFrame] {
        var taken: [PumpFrame] = []
        var spacing = SourceFrameSpacing()
        var last: Double?
        var playhead = 0.0
        while playhead < end {
            for _ in 0 ..< 40 {
                let s = spacing.seconds
                let target = LookaheadPull.target(last: last, spacing: s.isFinite ? s : 1.0 / 60,
                                                  playhead: playhead, windowEnd: playhead + lead)
                if let last, target <= last + 1e-6 { break }
                guard let frame = source.frame(atItemTime: CMTime(seconds: target, preferredTimescale: 600_000)) else { break }
                let seconds = frame.itemTime.seconds
                if let last { spacing.note(gap: seconds - last) }
                last = seconds
                taken.append(frame)
            }
            Thread.sleep(forTimeInterval: 0.002)
            playhead += 1.0 / 60
        }
        return taken
    }

    @Test("every frame comes out upscaled once the model is ready, in order, none skipped or repeated")
    func upscalesEveryFrame() {
        let upscaler = FakeUpscaler()
        let source = UpscalingFrameSource(wrapping: FakeSource(fps: 24), upscaler: upscaler)
        let taken = Self.play(source, to: 2)
        let steps = zip(taken, taken.dropFirst()).map { $1.itemTime.seconds - $0.itemTime.seconds }
        #expect(steps.allSatisfy { abs($0 - 1.0 / 24) < 0.001 }, "steps \(steps)")
        #expect(taken.count >= 46)
        // Only the first frame passes through as decoded (while it prepares).
        let upscaled = taken.filter { CVPixelBufferGetWidth($0.pixelBuffer) == 128 }
        #expect(upscaled.count >= taken.count - 2)
        #expect(upscaler.prepares == 1)
        for frame in upscaled {
            #expect(frame.sourceSize == CGSize(width: 64, height: 64))
            let k = Int((frame.itemTime.seconds * 24).rounded())
            #expect(Self.firstByte(frame.pixelBuffer) == UInt8(k % 250), "frame \(k)")
        }
        #expect(!taken.contains { CVPixelBufferGetWidth($0.pixelBuffer) == 64 && $0.sourceSize != nil })
    }

    @Test("a 60 fps source keeps up through the read-ahead")
    func keepsUpAtSixty() {
        let source = UpscalingFrameSource(wrapping: FakeSource(fps: 60), upscaler: FakeUpscaler())
        let taken = Self.play(source, to: 1.5)
        // Frames taken while the model loaded pass through, the pipeline's
        // three after them too (decided as they were taken).
        let settled = taken.drop { $0.itemTime.seconds < 0.3 }
        let steps = zip(settled, settled.dropFirst()).map { $1.itemTime.seconds - $0.itemTime.seconds }
        #expect(!steps.isEmpty && steps.allSatisfy { abs($0 - 1.0 / 60) < 0.001 })
        #expect(settled.allSatisfy { CVPixelBufferGetWidth($0.pixelBuffer) == 128 })
    }

    @Test("ticking only as often as the frames come, it still gets ahead of the playhead")
    func getsAheadAtFrameRateTicks() {
        // 24 ticks a second for a 24 fps source, as a throttled Mac window
        // drew: one frame per tick would hand each out at its time, never
        // ahead. The pipeline hands out several per tick until the lead fills.
        let source = UpscalingFrameSource(wrapping: FakeSource(fps: 24), upscaler: FakeUpscaler())
        var taken: [PumpFrame] = []
        var spacing = SourceFrameSpacing()
        var last: Double?
        var playhead = 0.0
        var leads: [Double] = []
        while playhead < 3 {
            for _ in 0 ..< 40 {
                let s = spacing.seconds
                let target = LookaheadPull.target(last: last, spacing: s.isFinite ? s : 1.0 / 60,
                                                  playhead: playhead, windowEnd: playhead + 0.15)
                if let last, target <= last + 1e-6 { break }
                guard let frame = source.frame(atItemTime: CMTime(seconds: target, preferredTimescale: 600_000)) else { break }
                if let last { spacing.note(gap: frame.itemTime.seconds - last) }
                last = frame.itemTime.seconds
                taken.append(frame)
            }
            if playhead > 1, let last { leads.append(last - playhead) }
            Thread.sleep(forTimeInterval: 0.01)
            playhead += 1.0 / 24
        }
        let steps = zip(taken, taken.dropFirst()).map { $1.itemTime.seconds - $0.itemTime.seconds }
        #expect(steps.allSatisfy { abs($0 - 1.0 / 24) < 0.001 })
        // Past the first second every tick holds at least two frames ahead.
        #expect(!leads.isEmpty && leads.allSatisfy { $0 >= 2.0 / 24 - 1e-6 }, "leads \(leads)")
    }

    @Test("HDR frames and frames it can't take pass through as decoded")
    func passesThrough() {
        let hdr = FakeUpscaler()
        let taken = Self.play(UpscalingFrameSource(wrapping: FakeSource(fps: 24, format: kCVPixelFormatType_64RGBAHalf),
                                                   upscaler: hdr), to: 1)
        #expect(hdr.made == 0 && hdr.prepares == 0)
        #expect(taken.count >= 22 && taken.allSatisfy { $0.sourceSize == nil })

        let off = FakeUpscaler()
        let source = UpscalingFrameSource(wrapping: FakeSource(fps: 24), upscaler: off)
        source.isEnabled = false
        let plain = Self.play(source, to: 1)
        #expect(off.made == 0)
        let steps = zip(plain, plain.dropFirst()).map { $1.itemTime.seconds - $0.itemTime.seconds }
        #expect(steps.allSatisfy { abs($0 - 1.0 / 24) < 0.001 })
    }

    @Test("an upscaler too slow for the frames turns itself off for the picture")
    func tooSlowTurnsOff() {
        // 30 ms a frame against 0.6 × 41.7 ms: over budget.
        let upscaler = FakeUpscaler(ms: 30)
        let taken = Self.play(UpscalingFrameSource(wrapping: FakeSource(fps: 24), upscaler: upscaler), to: 3)
        #expect(upscaler.made == UpscalingFrameSource.timingWindow)
        #expect(taken.suffix(20).allSatisfy { CVPixelBufferGetWidth($0.pixelBuffer) == 64 })
        let steps = zip(taken, taken.dropFirst()).map { $1.itemTime.seconds - $0.itemTime.seconds }
        #expect(steps.allSatisfy { abs($0 - 1.0 / 24) < 0.001 })

        // 20 ms is within it.
        let fits = FakeUpscaler(ms: 20)
        let kept = Self.play(UpscalingFrameSource(wrapping: FakeSource(fps: 24), upscaler: fits), to: 3)
        #expect(kept.suffix(20).allSatisfy { CVPixelBufferGetWidth($0.pixelBuffer) == 128 })
    }

    @Test("under Smooth Motion the in-between frames are made from upscaled frames and carry the decoded size")
    func underInterpolation() {
        let upscaling = UpscalingFrameSource(wrapping: FakeSource(fps: 24), upscaler: FakeUpscaler())
        let interpolator = InterpolatingFrameSourceTests.FakeInterpolator()
        let source = InterpolatingFrameSource(wrapping: upscaling, interpolator: interpolator)
        let taken = Self.play(source, to: 2)
        let settled = taken.drop { $0.itemTime.seconds < 0.5 }
        #expect(settled.count > 40)
        let mids = settled.filter(\.isMidFrame)
        #expect(mids.count > 15)
        #expect(mids.allSatisfy { $0.sourceSize == CGSize(width: 64, height: 64) })
        #expect(settled.filter { !$0.isMidFrame }.allSatisfy { CVPixelBufferGetWidth($0.pixelBuffer) == 128 })
    }

    @Test("a reset through the interpolator drops the frame the upscaler read ahead")
    func resetReachesTheUpscaler() {
        let inner = FakeSource(fps: 24)
        let upscaling = UpscalingFrameSource(wrapping: inner, upscaler: FakeUpscaler())
        let source = InterpolatingFrameSource(wrapping: upscaling,
                                              interpolator: InterpolatingFrameSourceTests.FakeInterpolator())
        _ = Self.play(source, to: 1)
        source.reset()
        // A loop back to the start: the source hands frame 0 again, and the
        // next frames come from there, not from what was read ahead before.
        inner.lastGiven = nil
        let again = Self.play(source, to: 0.5)
        #expect(again.first.map { $0.itemTime.seconds < 0.05 } == true)
        let steps = zip(again, again.dropFirst()).map { $1.itemTime.seconds - $0.itemTime.seconds }
        #expect(steps.allSatisfy { $0 > 0 && $0 < 0.05 })
    }

    #if !targetEnvironment(simulator)
    @Test("VideoToolbox's upscaler makes a 960×540 frame 1080p and keeps its colours")
    func videoToolbox() throws {
        guard #available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *), VTRealtimeUpscaler.isSupported else { return }
        #expect(VTRealtimeUpscaler.factor(width: 960, height: 540) == 2)
        #expect(VTRealtimeUpscaler.factor(width: 1280, height: 720) == 1.5)
        // 2× would be 1920×1440, over 1080p's pixels: 1.5× instead.
        #expect(VTRealtimeUpscaler.factor(width: 960, height: 720) == 1.5)
        #expect(VTRealtimeUpscaler.factor(width: 1920, height: 1080) == nil)
        #expect(UpscalingFrameSource.upscaledSize(width: 960, height: 540) == CGSize(width: 1920, height: 1080))

        let upscaler = VTRealtimeUpscaler()
        try #require(upscaler.prepare(width: 960, height: 540))
        let frame = try #require(FlowInterpolationBenchmark.pattern(width: 960, height: 540, shift: (0, 0)))
        CVBufferSetAttachment(frame, kCVImageBufferColorPrimariesKey, kCVImageBufferColorPrimaries_ITU_R_709_2, .shouldPropagate)
        CVBufferSetAttachment(frame, kCVImageBufferTransferFunctionKey, kCVImageBufferTransferFunction_ITU_R_709_2, .shouldPropagate)
        let done = DispatchSemaphore(value: 0)
        nonisolated(unsafe) var output: CVPixelBuffer?
        upscaler.upscale(frame) { out, _ in
            output = out
            done.signal()
        }
        done.wait()
        let out = try #require(output)
        #expect(CVPixelBufferGetWidth(out) == 1920 && CVPixelBufferGetHeight(out) == 1080)
        #expect(CVPixelBufferGetPixelFormatType(out) == kCVPixelFormatType_32BGRA)
        // Each channel's mean within a few code values of the input's.
        let before = Self.channelMeans(frame), after = Self.channelMeans(out)
        for c in 0 ..< 3 { #expect(abs(before[c] - after[c]) < 4, "channel \(c): \(before[c]) → \(after[c])") }
    }
    #endif

    static func channelMeans(_ buffer: CVPixelBuffer) -> [Double] {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let w = CVPixelBufferGetWidth(buffer), h = CVPixelBufferGetHeight(buffer)
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
        var sums = [0.0, 0.0, 0.0]
        for y in 0 ..< h { for x in 0 ..< w { for c in 0 ..< 3 { sums[c] += Double(base[y * stride + x * 4 + c]) } } }
        return sums.map { $0 / Double(w * h) }
    }
}
#endif
