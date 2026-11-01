#if canImport(CoreMedia)
import CoreMedia
import CoreVideo
import Foundation
import Testing
@testable import VergenceApple

/// The presenter driven the way a flat player's draw loop drives it: a tick
/// per display refresh, showing whatever it answers.
@Suite("Interpolated frame presenter")
struct InterpolatedFramePresenterTests {
    typealias Fakes = InterpolatingFrameSourceTests

    struct Shown {
        let seconds: Double
        let playhead: Double
        let luma: UInt8
    }

    /// Ticks at `hz` from `start` to `end` seconds of item time (the fake
    /// clock's item time is the host time).
    static func play(_ presenter: InterpolatedFramePresenter, from start: Double = 0, to end: Double,
                     hz: Double = 96, shown: inout [Shown]) {
        var playhead = start
        while playhead < end {
            if let frame = presenter.frame(forHostTime: playhead) {
                shown.append(Shown(seconds: frame.itemTime.seconds, playhead: playhead,
                                   luma: Fakes.firstByte(frame.pixelBuffer)))
            }
            Thread.sleep(forTimeInterval: 0.002)
            playhead += 1 / hz
        }
    }

    @Test("a 24 fps source shows at 48 fps, each frame once it is due and not before")
    func doublesOnTime() {
        let inner = Fakes.FakeSource(fps: 24, luma: { UInt8(10 + 4 * $0) })
        let source = InterpolatingFrameSource(wrapping: inner, interpolator: Fakes.FakeInterpolator(),
                                              thresholds: .flat)
        let presenter = InterpolatedFramePresenter(source: source)
        var shown: [Shown] = []
        Self.play(presenter, to: 2, shown: &shown)

        #expect(shown.allSatisfy { $0.seconds <= $0.playhead + 1e-6 }, "nothing shown early")
        let settled = shown.drop { $0.playhead < 0.5 }
        #expect(settled.count > 60)
        let steps = zip(settled, settled.dropFirst()).map { $1.seconds - $0.seconds }
        #expect(steps.allSatisfy { abs($0 - 1.0 / 48) < 0.001 }, "every half frame, none skipped")
        // Shown within a display tick of being due.
        #expect(settled.allSatisfy { $0.playhead - $0.seconds < 1.0 / 96 + 1e-6 })
        for frame in settled {
            #expect(abs(Double(frame.luma) - (10 + 4 * frame.seconds * 24)) <= 1, "luma at \(frame.seconds)")
        }
    }

    @Test("a 60 fps source shows every frame, none interpolated")
    func passesSixty() {
        let inner = Fakes.FakeSource(fps: 60)
        let interpolator = Fakes.FakeInterpolator()
        let source = InterpolatingFrameSource(wrapping: inner, interpolator: interpolator, thresholds: .flat)
        let presenter = InterpolatedFramePresenter(source: source)
        var shown: [Shown] = []
        Self.play(presenter, to: 1.5, hz: 120, shown: &shown)
        let settled = shown.drop { $0.playhead < 0.3 }
        let steps = zip(settled, settled.dropFirst()).map { $1.seconds - $0.seconds }
        #expect(steps.allSatisfy { abs($0 - 1.0 / 60) < 0.001 })
        #expect(interpolator.made == 0)
    }

    @Test("a seek back starts over at the new playhead and keeps doubling")
    func seekBack() {
        let inner = Fakes.FakeSource(fps: 24, luma: { UInt8(truncatingIfNeeded: 10 + 2 * $0) })
        let source = InterpolatingFrameSource(wrapping: inner, interpolator: Fakes.FakeInterpolator(),
                                              thresholds: .flat)
        let presenter = InterpolatedFramePresenter(source: source)
        var before: [Shown] = []
        Self.play(presenter, from: 0, to: 2, shown: &before)
        inner.lastGiven = nil // the output decodes from the seek point again
        var after: [Shown] = []
        Self.play(presenter, from: 0.5, to: 2, shown: &after)

        #expect(after.allSatisfy { $0.seconds <= $0.playhead + 1e-6 && $0.seconds >= 0.5 - 1.0 / 24 })
        let settled = after.drop { $0.playhead < 1.0 }
        #expect(settled.count > 30)
        let steps = zip(settled, settled.dropFirst()).map { $1.seconds - $0.seconds }
        #expect(steps.allSatisfy { abs($0 - 1.0 / 48) < 0.001 })
    }

    @Test("frames it can't interpolate (all cuts) aren't read ahead past the lead")
    func boundedReadAhead() {
        // Every frame a cut: the source still reads a real frame ahead of
        // each ask, as it does while the governor has it paused.
        let inner = Fakes.FakeSource(fps: 24, luma: { $0 % 2 == 0 ? 20 : 220 })
        let interpolator = Fakes.FakeInterpolator()
        let source = InterpolatingFrameSource(wrapping: inner, interpolator: interpolator, thresholds: .flat)
        let presenter = InterpolatedFramePresenter(source: source)
        var shown: [Shown] = []
        Self.play(presenter, to: 3, shown: &shown)
        #expect(interpolator.made == 0)
        let furthest = (inner.lastGiven.map { Double($0) / 24 }) ?? 0
        #expect(furthest <= 3 + InterpolatedFramePresenter.lead + 2.0 / 24, "read to \(furthest) s at a 3 s playhead")
        let settled = shown.drop { $0.playhead < 0.5 }
        let steps = zip(settled, settled.dropFirst()).map { $1.seconds - $0.seconds }
        #expect(steps.allSatisfy { abs($0 - 1.0 / 24) < 0.001 })
    }
}
#endif
