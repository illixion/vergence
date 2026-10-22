import Foundation
import Testing
@testable import VergenceApple

/// The stereo pump's frame cap and its keeping-up report, driven by a
/// simulated pump: 60 Hz ticks on a wall clock, each taking the newest frame
/// the source has, a tick that shows one taking `cost` seconds (the next one
/// waits for it), and a source whose frame k is due at wall time
/// k / (fps × speed).
@Suite("Stereo pacing")
struct StereoPacingTests {
    struct Run {
        var shown: [Int] = []
        var spacing = SourceFrameSpacing()
        var reports: [StereoPerformance] = []
        var steps: [Int] { zip(shown, shown.dropFirst()).map { $1 - $0 } }
    }

    static func simulate(fps: Double, speed: Double, seconds: Double, cost: Double,
                         capped: Bool) -> Run {
        let tick = 1.0 / 60.0
        var run = Run()
        var cap = FrameRateCap()
        var now = 0.0
        var next = 0.0
        var window = (start: 0.0, frames: 0, span: 0.0)
        var lastTaken: Int?
        var lastShown: Int?
        while now < seconds {
            now = max(now, next)
            next += tick
            let newest = Int((now * fps * speed + 1e-9).rounded(.down))
            guard newest != lastTaken else { continue }
            if let lastTaken { run.spacing.note(gap: Double(newest - lastTaken) / fps) }
            lastTaken = newest
            if capped, !cap.admits(frameAt: Double(newest) / fps, spacing: run.spacing.seconds, rate: speed) {
                continue
            }
            if let lastShown {
                window.frames += 1
                window.span += Double(newest - lastShown) / fps
            }
            lastShown = newest
            run.shown.append(newest)
            now += cost
            if now - window.start >= 2 {
                if let report = StereoPerformance.measured(
                    frames: window.frames, itemSpan: window.span, window: now - window.start,
                    spacing: run.spacing.seconds, maxShownPerSecond: 60, sourceHeight: 1080) {
                    run.reports.append(report)
                }
                window = (now, 0, 0)
            }
        }
        return run
    }

    @Test func atNormalSpeedEveryFrameIsShown() {
        // Real-time AI 3D on a 24 fps film: inference makes most ticks late.
        let run = Self.simulate(fps: 24, speed: 1, seconds: 10, cost: 0.038, capped: true)
        #expect(run.steps.allSatisfy { $0 == 1 })
        #expect(run.shown.count >= 239)
        #expect(!run.reports.isEmpty)
        #expect(run.reports.allSatisfy { !$0.isOverloaded && abs($0.sourceFramesPerSecond - 24) < 0.1 })
    }

    @Test func slowerEveryFrameIsShown() {
        let run = Self.simulate(fps: 30, speed: 0.5, seconds: 6, cost: 0.03, capped: true)
        #expect(run.steps.allSatisfy { $0 == 1 })
    }

    @Test func atNormalSpeedSixtyFramesKeepComing() {
        let run = Self.simulate(fps: 60, speed: 1, seconds: 5, cost: 0.004, capped: true)
        #expect(run.shown.count >= 290)
    }

    @Test func twiceAsFastShowsEveryOtherFrame() {
        let run = Self.simulate(fps: 24, speed: 2, seconds: 10, cost: 0.034, capped: true)
        // Past the first frames, which find the frame rate: 24 a second.
        #expect(run.steps.dropFirst(3).allSatisfy { $0 == 2 }, "\(run.steps)")
        #expect(!run.reports.isEmpty)
        #expect(run.reports.allSatisfy { !$0.isOverloaded && $0.deliveredFraction > 0.95 })
        #expect(run.reports.allSatisfy { abs($0.playbackRate - 2) < 0.1 })
    }

    @Test func halfAgainAsFastAlternates() {
        let run = Self.simulate(fps: 24, speed: 1.5, seconds: 10, cost: 0.034, capped: true)
        let steps = Array(run.steps.dropFirst(3))
        #expect(Set(steps) == [1, 2], "\(steps)")
        let mean = Double(steps.reduce(0, +)) / Double(steps.count)
        #expect(abs(mean - 1.5) < 0.05)
        #expect(run.reports.allSatisfy { !$0.isOverloaded })
    }

    @Test func uncappedFasterIsUneven() {
        // What the cap prevents: the inference takes every second or third frame.
        let run = Self.simulate(fps: 24, speed: 2, seconds: 10, cost: 0.034, capped: false)
        #expect(Set(run.steps.dropFirst(3)).count > 1)
    }

    @Test func aCheapPictureAtTwiceSixtyKeepsUp() {
        // Cached depth isn't capped; the pump's own 60 ticks are all it can show.
        let run = Self.simulate(fps: 60, speed: 2, seconds: 6, cost: 0.004, capped: false)
        #expect(!run.reports.isEmpty)
        #expect(run.reports.allSatisfy { !$0.isOverloaded })
    }

    @Test func aRealOverloadIsStillOne() {
        // 70 ms a frame against a 24 fps film: about 14 of its 24 frames.
        let run = Self.simulate(fps: 24, speed: 1, seconds: 10, cost: 0.07, capped: true)
        #expect(!run.reports.isEmpty)
        #expect(run.reports.allSatisfy { $0.isOverloaded }, "\(run.reports.map(\.deliveredFraction))")
    }

    @Test func aSeekBackIsShownAtOnce() {
        var cap = FrameRateCap()
        let s = 1.0 / 24
        var shown: [Bool] = []
        for pts in [10.0, 10 + s, 10 + 2 * s, 3.0, 3 + s, 3 + 2 * s] {
            shown.append(cap.admits(frameAt: pts, spacing: s, rate: 2))
        }
        #expect(shown == [true, false, true, true, false, true])
        // Back at normal speed nothing is left out.
        let normal = cap.admits(frameAt: 3 + 3 * s, spacing: s, rate: 1)
        #expect(normal)
    }

    @Test func oneShortGapDoesNotSetTheFrameRate() {
        var spacing = SourceFrameSpacing()
        spacing.note(gap: 1.0 / 24)
        #expect(spacing.seconds == .infinity)
        spacing.note(gap: 1.0 / 24)
        #expect(abs(spacing.seconds - 1.0 / 24) < 1e-9)
        // A glitch at a segment boundary, once: the film still reads as 24 fps.
        spacing.note(gap: 0.017)
        spacing.note(gap: 2.0 / 24)
        #expect(abs(spacing.seconds - 1.0 / 24) < 1e-9)
        // A seek isn't a frame.
        spacing.note(gap: 0.004)
        spacing.note(gap: 0.004)
        spacing.note(gap: 3)
        #expect(abs(spacing.seconds - 1.0 / 24) < 1e-9)
    }

    @Test func aDroppedRunDoesNotWidenIt() {
        var spacing = SourceFrameSpacing()
        for _ in 0 ..< 3 { spacing.note(gap: 1.0 / 30) }
        for _ in 0 ..< 50 { spacing.note(gap: 2.0 / 30) }
        #expect(abs(spacing.seconds - 1.0 / 30) < 1e-9)
    }
}
