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

    // MARK: Taking every frame in turn

    static func admitted(fps: Double, speed: Double, limits: Bool, minimumInterval: Double, frames: Int) -> [Int] {
        var cap = FrameRateCap()
        return (0 ..< frames).filter {
            cap.admits(frameAt: Double($0) / fps, spacing: 1 / fps, rate: speed,
                       limitsToNormalSpeed: limits, minimumInterval: minimumInterval)
        }
    }

    @Test func aPumpTakingEveryFrameShowsNoMoreThanItTicks() {
        // 120 fps at normal speed: every other frame, at 60.
        let shown = Self.admitted(fps: 120, speed: 1, limits: true, minimumInterval: 1.0 / 60, frames: 240)
        #expect(zip(shown, shown.dropFirst()).allSatisfy { $1 - $0 == 2 })
        // 60 fps is the tick rate already: every frame.
        #expect(Self.admitted(fps: 60, speed: 1, limits: true, minimumInterval: 1.0 / 60, frames: 120).count == 120)
        #expect(Self.admitted(fps: 59.94, speed: 1, limits: false, minimumInterval: 1.0 / 60, frames: 120).count == 120)
    }

    @Test func cachedDepthAtTwiceSpeedShowsWhatTheTicksAllow() {
        // 60 fps at 2×: every other frame (60 a second), as the old pump's
        // newest-frame-per-tick did.
        let sixty = Self.admitted(fps: 60, speed: 2, limits: false, minimumInterval: 1.0 / 60, frames: 240)
        #expect(zip(sixty, sixty.dropFirst()).allSatisfy { $1 - $0 == 2 })
        // 24 fps at 2× is 48 a second, under the ticks: every frame.
        #expect(Self.admitted(fps: 24, speed: 2, limits: false, minimumInterval: 1.0 / 60, frames: 96).count == 96)
    }

    @Test func theNextFrameIsAskedForInTurn() {
        let s = 1.0 / 60
        // Starting, or after a seek: the frame showing now.
        #expect(LookaheadPull.target(last: nil, spacing: s, playhead: 10, windowEnd: 10.075) == 10)
        // In turn, halfway past the next frame…
        #expect(abs(LookaheadPull.target(last: 10, spacing: s, playhead: 10, windowEnd: 10.075) - (10 + 1.5 * s)) < 1e-9)
        // …never past the window…
        #expect(LookaheadPull.target(last: 10.07, spacing: s, playhead: 10, windowEnd: 10.075) == 10.075)
        // …and from the playhead when the next one is late already.
        #expect(LookaheadPull.target(last: 9.9, spacing: s, playhead: 10, windowEnd: 10.075) == 10)
        // Spacing not known yet: the newest in the window.
        #expect(LookaheadPull.target(last: 10, spacing: .infinity, playhead: 10, windowEnd: 10.075) == 10.075)
    }

    @Test func theClockTellsASeekFromAPause() {
        // Playing on: no jump.
        #expect(!LookaheadClock.isJump(previous: 10, previousRate: 1, elapsed: 1.0 / 60, playhead: 10 + 1.0 / 60, rate: 1))
        // Paused, or stalled, in between: stopped short.
        #expect(!LookaheadClock.isJump(previous: 10, previousRate: 1, elapsed: 1.0 / 60, playhead: 10.005, rate: 0))
        // Resumed, or sped up, in between.
        #expect(!LookaheadClock.isJump(previous: 10, previousRate: 0, elapsed: 1.0 / 60, playhead: 10.01, rate: 1))
        #expect(!LookaheadClock.isJump(previous: 10, previousRate: 1, elapsed: 1.0 / 60, playhead: 10 + 2.0 / 60, rate: 2))
        // A seek either way, and a paused step of one frame.
        #expect(LookaheadClock.isJump(previous: 10, previousRate: 1, elapsed: 1.0 / 60, playhead: 40, rate: 1))
        #expect(LookaheadClock.isJump(previous: 10, previousRate: 1, elapsed: 1.0 / 60, playhead: 3, rate: 1))
        #expect(LookaheadClock.isJump(previous: 10, previousRate: 0, elapsed: 1.0 / 60, playhead: 10 + 1.0 / 24, rate: 0))
    }

    @Test func framesQueuedAheadAreRetimedWhenTheClockChanges() {
        #expect(!LookaheadClock.needsRetime(queuedFor: 5.050, due: 5.052))
        #expect(LookaheadClock.needsRetime(queuedFor: 5.050, due: nil))     // paused
        #expect(LookaheadClock.needsRetime(queuedFor: 5.050, due: 5.025))   // twice as fast
        #expect(LookaheadClock.isDiscontinuous(last: 3, playhead: 10, windowEnd: 10.075))
        #expect(LookaheadClock.isDiscontinuous(last: 30, playhead: 10, windowEnd: 10.075))
        #expect(!LookaheadClock.isDiscontinuous(last: 10.075, playhead: 10, windowEnd: 10))   // paused, frames ahead
    }

    // MARK: Held depth

    static let picture = [Float](repeating: 100, count: 16)
    static let anotherShot = [Float](repeating: 30, count: 16)

    static func decisions(fps: Double, speed: Double = 1, frames: Int,
                          signature: (Int) -> [Float]? = { _ in picture }) -> [HeldDepthPolicy.Decision] {
        var policy = HeldDepthPolicy()
        return (0 ..< frames).map { i in
            policy.decide(seconds: Double(i) / fps, shownAt: Double(i) / (fps * speed), signature: signature(i))
        }
    }

    @Test func sixtyFramesASecondReuseEveryOtherMap() {
        let d = Self.decisions(fps: 60, frames: 12)
        #expect(d == Array(repeating: [HeldDepthPolicy.Decision.infer, .hold], count: 6).flatMap { $0 })
        // 48 and 50 too.
        #expect(Self.decisions(fps: 48, frames: 4) == [.infer, .hold, .infer, .hold])
    }

    @Test func thirtyFramesASecondInferEveryFrame() {
        #expect(Self.decisions(fps: 30, frames: 6).allSatisfy { $0 == .infer })
        #expect(Self.decisions(fps: 24, frames: 6).allSatisfy { $0 == .infer })
        // Slowed to half speed, 60 fps shows 30 a second: every frame.
        #expect(Self.decisions(fps: 60, speed: 0.5, frames: 6).allSatisfy { $0 == .infer })
    }

    @Test func aMapIsNeverReusedAcrossACut() {
        let d = Self.decisions(fps: 60, frames: 4) { $0 >= 1 ? Self.anotherShot : Self.picture }
        #expect(d == [.infer, .infer, .hold, .infer])
        // No signature to judge by: infer.
        #expect(Self.decisions(fps: 60, frames: 4) { _ in nil }.allSatisfy { $0 == .infer })
    }

    @Test func aMapIsNeverReusedAcrossASeek() {
        var policy = HeldDepthPolicy()
        #expect(policy.decide(seconds: 10, shownAt: 0, signature: Self.picture) == .infer)
        // Back, or forward more than a held step, a frame later on screen.
        #expect(policy.decide(seconds: 9, shownAt: 0.016, signature: Self.picture) == .infer)
        #expect(policy.decide(seconds: 9.5, shownAt: 0.032, signature: Self.picture) == .infer)
        policy.forget()
        #expect(policy.decide(seconds: 9.5 + 1.0 / 60, shownAt: 0.048, signature: Self.picture) == .infer)
    }

    @Test func atTwiceSpeedTheMapRateStaysTheSame() {
        // 60 fps at 2×, capped to every other frame: shown 1/60 s apart, two
        // source frames apart; still one map for two frames shown.
        var policy = HeldDepthPolicy()
        let d = (0 ..< 6).map { k in
            policy.decide(seconds: Double(2 * k) / 60, shownAt: Double(k) / 60, signature: Self.picture)
        }
        #expect(d == [.infer, .hold, .infer, .hold, .infer, .hold])
    }

    // MARK: The pipeline's budget

    /// A clocked pump taking each frame `lead` ahead, inferring maps one at a
    /// time on the model's own thread and drawing frames in order on its
    /// own: which frames are drawn by the time they're due.
    struct Pipeline {
        var onTime = 0
        var late = 0
        var maps = 0
        var held = 0
        var maxDepthAge = 0.0
    }

    static func runPipeline(fps: Double, seconds: Double, inference: Double, draw: Double,
                            lead: Double = 0.075) -> Pipeline {
        var result = Pipeline()
        var policy = HeldDepthPolicy()
        var modelFree = 0.0, pumpFree = 0.0
        var keyReady = 0.0, keySeconds = 0.0
        let frames = Int(seconds * fps)
        let start = 0.5
        for i in 0 ..< frames {
            let item = Double(i) / fps
            let due = start + item
            // Taken at the first tick when it is within `lead`.
            let taken = max(0, ((due - lead) * 60).rounded(.up) / 60)
            let ready: Double
            switch policy.decide(seconds: item, shownAt: due, signature: picture) {
            case .infer:
                let begin = max(modelFree, taken)
                modelFree = begin + inference
                keyReady = modelFree
                keySeconds = item
                ready = keyReady
                result.maps += 1
            case .hold:
                ready = keyReady
                result.held += 1
            }
            result.maxDepthAge = max(result.maxDepthAge, item - keySeconds)
            pumpFree = max(pumpFree, ready, taken) + draw
            // The first frames start cold, with nothing taken ahead.
            if i >= Int(fps / 2) {
                if pumpFree <= due { result.onTime += 1 } else { result.late += 1 }
            }
        }
        return result
    }

    @Test func sixtyFramesASecondFitTheMeasuredBudget() {
        // Vision Pro (M5), 2026-10-22: a map ~24-28 ms, warp ~5 ms,
        // transfer ~2 ms. The old pump did them in turn and showed ~30 fps.
        let run = Self.runPipeline(fps: 60, seconds: 10, inference: 0.028, draw: 0.007)
        #expect(run.late == 0, "\(run)")
        #expect(run.held == run.maps || run.held == run.maps - 1)
        #expect(run.maxDepthAge <= 1.0 / 60 + 1e-9)
        // A slow map now and then still fits within the lead.
        #expect(Self.runPipeline(fps: 60, seconds: 10, inference: 0.032, draw: 0.009).late == 0)
    }

    @Test func filmFrameRatesInferEveryFrameOnTime() {
        let film = Self.runPipeline(fps: 24, seconds: 10, inference: 0.03, draw: 0.007)
        #expect(film.late == 0 && film.held == 0, "\(film)")
        let thirty = Self.runPipeline(fps: 30, seconds: 10, inference: 0.028, draw: 0.007)
        #expect(thirty.late == 0 && thirty.held == 0, "\(thirty)")
    }
}
