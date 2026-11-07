import Testing
@testable import VergenceCore

/// A film remuxed at 31.24 fps that repeats one frame in five, and videos
/// that don't.
@Suite struct RepeatedFramesTests {
    /// Frame-to-frame differences measured on such a film (a dark, fast scene).
    static let pulldown: [Float] = [14.83, 15.27, 0.61, 15.45, 15.10, 14.56, 14.44, 0.16, 14.53, 14.38, 14.38, 14.24, 0.22,
                                    14.23, 14.89, 14.63, 14.61, 0.58, 14.77, 14.97, 15.09, 15.93, 0.19, 16.33, 16.53, 15.57,
                                    15.33, 0.16, 15.94, 13.39, 19.18, 22.70, 0.69, 25.96, 29.29, 24.77, 20.01, 0.09, 24.03, 24.02]

    @Test func findsOneRepeatInFive() {
        let verdict = RepeatedFrames.judge([Self.pulldown, Array(Self.pulldown.dropFirst(3))])
        #expect(verdict.isPulldown)
        #expect(verdict.repeating == verdict.moving)
    }

    @Test func leavesOrdinaryAndStillVideoAlone() {
        let moving = (0 ..< 60).map { Float(10 + ($0 % 7)) }
        #expect(!RepeatedFrames.judge([moving]).isPulldown)
        // A still picture says nothing (and a little noise isn't a cadence).
        let still = (0 ..< 60).map { _ in Float(0.3) }
        #expect(RepeatedFrames.judge([still]).moving == 0)
        #expect(!RepeatedFrames.judge([still]).isPulldown)
        // A few repeats (a held frame here and there) aren't a pulldown.
        var some = moving
        for i in stride(from: 0, to: 60, by: 15) { some[i] = 0.1 }
        #expect(!RepeatedFrames.judge([some]).isPulldown)
    }

    @Test func dropsExactlyOneInFiveTheMostRepeated() {
        var dropper = RepeatDropper()
        var kept: [Int] = []
        for (i, d) in Self.pulldown.enumerated() { kept += dropper.add(i == 0 ? nil : d) }
        kept += dropper.finish()
        let dropped = Set(0 ..< Self.pulldown.count).subtracting(kept)
        #expect(dropped == [2, 7, 12, 17, 22, 27, 32, 37])
        #expect(kept == kept.sorted())
        // A partial stretch at the end is kept whole; the first frame never goes.
        var short = RepeatDropper()
        #expect(short.add(nil).isEmpty && short.add(0).isEmpty)
        #expect(short.finish() == [0, 1])
    }
}
