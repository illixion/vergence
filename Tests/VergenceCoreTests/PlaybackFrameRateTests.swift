import Testing
@testable import VergenceCore

/// Speed changes the frame rate a player shows and asks the display for.
@Suite("Playback frame rate")
struct PlaybackFrameRateTests {
    @Test func aMeasuredRateSnapsToTheStandardOne() {
        #expect(PlaybackFrameRate.snapped(23.975) == 24000.0 / 1001)
        #expect(PlaybackFrameRate.snapped(59.9) == 60000.0 / 1001)
        #expect(PlaybackFrameRate.snapped(119.88) == 120000.0 / 1001)
    }

    @Test func speedMultipliesTheFramesShown() {
        #expect(PlaybackFrameRate.shown(source: 60, speed: 2) == 120)
        #expect(PlaybackFrameRate.shown(source: 60, speed: 0.5) == 30)
        // Smooth Motion's doubled 23.976 at 2×: 95.9, a 96 Hz display.
        #expect(abs(PlaybackFrameRate.shown(source: 24000.0 / 1001, doubled: true, speed: 2) - 95.904) < 0.001)
        // Never past the display's fastest.
        #expect(PlaybackFrameRate.shown(source: 60, speed: 4) == 120)
    }

    @Test func aDrawLoopTicksFasterOnlyWhenFaster() {
        #expect(PlaybackFrameRate.drawRate(speed: 1) == 60)
        #expect(PlaybackFrameRate.drawRate(speed: 0.5) == 60)
        #expect(PlaybackFrameRate.drawRate(speed: 1.5) == 90)
        #expect(PlaybackFrameRate.drawRate(speed: 2) == 120)
        #expect(PlaybackFrameRate.drawRate(base: 96, speed: 2) == 120)
    }
}
