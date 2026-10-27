import XCTest
@testable import VergenceCore

final class StereoProcessingHoldPolicyTests: XCTestCase {
    func testStartupAndPauseRefillResumeOnlyWithReserve() {
        var policy = StereoProcessingHoldPolicy()
        XCTAssertTrue(policy.update(now: 1, clockRunning: false, readySeconds: 0, resumeSeconds: 0.5, lowWaterSeconds: 0.1))
        // The overdue frame becoming ready is not enough to restart.
        XCTAssertTrue(policy.update(now: 1.2, clockRunning: false, readySeconds: 0.1, resumeSeconds: 0.5, lowWaterSeconds: 0.1))
        XCTAssertFalse(policy.update(now: 1.5, clockRunning: false, readySeconds: 0.5, resumeSeconds: 0.5, lowWaterSeconds: 0.1))
        XCTAssertFalse(policy.update(now: 2, clockRunning: true, readySeconds: 0.2, resumeSeconds: 0.5, lowWaterSeconds: 0.1))
        XCTAssertTrue(policy.update(now: 3, clockRunning: true, readySeconds: 0.05, resumeSeconds: 0.5, lowWaterSeconds: 0.1))
        XCTAssertFalse(policy.update(now: 3.5, clockRunning: false, readySeconds: 0.5, resumeSeconds: 0.5, lowWaterSeconds: 0.1))
        XCTAssertFalse(policy.recoveryExhausted)
    }

    func testSustainedOverloadTimesOutAndBacksOff() {
        var policy = StereoProcessingHoldPolicy(maximumHold: 1, cooldown: 2)
        XCTAssertTrue(policy.update(now: 1, clockRunning: true, readySeconds: 0, resumeSeconds: 0.5, lowWaterSeconds: 0.1))
        XCTAssertFalse(policy.update(now: 2.1, clockRunning: false, readySeconds: 0.1, resumeSeconds: 0.5, lowWaterSeconds: 0.1))
        XCTAssertTrue(policy.recoveryExhausted)
        XCTAssertFalse(policy.update(now: 3, clockRunning: true, readySeconds: 0, resumeSeconds: 0.5, lowWaterSeconds: 0.1))
        XCTAssertTrue(policy.update(now: 4.2, clockRunning: true, readySeconds: 0, resumeSeconds: 0.5, lowWaterSeconds: 0.1))
        XCTAssertFalse(policy.recoveryExhausted)
    }

    func testRepeatedRefillsEventuallyReportSustainedOverload() {
        var policy = StereoProcessingHoldPolicy(maximumHold: 2, cooldown: 5)
        XCTAssertTrue(policy.update(now: 0, clockRunning: true, readySeconds: 0, resumeSeconds: 0.5, lowWaterSeconds: 0.1))
        XCTAssertFalse(policy.update(now: 1.1, clockRunning: false, readySeconds: 0.5, resumeSeconds: 0.5, lowWaterSeconds: 0.1))
        XCTAssertFalse(policy.recoveryExhausted)
        XCTAssertTrue(policy.update(now: 2, clockRunning: true, readySeconds: 0, resumeSeconds: 0.5, lowWaterSeconds: 0.1))
        XCTAssertFalse(policy.update(now: 3.1, clockRunning: false, readySeconds: 0.5, resumeSeconds: 0.5, lowWaterSeconds: 0.1))
        XCTAssertTrue(policy.recoveryExhausted)
        XCTAssertFalse(policy.update(now: 4, clockRunning: true, readySeconds: 0, resumeSeconds: 0.5, lowWaterSeconds: 0.1))
    }

    func testEndOfMediaDoesNotAskForAnImpossibleReserve() {
        var policy = StereoProcessingHoldPolicy()
        XCTAssertFalse(policy.update(now: 1, clockRunning: true, readySeconds: 0, resumeSeconds: 0, lowWaterSeconds: 0))
        policy.reset()
        XCTAssertFalse(policy.isHolding)
    }

    /// A paused AVPlayer can't fill the reserve: the hold ends once what it
    /// could hand out has depth, rather than at the timeout.
    func testSourceThatCannotRefillWhilePausedReleasesWhenExhausted() {
        var policy = StereoProcessingHoldPolicy()
        // Before the first frame: stopped with nothing to prepare, no hold.
        XCTAssertFalse(policy.update(now: 0, clockRunning: false, readySeconds: 0, resumeSeconds: 0.6,
                                     lowWaterSeconds: 0.12, sourceExhausted: true))
        // The first frame waits for its depth at the playhead.
        XCTAssertTrue(policy.update(now: 0.1, clockRunning: true, readySeconds: -0.01, resumeSeconds: 0.6,
                                    lowWaterSeconds: 0.12))
        XCTAssertTrue(policy.update(now: 0.12, clockRunning: false, readySeconds: -0.01, resumeSeconds: 0.6,
                                    lowWaterSeconds: 0.12))
        XCTAssertFalse(policy.update(now: 0.2, clockRunning: false, readySeconds: 0.04, resumeSeconds: 0.6,
                                     lowWaterSeconds: 0.12, sourceExhausted: true))
        XCTAssertFalse(policy.recoveryExhausted)
    }

    func testManyShortHoldsReportSustainedOverload() {
        var policy = StereoProcessingHoldPolicy(maximumRecoveries: 3)
        for index in 0 ..< 3 {
            let start = Double(index)
            XCTAssertTrue(policy.update(now: start, clockRunning: true, readySeconds: 0, resumeSeconds: 0.6,
                                        lowWaterSeconds: 0.12))
            XCTAssertFalse(policy.update(now: start + 0.1, clockRunning: false, readySeconds: 0.3,
                                         resumeSeconds: 0.6, lowWaterSeconds: 0.12, sourceExhausted: true))
        }
        XCTAssertTrue(policy.recoveryExhausted)
    }
}
