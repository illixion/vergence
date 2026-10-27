import XCTest
@testable import VergenceCore

final class StereoProcessingHoldPolicyTests: XCTestCase {
    func testHoldsForOverdueDepthAndReleasesWhenReady() {
        var policy = StereoProcessingHoldPolicy()
        XCTAssertFalse(policy.update(now: 1.11, clockRunning: true, waitingSince: 1))
        XCTAssertTrue(policy.update(now: 1.13, clockRunning: true, waitingSince: 1))
        XCTAssertTrue(policy.update(now: 1.2, clockRunning: false, waitingSince: 1))
        XCTAssertFalse(policy.update(now: 1.25, clockRunning: false, waitingSince: nil))
    }

    func testTimedOutHoldUsesCooldownBeforeRetry() {
        var policy = StereoProcessingHoldPolicy(triggerDelay: 0.1, maximumHold: 0.5, cooldown: 2)
        XCTAssertTrue(policy.update(now: 1.2, clockRunning: true, waitingSince: 1))
        XCTAssertFalse(policy.update(now: 1.71, clockRunning: false, waitingSince: 1))
        XCTAssertFalse(policy.update(now: 2, clockRunning: true, waitingSince: 1.8))
        XCTAssertTrue(policy.update(now: 3.8, clockRunning: true, waitingSince: 1.8))
    }

    func testUserPauseDoesNotTriggerHoldAndResetClearsState() {
        var policy = StereoProcessingHoldPolicy()
        XCTAssertFalse(policy.update(now: 3, clockRunning: false, waitingSince: 1))
        XCTAssertTrue(policy.update(now: 3, clockRunning: true, waitingSince: 1))
        policy.reset()
        XCTAssertFalse(policy.isHolding)
    }
}
