import Testing
@testable import VergenceCore

struct StereoLookaheadPolicyTests {
    @Test func thermalSlowdownGrowsLeadQuicklyAndRecoveryIsGradual() {
        #expect(StereoLookaheadPolicy.lead(after: 0.12, shown: 60, late: 10) == 0.145)
        #expect(StereoLookaheadPolicy.lead(after: 0.25, shown: 60, late: 10) == 0.25)
        #expect(StereoLookaheadPolicy.lead(after: 0.25, shown: 60, late: 0) == 0.245)
        #expect(StereoLookaheadPolicy.lead(after: 0.12, shown: 60, late: 0) == 0.12)
    }
    @Test func highResolutionDoesNotMultiplyQueuedMemory() {
        #expect(StereoLookaheadPolicy.capacity(width: 1920, height: 1080) == 12)
        #expect(StereoLookaheadPolicy.capacity(width: 3840, height: 2160) == 3)
        #expect(StereoLookaheadPolicy.capacity(width: 7680, height: 4320) == 1)
        #expect(StereoLookaheadPolicy.capacity(width: 0, height: 0) == 1)
        #expect(StereoLookaheadPolicy.capacity(width: .max, height: .max) == 1)
    }
}
