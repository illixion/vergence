import Testing
@testable import VergenceCore

struct StereoLookaheadPolicyTests {
    @Test func slowdownGrowsLeadAndRecoveryIsGradual() {
        #expect(StereoLookaheadPolicy.lead(after: 0.75, shown: 60, late: 10) == 0.85)
        #expect(StereoLookaheadPolicy.lead(after: 1.5, shown: 60, late: 10) == 1.5)
        #expect(StereoLookaheadPolicy.lead(after: 1, shown: 60, late: 0) == 0.99)
        #expect(StereoLookaheadPolicy.lead(after: 0.75, shown: 60, late: 0) == 0.75)
    }
    @Test func reserveFitsTheFrameBudgetAtEverySpeed() {
        let capacity = StereoLookaheadPolicy.capacity(width: 1920, height: 1080)
        #expect(capacity > 16)
        for rate in [0.5, 1, 2] {
            let reserve = StereoLookaheadPolicy.reserve(capacity: capacity, spacing: 1 / 60, rate: rate, lead: 0.75)
            #expect(reserve <= Double(capacity - 1) / 60 * max(1, rate))
            #expect(reserve <= 0.75 * rate)
        }
        #expect(StereoLookaheadPolicy.capacity(width: 3840, height: 2160) < capacity)
        #expect(StereoLookaheadPolicy.capacity(width: 7680, height: 4320) == 1)
        #expect(StereoLookaheadPolicy.capacity(width: .max, height: .max) == 1)
        #expect(StereoLookaheadPolicy.reserve(capacity: 1, spacing: 1 / 24, rate: 1, lead: 0.75) == 0)
    }
}
