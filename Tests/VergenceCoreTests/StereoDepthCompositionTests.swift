import Testing
@testable import VergenceCore

/// The measurement behind the shot-adaptive strength: a landscape (most pixels
/// far) must read as using most of the shift, an interior (depth spread
/// through the picture) as using about half, and the proposal must follow.
@Suite("Stereo depth composition")
struct StereoDepthCompositionTests {
    /// Bins over 0…1 with identity mapping and the legacy budget, so shown
    /// depth equals the bin's value.
    static func measure(_ fill: (Int) -> UInt32) -> DepthComposition? {
        let bins = (0..<DepthHistogram.binCount).map(fill)
        return DepthComposition.measure(bins: bins, lo: 0, hi: 1, mapping: .identity, budget: .legacy)
    }

    @Test func uniformDepthIsCentred() throws {
        let c = try #require(Self.measure { _ in 10 })
        #expect(abs(c.mean - 0.5) < 0.01)
        #expect(abs(c.p50 - 0.5) < 0.01)
        #expect(abs(c.farShare - 0.25) < 0.01)
        #expect(abs(c.spread - 0.9) < 0.01)
        #expect(abs(c.meanShift() - 0.5) < 0.01)
    }

    /// 90 % of the picture at the far end, a near subject in the rest: the
    /// horse-scene shape that needed about a third of the strength.
    @Test func landscapeIsAttenuated() throws {
        let c = try #require(Self.measure { i in i < 13 ? 70 : (i > 240 ? 10 : 0) })
        #expect(c.farShare > 0.85)
        #expect(c.meanShift() > 0.8)
        let scale = StereoStrengthPolicy.proposedScale(for: c)
        #expect(scale > 0.3 && scale < 0.4)
    }

    /// Depth spread through the picture keeps more of the strength than a
    /// landscape does.
    @Test func spreadDepthKeepsMoreStrength() throws {
        let interior = try #require(Self.measure { i in i > 100 ? 10 : 2 })
        let landscape = try #require(Self.measure { i in i < 13 ? 70 : (i > 240 ? 10 : 0) })
        #expect(StereoStrengthPolicy.proposedScale(for: interior) > StereoStrengthPolicy.proposedScale(for: landscape))
    }

    @Test func proposalIsBounded() {
        let allFar = DepthComposition(mean: 0, p05: 0, p20: 0, p50: 0, p80: 0, p95: 0, farShare: 1)
        let allNear = DepthComposition(mean: 1, p05: 1, p20: 1, p50: 1, p80: 1, p95: 1, farShare: 0)
        #expect(StereoStrengthPolicy.proposedScale(for: allFar) == max(StereoStrengthPolicy.minimumScale, StereoStrengthPolicy.targetMeanShift))
        #expect(StereoStrengthPolicy.proposedScale(for: allNear) == 1)
    }

    /// The budget's shoulder is applied: shot-space depth past 1 is shown just
    /// short of the glass, never past it.
    @Test func shownThroughTheBudget() throws {
        var bins = [UInt32](repeating: 0, count: DepthHistogram.binCount)
        bins[255] = 100
        let c = try #require(DepthComposition.measure(bins: bins, lo: 0, hi: 2, mapping: .identity, budget: .studio))
        #expect(c.mean > 0.9 && c.mean < 1)
    }

    @Test func emptyHistogramMeasuresNothing() {
        #expect(Self.measure { _ in 0 } == nil)
    }

    @Test func shotTrackerAveragesAndClosesAtCuts() throws {
        var tracker = DepthShotTracker()
        let far = DepthComposition(mean: 0.1, p05: 0, p20: 0, p50: 0.1, p80: 0.2, p95: 0.5, farShare: 0.9)
        let near = DepthComposition(mean: 0.5, p05: 0.1, p20: 0.3, p50: 0.5, p80: 0.7, p95: 0.9, farShare: 0.2)
        #expect(tracker.add(far, seconds: 0, cut: true) == nil)
        #expect(tracker.add(near, seconds: 1, cut: false) == nil)
        let closed = tracker.add(near, seconds: 2, cut: true)
        let shot = try #require(closed)
        #expect(shot.frames == 2)
        #expect(shot.start == 0 && shot.end == 1)
        #expect(abs(shot.composition.mean - 0.3) < 1e-6)
        #expect(tracker.current?.frames == 1)
        #expect(tracker.current?.start == 2)
    }
}
