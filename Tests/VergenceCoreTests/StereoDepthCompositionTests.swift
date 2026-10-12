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

@Suite struct DepthGapCompressorTests {
    /// 256 bins over 0…1 with the given shares of pixels at the given depths.
    private func bins(_ spread: [(depth: Float, share: Float)], pixels: Float = 100_000) -> [UInt32] {
        var b = [UInt32](repeating: 0, count: 256)
        for (d, share) in spread { b[min(255, Int(d * 256))] += UInt32(share * pixels) }
        return b
    }

    @Test func nearSubjectBeforeFarBackgroundIsSqueezed() {
        var c = DepthGapCompressor()
        // Background at 0–0.15, subject at 0.85–1, a thin outline ramp between.
        var spread: [(Float, Float)] = []
        for i in 0..<15 { spread.append((Float(i) / 100, 0.65 / 15)) }
        for i in 85..<100 { spread.append((Float(i) / 100, 0.3 / 15)) }
        for i in 15..<85 { spread.append((Float(i) / 100, 0.05 / 70)) }
        let curve = c.update(bins: bins(spread), lo: 0, hi: 1, mapping: .identity, dt: 1 / 30, cut: true)
        #expect(curve.span < 0.6)
        #expect(abs(curve.apply(1) - 1) < 1e-5)                  // the near end stays on the glass
        // The subject keeps its own roundness.
        #expect(abs((curve.apply(0.99) - curve.apply(0.88)) - 0.11) < 0.02)
        // Monotone.
        var last = -Float.infinity
        for i in 0...100 { let v = curve.apply(Float(i) / 100); #expect(v >= last); last = v }
    }

    @Test func depthSpreadThroughTheRoomIsLeftAlone() {
        var c = DepthGapCompressor()
        let spread = (0..<100).map { (Float($0) / 100, Float(0.01)) }
        let curve = c.update(bins: bins(spread), lo: 0, hi: 1, mapping: .identity, dt: 1 / 30, cut: true)
        #expect(abs(curve.span - 1) < 1e-4)
    }

    @Test func curveFollowsTheShotSlowlyAndRestartsAtACut() {
        var c = DepthGapCompressor()
        let room = bins((0..<100).map { (Float($0) / 100, Float(0.01)) })
        var gap: [(Float, Float)] = []
        for i in 0..<15 { gap.append((Float(i) / 100, 0.7 / 15)) }
        for i in 85..<100 { gap.append((Float(i) / 100, 0.3 / 15)) }
        let split = bins(gap)
        _ = c.update(bins: room, lo: 0, hi: 1, mapping: .identity, dt: 1 / 30, cut: true)
        let soon = c.update(bins: split, lo: 0, hi: 1, mapping: .identity, dt: 1 / 30, cut: false)
        #expect(soon.span > 0.95)                                // one frame barely moves it
        let atCut = c.update(bins: split, lo: 0, hi: 1, mapping: .identity, dt: 1 / 30, cut: true)
        #expect(atCut.span < 0.6)
    }

    /// A thin near object (a rail) is the only thing near: the rest of the
    /// picture must come forward rather than carry the full shift.
    @Test func sparseNearTailIsSqueezedToo() {
        var c = DepthGapCompressor()
        var spread: [(Float, Float)] = []
        for i in 0..<8 { spread.append((Float(i) / 100, 0.96 / 8)) }
        for i in 30..<100 { spread.append((Float(i) / 100, 0.04 / 70)) }
        let curve = c.update(bins: bins(spread), lo: 0, hi: 1, mapping: .identity, dt: 1 / 30, cut: true)
        #expect(abs(curve.apply(1) - 1) < 1e-5)
        #expect(curve.apply(0.04) > 0.5)
    }

    @Test func outsideTheRangeTheSlopeIsOne() {
        let curve = DepthGapCurve(knots: (0...32).map { 0.5 + Float($0) / 64 })
        #expect(abs(curve.apply(1.2) - 1.2) < 1e-5)
        #expect(abs(curve.apply(-0.1) - 0.4) < 1e-5)
        #expect(abs(curve.apply(0.5) - 0.75) < 1e-5)
    }
}
