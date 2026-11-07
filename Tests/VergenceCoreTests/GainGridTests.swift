import Foundation
import Testing
@testable import VergenceCore

/// The gain grid's pinned layout, its trilinear lookup, fitting a gain map,
/// and easing between keyframes without crossing a cut.
@Suite("Gain grid")
struct GainGridTests {
    @Test func aRecordHasThePinnedLayoutAndRoundTrips() throws {
        var values = [Float](repeating: 0, count: GainGrid.count)
        values[GainGrid.index(3, 4, 5)] = 1.5
        let grid = GainGrid(log2Gain: values)
        let record = grid.record(time: AnalysisTime(value: 7, timescale: 24))
        #expect(record.kind == .gainMap && record.eye == .mono && record.encoding == .float16)
        #expect((record.channels, record.width, record.height) == (8, 16, 9))
        let parsed = try AnalysisRecord.parse(AnalysisTrack.record(record).dropFirst(4))
        #expect(GainGrid(record: parsed) == grid)
        // Channel 5, row 4, column 3 in the planar payload.
        #expect(abs(parsed.values()[(5 * 9 + 4) * 16 + 3] - 1.5) < 0.001)
    }

    @Test func lookupIsTrilinearBetweenCellCentresAndClampsAtTheEdges() {
        // Gain rises with luma bin only: g = l.
        var values = [Float](repeating: 0, count: GainGrid.count)
        for l in 0 ..< 8 { for y in 0 ..< 9 { for x in 0 ..< 16 { values[GainGrid.index(x, y, l)] = Float(l) } } }
        let grid = GainGrid(log2Gain: values)
        #expect(abs(grid.lookup(u: 0.5, v: 0.5, luma: 2.5 / 8) - 2) < 1e-5)       // bin 2's centre
        #expect(abs(grid.lookup(u: 0.5, v: 0.5, luma: 3 / 8) - 2.5) < 1e-5)       // halfway to bin 3
        #expect(abs(grid.lookup(u: 0, v: 1, luma: 0) - 0) < 1e-5)                 // clamped below bin 0
        #expect(abs(grid.lookup(u: 1, v: 0, luma: 1) - 7) < 1e-5)                 // clamped above bin 7
    }

    @Test func fittingAMapGivesBrightPixelsTheirGainAndDarkOnesTheirs() {
        // A 64×36 frame: the right half bright (luma 0.9) with gain 4, the left dark (0.2) with gain 1.
        let w = 64, h = 36
        var gain = [Float](repeating: 1, count: w * h), luma = [Float](repeating: 0.2, count: w * h)
        for y in 0 ..< h { for x in w / 2 ..< w { gain[y * w + x] = 4; luma[y * w + x] = 0.9 } }
        let grid = GainGrid.fit(gain: gain, luma: luma, width: w, height: h)
        #expect(abs(grid.lookup(u: 0.9, v: 0.5, luma: 0.9) - 2) < 0.05)           // log2 4
        #expect(abs(grid.lookup(u: 0.1, v: 0.5, luma: 0.2)) < 0.05)
        // A bright pixel moving into the dark half keeps (most of) its boost: luma decides.
        #expect(grid.lookup(u: 0.1, v: 0.5, luma: 0.9) > 1.9)
    }

    @Test func highlightsSpreadFromTheMidTonesWhichStayAsSDR() {
        // GMNet's curve on film: log2 gain 0.1 per bin, so mid-tones would be lifted 1.3–1.5×.
        var values = [Float](repeating: 0, count: GainGrid.count)
        for l in 0 ..< 8 { for y in 0 ..< 9 { for x in 0 ..< 16 { values[GainGrid.index(x, y, l)] = 0.1 * Float(l) + (x == 0 ? 1 : 0) } } }
        let grid = GainGrid(log2Gain: values).expandingHighlights()
        // The pivot, luma 0.5, lies halfway between bins 3 and 4: bins 0–3 get no gain,
        // bin 4 twice its 0.05 over the pivot.
        #expect(grid.lookup(u: 0.5, v: 0.5, luma: 3.5 / 8) == 0)
        #expect(grid.lookup(u: 0.5, v: 0.5, luma: 0.2) == 0)
        #expect(abs(grid.lookup(u: 0.5, v: 0.5, luma: 4.5 / 8) - 0.1) < 1e-5)
        // Above it, twice the gain over the pivot: bin 7 had 0.35 over it, now 0.7.
        #expect(abs(grid.lookup(u: 0.5, v: 0.5, luma: 15.0 / 16) - 0.7) < 1e-5)
        // A cell's own mid-tone gain is the pivot, wherever it sits (column 0 is a stop up).
        #expect(abs(grid.lookup(u: 0.5 / 16, v: 0.5, luma: 15.0 / 16) - 0.7) < 1e-5)
        // Never past GMNet's peak.
        let steep = GainGrid(log2Gain: (0 ..< GainGrid.count).map { Float($0 / (16 * 9)) }).expandingHighlights()
        #expect(steep.log2Gain.max() == GainGrid.peakLog2Gain)
    }

    @Test func framesEaseBetweenKeyframesAndHoldBeforeACut() throws {
        let a = GainGrid.identity
        let b = GainGrid(log2Gain: [Float](repeating: 1, count: GainGrid.count))
        let c = GainGrid(log2Gain: [Float](repeating: 2, count: GainGrid.count))
        let keys = [GainGridSchedule.Keyframe(index: 0, grid: a, cut: false),
                    GainGridSchedule.Keyframe(index: 12, grid: b, cut: false),
                    GainGridSchedule.Keyframe(index: 17, grid: c, cut: true)]
        #expect(try #require(GainGridSchedule.grid(at: 6, keyframes: keys)).log2Gain[0] == 0.5)
        #expect(try #require(GainGridSchedule.grid(at: 12, keyframes: keys)).log2Gain[0] == 1)
        // 13…16 are the old shot: they keep its grid instead of fading toward the next shot's.
        #expect(try #require(GainGridSchedule.grid(at: 16, keyframes: keys)).log2Gain[0] == 1)
        #expect(try #require(GainGridSchedule.grid(at: 20, keyframes: keys)).log2Gain[0] == 2)
        #expect(GainGridSchedule.grid(at: -1, keyframes: keys) == nil)
        #expect(GainGridSchedule.keyframeInterval(frameRate: 24000.0 / 1001) == 12)
    }

    @Test func theCapScalesLogGains() {
        #expect(GainGrid.cappedScale(meanGain: 2, cap: 2.5) == 1)
        #expect(abs(GainGrid.cappedScale(meanGain: 4, cap: 2) - 0.5) < 1e-6)
        #expect(GainGrid.cappedScale(meanGain: 4, cap: 1) == 0)
    }
}
