import Foundation
import Testing
@testable import VergenceCore

/// Which grid a player shows: the frame's own, ramped in after engaging and
/// after a seek, held (never dropped) while a piece is missing, and refreshed
/// only every half second while stepped down.
@Suite("Gain grid playback")
struct GainGridPlaybackTests {
    /// Grids for frames 0…47 and 96…143 of a 24 fps title (a missing piece between), g = frame / 100.
    static func playback() throws -> GainGridPlayback {
        var data = try AnalysisTrack.header(AnalysisTrackHeader(mediaID: "t", producer: "test", models: [:],
                                                                sourceWidth: 64, sourceHeight: 36))
        for k in Array(0 ..< 48) + Array(96 ..< 144) {
            let grid = GainGrid(log2Gain: [Float](repeating: Float(k) / 100, count: GainGrid.count))
            data.append(AnalysisTrack.record(grid.record(time: AnalysisTime(value: Int64(k), timescale: 24))))
        }
        return GainGridPlayback(lookup: try AnalysisTrackIndex(data), frameRate: 24,
                                inflate: AnalysisTrackTests.identity)
    }

    static func g(_ frame: GainGridPlayback.Frame) -> Float { frame.grid.log2Gain[0] }

    @Test func itRampsInOverASecondAndShowsEachFramesOwnGrid() throws {
        let p = try Self.playback()
        let first = p.grid(at: 0)
        #expect(first.strength == 0 && abs(Self.g(first) - 0) < 0.001)
        let half = p.grid(at: 12.0 / 24)
        #expect(abs(half.strength - 0.5) < 0.01 && abs(Self.g(half) - 0.12) < 0.001)
        #expect(p.grid(at: 30.0 / 24).strength == 1)
    }

    @Test func aMissingPieceKeepsTheLastGridAndASeekRampsAgain() throws {
        let p = try Self.playback()
        for k in 0 ... 47 { _ = p.grid(at: Double(k) / 24) }
        // Frames 48…: no grids yet: the last one stays, at full strength.
        let held = p.grid(at: 48.0 / 24)
        #expect(abs(Self.g(held) - 0.47) < 0.001 && held.strength == 1)
        // A seek into the next piece: its grids, ramping from 0 again.
        let after = p.grid(at: 100.0 / 24)
        #expect(abs(Self.g(after) - 1.0) < 0.001 && after.strength == 0)
        // A seek back to before any grid: nothing new, strength 0 until one arrives.
        let back = try Self.playback()
        #expect(back.grid(at: 70.0 / 24).strength == 0)
    }

    @Test func steppedDownItRefreshesEveryHalfSecond() throws {
        let p = try Self.playback()
        _ = p.grid(at: 0, steppedDown: true)
        #expect(abs(Self.g(p.grid(at: 5.0 / 24, steppedDown: true)) - 0) < 0.001)
        #expect(abs(Self.g(p.grid(at: 12.0 / 24, steppedDown: true)) - 0.12) < 0.001)
        #expect(abs(Self.g(p.grid(at: 13.0 / 24, steppedDown: false)) - 0.13) < 0.001)
    }

    @Test func steppedDownItStillSnapsAtACut() throws {
        var data = try AnalysisTrack.header(AnalysisTrackHeader(mediaID: "t", producer: "test", models: [:],
                                                                sourceWidth: 64, sourceHeight: 36))
        // Frames 0…9 one shot (g 0), 10… the next (g 1).
        for k in 0 ..< 20 {
            let grid = GainGrid(log2Gain: [Float](repeating: k < 10 ? 0 : 1, count: GainGrid.count))
            data.append(AnalysisTrack.record(grid.record(time: AnalysisTime(value: Int64(k), timescale: 24))))
        }
        let p = GainGridPlayback(lookup: try AnalysisTrackIndex(data), frameRate: 24, inflate: AnalysisTrackTests.identity)
        _ = p.grid(at: 0, steppedDown: true)
        #expect(Self.g(p.grid(at: 9.0 / 24, steppedDown: true)) == 0)
        #expect(Self.g(p.grid(at: 10.0 / 24, steppedDown: true)) == 1)
    }
}
