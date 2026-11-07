import Foundation
import Testing
@testable import VergenceCore

/// The player finds a frame pair's fields by its own frame times: the pair's
/// flow and the disparity at each end, matched within a quarter of the
/// pair's spacing, with an offset for a timeline that starts elsewhere.
@Suite("Shared-field track lookup")
struct SharedFieldTrackTests {
    static let rate = 24000.0 / 1001

    /// `frames` real frames at 23.976 fps from `start` seconds: a flow pair
    /// per pair (channel c filled with c + k) and a disparity per frame (−k).
    static func track(frames: Int = 6, start: Double = 0) throws -> AnalysisTrackIndex {
        let header = AnalysisTrackHeader(mediaID: "t", producer: "test", models: [:], sourceWidth: 1920, sourceHeight: 1080)
        var data = try AnalysisTrack.header(header)
        func time(_ k: Int) -> AnalysisTime { AnalysisTime(seconds: start + Double(k) / rate) }
        for k in 0 ..< frames {
            data.append(AnalysisTrack.record(AnalysisRecord.encoding(
                [Float](repeating: -Float(k), count: 8 * 4), kind: .disparity, eye: .right, as: .int16,
                channels: 1, width: 8, height: 4, time: time(k), maxMagnitude: 1024).deflated(using: AnalysisTrackTests.identity)))
            guard k + 1 < frames else { continue }
            let flow = (0 ..< 5).flatMap { c in [Float](repeating: Float(c + k), count: 6 * 3) }
            data.append(AnalysisTrack.record(AnalysisRecord.encoding(
                flow, kind: .flowPair, as: .int16, channels: 5, width: 6, height: 3,
                time: time(k), endTime: time(k + 1), maxMagnitude: 1024)))
        }
        return try AnalysisTrackIndex(data)
    }

    @Test func aPairFindsItsFlowAndBothDisparities() throws {
        let provider = AnalysisTrackFieldProvider(index: try Self.track(), inflate: AnalysisTrackTests.identity)
        // The player's clock rounds: a millisecond off still matches.
        let fields = try #require(provider.fields(from: 2 / Self.rate + 0.001, to: 3 / Self.rate - 0.001))
        #expect(fields.flowWidth == 6 && fields.flowHeight == 3 && fields.disparityWidth == 8)
        #expect(abs(fields.flow[0] - 2) < 0.01 && abs(fields.flow[4 * 18] - 6) < 0.01)
        #expect(abs(fields.disparity0[0] + 2) < 0.01 && abs(fields.disparity1[0] + 3) < 0.01)
        // The next pair reuses frame 3's disparity.
        let next = try #require(provider.fields(from: 3 / Self.rate, to: 4 / Self.rate))
        #expect(abs(next.disparity0[0] + 3) < 0.01)
        #expect(provider.misses == 0)
    }

    @Test func aTwoDimensionalTrackNeedsNoDisparity() throws {
        // A 2D title's track (smooth-motion-2d) has flow pairs only: a stereo
        // provider finds nothing in it, a mono one finds every pair.
        let header = AnalysisTrackHeader(mediaID: "t", producer: "test", models: [:], sourceWidth: 1280, sourceHeight: 720)
        var data = try AnalysisTrack.header(header)
        for k in 0 ..< 3 {
            let flow = (0 ..< 5).flatMap { c in [Float](repeating: Float(c + k), count: 6 * 3) }
            data.append(AnalysisTrack.record(AnalysisRecord.encoding(
                flow, kind: .flowPair, as: .int16, channels: 5, width: 6, height: 3,
                time: AnalysisTime(seconds: Double(k) / 24), endTime: AnalysisTime(seconds: Double(k + 1) / 24),
                maxMagnitude: 1024)))
        }
        let index = try AnalysisTrackIndex(data)
        let stereo = AnalysisTrackFieldProvider(index: index, inflate: AnalysisTrackTests.identity)
        #expect(stereo.fields(from: 1.0 / 24, to: 2.0 / 24) == nil)
        let mono = AnalysisTrackFieldProvider(index: index, needsDisparity: false, inflate: AnalysisTrackTests.identity)
        let fields = try #require(mono.fields(from: 1.0 / 24, to: 2.0 / 24))
        #expect(!fields.hasDisparity && fields.flowWidth == 6 && abs(fields.flow[0] - 1) < 0.01)
        #expect(mono.frameRate.map { abs($0 - 24) < 0.001 } == true)
    }

    @Test func framesTheTrackDoesNotHoldFindNothing() throws {
        let provider = AnalysisTrackFieldProvider(index: try Self.track(), inflate: AnalysisTrackTests.identity)
        // Past the end, between real frames, and a pair two frames apart.
        #expect(provider.fields(from: 5 / Self.rate, to: 6 / Self.rate) == nil)
        #expect(provider.fields(from: 1.5 / Self.rate, to: 2.5 / Self.rate) == nil)
        #expect(provider.fields(from: 1 / Self.rate, to: 3 / Self.rate) == nil)
        #expect(provider.misses == 3)
    }

    @Test func theTrackSaysItsFrameRate() throws {
        // 23.976, which a player measuring HLS reads as ~24.
        let rate = try #require(AnalysisTrackFieldProvider(index: try Self.track(), inflate: AnalysisTrackTests.identity).frameRate)
        #expect(abs(rate - Self.rate) < 0.001)
    }

    @Test func anOffsetMovesThePlayersTimelineOntoTheTracks() throws {
        // A disc's track at its own times (600 s in); the player counts from 0.
        let index = try Self.track(start: 600)
        #expect(AnalysisTrackFieldProvider(index: index, inflate: AnalysisTrackTests.identity)
            .fields(from: 0, to: 1 / Self.rate) == nil)
        let shifted = AnalysisTrackFieldProvider(index: index, timeOffset: 600, inflate: AnalysisTrackTests.identity)
        #expect(shifted.fields(from: 0, to: 1 / Self.rate) != nil)
    }

    @Test func nearestPicksTheCloserRecordWithinTolerance() throws {
        let index = try Self.track()
        let t = AnalysisTime(seconds: 2.4 / Self.rate)
        #expect(index.nearest(.disparity, eye: .right, to: t, within: 0.5 / Self.rate)?.time
            == AnalysisTime(seconds: 2 / Self.rate))
        #expect(index.nearest(.disparity, eye: .right, to: AnalysisTime(seconds: 2.6 / Self.rate),
                              within: 0.5 / Self.rate)?.time == AnalysisTime(seconds: 3 / Self.rate))
        #expect(index.nearest(.disparity, eye: .right, to: t, within: 0.3 / Self.rate) == nil)
        #expect(index.nearest(.disparity, eye: .left, to: t, within: 1) == nil)
    }
}
