import Foundation
import Testing
@testable import VergenceCore

/// Analysis tracks carry flow, disparity and depth computed off the headset.
/// What has to hold: a field survives the trip within its quantisation step,
/// a stream parses the same however it is chunked, a file indexes without
/// reading payloads, and lookup by time finds the span a moment falls in.
@Suite("Analysis tracks")
struct AnalysisTrackTests {
    static let header = AnalysisTrackHeader(mediaID: "a1b2c3", producer: "test", models: ["flowPair": "RIFE 4.25 lite 480x288"],
                                            sourceWidth: 1920, sourceHeight: 1080)

    /// A smooth pan-like flow field with a few big values.
    static func flow(width: Int, height: Int, channels: Int = 4) -> [Float] {
        (0 ..< channels * width * height).map { i in
            let c = i / (width * height), p = i % (width * height)
            return Float(c % 2 == 0 ? 6 : -2.5) + Float(p % width) * 0.01 - Float(p / width) * 0.02 + (p == 7 ? 40 : 0)
        }
    }

    @Test func halfFloatsRoundTripAndHandleEdges() {
        for value: Float in [0, 1, -1, 0.5, 65504, -0.000061035156, 3.14159, 1e-7, 1e9, -1e9] {
            let back = HalfFloat.value(from: HalfFloat.bits(from: value))
            if abs(value) > 65504 {
                #expect(back.isInfinite && (back > 0) == (value > 0))
            } else if abs(value) < 6e-8 {
                #expect(back == 0)
            } else {
                #expect(abs(back - value) <= abs(value) / 1024 + 1e-7, "\(value) → \(back)")
            }
        }
        #expect(HalfFloat.value(from: HalfFloat.bits(from: .nan)).isNaN)
    }

    @Test(arguments: [AnalysisEncoding.float16, .int16])
    func flowSurvivesWithinItsStep(_ encoding: AnalysisEncoding) {
        let values = Self.flow(width: 32, height: 18)
        let record = AnalysisRecord.encoding(values, kind: .flowPair, as: encoding, channels: 4, width: 32, height: 18,
                                             time: AnalysisTime(seconds: 1), endTime: AnalysisTime(seconds: 1 + 1.0 / 24))
        #expect(record.payload.count == values.count * 2)
        let back = record.values()
        let worst = zip(values, back).map { abs($0 - $1) }.max() ?? 0
        // int16 over ±40 px: a step of 40/32767 px. float16 near 40: 1/32 px.
        #expect(worst <= (encoding == .int16 ? 40 / 32767 : 1.0 / 32) + 1e-5, "worst error \(worst)")
    }

    @Test func depthAsBytesSpansItsRange() {
        let depth = (0 ..< 64).map { Float($0) / 63 }
        let record = AnalysisRecord.encoding(depth, kind: .depth, as: .uint8, channels: 1, width: 8, height: 8,
                                             time: AnalysisTime(seconds: 2))
        let back = record.values()
        #expect(back.first == 0 && abs((back.last ?? 0) - 1) < 1e-6)
        #expect(zip(depth, back).allSatisfy { abs($0 - $1) <= 0.5 / 255 + 1e-6 })
        #expect(record.endTime == record.time)
    }

    static func track(records: Int = 10) throws -> (Data, [AnalysisRecord]) {
        var data = try AnalysisTrack.header(header)
        var written: [AnalysisRecord] = []
        for i in 0 ..< records {
            let t0 = AnalysisTime(value: Int64(i) * 1001, timescale: 24000)
            let t1 = AnalysisTime(value: Int64(i + 1) * 1001, timescale: 24000)
            let flow = AnalysisRecord.encoding(flow(width: 16, height: 9), kind: .flowPair, as: .int16, channels: 4,
                                               width: 16, height: 9, time: t0, endTime: t1)
            let disparity = AnalysisRecord.encoding((0 ..< 144).map { Float($0 % 16) - 8 }, kind: .disparity, eye: .left,
                                                    as: .float16, channels: 1, width: 16, height: 9, time: t0)
            for record in [flow, disparity] {
                data.append(AnalysisTrack.record(record))
                written.append(record)
            }
        }
        return (data, written)
    }

    @Test func aStreamParsesTheSameHoweverItIsChunked() throws {
        let (data, written) = try Self.track()
        for chunk in [1, 7, 4096, data.count] {
            var parser = AnalysisTrackParser()
            var read: [AnalysisRecord] = []
            var offset = 0
            while offset < data.count {
                let end = min(offset + chunk, data.count)
                read += try parser.append(data.subdata(in: offset ..< end))
                offset = end
            }
            #expect(parser.header == Self.header)
            #expect(read == written, "chunk \(chunk)")
        }
    }

    @Test func notATrackIsRefusedEarly() {
        var parser = AnalysisTrackParser()
        #expect(throws: AnalysisTrackError.notATrack) { _ = try parser.append(Data("GIF8".utf8)) }
        var corrupt = AnalysisTrackParser()
        var data = try! AnalysisTrack.header(Self.header)
        data.appendLittleEndian(UInt32(AnalysisTrack.maxRecordBytes + 1))
        #expect(throws: AnalysisTrackError.tooLarge) { _ = try corrupt.append(data) }
    }

    @Test func anIndexFindsTheSpanAMomentFallsIn() throws {
        let (data, written) = try Self.track()
        let index = try AnalysisTrackIndex(data)
        #expect(index.entries.count == written.count)
        // Halfway between frames 3 and 4 of 24000/1001.
        let mid = AnalysisTime(value: 3 * 1001 + 500, timescale: 24000)
        let flow = try #require(index.entry(.flowPair, at: mid))
        #expect(flow.time == AnalysisTime(value: 3 * 1001, timescale: 24000))
        #expect(try index.record(flow) == written[6])
        // Disparity is per real frame: the latest at or before.
        let disparity = try #require(index.entry(.disparity, eye: .left, at: mid))
        #expect(disparity.time == flow.time)
        #expect(index.entry(.flowPair, at: AnalysisTime(seconds: -1)) == nil)
        #expect(index.entry(.depth, at: mid) == nil)
    }

    @Test func aTruncatedFileIsAnErrorNotAShortIndex() throws {
        let (data, _) = try Self.track(records: 2)
        #expect(throws: AnalysisTrackError.truncated) { _ = try AnalysisTrackIndex(data.prefix(data.count - 3)) }
    }

    /// A stand-in for DEFLATE: the delta coding and framing are what this file owns.
    static func identity(_ data: Data) -> Data { data }
    static func identity(_ data: Data, _ size: Int) -> Data { data }

    @Test(arguments: [AnalysisEncoding.float16, .int16, .uint8])
    func deltaCodedRecordsSurviveATrack(_ encoding: AnalysisEncoding) throws {
        let values = Self.flow(width: 24, height: 10, channels: 2).map { encoding == .uint8 ? abs($0) / 50 : $0 }
        let plain = AnalysisRecord.encoding(values, kind: .flowPair, as: encoding, channels: 2, width: 24, height: 10,
                                            time: AnalysisTime(seconds: 3))
        let packed = plain.deflated(using: Self.identity)
        #expect(packed.isDeflated && packed.payload != plain.payload)
        var parser = AnalysisTrackParser()
        let read = try parser.append(try AnalysisTrack.header(Self.header) + AnalysisTrack.record(packed))
        #expect(read == [packed])
        #expect(try read[0].inflated(using: Self.identity) == plain)
        let index = try AnalysisTrackIndex(try AnalysisTrack.header(Self.header) + AnalysisTrack.record(packed))
        #expect(try index.record(index.entries[0]) == packed)
    }

    @Test func aWrongInflatedSizeIsAnError() {
        let record = AnalysisRecord.encoding([1, 2, 3, 4], kind: .depth, as: .int16, channels: 1, width: 2, height: 2,
                                             time: AnalysisTime(seconds: 0)).deflated(using: Self.identity)
        #expect(throws: AnalysisTrackError.payloadSize) {
            _ = try record.inflated { data, _ in data.prefix(2) }
        }
    }

    @Test func aModelsNaNIsStoredAsZero() {
        let record = AnalysisRecord.encoding([.nan, 2, -.infinity, .infinity], kind: .depth, as: .int16, channels: 1,
                                             width: 2, height: 2, time: AnalysisTime(seconds: 0))
        let back = record.values()
        #expect(back[0] == 0 && abs(back[1] - 2) < 1e-3 && back[2] < 0 && back[3] > 0)
        #expect(record.scale > 0 && record.scale.isFinite)
    }

    @Test func timesCompareAcrossTimescales() {
        #expect(AnalysisTime(value: 1001, timescale: 24000) < AnalysisTime(value: 3754, timescale: 90000))
        #expect(!(AnalysisTime(value: 1, timescale: 2) < AnalysisTime(value: 45000, timescale: 90000)))
        #expect(abs(AnalysisTime(seconds: 1.5).seconds - 1.5) < 1e-9)
    }
}
