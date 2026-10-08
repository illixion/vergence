import Foundation
import Testing
@testable import MVC3D

/// Builds BDAV (192-byte) packets carrying one PES per access unit.
struct StreamBuilder {
    var bytes = Data()
    private var continuity: [UInt16: UInt8] = [:]

    /// `declaresLength` fills PES_packet_length the way audio and subtitle
    /// PES do; video leaves it 0.
    mutating func pes(pid: UInt16, pts: Int64, payload: Data, declaresLength: Bool = false) {
        let length = declaresLength ? 3 + 5 + payload.count : 0
        var pes = Data([0, 0, 1, 0xE0, UInt8(length >> 8), UInt8(length & 0xFF), 0x80, 0x80, 5])
        pes.append(contentsOf: Self.timestamp(pts, marker: 0x2))
        pes.append(payload)
        var first = true
        var offset = 0
        while offset < pes.count {
            let room = 184
            let take = min(room, pes.count - offset)
            var packet = Data([0, 0, 0, 0, 0x47])
            let cc = continuity[pid, default: 0]
            continuity[pid] = (cc + 1) & 0xF
            packet.append((first ? 0x40 : 0) | UInt8(pid >> 8))
            packet.append(UInt8(pid & 0xFF))
            if take < room {
                // Adaptation field of stuffing pads the last packet to 188.
                let stuffing = room - take
                packet.append(0x30 | cc)
                packet.append(UInt8(stuffing - 1))
                if stuffing > 1 {
                    packet.append(0x00)
                    packet.append(contentsOf: repeatElement(0xFF, count: stuffing - 2))
                }
            } else {
                packet.append(0x10 | cc)
            }
            packet.append(pes[(pes.startIndex + offset) ..< (pes.startIndex + offset + take)])
            precondition(packet.count == 192)
            bytes.append(packet)
            offset += take
            first = false
        }
    }

    static func timestamp(_ t: Int64, marker: UInt8) -> [UInt8] {
        [marker << 4 | UInt8(t >> 29 & 0x0E) | 1, UInt8(t >> 22 & 0xFF), UInt8(t >> 14 & 0xFE) | 1,
         UInt8(t >> 7 & 0xFF), UInt8(t << 1 & 0xFE) | 1]
    }
}

func nal(_ type: UInt8, _ body: [UInt8] = [0xAA, 0xBB], startCode: [UInt8] = [0, 0, 0, 1]) -> Data {
    Data(startCode + [type] + body)
}

@Test func demuxerReassemblesPESAcrossPacketsAndChunks() throws {
    var builder = StreamBuilder()
    let big = Data((0 ..< 1000).map { UInt8($0 & 0x7F) | 0x80 })
    builder.pes(pid: 0x1011, pts: 90_000, payload: big)
    builder.pes(pid: 0x1100, pts: 90_000, payload: Data([1, 2, 3])) // audio, ignored
    builder.pes(pid: 0x1011, pts: 93_754, payload: Data([9, 9]))

    var demuxer = TransportStreamDemuxer(pids: [0x1011])
    var out: [PESPacket] = []
    // Odd chunk sizes so packets straddle feed calls.
    var i = 0
    while i < builder.bytes.count {
        let end = min(i + 301, builder.bytes.count)
        out += try demuxer.feed(builder.bytes[i ..< end])
        i = end
    }
    out += demuxer.flush()

    #expect(out.count == 2)
    #expect(out[0].pts == 90_000)
    #expect(out[0].payload == big)
    #expect(out[1].pts == 93_754)
    #expect(out[1].payload == Data([9, 9]))
}

@Test func demuxerDetectsPlainTransportStream() throws {
    var builder = StreamBuilder()
    for k in 0 ..< 4 { builder.pes(pid: 0x100, pts: Int64(k) * 3003, payload: Data([UInt8(k)])) }
    // Strip the 4-byte BDAV header from every packet.
    var plain = Data()
    for start in stride(from: 0, to: builder.bytes.count, by: 192) {
        plain.append(builder.bytes[(start + 4) ..< (start + 192)])
    }
    var demuxer = TransportStreamDemuxer(pids: [0x100])
    let out = try demuxer.feed(plain) + demuxer.flush()
    #expect(out.map(\.payload) == [Data([0]), Data([1]), Data([2]), Data([3])])
}

@Test func timestampsBeyond32BitsSurvive() throws {
    var builder = StreamBuilder()
    let pts: Int64 = (1 << 33) - 1
    builder.pes(pid: 0x1011, pts: pts, payload: Data([1]))
    var demuxer = TransportStreamDemuxer(pids: [0x1011])
    let out = try demuxer.feed(builder.bytes) + demuxer.flush()
    #expect(out.count == 1)
    #expect(out.first?.pts == pts, "got \(String(describing: out.first?.pts))")
}

@Test func annexBSplitsThreeAndFourByteStartCodes() {
    let data = nal(9, [0x10]) + nal(7, [1, 2, 0]) + nal(5, [3], startCode: [0, 0, 1])
    let units = AnnexB.nalUnits(in: data)
    #expect(units.map(\.type) == [9, 7, 5])
    // A trailing zero before the next start code is not part of the NAL.
    #expect(data[units[1].range] == Data([7, 1, 2]))
}

@Test func pairerJoinsViewsByPTSAndDropsTheDependentDelimiter() {
    var pairer = StereoAccessUnitPairer()
    let base = nal(9) + nal(5)
    let dependent = nal(24) + nal(15) + nal(20)
    var out = pairer.addBase(PESPacket(pid: 0x1011, pts: 100, dts: 100, payload: base))
    #expect(out.isEmpty)
    #expect(pairer.wantsDependent)
    out = pairer.addDependent(PESPacket(pid: 0x1012, pts: 100, dts: 100, payload: dependent))
    #expect(out.count == 1)
    #expect(out[0].hasDependentView)
    #expect(AnnexB.nalUnits(in: out[0].data).map(\.type) == [9, 5, 15, 20])
}

@Test func pairerKeepsDecodeOrderWhenDependentArrivesFirst() {
    var pairer = StereoAccessUnitPairer()
    _ = pairer.addDependent(PESPacket(pid: 0x1012, pts: 300, dts: nil, payload: nal(20)))
    _ = pairer.addDependent(PESPacket(pid: 0x1012, pts: 100, dts: nil, payload: nal(20)))
    var out = pairer.addBase(PESPacket(pid: 0x1011, pts: 300, dts: nil, payload: nal(1)))
    out += pairer.addBase(PESPacket(pid: 0x1011, pts: 100, dts: nil, payload: nal(1)))
    #expect(out.map(\.pts) == [300, 100])
    #expect(out.allSatisfy { $0.hasDependentView })
}

@Test func pairerFinishesUnpairedBaseAsMono() {
    var pairer = StereoAccessUnitPairer()
    _ = pairer.addBase(PESPacket(pid: 0x1011, pts: 100, dts: nil, payload: nal(1)))
    let out = pairer.finish()
    #expect(out.count == 1)
    #expect(!out[0].hasDependentView)
    #expect(pairer.unpairedBaseCount == 1)
}

@Test(arguments: [7, 192, 1000, 100_000])
func demuxerSkipsDamagedBytesAndCarriesOn(chunk: Int) throws {
    var builder = StreamBuilder()
    let payload = Data(repeating: 0x88, count: 600) // four packets per PES
    for i in 0 ..< 10 { builder.pes(pid: 0x1011, pts: Int64(i) * 3754, payload: payload) }
    var bytes = builder.bytes
    // Wreck the middle of the fourth PES: a packet's sync byte and the bytes after it,
    // as a bad read would.
    let damage = 3 * 4 * 192 + 2 * 192 + 4
    bytes.replaceSubrange(damage ..< damage + 300, with: Data(repeating: 0x00, count: 300))

    var demuxer = TransportStreamDemuxer(pids: [0x1011])
    var out: [PESPacket] = []
    for start in stride(from: 0, to: bytes.count, by: chunk) {
        out += try demuxer.feed(bytes[start ..< min(start + chunk, bytes.count)])
    }
    out += demuxer.flush()
    #expect(demuxer.resyncCount == 1)
    // The damaged PES is gone; every other one arrives whole.
    #expect(out.compactMap(\.pts) == [0, 1, 2, 4, 5, 6, 7, 8, 9].map { Int64($0) * 3754 })
    #expect(out.allSatisfy { $0.payload == payload })
}
