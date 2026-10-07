import Foundation
import Testing
@testable import MVC3D

/// Writes a CLPI shaped like a 3D disc's: the base view's EP_map in the CPI
/// block, or (for a dependent-view clip) an empty CPI and the EP_map in
/// ExtensionData 2/6.
struct ClipInfoBuilder {
    var pid: UInt16
    var entries: [ClipInfo.EntryPoint]
    var inExtension = false

    func build() -> Data {
        let cpi = buildCPI()
        let header = 40
        var data = Data("HDMV0200".utf8)
        if inExtension {
            let emptyCPI = u32(0)
            let block = cpi
            // One directory entry; the block follows the 12-byte header and the entry.
            let directory = u32(24) + Data([0, 0, 0, 1]) + u16(2) + u16(6) + u32(24) + u32(UInt32(block.count))
            let ext = u32(UInt32(directory.count + block.count)) + directory + block
            data += u32(0) + u32(0) + u32(UInt32(header)) + u32(0) + u32(UInt32(header + emptyCPI.count))
            data += Data(count: header - data.count) + emptyCPI + ext
        } else {
            data += u32(0) + u32(0) + u32(UInt32(header)) + u32(0) + u32(0)
            data += Data(count: header - data.count) + cpi
        }
        return data
    }

    private func buildCPI() -> Data {
        // Coarse entries start wherever the high bits of PTS or SPN change.
        var coarse = Data(), fine = Data()
        var coarseCount = 0
        var last: (UInt32, UInt32)?
        for (index, e) in entries.enumerated() {
            let ptsCoarse = UInt32(e.pts >> 19 & 0x3FFF), spnHigh = e.spn & ~0x1FFFF
            if last.map({ $0 != (ptsCoarse >> 1, spnHigh) }) ?? true {
                coarse += u32(UInt32(index) << 14 | ptsCoarse) + u32(e.spn)
                coarseCount += 1
                last = (ptsCoarse >> 1, spnHigh)
            }
            fine += u32(UInt32(e.pts >> 9 & 0x7FF) << 17 | e.spn & 0x1FFFF)
        }
        let streamMap = u32(UInt32(4 + coarse.count)) + coarse + fine
        let counts = UInt64(1) << 34 | UInt64(coarseCount) << 18 | UInt64(entries.count)
        let epMapHeader = 2 + 12
        var epMap = Data([0, 1]) + u16(pid)
        epMap += Data((0 ..< 6).map { UInt8(counts >> (40 - 8 * $0) & 0xFF) }) + u32(UInt32(epMapHeader))
        epMap += streamMap
        let body = u16(1) + epMap
        return u32(UInt32(body.count)) + body
    }

    private func u16(_ v: UInt16) -> Data { Data([UInt8(v >> 8), UInt8(v & 0xFF)]) }
    private func u32(_ v: UInt32) -> Data { Data([UInt8(v >> 24), UInt8(v >> 16 & 0xFF), UInt8(v >> 8 & 0xFF), UInt8(v & 0xFF)]) }
}

/// Entry points crossing both a coarse PTS boundary (bit 19) and a coarse
/// SPN boundary (bit 17).
private let entries: [ClipInfo.EntryPoint] = [
    .init(pts: 48_640, spn: 4),
    .init(pts: 138_752, spn: 896),
    .init(pts: 523_776, spn: 130_000),
    .init(pts: 614_400, spn: 131_100),
    .init(pts: 1_049_088, spn: 400_000),
    .init(pts: 614_984_192, spn: 156_579_520),
]

@Test(arguments: [false, true])
func entryPointsRoundTrip(inExtension: Bool) throws {
    let info = try ClipInfo(data: ClipInfoBuilder(pid: 0x1012, entries: entries, inExtension: inExtension).build())
    #expect(info.entryPoints[0x1012] == entries)
}

@Test func entryPointLookupFindsTheLastAtOrBefore() throws {
    let info = try ClipInfo(data: ClipInfoBuilder(pid: 0x1011, entries: entries).build())
    #expect(info.entryPoint(pid: 0x1011, atOrBefore: 0)?.spn == 4)
    #expect(info.entryPoint(pid: 0x1011, atOrBefore: 138_752)?.spn == 896)
    #expect(info.entryPoint(pid: 0x1011, atOrBefore: 600_000)?.spn == 130_000)
    let last = info.entryPoint(pid: 0x1011, atOrBefore: 1 << 32)
    #expect(last?.byteOffset == Int64(156_579_520) * 192)
    #expect(info.entryPoint(pid: 0x1012, atOrBefore: 0) == nil)
}

@Test func aClipWithoutEntryPointsParses() throws {
    let info = try ClipInfo(data: ClipInfoBuilder(pid: 0x1011, entries: []).build())
    #expect(info.entryPoints[0x1011] == [])
    #expect(info.entryPoint(pid: 0x1011, atOrBefore: 0) == nil)
}

@Test func notAClipInfoFile() {
    #expect(throws: FormatError.self) { try ClipInfo(data: Data("MPLS0200".utf8) + Data(count: 40)) }
}
