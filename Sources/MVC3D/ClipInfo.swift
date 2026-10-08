/*
 mvc3d - Blu-ray clip information (the .clpi files in BDMV/CLIPINF)

 Each clip has a CLPI whose EP_map lists the clip's entry points: for every
 random access picture of a video PID, its PTS and the source packet (192
 byte) where its access unit starts. That is what seeking needs.

 The EP_map is in the CPI block of a base clip. A 3D disc's dependent-view
 clip has an empty CPI and keeps its EP_map in ExtensionData 2/6 (CPI_SS),
 in the same format, one entry per base-view entry point.

 Entries are stored in two levels: coarse entries carry the high bits of
 PTS and SPN, fine entries the low bits.
 */

import Foundation

public struct ClipInfo: Sendable, Equatable {
    public struct EntryPoint: Sendable, Equatable {
        /// 90 kHz, the PES clock. Bit 0 to 8 are always 0 (the EP_map's resolution).
        public var pts: Int64
        /// Source packet number: the access unit starts at byte `spn * 192`.
        public var spn: UInt32
        public var byteOffset: Int64 { Int64(spn) * 192 }
    }

    /// Entry points per video PID, in PTS order.
    public var entryPoints: [UInt16: [EntryPoint]]

    public init(entryPoints: [UInt16: [EntryPoint]]) {
        self.entryPoints = entryPoints
    }

    public init(data: Data) throws {
        var header = BinaryReader(data)
        let magic = try header.ascii(4)
        guard magic == "HDMV" else { throw FormatError("not clip information (\(magic))") }
        _ = try header.ascii(4) // version
        _ = try header.u32() // SequenceInfo
        _ = try header.u32() // ProgramInfo
        let cpiStart = Int(try header.u32())
        _ = try header.u32() // ClipMark
        let extensionStart = Int(try header.u32())

        var points = try Self.parseCPI(BinaryReader(data, offset: cpiStart))
        if extensionStart > 0, extensionStart < data.count {
            for block in try ExtensionData.blocks(BinaryReader(data, offset: extensionStart)) where block.id == (2, 6) {
                points.merge(try Self.parseCPI(block.reader)) { base, _ in base }
            }
        }
        entryPoints = points
    }

    /// The last entry point at or before `pts`, else the first.
    public func entryPoint(pid: UInt16, atOrBefore pts: Int64) -> EntryPoint? {
        guard let list = entryPoints[pid], !list.isEmpty else { return nil }
        var low = 0, high = list.count
        while low < high {
            let mid = (low + high) / 2
            if list[mid].pts <= pts { low = mid + 1 } else { high = mid }
        }
        return list[max(low - 1, 0)]
    }

    /// The first entry point after `pts`, or nil past the last.
    public func entryPoint(pid: UInt16, after pts: Int64) -> EntryPoint? {
        guard let list = entryPoints[pid] else { return nil }
        var low = 0, high = list.count
        while low < high {
            let mid = (low + high) / 2
            if list[mid].pts <= pts { low = mid + 1 } else { high = mid }
        }
        return low < list.count ? list[low] : nil
    }

    // MARK: CPI / EP_map

    private static func parseCPI(_ base: BinaryReader) throws -> [UInt16: [EntryPoint]] {
        var r = base
        let length = try r.u32()
        guard length > 0 else { return [:] }
        let type = try r.u16() & 0xF
        guard type == 1 else { return [:] } // 1 = EP_map
        let mapStart = r.offset
        try r.skip(1)
        let streams = Int(try r.u8())
        var result: [UInt16: [EntryPoint]] = [:]
        for _ in 0 ..< streams {
            let pid = try r.u16()
            let a = try r.u16(), b = try r.u32()
            // 10 reserved, 4 EP_stream_type, 16 coarse count, 18 fine count
            let coarseCount = Int((UInt32(a) & 0x3) << 14 | b >> 18)
            let fineCount = Int(b & 0x3FFFF)
            let start = mapStart + Int(try r.u32())
            result[pid] = try parseStreamMap(BinaryReader(base.data, offset: start), coarse: coarseCount, fine: fineCount)
        }
        return result
    }

    private static func parseStreamMap(_ base: BinaryReader, coarse coarseCount: Int, fine fineCount: Int) throws -> [EntryPoint] {
        var r = base
        let fineStart = base.offset + Int(try r.u32())
        var coarse: [(fineID: Int, pts: UInt32, spn: UInt32)] = []
        coarse.reserveCapacity(coarseCount)
        for _ in 0 ..< coarseCount {
            let a = try r.u32() // 18 ref_to_EP_fine_id, 14 PTS_EP_coarse
            coarse.append((Int(a >> 14), a & 0x3FFF, try r.u32()))
        }
        var f = BinaryReader(base.data, offset: fineStart)
        var fine: [UInt32] = []
        fine.reserveCapacity(fineCount)
        for _ in 0 ..< fineCount { fine.append(try f.u32()) }

        var points: [EntryPoint] = []
        points.reserveCapacity(fineCount)
        for (index, c) in coarse.enumerated() {
            let end = index + 1 < coarse.count ? coarse[index + 1].fineID : fineCount
            guard c.fineID <= end, end <= fineCount else { throw FormatError("EP_map fine entries out of order") }
            for e in fine[c.fineID ..< end] {
                // 1 angle change, 3 I_end_position_offset, 11 PTS_EP_fine, 17 SPN_EP_fine.
                // The coarse PTS's low bit overlaps the fine PTS's high bit.
                let pts = Int64(c.pts & ~1) << 19 + Int64(e >> 17 & 0x7FF) << 9
                let spn = (c.spn & ~0x1FFFF) + (e & 0x1FFFF)
                points.append(EntryPoint(pts: pts, spn: spn))
            }
        }
        return points
    }
}

/// The ExtensionData directory shared by MPLS and CLPI files.
enum ExtensionData {
    struct Block {
        var id: (UInt16, UInt16)
        var reader: BinaryReader
    }

    static func blocks(_ base: BinaryReader) throws -> [Block] {
        var r = base
        let length = Int(try r.u32())
        guard length > 0 else { return [] }
        try r.skip(4) // data_block_start_address
        try r.skip(3)
        let entries = Int(try r.u8())
        var blocks: [Block] = []
        for _ in 0 ..< entries {
            let id1 = try r.u16(), id2 = try r.u16()
            let start = Int(try r.u32()), entryLength = Int(try r.u32())
            let block = BinaryReader(base.data, offset: base.offset + start)
            guard block.remaining >= entryLength else { throw FormatError("extension past the end") }
            blocks.append(Block(id: (id1, id2), reader: block))
        }
        return blocks
    }
}
