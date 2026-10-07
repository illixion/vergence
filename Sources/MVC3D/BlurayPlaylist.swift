/*
 mvc3d - Blu-ray playlists (the .mpls files in BDMV/PLAYLIST)

 A playlist is the title a player plays: an ordered list of PlayItems, each
 a span [IN, OUT) of one clip (BDMV/STREAM/<clip>.m2ts). Times are 45 kHz.

 3D (BD-ROM Part 3, "Blu-ray 3D") adds, in the playlist's ExtensionData:
 - ID 2/2, SubPath entries extension: a SubPath of type 8 ("out-of-mux
   synchronous, dependent view") whose SubPlayItems name the clip holding
   the second eye for each PlayItem, with the same IN/OUT.
 - ID 2/1, STN_table_SS: per PlayItem, the dependent view's PID (0x1012)
   among other 3D stream entries.

 Only what decoding needs is parsed. Multi-angle PlayItems keep angle 0.
 */

import Foundation

public struct BlurayPlaylist: Sendable, Equatable {
    public struct PlayItem: Sendable, Equatable {
        /// Base clip, five digits ("00007").
        public var clip: String
        /// Dependent-view clip for a 3D playlist.
        public var dependentClip: String?
        public var inTime: UInt32
        public var outTime: UInt32
        public var videoPID: UInt16?
        public var dependentVideoPID: UInt16?

        /// 90 kHz, the PES clock.
        public var inPTS: Int64 { Int64(inTime) * 2 }
        public var outPTS: Int64 { Int64(outTime) * 2 }
        public var duration: Double { Double(outTime &- inTime) / 45_000 }
    }

    public var items: [PlayItem]
    public var duration: Double { items.reduce(0) { $0 + $1.duration } }
    /// Every PlayItem has a dependent-view clip.
    public var isStereo: Bool { !items.isEmpty && items.allSatisfy { $0.dependentClip != nil } }

    public init(items: [PlayItem]) {
        self.items = items
    }

    public init(data: Data) throws {
        var header = BinaryReader(data)
        let magic = try header.ascii(4)
        guard magic == "MPLS" else { throw FormatError("not a playlist (\(magic))") }
        _ = try header.ascii(4) // version: 0100, 0200 (3D), 0300 (UHD)
        let playListStart = Int(try header.u32())
        _ = try header.u32() // PlayListMark
        let extensionStart = Int(try header.u32())

        var r = BinaryReader(data, offset: playListStart)
        _ = try r.u32() // length
        try r.skip(2)
        let playItemCount = Int(try r.u16())
        _ = try r.u16() // SubPaths: menus, PiP, audio; the 3D one is in the extension
        var items: [PlayItem] = []
        for _ in 0 ..< playItemCount {
            let length = Int(try r.u16())
            var item = try r.sub(length)
            items.append(try Self.parsePlayItem(&item))
        }

        if extensionStart > 0, extensionStart < data.count {
            try Self.applyExtensions(BinaryReader(data, offset: extensionStart), to: &items)
        }
        self.items = items
    }

    private static func parsePlayItem(_ r: inout BinaryReader) throws -> PlayItem {
        let clip = try r.ascii(5)
        _ = try r.ascii(4) // M2TS
        let flags = try r.u16()
        let multiAngle = flags & 0x10 != 0
        _ = try r.u8() // ref_to_STC_id
        let inTime = try r.u32()
        let outTime = try r.u32()
        try r.skip(8) // UO_mask_table
        try r.skip(1) // random_access_flag
        try r.skip(3) // still_mode, still_time
        if multiAngle {
            let angles = Int(try r.u8())
            try r.skip(1)
            try r.skip(max(angles - 1, 0) * 10) // further angles: clip name, codec, STC id
        }
        var item = PlayItem(clip: clip, inTime: inTime, outTime: outTime)
        let stnLength = Int(try r.u16())
        var stn = try r.sub(stnLength)
        item.videoPID = try parseSTNPrimaryVideo(&stn)
        return item
    }

    /// The first primary video stream's PID from an STN_table.
    private static func parseSTNPrimaryVideo(_ r: inout BinaryReader) throws -> UInt16? {
        try r.skip(2)
        let videoCount = try r.u8()
        try r.skip(11) // six other stream counts, 5 reserved
        guard videoCount > 0 else { return nil }
        return try parseStreamEntryPID(&r)
    }

    /// stream_entry(): length, type, then the PID where the type puts it.
    private static func parseStreamEntryPID(_ r: inout BinaryReader) throws -> UInt16? {
        let length = Int(try r.u8())
        var e = try r.sub(length)
        switch try e.u8() {
        case 1: return try e.u16()                       // in the main clip
        case 2: try e.skip(2); return try e.u16()        // SubPath id, subclip id
        case 3, 4: try e.skip(1); return try e.u16()     // SubPath id
        default: return nil
        }
    }

    // MARK: ExtensionData

    private static func applyExtensions(_ base: BinaryReader, to items: inout [PlayItem]) throws {
        for var block in try ExtensionData.blocks(base) {
            switch block.id {
            case (2, 1): try applySTNTableSS(&block.reader, to: &items)
            case (2, 2): try applySubPathExtension(&block.reader, to: &items)
            default: continue
            }
        }
    }

    /// SubPath entries extension: the type-8 SubPath's SubPlayItems name the
    /// dependent clip of each PlayItem (sync_PlayItem_id).
    private static func applySubPathExtension(_ r: inout BinaryReader, to items: inout [PlayItem]) throws {
        _ = try r.u32()
        let count = Int(try r.u16())
        for _ in 0 ..< count {
            let length = Int(try r.u32())
            var sp = try r.sub(length)
            try sp.skip(1)
            let type = try sp.u8()
            try sp.skip(3) // repeat flag, reserved
            let subItems = Int(try sp.u8())
            guard type == 8 else { continue }
            for index in 0 ..< subItems {
                let itemLength = Int(try sp.u16())
                var s = try sp.sub(itemLength)
                let clip = try s.ascii(5)
                try s.skip(4) // codec
                let flags = try s.u32()
                try s.skip(1) // STC id
                try s.skip(8) // IN, OUT
                let syncItem = Int(try s.u16())
                // One SubPlayItem per PlayItem in order; prefer the explicit sync id.
                let target = items.indices.contains(syncItem) ? syncItem : index
                if items.indices.contains(target) { items[target].dependentClip = clip }
                if flags & 1 != 0 { break } // multi-clip entries: angles, not handled
            }
        }
    }

    /// STN_table_SS, one per PlayItem: the dependent view's stream entry.
    private static func applySTNTableSS(_ r: inout BinaryReader, to items: inout [PlayItem]) throws {
        for index in items.indices {
            guard r.remaining >= 2 else { return }
            let length = Int(try r.u16())
            var t = try r.sub(length)
            try t.skip(2) // fixed offset flag, reserved
            if let pid = try parseStreamEntryPID(&t) { items[index].dependentVideoPID = pid }
        }
    }
}
