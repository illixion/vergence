/*
 Vergence - Spherical Video V2 metadata in MP4

 Players outside Apple's (YouTube, Android and Quest players) learn that a
 video is stereo from Google's Spherical Video V2 `st3d` box in the video's
 sample entry, and that an audio track is ambisonic from the Spatial Audio
 `SA3D` box in its sample entry
 (github.com/google/spatial-media, docs/spherical-video-v2-rfc.md and
 docs/spatial-audio-rfc.md). AVFoundation writes neither, so these are
 added afterwards: the `moov` box is rewritten with the new boxes, chunk
 offsets are shifted when `moov` comes before the media data, and the rest
 of the file is left as it is.

 Pure Foundation, so it is tested on Linux too.
 */

import Foundation

public enum SphericalMetadata {
    /// `st3d` stereo_mode.
    public enum StereoMode: UInt8, Sendable {
        case mono = 0, topBottom = 1, leftRight = 2
    }

    public struct Found: Sendable, Equatable {
        public var stereoMode: StereoMode?
        /// Ambisonic order per audio track that has an `SA3D` box, in track order.
        public var ambisonicOrders: [Int]
    }

    public enum Error: Swift.Error, LocalizedError {
        case notMP4(String)

        public var errorDescription: String? {
            switch self {
            case .notMP4(let why): "not an MP4/QuickTime file the metadata can be added to: \(why)"
            }
        }
    }

    /// Adds `st3d` to every video sample entry and `SA3D` (ACN, SN3D) to the
    /// audio tracks in `ambisonicTracks` (track ID: channel count, which must
    /// be (order + 1)²; an AAC sample entry's own channel count often just
    /// says 2), replacing earlier ones.
    public static func write(to url: URL, stereoMode: StereoMode?, ambisonicTracks: [UInt32: Int] = [:]) throws {
        let file = try FileHandle(forUpdating: url)
        defer { try? file.close() }
        let top = try topLevelBoxes(file)
        guard let moovIndex = top.firstIndex(where: { $0.type == "moov" }) else { throw Error.notMP4("no moov box") }
        let moovBox = top[moovIndex]
        try file.seek(toOffset: moovBox.offset)
        guard let moovData = try file.read(upToCount: Int(moovBox.size)), moovData.count == Int(moovBox.size) else {
            throw Error.notMP4("moov is cut short")
        }
        var moov = try Node.parse(moovData, path: [])
        moov.edit(stereoMode: stereoMode, ambisonicTracks: ambisonicTracks)
        var newMoov = moov.serialized()
        let delta = Int64(newMoov.count) - Int64(moovBox.size)

        let isLast = moovIndex == top.count - 1
        if isLast || delta == 0 {
            if !isLast { newMoov = moov.serialized() }
            try file.seek(toOffset: moovBox.offset)
            try file.write(contentsOf: newMoov)
            if delta < 0 { try file.truncate(atOffset: moovBox.offset + UInt64(newMoov.count)) }
            return
        }
        // moov sits before the media: every chunk offset past it moves.
        moov.shiftChunkOffsets(after: moovBox.offset, by: delta)
        newMoov = moov.serialized()
        let temporary = url.deletingLastPathComponent().appendingPathComponent(".\(url.lastPathComponent).st3d")
        FileManager.default.createFile(atPath: temporary.path, contents: nil)
        let out = try FileHandle(forWritingTo: temporary)
        do {
            try copy(file, from: 0, count: moovBox.offset, to: out)
            try out.write(contentsOf: newMoov)
            let rest = moovBox.offset + moovBox.size
            try copy(file, from: rest, count: try file.seekToEnd() - rest, to: out)
            try out.close()
            try file.close()
            // Not replaceItemAt: Linux's Foundation differs there.
            try FileManager.default.removeItem(at: url)
            try FileManager.default.moveItem(at: temporary, to: url)
        } catch {
            try? out.close()
            try? FileManager.default.removeItem(at: temporary)
            throw error
        }
    }

    /// What a file carries.
    public static func read(_ url: URL) throws -> Found {
        let file = try FileHandle(forReadingFrom: url)
        defer { try? file.close() }
        guard let moovBox = try topLevelBoxes(file).first(where: { $0.type == "moov" }) else { throw Error.notMP4("no moov box") }
        try file.seek(toOffset: moovBox.offset)
        guard let data = try file.read(upToCount: Int(moovBox.size)) else { throw Error.notMP4("moov is cut short") }
        let moov = try Node.parse(data, path: [])
        var found = Found(stereoMode: nil, ambisonicOrders: [])
        moov.visit { node in
            if node.type == "st3d", node.payload.count >= 5 { found.stereoMode = StereoMode(rawValue: node.payload[node.payload.startIndex + 4]) }
            if node.type == "SA3D", node.payload.count >= 6 {
                let p = node.payload.startIndex
                found.ambisonicOrders.append(Int(Node.u32(node.payload, p + 2)))
            }
        }
        return found
    }

    // MARK: - Boxes

    struct TopLevel {
        let type: String
        let offset: UInt64
        let size: UInt64
    }

    static func topLevelBoxes(_ file: FileHandle) throws -> [TopLevel] {
        let end = try file.seekToEnd()
        var offset: UInt64 = 0
        var boxes: [TopLevel] = []
        while offset + 8 <= end {
            try file.seek(toOffset: offset)
            guard let header = try file.read(upToCount: 16), header.count >= 8 else { break }
            var size = UInt64(Node.u32(header, 0))
            let type = String(decoding: header[4 ..< 8], as: UTF8.self)
            if size == 1 {
                guard header.count == 16 else { throw Error.notMP4("cut short at \(type)") }
                size = Node.u64(header, 8)
            } else if size == 0 {
                size = end - offset
            }
            guard size >= 8, offset + size <= end else { throw Error.notMP4("bad size for \(type) at \(offset)") }
            boxes.append(TopLevel(type: type, offset: offset, size: size))
            offset += size
        }
        return boxes
    }

    static func copy(_ from: FileHandle, from offset: UInt64, count: UInt64, to: FileHandle) throws {
        try from.seek(toOffset: offset)
        var left = count
        while left > 0 {
            guard let chunk = try from.read(upToCount: Int(min(left, 8 << 20))), !chunk.isEmpty else { break }
            try to.write(contentsOf: chunk)
            left -= UInt64(chunk.count)
        }
    }

    /// A box as a tree, deep enough to reach sample entries and chunk offsets.
    struct Node {
        var type: String
        /// Bytes after the 8-byte header and before the children (all of
        /// them for a leaf).
        var payload: Data
        var children: [Node]?

        /// Containers on the way to sample entries, with the bytes their
        /// own fields take before the first child.
        static func childOffset(_ type: String, payload: Data, parent: [String]) -> Int? {
            switch type {
            case "moov", "trak", "mdia", "minf", "stbl", "edts", "dinf": return 0
            case "stsd": return 8
            case "hvc1", "hev1", "avc1", "avc3", "mp4v", "dvh1", "dvhe":
                return parent.last == "stsd" ? 78 : nil
            case "mp4a", "ac-3", "ec-3", "Opus", "alac", "fLaC", "lpcm":
                guard parent.last == "stsd", payload.count >= 10 else { return nil }
                // A QuickTime sound description's version adds fields.
                let version = u16(payload, payload.startIndex + 8)
                return version == 1 ? 44 : version == 2 ? 64 : 28
            default: return nil
            }
        }

        static func parse(_ data: Data, path: [String]) throws -> Node {
            let base = data.startIndex
            guard data.count >= 8 else { throw Error.notMP4("box cut short") }
            var size = Int(u32(data, base))
            let type = String(decoding: data[base + 4 ..< base + 8], as: UTF8.self)
            var header = 8
            if size == 1 {
                size = Int(u64(data, base + 8))
                header = 16
            }
            guard size >= header, size <= data.count else { throw Error.notMP4("bad size for \(type)") }
            let body = data[base + header ..< base + size]
            guard header == 8, let fields = childOffset(type, payload: body, parent: path), fields <= body.count else {
                return Node(type: type, payload: Data(body), children: nil, largeHeader: header == 16)
            }
            var children: [Node] = []
            var offset = body.startIndex + fields
            while offset + 8 <= body.endIndex {
                let child = try parse(data[offset ..< body.endIndex], path: path + [type])
                children.append(child)
                offset += child.encodedSize(from: data, at: offset)
            }
            return Node(type: type, payload: Data(body.prefix(fields)), children: children)
        }

        var largeHeader = false

        init(type: String, payload: Data, children: [Node]?, largeHeader: Bool = false) {
            self.type = type
            self.payload = payload
            self.children = children
            self.largeHeader = largeHeader
        }

        func encodedSize(from data: Data, at offset: Int) -> Int {
            let size = Int(Node.u32(data, offset))
            return size == 1 ? Int(Node.u64(data, offset + 8)) : size
        }

        func serialized() -> Data {
            var body = payload
            for child in children ?? [] { body.append(child.serialized()) }
            var out = Data()
            if largeHeader || body.count + 8 > Int(UInt32.max) {
                out.append(contentsOf: Node.be32(1))
                out.append(contentsOf: Array(type.utf8))
                out.append(contentsOf: Node.be64(UInt64(body.count + 16)))
            } else {
                out.append(contentsOf: Node.be32(UInt32(body.count + 8)))
                out.append(contentsOf: Array(type.utf8))
            }
            out.append(body)
            return out
        }

        func visit(_ body: (Node) -> Void) {
            body(self)
            for child in children ?? [] { child.visit(body) }
        }

        func first(_ type: String) -> Node? {
            if self.type == type { return self }
            for child in children ?? [] { if let found = child.first(type) { return found } }
            return nil
        }

        mutating func edit(stereoMode: StereoMode?, ambisonicTracks: [UInt32: Int]) {
            guard var tracks = children else { return }
            for i in tracks.indices where tracks[i].type == "trak" {
                let handler = tracks[i].first("hdlr").map { $0.payload.count >= 12 ? String(decoding: $0.payload[$0.payload.startIndex + 8 ..< $0.payload.startIndex + 12], as: UTF8.self) : "" }
                let trackID = tracks[i].first("tkhd").map { tkhd -> UInt32 in
                    let p = tkhd.payload, version = p.first ?? 0
                    return Node.u32(p, p.startIndex + (version == 1 ? 20 : 12))
                } ?? 0
                tracks[i].editSampleEntries { entry in
                    if handler == "vide", let stereoMode {
                        entry.children?.removeAll { $0.type == "st3d" }
                        entry.children?.append(Node(type: "st3d", payload: Data([0, 0, 0, 0, stereoMode.rawValue]), children: nil))
                    }
                    if handler == "soun", let channels = ambisonicTracks[trackID] {
                        let order = Int(Double(channels).squareRoot().rounded()) - 1
                        guard order >= 1, (order + 1) * (order + 1) == channels else { return }
                        entry.children?.removeAll { $0.type == "SA3D" }
                        var sa3d = Data([0, 0]) // version 0, ambisonic_type 0 (periphonic)
                        sa3d.append(contentsOf: Node.be32(UInt32(order)))
                        sa3d.append(contentsOf: [0, 0]) // ACN, SN3D
                        sa3d.append(contentsOf: Node.be32(UInt32(channels)))
                        for c in 0 ..< channels { sa3d.append(contentsOf: Node.be32(UInt32(c))) }
                        entry.children?.append(Node(type: "SA3D", payload: sa3d, children: nil))
                    }
                }
            }
            children = tracks
        }

        mutating func editSampleEntries(_ body: (inout Node) -> Void) {
            guard children != nil else { return }
            if type == "stsd" {
                for i in children!.indices where children![i].children != nil { body(&children![i]) }
                return
            }
            for i in children!.indices { children![i].editSampleEntries(body) }
        }

        mutating func shiftChunkOffsets(after start: UInt64, by delta: Int64) {
            if type == "stco" || type == "co64" {
                let wide = type == "co64"
                var p = payload
                let count = Int(Node.u32(p, p.startIndex + 4))
                for i in 0 ..< count {
                    let at = p.startIndex + 8 + i * (wide ? 8 : 4)
                    let value = wide ? Node.u64(p, at) : UInt64(Node.u32(p, at))
                    guard value >= start else { continue }
                    let moved = UInt64(Int64(value) + delta)
                    p.replaceSubrange(at ..< at + (wide ? 8 : 4), with: wide ? Node.be64(moved) : Node.be32(UInt32(moved)))
                }
                payload = p
                return
            }
            guard children != nil else { return }
            for i in children!.indices { children![i].shiftChunkOffsets(after: start, by: delta) }
        }

        // Children of stbl that hold chunk offsets are leaves, so stbl must
        // be a container: it is (childOffset 0).

        static func u16(_ d: Data, _ i: Int) -> UInt16 { UInt16(d[i]) << 8 | UInt16(d[i + 1]) }
        static func u32(_ d: Data, _ i: Int) -> UInt32 { (0 ..< 4).reduce(0) { $0 << 8 | UInt32(d[i + $1]) } }
        static func u64(_ d: Data, _ i: Int) -> UInt64 { (0 ..< 8).reduce(0) { $0 << 8 | UInt64(d[i + $1]) } }
        static func be32(_ v: UInt32) -> [UInt8] { [24, 16, 8, 0].map { UInt8((v >> $0) & 0xFF) } }
        static func be64(_ v: UInt64) -> [UInt8] { (0 ..< 8).reversed().map { UInt8((v >> UInt64($0 * 8)) & 0xFF) } }
    }
}
