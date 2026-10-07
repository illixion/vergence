/*
 mvc3d - reading a Blu-ray disc image (UDF 2.50) without mounting it

 Read-only, and only as much of UDF (ECMA-167 + OSTA UDF 2.50) as Blu-ray
 discs use. The image is any `ByteSource`, so the same code reads a local
 `.iso` or one served over HTTP range requests.

 How a path resolves:
 1. Anchor Volume Descriptor Pointer, sector 256 → the Volume Descriptor
    Sequence: Partition Descriptors (where each partition starts) and the
    Logical Volume Descriptor (block size, partition maps, where the File Set
    Descriptor is).
 2. Blu-ray maps a *metadata partition* (UDF 2.50 type-2 map, "*UDF Metadata
    Partition") on top of the physical one: directories and file entries are
    addressed in it, and its blocks are found through the extents of the
    metadata file, itself a file entry in the physical partition.
 3. File Set Descriptor → root directory's file entry → File Identifier
    Descriptors, one per name → each file's (extended) file entry, whose
    allocation descriptors list where its bytes are.
 */

import Foundation

public final class UDFFileSystem: DiscFileSystem {
    public static let sectorSize = 2048

    private let image: ByteSource
    private var blockSize = UDFFileSystem.sectorSize
    /// Partition number → first sector.
    private var partitionStarts: [UInt16: Int64] = [:]
    private var maps: [PartitionMap] = []
    private var metadataExtents: [Extent] = []
    private var root: FileEntry!
    private var directoryCache: [String: [String: DirectoryEntry]] = [:]

    private enum PartitionMap {
        case physical(partitionNumber: UInt16)
        case metadata(partitionNumber: UInt16, fileLocation: UInt32)
    }

    /// Where a block lives: partition reference (index into the map table)
    /// and block within it.
    struct Address: Equatable { var partition: UInt16; var block: UInt32 }

    /// A run of a file's bytes on the image.
    struct Extent {
        var fileOffset: Int64
        var length: Int64
        /// Byte offset on the image; nil for an unrecorded extent (zeros).
        var imageOffset: Int64?
    }

    struct FileEntry {
        var size: Int64
        var extents: [Extent]
        /// Data held inside the entry itself (small directories).
        var embedded: Data?
    }

    struct DirectoryEntry {
        var isDirectory: Bool
        var icb: Address
    }

    public init(image: ByteSource) throws {
        self.image = image
        try readVolume()
    }

    // MARK: DiscFileSystem

    public func contentsOfDirectory(_ path: String) throws -> [String] {
        Array(try directory(at: path).values.map(\.name))
    }

    public func open(_ path: String) throws -> ByteSource {
        let (dir, name) = split(path)
        guard let entry = try directory(at: dir)[name.uppercased()], !entry.entry.isDirectory else {
            throw CocoaError(.fileReadNoSuchFile, userInfo: [NSFilePathErrorKey: path])
        }
        return UDFFile(image: image, entry: try fileEntry(at: entry.entry.icb))
    }

    // MARK: Volume

    private func readVolume() throws {
        var anchor = BinaryReader(try sector(256))
        guard try tag(&anchor) == 2 else { throw FormatError("no UDF anchor at sector 256") }
        let vdsLength = Int(try anchor.u32le())
        let vdsStart = Int64(try anchor.u32le())

        var fsd: Address?
        for i in 0 ..< Int64(max(vdsLength / Self.sectorSize, 1)) {
            let data = try sector(vdsStart + i)
            var r = BinaryReader(data)
            switch try tag(&r) {
            case 5: // Partition Descriptor
                var p = BinaryReader(data, offset: 22)
                let number = try p.u16le()
                var q = BinaryReader(data, offset: 188)
                partitionStarts[number] = Int64(try q.u32le())
            case 6: // Logical Volume Descriptor
                var l = BinaryReader(data, offset: 212)
                blockSize = Int(try l.u32le())
                var c = BinaryReader(data, offset: 248)
                _ = try c.u32le()
                fsd = Address(block: try c.u32le(), partition: try c.u16le())
                var m = BinaryReader(data, offset: 268)
                let count = Int(try m.u32le())
                var maps = BinaryReader(data, offset: 440)
                for _ in 0 ..< count { self.maps.append(try parseMap(&maps)) }
            case 8: // Terminating Descriptor
                break
            default:
                continue
            }
        }
        guard blockSize == Self.sectorSize else { throw FormatError("UDF block size \(blockSize) unsupported") }
        guard let fsd, !maps.isEmpty else { throw FormatError("no logical volume") }

        if let metadata = maps.firstIndex(where: { if case .metadata = $0 { true } else { false } }),
           case let .metadata(partitionNumber, location) = maps[metadata] {
            // The metadata file lives in the physical partition.
            guard let physical = maps.firstIndex(where: { if case let .physical(n) = $0 { n == partitionNumber } else { false } })
            else { throw FormatError("metadata partition without its physical partition") }
            metadataExtents = try fileEntry(at: Address(block: location, partition: UInt16(physical))).extents
        }

        var set = BinaryReader(try block(fsd))
        guard try tag(&set) == 256 else { throw FormatError("no File Set Descriptor") }
        var rootICB = BinaryReader(try block(fsd), offset: 400)
        root = try fileEntry(at: try longAddress(&rootICB))
    }

    private func parseMap(_ r: inout BinaryReader) throws -> PartitionMap {
        let type = try r.u8()
        let length = Int(try r.u8())
        var m = try r.sub(length - 2)
        switch type {
        case 1:
            _ = try m.u16le()
            return .physical(partitionNumber: try m.u16le())
        case 2:
            try m.skip(2)
            let identifier = String(decoding: try m.bytes(32).dropFirst().prefix(23), as: UTF8.self)
            _ = try m.u16le()
            let number = try m.u16le()
            guard identifier.hasPrefix("*UDF Metadata Partition") else {
                // Sparable / virtual maps are for rewritable and
                // multisession media, never pressed Blu-ray.
                throw FormatError("unsupported UDF partition map \(identifier)")
            }
            return .metadata(partitionNumber: number, fileLocation: try m.u32le())
        default:
            throw FormatError("unknown UDF partition map type \(type)")
        }
    }

    // MARK: Blocks and entries

    private func sector(_ n: Int64) throws -> Data {
        let data = try image.read(at: n * Int64(Self.sectorSize), count: Self.sectorSize)
        guard data.count == Self.sectorSize else { throw FormatError("image ends at sector \(n)") }
        return data
    }

    /// Image byte offset of a block in a partition.
    private func imageOffset(_ a: Address) throws -> Int64 {
        guard maps.indices.contains(Int(a.partition)) else { throw FormatError("bad partition reference \(a.partition)") }
        switch maps[Int(a.partition)] {
        case let .physical(number):
            guard let start = partitionStarts[number] else { throw FormatError("no partition \(number)") }
            return (start + Int64(a.block)) * Int64(blockSize)
        case .metadata:
            let offset = Int64(a.block) * Int64(blockSize)
            guard let e = metadataExtents.first(where: { offset >= $0.fileOffset && offset < $0.fileOffset + $0.length }),
                  let base = e.imageOffset else { throw FormatError("metadata block \(a.block) unmapped") }
            return base + offset - e.fileOffset
        }
    }

    private func block(_ a: Address) throws -> Data {
        let data = try image.read(at: try imageOffset(a), count: blockSize)
        guard data.count == blockSize else { throw FormatError("image ends in a block") }
        return data
    }

    /// Descriptor tag identifier; leaves the reader after the 16-byte tag.
    private func tag(_ r: inout BinaryReader) throws -> UInt16 {
        let id = try r.u16le()
        try r.skip(14)
        return id
    }

    private func longAddress(_ r: inout BinaryReader) throws -> Address {
        _ = try r.u32le()
        let a = Address(block: try r.u32le(), partition: try r.u16le())
        try r.skip(6)
        return a
    }

    /// A File Entry (261) or Extended File Entry (266), with its extents.
    func fileEntry(at a: Address) throws -> FileEntry {
        let data = try block(a)
        var r = BinaryReader(data)
        let id = try tag(&r)
        guard id == 261 || id == 266 else { throw FormatError("expected a file entry at \(a.block), found tag \(id)") }
        var flagsReader = BinaryReader(data, offset: 34)
        let adType = try flagsReader.u16le() & 0x7
        var sizeReader = BinaryReader(data, offset: 56)
        let size = Int64(try sizeReader.u64le())
        var lengths = BinaryReader(data, offset: id == 266 ? 208 : 168)
        let eaLength = Int(try lengths.u32le())
        let adLength = Int(try lengths.u32le())
        let adStart = (id == 266 ? 216 : 176) + eaLength
        guard adStart + adLength <= data.count else { throw FormatError("file entry overflows its block") }
        let descriptors = data[(data.startIndex + adStart) ..< (data.startIndex + adStart + adLength)]
        if adType == 3 {
            return FileEntry(size: size, extents: [], embedded: Data(descriptors))
        }
        var extents: [Extent] = []
        try appendExtents(descriptors, type: adType, partition: a.partition, to: &extents)
        return FileEntry(size: size, extents: extents, embedded: nil)
    }

    private func appendExtents(_ descriptors: Data, type: UInt16, partition: UInt16, to extents: inout [Extent]) throws {
        var r = BinaryReader(descriptors)
        var fileOffset = extents.last.map { $0.fileOffset + $0.length } ?? 0
        while r.remaining >= (type == 1 ? 16 : 8) {
            let raw = try r.u32le()
            let length = Int64(raw & 0x3FFF_FFFF)
            let kind = raw >> 30
            let address: Address
            switch type {
            case 0: address = Address(block: try r.u32le(), partition: partition)
            case 1:
                address = Address(block: try r.u32le(), partition: try r.u16le())
                try r.skip(6)
            default: throw FormatError("extended allocation descriptors unsupported")
            }
            if length == 0 { break }
            if kind == 3 {
                // Continuation: the list goes on in an Allocation Extent Descriptor.
                let next = try block(address)
                var t = BinaryReader(next)
                guard try tag(&t) == 258 else { throw FormatError("bad allocation extent") }
                _ = try t.u32le()
                let count = Int(try t.u32le())
                let more = next[(next.startIndex + 24) ..< (next.startIndex + min(24 + count, next.count))]
                try appendExtents(Data(more), type: type, partition: partition, to: &extents)
                return
            }
            let offset = kind == 0 ? try imageOffset(address) : nil // 1, 2: allocated or not, unrecorded
            extents.append(Extent(fileOffset: fileOffset, length: length, imageOffset: offset))
            fileOffset += length
        }
    }

    // MARK: Directories

    private func split(_ path: String) -> (String, String) {
        let parts = path.split(separator: "/").map(String.init)
        return (parts.dropLast().joined(separator: "/"), parts.last ?? "")
    }

    /// Upper-cased name → entry; Blu-ray names are ASCII and players match
    /// them case-insensitively.
    private func directory(at path: String) throws -> [String: (name: String, entry: DirectoryEntry)] {
        var current = root!
        var listing = try entries(of: current)
        for component in path.split(separator: "/") where !component.isEmpty {
            guard let next = listing[component.uppercased()], next.entry.isDirectory else {
                throw CocoaError(.fileReadNoSuchFile, userInfo: [NSFilePathErrorKey: path])
            }
            current = try fileEntry(at: next.entry.icb)
            listing = try entries(of: current)
        }
        return listing
    }

    private func entries(of directory: FileEntry) throws -> [String: (name: String, entry: DirectoryEntry)] {
        let data = try UDFFile(image: image, entry: directory).read(at: 0, count: Int(directory.size))
        var result: [String: (name: String, entry: DirectoryEntry)] = [:]
        var r = BinaryReader(data)
        while r.remaining >= 38 {
            let start = r.offset
            guard try tag(&r) == 257 else { break }
            try r.skip(2) // file version
            let characteristics = try r.u8()
            let nameLength = Int(try r.u8())
            let icb = try longAddress(&r)
            let implementationLength = Int(try r.u16le())
            try r.skip(implementationLength)
            let name = Self.decodeName(try r.bytes(nameLength))
            let total = 38 + implementationLength + nameLength
            r.offset = start + (total + 3) & ~3
            let deleted = characteristics & 0x04 != 0, parent = characteristics & 0x08 != 0
            if deleted || parent || name.isEmpty { continue }
            result[name.uppercased()] = (name, DirectoryEntry(isDirectory: characteristics & 0x02 != 0, icb: icb))
        }
        return result
    }

    /// OSTA compressed Unicode: 8 = one byte per character, 16 = UTF-16BE.
    static func decodeName(_ d: Data) -> String {
        guard let id = d.first else { return "" }
        let body = d.dropFirst()
        switch id {
        case 8: return String(decoding: body.map { UInt16($0) }, as: UTF16.self)
        case 16:
            let units = stride(from: body.startIndex, to: body.endIndex - 1, by: 2).map { UInt16(body[$0]) << 8 | UInt16(body[$0 + 1]) }
            return String(decoding: units, as: UTF16.self)
        default: return ""
        }
    }
}

extension UDFFileSystem.Address {
    init(block: UInt32, partition: UInt16) {
        self.partition = partition
        self.block = block
    }
}

/// A file inside the image, read through its extents.
final class UDFFile: ByteSource {
    private let image: ByteSource
    private let entry: UDFFileSystem.FileEntry
    let size: Int64

    init(image: ByteSource, entry: UDFFileSystem.FileEntry) {
        self.image = image
        self.entry = entry
        size = entry.size
    }

    func read(at offset: Int64, count: Int) throws -> Data {
        let end = min(offset + Int64(count), size)
        guard offset < end else { return Data() }
        if let embedded = entry.embedded {
            return embedded.subdata(in: Int(offset) ..< Int(min(end, Int64(embedded.count))))
        }
        var out = Data(capacity: Int(end - offset))
        var position = offset
        // Extents are sorted by file offset: find the first by binary search.
        var lo = 0, hi = entry.extents.count
        while lo < hi {
            let mid = (lo + hi) / 2
            let e = entry.extents[mid]
            if e.fileOffset + e.length <= position { lo = mid + 1 } else { hi = mid }
        }
        var index = lo
        while position < end, index < entry.extents.count {
            let e = entry.extents[index]
            let within = position - e.fileOffset
            let take = Int(min(e.length - within, end - position))
            if let base = e.imageOffset {
                let chunk = try image.read(at: base + within, count: take)
                guard chunk.count == take else { throw FormatError("image ends inside a file") }
                out.append(chunk)
            } else {
                out.append(Data(count: take))
            }
            position += Int64(take)
            index += 1
        }
        return out
    }
}
