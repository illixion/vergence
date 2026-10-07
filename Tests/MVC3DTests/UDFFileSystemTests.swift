import Foundation
import Testing
@testable import MVC3D

final class MemoryByteSource: ByteSource {
    let data: Data
    var size: Int64 { Int64(data.count) }
    init(_ data: Data) { self.data = data }
    func read(at offset: Int64, count: Int) throws -> Data {
        let start = Int(min(offset, size)), end = Int(min(offset + Int64(count), size))
        return data.subdata(in: start ..< end)
    }
}

/// A minimal Blu-ray-shaped UDF 2.50 image: physical partition at sector
/// 300, metadata partition whose file maps metadata blocks 0... onto
/// physical blocks 10..., directories with embedded entries, and a file in
/// two extents with a gap between them.
struct UDFImageBuilder {
    static let sector = 2048
    var image = Data(count: 420 * sector)
    let partitionStart = 300
    let metadataBase = 10 // metadata block n = physical block 10 + n

    mutating func put(_ bytes: Data, sector s: Int, at offset: Int = 0) {
        image.replaceSubrange((s * Self.sector + offset) ..< (s * Self.sector + offset + bytes.count), with: bytes)
    }

    func tag(_ id: UInt16) -> Data { le16(id) + Data(count: 14) }
    func le16(_ v: UInt16) -> Data { Data([UInt8(v & 0xFF), UInt8(v >> 8)]) }
    func le32(_ v: UInt32) -> Data { Data((0 ..< 4).map { UInt8(v >> (8 * $0) & 0xFF) }) }
    func le64(_ v: UInt64) -> Data { Data((0 ..< 8).map { UInt8(v >> (8 * $0) & 0xFF) }) }
    func longAD(length: UInt32, block: UInt32, partition: UInt16) -> Data {
        le32(length) + le32(block) + le16(partition) + Data(count: 6)
    }

    /// An Extended File Entry with the given allocation descriptors.
    func fileEntry(size: Int, adType: UInt16, descriptors: Data) -> Data {
        var fe = tag(266) + Data(count: 240)
        fe.replaceSubrange(34 ..< 36, with: le16(adType))
        fe.replaceSubrange(56 ..< 64, with: le64(UInt64(size)))
        fe.replaceSubrange(208 ..< 212, with: le32(0))
        fe.replaceSubrange(212 ..< 216, with: le32(UInt32(descriptors.count)))
        return fe.prefix(216) + descriptors
    }

    func fid(_ name: String, directory: Bool, icbBlock: UInt32, utf16: Bool = false) -> Data {
        let encoded: Data = utf16
            ? Data([16]) + Data(name.utf16.flatMap { [UInt8($0 >> 8), UInt8($0 & 0xFF)] })
            : Data([8]) + Data(name.utf8)
        var d = tag(257) + le16(1) + Data([directory ? 0x02 : 0x00, UInt8(encoded.count)])
        d += longAD(length: 2048, block: icbBlock, partition: 1) + le16(0) + encoded
        while d.count % 4 != 0 { d.append(0) }
        return d
    }

    func parentFID() -> Data {
        tag(257) + le16(1) + Data([0x0A, 0]) + longAD(length: 2048, block: 1, partition: 1) + le16(0) + Data(count: 2)
    }

    /// Writes a metadata-partition block.
    mutating func putMetadata(_ bytes: Data, block: Int) {
        put(bytes, sector: partitionStart + metadataBase + block)
    }

    mutating func build(fileContents: Data) -> Data {
        // Anchor → VDS at 257..259.
        put(tag(2) + le32(3 * 2048) + le32(257), sector: 256)
        var pd = tag(5) + Data(count: 200)
        pd.replaceSubrange(22 ..< 24, with: le16(0))
        pd.replaceSubrange(188 ..< 192, with: le32(UInt32(partitionStart)))
        put(pd, sector: 257)

        var lvd = tag(6) + Data(count: 600)
        lvd.replaceSubrange(212 ..< 216, with: le32(2048))
        lvd.replaceSubrange(248 ..< 264, with: longAD(length: 2048, block: 0, partition: 1)) // FSD: metadata block 0
        lvd.replaceSubrange(268 ..< 272, with: le32(2))
        let type1 = Data([1, 6]) + le16(1) + le16(0)
        var type2 = Data([2, 64]) + Data(count: 62)
        let identifier = Data([0]) + Data("*UDF Metadata Partition".utf8) + Data(count: 8)
        type2.replaceSubrange(4 ..< 36, with: identifier)
        type2.replaceSubrange(36 ..< 38, with: le16(1))
        type2.replaceSubrange(38 ..< 40, with: le16(0))   // on partition 0
        type2.replaceSubrange(40 ..< 44, with: le32(0))   // metadata file at physical block 0
        lvd.replaceSubrange(440 ..< 440 + 70, with: type1 + type2)
        put(lvd, sector: 258)
        put(tag(8), sector: 259)

        // Metadata file: one short_ad, 8 blocks at physical block 10.
        put(fileEntry(size: 8 * 2048, adType: 0, descriptors: le32(8 * 2048) + le32(UInt32(metadataBase))), sector: partitionStart)

        // FSD (metadata block 0): root directory ICB at metadata block 1.
        var fsd = tag(256) + Data(count: 500)
        fsd.replaceSubrange(400 ..< 416, with: longAD(length: 2048, block: 1, partition: 1))
        putMetadata(fsd, block: 0)

        // Root (block 1): BDMV directory at block 2.
        let rootFIDs = parentFID() + fid("BDMV", directory: true, icbBlock: 2)
        putMetadata(fileEntry(size: rootFIDs.count, adType: 3, descriptors: rootFIDs), block: 1)

        // BDMV (block 2): index.bdmv (block 3) and a UTF-16 named file (block 4).
        let bdmvFIDs = parentFID() + fid("index.bdmv", directory: false, icbBlock: 3)
            + fid("Ünïcode.bin", directory: false, icbBlock: 4, utf16: true)
        putMetadata(fileEntry(size: bdmvFIDs.count, adType: 3, descriptors: bdmvFIDs), block: 2)

        // index.bdmv: two long_ad extents on physical blocks 100 and 103.
        let first = 2048 + 100, second = fileContents.count - first
        precondition(second > 0 && second <= 2 * 2048)
        let ads = longAD(length: UInt32(first), block: 100, partition: 0) + longAD(length: UInt32(second), block: 103, partition: 0)
        putMetadata(fileEntry(size: fileContents.count, adType: 1, descriptors: ads), block: 3)
        put(fileContents.prefix(first), sector: partitionStart + 100)
        put(fileContents.dropFirst(first), sector: partitionStart + 103)

        putMetadata(fileEntry(size: 5, adType: 3, descriptors: Data("hello".utf8)), block: 4)
        return image
    }
}

@Test func readsDirectoriesAndFilesThroughTheMetadataPartition() throws {
    var builder = UDFImageBuilder()
    let contents = Data((0 ..< 5000).map { UInt8($0 % 251) })
    let fs = try UDFFileSystem(image: MemoryByteSource(builder.build(fileContents: contents)))

    #expect(try fs.contentsOfDirectory("").sorted() == ["BDMV"])
    #expect(try fs.contentsOfDirectory("BDMV").sorted() == ["index.bdmv", "Ünïcode.bin"])

    let file = try fs.open("BDMV/INDEX.BDMV") // names match case-insensitively
    #expect(file.size == 5000)
    #expect(try file.read(at: 0, count: 5000) == contents)
    // A read across the boundary between the two extents.
    #expect(try file.read(at: 2000, count: 500) == contents[2000 ..< 2500])
    #expect(try file.read(at: 4990, count: 100) == contents[4990...])
    #expect(try fs.open("BDMV/Ünïcode.bin").read(at: 0, count: 10) == Data("hello".utf8))
}

@Test func missingFilesAndNonImagesThrow() throws {
    var builder = UDFImageBuilder()
    let fs = try UDFFileSystem(image: MemoryByteSource(builder.build(fileContents: Data(count: 3000))))
    #expect(throws: (any Error).self) { try fs.open("BDMV/missing.bin") }
    #expect(throws: (any Error).self) { try fs.open("BDMV") }
    #expect(throws: FormatError.self) { try UDFFileSystem(image: MemoryByteSource(Data(count: 600 * 2048))) }
}
