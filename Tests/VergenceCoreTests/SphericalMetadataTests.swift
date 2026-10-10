import Foundation
import Testing
@testable import VergenceCore

@Suite struct SphericalMetadataTests {
    static func box(_ type: String, _ payload: [UInt8]) -> [UInt8] {
        let size = UInt32(payload.count + 8)
        return [24, 16, 8, 0].map { UInt8((size >> $0) & 0xFF) } + Array(type.utf8) + payload
    }

    static func be32(_ v: UInt32) -> [UInt8] { [24, 16, 8, 0].map { UInt8((v >> $0) & 0xFF) } }

    /// A track: tkhd (version 0) with `id`, hdlr `handler`, one sample entry
    /// and an stco pointing at `chunk`.
    static func track(id: UInt32, handler: String, entry: [UInt8], chunk: UInt32) -> [UInt8] {
        let tkhd = box("tkhd", [0, 0, 0, 0] + be32(0) + be32(0) + be32(id) + [UInt8](repeating: 0, count: 68))
        let hdlr = box("hdlr", [0, 0, 0, 0] + be32(0) + Array(handler.utf8) + [UInt8](repeating: 0, count: 13))
        let stsd = box("stsd", [0, 0, 0, 0] + be32(1) + entry)
        let stco = box("stco", [0, 0, 0, 0] + be32(1) + be32(chunk))
        let stbl = box("stbl", stsd + stco)
        return box("trak", tkhd + box("mdia", hdlr + box("minf", stbl)))
    }

    static let hvc1 = box("hvc1", [UInt8](repeating: 0, count: 78) + box("hvcC", [1, 2, 3]))
    /// mp4a version 0 (28 bytes of fields) saying 2 channels, with an esds.
    static let mp4a: [UInt8] = {
        var fields = [UInt8](repeating: 0, count: 28)
        fields[17] = 2
        return box("mp4a", fields + box("esds", [0, 0, 0, 0, 9]))
    }()

    /// ftyp, then moov and mdat in either order; chunk offsets point at the
    /// media's first byte.
    static func file(moovFirst: Bool) -> (data: Data, media: [UInt8]) {
        let ftyp = box("ftyp", Array("isom".utf8) + be32(0))
        let media: [UInt8] = Array(0 ..< 64)
        func moov(chunk: UInt32) -> [UInt8] {
            box("moov", track(id: 1, handler: "vide", entry: hvc1, chunk: chunk) + track(id: 2, handler: "soun", entry: mp4a, chunk: chunk))
        }
        let moovSize = moov(chunk: 0).count
        if moovFirst {
            let chunk = UInt32(ftyp.count + moovSize + 8)
            return (Data(ftyp + moov(chunk: chunk) + box("mdat", media)), media)
        }
        return (Data(ftyp + box("mdat", media) + moov(chunk: UInt32(ftyp.count + 8))), media)
    }

    /// The bytes the first stco points at.
    static func chunkBytes(_ url: URL, count: Int) throws -> [UInt8] {
        let data = try Data(contentsOf: url)
        let stco = try #require(data.range(of: Data("stco".utf8)))
        let offset = Int(SphericalMetadata.Node.u32(data, stco.upperBound + 8))
        return Array(data[offset ..< offset + count])
    }

    @Test(arguments: [true, false])
    func addsBoxesAndKeepsTheMediaFindable(moovFirst: Bool) throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("vergence-st3d-\(UUID().uuidString).mp4")
        defer { try? FileManager.default.removeItem(at: url) }
        let (data, media) = Self.file(moovFirst: moovFirst)
        try data.write(to: url)
        #expect(try Self.chunkBytes(url, count: media.count) == media)

        try SphericalMetadata.write(to: url, stereoMode: .leftRight, ambisonicTracks: [2: 4])
        var found = try SphericalMetadata.read(url)
        #expect(found.stereoMode == .leftRight)
        #expect(found.ambisonicOrders == [1])
        #expect(try Self.chunkBytes(url, count: media.count) == media)

        // Again, with something else: replaced, not added twice.
        try SphericalMetadata.write(to: url, stereoMode: .topBottom, ambisonicTracks: [2: 4])
        found = try SphericalMetadata.read(url)
        #expect(found.stereoMode == .topBottom)
        #expect(found.ambisonicOrders == [1])
        let rewritten = try Data(contentsOf: url)
        #expect(rewritten.ranges(of: Data("st3d".utf8)).count == 1)
        #expect(try Self.chunkBytes(url, count: media.count) == media)
    }

    @Test func theSpatialAudioBoxFollowsTheSpec() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("vergence-sa3d-\(UUID().uuidString).mp4")
        defer { try? FileManager.default.removeItem(at: url) }
        try Self.file(moovFirst: false).data.write(to: url)
        try SphericalMetadata.write(to: url, stereoMode: nil, ambisonicTracks: [2: 4])
        let data = try Data(contentsOf: url)
        let at = try #require(data.range(of: Data("SA3D".utf8))).lowerBound - 4
        // size 36, version 0, periphonic, order 1, ACN, SN3D, 4 channels, 0 1 2 3.
        var expected: [UInt8] = Self.be32(36)
        expected += Array("SA3D".utf8)
        expected += [0, 0]
        expected += Self.be32(1)
        expected += [0, 0]
        expected += Self.be32(4)
        for channel: UInt32 in 0 ..< 4 { expected += Self.be32(channel) }
        #expect(Array(data[at ..< at + 36]) == expected)
        // Not a 4-channel count: nothing written.
        try SphericalMetadata.write(to: url, stereoMode: nil, ambisonicTracks: [2: 5])
        #expect(try SphericalMetadata.read(url).ambisonicOrders == [1])
        #expect(try SphericalMetadata.read(url).stereoMode == nil)
    }
}
