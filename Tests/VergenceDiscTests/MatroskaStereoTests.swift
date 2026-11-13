import Foundation
import Testing
@testable import VergenceDisc

/// Matroska files built here around `Fixtures/x264-ref4-bframes2.264` (see
/// DecodeOrderTests): the stream twice over (two GOPs, 80 frames), one
/// Cluster per GOP, Cues behind a SeekHead after the Clusters, the way
/// MakeMKV and mkvmerge lay a file out. The picture is 2D; what is checked is
/// the container side: NAL rewriting, timestamps, Cues, the MVC records.
private let fixture = Bundle.module.url(forResource: "x264-ref4-bframes2", withExtension: "264", subdirectory: "Fixtures")!

/// Display index of each access unit of one GOP, in decode order.
private let displayOrder: [Int64] = [0, 3, 1, 2, 6, 4, 5, 9, 7, 8, 12, 10, 11, 15, 13, 14, 18, 16, 17, 21, 19, 20,
                                     24, 22, 23, 27, 25, 26, 30, 28, 29, 33, 31, 32, 36, 34, 35, 39, 37, 38]

/// A frame's time in milliseconds at 24000/1001 fps, as a remux rounds it.
private func milliseconds(frame: Int64) -> Int64 { (frame * 1001 * 1000 + 12_000) / 24_000 }

private func accessUnits() throws -> [Data] {
    let data = try Data(contentsOf: fixture)
    let starts = AnnexB.nalUnits(in: data).filter { $0.type == 9 }.map { $0.range.lowerBound - 3 }
    return zip(starts, starts.dropFirst() + [data.endIndex]).map { data[$0 ..< $1] }
}

/// EBML writing, with every size as an 8-byte number.
private enum Writer {
    static func id(_ id: UInt32) -> Data {
        var bytes = withUnsafeBytes(of: id.bigEndian) { Data($0) }
        while bytes.first == 0 { bytes.removeFirst() }
        return bytes
    }

    static func size(_ n: Int?) -> Data {
        guard let n else { return Data([0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]) }
        return Data([0x01]) + withUnsafeBytes(of: UInt64(n).bigEndian) { Data($0.dropFirst()) }
    }

    static func element(_ elementID: UInt32, _ body: Data, unknownSize: Bool = false) -> Data {
        id(elementID) + size(unknownSize ? nil : body.count) + body
    }

    static func uint(_ elementID: UInt32, _ value: UInt64) -> Data {
        element(elementID, withUnsafeBytes(of: value.bigEndian) { Data($0) })
    }

    static func string(_ elementID: UInt32, _ value: String) -> Data { element(elementID, Data(value.utf8)) }

    static func float(_ elementID: UInt32, _ value: Double) -> Data {
        element(elementID, withUnsafeBytes(of: value.bitPattern.bigEndian) { Data($0) })
    }
}

private struct SyntheticMatroska {
    var data = Data()
    /// Each Cluster's offset in the file.
    var clusterOffsets: [Int64] = []
}

/// - Parameters:
///   - codecPrivateTail: bytes after the avcC record (a legacy mvcC box).
///   - mvcMapping: an mvcC record carried as a BlockAdditionMapping.
///   - injectDelimiter: put a Blu-ray NAL 24 in front of every Block's slices.
///   - unknownSizeLastCluster: write the second Cluster with an unknown size.
private func makeMatroska(codecPrivateTail: Data = Data(), mvcMapping: Data? = nil, stereoMode: UInt64? = 13,
                          injectDelimiter: Bool = false, unknownSizeLastCluster: Bool = false) throws -> SyntheticMatroska {
    let units = try accessUnits()
    let first = AnnexB.nalUnits(in: units[0])
    let sps = units[0][first.first { $0.type == 7 }!.range]
    let pps = units[0][first.first { $0.type == 8 }!.range]
    var avcC = Data([1, sps[sps.startIndex + 1], sps[sps.startIndex + 2], sps[sps.startIndex + 3], 0xFF, 0xE1])
    avcC += Data([UInt8(sps.count >> 8), UInt8(sps.count & 0xFF)]) + sps
    avcC += Data([1, UInt8(pps.count >> 8), UInt8(pps.count & 0xFF)]) + pps
    if sps[sps.startIndex + 1] == 100 { avcC += Data([0xFD, 0xF8, 0xF8, 0x00]) } // High: chroma, depths, no SPS ext

    var video = Writer.uint(EBMLID.pixelWidth, 160) + Writer.uint(EBMLID.pixelHeight, 96)
    if let stereoMode { video += Writer.uint(EBMLID.stereoMode, stereoMode) }
    var entry = Writer.uint(EBMLID.trackNumber, 1) + Writer.uint(EBMLID.trackType, 1)
    entry += Writer.string(EBMLID.codecID, "V_MPEG4/ISO/AVC") + Writer.uint(EBMLID.defaultDuration, 41_708_333)
    entry += Writer.element(EBMLID.codecPrivate, avcC + codecPrivateTail) + Writer.element(EBMLID.video, video)
    if let mvcMapping {
        entry += Writer.element(EBMLID.blockAdditionMapping,
                                Writer.uint(EBMLID.blockAddIDType, 0x6D76_6343) + Writer.element(EBMLID.blockAddIDExtraData, mvcMapping))
    }
    let tracks = Writer.element(EBMLID.tracks, Writer.element(EBMLID.trackEntry, entry))
    let info = Writer.element(EBMLID.info, Writer.uint(EBMLID.timestampScale, 1_000_000) + Writer.float(EBMLID.duration, Double(milliseconds(frame: 80))))

    // Two GOPs: the fixture, then the fixture again 40 frames later.
    var clusters: [Data] = []
    for gop in 0 ..< 2 {
        let clusterTime = milliseconds(frame: Int64(gop) * 40)
        var body = Writer.uint(EBMLID.timestamp, UInt64(clusterTime))
        for (unit, display) in zip(units, displayOrder) {
            var payload = Data()
            for nal in AnnexB.nalUnits(in: unit) {
                if injectDelimiter, nal.type == 1 || nal.type == 5, payload.isEmpty == false {
                    payload += Data([0, 0, 0, 2, 0x18, 0x00])
                }
                payload += withUnsafeBytes(of: UInt32(nal.range.count).bigEndian) { Data($0) } + unit[nal.range]
            }
            let time = milliseconds(frame: Int64(gop) * 40 + display) - clusterTime
            var block = Data([0x81]) + withUnsafeBytes(of: Int16(time).bigEndian) { Data($0) }
            block.append(display == 0 ? 0x80 : 0x00)
            body += Writer.element(EBMLID.simpleBlock, block + payload)
        }
        clusters.append(Writer.element(EBMLID.cluster, body, unknownSize: unknownSizeLastCluster && gop == 1))
    }

    // Segment body: SeekHead, Info, Tracks, Clusters, Cues. The SeekHead is
    // written with a fixed-size position so its own length doesn't change.
    func seekHead(cues: UInt64) -> Data {
        Writer.element(EBMLID.seekHead, Writer.element(EBMLID.seek,
            Writer.element(EBMLID.seekID, Writer.id(EBMLID.cues)) + Writer.uint(EBMLID.seekPosition, cues)))
    }
    let head = Writer.element(EBMLID.ebml, Writer.string(EBMLID.docType, "matroska"))
    let segmentHeaderLength = Writer.id(EBMLID.segment).count + 8
    let beforeClusters = seekHead(cues: 0).count + info.count + tracks.count
    var result = SyntheticMatroska()
    var offset = Int64(head.count + segmentHeaderLength + beforeClusters)
    for cluster in clusters {
        result.clusterOffsets.append(offset)
        offset += Int64(cluster.count)
    }
    let segmentStart = Int64(head.count + segmentHeaderLength)
    var cuePoints = Data()
    for (gop, at) in result.clusterOffsets.enumerated() {
        cuePoints += Writer.element(EBMLID.cuePoint, Writer.uint(EBMLID.cueTime, UInt64(milliseconds(frame: Int64(gop) * 40)))
            + Writer.element(EBMLID.cueTrackPositions, Writer.uint(EBMLID.cueTrack, 1) + Writer.uint(EBMLID.cueClusterPosition, UInt64(at - segmentStart))))
    }
    let cues = Writer.element(EBMLID.cues, cuePoints)
    let body = seekHead(cues: UInt64(offset - segmentStart)) + info + tracks + clusters.reduce(Data(), +) + cues
    result.data = head + Writer.element(EBMLID.segment, body, unknownSize: unknownSizeLastCluster)
    return result
}

// MemoryByteSource: UDFFileSystemTests.swift.

/// A made-up MVC record: base SPS, subset SPS (type 15) and two PPS.
private func mvcRecord() -> Data {
    let sps = Data([0x67, 0x64, 0x00, 0x29, 0xAC])
    let subset = Data([0x6F, 0x80, 0x00, 0x29, 0x4B, 0x06])
    let pps = [Data([0x68, 0xEE, 0x3C]), Data([0x68, 0xDE, 0x3C, 0x80])]
    var record = Data([1, 0x80, 0x00, 0x29, 0xFF, 0xE2])
    for set in [sps, subset] { record += Data([0, UInt8(set.count)]) + set }
    record.append(UInt8(pps.count))
    for set in pps { record += Data([0, UInt8(set.count)]) + set }
    return record
}

private func decodedTimes(_ reader: MatroskaStereoReader, seek: Double? = nil) throws -> [Int64] {
    let decoder = try StereoDecoder(reader: reader)
    if let seek { try decoder.seek(to: seek) }
    var times: [Int64] = []
    try decoder.decode { frame in
        times.append(frame.pts ?? -1)
        return true
    }
    return times
}

@Test func headerGivesTheVideoTrackAndItsCues() throws {
    let file = try makeMatroska()
    let header = try MatroskaFile(source: MemoryByteSource(file.data))
    #expect(header.docType == "matroska")
    #expect(header.video?.number == 1)
    #expect(header.video?.width == 160)
    #expect(header.video?.stereoMode == 13)
    #expect(header.video?.avc?.lengthSize == 4)
    #expect(header.video?.isMVC == false)
    #expect(header.cues.map(\.clusterOffset) == file.clusterOffsets)
    #expect(abs((header.durationSeconds ?? 0) - 80 * 1001 / 24_000) < 0.002)
    #expect(header.firstClusterOffset == file.clusterOffsets.first)
    // Without Cues: only what tells an MVC track.
    #expect(try MatroskaFile(source: MemoryByteSource(file.data), readCues: false).cues.isEmpty)
}

@Test func blocksDecodeInDisplayOrderWithTheirTimes() throws {
    let file = try makeMatroska()
    let times = try decodedTimes(MatroskaStereoReader(source: MemoryByteSource(file.data)))
    #expect(times == (0 ..< 80).map { milliseconds(frame: $0) * 90 })
}

@Test func seekingStartsAtTheCueBeforeAndShowsFromTheTarget() throws {
    let file = try makeMatroska()
    let reader = try MatroskaStereoReader(source: MemoryByteSource(file.data))
    #expect(try reader.seek(to: 2.0) == milliseconds(frame: 40) * 90)
    let target = Double(milliseconds(frame: 50)) / 1000
    let times = try decodedTimes(reader, seek: target)
    #expect(times == (50 ..< 80).map { milliseconds(frame: $0) * 90 })
    // Back to the start after a seek: the parameter sets come again.
    #expect(try decodedTimes(reader, seek: 0).count == 80)
}

@Test func anUnknownSizeClusterEndsAtTheNextTopLevelElement() throws {
    let file = try makeMatroska(unknownSizeLastCluster: true)
    let times = try decodedTimes(MatroskaStereoReader(source: MemoryByteSource(file.data)))
    #expect(times.count == 80)
}

@Test func bluRayDependentDelimitersAreDropped() throws {
    let file = try makeMatroska(injectDelimiter: true)
    let reader = try MatroskaStereoReader(source: MemoryByteSource(file.data))
    let decoder = try StereoDecoder(reader: reader)
    var frames = 0
    try decoder.decode { _ in frames += 1; return true }
    #expect(frames == 80)
    #expect(decoder.unsupportedNALCount == 0)
    #expect(decoder.corruptNALCount == 0)
}

@Test func mvcConfigurationFromALegacyCodecPrivateBox() throws {
    let record = mvcRecord()
    // MakeMKV's box: its size field four bytes short of the record.
    let box = withUnsafeBytes(of: UInt32(record.count + 4).bigEndian) { Data($0) } + Data("mvcC".utf8) + record
    let header = try MatroskaFile(source: MemoryByteSource(makeMatroska(codecPrivateTail: box).data))
    let mvc = try #require(header.video?.mvc)
    #expect(mvc.sequenceParameterSets.map { $0.first! & 0x1F } == [7, 15])
    #expect(mvc.pictureParameterSets.count == 2)
    #expect(header.video?.isMVC == true)
    // The bare four-character code before the record, as the codec spec words it.
    let bare = try MatroskaFile(source: MemoryByteSource(makeMatroska(codecPrivateTail: Data("mvcC".utf8) + record).data))
    #expect(bare.video?.mvc == mvc)
}

@Test func mvcConfigurationFromABlockAdditionMapping() throws {
    let header = try MatroskaFile(source: MemoryByteSource(makeMatroska(mvcMapping: mvcRecord()).data))
    #expect(header.video?.mvc?.sequenceParameterSets.count == 2)
}

@Test func parameterSetsOfBothViewsGoAheadOfTheFirstUnit() throws {
    let file = try makeMatroska(mvcMapping: mvcRecord(), stereoMode: 14)
    let reader = try MatroskaStereoReader(source: MemoryByteSource(file.data))
    #expect(reader.track.baseIsRightEye)
    let first = try #require(try reader.next())
    // Both records' SPS (the base one, the made-up one, the subset SPS 15),
    // then every PPS, then the Block from its delimiter on.
    let leading = Array(AnnexB.nalUnits(in: first.data).map(\.type).prefix { $0 != 9 })
    #expect(leading == [7, 7, 15, 8, 8, 8])
    #expect(first.hasDependentView == false)
    // Only once: the next unit is the Block as it was.
    let second = try #require(try reader.next())
    #expect(AnnexB.nalUnits(in: second.data).first?.type == 9)
}

@Test func notMatroskaIsRefused() {
    #expect(throws: FormatError.self) { try MatroskaFile(source: MemoryByteSource(Data(repeating: 0x47, count: 400))) }
    #expect(!MatroskaFile.looksLikeMatroska(Data([0x47, 0x40, 0x00, 0x10])))
}
