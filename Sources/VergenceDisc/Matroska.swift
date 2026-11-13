/*
 Vergence - the parts of a Matroska file an MVC video track needs

 A 3D Blu-ray remuxed to Matroska (MakeMKV's "3D MVC", and the remuxes made
 from it) keeps both views in ONE track: `V_MPEG4/ISO/AVC` whose every Block is
 a whole stereo access unit, the base view's NAL units followed by the
 dependent view's (subset SPS 15, its PPS, the MVC slices, NAL 20). The
 dependent view's parameter sets are an `MVCDecoderConfigurationRecord`
 (ISO/IEC 14496-15), carried either as a BlockAdditionMapping of type `mvcC`
 or, the legacy way MakeMKV writes it, as an `mvcC` box appended to the
 CodecPrivate after the `avcC` record. Measured on a MakeMKV remux:
 CodecPrivate 421 bytes, `avcC` 129, then a 288-byte `mvcC` box holding the
 base SPS, the subset SPS and both PPS.

 Read here: the EBML header, the SeekHead, Info (timestamp scale and
 duration), Tracks (the first video track, its CodecPrivate, StereoMode and
 block addition mappings) and Cues (the video track's random access points).
 Clusters are read by `MatroskaStereoReader`. Everything comes from
 untrusted files: a malformed element throws `FormatError`, never traps.
 */

import Foundation

enum EBMLID {
    static let ebml: UInt32 = 0x1A45_DFA3
    static let docType: UInt32 = 0x4282
    static let segment: UInt32 = 0x1853_8067
    static let seekHead: UInt32 = 0x114D_9B74
    static let seek: UInt32 = 0x4DBB
    static let seekID: UInt32 = 0x53AB
    static let seekPosition: UInt32 = 0x53AC
    static let info: UInt32 = 0x1549_A966
    static let timestampScale: UInt32 = 0x2AD7B1
    static let duration: UInt32 = 0x4489
    static let tracks: UInt32 = 0x1654_AE6B
    static let trackEntry: UInt32 = 0xAE
    static let trackNumber: UInt32 = 0xD7
    static let trackType: UInt32 = 0x83
    static let codecID: UInt32 = 0x86
    static let codecPrivate: UInt32 = 0x63A2
    static let defaultDuration: UInt32 = 0x23E383
    static let video: UInt32 = 0xE0
    static let pixelWidth: UInt32 = 0xB0
    static let pixelHeight: UInt32 = 0xBA
    static let stereoMode: UInt32 = 0x53B8
    static let blockAdditionMapping: UInt32 = 0x41E4
    static let blockAddIDType: UInt32 = 0x41E7
    static let blockAddIDExtraData: UInt32 = 0x41ED
    static let contentEncodings: UInt32 = 0x6D80
    static let cues: UInt32 = 0x1C53_BB6B
    static let cuePoint: UInt32 = 0xBB
    static let cueTime: UInt32 = 0xB3
    static let cueTrackPositions: UInt32 = 0xB7
    static let cueTrack: UInt32 = 0xF7
    static let cueClusterPosition: UInt32 = 0xF1
    static let cueRelativePosition: UInt32 = 0xF0
    static let cluster: UInt32 = 0x1F43_B675
    static let timestamp: UInt32 = 0xE7
    static let simpleBlock: UInt32 = 0xA3
    static let blockGroup: UInt32 = 0xA0
    static let block: UInt32 = 0xA1
    static let referenceBlock: UInt32 = 0xFB
    static let tags: UInt32 = 0x1254_C367
    static let chapters: UInt32 = 0x1043_A770
    static let attachments: UInt32 = 0x1941_A469

    /// The Segment's children: where an unknown-size Cluster ends.
    static let topLevel: Set<UInt32> = [seekHead, info, tracks, cues, cluster, tags, chapters, attachments]
}

/// EBML variable-length numbers.
enum EBML {
    /// An element ID (marker bits kept) and its length, at `offset`.
    static func id(_ bytes: UnsafeRawBufferPointer, at offset: Int) -> (UInt32, Int)? {
        guard offset < bytes.count else { return nil }
        let first = bytes[offset]
        let length = first.leadingZeroBitCount + 1
        guard length <= 4, offset + length <= bytes.count else { return nil }
        var value = UInt32(first)
        for i in 1 ..< length { value = value << 8 | UInt32(bytes[offset + i]) }
        return (value, length)
    }

    /// An element size, nil for "unknown", and its length.
    static func size(_ bytes: UnsafeRawBufferPointer, at offset: Int) -> (UInt64?, Int)? {
        guard offset < bytes.count else { return nil }
        let first = bytes[offset]
        let length = first.leadingZeroBitCount + 1
        guard length <= 8, offset + length <= bytes.count else { return nil }
        var value = UInt64(first) & (0xFF >> UInt64(length))
        var allOnes = value == (0xFF >> UInt64(length))
        for i in 1 ..< length {
            let byte = bytes[offset + i]
            allOnes = allOnes && byte == 0xFF
            value = value << 8 | UInt64(byte)
        }
        return (allOnes ? nil : value, length)
    }

    static func unsigned(_ data: Data) -> UInt64 {
        data.prefix(8).reduce(0) { $0 << 8 | UInt64($1) }
    }

    static func float(_ data: Data) -> Double? {
        switch data.count {
        case 4: return Double(Float(bitPattern: UInt32(truncatingIfNeeded: unsigned(data))))
        case 8: return Double(bitPattern: unsigned(data))
        default: return nil
        }
    }
}

/// One element read from an in-memory buffer.
struct EBMLElement {
    var id: UInt32
    /// Offset of the body in the buffer it was read from.
    var bodyStart: Int
    /// Nil for an unknown size.
    var bodySize: Int?
    var headerLength: Int
}

/// Walks the children of an element held whole in memory.
struct EBMLChildren: Sequence, IteratorProtocol {
    let data: Data
    private var offset: Int
    private let end: Int

    init(_ data: Data, from start: Int = 0, to end: Int? = nil) {
        self.data = data
        offset = start
        self.end = Swift.min(end ?? data.count, data.count)
    }

    mutating func next() -> (id: UInt32, body: Data)? {
        guard offset < end else { return nil }
        let parsed: (UInt32, Int, UInt64?, Int)? = data.withUnsafeBytes { raw in
            guard let (id, idLength) = EBML.id(raw, at: offset),
                  let (size, sizeLength) = EBML.size(raw, at: offset + idLength) else { return nil }
            return (id, idLength, size, sizeLength)
        }
        guard let (id, idLength, size, sizeLength) = parsed else { offset = end; return nil }
        let bodyStart = offset + idLength + sizeLength
        // A child with an unknown size can't be skipped: stop here.
        guard let size, bodyStart + Int(clamping: size) <= end else { offset = end; return nil }
        let bodyEnd = bodyStart + Int(size)
        offset = bodyEnd
        return (id, data.subdata(in: data.startIndex + bodyStart ..< data.startIndex + bodyEnd))
    }
}

/// H.264 decoder configuration (`avcC`), and the MVC one (`mvcC`), which has
/// the same layout up to its parameter sets.
public struct AVCConfiguration: Sendable, Equatable {
    /// Bytes in each NAL unit's length prefix inside a Block.
    public var lengthSize: Int
    /// NAL units, header byte included: SPS (7) and subset SPS (15).
    public var sequenceParameterSets: [Data]
    public var pictureParameterSets: [Data]
    /// Bytes the record took, from its first byte.
    var recordLength: Int

    /// Parses an `AVCDecoderConfigurationRecord` (or MVC's, whose leading
    /// fields are the same). Trailing bytes are left for the caller.
    public init(record: Data) throws {
        var r = BinaryReader(record)
        guard try r.u8() == 1 else { throw FormatError("unknown avcC version") }
        let profile = try r.u8()
        try r.skip(2) // compatibility, level
        lengthSize = Int(try r.u8() & 0x03) + 1
        let spsCount = Int(try r.u8() & 0x1F)
        sequenceParameterSets = try (0 ..< spsCount).map { _ in try r.bytes(Int(r.u16())) }
        let ppsCount = Int(try r.u8())
        pictureParameterSets = try (0 ..< ppsCount).map { _ in try r.bytes(Int(r.u16())) }
        // High profiles add chroma format, bit depths and SPS extensions
        // (14496-15 5.3.3.1.2), which some writers leave out.
        if [100, 110, 122, 144].contains(profile), r.remaining >= 4 {
            var ext = r
            try ext.skip(3)
            let count = Int(try ext.u8())
            for _ in 0 ..< count { try ext.skip(Int(ext.u16())) }
            r = ext
        }
        recordLength = r.offset
    }

    /// The parameter sets as Annex B, every SPS before any PPS (a PPS names
    /// the SPS it refers to).
    var annexB: Data {
        var out = Data()
        for nal in sequenceParameterSets + pictureParameterSets where !nal.isEmpty {
            out.append(contentsOf: [0, 0, 0, 1])
            out.append(nal)
        }
        return out
    }
}

public struct MatroskaVideoTrack: Sendable {
    public var number: UInt64
    public var codecID: String
    public var width: Int?
    public var height: Int?
    /// Matroska's StereoMode, when stated: 13 = both eyes in one Block, left
    /// first; 14 = right first. (MakeMKV writes these for MVC; nothing in the
    /// spec reserves them for it.)
    public var stereoMode: Int?
    /// Nanoseconds per frame, when stated.
    public var defaultDuration: UInt64?
    /// The base view's configuration (H.264 tracks only).
    public var avc: AVCConfiguration?
    /// The dependent view's configuration: present for an MVC track.
    public var mvc: AVCConfiguration?
    /// The track's ContentEncodings are something other than none: its
    /// frames are compressed or encrypted, which this reader doesn't undo.
    public var hasContentEncoding = false

    /// Both views are in the track.
    public var isMVC: Bool { mvc != nil }
    /// The base view is the right eye (StereoMode 14). Blu-ray's base view is
    /// the left eye unless the disc says otherwise.
    public var baseIsRightEye: Bool { stereoMode == 14 }
}

public struct MatroskaCue: Sendable, Equatable {
    /// Timestamp ticks (see `MatroskaFile.timestampScale`).
    public var time: Int64
    /// Offset of the Cluster in the file.
    public var clusterOffset: Int64
}

/// A Matroska file's header, as far as stereo video needs it.
public struct MatroskaFile: Sendable {
    public var docType: String
    /// Nanoseconds per timestamp tick.
    public var timestampScale: Int64 = 1_000_000
    /// Seconds, when stated.
    public var durationSeconds: Double?
    public var video: MatroskaVideoTrack?
    /// The video track's random access points, by time.
    public var cues: [MatroskaCue] = []
    /// Where the Segment's body starts: positions in the SeekHead and Cues
    /// count from here.
    public var segmentStart: Int64
    /// Where the Segment ends (the file's end for an unknown size).
    public var segmentEnd: Int64
    public var firstClusterOffset: Int64?

    /// Whether the first bytes of a file are an EBML header (Matroska or WebM).
    public static func looksLikeMatroska(_ head: Data) -> Bool {
        head.count >= 4 && head.prefix(4).elementsEqual([0x1A, 0x45, 0xDF, 0xA3])
    }

    /// Ticks to 90 kHz.
    public func ninetyKHz(ticks: Int64) -> Int64 {
        // ticks * scale ns * 90_000 / 1e9, without overflow for long films.
        let ns = ticks.multipliedReportingOverflow(by: timestampScale)
        if !ns.overflow { return ns.partialValue / 100_000 * 9 + ns.partialValue % 100_000 * 9 / 100_000 }
        return Int64(Double(ticks) * Double(timestampScale) * 9e-5)
    }

    /// Reads the header. With `readCues` false the Cues (which can sit at the
    /// far end of the file) aren't fetched: enough to tell an MVC track.
    public init(source: ByteSource, readCues: Bool = true) throws {
        let head = try source.read(at: 0, count: 64 << 10)
        guard Self.looksLikeMatroska(head) else { throw FormatError("not a Matroska file") }
        var offset = 0
        guard let ebml = Self.element(in: head, at: offset), ebml.id == EBMLID.ebml, let ebmlSize = ebml.bodySize else {
            throw FormatError("bad EBML header")
        }
        let ebmlBody = head.subdata(in: ebml.bodyStart ..< min(head.count, ebml.bodyStart + ebmlSize))
        docType = EBMLChildren(ebmlBody).first { $0.id == EBMLID.docType }.map { String(decoding: $0.body, as: UTF8.self) } ?? "matroska"
        offset = ebml.bodyStart + ebmlSize
        guard let segment = Self.element(in: head, at: offset), segment.id == EBMLID.segment else {
            throw FormatError("no Segment after the EBML header")
        }
        let start = Int64(segment.bodyStart)
        segmentStart = start
        segmentEnd = min(segment.bodySize.map { start + Int64($0) } ?? source.size, source.size)

        // The Segment's children in file order, up to the first Cluster;
        // the SeekHead says where the ones after it are.
        var positions: [UInt32: Int64] = [:]
        var parsed: Set<UInt32> = []
        var position = segmentStart
        var video: MatroskaVideoTrack?
        while position < segmentEnd {
            let probe = try source.read(at: position, count: 16)
            guard let element = Self.element(in: probe, at: 0) else { break }
            if element.id == EBMLID.cluster {
                firstClusterOffset = position
                break
            }
            guard let size = element.bodySize else { break }
            let bodyOffset = position + Int64(element.headerLength)
            if [EBMLID.seekHead, EBMLID.info, EBMLID.tracks, EBMLID.cues].contains(element.id) {
                let body = try Self.read(source, at: bodyOffset, count: size, limit: 16 << 20)
                parse(element.id, body, positions: &positions, video: &video, readCues: readCues)
                parsed.insert(element.id)
            }
            position = bodyOffset + Int64(size)
        }
        // Elements only the SeekHead points at (Cues after the Clusters, or
        // Tracks after a large attachment).
        for id in [EBMLID.info, EBMLID.tracks] + (readCues ? [EBMLID.cues] : []) where !parsed.contains(id) {
            guard let relative = positions[id] else { continue }
            let at = segmentStart + relative
            let probe = try source.read(at: at, count: 16)
            guard let element = Self.element(in: probe, at: 0), element.id == id, let size = element.bodySize else { continue }
            let body = try Self.read(source, at: at + Int64(element.headerLength), count: size, limit: 16 << 20)
            parse(id, body, positions: &positions, video: &video, readCues: readCues)
        }
        self.video = video
        cues.sort { $0.time < $1.time }
    }

    private mutating func parse(_ id: UInt32, _ body: Data, positions: inout [UInt32: Int64],
                                video: inout MatroskaVideoTrack?, readCues: Bool) {
        switch id {
        case EBMLID.seekHead:
            for seek in EBMLChildren(body) where seek.id == EBMLID.seek {
                var target: UInt32?, at: Int64?
                for field in EBMLChildren(seek.body) {
                    if field.id == EBMLID.seekID { target = UInt32(truncatingIfNeeded: EBML.unsigned(field.body)) }
                    if field.id == EBMLID.seekPosition { at = Int64(clamping: EBML.unsigned(field.body)) }
                }
                if let target, let at, positions[target] == nil { positions[target] = at }
            }
        case EBMLID.info:
            for field in EBMLChildren(body) {
                if field.id == EBMLID.timestampScale, case let scale = Int64(clamping: EBML.unsigned(field.body)), scale > 0 {
                    timestampScale = scale
                }
                if field.id == EBMLID.duration, let ticks = EBML.float(field.body) { durationSeconds = ticks }
            }
            // Duration is in ticks; the scale may come after it.
            if let ticks = durationSeconds { durationSeconds = ticks * Double(timestampScale) / 1e9 }
        case EBMLID.tracks:
            for entry in EBMLChildren(body) where entry.id == EBMLID.trackEntry {
                if video == nil, let track = Self.videoTrack(entry.body) { video = track }
            }
        case EBMLID.cues:
            guard readCues else { return }
            let track = video?.number
            for point in EBMLChildren(body) where point.id == EBMLID.cuePoint {
                var time: Int64?
                var cluster: Int64?
                for field in EBMLChildren(point.body) {
                    if field.id == EBMLID.cueTime { time = Int64(clamping: EBML.unsigned(field.body)) }
                    if field.id == EBMLID.cueTrackPositions {
                        var cueTrack: UInt64?, at: Int64?
                        for p in EBMLChildren(field.body) {
                            if p.id == EBMLID.cueTrack { cueTrack = EBML.unsigned(p.body) }
                            if p.id == EBMLID.cueClusterPosition { at = Int64(clamping: EBML.unsigned(p.body)) }
                        }
                        // Cues read before Tracks (rare) can't be filtered by track.
                        if let at, track == nil || cueTrack == track { cluster = at }
                    }
                }
                if let time, let cluster { cues.append(MatroskaCue(time: time, clusterOffset: segmentStart + cluster)) }
            }
        default:
            break
        }
    }

    static func videoTrack(_ entry: Data) -> MatroskaVideoTrack? {
        var number: UInt64?, type: UInt64?, codec = "", codecPrivate = Data()
        var track = MatroskaVideoTrack(number: 0, codecID: "")
        var mvcExtra: Data?
        for field in EBMLChildren(entry) {
            switch field.id {
            case EBMLID.trackNumber: number = EBML.unsigned(field.body)
            case EBMLID.trackType: type = EBML.unsigned(field.body)
            case EBMLID.codecID: codec = String(decoding: field.body, as: UTF8.self).trimmingCharacters(in: .controlCharacters)
            case EBMLID.codecPrivate: codecPrivate = field.body
            case EBMLID.defaultDuration: track.defaultDuration = EBML.unsigned(field.body)
            case EBMLID.contentEncodings: track.hasContentEncoding = true
            case EBMLID.video:
                for v in EBMLChildren(field.body) {
                    if v.id == EBMLID.pixelWidth { track.width = Int(clamping: EBML.unsigned(v.body)) }
                    if v.id == EBMLID.pixelHeight { track.height = Int(clamping: EBML.unsigned(v.body)) }
                    if v.id == EBMLID.stereoMode { track.stereoMode = Int(clamping: EBML.unsigned(v.body)) }
                }
            case EBMLID.blockAdditionMapping:
                var kind: UInt64?, extra: Data?
                for m in EBMLChildren(field.body) {
                    if m.id == EBMLID.blockAddIDType { kind = EBML.unsigned(m.body) }
                    if m.id == EBMLID.blockAddIDExtraData { extra = m.body }
                }
                if kind == 0x6D76_6343, let extra { mvcExtra = extra } // "mvcC"
            default:
                break
            }
        }
        guard let number, type == 1 else { return nil }
        track.number = number
        track.codecID = codec
        if codec == "V_MPEG4/ISO/AVC", let avc = try? AVCConfiguration(record: codecPrivate) {
            track.avc = avc
            let tail = codecPrivate.dropFirst(avc.recordLength)
            track.mvc = (mvcExtra.flatMap { try? AVCConfiguration(record: $0) }) ?? Self.legacyMVC(Data(tail))
        }
        return track
    }

    /// The `mvcC` record after the `avcC` one in a CodecPrivate: as a whole
    /// box (size, "mvcC", record; what MakeMKV writes), or the four-character
    /// type straight before the record (the Matroska codec spec's wording).
    static func legacyMVC(_ tail: Data) -> AVCConfiguration? {
        let fourCC = Data("mvcC".utf8)
        let record: Data
        if tail.count > 8, tail.dropFirst(4).prefix(4) == fourCC {
            // The box's size field isn't trusted: MakeMKV's says 288 for a
            // 292-byte box. The record says where it ends.
            record = tail.dropFirst(8)
        } else if tail.count > 4, tail.prefix(4) == fourCC {
            record = tail.dropFirst(4)
        } else {
            return nil
        }
        return try? AVCConfiguration(record: Data(record))
    }

    static func element(in data: Data, at offset: Int) -> EBMLElement? {
        data.withUnsafeBytes { raw in
            guard let (id, idLength) = EBML.id(raw, at: offset),
                  let (size, sizeLength) = EBML.size(raw, at: offset + idLength) else { return nil }
            return EBMLElement(id: id, bodyStart: offset + idLength + sizeLength,
                               bodySize: size.map { Int(clamping: $0) }, headerLength: idLength + sizeLength)
        }
    }

    static func read(_ source: ByteSource, at offset: Int64, count: Int, limit: Int) throws -> Data {
        guard count <= limit else { throw FormatError("element of \(count) bytes is too large") }
        var data = Data()
        while data.count < count {
            let chunk = try source.read(at: offset + Int64(data.count), count: count - data.count)
            if chunk.isEmpty { throw FormatError("file ends inside an element") }
            data.append(chunk)
        }
        return data
    }
}
