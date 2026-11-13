/*
 Vergence - stereo access units from a Matroska MVC track

 Reads the video track's Blocks in file order from a Cluster on and turns each
 into a `StereoAccessUnit`: the Block's length-prefixed NAL units rewritten as
 Annex B (both views are already in it, base first; see `Matroska.swift`),
 with the Block's timestamp as the PTS in 90 kHz. The first unit after
 opening or seeking is preceded by the parameter sets of both views from the
 track's configuration, which edge264 needs before the first slice and a
 Block may only carry in band.

 Seeking goes to the Cue at or before the target: a Cluster that opens with a
 random access picture. Pictures between it and the target are decoded and
 dropped by the caller (`StereoDecoder`), as on a disc.

 Matroska timestamps are presentation times, so `dts` is left nil.
 */

import Foundation

public final class MatroskaStereoReader: StereoAccessUnitReader {
    public let file: MatroskaFile
    public let track: MatroskaVideoTrack
    private let source: ByteSource
    private let lengthSize: Int
    private let parameterSets: Data
    private var cursor: ElementCursor
    private var clusterTime: Int64 = 0
    /// Where the current Cluster ends; nil for an unknown size.
    private var clusterEnd: Int64?
    private var inCluster = false
    private var needsParameterSets = true

    /// Blocks skipped because they were laced (video never is in practice).
    public private(set) var skippedBlockCount = 0
    /// Places where an element couldn't be parsed and the rest of a Cluster
    /// was skipped.
    public private(set) var resyncCount = 0
    /// Blocks without a dependent view (2D stretches, or a damaged unit).
    public private(set) var unpairedBaseCount = 0

    /// - Parameter file: the header, if already read (`MatroskaFile(source:)`).
    public init(source: ByteSource, file: MatroskaFile? = nil) throws {
        let file = try file ?? MatroskaFile(source: source)
        guard let track = file.video, let avc = track.avc else { throw FormatError("no H.264 video track") }
        guard !track.hasContentEncoding else { throw FormatError("the video track's frames are compressed or encrypted") }
        self.source = source
        self.file = file
        self.track = track
        lengthSize = avc.lengthSize
        var sets = avc.annexB
        if let mvc = track.mvc {
            // SPS before PPS across both records: a PPS names its SPS.
            let all = AVCConfiguration(merging: avc, mvc)
            sets = all.annexB
        }
        parameterSets = sets
        cursor = ElementCursor(source: source, start: file.firstClusterOffset ?? file.segmentStart, end: file.segmentEnd)
    }

    /// Seconds of the stream, from its header.
    public var durationSeconds: Double? { file.durationSeconds }

    public func next() throws -> StereoAccessUnit? {
        while true {
            guard let element = try nextElement() else { return nil }
            switch element.id {
            case EBMLID.timestamp:
                clusterTime = Int64(clamping: EBML.unsigned(try cursor.body(element)))
            case EBMLID.simpleBlock:
                if let unit = try block(try cursor.body(element), simple: true) { return unit }
            case EBMLID.blockGroup:
                let group = try cursor.body(element)
                if let block = EBMLChildren(group).first(where: { $0.id == EBMLID.block }),
                   let unit = try self.block(block.body, simple: false) { return unit }
            default:
                try cursor.skip(element)
            }
        }
    }

    /// The next element inside a Cluster, entering Clusters as they come.
    private func nextElement() throws -> ElementCursor.Element? {
        while true {
            if inCluster, let end = clusterEnd, cursor.position >= end { inCluster = false }
            guard cursor.position < file.segmentEnd else { return nil }
            guard let element = try cursor.peek() else {
                // Damaged bytes: give up on this Cluster and look for the next.
                guard try resync() else { return nil }
                continue
            }
            if element.id == EBMLID.cluster {
                cursor.enter(element)
                inCluster = true
                clusterEnd = element.bodySize.map { cursor.position + $0 }
                clusterTime = 0
                continue
            }
            if !inCluster || EBMLID.topLevel.contains(element.id) {
                // Between Clusters (Cues, Tags...), or an unknown-size Cluster ended.
                inCluster = false
                guard element.bodySize != nil else { return nil }
                try cursor.skip(element)
                continue
            }
            guard element.bodySize != nil else { guard try resync() else { return nil }; continue }
            return element
        }
    }

    /// Finds the next Cluster ID after damaged bytes, within 32 MB.
    private func resync() throws -> Bool {
        resyncCount += 1
        inCluster = false
        let limit = min(file.segmentEnd, cursor.position + (32 << 20))
        var at = cursor.position + 1
        while at < limit {
            let chunk = try source.read(at: at, count: Int(min(1 << 20, limit - at)))
            if chunk.isEmpty { return false }
            if let found = chunk.withUnsafeBytes({ raw -> Int? in
                guard raw.count >= 4 else { return nil }
                for i in 0 ... raw.count - 4 where raw[i] == 0x1F && raw[i + 1] == 0x43 && raw[i + 2] == 0xB6 && raw[i + 3] == 0x75 {
                    return i
                }
                return nil
            }) {
                cursor.move(to: at + Int64(found))
                return true
            }
            at += Int64(max(1, chunk.count - 3))
        }
        return false
    }

    private func block(_ body: Data, simple: Bool) throws -> StereoAccessUnit? {
        let header: (track: UInt64, length: Int)? = body.withUnsafeBytes { raw in
            guard let (number, length) = EBML.size(raw, at: 0), let number else { return nil }
            return (number, length)
        }
        guard let header, header.track == track.number, body.count >= header.length + 3 else { return nil }
        let base = body.startIndex + header.length
        let relative = Int16(bitPattern: UInt16(body[base]) << 8 | UInt16(body[base + 1]))
        let flags = body[base + 2]
        guard flags & 0x06 == 0 else {
            skippedBlockCount += 1
            return nil
        }
        let payload = body[(base + 3)...]
        var data = Data(capacity: payload.count + 16 + (needsParameterSets ? parameterSets.count : 0))
        if needsParameterSets {
            data.append(parameterSets)
            needsParameterSets = false
        }
        var dependent = false
        var p = payload.startIndex
        while p + lengthSize <= payload.endIndex {
            var length = 0
            for i in 0 ..< lengthSize { length = length << 8 | Int(payload[p + i]) }
            p += lengthSize
            guard length > 0, p + length <= payload.endIndex else { break }
            let type = payload[p] & 0x1F
            // NAL 24 is Blu-ray's dependent-view delimiter, outside H.264.
            if type != 24 {
                if type == 20 { dependent = true }
                data.append(contentsOf: [0, 0, 0, 1])
                data.append(payload[p ..< p + length])
            }
            p += length
        }
        if !dependent { unpairedBaseCount += 1 }
        return StereoAccessUnit(pts: presentationTime(ticks: clusterTime + Int64(relative)), dts: nil, data: data,
                                hasDependentView: dependent)
    }

    /// A Block's time in 90 kHz. Matroska rounds it to a tick (usually a
    /// millisecond), so 23.976 fps frames come 41 or 42 ms apart: far under a
    /// display refresh, so it is passed on as is. (Snapping to the track's
    /// DefaultDuration doesn't work: a remux's frame grid is anchored on the
    /// disc's first PTS, not on zero.)
    func presentationTime(ticks: Int64) -> Int64 {
        file.ninetyKHz(ticks: ticks)
    }

    /// Moves to the Cue at or before `seconds` (from the file's time zero).
    /// Returns that Cue's time, 90 kHz.
    @discardableResult
    public func seek(to seconds: Double) throws -> Int64 {
        let ticks = Int64((max(seconds, 0) * 1e9 / Double(file.timestampScale)).rounded(.down))
        let cue = file.cues.last { $0.time <= ticks }
        cursor.move(to: cue?.clusterOffset ?? file.firstClusterOffset ?? file.segmentStart)
        inCluster = false
        clusterEnd = nil
        needsParameterSets = true
        return cue.map { file.ninetyKHz(ticks: $0.time) } ?? 0
    }
}

extension AVCConfiguration {
    /// One record's parameter sets followed by another's, the base view's
    /// SPS first and duplicates dropped.
    init(merging base: AVCConfiguration, _ other: AVCConfiguration) {
        self = base
        for sps in other.sequenceParameterSets where !sequenceParameterSets.contains(sps) { sequenceParameterSets.append(sps) }
        for pps in other.pictureParameterSets where !pictureParameterSets.contains(pps) { pictureParameterSets.append(pps) }
    }
}

/// Sequential element reading from a byte source, buffered in large chunks.
struct ElementCursor {
    struct Element {
        var id: UInt32
        var headerLength: Int
        /// Nil for an unknown size.
        var bodySize: Int64?
    }

    let source: ByteSource
    let end: Int64
    private(set) var position: Int64
    private var buffer = Data()
    private var bufferStart: Int64 = 0
    private let chunkSize = 1 << 20

    init(source: ByteSource, start: Int64, end: Int64) {
        self.source = source
        self.end = end
        position = start
        bufferStart = start
    }

    mutating func move(to offset: Int64) {
        position = offset
        if offset < bufferStart || offset > bufferStart + Int64(buffer.count) {
            buffer.removeAll(keepingCapacity: true)
            bufferStart = offset
        }
    }

    /// Makes `count` bytes from `position` available, or as many as remain.
    private mutating func fill(_ count: Int) throws -> Int {
        let have = Int(bufferStart + Int64(buffer.count) - position)
        if have >= count { return count }
        // Drop what is behind the position before growing.
        if position > bufferStart {
            buffer.removeSubrange(buffer.startIndex ..< buffer.startIndex + Int(position - bufferStart))
            bufferStart = position
        }
        while Int64(buffer.count) < Int64(count), bufferStart + Int64(buffer.count) < end {
            let from = bufferStart + Int64(buffer.count)
            let want = Int(min(Int64(max(chunkSize, count - buffer.count)), end - from))
            let chunk = try source.read(at: from, count: want)
            if chunk.isEmpty { break }
            buffer.append(chunk)
        }
        return min(count, buffer.count)
    }

    /// The element header at the position, without consuming it.
    mutating func peek() throws -> Element? {
        let available = try fill(12)
        guard available > 0 else { return nil }
        let start = Int(position - bufferStart)
        return buffer.withUnsafeBytes { raw -> Element? in
            let view = UnsafeRawBufferPointer(rebasing: raw[start ..< start + available])
            guard let (id, idLength) = EBML.id(view, at: 0), let (size, sizeLength) = EBML.size(view, at: idLength) else {
                return nil
            }
            return Element(id: id, headerLength: idLength + sizeLength, bodySize: size.map { Int64(clamping: $0) })
        }
    }

    /// Steps into an element's body (for a master element).
    mutating func enter(_ element: Element) {
        position += Int64(element.headerLength)
    }

    mutating func skip(_ element: Element) throws {
        guard let size = element.bodySize else { throw FormatError("can't skip an element of unknown size") }
        move(to: position + Int64(element.headerLength) + size)
    }

    /// Reads an element's body and moves past it.
    mutating func body(_ element: Element) throws -> Data {
        guard let size = element.bodySize, size <= 256 << 20 else { throw FormatError("element too large") }
        position += Int64(element.headerLength)
        let count = Int(size)
        let available = try fill(count)
        guard available == count else { throw FormatError("file ends inside an element") }
        let start = buffer.startIndex + Int(position - bufferStart)
        let data = buffer.subdata(in: start ..< start + count)
        position += size
        return data
    }
}
