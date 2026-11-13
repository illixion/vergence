/*
 Vergence - a lone transport stream holding both views

 A 3D Blu-ray's feature also turns up as a single transport stream file
 with both views in it: the disc's SSIF (its two clips interleaved in
 extents, the way a player reads them) copied out as an .m2ts, a tsMuxeR
 M2TS made from the demuxed views (packets interleaved finely), or a 3D
 camcorder's AVCHD clip. One file carries the base view (stream type 0x1B;
 PID 0x1011 on a disc), the dependent view (stream type 0x20, MVC), the
 audio and the PGS subtitles, with no playlist or clip information beside
 it.

 `TransportStreamFile` makes it a disc of one clip. The PMT names the
 streams; the base view's first and last timestamps make a one-item
 playlist (`playlist`); entry points are found by searching the file
 instead of read from an EP map. The playlist readers (`PlaylistDecoder`,
 `PlaylistStereoReader`, `PlaylistAudioReader`) then work unchanged, and
 the two views are read by two readers of the same bytes, so neither
 interleave needs buffering: the readers stay within an extent of each
 other, in the same few cached blocks of a remote file.

 A search on the base view: an interpolation search on its decode times
 gets to within a few megabytes (about a second of a Blu-ray) of the
 target, reading a little of the file at each guess; a scan forward from
 there finds the last random access picture at or before the target: an
 IDR, or an I picture, which a Blu-ray's open GOPs start with. The
 dependent view's entry is its access unit with the same PTS, close to the
 base view's in a finely interleaved file and an extent away in an SSIF.

 Times are the stream's own clock, as a disc's clips' are: the playlist's
 IN time is the base view's first PTS.
 */

import Foundation

public final class TransportStreamFile: ClipSource, @unchecked Sendable {
    /// The one clip's name in `playlist`.
    public static let clipName = "00000"

    /// An elementary stream the PMT lists.
    public struct Stream: Sendable, Equatable {
        public var pid: UInt16
        public var streamType: UInt8
        /// ISO 639-2, when the PMT carries a language descriptor (a disc's
        /// clips don't: their languages are in the playlist).
        public var language: String?
        /// The Blu-ray stream_coding_type it stands for (`BlurayPlaylist.Stream`):
        /// a disc's own, or one mapped from a plain stream's AC-3/E-AC-3/DTS
        /// signalling. Nil for anything else.
        public var codingType: UInt8?
    }

    public let source: ByteSource
    /// 192 (BDAV, `.m2ts`) or 188 (plain `.ts`).
    public let packetSize: Int
    public let streams: [Stream]
    public let basePID: UInt16
    /// The MVC dependent view, nil for a 2D stream.
    public let dependentPID: UInt16?
    /// The base view's first and last PTS (the stream's clock, 90 kHz).
    public let firstPTS: Int64
    public let lastPTS: Int64
    /// One picture, 90 kHz, from the first pictures' timestamps.
    public let frameTicks: Int64
    /// One PlayItem spanning the file, clip `clipName` for both views.
    public let playlist: BlurayPlaylist
    public var isStereo: Bool { dependentPID != nil }

    private let lock = NSLock()
    /// The last base-view entry found: the dependent view's is near it.
    private var lastBaseEntry: (pts: Int64, offset: Int64)?

    /// Read for the PMT and the first timestamps; an SSIF that opens with
    /// a long dependent-view extent needs the larger one.
    private static let headBytes = [4 << 20, 48 << 20]

    /// Whether `head` (the file's first kilobyte or so) is a transport
    /// stream: sync bytes at packet intervals.
    public static func looksLikeTransportStream(_ head: Data) -> Bool {
        packetSize(head, packets: min(5, head.count / 192)) != nil
    }

    /// The streams of the first program, from the PAT and PMT in `head` (a
    /// few hundred kilobytes from the start of the file are plenty), or nil
    /// when it holds no PMT. An SSIF carries both clips' PMTs, the dependent
    /// one (the MVC stream alone) first: every PMT seen is merged.
    public static func programStreams(in head: Data) -> (packetSize: Int, streams: [Stream])? {
        guard let size = packetSize(head, packets: min(5, head.count / 192)),
              let streams = PSI.programStreams(head, packetSize: size) else { return nil }
        return (size, streams)
    }

    /// Reads the PMT and the base view's first timestamps from the start of
    /// the file, and its last from the end.
    public init(source: ByteSource) throws {
        self.source = source
        var opened: (head: Data, size: Int, streams: [Stream], base: Stream, starts: [PESStart])?
        for count in Self.headBytes where opened == nil {
            let head = try source.read(at: 0, count: Int(min(source.size, Int64(count))))
            guard let (size, streams) = Self.programStreams(in: head) else { continue }
            guard let base = streams.first(where: { $0.streamType == 0x1B }) else { continue }
            let starts = Self.starts(in: head, at: 0, pid: base.pid, packetSize: size)
            if !starts.isEmpty { opened = (head, size, streams, base, starts) }
            if head.count < count { break }
        }
        guard let (_, size, streams, base, first) = opened else {
            throw FormatError("no H.264 video in a transport stream program")
        }
        packetSize = size
        self.streams = streams
        basePID = base.pid
        dependentPID = streams.first { $0.streamType == 0x20 }?.pid

        guard let earliest = first.prefix(32).compactMap(\.pts).min(by: { PlaylistStereoReader.offset($0, from: $1) < 0 }) else {
            throw FormatError("no timestamps on the video")
        }
        firstPTS = earliest
        let times = first.prefix(64).compactMap(\.pts).map { PlaylistStereoReader.offset($0, from: earliest) }.sorted()
        frameTicks = zip(times, times.dropFirst()).map { $1 - $0 }.filter { $0 >= 900 }.min() ?? 3754

        // The last PTS, from the end: an SSIF can end with a long extent of
        // the other view, so look further back when the tail has none.
        var last: Int64?
        for tail in [4 << 20, 32 << 20] where last == nil {
            let start = max(0, source.size - Int64(tail)) / Int64(size) * Int64(size)
            let bytes = try source.read(at: start, count: Int(source.size - start))
            last = Self.starts(in: bytes, at: start, pid: base.pid, packetSize: size).compactMap(\.pts)
                .max { PlaylistStereoReader.offset($0, from: earliest) < PlaylistStereoReader.offset($1, from: earliest) }
            if start == 0 { break }
        }
        lastPTS = last ?? earliest

        let mask: Int64 = (1 << 33) - 1
        let item = BlurayPlaylist.PlayItem(
            clip: Self.clipName,
            dependentClip: dependentPID == nil ? nil : Self.clipName,
            inTime: UInt32(truncatingIfNeeded: firstPTS / 2),
            outTime: UInt32(truncatingIfNeeded: ((lastPTS + frameTicks) & mask) / 2),
            videoPID: base.pid,
            dependentVideoPID: dependentPID,
            audioStreams: streams.compactMap { stream in
                stream.codingType.flatMap { (0x80 ... 0x86).contains($0) || $0 == 0xA1 || $0 == 0xA2 ? $0 : nil }
                    .map { BlurayPlaylist.Stream(pid: stream.pid, codingType: $0, language: stream.language) }
            },
            subtitleStreams: streams.compactMap { stream in
                stream.codingType.flatMap { $0 == 0x90 || $0 == 0x92 ? $0 : nil }
                    .map { BlurayPlaylist.Stream(pid: stream.pid, codingType: $0, language: stream.language) }
            })
        playlist = BlurayPlaylist(items: [item])
    }

    /// The same file on another reader of its bytes (a second HTTP reader,
    /// for audio read apart from the picture), without reading it again.
    public func reopened(on source: ByteSource) -> TransportStreamFile {
        TransportStreamFile(source: source, like: self)
    }

    private init(source: ByteSource, like other: TransportStreamFile) {
        self.source = source
        packetSize = other.packetSize
        streams = other.streams
        basePID = other.basePID
        dependentPID = other.dependentPID
        firstPTS = other.firstPTS
        lastPTS = other.lastPTS
        frameTicks = other.frameTicks
        playlist = other.playlist
    }

    /// Whether a TrueHD track carries the AC-3 core a Blu-ray's always
    /// does (PES that open with the AC-3 sync word, between the TrueHD
    /// ones), from its first PES in the file's first few megabytes. A
    /// remux made from Matroska has TrueHD alone. True when the track can't
    /// be found there, so it isn't written off unread.
    public func carriesAC3Frames(pid: UInt16) throws -> Bool {
        var seen = 0, ac3 = false
        try scan(pid: pid, from: 0, until: 8 << 20, classify: true) { start in
            seen += 1
            ac3 = start.opensWithAC3
            return ac3 || seen >= 16
        }
        return ac3 || seen == 0
    }

    // MARK: ClipSource

    public func clip(_ name: String) throws -> ByteSource { source }

    public func entryPoint(clip: String, pid: UInt16, atOrBefore pts: Int64) throws -> ClipInfo.EntryPoint? {
        if pid == basePID { return try baseEntry(atOrBefore: pts) }
        if pid == dependentPID { return try dependentEntry(at: pts) }
        return nil
    }

    public func entryPoint(clip: String, pid: UInt16, after pts: Int64) throws -> ClipInfo.EntryPoint? {
        guard pid == basePID, linear(pts) < linear(lastPTS) else { return nil }
        let from = try baseEntry(atOrBefore: pts).byteOffset
        var next: PESStart?
        try scan(pid: basePID, from: from, classify: true) { start in
            guard start.randomAccess, let time = start.pts.map(linear), time > linear(pts) else { return false }
            next = start
            return true
        }
        return next.map(entry)
    }

    // MARK: Search

    /// Time since the first picture, on the 33-bit clock.
    private func linear(_ pts: Int64) -> Int64 {
        PlaylistStereoReader.offset(pts, from: firstPTS)
    }

    private func entry(_ start: PESStart) -> ClipInfo.EntryPoint {
        ClipInfo.EntryPoint(pts: start.pts ?? firstPTS, spn: UInt32(start.offset / Int64(packetSize)), packetSize: packetSize)
    }

    /// The last random access picture of the base view presented at or
    /// before `pts`; the first picture of the file when none is.
    private func baseEntry(atOrBefore pts: Int64) throws -> ClipInfo.EntryPoint {
        let target = linear(pts)
        // A Blu-ray's GOPs are at most a second; look further back if this
        // stream's are longer.
        var back: Int64 = 108_000
        while true {
            let from = target - back <= 0 ? 0 : try offset(pid: basePID, decodingAtOrBefore: target - back)
            var best: PESStart?
            try scan(pid: basePID, from: from, classify: true) { [self] start in
                guard let order = (start.dts ?? start.pts).map(linear) else { return false }
                if order > target { return true }
                if start.randomAccess, let shown = start.pts.map(linear), shown <= target { best = start }
                return false
            }
            if let best {
                lock.withLock { lastBaseEntry = (best.pts ?? firstPTS, best.offset) }
                return entry(best)
            }
            if from == 0 { return ClipInfo.EntryPoint(pts: firstPTS, spn: 0, packetSize: packetSize) }
            back *= 4
        }
    }

    /// The dependent view's access unit presented at `pts` (a base view
    /// entry point's), or the last one decoded before it when no unit has
    /// that time. Never nil for a stereo file: the dependent view's reader
    /// can't be left at the start of the file.
    private func dependentEntry(at pts: Int64) throws -> ClipInfo.EntryPoint? {
        guard let pid = dependentPID else { return nil }
        let target = linear(pts)
        var match: PESStart?, before: PESStart?, first: PESStart?
        let look: (PESStart) -> Bool = { [self] start in
            if first == nil { first = start }
            if start.pts == pts { match = start; return true }
            guard let order = (start.dts ?? start.pts).map(linear) else { return false }
            if order > target { return true }
            before = start
            return false
        }
        // Next to the base view's entry, in most files.
        if let near = lock.withLock({ lastBaseEntry }), near.pts == pts {
            let from = max(0, near.offset - (2 << 20)) / Int64(packetSize) * Int64(packetSize)
            try scan(pid: pid, from: from, until: near.offset + (2 << 20), classify: false, look)
            if let match { return entry(match) }
        }
        match = nil; before = nil; first = nil
        let from = target <= 45_000 ? 0 : try offset(pid: pid, decodingAtOrBefore: target - 45_000)
        try scan(pid: pid, from: from, classify: false, look)
        return (match ?? before ?? first).map(entry) ?? ClipInfo.EntryPoint(pts: pts, spn: 0, packetSize: packetSize)
    }

    /// A byte offset where `pid`'s decode time is at or a little before
    /// `target` (time since the first picture): the start of one of its PES
    /// decoded at most half a second before `target`, or within 256 kB of
    /// the PES that passes it.
    private func offset(pid: UInt16, decodingAtOrBefore target: Int64) throws -> Int64 {
        let step = Int64(packetSize)
        var low: (offset: Int64, time: Int64) = (0, 0)
        var high: (offset: Int64, time: Int64) = (source.size, linear(lastPTS) + frameTicks)
        var bisect = false
        for _ in 0 ..< 64 {
            let span = high.offset - low.offset
            guard span > 256 << 10, target - low.time > 45_000 else { break }
            var guess: Int64
            if bisect {
                guess = low.offset + span / 2
            } else {
                let fraction = Double(target - low.time) / Double(max(high.time - low.time, 1))
                guess = low.offset + Int64(Double(span) * min(max(fraction, 0), 1))
                guess = min(max(guess, low.offset + span / 16), high.offset - span / 16)
            }
            guess = guess / step * step
            let found = try firstStart(pid: pid, at: guess, before: high.offset)
            if let found, let time = (found.dts ?? found.pts).map(linear), time <= target {
                low = (found.offset, time)
            } else {
                high = (guess, found.flatMap { ($0.dts ?? $0.pts).map(linear) } ?? high.time)
            }
            // Interpolation that doesn't halve the bracket gives way to a bisection.
            bisect = high.offset - low.offset > span / 2
        }
        return low.offset
    }

    /// The first PES start on `pid` at or after `offset`, looking no
    /// further than `limit` or 16 MB.
    private func firstStart(pid: UInt16, at offset: Int64, before limit: Int64) throws -> PESStart? {
        var found: PESStart?
        try scan(pid: pid, from: offset, until: min(limit, offset + (16 << 20)), classify: false) { start in
            found = start
            return true
        }
        return found
    }

    // MARK: Scanning

    /// A PES start on one PID, as a scan of the file finds it.
    struct PESStart {
        /// Its first packet's byte offset.
        var offset: Int64
        var pts: Int64?
        var dts: Int64?
        /// An IDR or I picture (only worked out when a scan classifies).
        var randomAccess = false
        /// The payload opens with the AC-3 sync word (when classified).
        var opensWithAC3 = false
    }

    /// Calls `found` with each PES start on `pid` from byte `from` (a packet
    /// boundary) until it returns true or `until`. Classifying reads the
    /// start of each payload, so a start is reported once a few packets
    /// more of its PID have gone by.
    private func scan(pid: UInt16, from: Int64, until: Int64? = nil, classify: Bool,
                      _ found: (PESStart) -> Bool) throws {
        let end = min(until ?? source.size, source.size)
        // Small reads first (a probe usually finds its PES in the first
        // few packets), larger ones for a long scan.
        var chunk = Int64(packetSize) * 256
        var offset = from
        var pending: (start: PESStart, payload: Data)?
        func finish(_ open: (start: PESStart, payload: Data)) -> Bool {
            var start = open.start
            start.randomAccess = Self.isRandomAccess(open.payload)
            start.opensWithAC3 = open.payload.starts(with: [0x0B, 0x77])
            return found(start)
        }
        while offset < end {
            let data = try source.read(at: offset, count: Int(min(chunk, end - offset)))
            if data.isEmpty { break }
            var stopped = false
            data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                let bytes = raw.bindMemory(to: UInt8.self)
                let lead = packetSize - 188
                var at = 0
                while at + packetSize <= bytes.count {
                    defer { at += packetSize }
                    let p = UnsafeBufferPointer(rebasing: bytes[(at + lead) ..< (at + packetSize)])
                    guard p[0] == 0x47, UInt16(p[1] & 0x1F) << 8 | UInt16(p[2]) == pid else { continue }
                    let adaptation = (p[3] >> 4) & 0x3
                    var payload = 4
                    if adaptation & 0x2 != 0 { payload += 1 + Int(p[4]) }
                    guard adaptation & 0x1 != 0, payload < 188 else { continue }
                    let body = UnsafeBufferPointer(rebasing: p[payload...])
                    if p[1] & 0x40 != 0 {
                        if let open = pending, finish(open) { stopped = true; return }
                        pending = nil
                        guard let (times, headerLength) = TransportStreamDemuxer.parsePESHeader(body) else { continue }
                        let start = PESStart(offset: offset + Int64(at), pts: times.pts, dts: times.dts)
                        if classify {
                            pending = (start, Data(UnsafeBufferPointer(rebasing: body[headerLength...])))
                        } else if found(start) {
                            stopped = true
                            return
                        }
                    } else if var open = pending {
                        open.payload.append(contentsOf: body)
                        if open.payload.count >= Self.classifyBytes {
                            pending = nil
                            if finish(open) { stopped = true; return }
                        } else {
                            pending = open
                        }
                    }
                }
            }
            if stopped { return }
            offset += Int64(data.count)
            chunk = min(chunk * 2, Int64(packetSize) * 8192)
        }
        if let open = pending { _ = finish(open) }
    }

    /// Enough of an access unit to reach its first slice header past the
    /// delimiter, parameter sets and SEI a Blu-ray puts first.
    private static let classifyBytes = 4096

    /// Whether an access unit's first bytes start a random access picture:
    /// an IDR slice, or an I (or SI) slice; parameter sets with no slice in
    /// sight count too.
    static func isRandomAccess(_ payload: Data) -> Bool {
        var parameterSets = false
        for nal in AnnexB.nalUnits(in: payload) {
            switch nal.type {
            case 5: return true
            case 1:
                var bits = ExpGolomb(payload[(nal.range.lowerBound + 1) ..< nal.range.upperBound])
                _ = bits.unsigned() // first_mb_in_slice
                return [2, 4, 7, 9].contains(bits.unsigned()) // slice_type I, SI
            case 7: parameterSets = true
            default: continue
            }
        }
        return parameterSets
    }

    /// The PES starts on `pid` among the whole packets of `bytes`, which sit
    /// at byte `offset` of the file.
    private static func starts(in bytes: Data, at offset: Int64, pid: UInt16, packetSize: Int) -> [PESStart] {
        var starts: [PESStart] = []
        bytes.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            let b = raw.bindMemory(to: UInt8.self)
            let lead = packetSize - 188
            var at = 0
            while at + packetSize <= b.count {
                defer { at += packetSize }
                let p = UnsafeBufferPointer(rebasing: b[(at + lead) ..< (at + packetSize)])
                guard p[0] == 0x47, p[1] & 0x40 != 0, UInt16(p[1] & 0x1F) << 8 | UInt16(p[2]) == pid else { continue }
                let adaptation = (p[3] >> 4) & 0x3
                var payload = 4
                if adaptation & 0x2 != 0 { payload += 1 + Int(p[4]) }
                guard adaptation & 0x1 != 0, payload < 188,
                      let (times, _) = TransportStreamDemuxer.parsePESHeader(UnsafeBufferPointer(rebasing: p[payload...])) else { continue }
                starts.append(PESStart(offset: offset + Int64(at), pts: times.pts, dts: times.dts))
            }
        }
        return starts
    }

    /// 192 or 188: whichever puts a sync byte at the start of `packets`
    /// packets in a row from the start of `head`.
    private static func packetSize(_ head: Data, packets: Int) -> Int? {
        guard packets >= 2 else { return nil }
        return head.withUnsafeBytes { raw in
            TransportStreamDemuxer.detectPacketSize(raw.bindMemory(to: UInt8.self), packets: packets)
        }
    }
}

/// Unsigned Exp-Golomb codes from the start of a NAL's payload (emulation
/// prevention bytes skipped).
private struct ExpGolomb {
    private let bytes: [UInt8]
    private var position = 0

    init(_ data: some Collection<UInt8>) {
        var out: [UInt8] = []
        var zeros = 0
        for byte in data.prefix(64) {
            if zeros >= 2, byte == 3 { zeros = 0; continue }
            out.append(byte)
            zeros = byte == 0 ? zeros + 1 : 0
        }
        bytes = out
    }

    private mutating func bit() -> Int {
        defer { position += 1 }
        let byte = position / 8
        return byte < bytes.count ? Int(bytes[byte] >> (7 - position % 8)) & 1 : 1
    }

    mutating func unsigned() -> Int {
        var zeros = 0
        while bit() == 0, zeros < 31 { zeros += 1 }
        var value = 0
        for _ in 0 ..< zeros { value = value << 1 | bit() }
        return (1 << zeros) - 1 + value
    }
}

/// The PAT and the first program's PMTs.
private enum PSI {
    static func programStreams(_ head: Data, packetSize: Int) -> [TransportStreamFile.Stream]? {
        var pmtPIDs: [UInt16] = []
        for pat in sections(in: head, pid: 0, packetSize: packetSize, tableID: 0x00) {
            // program_number 16, reserved 3, PID 13, after an 8-byte header; CRC last.
            var at = 8
            while at + 4 <= pat.count - 4 {
                let program = UInt16(pat[at]) << 8 | UInt16(pat[at + 1])
                let pid = UInt16(pat[at + 2] & 0x1F) << 8 | UInt16(pat[at + 3])
                if program != 0 {
                    if !pmtPIDs.contains(pid) { pmtPIDs.append(pid) }
                    break
                }
                at += 4
            }
        }
        var streams: [TransportStreamFile.Stream] = []
        var found = false
        for pid in pmtPIDs {
            for pmt in sections(in: head, pid: pid, packetSize: packetSize, tableID: 0x02) where pmt.count >= 12 {
                found = true
                for stream in elementaryStreams(pmt) where !streams.contains(where: { $0.pid == stream.pid }) {
                    streams.append(stream)
                }
            }
        }
        return found ? streams : nil
    }

    private static func elementaryStreams(_ pmt: [UInt8]) -> [TransportStreamFile.Stream] {
        var streams: [TransportStreamFile.Stream] = []
        var at = 12 + (Int(pmt[10] & 0x0F) << 8 | Int(pmt[11]))
        while at + 5 <= pmt.count - 4 {
            let type = pmt[at]
            let pid = UInt16(pmt[at + 1] & 0x1F) << 8 | UInt16(pmt[at + 2])
            let infoLength = Int(pmt[at + 3] & 0x0F) << 8 | Int(pmt[at + 4])
            let infoEnd = min(at + 5 + infoLength, pmt.count - 4)
            var language: String?
            var signalled: UInt8?
            var d = at + 5
            while d + 2 <= infoEnd {
                let tag = pmt[d], length = Int(pmt[d + 1])
                let body = pmt[min(d + 2, infoEnd) ..< min(d + 2 + length, infoEnd)]
                switch tag {
                case 0x0A where body.count >= 3:
                    language = String(bytes: body.prefix(3), encoding: .ascii).map { $0.lowercased() }
                case 0x05 where body.count >= 4:
                    switch String(bytes: body.prefix(4), encoding: .ascii) {
                    case "AC-3": signalled = 0x81
                    case "EAC3": signalled = 0x84
                    case "DTS1", "DTS2", "DTS3": signalled = 0x82
                    default: break
                    }
                case 0x6A: signalled = 0x81
                case 0x7A: signalled = 0x84
                case 0x7B: signalled = 0x82
                default: break
                }
                d += 2 + length
            }
            let coding: UInt8? = switch type {
            case 0x80 ... 0x86, 0xA1, 0xA2, 0x90, 0x92: type
            case 0x87: 0x84 // ATSC E-AC-3
            case 0x06: signalled
            default: nil
            }
            streams.append(.init(pid: pid, streamType: type, language: language, codingType: coding))
            at = infoEnd
        }
        return streams
    }

    /// The distinct complete sections of `tableID` on `pid`, each from its
    /// table_id to its CRC, in the order they first appear.
    private static func sections(in head: Data, pid: UInt16, packetSize: Int, tableID: UInt8) -> [[UInt8]] {
        var collected: [UInt8]?
        var sections: [[UInt8]] = []
        head.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            let b = raw.bindMemory(to: UInt8.self)
            let lead = packetSize - 188
            var at = 0
            while at + packetSize <= b.count {
                defer { at += packetSize }
                let p = UnsafeBufferPointer(rebasing: b[(at + lead) ..< (at + packetSize)])
                guard p[0] == 0x47, UInt16(p[1] & 0x1F) << 8 | UInt16(p[2]) == pid else { continue }
                let adaptation = (p[3] >> 4) & 0x3
                var payload = 4
                if adaptation & 0x2 != 0 { payload += 1 + Int(p[4]) }
                guard adaptation & 0x1 != 0, payload < 188 else { continue }
                if p[1] & 0x40 != 0 {
                    let pointer = Int(p[payload])
                    guard payload + 1 + pointer < 188, p[payload + 1 + pointer] == tableID else { collected = nil; continue }
                    collected = Array(p[(payload + 1 + pointer)...])
                } else if collected != nil {
                    collected! += p[payload...]
                }
                if let section = collected, section.count >= 3 {
                    let length = 3 + (Int(section[1] & 0x0F) << 8 | Int(section[2]))
                    if section.count >= length {
                        let complete = Array(section.prefix(length))
                        if !sections.contains(complete) { sections.append(complete) }
                        collected = nil
                    }
                }
            }
        }
        return sections
    }
}
