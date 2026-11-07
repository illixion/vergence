/*
 Vergence - analysis tracks: per-frame results computed somewhere else

 The headset can't afford the best models live (flow networks fit its budget
 only at a quarter of the eye's size, see Hypnos
 `docs/plans/blu-ray-3d-smooth-motion.md`). A Mac, servo or the media server
 can: it decodes the same video itself and sends back only what it computed,
 small fields keyed by media time, never pictures. The same records come from
 any producer: a live Mac over a paired channel, a file written ahead of time
 next to a pre-processed AI 3D cache, or a server.

 A track is a header then records, each length-prefixed, so it streams over a
 socket as well as it sits in a file, and a reader can index a file without
 decoding a single payload.

   track   = "VGAT" u8(version) u32(header JSON length) header-JSON  record*
   record  = u32(length of what follows)  u8(kind) u8(eye) u8(encoding) u8(channels)
             u16(width) u16(height)  i64(time value) i64(end value) i32(timescale)
             f32(scale) f32(offset)  payload

 All integers and floats little-endian. A sample's value is `raw * scale +
 offset`, where raw is a half float (`float16`, scale 1 offset 0), a signed
 16-bit integer or an unsigned byte. Samples are planar: channel by channel,
 rows top to bottom.

 The encoding byte's top bit (0x80) marks a deflated payload: each row's raw
 samples are stored as differences from the sample before them in the row
 (wrapping, in the raw integer type; the first against 0), then the whole
 payload is compressed with raw DEFLATE (RFC 1951, no zlib header). Flow
 fields are smooth, so this takes a 120×72 five-channel flow pair from 86 KB
 to about 20 KB at a half-pixel step. DEFLATE itself comes from the platform
 (`deflated(using:)` and `inflated(using:)` take it), so this file stays
 Foundation only.
 */

import Foundation

public enum AnalysisKind: UInt8, Sendable, Codable, CaseIterable {
    /// Optical flow between two real frames, both ways: channels F0→1 (x, y),
    /// then F1→0 (x, y), in pixels of the record's width and height. `time`
    /// is frame 0's, `endTime` frame 1's. An optional fifth channel is frame
    /// 0's blend weight at the midpoint (RIFE's mask), 0…1. A producer running
    /// a timestep network (RIFE) once at t = ½ stores F0→1 ≈ 2·F½→1 and
    /// F1→0 ≈ 2·F½→0, so every consumer does the same flow reversal. The
    /// mask channel says that is what happened: halving the two flows then
    /// gives the network's own F½→1 and F½→0, which a consumer should use as
    /// they are at t = ½ (they carry its occlusion handling; the reversal
    /// assumes straight-line motion).
    case flowPair = 1
    /// Disparity between the eyes of one real pair, laid out on the record's
    /// `eye`: channel 0 is the horizontal offset from that eye's pixel to the
    /// matching pixel in the other eye, in pixels of the record's width
    /// (channel 1, if present, the vertical one). `.left` is left→right,
    /// `.right` right→left (what Smooth Motion's shared field reads).
    case disparity = 2
    /// Relative inverse depth, near = high, 0…1.
    case depth = 3
    /// AI HDR's gain for the frame as a bilateral grid (`GainGrid` pins it):
    /// eye mono, float16, 8 channels (luma bins) of 16×9 cells, each the log2
    /// of the linear gain for pixels in that cell whose non-linear luma falls
    /// in that bin, sampled trilinearly at cell and bin centres.
    case gainMap = 4
}

public enum AnalysisEye: UInt8, Sendable, Codable {
    case mono = 0
    case left = 1
    case right = 2
}

public enum AnalysisEncoding: UInt8, Sendable, Codable {
    case float16 = 1
    case int16 = 2
    case uint8 = 3

    public var bytesPerSample: Int {
        switch self {
        case .float16, .int16: 2
        case .uint8: 1
        }
    }
}

/// Media time as a rational, like `CMTime` but portable.
public struct AnalysisTime: Hashable, Comparable, Sendable, Codable {
    public var value: Int64
    public var timescale: Int32

    public init(value: Int64, timescale: Int32) {
        self.value = value
        self.timescale = timescale
    }

    public init(seconds: Double, timescale: Int32 = 90_000) {
        self.init(value: Int64((seconds * Double(timescale)).rounded()), timescale: timescale)
    }

    public var seconds: Double { Double(value) / Double(timescale) }

    public static func < (a: AnalysisTime, b: AnalysisTime) -> Bool {
        // Cross-multiply in 128 bits' worth of care: timescales are ≤ 2^31.
        a.value.multipliedFullWidth(by: Int64(b.timescale)) < b.value.multipliedFullWidth(by: Int64(a.timescale))
    }
}

public struct AnalysisRecord: Sendable, Equatable {
    public var kind: AnalysisKind
    public var eye: AnalysisEye
    public var encoding: AnalysisEncoding
    public var channels: Int
    public var width: Int
    public var height: Int
    public var time: AnalysisTime
    /// The second real frame for pair kinds; equal to `time` otherwise.
    public var endTime: AnalysisTime
    public var scale: Float
    public var offset: Float
    public var payload: Data
    /// The payload is row-delta coded and DEFLATE-compressed (see the file
    /// header); `values()` needs `inflated(using:)` first.
    public var isDeflated: Bool

    public init(kind: AnalysisKind, eye: AnalysisEye = .mono, encoding: AnalysisEncoding, channels: Int,
                width: Int, height: Int, time: AnalysisTime, endTime: AnalysisTime? = nil,
                scale: Float = 1, offset: Float = 0, payload: Data, isDeflated: Bool = false) {
        self.kind = kind
        self.eye = eye
        self.encoding = encoding
        self.channels = channels
        self.width = width
        self.height = height
        self.time = time
        self.endTime = endTime ?? time
        self.scale = scale
        self.offset = offset
        self.payload = payload
        self.isDeflated = isDeflated
    }

    public var sampleCount: Int { channels * width * height }
    public var rawByteCount: Int { sampleCount * encoding.bytesPerSample }

    /// The record with its payload delta coded and compressed by `deflate`
    /// (raw DEFLATE). Already deflated: unchanged.
    public func deflated(using deflate: (Data) throws -> Data) rethrows -> AnalysisRecord {
        guard !isDeflated else { return self }
        var copy = self
        copy.payload = try deflate(rowDeltas(payload, encode: true))
        copy.isDeflated = true
        return copy
    }

    /// The record with a plain payload, `inflate` undoing raw DEFLATE given
    /// the size it must come to. Throws `payloadSize` if it doesn't.
    public func inflated(using inflate: (Data, Int) throws -> Data) throws -> AnalysisRecord {
        guard isDeflated else { return self }
        let plain = try inflate(payload, rawByteCount)
        guard plain.count == rawByteCount else { throw AnalysisTrackError.payloadSize }
        var copy = self
        copy.payload = rowDeltas(plain, encode: false)
        copy.isDeflated = false
        return copy
    }

    /// Row-wise differences (encode) or their running sums (decode), wrapping
    /// in the raw integer type. Half floats are differenced as bit patterns.
    private func rowDeltas(_ data: Data, encode: Bool) -> Data {
        var out = data
        let rows = channels * height
        out.withUnsafeMutableBytes { raw in
            if encoding.bytesPerSample == 2 {
                let samples = raw.bindMemory(to: UInt16.self)
                for row in 0 ..< rows {
                    var previous: UInt16 = 0
                    for x in 0 ..< width {
                        let i = row * width + x
                        let value = UInt16(littleEndian: samples[i])
                        if encode {
                            samples[i] = (value &- previous).littleEndian
                            previous = value
                        } else {
                            previous = value &+ previous
                            samples[i] = previous.littleEndian
                        }
                    }
                }
            } else {
                let samples = raw.bindMemory(to: UInt8.self)
                for row in 0 ..< rows {
                    var previous: UInt8 = 0
                    for x in 0 ..< width {
                        let i = row * width + x
                        let value = samples[i]
                        if encode {
                            samples[i] = value &- previous
                            previous = value
                        } else {
                            previous = value &+ previous
                            samples[i] = previous
                        }
                    }
                }
            }
        }
        return out
    }

    /// Encodes planar float samples. `int16` picks a scale from the largest
    /// magnitude (`maxMagnitude`, or the data's own) so the full range is used;
    /// `uint8` maps `range` onto 0…255.
    public static func encoding(_ values: [Float], kind: AnalysisKind, eye: AnalysisEye = .mono,
                                as encoding: AnalysisEncoding, channels: Int, width: Int, height: Int,
                                time: AnalysisTime, endTime: AnalysisTime? = nil,
                                maxMagnitude: Float? = nil, range: ClosedRange<Float> = 0 ... 1) -> AnalysisRecord {
        precondition(values.count == channels * width * height, "sample count")
        var payload = Data(capacity: values.count * encoding.bytesPerSample)
        var scale: Float = 1
        var offset: Float = 0
        switch encoding {
        case .float16:
            for value in values { payload.appendLittleEndian(HalfFloat.bits(from: value)) }
        case .int16:
            let limit = maxMagnitude ?? max(values.lazy.filter(\.isFinite).map(abs).max() ?? 0, 1e-6)
            scale = limit / Float(Int16.max)
            for value in values {
                // A model's NaN (an fp16 overflow) must not take the producer down: store 0.
                let raw = value.isNaN ? 0 : (value / scale).rounded().clamped(to: Float(-Int16.max) ... Float(Int16.max))
                payload.appendLittleEndian(UInt16(bitPattern: Int16(raw)))
            }
        case .uint8:
            offset = range.lowerBound
            scale = max(range.upperBound - range.lowerBound, 1e-12) / 255
            for value in values {
                payload.append(value.isNaN ? 0 : UInt8(((value - offset) / scale).rounded().clamped(to: 0 ... 255)))
            }
        }
        return AnalysisRecord(kind: kind, eye: eye, encoding: encoding, channels: channels, width: width,
                              height: height, time: time, endTime: endTime, scale: scale, offset: offset,
                              payload: payload)
    }

    /// The samples as floats, planar. A deflated record must be inflated first.
    public func values() -> [Float] {
        precondition(!isDeflated, "inflate the record first")
        var out = [Float]()
        out.reserveCapacity(sampleCount)
        payload.withUnsafeBytes { raw in
            switch encoding {
            case .float16:
                for i in 0 ..< sampleCount {
                    out.append(HalfFloat.value(from: raw.loadLittleEndian(UInt16.self, at: i * 2)) * scale + offset)
                }
            case .int16:
                for i in 0 ..< sampleCount {
                    out.append(Float(Int16(bitPattern: raw.loadLittleEndian(UInt16.self, at: i * 2))) * scale + offset)
                }
            case .uint8:
                for i in 0 ..< sampleCount { out.append(Float(raw[i]) * scale + offset) }
            }
        }
        return out
    }

    static let headerBytes = 4 + 4 + 8 + 8 + 4 + 4 + 4

    func framed() -> Data {
        var body = Data(capacity: Self.headerBytes + payload.count)
        body.append(kind.rawValue)
        body.append(eye.rawValue)
        body.append(encoding.rawValue | (isDeflated ? 0x80 : 0))
        body.append(UInt8(channels))
        body.appendLittleEndian(UInt16(width))
        body.appendLittleEndian(UInt16(height))
        body.appendLittleEndian(UInt64(bitPattern: time.value))
        body.appendLittleEndian(UInt64(bitPattern: endTime.value))
        body.appendLittleEndian(UInt32(bitPattern: time.timescale))
        body.appendLittleEndian(scale.bitPattern)
        body.appendLittleEndian(offset.bitPattern)
        body.append(payload)
        var frame = Data(capacity: 4 + body.count)
        frame.appendLittleEndian(UInt32(body.count))
        frame.append(body)
        return frame
    }

    /// Parses a record body (after its length prefix). For indexing, pass
    /// only the fixed header and the body's declared length: the payload is
    /// then left empty.
    static func parse(_ body: Data, declaredLength: Int? = nil) throws -> AnalysisRecord {
        guard body.count >= headerBytes else { throw AnalysisTrackError.truncated }
        let length = declaredLength ?? body.count
        return try body.withUnsafeBytes { raw in
            guard let kind = AnalysisKind(rawValue: raw[0]), let eye = AnalysisEye(rawValue: raw[1]),
                  let encoding = AnalysisEncoding(rawValue: raw[2] & 0x7F) else { throw AnalysisTrackError.unknownValue }
            let isDeflated = raw[2] & 0x80 != 0
            let channels = Int(raw[3])
            let width = Int(raw.loadLittleEndian(UInt16.self, at: 4))
            let height = Int(raw.loadLittleEndian(UInt16.self, at: 6))
            let value = Int64(bitPattern: raw.loadLittleEndian(UInt64.self, at: 8))
            let end = Int64(bitPattern: raw.loadLittleEndian(UInt64.self, at: 16))
            let timescale = Int32(bitPattern: raw.loadLittleEndian(UInt32.self, at: 24))
            let scale = Float(bitPattern: raw.loadLittleEndian(UInt32.self, at: 28))
            let offset = Float(bitPattern: raw.loadLittleEndian(UInt32.self, at: 32))
            guard timescale > 0 else { throw AnalysisTrackError.unknownValue }
            let expected = channels * width * height * encoding.bytesPerSample
            guard isDeflated ? length > headerBytes : length - headerBytes == expected else {
                throw AnalysisTrackError.payloadSize
            }
            let payload = declaredLength == nil ? Data(body[(body.startIndex + headerBytes)...]) : Data()
            return AnalysisRecord(kind: kind, eye: eye, encoding: encoding, channels: channels, width: width,
                                  height: height, time: AnalysisTime(value: value, timescale: timescale),
                                  endTime: AnalysisTime(value: end, timescale: timescale), scale: scale,
                                  offset: offset, payload: payload, isDeflated: isDeflated)
        }
    }
}

/// What a track is about, at its start.
public struct AnalysisTrackHeader: Codable, Sendable, Equatable {
    /// The producer's opaque id for the media (a hash, never a title or path).
    public var mediaID: String
    /// Who made it ("vergence 0.9 / M2 Max").
    public var producer: String
    /// Which model made each kind, with its working size ("RIFE 4.25 lite 480x288").
    public var models: [String: String]
    /// The source's picture size, so a consumer can scale flow and disparity
    /// from the record's size to its own.
    public var sourceWidth: Int
    public var sourceHeight: Int
    /// A worker's piece only: the video stream ended at this frame
    /// (exclusive), before the piece did (a container whose audio runs past
    /// its picture), so the coordinator can end the title there.
    public var videoEndFrame: Int64?

    public init(mediaID: String, producer: String, models: [String: String] = [:], sourceWidth: Int, sourceHeight: Int,
                videoEndFrame: Int64? = nil) {
        self.mediaID = mediaID
        self.producer = producer
        self.models = models
        self.sourceWidth = sourceWidth
        self.sourceHeight = sourceHeight
        self.videoEndFrame = videoEndFrame
    }
}

public enum AnalysisTrackError: Error, Equatable {
    case notATrack
    case unsupportedVersion(UInt8)
    case truncated
    case unknownValue
    case payloadSize
    case tooLarge
}

public enum AnalysisTrack {
    public static let magic = Data("VGAT".utf8)
    public static let version: UInt8 = 1
    /// No record legitimately comes near this (a 1920×1088 4-channel half
    /// float field is 16.7 MB); a length past it means a corrupt stream.
    public static let maxRecordBytes = 64 << 20

    /// The track's opening bytes.
    public static func header(_ header: AnalysisTrackHeader) throws -> Data {
        let json = try JSONEncoder().encode(header)
        var data = magic
        data.append(version)
        data.appendLittleEndian(UInt32(json.count))
        data.append(json)
        return data
    }

    public static func record(_ record: AnalysisRecord) -> Data { record.framed() }
}

/// Reads a track as it arrives: feed bytes, take the header once it is
/// whole, then records as each completes. Works the same on a file read in
/// pieces and on a socket.
public struct AnalysisTrackParser: Sendable {
    public private(set) var header: AnalysisTrackHeader?
    private var buffer = Data()

    public init() {}

    /// Appends bytes and returns every record they completed.
    public mutating func append(_ bytes: Data) throws -> [AnalysisRecord] {
        buffer.append(bytes)
        var records: [AnalysisRecord] = []
        if header == nil {
            guard let parsed = try Self.takeHeader(&buffer) else { return [] }
            header = parsed
        }
        while let body = try Self.takeRecordBody(&buffer) {
            records.append(try AnalysisRecord.parse(body))
        }
        return records
    }

    static func takeHeader(_ buffer: inout Data) throws -> AnalysisTrackHeader? {
        let fixed = AnalysisTrack.magic.count + 1 + 4
        guard buffer.count >= fixed else {
            if !AnalysisTrack.magic.starts(with: buffer.prefix(AnalysisTrack.magic.count)) {
                throw AnalysisTrackError.notATrack
            }
            return nil
        }
        guard buffer.prefix(4) == AnalysisTrack.magic else { throw AnalysisTrackError.notATrack }
        let version = buffer[buffer.startIndex + 4]
        guard version == AnalysisTrack.version else { throw AnalysisTrackError.unsupportedVersion(version) }
        let length = Int(buffer.withUnsafeBytes { $0.loadLittleEndian(UInt32.self, at: 5) })
        guard length < 1 << 20 else { throw AnalysisTrackError.tooLarge }
        guard buffer.count >= fixed + length else { return nil }
        let json = buffer.subdata(in: (buffer.startIndex + fixed) ..< (buffer.startIndex + fixed + length))
        buffer.removeFirst(fixed + length)
        return try JSONDecoder().decode(AnalysisTrackHeader.self, from: json)
    }

    static func takeRecordBody(_ buffer: inout Data) throws -> Data? {
        guard buffer.count >= 4 else { return nil }
        let length = Int(buffer.withUnsafeBytes { $0.loadLittleEndian(UInt32.self, at: 0) })
        guard length <= AnalysisTrack.maxRecordBytes else { throw AnalysisTrackError.tooLarge }
        guard buffer.count >= 4 + length else { return nil }
        let body = buffer.subdata(in: (buffer.startIndex + 4) ..< (buffer.startIndex + 4 + length))
        buffer.removeFirst(4 + length)
        return body
    }
}

/// A whole track in memory (or mapped from a file), indexed without decoding
/// payloads, for lookup by time.
public struct AnalysisTrackIndex: Sendable {
    public struct Entry: Sendable, Equatable {
        public let kind: AnalysisKind
        public let eye: AnalysisEye
        public let time: AnalysisTime
        public let endTime: AnalysisTime
        /// Where the record's body starts and how long it is.
        public let range: Range<Int>
    }

    public let header: AnalysisTrackHeader
    /// In file order.
    public let entries: [Entry]
    private let data: Data
    /// Per kind and eye, indices into `entries` sorted by time, so a lookup is
    /// a binary search (a two-hour film holds about 340,000 records).
    private let byKey: [Key: [Int]]

    private struct Key: Hashable {
        let kind: AnalysisKind
        let eye: AnalysisEye
    }

    public init(_ data: Data) throws {
        var rest = data
        guard let header = try AnalysisTrackParser.takeHeader(&rest) else { throw AnalysisTrackError.truncated }
        var entries: [Entry] = []
        var position = data.count - rest.count
        while position < data.count {
            guard data.count - position >= 4 else { throw AnalysisTrackError.truncated }
            let length = Int(data.withUnsafeBytes { $0.loadLittleEndian(UInt32.self, at: position) })
            guard length <= AnalysisTrack.maxRecordBytes, data.count - position - 4 >= length else {
                throw AnalysisTrackError.truncated
            }
            let start = position + 4
            let body = data.subdata(in: (data.startIndex + start) ..< (data.startIndex + start + min(length, AnalysisRecord.headerBytes)))
            let record = try AnalysisRecord.parse(body, declaredLength: length)
            entries.append(Entry(kind: record.kind, eye: record.eye, time: record.time, endTime: record.endTime,
                                 range: start ..< start + length))
            position = start + length
        }
        self.header = header
        self.entries = entries
        self.data = data
        var byKey: [Key: [Int]] = [:]
        for (i, entry) in entries.enumerated() { byKey[Key(kind: entry.kind, eye: entry.eye), default: []].append(i) }
        // Stable: of two records at one time, the later in the file stays later.
        self.byKey = byKey.mapValues { indices in
            indices.enumerated().sorted { a, b in
                let ta = entries[a.element].time, tb = entries[b.element].time
                return ta == tb ? a.offset < b.offset : ta < tb
            }.map(\.element)
        }
    }

    /// Of `kind` and `eye`, how many records start at or before `time`.
    private func countAtOrBefore(_ indices: [Int], _ time: AnalysisTime) -> Int {
        var low = 0, high = indices.count
        while low < high {
            let mid = (low + high) / 2
            if time < entries[indices[mid]].time { high = mid } else { low = mid + 1 }
        }
        return low
    }

    /// The record of `kind` and `eye` whose span holds `time` (for pair kinds)
    /// or whose time is nearest at or before it (otherwise).
    public func entry(_ kind: AnalysisKind, eye: AnalysisEye = .mono, at time: AnalysisTime) -> Entry? {
        guard let indices = byKey[Key(kind: kind, eye: eye)] else { return nil }
        let count = countAtOrBefore(indices, time)
        guard count > 0 else { return nil }
        let latest = entries[indices[count - 1]]
        // A pair record is found by its span; spans don't overlap, so the latest start is the one.
        return latest
    }

    /// The record of `kind` and `eye` whose time is nearest `time`, if it is
    /// within `tolerance` seconds of it.
    public func nearest(_ kind: AnalysisKind, eye: AnalysisEye = .mono, to time: AnalysisTime,
                        within tolerance: Double) -> Entry? {
        guard let indices = byKey[Key(kind: kind, eye: eye)] else { return nil }
        let count = countAtOrBefore(indices, time)
        var best: Entry?
        for i in [count - 1, count] where indices.indices.contains(i) {
            let entry = entries[indices[i]]
            let distance = abs(entry.time.seconds - time.seconds)
            if distance <= tolerance, best.map({ distance < abs($0.time.seconds - time.seconds) }) ?? true {
                best = entry
            }
        }
        return best
    }

    public func record(_ entry: Entry) throws -> AnalysisRecord {
        try AnalysisRecord.parse(data.subdata(in: (data.startIndex + entry.range.lowerBound) ..< (data.startIndex + entry.range.upperBound)))
    }
}

// MARK: - Half floats and little-endian helpers

/// IEEE 754 binary16 by hand: `Float16` doesn't exist on Intel Macs.
public enum HalfFloat {
    public static func bits(from value: Float) -> UInt16 {
        let f = value.bitPattern
        let sign = UInt16((f >> 16) & 0x8000)
        let exponent = Int((f >> 23) & 0xFF) - 127 + 15
        var mantissa = f & 0x7F_FFFF
        if (f & 0x7FFF_FFFF) > 0x7F80_0000 { return sign | 0x7E00 } // NaN
        if exponent >= 0x1F { return sign | 0x7C00 } // overflow → infinity
        if exponent <= 0 {
            guard exponent >= -10 else { return sign } // underflow → zero
            mantissa |= 0x80_0000
            let shift = UInt32(14 - exponent)
            var half = UInt16(mantissa >> shift)
            if (mantissa >> (shift - 1)) & 1 != 0 { half += 1 } // round half up
            return sign | half
        }
        var half = sign | UInt16(exponent << 10) | UInt16(mantissa >> 13)
        if mantissa & 0x1000 != 0 { half &+= 1 } // round half up; may carry into the exponent
        return half
    }

    public static func value(from bits: UInt16) -> Float {
        let sign = UInt32(bits & 0x8000) << 16
        let exponent = Int((bits >> 10) & 0x1F)
        let mantissa = UInt32(bits & 0x3FF)
        if exponent == 0 {
            if mantissa == 0 { return Float(bitPattern: sign) }
            let magnitude = Float(mantissa) / 1024 * pow(2, -14)
            return sign == 0 ? magnitude : -magnitude
        }
        if exponent == 0x1F { return Float(bitPattern: sign | 0x7F80_0000 | (mantissa << 13)) }
        return Float(bitPattern: sign | UInt32(exponent - 15 + 127) << 23 | (mantissa << 13))
    }
}

extension Data {
    mutating func appendLittleEndian<T: FixedWidthInteger>(_ value: T) {
        Swift.withUnsafeBytes(of: value.littleEndian) { append(contentsOf: $0) }
    }
}

extension UnsafeRawBufferPointer {
    func loadLittleEndian<T: FixedWidthInteger>(_: T.Type, at offset: Int) -> T {
        T(littleEndian: loadUnaligned(fromByteOffset: offset, as: T.self))
    }
}

extension Comparable {
    func clamped(to range: ClosedRange<Self>) -> Self { Swift.min(Swift.max(self, range.lowerBound), range.upperBound) }
}
