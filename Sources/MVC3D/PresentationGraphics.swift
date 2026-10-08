/*
 mvc3d - Blu-ray subtitles (Presentation Graphics, "PGS")

 A Blu-ray subtitle track is a stream of segments: a composition (PCS)
 naming which objects show where, windows (WDS), palettes (PDS: 256 entries
 of Y'CbCr + alpha) and objects (ODS: run-length coded 8-bit bitmaps,
 possibly split over several segments), ending in an END segment that
 completes a display set. A display set takes effect at its composition's
 PTS and lasts until the next one; a composition with no objects clears
 the screen.

 Segments come either from the PES packets of the subtitle PID in a clip
 (`feed(pes:)`, PTS from the PES header) or from a `.sup` file
 (`SupFile`), which prefixes each segment with "PG" and its timestamps.

 The decoder keeps objects compressed and hands out `PGSCaption`s that
 decode to RGBA on request: a film has thousands of captions, and their
 bitmaps at screen size would not fit in memory together.

 Colours follow the Blu-ray specification: BT.709 Y'CbCr, limited range
 (Y' 16-235), straight alpha. An HD disc's PGS is authored for a 1920x1080
 frame regardless of the video's crop.
 */

import Foundation

/// One display set: what the screen shows from `pts` until the next caption
/// of the same track.
public struct PGSCaption: Sendable {
    /// 90 kHz, on the timeline of whatever fed the decoder.
    public var pts: Int64
    /// The frame the positions are relative to (normally 1920x1080).
    public var videoWidth: Int
    public var videoHeight: Int
    /// Empty for a caption that clears the screen.
    public var objects: [PlacedObject]
    /// 256 premultiplied RGBA entries.
    public var palette: [UInt32]
    /// A composition from an epoch start or acquisition point, which a
    /// player joining mid-stream can show on its own.
    public var isRefresh: Bool

    public struct PlacedObject: Sendable {
        public var x: Int
        public var y: Int
        public var bitmap: PGSBitmap
        /// The part of the bitmap shown, in bitmap coordinates.
        public var crop: (x: Int, y: Int, width: Int, height: Int)?
    }

    public var isEmpty: Bool { objects.isEmpty }

    /// The area the objects cover, in frame coordinates.
    public var bounds: (x: Int, y: Int, width: Int, height: Int)? {
        guard !objects.isEmpty else { return nil }
        var minX = Int.max, minY = Int.max, maxX = Int.min, maxY = Int.min
        for object in objects {
            let size = object.shownSize
            minX = min(minX, object.x)
            minY = min(minY, object.y)
            maxX = max(maxX, object.x + size.width)
            maxY = max(maxY, object.y + size.height)
        }
        return (minX, minY, maxX - minX, maxY - minY)
    }

    /// The caption over its `bounds`, premultiplied RGBA, rows top to
    /// bottom. Nil for an empty caption or a bitmap that doesn't decode.
    public func renderRGBA() -> (x: Int, y: Int, width: Int, height: Int, pixels: [UInt32])? {
        guard let bounds, bounds.width > 0, bounds.height > 0 else { return nil }
        var pixels = [UInt32](repeating: 0, count: bounds.width * bounds.height)
        var drew = false
        for object in objects {
            guard let indexes = object.bitmap.decodeIndexes() else { continue }
            let crop = object.crop ?? (0, 0, object.bitmap.width, object.bitmap.height)
            let size = object.shownSize
            for row in 0 ..< size.height {
                let sourceRow = (crop.y + row) * object.bitmap.width + crop.x
                let target = (object.y - bounds.y + row) * bounds.width + (object.x - bounds.x)
                for column in 0 ..< size.width {
                    let color = palette[Int(indexes[sourceRow + column])]
                    if color != 0 { pixels[target + column] = color }
                }
            }
            drew = true
        }
        return drew ? (bounds.x, bounds.y, bounds.width, bounds.height, pixels) : nil
    }
}

extension PGSCaption.PlacedObject {
    var shownSize: (width: Int, height: Int) {
        guard let crop else { return (bitmap.width, bitmap.height) }
        return (min(crop.width, max(bitmap.width - crop.x, 0)), min(crop.height, max(bitmap.height - crop.y, 0)))
    }
}

/// An object's run-length coded bitmap.
public struct PGSBitmap: Sendable {
    public var width: Int
    public var height: Int
    var rle: Data

    /// Palette indexes, row by row, or nil when the data runs short or
    /// overflows (a damaged object).
    public func decodeIndexes() -> [UInt8]? {
        guard width > 0, height > 0 else { return nil }
        var out = [UInt8](repeating: 0, count: width * height)
        var x = 0, y = 0
        let ok = rle.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> Bool in
            let b = raw.bindMemory(to: UInt8.self)
            var i = 0
            func run(_ count: Int, _ color: UInt8) -> Bool {
                guard y < height else { return false }
                let n = min(count, width - x)
                if color != 0 {
                    let start = y * width + x
                    for k in 0 ..< n { out[start + k] = color }
                }
                x += n
                return true
            }
            while i < b.count, y < height {
                let first = b[i]; i += 1
                if first != 0 {
                    if !run(1, first) { return false }
                    continue
                }
                guard i < b.count else { break }
                let flags = b[i]; i += 1
                if flags == 0 {
                    // End of line.
                    x = 0
                    y += 1
                    continue
                }
                var count = Int(flags & 0x3F)
                if flags & 0x40 != 0 {
                    guard i < b.count else { return false }
                    count = count << 8 | Int(b[i]); i += 1
                }
                var color: UInt8 = 0
                if flags & 0x80 != 0 {
                    guard i < b.count else { return false }
                    color = b[i]; i += 1
                }
                if !run(count, color) { return false }
            }
            return true
        }
        return ok ? out : nil
    }
}

/// Turns PGS segments into captions.
public struct PGSDecoder {
    private struct Composition {
        var pts: Int64
        var width: Int
        var height: Int
        var state: UInt8
        var paletteID: UInt8
        var objects: [(id: UInt16, x: Int, y: Int, crop: (x: Int, y: Int, width: Int, height: Int)?)]
    }

    private struct PendingObject {
        var width: Int
        var height: Int
        var expected: Int
        var data: Data
    }

    private var composition: Composition?
    private var palettes: [UInt8: [UInt32]] = [:]
    private var objects: [UInt16: PGSBitmap] = [:]
    private var pending: [UInt16: PendingObject] = [:]
    /// Waiting for an epoch start or acquisition point: after a seek, a
    /// normal composition refers to objects this decoder never saw.
    private var needsRefresh = true

    public init() {}

    /// Forgets everything, for a seek.
    public mutating func reset() {
        composition = nil
        palettes = [:]
        objects = [:]
        pending = [:]
        needsRefresh = true
    }

    /// One PES payload of a subtitle PID; returns the captions it completed.
    public mutating func feed(pes payload: Data, pts: Int64?) -> [PGSCaption] {
        var captions: [PGSCaption] = []
        var offset = payload.startIndex
        while offset + 3 <= payload.endIndex {
            let type = payload[offset]
            let length = Int(payload[offset + 1]) << 8 | Int(payload[offset + 2])
            let body = offset + 3
            guard body + length <= payload.endIndex else { break }
            if let caption = segment(type: type, payload.subdata(in: body ..< body + length), pts: pts ?? 0) {
                captions.append(caption)
            }
            offset = body + length
        }
        return captions
    }

    /// One segment's type and body, as a `.sup` file carries them.
    public mutating func feed(segment type: UInt8, body: Data, pts: Int64) -> PGSCaption? {
        segment(type: type, body, pts: pts)
    }

    private mutating func segment(type: UInt8, _ s: Data, pts: Int64) -> PGSCaption? {
        let b = [UInt8](s)
        func u16(_ i: Int) -> Int { Int(b[i]) << 8 | Int(b[i + 1]) }
        switch type {
        case 0x16: // presentation composition
            guard b.count >= 11 else { return nil }
            let state = b[7]
            if state & 0x80 != 0 {
                // Epoch start: a new set of objects and palettes.
                objects = [:]
                palettes = [:]
                pending = [:]
            }
            if state & 0xC0 != 0 { needsRefresh = false }
            var placed: [(UInt16, Int, Int, (Int, Int, Int, Int)?)] = []
            var i = 11
            for _ in 0 ..< Int(b[10]) {
                guard i + 8 <= b.count else { break }
                let id = UInt16(u16(i))
                let cropped = b[i + 3] & 0x40 != 0
                let x = u16(i + 4), y = u16(i + 6)
                i += 8
                var crop: (Int, Int, Int, Int)?
                if cropped {
                    guard i + 8 <= b.count else { break }
                    crop = (u16(i), u16(i + 2), u16(i + 4), u16(i + 6))
                    i += 8
                }
                placed.append((id, x, y, crop))
            }
            composition = Composition(pts: pts, width: u16(0), height: u16(2), state: state, paletteID: b[9],
                                      objects: placed.map { (id: $0.0, x: $0.1, y: $0.2, crop: $0.3.map { (x: $0.0, y: $0.1, width: $0.2, height: $0.3) }) })
        case 0x14: // palette
            guard b.count >= 2 else { return nil }
            var palette = palettes[b[0]] ?? [UInt32](repeating: 0, count: 256)
            var i = 2
            while i + 5 <= b.count {
                palette[Int(b[i])] = Self.rgba(y: b[i + 1], cr: b[i + 2], cb: b[i + 3], alpha: b[i + 4])
                i += 5
            }
            palettes[b[0]] = palette
        case 0x15: // object
            guard b.count >= 4 else { return nil }
            let id = UInt16(u16(0))
            let sequence = b[3]
            if sequence & 0x80 != 0 {
                guard b.count >= 11 else { return nil }
                let length = Int(b[4]) << 16 | Int(b[5]) << 8 | Int(b[6])
                // The length counts the width and height that follow it.
                pending[id] = PendingObject(width: u16(7), height: u16(9), expected: max(length - 4, 0),
                                            data: Data(b[11...]))
            } else if var object = pending[id] {
                object.data.append(contentsOf: b[4...])
                pending[id] = object
            }
            if sequence & 0x40 != 0, let object = pending.removeValue(forKey: id) {
                objects[id] = PGSBitmap(width: object.width, height: object.height, rle: object.data)
            }
        case 0x80: // end of display set
            guard let composition else { return nil }
            self.composition = nil
            guard !needsRefresh else { return nil }
            let palette = palettes[composition.paletteID] ?? [UInt32](repeating: 0, count: 256)
            let placed = composition.objects.compactMap { entry -> PGSCaption.PlacedObject? in
                guard let bitmap = objects[entry.id] else { return nil }
                return PGSCaption.PlacedObject(x: entry.x, y: entry.y, bitmap: bitmap, crop: entry.crop)
            }
            return PGSCaption(pts: composition.pts, videoWidth: composition.width, videoHeight: composition.height,
                              objects: placed, palette: palette, isRefresh: composition.state & 0xC0 != 0)
        default: // window definitions and anything unknown: positions come from the composition
            break
        }
        return nil
    }

    /// BT.709 limited-range Y'CbCr to premultiplied RGBA, packed as bytes
    /// R, G, B, A in memory order.
    static func rgba(y: UInt8, cr: UInt8, cb: UInt8, alpha: UInt8) -> UInt32 {
        guard alpha > 0 else { return 0 }
        let yy = (Double(y) - 16) * 255 / 219
        let pb = (Double(cb) - 128) * 255 / 224
        let pr = (Double(cr) - 128) * 255 / 224
        let a = Double(alpha) / 255
        func clamp(_ v: Double) -> UInt32 { UInt32(max(0, min(255, (v * a).rounded()))) }
        let r = clamp(yy + 1.5748 * pr)
        let g = clamp(yy - 0.1873 * pb - 0.4681 * pr)
        let bl = clamp(yy + 1.8556 * pb)
        return r | g << 8 | bl << 16 | UInt32(alpha) << 24
    }
}

/// A `.sup` file: PGS segments, each prefixed with "PG", PTS and DTS.
public enum SupFile {
    /// Every caption in the file, in order.
    public static func captions(_ data: Data) -> [PGSCaption] {
        var decoder = PGSDecoder()
        var captions: [PGSCaption] = []
        var i = data.startIndex
        while i + 13 <= data.endIndex {
            guard data[i] == 0x50, data[i + 1] == 0x47 else { break }
            let pts = (2 ..< 6).reduce(Int64(0)) { $0 << 8 | Int64(data[i + $1]) }
            let type = data[i + 10]
            let length = Int(data[i + 11]) << 8 | Int(data[i + 12])
            let body = i + 13
            guard body + length <= data.endIndex else { break }
            if let caption = decoder.feed(segment: type, body: data.subdata(in: body ..< body + length), pts: pts) {
                captions.append(caption)
            }
            i = body + length
        }
        return captions
    }
}
