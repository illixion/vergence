import Foundation
import Testing
@testable import VergenceDisc

/// PGS segments, built the way a disc carries them.
enum PGS {
    static func segment(_ type: UInt8, _ body: [UInt8]) -> Data {
        Data([type, UInt8(body.count >> 8), UInt8(body.count & 0xFF)] + body)
    }

    static func u16(_ v: Int) -> [UInt8] { [UInt8(v >> 8), UInt8(v & 0xFF)] }

    /// A composition placing `objects` (id, x, y); state 0x80 = epoch start.
    /// `objectFlags` per object: 0x40 forced, 0x80 cropped (a crop rectangle
    /// of the object's own top-left 2x1 pixels follows).
    static func composition(state: UInt8, objects: [(UInt16, Int, Int)], paletteUpdate: Bool = false,
                            objectFlags: UInt8 = 0) -> Data {
        var body = u16(1920) + u16(1080) + [0x10] + u16(1) + [state, paletteUpdate ? 0x80 : 0, 0, UInt8(objects.count)]
        for (id, x, y) in objects {
            body += u16(Int(id)) + [0, objectFlags] + u16(x) + u16(y)
            if objectFlags & 0x80 != 0 { body += u16(0) + u16(0) + u16(2) + u16(1) }
        }
        return segment(0x16, body)
    }

    /// Palette 0: index 1 opaque white, index 2 half-transparent black.
    static func palette(version: UInt8 = 0, whiteY: UInt8 = 235) -> Data {
        segment(0x14, [0, version, 1, whiteY, 128, 128, 255, 2, 16, 128, 128, 128])
    }

    /// An object split into `parts` segments.
    static func object(id: UInt16, width: Int, height: Int, rle: [UInt8], parts: Int = 1) -> [Data] {
        let length = rle.count + 4
        let chunk = (rle.count + parts - 1) / parts
        return (0 ..< parts).map { i in
            let slice = Array(rle[min(i * chunk, rle.count) ..< min((i + 1) * chunk, rle.count)])
            var flags: UInt8 = 0
            if i == 0 { flags |= 0x80 }
            if i == parts - 1 { flags |= 0x40 }
            var body = u16(Int(id)) + [0, flags]
            if i == 0 { body += [UInt8(length >> 16), UInt8(length >> 8 & 0xFF), UInt8(length & 0xFF)] + u16(width) + u16(height) }
            return segment(0x15, body + slice)
        }
    }

    static let end = segment(0x80, [])
}

/// 4x2: row 0 is white, white, (2 transparent); row 1 is a run of 4 colour-2
/// pixels coded long-form.
private let rle: [UInt8] = [1, 1, 0, 0x02, 0, 0, 0, 0xC0, 4, 2, 0, 0]

@Test func pgsDisplaySetDecodesToPlacedPixels() throws {
    var decoder = PGSDecoder()
    var captions: [PGSCaption] = []
    for segment in [PGS.composition(state: 0x80, objects: [(7, 100, 900)]), PGS.palette()]
        + PGS.object(id: 7, width: 4, height: 2, rle: rle, parts: 3) + [PGS.end] {
        captions += decoder.feed(pes: segment, pts: 9000)
    }
    #expect(captions.count == 1)
    let caption = try #require(captions.first)
    #expect(caption.pts == 9000)
    #expect(caption.isRefresh)
    #expect(caption.bounds?.x == 100 && caption.bounds?.y == 900)
    let image = try #require(caption.renderRGBA())
    #expect(image.width == 4 && image.height == 2)
    // Opaque white (limited range 235 -> 255), then nothing, then half-alpha black.
    #expect(image.pixels[0] == 0xFFFF_FFFF)
    #expect(image.pixels[1] == 0xFFFF_FFFF)
    #expect(image.pixels[2] == 0)
    #expect(image.pixels[4] == 0x8000_0000)
    #expect(image.pixels[7] == 0x8000_0000)
}

@Test func pgsEmptyCompositionClears() {
    var decoder = PGSDecoder()
    var captions: [PGSCaption] = []
    let show = [PGS.composition(state: 0x80, objects: [(0, 0, 0)]), PGS.palette()]
        + PGS.object(id: 0, width: 4, height: 2, rle: rle) + [PGS.end]
    for segment in show { captions += decoder.feed(pes: segment, pts: 0) }
    for segment in [PGS.composition(state: 0, objects: []), PGS.end] { captions += decoder.feed(pes: segment, pts: 180_000) }
    #expect(captions.count == 2)
    #expect(captions[1].isEmpty)
    #expect(captions[1].pts == 180_000)
}

@Test func pgsWaitsForARefreshAfterAReset() {
    var decoder = PGSDecoder()
    decoder.reset()
    var captions: [PGSCaption] = []
    // A normal composition refers to objects defined before the seek point.
    for segment in [PGS.composition(state: 0, objects: [(3, 0, 0)]), PGS.end] { captions += decoder.feed(pes: segment, pts: 0) }
    #expect(captions.isEmpty)
    // An acquisition point carries everything it needs.
    for segment in [PGS.composition(state: 0x40, objects: [(3, 0, 0)]), PGS.palette()]
        + PGS.object(id: 3, width: 4, height: 2, rle: rle) + [PGS.end] {
        captions += decoder.feed(pes: segment, pts: 90)
    }
    #expect(captions.count == 1)
    #expect(captions.first?.renderRGBA() != nil)
}

@Test func pgsPaletteUpdateReusesObjects() throws {
    var decoder = PGSDecoder()
    var captions: [PGSCaption] = []
    for segment in [PGS.composition(state: 0x80, objects: [(1, 0, 0)]), PGS.palette()]
        + PGS.object(id: 1, width: 4, height: 2, rle: rle) + [PGS.end] {
        captions += decoder.feed(pes: segment, pts: 0)
    }
    // A fade: same object, new palette version with darker white.
    for segment in [PGS.composition(state: 0, objects: [(1, 0, 0)], paletteUpdate: true), PGS.palette(version: 1, whiteY: 16), PGS.end] {
        captions += decoder.feed(pes: segment, pts: 3000)
    }
    #expect(captions.count == 2)
    let faded = try #require(captions[1].renderRGBA())
    #expect(faded.pixels[0] == 0xFF00_0000)
}

@Test func supFileCaptionsCarryTheirTimestamps() {
    var file = Data()
    func add(_ segment: Data, pts: UInt32) {
        file.append(contentsOf: [0x50, 0x47])
        file.append(contentsOf: withUnsafeBytes(of: pts.bigEndian, Array.init))
        file.append(contentsOf: [0, 0, 0, 0])
        file.append(segment)
    }
    for segment in [PGS.composition(state: 0x80, objects: [(0, 10, 20)]), PGS.palette()]
        + PGS.object(id: 0, width: 4, height: 2, rle: rle) + [PGS.end] {
        add(segment, pts: 45_000)
    }
    let captions = SupFile.captions(file)
    #expect(captions.count == 1)
    #expect(captions.first?.pts == 45_000)
}

@Test func streamReaderHandsOutSidePIDsAlongsideTheVideo() throws {
    var builder = StreamBuilder()
    builder.pes(pid: 0x1200, pts: 1000, payload: Data([0xAB]), declaresLength: true) // subtitle, muxed ahead
    builder.pes(pid: 0x1011, pts: 3000, payload: nal(9) + nal(5))
    builder.pes(pid: 0x1100, pts: 3000, payload: Data([0xCD]), declaresLength: true) // audio
    builder.pes(pid: 0x1011, pts: 6754, payload: nal(9) + nal(1))
    builder.pes(pid: 0x1300, pts: 6754, payload: Data([0xEE]), declaresLength: true) // not asked for
    builder.pes(pid: 0x1011, pts: 10_508, payload: nal(9) + nal(1))

    let reader = StereoStreamReader(base: MemoryByteSource(builder.bytes), dependent: nil, sidePIDs: [0x1100, 0x1200])
    var side: [PESPacket] = []
    var units = 0
    while try reader.next() != nil {
        units += 1
        side += reader.takeSidePackets()
        // Complete side PES come out with the picture they're muxed by,
        // not when the next PES on their PID starts.
        if units == 1 { #expect(side.map(\.pid) == [0x1200, 0x1100]) }
    }
    side += reader.takeSidePackets()
    // The last picture is dropped as possibly cut short; nothing follows it.
    #expect(units == 2)
    #expect(side.map(\.pid) == [0x1200, 0x1100])
    #expect(side.map(\.pts) == [1000, 3000])
}

/// A forced-only track flags every caption 0x40 (forced). Read as the cropped
/// flag, it sent the parser looking for a crop rectangle that isn't there and
/// every caption of the track came out empty (Baby Driver UHD, 2026-10-10).
@Test func pgsForcedCaptionsKeepTheirObjects() throws {
    var decoder = PGSDecoder()
    var captions: [PGSCaption] = []
    for segment in [PGS.composition(state: 0x80, objects: [(7, 100, 900)], objectFlags: 0x40), PGS.palette()]
        + PGS.object(id: 7, width: 4, height: 2, rle: rle) + [PGS.end] {
        captions += decoder.feed(pes: segment, pts: 9000)
    }
    let caption = try #require(captions.first)
    #expect(!caption.isEmpty)
    #expect(caption.bounds?.x == 100 && caption.bounds?.width == 4)
}

@Test func pgsCroppedCaptionsReadTheirCropRectangle() throws {
    var decoder = PGSDecoder()
    var captions: [PGSCaption] = []
    for segment in [PGS.composition(state: 0x80, objects: [(7, 100, 900)], objectFlags: 0x80), PGS.palette()]
        + PGS.object(id: 7, width: 4, height: 2, rle: rle) + [PGS.end] {
        captions += decoder.feed(pes: segment, pts: 9000)
    }
    let caption = try #require(captions.first)
    #expect(caption.objects.first?.crop?.width == 2)
}
