import Foundation
import Testing
@testable import VergenceDisc

/// A synthetic 3D stream in one file: 24 fps pictures in GOPs, each a base
/// PES (an I picture opening the GOP, P pictures after it) and a dependent
/// PES at the same PTS, decoded a frame before shown, with PAT and PMT.
struct TwoViewStream: Sendable {
    static let start: Int64 = 54_000_000 // 600 s, where tsMuxeR and many discs begin
    static let frame: Int64 = 3754

    var frames = 480
    var gop = 24
    /// Base pictures per extent, dependent extent first, the way an SSIF
    /// interleaves its two clips (each with its own PMT); 0 interleaves
    /// picture by picture under one PMT.
    var extent = 0
    /// Bytes of filler per picture, per view.
    var payload = 64
    var idr = false

    static func pts(_ k: Int) -> Int64 { start + Int64(k) * frame }

    func base(_ k: Int) -> Data {
        let filler = [UInt8](repeating: 0x55, count: payload)
        guard k % gop == 0 else { return nal(9, [0x30]) + nal(1, [0xE0] + filler) } // P slice
        // An I slice (first_mb 0, slice_type 7), or an IDR.
        return nal(9, [0x10]) + nal(7, [0x64, 0, 0x28]) + nal(8, [0xEE]) + nal(idr ? 5 : 1, [0x88] + filler)
    }

    func dependent(_ k: Int) -> Data {
        nal(24, [0x01]) + (k % gop == 0 ? nal(15, [0x76, 0, 0x28]) : Data()) + nal(20, [0x80] + [UInt8](repeating: 0x66, count: payload))
    }

    private static func stream(_ type: UInt8, _ pid: UInt16, language: String? = nil) -> [UInt8] {
        // Typed step by step: Linux's type checker gives up on the one-expression form.
        var descriptor: [UInt8] = []
        if let language {
            descriptor = [0x0A, 4]
            descriptor += Array(language.utf8)
            descriptor.append(0)
        }
        let high: UInt8 = 0xE0 | UInt8(pid >> 8)
        let entry: [UInt8] = [type, high, UInt8(pid & 0xFF), 0xF0, UInt8(descriptor.count)]
        return entry + descriptor
    }

    private static func programTables(_ builder: inout StreamBuilder, streams: [[UInt8]]) {
        builder.section(pid: 0, tableID: 0x00, body: [0, 1, 0xE1, 0x00]) // program 1 -> PMT 0x100
        builder.section(pid: 0x100, tableID: 0x02, body: [0xF0, 0x01, 0xF0, 0x00] + streams.flatMap { $0 })
    }

    static let baseStreams = [stream(0x1B, 0x1011), stream(0x83, 0x1100, language: "eng"), stream(0x90, 0x1200)]

    func bytes(plain: Bool = false) -> Data {
        var builder = StreamBuilder()
        let picture = { (b: inout StreamBuilder, pid: UInt16, k: Int, data: Data) in
            b.pes(pid: pid, pts: Self.pts(k), dts: Self.pts(k) - Self.frame, payload: data)
        }
        if extent == 0 {
            Self.programTables(&builder, streams: Self.baseStreams + [Self.stream(0x20, 0x1012)])
            for k in 0 ..< frames {
                picture(&builder, 0x1011, k, base(k))
                builder.pes(pid: 0x1100, pts: Self.pts(k), payload: Data([0xAB, 0xCD]), declaresLength: true)
                picture(&builder, 0x1012, k, dependent(k))
            }
        } else {
            for first in stride(from: 0, to: frames, by: extent) {
                let range = first ..< min(first + extent, frames)
                Self.programTables(&builder, streams: [Self.stream(0x20, 0x1012)])
                for k in range { picture(&builder, 0x1012, k, dependent(k)) }
                Self.programTables(&builder, streams: Self.baseStreams)
                for k in range { picture(&builder, 0x1011, k, base(k)) }
            }
        }
        guard plain else { return builder.bytes }
        var stripped = Data()
        for at in stride(from: 0, to: builder.bytes.count, by: 192) {
            stripped.append(builder.bytes[(at + 4) ..< (at + 192)])
        }
        return stripped
    }
}

/// Counts the bytes read, to bound what a seek costs.
private final class CountingSource: ByteSource {
    let inner: ByteSource
    var bytesRead = 0
    init(_ inner: ByteSource) { self.inner = inner }
    var size: Int64 { inner.size }
    func read(at offset: Int64, count: Int) throws -> Data {
        let data = try inner.read(at: offset, count: count)
        bytesRead += data.count
        return data
    }
}

@Test func transportStreamFileReadsTheProgramAndTimeline() throws {
    let file = try TransportStreamFile(source: MemoryByteSource(TwoViewStream().bytes()))
    #expect(file.packetSize == 192)
    #expect(file.basePID == 0x1011)
    #expect(file.dependentPID == 0x1012)
    #expect(file.isStereo)
    #expect(file.firstPTS == TwoViewStream.start)
    #expect(file.lastPTS == TwoViewStream.pts(479))
    #expect(file.frameTicks == TwoViewStream.frame)
    let item = try #require(file.playlist.items.first)
    #expect(file.playlist.isStereo)
    #expect(item.inPTS == TwoViewStream.start)
    #expect(abs(item.duration - 480.0 * 3754 / 90_000) < 0.001)
    #expect(item.videoPID == 0x1011 && item.dependentVideoPID == 0x1012)
    #expect(item.audioStreams == [BlurayPlaylist.Stream(pid: 0x1100, codingType: 0x83, language: "eng")])
    #expect(item.subtitleStreams == [BlurayPlaylist.Stream(pid: 0x1200, codingType: 0x90, language: nil)])
}

@Test func aTwoDimensionalStreamHasNoDependentView() throws {
    var builder = StreamBuilder()
    builder.section(pid: 0, tableID: 0x00, body: [0, 1, 0xE1, 0x00])
    builder.section(pid: 0x100, tableID: 0x02, body: [0xF0, 0x01, 0xF0, 0x00, 0x1B, 0xF0, 0x11, 0xF0, 0x00])
    builder.pes(pid: 0x1011, pts: 90_000, payload: nal(5, [0x88]))
    let file = try TransportStreamFile(source: MemoryByteSource(builder.bytes))
    #expect(!file.isStereo)
    #expect(file.playlist.items.first?.dependentClip == nil)
}

@Test func somethingElseIsNotATransportStream() {
    #expect(!TransportStreamFile.looksLikeTransportStream(Data(repeating: 0x1A, count: 2048)))
    #expect(throws: (any Error).self) { try TransportStreamFile(source: MemoryByteSource(Data(repeating: 0, count: 4096))) }
}

/// Fine interleave (tsMuxeR), SSIF extents with two PMTs, plain 188-byte
/// packets, IDR GOPs, and GOPs longer than the search's first look back.
@Test(arguments: [
    ("fine", TwoViewStream(), false),
    ("ssif", TwoViewStream(extent: 30), false),
    ("plain", TwoViewStream(), true),
    ("idr", TwoViewStream(idr: true), false),
    ("long GOP", TwoViewStream(gop: 96), false),
])
func seekingLandsOnTheGOPStartAtOrBeforeTheTarget(name: String, stream: TwoViewStream, plain: Bool) throws {
    let file = try TransportStreamFile(source: MemoryByteSource(stream.bytes(plain: plain)))
    #expect(file.packetSize == (plain ? 188 : 192))
    #expect(file.isStereo, "\(name)")
    let reader = file.stereoReader(for: file.playlist)
    for seconds in [0, 3.3, 7.0, 12.5, 19.9] {
        let frame = Int(seconds * 90_000 / Double(TwoViewStream.frame))
        let expected = Int64(frame / stream.gop * stream.gop) * TwoViewStream.frame
        let entry = try reader.seek(to: seconds)
        #expect(entry == expected, "\(name) at \(seconds) s")
        var units: [StereoAccessUnit] = []
        while units.count < 30, let unit = try reader.next() { units.append(unit) }
        #expect(units.first?.pts == expected, "\(name) at \(seconds) s")
        #expect(units.allSatisfy { $0.hasDependentView }, "\(name) at \(seconds) s")
        #expect(zip(units, units.dropFirst()).allSatisfy { $1.pts! - $0.pts! == TwoViewStream.frame })
    }
    #expect(reader.unpairedBaseCount == 0)
}

@Test func entryPointsAreAtPacketBoundaries() throws {
    for plain in [false, true] {
        let file = try TransportStreamFile(source: MemoryByteSource(TwoViewStream().bytes(plain: plain)))
        let entry = try #require(try file.entryPoint(clip: TransportStreamFile.clipName, pid: 0x1011, atOrBefore: TwoViewStream.pts(100)))
        #expect(entry.pts == TwoViewStream.pts(96))
        #expect(entry.packetSize == file.packetSize)
        let bytes = try file.source.read(at: entry.byteOffset, count: file.packetSize)
        #expect(bytes[file.packetSize - 188] == 0x47)
        #expect(bytes[file.packetSize - 188 + 1] & 0x40 != 0) // a PES starts here
        let dependent = try #require(try file.entryPoint(clip: TransportStreamFile.clipName, pid: 0x1012, atOrBefore: entry.pts))
        #expect(dependent.pts == entry.pts)
        #expect(dependent.byteOffset > entry.byteOffset) // muxed right after it
        let next = try file.entryPoint(clip: TransportStreamFile.clipName, pid: 0x1011, after: entry.pts)
        #expect(next?.pts == TwoViewStream.pts(120))
    }
}

@Test func aSeekReadsAFewSecondsOfTheFileNotAllOfIt() throws {
    // Ten minutes, about 4.4 kB a picture over both views: 63 MB.
    let stream = TwoViewStream(frames: 24 * 600, payload: 2000)
    let data = stream.bytes()
    let source = CountingSource(MemoryByteSource(data))
    let file = try TransportStreamFile(source: source)
    let perSecond = data.count / 600
    for seconds in [37.0, 290.0, 555.5, 123.4] {
        source.bytesRead = 0
        let entry = try #require(try file.entryPoint(clip: TransportStreamFile.clipName, pid: 0x1011, atOrBefore: TwoViewStream.start + Int64(seconds * 90_000)))
        let expected = TwoViewStream.pts(Int(seconds * 90_000 / Double(TwoViewStream.frame)) / 24 * 24)
        #expect(entry.pts == expected)
        // A scan of two seconds or so, and a dozen probes of 256 packets
        // (a lot of this stream, little of a Blu-ray's 5 MB a second).
        let probes = 12 * 256 * 192
        #expect(source.bytesRead < 2 * perSecond + probes, "\(seconds) s: \(source.bytesRead) bytes, \(perSecond) a second")
    }
}

@Test func isRandomAccessTellsIntraFromPredictedPictures() {
    #expect(TransportStreamFile.isRandomAccess(nal(9) + nal(5, [0xB8])))
    #expect(TransportStreamFile.isRandomAccess(nal(9) + nal(6, [5, 1, 0]) + nal(1, [0x88]))) // I, slice_type 7
    #expect(TransportStreamFile.isRandomAccess(nal(1, [0xB0]))) // I, slice_type 2
    #expect(!TransportStreamFile.isRandomAccess(nal(9) + nal(1, [0xE0]))) // P
    #expect(!TransportStreamFile.isRandomAccess(nal(1, [0xA0]))) // B
    #expect(TransportStreamFile.isRandomAccess(nal(9) + nal(7, [0x64]))) // parameter sets, slice cut off
    #expect(!TransportStreamFile.isRandomAccess(nal(9)))
}

@Test func aTrueHDTrackWithoutTheAC3CoreIsToldApart() throws {
    // The synthetic stream's 0x1100 PES carry two bytes of TrueHD-ish data.
    let bare = try TransportStreamFile(source: MemoryByteSource(TwoViewStream().bytes()))
    #expect(try !bare.carriesAC3Frames(pid: 0x1100))
    #expect(try bare.carriesAC3Frames(pid: 0x1300)) // not in the file: not written off

    var builder = StreamBuilder()
    builder.section(pid: 0, tableID: 0x00, body: [0, 1, 0xE1, 0x00])
    builder.section(pid: 0x100, tableID: 0x02, body: [0xF0, 0x01, 0xF0, 0x00, 0x1B, 0xF0, 0x11, 0xF0, 0x00, 0x83, 0xF1, 0x00, 0xF0, 0x00])
    for k in 0 ..< 4 {
        builder.pes(pid: 0x1011, pts: TwoViewStream.pts(k), payload: nal(5, [0x88]))
        builder.pes(pid: 0x1100, pts: TwoViewStream.pts(k), payload: Data([0xF8, 0x72, 0x6F, 0xBA]), declaresLength: true)
        builder.pes(pid: 0x1100, pts: TwoViewStream.pts(k), payload: Data([0x0B, 0x77, 0x01, 0x02]), declaresLength: true)
    }
    let cored = try TransportStreamFile(source: MemoryByteSource(builder.bytes))
    #expect(try cored.carriesAC3Frames(pid: 0x1100))
}
