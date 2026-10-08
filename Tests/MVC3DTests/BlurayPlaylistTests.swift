import Foundation
import Testing
@testable import MVC3D

/// Writes an MPLS shaped like a 3D Blu-ray's feature playlist.
struct PlaylistBuilder {
    struct Item { var clip: String; var dependent: String?; var inTime: UInt32; var outTime: UInt32 }
    var items: [Item]
    var multiAngleFirstItem = false
    /// Primary audio and PGS streams on every item: PID, coding type, language.
    var audio: [(UInt16, UInt8, String)] = []
    var subtitles: [(UInt16, String)] = []

    func build() -> Data {
        var playList = Data()
        playList += u16(0) + u16(UInt16(items.count)) + u16(0)
        for (index, item) in items.enumerated() {
            var body = Data(item.clip.utf8) + Data("M2TS".utf8)
            let multiAngle = index == 0 && multiAngleFirstItem
            body += u16(multiAngle ? 0x11 : 0x01) + Data([0]) + u32(item.inTime) + u32(item.outTime)
            body += Data(count: 8) + Data([0]) + Data([0]) + u16(0)
            if multiAngle {
                body += Data([2, 0]) + Data("00099".utf8) + Data("M2TS".utf8) + Data([0])
            }
            // STN_table: one primary video stream, PID 0x1011, then audio and PGS.
            var stn = u16(0) + Data([1, UInt8(audio.count), UInt8(subtitles.count), 0, 0, 0, 0]) + Data(count: 5)
            stn += Data([9, 1]) + u16(0x1011) + Data(count: 6) + Data([5, 0x1B, 0x61, 0, 0, 0])
            for (pid, type, language) in audio {
                stn += Data([9, 1]) + u16(pid) + Data(count: 6) + Data([5, type, 0x31]) + Data(language.utf8)
            }
            for (pid, language) in subtitles {
                stn += Data([9, 1]) + u16(pid) + Data(count: 6) + Data([5, 0x90]) + Data(language.utf8) + Data([0])
            }
            body += u16(UInt16(stn.count)) + stn
            playList += u16(UInt16(body.count)) + body
        }
        playList = u32(UInt32(playList.count)) + playList

        var ext = Data()
        if items.contains(where: { $0.dependent != nil }) {
            // STN_table_SS (2/1): per item, the dependent view in SubPath 1.
            var ss = Data()
            for _ in items {
                let entry = u16(0) + Data([9, 2, 1, 0]) + u16(0x1012) + Data(count: 4) + Data([5, 0x20, 0x61, 0, 0, 0])
                ss += u16(UInt16(entry.count)) + entry
            }
            // SubPath extension (2/2): one type-8 SubPath, a SubPlayItem per item.
            var subItems = Data()
            for (index, item) in items.enumerated() {
                let s = Data((item.dependent ?? "00000").utf8) + Data("M2TS".utf8) + u32(0x02) + Data([0])
                    + u32(item.inTime) + u32(item.outTime) + u16(UInt16(index)) + u32(item.inTime)
                subItems += u16(UInt16(s.count)) + s
            }
            let subPathBody = Data([0, 8]) + u16(0) + Data([0, UInt8(items.count)]) + subItems
            let subPath = u32(UInt32(subPathBody.count)) + subPathBody
            let spExt = u32(UInt32(2 + subPath.count)) + u16(1) + subPath

            let headerSize = 12 + 2 * 12
            let firstStart = UInt32(headerSize)
            let secondStart = firstStart + UInt32(ss.count)
            var block = u32(firstStart) + Data([0, 0, 0, 2])
            block += u16(2) + u16(1) + u32(firstStart) + u32(UInt32(ss.count))
            block += u16(2) + u16(2) + u32(secondStart) + u32(UInt32(spExt.count))
            block = u32(UInt32(block.count + ss.count + spExt.count)) + block
            ext = block + ss + spExt
        }

        let playListStart: UInt32 = 0x3A
        let markStart = playListStart + UInt32(playList.count)
        let mark = u32(2) + u16(0)
        let extStart = ext.isEmpty ? 0 : markStart + UInt32(mark.count)
        var file = Data("MPLS0200".utf8) + u32(playListStart) + u32(markStart) + u32(extStart)
        file += Data(count: Int(playListStart) - file.count)
        return file + playList + mark + ext
    }

    private func u16(_ v: UInt16) -> Data { Data([UInt8(v >> 8), UInt8(v & 0xFF)]) }
    private func u32(_ v: UInt32) -> Data { Data([UInt8(v >> 24), UInt8(v >> 16 & 0xFF), UInt8(v >> 8 & 0xFF), UInt8(v & 0xFF)]) }
}

@Test func parsesA3DFeaturePlaylist() throws {
    let data = PlaylistBuilder(items: [
        .init(clip: "00007", dependent: "00008", inTime: 24_377, outTime: 307_495_916),
    ]).build()
    let playlist = try BlurayPlaylist(data: data)
    #expect(playlist.items.count == 1)
    let item = try #require(playlist.items.first)
    #expect(item.clip == "00007")
    #expect(item.dependentClip == "00008")
    #expect(item.videoPID == 0x1011)
    #expect(item.dependentVideoPID == 0x1012)
    #expect(item.inPTS == 48_754)
    #expect(playlist.isStereo)
    #expect(abs(playlist.duration - 6832.70) < 0.01)
}

@Test func listsTheAudioAndSubtitleStreamsWithTheirLanguages() throws {
    var builder = PlaylistBuilder(items: [.init(clip: "00007", dependent: "00008", inTime: 0, outTime: 45_000)])
    builder.audio = [(0x1100, 0x83, "deu"), (0x1101, 0x83, "eng"), (0x1102, 0x81, "fra")]
    builder.subtitles = [(0x1200, "deu"), (0x1201, "eng")]
    let item = try #require(try BlurayPlaylist(data: builder.build()).items.first)
    #expect(item.videoPID == 0x1011)
    #expect(item.audioStreams == [
        .init(pid: 0x1100, codingType: 0x83, language: "deu"),
        .init(pid: 0x1101, codingType: 0x83, language: "eng"),
        .init(pid: 0x1102, codingType: 0x81, language: "fra"),
    ])
    #expect(item.audioStreams.map(\.codecName) == ["truehd", "truehd", "ac3"])
    #expect(item.subtitleStreams.map(\.pid) == [0x1200, 0x1201])
    #expect(item.subtitleStreams.map(\.language) == ["deu", "eng"])
    #expect(item.dependentVideoPID == 0x1012)
}

@Test func parsesMultipleItemsAndAMultiAngleItem() throws {
    var builder = PlaylistBuilder(items: [
        .init(clip: "00010", dependent: "00011", inTime: 0, outTime: 45_000 * 60),
        .init(clip: "00012", dependent: "00013", inTime: 900, outTime: 45_000 * 90),
    ])
    builder.multiAngleFirstItem = true
    let playlist = try BlurayPlaylist(data: builder.build())
    #expect(playlist.items.map(\.clip) == ["00010", "00012"])
    #expect(playlist.items.map(\.dependentClip) == ["00011", "00013"])
    #expect(playlist.isStereo)
}

@Test func a2DPlaylistHasNoDependentClips() throws {
    let playlist = try BlurayPlaylist(data: PlaylistBuilder(items: [
        .init(clip: "00001", dependent: nil, inTime: 0, outTime: 45_000),
    ]).build())
    #expect(playlist.items.first?.dependentClip == nil)
    #expect(!playlist.isStereo)
}

@Test func rejectsWhatIsNotAPlaylist() {
    #expect(throws: FormatError.self) { try BlurayPlaylist(data: Data("HDMV0200".utf8) + Data(count: 40)) }
    #expect(throws: FormatError.self) { try BlurayPlaylist(data: Data("MPLS02".utf8)) }
}

@Test func playlistOffsetsWrapOnThe33BitClock() {
    #expect(PlaylistStereoReader.offset(48_754 + 3_754, from: 48_754) == 3_754)
    #expect(PlaylistStereoReader.offset(48_754 - 3_754, from: 48_754) == -3_754)
    #expect(PlaylistStereoReader.offset(100, from: (1 << 33) - 200) == 300)
}
