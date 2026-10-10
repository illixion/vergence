/*
 Vergence - a Blu-ray disc: its playlists and clips

 The disc is reached through `DiscFileSystem`, so the same code reads a BDMV
 folder (a rip, a mounted disc) or, through `UDFFileSystem`, a disc image
 straight from its bytes, local or remote.
 */

import Foundation

public protocol DiscFileSystem: AnyObject {
    /// File names in a directory, relative to the disc root ("BDMV/PLAYLIST").
    func contentsOfDirectory(_ path: String) throws -> [String]
    func open(_ path: String) throws -> ByteSource
}

extension DiscFileSystem {
    func readAll(_ path: String) throws -> Data {
        let source = try open(path)
        return try source.read(at: 0, count: Int(source.size))
    }
}

/// A folder holding `BDMV` (or the `BDMV` folder itself).
public final class DirectoryFileSystem: DiscFileSystem {
    private let root: URL

    public init(root: URL) {
        self.root = root.lastPathComponent.uppercased() == "BDMV" ? root.deletingLastPathComponent() : root
    }

    public func contentsOfDirectory(_ path: String) throws -> [String] {
        try FileManager.default.contentsOfDirectory(atPath: root.appendingPathComponent(path).path)
    }

    public func open(_ path: String) throws -> ByteSource {
        try FileByteSource(url: root.appendingPathComponent(path))
    }
}

public final class BlurayDisc {
    public let fileSystem: DiscFileSystem

    public init(fileSystem: DiscFileSystem) {
        self.fileSystem = fileSystem
    }

    /// Every playlist that parses, by number ("01005").
    public func playlists() throws -> [(name: String, playlist: BlurayPlaylist)] {
        try fileSystem.contentsOfDirectory("BDMV/PLAYLIST")
            .filter { $0.uppercased().hasSuffix(".MPLS") }
            .sorted()
            .compactMap { file in
                guard let playlist = try? BlurayPlaylist(data: fileSystem.readAll("BDMV/PLAYLIST/\(file)")) else { return nil }
                return (String(file.prefix(5)), playlist)
            }
    }

    /// The feature: the longest stereo playlist, else the longest of all.
    /// Ties go to the playlist with fewer items (obfuscated discs pad decoy
    /// playlists with many short repeated items), then the lower number.
    public func mainPlaylist(preferStereo: Bool = true) throws -> (name: String, playlist: BlurayPlaylist)? {
        let all = try playlists()
        let pool = preferStereo && all.contains(where: { $0.playlist.isStereo }) ? all.filter(\.playlist.isStereo) : all
        return pool.max { a, b in
            if abs(a.playlist.duration - b.playlist.duration) > 1 { return a.playlist.duration < b.playlist.duration }
            if a.playlist.items.count != b.playlist.items.count { return a.playlist.items.count > b.playlist.items.count }
            return a.name > b.name
        }
    }

    public func clip(_ name: String) throws -> ByteSource {
        try fileSystem.open("BDMV/STREAM/\(name).m2ts")
    }

    /// The clip's CLPI, or nil when it is missing or unreadable (the clip
    /// still plays from its start; it just can't seek).
    public func clipInfo(_ name: String) -> ClipInfo? {
        if let cached = clipInfoCache[name] { return cached }
        let info = try? ClipInfo(data: fileSystem.readAll("BDMV/CLIPINF/\(name).clpi"))
        clipInfoCache[name] = info
        return info
    }

    private var clipInfoCache: [String: ClipInfo?] = [:]

    /// Stereo access units of a playlist, in decode order, timestamps on the
    /// playlist's own timeline (0 = start of the first PlayItem).
    public func stereoReader(for playlist: BlurayPlaylist, stereo: Bool = true) -> PlaylistStereoReader {
        PlaylistStereoReader(disc: self, playlist: playlist, stereo: stereo && playlist.isStereo)
    }

    /// A byte range of a PlayItem's base clip, which carries the audio and
    /// subtitles along with the base view.
    public struct ClipSpan: Sendable, Equatable {
        public var clip: String
        public var start: Int64
        /// Exclusive.
        public var end: Int64
        /// Playlist time of the entry point at `start` (or of the item's IN
        /// time when the clip has no entry points), 90 kHz.
        public var playlistTime: Int64
    }

    /// What to read, clip by clip, to get every stream of a playlist from
    /// `seconds` on: from the entry point at or before it in the first item,
    /// each later item from the entry point at or before its IN time, and
    /// each item up to the first entry point after its OUT time, so the
    /// audio of the item's last pictures is included. Timestamps inside are
    /// the clips' own, so they jump between items.
    public func clipSpans(of playlist: BlurayPlaylist, from seconds: Double = 0) throws -> [ClipSpan] {
        let target = Int64((max(seconds, 0) * 90_000).rounded())
        var spans: [ClipSpan] = []
        var itemStart: Int64 = 0
        for item in playlist.items {
            defer { itemStart += item.outPTS - item.inPTS }
            let itemEnd = itemStart + item.outPTS - item.inPTS
            guard itemEnd > target else { continue }
            let pid = item.videoPID ?? TransportStreamDemuxer.blurayBaseVideoPID
            let size = try clip(item.clip).size
            let clipPTS = item.inPTS + max(target - itemStart, 0)
            let info = clipInfo(item.clip)
            let entry = info?.entryPoint(pid: pid, atOrBefore: clipPTS)
            let after = info?.entryPoint(pid: pid, after: item.outPTS)
            spans.append(ClipSpan(
                clip: item.clip,
                start: entry?.byteOffset ?? 0,
                end: min(after?.byteOffset ?? size, size),
                playlistTime: itemStart + PlaylistStereoReader.offset(entry?.pts ?? item.inPTS, from: item.inPTS)))
        }
        return spans
    }
}

public final class PlaylistStereoReader {
    private let disc: BlurayDisc
    public let playlist: BlurayPlaylist
    private let stereo: Bool
    private var itemIndex = -1
    private var current: StereoStreamReader?
    /// Playlist time at which each PlayItem starts, 90 kHz.
    private let itemStarts: [Int64]
    private var finishedUnpaired = 0
    private var finishedResyncs = 0

    public var unpairedBaseCount: Int { finishedUnpaired + (current?.unpairedBaseCount ?? 0) }
    public var resyncCount: Int { finishedResyncs + (current?.resyncCount ?? 0) }

    /// PIDs of the base clips (audio, subtitles) to collect alongside the
    /// video, from the next item or seek on.
    public var sidePIDs: Set<UInt16> = []
    private var side: [PESPacket] = []

    /// The side PIDs' PES read so far, timestamps on the playlist timeline
    /// like the video's. Muxed somewhat ahead of the pictures they go with.
    public func takeSidePackets() -> [PESPacket] {
        collectSide()
        defer { side.removeAll() }
        return side
    }

    private func collectSide() {
        guard let current, itemIndex >= 0, itemIndex < playlist.items.count else { return }
        let item = playlist.items[itemIndex]
        for var packet in current.takeSidePackets() {
            if let pts = packet.pts { packet.pts = itemStarts[itemIndex] + Self.offset(pts, from: item.inPTS) }
            if let dts = packet.dts { packet.dts = itemStarts[itemIndex] + Self.offset(dts, from: item.inPTS) }
            side.append(packet)
        }
    }

    init(disc: BlurayDisc, playlist: BlurayPlaylist, stereo: Bool) {
        self.disc = disc
        self.playlist = playlist
        self.stereo = stereo
        var start: Int64 = 0
        itemStarts = playlist.items.map { item in
            defer { start += item.outPTS - item.inPTS }
            return start
        }
    }

    /// The next access unit. Each PlayItem starts at the entry point at or
    /// before its IN time and ends at its OUT time, so a picture shortly
    /// before IN (the references the first shown pictures need) comes out
    /// with a timestamp before the item's start, for the caller to drop.
    public func next() throws -> StereoAccessUnit? {
        while true {
            if let current, var unit = try current.next() {
                let item = playlist.items[itemIndex]
                // Every picture shown before OUT is decoded before it.
                if let order = unit.dts ?? unit.pts, Self.offset(order, from: item.outPTS) >= 0 {
                    endItem()
                    continue
                }
                if let pts = unit.pts { unit.pts = itemStarts[itemIndex] + Self.offset(pts, from: item.inPTS) }
                if let dts = unit.dts { unit.dts = itemStarts[itemIndex] + Self.offset(dts, from: item.inPTS) }
                return unit
            }
            if current != nil { endItem() }
            guard itemIndex + 1 < playlist.items.count else { itemIndex = playlist.items.count; return nil }
            try startItem(itemIndex + 1, at: playlist.items[itemIndex + 1].inPTS)
        }
    }

    /// Moves to `seconds` into the playlist. Returns the playlist time, in
    /// 90 kHz, of the entry point reading resumes at: the random access
    /// picture at or before the target. Pictures between it and the target
    /// have to be decoded and dropped.
    @discardableResult
    public func seek(to seconds: Double) throws -> Int64 {
        let target = Int64((max(seconds, 0) * 90_000).rounded())
        guard !playlist.items.isEmpty else { return 0 }
        let index = itemStarts.lastIndex { $0 <= target } ?? 0
        let item = playlist.items[index]
        let clipPTS = min(item.inPTS + target - itemStarts[index], item.outPTS - 1)
        if current != nil { endItem() }
        side.removeAll()
        let entry = try startItem(index, at: clipPTS)
        return itemStarts[index] + Self.offset(entry, from: item.inPTS)
    }

    /// Opens a PlayItem's clips at the entry point at or before `clipPTS`,
    /// returning that entry point's clip PTS.
    @discardableResult
    private func startItem(_ index: Int, at clipPTS: Int64) throws -> Int64 {
        itemIndex = index
        let item = playlist.items[index]
        let basePID = item.videoPID ?? TransportStreamDemuxer.blurayBaseVideoPID
        let dependentPID = item.dependentVideoPID ?? TransportStreamDemuxer.blurayDependentVideoPID
        let dependentClip = stereo ? item.dependentClip : nil
        let reader = StereoStreamReader(
            base: try disc.clip(item.clip),
            dependent: try dependentClip.map { try disc.clip($0) },
            basePID: basePID, dependentPID: dependentPID, sidePIDs: sidePIDs)
        current = reader
        guard let entry = disc.clipInfo(item.clip)?.entryPoint(pid: basePID, atOrBefore: clipPTS) else {
            return item.inPTS
        }
        // The dependent view's entry points sit at the base view's PTS.
        let dependentEntry = dependentClip.flatMap { disc.clipInfo($0)?.entryPoint(pid: dependentPID, atOrBefore: entry.pts) }
        if entry.spn > 0 || (dependentEntry?.spn ?? 0) > 0 {
            reader.seek(baseOffset: entry.byteOffset, dependentOffset: dependentEntry?.byteOffset)
        }
        return entry.pts
    }

    private func endItem() {
        collectSide()
        finishedUnpaired += current?.unpairedBaseCount ?? 0
        finishedResyncs += current?.resyncCount ?? 0
        current = nil
    }

    /// `pts - start` on the 33-bit clock: wraps forward, and a picture just
    /// before IN (a reference decoded ahead of it) comes out slightly negative.
    public static func offset(_ pts: Int64, from start: Int64) -> Int64 {
        let wrapped = (pts - start) & ((1 << 33) - 1)
        return wrapped >= 1 << 32 ? wrapped - (1 << 33) : wrapped
    }
}
