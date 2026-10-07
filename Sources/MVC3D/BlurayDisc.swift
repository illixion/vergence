/*
 mvc3d - a Blu-ray disc: its playlists and clips

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

    /// Stereo access units of a playlist, in decode order, timestamps on the
    /// playlist's own timeline (0 = start of the first PlayItem).
    public func stereoReader(for playlist: BlurayPlaylist, stereo: Bool = true) -> PlaylistStereoReader {
        PlaylistStereoReader(disc: self, playlist: playlist, stereo: stereo && playlist.isStereo)
    }
}

public final class PlaylistStereoReader {
    private let disc: BlurayDisc
    public let playlist: BlurayPlaylist
    private let stereo: Bool
    private var itemIndex = -1
    private var current: StereoStreamReader?
    /// Playlist time at which the current PlayItem starts, 90 kHz.
    private var itemStart: Int64 = 0
    public private(set) var unpairedBaseCount = 0

    init(disc: BlurayDisc, playlist: BlurayPlaylist, stereo: Bool) {
        self.disc = disc
        self.playlist = playlist
        self.stereo = stereo
    }

    public func next() throws -> StereoAccessUnit? {
        while true {
            if let current, var unit = try current.next() {
                let item = playlist.items[itemIndex]
                // PTS wraps at 2^33; a clip's timeline starts at its IN time.
                if let pts = unit.pts { unit.pts = itemStart + Self.offset(pts, from: item.inPTS) }
                if let dts = unit.dts { unit.dts = itemStart + Self.offset(dts, from: item.inPTS) }
                return unit
            }
            if let current { unpairedBaseCount += current.unpairedBaseCount }
            if itemIndex >= 0 { itemStart += playlist.items[itemIndex].outPTS - playlist.items[itemIndex].inPTS }
            itemIndex += 1
            guard itemIndex < playlist.items.count else { current = nil; return nil }
            let item = playlist.items[itemIndex]
            current = StereoStreamReader(
                base: try disc.clip(item.clip),
                dependent: stereo ? try item.dependentClip.map { try disc.clip($0) } : nil,
                basePID: item.videoPID ?? TransportStreamDemuxer.blurayBaseVideoPID,
                dependentPID: item.dependentVideoPID ?? TransportStreamDemuxer.blurayDependentVideoPID)
        }
    }

    /// `pts - start` on the 33-bit clock: wraps forward, and a picture just
    /// before IN (a reference decoded ahead of it) comes out slightly negative.
    static func offset(_ pts: Int64, from start: Int64) -> Int64 {
        let wrapped = (pts - start) & ((1 << 33) - 1)
        return wrapped >= 1 << 32 ? wrapped - (1 << 33) : wrapped
    }
}
