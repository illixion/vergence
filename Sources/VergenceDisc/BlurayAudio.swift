/*
Vergence - a disc's audio as PCM, for players without a lossless decoder

 Blu-ray's TrueHD tracks carry an AC-3 core on the same PID: the PES
 packets alternate between TrueHD access units and complete AC-3 frames,
 which start with the AC-3 sync word. Plain AC-3 and E-AC-3 tracks are just
 those frames. Apple's decoders handle AC-3 and E-AC-3, so this is the
 audio a player can always produce on the device: up to 5.1. The frames
 are decoded by an `AudioFrameDecoder`: VergenceApple's `AC3Decoder` on
 Apple platforms; elsewhere the caller brings one.

 `PlaylistAudioReader` reads any window of a playlist: it starts at the
 video entry point a second ahead (audio is muxed after the pictures at
 the same time, and the EP_map indexes only the video), demuxes the audio
 PID and decodes until the window is full. Output is planar Float32 at
 48 kHz in a fixed 5.1 order, L R C LFE Ls Rs; a stereo or mono track
 leaves the rest silent.

 DTS (the other common disc codec) has no Apple decoder, and LPCM isn't
 handled yet.
 */

import Foundation

/// One AC-3 or E-AC-3 frame's header fields.
public struct AC3FrameHeader: Equatable, Sendable {
    public var isEnhanced: Bool
    public var byteCount: Int
    public var sampleCount: Int
    public var sampleRate: Int
    /// acmod: 1 C, 2 L R, 3 L C R, 4 L R S, 5 L C R S, 6 L R Ls Rs, 7 L C R Ls Rs.
    public var channelMode: Int
    public var hasLFE: Bool
    public var channelCount: Int { [2, 1, 2, 3, 3, 4, 4, 5][channelMode] + (hasLFE ? 1 : 0) }

    /// Words per frame for 48 kHz AC-3, by frmsizecod / 2.
    private static let ac3Words48k = [64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1152, 1280]

    /// The header at the start of `bytes`, or nil when they don't start an
    /// AC-3 or E-AC-3 frame (TrueHD, damage).
    public init?(_ bytes: some RandomAccessCollection<UInt8>) {
        let b = Array(bytes.prefix(8))
        guard b.count >= 8, b[0] == 0x0B, b[1] == 0x77 else { return nil }
        let bsid = Int(b[5] >> 3)
        var reader = BitReader(b)
        if bsid <= 10 {
            let fscod = Int(b[4] >> 6)
            let frmsizecod = Int(b[4] & 0x3F)
            guard fscod == 0, frmsizecod < 38 else { return nil }
            isEnhanced = false
            byteCount = Self.ac3Words48k[frmsizecod / 2] * 2
            sampleCount = 1536
            sampleRate = 48_000
            // syncword 16, crc1 16, fscod 2, frmsizecod 6, bsid 5, bsmod 3
            reader.skip(48)
        } else if bsid == 16 {
            isEnhanced = true
            // syncword 16, strmtyp 2, substreamid 3, frmsiz 11
            reader.skip(16)
            let streamType = reader.read(2)
            reader.skip(3)
            byteCount = (reader.read(11) + 1) * 2
            // Dependent substreams extend an independent one's channels; the
            // independent stream alone is a complete 5.1 or less.
            guard streamType != 1 else { return nil }
            let fscod = reader.read(2)
            guard fscod == 0 else { return nil }
            let blocks = [1, 2, 3, 6][reader.read(2)]
            sampleCount = 256 * blocks
            sampleRate = 48_000
        } else {
            return nil
        }
        channelMode = reader.read(3)
        if !isEnhanced {
            if channelMode & 1 != 0, channelMode != 1 { reader.skip(2) } // cmixlev
            if channelMode & 4 != 0 { reader.skip(2) } // surmixlev
            if channelMode == 2 { reader.skip(2) } // dsurmod
        }
        hasLFE = reader.read(1) == 1
    }
}

private struct BitReader {
    let bytes: [UInt8]
    var position = 0

    init(_ bytes: [UInt8]) { self.bytes = bytes }

    mutating func skip(_ bits: Int) { position += bits }

    mutating func read(_ bits: Int) -> Int {
        var value = 0
        for _ in 0 ..< bits {
            let byte = position / 8
            let bit = byte < bytes.count ? Int(bytes[byte] >> (7 - position % 8)) & 1 : 0
            value = value << 1 | bit
            position += 1
        }
        return value
    }
}

/// Decodes one compressed audio frame to planar Float32 in the 5.1 order
/// L R C LFE Ls Rs (`outputChannels` planes, equal lengths).
public protocol AudioFrameDecoder: AnyObject {
    /// Forgets decoder state, for a jump in the stream.
    func reset()
    /// One frame's samples, or nil when it doesn't decode.
    func decode(_ frame: Data) -> [[Float]]?
}

/// Planes an `AudioFrameDecoder` hands back.
public let audioOutputChannels = 6

/// A playlist's audio track as 5.1 PCM, read by time window.
public final class PlaylistAudioReader {
    public let playlist: BlurayPlaylist
    public let pid: UInt16
    public static let sampleRate = 48_000
    private let disc: BlurayDisc
    private let itemStarts: [Int64]
    private let decoder: any AudioFrameDecoder
    /// Where the last read ended, to continue without seeking: the clip,
    /// its byte offset, and the demuxer's state there.
    private var resume: (item: Int, offset: Int64, demuxer: TransportStreamDemuxer, pending: [PESPacket], nextPTS: Int64)?

    /// - Parameters:
    ///   - pid: an AC-3, E-AC-3 or TrueHD (its AC-3 core) PID.
    ///   - decoder: decodes the track's AC-3 frames.
    public init(disc: BlurayDisc, playlist: BlurayPlaylist, pid: UInt16, decoder: any AudioFrameDecoder) {
        self.decoder = decoder
        self.disc = disc
        self.playlist = playlist
        self.pid = pid
        var start: Int64 = 0
        itemStarts = playlist.items.map { item in
            defer { start += item.outPTS - item.inPTS }
            return start
        }
    }

    /// `frames` samples per channel from playlist time `startPTS` (90 kHz):
    /// 6 planes, L R C LFE Ls Rs. Silence where the track has nothing.
    public func read(fromPTS startPTS: Int64, frames: Int) throws -> [[Float]] {
        var planes = [[Float]](repeating: [Float](repeating: 0, count: frames), count: audioOutputChannels)
        let endPTS = startPTS + Int64(frames) * 90_000 / Int64(Self.sampleRate)
        for index in playlist.items.indices {
            let item = playlist.items[index]
            let itemStart = itemStarts[index], itemEnd = itemStart + item.outPTS - item.inPTS
            guard itemEnd > startPTS, itemStart < endPTS else { continue }
            try readItem(index, from: max(startPTS, itemStart), to: min(endPTS, itemEnd), windowStart: startPTS, into: &planes)
        }
        return planes
    }

    private func readItem(_ index: Int, from: Int64, to: Int64, windowStart: Int64, into planes: inout [[Float]]) throws {
        let item = playlist.items[index]
        let clip = try disc.clip(item.clip)
        let clipFrom = item.inPTS + (from - itemStarts[index])
        var demuxer: TransportStreamDemuxer
        var offset: Int64
        var pending: [PESPacket] = []
        // A read that carries on from the last one keeps going (and keeps
        // the decoder's state); anything else starts at an entry point.
        if let resume, resume.item == index, abs(resume.nextPTS - from) < 9000 {
            demuxer = resume.demuxer
            offset = resume.offset
            pending = resume.pending
        } else {
            let videoPID = item.videoPID ?? TransportStreamDemuxer.blurayBaseVideoPID
            let entry = disc.clipInfo(item.clip)?.entryPoint(pid: videoPID, atOrBefore: clipFrom - 90_000)
            demuxer = TransportStreamDemuxer(pids: [pid])
            offset = entry?.byteOffset ?? 0
            decoder.reset()
        }
        let frames = planes[0].count
        var reachedEnd = false
        while !reachedEnd {
            if pending.isEmpty {
                let chunk = try clip.read(at: offset, count: 1 << 20)
                if chunk.isEmpty { break }
                offset += Int64(chunk.count)
                pending = try demuxer.feed(chunk)
                continue
            }
            while !pending.isEmpty {
                let packet = pending[0]
                guard let pts = packet.pts, let header = AC3FrameHeader(packet.payload) else { pending.removeFirst(); continue }
                let playlistPTS = itemStarts[index] + PlaylistStereoReader.offset(pts, from: item.inPTS)
                let frameEnd = playlistPTS + Int64(header.sampleCount) * 90_000 / Int64(Self.sampleRate)
                // Decode a frame or two ahead of the window for the decoder's overlap.
                guard frameEnd > from - 2 * 2880 else { pending.removeFirst(); continue }
                // The next window's first frame stays pending for it.
                if playlistPTS >= to {
                    reachedEnd = true
                    break
                }
                pending.removeFirst()
                guard let samples = decoder.decode(packet.payload.prefix(header.byteCount)) else { continue }
                // Samples land where their timestamp says, clipped to the item.
                let first = Int(((playlistPTS - windowStart) * Int64(Self.sampleRate)) / 90_000)
                for (k, sample) in samples[0].indices.enumerated() {
                    let position = first + k
                    guard position >= 0, position < frames else { continue }
                    let time = windowStart + Int64(position) * 90_000 / Int64(Self.sampleRate)
                    guard time >= from, time < to else { continue }
                    for channel in 0 ..< audioOutputChannels { planes[channel][position] = samples[channel][sample] }
                }
            }
        }
        resume = reachedEnd ? (index, offset, demuxer, pending, to) : nil
    }
}
