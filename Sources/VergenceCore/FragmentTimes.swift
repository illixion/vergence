/*
 Vergence - setting a fragmented-MP4 segment's times

 A segment's place on the timeline is each track fragment's base decode
 time (tfdt). An HLS writer sets it from its own session, and shifts the
 whole session when something starts before zero (an AAC track's priming),
 so two runs that start at different points disagree about where the same
 frame plays. `retime` moves a segment so that one track's first sample
 presents at a time the caller knows (the source's own), and every other
 track by the same amount, which keeps the writer's own placing of the
 tracks against each other.
 */

import Foundation

public enum FragmentTimes {
    public struct Failure: Error, CustomStringConvertible {
        public let description: String
    }

    /// Track id → timescale, from an init segment's moov (tkhd + mdhd).
    public static func timescales(initSegment data: Data) -> [UInt32: UInt32] {
        var result: [UInt32: UInt32] = [:]
        for (type, body) in boxes(data, data.startIndex ..< data.endIndex) where type == "moov" {
            for (type, trak) in boxes(data, body) where type == "trak" {
                var id: UInt32?, scale: UInt32?
                for (type, box) in boxes(data, trak) {
                    if type == "tkhd" { id = u32(data, box.lowerBound + (data[box.lowerBound] == 1 ? 20 : 12)) }
                    if type == "mdia" {
                        for (type, mdhd) in boxes(data, box) where type == "mdhd" {
                            scale = u32(data, mdhd.lowerBound + (data[mdhd.lowerBound] == 1 ? 20 : 12))
                        }
                    }
                }
                if let id, let scale { result[id] = scale }
            }
        }
        return result
    }

    /// Where each track's first sample in the segment presents now, in its timescale.
    public static func firstPresentation(_ data: Data) -> [UInt32: Int64] {
        var result: [UInt32: Int64] = [:]
        for fragment in fragments(data) where result[fragment.track] == nil {
            result[fragment.track] = Int64(fragment.decodeTime) + Int64(fragment.firstCompositionOffset)
        }
        return result
    }

    /// Shifts the segment so that `track`'s first sample presents at
    /// `presentation` (in its timescale), and every other track in
    /// `timescales` by the same time, or by `othersBy` seconds when given.
    /// Returns the shift, in seconds, that `track` got.
    ///
    /// A writer's later segments drift from the frames' own times by a tick
    /// or so (it times them by durations), while its audio stays contiguous:
    /// a caller fixing a run's video segment by segment keeps the audio on
    /// the shift its first segment got, or every join would move it a sample.
    @discardableResult
    public static func retime(_ data: inout Data, track: UInt32, to presentation: Int64, timescales: [UInt32: UInt32],
                              othersBy: Double? = nil) throws -> Double {
        guard let current = firstPresentation(data)[track], let scale = timescales[track], scale > 0 else {
            throw Failure(description: "the segment has no samples of track \(track)")
        }
        let own = Double(presentation - current) / Double(scale)
        let shift = othersBy ?? own
        for fragment in fragments(data) {
            guard let own = timescales[fragment.track] else { continue }
            let delta = fragment.track == track ? presentation - current : Int64((shift * Double(own)).rounded())
            let value = Int64(fragment.decodeTime) + delta
            guard value >= 0 else { throw Failure(description: "track \(fragment.track) would start before zero") }
            if fragment.tfdtVersion == 1 {
                put64(&data, fragment.tfdtValueOffset, UInt64(value))
            } else {
                guard value <= Int64(UInt32.max) else { throw Failure(description: "track \(fragment.track)'s time needs a 64-bit tfdt") }
                put32(&data, fragment.tfdtValueOffset, UInt32(value))
            }
        }
        return own
    }

    /// Numbers the segment's movie fragments (mfhd) after `segment`, so they
    /// count up through the stream whichever run made each segment: a run
    /// numbers its own from 1, and AVFoundation stops reading at a number
    /// that doesn't follow the last. An HLS writer makes one fragment per
    /// segment, so segment n's is n + 1.
    public static func renumber(_ data: inout Data, segment: Int) {
        var index = 0
        for (type, moof) in boxes(data, data.startIndex ..< data.endIndex) where type == "moof" {
            for (type, mfhd) in boxes(data, moof) where type == "mfhd" {
                put32(&data, mfhd.lowerBound + 4, UInt32(truncatingIfNeeded: segment + index + 1))
                index += 1
            }
        }
    }

    // MARK: Parsing

    struct Fragment {
        var track: UInt32
        var decodeTime: UInt64
        var tfdtVersion: UInt8
        var tfdtValueOffset: Int
        var firstCompositionOffset: Int32
    }

    static func fragments(_ data: Data) -> [Fragment] {
        var result: [Fragment] = []
        for (type, moof) in boxes(data, data.startIndex ..< data.endIndex) where type == "moof" {
            for (type, traf) in boxes(data, moof) where type == "traf" {
                var track: UInt32?, tfdt: (UInt8, Int, UInt64)?, offset: Int32 = 0, sawRun = false
                for (type, box) in boxes(data, traf) {
                    switch type {
                    case "tfhd": track = u32(data, box.lowerBound + 4)
                    case "tfdt":
                        let version = data[box.lowerBound]
                        let at = box.lowerBound + 4
                        tfdt = (version, at, version == 1 ? u64(data, at) : UInt64(u32(data, at)))
                    case "trun" where !sawRun:
                        sawRun = true
                        let flags = u32(data, box.lowerBound) & 0xFFFFFF
                        guard flags & 0x800 != 0, u32(data, box.lowerBound + 4) > 0 else { break }
                        var at = box.lowerBound + 8
                        if flags & 0x1 != 0 { at += 4 }
                        if flags & 0x4 != 0 { at += 4 }
                        if flags & 0x100 != 0 { at += 4 }
                        if flags & 0x200 != 0 { at += 4 }
                        if flags & 0x400 != 0 { at += 4 }
                        offset = Int32(bitPattern: u32(data, at))
                    default: break
                    }
                }
                if let track, let tfdt {
                    result.append(Fragment(track: track, decodeTime: tfdt.2, tfdtVersion: tfdt.0, tfdtValueOffset: tfdt.1,
                                           firstCompositionOffset: offset))
                }
            }
        }
        return result
    }

    /// The child boxes in `range`: type, and the range of each one's body.
    static func boxes(_ data: Data, _ range: Range<Int>) -> [(String, Range<Int>)] {
        var result: [(String, Range<Int>)] = []
        var at = range.lowerBound
        while at + 8 <= range.upperBound {
            var size = Int(u32(data, at)), header = 8
            if size == 1, at + 16 <= range.upperBound { size = Int(u64(data, at + 8)); header = 16 }
            if size == 0 { size = range.upperBound - at }
            guard size >= header, at + size <= range.upperBound else { break }
            let type = String(decoding: data[at + 4 ..< at + 8], as: UTF8.self)
            result.append((type, at + header ..< at + size))
            at += size
        }
        return result
    }

    static func u32(_ d: Data, _ i: Int) -> UInt32 {
        (UInt32(d[i]) << 24) | (UInt32(d[i + 1]) << 16) | (UInt32(d[i + 2]) << 8) | UInt32(d[i + 3])
    }

    static func u64(_ d: Data, _ i: Int) -> UInt64 { (UInt64(u32(d, i)) << 32) | UInt64(u32(d, i + 4)) }

    static func put32(_ d: inout Data, _ i: Int, _ v: UInt32) {
        for k in 0 ..< 4 { d[i + k] = UInt8((v >> (24 - 8 * UInt32(k))) & 0xFF) }
    }

    static func put64(_ d: inout Data, _ i: Int, _ v: UInt64) {
        put32(&d, i, UInt32(v >> 32))
        put32(&d, i + 4, UInt32(v & 0xFFFF_FFFF))
    }
}
