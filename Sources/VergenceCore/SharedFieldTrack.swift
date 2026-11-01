/*
 Vergence - the player's half of Smooth Motion for true stereo: finding a
 frame pair's fields

 The producer (`SharedFieldAnalysis`, `vergence analyze`) writes a `flowPair`
 record per real frame pair and a right-eye `disparity` record per real
 frame. To make the in-between pair, the player needs three of them: the
 pair's flow and mask, and the disparity at each end. `SharedFields` is those
 three decoded; a `SharedFieldProvider` finds them by the player's frame times,
 from a file, a server's chunks (`StreamingAnalysisTrack`), or a paired
 Mac's live channel later.

 Foundation only: a Quest or a server reads tracks the same way.
 */

import Foundation

/// What the in-between pair of two real stereo frames is made from, in the
/// units the producer stores (`SharedFieldAnalysis`): pixels of each field's
/// own grid.
public struct SharedFields: Sendable {
    /// F0→1 (x, y), F1→0 (x, y), then frame 0's blend weight at ½, planar.
    public var flow: [Float]
    public var flowWidth: Int
    public var flowHeight: Int
    /// Right→left disparity of the first and second real frame, x only.
    public var disparity0: [Float]
    public var disparity1: [Float]
    public var disparityWidth: Int
    public var disparityHeight: Int

    public init(flow: [Float], flowWidth: Int, flowHeight: Int, disparity0: [Float], disparity1: [Float],
                disparityWidth: Int, disparityHeight: Int) {
        precondition(flow.count == 5 * flowWidth * flowHeight, "flow is 5 channels")
        precondition(disparity0.count == disparityWidth * disparityHeight && disparity1.count == disparity0.count,
                     "disparity is 1 channel")
        self.flow = flow
        self.flowWidth = flowWidth
        self.flowHeight = flowHeight
        self.disparity0 = disparity0
        self.disparity1 = disparity1
        self.disparityWidth = disparityWidth
        self.disparityHeight = disparityHeight
    }
}

/// Finds the fields for a pair of real frames by their presentation times on
/// the player's timeline, in seconds. Called on the interpolator's work queue,
/// one call at a time.
public protocol SharedFieldProvider: AnyObject, Sendable {
    func fields(from start: Double, to end: Double) -> SharedFields?
    /// The source's frame rate as the fields were made at, when known. A
    /// player whose own measure is rough (HLS reads ~24 for 23.976) can state
    /// the display rate from this.
    var frameRate: Double? { get }
}

extension SharedFieldProvider {
    public var frameRate: Double? { nil }
}

/// Where a provider finds its records: a whole track in memory
/// (`AnalysisTrackIndex`), or the chunks of one fetched as it plays
/// (`StreamingAnalysisTrack`).
public protocol AnalysisRecordLookup: Sendable {
    /// The record of `kind` and `eye` nearest `time`, within `tolerance`
    /// seconds, and the index holding it.
    func nearestRecord(_ kind: AnalysisKind, eye: AnalysisEye, to time: AnalysisTime,
                       within tolerance: Double) -> (index: AnalysisTrackIndex, entry: AnalysisTrackIndex.Entry)?
    /// The player is about to ask for `seconds` (on the track's timeline).
    func noteDemand(at seconds: Double)
    /// Whether the records around `seconds` are at hand, so a miss there
    /// means the track has none (not that they haven't arrived).
    func holdsRecords(at seconds: Double) -> Bool
}

extension AnalysisRecordLookup {
    public func noteDemand(at seconds: Double) {}
    public func holdsRecords(at seconds: Double) -> Bool { true }
}

extension AnalysisTrackIndex: AnalysisRecordLookup {
    public func nearestRecord(_ kind: AnalysisKind, eye: AnalysisEye, to time: AnalysisTime,
                              within tolerance: Double) -> (index: AnalysisTrackIndex, entry: Entry)? {
        nearest(kind, eye: eye, to: time, within: tolerance).map { (self, $0) }
    }
}

/// Fields from an analysis track: a `.vgat` file, mapped, or one streamed in
/// chunks (`StreamingAnalysisTrack`).
///
/// Frames are matched to records by time, within a quarter of the pair's
/// spacing, so a track made at the source's own times (90 kHz for a disc)
/// matches whatever rounding the player's clock adds. `timeOffset` is added
/// to the player's times first, for a player whose timeline starts somewhere
/// other than the source's.
public final class AnalysisTrackFieldProvider: SharedFieldProvider, @unchecked Sendable {
    public let lookup: any AnalysisRecordLookup
    public let timeOffset: Double
    private let inflate: (Data, Int) throws -> Data
    private let lock = NSLock()
    /// The last disparity decoded: a pair's second frame is the next pair's first.
    private var lastDisparity: (time: AnalysisTime, values: [Float], width: Int, height: Int)?
    public private(set) var misses = 0
    public let frameRate: Double?

    /// - Parameter inflate: raw DEFLATE, as `AnalysisRecord.inflated(using:)` takes it.
    public convenience init(index: AnalysisTrackIndex, timeOffset: Double = 0,
                            inflate: @escaping (Data, Int) throws -> Data) {
        self.init(lookup: index, frameRate: Self.frameRate(of: index), timeOffset: timeOffset, inflate: inflate)
    }

    /// - Parameter frameRate: the rate the track was made at, when its producer says so.
    public init(lookup: any AnalysisRecordLookup, frameRate: Double?, timeOffset: Double = 0,
                inflate: @escaping (Data, Int) throws -> Data) {
        self.lookup = lookup
        self.timeOffset = timeOffset
        self.inflate = inflate
        self.frameRate = frameRate
    }

    /// The track's first pairs' mean spacing, as a rate: each pair's own
    /// spacing is rounded to the timescale (3753 or 3754 at 90 kHz for
    /// 23.976), which the span averages out. Gaps (no pair) don't count.
    static func frameRate(of index: AnalysisTrackIndex) -> Double? {
        var total = 0.0, count = 0
        var previousEnd: AnalysisTime?
        for entry in index.entries where entry.kind == .flowPair {
            let spacing = entry.endTime.seconds - entry.time.seconds
            // Only back-to-back pairs: a cut's missing pair would read as one long spacing.
            if spacing > 0, previousEnd == nil || previousEnd == entry.time {
                total += spacing
                count += 1
            }
            previousEnd = entry.endTime
            if count == 240 { break }
        }
        return count > 0 ? Double(count) / total : nil
    }

    public func fields(from start: Double, to end: Double) -> SharedFields? {
        lock.lock()
        defer { lock.unlock() }
        lookup.noteDemand(at: start + timeOffset)
        guard let fields = lookUp(start + timeOffset, end + timeOffset) else {
            misses += 1
            if (misses == 1 || misses % 240 == 0), lookup.holdsRecords(at: start + timeOffset) {
                reportMiss(start + timeOffset, end + timeOffset)
            }
            return nil
        }
        return fields
    }

    private func lookUp(_ start: Double, _ end: Double) -> SharedFields? {
        guard end > start else { return nil }
        let tolerance = (end - start) / 4
        let t0 = AnalysisTime(seconds: start), t1 = AnalysisTime(seconds: end)
        guard let (index, pairEntry) = lookup.nearestRecord(.flowPair, eye: .mono, to: t0, within: tolerance),
              abs(pairEntry.endTime.seconds - end) <= tolerance,
              let pair = try? index.record(pairEntry).inflated(using: inflate),
              pair.channels == 5,
              let d0 = disparity(at: t0, within: tolerance),
              let d1 = disparity(at: t1, within: tolerance),
              d0.width == d1.width, d0.height == d1.height else { return nil }
        return SharedFields(flow: pair.values(), flowWidth: pair.width, flowHeight: pair.height,
                            disparity0: d0.values, disparity1: d1.values,
                            disparityWidth: d0.width, disparityHeight: d0.height)
    }

    /// Says how far the nearest pair is, so a player whose clock starts
    /// somewhere other than the track's shows it in the first miss's line
    /// (`timeOffset` is then that distance).
    private func reportMiss(_ start: Double, _ end: Double) {
        let nearest = lookup.nearestRecord(.flowPair, eye: .mono, to: AnalysisTime(seconds: start), within: 5)
        let offset = nearest.map { ($0.entry.time.seconds - start) * 1000 }
        VergenceLog.pipeline.notice(
            "Smooth Motion track: no fields for \(start, format: .fixed(precision: 3), privacy: .public)-\(end, format: .fixed(precision: 3), privacy: .public) s (miss \(self.misses, privacy: .public)); nearest pair \(offset.map { String(format: "%+.1f ms away", $0) } ?? "over 5 s away", privacy: .public)")
    }

    private func disparity(at time: AnalysisTime, within tolerance: Double)
        -> (values: [Float], width: Int, height: Int)? {
        guard let (index, entry) = lookup.nearestRecord(.disparity, eye: .right, to: time, within: tolerance) else { return nil }
        if let last = lastDisparity, last.time == entry.time { return (last.values, last.width, last.height) }
        guard let record = try? index.record(entry).inflated(using: inflate), record.channels == 1 else { return nil }
        let values = record.values()
        lastDisparity = (entry.time, values, record.width, record.height)
        return (values, record.width, record.height)
    }
}
