/*
 Vergence - which gain grid a player shows, and how strongly

 A player hands each frame's media time to `grid(at:steppedDown:)` and gets
 back the grid to look its pixels up in and the strength to apply it at
 (Hypnos `docs/plans/analysis-workers.md`, "AI HDR for video, in short"):

 - **The track's own grid for the frame,** matched within half a frame.
   The worker already eased the grids between keyframes within a shot and
   snapped them at cuts, so a player that shows every frame's grid snaps
   where the picture cuts, and nowhere else.
 - **Ramp in** over a second of media time from the first grid after it
   engages and after every seek (a jump of more than a second, or back), so
   highlights don't jump on.
 - **Never off mid-shot.** A frame with no grid (a piece not here yet)
   keeps the last one, at the strength it had; before the first grid after
   a seek the strength is 0, and the picture stays on its HDR path, SDR-
   looking, until the grid arrives and ramps in.
 - **Stepping down** (the host is out of time; depth comes first, then
   interpolation, then this): the eased grid is refreshed only every half
   second (the keyframe spacing), but a grid that jumps from the one shown
   (a cut, where the worker snapped) is still taken at once. HDR itself
   never goes off.

 Thread safe; Foundation only (a Quest's player can share it).
 */

import Foundation

public final class GainGridPlayback: @unchecked Sendable {
    public let lookup: any AnalysisRecordLookup
    public let frameRate: Double
    private let inflate: (Data, Int) throws -> Data
    /// The most the frame's average gain may reach (linear), applied by the
    /// shader: below the window brightness limiter.
    public var cap: Float
    public static let rampSeconds = 1.0
    /// How often a stepped-down player still takes a new grid.
    public static let steppedDownRefreshSeconds = GainGridSchedule.keyframeSeconds
    public static let defaultCap: Float = 2.5
    /// A mean change in log2 gain this large between the shown grid and the
    /// frame's is a cut, which a stepped-down player still snaps to.
    public static let cutChange: Float = 0.2

    private let lock = NSLock()
    private var current: GainGrid?
    private var currentSeconds = -Double.infinity
    private var lastSeconds: Double?
    private var rampStart: Double?
    private var lastSource: String?

    public init(lookup: any AnalysisRecordLookup, frameRate: Double, cap: Float = GainGridPlayback.defaultCap,
                inflate: @escaping (Data, Int) throws -> Data) {
        self.lookup = lookup
        self.frameRate = frameRate
        self.cap = cap
        self.inflate = inflate
    }

    public struct Frame: Sendable {
        public let grid: GainGrid
        /// 0…1, the ramp.
        public let strength: Float
    }

    /// The grid and strength for the frame at `seconds`.
    public func grid(at seconds: Double, steppedDown: Bool = false) -> Frame {
        lock.lock()
        defer { lock.unlock() }
        lookup.noteDemand(at: seconds)
        if let last = lastSeconds, seconds < last - 0.1 || seconds > last + 1 {
            // A seek: ramp in again from the next grid.
            rampStart = nil
        }
        lastSeconds = seconds
        let due = !steppedDown || seconds - currentSeconds >= Self.steppedDownRefreshSeconds || seconds < currentSeconds
        let found = trackGrid(at: seconds)
        let cut = found.flatMap { found in current.map { Self.meanChange($0, found) >= Self.cutChange } } ?? false
        if due || cut, let found {
            current = found
            currentSeconds = seconds
            if rampStart == nil { rampStart = seconds }
            note("the analysis track")
        } else if current == nil {
            note("none yet")
        }
        guard let current, let rampStart else { return Frame(grid: .identity, strength: 0) }
        let strength = Float(min(max((seconds - rampStart) / Self.rampSeconds, 0), 1))
        return Frame(grid: current, strength: strength)
    }

    static func meanChange(_ a: GainGrid, _ b: GainGrid) -> Float {
        zip(a.log2Gain, b.log2Gain).reduce(Float(0)) { $0 + abs($1.0 - $1.1) } / Float(GainGrid.count)
    }

    private func trackGrid(at seconds: Double) -> GainGrid? {
        guard let (index, entry) = lookup.nearestRecord(.gainMap, eye: .mono, to: AnalysisTime(seconds: seconds),
                                                        within: 0.5 / max(frameRate, 1)),
              let record = try? index.record(entry).inflated(using: inflate) else { return nil }
        return GainGrid(record: record)
    }

    private func note(_ source: String) {
        guard source != lastSource else { return }
        lastSource = source
        VergenceLog.pipeline.notice("AI HDR: gain grids from \(source, privacy: .public)")
    }
}
