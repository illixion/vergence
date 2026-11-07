/*
 Vergence - an analysis track read in chunks as it plays

 A film's track is about 4 GB, too much to download before playing and too
 slow to index over a network. A server (Hypnos's Jellyfin plugin,
 `SmoothMotionService`) serves it in chunks instead: chunk n is the track's
 header plus the records whose time falls in [n·chunkSeconds,
 (n+1)·chunkSeconds), so each is a small track of its own (about 4.4 MB for
 10 s of a 1080p film). This keeps the chunks around the playhead: the one
 it is in, `lookAhead` after it, and one behind, fetching the nearest
 missing first, a couple at a time. A seek lands in a chunk that isn't
 there, so the frames pass through un-interpolated until it arrives (a
 fraction of a second on a home network).

 A coordinator (`vergenced`) serves chunks while workers are still making
 them: a chunk can arrive partial, holding the pieces done so far. Its
 records are used at once, and it is fetched again every
 `partialRefetchSeconds` while it stays in the window, until it comes whole.
 `onDemand` hears every playhead the player asks about, so the caller can
 tell the coordinator where it is playing.

 Foundation only, like the rest of VergenceCore: the fetch is the caller's.
 */

import Foundation

/// A chunk as it arrived: a track (header, then records), and whether it is
/// whole or still being made.
public struct StreamingAnalysisChunk: Sendable {
    public var data: Data
    public var isComplete: Bool

    public init(data: Data, isComplete: Bool = true) {
        self.data = data
        self.isComplete = isComplete
    }
}

public final class StreamingAnalysisTrack: AnalysisRecordLookup, @unchecked Sendable {
    /// Chunk `n`'s bytes (a track: header, then records).
    public typealias Fetch = @Sendable (_ chunk: Int) async throws -> Data
    /// Chunk `n`, which may still be partial.
    public typealias ChunkFetch = @Sendable (_ chunk: Int) async throws -> StreamingAnalysisChunk

    public let chunkSeconds: Double
    public let chunkCount: Int
    /// Chunks kept and fetched ahead of the playhead's.
    public let lookAhead: Int
    private let fetch: ChunkFetch
    private let onDemand: (@Sendable (Double) -> Void)?
    private let retrySeconds: TimeInterval
    private let partialRefetchSeconds: TimeInterval
    private let lock = NSLock()
    private var chunks: [Int: AnalysisTrackIndex] = [:]
    /// Partial chunks held, and when each was fetched.
    private var partial: [Int: Date] = [:]
    private var inFlight: Set<Int> = []
    private var failedAt: [Int: Date] = [:]
    private var playhead: Int?
    private var failures = 0
    private var loggedFirst = false
    /// How many fetches run at once.
    private static let concurrentFetches = 2

    /// - Parameter retrySeconds: how long after a failure a chunk is tried again.
    public convenience init(chunkSeconds: Double, chunkCount: Int, lookAhead: Int = 3,
                            retrySeconds: TimeInterval = 5, fetch: @escaping Fetch) {
        self.init(chunkSeconds: chunkSeconds, chunkCount: chunkCount, lookAhead: lookAhead,
                  retrySeconds: retrySeconds) { n in StreamingAnalysisChunk(data: try await fetch(n)) }
    }

    /// - Parameters:
    ///   - retrySeconds: how long after a failure (or a chunk with nothing
    ///     ready, which a coordinator answers 404) a chunk is tried again.
    ///   - partialRefetchSeconds: how often a partial chunk is fetched again.
    ///   - onDemand: called with every playhead the player asks about (on
    ///     the track's timeline), on the player's thread; keep it cheap.
    public init(chunkSeconds: Double, chunkCount: Int, lookAhead: Int = 3, retrySeconds: TimeInterval = 5,
                partialRefetchSeconds: TimeInterval = 1.5, onDemand: (@Sendable (Double) -> Void)? = nil,
                fetchChunk: @escaping ChunkFetch) {
        precondition(chunkSeconds > 0, "chunks have a length")
        self.chunkSeconds = chunkSeconds
        self.chunkCount = chunkCount
        self.lookAhead = lookAhead
        self.retrySeconds = retrySeconds
        self.partialRefetchSeconds = partialRefetchSeconds
        self.onDemand = onDemand
        self.fetch = fetchChunk
    }

    /// The chunks held now, for tests and the log.
    public var heldChunks: [Int] {
        lock.lock()
        defer { lock.unlock() }
        return chunks.keys.sorted()
    }

    /// The held chunks that are still partial.
    public var partialChunks: [Int] {
        lock.lock()
        defer { lock.unlock() }
        return partial.keys.sorted()
    }

    private func chunk(at seconds: Double) -> Int {
        max(0, Int((seconds / chunkSeconds).rounded(.down)))
    }

    // MARK: AnalysisRecordLookup

    public func nearestRecord(_ kind: AnalysisKind, eye: AnalysisEye, to time: AnalysisTime,
                              within tolerance: Double) -> (index: AnalysisTrackIndex, entry: AnalysisTrackIndex.Entry)? {
        let c = chunk(at: time.seconds)
        lock.lock()
        let near = (c - 1 ... c + 1).compactMap { chunks[$0] }
        lock.unlock()
        // A record near a chunk's edge may be in the next or previous one.
        var best: (index: AnalysisTrackIndex, entry: AnalysisTrackIndex.Entry)?
        for index in near {
            guard let entry = index.nearest(kind, eye: eye, to: time, within: tolerance) else { continue }
            if best.map({ abs(entry.time.seconds - time.seconds) < abs($0.entry.time.seconds - time.seconds) }) ?? true {
                best = (index, entry)
            }
        }
        return best
    }

    public func noteDemand(at seconds: Double) {
        onDemand?(seconds)
        let c = chunk(at: seconds)
        lock.lock()
        if playhead != c {
            playhead = c
            // Only the window around the playhead stays.
            chunks = chunks.filter { $0.key >= c - 1 && $0.key <= c + lookAhead + 1 }
            partial = partial.filter { chunks[$0.key] != nil }
        }
        lock.unlock()
        schedule()
    }

    /// True once the chunk holding `seconds` is here whole: a miss in a
    /// partial chunk may only mean its piece isn't done yet.
    public func holdsRecords(at seconds: Double) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        let c = chunk(at: seconds)
        return chunks[c] != nil && partial[c] == nil
    }

    /// Fetches the chunk holding `seconds` and the ones after it now, before
    /// the player asks (at open, or after a seek).
    public func prefetch(at seconds: Double) {
        noteDemand(at: seconds)
    }

    /// Waits until the chunk holding `seconds` is here (or `timeout` passes).
    public func waitForChunk(at seconds: Double, timeout: TimeInterval = 10) async -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            // Asking again is what retries a failed chunk, as playback does every frame.
            prefetch(at: seconds)
            if holdsRecords(at: seconds) { return true }
            try? await Task.sleep(nanoseconds: 20_000_000)
        }
        return holdsRecords(at: seconds)
    }

    // MARK: Fetching

    private func schedule() {
        lock.lock()
        guard let c = playhead else { lock.unlock(); return }
        var starting: [Int] = []
        let now = Date()
        for n in c ... c + lookAhead where n < chunkCount {
            guard inFlight.count + starting.count < Self.concurrentFetches else { break }
            if inFlight.contains(n) { continue }
            if chunks[n] != nil {
                // A partial chunk is asked for again once in a while; a whole one never.
                guard let fetched = partial[n], now.timeIntervalSince(fetched) >= partialRefetchSeconds else { continue }
            }
            if let failed = failedAt[n], now.timeIntervalSince(failed) < retrySeconds { continue }
            starting.append(n)
        }
        inFlight.formUnion(starting)
        lock.unlock()
        for n in starting {
            Task.detached(priority: .userInitiated) { [weak self, fetch] in
                let result: Result<(AnalysisTrackIndex, Bool), Error>
                do {
                    let chunk = try await fetch(n)
                    result = .success((try AnalysisTrackIndex(chunk.data), chunk.isComplete))
                } catch {
                    result = .failure(error)
                }
                self?.finished(n, result)
            }
        }
    }

    private func finished(_ n: Int, _ result: Result<(AnalysisTrackIndex, Bool), Error>) {
        lock.lock()
        inFlight.remove(n)
        var logFirst = false, logFailure: (Int, Error)?
        switch result {
        case .success(let (index, isComplete)):
            failedAt[n] = nil
            if let c = playhead, n >= c - 1, n <= c + lookAhead + 1 {
                chunks[n] = index
                partial[n] = isComplete ? nil : Date()
            }
            if !loggedFirst { loggedFirst = true; logFirst = true }
        case .failure(let error):
            failedAt[n] = Date()
            // A partial chunk already held stays; it is tried again after the retry.
            failures += 1
            if failures == 1 || failures % 10 == 0 { logFailure = (failures, error) }
        }
        lock.unlock()
        if logFirst {
            VergenceLog.pipeline.info("Smooth Motion track: first chunk (\(n, privacy: .public) of \(self.chunkCount, privacy: .public)) here")
        }
        if let (count, error) = logFailure {
            VergenceLog.pipeline.error(
                "Smooth Motion track: chunk \(n, privacy: .public) failed (\(count, privacy: .public) so far): \(String(describing: type(of: error)), privacy: .public)")
        }
        schedule()
    }
}
