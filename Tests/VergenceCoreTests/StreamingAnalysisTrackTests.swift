import Foundation
import Testing
@testable import VergenceCore

/// A track served in chunks (header + the records of `chunkSeconds`) gives
/// the same fields as the whole file once the playhead's chunks are here,
/// across chunk edges too; it keeps only the window around the playhead and
/// tries a failed chunk again.
@Suite("Streaming analysis track")
struct StreamingAnalysisTrackTests {
    static let rate = SharedFieldTrackTests.rate

    /// The test track (`frames` frames) cut into chunks the way the server cuts them.
    static func chunks(frames: Int, seconds: Double) throws -> [Data] {
        let header = AnalysisTrackHeader(mediaID: "t", producer: "test", models: [:], sourceWidth: 1920, sourceHeight: 1080)
        let headerBytes = try AnalysisTrack.header(header)
        let whole = try SharedFieldTrackTests.track(frames: frames)
        var out: [Data] = []
        for entry in whole.entries {
            let n = Int((entry.time.seconds / seconds).rounded(.down))
            while out.count <= n { out.append(headerBytes) }
            out[n].append(AnalysisTrack.record(try whole.record(entry)))
        }
        return out
    }

    /// Counts fetches and can fail chunks on demand.
    final class Server: @unchecked Sendable {
        let chunks: [Data]
        private let lock = NSLock()
        private var _requests: [Int] = []
        private var failing: Set<Int> = []
        init(_ chunks: [Data]) { self.chunks = chunks }
        var requests: [Int] { lock.lock(); defer { lock.unlock() }; return _requests }
        func fail(_ n: Int) { lock.lock(); failing.insert(n); lock.unlock() }
        func heal() { lock.lock(); failing.removeAll(); lock.unlock() }
        func fetch(_ n: Int) throws -> Data {
            lock.lock()
            _requests.append(n)
            let fails = failing.contains(n)
            lock.unlock()
            if fails { throw URLError(.timedOut) }
            return chunks[n]
        }
    }

    static func provider(_ track: StreamingAnalysisTrack) -> AnalysisTrackFieldProvider {
        AnalysisTrackFieldProvider(lookup: track, frameRate: rate, inflate: AnalysisTrackTests.identity)
    }

    @Test func pairsMatchTheWholeTrackAcrossChunkEdges() async throws {
        // 48 frames in 0.5 s chunks: an edge every 12 frames.
        let server = Server(try Self.chunks(frames: 48, seconds: 0.5))
        let track = StreamingAnalysisTrack(chunkSeconds: 0.5, chunkCount: server.chunks.count, lookAhead: 4) { try server.fetch($0) }
        let streamed = Self.provider(track)
        let whole = AnalysisTrackFieldProvider(index: try SharedFieldTrackTests.track(frames: 48), inflate: AnalysisTrackTests.identity)
        #expect(await track.waitForChunk(at: 0))
        // Let the look-ahead arrive.
        for _ in 0 ..< 100 where track.heldChunks.count < 4 { try await Task.sleep(nanoseconds: 10_000_000) }
        for k in 0 ..< 46 {
            let a = Double(k) / Self.rate, b = Double(k + 1) / Self.rate
            let s = try #require(streamed.fields(from: a, to: b), "pair \(k)")
            let w = try #require(whole.fields(from: a, to: b))
            #expect(s.flow == w.flow && s.disparity0 == w.disparity0 && s.disparity1 == w.disparity1)
            if k % 6 == 0 {
                // Keep the fetches ahead of the asking.
                for _ in 0 ..< 100 where !track.holdsRecords(at: b + 0.5) && b + 0.5 < 2 {
                    try await Task.sleep(nanoseconds: 10_000_000)
                }
            }
        }
        #expect(streamed.misses == 0)
    }

    @Test func aSeekKeepsOnlyTheWindowAroundThePlayhead() async throws {
        let server = Server(try Self.chunks(frames: 240, seconds: 1))
        let track = StreamingAnalysisTrack(chunkSeconds: 1, chunkCount: server.chunks.count, lookAhead: 2) { try server.fetch($0) }
        #expect(await track.waitForChunk(at: 0.2))
        #expect(await track.waitForChunk(at: 7.5))
        for _ in 0 ..< 100 where track.heldChunks != [7, 8, 9] { try await Task.sleep(nanoseconds: 10_000_000) }
        #expect(track.heldChunks == [7, 8, 9])
        // Nothing between the two places was fetched.
        #expect(!server.requests.contains(4))
        // A pair there is found (frames 180-181 are 7.5 s in).
        #expect(Self.provider(track).fields(from: 180 / Self.rate, to: 181 / Self.rate) != nil)
    }

    @Test func aFailedChunkIsTriedAgain() async throws {
        let server = Server(try Self.chunks(frames: 48, seconds: 1))
        server.fail(0)
        let track = StreamingAnalysisTrack(chunkSeconds: 1, chunkCount: server.chunks.count) { try server.fetch($0) }
        let provider = Self.provider(track)
        #expect(provider.fields(from: 0, to: 1 / Self.rate) == nil)
        try await Task.sleep(nanoseconds: 100_000_000)
        #expect(!track.holdsRecords(at: 0))
        server.heal()
        // Within the retry interval nothing is asked again; after it, the chunk arrives.
        #expect(await track.waitForChunk(at: 0, timeout: 8))
        #expect(provider.fields(from: 0, to: 1 / Self.rate) != nil)
    }

    #if canImport(Darwin)
    /// A real server's chunks (`VERGENCE_TEST_CHUNKS`: a folder of 0.vgat,
    /// 1.vgat, … as `GET SmoothMotion/{id}/Chunks/{n}` served them, plus
    /// track.vgat from `/Track`) give every pair the whole track gives.
    /// A coordinator serves a chunk with only some pieces done: its records
    /// are used at once, it isn't taken for whole, and it is fetched again
    /// until it is.
    @Test func aPartialChunkIsUsedAndFetchedAgainUntilWhole() async throws {
        let whole = try Self.chunks(frames: 48, seconds: 1)
        // The first half second of chunk 0's records, as a partial chunk.
        let header = try AnalysisTrack.header(AnalysisTrackHeader(mediaID: "t", producer: "test", models: [:],
                                                                  sourceWidth: 1920, sourceHeight: 1080))
        let index0 = try AnalysisTrackIndex(whole[0])
        var firstHalf = header
        for entry in index0.entries where entry.time.seconds < 0.5 {
            firstHalf.append(AnalysisTrack.record(try index0.record(entry)))
        }
        final class Flags: @unchecked Sendable {
            private let lock = NSLock()
            private var _done = false, _count = 0
            func finish() { lock.lock(); _done = true; lock.unlock() }
            var done: Bool { lock.lock(); defer { lock.unlock() }; return _done }
            func count() { lock.lock(); _count += 1; lock.unlock() }
            var counted: Int { lock.lock(); defer { lock.unlock() }; return _count }
        }
        let pieces = Flags(), demand = Flags()
        let partialData = firstHalf
        let track = StreamingAnalysisTrack(chunkSeconds: 1, chunkCount: whole.count, lookAhead: 0, retrySeconds: 0.05,
                                           partialRefetchSeconds: 0.05, onDemand: { _ in demand.count() }) { n in
            n == 0 && !pieces.done ? StreamingAnalysisChunk(data: partialData, isComplete: false)
                : StreamingAnalysisChunk(data: whole[n])
        }
        let provider = Self.provider(track)
        track.prefetch(at: 0)
        for _ in 0 ..< 200 where track.heldChunks.isEmpty { try await Task.sleep(nanoseconds: 5_000_000) }
        #expect(track.partialChunks == [0])
        #expect(!track.holdsRecords(at: 0.1))
        #expect(provider.fields(from: 0, to: 1 / Self.rate) != nil, "a partial chunk's records are used")
        #expect(provider.fields(from: 20 / Self.rate, to: 21 / Self.rate) == nil, "its missing piece isn't there yet")
        pieces.finish()
        for _ in 0 ..< 200 where !track.partialChunks.isEmpty {
            track.noteDemand(at: 0.1)
            try await Task.sleep(nanoseconds: 5_000_000)
        }
        #expect(track.partialChunks.isEmpty && track.holdsRecords(at: 0.1))
        #expect(provider.fields(from: 20 / Self.rate, to: 21 / Self.rate) != nil)
        #expect(demand.counted > 0, "the caller hears where the player is")
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["VERGENCE_TEST_CHUNKS"] != nil))
    func aServersChunksGiveTheWholeTracksFields() async throws {
        let folder = URL(fileURLWithPath: try #require(ProcessInfo.processInfo.environment["VERGENCE_TEST_CHUNKS"]))
        let whole = try AnalysisTrackIndex(Data(contentsOf: folder.appending(path: "track.vgat")))
        var chunks: [Data] = []
        while let data = try? Data(contentsOf: folder.appending(path: "\(chunks.count).vgat")) { chunks.append(data) }
        let served = chunks
        let track = StreamingAnalysisTrack(chunkSeconds: 10, chunkCount: served.count, lookAhead: served.count) { served[$0] }
        #expect(await track.waitForChunk(at: 0))
        for _ in 0 ..< 200 where track.heldChunks.count < served.count { try await Task.sleep(nanoseconds: 10_000_000) }
        let inflate: (Data, Int) throws -> Data = { data, size in
            let out = try (data as NSData).decompressed(using: .zlib) as Data
            guard out.count == size else { throw AnalysisTrackError.truncated }
            return out
        }
        let streamed = AnalysisTrackFieldProvider(lookup: track, frameRate: nil, inflate: inflate)
        let reference = AnalysisTrackFieldProvider(index: whole, inflate: inflate)
        var pairs = 0
        for entry in whole.entries where entry.kind == .flowPair {
            let a = entry.time.seconds, b = entry.endTime.seconds
            let s = try #require(streamed.fields(from: a, to: b), "pair at \(a)")
            let w = try #require(reference.fields(from: a, to: b))
            #expect(s.flow == w.flow && s.disparity0 == w.disparity0 && s.disparity1 == w.disparity1)
            pairs += 1
        }
        #expect(pairs > 0 && streamed.misses == 0)
        print("streamed \(pairs) pairs from \(served.count) chunks")
    }
    #endif
}

