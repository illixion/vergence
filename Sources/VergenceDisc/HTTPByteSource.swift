/*
 Vergence - a remote file read with HTTP range requests

 Lets a disc image be read in place from a media server (Jellyfin's
 `/Items/{id}/Download`, any static file server). Reads are served from a
 cache of fixed-size blocks; reading a block also fetches the next
 `readAhead` blocks in the background, so a sequential reader rides out a
 slow request (Wi-Fi, a server busy with another reader of the same file)
 rather than waiting on it. One block ahead, under a second of a 3D
 Blu-ray, showed as the picture freezing for a beat. A Blu-ray clip pair is
 two such readers (base and dependent view) at different offsets, and the
 UDF reader adds small scattered reads, so the cache holds several blocks
 per reader.

 `read` blocks the calling thread until the bytes arrive: call it off the
 main thread, as the decoder does.
 */

import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

public final class HTTPByteSource: ByteSource, @unchecked Sendable {
    public enum Error: Swift.Error, Equatable {
        /// The server answered without honouring the range (no 206).
        case rangesNotSupported(status: Int)
        case badStatus(Int)
        case missingSize
    }

    public let url: URL
    public private(set) var size: Int64 = 0
    private let headers: [String: String]
    private let session: URLSession
    private let blockSize: Int
    private let capacity: Int
    private let readAhead: Int
    private let timeout: TimeInterval

    private let lock = NSLock()
    private var blocks: [Int64: Data] = [:]
    /// Most recently used last.
    private var recency: [Int64] = []
    private var inFlight: [Int64: DispatchGroup] = [:]
    private var failures: [Int64: Swift.Error] = [:]

    /// Opens `url`, learning its size from a one-byte range request.
    /// - Parameters:
    ///   - headers: sent with every request (an `Authorization` header, say).
    ///   - blockSize: bytes per request.
    ///   - cacheBlocks: blocks kept in memory.
    ///   - readAhead: blocks fetched ahead of a read.
    public init(url: URL, headers: [String: String] = [:], session: URLSession = .shared,
                blockSize: Int = 4 << 20, cacheBlocks: Int = 16, readAhead: Int = 3,
                timeout: TimeInterval = 30) throws {
        self.url = url
        self.headers = headers
        self.session = session
        self.blockSize = blockSize
        self.readAhead = max(readAhead, 1)
        // Room for two sequential readers' read-ahead plus scattered reads.
        capacity = max(cacheBlocks, 2 * (self.readAhead + 1) + 2)
        self.timeout = timeout
        let (_, response) = try fetch(range: 0 ... 0)
        guard let total = Self.totalSize(response) else { throw Error.missingSize }
        size = total
    }

    public func read(at offset: Int64, count: Int) throws -> Data {
        guard offset >= 0, offset < size, count > 0 else { return Data() }
        let end = min(offset + Int64(count), size)
        var result = Data(capacity: Int(end - offset))
        var position = offset
        while position < end {
            let index = position / Int64(blockSize)
            let block = try self.block(index)
            let start = Int(position - index * Int64(blockSize))
            let take = min(block.count - start, Int(end - position))
            guard take > 0 else { break }
            result.append(block[block.startIndex + start ..< block.startIndex + start + take])
            position += Int64(take)
        }
        // A reader moving forward wants the next blocks soon.
        let last = (end - 1) / Int64(blockSize)
        for ahead in 1 ... Int64(readAhead) { prefetch(last + ahead) }
        return result
    }

    /// Drops cached blocks, for a seek far away (they would only age out).
    public func discardCache() {
        lock.withLock {
            blocks.removeAll()
            recency.removeAll()
        }
    }

    // MARK: Blocks

    private func block(_ index: Int64) throws -> Data {
        var loaded = false
        while true {
            lock.lock()
            if let data = blocks[index] {
                touch(index)
                lock.unlock()
                return data
            }
            // A failed prefetch gets one more try; a failed load of our own throws.
            if let error = failures.removeValue(forKey: index), loaded {
                lock.unlock()
                throw error
            }
            if let group = inFlight[index] {
                lock.unlock()
                group.wait()
                continue
            }
            let group = DispatchGroup()
            group.enter()
            inFlight[index] = group
            lock.unlock()
            load(index, group: group)
            loaded = true
        }
    }

    private func prefetch(_ index: Int64) {
        guard index * Int64(blockSize) < size else { return }
        lock.lock()
        guard blocks[index] == nil, inFlight[index] == nil else { lock.unlock(); return }
        let group = DispatchGroup()
        group.enter()
        inFlight[index] = group
        lock.unlock()
        DispatchQueue.global(qos: .userInitiated).async { self.load(index, group: group) }
    }

    /// Fetches a block (retrying twice) and files the result for waiters.
    private func load(_ index: Int64, group: DispatchGroup) {
        let start = index * Int64(blockSize)
        let last = min(start + Int64(blockSize), size) - 1
        var outcome: Result<Data, Swift.Error> = .failure(Error.missingSize)
        for attempt in 0 ..< 3 {
            do {
                let (data, _) = try fetch(range: start ... last)
                guard Int64(data.count) == last - start + 1 else { throw URLError(.cannotDecodeContentData) }
                outcome = .success(data)
                break
            } catch let error as Error {
                outcome = .failure(error)
                break // the server's answer won't change
            } catch {
                outcome = .failure(error)
                if attempt < 2 { Thread.sleep(forTimeInterval: 0.2 * Double(attempt + 1)) }
            }
        }
        lock.withLock {
            inFlight[index] = nil
            switch outcome {
            case .success(let data):
                blocks[index] = data
                touch(index)
                while blocks.count > capacity, let oldest = recency.first {
                    recency.removeFirst()
                    blocks[oldest] = nil
                }
            case .failure(let error):
                failures[index] = error
            }
        }
        group.leave()
    }

    /// Call with the lock held.
    private func touch(_ index: Int64) {
        if let i = recency.lastIndex(of: index) { recency.remove(at: i) }
        recency.append(index)
    }

    // MARK: HTTP

    private func fetch(range: ClosedRange<Int64>) throws -> (Data, HTTPURLResponse) {
        var request = URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: timeout)
        for (name, value) in headers { request.setValue(value, forHTTPHeaderField: name) }
        request.setValue("bytes=\(range.lowerBound)-\(range.upperBound)", forHTTPHeaderField: "Range")
        // Compressed bodies would make the byte counts meaningless.
        request.setValue("identity", forHTTPHeaderField: "Accept-Encoding")

        // Written by the completion handler before it signals, read after the wait.
        final class Box: @unchecked Sendable {
            var result: Result<(Data, HTTPURLResponse), Swift.Error> = .failure(URLError(.unknown))
        }
        let box = Box()
        let done = DispatchSemaphore(value: 0)
        let task = session.dataTask(with: request) { data, response, error in
            if let error {
                box.result = .failure(error)
            } else if let response = response as? HTTPURLResponse {
                box.result = .success((data ?? Data(), response))
            } else {
                box.result = .failure(URLError(.badServerResponse))
            }
            done.signal()
        }
        task.resume()
        done.wait()
        let (data, response) = try box.result.get()
        switch response.statusCode {
        case 206: return (data, response)
        case 200: throw Error.rangesNotSupported(status: 200)
        default: throw Error.badStatus(response.statusCode)
        }
    }

    /// The total from `Content-Range: bytes 0-0/12345`.
    static func totalSize(_ response: HTTPURLResponse) -> Int64? {
        guard let value = response.value(forHTTPHeaderField: "Content-Range"),
              let slash = value.lastIndex(of: "/") else { return nil }
        return Int64(value[value.index(after: slash)...].trimmingCharacters(in: .whitespaces))
    }
}
