import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
import Testing
@testable import VergenceDisc

/// Serves `body` for any URL, honouring `Range` the way a static file server does.
final class RangeServer: URLProtocol, @unchecked Sendable {
    struct State {
        var body = Data()
        var honoursRanges = true
        /// Requests to fail with a network error before succeeding again.
        var failNext = 0
        var requests: [String] = []
    }

    nonisolated(unsafe) static var state = State()
    static let lock = NSLock()

    static func session(body: Data, honoursRanges: Bool = true) -> URLSession {
        lock.withLock { state = State(body: body, honoursRanges: honoursRanges) }
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [RangeServer.self]
        return URLSession(configuration: configuration)
    }

    static var requests: [String] { lock.withLock { state.requests } }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func stopLoading() {}

    override func startLoading() {
        let range = request.value(forHTTPHeaderField: "Range") ?? ""
        let (body, honours, fail) = Self.lock.withLock { () -> (Data, Bool, Bool) in
            Self.state.requests.append(range)
            let fail = Self.state.failNext > 0
            if fail { Self.state.failNext -= 1 }
            return (Self.state.body, Self.state.honoursRanges, fail)
        }
        if fail {
            client?.urlProtocol(self, didFailWithError: URLError(.networkConnectionLost))
            return
        }
        var status = 200
        var headers = ["Content-Length": "\(body.count)"]
        var payload = body
        if honours, range.hasPrefix("bytes=") {
            let bounds = range.dropFirst(6).split(separator: "-").compactMap { Int($0) }
            let start = bounds[0], end = min(bounds[1], body.count - 1)
            payload = body.subdata(in: start ..< end + 1)
            status = 206
            headers = ["Content-Range": "bytes \(start)-\(end)/\(body.count)", "Content-Length": "\(payload.count)"]
        }
        let response = HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: "HTTP/1.1", headerFields: headers)!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: payload)
        client?.urlProtocolDidFinishLoading(self)
    }
}

private let url = URL(string: "https://media.example/Items/1/Download")!
private let body = Data((0 ..< 10_000).map { UInt8(truncatingIfNeeded: $0 &* 7 &+ $0 >> 8) })

// The mock's state is global, so these run one at a time.
@Suite(.serialized)
struct HTTPByteSourceTests {
    @Test func readsMatchTheFile() throws {
        let source = try HTTPByteSource(url: url, session: RangeServer.session(body: body), blockSize: 1000, cacheBlocks: 3)
        #expect(source.size == 10_000)
        for (offset, count) in [(0, 10), (995, 10), (1500, 3000), (9990, 100), (0, 10_000)] {
            #expect(try source.read(at: Int64(offset), count: count) == body[offset ..< min(offset + count, body.count)])
        }
        #expect(try source.read(at: 10_000, count: 10).isEmpty)
    }

    @Test func sequentialReadsAreServedFromTheCache() throws {
        let source = try HTTPByteSource(url: url, session: RangeServer.session(body: body), blockSize: 1000, cacheBlocks: 4)
        for offset in stride(from: 0, to: 3000, by: 100) { _ = try source.read(at: Int64(offset), count: 100) }
        // The size probe, then each block once, whether a read or a prefetch got it first (the
        // read-ahead of blocks 3-5, three past the last read, may or may not have started yet).
        let ranges = RangeServer.requests
        #expect(ranges.first == "bytes=0-0")
        #expect(Set(ranges).count == ranges.count)
        #expect(Set(ranges.dropFirst()).isSuperset(of: ["bytes=0-999", "bytes=1000-1999", "bytes=2000-2999"]))
        #expect(Set(ranges.dropFirst()).isSubset(of: (0 ..< 6).map { "bytes=\($0 * 1000)-\($0 * 1000 + 999)" }))
    }

    @Test func aServerIgnoringRangesIsRejected() {
        #expect(throws: HTTPByteSource.Error.rangesNotSupported(status: 200)) {
            try HTTPByteSource(url: url, session: RangeServer.session(body: body, honoursRanges: false))
        }
    }

    @Test func aDroppedConnectionIsRetried() throws {
        let session = RangeServer.session(body: body)
        let source = try HTTPByteSource(url: url, session: session, blockSize: 1000)
        RangeServer.lock.withLock { RangeServer.state.failNext = 2 }
        #expect(try source.read(at: 5000, count: 100) == body[5000 ..< 5100])
    }

    @Test func aDiscImageReadsOverHTTP() throws {
        // The synthetic UDF image from UDFFileSystemTests, served remotely.
        var builder = UDFImageBuilder()
        let contents = Data((0 ..< 5000).map { UInt8($0 % 251) })
        let image = builder.build(fileContents: contents)
        let source = try HTTPByteSource(url: url, session: RangeServer.session(body: image), blockSize: 64 << 10)
        let remote = try UDFFileSystem(image: source)
        #expect(try remote.contentsOfDirectory("BDMV").sorted() == ["index.bdmv", "Ünïcode.bin"])
        #expect(try remote.open("BDMV/index.bdmv").read(at: 0, count: 5000) == contents)
    }
}
