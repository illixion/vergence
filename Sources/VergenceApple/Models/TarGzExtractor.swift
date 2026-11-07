/*
 Vergence - unpacks a .tar.gz into a folder, streaming

 Some depth models are published as a gzipped tar of their .mlpackage (a
 folder), and an app can't run `tar`. This reads the archive in pieces,
 inflates it with Apple's Compression (raw DEFLATE after the gzip header) and
 writes each regular file as its tar entry arrives, so a 600 MB model never
 sits in memory. ustar names (with their prefix) only; pax headers and macOS's
 `._` resource-fork entries are skipped; any path leaving the destination is
 refused.
 */

#if canImport(Compression)
import Compression
import Foundation

public enum TarGzExtractor {
    public enum Failure: Error, LocalizedError {
        case notGzip
        case corrupt(String)
        case unsafePath(String)

        public var errorDescription: String? {
            switch self {
            case .notGzip: "the archive isn't gzip"
            case .corrupt(let detail): "the archive is damaged (\(detail))"
            case .unsafePath(let path): "the archive holds a path outside its folder (\(path))"
            }
        }
    }

    /// Unpacks `archive` into `destination` (created if needed). Returns the
    /// top-level names it wrote.
    @discardableResult
    public static func extract(_ archive: URL, into destination: URL) throws -> Set<String> {
        let input = try FileHandle(forReadingFrom: archive)
        defer { try? input.close() }
        var header = try input.read(upToCount: 10) ?? Data()
        guard header.count == 10, header[0] == 0x1F, header[1] == 0x8B, header[2] == 8 else { throw Failure.notGzip }
        let flags = header[3]
        // Optional gzip fields before the DEFLATE stream.
        var pending = Data()
        func readExactly(_ count: Int) throws -> Data {
            var out = Data()
            while out.count < count {
                if !pending.isEmpty {
                    let take = min(count - out.count, pending.count)
                    out.append(pending.prefix(take))
                    pending.removeFirst(take)
                    continue
                }
                guard let more = try input.read(upToCount: max(count - out.count, 65536)), !more.isEmpty else {
                    throw Failure.corrupt("gzip header")
                }
                pending.append(more)
            }
            return out
        }
        func skipZeroTerminated() throws {
            while try readExactly(1)[0] != 0 {}
        }
        if flags & 0x04 != 0 {
            let length = try readExactly(2)
            _ = try readExactly(Int(length[0]) | Int(length[1]) << 8)
        }
        if flags & 0x08 != 0 { try skipZeroTerminated() }
        if flags & 0x10 != 0 { try skipZeroTerminated() }
        if flags & 0x02 != 0 { _ = try readExactly(2) }
        header.removeAll()

        let tar = TarWriter(destination: destination)
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        let stream = UnsafeMutablePointer<compression_stream>.allocate(capacity: 1)
        defer { stream.deallocate() }
        guard compression_stream_init(stream, COMPRESSION_STREAM_DECODE, COMPRESSION_ZLIB) == COMPRESSION_STATUS_OK else {
            throw Failure.corrupt("inflater")
        }
        defer { compression_stream_destroy(stream) }
        let outSize = 1 << 20
        let output = UnsafeMutablePointer<UInt8>.allocate(capacity: outSize)
        defer { output.deallocate() }
        var finished = false
        while !finished {
            var chunk = pending
            pending = Data()
            if chunk.isEmpty { chunk = try input.read(upToCount: 1 << 20) ?? Data() }
            let last = chunk.isEmpty
            try chunk.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                stream.pointee.src_ptr = raw.bindMemory(to: UInt8.self).baseAddress ?? UnsafePointer(output)
                stream.pointee.src_size = raw.count
                repeat {
                    stream.pointee.dst_ptr = output
                    stream.pointee.dst_size = outSize
                    let status = compression_stream_process(stream, last ? Int32(COMPRESSION_STREAM_FINALIZE.rawValue) : 0)
                    guard status != COMPRESSION_STATUS_ERROR else { throw Failure.corrupt("deflate") }
                    let produced = outSize - stream.pointee.dst_size
                    if produced > 0 { try tar.consume(UnsafeBufferPointer(start: output, count: produced)) }
                    if status == COMPRESSION_STATUS_END { finished = true; break }
                    // The inflater hands its output over in blocks: call again while it still has
                    // input or still gives something; stop for more input once it has neither.
                    if produced == 0 && (stream.pointee.src_size == 0 || last) { break }
                } while true
            }
            if last && !finished { throw Failure.corrupt("truncated") }
        }
        try tar.finish()
        return tar.topLevel
    }
}

/// A tar stream's state: 512-byte headers, then each file's bytes padded to 512.
private final class TarWriter {
    let destination: URL
    var topLevel: Set<String> = []
    private var header = Data()
    private var file: FileHandle?
    private var remaining = 0
    private var padding = 0
    private var ended = false

    init(destination: URL) { self.destination = destination }

    func consume(_ bytes: UnsafeBufferPointer<UInt8>) throws {
        var i = 0
        while i < bytes.count, !ended {
            if remaining > 0 {
                let take = min(remaining, bytes.count - i)
                try file?.write(contentsOf: UnsafeRawBufferPointer(rebasing: UnsafeRawBufferPointer(bytes)[i ..< i + take]))
                remaining -= take
                i += take
                if remaining == 0 { try file?.close(); file = nil }
                continue
            }
            if padding > 0 {
                let take = min(padding, bytes.count - i)
                padding -= take
                i += take
                continue
            }
            let take = min(512 - header.count, bytes.count - i)
            header.append(contentsOf: UnsafeBufferPointer(rebasing: bytes[i ..< i + take]))
            i += take
            if header.count == 512 {
                try begin(header)
                header.removeAll(keepingCapacity: true)
            }
        }
    }

    private func begin(_ block: Data) throws {
        let bytes = [UInt8](block)
        if bytes.allSatisfy({ $0 == 0 }) { ended = true; return }
        func text(_ range: Range<Int>) -> String {
            let slice = bytes[range].prefix { $0 != 0 }
            return String(decoding: slice, as: UTF8.self)
        }
        let sizeText = text(124 ..< 136).trimmingCharacters(in: .whitespaces)
        guard let size = Int(sizeText.isEmpty ? "0" : sizeText, radix: 8) else { throw TarGzExtractor.Failure.corrupt("size") }
        let type = bytes[156]
        var name = text(0 ..< 100)
        if text(257 ..< 262) == "ustar" {
            let prefix = text(345 ..< 500)
            if !prefix.isEmpty { name = prefix + "/" + name }
        }
        remaining = 0
        padding = (512 - size % 512) % 512
        let skip = type == UInt8(ascii: "x") || type == UInt8(ascii: "g")
            || name.split(separator: "/").contains { $0.hasPrefix("._") } || name.hasPrefix("._")
        let components = name.split(separator: "/").map(String.init)
        guard !components.contains(".."), !name.hasPrefix("/") else { throw TarGzExtractor.Failure.unsafePath(name) }
        let target = destination.appendingPathComponent(name)
        if !skip, let first = components.first, first != "." { topLevel.insert(first) }
        if skip || (type != 0 && type != UInt8(ascii: "0") && type != UInt8(ascii: "5")) {
            // Skipped or another kind of entry (a link): pass over its bytes.
            remaining = 0
            padding += size
            return
        }
        if type == UInt8(ascii: "5") {
            try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
            return
        }
        try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: target.path, contents: nil)
        if size > 0 {
            file = try FileHandle(forWritingTo: target)
            remaining = size
        }
    }

    func finish() throws {
        try file?.close()
        file = nil
        guard remaining == 0 else { throw TarGzExtractor.Failure.corrupt("a file was cut short") }
    }
}
#endif
