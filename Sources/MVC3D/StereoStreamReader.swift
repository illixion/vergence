/*
 mvc3d - reading stereo access units from Blu-ray clip files

 A 3D clip is two files on disc: `STREAM/<n>.m2ts` (base view, with audio and
 subtitles) and the dependent-view clip it names, typically `<n+1>.m2ts`.
 (The disc's `SSIF/<n>.ssif` is the same data interleaved for optical
 playback; the two separate files are simpler to read.) The reader pulls
 from whichever file is behind, so pairing never buffers more than a few
 access units of either.
 */

import Foundation

/// Random-access bytes: a local file now; an HTTP range reader or a disc
/// image's file later.
public protocol ByteSource: AnyObject {
    var size: Int64 { get }
    /// Reads up to `count` bytes at `offset`; an empty result is the end.
    func read(at offset: Int64, count: Int) throws -> Data
}

/// A local file read with `pread`. (FileHandle's reads return autoreleased
/// NSData on Darwin, which a thread without an autorelease pool, such as a
/// command line tool's main thread, never frees: reading a film that way
/// kept every byte of it in memory.)
public final class FileByteSource: ByteSource {
    private let descriptor: Int32
    public let size: Int64

    public init(url: URL) throws {
        descriptor = open(url.path, O_RDONLY)
        guard descriptor >= 0 else { throw CocoaError(.fileReadNoSuchFile, userInfo: [NSFilePathErrorKey: url.path]) }
        var info = stat()
        fstat(descriptor, &info)
        size = Int64(info.st_size)
    }

    deinit {
        close(descriptor)
    }

    public func read(at offset: Int64, count: Int) throws -> Data {
        var data = Data(count: count)
        let read = data.withUnsafeMutableBytes { pread(descriptor, $0.baseAddress, count, off_t(offset)) }
        guard read >= 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        data.count = read
        return data
    }
}

/// Sequential PES from one PID of one source.
final class PESStream {
    private let source: ByteSource
    private var demuxer: TransportStreamDemuxer
    private var offset: Int64 = 0
    private var queue: [PESPacket] = []
    private var ended = false
    private let chunkSize: Int

    init(source: ByteSource, pid: UInt16, chunkSize: Int = 1 << 20) {
        self.source = source
        self.chunkSize = chunkSize
        demuxer = TransportStreamDemuxer(pids: [pid])
    }

    /// The next PES, or nil at the end of the source.
    func next() throws -> PESPacket? {
        while queue.isEmpty {
            if ended { return nil }
            let chunk = try source.read(at: offset, count: chunkSize)
            if chunk.isEmpty {
                ended = true
                // A final PES cut short by the end of the file is dropped:
                // a truncated slice is worse than none.
                _ = demuxer.flush()
                return nil
            }
            offset += Int64(chunk.count)
            queue = try demuxer.feed(chunk)
        }
        return queue.removeFirst()
    }

    /// Moves to `byteOffset` (rounded down to a packet boundary by the
    /// caller) and forgets anything buffered.
    func seek(to byteOffset: Int64) {
        offset = byteOffset
        queue.removeAll()
        ended = false
        demuxer.reset()
    }
}

public final class StereoStreamReader {
    private let base: PESStream
    private let dependent: PESStream?
    private var pairer = StereoAccessUnitPairer()
    private var ready: [StereoAccessUnit] = []
    private var baseEnded = false
    private var dependentEnded = false

    public var unpairedBaseCount: Int { pairer.unpairedBaseCount }

    /// - Parameters:
    ///   - base: the clip carrying PID 0x1011.
    ///   - dependent: the clip carrying PID 0x1012, or nil to decode the
    ///     base view alone.
    public init(base: ByteSource, dependent: ByteSource?,
                basePID: UInt16 = TransportStreamDemuxer.blurayBaseVideoPID,
                dependentPID: UInt16 = TransportStreamDemuxer.blurayDependentVideoPID) {
        self.base = PESStream(source: base, pid: basePID)
        self.dependent = dependent.map { PESStream(source: $0, pid: dependentPID) }
        dependentEnded = dependent == nil
    }

    /// The next stereo access unit in decode order, or nil at the end.
    public func next() throws -> StereoAccessUnit? {
        while ready.isEmpty {
            if baseEnded && dependentEnded {
                ready = pairer.finish()
                if ready.isEmpty { return nil }
                break
            }
            if !dependentEnded && (pairer.wantsDependent || baseEnded), let dependent {
                if let pes = try dependent.next() { ready += pairer.addDependent(pes) } else { dependentEnded = true }
            } else if !baseEnded {
                if let pes = try base.next() {
                    ready += dependent == nil ? [Self.mono(pes)] : pairer.addBase(pes)
                } else {
                    baseEnded = true
                }
            } else {
                dependentEnded = true
            }
        }
        return ready.removeFirst()
    }

    private static func mono(_ pes: PESPacket) -> StereoAccessUnit {
        StereoAccessUnit(pts: pes.pts, dts: pes.dts, data: pes.payload, hasDependentView: false)
    }
}
