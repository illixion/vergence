/*
 mvc3d - MPEG transport stream demuxer

 Reassembles the PES packets of chosen PIDs from an MPEG-2 transport stream.
 Blu-ray stores streams as BDAV (`.m2ts`): 192-byte packets, a 4-byte
 TP_extra_header (copy permission + arrival timestamp) ahead of each ordinary
 188-byte TS packet. Plain `.ts` is accepted too; the packet size is detected
 from the sync bytes.

 Only what video needs is parsed: no PSI tables (the caller names the PIDs,
 which Blu-ray fixes: 0x1011 base view, 0x1012 dependent view), no scrambling,
 no PCR. A PES is emitted when the next one on its PID starts, or on `flush`.

 Damaged bytes (a sync byte missing) don't stop it: the PES being assembled
 is dropped and reading resumes where sync bytes recur at packet intervals.
 */

import Foundation

/// One elementary-stream packet: for Blu-ray video, exactly one access unit.
public struct PESPacket: Sendable {
    public var pid: UInt16
    /// 90 kHz presentation timestamp, if the header carried one.
    public var pts: Int64?
    /// 90 kHz decoding timestamp; equals `pts` when the header omits it.
    public var dts: Int64?
    public var payload: Data
}

public struct TransportStreamDemuxer {
    public static let blurayBaseVideoPID: UInt16 = 0x1011
    public static let blurayDependentVideoPID: UInt16 = 0x1012

    private let pids: Set<UInt16>
    private var assembling: [UInt16: PESAssembly] = [:]
    /// Bytes carried between `feed` calls when a packet straddles two chunks.
    private var remainder = Data()
    private var packetSize: Int?
    /// Times sync was lost and found again.
    public private(set) var resyncCount = 0
    private var synced = true

    private struct PESAssembly {
        var header: (pts: Int64?, dts: Int64?)
        var payload: Data
    }

    public init(pids: Set<UInt16>) {
        self.pids = pids
    }

    /// Feeds the next chunk of the stream, in order, and returns every PES
    /// it completed.
    public mutating func feed(_ chunk: Data) throws -> [PESPacket] {
        try process(chunk, final: false)
    }

    private mutating func process(_ chunk: Data, final: Bool) throws -> [PESPacket] {
        var buffer = remainder
        buffer.append(chunk)
        var output: [PESPacket] = []
        try buffer.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            let bytes = raw.bindMemory(to: UInt8.self)
            if packetSize == nil { packetSize = Self.detectPacketSize(bytes, packets: final ? 1 : 3) }
            guard let size = packetSize else { return }
            let lead = size - 188
            var offset = 0
            while offset + size <= bytes.count {
                let packet = UnsafeBufferPointer(rebasing: bytes[(offset + lead) ..< (offset + size)])
                if packet[0] != 0x47 {
                    if synced {
                        resyncCount += 1
                        synced = false
                        assembling.removeAll()
                    }
                    // The next spot with a sync byte where two packets in a row put one.
                    var probe = offset + 1
                    while probe + lead + size < bytes.count,
                          bytes[probe + lead] != 0x47 || bytes[probe + lead + size] != 0x47 {
                        probe += 1
                    }
                    if probe + lead + size >= bytes.count {
                        // Not in this chunk: keep its last packet's worth for the next.
                        offset = max(offset + 1, bytes.count - size)
                        break
                    }
                    offset = probe
                    continue
                }
                synced = true
                if let pes = handle(packet) { output.append(pes) }
                offset += size
            }
            remainder = Data(bytes[offset...])
        }
        if packetSize == nil { remainder = buffer }
        return output
    }

    /// Emits the PES still being assembled on each PID (end of stream).
    /// A stream too short to have shown its packet size gets it detected
    /// from a single packet here.
    public mutating func flush() -> [PESPacket] {
        let early = packetSize == nil && !remainder.isEmpty ? ((try? process(Data(), final: true)) ?? []) : []
        let done = assembling.map { PESPacket(pid: $0.key, pts: $0.value.header.pts, dts: $0.value.header.dts, payload: $0.value.payload) }
        assembling.removeAll()
        return early + done.sorted { $0.pid < $1.pid }
    }

    /// Forgets partial packets, for a seek to another position in the file.
    public mutating func reset() {
        assembling.removeAll()
        remainder.removeAll()
    }

    // MARK: Packets

    private mutating func handle(_ p: UnsafeBufferPointer<UInt8>) -> PESPacket? {
        let pid = UInt16(p[1] & 0x1F) << 8 | UInt16(p[2])
        guard pids.contains(pid) else { return nil }
        let payloadUnitStart = p[1] & 0x40 != 0
        let adaptation = (p[3] >> 4) & 0x3
        var start = 4
        if adaptation & 0x2 != 0 { start += 1 + Int(p[4]) }
        guard adaptation & 0x1 != 0, start < 188 else { return nil }
        let body = UnsafeBufferPointer(rebasing: p[start...])

        var finished: PESPacket?
        if payloadUnitStart {
            if let previous = assembling.removeValue(forKey: pid) {
                finished = PESPacket(pid: pid, pts: previous.header.pts, dts: previous.header.dts, payload: previous.payload)
            }
            guard let (header, headerLength) = Self.parsePESHeader(body) else { return finished }
            var payload = Data()
            payload.reserveCapacity(256 * 1024)
            payload.append(contentsOf: UnsafeBufferPointer(rebasing: body[headerLength...]))
            assembling[pid] = PESAssembly(header: header, payload: payload)
        } else {
            assembling[pid]?.payload.append(contentsOf: body)
        }
        return finished
    }

    /// The PES header's timestamps and total length, or nil if this is not
    /// a PES start (missing 00 00 01 prefix).
    static func parsePESHeader(_ b: UnsafeBufferPointer<UInt8>) -> ((pts: Int64?, dts: Int64?), Int)? {
        guard b.count >= 9, b[0] == 0, b[1] == 0, b[2] == 1 else { return nil }
        let flags = b[7]
        let headerLength = 9 + Int(b[8])
        guard headerLength <= b.count else { return nil }
        var pts: Int64?
        var dts: Int64?
        if flags & 0x80 != 0, b.count >= 14 { pts = timestamp(b, at: 9) }
        if flags & 0x40 != 0, b.count >= 19 { dts = timestamp(b, at: 14) }
        return ((pts, dts ?? pts), headerLength)
    }

    private static func timestamp(_ b: UnsafeBufferPointer<UInt8>, at i: Int) -> Int64 {
        Int64(b[i] >> 1 & 0x07) << 30 | Int64(b[i + 1]) << 22 | Int64(b[i + 2] >> 1) << 15
            | Int64(b[i + 3]) << 7 | Int64(b[i + 4] >> 1)
    }

    /// 192 for BDAV, 188 for plain TS: whichever puts a sync byte at the
    /// start of `packets` packets in a row.
    static func detectPacketSize(_ b: UnsafeBufferPointer<UInt8>, packets: Int = 3) -> Int? {
        for size in [192, 188] {
            let lead = size - 188
            guard b.count >= packets * size else { continue }
            if (0 ..< packets).allSatisfy({ b[lead + $0 * size] == 0x47 }) { return size }
        }
        return nil
    }
}
