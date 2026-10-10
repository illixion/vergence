/*
 Vergence - joining the two views of a Blu-ray 3D video into one MVC stream

 A 3D Blu-ray carries its video as two elementary streams: the base view
 (plain H.264, PID 0x1011, what every 2D player shows) and the dependent view
 (MVC NAL units only, PID 0x1012), each access unit in its own PES. The two
 halves of one stereo access unit share a PTS. An MVC decoder wants them as a
 single Annex B stream, base view first, dependent view right after it.

 The dependent view opens each access unit with NAL type 24, a Blu-ray
 specific "dependent delimiter" outside the H.264 spec. It is dropped.
 */

import Foundation

/// A complete stereo access unit, base view then dependent view, in Annex B.
public struct StereoAccessUnit: Sendable {
    public var pts: Int64?
    public var dts: Int64?
    /// Annex B bytes with 4-byte start codes.
    public var data: Data
    /// False when no dependent view arrived for this PTS: the decoder still
    /// produces the base picture, shown to both eyes.
    public var hasDependentView: Bool
}

public struct StereoAccessUnitPairer {
    /// Base-view access units waiting for their dependent halves, in
    /// decode order.
    private var pendingBase: [PESPacket] = []
    private var pendingDependent: [Int64: Data] = [:]
    /// Dependent units whose base already left without them. Kept briefly
    /// in case of reordering, then dropped.
    private let maxPending = 32

    public private(set) var unpairedBaseCount = 0

    public init() {}

    public mutating func addBase(_ pes: PESPacket) -> [StereoAccessUnit] {
        pendingBase.append(pes)
        return drain(force: false)
    }

    public mutating func addDependent(_ pes: PESPacket) -> [StereoAccessUnit] {
        guard let pts = pes.pts else { return [] }
        pendingDependent[pts] = Self.stripDependentDelimiters(pes.payload)
        if pendingDependent.count > maxPending * 4 {
            // Only possible if the base stream stopped: drop the oldest.
            let floor = pendingDependent.keys.sorted()[pendingDependent.count - maxPending]
            pendingDependent = pendingDependent.filter { $0.key >= floor }
        }
        return drain(force: false)
    }

    /// End of stream: emits what is left, unpaired.
    public mutating func finish() -> [StereoAccessUnit] {
        drain(force: true)
    }

    public mutating func reset() {
        pendingBase.removeAll()
        pendingDependent.removeAll()
    }

    /// True while the caller should read more of the dependent stream before
    /// more base, which keeps the two readers in step.
    public var wantsDependent: Bool {
        !pendingBase.isEmpty && pendingBase.first.flatMap(\.pts).map { pendingDependent[$0] == nil } ?? false
    }

    private mutating func drain(force: Bool) -> [StereoAccessUnit] {
        var out: [StereoAccessUnit] = []
        while let head = pendingBase.first {
            if let pts = head.pts, let dependent = pendingDependent.removeValue(forKey: pts) {
                out.append(Self.join(base: head, dependent: dependent))
            } else if force || pendingBase.count > maxPending || head.pts == nil {
                unpairedBaseCount += 1
                out.append(Self.join(base: head, dependent: nil))
            } else {
                break
            }
            pendingBase.removeFirst()
        }
        return out
    }

    private static func join(base: PESPacket, dependent: Data?) -> StereoAccessUnit {
        var data = Data(capacity: base.payload.count + (dependent?.count ?? 0) + 8)
        data.append(base.payload)
        if let dependent { data.append(dependent) }
        return StereoAccessUnit(pts: base.pts, dts: base.dts, data: data, hasDependentView: dependent != nil)
    }

    /// The dependent payload without NAL 24, every start code made 4 bytes.
    static func stripDependentDelimiters(_ payload: Data) -> Data {
        var out = Data(capacity: payload.count)
        for nal in AnnexB.nalUnits(in: payload) where nal.type != 24 {
            out.append(contentsOf: [0, 0, 0, 1])
            out.append(payload[nal.range])
        }
        return out
    }
}

public enum AnnexB {
    public struct NALUnit: Sendable {
        /// The NAL bytes, header included, start code excluded.
        public var range: Range<Data.Index>
        public var type: UInt8
    }

    /// Splits Annex B bytes at their 00 00 01 start codes. Trailing zero
    /// bytes before the next start code belong to it, not to the NAL.
    public static func nalUnits(in data: Data) -> [NALUnit] {
        var units: [NALUnit] = []
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            let b = raw.bindMemory(to: UInt8.self)
            let base = data.startIndex
            var starts: [Int] = []
            var i = 0
            while i + 2 < b.count {
                if b[i + 2] > 1 { i += 3; continue }
                if b[i] == 0, b[i + 1] == 0, b[i + 2] == 1 {
                    starts.append(i + 3)
                    i += 3
                } else {
                    i += 1
                }
            }
            for (k, s) in starts.enumerated() where s < b.count {
                var end = k + 1 < starts.count ? starts[k + 1] - 3 : b.count
                while end > s, b[end - 1] == 0 { end -= 1 }
                units.append(NALUnit(range: (base + s) ..< (base + end), type: b[s] & 0x1F))
            }
        }
        return units
    }
}
