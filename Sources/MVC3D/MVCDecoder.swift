/*
 mvc3d - stereo H.264 (MVC) decoding through edge264

 Feeds stereo access units to edge264 NAL by NAL and hands back each decoded
 picture pair. Frames come out in display order; edge264 does not carry
 timestamps, so each output frame takes the smallest PTS still waiting, which
 holds for frame-coded streams (one picture pair per access unit), the only
 kind edge264 decodes and the only kind 3D Blu-ray uses.

 With worker threads edge264 parses a NAL and returns before the slice is
 decoded, so the bytes must outlive the call: every access unit is copied
 into a buffer that edge264 releases through its unref callback.
 */

import CEdge264
import Foundation

/// One plane of a decoded picture. Valid only inside the frame callback.
public struct PlaneView {
    public var base: UnsafePointer<UInt8>
    public var width: Int
    public var height: Int
    public var stride: Int

    public func row(_ y: Int) -> UnsafeBufferPointer<UInt8> {
        UnsafeBufferPointer(start: base + y * stride, count: width)
    }
}

/// A decoded picture pair, 8-bit 4:2:0. Valid only inside the frame
/// callback; copy out what must outlive it.
public struct StereoFrame {
    public var pts: Int64?
    /// Y, Cb, Cr of the base view: the left eye on Blu-ray.
    public var left: [PlaneView]
    /// Y, Cb, Cr of the dependent view, or nil for a mono picture.
    public var right: [PlaneView]?
    public var width: Int { left[0].width }
    public var height: Int { left[0].height }
}

public final class MVCDecoder {
    public enum Error: Swift.Error, Equatable {
        case allocationFailed
        case unsupportedStream
        /// edge264 found an invalid bitstream. Decoding goes on; the
        /// picture may show artefacts.
        case corruptStream
    }

    private var decoder: OpaquePointer?
    private var pendingPTS: [Int64] = []
    public private(set) var corruptNALCount = 0
    public private(set) var unsupportedNALCount = 0

    /// - Parameter threads: worker threads; 0 decodes on the calling thread,
    ///   -1 uses every core.
    public init(threads: Int = 0) throws {
        guard let decoder = edge264_alloc(Int32(threads), nil, nil, 0, nil, nil, nil) else { throw Error.allocationFailed }
        self.decoder = decoder
    }

    deinit {
        edge264_free(&decoder)
    }

    /// Decodes one access unit and calls `output` for every frame that
    /// became ready.
    public func decode(_ unit: StereoAccessUnit, output: (StereoFrame) throws -> Void) throws {
        if let pts = unit.pts { insertPending(pts) }
        let buffer = NALBuffer(unit.data)
        let start = buffer.base
        let end = start + buffer.count
        var nal = edge264_find_start_code(start, end, 0) ?? end
        while nal < end {
            nal += 3
            let next = edge264_find_start_code(nal, end, 0) ?? end
            // A 4-byte start code is a 3-byte one after a zero byte.
            var nalEnd = next
            while nalEnd > nal, nalEnd[-1] == 0 { nalEnd -= 1 }
            if nalEnd > nal { try submit(nal, nalEnd, buffer: buffer, output: output) }
            nal = next
        }
        try drainFrames(output)
    }

    /// End of stream: outputs every frame still held for reordering.
    public func finish(output: (StereoFrame) throws -> Void) throws {
        var dummy: UInt8 = 0
        while true {
            let ret = withUnsafePointer(to: &dummy) { edge264_decode_NAL(decoder, $0, $0, nil, nil) }
            try drainFrames(output)
            if ret == ENODATA || ret == EINVAL { break }
            if ret != 0 && ret != ENOBUFS { break }
        }
        pendingPTS.removeAll()
    }

    /// Forgets all state, for a seek. The next unit must start a GOP.
    public func reset() {
        edge264_flush(decoder)
        pendingPTS.removeAll()
    }

    private func submit(_ nal: UnsafePointer<UInt8>, _ end: UnsafePointer<UInt8>, buffer: NALBuffer,
                        output: (StereoFrame) throws -> Void) throws {
        while true {
            let ref = Unmanaged.passRetained(buffer).toOpaque()
            let ret = edge264_decode_NAL(decoder, nal, end, { _, arg in
                Unmanaged<NALBuffer>.fromOpaque(arg!).release()
            }, ref)
            if ret != 0 { Unmanaged<NALBuffer>.fromOpaque(ref).release() }
            switch ret {
            case 0: return
            case ENOBUFS: try drainFrames(output) // room needed, then the same NAL again
            case ENOTSUP: unsupportedNALCount += 1; return
            case EBADMSG: corruptNALCount += 1; return
            case ENOMEM: throw Error.allocationFailed
            default: return
            }
        }
    }

    private func drainFrames(_ output: (StereoFrame) throws -> Void) throws {
        var frame = Edge264Frame()
        while edge264_get_frame(decoder, &frame, 0) == 0 {
            let pts = pendingPTS.isEmpty ? nil : pendingPTS.removeFirst()
            try output(StereoFrame(pts: pts, left: Self.planes(frame.samples, frame),
                                   right: frame.samples_mvc.0 == nil ? nil : Self.planes(frame.samples_mvc, frame)))
        }
    }

    private static func planes(_ s: (UnsafePointer<UInt8>?, UnsafePointer<UInt8>?, UnsafePointer<UInt8>?),
                               _ f: Edge264Frame) -> [PlaneView] {
        [PlaneView(base: s.0!, width: Int(f.width_Y), height: Int(f.height_Y), stride: Int(f.stride_Y)),
         PlaneView(base: s.1!, width: Int(f.width_C), height: Int(f.height_C), stride: Int(f.stride_C)),
         PlaneView(base: s.2!, width: Int(f.width_C), height: Int(f.height_C), stride: Int(f.stride_C))]
    }

    private func insertPending(_ pts: Int64) {
        // Decode order is nearly display order; insert from the back.
        var i = pendingPTS.endIndex
        while i > pendingPTS.startIndex, pendingPTS[i - 1] > pts { i -= 1 }
        pendingPTS.insert(pts, at: i)
    }
}

/// An access unit's bytes, at a fixed address for as long as edge264 holds
/// a reference. Zero padding on both sides keeps edge264's aligned 16-byte
/// vector reads inside the allocation.
private final class NALBuffer: @unchecked Sendable {
    /// The first byte of the access unit.
    let base: UnsafePointer<UInt8>
    let count: Int
    private let allocation: UnsafeMutablePointer<UInt8>
    private static let padding = 64

    init(_ data: Data) {
        count = data.count
        allocation = .allocate(capacity: count + 2 * Self.padding)
        allocation.initialize(repeating: 0, count: count + 2 * Self.padding)
        data.copyBytes(to: allocation + Self.padding, count: count)
        base = UnsafePointer(allocation + Self.padding)
    }

    deinit {
        allocation.deallocate()
    }
}
