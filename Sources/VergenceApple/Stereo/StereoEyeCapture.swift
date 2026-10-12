/*
 Vergence - one frame's eyes, for a debug readout

 A screenshot of the window shows the picture once; what differs between the
 eyes (a warp artefact in one of them, a strip that reads as a different
 angle) needs both, exactly as the warp drew them. A host asks for the next
 frame (`request`), the pump hands over its two BGRA eye buffers, the decoded
 source frame and the depth map it warped with, and the host reads them back
 (`latest`). Nothing is kept unless asked for.
 */

#if canImport(Metal)
import CoreVideo
import Metal
import VergenceCore

public enum StereoEyeCapture {
    public struct Frame: @unchecked Sendable {
        public let left: CVPixelBuffer
        public let right: CVPixelBuffer
        public let source: CVPixelBuffer
        /// The warp's depth map (shot-space or cached units, before the depth
        /// budget), row-major; nil when the frame had none.
        public let depth: (width: Int, height: Int, values: [Float])?
        public let itemSeconds: Double
        public let depthStrength: Float
        public let convergence: Float
    }

    private struct State {
        var requested = false
        var recomputesDepth = false
        var latest: Frame?
        var generation = 0
    }

    private static let state = VergenceLock(State())

    /// Asks the pump to keep the next frame it draws. Returns the generation
    /// to wait for with `latest(after:)`.
    /// `recomputingDepth`: on a paused picture, compute the frame's depth
    /// afresh (as the first frame of a shot) instead of showing the map on
    /// screen, so a changed setting shows on it.
    public static func request(recomputingDepth: Bool = false) -> Int {
        state.withLock {
            $0.requested = true
            $0.recomputesDepth = recomputingDepth
            return $0.generation
        }
    }

    /// The last captured frame, if one arrived after `generation`.
    public static func latest(after generation: Int) -> Frame? {
        state.withLock { $0.generation > generation ? $0.latest : nil }
    }

    public static var latest: Frame? { state.withLock { $0.latest } }

    /// Whether a capture is waiting, without taking it.
    static var isRequested: Bool { state.withLock { $0.requested } }
    static var recomputesDepth: Bool { state.withLock { $0.requested && $0.recomputesDepth } }

    static func takeRequest() -> Bool {
        state.withLock {
            defer { $0.requested = false }
            return $0.requested
        }
    }

    /// Keeps copies: the pump's eye buffers and the decoder's frame come from
    /// small pools, and holding them would starve playback.
    static func deliver(_ frame: Frame) {
        guard let left = copy(frame.left), let right = copy(frame.right), let source = copy(frame.source) else { return }
        let frame = Frame(left: left, right: right, source: source, depth: frame.depth, itemSeconds: frame.itemSeconds,
                          depthStrength: frame.depthStrength, convergence: frame.convergence)
        state.withLock {
            $0.latest = frame
            $0.generation += 1
        }
    }

    /// A plain-memory copy of a single-plane buffer (BGRA here).
    static func copy(_ buffer: CVPixelBuffer) -> CVPixelBuffer? {
        guard !CVPixelBufferIsPlanar(buffer) else { return nil }
        let width = CVPixelBufferGetWidth(buffer), height = CVPixelBufferGetHeight(buffer)
        var out: CVPixelBuffer?
        guard CVPixelBufferCreate(nil, width, height, CVPixelBufferGetPixelFormatType(buffer), nil, &out) == kCVReturnSuccess,
              let out else { return nil }
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        CVPixelBufferLockBaseAddress(out, [])
        defer {
            CVPixelBufferUnlockBaseAddress(out, [])
            CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
        }
        guard let from = CVPixelBufferGetBaseAddress(buffer), let to = CVPixelBufferGetBaseAddress(out) else { return nil }
        let fromRow = CVPixelBufferGetBytesPerRow(buffer), toRow = CVPixelBufferGetBytesPerRow(out)
        for y in 0..<height {
            memcpy(to + y * toRow, from + y * fromRow, min(fromRow, toRow))
        }
        CVBufferPropagateAttachments(buffer, out)
        return out
    }

    /// A readback of the depth texture, read once the command buffer is done.
    struct DepthReadback {
        let buffer: MTLBuffer?
        let width: Int
        let height: Int
        let format: MTLPixelFormat

        func read() -> (width: Int, height: Int, values: [Float])? {
            guard let buffer else { return nil }
            let count = width * height
            switch format {
            case .r16Float:
                // Decoded by hand: Float16 doesn't exist on Intel Macs.
                let ptr = buffer.contents().bindMemory(to: UInt16.self, capacity: count)
                return (width, height, (0..<count).map { Self.halfToFloat(ptr[$0]) })
            case .r32Float:
                let ptr = buffer.contents().bindMemory(to: Float.self, capacity: count)
                return (width, height, Array(UnsafeBufferPointer(start: ptr, count: count)))
            default:
                return nil
            }
        }

        static func halfToFloat(_ h: UInt16) -> Float {
            let sign: UInt32 = UInt32(h & 0x8000) << 16
            let exponent = Int((h >> 10) & 0x1F)
            var mantissa = UInt32(h & 0x3FF)
            switch exponent {
            case 0:
                guard mantissa != 0 else { return Float(bitPattern: sign) }
                // Subnormal: normalise.
                var e = -1
                repeat { e += 1; mantissa <<= 1 } while mantissa & 0x400 == 0
                mantissa &= 0x3FF
                return Float(bitPattern: sign | UInt32(127 - 15 - e) << 23 | mantissa << 13)
            case 31:
                return Float(bitPattern: sign | 0x7F80_0000 | mantissa << 13)
            default:
                return Float(bitPattern: sign | UInt32(exponent - 15 + 127) << 23 | mantissa << 13)
            }
        }
    }

    static func encodeDepthReadback(_ depth: PumpFrameDepth?, device: MTLDevice, commandBuffer: MTLCommandBuffer) -> DepthReadback {
        guard let texture = depth?.texture,
              texture.pixelFormat == .r16Float || texture.pixelFormat == .r32Float else {
            return DepthReadback(buffer: nil, width: 0, height: 0, format: .invalid)
        }
        let bytesPerPixel = texture.pixelFormat == .r16Float ? 2 : 4
        let rowBytes = texture.width * bytesPerPixel
        guard let buffer = device.makeBuffer(length: rowBytes * texture.height, options: .storageModeShared),
              let blit = commandBuffer.makeBlitCommandEncoder() else {
            return DepthReadback(buffer: nil, width: 0, height: 0, format: .invalid)
        }
        blit.copy(from: texture, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                  sourceSize: MTLSize(width: texture.width, height: texture.height, depth: 1),
                  to: buffer, destinationOffset: 0, destinationBytesPerRow: rowBytes,
                  destinationBytesPerImage: rowBytes * texture.height)
        blit.endEncoding()
        return DepthReadback(buffer: buffer, width: texture.width, height: texture.height, format: texture.pixelFormat)
    }
}
#endif
