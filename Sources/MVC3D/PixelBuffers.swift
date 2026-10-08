/*
 mvc3d - decoded frames as CoreVideo pixel buffers

 edge264 hands out planar 4:2:0 (I420) in its own memory, valid only during
 the frame callback. Players on Apple platforms want IOSurface-backed
 bi-planar buffers ('420v': luma plane, then interleaved CbCr), so each frame
 is copied out into buffers from a pool: one per eye, or both eyes packed
 into one (top and bottom or side by side), which is the form a player that
 already crops eyes out of a frame-packed video takes.

 Blu-ray video is limited range BT.709; the buffers are tagged so.
 */

#if canImport(CoreVideo)
import CoreVideo
import Foundation

public final class StereoPixelBufferMaker {
    public enum Layout: Equatable, Sendable {
        /// One buffer per eye.
        case separate
        /// Both eyes in one buffer.
        case packed(StereoPacking)
    }

    public enum Error: Swift.Error, Equatable {
        case poolCreationFailed(CVReturn)
        case bufferCreationFailed(CVReturn)
    }

    public let layout: Layout
    private var pool: CVPixelBufferPool?
    private var poolSize = (width: 0, height: 0)

    public init(layout: Layout) {
        self.layout = layout
    }

    /// Copies a frame out. With `.separate`, `right` is nil for a frame
    /// without a second view; packed layouts repeat the left eye then.
    public func makeBuffers(_ frame: StereoFrame) throws -> (left: CVPixelBuffer, right: CVPixelBuffer?) {
        let right = frame.right ?? frame.left
        switch layout {
        case .separate:
            let l = try buffer(width: frame.width, height: frame.height)
            try copy(frame.left, into: l, x: 0, y: 0)
            guard frame.right != nil else { return (l, nil) }
            let r = try buffer(width: frame.width, height: frame.height)
            try copy(right, into: r, x: 0, y: 0)
            return (l, r)
        case .packed(let packing):
            let (w, h) = packing.packedSize(width: frame.width, height: frame.height)
            let b = try buffer(width: w, height: h)
            try copy(frame.left, into: b, x: 0, y: 0)
            switch packing {
            case .topAndBottom: try copy(right, into: b, x: 0, y: frame.height)
            case .sideBySide: try copy(right, into: b, x: frame.width, y: 0)
            }
            return (b, nil)
        }
    }

    private func buffer(width: Int, height: Int) throws -> CVPixelBuffer {
        if pool == nil || poolSize != (width, height) {
            let attributes: [CFString: Any] = [
                kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
                kCVPixelBufferWidthKey: width,
                kCVPixelBufferHeightKey: height,
                kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary,
                kCVPixelBufferMetalCompatibilityKey: true,
            ]
            var created: CVPixelBufferPool?
            let status = CVPixelBufferPoolCreate(nil, [kCVPixelBufferPoolMinimumBufferCountKey: 4] as CFDictionary,
                                                 attributes as CFDictionary, &created)
            guard status == kCVReturnSuccess, let created else { throw Error.poolCreationFailed(status) }
            pool = created
            poolSize = (width, height)
        }
        var out: CVPixelBuffer?
        let status = CVPixelBufferPoolCreatePixelBuffer(nil, pool!, &out)
        guard status == kCVReturnSuccess, let out else { throw Error.bufferCreationFailed(status) }
        CVBufferSetAttachment(out, kCVImageBufferColorPrimariesKey, kCVImageBufferColorPrimaries_ITU_R_709_2, .shouldPropagate)
        CVBufferSetAttachment(out, kCVImageBufferTransferFunctionKey, kCVImageBufferTransferFunction_ITU_R_709_2, .shouldPropagate)
        CVBufferSetAttachment(out, kCVImageBufferYCbCrMatrixKey, kCVImageBufferYCbCrMatrix_ITU_R_709_2, .shouldPropagate)
        return out
    }

    /// Writes one eye's I420 planes at (x, y) of a '420v' buffer.
    private func copy(_ planes: [PlaneView], into buffer: CVPixelBuffer, x: Int, y: Int) throws {
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        let luma = planes[0], cb = planes[1], cr = planes[2]

        let yBase = CVPixelBufferGetBaseAddressOfPlane(buffer, 0)!.assumingMemoryBound(to: UInt8.self)
        let yStride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 0)
        for row in 0 ..< luma.height {
            (yBase + (y + row) * yStride + x).update(from: luma.base + row * luma.stride, count: luma.width)
        }

        let cBase = CVPixelBufferGetBaseAddressOfPlane(buffer, 1)!.assumingMemoryBound(to: UInt8.self)
        let cStride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 1)
        for row in 0 ..< cb.height {
            let out = cBase + (y / 2 + row) * cStride + x // x/2 pairs of 2 bytes
            let u = cb.base + row * cb.stride, v = cr.base + row * cr.stride
            for i in 0 ..< cb.width {
                out[2 * i] = u[i]
                out[2 * i + 1] = v[i]
            }
        }
    }
}
#endif
