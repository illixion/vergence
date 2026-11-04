#if canImport(CoreVideo)
import CoreVideo
import VergenceCore

extension LumaSignature {
    /// Luma samples on the grid from a decoded BGRA frame, or a half-float
    /// RGBA one (an HDR picture, still PQ/HLG-encoded: its code values are
    /// read on the same 0-255 scale), on the 0-255 scale; nil for another
    /// format or a frame smaller than the grid. Shared by the converter's
    /// static-frame skip and real-time depth's held-map check, so both read
    /// the picture the same way.
    static func sample(_ frame: CVPixelBuffer) -> [Float]? {
        let format = CVPixelBufferGetPixelFormatType(frame)
        guard format == kCVPixelFormatType_32BGRA || format == kCVPixelFormatType_64RGBAHalf,
              CVPixelBufferLockBaseAddress(frame, .readOnly) == kCVReturnSuccess else { return nil }
        defer { CVPixelBufferUnlockBaseAddress(frame, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(frame) else { return nil }
        let width = CVPixelBufferGetWidth(frame)
        let height = CVPixelBufferGetHeight(frame)
        guard width >= gridSize, height >= gridSize else { return nil }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(frame)

        var signature = [Float](repeating: 0, count: gridSize * gridSize)
        if format == kCVPixelFormatType_64RGBAHalf {
            for gy in 0..<gridSize {
                let y = gy * (height - 1) / (gridSize - 1)
                let row = (base + y * bytesPerRow).assumingMemoryBound(to: UInt16.self)
                for gx in 0..<gridSize {
                    let px = row + gx * (width - 1) / (gridSize - 1) * 4 // RGBA
                    let red = HalfFloat.value(from: px[0])
                    let green = HalfFloat.value(from: px[1])
                    let blue = HalfFloat.value(from: px[2])
                    let luma: Float = red + 2 * green + blue
                    signature[gy * gridSize + gx] = min(max(luma, 0), 4) * (255 / 4)
                }
            }
            return signature
        }
        let ptr = base.assumingMemoryBound(to: UInt8.self)
        for gy in 0..<gridSize {
            let y = gy * (height - 1) / (gridSize - 1)
            let row = ptr + y * bytesPerRow
            for gx in 0..<gridSize {
                let x = gx * (width - 1) / (gridSize - 1)
                let px = row + x * 4 // BGRA
                // Integer luma approximation: (B + 2G + R) / 4.
                let luma = UInt16(px[0]) + 2 * UInt16(px[1]) + UInt16(px[2])
                signature[gy * gridSize + gx] = Float(luma) * 0.25
            }
        }
        return signature
    }

}
#endif
