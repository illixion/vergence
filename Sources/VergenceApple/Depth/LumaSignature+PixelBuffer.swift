#if canImport(CoreVideo)
import CoreVideo

extension LumaSignature {
    /// Luma samples on the grid from a decoded BGRA frame, on the 0-255
    /// scale; nil for another format or a frame smaller than the grid.
    /// Shared by the converter's static-frame skip and real-time depth's
    /// held-map check, so both read the picture the same way.
    static func sample(_ frame: CVPixelBuffer) -> [Float]? {
        guard CVPixelBufferGetPixelFormatType(frame) == kCVPixelFormatType_32BGRA,
              CVPixelBufferLockBaseAddress(frame, .readOnly) == kCVReturnSuccess else { return nil }
        defer { CVPixelBufferUnlockBaseAddress(frame, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(frame) else { return nil }
        let width = CVPixelBufferGetWidth(frame)
        let height = CVPixelBufferGetHeight(frame)
        guard width >= gridSize, height >= gridSize else { return nil }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(frame)
        let ptr = base.assumingMemoryBound(to: UInt8.self)

        var signature = [Float](repeating: 0, count: gridSize * gridSize)
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
