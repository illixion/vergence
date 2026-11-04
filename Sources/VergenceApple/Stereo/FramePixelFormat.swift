#if canImport(Metal) && canImport(CoreVideo)
import CoreVideo
import Metal

/// The two frame formats the stereo pump and the interpolators work in: BGRA
/// for SDR, and half-float RGBA for HDR, whose values stay PQ or HLG code
/// values (0-1), so 8 bits don't band the picture. A frame keeps its colour
/// tags either way.
public enum FramePixelFormat {
    public static let sdr = kCVPixelFormatType_32BGRA
    public static let hdr = kCVPixelFormatType_64RGBAHalf

    /// True for a format the pump and interpolators take.
    public static func isSupported(_ format: OSType) -> Bool { format == sdr || format == hdr }

    /// The Metal view of a frame of `format`.
    public static func metal(_ format: OSType) -> MTLPixelFormat? {
        switch format {
        case sdr: .bgra8Unorm
        case hdr: .rgba16Float
        default: nil
        }
    }

    /// The frame format for a picture with this transfer function
    /// (`kCVImageBufferTransferFunctionKey`): half float for PQ and HLG.
    public static func forTransfer(_ transfer: CFTypeRef?) -> OSType {
        isHDR(transfer: transfer) ? hdr : sdr
    }

    public static func isHDR(transfer: CFTypeRef?) -> Bool {
        guard let transfer = transfer as? String else { return false }
        return transfer == (kCVImageBufferTransferFunction_SMPTE_ST_2084_PQ as String)
            || transfer == (kCVImageBufferTransferFunction_ITU_R_2100_HLG as String)
    }

    public static func isHDR(_ pixelBuffer: CVPixelBuffer) -> Bool {
        isHDR(transfer: CVBufferCopyAttachment(pixelBuffer, kCVImageBufferTransferFunctionKey, nil))
    }

    /// A Metal-compatible, IOSurface-backed pool of frames of `format`.
    public static func pool(format: OSType, width: Int, height: Int) -> CVPixelBufferPool? {
        let attributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: format,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
        ]
        var pool: CVPixelBufferPool?
        CVPixelBufferPoolCreate(nil, nil, attributes as CFDictionary, &pool)
        return pool
    }

    /// A Metal texture over `buffer` in its own format, with the CVMetalTexture
    /// that keeps its IOSurface alive.
    public static func texture(_ buffer: CVPixelBuffer, cache: CVMetalTextureCache)
        -> (texture: MTLTexture, held: CVMetalTexture)? {
        guard let format = metal(CVPixelBufferGetPixelFormatType(buffer)) else { return nil }
        var cvTexture: CVMetalTexture?
        guard CVMetalTextureCacheCreateTextureFromImage(
            nil, cache, buffer, nil, format,
            CVPixelBufferGetWidth(buffer), CVPixelBufferGetHeight(buffer), 0, &cvTexture) == kCVReturnSuccess,
            let cvTexture, let texture = CVMetalTextureGetTexture(cvTexture) else { return nil }
        return (texture, cvTexture)
    }
}
#endif
