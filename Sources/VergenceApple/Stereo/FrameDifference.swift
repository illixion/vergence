#if canImport(CoreVideo)
import CoreVideo

/// How much a decoded frame differs from another, read on the same luma grid
/// the converter's static-frame skip uses (`LumaSignature`): what finds the
/// frames a pulldown repeats (`RepeatedFrames`).
public enum FrameDifference {
    /// The frame's luma samples; nil for a format other than BGRA or
    /// half-float RGBA, or a frame smaller than the grid.
    public static func signature(_ frame: CVPixelBuffer) -> [Float]? { LumaSignature.sample(frame) }

    /// Mean absolute difference of two signatures, on a 0-255 scale.
    public static func between(_ a: [Float], _ b: [Float]) -> Float? { LumaSignature.meanDelta(a, b) }
}
#endif
