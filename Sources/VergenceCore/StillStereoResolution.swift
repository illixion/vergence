import Foundation

/// Additional working-set bound for a one-shot stereo conversion. The app
/// first applies its normal dynamic-resolution/global/per-image policy.
public enum StillStereoResolution {
    public static let maximumLongSide = 4096
    public static let defaultWorkingBytes = 256 * 1024 * 1024

    /// Reserve 40 bytes per source pixel: decoded colour, half-float source
    /// and eyes, 10-bit presentation pair, plus depth/scratch headroom. Codec
    /// internals are outside this estimate (notably some full-decode JXL paths).
    public static func longSide(width: Int, height: Int, requested: Double,
                                workingBytes: Int = defaultWorkingBytes) -> Int {
        guard width > 0, height > 0, requested.isFinite, requested > 0, workingBytes > 0 else { return 1 }
        let native = Double(max(width, height))
        let aspectArea = Double(width) / native * (Double(height) / native)
        let budgetSide = sqrt(Double(workingBytes) / (40 * aspectArea))
        return max(1, Int(min(native, requested, Double(maximumLongSide), budgetSide).rounded(.down)))
    }
}
