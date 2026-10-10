/*
 Vergence - how many bits a spatial video gets

 Scaled to what is being encoded rather than one number for everything: bits
 per pixel of one eye times its pixels per second, plus the second eye,
 which MV-HEVC predicts from the first and costs well under a second stream.
 `.standard` gives two 1080p24 eyes about 10 Mbps, two 4K30 eyes about
 50 Mbps. Never more than a margin above what the source itself had: extra
 bits can't add detail the source never carried.
 */

import Foundation

public enum SpatialQuality: String, Sendable, CaseIterable, Codable {
    case low, standard, high, max

    /// Bits per pixel per frame for the first eye.
    var bitsPerPixel: Double {
        switch self {
        case .low: 0.07
        case .standard: 0.12
        case .high: 0.18
        case .max: 0.3
        }
    }
}

public enum SpatialBitrate {
    /// What the second eye costs relative to the first.
    static let secondEyeShare = 0.6
    /// How far above the source's own rate the output may go.
    static let sourceMargin = 1.25

    /// Bits per second for two eyes of `eyeWidth`×`eyeHeight` at `frameRate`.
    /// `sourceBitsPerSecond` (the packed source's video rate, when known)
    /// caps it. With `secondEye` false, the size is one ordinary picture
    /// (both eyes packed into it) and nothing is added for a predicted view.
    public static func bitsPerSecond(eyeWidth: Int, eyeHeight: Int, frameRate: Double,
                                     quality: SpatialQuality = .standard, sourceBitsPerSecond: Double? = nil,
                                     secondEye: Bool = true) -> Int {
        let rate = frameRate > 0 ? frameRate : 30
        var bits = Double(eyeWidth * eyeHeight) * rate * quality.bitsPerPixel * (secondEye ? 1 + secondEyeShare : 1)
        if let source = sourceBitsPerSecond, source > 0, quality != .max {
            bits = min(bits, source * sourceMargin)
        }
        // Below this even a small picture breaks up.
        return Int(max(bits, 1_000_000).rounded())
    }
}
