/*
 Vergence - spatial video metadata

 What Apple's players need besides the two eyes before they treat a file as
 spatial video: the distance between the cameras, the horizontal field of
 view, a disparity adjustment and the projection. A file missing the field
 of view or the projection plays as ordinary 3D in a window at best, so
 `vergence make` always writes all of them, from a preset unless told
 otherwise.

 The values are stored in a `vexu` box in the video's sample description
 (Apple calls the format "Apple HEVC Stereo Video"); VideoToolbox writes it
 from the compression properties of the same names.
 */

import Foundation

/// The camera description written into a spatial video.
public struct SpatialCameraMetadata: Sendable, Equatable, Codable {
    public enum Projection: String, Sendable, Codable, CaseIterable {
        /// An ordinary flat picture.
        case rectilinear
    }

    /// Distance between the two camera centres, in millimetres.
    public var baselineMillimeters: Double
    /// Horizontal field of view of one eye, in degrees.
    public var horizontalFieldOfViewDegrees: Double
    /// Shifts the eyes apart (positive) or together, as a fraction of the
    /// picture's width, -1...1. Players use it to place the picture at a
    /// comfortable depth.
    public var horizontalDisparityAdjustment: Double
    public var projection: Projection

    public init(baselineMillimeters: Double, horizontalFieldOfViewDegrees: Double,
                horizontalDisparityAdjustment: Double = 0, projection: Projection = .rectilinear) {
        self.baselineMillimeters = baselineMillimeters
        self.horizontalFieldOfViewDegrees = horizontalFieldOfViewDegrees
        self.horizontalDisparityAdjustment = horizontalDisparityAdjustment
        self.projection = projection
    }

    /// The baseline as VideoToolbox stores it: micrometres.
    public var baselineMicrometers: UInt32 { UInt32(max(0, (baselineMillimeters * 1000).rounded())) }
    /// The field of view as VideoToolbox stores it: thousandths of a degree.
    public var fieldOfViewThousandthsOfDegree: UInt32 {
        UInt32(max(0, (horizontalFieldOfViewDegrees * 1000).rounded()))
    }
    /// The adjustment as VideoToolbox stores it: -10000...10000.
    public var disparityAdjustmentNormalized: Int32 {
        Int32((min(1, max(-1, horizontalDisparityAdjustment)) * 10_000).rounded())
    }

    /// Problems a player would have with these values, empty when none.
    public var problems: [String] {
        var found: [String] = []
        if !(baselineMillimeters > 0) { found.append("the camera baseline must be above 0 mm") }
        if !(horizontalFieldOfViewDegrees > 0 && horizontalFieldOfViewDegrees < 180) {
            found.append("a flat picture's field of view must be between 0 and 180 degrees")
        }
        if abs(horizontalDisparityAdjustment) > 1 { found.append("the disparity adjustment must be between -1 and 1") }
        return found
    }
}

/// Camera values for kinds of source, so nobody has to look them up.
public enum SpatialPreset: String, Sendable, CaseIterable, Codable {
    /// Stereo footage of unknown origin: a human eye distance and a field of
    /// view that shows the picture at a comfortable size.
    case generic
    /// Recorded on an iPhone with spatial video capture (iPhone 15 Pro and
    /// later): its two rear cameras are about 19 mm apart.
    case iphone
    /// Recorded on Apple Vision Pro: its cameras sit about an eye distance
    /// apart.
    case visionPro = "avp"
    /// A 3D film (a Blu-ray, a cinema release): shot for a screen, so the
    /// baseline is nominal, and a field of view like a good cinema seat.
    case film

    public var metadata: SpatialCameraMetadata {
        switch self {
        case .generic:
            SpatialCameraMetadata(baselineMillimeters: 65, horizontalFieldOfViewDegrees: 90)
        case .iphone:
            // The values `spatial` documents for iPhone 15 Pro-style video.
            SpatialCameraMetadata(baselineMillimeters: 19.24, horizontalFieldOfViewDegrees: 63.4,
                                  horizontalDisparityAdjustment: 0.02)
        case .visionPro:
            // Read from a spatial video an Apple Vision Pro (M5, visionOS
            // 27.2) recorded.
            SpatialCameraMetadata(baselineMillimeters: 63.664, horizontalFieldOfViewDegrees: 71.59,
                                  horizontalDisparityAdjustment: 0.0293)
        case .film:
            SpatialCameraMetadata(baselineMillimeters: 65, horizontalFieldOfViewDegrees: 60)
        }
    }

    public var summary: String {
        switch self {
        case .generic: "unknown stereo footage"
        case .iphone: "iPhone spatial capture"
        case .visionPro: "Apple Vision Pro capture"
        case .film: "3D films"
        }
    }
}
