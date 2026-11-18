/*
 Vergence - VR projections

 How one eye of a VR video or photo maps onto the viewer's surroundings:
 flat (an ordinary picture, shown on a screen), equirectangular (360° or
 180° of longitude by 180° of latitude: VR180, VR360) or fisheye (a circle
 or an ellipse covering a field of view, as cameras with fisheye lenses and
 the MKX/RF52/VRCA formats record it). Packing the two eyes into a frame is
 `PackedStereoLayout`'s job; this is what is inside one eye.

 A fisheye eye is an ellipse in the eye's own picture (0...1 both ways,
 origin top left), so the two equal circles of an over-under file and the
 two tall ovals of a full side-by-side one are the same model: the ovals
 are circles squeezed by the packing or by an anamorphic lens. The lens is
 equidistant (the distance from the centre is proportional to the angle
 from straight ahead), which is how these formats are made.

 Pure geometry, no platform gate: the sphere mesh (`VRSphereMesh`) and the
 detection (`VRFormatDetector`) are unit-tested on Linux, and the Quest port
 can use them as they are.
 */

import Foundation

/// An ellipse in one eye's picture, in that picture's normalized
/// coordinates (0...1, origin top left).
public struct EyeEllipse: Hashable, Sendable, Codable {
    public var centerX: Double
    public var centerY: Double
    public var radiusX: Double
    public var radiusY: Double

    public init(centerX: Double = 0.5, centerY: Double = 0.5, radiusX: Double = 0.5, radiusY: Double = 0.5) {
        self.centerX = centerX
        self.centerY = centerY
        self.radiusX = radiusX
        self.radiusY = radiusY
    }

    /// The ellipse touching all four edges of the eye: what an eye that is
    /// all fisheye picture, with no black border, looks like.
    public static let inscribed = EyeEllipse()
}

public enum VRProjection: Hashable, Sendable, Codable {
    /// An ordinary picture, shown on a screen.
    case flat
    /// `horizontalDegrees` of longitude (360 or 180) by 180° of latitude.
    case equirectangular(horizontalDegrees: Double)
    /// An equidistant fisheye covering `fieldOfView` degrees across the
    /// ellipse's horizontal diameter.
    case fisheye(fieldOfView: Double, ellipse: EyeEllipse)

    public static let vr180: VRProjection = .equirectangular(horizontalDegrees: 180)
    public static let vr360: VRProjection = .equirectangular(horizontalDegrees: 360)
    public static func fisheye(_ fieldOfView: Double) -> VRProjection {
        .fisheye(fieldOfView: fieldOfView, ellipse: .inscribed)
    }

    /// Shown around the viewer rather than on a screen.
    public var isImmersive: Bool {
        if case .flat = self { return false }
        return true
    }

    /// How far it reaches to each side of straight ahead, in degrees.
    public var halfWidthDegrees: Double {
        switch self {
        case .flat: 0
        case .equirectangular(let degrees): min(max(degrees, 1), 360) / 2
        case .fisheye(let fieldOfView, _): min(max(fieldOfView, 1), 360) / 2
        }
    }

    /// A short name for menus and badges: "180°", "360°", "Fisheye 190°".
    public var label: String {
        switch self {
        case .flat: "Flat"
        case .equirectangular(let degrees): "\(Int(degrees.rounded()))°"
        case .fisheye(let fieldOfView, _): "Fisheye \(Int(fieldOfView.rounded()))°"
        }
    }

    /// This projection with the fisheye ellipse replaced, when it is one.
    public func withEllipse(_ ellipse: EyeEllipse) -> VRProjection {
        guard case .fisheye(let fieldOfView, _) = self else { return self }
        return .fisheye(fieldOfView: fieldOfView, ellipse: ellipse)
    }

    /// Where the direction `direction` (viewer at the origin looking down
    /// -Z, +Y up, +X right; need not be unit length) lands in one eye's
    /// picture, or nil when this projection doesn't cover it.
    public func eyeCoordinate(of direction: SIMD3<Double>) -> SIMD2<Double>? {
        let length = (direction * direction).sum().squareRoot()
        guard length > 0 else { return nil }
        let d = direction / length
        switch self {
        case .flat:
            return nil
        case .equirectangular:
            let half = halfWidthDegrees * .pi / 180
            // Straight up or down has no longitude (and atan2(0, -0) is π).
            let longitude = abs(d.x) < 1e-12 && abs(d.z) < 1e-12 ? 0 : atan2(d.x, -d.z)
            let latitude = asin(max(-1, min(1, d.y)))
            guard abs(longitude) <= half + 1e-9 else { return nil }
            return SIMD2((longitude + half) / (2 * half), 0.5 - latitude / .pi)
        case .fisheye(_, let ellipse):
            let half = halfWidthDegrees * .pi / 180
            let theta = acos(max(-1, min(1, -d.z)))
            guard theta <= half + 1e-9 else { return nil }
            let r = theta / half
            let psi = atan2(d.y, d.x)
            return SIMD2(ellipse.centerX + ellipse.radiusX * r * cos(psi),
                         ellipse.centerY - ellipse.radiusY * r * sin(psi))
        }
    }
}

/// Everything about how a VR video or photo is laid out: how the eyes are
/// packed (nil for a mono picture) and what is inside each.
public struct VRFormat: Hashable, Sendable, Codable {
    public var layout: PackedStereoLayout?
    public var projection: VRProjection

    public init(layout: PackedStereoLayout?, projection: VRProjection) {
        self.layout = layout
        self.projection = projection
    }

    public var isStereo: Bool { layout != nil }

    /// The part of the packed frame that holds `eye`:
    /// `packedUV = eyeUV * scale + offset`. The whole frame for a mono
    /// picture.
    public func eyeRegion(_ eye: PackedStereoLayout.Eye, mirrored: Bool = false)
        -> (scale: SIMD2<Float>, offset: SIMD2<Float>) {
        layout?.eyeRegion(eye, mirrored: mirrored) ?? (SIMD2(1, 1), SIMD2(0, 0))
    }
}
