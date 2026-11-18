/*
 Vergence - telling a VR video or photo from its name, tags and picture

 VR files rarely say what they are inside the file (some MP4s carry
 Spherical Video's `sv3d`, some MKVs a Projection element; the container
 readers pass that in as `declared`). The common convention is the name, as
 the VR players DeoVR and HereSphere read it:

     Clip_180_LR.mp4         180°, side by side
     Clip_360_TB.mp4         360°, over under
     Clip_3dh.mp4, _3dv      side by side / over under (no projection said)
     Clip_FISHEYE190_LR.mp4  equidistant fisheye, 190°
     Clip_MKX200.mp4         fisheye, 200° (MKX220, VRCA220: 220°; RF52: 190°)
     Clip_180x180_3dh.mp4    180° side by side

 Stash tags are read the same way, one tag at a time ("VR180", "Fisheye",
 "360", "SBS"). A fisheye without a stated field of view is taken as 180°,
 and its ellipse is fitted from the picture (`FisheyeEllipseFit`) when one
 is at hand.
 */

import Foundation

public enum VRFormatDetector {
    /// Field of view per fisheye format name.
    static let fisheyeNames: [String: Double] = [
        "fisheye": 180, "fisheye180": 180, "fisheye190": 190, "fisheye200": 200, "fisheye220": 220,
        "mkx200": 200, "mkx220": 220, "vrca220": 220, "rf52": 190,
    ]

    /// What `words` (a file name or one tag, any case, any separators)
    /// say, or nil when they say nothing about VR or stereo.
    public static func format(fromWords words: String) -> VRFormat? {
        let lower = words.lowercased()
        let tokens = Set(lower.split { !$0.isLetter && !$0.isNumber }.map(String.init))
        var projection: VRProjection?
        if let fov = tokens.lazy.compactMap({ fisheyeNames[$0] }).max() {
            projection = .fisheye(fov)
        } else if tokens.contains("360") || tokens.contains("vr360") || tokens.contains("360x180") {
            projection = .vr360
        } else if tokens.contains("180") || tokens.contains("vr180") || tokens.contains("180x180") {
            projection = .vr180
        }
        var layout = StereoLayoutDetector.layout(fromFileName: words)?.layout
        if layout == nil {
            if tokens.contains("3dh") { layout = PackedStereoLayout(arrangement: .sideBySide, isHalfResolution: false) }
            if tokens.contains("3dv") { layout = PackedStereoLayout(arrangement: .overUnder, isHalfResolution: false) }
        }
        switch (layout, projection) {
        case (nil, nil): return nil
        case (let layout, let projection?): return VRFormat(layout: layout, projection: projection)
        case (let layout?, nil): return VRFormat(layout: layout, projection: .flat)
        }
    }

    /// What a set of tags says together: the projection from whichever tag
    /// names one, the packing from whichever names one. Fisheye wins over
    /// equirectangular when both are tagged (it is the more specific).
    public static func format(fromTags tags: [String]) -> VRFormat? {
        let found = tags.compactMap { format(fromWords: $0) }
        guard !found.isEmpty else { return nil }
        let layout = found.lazy.compactMap(\.layout).first
        let projections = found.map(\.projection).filter(\.isImmersive)
        let projection = projections.first { if case .fisheye = $0 { true } else { false } }
            ?? projections.first ?? .flat
        return VRFormat(layout: layout, projection: projection)
    }

    /// The best reading of everything known about a file, strongest first:
    /// what the container declares, then tags, then the name. A VR
    /// projection with no packing named anywhere gets the packing its
    /// frame's shape implies (`packing(for:width:height:)`), mono when the
    /// shape is unknown; a fisheye's ellipse is refined by `fittedEllipse`.
    public static func detect(declared: VRFormat? = nil, tags: [String] = [], fileName: String? = nil,
                              frameWidth: Int? = nil, frameHeight: Int? = nil,
                              fittedEllipse: EyeEllipse? = nil) -> VRFormat? {
        let fromTags = format(fromTags: tags)
        let fromName = fileName.flatMap { format(fromWords: ($0 as NSString).deletingPathExtension) }
        let candidates = [declared, fromTags, fromName].compactMap { $0 }
        guard !candidates.isEmpty else { return nil }
        var projection = candidates.lazy.map(\.projection).first(where: \.isImmersive) ?? .flat
        var layout = candidates.lazy.compactMap(\.layout).first
        if layout == nil, projection.isImmersive, let frameWidth, let frameHeight {
            layout = packing(for: projection, width: frameWidth, height: frameHeight)
        }
        if let fittedEllipse { projection = projection.withEllipse(fittedEllipse) }
        return VRFormat(layout: layout, projection: projection)
    }

    /// The packing a VR frame of this shape has when nothing names it. An eye
    /// of a 180° or fisheye picture is about square and a 360° one about 2:1,
    /// so the frame's shape tells mono, side by side and over under apart:
    /// 180° 2:1 is side by side, 1:2 over under, 1:1 mono; 360° 4:1 side by
    /// side, 1:1 over under, 2:1 mono.
    public static func packing(for projection: VRProjection, width: Int, height: Int) -> PackedStereoLayout? {
        guard width > 0, height > 0, projection.isImmersive else { return nil }
        let aspect = Double(width) / Double(height)
        let eyeAspect: Double = if case .equirectangular(let degrees) = projection, degrees > 270 { 2 } else { 1 }
        let ratio = aspect / eyeAspect
        if ratio > 1.5 { return PackedStereoLayout(arrangement: .sideBySide, isHalfResolution: false) }
        if ratio < 0.7 { return PackedStereoLayout(arrangement: .overUnder, isHalfResolution: false) }
        return nil
    }
}

/// Finds the picture circle (or ellipse) of a fisheye eye: a bright disc on
/// black, the corners outside it dark.
public enum FisheyeEllipseFit {
    /// Below this, a pixel is the black outside the picture circle.
    static let blackLevel: UInt8 = 18

    /// The ellipse in `eye` (one eye's picture, already cut from the packed
    /// frame), or nil when the eye doesn't look like a fisheye: no dark
    /// corners, or nothing bright.
    public static func fit(_ eye: LumaPlane) -> EyeEllipse? {
        let w = eye.width, h = eye.height
        guard w >= 16, h >= 16 else { return nil }
        func luma(_ x: Int, _ y: Int) -> UInt8 { eye.pixels[y * w + x] }
        // Corners: a fisheye's are black; an equirectangular or flat
        // picture's are not.
        let patch = max(2, min(w, h) / 24)
        for (cx, cy) in [(0, 0), (w - patch, 0), (0, h - patch), (w - patch, h - patch)] {
            var sum = 0
            for y in cy ..< cy + patch { for x in cx ..< cx + patch { sum += Int(luma(x, y)) } }
            guard sum / (patch * patch) < Int(blackLevel) else { return nil }
        }
        // The extent of bright pixels along the middle row and column gives
        // the axes; the widest row and tallest column give them more
        // robustly when the middle is dark in places (a night scene).
        var minX = w, maxX = -1, minY = h, maxY = -1
        let step = max(1, min(w, h) / 256)
        for y in stride(from: 0, to: h, by: step) {
            var first = -1, last = -1
            for x in stride(from: 0, to: w, by: step) where luma(x, y) >= blackLevel {
                if first < 0 { first = x }
                last = x
            }
            if first >= 0 { minX = min(minX, first); maxX = max(maxX, last); minY = min(minY, y); maxY = max(maxY, y) }
        }
        guard maxX > minX, maxY > minY else { return nil }
        let rx = Double(maxX - minX + step) / 2 / Double(w)
        let ry = Double(maxY - minY + step) / 2 / Double(h)
        // Too small to be the picture circle: a dark frame with a lamp in it.
        guard rx > 0.3, ry > 0.3 else { return nil }
        return EyeEllipse(centerX: (Double(minX + maxX + step) / 2) / Double(w),
                          centerY: (Double(minY + maxY + step) / 2) / Double(h),
                          radiusX: rx, radiusY: ry)
    }
}
