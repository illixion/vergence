/*
 Vergence - telling how a video packs its eyes

 `vergence make` works without being told the layout. Three kinds of
 evidence, strongest first:

 1. The container says so (Spherical Video V2's `st3d`, read by the
    platform layer and passed in as `declared`).
 2. The file name says so: `Film.3D.HSBS.mkv`, `clip_sbs.mp4`, `Movie (OU)`.
    Whole words only, so "house" is not "ou".
 3. The picture says so. Two halves of a stereo frame show the same scene a
    few pixels apart; two halves of an ordinary frame don't. Each candidate
    split is scored by the best correlation of its halves over small
    horizontal shifts (the disparity), averaged over frames from across the
    video.

 Full or half resolution per eye comes from the frame's shape when only one
 reading gives an eye a real picture's shape (`PackedStereoLayout.correctingResolution`),
 else from the name, else full.
 */

#if canImport(CoreGraphics)
import CoreGraphics
#endif
import Foundation

/// An 8-bit greyscale picture, small enough to compare quickly.
public struct LumaPlane: Sendable {
    public let width: Int
    public let height: Int
    public let pixels: [UInt8]

    public init(width: Int, height: Int, pixels: [UInt8]) {
        precondition(pixels.count == width * height, "pixel count does not match the size")
        self.width = width
        self.height = height
        self.pixels = pixels
    }
}

/// What the evidence says about a video's layout.
public struct StereoLayoutDetection: Sendable, Equatable {
    public enum Source: String, Sendable {
        case container, fileName = "file name", picture, frameShape = "frame shape", fallback
    }

    /// Nil when the video looks like ordinary 2D.
    public var layout: PackedStereoLayout?
    /// 0...1: how sure the arrangement is.
    public var confidence: Double
    /// Each finding in a sentence, for the user.
    public var reasons: [String]
    /// What decided the arrangement.
    public var source: Source

    public var format: StereoscopicFormat? {
        layout.map { layout in
            switch (layout.arrangement, layout.isHalfResolution) {
            case (.sideBySide, false): .sideBySide
            case (.sideBySide, true): .halfSideBySide
            case (.overUnder, false): .overUnder
            case (.overUnder, true): .halfOverUnder
            }
        }
    }
}

public enum StereoLayoutDetector {
    /// Similarity scores from the picture analysis, -1...1 (1 = identical
    /// halves after the best shift).
    public struct PictureScores: Sendable, Equatable {
        public var sideBySide: Double
        public var overUnder: Double
        public var frames: Int
    }

    /// Scores above this mean the two halves show the same scene.
    static let stereoThreshold = 0.6
    /// The winner must beat the other split by this much.
    static let margin = 0.15

    /// The layout a file name declares, if any. The name is split into words
    /// on anything but letters and digits.
    public static func layout(fromFileName name: String) -> (layout: PackedStereoLayout, halfIsExplicit: Bool)? {
        let words = Set(name.lowercased().split { !$0.isLetter && !$0.isNumber }.map(String.init))
        let joined = name.lowercased()
        func has(_ any: String...) -> Bool { any.contains { words.contains($0) } }
        func phrase(_ any: String...) -> Bool { any.contains { joined.contains($0) } }
        if has("hsbs") || phrase("half-sbs", "half sbs", "half_sbs", "half.sbs") {
            return (PackedStereoLayout(arrangement: .sideBySide, isHalfResolution: true), true)
        }
        if has("hou", "htab", "htb") || phrase("half-ou", "half ou", "half_ou", "half.ou", "half-tab", "half_tab") {
            return (PackedStereoLayout(arrangement: .overUnder, isHalfResolution: true), true)
        }
        if has("fsbs") || phrase("full-sbs", "full sbs", "full_sbs", "full.sbs") {
            return (PackedStereoLayout(arrangement: .sideBySide, isHalfResolution: false), true)
        }
        if has("fou", "ftab") || phrase("full-ou", "full ou", "full_ou", "full.ou") {
            return (PackedStereoLayout(arrangement: .overUnder, isHalfResolution: false), true)
        }
        if has("sbs", "lr") || phrase("side-by-side", "side by side", "sidebyside", "side_by_side") {
            return (PackedStereoLayout(arrangement: .sideBySide, isHalfResolution: false), false)
        }
        if has("ou", "tab", "tb", "overunder") || phrase("over-under", "over under", "top-bottom", "top-and-bottom", "top_bottom") {
            return (PackedStereoLayout(arrangement: .overUnder, isHalfResolution: false), false)
        }
        return nil
    }

    /// Scores both splits over `frames`.
    public static func scores(_ frames: [LumaPlane]) -> PictureScores {
        var sbs = 0.0, ou = 0.0, counted = 0
        for frame in frames where frame.width >= 16 && frame.height >= 16 {
            guard let a = sideBySideScore(frame), let b = overUnderScore(frame) else { continue }
            sbs += a
            ou += b
            counted += 1
        }
        guard counted > 0 else { return PictureScores(sideBySide: 0, overUnder: 0, frames: 0) }
        return PictureScores(sideBySide: sbs / Double(counted), overUnder: ou / Double(counted), frames: counted)
    }

    /// Puts the evidence together. `declared` is what the container says;
    /// `frameSize` is the decoded frame's size in pixels.
    public static func detect(declared: PackedStereoLayout?, fileName: String, frameSize: CGSize,
                              pictures: [LumaPlane]) -> StereoLayoutDetection {
        var reasons: [String] = []
        if let declared {
            reasons.append("the file's stereo metadata says \(describe(declared.arrangement))")
            return StereoLayoutDetection(layout: declared, confidence: 1, reasons: reasons, source: .container)
        }

        let named = layout(fromFileName: fileName)
        let scores = scores(pictures)
        if scores.frames > 0 {
            reasons.append(String(format: "halves compared over %d frames: side by side %.2f, over-under %.2f",
                                  scores.frames, scores.sideBySide, scores.overUnder))
        }

        let pictureArrangement: PackedStereoLayout.Arrangement? = {
            let best = max(scores.sideBySide, scores.overUnder)
            guard best >= stereoThreshold, abs(scores.sideBySide - scores.overUnder) >= margin else { return nil }
            return scores.sideBySide > scores.overUnder ? .sideBySide : .overUnder
        }()

        let arrangement: PackedStereoLayout.Arrangement
        var confidence: Double
        let source: StereoLayoutDetection.Source
        switch (named, pictureArrangement) {
        case let (named?, picture?) where named.layout.arrangement != picture:
            reasons.append("the file name says \(describe(named.layout.arrangement)), but the picture looks \(describe(picture)); going by the picture")
            arrangement = picture
            confidence = 0.7
            source = .picture
        case let (named?, picture):
            reasons.append("the file name says \(describe(named.layout.arrangement))")
            if picture != nil { reasons.append("and the picture agrees") }
            arrangement = named.layout.arrangement
            confidence = picture != nil ? 1 : (scores.frames > 0 ? 0.75 : 0.85)
            source = .fileName
        case let (nil, picture?):
            reasons.append("the picture's halves match \(describe(picture))")
            arrangement = picture
            confidence = min(1, 0.5 + max(scores.sideBySide, scores.overUnder) / 2)
            source = .picture
        case (nil, nil):
            reasons.append(scores.frames > 0 ? "neither split of the picture has matching halves: it looks like 2D video"
                                             : "nothing says it is 3D")
            return StereoLayoutDetection(layout: nil, confidence: scores.frames > 0 ? 0.8 : 0.3, reasons: reasons, source: .picture)
        }

        // Full or half: the frame's shape first, then the name.
        let explicitHalf = named.flatMap { $0.layout.arrangement == arrangement && $0.halfIsExplicit ? $0.layout.isHalfResolution : nil }
        let guess = PackedStereoLayout(arrangement: arrangement, isHalfResolution: explicitHalf ?? false)
        var layout = guess.correctingResolution(forPacked: frameSize)
        if layout.isHalfResolution != guess.isHalfResolution {
            reasons.append("each eye is \(layout.isHalfResolution ? "squeezed to half" : "full") resolution: only that gives a \(Int(frameSize.width))×\(Int(frameSize.height)) frame a picture-shaped eye")
        } else if let explicitHalf {
            reasons.append("the file name says \(explicitHalf ? "half" : "full") resolution per eye")
        } else {
            let other = PackedStereoLayout(arrangement: arrangement, isHalfResolution: true).correctingResolution(forPacked: frameSize)
            if other.isHalfResolution == layout.isHalfResolution {
                reasons.append("each eye is full resolution: the frame's shape fits that")
            } else {
                reasons.append("full or half resolution per eye can't be told from a \(Int(frameSize.width))×\(Int(frameSize.height)) frame; assuming full (--layout h\(arrangement == .sideBySide ? "sbs" : "ou") if the picture looks stretched)")
                confidence = min(confidence, 0.6)
            }
        }
        layout.eyesReversed = false
        return StereoLayoutDetection(layout: layout, confidence: confidence, reasons: reasons, source: source)
    }

    static func describe(_ arrangement: PackedStereoLayout.Arrangement) -> String {
        arrangement == .sideBySide ? "side by side" : "over-under"
    }

    // MARK: - Picture analysis

    /// Left half against right half.
    static func sideBySideScore(_ frame: LumaPlane) -> Double? {
        let half = frame.width / 2
        return bestShiftCorrelation(frame, a: (0, 0), b: (half, 0), width: half, height: frame.height)
    }

    /// Top half against bottom half.
    static func overUnderScore(_ frame: LumaPlane) -> Double? {
        let half = frame.height / 2
        return bestShiftCorrelation(frame, a: (0, 0), b: (0, half), width: frame.width, height: half)
    }

    /// The highest normalized cross-correlation between two regions of
    /// `frame`, trying horizontal shifts up to 6% of the region's width (the
    /// eyes differ by their disparity). Nil when a region is flat.
    static func bestShiftCorrelation(_ frame: LumaPlane, a: (x: Int, y: Int), b: (x: Int, y: Int),
                                     width: Int, height: Int) -> Double? {
        let maxShift = max(1, width * 6 / 100)
        // Leave out the edges the shift would run off, and a border where
        // letterboxing and logos live.
        let inset = maxShift + width / 20
        let top = height / 10
        guard width - 2 * inset >= 8, height - 2 * top >= 8 else { return nil }
        var best: Double?
        frame.pixels.withUnsafeBufferPointer { p in
            for shift in stride(from: -maxShift, through: maxShift, by: max(1, maxShift / 12)) {
                var sa = 0.0, sb = 0.0, saa = 0.0, sbb = 0.0, sab = 0.0, n = 0.0
                for y in top ..< height - top {
                    let rowA = (a.y + y) * frame.width + a.x
                    let rowB = (b.y + y) * frame.width + b.x + shift
                    for x in inset ..< width - inset {
                        let va = Double(p[rowA + x]), vb = Double(p[rowB + x])
                        sa += va; sb += vb; saa += va * va; sbb += vb * vb; sab += va * vb
                        n += 1
                    }
                }
                let cov = sab - sa * sb / n
                let varA = saa - sa * sa / n, varB = sbb - sb * sb / n
                // A flat region (a black screen) matches anything.
                guard varA / n > 4, varB / n > 4 else { continue }
                let r = cov / (varA * varB).squareRoot()
                if r > (best ?? -2) { best = r }
            }
        }
        return best
    }
}
