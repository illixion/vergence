/*
 Vergence - working out how a video is laid out

 A player has to show 2D video, packed stereo 3D (side by side, over under,
 full or half resolution, either eye first) and VR (180° and 360°
 equirectangular, fisheye, mono or stereo) without being told which. Names
 and tags are often missing or wrong, so the decoded picture is evidence
 too. `VideoFormatAnalyzer.analyze` takes a few small greyscale frames from
 across the video and puts everything together, strongest first:

 1. The container's metadata (`declared`) wins outright, confidence 1.
 2. Tags and the file name (`VRFormatDetector`, `StereoLayoutDetector`)
    give the projection (180, 360, fisheye and its field of view) and a
    packing. A projection named this way stands (0.9), except that a named
    180° equirectangular gives way to fisheye circles in the picture.
 3. The picture:
    - Packing: the two candidate halves are compared as
      `StereoLayoutDetector` does (best-shift correlation >= 0.6, beating
      the other split by 0.15). A confident picture overrides a named
      packing (0.7). The split is then checked block by block: of the
      eye's textured blocks at least 30% must find a sharp match in the
      other half, or the halves only share coarse structure (a slide with
      lines of text across it) and the frame is not taken as stereo.
    - Eye order: each textured block of eye A (the left or top half) is
      matched in eye B along the row, +-6% of the eye's width, by
      normalized cross-correlation (peak >= 0.8, no second peak within
      0.05, sub-pixel by a parabola). The signed disparity is d = xA - xB.
      A parallel VR rig puts everything at d >= 0 in the true left eye
      (zero at infinity), so when at least 67% of the non-zero blocks
      (|d| >= 0.5 px) are negative, A is the right eye. Flat 3D films are
      converged and most of the picture sits behind the screen, where the
      true left eye has d < 0, so there 75% positive means reversed. Either
      way it needs 12 non-zero blocks across the frames and a median |d|
      of 0.3% of the eye's width; otherwise the order is left alone
      (`eyeOrderSource == .none`). A name's `RL` token is the fallback.
    - Projection, when nothing names one: an eye whose region outside a
      fitted ellipse (`FisheyeEllipseFit`) is black while the inside isn't,
      in 60% of the frames, is a fisheye (190° unless a name says, 0.85
      stereo, 0.75 mono). An eye of about 2:1 (1.8-2.2) whose left and
      right edge columns continue into each other (correlation >= 0.6 and
      0.3 above two unrelated columns, after each row's mean is taken
      out, in 2/3 of the frames) is 360° (0.85 stereo, 0.75 mono). A
      stereo eye of about 1:1 (0.9-1.1) at least 1440 px across is 180°
      (0.7 from 1920 px, else 0.6); a mono square frame only from 2880 px
      (0.6). Anything else is flat (0.85 with frames to look at).
    - Full or half resolution: for VR from the eye shape the projection
      needs (1:1 for 180° and fisheye, 2:1 for 360°), for flat video from
      `PackedStereoLayout.correctingResolution` and the name.
    - Near-black and flat frames (standard deviation under 5) are not
      evidence. With none left, 2D comes out at 0.3.
 4. The whole answer's confidence is the lower of the projection's and the
    packing's; a stereo answer whose eye order the picture could not
    settle is capped at 0.85. Ordinary 2D with frames to look at is 0.85.
 */

#if canImport(CoreGraphics)
import CoreGraphics
#endif
import Foundation

/// What a video is: 2D, packed 3D or VR, how sure that is and why.
public struct VideoFormatAnalysis: Sendable, Equatable, Codable {
    public enum Source: String, Sendable, Codable { case container, name, picture, shape, none }

    /// What to show: projection .flat + nil layout = ordinary 2D; .flat +
    /// layout = packed 3D; immersive projection = VR (layout nil = mono).
    /// `layout.eyesReversed` is set when the picture says so.
    public var format: VRFormat
    /// 0...1, how sure the whole answer is.
    public var confidence: Double
    public var projectionSource: Source
    public var layoutSource: Source
    public var eyeOrderSource: Source
    /// One short user-facing sentence per finding.
    public var reasons: [String]

    public var isStereo: Bool { format.layout != nil }
    public var isVR: Bool { format.projection.isImmersive }

    public init(format: VRFormat, confidence: Double, projectionSource: Source, layoutSource: Source,
                eyeOrderSource: Source, reasons: [String]) {
        self.format = format
        self.confidence = confidence
        self.projectionSource = projectionSource
        self.layoutSource = layoutSource
        self.eyeOrderSource = eyeOrderSource
        self.reasons = reasons
    }
}

public enum VideoFormatAnalyzer {
    // MARK: Thresholds (see the header)

    static let minFrameDeviation = 5.0
    static let minMatchedBlockFraction = 0.3
    static let minNonZeroBlocks = 12
    static let minMedianDisparity = 0.003
    static let vrOrderMajority = 0.67
    static let flatOrderMajority = 0.75
    static let squareEye: ClosedRange<Double> = 0.9...1.1
    static let panoramaEye: ClosedRange<Double> = 1.8...2.2

    /// `frames`: small greyscale pictures (long side <= ~640 px) of decoded
    /// frames from across the video; `frameSize`: the decoded frame size in
    /// pixels.
    public static func analyze(frames: [LumaPlane], frameSize: CGSize, fileName: String? = nil,
                               tags: [String] = [], declared: VRFormat? = nil) -> VideoFormatAnalysis {
        if let declared {
            return VideoFormatAnalysis(format: declared, confidence: 1, projectionSource: .container,
                                       layoutSource: .container, eyeOrderSource: .container,
                                       reasons: ["The file's metadata says it is \(describe(declared))."])
        }
        var reasons: [String] = []
        let usable = frames.filter(isUsable)
        let frameW = Double(frameSize.width), frameH = Double(frameSize.height)

        // MARK: Names
        let baseName = fileName.map { ($0 as NSString).deletingPathExtension }
        let named = VRFormatDetector.detect(tags: tags, fileName: fileName)
        let namedProjection = named?.projection.isImmersive == true ? named?.projection : nil
        let nameStrings = tags + (baseName.map { [$0] } ?? [])
        let nameReversed = nameStrings.contains { words($0).contains("rl") }
        var namedLayout = named?.layout
        if namedLayout == nil, nameReversed {
            namedLayout = PackedStereoLayout(arrangement: .sideBySide, isHalfResolution: false)
        }
        if let namedProjection { reasons.append("The name says \(namedProjection.label).") }
        if let namedLayout { reasons.append("The name says \(describe(namedLayout.arrangement)).") }

        // MARK: Packing from the picture
        let scores = StereoLayoutDetector.scores(usable)
        var pictureArrangement: PackedStereoLayout.Arrangement? = {
            let best = max(scores.sideBySide, scores.overUnder)
            guard scores.frames > 0, best >= StereoLayoutDetector.stereoThreshold,
                  abs(scores.sideBySide - scores.overUnder) >= StereoLayoutDetector.margin else { return nil }
            return scores.sideBySide > scores.overUnder ? .sideBySide : .overUnder
        }()
        var blocks: [PackedStereoLayout.Arrangement: BlockDisparities] = [:]
        func disparities(_ arrangement: PackedStereoLayout.Arrangement) -> BlockDisparities {
            if let known = blocks[arrangement] { return known }
            var total = BlockDisparities()
            for frame in usable { total.add(blockDisparities(frame, arrangement)) }
            blocks[arrangement] = total
            return total
        }
        if let arrangement = pictureArrangement {
            let found = disparities(arrangement)
            if found.textured >= 20, found.matchedFraction < minMatchedBlockFraction {
                reasons.append("The two halves look alike only roughly; their details don't match, so it isn't 3D.")
                pictureArrangement = nil
            }
        }

        var arrangement: PackedStereoLayout.Arrangement?
        var layoutConfidence: Double
        var layoutSource: VideoFormatAnalysis.Source
        let pictureBest = max(scores.sideBySide, scores.overUnder)
        switch (namedLayout?.arrangement, pictureArrangement) {
        case let (named?, picture?) where named != picture:
            reasons.append("The two halves show the same scene \(describe(picture)), not \(describe(named)) as the name says.")
            arrangement = picture
            layoutConfidence = 0.7
            layoutSource = .picture
        case let (named?, picture):
            if picture != nil { reasons.append("The two halves show the same scene: \(describe(named)).") }
            arrangement = named
            layoutConfidence = picture != nil ? 1 : (usable.isEmpty ? 0.85 : 0.75)
            layoutSource = .name
        case let (nil, picture?):
            reasons.append("The two halves show the same scene: \(describe(picture)).")
            arrangement = picture
            layoutConfidence = min(1, 0.5 + pictureBest / 2)
            layoutSource = .picture
        case (nil, nil):
            arrangement = nil
            layoutConfidence = usable.isEmpty ? 0.3 : 0.85
            layoutSource = usable.isEmpty ? .none : .picture
        }
        // A named VR projection with no packing named or seen: the frame's shape.
        if arrangement == nil, let namedProjection, frameW > 0, frameH > 0,
           let shaped = VRFormatDetector.packing(for: namedProjection, width: Int(frameW), height: Int(frameH)) {
            reasons.append("The frame's shape fits \(namedProjection.label) \(describe(shaped.arrangement)).")
            arrangement = shaped.arrangement
            layoutConfidence = 0.7
            layoutSource = .shape
        }

        // MARK: Projection
        // Eye aspect for the full and the half reading of the packing.
        func eyeAspects(_ arrangement: PackedStereoLayout.Arrangement?) -> (full: Double, half: Double) {
            guard frameW > 0, frameH > 0 else { return (0, 0) }
            switch arrangement {
            case nil: return (frameW / frameH, frameW / frameH)
            case .sideBySide?: return (frameW / 2 / frameH, frameW / frameH)
            case .overUnder?: return (frameW / (frameH / 2), frameW / frameH)
            }
        }
        let eyePixelSide: Double = {
            switch arrangement {
            case nil: min(frameW, frameH)
            case .sideBySide?: min(frameW / 2, frameH)
            case .overUnder?: min(frameW, frameH / 2)
            }
        }()
        let aspects = eyeAspects(arrangement)
        let stereo = arrangement != nil
        let fittedFisheye = usable.isEmpty ? nil : fisheyeEllipse(usable, arrangement)

        var projection: VRProjection = .flat
        var projectionConfidence: Double
        var projectionSource: VideoFormatAnalysis.Source
        if let namedProjection {
            projection = namedProjection
            projectionConfidence = 0.9
            projectionSource = .name
            if let fittedFisheye {
                if case .fisheye = namedProjection {
                    projection = namedProjection.withEllipse(fittedFisheye)
                } else if namedProjection.halfWidthDegrees <= 90 {
                    reasons.append("Each eye is a round picture on black, so it is a fisheye, not equirectangular.")
                    projection = .fisheye(fieldOfView: 190, ellipse: fittedFisheye)
                    projectionConfidence = 0.8
                    projectionSource = .picture
                }
            }
        } else if let fittedFisheye {
            reasons.append("Each eye is a round picture on black: fisheye.")
            projection = .fisheye(fieldOfView: 190, ellipse: fittedFisheye)
            projectionConfidence = stereo ? 0.85 : 0.75
            projectionSource = .picture
        } else if [aspects.full, aspects.half].contains(where: panoramaEye.contains), hasWrap(usable, arrangement) {
            reasons.append("Each eye's left and right edges join up: 360°.")
            projection = .vr360
            projectionConfidence = stereo ? 0.85 : 0.75
            projectionSource = .picture
        } else if stereo, eyePixelSide >= 1440, [aspects.full, aspects.half].contains(where: squareEye.contains) {
            reasons.append("Each eye is square, as 180° video is.")
            projection = .vr180
            projectionConfidence = eyePixelSide >= 1920 ? 0.7 : 0.6
            projectionSource = .shape
        } else if !stereo, eyePixelSide >= 2880, squareEye.contains(aspects.full) {
            reasons.append("The frame is large and square, as 180° video is.")
            projection = .vr180
            projectionConfidence = 0.6
            projectionSource = .shape
        } else if !usable.isEmpty {
            projectionConfidence = 0.85
            projectionSource = .picture
        } else if namedLayout != nil {
            projectionConfidence = 0.8
            projectionSource = .name
        } else {
            projectionConfidence = 0.4
            projectionSource = .none
        }

        // MARK: Full or half resolution
        var layout: PackedStereoLayout?
        if let arrangement {
            if projection.isImmersive {
                let target: Double = if case .equirectangular(let degrees) = projection, degrees > 270 { 2 } else { 1 }
                let half = abs(log(max(aspects.half, 1e-6) / target)) < abs(log(max(aspects.full, 1e-6) / target))
                layout = PackedStereoLayout(arrangement: arrangement, isHalfResolution: half)
                if half { reasons.append("Each eye is squeezed to half resolution: only that gives \(projection.label) its shape.") }
            } else {
                let explicitHalf = nameStrings.lazy
                    .compactMap { StereoLayoutDetector.layout(fromFileName: $0) }
                    .first { $0.layout.arrangement == arrangement && $0.halfIsExplicit }?.layout.isHalfResolution
                let guess = PackedStereoLayout(arrangement: arrangement, isHalfResolution: explicitHalf ?? false)
                let corrected = guess.correctingResolution(forPacked: frameSize)
                let other = PackedStereoLayout(arrangement: arrangement, isHalfResolution: !guess.isHalfResolution)
                    .correctingResolution(forPacked: frameSize)
                if corrected.isHalfResolution != guess.isHalfResolution || other.isHalfResolution == corrected.isHalfResolution {
                    if corrected.isHalfResolution { reasons.append("Each eye is squeezed to half resolution: the frame's shape fits that.") }
                } else if explicitHalf == nil {
                    reasons.append("Full or half resolution per eye can't be told from the frame's shape; assuming full.")
                    layoutConfidence = min(layoutConfidence, 0.6)
                }
                layout = corrected
            }
        }

        // MARK: Eye order
        var eyeOrderSource: VideoFormatAnalysis.Source = .none
        if let arrangement, var current = layout {
            let found = disparities(arrangement)
            switch eyeOrder(found, immersive: projection.isImmersive) {
            case let .decided(reversed, share)?:
                current.eyesReversed = reversed
                eyeOrderSource = .picture
                let percent = Int((share * 100).rounded())
                if reversed {
                    reasons.append("Near things sit the wrong way round between the eyes in \(percent)% of the picture: the eyes are stored swapped.")
                } else {
                    reasons.append("Near and far things sit as they should between the eyes in \(percent)% of the picture: the eyes are in the usual order.")
                }
            case nil:
                if nameReversed {
                    current.eyesReversed = true
                    eyeOrderSource = .name
                    reasons.append("The name says the right eye comes first.")
                }
            }
            layout = current
        }

        // MARK: Confidence
        var confidence = min(projectionConfidence, layoutConfidence)
        if layout != nil, eyeOrderSource != .picture, !usable.isEmpty { confidence = min(confidence, 0.85) }
        if usable.isEmpty, !frames.isEmpty { reasons.append("The frames are too dark or flat to judge.") }
        if layout == nil, !projection.isImmersive {
            reasons.append(usable.isEmpty ? "Nothing says it is 3D or VR." : "Nothing in the picture looks 3D or VR: ordinary 2D video.")
        }
        let format = VRFormat(layout: layout, projection: projection)
        return VideoFormatAnalysis(format: format, confidence: confidence, projectionSource: projectionSource,
                                   layoutSource: layout == nil && !usable.isEmpty ? .picture : layoutSource,
                                   eyeOrderSource: eyeOrderSource, reasons: reasons)
    }

    // MARK: - Descriptions

    static func describe(_ arrangement: PackedStereoLayout.Arrangement) -> String {
        StereoLayoutDetector.describe(arrangement)
    }

    static func describe(_ format: VRFormat) -> String {
        let packing = format.layout.map { describe($0.arrangement) }
        switch (format.projection.isImmersive, packing) {
        case (false, nil): return "ordinary 2D"
        case (false, let packing?): return "3D \(packing)"
        case (true, nil): return "\(format.projection.label) mono"
        case (true, let packing?): return "\(format.projection.label) \(packing)"
        }
    }

    static func words(_ string: String) -> Set<String> {
        Set(string.lowercased().split { !$0.isLetter && !$0.isNumber }.map(String.init))
    }

    // MARK: - Frames

    /// Not near black and not flat.
    static func isUsable(_ frame: LumaPlane) -> Bool {
        guard frame.width >= 16, frame.height >= 16 else { return false }
        let step = max(1, frame.pixels.count / 20_000)
        var sum = 0.0, sumSquares = 0.0, n = 0.0
        for i in stride(from: 0, to: frame.pixels.count, by: step) {
            let v = Double(frame.pixels[i])
            sum += v; sumSquares += v * v; n += 1
        }
        let mean = sum / n
        return (sumSquares / n - mean * mean).squareRoot() >= minFrameDeviation
    }

    struct Region {
        var x: Int, y: Int, width: Int, height: Int
    }

    /// Eye A (left or top) and eye B; the whole frame twice for mono.
    static func eyeRegions(_ frame: LumaPlane, _ arrangement: PackedStereoLayout.Arrangement?) -> [Region] {
        switch arrangement {
        case nil:
            return [Region(x: 0, y: 0, width: frame.width, height: frame.height)]
        case .sideBySide?:
            let w = frame.width / 2
            return [Region(x: 0, y: 0, width: w, height: frame.height), Region(x: w, y: 0, width: w, height: frame.height)]
        case .overUnder?:
            let h = frame.height / 2
            return [Region(x: 0, y: 0, width: frame.width, height: h), Region(x: 0, y: h, width: frame.width, height: h)]
        }
    }

    static func crop(_ frame: LumaPlane, _ r: Region) -> LumaPlane {
        var pixels = [UInt8]()
        pixels.reserveCapacity(r.width * r.height)
        for y in r.y ..< r.y + r.height {
            let row = y * frame.width + r.x
            pixels.append(contentsOf: frame.pixels[row ..< row + r.width])
        }
        return LumaPlane(width: r.width, height: r.height, pixels: pixels)
    }

    // MARK: - Block disparity

    struct BlockDisparities {
        /// Blocks with enough horizontal texture to match.
        var textured = 0
        /// Of those, blocks with one sharp match.
        var matched = 0
        /// d = xA - xB of matched blocks with |d| >= 0.5 px, as a fraction
        /// of the eye's width.
        var nonZero: [Double] = []

        var matchedFraction: Double { textured > 0 ? Double(matched) / Double(textured) : 0 }

        mutating func add(_ other: BlockDisparities) {
            textured += other.textured
            matched += other.matched
            nonZero += other.nonZero
        }
    }

    enum EyeOrder: Equatable {
        /// `share`: the fraction of non-zero blocks agreeing.
        case decided(reversed: Bool, share: Double)
    }

    static func eyeOrder(_ found: BlockDisparities, immersive: Bool) -> EyeOrder? {
        let n = found.nonZero.count
        guard n >= minNonZeroBlocks else { return nil }
        let magnitudes = found.nonZero.map(abs).sorted()
        guard magnitudes[n / 2] >= minMedianDisparity else { return nil }
        let positive = Double(found.nonZero.filter { $0 > 0 }.count) / Double(n)
        // The share that says the eyes are in order: positive for a parallel
        // VR rig, negative (behind the screen) for a converged flat film.
        let inOrder = immersive ? positive : 1 - positive
        let majority = immersive ? vrOrderMajority : flatOrderMajority
        if inOrder >= majority { return .decided(reversed: false, share: inOrder) }
        if 1 - inOrder >= majority { return .decided(reversed: true, share: 1 - inOrder) }
        return nil
    }

    /// Matches eye A's textured blocks along the row in eye B.
    static func blockDisparities(_ frame: LumaPlane, _ arrangement: PackedStereoLayout.Arrangement) -> BlockDisparities {
        let regions = eyeRegions(frame, arrangement)
        let a = regions[0], b = regions[1]
        let w = a.width, h = a.height
        let block = max(8, min(w, h) / 16)
        let range = max(3, w * 6 / 100)
        var result = BlockDisparities()
        guard w - 2 * (range + 1) >= block, h >= block * 2 else { return result }
        let stride = frame.width
        let n = Double(block * block)
        var scores = [Double](repeating: -2, count: 2 * range + 1)
        frame.pixels.withUnsafeBufferPointer { p in
            for by in Swift.stride(from: h / 20, through: h - h / 20 - block, by: block) {
                for bx in Swift.stride(from: range + 1, through: w - range - 1 - block, by: block) {
                    // Eye A's block: needs contrast and vertical edges to tell
                    // horizontal shifts apart.
                    var sa = 0, saa = 0, gradient = 0
                    for y in 0 ..< block {
                        let row = (a.y + by + y) * stride + a.x + bx
                        for x in 0 ..< block {
                            let v = Int(p[row + x])
                            sa += v; saa += v * v
                            if x > 0 { gradient += abs(v - Int(p[row + x - 1])) }
                        }
                    }
                    let meanA = Double(sa) / n
                    let varA = Double(saa) / n - meanA * meanA
                    guard varA >= 36, Double(gradient) / Double(block * (block - 1)) >= 2 else { continue }
                    result.textured += 1

                    for s in -range ... range {
                        var sb = 0, sbb = 0, sab = 0
                        for y in 0 ..< block {
                            let rowA = (a.y + by + y) * stride + a.x + bx
                            let rowB = (b.y + by + y) * stride + b.x + bx + s
                            for x in 0 ..< block {
                                let va = Int(p[rowA + x]), vb = Int(p[rowB + x])
                                sb += vb; sbb += vb * vb; sab += va * vb
                            }
                        }
                        let meanB = Double(sb) / n
                        let varB = Double(sbb) / n - meanB * meanB
                        let cov = Double(sab) / n - meanA * meanB
                        scores[s + range] = varB > 1 ? cov / (varA * varB).squareRoot() : -1
                    }
                    var best = 0
                    for i in scores.indices where scores[i] > scores[best] { best = i }
                    let peak = scores[best]
                    guard peak >= 0.8, best > 0, best < scores.count - 1 else { continue }
                    // A second peak almost as high: a repeating pattern.
                    let ambiguous = scores.indices.contains { i in
                        abs(i - best) >= 2 && scores[i] > peak - 0.05
                            && (i == 0 || scores[i] >= scores[i - 1])
                            && (i == scores.count - 1 || scores[i] >= scores[i + 1])
                    }
                    guard !ambiguous else { continue }
                    result.matched += 1
                    let l = scores[best - 1], r = scores[best + 1]
                    let curvature = l - 2 * peak + r
                    let offset = curvature < 0 ? max(-0.5, min(0.5, 0.5 * (l - r) / curvature)) : 0
                    let d = -(Double(best - range) + offset)
                    if abs(d) >= 0.5 { result.nonZero.append(d / Double(w)) }
                }
            }
        }
        return result
    }

    // MARK: - Fisheye

    /// The averaged picture ellipse when the eyes are round pictures on
    /// black in most frames, else nil.
    static func fisheyeEllipse(_ frames: [LumaPlane], _ arrangement: PackedStereoLayout.Arrangement?) -> EyeEllipse? {
        var fits: [EyeEllipse] = []
        for frame in frames {
            let eyes = eyeRegions(frame, arrangement).map { crop(frame, $0) }
            let found = eyes.compactMap { eye in FisheyeEllipseFit.fit(eye).flatMap { isRoundPicture(eye, $0) ? $0 : nil } }
            if found.count == eyes.count { fits.append(found[0]) }
        }
        guard !fits.isEmpty, Double(fits.count) >= (0.6 * Double(frames.count)).rounded(.up) else { return nil }
        let c = Double(fits.count)
        return EyeEllipse(centerX: fits.map(\.centerX).reduce(0, +) / c, centerY: fits.map(\.centerY).reduce(0, +) / c,
                          radiusX: fits.map(\.radiusX).reduce(0, +) / c, radiusY: fits.map(\.radiusY).reduce(0, +) / c)
    }

    /// Outside the ellipse (with a margin) is black, inside mostly isn't,
    /// and there is enough outside to tell: a letterboxed or windowboxed
    /// rectangle has bright corners outside any ellipse through its edges.
    static func isRoundPicture(_ eye: LumaPlane, _ e: EyeEllipse) -> Bool {
        let step = max(1, min(eye.width, eye.height) / 96)
        var outside = 0, outsideDark = 0, inside = 0, insideDark = 0, total = 0
        for y in Swift.stride(from: 0, to: eye.height, by: step) {
            let v = ((Double(y) + 0.5) / Double(eye.height) - e.centerY) / e.radiusY
            for x in Swift.stride(from: 0, to: eye.width, by: step) {
                let u = ((Double(x) + 0.5) / Double(eye.width) - e.centerX) / e.radiusX
                let r2 = u * u + v * v
                let luma = eye.pixels[y * eye.width + x]
                total += 1
                if r2 > 1.1 * 1.1 {
                    outside += 1
                    if luma < 24 { outsideDark += 1 }
                } else if r2 < 0.85 * 0.85 {
                    inside += 1
                    if luma < FisheyeEllipseFit.blackLevel { insideDark += 1 }
                }
            }
        }
        guard outside * 33 >= total, inside > 0 else { return false }
        return Double(outsideDark) >= 0.9 * Double(outside) && Double(insideDark) <= 0.5 * Double(inside)
    }

    // MARK: - 360° wrap

    /// The eyes' left and right edges continue into each other in at least
    /// two thirds of the frames that can tell.
    static func hasWrap(_ frames: [LumaPlane], _ arrangement: PackedStereoLayout.Arrangement?) -> Bool {
        var telling = 0, wrapping = 0
        for frame in frames {
            guard let (wrap, unrelated) = wrapCorrelation(frame, eyeRegions(frame, arrangement)[0]) else { continue }
            telling += 1
            if wrap >= 0.6, wrap - unrelated >= 0.3 { wrapping += 1 }
        }
        return telling > 0 && Double(wrapping) * 3 >= Double(telling) * 2
    }

    /// Correlation of the region's last two columns with its first two
    /// (across the seam), and of unrelated column pairs, with each row's
    /// mean taken out so a sky-to-ground gradient doesn't count.
    static func wrapCorrelation(_ frame: LumaPlane, _ r: Region) -> (wrap: Double, unrelated: Double)? {
        let w = r.width
        guard w >= 16, r.height >= 16 else { return nil }
        let rows = Array(r.y + r.height / 20 ..< r.y + r.height - r.height / 20)
        func column(_ x0: Int) -> [Double] {
            rows.map { y in
                let row = y * frame.width + r.x
                var sum = 0
                for x in 0 ..< w { sum += Int(frame.pixels[row + x]) }
                let mean = Double(sum) / Double(w)
                return (Double(frame.pixels[row + x0]) + Double(frame.pixels[row + x0 + 1])) / 2 - mean
            }
        }
        func correlation(_ a: [Double], _ b: [Double]) -> Double? {
            let n = Double(a.count)
            let ma = a.reduce(0, +) / n, mb = b.reduce(0, +) / n
            var cov = 0.0, va = 0.0, vb = 0.0
            for i in a.indices {
                cov += (a[i] - ma) * (b[i] - mb); va += (a[i] - ma) * (a[i] - ma); vb += (b[i] - mb) * (b[i] - mb)
            }
            guard va / n >= 9, vb / n >= 9 else { return nil }
            return cov / (va * vb).squareRoot()
        }
        let first = column(0), last = column(w - 2)
        guard let wrap = correlation(last, first) else { return nil }
        let unrelated = [correlation(first, column(w / 2)), correlation(column(w / 4), column(3 * w / 4))]
            .compactMap { $0 }
        guard !unrelated.isEmpty else { return nil }
        return (wrap, unrelated.reduce(0, +) / Double(unrelated.count))
    }
}
