#if canImport(CoreGraphics)
import CoreGraphics
#endif
import Foundation
import Testing
@testable import VergenceCore

@Suite struct StereoLayoutDetectionTests {
    /// Noise, deterministic.
    static func texture(width: Int, height: Int, seed: UInt64) -> [UInt8] {
        var state = seed &* 0x9E37_79B9_7F4A_7C15 | 1
        return (0 ..< width * height).map { _ in
            state ^= state << 13; state ^= state >> 7; state ^= state << 17
            return UInt8(truncatingIfNeeded: state >> 24)
        }
    }

    /// A scene seen twice, `disparity` pixels apart, packed as `arrangement`.
    static func stereoFrame(_ arrangement: PackedStereoLayout.Arrangement, eye: (w: Int, h: Int), disparity: Int, seed: UInt64) -> LumaPlane {
        let sceneWidth = eye.w + disparity
        let scene = texture(width: sceneWidth, height: eye.h, seed: seed)
        func eyePixels(_ shift: Int) -> [[UInt8]] {
            (0 ..< eye.h).map { y in Array(scene[y * sceneWidth + shift ..< y * sceneWidth + shift + eye.w]) }
        }
        let left = eyePixels(0), right = eyePixels(disparity)
        switch arrangement {
        case .sideBySide:
            return LumaPlane(width: eye.w * 2, height: eye.h, pixels: (0 ..< eye.h).flatMap { left[$0] + right[$0] })
        case .overUnder:
            return LumaPlane(width: eye.w, height: eye.h * 2, pixels: left.flatMap { $0 } + right.flatMap { $0 })
        }
    }

    @Test func fileNamesAreReadWordByWord() throws {
        #expect(StereoLayoutDetector.layout(fromFileName: "Film.2010.3D.HSBS.1080p.mkv")?.layout
            == PackedStereoLayout(arrangement: .sideBySide, isHalfResolution: true))
        #expect(StereoLayoutDetector.layout(fromFileName: "Concert (Half-OU).mp4")?.layout
            == PackedStereoLayout(arrangement: .overUnder, isHalfResolution: true))
        let sbs = try #require(StereoLayoutDetector.layout(fromFileName: "clip_sbs.mp4"))
        #expect(sbs.layout.arrangement == .sideBySide && !sbs.halfIsExplicit)
        #expect(StereoLayoutDetector.layout(fromFileName: "Movie.3D.TAB.mov")?.layout.arrangement == .overUnder)
        // Not a layout: "ou" inside a word, or no word at all.
        #expect(StereoLayoutDetector.layout(fromFileName: "Our House Tour.mp4") == nil)
        #expect(StereoLayoutDetector.layout(fromFileName: "holiday.mp4") == nil)
    }

    @Test func picturesTellSideBySideFromOverUnderFromFlat() {
        let sbs = (0 ..< 3).map { Self.stereoFrame(.sideBySide, eye: (160, 90), disparity: 6, seed: UInt64($0) + 1) }
        let ou = (0 ..< 3).map { Self.stereoFrame(.overUnder, eye: (160, 90), disparity: 6, seed: UInt64($0) + 10) }
        let flat = (0 ..< 3).map { LumaPlane(width: 320, height: 180, pixels: Self.texture(width: 320, height: 180, seed: UInt64($0) + 20)) }

        let a = StereoLayoutDetector.scores(sbs)
        #expect(a.sideBySide > 0.9 && a.overUnder < 0.3)
        let b = StereoLayoutDetector.scores(ou)
        #expect(b.overUnder > 0.9 && b.sideBySide < 0.3)
        let c = StereoLayoutDetector.scores(flat)
        #expect(c.sideBySide < 0.3 && c.overUnder < 0.3)

        let detected = StereoLayoutDetector.detect(declared: nil, fileName: "clip.mp4", frameSize: CGSize(width: 3840, height: 1080), pictures: sbs)
        #expect(detected.layout == PackedStereoLayout(arrangement: .sideBySide, isHalfResolution: false))
        #expect(detected.source == .picture)
        #expect(StereoLayoutDetector.detect(declared: nil, fileName: "clip.mp4", frameSize: CGSize(width: 1920, height: 1080),
                                            pictures: flat).layout == nil)
    }

    @Test func aBlackPictureIsNotEvidence() {
        let black = LumaPlane(width: 320, height: 180, pixels: [UInt8](repeating: 16, count: 320 * 180))
        let scores = StereoLayoutDetector.scores([black])
        #expect(scores.frames == 0)
        let detected = StereoLayoutDetector.detect(declared: nil, fileName: "x.mp4", frameSize: CGSize(width: 1920, height: 1080), pictures: [black])
        #expect(detected.layout == nil)
    }

    @Test func theFrameShapeDecidesFullOrHalf() {
        let sbs = [Self.stereoFrame(.sideBySide, eye: (160, 90), disparity: 6, seed: 3)]
        let ou = [Self.stereoFrame(.overUnder, eye: (160, 90), disparity: 6, seed: 4)]
        func detect(_ name: String, _ w: Int, _ h: Int, _ pictures: [LumaPlane]) -> PackedStereoLayout? {
            StereoLayoutDetector.detect(declared: nil, fileName: name, frameSize: CGSize(width: w, height: h), pictures: pictures).layout
        }
        #expect(detect("a.mp4", 1920, 1080, sbs)?.isHalfResolution == true)
        #expect(detect("a.mp4", 3840, 1080, sbs)?.isHalfResolution == false)
        #expect(detect("a.mp4", 1920, 1080, ou)?.isHalfResolution == true)
        #expect(detect("a.mp4", 1920, 2160, ou)?.isHalfResolution == false)
        // A "full SBS" name on a 1920×1080 frame: the shape wins.
        #expect(detect("a.FSBS.mp4", 1920, 1080, sbs)?.isHalfResolution == true)
        // 2880×1080 fits both (4:3 full, or 2.67:1 half): the name decides,
        // else full with lower confidence.
        #expect(detect("a.HSBS.mp4", 2880, 1080, sbs)?.isHalfResolution == true)
        let ambiguous = StereoLayoutDetector.detect(declared: nil, fileName: "a.mp4", frameSize: CGSize(width: 2880, height: 1080), pictures: sbs)
        #expect(ambiguous.layout?.isHalfResolution == false)
        #expect(ambiguous.confidence <= 0.6)
    }

    @Test func thePictureOverrulesAWrongName() {
        let ou = [Self.stereoFrame(.overUnder, eye: (160, 90), disparity: 6, seed: 5)]
        let detected = StereoLayoutDetector.detect(declared: nil, fileName: "clip.SBS.mp4", frameSize: CGSize(width: 1920, height: 2160), pictures: ou)
        #expect(detected.layout?.arrangement == .overUnder)
        #expect(detected.confidence < 1)
    }

    @Test func declaredMetadataWins() {
        let declared = PackedStereoLayout(arrangement: .overUnder, isHalfResolution: false)
        let detected = StereoLayoutDetector.detect(declared: declared, fileName: "clip.SBS.mp4", frameSize: CGSize(width: 1920, height: 2160), pictures: [])
        #expect(detected.layout == declared && detected.source == .container)
    }
}

@Suite struct SpatialMetadataTests {
    @Test func everyPresetIsComplete() {
        for preset in SpatialPreset.allCases {
            #expect(preset.metadata.problems.isEmpty, "\(preset)")
        }
    }

    @Test func valuesAreStoredInVideoToolboxUnits() {
        let iphone = SpatialPreset.iphone.metadata
        #expect(iphone.baselineMicrometers == 19_240)
        #expect(iphone.fieldOfViewThousandthsOfDegree == 63_400)
        #expect(iphone.disparityAdjustmentNormalized == 200)
        var negative = iphone
        negative.horizontalDisparityAdjustment = -0.02
        #expect(negative.disparityAdjustmentNormalized == -200)
    }

    @Test func theVisionProPresetMatchesItsOwnRecording() {
        let avp = SpatialPreset(rawValue: "avp")?.metadata
        #expect(avp?.baselineMicrometers == 63_664)
        #expect(avp?.fieldOfViewThousandthsOfDegree == 71_590)
        #expect(avp?.disparityAdjustmentNormalized == 293)
    }

    @Test func impossibleValuesAreReported() {
        var bad = SpatialPreset.generic.metadata
        bad.baselineMillimeters = 0
        bad.horizontalFieldOfViewDegrees = 200
        bad.horizontalDisparityAdjustment = 1.5
        #expect(bad.problems.count == 3)
    }
}

@Suite struct SpatialBitrateTests {
    @Test func scalesWithPixelsAndFrameRate() {
        let hd24 = SpatialBitrate.bitsPerSecond(eyeWidth: 1920, eyeHeight: 1080, frameRate: 24)
        #expect((9_000_000 ... 10_500_000).contains(hd24))
        let uhd30 = SpatialBitrate.bitsPerSecond(eyeWidth: 3840, eyeHeight: 2160, frameRate: 30)
        #expect(abs(Double(uhd30) - Double(hd24) * 4 * 30 / 24) < 2)
        #expect(SpatialBitrate.bitsPerSecond(eyeWidth: 1920, eyeHeight: 1080, frameRate: 24, quality: .high) > hd24)
    }

    @Test func neverFarAboveTheSource() {
        let capped = SpatialBitrate.bitsPerSecond(eyeWidth: 1920, eyeHeight: 1080, frameRate: 24, sourceBitsPerSecond: 4_000_000)
        #expect(capped == 5_000_000)
        // --quality max ignores the source.
        #expect(SpatialBitrate.bitsPerSecond(eyeWidth: 1920, eyeHeight: 1080, frameRate: 24, quality: .max,
                                             sourceBitsPerSecond: 4_000_000) > 5_000_000)
        #expect(SpatialBitrate.bitsPerSecond(eyeWidth: 64, eyeHeight: 64, frameRate: 24) == 1_000_000)
    }
}
