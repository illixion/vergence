#if canImport(CoreGraphics)
import CoreGraphics
#endif
import Foundation
import Testing
@testable import VergenceCore

/// Procedural scenes: a textured background and textured ellipses at
/// different depths, seen by two eyes.
private struct Scene {
    struct Object {
        var cx, cy, rx, ry: Double
        /// d = x(left eye) - x(right eye), in pixels.
        var disparity: Double
        var texture: Noise
    }

    /// Value noise, several octaves, bilinear. Periodic across `width` when
    /// asked (a 360° panorama's seam).
    struct Noise {
        struct Octave { var cell: Double; var amplitude: Double; var columns: Int; var rows: Int; var grid: [Double] }
        var octaves: [Octave]
        var offset: Double
        var period: Double?

        init(seed: UInt64, width: Int, height: Int, periodic: Bool) {
            var state = seed &* 0x9E37_79B9_7F4A_7C15 | 1
            func next() -> Double {
                state ^= state << 13; state ^= state >> 7; state ^= state << 17
                return Double(state >> 11) / Double(1 << 53) * 2 - 1
            }
            let span = periodic ? Double(width) : Double(width + 96)
            octaves = [(24.0, 55.0), (9, 40), (4, 32), (2, 22)].map { cell, amplitude in
                let columns = max(1, Int((span / cell).rounded()))
                let rows = Int(Double(height) / cell) + 2
                return Octave(cell: span / Double(columns), amplitude: amplitude, columns: columns, rows: rows,
                              grid: (0 ..< columns * rows).map { _ in next() })
            }
            offset = periodic ? 0 : 48
            period = periodic ? Double(width) : nil
        }

        func value(_ x: Double, _ y: Double) -> Double {
            var x = x + offset
            if let period { x = x.truncatingRemainder(dividingBy: period); if x < 0 { x += period } }
            var sum = 128.0
            for o in octaves {
                let gx = x / o.cell, gy = max(0, y) / o.cell
                let x0 = Int(gx.rounded(.down)), y0 = min(Int(gy), o.rows - 2)
                let fx = gx - Double(x0), fy = gy - Double(y0)
                func g(_ i: Int, _ j: Int) -> Double {
                    o.grid[(j * o.columns) + ((i % o.columns) + o.columns) % o.columns]
                }
                let top = g(x0, y0) * (1 - fx) + g(x0 + 1, y0) * fx
                let bottom = g(x0, y0 + 1) * (1 - fx) + g(x0 + 1, y0 + 1) * fx
                sum += o.amplitude * (top * (1 - fy) + bottom * fy)
            }
            return sum
        }
    }

    var width: Int
    var height: Int
    var background: Noise
    var backgroundDisparity: Double
    var objects: [Object]

    /// `vr`: a parallel rig (background at infinity, objects in front);
    /// otherwise a converged film (most of it behind the screen).
    init(width: Int, height: Int, seed: UInt64, vr: Bool, periodic: Bool = false) {
        self.width = width
        self.height = height
        let scale = Double(width) / 192
        background = Noise(seed: seed, width: width, height: height, periodic: periodic)
        backgroundDisparity = vr ? 0 : -4 * scale
        let w = Double(width), h = Double(height)
        let specs: [(Double, Double, Double, Double, Double)] = vr
            ? [(0.3, 0.4, 0.16, 0.2, 3), (0.65, 0.55, 0.14, 0.22, 6), (0.45, 0.75, 0.12, 0.12, 9)]
            : [(0.3, 0.4, 0.14, 0.18, -2), (0.68, 0.5, 0.12, 0.2, -3), (0.5, 0.78, 0.07, 0.08, 3)]
        objects = specs.enumerated().map { i, s in
            Object(cx: s.0 * w, cy: s.1 * h, rx: s.2 * w, ry: s.3 * h, disparity: s.4 * scale,
                   texture: Noise(seed: seed &+ UInt64(i + 1) &* 977, width: width, height: height, periodic: false))
        }
    }

    /// One eye: +1 left, -1 right.
    func eye(_ side: Double, fisheye: Bool = false) -> [UInt8] {
        var pixels = [UInt8](repeating: 0, count: width * height)
        for y in 0 ..< height {
            for x in 0 ..< width {
                let px = Double(x) + 0.5, py = Double(y) + 0.5
                if fisheye {
                    let u = (px / Double(width) - 0.5) / 0.5, v = (py / Double(height) - 0.5) / 0.5
                    if u * u + v * v > 1 { pixels[y * width + x] = 3; continue }
                }
                var value: Double?
                for o in objects {
                    let sx = px - side * o.disparity / 2
                    let u = (sx - o.cx) / o.rx, v = (py - o.cy) / o.ry
                    if u * u + v * v <= 1 { value = o.texture.value(sx, py); break }
                }
                let v = value ?? background.value(px - side * backgroundDisparity / 2, py)
                pixels[y * width + x] = UInt8(max(24, min(240, v)))
            }
        }
        return pixels
    }

    func plane() -> LumaPlane { LumaPlane(width: width, height: height, pixels: eye(1)) }

    func packed(_ arrangement: PackedStereoLayout.Arrangement, swapped: Bool = false, fisheye: Bool = false) -> LumaPlane {
        var a = eye(1, fisheye: fisheye), b = eye(-1, fisheye: fisheye)
        if swapped { swap(&a, &b) }
        switch arrangement {
        case .sideBySide:
            var pixels = [UInt8]()
            for y in 0 ..< height {
                pixels += a[y * width ..< (y + 1) * width]
                pixels += b[y * width ..< (y + 1) * width]
            }
            return LumaPlane(width: width * 2, height: height, pixels: pixels)
        case .overUnder:
            return LumaPlane(width: width, height: height * 2, pixels: a + b)
        }
    }
}

@Suite("Video format analysis")
struct VideoFormatAnalysisTests {
    static func frames(_ count: Int = 3, seed: UInt64, _ make: (UInt64) -> LumaPlane) -> [LumaPlane] {
        (0 ..< count).map { make(seed &+ UInt64($0) &* 31) }
    }

    static func size(_ w: Int, _ h: Int) -> CGSize { CGSize(width: w, height: h) }

    // MARK: Flat 3D

    @Test("Full side by side, eyes in order and swapped")
    func fullSideBySide() {
        for swapped in [false, true] {
            let frames = Self.frames(seed: 1) { Scene(width: 192, height: 108, seed: $0, vr: false).packed(.sideBySide, swapped: swapped) }
            let result = VideoFormatAnalyzer.analyze(frames: frames, frameSize: Self.size(3840, 1080))
            #expect(result.format == VRFormat(layout: PackedStereoLayout(arrangement: .sideBySide, isHalfResolution: false,
                                                                         eyesReversed: swapped), projection: .flat), "\(result.reasons)")
            #expect(result.layoutSource == .picture && result.eyeOrderSource == .picture)
            #expect(result.confidence >= 0.7)
            #expect(result.isStereo && !result.isVR)
        }
    }

    @Test("Half side by side is told by the frame's shape")
    func halfSideBySide() {
        let frames = Self.frames(seed: 2) { Scene(width: 160, height: 180, seed: $0, vr: false).packed(.sideBySide) }
        let result = VideoFormatAnalyzer.analyze(frames: frames, frameSize: Self.size(1920, 1080))
        #expect(result.format.layout == PackedStereoLayout(arrangement: .sideBySide, isHalfResolution: true), "\(result.reasons)")
        #expect(result.format.projection == .flat)
        #expect(result.eyeOrderSource == .picture)
    }

    @Test("Full over-under, eyes swapped")
    func fullOverUnder() {
        let frames = Self.frames(seed: 3) { Scene(width: 192, height: 108, seed: $0, vr: false).packed(.overUnder, swapped: true) }
        let result = VideoFormatAnalyzer.analyze(frames: frames, frameSize: Self.size(1920, 2160))
        #expect(result.format == VRFormat(layout: PackedStereoLayout(arrangement: .overUnder, isHalfResolution: false,
                                                                     eyesReversed: true), projection: .flat), "\(result.reasons)")
    }

    // MARK: VR

    @Test("180° fisheye over-under: two black-cornered circles, either eye order")
    func fisheyeOverUnder() throws {
        for swapped in [false, true] {
            let frames = Self.frames(seed: 4) { Scene(width: 160, height: 160, seed: $0, vr: true).packed(.overUnder, swapped: swapped, fisheye: true) }
            let result = VideoFormatAnalyzer.analyze(frames: frames, frameSize: Self.size(1920, 3840))
            guard case .fisheye(let fov, let ellipse) = result.format.projection else {
                Issue.record("not fisheye: \(result.reasons)"); continue
            }
            #expect(fov == 190)
            #expect(abs(ellipse.radiusX - 0.5) < 0.05 && abs(ellipse.centerY - 0.5) < 0.05)
            #expect(result.format.layout == PackedStereoLayout(arrangement: .overUnder, isHalfResolution: false, eyesReversed: swapped),
                    "\(result.reasons)")
            #expect(result.projectionSource == .picture && result.eyeOrderSource == .picture)
            #expect(result.confidence >= 0.7)
        }
    }

    @Test("180° equirectangular side by side (2:1 frame), either eye order")
    func equirect180SideBySide() {
        for swapped in [false, true] {
            let frames = Self.frames(seed: 5) { Scene(width: 192, height: 192, seed: $0, vr: true).packed(.sideBySide, swapped: swapped) }
            let result = VideoFormatAnalyzer.analyze(frames: frames, frameSize: Self.size(5760, 2880))
            #expect(result.format == VRFormat(layout: PackedStereoLayout(arrangement: .sideBySide, isHalfResolution: false,
                                                                         eyesReversed: swapped), projection: .vr180), "\(result.reasons)")
            #expect(result.projectionSource == .shape)
            #expect(result.confidence >= 0.7)
        }
    }

    @Test("180° equirectangular over-under (1:2 frame)")
    func equirect180OverUnder() {
        let frames = Self.frames(seed: 6) { Scene(width: 160, height: 160, seed: $0, vr: true).packed(.overUnder) }
        let result = VideoFormatAnalyzer.analyze(frames: frames, frameSize: Self.size(2880, 5760))
        #expect(result.format == VRFormat(layout: PackedStereoLayout(arrangement: .overUnder, isHalfResolution: false),
                                          projection: .vr180), "\(result.reasons)")
        #expect(result.eyeOrderSource == .picture)
    }

    @Test("360° over-under (1:1 frame) with a seamless panorama, either eye order")
    func equirect360OverUnder() {
        for swapped in [false, true] {
            let frames = Self.frames(seed: 7) {
                Scene(width: 256, height: 128, seed: $0, vr: true, periodic: true).packed(.overUnder, swapped: swapped)
            }
            let result = VideoFormatAnalyzer.analyze(frames: frames, frameSize: Self.size(5760, 5760))
            #expect(result.format == VRFormat(layout: PackedStereoLayout(arrangement: .overUnder, isHalfResolution: false,
                                                                         eyesReversed: swapped), projection: .vr360), "\(result.reasons)")
            #expect(result.projectionSource == .picture)
            #expect(result.confidence >= 0.7)
        }
    }

    @Test("Mono 360° (2:1 frame) whose edges wrap")
    func mono360() {
        let frames = Self.frames(seed: 8) { Scene(width: 320, height: 160, seed: $0, vr: true, periodic: true).plane() }
        let result = VideoFormatAnalyzer.analyze(frames: frames, frameSize: Self.size(7680, 3840))
        #expect(result.format == VRFormat(layout: nil, projection: .vr360), "\(result.reasons)")
        #expect(result.isVR && !result.isStereo)
        #expect(result.confidence >= 0.7)
    }

    // MARK: 2D

    @Test("Ordinary 2D, 16:9 and 9:16, letterboxed, and a slide, stays 2D")
    func ordinary2D() {
        let landscape = Self.frames(seed: 9) { Scene(width: 320, height: 180, seed: $0, vr: false).plane() }
        let portrait = Self.frames(seed: 10) { Scene(width: 180, height: 320, seed: $0, vr: false).plane() }
        let letterboxed = Self.frames(seed: 11) { seed in
            let picture = Scene(width: 320, height: 134, seed: seed, vr: false).plane()
            let bar = [UInt8](repeating: 2, count: 320 * 23)
            return LumaPlane(width: 320, height: 180, pixels: bar + picture.pixels + bar)
        }
        let wide = Self.frames(seed: 12) { Scene(width: 320, height: 160, seed: $0, vr: false).plane() }
        let cases: [(String, [LumaPlane], CGSize)] = [
            ("16:9", landscape, Self.size(1920, 1080)), ("9:16", portrait, Self.size(1080, 1920)),
            ("letterboxed", letterboxed, Self.size(1920, 1080)), ("slide", [Self.slide(seed: 13)], Self.size(1920, 1080)),
            ("2:1 not a panorama", wide, Self.size(3840, 1920)),
        ]
        for (name, frames, size) in cases {
            let result = VideoFormatAnalyzer.analyze(frames: frames, frameSize: size)
            #expect(result.format == VRFormat(layout: nil, projection: .flat), "\(name): \(result.reasons)")
            #expect(result.confidence >= 0.7, "\(name)")
            #expect(result.eyeOrderSource == .none)
        }
    }

    /// Lines of "text" across a light background: rows line up between the
    /// halves, the glyphs don't.
    static func slide(seed: UInt64) -> LumaPlane {
        let w = 320, h = 180
        var pixels = [UInt8](repeating: 232, count: w * h)
        var state = seed &* 0x9E37_79B9_7F4A_7C15 | 1
        func next(_ n: Int) -> Int {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17
            return Int(state >> 33) % n
        }
        for line in stride(from: 24, to: h - 20, by: 18) {
            var x = 16
            let end = w - 16 - next(60)
            while x < end {
                let glyph = 3 + next(5)
                for y in line ..< line + 9 where next(5) > 0 {
                    for gx in x ..< min(x + glyph, end) { pixels[y * w + gx] = 30 }
                }
                x += glyph + 1 + next(3) + (next(6) == 0 ? 5 : 0)
            }
        }
        return LumaPlane(width: w, height: h, pixels: pixels)
    }

    @Test("Halves that share only coarse shapes (same blobs, different grain) are not 3D")
    func coarselySimilarHalves() {
        var state: UInt64 = 0x1234_5678_9ABC_DEF1
        var pixels = [UInt8]()
        for y in 0 ..< 180 {
            for _ in 0 ..< 2 {
                for x in 0 ..< 160 {
                    state ^= state << 13; state ^= state >> 7; state ^= state << 17
                    let grain = 0.35 * (Double(state >> 56) - 128)
                    let blobs = 70 * sin(Double(x) / 13) * cos(Double(y) / 17)
                    pixels.append(UInt8(max(0, min(255, 128 + blobs + grain))))
                }
            }
        }
        let frame = LumaPlane(width: 320, height: 180, pixels: pixels)
        #expect(StereoLayoutDetector.scores([frame]).sideBySide >= StereoLayoutDetector.stereoThreshold,
                "the whole-half comparison alone would call it side by side")
        let result = VideoFormatAnalyzer.analyze(frames: [frame], frameSize: Self.size(1920, 1080))
        #expect(result.format == VRFormat(layout: nil, projection: .flat), "\(result.reasons)")
    }

    @Test("Black frames are no evidence: 2D, low confidence")
    func black() {
        let black = LumaPlane(width: 320, height: 180, pixels: [UInt8](repeating: 16, count: 320 * 180))
        let result = VideoFormatAnalyzer.analyze(frames: [black, black], frameSize: Self.size(1920, 1080))
        #expect(result.format == VRFormat(layout: nil, projection: .flat))
        #expect(result.confidence < 0.5)
        #expect(result.projectionSource == .none && result.layoutSource == .none)
    }

    // MARK: Precedence

    @Test("The picture's packing beats the name's")
    func pictureBeatsName() {
        let frames = Self.frames(seed: 14) { Scene(width: 192, height: 108, seed: $0, vr: false).packed(.sideBySide) }
        let result = VideoFormatAnalyzer.analyze(frames: frames, frameSize: Self.size(3840, 1080), fileName: "Film.3D.OU.mkv")
        #expect(result.format.layout?.arrangement == .sideBySide, "\(result.reasons)")
        #expect(result.layoutSource == .picture)
        #expect(result.confidence <= 0.7)
    }

    @Test("A name gives the projection; its packing comes from the frame's shape without a picture")
    func nameWithoutPicture() {
        let result = VideoFormatAnalyzer.analyze(frames: [], frameSize: Self.size(5760, 2880), fileName: "Clip_MKX200.mp4")
        #expect(result.format == VRFormat(layout: PackedStereoLayout(arrangement: .sideBySide, isHalfResolution: false),
                                          projection: .fisheye(200)))
        #expect(result.projectionSource == .name && result.layoutSource == .shape && result.eyeOrderSource == .none)
        let reversed = VideoFormatAnalyzer.analyze(frames: [], frameSize: Self.size(5760, 2880), fileName: "Clip_180_RL.mp4")
        #expect(reversed.format.layout == PackedStereoLayout(arrangement: .sideBySide, isHalfResolution: false, eyesReversed: true))
        #expect(reversed.eyeOrderSource == .name)
    }

    @Test("Declared metadata wins over everything")
    func declaredWins() throws {
        let frames = Self.frames(seed: 15) { Scene(width: 192, height: 108, seed: $0, vr: false).packed(.sideBySide) }
        let declared = VRFormat(layout: PackedStereoLayout(arrangement: .overUnder, isHalfResolution: true), projection: .vr360)
        let result = VideoFormatAnalyzer.analyze(frames: frames, frameSize: Self.size(3840, 1080), fileName: "x_180_LR.mp4",
                                                 tags: ["Fisheye"], declared: declared)
        #expect(result.format == declared)
        #expect(result.confidence == 1)
        #expect(result.projectionSource == .container && result.layoutSource == .container)
        let decoded = try JSONDecoder().decode(VideoFormatAnalysis.self, from: JSONEncoder().encode(result))
        #expect(decoded == result)
    }
}
