import Foundation
import Testing
@testable import VergenceCore

@Suite("VR projections")
struct VRProjectionTests {
    @Test("Equirectangular: straight ahead is the middle, edges are the longitude limits")
    func equirectangular() throws {
        let p = VRProjection.vr180
        let ahead = try #require(p.eyeCoordinate(of: SIMD3(0, 0, -1)))
        #expect(abs(ahead.x - 0.5) < 1e-9 && abs(ahead.y - 0.5) < 1e-9)
        let right = try #require(p.eyeCoordinate(of: SIMD3(1, 0, 0)))
        #expect(abs(right.x - 1) < 1e-9)
        let up = try #require(p.eyeCoordinate(of: SIMD3(0, 1, 0)))
        #expect(abs(up.y) < 1e-9)
        #expect(p.eyeCoordinate(of: SIMD3(0, 0, 1)) == nil, "behind is outside 180°")
        let behind = try #require(VRProjection.vr360.eyeCoordinate(of: SIMD3(0, 0, 1)))
        #expect(abs(abs(behind.x - 0.5) - 0.5) < 1e-9, "behind is the 360° seam")
    }

    @Test("Fisheye: angle from ahead is the distance from the ellipse centre")
    func fisheye() throws {
        let ellipse = EyeEllipse(centerX: 0.5, centerY: 0.5, radiusX: 0.4, radiusY: 0.45)
        let p = VRProjection.fisheye(fieldOfView: 180, ellipse: ellipse)
        let right = try #require(p.eyeCoordinate(of: SIMD3(1, 0, 0)))
        #expect(abs(right.x - 0.9) < 1e-9 && abs(right.y - 0.5) < 1e-9)
        let up = try #require(p.eyeCoordinate(of: SIMD3(0, 1, 0)))
        #expect(abs(up.y - 0.05) < 1e-9, "up is towards the top (smaller v)")
        let halfway = try #require(p.eyeCoordinate(of: SIMD3(sin(.pi / 4), 0, -cos(.pi / 4))))
        #expect(abs(halfway.x - 0.7) < 1e-9, "45° of 90° is halfway out")
        #expect(p.eyeCoordinate(of: SIMD3(0, 0, 1)) == nil)
        let wide = VRProjection.fisheye(200)
        #expect(wide.eyeCoordinate(of: SIMD3(sin(.pi * 0.55), 0, -cos(.pi * 0.55))) != nil, "200° reaches past the side")
    }

    @Test("The mesh faces inward and covers only what the projection holds")
    func meshWinding() throws {
        for projection in [VRProjection.vr180, .vr360, .fisheye(190)] {
            let mesh = try #require(VRSphereMesh.make(for: projection, detail: 24))
            #expect(mesh.positions.count == mesh.eyeCoordinates.count)
            #expect(mesh.indices.count % 3 == 0)
            var inward = 0, outward = 0
            for t in stride(from: 0, to: mesh.indices.count, by: 3) {
                let a = mesh.positions[Int(mesh.indices[t])], b = mesh.positions[Int(mesh.indices[t + 1])]
                let c = mesh.positions[Int(mesh.indices[t + 2])]
                let n = cross(b - a, c - a)
                let area = (n * n).sum()
                guard area > 1e-12 else { continue } // a pole's degenerate sliver
                if (n * (a + b + c)).sum() < 0 { inward += 1 } else { outward += 1 }
            }
            #expect(outward == 0, "\(projection.label): \(outward) triangles face outward")
            #expect(inward > 0, "\(projection.label): inward \(inward) outward \(outward)")
            #expect(mesh.eyeCoordinates.allSatisfy { $0.x >= -1e-5 && $0.x <= 1 + 1e-5 && $0.y >= -1e-5 && $0.y <= 1 + 1e-5 })
        }
        #expect(VRSphereMesh.make(for: .flat) == nil)
    }

    @Test("Interpolating the fisheye per vertex stays within a pixel of a 4K eye")
    func fisheyeInterpolationError() throws {
        let projection = VRProjection.fisheye(fieldOfView: 200, ellipse: EyeEllipse(radiusX: 0.48, radiusY: 0.5))
        let mesh = try #require(VRSphereMesh.make(for: projection))
        var worst = 0.0
        for t in stride(from: 0, to: mesh.indices.count, by: 3 * 7) {
            let i = (0 ..< 3).map { Int(mesh.indices[t + $0]) }
            let mid = (mesh.positions[i[0]] + mesh.positions[i[1]] + mesh.positions[i[2]]) / 3
            let interpolated = (mesh.eyeCoordinates[i[0]] + mesh.eyeCoordinates[i[1]] + mesh.eyeCoordinates[i[2]]) / 3
            guard let exact = projection.eyeCoordinate(of: SIMD3<Double>(mid)) else { continue }
            let error = SIMD2<Double>(interpolated) - exact
            worst = max(worst, (error * error).sum().squareRoot())
        }
        #expect(worst * 4096 < 1.5, "worst \(worst * 4096) px")
    }

    private func cross(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> SIMD3<Float> {
        SIMD3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)
    }
}

@Suite("VR format detection")
struct VRFormatDetectionTests {
    private let sbs = PackedStereoLayout(arrangement: .sideBySide, isHalfResolution: false)
    private let ou = PackedStereoLayout(arrangement: .overUnder, isHalfResolution: false)

    @Test("DeoVR-style names")
    func names() {
        #expect(VRFormatDetector.format(fromWords: "Clip_180_LR") == VRFormat(layout: sbs, projection: .vr180))
        #expect(VRFormatDetector.format(fromWords: "Clip_360_TB") == VRFormat(layout: ou, projection: .vr360))
        #expect(VRFormatDetector.format(fromWords: "Clip_FISHEYE190_LR") == VRFormat(layout: sbs, projection: .fisheye(190)))
        #expect(VRFormatDetector.format(fromWords: "clip MKX200 3dh") == VRFormat(layout: sbs, projection: .fisheye(200)))
        #expect(VRFormatDetector.format(fromWords: "Clip_180x180_3dv") == VRFormat(layout: ou, projection: .vr180))
        #expect(VRFormatDetector.format(fromWords: "Film.3D.HSBS")?.projection == .flat)
        #expect(VRFormatDetector.format(fromWords: "Holiday 2024") == nil)
        #expect(VRFormatDetector.format(fromWords: "houses") == nil)
    }

    @Test("Tags combine: projection from one, packing from another, fisheye most specific")
    func tags() {
        #expect(VRFormatDetector.format(fromTags: ["VR180", "SBS"]) == VRFormat(layout: sbs, projection: .vr180))
        #expect(VRFormatDetector.format(fromTags: ["180", "Fisheye", "OU"]) == VRFormat(layout: ou, projection: .fisheye(180)))
        #expect(VRFormatDetector.format(fromTags: ["Outdoor"]) == nil)
    }

    @Test("Packing from the frame's shape when nothing names it")
    func shape() {
        #expect(VRFormatDetector.detect(fileName: "Clip_180.mp4", frameWidth: 5760, frameHeight: 2880)?.layout == sbs)
        #expect(VRFormatDetector.detect(fileName: "Clip_180.mp4", frameWidth: 2880, frameHeight: 2880)?.layout == nil)
        #expect(VRFormatDetector.detect(fileName: "Clip_360.mp4", frameWidth: 4096, frameHeight: 4096)?.layout == ou)
        #expect(VRFormatDetector.detect(fileName: "Clip_360.mp4", frameWidth: 4096, frameHeight: 2048)?.layout == nil)
        #expect(VRFormatDetector.detect(fileName: "fisheye.mp4", frameWidth: 2048, frameHeight: 4096)?.layout == ou)
        #expect(VRFormatDetector.detect(tags: ["VR"], fileName: "x.mp4", frameWidth: 3840, frameHeight: 1920) == nil
                || VRFormatDetector.detect(tags: ["VR"], fileName: "x.mp4", frameWidth: 3840, frameHeight: 1920)?.projection == .flat)
    }

    @Test("Declared metadata beats tags beats the name; a fitted ellipse refines a fisheye")
    func precedence() {
        let declared = VRFormat(layout: ou, projection: .vr360)
        let found = VRFormatDetector.detect(declared: declared, tags: ["180", "SBS"], fileName: "x_fisheye_lr.mp4")
        #expect(found == declared)
        let ellipse = EyeEllipse(centerX: 0.5, centerY: 0.5, radiusX: 0.42, radiusY: 0.5)
        let fisheye = VRFormatDetector.detect(fileName: "x_fisheye190_lr.mp4", fittedEllipse: ellipse)
        #expect(fisheye?.projection == .fisheye(fieldOfView: 190, ellipse: ellipse))
    }

    @Test("Fits the picture ellipse of a fisheye eye and refuses a full-frame picture")
    func ellipseFit() throws {
        let w = 400, h = 480
        var pixels = [UInt8](repeating: 2, count: w * h)
        // A tall oval: centre (200, 240), radii 160 × 230.
        for y in 0 ..< h { for x in 0 ..< w {
            let dx = Double(x - 200) / 160, dy = Double(y - 240) / 230
            if dx * dx + dy * dy <= 1 { pixels[y * w + x] = 120 }
        } }
        let ellipse = try #require(FisheyeEllipseFit.fit(LumaPlane(width: w, height: h, pixels: pixels)))
        #expect(abs(ellipse.centerX - 0.5) < 0.01 && abs(ellipse.centerY - 0.5) < 0.01)
        #expect(abs(ellipse.radiusX - 160.0 / 400) < 0.01)
        #expect(abs(ellipse.radiusY - 230.0 / 480) < 0.01)
        let full = LumaPlane(width: w, height: h, pixels: [UInt8](repeating: 120, count: w * h))
        #expect(FisheyeEllipseFit.fit(full) == nil)
    }
}
