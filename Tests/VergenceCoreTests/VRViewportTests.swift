import Foundation
import Testing
@testable import VergenceCore

struct VRViewportTests {
    @Test func centreLooksWhereTheViewIsTurned() {
        let straight = VRViewport().direction(u: 0, v: 0)
        #expect(abs(straight.x) < 1e-9 && abs(straight.y) < 1e-9 && straight.z < 0)
        let right = VRViewport(yawDegrees: 30).direction(u: 0, v: 0)
        let longitude = atan2(right.x, -right.z) * 180 / .pi
        #expect(abs(longitude - 30) < 1e-6)
        let up = VRViewport(pitchDegrees: 20).direction(u: 0, v: 0)
        let latitude = atan2(up.y, -up.z) * 180 / .pi
        #expect(abs(latitude - 20) < 1e-6)
        // The panel's right edge looks right, its top edge up.
        let edge = VRViewport(fieldOfViewDegrees: 90).direction(u: 0.5, v: 0)
        #expect(abs(atan2(edge.x, -edge.z) * 180 / .pi - 45) < 1e-6)
        #expect(VRViewport().direction(u: 0, v: 0.5).y > 0)
    }

    @Test func panelSamplesThePictureAlongItsRays() throws {
        let mesh = try #require(VRSphereMesh.viewport(VRViewport(yawDegrees: 30), projection: .vr180, detail: 4))
        #expect(mesh.positions.count == 25)
        #expect(mesh.indices.count == 4 * 4 * 6)
        // The centre vertex (row 2, column 2) lands 30° right of centre: u = (30 + 90) / 180.
        let centre = mesh.eyeCoordinates[2 * 5 + 2]
        #expect(abs(centre.x - Float(120.0 / 180)) < 1e-4)
        #expect(abs(centre.y - 0.5) < 1e-4)
        // Top row samples above the horizon (v smaller), left column left of the centre.
        #expect(mesh.eyeCoordinates[2].y < 0.5)
        #expect(mesh.eyeCoordinates[2 * 5].x < centre.x)
        let fisheye = try #require(VRSphereMesh.viewport(VRViewport(), projection: .fisheye(180), detail: 4))
        let middle = fisheye.eyeCoordinates[2 * 5 + 2]
        #expect(abs(middle.x - 0.5) < 1e-4 && abs(middle.y - 0.5) < 1e-4)
        #expect(VRSphereMesh.viewport(VRViewport(), projection: .flat) == nil)
    }

    @Test func clampingKeepsTheViewInsideThePicture() {
        let view = VRViewport(yawDegrees: 170, pitchDegrees: 89, fieldOfViewDegrees: 90).clamped(to: .vr180)
        #expect(view.yawDegrees == 45)
        #expect(view.pitchDegrees == 45)
        let around = VRViewport(yawDegrees: 200).clamped(to: .vr360)
        #expect(abs(around.yawDegrees - -160) < 1e-9)
        // Nothing past the picture's edge is sampled in a clamped view.
        let mesh = VRSphereMesh.viewport(view, projection: .vr180, detail: 8)!
        #expect(mesh.eyeCoordinates.allSatisfy { $0.x >= -1e-4 && $0.x <= 1 + 1e-4 })
    }
}
