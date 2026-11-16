#if canImport(Metal)
import Foundation
import Testing
import VergenceCore
@testable import VergenceApple

/// The picture's sides and the viewer's offset, on the real shaders.
///
/// Each eye's image is shifted sideways, so at one side it has nothing to
/// show. That strip used to be hidden by fading the depth to nothing over the
/// outer 5 % of the width, which bends a far background up to the glass in a
/// band down each side: one eye's band squeezed, the other's stretched. The
/// crop draws the pass a little larger instead, so the shift is the same
/// right up to the edge.
@Suite("Stereo edge crop and viewer offset", .serialized)
struct StereoEdgeCropTests {
    private static let width = 400, height = 60
    private static let strength: Float = 0.02

    /// A horizontal ramp (so a pixel's value says which source column it
    /// shows) over a flat depth: 0 is the far end, the whole shift.
    private func picture(depth: Float = 0) -> LayeredPicture {
        var bgra = [UInt8](repeating: 255, count: Self.width * Self.height * 4)
        for y in 0..<Self.height {
            for x in 0..<Self.width {
                let v = UInt8((40 + 200 * (Float(x) + 0.5) / Float(Self.width)).rounded())
                let i = (y * Self.width + x) * 4
                bgra[i] = v; bgra[i + 1] = v; bgra[i + 2] = v
            }
        }
        return LayeredPicture(width: Self.width, height: Self.height, bgra: bgra, depthWidth: 64, depthHeight: 16,
                              depth: [Float](repeating: depth, count: 64 * 16))
    }

    /// The source column each pixel of the middle row shows, nil where the eye is empty (black).
    private func columns(_ eye: [UInt8]) -> [Float?] {
        let row = Self.height / 2
        return (0..<Self.width).map { x in
            let v = Float(eye[(row * Self.width + x) * 4])
            return v < 20 ? nil : (v - 40) / 200 * Float(Self.width) - 0.5
        }
    }

    /// How far the picture is shifted at a column: positive = drawn to the right of its source.
    private func shift(_ columns: [Float?], at x: Int) -> Float? { columns[x].map { Float(x) - $0 } }

    @Test("Without the fade or the crop, each eye has an empty strip at one side")
    func stripsNeedHiding() throws {
        let eyes = try LayeredBench().render(picture(), strength: Self.strength, layers: false, cropsEdges: false)
        let left = columns(eyes.left), right = columns(eyes.right)
        // Far content: the left eye's image moves left, the right eye's right.
        #expect(left[Self.width - 3] == nil)
        #expect(right[2] == nil)
        #expect(left[2] != nil && right[Self.width - 3] != nil)
    }

    @Test("The fade bends the sides: the shift there is not the shift in the middle")
    func fadeBendsTheSides() throws {
        let eyes = try LayeredBench().render(picture(), strength: Self.strength, layers: false)
        let left = columns(eyes.left)
        let middle = try #require(shift(left, at: Self.width / 2))
        // The right side, where the left eye's image moves inward.
        let side = try #require(shift(left, at: Self.width - 5))
        #expect(abs(middle + Self.strength * Float(Self.width)) < 1.5)
        #expect(abs(side) < abs(middle) * 0.5)
    }

    @Test("The crop leaves no empty strip and the same picture scale right up to the sides")
    func cropKeepsTheSidesStraight() throws {
        let eyes = try LayeredBench().render(picture(), strength: Self.strength, layers: false, cropsEdges: true)
        let zoom = StereoEdge.zoom(strength: Self.strength, convergence: 1, viewerReach: 0)
        for (name, eye, sign) in [("left", eyes.left, Float(1)), ("right", eyes.right, Float(-1))] {
            let shown = columns(eye)
            #expect(!shown.contains { $0 == nil }, "\(name) eye has an empty column")
            // Every column shows the source the enlarged, shifted picture puts there.
            let centre = Float(Self.width) / 2
            for x in stride(from: 4, to: Self.width - 4, by: 7) {
                let expected = centre + (Float(x) + 0.5 - centre) / zoom + sign * Self.strength * Float(Self.width) - 0.5
                let got = try #require(shown[x])
                #expect(abs(got - expected) < 1.6, "\(name) eye column \(x): \(got) vs \(expected)")
            }
        }
    }

    @Test("A viewer one half eye spacing to the right sees the left eye's picture where the middle was")
    func viewerOffsetMovesBothEyes() throws {
        let bench = try LayeredBench()
        let centred = try bench.render(picture(), strength: Self.strength, layers: false, cropsEdges: false)
        let moved = try bench.render(picture(), strength: Self.strength, layers: false, viewerOffset: 1, cropsEdges: false)
        let mid = Self.width / 2
        // The viewer's left eye is now where the middle of the head was: no shift at all.
        #expect(abs(try #require(shift(columns(moved.left), at: mid))) < 1)
        // The right eye is a whole eye spacing from there: twice the shift it had.
        let was = try #require(shift(columns(centred.right), at: mid))
        let now = try #require(shift(columns(moved.right), at: mid))
        #expect(abs(now - 2 * was) < 1.5)
        // Far content follows the head: it moved right for a viewer who moved right.
        #expect(now > was)
    }

    @Test("The crop is sized for the farthest the viewer's offset can take an eye")
    func cropCoversTheViewerReach() throws {
        let eyes = try LayeredBench().render(picture(), strength: Self.strength, layers: false,
                                             viewerOffset: 1.5, cropsEdges: true)
        #expect(!columns(eyes.left).contains { $0 == nil })
        #expect(!columns(eyes.right).contains { $0 == nil })
    }
}
#endif
