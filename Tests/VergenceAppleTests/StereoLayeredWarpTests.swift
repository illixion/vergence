#if canImport(Metal)
import Foundation
import Metal
import Testing
import VergenceCore
@testable import VergenceApple

private let sceneWidth = 960, sceneHeight = 540

/// A subject's outline across a textured background: the coat in
/// docs/investigations/ai-3d-coat-stairstepping-2026-10-29.md, as a soft diagonal edge whose depth
/// edge is a little softer than its colour edge, as a depth model's is. Lengths are in pixels of a
/// 960-wide picture and grow with `scale`; the depth map stays as small as a model's is.
struct SyntheticScene {
    /// The picture is 960 × `scale` wide.
    var scale: Float = 1
    /// What lengths grow with, `scale` unless set: a wider picture needn't have wider edges.
    var detail: Float?
    var foregroundOnLeft = true
    var foreground = SIMD3<Float>(220, 190, 40)
    var background = SIMD3<Float>(30, 90, 200)
    var foregroundDepth: Float = 0.85
    var backgroundDepth: Float = 0.15
    var slope: Float = 0.5
    /// Logistic scales, in picture pixels (a 90–10 % width is 4.39 of them).
    var colorScale: Float = 1.2
    var depthScale: Float = 1.5
    /// How far the depth edge sits from the colour edge, in picture pixels.
    var depthShift: Float = 0
    var stripes = true
    /// A bar this wide centred on the edge, in place of the half plane behind it.
    var barWidth: Float?
    var strength: Float = 0.016
    var budget = DepthBudget.legacy

    var width: Int { Int((Float(sceneWidth) * scale).rounded()) }
    var height: Int { Int((Float(sceneHeight) * scale).rounded()) }
    var lengths: Float { detail ?? scale }
    var pixelsPerDepth: Float { strength * Float(width) }

    func edge(atRow y: Float) -> Float { Float(width) / 2 + slope * (y - Float(height) / 2) }

    private func share(_ x: Float, _ y: Float, scale softness: Float, shift: Float) -> Float {
        let softness = softness * lengths
        let centre = edge(atRow: y) + shift
        if let barWidth {
            func rise(_ u: Float) -> Float { 1 / (1 + exp(-u / softness)) }
            return rise(x - centre + barWidth * lengths / 2) - rise(x - centre - barWidth * lengths / 2)
        }
        let u = (x - centre) / softness
        return foregroundOnLeft ? 1 / (1 + exp(u)) : 1 / (1 + exp(-u))
    }

    func picture(depthSize: (Int, Int) = (448, 252)) -> LayeredPicture {
        var bgra = [UInt8](repeating: 255, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let f = share(Float(x) + 0.5, Float(y) + 0.5, scale: colorScale, shift: 0)
                var back = background
                if stripes { back.y += 25 * sin(2 * .pi * Float(x) / (11 * lengths)) }
                let color = foreground * f + back * (1 - f)
                let i = (y * width + x) * 4
                bgra[i] = UInt8(max(0, min(255, color.z.rounded())))
                bgra[i + 1] = UInt8(max(0, min(255, color.y.rounded())))
                bgra[i + 2] = UInt8(max(0, min(255, color.x.rounded())))
            }
        }
        var depth = [Float](repeating: 0, count: depthSize.0 * depthSize.1)
        for j in 0..<depthSize.1 {
            for i in 0..<depthSize.0 {
                let x = (Float(i) + 0.5) * Float(width) / Float(depthSize.0)
                let y = (Float(j) + 0.5) * Float(height) / Float(depthSize.1)
                let f = share(x, y, scale: depthScale, shift: depthShift * lengths)
                depth[j * depthSize.0 + i] = backgroundDepth + (foregroundDepth - backgroundDepth) * f
            }
        }
        return LayeredPicture(width: width, height: height, bgra: bgra, depthWidth: depthSize.0,
                              depthHeight: depthSize.1, depth: depth, budget: budget)
    }
}

/// Where the red channel crosses its 90, 50 and 10 % levels along each row of the band, scanning
/// from the subject's side, and how straight the 50 % crossings run.
struct Silhouette {
    var width: Float
    var position: Float
    var contour: Float

    init?(_ bgra: [UInt8], pictureWidth: Int, rows: Range<Int>, around edge: (Int) -> Float, reach: Int,
          foregroundOnLeft: Bool, foregroundRed: Float, backgroundRed: Float) {
        var widths: [Float] = [], crossings: [(Float, Float)] = []
        for y in rows {
            let lo = max(1, Int(edge(y)) - reach), hi = min(pictureWidth - 2, Int(edge(y)) + reach)
            let order = foregroundOnLeft ? Array(lo...hi) : Array((lo...hi).reversed())
            func share(_ x: Int) -> Float {
                (Float(bgra[(y * pictureWidth + x) * 4 + 2]) - backgroundRed) / (foregroundRed - backgroundRed)
            }
            func crossing(_ level: Float) -> Float? {
                for k in 1..<order.count where share(order[k]) < level && share(order[k - 1]) >= level {
                    let a = share(order[k - 1]), b = share(order[k])
                    let t = (a - level) / (a - b)
                    return Float(order[k - 1]) + t * Float(order[k] - order[k - 1]) + 0.5
                }
                return nil
            }
            guard let top = crossing(0.9), let middle = crossing(0.5), let bottom = crossing(0.1) else {
                return nil
            }
            widths.append(abs(bottom - top))
            crossings.append((Float(y), middle))
        }
        let n = Float(crossings.count)
        let meanY = crossings.reduce(0) { $0 + $1.0 } / n, meanX = crossings.reduce(0) { $0 + $1.1 } / n
        let slope = crossings.reduce(0) { $0 + ($1.0 - meanY) * ($1.1 - meanX) }
            / crossings.reduce(0) { $0 + ($1.0 - meanY) * ($1.0 - meanY) }
        let residual = crossings.reduce(Float(0)) { sum, c in
            let fit = meanX + slope * (c.0 - meanY)
            return sum + (c.1 - fit) * (c.1 - fit)
        }
        width = widths.reduce(0, +) / n
        position = meanX
        contour = (residual / n).squareRoot()
    }
}

@Suite("Layered silhouettes", .serialized)
struct StereoLayeredWarpTests {
    private func measure(_ bytes: [UInt8], scene: SyntheticScene) -> Silhouette? {
        let rows = Int(150 * scene.scale)..<Int(390 * scene.scale)
        return Silhouette(bytes, pictureWidth: scene.width, rows: rows,
                          around: { scene.edge(atRow: Float($0)) },
                          reach: Int(40 * scene.lengths), foregroundOnLeft: scene.foregroundOnLeft,
                          foregroundRed: scene.foreground.x, backgroundRed: scene.background.x)
    }

    private func largestDifference(_ a: [UInt8], _ b: [UInt8]) -> Int {
        zip(a, b).reduce(0) { max($0, abs(Int($1.0) - Int($1.1))) }
    }

    /// The picture as a horizontal mirror shows it, its depth flipped with it.
    private func flipped(_ picture: LayeredPicture) -> LayeredPicture {
        var result = picture
        for y in 0..<picture.height {
            for x in 0..<picture.width {
                let from = (y * picture.width + picture.width - 1 - x) * 4, to = (y * picture.width + x) * 4
                for channel in 0..<4 { result.bgra[to + channel] = picture.bgra[from + channel] }
            }
        }
        for y in 0..<picture.depthHeight {
            for x in 0..<picture.depthWidth {
                result.depth[y * picture.depthWidth + x] =
                    picture.depth[y * picture.depthWidth + picture.depthWidth - 1 - x]
            }
        }
        return result
    }

    private func checkOutline(_ scene: SyntheticScene) throws {
        let bench = try LayeredBench()
        let picture = scene.picture()
        let source = try #require(measure(picture.bgra, scene: scene))
        let mesh = try bench.render(picture, strength: scene.strength, layers: false)
        let layered = try bench.render(picture, strength: scene.strength, layers: true)

        for (name, bytes, meshBytes, sign) in [("left", layered.left, mesh.left, Float(1)),
                                               ("right", layered.right, mesh.right, Float(-1))] {
            let eye = try #require(measure(bytes, scene: scene), "\(name) eye lost the outline")
            let expected = source.position + sign * scene.pixelsPerDepth * (scene.foregroundDepth - 1)
            #expect(abs(eye.width - source.width) < 0.4 * source.width,
                    "\(name) width \(eye.width) against the source's \(source.width)")
            #expect(abs(eye.position - expected) < scene.lengths,
                    "\(name) outline at \(eye.position), expected \(expected)")
            #expect(eye.contour < 0.35 * scene.lengths, "\(name) contour RMS \(eye.contour)")

            // The mesh folds one eye's outline to a sliver and stretches the other's; the layers have
            // to be clearly closer to the source than that, or this test measures nothing.
            let connected = try #require(measure(meshBytes, scene: scene), "\(name) mesh lost the outline")
            #expect(abs(eye.width - source.width) < 0.5 * abs(connected.width - source.width),
                    "\(name) width \(eye.width) against the mesh's \(connected.width)")
        }
    }

    @Test("A soft outline keeps its width and its place in both eyes", arguments: [true, false])
    func outlineKeepsWidthAndPlace(foregroundOnLeft: Bool) throws {
        var scene = SyntheticScene()
        scene.foregroundOnLeft = foregroundOnLeft
        try checkOutline(scene)
    }

    // The layers are composed at no more than `maxWidth`, so a wider picture is the same scene in
    // fewer pixels. This one is what the base scene looks like at that width: the same outline and
    // the same shift in composed pixels, not a wider outline and a stronger stereo effect, which
    // asks something else of the layers (they follow the app's pinned strength, not twice it).
    @Test("A picture wider than the layers are composed at keeps it too", arguments: [2.4, 4.0] as [Float])
    func outlineAboveTheCompositionWidth(scale: Float) throws {
        let composed = Float(StereoLayeredWarp.maxWidth) / Float(sceneWidth)
        var scene = SyntheticScene()
        scene.scale = scale
        scene.detail = scale / composed
        scene.strength /= composed
        #expect(scene.width > StereoLayeredWarp.maxWidth)
        try checkOutline(scene)
    }

    enum Inert: CaseIterable, CustomTestStringConvertible {
        case flatDepth, lowContrast, thinBar

        var testDescription: String {
            switch self {
            case .flatDepth: "no step in depth"
            case .lowContrast: "no step in colour"
            case .thinBar: "a bar narrower than the margin"
            }
        }

        var scene: SyntheticScene {
            var scene = SyntheticScene()
            switch self {
            case .flatDepth:
                scene.backgroundDepth = scene.foregroundDepth
            case .lowContrast:
                scene.foreground = scene.background + SIMD3(10, 5, 5)
                scene.stripes = false
            case .thinBar:
                scene.slope = 0
                scene.barWidth = 6
            }
            return scene
        }
    }

    @Test("Where there is nothing it can separate the mesh is left as it is", arguments: Inert.allCases)
    func leavesTheMeshAlone(_ kind: Inert) throws {
        let bench = try LayeredBench()
        let scene = kind.scene
        let picture = scene.picture()
        let mesh = try bench.render(picture, strength: scene.strength, layers: false)
        let layered = try bench.render(picture, strength: scene.strength, layers: true)
        #expect(largestDifference(layered.left, mesh.left) <= 1, "left eye")
        #expect(largestDifference(layered.right, mesh.right) <= 1, "right eye")
    }

    @Test("A mirrored picture is drawn as the flipped picture is")
    func mirrorIsTheFlippedPicture() throws {
        let bench = try LayeredBench()
        let scene = SyntheticScene()
        let picture = scene.picture()
        let mirrored = try bench.render(picture, strength: scene.strength, mirror: true, layers: true)
        let reference = try bench.render(flipped(picture), strength: scene.strength, layers: true)
        #expect(largestDifference(mirrored.left, reference.left) <= 6, "left eye")
        #expect(largestDifference(mirrored.right, reference.right) <= 6, "right eye")

        // Agreeing would mean little if the layers weren't drawn in the mirrored picture at all.
        let mesh = try bench.render(picture, strength: scene.strength, mirror: true, layers: false)
        #expect(largestDifference(mirrored.left, mesh.left) > 40, "left eye is the mesh's")
        #expect(largestDifference(mirrored.right, mesh.right) > 40, "right eye is the mesh's")
    }
}
#endif
