#if canImport(Metal)
import Foundation
import Testing
import VergenceCore
@testable import VergenceApple

/// Ground-truth check for optional background texture continuation.
/// The analytic scene is independent of the reconstruction: a soft opaque disc over stripes,
/// with known constant layer depths, so each eye's correct image is two simple translations.
@Suite("Directional background continuation", .serialized)
struct StereoDirectionalBackgroundTests {
    @Test(arguments: [Float(0), .pi / 6, .pi / 3, .pi / 2])
    func preservesStripesAcrossSoftDepth(angle: Float) throws {
        let width = 960, height = 540, depthWidth = 448, depthHeight = 252
        let strength: Float = 0.016, foregroundDepth: Float = 0.78, backgroundDepth: Float = 0.2
        let foreground = SIMD3<Float>(95, 65, 45)
        func alpha(_ x: Float, _ y: Float, softness: Float) -> Float {
            let radius = hypot(x - 480, y - 240)
            return 1 / (1 + exp(min(60, max(-60, (radius - 95) / softness))))
        }
        func background(_ x: Float, _ y: Float) -> SIMD3<Float> {
            let stripe = 30 * sin((x * cos(angle) + y * sin(angle)) * 2 * .pi / 24)
            return SIMD3(90 + stripe, 100 + stripe, 105 + stripe)
        }
        func code(_ value: Float) -> UInt8 { UInt8(min(255, max(0, value.rounded()))) }
        var pixels = [UInt8](repeating: 255, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let a = alpha(Float(x), Float(y), softness: 0.9)
                let color = foreground * a + background(Float(x), Float(y)) * (1 - a)
                let at = (y * width + x) * 4
                pixels[at] = code(color.z); pixels[at + 1] = code(color.y); pixels[at + 2] = code(color.x)
            }
        }
        var depth = [Float](repeating: 0, count: depthWidth * depthHeight)
        for y in 0..<depthHeight {
            for x in 0..<depthWidth {
                let a = alpha((Float(x) + 0.5) * Float(width) / Float(depthWidth) - 0.5,
                              (Float(y) + 0.5) * Float(height) / Float(depthHeight) - 0.5, softness: 6)
                depth[y * depthWidth + x] = backgroundDepth + (foregroundDepth - backgroundDepth) * a
            }
        }
        let picture = LayeredPicture(width: width, height: height, bgra: pixels,
                                     depthWidth: depthWidth, depthHeight: depthHeight, depth: depth, budget: .studio)
        let bench = try LayeredBench()
        let eyes = try bench.render(picture, strength: strength, layers: true, dense: true, directional: true)
        for (eye, sign) in [(eyes.left, Float(1)), (eyes.right, Float(-1))] {
            let foregroundShift = sign * strength * Float(width) * (DepthBudget.studio.apply(foregroundDepth) - 1)
            let backgroundShift = sign * strength * Float(width) * (DepthBudget.studio.apply(backgroundDepth) - 1)
            var squared: Double = 0
            var channels = 0
            for y in 0..<height {
                for x in 0..<width {
                    let radius = hypot(Float(x - 480), Float(y - 240))
                    guard radius > 70, radius < 125 else { continue }
                    let a = alpha(Float(x) - foregroundShift, Float(y), softness: 0.9)
                    let expected = foreground * a + background(Float(x) - backgroundShift, Float(y)) * (1 - a)
                    let at = (y * width + x) * 4
                    for (channel, value) in [(0, expected.z), (1, expected.y), (2, expected.x)] {
                        let difference = Double(Int(eye[at + channel]) - Int(code(value)))
                        squared += difference * difference; channels += 1
                    }
                }
            }
            let rms = sqrt(squared / Double(channels))
            #expect(rms < 5, "Both eyes should preserve the stripe phase and foreground coverage: RMS \(rms) code values")
        }
    }
}
#endif
