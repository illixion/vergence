#if os(macOS)
import CoreImage
import Foundation
import ImageIO
import Metal
import Testing
@testable import VergenceApple
import VergenceCore

/// Explicitly invoked by scripts/still-stereo-lab.sh. Personal pictures and
/// outputs stay outside the repository; ordinary test runs use synthetic data.
struct StillStereoLabTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["VERGENCE_STILL_IMAGE"] != nil))
    func exportComparison() throws {
        let env = ProcessInfo.processInfo.environment
        let imageURL = URL(fileURLWithPath: try #require(env["VERGENCE_STILL_IMAGE"]))
        let modelURL = URL(fileURLWithPath: try #require(env["VERGENCE_STILL_MODEL"]))
        let output = URL(fileURLWithPath: try #require(env["VERGENCE_STILL_OUTPUT"]))
        let requested = Double(env["VERGENCE_STILL_SIZE"] ?? "2048") ?? 2048
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let start = Date()
        let src = try #require(CGImageSourceCreateWithURL(imageURL as CFURL,
            [kCGImageSourceShouldCache: false] as CFDictionary))
        let properties = try #require(CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any])
        let nativeWidth = try #require(properties[kCGImagePropertyPixelWidth] as? Int)
        let nativeHeight = try #require(properties[kCGImagePropertyPixelHeight] as? Int)
        let size = StillStereoResolution.longSide(width: nativeWidth, height: nativeHeight, requested: requested)
        let image = try #require(CGImageSourceCreateThumbnailAtIndex(src, 0, [
            kCGImageSourceThumbnailMaxPixelSize: size,
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceShouldCacheImmediately: true] as CFDictionary))
        let decodeSeconds = Date().timeIntervalSince(start)
        let device = try #require(MTLCreateSystemDefaultDevice())
        let provider = try #require(CoreMLDepthProvider(device: device, modelURL: modelURL))
        let inferenceStart = Date()
        let depth = try #require(provider.stillDepth(for: image, preservePrecision: true))
        let inferenceSeconds = Date().timeIntervalSince(inferenceStart)
        let shaderURL = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Sources/VergenceApple/Stereo/StereoShaders.metal")
        let library = try device.makeLibrary(source: String(contentsOf: shaderURL, encoding: .utf8), options: nil)
        let renderer = try #require(StillStereoRenderer(device: device, library: library))
        let context = CIContext(mtlDevice: device)
        let srgb = CGColorSpace(name: CGColorSpace.sRGB)!
        func write(_ ci: CIImage, name: String) throws {
            let cg = try #require(context.createCGImage(ci, from: ci.extent, format: .RGBA8, colorSpace: srgb))
            let dest = try #require(CGImageDestinationCreateWithURL(output.appendingPathComponent(name + ".png") as CFURL,
                "public.png" as CFString, 1, nil))
            CGImageDestinationAddImage(dest, cg, nil)
            #expect(CGImageDestinationFinalize(dest))
        }
        func oriented(_ texture: MTLTexture, colorSpace: CGColorSpace) throws -> CIImage {
            let ci = try #require(CIImage(mtlTexture: texture, options: [.colorSpace: colorSpace]))
            return ci.transformed(by: CGAffineTransform(translationX: 0, y: CGFloat(texture.height)).scaledBy(x: 1, y: -1))
        }
        try write(CIImage(cgImage: image), name: "source")
        var renderTimes: [String: Double] = [:]
        for label in ["raw", "refined", "separated"] {
            let refined = label != "raw"
            let began = Date()
            let pair = try renderer.render(image: image, depth: depth, refine: refined, separatesBoundaries: label == "separated")
            renderTimes[label] = Date().timeIntervalSince(began)
            let left = try oriented(pair.left, colorSpace: pair.colorSpace)
            let right = try oriented(pair.right, colorSpace: pair.colorSpace)
            try write(left, name: label + "-left")
            try write(right, name: label + "-right")
            let red = left.applyingFilter("CIColorMatrix", parameters: [
                "inputRVector": CIVector(x: 1, y: 0, z: 0, w: 0),
                "inputGVector": CIVector(x: 0, y: 0, z: 0, w: 0),
                "inputBVector": CIVector(x: 0, y: 0, z: 0, w: 0)])
            let cyan = right.applyingFilter("CIColorMatrix", parameters: [
                "inputRVector": CIVector(x: 0, y: 0, z: 0, w: 0)])
            try write(red.applyingFilter("CIAdditionCompositing", parameters: [kCIInputBackgroundImageKey: cyan]),
                name: label + "-anaglyph")
            let d = try oriented(pair.depth, colorSpace: CGColorSpaceCreateDeviceGray())
            try write(d, name: label + "-depth")
        }
        var usage = rusage()
        getrusage(RUSAGE_SELF, &usage)
        let report: [String: Any] = ["nativeWidth": nativeWidth, "nativeHeight": nativeHeight,
            "width": image.width, "height": image.height, "requestedLongSide": requested,
            "depthWidth": depth.width, "depthHeight": depth.height, "model": provider.modelName,
            "decodeSeconds": decodeSeconds, "inferenceSeconds": inferenceSeconds,
            "renderSeconds": renderTimes, "processPeakResidentBytes": usage.ru_maxrss]
        let json = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        try json.write(to: output.appendingPathComponent("report.json"))
        print(String(decoding: json, as: UTF8.self))
    }
}
#endif
