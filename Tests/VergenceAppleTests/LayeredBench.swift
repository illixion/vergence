#if canImport(Metal)
import CoreGraphics
import Foundation
import ImageIO
import Metal
import UniformTypeIdentifiers
import VergenceCore
@testable import VergenceApple

struct LayeredBenchError: Error, CustomStringConvertible {
    let description: String
}

/// A picture and its depth map, as the pump hands them to the warp.
struct LayeredPicture {
    var width: Int
    var height: Int
    /// 8-bit BGRA, premultiplied, `width * height * 4` bytes.
    var bgra: [UInt8]
    var depthWidth: Int
    var depthHeight: Int
    var depth: [Float]
    var uvScale = SIMD2<Float>(1, 1)
    var uvOffset = SIMD2<Float>(0, 0)
    var valueScale: Float = 1
    var valueBias: Float = 0
    var budget = DepthBudget.legacy
}

struct LayeredEyes {
    let width: Int
    let height: Int
    /// 8-bit BGRA.
    let left: [UInt8]
    let right: [UInt8]
    /// What the GPU spent on the whole command buffer: the split (when layered) and both eyes.
    let gpuMilliseconds: Double
}

/// Drives the real shaders the way `StereoPump` does: one command buffer, the layers split once,
/// then each eye's mesh pass with the patch drawn into it. The shaders are compiled from source,
/// because SwiftPM's CLI leaves out Metal resources, so under `swift test` the shipped library
/// doesn't exist.
struct LayeredBench {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let mesh: StereoMeshWarp
    let layered: StereoLayeredWarp

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw LayeredBenchError(description: "no Metal device")
        }
        self.device = device
        self.queue = queue
        let path = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Sources/VergenceApple/Stereo/StereoShaders.metal")
        let library = try device.makeLibrary(source: String(contentsOf: path, encoding: .utf8), options: nil)
        guard let resources = StereoWarpResources(device: device, library: library),
              let layered = StereoLayeredWarp(device: device, library: library, pixelFormat: .bgra8Unorm,
                                              sampleCount: resources.msaaSampleCount) else {
            throw LayeredBenchError(description: "renderer unavailable")
        }
        mesh = StereoMeshWarp(resources: resources)
        self.layered = layered
    }

    func render(_ picture: LayeredPicture, strength: Float, convergence: Float = 1, mirror: Bool = false,
                layers: Bool, dense: Bool = false,
                tone: SIMD3<Float> = SIMD3(0, 1, 1)) throws -> LayeredEyes {
        func texture(_ format: MTLPixelFormat, _ width: Int, _ height: Int,
                     usage: MTLTextureUsage) throws -> MTLTexture {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: width,
                                                                      height: height, mipmapped: false)
            descriptor.storageMode = .shared
            descriptor.usage = usage
            guard let made = device.makeTexture(descriptor: descriptor) else {
                throw LayeredBenchError(description: "texture \(width)x\(height) unavailable")
            }
            return made
        }
        let source = try texture(.bgra8Unorm, picture.width, picture.height, usage: .shaderRead)
        source.replace(region: MTLRegionMake2D(0, 0, picture.width, picture.height), mipmapLevel: 0,
                       withBytes: picture.bgra, bytesPerRow: picture.width * 4)
        let depthTexture = try texture(.r32Float, picture.depthWidth, picture.depthHeight, usage: .shaderRead)
        depthTexture.replace(region: MTLRegionMake2D(0, 0, picture.depthWidth, picture.depthHeight),
                             mipmapLevel: 0, withBytes: picture.depth, bytesPerRow: picture.depthWidth * 4)
        let frameDepth = PumpFrameDepth(texture: depthTexture, uvScale: picture.uvScale,
                                        uvOffset: picture.uvOffset, valueScale: picture.valueScale,
                                        valueBias: picture.valueBias, budget: picture.budget)
        let targets = try (0..<2).map { _ in
            try texture(.bgra8Unorm, picture.width, picture.height,
                        usage: [.renderTarget, .shaderRead, .shaderWrite])
        }

        guard let command = queue.makeCommandBuffer() else {
            throw LayeredBenchError(description: "no command buffer")
        }
        let split = layers && layered.prepare(source: source, depth: frameDepth, strength: strength,
                                              convergence: convergence, commandBuffer: command)
        for (eye, sign) in [(0, Float(1)), (1, Float(-1))] {
            let uniforms = VideoStereoUniforms(
                brightness: tone.x, contrast: tone.y, saturation: tone.z, depthStrength: strength,
                convergence: convergence, eyeSign: sign, mirror: mirror ? 1 : 0, useDepth: 1,
                depthUVScale: picture.uvScale, depthUVOffset: picture.uvOffset,
                depthValueScale: picture.valueScale, depthValueBias: picture.valueBias,
                depthBudget: picture.budget.knees)
            let drawn = mesh.encode(
                source: source, depth: depthTexture, destination: targets[eye], uniforms: uniforms,
                dense: dense, eye: eye, commandBuffer: command,
                overlay: split ? { encoder in
                    layered.encodePatch(into: encoder, eyeSign: sign, mirror: mirror, brightness: tone.x,
                                        contrast: tone.y, saturation: tone.z, extendedRange: false)
                } : nil)
            if !drawn { throw LayeredBenchError(description: "mesh pass \(eye) failed") }
        }
        command.commit()
        command.waitUntilCompleted()
        if command.status != .completed {
            throw LayeredBenchError(description: "command failed: \(String(describing: command.error))")
        }
        func bytes(_ texture: MTLTexture) -> [UInt8] {
            var result = [UInt8](repeating: 0, count: picture.width * picture.height * 4)
            texture.getBytes(&result, bytesPerRow: picture.width * 4,
                             from: MTLRegionMake2D(0, 0, picture.width, picture.height), mipmapLevel: 0)
            return result
        }
        return LayeredEyes(width: picture.width, height: picture.height,
                           left: bytes(targets[0]), right: bytes(targets[1]),
                           gpuMilliseconds: (command.gpuEndTime - command.gpuStartTime) * 1000)
    }

    /// One channel of a private layer texture as floats, for the lab's debug maps. Call after `render`.
    func plane(of texture: MTLTexture, channel: Int) throws -> [Float] {
        let shared = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: texture.pixelFormat,
            width: texture.width, height: texture.height, mipmapped: false)
        shared.storageMode = .shared
        guard let copy = device.makeTexture(descriptor: shared),
              let command = queue.makeCommandBuffer(), let blit = command.makeBlitCommandEncoder() else {
            throw LayeredBenchError(description: "readback unavailable")
        }
        blit.copy(from: texture, to: copy)
        blit.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        let (channels, bytesPerChannel): (Int, Int)
        switch texture.pixelFormat {
        case .rgba32Float: (channels, bytesPerChannel) = (4, 4)
        case .rgba16Float: (channels, bytesPerChannel) = (4, 2)
        case .rg16Float: (channels, bytesPerChannel) = (2, 2)
        default: throw LayeredBenchError(description: "no readback for \(texture.pixelFormat)")
        }
        let rowBytes = texture.width * channels * bytesPerChannel
        var raw = [UInt8](repeating: 0, count: rowBytes * texture.height)
        copy.getBytes(&raw, bytesPerRow: rowBytes,
                      from: MTLRegionMake2D(0, 0, texture.width, texture.height), mipmapLevel: 0)
        var result = [Float](repeating: 0, count: texture.width * texture.height)
        raw.withUnsafeBytes { buffer in
            for i in 0..<result.count {
                let offset = (i * channels + channel) * bytesPerChannel
                result[i] = bytesPerChannel == 4
                    ? buffer.loadUnaligned(fromByteOffset: offset, as: Float.self)
                    : Self.float(fromHalf: buffer.loadUnaligned(fromByteOffset: offset, as: UInt16.self))
            }
        }
        return result
    }

    static func float(fromHalf bits: UInt16) -> Float {
        let sign: Float = bits & 0x8000 != 0 ? -1 : 1
        let exponent = Int((bits >> 10) & 0x1F)
        let fraction = Float(bits & 0x3FF)
        if exponent == 0 { return sign * fraction * 5.9604645e-8 }
        if exponent == 31 { return fraction == 0 ? sign * .infinity : .nan }
        return sign * (1 + fraction / 1024) * Float(sign: .plus, exponent: exponent - 15, significand: 1)
    }
}

// MARK: - Files

enum LayeredFiles {
    /// 8-bit premultiplied BGRA in sRGB, the way a pump frame arrives.
    static func loadPicture(_ url: URL) throws -> (width: Int, height: Int, bgra: [UInt8]) {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw LayeredBenchError(description: "cannot read \(url.path)")
        }
        let (width, height) = (image.width, image.height)
        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        let drawn = bytes.withUnsafeMutableBytes { buffer -> Bool in
            guard let context = CGContext(
                data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8,
                bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
            else { return false }
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard drawn else { throw LayeredBenchError(description: "cannot decode \(url.path)") }
        return (width, height, bytes)
    }

    /// A capture directory: `source.png`, `depth.f32` (little-endian float32, top-left origin) and
    /// the `depth.json` that describes how the pump held that depth.
    static func loadCapture(_ directory: URL) throws -> LayeredPicture {
        let (width, height, preview) = try loadPicture(directory.appendingPathComponent("source.png"))
        let rawSource = directory.appendingPathComponent("source.bgra")
        let bgra = FileManager.default.fileExists(atPath: rawSource.path)
            ? [UInt8](try Data(contentsOf: rawSource)) : preview
        guard bgra.count == width * height * 4 else {
            throw LayeredBenchError(description: "source.bgra has the wrong byte count")
        }
        let info = try JSONSerialization.jsonObject(
            with: Data(contentsOf: directory.appendingPathComponent("depth.json"))) as? [String: Any] ?? [:]
        func number(_ key: String) throws -> Float {
            guard let value = info[key] as? NSNumber else {
                throw LayeredBenchError(description: "depth.json has no \(key)")
            }
            return value.floatValue
        }
        func pair(_ key: String) throws -> SIMD2<Float> {
            guard let values = info[key] as? [NSNumber], values.count == 2 else {
                throw LayeredBenchError(description: "depth.json has no \(key)")
            }
            return SIMD2(values[0].floatValue, values[1].floatValue)
        }
        let depthWidth = Int(try number("width")), depthHeight = Int(try number("height"))
        let data = try Data(contentsOf: directory.appendingPathComponent("depth.f32"))
        guard data.count == depthWidth * depthHeight * 4 else {
            throw LayeredBenchError(description: "depth.f32 is \(data.count) bytes, expected \(depthWidth * depthHeight * 4)")
        }
        let depth = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        let knees = try pair("depthBudget")
        return LayeredPicture(
            width: width, height: height, bgra: bgra, depthWidth: depthWidth, depthHeight: depthHeight,
            depth: depth, uvScale: try pair("uvScale"), uvOffset: try pair("uvOffset"),
            valueScale: try number("valueScale"), valueBias: try number("valueBias"),
            budget: DepthBudget(kneeLo: knees.x, kneeHi: knees.y))
    }

    static func writePNG(bgra: [UInt8], width: Int, height: Int, to url: URL) throws {
        var bytes = bgra
        try bytes.withUnsafeMutableBytes { buffer in
            guard let context = CGContext(
                    data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8,
                    bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                    bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue),
                  let image = context.makeImage(),
                  let destination = CGImageDestinationCreateWithURL(
                    url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
                throw LayeredBenchError(description: "cannot write \(url.path)")
            }
            CGImageDestinationAddImage(destination, image, nil)
            guard CGImageDestinationFinalize(destination) else {
                throw LayeredBenchError(description: "cannot finish \(url.path)")
            }
        }
    }

    /// A 0…1 plane as a grey picture.
    static func writePlane(_ values: [Float], width: Int, height: Int, to url: URL) throws {
        var bytes = [UInt8](repeating: 255, count: width * height * 4)
        for i in 0..<(width * height) {
            let v = UInt8(max(0, min(1, values[i])) * 255 + 0.5)
            bytes[i * 4] = v
            bytes[i * 4 + 1] = v
            bytes[i * 4 + 2] = v
        }
        try writePNG(bgra: bytes, width: width, height: height, to: url)
    }
}
#endif
