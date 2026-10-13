#if canImport(Metal)
import CoreVideo
import Foundation
import Metal
import simd
import Testing
@testable import VergenceApple
@testable import VergenceCore

/// Silhouette fidelity of the realtime refiner's output, as the warp reads it.
/// The warp samples depth bilinearly, so the displaced silhouette sits where
/// that bilinear field crosses halfway between subject and background. A
/// hard-stepped map at model resolution puts that crossing on depth texel
/// boundaries (~3.7 video px at 1080p) and a smooth edge stairsteps; the
/// guided upsample has to pull it back onto the image edge.
@Suite("Realtime depth edges", .serialized)
struct RealtimeDepthEdgeTests {
    static let videoWidth = 1920, videoHeight = 1080
    static let depthWidth = 518, depthHeight = 392
    // A disc in video pixels: its edge sweeps every slope.
    static let center = SIMD2<Float>(960, 540)
    static let radius: Float = 300

    @Test func refinedSilhouetteFollowsTheImageEdge() throws {
        guard let metal = VergenceMetal.shared else { return }
        let device = metal.device
        let queue = try #require(device.makeCommandQueue())
        var cache: CVMetalTextureCache?
        CVMetalTextureCacheCreate(nil, nil, device, nil, &cache)
        let textureCache = try #require(cache)
        let refiner = try #require(RealtimeDepthRefiner(device: device, commandQueue: queue, textureCache: textureCache))

        let frame = try #require(Self.discFrame())
        let raw = try #require(Self.discDepth(device: device))
        var out: MTLTexture?
        // A few frames, so the lock and the EMA are past their first-frame path.
        for i in 0..<4 {
            out = refiner.refine(raw: raw, frame: frame, frameSeconds: Double(i) / 30)
        }
        let depth = try #require(out)
        let (w, h, values) = try #require(Self.read(depth, device: device, queue: queue))

        let uv = CoreMLDepthProvider.letterboxUVTransform(
            videoWidth: Self.videoWidth, videoHeight: Self.videoHeight, depthWidth: w, depthHeight: h
        )
        // Bilinear read at a video pixel, the way the warp samples the map.
        func sample(_ x: Float, _ y: Float) -> Float {
            let u = (x + 0.5) / Float(Self.videoWidth) * uv.scale.x + uv.offset.x
            let v = (y + 0.5) / Float(Self.videoHeight) * uv.scale.y + uv.offset.y
            let fx = u * Float(w) - 0.5, fy = v * Float(h) - 0.5
            let x0 = max(0, min(w - 1, Int(fx.rounded(.down)))), y0 = max(0, min(h - 1, Int(fy.rounded(.down))))
            let x1 = min(w - 1, x0 + 1), y1 = min(h - 1, y0 + 1)
            let tx = max(0, min(1, fx - Float(x0))), ty = max(0, min(1, fy - Float(y0)))
            let a = values[y0 * w + x0] * (1 - tx) + values[y0 * w + x1] * tx
            let b = values[y1 * w + x0] * (1 - tx) + values[y1 * w + x1] * tx
            return a * (1 - ty) + b * ty
        }
        let fg = sample(Self.center.x, Self.center.y)
        let bg = sample(Self.center.x + Self.radius * 1.5, Self.center.y)
        let half = (fg + bg) / 2
        #expect(fg > bg)

        // Right-hand edge, every video row across the disc's middle band
        // (slopes from vertical to 60°): the crossing's distance from the
        // true circle.
        var errors: [Float] = []
        let span = Int(Self.radius * 0.85)
        for dy in -span...span {
            let y = Self.center.y + Float(dy)
            let truth = Self.center.x + (Self.radius * Self.radius - Float(dy * dy)).squareRoot()
            var x = truth - 20
            var previous = sample(x, y)
            var crossing: Float?
            while x < truth + 20 {
                let next = sample(x + 0.25, y)
                if previous >= half, next < half {
                    crossing = x + 0.25 * (previous - half) / max(previous - next, 1e-6)
                    break
                }
                previous = next
                x += 0.25
            }
            errors.append(abs((crossing ?? (truth + 20)) - truth))
        }
        let rms = (errors.reduce(0) { $0 + $1 * $1 } / Float(errors.count)).squareRoot()
        let worst = errors.max() ?? 0
        // Stairstepping is the row-to-row part of the error: what's left after
        // a 9-row moving average (a steady offset reads as a slightly
        // smaller subject, not as steps).
        let wobble = (4..<(errors.count - 4)).map { i -> Float in
            let local = errors[(i - 4)...(i + 4)].reduce(0, +) / 9
            return errors[i] - local
        }
        let wobbleRMS = (wobble.reduce(0) { $0 + $1 * $1 } / Float(wobble.count)).squareRoot()
        print(String(format: "realtime silhouette error vs image edge: rms %.2f px, worst %.2f px, row-to-row wobble %.2f px (map %dx%d)",
                     rms, worst, wobbleRMS, w, h))
        // At 1× the crossing wanders across a whole depth texel (~3.7 px).
        #expect(rms < 1.0)
        #expect(worst < 2.0)
        #expect(wobbleRMS < 0.5)
    }

    /// The model draws outlines soft: depth ramps from subject to background
    /// over several of its texels. The warp stretches the picture across that
    /// whole ramp, so the edge snap must turn it back into a step on the
    /// image edge (and leave it alone when switched off).
    @Test func softOutlineIsSnappedToTheImageEdge() throws {
        guard let metal = VergenceMetal.shared else { return }
        // The snap alone: the gap curve would squeeze the disc's step too.
        let compressed = DepthRefinement.compressesGaps
        DepthRefinement.compressesGaps = false
        defer { DepthRefinement.compressesGaps = compressed }
        let device = metal.device
        let queue = try #require(device.makeCommandQueue())
        var cache: CVMetalTextureCache?
        CVMetalTextureCacheCreate(nil, nil, device, nil, &cache)
        let textureCache = try #require(cache)
        // Low contrast, as a subject against a similar background is in film:
        // the refiner's luma-guided bilateral sharpens a high-contrast edge by
        // itself, so only a faint one shows what the snap adds.
        let frame = try #require(Self.discFrame(inside: 0.5, outside: 0.42))
        // Ramp over 6 model texels (~22 video px), centred on the edge.
        let raw = try #require(Self.discDepth(device: device, rampTexels: 6))

        func rampWidth(snapping: Bool) throws -> (width: Float, centre: Float) {
            let saved = DepthRefinement.snapsEdges
            DepthRefinement.snapsEdges = snapping
            defer { DepthRefinement.snapsEdges = saved }
            let refiner = try #require(RealtimeDepthRefiner(device: device, commandQueue: queue, textureCache: textureCache))
            var out: MTLTexture?
            for i in 0..<4 { out = refiner.refine(raw: raw, frame: frame, frameSeconds: Double(i) / 30) }
            let depth = try #require(out)
            let (w, h, values) = try #require(Self.read(depth, device: device, queue: queue))
            let uv = CoreMLDepthProvider.letterboxUVTransform(
                videoWidth: Self.videoWidth, videoHeight: Self.videoHeight, depthWidth: w, depthHeight: h
            )
            // Along the middle row, video px → nearest depth texel.
            func at(_ x: Float) -> Float {
                let u = (x + 0.5) / Float(Self.videoWidth) * uv.scale.x + uv.offset.x
                let v = (Self.center.y + 0.5) / Float(Self.videoHeight) * uv.scale.y + uv.offset.y
                let i = max(0, min(w - 1, Int(u * Float(w)))), j = max(0, min(h - 1, Int(v * Float(h))))
                return values[j * w + i]
            }
            let edge = Self.center.x + Self.radius
            let fg = at(Self.center.x), bg = at(edge + 80)
            var hi: Float?, lo: Float?, mid: Float?
            var x = edge - 60
            while x < edge + 60 {
                let t = (at(x) - bg) / max(fg - bg, 1e-6)
                if hi == nil, t < 0.9 { hi = x }
                if mid == nil, t < 0.5 { mid = x }
                if lo == nil, t < 0.1 { lo = x }
                x += 0.5
            }
            return ((lo ?? x) - (hi ?? x), (mid ?? x) - edge)
        }
        let soft = try rampWidth(snapping: false)
        let snapped = try rampWidth(snapping: true)
        print(String(format: "soft outline ramp 10–90%%: %.1f px unsnapped, %.1f px snapped (centre %.1f / %.1f px from the edge)",
                     soft.width, snapped.width, soft.centre, snapped.centre))
        #expect(soft.width > 10)
        #expect(snapped.width < soft.width / 2)
        #expect(abs(snapped.centre) < 3)
    }

    /// A thin near rail on pale ground with dark steps right under it, the
    /// model's depth soft above and below the rail (Movie A's corral). The steps'
    /// top rows must not keep the rail's depth (they shear like a smudge), and
    /// the rail must stay near.
    @Test func stepsUnderANearRailKeepTheirOwnDepth() throws {
        guard let metal = VergenceMetal.shared else { return }
        let compressed = DepthRefinement.compressesGaps
        DepthRefinement.compressesGaps = false
        defer { DepthRefinement.compressesGaps = compressed }
        let device = metal.device
        let queue = try #require(device.makeCommandQueue())
        var cache: CVMetalTextureCache?
        CVMetalTextureCacheCreate(nil, nil, device, nil, &cache)
        let textureCache = try #require(cache)
        let railTop: Float = 500, railBottom: Float = 530
        let steps = (x: Float(900)...Float(1000), y: railBottom...Float(600))
        let frame = try #require(Self.frame { x, y in
            if (railTop..<railBottom).contains(y) { return 0.35 }
            if steps.x.contains(x), steps.y.contains(y) { return 0.12 }
            return 0.7
        })
        let raw = try #require(Self.depth(device: device) { _, y in
            let outside = max(0, max(railTop - y, y - railBottom))
            return 0.2 + 0.8 * max(0, 1 - outside / 30)
        })

        func sample(snapping: Bool) throws -> (stepsTop: Float, railCore: Float) {
            let saved = DepthRefinement.snapsEdges
            DepthRefinement.snapsEdges = snapping
            defer { DepthRefinement.snapsEdges = saved }
            let refiner = try #require(RealtimeDepthRefiner(device: device, commandQueue: queue, textureCache: textureCache))
            var out: MTLTexture?
            for i in 0..<4 { out = refiner.refine(raw: raw, frame: frame, frameSeconds: Double(i) / 30) }
            let depth = try #require(out)
            let (w, h, values) = try #require(Self.read(depth, device: device, queue: queue))
            func at(_ x: Float, _ y: Float) -> Float {
                let i = max(0, min(w - 1, Int(x / Float(Self.videoWidth) * Float(w))))
                let j = max(0, min(h - 1, Int(y / Float(Self.videoHeight) * Float(h))))
                return values[j * w + i]
            }
            let near = at(400, 515), far = at(400, 800)
            func t(_ v: Float) -> Float { (v - far) / max(near - far, 1e-6) }
            var top: Float = 0, n: Float = 0
            for y in stride(from: railBottom + 3, to: railBottom + 18, by: 2) {
                for x in stride(from: Float(920), to: 980, by: 6) { top += t(at(x, y)); n += 1 }
            }
            return (top / n, t(at(950, 515)))
        }
        let soft = try sample(snapping: false)
        let snapped = try sample(snapping: true)
        print(String(format: "steps under a rail, top rows: %.2f of the rail's depth unsnapped, %.2f snapped; rail core %.2f",
                     soft.stepsTop, snapped.stepsTop, snapped.railCore))
        #expect(soft.stepsTop > 0.3)
        #expect(snapped.stepsTop < soft.stepsTop / 2)
        #expect(snapped.railCore > 0.85)
    }

    /// Ground running off to the horizon: depth falls gently from the bottom
    /// of the frame to the top, with dark tufts on pale ground. Not an outline,
    /// so the snap must leave it as it is (it once pushed the tufts far, as
    /// speckles).
    @Test func slopingGroundIsLeftAlone() throws {
        guard let metal = VergenceMetal.shared else { return }
        let compressed = DepthRefinement.compressesGaps
        DepthRefinement.compressesGaps = false
        defer { DepthRefinement.compressesGaps = compressed }
        let device = metal.device
        let queue = try #require(device.makeCommandQueue())
        var cache: CVMetalTextureCache?
        CVMetalTextureCacheCreate(nil, nil, device, nil, &cache)
        let textureCache = try #require(cache)
        let frame = try #require(Self.frame { x, y in
            // Tufts: 12 px dark blobs on a 40 px lattice.
            let fx = x.truncatingRemainder(dividingBy: 40) - 20, fy = y.truncatingRemainder(dividingBy: 40) - 20
            return fx * fx + fy * fy < 36 ? 0.2 : 0.7
        })
        let raw = try #require(Self.depth(device: device) { _, y in 0.1 + 0.9 * y / Float(Self.videoHeight) })

        func values(snapping: Bool) throws -> [Float] {
            let saved = DepthRefinement.snapsEdges
            DepthRefinement.snapsEdges = snapping
            defer { DepthRefinement.snapsEdges = saved }
            let refiner = try #require(RealtimeDepthRefiner(device: device, commandQueue: queue, textureCache: textureCache))
            var out: MTLTexture?
            for i in 0..<4 { out = refiner.refine(raw: raw, frame: frame, frameSeconds: Double(i) / 30) }
            let depth = try #require(out)
            return try #require(Self.read(depth, device: device, queue: queue)).2
        }
        let soft = try values(snapping: false), snapped = try values(snapping: true)
        let lo = soft.min() ?? 0, hi = soft.max() ?? 1
        let worst = zip(soft, snapped).map { abs($0 - $1) }.max() ?? 0
        print(String(format: "sloping ground: snap moved depth by at most %.3f of the range", worst / max(hi - lo, 1e-6)))
        #expect(worst / max(hi - lo, 1e-6) < 0.02)
    }

    /// A dark cap with a lit rim in front of dark trees (Movie A, walking away):
    /// inside the cap the colour matches the trees, but the rim stands between
    /// them, so the snap must not cut a far hole into the cap.
    @Test func darkSubjectBeforeDarkBackgroundKeepsItsInside() throws {
        guard let metal = VergenceMetal.shared else { return }
        let compressed = DepthRefinement.compressesGaps
        DepthRefinement.compressesGaps = false
        defer { DepthRefinement.compressesGaps = compressed }
        let device = metal.device
        let queue = try #require(device.makeCommandQueue())
        var cache: CVMetalTextureCache?
        CVMetalTextureCacheCreate(nil, nil, device, nil, &cache)
        let textureCache = try #require(cache)
        let frame = try #require(Self.frame { x, y in
            let r = simd_length(SIMD2<Float>(x, y) - Self.center)
            if r > Self.radius { return 0.25 }
            return r > Self.radius - 6 ? 0.6 : 0.27
        })
        let raw = try #require(Self.discDepth(device: device, rampTexels: 10))

        func inside(snapping: Bool) throws -> Float {
            let saved = DepthRefinement.snapsEdges
            DepthRefinement.snapsEdges = snapping
            defer { DepthRefinement.snapsEdges = saved }
            let refiner = try #require(RealtimeDepthRefiner(device: device, commandQueue: queue, textureCache: textureCache))
            var out: MTLTexture?
            for i in 0..<4 { out = refiner.refine(raw: raw, frame: frame, frameSeconds: Double(i) / 30) }
            let depth = try #require(out)
            let (w, h, values) = try #require(Self.read(depth, device: device, queue: queue))
            func at(_ x: Float, _ y: Float) -> Float {
                values[max(0, min(h - 1, Int(y / Float(Self.videoHeight) * Float(h)))) * w
                       + max(0, min(w - 1, Int(x / Float(Self.videoWidth) * Float(w))))]
            }
            let near = at(Self.center.x, Self.center.y), far = at(Self.center.x + Self.radius + 120, Self.center.y)
            // The least near texel 8–24 px inside the rim, along the middle row.
            var worst: Float = 1
            for dx in stride(from: Float(14), to: 30, by: 1) {
                worst = min(worst, (at(Self.center.x + Self.radius - dx, Self.center.y) - far) / max(near - far, 1e-6))
            }
            return worst
        }
        let soft = try inside(snapping: false), snapped = try inside(snapping: true)
        print(String(format: "dark cap before dark trees, just inside the rim: %.2f near unsnapped, %.2f snapped", soft, snapped))
        #expect(snapped > soft - 0.05)
    }

    /// A tan glove before tan hills, with the white horse below it (Movie A):
    /// the model's depth is soft across the glove's top, and the glove is
    /// coloured like the hills, not like what is near below it. It must keep
    /// its top.
    @Test func nearObjectColouredLikeTheBackgroundKeepsItsTop() throws {
        guard let metal = VergenceMetal.shared else { return }
        let compressed = DepthRefinement.compressesGaps
        DepthRefinement.compressesGaps = false
        defer { DepthRefinement.compressesGaps = compressed }
        let device = metal.device
        let queue = try #require(device.makeCommandQueue())
        var cache: CVMetalTextureCache?
        CVMetalTextureCacheCreate(nil, nil, device, nil, &cache)
        let textureCache = try #require(cache)
        let gloveTop: Float = 500, gloveBottom: Float = 600
        let frame = try #require(Self.frame { x, y in
            if y < gloveTop { return 0.6 }                                  // hills
            if y < gloveBottom, (800...1100).contains(x) { return 0.62 }    // glove
            return y < gloveBottom ? 0.6 : 0.9                              // hills beside it, horse below
        })
        let raw = try #require(Self.depth(device: device) { x, y in
            let near = (800...1100).contains(x) ? max(0, min(1, (y - gloveTop) / 30 + 0.5)) : (y >= gloveBottom ? 1 : 0)
            return 0.2 + 0.8 * near
        })
        func top(snapping: Bool) throws -> Float {
            let saved = DepthRefinement.snapsEdges
            DepthRefinement.snapsEdges = snapping
            defer { DepthRefinement.snapsEdges = saved }
            let refiner = try #require(RealtimeDepthRefiner(device: device, commandQueue: queue, textureCache: textureCache))
            var out: MTLTexture?
            for i in 0..<4 { out = refiner.refine(raw: raw, frame: frame, frameSeconds: Double(i) / 30) }
            let depth = try #require(out)
            let (w, h, values) = try #require(Self.read(depth, device: device, queue: queue))
            func at(_ x: Float, _ y: Float) -> Float {
                values[max(0, min(h - 1, Int(y / Float(Self.videoHeight) * Float(h)))) * w
                       + max(0, min(w - 1, Int(x / Float(Self.videoWidth) * Float(w))))]
            }
            let near = at(950, 580), far = at(950, 300)
            var sum: Float = 0, n: Float = 0
            for y in stride(from: gloveTop + 2, to: gloveTop + 14, by: 2) {
                for x in stride(from: Float(860), to: 1040, by: 12) { sum += (at(x, y) - far) / max(near - far, 1e-6); n += 1 }
            }
            return sum / n
        }
        let soft = try top(snapping: false), snapped = try top(snapping: true)
        print(String(format: "glove before hills, its top rows: %.2f near unsnapped, %.2f snapped", soft, snapped))
        #expect(snapped >= soft - 0.05)
    }

    /// A full-HD BGRA frame, grey from `value(x, y)` (video pixels).
    static func frame(_ value: (Float, Float) -> Float) -> CVPixelBuffer? {
        var pb: CVPixelBuffer?
        let attrs: [CFString: Any] = [kCVPixelBufferMetalCompatibilityKey: true, kCVPixelBufferIOSurfacePropertiesKey: [:]]
        guard CVPixelBufferCreate(nil, videoWidth, videoHeight, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb) == kCVReturnSuccess,
              let pb else { return nil }
        CVPixelBufferLockBaseAddress(pb, [])
        defer { CVPixelBufferUnlockBaseAddress(pb, []) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return nil }
        let stride = CVPixelBufferGetBytesPerRow(pb)
        for y in 0..<videoHeight {
            let row = base.advanced(by: y * stride).assumingMemoryBound(to: UInt8.self)
            for x in 0..<videoWidth {
                let v = UInt8(max(0, min(1, value(Float(x) + 0.5, Float(y) + 0.5))) * 255)
                row[x * 4] = v; row[x * 4 + 1] = v; row[x * 4 + 2] = v; row[x * 4 + 3] = 255
            }
        }
        return pb
    }

    /// Model-resolution depth from `value(x, y)` in video pixels.
    static func depth(device: MTLDevice, _ value: (Float, Float) -> Float) -> MTLTexture? {
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r16Float, width: depthWidth, height: depthHeight, mipmapped: false)
        desc.usage = [.shaderRead]
        guard let texture = device.makeTexture(descriptor: desc) else { return nil }
        let uv = CoreMLDepthProvider.letterboxUVTransform(
            videoWidth: videoWidth, videoHeight: videoHeight, depthWidth: depthWidth, depthHeight: depthHeight
        )
        var values = [Float16](repeating: 0, count: depthWidth * depthHeight)
        for j in 0..<depthHeight {
            for i in 0..<depthWidth {
                let u = (Float(i) + 0.5) / Float(depthWidth), v = (Float(j) + 0.5) / Float(depthHeight)
                let x = (u - uv.offset.x) / uv.scale.x * Float(videoWidth), y = (v - uv.offset.y) / uv.scale.y * Float(videoHeight)
                values[j * depthWidth + i] = Float16(value(x, y))
            }
        }
        values.withUnsafeBytes {
            texture.replace(region: MTLRegionMake2D(0, 0, depthWidth, depthHeight), mipmapLevel: 0,
                            withBytes: $0.baseAddress!, bytesPerRow: depthWidth * 2)
        }
        return texture
    }

    /// Anti-aliased white disc on grey, BGRA, full HD.
    static func discFrame(inside: Float = 0.95, outside: Float = 0.25) -> CVPixelBuffer? {
        var pb: CVPixelBuffer?
        let attrs: [CFString: Any] = [kCVPixelBufferMetalCompatibilityKey: true, kCVPixelBufferIOSurfacePropertiesKey: [:]]
        guard CVPixelBufferCreate(nil, videoWidth, videoHeight, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb) == kCVReturnSuccess,
              let pb else { return nil }
        CVPixelBufferLockBaseAddress(pb, [])
        defer { CVPixelBufferUnlockBaseAddress(pb, []) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return nil }
        let stride = CVPixelBufferGetBytesPerRow(pb)
        for y in 0..<videoHeight {
            let row = base.advanced(by: y * stride).assumingMemoryBound(to: UInt8.self)
            for x in 0..<videoWidth {
                let d = simd_length(SIMD2<Float>(Float(x) + 0.5, Float(y) + 0.5) - center)
                let coverage = max(0, min(1, radius - d + 0.5))
                let v = UInt8((outside + (inside - outside) * coverage) * 255)
                row[x * 4] = v; row[x * 4 + 1] = v; row[x * 4 + 2] = v; row[x * 4 + 3] = 255
            }
        }
        return pb
    }

    /// The model's view of it: a hard near/far step on the depth texel lattice.
    static func discDepth(device: MTLDevice, rampTexels: Float = 0) -> MTLTexture? {
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r16Float, width: depthWidth, height: depthHeight, mipmapped: false)
        desc.usage = [.shaderRead]
        guard let texture = device.makeTexture(descriptor: desc) else { return nil }
        let uv = CoreMLDepthProvider.letterboxUVTransform(
            videoWidth: videoWidth, videoHeight: videoHeight, depthWidth: depthWidth, depthHeight: depthHeight
        )
        var values = [Float16](repeating: 0, count: depthWidth * depthHeight)
        for j in 0..<depthHeight {
            for i in 0..<depthWidth {
                let u = (Float(i) + 0.5) / Float(depthWidth), v = (Float(j) + 0.5) / Float(depthHeight)
                let video = SIMD2((u - uv.offset.x) / uv.scale.x * Float(videoWidth), (v - uv.offset.y) / uv.scale.y * Float(videoHeight))
                // Signed distance from the edge in depth texels (positive inside).
                let inside = (radius - simd_length(video - center)) / Float(videoWidth) * Float(depthWidth) / uv.scale.x
                let t: Float = rampTexels > 0 ? max(0, min(1, inside / rampTexels + 0.5)) : (inside > 0 ? 1 : 0)
                values[j * depthWidth + i] = Float16(0.2 + 0.8 * t)
            }
        }
        values.withUnsafeBytes {
            texture.replace(region: MTLRegionMake2D(0, 0, depthWidth, depthHeight), mipmapLevel: 0,
                            withBytes: $0.baseAddress!, bytesPerRow: depthWidth * 2)
        }
        return texture
    }

    static func read(_ texture: MTLTexture, device: MTLDevice, queue: MTLCommandQueue) -> (Int, Int, [Float])? {
        let w = texture.width, h = texture.height
        guard let cmd = queue.makeCommandBuffer(), let blit = cmd.makeBlitCommandEncoder(),
              let bytes = device.makeBuffer(length: w * h * 2, options: .storageModeShared) else { return nil }
        blit.copy(from: texture, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                  sourceSize: MTLSize(width: w, height: h, depth: 1), to: bytes, destinationOffset: 0,
                  destinationBytesPerRow: w * 2, destinationBytesPerImage: w * h * 2)
        blit.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        let ptr = bytes.contents().bindMemory(to: Float16.self, capacity: w * h)
        return (w, h, (0..<(w * h)).map { Float(ptr[$0]) })
    }
}
#endif
