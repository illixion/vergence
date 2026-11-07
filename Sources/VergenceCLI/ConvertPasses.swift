/*
 Vergence - `convert`'s optional passes for old SD films: `--antialias` and `--ai-hdr`

 `--antialias`: a 2003 SD CG film after NTSC deinterlacing has diagonal
 edges in stairs a line tall (each field line doubled). An upscaler sharpens
 those steps into crisp ragged edges. Before anything else sees the picture,
 ffmpeg re-makes each field's missing lines by edge-slope tracing (`estdif`,
 built into every ffmpeg, LGPL, no weights), both fields' versions are
 averaged (the "daa" anti-aliasing of old AviSynth scripts), and that is used
 only along edges (a Sobel mask, blurred), so texture away from edges keeps
 its detail. On a 3D film's bed edge, against nothing and against NNEDI3 in
 the same arrangement (GPL weights) and both directions: edge raggedness
 0.0565 (masked 0.0564) against 0.0655 untouched and 0.0612 / 0.0592 with
 NNEDI3, keeping more detail than either. The result goes to a ProRes
 intermediate of the part being converted, in the work folder.

 `--ai-hdr`: GMNet's gain grids made exactly as the `ai-hdr` workload makes
 them (`GainGridProducer`: keyframes every half second and at cuts, eased
 within a shot, the 16×9×8 grid, stored and read back as its records), from
 the left eye of a 3D source; the picture pass then applies them with
 `GainGridPass`, the players' own shader: mean gain capped at 2.5×, BT.2020
 PQ with SDR white at 203 nits, both eyes with the left eye's grid and scale.
 */

#if canImport(AVFoundation) && os(macOS)
import AVFoundation
import CoreML
import Foundation
import VergenceApple
import VergenceCommandLine

enum Antialias {
    /// The ffmpeg filter graph (one input, one output): both fields
    /// re-interpolated by edge-slope tracing, averaged, kept along edges.
    /// `vergence-convert` uses the same graph.
    static let filter = "format=yuv420p10le,split=4[o][a][b][m];"
        + "[a]estdif=mode=frame:parity=tff:deint=all[t];[b]estdif=mode=frame:parity=bff:deint=all[u];"
        + "[t][u]blend=all_mode=average[aa];[m]sobel=planes=7:scale=4,gblur=sigma=1.5:planes=7[mk];[o][aa][mk]maskedmerge"

    /// `input` from `start` for `length` seconds (nil: to the end),
    /// antialiased, its audio copied, into a ProRes 422 file at `output`.
    static func make(_ input: URL, start: Double, length: Double?, into output: URL) throws {
        let ffmpeg = try FFmpeg.required(for: "--antialias")
        var args = ["-nostdin", "-v", "error", "-y"]
        if start > 0 { args += ["-ss", String(format: "%.6f", start)] }
        args += ["-i", input.path]
        if let length, length.isFinite { args += ["-t", String(format: "%.6f", length)] }
        args += ["-map", "0:v:0", "-map", "0:a?", "-fps_mode", "passthrough", "-vf", filter,
                 "-c:v", "prores_ks", "-profile:v", "2", "-pix_fmt", "yuv422p10le", "-c:a", "copy", "-f", "mov", output.path]
        try ffmpeg.run(ffmpeg.ffmpeg, args, capture: false)
    }
}

/// GMNet's grid for every frame of a part, by the frame's index from its start.
struct AIHDRGrids {
    let grids: [GainGrid]
    let frameRate: Double
    let keyframes: Int

    /// The grid for a frame shown `seconds` after the part's first.
    func grid(at seconds: Double) -> GainGrid? {
        guard !grids.isEmpty else { return nil }
        return grids[min(max(0, Int((seconds * frameRate).rounded())), grids.count - 1)]
    }

    /// --gain-model, VERGENCE_GAIN_MODEL, else Convolution's conversion beside a Hypnos checkout.
    static func model(_ o: Options) throws -> URL {
        let given = o.gainModel ?? ProcessInfo.processInfo.environment["VERGENCE_GAIN_MODEL"]
        let candidates = given.map { [URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath)] } ?? [
            URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
                .deletingLastPathComponent().appendingPathComponent("Hypnos/Hypnos/Hypnos/Resources/GainMapGMNet.mlpackage"),
        ]
        guard let found = candidates.first(where: { FileManager.default.fileExists(atPath: $0.path) }) else {
            throw FormatError("--ai-hdr needs GMNet as Core ML: pass --gain-model PATH (GainMapGMNet.mlpackage, Convolution's "
                + "conversion, as a worker's gainModel) or set VERGENCE_GAIN_MODEL")
        }
        return found
    }

    /// Decodes the part with ffmpeg (the left eye of a frame-packed one, as
    /// the worker sees a 2D title) and runs the worker's producer over it.
    static func make(_ input: URL, start: Double, length: Double?, frameRate: Double, eye: PackedStereoLayout?,
                     model: URL) async throws -> AIHDRGrids {
        let producer = try await GainGridProducer(modelURL: model, computeUnits: .all)
        producer.prepare(frameRate: frameRate)
        let scale = Int32(max(1, (frameRate * 1000).rounded()))
        producer.timeScaleNum = scale
        producer.timeScaleDen = 1000
        producer.begin(continuing: false)
        let asset = AVURLAsset(url: input)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else { throw SpatialVideoError.noVideoTrack }
        let size = try await track.load(.naturalSize)
        var w = Int(size.width), h = Int(size.height)
        var crop: String?
        if let eye {
            let first = !eye.eyesReversed
            if eye.arrangement == .sideBySide {
                w /= 2
                crop = "crop=\(w):\(h):\(first ? 0 : w):0"
            } else {
                h /= 2
                crop = "crop=\(w):\(h):0:\(first ? 0 : h)"
            }
        }
        w &= ~1
        h &= ~1
        var args = ["-nostdin", "-v", "error"]
        if start > 0 { args += ["-ss", String(format: "%.6f", start)] }
        args += ["-i", input.path]
        if let length, length.isFinite { args += ["-t", String(format: "%.6f", length)] }
        args += ["-map", "0:v:0", "-fps_mode", "passthrough", "-vf", (crop.map { $0 + "," } ?? "") + "scale=\(w):\(h)",
                 "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"]
        let ffmpeg = try FFmpeg.required(for: "--ai-hdr")
        let process = Process()
        process.executableURL = ffmpeg.ffmpeg
        process.arguments = args
        let output = Pipe()
        process.standardOutput = output
        process.standardError = FileHandle.nullDevice
        try process.run()
        let reader = output.fileHandleForReading
        let frameBytes = w * h * 3 / 2
        var index: Int64 = 0
        while true {
            var frame = Data()
            while frame.count < frameBytes, let chunk = try reader.read(upToCount: frameBytes - frame.count), !chunk.isEmpty {
                frame.append(chunk)
            }
            guard frame.count == frameBytes else { break }
            try frame.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                let base = raw.bindMemory(to: UInt8.self).baseAddress!
                let planes = [AnalysisPlane(base: base, width: w, height: h, stride: w),
                              AnalysisPlane(base: base + w * h, width: w / 2, height: h / 2, stride: w / 2),
                              AnalysisPlane(base: base + w * h * 5 / 4, width: w / 2, height: h / 2, stride: w / 2)]
                _ = try producer.frame(DecodedFrame(left: planes, right: nil), index: index,
                                       time: AnalysisTime(value: index * 1000, timescale: scale), piece: 0 ..< Int64.max)
            }
            index += 1
        }
        process.waitUntilExit()
        guard index > 0 else { throw FormatError("--ai-hdr: ffmpeg decoded no frames") }
        let records = try producer.finish(piece: 0 ..< index)
        let grids = records.compactMap { GainGrid(record: $0) }
        guard grids.count == Int(index) else { throw FormatError("--ai-hdr: \(grids.count) grids for \(index) frames") }
        let keyframes = Int(index / GainGridSchedule.keyframeInterval(frameRate: frameRate))
        return AIHDRGrids(grids: grids, frameRate: frameRate, keyframes: keyframes)
    }
}

/// Bakes AI HDR into packed eyes: each eye made BT.2020 PQ by
/// `GainGridPass` with one grid and one capped scale, from the first eye.
final class AIHDRBaker {
    let pass: GainGridPass
    private let queue: MTLCommandQueue
    private var cache: CVMetalTextureCache?
    private var eyes: [MTLTexture] = []

    init() throws {
        guard let pass = GainGridPass(), let queue = pass.device.makeCommandQueue() else {
            throw FormatError("--ai-hdr: the gain-grid shaders aren't available")
        }
        self.pass = pass
        self.queue = queue
        CVMetalTextureCacheCreate(nil, nil, pass.device, nil, &cache)
    }

    /// `packed` (half-float RGBA, SDR code values) rewritten in place: the
    /// eyes at `origins`, each `eye` in size.
    func bake(_ packed: CVPixelBuffer, grid: GainGrid, eye: (width: Int, height: Int), origins: [(x: Int, y: Int)]) throws {
        guard let cache else { throw FormatError("--ai-hdr: no texture cache") }
        var cvTexture: CVMetalTexture?
        CVMetalTextureCacheCreateTextureFromImage(nil, cache, packed, nil, .rgba16Float, CVPixelBufferGetWidth(packed),
                                                  CVPixelBufferGetHeight(packed), 0, &cvTexture)
        guard let cvTexture, let picture = CVMetalTextureGetTexture(cvTexture) else { throw FormatError("--ai-hdr: no texture") }
        if eyes.count != origins.count || eyes.first?.width != eye.width || eyes.first?.height != eye.height {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: eye.width, height: eye.height,
                                                             mipmapped: false)
            d.usage = [.shaderRead, .renderTarget]
            d.storageMode = .private
            eyes = try origins.map { _ in
                guard let t = pass.device.makeTexture(descriptor: d) else { throw FormatError("--ai-hdr: out of GPU memory") }
                return t
            }
        }
        guard let commands = queue.makeCommandBuffer(), let blit = commands.makeBlitCommandEncoder() else {
            throw FormatError("--ai-hdr: no command buffer")
        }
        let size = MTLSize(width: eye.width, height: eye.height, depth: 1)
        for (texture, origin) in zip(eyes, origins) {
            blit.copy(from: picture, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: origin.x, y: origin.y, z: 0),
                      sourceSize: size, to: texture, destinationSlice: 0, destinationLevel: 0, destinationOrigin: MTLOrigin())
        }
        blit.endEncoding()
        guard let slot = pass.encodeMean(of: eyes[0], grid: grid, cap: GainGridPlayback.defaultCap, strength: 1, into: commands),
              pass.encodePQEyes(eyes, slot: slot, into: commands), let back = commands.makeBlitCommandEncoder() else {
            throw FormatError("--ai-hdr: the gain-grid pass failed")
        }
        for (texture, origin) in zip(eyes, origins) {
            back.copy(from: texture, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(), sourceSize: size, to: picture,
                      destinationSlice: 0, destinationLevel: 0, destinationOrigin: MTLOrigin(x: origin.x, y: origin.y, z: 0))
        }
        back.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        withExtendedLifetime(cvTexture) {}
        if let error = commands.error { throw FormatError("--ai-hdr: the GPU failed: \(error.localizedDescription)") }
    }
}
#endif
