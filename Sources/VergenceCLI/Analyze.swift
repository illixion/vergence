/*
 vergence analyze: Smooth Motion's analysis track for a stereo video

   vergence analyze DISC [--playlist NNNNN] [--seek S] [--start N] [--frames N] --models DIR [--output FILE.vgat]
   vergence analyze BASE.m2ts [DEPENDENT.m2ts] --models DIR [--output FILE.vgat]
   vergence analyze --raw WxH [--rate N/D] FILE|- --models DIR --output FILE.vgat

 For every real frame pair, what the headset needs to make in-between frames
 for both eyes without running a network itself (Hypnos
 `docs/plans/blu-ray-3d-smooth-motion.md`, option E): the left eye's motion
 from each frame to the next (RIFE at 480×288, a `flowPair` record per pair)
 and each frame's right→left disparity (NeuFlow at 512×288, a `disparity`
 record on the right eye), stored small and deflated (`SharedFieldAnalysis`).

 --models is a folder of the Core ML conversions made by Hypnos's
 `scripts/convert-interp-model.py`: RIFE*480x288 and NeuFlow*512x288
 (.mlpackage or .mlmodelc). --raw reads packed top/bottom I420, left on top,
 which is what `vergence decode` writes and what ffmpeg makes of any stereo
 file:

   ffmpeg -i clip.mkv -f rawvideo -pix_fmt yuv420p - | vergence analyze --raw 1920x1080 - --models M -o clip.vgat

 Times are the source's (90 kHz for a disc, on the playlist's own clock:
 the first clip's in-time plus the PTS, as Convolution's film player counts;
 frame index / --rate for raw), so a player matches records to frames by
 presentation time.
 */

#if canImport(CoreML)
import CoreML
#endif
import Foundation
import VergenceApple
import VergenceCommandLine
import VergenceCore
import VergenceDisc

/// One stereo frame as the producer sees it: borrowed planes, valid only in the callback.
struct AnalysisInputFrame {
    var left: [AnalysisPlane]
    var right: [AnalysisPlane]?
    var time: AnalysisTime
}

#if canImport(CoreML) && canImport(Compression)
func analyze(_ o: Options) async throws {
    guard let modelsPath = o.models else { usage("analyze needs --models DIR") }
    let folder = URL(fileURLWithPath: modelsPath)
    let flowURL = try findModel(in: folder, prefix: "RIFE", size: "480x288")
    let disparityURL = try findModel(in: folder, prefix: "NeuFlow", size: "512x288")
    // VERGENCE_COMPUTE_UNITS=ane|gpu|cpu|all picks where Core ML runs the models (default all).
    let units: MLComputeUnits = switch ProcessInfo.processInfo.environment["VERGENCE_COMPUTE_UNITS"] {
    case "ane": .cpuAndNeuralEngine
    case "gpu": .cpuAndGPU
    case "cpu": .cpuOnly
    default: .all
    }
    let analyzer = try await StereoFlowAnalyzer(flowModel: flowURL, disparityModel: disparityURL, computeUnits: units)
    log("vergence: flow \(analyzer.flowModelName), disparity \(analyzer.disparityModelName)")

    let outputURL: URL
    if o.output != "-" {
        outputURL = URL(fileURLWithPath: o.output)
    } else if o.raw == nil, let input = o.positional.first {
        let source = URL(fileURLWithPath: input)
        outputURL = source.deletingLastPathComponent()
            .appendingPathComponent(source.deletingPathExtension().lastPathComponent + ".vgat")
    } else {
        usage("analyze --raw needs --output FILE.vgat")
    }
    FileManager.default.createFile(atPath: outputURL.path, contents: nil)
    let out = try FileHandle(forWritingTo: outputURL)
    defer { try? out.close() }

    let flowSize = SharedFieldAnalysis.flowModelSize, disparitySize = SharedFieldAnalysis.disparityModelSize
    var previous: (left: [Float], time: AnalysisTime)?
    var wroteHeader = false
    var frames = 0, pairs = 0, bytes = 0, mono = 0
    let clock = ContinuousClock()
    let began = clock.now
    let last = o.frames == Int.max ? Int.max : o.frames
    var index = 0
    // Where a frame's time goes: resizing, the model runs (together), -, encoding and writing.
    var spent = [Duration](repeating: .zero, count: 4)
    func timed<T>(_ stage: Int, _ work: () throws -> T) rethrows -> T {
        let start = clock.now
        defer { spent[stage] += clock.now - start }
        return try work()
    }

    func write(_ record: AnalysisRecord) throws {
        let data = try timed(3) { AnalysisTrack.record(try record.deflated()) }
        try out.write(contentsOf: data)
        bytes += data.count
    }

    try forEachAnalysisFrame(o) { frame in
        defer { index += 1 }
        guard index >= o.start else { return true }
        guard frames < last else { return false }
        let y = frame.left[0]
        if !wroteHeader {
            let header = AnalysisTrackHeader(
                mediaID: mediaID(o), producer: "vergence analyze",
                models: ["flowPair": "\(analyzer.flowModelName) \(flowSize.width)x\(flowSize.height)",
                         "disparity": "\(analyzer.disparityModelName) \(disparitySize.width)x\(disparitySize.height)"],
                sourceWidth: y.width, sourceHeight: y.height)
            try out.write(contentsOf: try AnalysisTrack.header(header))
            wroteHeader = true
            log("vergence: \(y.width)x\(y.height) per eye -> \(outputURL.lastPathComponent)")
        }
        let left = timed(0) {
            SharedFieldAnalysis.rgbPlanar(y: frame.left[0], cb: frame.left[1], cr: frame.left[2],
                                          width: flowSize.width, height: flowSize.height)
        }
        let eyes = frame.right.map { right in
            timed(0) {
                (SharedFieldAnalysis.rgbPlanar(y: frame.left[0], cb: frame.left[1], cr: frame.left[2],
                                               width: disparitySize.width, height: disparitySize.height),
                 SharedFieldAnalysis.rgbPlanar(y: right[0], cb: right[1], cr: right[2],
                                               width: disparitySize.width, height: disparitySize.height))
            }
        }
        // The two runs overlap: RIFE usually lands on the Neural Engine, the fp32 NeuFlow on the GPU.
        let results = ModelRuns()
        let start = clock.now
        DispatchQueue.concurrentPerform(iterations: 2) { job in
            do {
                if job == 0, let previous {
                    results.flow = try analyzer.flow(from: previous.left, to: left)
                } else if job == 1, let (l, r) = eyes {
                    results.disparity = try analyzer.disparity(left: l, right: r)
                }
            } catch {
                results.error = error
            }
        }
        spent[1] += clock.now - start
        if let error = results.error { throw error }
        if let previous, let (flow, mask) = results.flow {
            try write(SharedFieldAnalysis.flowPairRecord(flow: flow, mask: mask, width: flowSize.width,
                                                         height: flowSize.height, time: previous.time,
                                                         endTime: frame.time))
            pairs += 1
        }
        if let rightToLeft = results.disparity {
            try write(SharedFieldAnalysis.disparityRecord(rightToLeft: rightToLeft, width: disparitySize.width,
                                                          height: disparitySize.height, time: frame.time))
        } else {
            mono += 1
        }
        previous = (left, frame.time)
        frames += 1
        if frames % 240 == 0 {
            let seconds = (clock.now - began) / .seconds(1)
            log(String(format: "vergence: %d frames, %.1f fps, %.1f KB a frame", frames, Double(frames) / seconds,
                       Double(bytes) / 1024 / Double(frames)))
        }
        return true
    }
    let seconds = (clock.now - began) / .seconds(1)
    log(String(format: "vergence: %d frames, %d flow pairs in %.1f s (%.1f fps), %.1f KB a frame, %d without a second view",
               frames, pairs, seconds, Double(frames) / max(seconds, 1e-9),
               Double(bytes) / 1024 / Double(max(frames, 1)), mono))
    let ms = spent.map { $0 / .milliseconds(1) / Double(max(frames, 1)) }
    log(String(format: "vergence: a frame: resize %.1f ms, models %.1f ms (flow and disparity together), encode %.1f ms",
               ms[0], ms[1], ms[3]))
}
#else
func analyze(_ o: Options) async throws {
    throw FormatError("analyze needs Core ML (Apple platforms only)")
}
#endif

/// What one frame's two concurrent model runs produced.
private final class ModelRuns: @unchecked Sendable {
    var flow: (flow: [Float], mask: [Float])?
    var disparity: [Float]?
    var error: Error?
}

/// The flow or disparity model in `folder`: name starts with `prefix` and
/// mentions `size`; a one-way NeuFlow (cheaper, same answer) wins over a
/// two-way one.
func findModel(in folder: URL, prefix: String, size: String) throws -> URL {
    let entries = (try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)) ?? []
    let matches = entries.filter {
        ["mlpackage", "mlmodelc"].contains($0.pathExtension) && $0.lastPathComponent.hasPrefix(prefix)
            && $0.lastPathComponent.contains(size)
    }.sorted { a, b in
        let a1 = a.lastPathComponent.contains("1way"), b1 = b.lastPathComponent.contains("1way")
        return a1 != b1 ? a1 : a.lastPathComponent < b.lastPathComponent
    }
    guard let first = matches.first else {
        throw FormatError("no \(prefix) model at \(size) in \(folder.path) (see Hypnos scripts/convert-interp-model.py)")
    }
    return first
}

/// An opaque id for the media: a hash of what was asked for, never the name itself.
func mediaID(_ o: Options) -> String {
    var hash: UInt64 = 0xCBF2_9CE4_8422_2325
    for byte in (o.positional + [o.playlist ?? "", o.raw.map { "\($0.width)x\($0.height)" } ?? ""])
        .joined(separator: "\u{0}").utf8 {
        hash = (hash ^ UInt64(byte)) &* 0x100_0000_01B3
    }
    return String(hash, radix: 16)
}

/// Frames from a disc or clip pair (through `frameSource`), or raw packed I420.
func forEachAnalysisFrame(_ o: Options, _ body: (AnalysisInputFrame) throws -> Bool) throws {
    guard let size = o.raw else {
        let source = try frameSource(o)
        try source.run { frame in
            func planes(_ views: [PlaneView]) -> [AnalysisPlane] {
                views.map { AnalysisPlane(base: $0.base, width: $0.width, height: $0.height, stride: $0.stride) }
            }
            return try body(AnalysisInputFrame(left: planes(frame.left), right: frame.right.map(planes),
                                               time: AnalysisTime(value: (frame.pts ?? 0) + source.timeOrigin,
                                                                  timescale: 90_000)))
        }
        log("vergence: \(source.summary())")
        return
    }
    guard o.positional.count == 1 else { usage("analyze --raw reads one FILE or -") }
    let input = o.positional[0] == "-" ? FileHandle.standardInput
        : try FileHandle(forReadingFrom: URL(fileURLWithPath: o.positional[0]))
    let w = size.width, h = size.height
    let lumaBytes = w * 2 * h, chromaBytes = (w / 2) * h
    let frameBytes = lumaBytes + 2 * chromaBytes
    var index: Int64 = 0
    while true {
        var data = Data()
        while data.count < frameBytes {
            guard let chunk = try input.read(upToCount: frameBytes - data.count), !chunk.isEmpty else { break }
            data.append(chunk)
        }
        guard data.count == frameBytes else {
            if !data.isEmpty { log("vergence: ignored a partial frame at the end (\(data.count) bytes)") }
            return
        }
        let time = AnalysisTime(value: index * Int64(o.rate.denominator), timescale: Int32(o.rate.numerator))
        index += 1
        let more = try data.withUnsafeBytes { raw -> Bool in
            let base = raw.bindMemory(to: UInt8.self).baseAddress!
            func eye(_ e: Int) -> [AnalysisPlane] {
                [AnalysisPlane(base: base + e * w * h, width: w, height: h, stride: w),
                 AnalysisPlane(base: base + lumaBytes + e * (w / 2) * (h / 2), width: w / 2, height: h / 2, stride: w / 2),
                 AnalysisPlane(base: base + lumaBytes + chromaBytes + e * (w / 2) * (h / 2), width: w / 2,
                               height: h / 2, stride: w / 2)]
            }
            return try body(AnalysisInputFrame(left: eye(0), right: eye(1), time: time))
        }
        if !more { return }
    }
}
