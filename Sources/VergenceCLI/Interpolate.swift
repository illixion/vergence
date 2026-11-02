/*
 vergence interpolate: the player's half of Smooth Motion for true stereo, on a Mac

   vergence interpolate --raw WxH [--rate N/D] FILE|- --fields FILE.vgat --output FILE.bgra

 Reads packed top/bottom BGRA frames (left on top, each eye W×H) and, for
 each pair of consecutive frames, writes the in-between frame made from the
 analysis track by `SharedFieldInterpolator`, the Metal pass the headset runs.
 Frame k's time is k / --rate, the same as `vergence analyze --raw` gives it,
 so a track made from the same frames lines up. A pair the track has no fields
 for is written as zeros, so output frame k is always the one between input
 frames k and k + 1. Hypnos `scripts/check-analysis-track.py --metal` scores
 the output against the real frames and the PyTorch reference:

   ffmpeg -i clip.mkv -vf "select=not(mod(n\\,2))" -fps_mode passthrough -f rawvideo -pix_fmt bgra - \\
     | vergence interpolate --raw 1920x1080 - --fields clip.vgat -o clip-mid.bgra

 Without --fields it reads plain (mono) WxH BGRA frames and makes each
 in-between frame with VideoToolbox's low-latency frame interpolation, the
 interpolator a 2D video's Smooth Motion uses on the headset
 (`VTMidFrameInterpolator`), so its artefacts can be looked at on a Mac.
 Its frames go through `InterpolationGuard` as on the headset; --no-guard
 writes VideoToolbox's frames as they come.
 */

import Foundation
import VergenceApple
import VergenceCommandLine
import VergenceCore

#if canImport(Metal) && canImport(Compression)
import CoreVideo

func interpolate(_ o: Options) throws {
    guard let size = o.raw else { usage("interpolate needs --raw WxH (one eye)") }
    guard let trackPath = o.fields else { return try interpolateMono(o, width: size.width, height: size.height) }
    guard o.output != "-" else { usage("interpolate needs --output FILE.bgra") }
    guard o.positional.count == 1 else { usage("interpolate reads one FILE or -") }
    let index = try AnalysisTrackIndex(Data(contentsOf: URL(fileURLWithPath: trackPath), options: .mappedIfSafe))
    let provider = AnalysisTrackFieldProvider(index: index, inflate: AnalysisDeflate.inflate)
    guard let interpolator = SharedFieldInterpolator(provider: provider) else {
        throw FormatError("no Metal device, or the shared-field kernel didn't compile")
    }
    let w = size.width, h = 2 * size.height
    guard interpolator.prepare(width: w, height: h, scale: 1) else { throw FormatError("can't interpolate \(w)x\(h)") }

    let input = o.positional[0] == "-" ? FileHandle.standardInput
        : try FileHandle(forReadingFrom: URL(fileURLWithPath: o.positional[0]))
    FileManager.default.createFile(atPath: o.output, contents: nil)
    let out = try FileHandle(forWritingTo: URL(fileURLWithPath: o.output))
    defer { try? out.close() }

    let frameBytes = w * h * 4
    var previous: (buffer: CVPixelBuffer, time: Double)?
    var k: Int64 = 0, made = 0, missed = 0
    var gpuMs: [Double] = []
    let zeros = Data(count: frameBytes)
    // A pool per frame, as in analyze: FileHandle's chunks are autoreleased.
    while k < Int64(min(o.frames, Int(Int32.max))), try autoreleasepool(invoking: { () throws -> Bool in
        var data = Data()
        while data.count < frameBytes {
            guard let chunk = try input.read(upToCount: frameBytes - data.count), !chunk.isEmpty else { break }
            data.append(chunk)
        }
        guard data.count == frameBytes else { return false }
        let time = Double(k) * Double(o.rate.denominator) / Double(o.rate.numerator)
        k += 1
        let buffer = try bgraBuffer(data, width: w, height: h)
        if let previous {
            let started = Date()
            if let fields = provider.fields(from: previous.time, to: time),
               let mid = interpolator.midFrame(between: previous.buffer, and: buffer, fields: fields) {
                gpuMs.append(Date().timeIntervalSince(started) * 1000)
                try out.write(contentsOf: bytes(of: mid))
                made += 1
            } else {
                try out.write(contentsOf: zeros)
                missed += 1
            }
        }
        previous = (buffer, time)
        return true
    }) {}
    let median = gpuMs.isEmpty ? .nan : gpuMs.sorted()[gpuMs.count / 2]
    log(String(format: "vergence: %d in-between frames, %d pairs without fields; %.1f ms median a frame (lookup, upload, pass)",
               made, missed, median))
}

/// Mono frames through VideoToolbox's interpolator, as a 2D video's Smooth Motion makes them.
private func interpolateMono(_ o: Options, width w: Int, height h: Int) throws {
    guard o.output != "-" else { usage("interpolate needs --output FILE.bgra") }
    guard o.positional.count == 1 else { usage("interpolate reads one FILE or -") }
    guard #available(macOS 26.0, *), VTMidFrameInterpolator.isSupported else {
        throw FormatError("VideoToolbox's low-latency frame interpolation isn't supported here")
    }
    let interpolator = VTMidFrameInterpolator(guardsResults: !o.noGuard)
    guard interpolator.prepare(width: w, height: h, scale: 1) else { throw FormatError("can't interpolate \(w)x\(h)") }
    let input = o.positional[0] == "-" ? FileHandle.standardInput
        : try FileHandle(forReadingFrom: URL(fileURLWithPath: o.positional[0]))
    FileManager.default.createFile(atPath: o.output, contents: nil)
    let out = try FileHandle(forWritingTo: URL(fileURLWithPath: o.output))
    defer { try? out.close() }
    let frameBytes = w * h * 4
    let zeros = Data(count: frameBytes)
    var previous: CVPixelBuffer?
    var k = 0, made = 0, failed = 0
    var ms: [Double] = [], guardMs: [Double] = []
    while k < o.frames, try autoreleasepool(invoking: { () throws -> Bool in
        var data = Data()
        while data.count < frameBytes {
            guard let chunk = try input.read(upToCount: frameBytes - data.count), !chunk.isEmpty else { break }
            data.append(chunk)
        }
        guard data.count == frameBytes else { return false }
        k += 1
        let buffer = try bgraBuffer(data, width: w, height: h)
        if let previous {
            let done = DispatchSemaphore(value: 0)
            nonisolated(unsafe) var result: (CVPixelBuffer?, Double) = (nil, 0)
            interpolator.midFrame(between: previous, and: buffer) { frame, processMs in
                result = (frame, processMs)
                done.signal()
            }
            done.wait()
            if let mid = result.0 {
                ms.append(result.1)
                if !o.noGuard { guardMs.append(interpolator.guardGPUMilliseconds) }
                try out.write(contentsOf: bytes(of: mid))
                made += 1
            } else {
                try out.write(contentsOf: zeros)
                failed += 1
            }
        }
        previous = buffer
        return true
    }) {}
    let median = ms.isEmpty ? .nan : ms.sorted()[ms.count / 2]
    let guardMedian = guardMs.isEmpty ? 0 : guardMs.sorted()[guardMs.count / 2]
    log(String(format: "vergence: %d in-between frames (VideoToolbox), %d failed; %.1f ms median a frame%@",
               made, failed, median,
               guardMs.isEmpty ? "" : String(format: " (guard GPU %.2f ms median, %.2f max)", guardMedian, guardMs.max() ?? 0)))
}

private func bgraBuffer(_ data: Data, width: Int, height: Int) throws -> CVPixelBuffer {
    var buffer: CVPixelBuffer?
    let attributes: [String: Any] = [kCVPixelBufferMetalCompatibilityKey as String: true,
                                     kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any]()]
    CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA, attributes as CFDictionary, &buffer)
    guard let buffer else { throw FormatError("couldn't make a \(width)x\(height) buffer") }
    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
    let stride = CVPixelBufferGetBytesPerRow(buffer)
    data.withUnsafeBytes { raw in
        let src = raw.bindMemory(to: UInt8.self).baseAddress!
        for y in 0 ..< height { (base + y * stride).update(from: src + y * width * 4, count: width * 4) }
    }
    return buffer
}

private func bytes(of buffer: CVPixelBuffer) -> Data {
    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
    let width = CVPixelBufferGetWidth(buffer), height = CVPixelBufferGetHeight(buffer)
    let stride = CVPixelBufferGetBytesPerRow(buffer)
    let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
    var data = Data(count: width * height * 4)
    data.withUnsafeMutableBytes { raw in
        let dst = raw.bindMemory(to: UInt8.self).baseAddress!
        for y in 0 ..< height { (dst + y * width * 4).update(from: base + y * stride, count: width * 4) }
    }
    return data
}
#else
func interpolate(_ o: Options) throws {
    throw FormatError("interpolate needs Metal (Apple platforms only)")
}
#endif
