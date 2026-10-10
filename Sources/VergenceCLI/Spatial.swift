/*
 Vergence - `make` and `inspect`
 */

import Foundation
import VergenceApple
import VergenceDisc

/// The files `make` would convert: each argument, or every video in a
/// folder argument (recursively), leaving out its own outputs.
func videoInputs(_ paths: [String]) -> [URL] {
    let extensions: Set<String> = ["mov", "mp4", "m4v", "mkv", "webm", "avi", "ts", "m2ts", "mts", "mpg", "mpeg", "wmv"]
    func isInput(_ url: URL) -> Bool {
        let name = url.lastPathComponent.lowercased()
        return !name.hasPrefix(".") && extensions.contains(url.pathExtension.lowercased())
            && !name.hasSuffix(".spatial.mov") && !name.hasSuffix(".partial.mov")
    }
    var found: [URL] = []
    for path in paths {
        let url = URL(fileURLWithPath: path)
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory) else {
            log("vergence: \(path): no such file")
            continue
        }
        guard isDirectory.boolValue else {
            found.append(url)
            continue
        }
        let walker = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.isRegularFileKey],
                                                    options: [.skipsHiddenFiles])
        while let next = walker?.nextObject() as? URL {
            if isInput(next) { found.append(next) }
        }
    }
    return found.sorted { $0.path.localizedStandardCompare($1.path) == .orderedAscending }
}

/// Where `input`'s spatial video goes.
func outputURL(for input: URL, _ o: Options, single: Bool) -> URL {
    let name = input.deletingPathExtension().lastPathComponent + ".spatial.mov"
    guard o.output != "-" else { return input.deletingLastPathComponent().appendingPathComponent(name) }
    let target = URL(fileURLWithPath: o.output)
    var isDirectory: ObjCBool = false
    let exists = FileManager.default.fileExists(atPath: target.path, isDirectory: &isDirectory)
    if single, !(exists && isDirectory.boolValue), !o.output.hasSuffix("/") { return target }
    return target.appendingPathComponent(name)
}

func make(_ o: Options) async throws {
    #if !canImport(AVFoundation) || os(tvOS)
    throw FormatError("making spatial video needs Apple's encoder (macOS)")
    #else
    let inputs = videoInputs(o.positional)
    guard !inputs.isEmpty else { usage("make needs a video file or a folder of them") }
    if inputs.count > 1, o.output != "-" {
        try FileManager.default.createDirectory(at: URL(fileURLWithPath: o.output), withIntermediateDirectories: true)
    }
    var made = 0, skipped = 0, failed = 0
    for input in inputs {
        let output = outputURL(for: input, o, single: inputs.count == 1)
        do {
            switch try await makeOne(input, output, o) {
            case true: made += 1
            case false: skipped += 1
            }
        } catch {
            log("vergence: \(input.lastPathComponent): \(error.localizedDescription)")
            failed += 1
        }
    }
    if inputs.count > 1 { log("vergence: \(made) made, \(skipped) skipped, \(failed) failed") }
    if failed > 0 { exit(1) }
    #endif
}

#if canImport(AVFoundation) && !os(tvOS)
/// Converts one file. False when there was nothing to do.
func makeOne(_ input: URL, _ output: URL, _ o: Options) async throws -> Bool {
    let name = input.lastPathComponent
    if FileManager.default.fileExists(atPath: output.path), !o.force {
        log("vergence: \(name): \(output.lastPathComponent) is already made (--force to redo)")
        return false
    }
    // AVFoundation reads QuickTime and MP4; everything else goes through
    // ffmpeg, but only once the file is known to be stereo.
    let native = FFmpeg.isNative(input)
    var nativeSource: SpatialVideoSource?
    var ffmpeg: FFmpeg?
    var probe: FFmpeg.Probe?
    let frameSize: CGSize
    let declared: PackedStereoLayout?
    if native {
        let source = try await SpatialVideoSource.open(input)
        if source.isMultiviewStereo {
            log("vergence: \(name): already MV-HEVC stereo (vergence inspect shows its metadata)")
            return false
        }
        nativeSource = source
        frameSize = source.frameSize
        declared = source.declaredLayout
    } else {
        let tool = try FFmpeg.required(for: "reading .\(input.pathExtension)")
        let probed = try tool.probe(input)
        guard let video = probed.video, let w = video.width, let h = video.height else { throw FormatError("no video stream") }
        ffmpeg = tool
        probe = probed
        frameSize = CGSize(width: w, height: h)
        declared = video.declaredLayout.map {
            var corrected = $0.correctingResolution(forPacked: frameSize)
            corrected.eyesReversed = $0.eyesReversed
            return corrected
        }
    }

    var layout: PackedStereoLayout
    var notes: [String] = []
    if o.layout != "auto" {
        let arrangement: PackedStereoLayout.Arrangement = o.layout.contains("sbs") ? .sideBySide : .overUnder
        layout = PackedStereoLayout(arrangement: arrangement, isHalfResolution: o.layout.hasPrefix("h"))
        let shaped = layout.correctingResolution(forPacked: frameSize)
        if shaped.isHalfResolution != layout.isHalfResolution {
            notes.append("--layout \(o.layout) gives each eye an odd shape for a \(Int(frameSize.width))×\(Int(frameSize.height)) frame; did you mean \(shaped.isHalfResolution ? "h" + o.layout : String(o.layout.dropFirst()))?")
        }
        notes.append("layout \(o.layout) as given")
    } else {
        let pictures: [LumaPlane]
        if let nativeSource { pictures = try await nativeSource.samplePictures() } else { pictures = try ffmpeg!.pictures(input, probe: probe!) }
        let detection = StereoLayoutDetector.detect(declared: declared, fileName: name, frameSize: frameSize, pictures: pictures)
        guard let found = detection.layout, let format = detection.format else {
            log("vergence: \(name): looks like 2D video (\(detection.reasons.joined(separator: "; "))); --layout converts it anyway")
            return false
        }
        layout = found
        notes.append(String(format: "layout %@ (%@, %.0f%% sure): %@", format.rawValue, detection.source.rawValue,
                            detection.confidence * 100, detection.reasons.joined(separator: "; ")))
        if detection.confidence < 0.6 { notes.append("unsure: check the result, or pass --layout") }
    }
    // A container's "eyes inverted" and --swap-eyes cancel out.
    layout.eyesReversed = layout.eyesReversed != o.swapEyes

    var camera = o.preset.metadata
    if let v = o.baseline { camera.baselineMillimeters = v }
    if let v = o.hfov { camera.horizontalFieldOfViewDegrees = v }
    if let v = o.adjust { camera.horizontalDisparityAdjustment = v }
    guard camera.problems.isEmpty else { throw SpatialVideoError.invalidMetadata(camera.problems) }

    // Remux what AVFoundation can't read into a QuickTime file beside the
    // output, gone again when this returns.
    var remuxed: URL?
    defer { if let remuxed { try? FileManager.default.removeItem(at: remuxed) } }
    if let ffmpeg, let probe {
        let temporary = output.deletingLastPathComponent()
            .appendingPathComponent("." + input.deletingPathExtension().lastPathComponent + ".remux.mov")
        let plan = try ffmpeg.remuxPlan(probe, to: temporary, input: input)
        if o.dryRun {
            log("vergence: \(name) -> \(output.path)")
            for note in notes { log("  \(note)") }
            log("  remux with ffmpeg first: \(plan.notes.joined(separator: ", "))")
            return false
        }
        log("vergence: \(name): remuxing with ffmpeg (\(plan.notes.joined(separator: ", ")))")
        remuxed = temporary
        try ffmpeg.remux(plan)
    }
    let source: SpatialVideoSource
    if let nativeSource { source = nativeSource } else { source = try await SpatialVideoSource.open(remuxed!) }
    // A ProRes intermediate is 10-bit whatever the source was.
    let sourceTenBit = probe?.video.map { video in
        let format = video.pix_fmt ?? ""
        return format.contains("10") || format.contains("12") || format.contains("16")
            || ["smpte2084", "arib-std-b67"].contains(video.color_transfer ?? "")
    }
    let job = SpatialVideoJob(source: source, output: output, layout: layout, camera: camera,
                              quality: o.quality, bitsPerSecond: o.bitrate, includeAudio: o.includeAudio, tenBit: sourceTenBit)

    let eye = job.eyeSize
    log("vergence: \(name) -> \(output.path)")
    for note in notes { log("  \(note)") }
    log(String(format: "  eyes %d×%d%@%@, %.1f Mbps, %@", eye.width, eye.height, layout.eyesReversed ? " (swapped)" : "",
               job.encodesTenBit ? ", 10-bit" + (source.isHDR ? " HDR" : "") : "",
               Double(job.resolvedBitsPerSecond) / 1_000_000,
               o.includeAudio ? (source.audioTracks.isEmpty ? "no audio" : "audio copied") : "audio left out"))
    log(String(format: "  camera (%@ preset%@): baseline %.2f mm, field of view %.1f°, disparity adjustment %.3f, %@",
               o.preset.rawValue, o.baseline ?? o.hfov ?? o.adjust != nil ? " + overrides" : "",
               camera.baselineMillimeters, camera.horizontalFieldOfViewDegrees, camera.horizontalDisparityAdjustment,
               camera.projection.rawValue))
    if o.dryRun { return false }

    let began = Date()
    let interactive = isatty(STDERR_FILENO) != 0
    let result = try await SpatialVideoMaker.make(job) { fraction in
        guard interactive else { return }
        FileHandle.standardError.write(Data(String(format: "\r  %3.0f%%", fraction * 100).utf8))
    }
    if interactive { FileHandle.standardError.write(Data("\r".utf8)) }
    let seconds = Date().timeIntervalSince(began)
    log(String(format: "  done: %d frames in %.1f s (%.1f fps)%@", result.frames, seconds, Double(result.frames) / max(seconds, 0.001),
               result.audioTracksSkipped > 0 ? ", \(result.audioTracksSkipped) audio track(s) QuickTime can't hold left out" : ""))
    return true
}
#endif

func inspect(_ o: Options) async throws {
    guard o.positional.count == 1 else { usage("inspect takes one file") }
    let path = o.positional[0]
    if let disc = try? openDisc(path), (try? disc.mainPlaylist()) != nil {
        print("3D Blu-ray disc (decrypted)" )
        try info(o)
        return
    }
    let url = URL(fileURLWithPath: path)
    #if !canImport(AVFoundation) || os(tvOS)
    // Without AVFoundation, ffmpeg reads everything.
    return try inspectWithFFmpeg(url, shown: path)
    #else
    if !FFmpeg.isNative(url) { return try inspectWithFFmpeg(url, shown: path) }
    let report = try await SpatialVideoInspector.inspect(url)
    let width = report.fields.map(\.name.count).max() ?? 0
    for field in report.fields {
        print(field.name.padding(toLength: width, withPad: " ", startingAt: 0) + "  " + field.value)
    }
    print("")
    if report.isCompleteSpatialVideo {
        print("Spatial video: Photos and Files show it as spatial.")
    } else if report.source.isMultiviewStereo {
        print("Stereo, but players will miss:")
        for line in report.missing { print("  - \(line)") }
        print("vergence make can't re-tag MV-HEVC yet; re-export it from its source with vergence make.")
    } else {
        let pictures = try await report.source.samplePictures()
        let detection = StereoLayoutDetector.detect(declared: report.source.declaredLayout, fileName: url.lastPathComponent,
                                                    frameSize: report.source.frameSize, pictures: pictures)
        if let format = detection.format {
            print(String(format: "Frame-packed stereo, %@ (%.0f%% sure):", format.displayName, detection.confidence * 100))
            for reason in detection.reasons { print("  - \(reason)") }
            print("vergence make \(shellQuoted(path)) turns it into spatial video.")
        } else {
            print("2D video:")
            for reason in detection.reasons { print("  - \(reason)") }
        }
    }
    #endif
}

func shellQuoted(_ s: String) -> String {
    s.allSatisfy { $0.isLetter || $0.isNumber || "/._-+".contains($0) } ? s : "'" + s.replacingOccurrences(of: "'", with: "'\\''") + "'"
}

/// inspect for what AVFoundation can't open: ffprobe's view of it, and the
/// layout it looks like.
func inspectWithFFmpeg(_ url: URL, shown: String) throws {
    let ffmpeg = try FFmpeg.required(for: "reading .\(url.pathExtension)")
    let probe = try ffmpeg.probe(url)
    guard let video = probe.video, let w = video.width, let h = video.height else { throw FormatError("no video stream") }
    var fields: [(String, String)] = []
    fields.append(("container", probe.format?.format_name ?? "?"))
    fields.append(("video", "\(video.codec_name ?? "?") \(w)×\(h), " + String(format: "%.3f fps", video.frameRate) + ", \(video.pix_fmt ?? "?")"
        + (Double(video.bit_rate ?? "").map { String(format: ", %.1f Mbps", $0 / 1_000_000) } ?? "")))
    fields.append(("colour", "\(video.color_primaries ?? "untagged") / \(video.color_transfer ?? "untagged")"))
    fields.append(("duration", String(format: "%.1f s", probe.duration)))
    fields.append(("audio", probe.audio.isEmpty ? "none" : probe.audio.map {
        "\($0.codec_name ?? "?") \($0.channel_layout ?? "\($0.channels ?? 0) ch")" + ($0.tags?.language.map { " \($0)" } ?? "")
    }.joined(separator: "; ")))
    let frameSize = CGSize(width: w, height: h)
    let declared = video.declaredLayout.map {
        var corrected = $0.correctingResolution(forPacked: frameSize)
        corrected.eyesReversed = $0.eyesReversed
        return corrected
    }
    if let declared {
        fields.append(("stereo", "\(declared.arrangement == .sideBySide ? "side by side" : "over-under") (container metadata)"
            + (declared.eyesReversed ? ", eyes swapped" : "")))
    }
    let width = fields.map(\.0.count).max() ?? 0
    for (name, value) in fields { print(name.padding(toLength: width, withPad: " ", startingAt: 0) + "  " + value) }
    print("")
    let detection = StereoLayoutDetector.detect(declared: declared, fileName: url.lastPathComponent, frameSize: frameSize,
                                                pictures: try ffmpeg.pictures(url, probe: probe))
    if let format = detection.format {
        print(String(format: "Frame-packed stereo, %@ (%.0f%% sure):", format.displayName, detection.confidence * 100))
        for reason in detection.reasons { print("  - \(reason)") }
        print("vergence make \(shellQuoted(shown)) turns it into spatial video (remuxing it with ffmpeg first).")
    } else {
        print("2D video:")
        for reason in detection.reasons { print("  - \(reason)") }
    }
}
