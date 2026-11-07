/*
 Vergence - `convert`: any 2D video to an AI 3D stereo copy

   vergence convert VIDEO|FOLDER... [--output FILE|DIR] [--layout fsbs|hsbs|tab|htab|spatial|2d]
                    [--strength S] [--convergence C] [--sample SECONDS] [--start S] [--frames N]
                    [--model PATH] [--download-model] [--depth-track FILE.vgat]
                    [--quality low|standard|high|max] [--bitrate MBPS] [--max-eye PIXELS] [--no-audio]
                    [--upscale 2x] [--upscale-model PATH.pth] [--input-layout auto|2d|sbs|hsbs|tab|htab]
                    [--keep-repeated-frames] [--antialias] [--ai-hdr] [--gain-model PATH]
                    [--dry-run] [--force]

 The offline path Convolution's Pre-Process takes, run to a file instead of a
 cache, in two passes over the video:
 1. Depth (`OfflineDepthConversion`, i.e. `DepthConverter`): Core ML inference
    on every frame, joint-bilateral refinement on the frame's own edges, the
    ±2-frame window cut at scene changes, each shot's scale locked by Stable
    Depth anchoring. It goes into a hidden folder beside the output, kept if
    the run stops, so running again picks up from the finished depth.
 2. Picture: each frame decoded again with its depth (`CachedDepthSource`,
    matched by time), both eyes drawn by the depth-displaced mesh exactly as
    cached playback does (`DepthStereoFrameRenderer`, with the soft-knee depth
    budget), packed side by side or top and bottom, encoded HEVC. 10-bit
    sources and HDR (PQ, HLG) stay 10-bit with their colour tags and mastering
    metadata; audio tracks MP4 holds (AAC, AC-3, E-AC-3, ALAC, MP3) are copied.

 Then Google's `st3d` box and a name players and media servers read the layout
 from: VIDEO.3D.FSBS.mp4, .3D.HSBS.mp4, .3D.FTAB.mp4, .3D.HTAB.mp4. `--layout
 spatial` encodes full side by side to a temporary file and makes MV-HEVC
 spatial video from it the way `make` does (VIDEO.spatial.mov, `--preset`
 and friends apply).

 `--upscale 2x` puts Real-ESRGAN's compact network (`FrameUpscaler`) in
 front of the picture pass: a 2D source is upscaled, then made 3D from depth
 as above (depth is still made at the source's size), or with `--layout 2d`
 only upscaled (VIDEO.2x.mp4). A side-by-side or top-and-bottom source
 (declared, named, or seen in its picture; `--input-layout` to say) is
 upscaled eye by eye in one batch and packed in the output layout, with no
 depth at all (VIDEO.2x.3D.FSBS.mp4).

 A source that repeats one frame in five (a pulldown, often at an odd rate
 after a remux: `RepeatedFrames`) has the repeats dropped and the rest
 spaced evenly at four fifths of the rate, unless `--keep-repeated-frames`.

 Not done here: subtitles (left out), Smooth Motion in-between frames,
 MKV output (audio MP4 can't hold is left out, with a note; a file ffmpeg
 remuxes first has had TrueHD/DTS made ALAC or AAC already).
 */

import Foundation
import VergenceApple
import VergenceCommandLine
import VergenceDisc
#if canImport(AVFoundation) && os(macOS)
import AVFoundation
import CoreMedia
import CoreVideo
import VideoToolbox
#endif

func convert(_ o: Options) async throws {
    #if !(canImport(AVFoundation) && os(macOS))
    throw FormatError("convert needs Apple's frameworks (macOS)")
    #else
    let inputs = videoInputs(o.positional, forSideBySide: true).filter {
        let name = $0.lastPathComponent.lowercased()
        return !name.contains(".3d.") && !name.hasSuffix(".spatial.mov")
    }
    guard !inputs.isEmpty else { usage("convert needs a video file or a folder of them") }
    if o.depthTrack != nil, inputs.count > 1 { usage("--depth-track takes one video") }
    if inputs.count > 1, o.output != "-" {
        try FileManager.default.createDirectory(at: URL(fileURLWithPath: o.output), withIntermediateDirectories: true)
    }
    // A dry run only says what it would do: no shaders, no model.
    if !o.dryRun { try prepareShaders() }
    // The depth model is looked up when a file first needs depth (an
    // upscaled 3D source or --layout 2d needs none); a bad one stops the run.
    let models = ConvertModels(o)
    if !o.dryRun, o.upscale != nil { _ = try await models.upscaler() }
    if !o.dryRun, o.upscale == nil || (o.convertLayout != .flat && o.inputLayout == "2d") { _ = try await models.depth() }
    var made = 0, skipped = 0, failed = 0
    for input in inputs {
        let output = OutputNames.converted(for: input, o, single: inputs.count == 1)
        do {
            if try await convertOne(input, output, models: models, o) { made += 1 } else { skipped += 1 }
        } catch {
            log("vergence: \(input.lastPathComponent): \(describe(error))")
            failed += 1
        }
    }
    if inputs.count > 1 { log("vergence: \(made) converted, \(skipped) skipped, \(failed) failed") }
    if failed > 0 { exit(1) }
    #endif
}

#if canImport(AVFoundation) && os(macOS)

/// An error as a sentence: our own errors say what went wrong in their
/// description, Foundation's in their localized one.
func describe(_ error: Error) -> String {
    switch error {
    case let e as FormatError: e.description
    case let e as UpscaleNetwork.Failure: e.description
    case let e as FrameUpscaler.Failure: e.description
    default: error.localizedDescription
    }
}

/// The warp's shaders. A SwiftPM build compiles no Metal library, so a tool
/// built that way compiles `StereoShaders.metal` from its source tree (or
/// wherever VERGENCE_SHADERS points); one built with Xcode has its library.
func prepareShaders() throws {
    if ProcessInfo.processInfo.environment["VERGENCE_SHADERS"] == nil {
        let source = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("VergenceApple/Stereo/StereoShaders.metal")
        if FileManager.default.fileExists(atPath: source.path) { VergenceMetal.useShaders(at: source) }
    }
    guard VergenceMetal.shared != nil else {
        throw FormatError("the stereo shaders aren't available: set VERGENCE_SHADERS to Sources/VergenceApple/Stereo/StereoShaders.metal")
    }
}

/// --model, VERGENCE_DEPTH_MODEL, else the installed Depth Anything V2 Small
/// (downloaded first with --download-model).
@MainActor func depthModel(_ o: Options) async throws -> URL {
    try checkDepthModelLicence(try await resolvedDepthModel(o))
}

/// Refuses a model whose weights restrict their use (`DepthModelLicence`)
/// unless this build was made to take one (`-Xswiftc
/// -DVERGENCE_NONCOMMERCIAL_MODELS`), and says the licence when it was.
func checkDepthModelLicence(_ url: URL) throws -> URL {
    guard let restriction = DepthModelLicence.restriction(forModelAt: url) else { return url }
    let name = url.deletingPathExtension().lastPathComponent
    #if VERGENCE_NONCOMMERCIAL_MODELS
    log("vergence: \(name)'s weights are \(restriction.name)\(restriction.url.isEmpty ? "" : " (\(restriction.url))"): "
        + "\(restriction.note). Use it, and what you make with it, for non-commercial work only.")
    return url
    #else
    throw FormatError("\(name)'s weights are \(restriction.name) (\(restriction.note)); this build takes only "
        + "Apache-2.0 models such as Depth Anything V2 Small. Build with -Xswiftc -DVERGENCE_NONCOMMERCIAL_MODELS "
        + "to use it for non-commercial work.")
    #endif
}

@MainActor private func resolvedDepthModel(_ o: Options) async throws -> URL {
    if let path = o.model ?? ProcessInfo.processInfo.environment["VERGENCE_DEPTH_MODEL"] {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        guard FileManager.default.fileExists(atPath: url.path) else { throw FormatError("no depth model at \(path)") }
        return url
    }
    let preferred = DepthModelManager.variants[0]
    func installed() -> URL? {
        DepthModelStore.modelURL(named: preferred.name)
            ?? DepthModelStore.installedModelNames().first.flatMap { DepthModelStore.modelURL(named: $0) }
    }
    if let url = installed() { return url }
    guard o.downloadModel else {
        throw FormatError("no depth model: pass --model PATH (Depth Anything V2 Small as .mlpackage or .mlmodelc), "
            + "set VERGENCE_DEPTH_MODEL, or add --download-model to fetch it from Apple's Core ML repository (Apache-2.0, about 50 MB)")
    }
    log("vergence: downloading \(preferred.displayName) from huggingface.co/apple/coreml-depth-anything-v2-small")
    let manager = DepthModelManager.shared
    await manager.download(preferred)
    if let url = installed() { return url }
    throw FormatError(manager.errorMessage ?? "the model download failed")
}

/// The models a run needs, each looked up once.
@MainActor final class ConvertModels {
    let options: Options
    private var depthURL: URL?
    private var network: UpscaleNetwork?
    nonisolated init(_ o: Options) { options = o }

    func depth() async throws -> URL {
        if let depthURL { return depthURL }
        let url = try await depthModel(options)
        depthURL = url
        return url
    }

    func upscaler() async throws -> UpscaleNetwork {
        if let network { return network }
        let made = try await upscaleNetwork(options)
        network = made
        return made
    }
}

/// What the picture pass makes.
enum PictureMode {
    /// Both eyes warped from depth (the frame upscaled first with --upscale).
    case depth
    /// A frame-packed 3D source, each eye upscaled; no depth.
    case stereoUpscale(PackedStereoLayout)
    /// --layout 2d: upscaled, still 2D.
    case flatUpscale
}

/// Converts one file. False when there was nothing to do.
func convertOne(_ input: URL, _ output: URL, models: ConvertModels, _ o: Options) async throws -> Bool {
    let name = input.lastPathComponent
    if FileManager.default.fileExists(atPath: output.path), !o.force {
        log("vergence: \(name): \(output.lastPathComponent) is already made (--force to redo)")
        return false
    }
    // Everything temporary sits in one hidden folder beside the output: the
    // depth (kept when a run stops, so the next one starts from it), a remux,
    // the spatial layout's side-by-side intermediate.
    let work = output.deletingLastPathComponent().appendingPathComponent("." + output.lastPathComponent + ".work")
    try FileManager.default.createDirectory(at: work, withIntermediateDirectories: true)
    // Kept only while it holds finished depth worth starting again from.
    defer {
        if DepthCacheStore.entry(at: work.appendingPathComponent("depth")) == nil {
            try? FileManager.default.removeItem(at: work)
        }
    }
    var readable = input
    // A ProRes intermediate is 10-bit whatever the source was.
    var eightBitSource = false
    // A layout the container declares that the remux drops (Matroska's StereoMode).
    var containerLayout: PackedStereoLayout?
    if !FFmpeg.isNative(input) {
        let ffmpeg = try FFmpeg.required(for: "reading .\(input.pathExtension)")
        let probe = try ffmpeg.probe(input)
        if let video = probe.video, let declared = video.declaredLayout, let w = video.width, let h = video.height {
            var corrected = declared.correctingResolution(forPacked: CGSize(width: w, height: h))
            corrected.eyesReversed = declared.eyesReversed
            containerLayout = corrected
        }
        let remuxed = work.appendingPathComponent("source.mov")
        let plan = try ffmpeg.remuxPlan(probe, to: remuxed, input: input)
        if !o.dryRun {
            log("vergence: \(name): remuxing with ffmpeg (\(plan.notes.joined(separator: ", ")))")
            try ffmpeg.remux(plan)
            readable = remuxed
        }
        let format = probe.video?.pix_fmt ?? ""
        eightBitSource = !(format.contains("10") || format.contains("12") || format.contains("16")
            || ["smpte2084", "arib-std-b67"].contains(probe.video?.color_transfer ?? ""))
        if o.dryRun {
            log("vergence: \(name): remux with ffmpeg first: \(plan.notes.joined(separator: ", "))")
            return false
        }
    }
    let source = try await SpatialVideoSource.open(readable)
    if source.isMultiviewStereo {
        log("vergence: \(name): already 3D (MV-HEVC spatial video)\(o.upscale != nil ? "; upscaling spatial video isn't done yet" : ""), skipped")
        return false
    }
    // 2D or 3D: what --input-layout says, else what the file declares, and
    // when upscaling (where a 3D source is worth taking), its name and picture.
    var stereoSource: PackedStereoLayout?
    var stereoNote: String?
    let declared = source.declaredLayout ?? containerLayout
    switch o.inputLayout {
    case "auto":
        if let declared {
            stereoSource = declared
            stereoNote = "declared by the file"
        } else if o.upscale != nil {
            let pictures = (try? await source.samplePictures()) ?? []
            let detection = StereoLayoutDetector.detect(declared: nil, fileName: input.lastPathComponent,
                                                        frameSize: source.frameSize, pictures: pictures)
            if let found = detection.layout, detection.confidence >= 0.6 {
                stereoSource = found
                stereoNote = String(format: "%.0f%% sure: %@", detection.confidence * 100, detection.reasons.joined(separator: "; "))
            }
        }
    case "2d": break
    default:
        let given = PackedStereoLayout(arrangement: o.inputLayout.contains("sbs") ? .sideBySide : .overUnder,
                                       isHalfResolution: o.inputLayout.hasPrefix("h"))
        stereoSource = given
        stereoNote = "--input-layout \(o.inputLayout)"
    }
    if let stereoSource, o.upscale == nil {
        log("vergence: \(name): already 3D (\(stereoSource.arrangement == .sideBySide ? "side by side" : "over-under")), skipped"
            + " (--upscale 2x upscales its eyes)")
        return false
    }
    let layout = o.convertLayout
    let mode: PictureMode
    if let stereoSource {
        guard layout != .flat else {
            throw FormatError("it is 3D (\(stereoNote ?? "")); --layout 2d is for 2D sources (--input-layout 2d if it isn't 3D)")
        }
        mode = .stereoUpscale(stereoSource)
    } else {
        mode = layout == .flat ? .flatUpscale : .depth
    }
    let upscaling = o.upscale != nil
    let network = upscaling ? try await models.upscaler() : nil
    let model: URL? = if case .depth = mode, !o.dryRun { try await models.depth() } else { nil }

    // The part to convert.
    let fps = source.frameRate > 0 ? source.frameRate : 30
    let start = o.seek ?? o.sample.map { max(0, (source.duration - $0) / 2) } ?? 0
    guard start < source.duration else { throw FormatError("--start \(start) is past the end (\(clockTime(source.duration)))") }
    var length = min(o.sample ?? .infinity, source.duration - start)
    // The depth window looks two frames ahead; give it those.
    if o.frames != Int.max { length = min(length, (Double(o.frames) + 2.5) / fps) }
    let whole = start == 0 && length >= source.duration
    let range = whole ? nil : CMTimeRange(start: CMTime(seconds: start, preferredTimescale: 90_000),
                                          duration: CMTime(seconds: length, preferredTimescale: 90_000))
    if o.aiHDR, source.isHDR { throw FormatError("--ai-hdr makes SDR video HDR; this is HDR already") }
    // Each eye as it shows (a squeezed eye stretched back out), then as it
    // is made: twice that with --upscale, capped by --max-eye.
    let frame = (width: Double(source.frameSize.width), height: Double(source.frameSize.height))
    let sourceEye: (width: Double, height: Double)
    var regions: [FrameUpscaler.Region] = [.init(x: 0, y: 0, width: frame.width, height: frame.height)]
    if case .stereoUpscale(let packing) = mode {
        let sideBySide = packing.arrangement == .sideBySide
        let half = (width: sideBySide ? frame.width / 2 : frame.width, height: sideBySide ? frame.height : frame.height / 2)
        sourceEye = packing.isHalfResolution
            ? (sideBySide ? (half.width * 2, half.height) : (half.width, half.height * 2)) : half
        let first = FrameUpscaler.Region(x: 0, y: 0, width: half.width, height: half.height)
        let second = FrameUpscaler.Region(x: sideBySide ? half.width : 0, y: sideBySide ? 0 : half.height,
                                          width: half.width, height: half.height)
        regions = packing.eyesReversed ? [second, first] : [first, second]
    } else {
        sourceEye = frame
    }
    let eye = eyeSize(width: sourceEye.width * (upscaling ? 2 : 1), height: sourceEye.height * (upscaling ? 2 : 1), maxEye: o.maxEye)
    // The upscaler's input: half the eye made, even.
    let modelEye = (width: eye.width / 2 & ~1, height: eye.height / 2 & ~1)
    let made = upscaling ? (width: modelEye.width * 2, height: modelEye.height * 2) : eye
    let packed = layout == .flat ? made
        : layout.isSideBySide ? (width: made.width * 2, height: made.height) : (width: made.width, height: made.height * 2)
    // Half layouts squeeze both eyes into one frame's size.
    let picture = layout.isHalf ? made : packed
    // AI HDR is 10-bit BT.2020 PQ whatever the source was.
    let tenBit = source.needsTenBit && !eightBitSource || o.aiHDR
    let strength = Float(o.strength ?? Pseudo3DSettings.default.depthStrength)
    let convergence = Float(o.convergence ?? Pseudo3DSettings.default.convergence)

    // Frames a pulldown repeats, judged on a few stretches across the file.
    let repeats = o.dryRun ? nil : try await repeatedFrames(readable)
    let dropRepeats = repeats?.isPulldown == true && !o.keepRepeatedFrames
    let outputFPS = dropRepeats ? fps * 4 / 5 : fps

    // What the encoder may spend: the source's own rate, scaled for what is new.
    let sourceShare: Double = switch mode {
    case .depth: (layout.isHalf ? 1.2 : 2) * (upscaling ? 2.5 : 1)
    case .stereoUpscale: 2.5 * (layout.isHalf ? 0.6 : 1)
    case .flatUpscale: 2.5
    }
    let bitsPerSecond = o.bitrate ?? SpatialBitrate.bitsPerSecond(
        eyeWidth: picture.width, eyeHeight: picture.height, frameRate: outputFPS,
        quality: layout == .spatial ? .max : o.quality,
        sourceBitsPerSecond: source.videoBitsPerSecond.map { $0 * sourceShare }, secondEye: false)

    log("vergence: \(name) -> \(output.path)")
    let settings: String = switch mode {
    case .depth: String(format: "strength %.4f, convergence %.2f, model %@", strength, convergence,
                        model?.deletingPathExtension().lastPathComponent ?? "not looked up")
    case .stereoUpscale(let packing):
        "3D source (\(packing.arrangement == .sideBySide ? "side by side" : "top and bottom")"
            + "\(packing.isHalfResolution ? ", half" : "")\(packing.eyesReversed ? ", eyes reversed" : ""); \(stereoNote ?? "")), no depth"
    case .flatUpscale: "2D"
    }
    log(String(format: "  %@, %d×%d (%@ %d×%d%@), HEVC %.1f Mbps%@, %@",
               layout == .flat ? "2D" : layoutDescription(layout), picture.width, picture.height,
               layout == .flat ? "picture" : "eyes", made.width, made.height,
               layout.isHalf ? ", squeezed to half" : "", Double(bitsPerSecond) / 1_000_000,
               tenBit ? (source.isHDR ? ", 10-bit HDR" : ", 10-bit") : "", settings))
    if let network {
        log("  upscaled 2× from \(modelEye.width)×\(modelEye.height) by \(network.name) (Real-ESRGAN, \(network.convolutions) convolutions)")
    }
    if o.antialias { log("  antialiased: each eye's straight edges averaged along themselves (edge-tangent AA, last before encoding)") }
    if o.aiHDR { log("  AI HDR: GMNet gain grids, highlights spread from the mid-tones, BT.2020 PQ, SDR white at 203 nits, average gain held to 2.5×") }
    if let repeats, repeats.isPulldown {
        log(String(format: "  frames: one in five repeats the one before (%d of %d moving stretches): %@", repeats.repeating,
                   repeats.moving, dropRepeats ? String(format: "dropped, %.3f fps becomes %.3f", fps, outputFPS)
                                               : "kept (--keep-repeated-frames)"))
    }
    if !whole {
        log(String(format: "  from %@ for %.1f s%@", clockTime(start), length,
                   o.frames != Int.max ? ", \(o.frames) frames" : ""))
    }
    guard !o.dryRun else { return false }
    if case .depth = mode, model == nil { return false }
    let upscaler: FrameUpscaler? = if let network {
        try await FrameUpscaler(network: network, width: modelEye.width, height: modelEye.height, eyes: regions.count)
    } else { nil }

    let began = Date()
    // Progress is redrawn in place on a terminal; with VERGENCE_PROGRESS=1 (a
    // caller reading a pipe, as the desktop app on Windows does) it comes as lines.
    let terminal = isatty(STDERR_FILENO) != 0
    let interactive = terminal || ProcessInfo.processInfo.environment["VERGENCE_PROGRESS"] == "1"
    func report(_ label: String) -> @Sendable (Double) -> Void {
        { fraction in
            guard interactive else { return }
            FileHandle.standardError.write(Data(String(format: terminal ? "\r  %@ %3.0f%%" : "  %@ %3.0f%%\n", label,
                                                       fraction * 100).utf8))
        }
    }

    // Pass 1: depth, or the finished depth of a run that stopped.
    let depthFolder = work.appendingPathComponent("depth")
    let expectedDuration = whole ? source.duration : min(source.duration, start + length) - start
    var entry: DepthCacheStore.Entry?
    if let model {
    if let earlier = DepthCacheStore.entry(at: depthFolder),
       earlier.meta.modelName == model.deletingPathExtension().lastPathComponent,
       abs(earlier.meta.duration - expectedDuration) < 0.05,
       whole || abs((earlier.meta.framePTS.first ?? -1) - start) < 1.5 / fps {
        entry = earlier
        log("  depth: \(earlier.meta.frameCount) frames from an earlier run")
    } else {
        let depthBegan = Date()
        let made = try await OfflineDepthConversion.convert(video: readable, into: depthFolder, model: model, timeRange: range,
                                                            progress: report("depth"))
        entry = made
        if terminal { FileHandle.standardError.write(Data("\r".utf8)) }
        let seconds = Date().timeIntervalSince(depthBegan)
        log(String(format: "  depth: %d frames in %.1f s (%.1f fps)", made.meta.frameCount, seconds,
                   Double(made.meta.frameCount) / max(seconds, 0.001)))
    }
    if let path = o.depthTrack, let entry {
        let url = URL(fileURLWithPath: path)
        let records = try await OfflineDepthConversion.writeAnalysisTrack(
            entry, to: url, mediaID: mediaID(of: input), producer: "vergence convert",
            sourceWidth: Int(source.frameSize.width), sourceHeight: Int(source.frameSize.height))
        log("  depth track: \(records) records -> \(url.path)")
    }
    }

    // AI HDR's grids, as the ai-hdr workload makes them (from the left eye of a 3D source).
    var grids: AIHDRGrids?
    if o.aiHDR {
        let began = Date()
        let eye: PackedStereoLayout? = if case .stereoUpscale(let packing) = mode { packing } else { nil }
        grids = try await AIHDRGrids.make(readable, start: start, length: whole ? nil : length, frameRate: fps, eye: eye,
                                          model: try AIHDRGrids.model(o))
        let seconds = Date().timeIntervalSince(began)
        log(String(format: "  AI HDR grids: %d frames, %d keyframes in %.1f s (%.1f fps)", grids!.grids.count, grids!.keyframes,
                   seconds, Double(grids!.grids.count) / max(seconds, 0.001)))
    }

    // Pass 2: both eyes, encoded.
    let pictureBegan = Date()
    let stereoTarget = layout == .spatial ? work.appendingPathComponent("stereo.mov") : output
    let job = StereoEncodeJob(
        input: readable, output: stereoTarget, fileType: layout == .spatial ? .mov : .mp4, source: source, entry: entry,
        range: range, frameLimit: o.frames == Int.max ? nil : o.frames,
        arrangement: layout.isSideBySide ? .sideBySide : .topAndBottom, eye: made, picture: picture,
        strength: strength, convergence: convergence, bitsPerSecond: bitsPerSecond, tenBit: tenBit,
        includeAudio: o.includeAudio, progress: report("picture"), mode: mode, upscaler: upscaler, regions: regions,
        dropRepeats: dropRepeats, outputFrameRate: outputFPS, gains: grids,
        antialias: o.antialias && !o.dryRun ? try EdgeAntialias() : nil)
    let result = try await encodeStereo(job)
    if terminal { FileHandle.standardError.write(Data("\r".utf8)) }
    let pictureSeconds = Date().timeIntervalSince(pictureBegan)
    log(String(format: "  picture: %d frames in %.1f s (%.1f fps)%@%@; %d audio track(s) copied%@", result.frames, pictureSeconds,
               Double(result.frames) / max(pictureSeconds, 0.001),
               result.flatFrames > 0 ? ", \(result.flatFrames) without depth (flat)" : "",
               result.droppedRepeats > 0 ? ", \(result.droppedRepeats) repeats dropped" : "", result.audioCopied,
               result.audioSkipped.isEmpty ? "" : ", left out (MP4 can't hold them): \(result.audioSkipped.joined(separator: ", "))"))

    if layout == .spatial {
        var camera = o.preset.metadata
        if let v = o.baseline { camera.baselineMillimeters = v }
        if let v = o.hfov { camera.horizontalFieldOfViewDegrees = v }
        if let v = o.adjust { camera.horizontalDisparityAdjustment = v }
        guard camera.problems.isEmpty else { throw SpatialVideoError.invalidMetadata(camera.problems) }
        let stereo = try await SpatialVideoSource.open(stereoTarget)
        let spatial = SpatialVideoJob(source: stereo, output: output,
                                      layout: PackedStereoLayout(arrangement: .sideBySide, isHalfResolution: false),
                                      camera: camera, quality: o.quality, bitsPerSecond: o.bitrate,
                                      includeAudio: o.includeAudio, tenBit: tenBit)
        let made = try await SpatialVideoMaker.make(spatial, progress: report("spatial"))
        if terminal { FileHandle.standardError.write(Data("\r".utf8)) }
        log(String(format: "  spatial video: %d frames, %.1f Mbps", made.frames, Double(made.bitsPerSecond) / 1_000_000))
    } else if layout != .flat {
        try SphericalMetadata.write(to: output, stereoMode: layout.isSideBySide ? .leftRight : .topBottom)
    }
    try? FileManager.default.removeItem(at: work)
    let seconds = Date().timeIntervalSince(began)
    log(String(format: "  done: %d frames in %.1f s, %.1f fps overall", result.frames, seconds,
               Double(result.frames) / max(seconds, 0.001)))
    return true
}

func layoutDescription(_ layout: ConvertLayout) -> String {
    switch layout {
    case .fullSideBySide: "full side by side"
    case .halfSideBySide: "half side by side"
    case .fullTopAndBottom: "full top and bottom"
    case .halfTopAndBottom: "half top and bottom"
    case .spatial: "spatial video (MV-HEVC)"
    case .flat: "2D"
    }
}

/// One eye: the frame's size, capped by --max-eye, even.
func eyeSize(_ source: SpatialVideoSource, maxEye: Int?) -> (width: Int, height: Int) {
    eyeSize(width: Double(source.frameSize.width), height: Double(source.frameSize.height), maxEye: maxEye)
}

func eyeSize(width: Double, height: Double, maxEye: Int?) -> (width: Int, height: Int) {
    var w = width, h = height
    if let cap = maxEye, max(w, h) > Double(cap) {
        let scale = Double(cap) / max(w, h)
        w *= scale
        h *= scale
    }
    return (Int(w.rounded()) & ~1, Int(h.rounded()) & ~1)
}

/// An opaque id for the media in an analysis track: its name and size
/// hashed, never the name itself.
func mediaID(of url: URL) -> String {
    let size = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
    var hash: UInt64 = 0xCBF2_9CE4_8422_2325
    for byte in "\(url.lastPathComponent)\u{0}\(size)".utf8 { hash = (hash ^ UInt64(byte)) &* 0x100_0000_01B3 }
    return String(hash, radix: 16)
}

// MARK: - Pass 2

struct StereoEncodeJob: @unchecked Sendable {
    var input: URL
    var output: URL
    var fileType: AVFileType
    var source: SpatialVideoSource
    var entry: DepthCacheStore.Entry?
    var range: CMTimeRange?
    var frameLimit: Int?
    var arrangement: DepthStereoFrameRenderer.Arrangement
    var eye: (width: Int, height: Int)
    var picture: (width: Int, height: Int)
    var strength: Float
    var convergence: Float
    var bitsPerSecond: Int
    var tenBit: Bool
    var includeAudio: Bool
    var progress: @Sendable (Double) -> Void
    var mode: PictureMode = .depth
    /// Upscales each frame (both eyes of a 3D one) first.
    var upscaler: FrameUpscaler?
    /// Where each eye is in the source frame, for the upscaler.
    var regions: [FrameUpscaler.Region] = []
    /// Drop one frame in five, the repeat, and space the rest evenly.
    var dropRepeats = false
    var outputFrameRate: Double = 30
    /// AI HDR: each frame's grid, by its time from the part's start.
    var gains: AIHDRGrids?
    /// --antialias: each eye of the finished picture, before AI HDR.
    var antialias: EdgeAntialias?
}

struct StereoEncodeResult: Sendable {
    var frames = 0
    var flatFrames = 0
    var droppedRepeats = 0
    var audioCopied = 0
    var audioSkipped: [String] = []
}

/// Codecs an MP4 (and QuickTime) file takes as they are.
let copiedAudioCodecs: Set<String> = ["aac", "ac3", "eac3", "alac", "mp3"]

func encodeStereo(_ job: StereoEncodeJob) async throws -> StereoEncodeResult {
    let asset = AVURLAsset(url: job.input)
    guard let videoTrack = try await asset.loadTracks(withMediaType: .video).first else { throw SpatialVideoError.noVideoTrack }
    var audio: [(track: AVAssetTrack, format: CMFormatDescription, codec: String)] = []
    var skipped: [String] = []
    if job.includeAudio {
        let infos = try await AudioTrackReader.tracks(job.input)
        for (track, info) in zip(try await asset.loadTracks(withMediaType: .audio), infos) {
            guard copiedAudioCodecs.contains(info.codec), let format = try await track.load(.formatDescriptions).first else {
                skipped.append(info.codec)
                continue
            }
            audio.append((track, format, info.codec))
        }
    }
    let videoStart = try await videoTrack.load(.timeRange).start
    let videoFormat = try await videoTrack.load(.formatDescriptions).first
    // The frame loop blocks (decoding, the GPU, the encoder), so it runs on a
    // thread of its own, not Swift concurrency's pool.
    struct Inputs: @unchecked Sendable {
        let asset: AVURLAsset
        let video: AVAssetTrack
        let videoStart: CMTime
        let videoFormat: CMFormatDescription?
        let audio: [(track: AVAssetTrack, format: CMFormatDescription, codec: String)]
    }
    let inputs = Inputs(asset: asset, video: videoTrack, videoStart: videoStart, videoFormat: videoFormat, audio: audio)
    var result = try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<StereoEncodeResult, Error>) in
        Thread.detachNewThread {
            do { continuation.resume(returning: try encodeStereoFrames(job, inputs.asset, inputs.video, inputs.videoStart,
                                                                             inputs.videoFormat, inputs.audio)) }
            catch { continuation.resume(throwing: error) }
        }
    }
    result.audioSkipped = skipped
    return result
}

private func encodeStereoFrames(_ job: StereoEncodeJob, _ asset: AVURLAsset, _ videoTrack: AVAssetTrack, _ videoStart: CMTime,
                                _ videoFormat: CMFormatDescription?,
                                _ audio: [(track: AVAssetTrack, format: CMFormatDescription, codec: String)]) throws -> StereoEncodeResult {
    guard let metal = VergenceMetal.shared, let renderer = DepthStereoFrameRenderer() else {
        throw FormatError("the stereo warp isn't available (no Metal or no shaders)")
    }
    if case .depth = job.mode, job.entry == nil { throw FormatError("no depth to warp from") }
    if job.upscaler == nil, !{ if case .depth = job.mode { true } else { false } }() {
        throw FormatError("nothing to upscale with")
    }
    let source = job.source
    let hdr = source.isHDR
    // AI HDR draws in half floats (SDR code values) and writes PQ into them.
    let aiHDR = job.gains != nil
    let frameFormat = hdr || aiHDR ? FramePixelFormat.hdr : FramePixelFormat.sdr
    let encodeFormat = job.tenBit ? kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange : kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
    let packedSize: (width: Int, height: Int) = if case .flatUpscale = job.mode { job.eye }
        else { job.arrangement.packedSize(width: job.eye.width, height: job.eye.height) }

    let reader = try AVAssetReader(asset: asset)
    if let range = job.range { reader.timeRange = range }
    var videoSettings: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String: frameFormat,
        kCVPixelBufferMetalCompatibilityKey as String: true,
        kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
    ]
    // The upscaler reads the frame as it is; the warp takes it at the eye's size.
    if job.upscaler == nil, job.eye.width != Int(source.frameSize.width) || job.eye.height != Int(source.frameSize.height) {
        videoSettings[kCVPixelBufferWidthKey as String] = job.eye.width
        videoSettings[kCVPixelBufferHeightKey as String] = job.eye.height
    }
    let videoOutput = AVAssetReaderTrackOutput(track: videoTrack, outputSettings: videoSettings)
    videoOutput.alwaysCopiesSampleData = false
    reader.add(videoOutput)
    var audioOutputs: [(AVAssetReaderTrackOutput, CMFormatDescription)] = []
    for track in audio {
        let output = AVAssetReaderTrackOutput(track: track.track, outputSettings: nil)
        guard reader.canAdd(output) else { continue }
        reader.add(output)
        audioOutputs.append((output, track.format))
    }
    guard reader.startReading() else { throw SpatialVideoError.unreadable(reader.error?.localizedDescription ?? "unknown") }
    let feed = FrameFeed(videoOutput, dropRepeats: job.dropRepeats,
                         frameDuration: CMTime(seconds: 1 / job.outputFrameRate, preferredTimescale: 90_000))
    // The first frame, before the encoder is set up: an HDR picture's
    // mastering display and light levels often come only with its frames.
    var nextVideo = feed.next()
    guard let firstFrame = nextVideo.flatMap({ CMSampleBufferGetImageBuffer($0.sample) }) else {
        throw SpatialVideoError.unreadable(reader.error?.localizedDescription ?? "no video frames in the range")
    }

    let partial = job.output.deletingPathExtension().appendingPathExtension("partial." + job.output.pathExtension)
    try? FileManager.default.removeItem(at: partial)
    let writer = try AVAssetWriter(outputURL: partial, fileType: job.fileType)
    // moov at the end, so the st3d box can be added without moving media.
    writer.shouldOptimizeForNetworkUse = false
    let primaries = aiHDR ? AVVideoColorPrimaries_ITU_R_2020
        : source.colorPrimaries ?? (hdr ? AVVideoColorPrimaries_ITU_R_2020 : AVVideoColorPrimaries_ITU_R_709_2)
    let transferFunction = aiHDR ? AVVideoTransferFunction_SMPTE_ST_2084_PQ
        : source.transferFunction ?? (hdr ? AVVideoTransferFunction_SMPTE_ST_2084_PQ : AVVideoTransferFunction_ITU_R_709_2)
    let matrix = aiHDR ? AVVideoYCbCrMatrix_ITU_R_2020
        : source.yCbCrMatrix ?? (hdr ? AVVideoYCbCrMatrix_ITU_R_2020 : AVVideoYCbCrMatrix_ITU_R_709_2)
    var compression: [String: Any] = [
        AVVideoAverageBitRateKey: job.bitsPerSecond,
        AVVideoExpectedSourceFrameRateKey: job.outputFrameRate,
        AVVideoMaxKeyFrameIntervalDurationKey: 2,
        AVVideoAllowFrameReorderingKey: true,
    ]
    if job.tenBit { compression[AVVideoProfileLevelKey] = kVTProfileLevel_HEVC_Main10_AutoLevel as String }
    if aiHDR {
        // HDR10 static metadata for an SDR picture made HDR (GainGridPass.staticMetadata).
        compression[kVTCompressionPropertyKey_MasteringDisplayColorVolume as String] = GainGridPass.staticMetadata.masteringDisplay
        compression[kVTCompressionPropertyKey_ContentLightLevelInfo as String] = GainGridPass.staticMetadata.contentLightLevel
    }
    if hdr {
        for (extensionKey, frameKey, propertyKey) in [
            (kCMFormatDescriptionExtension_MasteringDisplayColorVolume, kCVImageBufferMasteringDisplayColorVolumeKey,
             kVTCompressionPropertyKey_MasteringDisplayColorVolume),
            (kCMFormatDescriptionExtension_ContentLightLevelInfo, kCVImageBufferContentLightLevelInfoKey,
             kVTCompressionPropertyKey_ContentLightLevelInfo),
        ] {
            let value = videoFormat.flatMap { CMFormatDescriptionGetExtension($0, extensionKey: extensionKey) }
                ?? CVBufferCopyAttachment(firstFrame, frameKey, nil)
            if let value { compression[propertyKey as String] = value }
        }
    }
    let videoInput = AVAssetWriterInput(mediaType: .video, outputSettings: [
        AVVideoCodecKey: AVVideoCodecType.hevc, AVVideoWidthKey: job.picture.width, AVVideoHeightKey: job.picture.height,
        AVVideoCompressionPropertiesKey: compression,
        AVVideoColorPropertiesKey: [AVVideoColorPrimariesKey: primaries, AVVideoTransferFunctionKey: transferFunction,
                                    AVVideoYCbCrMatrixKey: matrix],
    ])
    videoInput.expectsMediaDataInRealTime = false
    let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: videoInput, sourcePixelBufferAttributes: [
        kCVPixelBufferPixelFormatTypeKey as String: encodeFormat,
        kCVPixelBufferWidthKey as String: job.picture.width, kCVPixelBufferHeightKey as String: job.picture.height,
        kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
    ])
    guard writer.canAdd(videoInput) else {
        throw SpatialVideoError.writer("the encoder refused \(job.picture.width)×\(job.picture.height) HEVC")
    }
    writer.add(videoInput)

    // A track the writer refuses is still read (and dropped), or the reader
    // would stall waiting for it to be taken.
    var pairs: [(AVAssetReaderTrackOutput, AVAssetWriterInput?)] = []
    for (output, format) in audioOutputs {
        let input = AVAssetWriterInput(mediaType: .audio, outputSettings: nil, sourceFormatHint: format)
        input.expectsMediaDataInRealTime = false
        if writer.canAdd(input) {
            writer.add(input)
            pairs.append((output, input))
        } else {
            pairs.append((output, nil))
        }
    }

    guard let packedPool = FramePixelFormat.pool(format: frameFormat, width: packedSize.width, height: packedSize.height)
    else { throw SpatialVideoError.writer("can't make packed buffers") }
    // A 2D frame upscaled before the warp.
    let upscaledPool = job.upscaler != nil && { if case .depth = job.mode { true } else { false } }()
        ? FramePixelFormat.pool(format: frameFormat, width: job.eye.width, height: job.eye.height) : nil
    var transferOut: VTPixelTransferSession?
    VTPixelTransferSessionCreate(allocator: kCFAllocatorDefault, pixelTransferSessionOut: &transferOut)
    guard let transfer = transferOut else { throw SpatialVideoError.writer("can't create a pixel transfer session") }
    defer { VTPixelTransferSessionInvalidate(transfer) }
    // The source's colour in and out: only the format and size change.
    VTSessionSetProperty(transfer, key: kVTPixelTransferPropertyKey_DestinationColorPrimaries, value: primaries as CFString)
    VTSessionSetProperty(transfer, key: kVTPixelTransferPropertyKey_DestinationTransferFunction, value: transferFunction as CFString)
    VTSessionSetProperty(transfer, key: kVTPixelTransferPropertyKey_DestinationYCbCrMatrix, value: matrix as CFString)

    let depthSource = job.entry.map { CachedDepthSource(entry: $0, device: metal.device) }
    let baker = job.gains == nil ? nil : try AIHDRBaker()
    defer { depthSource?.invalidate() }

    guard writer.startWriting() else { throw SpatialVideoError.writer(writer.error?.localizedDescription ?? "unknown") }

    var result = StereoEncodeResult(audioCopied: pairs.filter { $0.1 != nil }.count)
    let first = job.range?.start.seconds ?? videoStart.seconds
    let span = max(job.range?.duration.seconds ?? source.duration, 0.001)
    let frameDuration = CMTime(seconds: 1 / job.outputFrameRate, preferredTimescale: 90_000)
    var sessionStarted = false
    var lastEnd: CMTime?
    var videoDone = false
    var audioDone = pairs.map { _ in false }

    do {
        while !videoDone || audioDone.contains(false) {
            var moved = false
            if !videoDone, videoInput.isReadyForMoreMediaData {
                moved = true
                if job.frameLimit.map({ result.frames >= $0 }) == true {
                    videoInput.markAsFinished()
                    videoDone = true
                } else if let (sample, time) = nextVideo ?? feed.next() {
                    nextVideo = nil
                    if let frame = CMSampleBufferGetImageBuffer(sample) {
                        if !sessionStarted {
                            writer.startSession(atSourceTime: time)
                            sessionStarted = true
                        }
                        try autoreleasepool {
                            var packedOut: CVPixelBuffer?
                            CVPixelBufferPoolCreatePixelBuffer(nil, packedPool, &packedOut)
                            guard let packed = packedOut else { throw SpatialVideoError.writer("out of packed buffers") }
                            switch job.mode {
                            case .depth:
                                // Depth by the frame's own time, wherever it lands in the output.
                                let depth = depthSource?.frameDepth(itemTime: CMSampleBufferGetPresentationTimeStamp(sample), frame: frame)
                                if depth == nil { result.flatFrames += 1 }
                                var picture = frame
                                if let upscaler = job.upscaler, let upscaledPool {
                                    var upscaledOut: CVPixelBuffer?
                                    CVPixelBufferPoolCreatePixelBuffer(nil, upscaledPool, &upscaledOut)
                                    guard let upscaled = upscaledOut else { throw SpatialVideoError.writer("out of upscaled buffers") }
                                    try upscaler.upscale(frame, regions: job.regions, into: upscaled, at: [(0, 0)])
                                    picture = upscaled
                                }
                                try renderer.render(frame: picture, depth: depth, strength: job.strength,
                                                    convergence: job.convergence, arrangement: job.arrangement, into: packed)
                            case .stereoUpscale:
                                let second = job.arrangement == .sideBySide ? (job.eye.width, 0) : (0, job.eye.height)
                                try job.upscaler?.upscale(frame, regions: job.regions, into: packed, at: [(0, 0), second])
                            case .flatUpscale:
                                try job.upscaler?.upscale(frame, regions: job.regions, into: packed, at: [(0, 0)])
                            }
                            if let antialias = job.antialias {
                                let eyes: [EdgeAntialias.Eye]
                                if case .flatUpscale = job.mode {
                                    eyes = [.init(x: 0, y: 0, width: job.eye.width, height: job.eye.height)]
                                } else if job.arrangement == .sideBySide {
                                    eyes = [.init(x: 0, y: 0, width: job.eye.width, height: job.eye.height),
                                            .init(x: job.eye.width, y: 0, width: job.eye.width, height: job.eye.height)]
                                } else {
                                    eyes = [.init(x: 0, y: 0, width: job.eye.width, height: job.eye.height),
                                            .init(x: 0, y: job.eye.height, width: job.eye.width, height: job.eye.height)]
                                }
                                try antialias.apply(packed, eyes: eyes)
                            }
                            if let gains = job.gains, let baker,
                               let grid = gains.grid(at: CMSampleBufferGetPresentationTimeStamp(sample).seconds - first) {
                                // The left (first) eye's grid and capped scale for every eye, as a player draws them.
                                let origins: [(x: Int, y: Int)]
                                if case .flatUpscale = job.mode { origins = [(0, 0)] }
                                else if job.arrangement == .sideBySide { origins = [(0, 0), (job.eye.width, 0)] }
                                else { origins = [(0, 0), (0, job.eye.height)] }
                                try baker.bake(packed, grid: grid, eye: job.eye, origins: origins)
                                // PQ BT.2020 already: the transfer only changes the format.
                                CVBufferSetAttachment(packed, kCVImageBufferColorPrimariesKey, kCVImageBufferColorPrimaries_ITU_R_2020, .shouldPropagate)
                                CVBufferSetAttachment(packed, kCVImageBufferTransferFunctionKey, kCVImageBufferTransferFunction_SMPTE_ST_2084_PQ,
                                                      .shouldPropagate)
                                CVBufferSetAttachment(packed, kCVImageBufferYCbCrMatrixKey, kCVImageBufferYCbCrMatrix_ITU_R_2020, .shouldPropagate)
                            }
                            guard let pool = adaptor.pixelBufferPool else { throw SpatialVideoError.writer("the writer has no buffer pool") }
                            var encodedOut: CVPixelBuffer?
                            CVPixelBufferPoolCreatePixelBuffer(nil, pool, &encodedOut)
                            guard let encoded = encodedOut else { throw SpatialVideoError.writer("out of pixel buffers") }
                            let status = VTPixelTransferSessionTransferImage(transfer, from: packed, to: encoded)
                            guard status == noErr else { throw SpatialVideoError.writer("converting a frame failed (\(status))") }
                            if hdr { copyHDRMetadata(from: frame, to: encoded) }
                            guard adaptor.append(encoded, withPresentationTime: time) else {
                                throw SpatialVideoError.writer(writer.error?.localizedDescription ?? "frame \(result.frames) was refused")
                            }
                        }
                        result.frames += 1
                        lastEnd = time + (!job.dropRepeats && CMSampleBufferGetDuration(sample).isValid
                                          && CMSampleBufferGetDuration(sample) > .zero ? CMSampleBufferGetDuration(sample) : frameDuration)
                        if result.frames % 24 == 0 { job.progress(min(1, (time.seconds - first) / span)) }
                    }
                } else {
                    videoInput.markAsFinished()
                    videoDone = true
                }
            }
            // Audio waits until the picture has started the session, and stops
            // where a --frames limit ended the picture.
            if sessionStarted {
                for (i, (output, input)) in pairs.enumerated() where !audioDone[i] && input?.isReadyForMoreMediaData != false {
                    moved = true
                    guard let sample = output.copyNextSampleBuffer() else {
                        input?.markAsFinished()
                        audioDone[i] = true
                        continue
                    }
                    if videoDone, let lastEnd, CMSampleBufferGetPresentationTimeStamp(sample) >= lastEnd {
                        input?.markAsFinished()
                        audioDone[i] = true
                        continue
                    }
                    guard let input else { continue }
                    guard input.append(sample) else { throw SpatialVideoError.writer(writer.error?.localizedDescription ?? "audio refused") }
                }
            } else if videoDone {
                throw SpatialVideoError.unreadable("no video frames in the range")
            }
            if reader.status == .failed { throw SpatialVideoError.unreadable(reader.error?.localizedDescription ?? "unknown") }
            if writer.status == .failed { throw SpatialVideoError.writer(writer.error?.localizedDescription ?? "unknown") }
            if !moved { Thread.sleep(forTimeInterval: 0.002) }
        }
        if let lastEnd { writer.endSession(atSourceTime: lastEnd) }
        let finished = DispatchSemaphore(value: 0)
        writer.finishWriting { finished.signal() }
        finished.wait()
        guard writer.status == .completed else { throw SpatialVideoError.writer(writer.error?.localizedDescription ?? "unknown") }
    } catch {
        reader.cancelReading()
        writer.cancelWriting()
        try? FileManager.default.removeItem(at: partial)
        throw error
    }
    reader.cancelReading()
    try? FileManager.default.removeItem(at: job.output)
    try FileManager.default.moveItem(at: partial, to: job.output)
    job.progress(1)
    result.droppedRepeats = feed.dropped
    return result
}

/// The video's frames in order, with the times to show them at: their own,
/// or with a pulldown's repeats dropped (one in five, `RepeatDropper`), the
/// rest evenly spaced from the first.
final class FrameFeed {
    private let output: AVAssetReaderTrackOutput
    private let dropRepeats: Bool
    private let frameDuration: CMTime
    private var dropper = RepeatDropper()
    private var held: [Int: CMSampleBuffer] = [:]
    private var ready: [CMSampleBuffer] = []
    private var previous: [Float]?
    private var count = 0
    private var emitted = 0
    private var first: CMTime?
    private var ended = false
    private(set) var dropped = 0

    init(_ output: AVAssetReaderTrackOutput, dropRepeats: Bool, frameDuration: CMTime) {
        self.output = output
        self.dropRepeats = dropRepeats
        self.frameDuration = frameDuration
    }

    func next() -> (sample: CMSampleBuffer, time: CMTime)? {
        while ready.isEmpty, !ended { pull() }
        guard !ready.isEmpty else { return nil }
        let sample = ready.removeFirst()
        let own = CMSampleBufferGetPresentationTimeStamp(sample)
        guard dropRepeats else { return (sample, own) }
        let start = first ?? own
        first = start
        defer { emitted += 1 }
        return (sample, start + CMTimeMultiply(frameDuration, multiplier: Int32(emitted)))
    }

    private func pull() {
        guard let sample = output.copyNextSampleBuffer() else {
            ended = true
            for index in dropper.finish() { if let s = held.removeValue(forKey: index) { ready.append(s) } }
            return
        }
        guard let frame = CMSampleBufferGetImageBuffer(sample) else { return }
        guard dropRepeats else { ready.append(sample); return }
        let signature = FrameDifference.signature(frame)
        let difference = zip(previous, signature).flatMap { FrameDifference.between($0, $1) }
        previous = signature
        held[count] = sample
        count += 1
        let kept = dropper.add(difference)
        guard !kept.isEmpty else { return }
        for index in kept { if let s = held.removeValue(forKey: index) { ready.append(s) } }
        dropped += held.count
        held.removeAll()
    }
}

private func zip(_ a: [Float]?, _ b: [Float]?) -> ([Float], [Float])? {
    guard let a, let b else { return nil }
    return (a, b)
}

/// Whether the video repeats one frame in five: up to six stretches of 60
/// frames across it, each frame's difference to the one before.
func repeatedFrames(_ url: URL) async throws -> RepeatedFrames.Verdict? {
    let asset = AVURLAsset(url: url)
    guard let track = try await asset.loadTracks(withMediaType: .video).first else { return nil }
    let (size, rate) = try await track.load(.naturalSize, .nominalFrameRate)
    let span = try await asset.load(.duration).seconds
    guard span > 0, size.width > 0 else { return nil }
    let seconds = min(span, 61 / Double(max(rate, 1)))
    let stretches = span > 30 ? 6 : 1
    var runs: [[Float]] = []
    for i in 0 ..< stretches {
        let at = (span - seconds) * (stretches == 1 ? 0 : (0.1 + 0.8 * Double(i) / Double(stretches - 1)))
        let reader = try AVAssetReader(asset: asset)
        reader.timeRange = CMTimeRange(start: CMTime(seconds: at, preferredTimescale: 90_000),
                                       duration: CMTime(seconds: seconds, preferredTimescale: 90_000))
        let width = min(Int(size.width), 384) & ~1
        let height = max(48, Int((size.height * CGFloat(width) / size.width).rounded()) & ~1)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width, kCVPixelBufferHeightKey as String: height,
        ])
        output.alwaysCopiesSampleData = false
        reader.add(output)
        guard reader.startReading() else { continue }
        var previous: [Float]?
        var run: [Float] = []
        while let sample = output.copyNextSampleBuffer() {
            guard let frame = CMSampleBufferGetImageBuffer(sample), let signature = FrameDifference.signature(frame) else { continue }
            if let previous, let d = FrameDifference.between(previous, signature) { run.append(d) }
            previous = signature
        }
        reader.cancelReading()
        runs.append(run)
    }
    return RepeatedFrames.judge(runs)
}

/// An HDR picture's mastering display and light levels, for the encoder.
private func copyHDRMetadata(from frame: CVPixelBuffer, to encoded: CVPixelBuffer) {
    for key in [kCVImageBufferMasteringDisplayColorVolumeKey, kCVImageBufferContentLightLevelInfoKey,
                kCVImageBufferAmbientViewingEnvironmentKey] {
        if let value = CVBufferCopyAttachment(frame, key, nil) {
            CVBufferSetAttachment(encoded, key, value, .shouldPropagate)
        }
    }
}
#endif
