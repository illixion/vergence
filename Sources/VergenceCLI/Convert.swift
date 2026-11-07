/*
 Vergence - `convert`: any 2D video to an AI 3D stereo copy

   vergence convert VIDEO|FOLDER... [--output FILE|DIR] [--layout fsbs|hsbs|tab|htab|spatial]
                    [--strength S] [--convergence C] [--sample SECONDS] [--start S] [--frames N]
                    [--model PATH] [--download-model] [--depth-track FILE.vgat]
                    [--quality low|standard|high|max] [--bitrate MBPS] [--max-eye PIXELS] [--no-audio]
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
    let model = o.dryRun ? nil : try await depthModel(o)
    var made = 0, skipped = 0, failed = 0
    for input in inputs {
        let output = OutputNames.converted(for: input, o, single: inputs.count == 1)
        do {
            if try await convertOne(input, output, model: model, o) { made += 1 } else { skipped += 1 }
        } catch {
            log("vergence: \(input.lastPathComponent): \(error.localizedDescription)")
            failed += 1
        }
    }
    if inputs.count > 1 { log("vergence: \(made) converted, \(skipped) skipped, \(failed) failed") }
    if failed > 0 { exit(1) }
    #endif
}

#if canImport(AVFoundation) && os(macOS)

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

/// Converts one file. False when there was nothing to do.
func convertOne(_ input: URL, _ output: URL, model: URL?, _ o: Options) async throws -> Bool {
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
    if !FFmpeg.isNative(input) {
        let ffmpeg = try FFmpeg.required(for: "reading .\(input.pathExtension)")
        let probe = try ffmpeg.probe(input)
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
        log("vergence: \(name): already 3D (MV-HEVC spatial video), skipped")
        return false
    }
    if let declared = source.declaredLayout {
        log("vergence: \(name): already 3D (\(declared.arrangement == .sideBySide ? "side by side" : "over-under")), skipped")
        return false
    }

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

    let layout = o.convertLayout
    let eye = eyeSize(source, maxEye: o.maxEye)
    let packed = layout.isSideBySide ? (width: eye.width * 2, height: eye.height) : (width: eye.width, height: eye.height * 2)
    // Half layouts squeeze both eyes into one frame's size.
    let picture = layout.isHalf ? eye : packed
    let tenBit = source.needsTenBit && !eightBitSource
    let strength = Float(o.strength ?? Pseudo3DSettings.default.depthStrength)
    let convergence = Float(o.convergence ?? Pseudo3DSettings.default.convergence)
    let bitsPerSecond = o.bitrate ?? SpatialBitrate.bitsPerSecond(
        eyeWidth: picture.width, eyeHeight: picture.height, frameRate: fps,
        quality: layout == .spatial ? .max : o.quality,
        sourceBitsPerSecond: source.videoBitsPerSecond.map { $0 * (layout.isHalf ? 1.2 : 2) }, secondEye: false)

    log("vergence: \(name) -> \(output.path)")
    log(String(format: "  %@, %d×%d (eyes %d×%d%@), HEVC %.1f Mbps%@, strength %.4f, convergence %.2f, model %@",
               layoutDescription(layout), picture.width, picture.height, eye.width, eye.height,
               layout.isHalf ? ", squeezed to half" : "", Double(bitsPerSecond) / 1_000_000,
               tenBit ? (source.isHDR ? ", 10-bit HDR" : ", 10-bit") : "", strength, convergence,
               model?.deletingPathExtension().lastPathComponent ?? "not looked up"))
    if !whole {
        log(String(format: "  from %@ for %.1f s%@", clockTime(start), length,
                   o.frames != Int.max ? ", \(o.frames) frames" : ""))
    }
    guard !o.dryRun, let model else { return false }

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
    let entry: DepthCacheStore.Entry
    if let earlier = DepthCacheStore.entry(at: depthFolder),
       earlier.meta.modelName == model.deletingPathExtension().lastPathComponent,
       abs(earlier.meta.duration - expectedDuration) < 0.05,
       whole || abs((earlier.meta.framePTS.first ?? -1) - start) < 1.5 / fps {
        entry = earlier
        log("  depth: \(earlier.meta.frameCount) frames from an earlier run")
    } else {
        let depthBegan = Date()
        entry = try await OfflineDepthConversion.convert(video: readable, into: depthFolder, model: model, timeRange: range,
                                                         progress: report("depth"))
        if terminal { FileHandle.standardError.write(Data("\r".utf8)) }
        let seconds = Date().timeIntervalSince(depthBegan)
        log(String(format: "  depth: %d frames in %.1f s (%.1f fps)", entry.meta.frameCount, seconds,
                   Double(entry.meta.frameCount) / max(seconds, 0.001)))
    }
    if let path = o.depthTrack {
        let url = URL(fileURLWithPath: path)
        let records = try await OfflineDepthConversion.writeAnalysisTrack(
            entry, to: url, mediaID: mediaID(of: input), producer: "vergence convert",
            sourceWidth: Int(source.frameSize.width), sourceHeight: Int(source.frameSize.height))
        log("  depth track: \(records) records -> \(url.path)")
    }

    // Pass 2: both eyes, encoded.
    let pictureBegan = Date()
    let stereoTarget = layout == .spatial ? work.appendingPathComponent("stereo.mov") : output
    let job = StereoEncodeJob(
        input: readable, output: stereoTarget, fileType: layout == .spatial ? .mov : .mp4, source: source, entry: entry,
        range: range, frameLimit: o.frames == Int.max ? nil : o.frames,
        arrangement: layout.isSideBySide ? .sideBySide : .topAndBottom, eye: eye, picture: picture,
        strength: strength, convergence: convergence, bitsPerSecond: bitsPerSecond, tenBit: tenBit,
        includeAudio: o.includeAudio, progress: report("picture"))
    let result = try await encodeStereo(job)
    if terminal { FileHandle.standardError.write(Data("\r".utf8)) }
    let pictureSeconds = Date().timeIntervalSince(pictureBegan)
    log(String(format: "  picture: %d frames in %.1f s (%.1f fps)%@; %d audio track(s) copied%@", result.frames, pictureSeconds,
               Double(result.frames) / max(pictureSeconds, 0.001),
               result.flatFrames > 0 ? ", \(result.flatFrames) without depth (flat)" : "", result.audioCopied,
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
    } else {
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
    }
}

/// One eye: the frame's size, capped by --max-eye, even.
func eyeSize(_ source: SpatialVideoSource, maxEye: Int?) -> (width: Int, height: Int) {
    var w = Double(source.frameSize.width), h = Double(source.frameSize.height)
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
    var entry: DepthCacheStore.Entry
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
}

struct StereoEncodeResult: Sendable {
    var frames = 0
    var flatFrames = 0
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
    let source = job.source
    let hdr = source.isHDR
    let frameFormat = hdr ? FramePixelFormat.hdr : FramePixelFormat.sdr
    let encodeFormat = job.tenBit ? kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange : kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
    let packedSize = job.arrangement.packedSize(width: job.eye.width, height: job.eye.height)

    let reader = try AVAssetReader(asset: asset)
    if let range = job.range { reader.timeRange = range }
    var videoSettings: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String: frameFormat,
        kCVPixelBufferMetalCompatibilityKey as String: true,
        kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
    ]
    if job.eye.width != Int(source.frameSize.width) || job.eye.height != Int(source.frameSize.height) {
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
    // The first frame, before the encoder is set up: an HDR picture's
    // mastering display and light levels often come only with its frames.
    var nextVideo = videoOutput.copyNextSampleBuffer()
    guard let firstFrame = nextVideo.flatMap(CMSampleBufferGetImageBuffer) else {
        throw SpatialVideoError.unreadable(reader.error?.localizedDescription ?? "no video frames in the range")
    }

    let partial = job.output.deletingPathExtension().appendingPathExtension("partial." + job.output.pathExtension)
    try? FileManager.default.removeItem(at: partial)
    let writer = try AVAssetWriter(outputURL: partial, fileType: job.fileType)
    // moov at the end, so the st3d box can be added without moving media.
    writer.shouldOptimizeForNetworkUse = false
    let primaries = source.colorPrimaries ?? (hdr ? AVVideoColorPrimaries_ITU_R_2020 : AVVideoColorPrimaries_ITU_R_709_2)
    let transferFunction = source.transferFunction ?? (hdr ? AVVideoTransferFunction_SMPTE_ST_2084_PQ : AVVideoTransferFunction_ITU_R_709_2)
    let matrix = source.yCbCrMatrix ?? (hdr ? AVVideoYCbCrMatrix_ITU_R_2020 : AVVideoYCbCrMatrix_ITU_R_709_2)
    var compression: [String: Any] = [
        AVVideoAverageBitRateKey: job.bitsPerSecond,
        AVVideoExpectedSourceFrameRateKey: source.frameRate > 0 ? source.frameRate : 30,
        AVVideoMaxKeyFrameIntervalDurationKey: 2,
        AVVideoAllowFrameReorderingKey: true,
    ]
    if job.tenBit { compression[AVVideoProfileLevelKey] = kVTProfileLevel_HEVC_Main10_AutoLevel as String }
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
    var transferOut: VTPixelTransferSession?
    VTPixelTransferSessionCreate(allocator: kCFAllocatorDefault, pixelTransferSessionOut: &transferOut)
    guard let transfer = transferOut else { throw SpatialVideoError.writer("can't create a pixel transfer session") }
    defer { VTPixelTransferSessionInvalidate(transfer) }
    // The source's colour in and out: only the format and size change.
    VTSessionSetProperty(transfer, key: kVTPixelTransferPropertyKey_DestinationColorPrimaries, value: primaries as CFString)
    VTSessionSetProperty(transfer, key: kVTPixelTransferPropertyKey_DestinationTransferFunction, value: transferFunction as CFString)
    VTSessionSetProperty(transfer, key: kVTPixelTransferPropertyKey_DestinationYCbCrMatrix, value: matrix as CFString)

    let depthSource = CachedDepthSource(entry: job.entry, device: metal.device)
    defer { depthSource.invalidate() }

    guard writer.startWriting() else { throw SpatialVideoError.writer(writer.error?.localizedDescription ?? "unknown") }

    var result = StereoEncodeResult(audioCopied: pairs.filter { $0.1 != nil }.count)
    let first = job.range?.start.seconds ?? videoStart.seconds
    let span = max(job.range?.duration.seconds ?? source.duration, 0.001)
    let frameDuration = CMTime(seconds: 1 / (source.frameRate > 0 ? source.frameRate : 30), preferredTimescale: 90_000)
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
                } else if let sample = nextVideo ?? videoOutput.copyNextSampleBuffer() {
                    nextVideo = nil
                    if let frame = CMSampleBufferGetImageBuffer(sample) {
                        let time = CMSampleBufferGetPresentationTimeStamp(sample)
                        if !sessionStarted {
                            writer.startSession(atSourceTime: time)
                            sessionStarted = true
                        }
                        try autoreleasepool {
                            let depth = depthSource.frameDepth(itemTime: time, frame: frame)
                            if depth == nil { result.flatFrames += 1 }
                            var packedOut: CVPixelBuffer?
                            CVPixelBufferPoolCreatePixelBuffer(nil, packedPool, &packedOut)
                            guard let packed = packedOut else { throw SpatialVideoError.writer("out of packed buffers") }
                            try renderer.render(frame: frame, depth: depth, strength: job.strength,
                                                convergence: job.convergence, arrangement: job.arrangement, into: packed)
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
                        lastEnd = time + (CMSampleBufferGetDuration(sample).isValid && CMSampleBufferGetDuration(sample) > .zero
                                          ? CMSampleBufferGetDuration(sample) : frameDuration)
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
    return result
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
