/*
 Vergence - making and inspecting spatial video

 `SpatialVideoSource.open` reads what a file is (size, frame rate, colour,
 whether it is already MV-HEVC, a stereo layout its metadata declares) and
 can sample small greyscale pictures for `StereoLayoutDetector`.

 `SpatialVideoMaker.make` turns a frame-packed stereo file into MV-HEVC with
 the complete spatial metadata (`SpatialCameraMetadata`):
 - each eye is cropped out and, for a half format, stretched back to its
   display size by VideoToolbox, in the source's own pixel format, so 10-bit
   and HDR (PQ, HLG) sources stay 10-bit with their colour tags;
 - frames keep the source's timestamps (no re-timing, so variable frame rate
   stays in sync);
 - audio tracks are copied as they are;
 - it writes to `<output>.partial` and renames when done, so a file at the
   output path is always complete.

 `SpatialVideoInspector.inspect` reports what a file carries and what a
 player would miss.
 */

#if canImport(AVFoundation) && !os(tvOS)

import AVFoundation
import CoreGraphics
import CoreMedia
import CoreVideo
import VideoToolbox
import VergenceCore

public enum SpatialVideoError: Error, LocalizedError, Sendable {
    case noVideoTrack
    case unreadable(String)
    case alreadySpatial
    case invalidMetadata([String])
    case writer(String)

    public var errorDescription: String? {
        switch self {
        case .noVideoTrack: "the file has no video track"
        case .unreadable(let why): "can't read the video: \(why)"
        case .alreadySpatial: "the video is already MV-HEVC stereo"
        case .invalidMetadata(let problems): problems.joined(separator: "; ")
        case .writer(let why): "can't write the spatial video: \(why)"
        }
    }
}

/// What a video file is, as far as making spatial video goes.
public struct SpatialVideoSource: Sendable {
    public let url: URL
    /// The coded picture's size in pixels. A pixel aspect ratio is ignored:
    /// it is how players stretch a squeezed frame-packed file, and the
    /// layout already says how to un-squeeze each eye.
    public let frameSize: CGSize
    public let frameRate: Double
    public let duration: Double
    /// The video track's data rate, bits per second, when the file says.
    public let videoBitsPerSecond: Double?
    public let codec: String
    public let bitsPerComponent: Int
    public let colorPrimaries: String?
    public let transferFunction: String?
    public let yCbCrMatrix: String?
    /// Already MV-HEVC with two views.
    public let isMultiviewStereo: Bool
    /// A frame-packed layout the file's metadata declares.
    public let declaredLayout: PackedStereoLayout?
    public let audioTracks: [String]

    public var isHDR: Bool {
        transferFunction == (kCMFormatDescriptionTransferFunction_SMPTE_ST_2084_PQ as String)
            || transferFunction == (kCMFormatDescriptionTransferFunction_ITU_R_2100_HLG as String)
    }

    /// Ten bits a component on the way through, so nothing is lost.
    public var needsTenBit: Bool { bitsPerComponent > 8 || isHDR }

    public static func open(_ url: URL) async throws -> SpatialVideoSource {
        let asset = AVURLAsset(url: url)
        let track: AVAssetTrack
        do {
            guard let first = try await asset.loadTracks(withMediaType: .video).first else { throw SpatialVideoError.noVideoTrack }
            track = first
        } catch let error as SpatialVideoError {
            throw error
        } catch {
            throw SpatialVideoError.unreadable(error.localizedDescription)
        }
        let (size, rate, dataRate, descriptions) = try await track.load(.naturalSize, .nominalFrameRate, .estimatedDataRate, .formatDescriptions)
        let duration = try await asset.load(.duration).seconds
        let multiview = try await track.load(.mediaCharacteristics).contains(.containsStereoMultiviewVideo)
        let description = descriptions.first
        let coded = description.map { CMVideoFormatDescriptionGetDimensions($0) }
        let pixels = coded.map { CGSize(width: Int($0.width), height: Int($0.height)) } ?? size
        let extensions = description.flatMap { CMFormatDescriptionGetExtensions($0) as? [String: Any] } ?? [:]

        var declared: PackedStereoLayout?
        if #available(macOS 15, iOS 18, visionOS 2, *),
           let packing = extensions[kCMFormatDescriptionExtension_ViewPackingKind as String] as? String {
            if packing == kCMFormatDescriptionViewPackingKind_SideBySide as String {
                declared = PackedStereoLayout(arrangement: .sideBySide, isHalfResolution: false).correctingResolution(forPacked: pixels)
            } else if packing == kCMFormatDescriptionViewPackingKind_OverUnder as String {
                declared = PackedStereoLayout(arrangement: .overUnder, isHalfResolution: false).correctingResolution(forPacked: pixels)
            }
        }

        var audio: [String] = []
        for audioTrack in try await asset.loadTracks(withMediaType: .audio) {
            guard let format = try await audioTrack.load(.formatDescriptions).first,
                  let basic = CMAudioFormatDescriptionGetStreamBasicDescription(format)?.pointee else { continue }
            audio.append("\(fourCharacterCode(basic.mFormatID)) \(basic.mChannelsPerFrame) ch \(Int(basic.mSampleRate)) Hz")
        }

        return SpatialVideoSource(
            url: url, frameSize: pixels, frameRate: Double(rate), duration: duration,
            videoBitsPerSecond: dataRate > 0 ? Double(dataRate) : nil,
            codec: description.map { fourCharacterCode(CMFormatDescriptionGetMediaSubType($0)) } ?? "?",
            bitsPerComponent: bitDepth(description, extensions),
            colorPrimaries: extensions[kCMFormatDescriptionExtension_ColorPrimaries as String] as? String,
            transferFunction: extensions[kCMFormatDescriptionExtension_TransferFunction as String] as? String,
            yCbCrMatrix: extensions[kCMFormatDescriptionExtension_YCbCrMatrix as String] as? String,
            isMultiviewStereo: multiview, declaredLayout: declared, audioTracks: audio)
    }

    /// `count` greyscale pictures from across the video, at most `maxWidth`
    /// wide, for the layout detector.
    public func samplePictures(count: Int = 8, maxWidth: Int = 480) async throws -> [LumaPlane] {
        let generator = AVAssetImageGenerator(asset: AVURLAsset(url: url))
        generator.appliesPreferredTrackTransform = false
        let scale = min(1, Double(maxWidth) / max(frameSize.width, 1))
        generator.maximumSize = CGSize(width: (frameSize.width * scale).rounded(), height: (frameSize.height * scale).rounded())
        generator.requestedTimeToleranceBefore = CMTime(seconds: 1, preferredTimescale: 600)
        generator.requestedTimeToleranceAfter = CMTime(seconds: 1, preferredTimescale: 600)
        // Skip the ends: titles, fades and black.
        let times = (0 ..< count).map { i in
            CMTime(seconds: duration * (0.1 + 0.8 * Double(i) / Double(max(count - 1, 1))), preferredTimescale: 600)
        }
        var planes: [LumaPlane] = []
        for await result in generator.images(for: times) {
            guard let image = try? result.image, let plane = Self.luma(image) else { continue }
            planes.append(plane)
        }
        return planes
    }

    static func luma(_ image: CGImage) -> LumaPlane? {
        let width = image.width, height = image.height
        var pixels = [UInt8](repeating: 0, count: width * height)
        let drawn = pixels.withUnsafeMutableBytes { raw -> Bool in
            guard let context = CGContext(data: raw.baseAddress, width: width, height: height, bitsPerComponent: 8,
                                          bytesPerRow: width, space: CGColorSpaceCreateDeviceGray(),
                                          bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return false }
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        return drawn ? LumaPlane(width: width, height: height, pixels: pixels) : nil
    }

    static func bitDepth(_ description: CMFormatDescription?, _ extensions: [String: Any]) -> Int {
        if let bits = extensions[kCMFormatDescriptionExtension_BitsPerComponent as String] as? Int { return bits }
        // HEVC Main 10 and ProRes don't always say; their pixel formats do.
        if let description, [kCMVideoCodecType_AppleProRes422, kCMVideoCodecType_AppleProRes422HQ,
                             kCMVideoCodecType_AppleProRes4444].contains(CMFormatDescriptionGetMediaSubType(description)) {
            return 10
        }
        return 8
    }
}

/// What to make.
public struct SpatialVideoJob: Sendable {
    public var source: SpatialVideoSource
    public var output: URL
    public var layout: PackedStereoLayout
    public var camera: SpatialCameraMetadata
    public var quality: SpatialQuality
    /// Overrides the computed rate.
    public var bitsPerSecond: Int?
    public var includeAudio: Bool

    public init(source: SpatialVideoSource, output: URL, layout: PackedStereoLayout, camera: SpatialCameraMetadata,
                quality: SpatialQuality = .standard, bitsPerSecond: Int? = nil, includeAudio: Bool = true) {
        self.source = source
        self.output = output
        self.layout = layout
        self.camera = camera
        self.quality = quality
        self.bitsPerSecond = bitsPerSecond
        self.includeAudio = includeAudio
    }

    /// One eye's output size, even in both directions.
    public var eyeSize: (width: Int, height: Int) {
        let eye = layout.eyeSize(packed: source.frameSize)
        return (Int(eye.width) & ~1, Int(eye.height) & ~1)
    }

    public var resolvedBitsPerSecond: Int {
        bitsPerSecond ?? SpatialBitrate.bitsPerSecond(eyeWidth: eyeSize.width, eyeHeight: eyeSize.height,
                                                      frameRate: source.frameRate, quality: quality,
                                                      sourceBitsPerSecond: source.videoBitsPerSecond)
    }
}

public struct SpatialVideoResult: Sendable {
    public let frames: Int
    public let audioTracksCopied: Int
    public let audioTracksSkipped: Int
    public let bitsPerSecond: Int
    public let tenBit: Bool
}

public enum SpatialVideoMaker {
    /// Encodes `job`, reporting progress 0...1 as it goes.
    public static func make(_ job: SpatialVideoJob, progress: @Sendable (Double) -> Void = { _ in }) async throws -> SpatialVideoResult {
        #if targetEnvironment(simulator)
        throw SpatialVideoError.writer("MV-HEVC encoding needs a real device")
        #else
        let problems = job.camera.problems
        guard problems.isEmpty else { throw SpatialVideoError.invalidMetadata(problems) }
        guard !job.source.isMultiviewStereo else { throw SpatialVideoError.alreadySpatial }

        let asset = AVURLAsset(url: job.source.url)
        guard let videoTrack = try await asset.loadTracks(withMediaType: .video).first else { throw SpatialVideoError.noVideoTrack }
        let audioTracks = job.includeAudio ? try await asset.loadTracks(withMediaType: .audio) : []
        let audioFormats = try await audioTracks.asyncMap { try await $0.load(.formatDescriptions).first }

        let partial = job.output.deletingPathExtension().appendingPathExtension("partial.mov")
        try? FileManager.default.removeItem(at: partial)

        let tenBit = job.source.needsTenBit
        let pixelFormat = tenBit ? kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange : kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
        let (eyeWidth, eyeHeight) = job.eyeSize
        let bitsPerSecond = job.resolvedBitsPerSecond

        let reader: AVAssetReader
        let writer: AVAssetWriter
        do {
            reader = try AVAssetReader(asset: asset)
            writer = try AVAssetWriter(outputURL: partial, fileType: .mov)
        } catch {
            throw SpatialVideoError.writer(error.localizedDescription)
        }
        let videoOutput = AVAssetReaderTrackOutput(track: videoTrack, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: pixelFormat,
            kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
        ])
        videoOutput.alwaysCopiesSampleData = false
        reader.add(videoOutput)

        let videoInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings(job, width: eyeWidth, height: eyeHeight,
                                                                                            bitsPerSecond: bitsPerSecond, tenBit: tenBit))
        videoInput.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputTaggedPixelBufferGroupAdaptor(assetWriterInput: videoInput, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: pixelFormat,
            kCVPixelBufferWidthKey as String: eyeWidth,
            kCVPixelBufferHeightKey as String: eyeHeight,
            kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
        ])
        guard writer.canAdd(videoInput) else { throw SpatialVideoError.writer("the encoder refused the MV-HEVC settings") }
        writer.add(videoInput)

        // Audio is copied as it is; a codec QuickTime can't hold is left out.
        var audioPairs: [(AVAssetReaderTrackOutput, AVAssetWriterInput)] = []
        var skipped = 0
        for (track, format) in zip(audioTracks, audioFormats) {
            let output = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
            let input = AVAssetWriterInput(mediaType: .audio, outputSettings: nil, sourceFormatHint: format)
            input.expectsMediaDataInRealTime = false
            guard reader.canAdd(output), writer.canAdd(input) else { skipped += 1; continue }
            reader.add(output)
            writer.add(input)
            audioPairs.append((output, input))
        }

        guard reader.startReading() else { throw SpatialVideoError.unreadable(reader.error?.localizedDescription ?? "unknown") }
        guard writer.startWriting() else { throw SpatialVideoError.writer(writer.error?.localizedDescription ?? "unknown") }
        writer.startSession(atSourceTime: try await videoTrack.load(.timeRange).start)

        var transfer: VTPixelTransferSession?
        VTPixelTransferSessionCreate(allocator: kCFAllocatorDefault, pixelTransferSessionOut: &transfer)
        guard let transfer else { throw SpatialVideoError.writer("can't create a pixel transfer session") }
        defer { VTPixelTransferSessionInvalidate(transfer) }
        VTSessionSetProperty(transfer, key: kVTPixelTransferPropertyKey_ScalingMode, value: kVTScalingMode_CropSourceToCleanAperture)

        var frames = 0
        var videoDone = false
        var audioDone = audioPairs.map { _ in false }
        let duration = max(job.source.duration, 0.001)
        do {
            while !videoDone || audioDone.contains(false) {
                var moved = false
                if !videoDone, videoInput.isReadyForMoreMediaData {
                    moved = true
                    if let sample = videoOutput.copyNextSampleBuffer() {
                        if let picture = CMSampleBufferGetImageBuffer(sample) {
                            let time = CMSampleBufferGetPresentationTimeStamp(sample)
                            let eyes = try cropEyes(picture, layout: job.layout, transfer: transfer, adaptor: adaptor)
                            guard adaptor.appendTaggedBuffers(eyes, withPresentationTime: time) else {
                                throw SpatialVideoError.writer(writer.error?.localizedDescription ?? "frame \(frames) was refused")
                            }
                            frames += 1
                            if frames % 24 == 0 { progress(min(1, time.seconds / duration)) }
                        }
                    } else {
                        videoInput.markAsFinished()
                        videoDone = true
                    }
                }
                for (i, (output, input)) in audioPairs.enumerated() where !audioDone[i] && input.isReadyForMoreMediaData {
                    moved = true
                    if let sample = output.copyNextSampleBuffer() {
                        guard input.append(sample) else {
                            throw SpatialVideoError.writer(writer.error?.localizedDescription ?? "an audio sample was refused")
                        }
                    } else {
                        input.markAsFinished()
                        audioDone[i] = true
                    }
                }
                if reader.status == .failed { throw SpatialVideoError.unreadable(reader.error?.localizedDescription ?? "unknown") }
                if writer.status == .failed { throw SpatialVideoError.writer(writer.error?.localizedDescription ?? "unknown") }
                if !moved { try await Task.sleep(for: .milliseconds(2)) }
            }
            await writer.finishWriting()
            guard writer.status == .completed else { throw SpatialVideoError.writer(writer.error?.localizedDescription ?? "unknown") }
        } catch {
            reader.cancelReading()
            writer.cancelWriting()
            try? FileManager.default.removeItem(at: partial)
            throw error
        }
        try? FileManager.default.removeItem(at: job.output)
        try FileManager.default.moveItem(at: partial, to: job.output)
        progress(1)
        return SpatialVideoResult(frames: frames, audioTracksCopied: audioPairs.count, audioTracksSkipped: skipped,
                                  bitsPerSecond: bitsPerSecond, tenBit: tenBit)
        #endif
    }

    static func videoSettings(_ job: SpatialVideoJob, width: Int, height: Int, bitsPerSecond: Int, tenBit: Bool) -> [String: Any] {
        let camera = job.camera
        var compression: [String: Any] = [
            AVVideoAverageBitRateKey: bitsPerSecond,
            AVVideoExpectedSourceFrameRateKey: job.source.frameRate > 0 ? job.source.frameRate : 30,
            AVVideoMaxKeyFrameIntervalDurationKey: 2,
            AVVideoAllowFrameReorderingKey: true,
            kVTCompressionPropertyKey_MVHEVCVideoLayerIDs as String: [0, 1],
            kVTCompressionPropertyKey_MVHEVCViewIDs as String: [0, 1],
            kVTCompressionPropertyKey_MVHEVCLeftAndRightViewIDs as String: [0, 1],
            kVTCompressionPropertyKey_HasLeftStereoEyeView as String: true,
            kVTCompressionPropertyKey_HasRightStereoEyeView as String: true,
            kVTCompressionPropertyKey_HeroEye as String: kCMFormatDescriptionHeroEye_Left as String,
            kVTCompressionPropertyKey_StereoCameraBaseline as String: camera.baselineMicrometers,
            kVTCompressionPropertyKey_HorizontalDisparityAdjustment as String: camera.disparityAdjustmentNormalized,
        ]
        if #available(macOS 14.4, iOS 17.4, visionOS 1.1, *) {
            compression[kVTCompressionPropertyKey_HorizontalFieldOfView as String] = camera.fieldOfViewThousandthsOfDegree
        }
        if #available(macOS 15, iOS 18, visionOS 2, *) {
            compression[kVTCompressionPropertyKey_ProjectionKind as String] = kCMFormatDescriptionProjectionKind_Rectilinear as String
        }
        if tenBit {
            compression[AVVideoProfileLevelKey] = kVTProfileLevel_HEVC_Main10_AutoLevel as String
            compression[kVTCompressionPropertyKey_HDRMetadataInsertionMode as String] = kVTHDRMetadataInsertionMode_Auto as String
        }
        let source = job.source
        return [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: compression,
            AVVideoColorPropertiesKey: [
                AVVideoColorPrimariesKey: source.colorPrimaries ?? AVVideoColorPrimaries_ITU_R_709_2,
                AVVideoTransferFunctionKey: source.transferFunction ?? AVVideoTransferFunction_ITU_R_709_2,
                AVVideoYCbCrMatrixKey: source.yCbCrMatrix ?? AVVideoYCbCrMatrix_ITU_R_709_2,
            ],
        ]
    }

    /// Both eyes of a packed picture as tagged buffers, left first. Each eye
    /// is the source cropped through its clean aperture and scaled by the
    /// transfer session, so a half format comes out at full display size.
    static func cropEyes(_ picture: CVPixelBuffer, layout: PackedStereoLayout, transfer: VTPixelTransferSession,
                         adaptor: AVAssetWriterInputTaggedPixelBufferGroupAdaptor) throws -> [CMTaggedBuffer] {
        guard let pool = adaptor.pixelBufferPool else { throw SpatialVideoError.writer("the writer has no buffer pool") }
        let width = Double(CVPixelBufferGetWidth(picture)), height = Double(CVPixelBufferGetHeight(picture))
        let saved = CVBufferCopyAttachment(picture, kCVImageBufferCleanApertureKey, nil)
        defer {
            if let saved { CVBufferSetAttachment(picture, kCVImageBufferCleanApertureKey, saved, .shouldPropagate) }
            else { CVBufferRemoveAttachment(picture, kCVImageBufferCleanApertureKey) }
        }
        var eyes: [CMTaggedBuffer] = []
        for (eye, layerID) in [(PackedStereoLayout.Eye.left, 0), (.right, 1)] {
            let region = layout.eyeRegion(eye)
            let cropWidth = width * Double(region.scale.x), cropHeight = height * Double(region.scale.y)
            // Offsets are from the picture's centre to the crop's centre.
            let centerX = width * Double(region.offset.x) + cropWidth / 2 - width / 2
            let centerY = height * Double(region.offset.y) + cropHeight / 2 - height / 2
            let aperture: [CFString: Any] = [
                kCVImageBufferCleanApertureWidthKey: cropWidth,
                kCVImageBufferCleanApertureHeightKey: cropHeight,
                kCVImageBufferCleanApertureHorizontalOffsetKey: centerX,
                kCVImageBufferCleanApertureVerticalOffsetKey: centerY,
            ]
            CVBufferSetAttachment(picture, kCVImageBufferCleanApertureKey, aperture as CFDictionary, .shouldPropagate)
            var out: CVPixelBuffer?
            CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &out)
            guard let out else { throw SpatialVideoError.writer("out of pixel buffers") }
            let status = VTPixelTransferSessionTransferImage(transfer, from: picture, to: out)
            guard status == noErr else { throw SpatialVideoError.writer("cropping an eye failed (\(status))") }
            CVBufferRemoveAttachment(out, kCVImageBufferCleanApertureKey)
            eyes.append(CMTaggedBuffer(tags: [.videoLayerID(Int64(layerID)), .stereoView(eye == .left ? .leftEye : .rightEye)],
                                       pixelBuffer: out))
        }
        return eyes
    }
}

/// What a file carries, for `vergence inspect`.
public struct SpatialVideoReport: Sendable {
    public struct Field: Sendable {
        public let name: String
        public let value: String
    }

    public let source: SpatialVideoSource
    public let fields: [Field]
    /// Why a player wouldn't treat it as spatial video; empty when it would.
    public let missing: [String]
    /// Plays as spatial video in Photos and Files.
    public var isCompleteSpatialVideo: Bool { source.isMultiviewStereo && missing.isEmpty }
}

public enum SpatialVideoInspector {
    public static func inspect(_ url: URL) async throws -> SpatialVideoReport {
        let source = try await SpatialVideoSource.open(url)
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .video).first,
              let description = try await track.load(.formatDescriptions).first else { throw SpatialVideoError.noVideoTrack }
        let ext = CMFormatDescriptionGetExtensions(description) as? [String: Any] ?? [:]
        func value(_ key: CFString) -> Any? { ext[key as String] }

        var fields: [SpatialVideoReport.Field] = []
        func add(_ name: String, _ value: String) { fields.append(.init(name: name, value: value)) }
        add("video", "\(source.codec) \(Int(source.frameSize.width))×\(Int(source.frameSize.height)), "
            + String(format: "%.3f fps, %d-bit", source.frameRate, source.bitsPerComponent)
            + (source.videoBitsPerSecond.map { String(format: ", %.1f Mbps", $0 / 1_000_000) } ?? ""))
        add("colour", [source.colorPrimaries, source.transferFunction, source.yCbCrMatrix].map { $0 ?? "untagged" }.joined(separator: " / ")
            + (source.isHDR ? " (HDR)" : ""))
        add("duration", String(format: "%.1f s", source.duration))
        add("audio", source.audioTracks.isEmpty ? "none" : source.audioTracks.joined(separator: "; "))
        add("stereo", source.isMultiviewStereo ? "MV-HEVC, two views" : "one view")

        var missing: [String] = []
        if source.isMultiviewStereo {
            let left = value(kCMFormatDescriptionExtension_HasLeftStereoEyeView) as? Bool ?? false
            let right = value(kCMFormatDescriptionExtension_HasRightStereoEyeView) as? Bool ?? false
            add("eyes", [left ? "left" : nil, right ? "right" : nil].compactMap { $0 }.joined(separator: " + ").ifEmpty("not labelled"))
            if !(left && right) { missing.append("the views aren't labelled as left and right eye") }
            if let hero = value(kCMFormatDescriptionExtension_HeroEye) as? String { add("hero eye", hero) }

            if let baseline = (value(kCMFormatDescriptionExtension_StereoCameraBaseline) as? NSNumber)?.doubleValue {
                add("camera baseline", String(format: "%.2f mm", baseline / 1000))
            } else {
                missing.append("no camera baseline")
            }
            if #available(macOS 14.4, iOS 17.4, visionOS 1.1, *),
               let fov = (value(kCMFormatDescriptionExtension_HorizontalFieldOfView) as? NSNumber)?.doubleValue {
                add("field of view", String(format: "%.2f°", fov / 1000))
            } else {
                missing.append("no horizontal field of view (Photos and Files need it to show the video as spatial)")
            }
            if let adjust = (value(kCMFormatDescriptionExtension_HorizontalDisparityAdjustment) as? NSNumber)?.doubleValue {
                add("disparity adjustment", String(format: "%.4f", adjust / 10_000))
            }
            if #available(macOS 15, iOS 18, visionOS 2, *) {
                if let projection = value(kCMFormatDescriptionExtension_ProjectionKind) as? String {
                    add("projection", projection)
                } else {
                    missing.append("no projection")
                }
            }
        } else if let declared = source.declaredLayout {
            add("packing", "\(declared.arrangement == .sideBySide ? "side by side" : "over-under") (from the file's metadata)")
        }
        return SpatialVideoReport(source: source, fields: fields, missing: missing)
    }
}

private func fourCharacterCode(_ code: FourCharCode) -> String {
    let bytes = [24, 16, 8, 0].map { UInt8((code >> $0) & 0xFF) }
    guard bytes.allSatisfy({ $0 >= 32 && $0 < 127 }) else { return String(code) }
    return String(decoding: bytes, as: UTF8.self).trimmingCharacters(in: .whitespaces)
}

private extension String {
    func ifEmpty(_ fallback: String) -> String { isEmpty ? fallback : self }
}

private extension Array {
    func asyncMap<T>(_ transform: (Element) async throws -> T) async rethrows -> [T] {
        var out: [T] = []
        out.reserveCapacity(count)
        for element in self { out.append(try await transform(element)) }
        return out
    }
}
#endif
