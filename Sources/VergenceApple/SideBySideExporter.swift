/*
 Vergence - spatial video to side by side, for everything that isn't Apple

 A Quest, an Android phone or YouTube can't decode MV-HEVC's second view;
 their players crop eyes out of one ordinary picture. `SideBySideExporter`
 decodes both views of a spatial video (a Vision Pro or iPhone recording,
 or `make`'s output) and writes one HEVC picture with the left eye on the
 left, as MP4:
 - full resolution per eye by default (a Vision Pro's 2200×2200 eyes make a
   4400×2200 picture, which Quest 3 decodes), or squeezed to half width,
   or with each eye capped at a size for older decoders;
 - colour converted to BT.709 (a Vision Pro records Display P3, which most
   Android players would show oversaturated); HDR stays as it is;
 - the stereo audio copied, and an ambisonic track (a Vision Pro's APAC)
   decoded and re-encoded as 4-channel AAC, the form YouTube's spatial
   audio takes;
 - Google's `st3d` (left-right) and `SA3D` boxes added afterwards
   (`SphericalMetadata`), which non-Apple players read.
 */

#if canImport(AVFoundation) && !os(tvOS)

import AVFoundation
import CoreMedia
import CoreVideo
import VideoToolbox
import VergenceCore

public struct SideBySideJob: Sendable {
    public var source: SpatialVideoSource
    public var output: URL
    /// Squeeze each eye to half width, so the picture is one eye's size.
    public var halfWidth: Bool
    /// Cap each eye's longer side (keeping its shape); nil keeps it.
    public var maxEyeSide: Int?
    public var quality: SpatialQuality
    public var bitsPerSecond: Int?
    public var includeAmbisonics: Bool

    public init(source: SpatialVideoSource, output: URL, halfWidth: Bool = false, maxEyeSide: Int? = nil,
                quality: SpatialQuality = .standard, bitsPerSecond: Int? = nil, includeAmbisonics: Bool = true) {
        self.source = source
        self.output = output
        self.halfWidth = halfWidth
        self.maxEyeSide = maxEyeSide
        self.quality = quality
        self.bitsPerSecond = bitsPerSecond
        self.includeAmbisonics = includeAmbisonics
    }

    /// One eye as stored in the picture (squeezed when `halfWidth`), even.
    public var storedEye: (width: Int, height: Int) {
        var w = Double(source.frameSize.width), h = Double(source.frameSize.height)
        if let cap = maxEyeSide, max(w, h) > Double(cap) {
            let scale = Double(cap) / max(w, h)
            w *= scale
            h *= scale
        }
        if halfWidth { w /= 2 }
        return (Int(w.rounded()) & ~1, Int(h.rounded()) & ~1)
    }

    public var pictureSize: (width: Int, height: Int) { (storedEye.width * 2, storedEye.height) }

    public var resolvedBitsPerSecond: Int {
        // One ordinary picture of both eyes: the second eye isn't predicted
        // from the first, so it costs about as much as the first.
        bitsPerSecond ?? SpatialBitrate.bitsPerSecond(
            eyeWidth: pictureSize.width, eyeHeight: pictureSize.height, frameRate: source.frameRate,
            quality: quality, sourceBitsPerSecond: source.videoBitsPerSecond.map { $0 * 1.6 }, secondEye: false)
    }
}

public struct SideBySideResult: Sendable {
    public let frames: Int
    public let bitsPerSecond: Int
    public let audioTracksCopied: Int
    public let ambisonicOrder: Int?
}

public enum SideBySideExporter {
    public static func export(_ job: SideBySideJob, progress: @Sendable (Double) -> Void = { _ in }) async throws -> SideBySideResult {
        guard job.source.isMultiviewStereo else { throw SpatialVideoError.unreadable("not MV-HEVC stereo; vergence make turns side-by-side files into spatial video, not back") }
        let asset = AVURLAsset(url: job.source.url)
        guard let videoTrack = try await asset.loadTracks(withMediaType: .video).first else { throw SpatialVideoError.noVideoTrack }
        let tracks = try await AudioTrackReader.tracks(job.source.url)
        let audioTracks = try await asset.loadTracks(withMediaType: .audio)

        let partial = job.output.deletingPathExtension().appendingPathExtension("partial.mp4")
        try? FileManager.default.removeItem(at: partial)
        let hdr = job.source.isHDR
        let tenBit = job.source.needsTenBit
        let pixelFormat = tenBit ? kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange : kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
        let eye = job.storedEye, picture = job.pictureSize
        let bitsPerSecond = job.resolvedBitsPerSecond

        let reader = try AVAssetReader(asset: asset)
        let writer = try AVAssetWriter(outputURL: partial, fileType: .mp4)
        // moov at the end, so the boxes can be added without moving media.
        writer.shouldOptimizeForNetworkUse = false

        let videoOutput = AVAssetReaderTrackOutput(track: videoTrack, outputSettings: [
            AVVideoDecompressionPropertiesKey: [kVTDecompressionPropertyKey_RequestedMVHEVCVideoLayerIDs: [0, 1]],
            kCVPixelBufferPixelFormatTypeKey as String: pixelFormat,
            kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
        ])
        videoOutput.alwaysCopiesSampleData = false
        reader.add(videoOutput)

        let color: [String: Any] = hdr ? [
            AVVideoColorPrimariesKey: job.source.colorPrimaries ?? AVVideoColorPrimaries_ITU_R_2020,
            AVVideoTransferFunctionKey: job.source.transferFunction ?? AVVideoTransferFunction_SMPTE_ST_2084_PQ,
            AVVideoYCbCrMatrixKey: job.source.yCbCrMatrix ?? AVVideoYCbCrMatrix_ITU_R_2020,
        ] : [
            AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
            AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2,
            AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2,
        ]
        var compression: [String: Any] = [
            AVVideoAverageBitRateKey: bitsPerSecond,
            AVVideoExpectedSourceFrameRateKey: job.source.frameRate > 0 ? job.source.frameRate : 30,
            AVVideoMaxKeyFrameIntervalDurationKey: 1,
            AVVideoAllowFrameReorderingKey: true,
        ]
        if tenBit { compression[AVVideoProfileLevelKey] = kVTProfileLevel_HEVC_Main10_AutoLevel as String }
        let videoInput = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.hevc, AVVideoWidthKey: picture.width, AVVideoHeightKey: picture.height,
            AVVideoCompressionPropertiesKey: compression, AVVideoColorPropertiesKey: color,
        ])
        videoInput.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: videoInput, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: pixelFormat,
            kCVPixelBufferWidthKey as String: picture.width, kCVPixelBufferHeightKey as String: picture.height,
            kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
        ])
        guard writer.canAdd(videoInput) else { throw SpatialVideoError.writer("the encoder refused \(picture.width)×\(picture.height) HEVC") }
        writer.add(videoInput)

        // Audio: plain tracks copied, the ambisonic one re-encoded as AAC.
        var pairs: [(AVAssetReaderTrackOutput, AVAssetWriterInput)] = []
        var ambisonicOrder: Int?
        var copied = 0
        for (track, info) in zip(audioTracks, tracks) {
            if let order = info.ambisonicOrder {
                guard job.includeAmbisonics, ambisonicOrder == nil, order == 1 else { continue }
                let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
                    AVFormatIDKey: kAudioFormatLinearPCM, AVLinearPCMBitDepthKey: 32, AVLinearPCMIsFloatKey: true,
                    AVLinearPCMIsNonInterleaved: false, AVLinearPCMIsBigEndianKey: false,
                ])
                // AAC knows no ambisonic layout; four discrete channels in
                // ACN order, which SA3D then describes.
                var layout = AudioChannelLayout()
                layout.mChannelLayoutTag = kAudioChannelLayoutTag_Quadraphonic
                let input = AVAssetWriterInput(mediaType: .audio, outputSettings: [
                    AVFormatIDKey: kAudioFormatMPEG4AAC, AVSampleRateKey: info.sampleRate, AVNumberOfChannelsKey: 4,
                    AVEncoderBitRateKey: 256_000,
                    AVChannelLayoutKey: Data(bytes: &layout, count: MemoryLayout<AudioChannelLayout>.size),
                ])
                input.expectsMediaDataInRealTime = false
                guard reader.canAdd(output), writer.canAdd(input) else { continue }
                reader.add(output)
                writer.add(input)
                pairs.append((output, input))
                ambisonicOrder = order
            } else {
                guard let format = try await track.load(.formatDescriptions).first,
                      ["aac", "ac3", "eac3", "alac", "mp3"].contains(info.codec) else { continue }
                let output = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
                let input = AVAssetWriterInput(mediaType: .audio, outputSettings: nil, sourceFormatHint: format)
                input.expectsMediaDataInRealTime = false
                guard reader.canAdd(output), writer.canAdd(input) else { continue }
                reader.add(output)
                writer.add(input)
                pairs.append((output, input))
                copied += 1
            }
        }

        guard reader.startReading() else { throw SpatialVideoError.unreadable(reader.error?.localizedDescription ?? "unknown") }
        guard writer.startWriting() else { throw SpatialVideoError.writer(writer.error?.localizedDescription ?? "unknown") }
        writer.startSession(atSourceTime: try await videoTrack.load(.timeRange).start)

        var transfer: VTPixelTransferSession?
        VTPixelTransferSessionCreate(allocator: kCFAllocatorDefault, pixelTransferSessionOut: &transfer)
        guard let transfer else { throw SpatialVideoError.writer("can't create a pixel transfer session") }
        defer { VTPixelTransferSessionInvalidate(transfer) }
        if !hdr {
            VTSessionSetProperty(transfer, key: kVTPixelTransferPropertyKey_DestinationColorPrimaries, value: kCVImageBufferColorPrimaries_ITU_R_709_2)
            VTSessionSetProperty(transfer, key: kVTPixelTransferPropertyKey_DestinationTransferFunction, value: kCVImageBufferTransferFunction_ITU_R_709_2)
            VTSessionSetProperty(transfer, key: kVTPixelTransferPropertyKey_DestinationYCbCrMatrix, value: kCVImageBufferYCbCrMatrix_ITU_R_709_2)
        }
        var eyePool: CVPixelBufferPool?
        CVPixelBufferPoolCreate(nil, nil, [
            kCVPixelBufferPixelFormatTypeKey as String: pixelFormat,
            kCVPixelBufferWidthKey as String: eye.width, kCVPixelBufferHeightKey as String: eye.height,
            kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
        ] as CFDictionary, &eyePool)
        guard let eyePool else { throw SpatialVideoError.writer("can't make eye buffers") }

        var frames = 0
        var videoDone = false
        var audioDone = pairs.map { _ in false }
        let duration = max(job.source.duration, 0.001)
        do {
            while !videoDone || audioDone.contains(false) {
                var moved = false
                if !videoDone, videoInput.isReadyForMoreMediaData {
                    moved = true
                    if let sample = videoOutput.copyNextSampleBuffer() {
                        if let views = sample.taggedBuffers {
                            let time = CMSampleBufferGetPresentationTimeStamp(sample)
                            let packed = try pack(views, transfer: transfer, eyePool: eyePool, adaptor: adaptor)
                            guard adaptor.append(packed, withPresentationTime: time) else {
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
                for (i, (output, input)) in pairs.enumerated() where !audioDone[i] && input.isReadyForMoreMediaData {
                    moved = true
                    if let sample = output.copyNextSampleBuffer() {
                        guard input.append(sample) else { throw SpatialVideoError.writer(writer.error?.localizedDescription ?? "audio refused") }
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

            // The writer numbers tracks in the order they were added.
            var ambisonicTracks: [UInt32: Int] = [:]
            if ambisonicOrder != nil {
                let written = try await AVURLAsset(url: partial).loadTracks(withMediaType: .audio)
                for track in written {
                    if let format = try await track.load(.formatDescriptions).first,
                       CMAudioFormatDescriptionGetStreamBasicDescription(format)?.pointee.mChannelsPerFrame == 4 {
                        ambisonicTracks[UInt32(track.trackID)] = 4
                    }
                }
            }
            try SphericalMetadata.write(to: partial, stereoMode: .leftRight, ambisonicTracks: ambisonicTracks)
        } catch {
            reader.cancelReading()
            writer.cancelWriting()
            try? FileManager.default.removeItem(at: partial)
            throw error
        }
        try? FileManager.default.removeItem(at: job.output)
        try FileManager.default.moveItem(at: partial, to: job.output)
        progress(1)
        return SideBySideResult(frames: frames, bitsPerSecond: bitsPerSecond, audioTracksCopied: copied, ambisonicOrder: ambisonicOrder)
    }

    /// Both views scaled into one picture, left eye on the left.
    static func pack(_ views: [CMTaggedBuffer], transfer: VTPixelTransferSession, eyePool: CVPixelBufferPool,
                     adaptor: AVAssetWriterInputPixelBufferAdaptor) throws -> CVPixelBuffer {
        func view(_ which: CMStereoViewComponents) -> CVPixelBuffer? {
            for tagged in views {
                guard case .pixelBuffer(let buffer) = tagged.buffer,
                      tagged.tags.first(matchingCategory: .stereoView)?.value(onlyIfMatching: .stereoView) == which else { continue }
                return buffer
            }
            return nil
        }
        guard let left = view(.leftEye), let right = view(.rightEye) else { throw SpatialVideoError.unreadable("a frame is missing an eye") }
        guard let pool = adaptor.pixelBufferPool else { throw SpatialVideoError.writer("the writer has no buffer pool") }
        var out: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &out)
        guard let out else { throw SpatialVideoError.writer("out of pixel buffers") }

        CVPixelBufferLockBaseAddress(out, [])
        defer { CVPixelBufferUnlockBaseAddress(out, []) }
        for (index, source) in [left, right].enumerated() {
            var scaled: CVPixelBuffer?
            CVPixelBufferPoolCreatePixelBuffer(nil, eyePool, &scaled)
            guard let scaled else { throw SpatialVideoError.writer("out of eye buffers") }
            let status = VTPixelTransferSessionTransferImage(transfer, from: source, to: scaled)
            guard status == noErr else { throw SpatialVideoError.writer("scaling an eye failed (\(status))") }
            CVPixelBufferLockBaseAddress(scaled, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(scaled, .readOnly) }
            // Both planes of 4:2:0 biplanar are as many bytes wide as the
            // eye is pixels (chroma pairs at half resolution), times the
            // component size.
            for plane in 0 ..< 2 {
                guard let from = CVPixelBufferGetBaseAddressOfPlane(scaled, plane),
                      let to = CVPixelBufferGetBaseAddressOfPlane(out, plane) else { throw SpatialVideoError.writer("no plane \(plane)") }
                let fromStride = CVPixelBufferGetBytesPerRowOfPlane(scaled, plane)
                let toStride = CVPixelBufferGetBytesPerRowOfPlane(out, plane)
                let rows = CVPixelBufferGetHeightOfPlane(scaled, plane)
                let rowBytes = CVPixelBufferGetWidthOfPlane(scaled, plane) * (plane == 0 ? 1 : 2) * bytesPerComponent(scaled)
                for row in 0 ..< rows {
                    memcpy(to + row * toStride + index * rowBytes, from + row * fromStride, rowBytes)
                }
            }
        }
        if let attachments = CVBufferCopyAttachments(left, .shouldPropagate) as? [CFString: Any], hdrPassThrough(attachments) {
            CVBufferSetAttachments(out, attachments as CFDictionary, .shouldPropagate)
        }
        return out
    }

    static func bytesPerComponent(_ buffer: CVPixelBuffer) -> Int {
        CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange ? 2 : 1
    }

    /// HDR keeps the source's tags (and mastering metadata) on the picture.
    static func hdrPassThrough(_ attachments: [CFString: Any]) -> Bool {
        let transfer = attachments[kCVImageBufferTransferFunctionKey] as? String
        return transfer == (kCVImageBufferTransferFunction_SMPTE_ST_2084_PQ as String)
            || transfer == (kCVImageBufferTransferFunction_ITU_R_2100_HLG as String)
    }
}
#endif
