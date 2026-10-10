#if os(macOS)
import AVFoundation
import CoreMedia
import CoreVideo
import Testing
import VideoToolbox
@testable import VergenceApple
@testable import VergenceCore

/// End to end: a synthetic side-by-side clip in, a spatial video out that
/// `inspect` calls complete, with each eye holding its own half.
@Suite struct SpatialVideoMakerTests {
    static let eye = (width: 256, height: 144)
    static let disparity = 8
    static let frames = 12

    /// A noise scene seen by two eyes `disparity` pixels apart, side by side,
    /// with the left eye's top-left pixel marked bright and the right's dark.
    /// `audio` adds a stereo AAC track (48 kHz) as long as the video; `hdr`
    /// encodes 10-bit HEVC tagged BT.2020 PQ instead of 8-bit H.264.
    static func writeSideBySideClip(to url: URL, frames: Int = frames, audio: Bool = false, hdr: Bool = false) async throws {
        let width = eye.width * 2, height = eye.height
        let writer = try AVAssetWriter(outputURL: url, fileType: hdr ? .mov : .mp4)
        var settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: width, AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: [AVVideoAverageBitRateKey: 8_000_000],
        ]
        if hdr {
            settings[AVVideoCodecKey] = AVVideoCodecType.hevc
            settings[AVVideoCompressionPropertiesKey] = [
                AVVideoAverageBitRateKey: 8_000_000,
                AVVideoProfileLevelKey: kVTProfileLevel_HEVC_Main10_AutoLevel as String,
            ]
            settings[AVVideoColorPropertiesKey] = [
                AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_2020,
                AVVideoTransferFunctionKey: AVVideoTransferFunction_SMPTE_ST_2084_PQ,
                AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_2020,
            ]
        }
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        input.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width, kCVPixelBufferHeightKey as String: height,
        ])
        writer.add(input)
        var audioInput: AVAssetWriterInput?
        if audio {
            let made = AVAssetWriterInput(mediaType: .audio, outputSettings: [
                AVFormatIDKey: kAudioFormatMPEG4AAC, AVSampleRateKey: audioRate, AVNumberOfChannelsKey: 2,
                AVEncoderBitRateKey: 128_000,
            ])
            made.expectsMediaDataInRealTime = false
            writer.add(made)
            audioInput = made
        }
        #expect(writer.startWriting())
        writer.startSession(atSourceTime: .zero)
        let sceneWidth = eye.width + disparity
        func picture(_ frame: Int) throws -> CVPixelBuffer {
            var state = UInt64(frame + 1) &* 0x9E37_79B9_7F4A_7C15 | 1
            let scene = (0 ..< sceneWidth * height).map { _ -> UInt8 in
                state ^= state << 13; state ^= state >> 7; state ^= state << 17
                return UInt8(truncatingIfNeeded: state >> 24)
            }
            let pool = try #require(adaptor.pixelBufferPool)
            var buffer: CVPixelBuffer?
            CVPixelBufferPoolCreatePixelBuffer(nil, pool, &buffer)
            let picture = try #require(buffer)
            CVPixelBufferLockBaseAddress(picture, [])
            let base = try #require(CVPixelBufferGetBaseAddress(picture)).assumingMemoryBound(to: UInt8.self)
            let stride = CVPixelBufferGetBytesPerRow(picture)
            for y in 0 ..< height {
                for x in 0 ..< width {
                    let isRight = x >= eye.width
                    let v = scene[y * sceneWidth + (isRight ? x - eye.width + disparity : x)]
                    let p = base + y * stride + x * 4
                    p[0] = v; p[1] = v; p[2] = v; p[3] = 255
                }
            }
            // A bright left eye band and a dark right eye band at the top.
            for y in 0 ..< 16 {
                for x in 0 ..< width {
                    let p = base + y * stride + x * 4
                    let v: UInt8 = x < eye.width ? 235 : 16
                    p[0] = v; p[1] = v; p[2] = v
                }
            }
            CVPixelBufferUnlockBaseAddress(picture, [])
            return picture
        }
        // As long as the video, in tenth-of-a-second chunks of a 440 Hz tone.
        let audioSamples = audio ? frames * audioRate / 24 : 0
        var frame = 0, written = 0
        while frame < frames || written < audioSamples {
            var moved = false
            if frame < frames, input.isReadyForMoreMediaData {
                #expect(adaptor.append(try picture(frame), withPresentationTime: CMTime(value: CMTimeValue(frame), timescale: 24)))
                frame += 1
                if frame == frames { input.markAsFinished() }
                moved = true
            }
            if written < audioSamples, let audioInput, audioInput.isReadyForMoreMediaData {
                let count = min(audioRate / 10, audioSamples - written)
                #expect(audioInput.append(try Self.tone(at: written, count: count)))
                written += count
                if written == audioSamples { audioInput.markAsFinished() }
                moved = true
            }
            if !moved { try await Task.sleep(for: .milliseconds(2)) }
        }
        await writer.finishWriting()
        #expect(writer.status == .completed, "\(String(describing: writer.error))")
    }

    static let audioRate = 48_000

    /// `count` frames of a stereo 16-bit 440 Hz tone starting at frame `start`.
    static func tone(at start: Int, count: Int) throws -> CMSampleBuffer {
        var description = AudioStreamBasicDescription(
            mSampleRate: Double(audioRate), mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: kAudioFormatFlagIsSignedInteger | kAudioFormatFlagIsPacked, mBytesPerPacket: 4, mFramesPerPacket: 1,
            mBytesPerFrame: 4, mChannelsPerFrame: 2, mBitsPerChannel: 16, mReserved: 0)
        var format: CMAudioFormatDescription?
        CMAudioFormatDescriptionCreate(allocator: nil, asbd: &description, layoutSize: 0, layout: nil, magicCookieSize: 0,
                                       magicCookie: nil, extensions: nil, formatDescriptionOut: &format)
        var samples = [Int16](repeating: 0, count: count * 2)
        for i in 0 ..< count {
            let v = Int16(8000 * sin(2 * Double.pi * 440 * Double(start + i) / Double(audioRate)))
            samples[2 * i] = v; samples[2 * i + 1] = v
        }
        var block: CMBlockBuffer?
        CMBlockBufferCreateWithMemoryBlock(allocator: nil, memoryBlock: nil, blockLength: count * 4, blockAllocator: nil,
                                           customBlockSource: nil, offsetToData: 0, dataLength: count * 4, flags: 0,
                                           blockBufferOut: &block)
        let data = try #require(block)
        samples.withUnsafeBytes { _ = CMBlockBufferReplaceDataBytes(with: $0.baseAddress!, blockBuffer: data, offsetIntoDestination: 0, dataLength: $0.count) }
        var sample: CMSampleBuffer?
        CMAudioSampleBufferCreateReadyWithPacketDescriptions(allocator: nil, dataBuffer: data, formatDescription: try #require(format),
                                                             sampleCount: count,
                                                             presentationTimeStamp: CMTime(value: CMTimeValue(start), timescale: CMTimeScale(audioRate)),
                                                             packetDescriptions: nil, sampleBufferOut: &sample)
        return try #require(sample)
    }

    @Test func makesCompleteSpatialVideoFromAnUntaggedSideBySideClip() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("vergence-make-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let input = directory.appendingPathComponent("clip.mp4")
        try await Self.writeSideBySideClip(to: input)

        let source = try await SpatialVideoSource.open(input)
        #expect(source.frameSize == CGSize(width: 512, height: 144))
        #expect(!source.isMultiviewStereo)
        let detection = StereoLayoutDetector.detect(declared: source.declaredLayout, fileName: "clip.mp4",
                                                    frameSize: source.frameSize, pictures: try await source.samplePictures(count: 4))
        let layout = try #require(detection.layout)
        #expect(layout == PackedStereoLayout(arrangement: .sideBySide, isHalfResolution: false))

        let output = directory.appendingPathComponent("clip.spatial.mov")
        let result = try await SpatialVideoMaker.make(SpatialVideoJob(source: source, output: output, layout: layout,
                                                                       camera: SpatialPreset.iphone.metadata))
        #expect(result.frames == Self.frames)
        #expect(!FileManager.default.fileExists(atPath: directory.appendingPathComponent("clip.spatial.partial.mov").path))

        let report = try await SpatialVideoInspector.inspect(output)
        #expect(report.isCompleteSpatialVideo, "missing: \(report.missing)")
        #expect(report.source.frameSize == CGSize(width: Self.eye.width, height: Self.eye.height))
        let fields = Dictionary(uniqueKeysWithValues: report.fields.map { ($0.name, $0.value) })
        #expect(fields["camera baseline"] == "19.24 mm")
        #expect(fields["field of view"] == "63.40°")
        #expect(fields["disparity adjustment"] == "0.0200")

        // Each eye holds its own half: the marker band is bright on the left
        // and dark on the right.
        let means = try await Self.markerMeans(output)
        #expect(means.left > 200 && means.right < 50, "left \(means.left), right \(means.right)")

        // And back to side by side for players without MV-HEVC.
        let sbs = directory.appendingPathComponent("clip.3D.FSBS.mp4")
        let exported = try await SideBySideExporter.export(SideBySideJob(source: try await SpatialVideoSource.open(output), output: sbs))
        #expect(exported.frames == Self.frames)
        let flat = try await SpatialVideoSource.open(sbs)
        #expect(!flat.isMultiviewStereo)
        #expect(flat.frameSize == CGSize(width: Self.eye.width * 2, height: Self.eye.height))
        #expect(try SphericalMetadata.read(sbs).stereoMode == .leftRight)
        let halves = try await Self.markerMeansSideBySide(sbs)
        #expect(halves.left > 200 && halves.right < 50, "left \(halves.left), right \(halves.right)")

        // Already spatial: refused rather than re-encoded.
        let again = try await SpatialVideoSource.open(output)
        await #expect(throws: SpatialVideoError.self) {
            try await SpatialVideoMaker.make(SpatialVideoJob(source: again, output: directory.appendingPathComponent("x.mov"),
                                                             layout: layout, camera: SpatialPreset.generic.metadata))
        }
    }

    @Test func streamSegmentsFromTwoRunsJoinIntoTheWholeVideo() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("vergence-segments-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let clip = directory.appendingPathComponent("clip.mp4"), spatial = directory.appendingPathComponent("clip.spatial.mov")
        try await Self.writeSideBySideClip(to: clip, frames: 48)
        _ = try await SpatialVideoMaker.make(SpatialVideoJob(source: try await SpatialVideoSource.open(clip), output: spatial,
                                                              layout: PackedStereoLayout(arrangement: .sideBySide, isHalfResolution: false),
                                                              camera: SpatialPreset.generic.metadata))
        let job = SideBySideJob(source: try await SpatialVideoSource.open(spatial), output: spatial)
        let segments = directory.appendingPathComponent("segments")
        // Two seconds in half-second segments, made out of order by two runs.
        let late = try await SideBySideSegmenter.run(SideBySideSegments(job: job, directory: segments, segmentSeconds: 0.5, first: 2))
        let early = try await SideBySideSegmenter.run(SideBySideSegments(job: job, directory: segments, segmentSeconds: 0.5, first: 0, last: 1))
        #expect(early.segments == [0, 1] && late.segments == [2, 3])
        #expect(early.frames + late.frames == 48)

        var joined = try Data(contentsOf: segments.appendingPathComponent(SideBySideSegments.initName))
        for n in 0 ..< 4 { joined += try Data(contentsOf: segments.appendingPathComponent(SideBySideSegments.segmentName(n))) }
        let file = directory.appendingPathComponent("joined.mp4")
        try joined.write(to: file)

        // Every frame once, at its own time, a keyframe opening every
        // segment. (AVFoundation starts a fragmented file's timeline at its
        // first fragment, so the offset doesn't show here.)
        let asset = AVURLAsset(url: file)
        let track = try #require(try await asset.loadTracks(withMediaType: .video).first)
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
        reader.add(output)
        #expect(reader.startReading())
        var times: [Double] = [], keys: [Int] = []
        while let sample = output.copyNextSampleBuffer() {
            guard CMSampleBufferGetNumSamples(sample) > 0 else { continue }
            let attachments = CMSampleBufferGetSampleAttachmentsArray(sample, createIfNecessary: false) as? [[CFString: Any]]
            if attachments?.first?[kCMSampleAttachmentKey_NotSync] as? Bool != true { keys.append(times.count) }
            times.append(CMSampleBufferGetPresentationTimeStamp(sample).seconds)
        }
        #expect(times.count == 48, "\(times)")
        // In the segment itself: 1 s in, plus the offset.
        let scale = try #require(FragmentTimes.timescales(initSegment: try Data(contentsOf: segments.appendingPathComponent(SideBySideSegments.initName)))[1])
        let start = try #require(FragmentTimes.firstPresentation(try Data(contentsOf: segments.appendingPathComponent(SideBySideSegments.segmentName(2))))[1])
        #expect(abs(Double(start) / Double(scale) - (1 + SideBySideSegments.timeOffset.seconds)) < 0.001)
        #expect(zip(times, (0 ..< 48).map { Double($0) / 24 }).allSatisfy { abs($0 - $1) < 0.001 }, "\(times)")
        #expect(keys == [0, 12, 24, 36], "\(keys)")
        let halves = try await Self.markerMeansSideBySide(file)
        #expect(halves.left > 200 && halves.right < 50, "left \(halves.left), right \(halves.right)")
    }

    /// The mean luma of the marker band in each half of a packed picture.
    static func markerMeansSideBySide(_ url: URL) async throws -> (left: Double, right: Double) {
        let asset = AVURLAsset(url: url)
        let track = try #require(try await asset.loadTracks(withMediaType: .video).first)
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
        ])
        reader.add(output)
        #expect(reader.startReading())
        let picture = try #require(output.copyNextSampleBuffer().flatMap(CMSampleBufferGetImageBuffer))
        CVPixelBufferLockBaseAddress(picture, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(picture, .readOnly) }
        let base = try #require(CVPixelBufferGetBaseAddressOfPlane(picture, 0)).assumingMemoryBound(to: UInt8.self)
        let stride = CVPixelBufferGetBytesPerRowOfPlane(picture, 0)
        func mean(_ x0: Int) -> Double {
            var sum = 0.0
            for y in 2 ..< 12 { for x in x0 + 8 ..< x0 + 248 { sum += Double(base[y * stride + x]) } }
            return sum / Double(10 * 240)
        }
        reader.cancelReading()
        return (mean(0), mean(eye.width))
    }

    /// The mean luma of the marker band in each eye's first frame.
    static func markerMeans(_ url: URL) async throws -> (left: Double, right: Double) {
        let asset = AVURLAsset(url: url)
        let track = try #require(try await asset.loadTracks(withMediaType: .video).first)
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            AVVideoDecompressionPropertiesKey: [kVTDecompressionPropertyKey_RequestedMVHEVCVideoLayerIDs: [0, 1]],
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
        ])
        reader.add(output)
        #expect(reader.startReading())
        let sample = try #require(output.copyNextSampleBuffer())
        let group = try #require(sample.taggedBuffers)
        var left: Double?, right: Double?
        for tagged in group {
            guard case .pixelBuffer(let picture) = tagged.buffer,
                  let view = tagged.tags.first(matchingCategory: .stereoView)?.value(onlyIfMatching: .stereoView) else { continue }
            CVPixelBufferLockBaseAddress(picture, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(picture, .readOnly) }
            let base = try #require(CVPixelBufferGetBaseAddressOfPlane(picture, 0)).assumingMemoryBound(to: UInt8.self)
            let stride = CVPixelBufferGetBytesPerRowOfPlane(picture, 0)
            var sum = 0.0
            for y in 2 ..< 12 { for x in 8 ..< 248 { sum += Double(base[y * stride + x]) } }
            if view == .leftEye { left = sum / Double(10 * 240) } else if view == .rightEye { right = sum / Double(10 * 240) }
        }
        reader.cancelReading()
        return (try #require(left), try #require(right))
    }
}
#endif
