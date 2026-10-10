#if os(macOS)
import AVFoundation
import CoreMedia
import CoreVideo
import Testing
import VideoToolbox
@testable import VergenceApple
@testable import VergenceCore
import VergenceTestSupport

/// End to end: a synthetic side-by-side clip in, a spatial video out that
/// `inspect` calls complete, with each eye holding its own half.
@Suite struct SpatialVideoMakerTests {
    typealias Clip = SyntheticClip
    static let eye = Clip.eye
    static let disparity = Clip.disparity
    static let frames = Clip.frames

    static func writeSideBySideClip(to url: URL, frames: Int = frames, audio: Bool = false, hdr: Bool = false) async throws {
        try await Clip.writeSideBySide(to: url, frames: frames, audio: audio, hdr: hdr)
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
