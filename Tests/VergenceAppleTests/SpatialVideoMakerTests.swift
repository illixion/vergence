#if os(macOS)
import AVFoundation
import CoreMedia
import CoreVideo
import Testing
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
    static func writeSideBySideClip(to url: URL) async throws {
        let width = eye.width * 2, height = eye.height
        let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: width, AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: [AVVideoAverageBitRateKey: 8_000_000],
        ])
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width, kCVPixelBufferHeightKey as String: height,
        ])
        writer.add(input)
        #expect(writer.startWriting())
        writer.startSession(atSourceTime: .zero)
        let sceneWidth = eye.width + disparity
        for frame in 0 ..< frames {
            var state = UInt64(frame + 1) &* 0x9E37_79B9_7F4A_7C15 | 1
            let scene = (0 ..< sceneWidth * height).map { _ -> UInt8 in
                state ^= state << 13; state ^= state >> 7; state ^= state << 17
                return UInt8(truncatingIfNeeded: state >> 24)
            }
            while !input.isReadyForMoreMediaData { try await Task.sleep(for: .milliseconds(2)) }
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
            #expect(adaptor.append(picture, withPresentationTime: CMTime(value: CMTimeValue(frame), timescale: 24)))
        }
        input.markAsFinished()
        await writer.finishWriting()
        #expect(writer.status == .completed)
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
