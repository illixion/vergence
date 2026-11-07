#if os(macOS)
import AVFoundation
import Foundation
import Testing
import VergenceCore
import VergenceTestSupport

/// `vergence convert` run for real on a short synthetic clip: Core ML depth,
/// the Metal warp, HEVC side by side with its st3d box and audio. Needs a
/// depth model, which the repository doesn't carry: VERGENCE_DEPTH_MODEL, or
/// one installed where the apps keep it (Application Support/DepthModels).
@Suite(.serialized) struct ConvertRunTests {
    typealias CLI = CommandLineRunTests

    static let model: URL? = {
        if let path = ProcessInfo.processInfo.environment["VERGENCE_DEPTH_MODEL"] {
            return URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        }
        let store = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/DepthModels")
        let names = (try? FileManager.default.contentsOfDirectory(atPath: store.path)) ?? []
        return names.sorted().first { $0.hasPrefix("DepthAnythingV2Small") && ($0.hasSuffix(".mlpackage") || $0.hasSuffix(".mlmodelc")) }
            .map { store.appendingPathComponent($0) }
    }()

    @Test(.enabled(if: model != nil, "no depth model: set VERGENCE_DEPTH_MODEL to a Depth Anything V2 Small .mlpackage or .mlmodelc"))
    func convertsAClipToSideBySideWithDifferentEyes() async throws {
        let directory = try CLI.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let clip = directory.appendingPathComponent("clip.mp4")
        try await SyntheticClip.writeSideBySide(to: clip, frames: 24, audio: true)
        let model = try #require(Self.model)

        let run = try CLI.run(["convert", "clip.mp4", "--model", model.path, "--strength", "0.03",
                               "--depth-track", "clip.vgat"], in: directory)
        #expect(run.status == 0, "\(run.stderr)")
        let output = directory.appendingPathComponent("clip.3D.FSBS.mp4")
        #expect(FileManager.default.fileExists(atPath: output.path), "\(run.stderr)")
        #expect(run.stderr.contains("depth: 24 frames") && run.stderr.contains("picture: 24 frames"), "\(run.stderr)")
        #expect(!run.stderr.contains("without depth"), "\(run.stderr)")
        // Nothing temporary left behind.
        #expect(try FileManager.default.contentsOfDirectory(atPath: directory.path).sorted()
            == ["clip.3D.FSBS.mp4", "clip.mp4", "clip.vgat"])

        // HEVC, twice the frame's width, Google's st3d box saying left-right, the audio copied.
        let asset = AVURLAsset(url: output)
        let video = try #require(try await asset.loadTracks(withMediaType: .video).first)
        let format = try #require(try await video.load(.formatDescriptions).first)
        #expect(CMFormatDescriptionGetMediaSubType(format) == kCMVideoCodecType_HEVC)
        let size = try await video.load(.naturalSize)
        #expect(Int(size.width) == SyntheticClip.eye.width * 4 && Int(size.height) == SyntheticClip.eye.height)
        #expect(try SphericalMetadata.read(output).stereoMode == .leftRight)
        #expect(try await asset.loadTracks(withMediaType: .audio).count == 1)

        // The eyes differ: the warp moved the picture.
        let difference = try Self.meanEyeDifference(asset, video)
        #expect(difference > 2, "left and right eyes are nearly the same (mean difference \(difference))")

        // The depth as analysis-track records, one a frame.
        let track = try AnalysisTrackIndex(Data(contentsOf: directory.appendingPathComponent("clip.vgat")))
        #expect(track.entries.count == 24 && track.entries.allSatisfy { $0.kind == .depth })
        #expect(track.header.sourceWidth == SyntheticClip.eye.width * 2 && track.header.sourceHeight == SyntheticClip.eye.height)
    }

    /// The mean absolute luma difference between the left and right halves of
    /// the middle frame, 0…255.
    static func meanEyeDifference(_ asset: AVURLAsset, _ track: AVAssetTrack) throws -> Double {
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
        ])
        reader.add(output)
        #expect(reader.startReading())
        var frame: CVPixelBuffer?
        for _ in 0 ..< 12 {
            guard let sample = output.copyNextSampleBuffer() else { break }
            frame = CMSampleBufferGetImageBuffer(sample)
        }
        reader.cancelReading()
        let picture = try #require(frame)
        CVPixelBufferLockBaseAddress(picture, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(picture, .readOnly) }
        let base = try #require(CVPixelBufferGetBaseAddressOfPlane(picture, 0)).assumingMemoryBound(to: UInt8.self)
        let stride = CVPixelBufferGetBytesPerRowOfPlane(picture, 0)
        let width = CVPixelBufferGetWidthOfPlane(picture, 0), height = CVPixelBufferGetHeightOfPlane(picture, 0)
        let eye = width / 2
        var total = 0
        // Below the synthetic clip's marker band, which differs by design.
        for y in 24 ..< height {
            for x in 0 ..< eye {
                total += abs(Int(base[y * stride + x]) - Int(base[y * stride + eye + x]))
            }
        }
        return Double(total) / Double(eye * (height - 24))
    }
}
#endif
