#if os(macOS)
import AVFoundation
import Foundation
import Testing
import VergenceCommandLine
import VergenceCore
import VergenceTestSupport

/// `vergence convert --upscale` run for real on the synthetic side-by-side
/// clip, with the tiny network the upscaler's own tests use (its picture is
/// meaningless; the run around it is what is checked): a 3D source found by
/// its picture and upscaled eye by eye, and a 2D one upscaled only.
@Suite(.serialized) struct ConvertUpscaleRunTests {
    typealias CLI = CommandLineRunTests

    static let network = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("VergenceAppleTests/Fixtures/tiny-srvgg.pth")

    static func size(_ url: URL) async throws -> CGSize {
        let video = try #require(try await AVURLAsset(url: url).loadTracks(withMediaType: .video).first)
        return try await video.load(.naturalSize)
    }

    @Test func upscalesEachEyeOfA3DSource() async throws {
        let directory = try CLI.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try await SyntheticClip.writeSideBySide(to: directory.appendingPathComponent("clip.mp4"), frames: 24, audio: true)

        let run = try CLI.run(["convert", "clip.mp4", "--upscale", "2x", "--upscale-model", Self.network.path], in: directory)
        #expect(run.status == 0, "\(run.stderr)")
        let output = directory.appendingPathComponent("clip.2x.3D.FSBS.mp4")
        #expect(run.stderr.contains("3D source (side by side"), "\(run.stderr)")
        #expect(run.stderr.contains("upscaled 2× from \(SyntheticClip.eye.width)×\(SyntheticClip.eye.height)"), "\(run.stderr)")
        #expect(!run.stderr.contains("  depth"), "no depth for a 3D source: \(run.stderr)")
        #expect(run.stderr.contains("picture: 24 frames"), "\(run.stderr)")
        let size = try await Self.size(output)
        #expect(Int(size.width) == SyntheticClip.eye.width * 4 && Int(size.height) == SyntheticClip.eye.height * 2)
        #expect(try SphericalMetadata.read(output).stereoMode == .leftRight)
        #expect(try await AVURLAsset(url: output).loadTracks(withMediaType: .audio).count == 1)
        #expect(try FileManager.default.contentsOfDirectory(atPath: directory.path).sorted() == ["clip.2x.3D.FSBS.mp4", "clip.mp4"])

        // Half top and bottom out: the eyes squeezed into one eye's size.
        let half = try CLI.run(["convert", "clip.mp4", "--upscale", "2x", "--upscale-model", Self.network.path,
                                "--layout", "htab", "--input-layout", "sbs", "--no-audio"], in: directory)
        #expect(half.status == 0, "\(half.stderr)")
        let squeezed = try await Self.size(directory.appendingPathComponent("clip.2x.3D.HTAB.mp4"))
        #expect(Int(squeezed.width) == SyntheticClip.eye.width * 2 && Int(squeezed.height) == SyntheticClip.eye.height * 2)
    }

    @Test func upscalesA2DSourceOnly() async throws {
        let directory = try CLI.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try await SyntheticClip.writeSideBySide(to: directory.appendingPathComponent("clip.mp4"), frames: 12)

        let run = try CLI.run(["convert", "clip.mp4", "--upscale", "2x", "--upscale-model", Self.network.path,
                               "--input-layout", "2d", "--layout", "2d", "--max-eye", "800"], in: directory)
        #expect(run.status == 0, "\(run.stderr)")
        let output = directory.appendingPathComponent("clip.2x.mp4")
        let size = try await Self.size(output)
        // 512×144 made 1024×288, then held to 800 wide.
        #expect(Int(size.width) == 800 && Int(size.height) == 224, "\(size)")
        #expect(try SphericalMetadata.read(output).stereoMode == nil)

        // --layout 2d needs --upscale; a 3D source can't be made 2D.
        let refused = try CLI.run(["convert", "clip.mp4", "--layout", "2d"], in: directory)
        #expect(refused.status == 64 && refused.stderr.contains("--layout 2d makes no 3D"), "\(refused.stderr)")
        let stereo = try CLI.run(["convert", "clip.mp4", "--upscale", "2x", "--upscale-model", Self.network.path,
                                  "--input-layout", "sbs", "--layout", "2d", "--force"], in: directory)
        #expect(stereo.status == 1 && stereo.stderr.contains("--layout 2d is for 2D sources"), "\(stereo.stderr)")
    }

    static let gainModel: URL? = {
        let path = ProcessInfo.processInfo.environment["VERGENCE_GAIN_MODEL"]
            ?? URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
                .deletingLastPathComponent().appendingPathComponent("Hypnos/Hypnos/Hypnos/Resources/GainMapGMNet.mlpackage").path
        return FileManager.default.fileExists(atPath: path) ? URL(fileURLWithPath: path) : nil
    }()

    /// --antialias and --ai-hdr on a 3D source: 10-bit HEVC tagged BT.2020 PQ
    /// with HDR10 static metadata, the eyes upscaled.
    @Test(.enabled(if: gainModel != nil, "no GMNet: set VERGENCE_GAIN_MODEL to GainMapGMNet.mlpackage"))
    func antialiasesAndMakesHDR() async throws {
        let directory = try CLI.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try await SyntheticClip.writeSideBySide(to: directory.appendingPathComponent("clip.mp4"), frames: 24, audio: true)
        let run = try CLI.run(["convert", "clip.mp4", "--upscale", "2x", "--upscale-model", Self.network.path, "--antialias",
                               "--ai-hdr", "--gain-model", try #require(Self.gainModel).path], in: directory)
        #expect(run.status == 0, "\(run.stderr)")
        #expect(run.stderr.contains("antialiased:") && run.stderr.contains("AI HDR grids: 24 frames"), "\(run.stderr)")
        let output = directory.appendingPathComponent("clip.2x.hdr.3D.FSBS.mp4")
        let video = try #require(try await AVURLAsset(url: output).loadTracks(withMediaType: .video).first)
        let format = try #require(try await video.load(.formatDescriptions).first)
        let extensions = CMFormatDescriptionGetExtensions(format) as? [String: Any] ?? [:]
        #expect(extensions[kCMFormatDescriptionExtension_TransferFunction as String] as? String
            == kCMFormatDescriptionTransferFunction_SMPTE_ST_2084_PQ as String)
        #expect(extensions[kCMFormatDescriptionExtension_ColorPrimaries as String] as? String
            == kCMFormatDescriptionColorPrimaries_ITU_R_2020 as String)
        #expect(extensions[kCMFormatDescriptionExtension_MasteringDisplayColorVolume as String] != nil)
        #expect(extensions[kCMFormatDescriptionExtension_ContentLightLevelInfo as String] != nil)
        #expect(try SphericalMetadata.read(output).stereoMode == .leftRight)
        // An HDR source isn't made HDR again; --ai-hdr names its output.
        #expect(OutputNames.converted(for: URL(fileURLWithPath: "/f/a.mkv"), try parse(["convert", "a", "--ai-hdr", "--upscale", "2x"]),
                                      single: true).lastPathComponent == "a.2x.hdr.3D.FSBS.mp4")
    }
}
#endif
