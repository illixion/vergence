#if canImport(AVFoundation) && os(macOS)
import AVFoundation
import Foundation
import Testing
@testable import VergenceCLI

/// The depth worker's intermediate for a piece with context before it, cut
/// from a Matroska file (millisecond timestamps, as the dev titles that
/// failed with "the depth pass gave 57 of 74 frames"): every frame wanted
/// comes out, frame k at k / rate, as AVFoundation (the depth pass) reads it.
@Suite("Depth piece intermediate", .enabled(if: FileManager.default.isExecutableFile(atPath: "/opt/homebrew/bin/ffmpeg")))
struct DepthIntermediateTests {
    static func run(_ arguments: [String]) throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/opt/homebrew/bin/ffmpeg")
        process.arguments = arguments
        process.standardError = FileHandle.nullDevice
        try process.run()
        process.waitUntilExit()
        #expect(process.terminationStatus == 0)
    }

    @Test func aMatroskaPieceKeepsEveryFrameOnTheGrid() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("title.mkv")
        try Self.run(["-nostdin", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=24000/1001", "-t", "8",
                      "-c:v", "libx264", "-g", "48", "-y", source.path])
        let input = PieceInput(url: source.path, width: 320, height: 180, codec: "h264", stereo: "mono", rate: (24000, 1001))
        // Piece 1 (frames 48…95) with a second of context before it and two after: 74 frames from 24.
        let intermediate = root.appendingPathComponent("piece.mov")
        try Self.run(DepthProducer.intermediateArguments(input: input, first: 24, wanted: 74, into: intermediate))
        let asset = AVURLAsset(url: intermediate)
        let track = try #require(try await asset.loadTracks(withMediaType: .video).first)
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
        reader.add(output)
        #expect(reader.startReading())
        var times: [Double] = []
        while let sample = output.copyNextSampleBuffer() {
            if CMSampleBufferGetNumSamples(sample) > 0 { times.append(CMSampleBufferGetPresentationTimeStamp(sample).seconds) }
        }
        #expect(times.count == 74)
        let rate = 24000.0 / 1001
        #expect(times.enumerated().allSatisfy { abs($0.element * rate - Double($0.offset)) < 0.01 })
    }
}
#endif
