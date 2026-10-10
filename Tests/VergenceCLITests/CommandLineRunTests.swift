import Foundation
import Testing
import VergenceCommandLine
#if os(macOS)
import VergenceTestSupport
#endif

/// The built `vergence` binary, run as a user would: exit codes and messages
/// for a bad command line, and (macOS) make, sbs --info and sbs --segments on
/// a synthetic clip.
@Suite struct CommandLineRunTests {
    struct Run {
        var status: Int32
        var stdout: String
        var stderr: String
    }

    /// In the build's products folder, beside the tests; VERGENCE_BINARY
    /// points elsewhere.
    static let binary: URL? = {
        if let path = ProcessInfo.processInfo.environment["VERGENCE_BINARY"] { return URL(fileURLWithPath: path) }
        // macOS: the .xctest bundle's folder. Linux: the test executable's,
        // which is also the main bundle itself.
        let folders = [Bundle(for: BundleMarker.self), Bundle.main].flatMap { bundle in
            [bundle.bundleURL.deletingLastPathComponent(), bundle.bundleURL, bundle.executableURL?.deletingLastPathComponent()]
                .compactMap { $0 }
        }
        return folders.map { $0.appendingPathComponent("vergence") }
            .first { FileManager.default.isExecutableFile(atPath: $0.path) }
    }()

    private final class BundleMarker {}

    static func run(_ args: [String], in directory: URL? = nil) throws -> Run {
        let binary = try #require(binary, "no vergence binary beside the tests (set VERGENCE_BINARY)")
        let process = Process()
        process.executableURL = binary
        process.arguments = args
        if let directory { process.currentDirectoryURL = directory }
        var environment = ProcessInfo.processInfo.environment
        environment["VERGENCE_HTTP_HEADER"] = nil
        process.environment = environment
        // Files rather than pipes, so a chatty run can't fill a pipe and stall.
        let scratch = FileManager.default.temporaryDirectory.appendingPathComponent("vergence-run-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: scratch) }
        let out = scratch.appendingPathComponent("stdout"), err = scratch.appendingPathComponent("stderr")
        FileManager.default.createFile(atPath: out.path, contents: nil)
        FileManager.default.createFile(atPath: err.path, contents: nil)
        process.standardOutput = try FileHandle(forWritingTo: out)
        process.standardError = try FileHandle(forWritingTo: err)
        try process.run()
        process.waitUntilExit()
        return Run(status: process.terminationStatus,
                   stdout: try String(contentsOf: out, encoding: .utf8), stderr: try String(contentsOf: err, encoding: .utf8))
    }

    static func temporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("vergence-cli-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    /// Usage problems exit 64 (EX_USAGE) with the problem and the usage text.
    static let usageProblems: [(args: [String], message: String?)] = [
        ([], nil),
        (["teleport"], "unknown command teleport"),
        (["make", "a.mp4", "--bogus"], "unknown option --bogus"),
        (["sbs", "a.mov", "--segments"], "--segments needs a value"),
        (["sbs", "a.mov", "--segment-seconds", "four"], "--segment-seconds takes a number, not \"four\""),
        (["sbs", "a.mov", "--segments", "s", "--first", "3", "--last", "1"], "--last 1 comes before --first 3"),
        (["decode", "d", "--frames", "1e30"], "--frames takes a whole number"),
    ]

    @Test(arguments: usageProblems)
    func badCommandLinesExitWithUsage(_ args: [String], _ message: String?) throws {
        let run = try Self.run(args)
        #expect(run.status == 64)
        if let message { #expect(run.stderr.contains("vergence: \(message)"), "\(run.stderr)") }
        #expect(run.stderr.contains("usage: vergence make"))
        #expect(run.stdout.isEmpty)
    }

    #if os(macOS)
    @Test func aMissingInputIsNamed() throws {
        let directory = try Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        for command in ["make", "sbs"] {
            let run = try Self.run([command, "nothing.mov"], in: directory)
            #expect(run.status == 64)
            #expect(run.stderr.contains("vergence: nothing.mov: no such file"), "\(run.stderr)")
            #expect(run.stderr.contains("vergence: \(command) needs a"), "\(run.stderr)")
        }
        let info = try Self.run(["sbs", "nothing.mov", "--info"], in: directory)
        #expect(info.status == 64 && info.stdout.isEmpty)
    }
    #endif
}

#if os(macOS)
/// make, then sbs three ways, on one synthetic side-by-side clip of a second.
@Suite(.serialized) struct SpatialCommandRunTests {
    typealias CLI = CommandLineRunTests

    @Test func makeThenSideBySideInfoSegmentsAndFile() async throws {
        let directory = try CLI.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let clip = directory.appendingPathComponent("clip.mp4")
        try await SyntheticClip.writeSideBySide(to: clip, frames: 24)

        // make finds the layout by itself and writes beside the input.
        let make = try CLI.run(["make", "clip.mp4"], in: directory)
        #expect(make.status == 0, "\(make.stderr)")
        let spatial = directory.appendingPathComponent("clip.spatial.mov")
        #expect(FileManager.default.fileExists(atPath: spatial.path), "\(make.stderr)")
        #expect(make.stderr.contains("clip.mp4 -> ") && make.stderr.contains("/clip.spatial.mov"), "\(make.stderr)")
        // Run again: already made, skipped, still a success.
        let again = try CLI.run(["make", "clip.mp4"], in: directory)
        #expect(again.status == 0)
        #expect(again.stderr.contains("clip.spatial.mov is already made"), "\(again.stderr)")

        // --info: JSON on stdout, nothing written.
        let info = try infoJSON(["sbs", "clip.spatial.mov", "--info", "--segment-seconds", "0.5"], in: directory)
        #expect(info["spatial"] as? Bool == true)
        #expect(info["width"] as? Int == 512 && info["height"] as? Int == 144)
        #expect(info["eyeWidth"] as? Int == 256 && info["eyeHeight"] as? Int == 144)
        #expect(info["halfWidth"] as? Bool == false && info["hdr"] as? Bool == false)
        #expect(info["frameRate"] as? Double == 24)
        #expect(abs((info["durationSeconds"] as? Double ?? 0) - 1) < 0.01)
        #expect(info["segmentSeconds"] as? Double == 0.5 && info["segments"] as? Int == 2)
        #expect((info["bitsPerSecond"] as? Int ?? 0) > 0)
        #expect(info["audio"] as? [String] == [])
        #expect(Set(info.keys) == ["spatial", "durationSeconds", "frameRate", "width", "height", "eyeWidth", "eyeHeight",
                                   "halfWidth", "bitsPerSecond", "hdr", "segmentSeconds", "segments", "audio"])
        let half = try infoJSON(["sbs", "clip.spatial.mov", "--info", "--half"], in: directory)
        #expect(half["width"] as? Int == 256 && half["eyeWidth"] as? Int == 128 && half["halfWidth"] as? Bool == true)
        #expect(half["segmentSeconds"] as? Double == 4 && half["segments"] as? Int == 1)
        let capped = try infoJSON(["sbs", "clip.spatial.mov", "--info", "--max-eye", "128"], in: directory)
        #expect(capped["eyeWidth"] as? Int == 128 && capped["eyeHeight"] as? Int == 72)
        // A flat video is answered, not refused.
        let flat = try infoJSON(["sbs", "clip.mp4", "--info"], in: directory)
        #expect(flat as NSDictionary == ["spatial": false] as NSDictionary)
        #expect(try FileManager.default.contentsOfDirectory(atPath: directory.path).sorted() == ["clip.mp4", "clip.spatial.mov"])

        // --segments: the playlist, the init segment and N.m4s in the folder.
        let segments = try CLI.run(["sbs", "clip.spatial.mov", "--segments", "seg", "--segment-seconds", "0.5"], in: directory)
        #expect(segments.status == 0, "\(segments.stderr)")
        #expect(segments.stdout.isEmpty)
        let folder = directory.appendingPathComponent("seg")
        #expect(try FileManager.default.contentsOfDirectory(atPath: folder.path).sorted() == ["0.m4s", "1.m4s", "index.m3u8", "init.mp4"])
        let playlist = try String(contentsOf: folder.appendingPathComponent(OutputNames.segmentPlaylist), encoding: .utf8)
        #expect(playlist.contains("#EXT-X-MAP:URI=\"init.mp4\""), "\(playlist)")
        #expect(playlist.contains("\n0.m4s") && playlist.contains("\n1.m4s") && !playlist.contains("2.m4s"), "\(playlist)")
        // A range past the end is refused at run time (the grid is known
        // only once the file is open).
        let past = try CLI.run(["sbs", "clip.spatial.mov", "--segments", "seg", "--segment-seconds", "0.5", "--first", "5"],
                               in: directory)
        #expect(past.status == 1 && past.stderr.contains("outside 0...1"), "\(past.stderr)")

        // The file: named for players, full width unless --half.
        let dry = try CLI.run(["sbs", "clip.spatial.mov", "--half", "--dry-run"], in: directory)
        #expect(dry.status == 0 && dry.stderr.contains("/clip.spatial.3D.HSBS.mp4"), "\(dry.stderr)")
        #expect(!FileManager.default.fileExists(atPath: directory.appendingPathComponent("clip.spatial.3D.HSBS.mp4").path))
        let file = try CLI.run(["sbs", "clip.spatial.mov"], in: directory)
        #expect(file.status == 0, "\(file.stderr)")
        #expect(FileManager.default.fileExists(atPath: directory.appendingPathComponent("clip.spatial.3D.FSBS.mp4").path))
        // And sbs leaves its own outputs out of a folder.
        let folderRun = try CLI.run(["sbs", ".", "--dry-run"], in: directory)
        #expect(!folderRun.stderr.contains("3D.FSBS.mp4 ->"), "\(folderRun.stderr)")
    }

    func infoJSON(_ args: [String], in directory: URL) throws -> [String: Any] {
        let run = try CLI.run(args, in: directory)
        #expect(run.status == 0, "\(run.stderr)")
        #expect(run.stdout.hasSuffix("}\n"))
        let object = try JSONSerialization.jsonObject(with: Data(run.stdout.utf8))
        return try #require(object as? [String: Any], "\(run.stdout)")
    }
}
#endif
