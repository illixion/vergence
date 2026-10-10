import Foundation
import Testing
import VergenceCommandLine
import VergenceCore
import VergenceDisc

/// The command line as parsed, before anything is opened: every subcommand's
/// options, and what's refused before a file is touched.
@Suite struct OptionsTests {
    @Test func makeTakesEveryOption() throws {
        let o = try parse(["make", "clip.mp4", "--layout", "HSBS", "--swap-eyes", "--preset", "iphone", "--baseline", "64.5",
                           "--hfov=70", "--adjust", "-0.02", "--quality", "high", "--bitrate", "20M", "--no-audio",
                           "--dry-run", "--force", "-o", "out"])
        var expected = Options()
        expected.command = "make"
        expected.positional = ["clip.mp4"]
        expected.layout = "hsbs"
        expected.swapEyes = true
        expected.preset = .iphone
        expected.baseline = 64.5
        expected.hfov = 70
        expected.adjust = -0.02
        expected.quality = .high
        expected.bitrate = 20_000_000
        expected.bitrateText = "20M"
        expected.includeAudio = false
        expected.dryRun = true
        expected.force = true
        expected.output = "out"
        #expect(o == expected)
    }

    @Test func makeTakesFoldersAndShortAliases() throws {
        let o = try parse(["make", "a", "b/", "-f", "ou", "-r", "-q", "max", "-b", "750K", "-n", "-y"])
        #expect(o.positional == ["a", "b/"])
        #expect(o.layout == "ou" && o.swapEyes && o.quality == .max && o.bitrate == 750_000 && o.dryRun && o.force)
        // Defaults: nothing given, nothing changed.
        let plain = try parse(["make", "a"])
        #expect(plain.layout == "auto" && plain.preset == .generic && plain.quality == .standard && plain.bitrate == nil)
        #expect(plain.output == "-" && plain.includeAudio && !plain.swapEyes)
    }

    @Test func sideBySideOptions() throws {
        let o = try parse(["sbs", "a.mov", "--half", "--max-eye", "1920", "--no-ambisonics", "--output=out/"])
        #expect(o.command == "sbs" && o.positional == ["a.mov"])
        #expect(o.half && o.maxEye == 1920 && !o.ambisonics && o.output == "out/")
        #expect(o.segments == nil && !o.info)
    }

    @Test func sideBySideSegmentOptions() throws {
        let o = try parse(["sbs", "a.mov", "--segments", "seg", "--segment-seconds", "6", "--first", "2", "--last=5"])
        #expect(o.segments == "seg" && o.segmentSeconds == 6 && o.firstSegment == 2 && o.lastSegment == 5)
        let defaults = try parse(["sbs", "a.mov", "--segments", "seg"])
        #expect(defaults.segmentSeconds == 4 && defaults.firstSegment == 0 && defaults.lastSegment == nil)
        // One segment is a range too.
        #expect(try parse(["sbs", "a.mov", "--segments", "s", "--first", "3", "--last", "3"]).lastSegment == 3)
    }

    @Test func sideBySideInfo() throws {
        let o = try parse(["quest", "a.mov", "--info", "--segment-seconds", "0.5", "--half"])
        #expect(o.command == "quest" && o.info && o.segmentSeconds == 0.5 && o.half)
    }

    @Test func discCommands() throws {
        let o = try parse(["decode", "disc.iso", "--playlist", "00800", "--packing", "sbs", "--seek", "3600.5",
                           "--start", "5", "--frames", "240", "--threads", "-1", "-o", "-"])
        #expect(o.command == "decode" && o.positional == ["disc.iso"] && o.playlist == "00800")
        #expect(o.packing == .sideBySide && o.seek == 3600.5 && o.start == 5 && o.frames == 240 && o.threads == -1 && o.output == "-")
        // A frame count may be written as a float as long as it is whole.
        #expect(try parse(["decode", "d", "--frames", "2.4e2"]).frames == 240)
        #expect(try parse(["decode", "a.m2ts", "b.m2ts"]).positional == ["a.m2ts", "b.m2ts"])
        #expect(try parse(["subtitles", "d", "--pid", "0x1203"]).pid == 0x1203)
        #expect(try parse(["streams", "d", "--pid", "1100"]).pid == 0x1100)
        let audio = try parse(["audio", "v.mov", "--track", "2", "--bitrate", "256", "--output", "a.opus"])
        #expect(audio.track == 2 && audio.bitrateText == "256" && audio.output == "a.opus")
    }

    @Test func negativeNumbersAreValuesAndPositionalsNotOptions() throws {
        #expect(try parse(["make", "a", "--adjust", "-0.02"]).adjust == -0.02)
        #expect(try parse(["make", "a", "--adjust=-0.02"]).adjust == -0.02)
        // A bare "-" (stdin/stdout) and a bare number are positional.
        #expect(try parse(["cat", "-", "-5"]).positional == ["-", "-5"])
    }

    @Test func headersFromOptionsAndEnvironment() throws {
        let o = try parse(["info", "https://x/disc.iso", "-H", "Authorization: Bearer t", "--header=X-A:  b "],
                          environment: ["VERGENCE_HTTP_HEADER": "Cookie: c=1"])
        #expect(o.headers == ["Authorization": "Bearer t", "X-A": "b", "Cookie": "c=1"])
        #expect(try parse(["info", "d"], environment: ["PATH": "/bin"]).headers.isEmpty)
    }

    /// Refused with a message that names the problem.
    static let refused: [(args: [String], message: String)] = [
        (["make", "a", "--bogus"], "unknown option --bogus"),
        (["sbs", "a", "--segments"], "--segments needs a value"),
        (["sbs", "a", "--segment-seconds", "four"], "--segment-seconds takes a number, not \"four\""),
        (["sbs", "a", "--segment-seconds", "nan"], "--segment-seconds takes a number"),
        (["sbs", "a", "--segment-seconds", "0.1"], "--segment-seconds must be 0.5...60"),
        (["sbs", "a", "--segment-seconds=61"], "--segment-seconds must be 0.5...60"),
        (["sbs", "a", "--first", "5", "--last", "2"], "--last 2 comes before --first 5"),
        (["sbs", "a", "--first", "-1"], "--first must be 0 or more"),
        (["sbs", "a", "--first", "1.5"], "--first takes a whole number, not \"1.5\""),
        (["decode", "d", "--frames", "1e30"], "--frames takes a whole number"),
        (["decode", "d", "--start", "x"], "--start takes a whole number"),
        (["make", "a", "--half=yes"], "--half takes no value"),
        (["make", "a", "--preset", "gopro"], "unknown --preset gopro (generic, iphone, avp, film)"),
        (["make", "a", "--layout", "3d"], "unknown --layout 3d"),
        (["make", "a", "--quality", "ultra"], "unknown --quality ultra"),
        (["make", "a", "--bitrate", "fast"], "bad --bitrate"),
        (["make", "a", "--bitrate", "1e300M"], "bad --bitrate"),
        (["sbs", "a", "--max-eye", "0"], "--max-eye takes a size in pixels"),
        (["decode", "d", "--packing", "diagonal"], "unknown --packing diagonal"),
        (["decode", "d", "--pid", "zz"], "bad --pid zz"),
        (["info", "d", "-H", "no colon"], "a header is \"Name: value\""),
    ]

    @Test(arguments: refused)
    func refuses(_ args: [String], _ message: String) {
        #expect {
            _ = try parse(args)
        } throws: { error in
            (error as? UsageError)?.problem?.contains(message) == true
        }
    }

    @Test func refusesNoCommandAndABadEnvironmentHeader() {
        #expect(throws: UsageError()) { _ = try parse([]) }
        #expect(throws: UsageError.self) { _ = try parse(["info", "d"], environment: ["VERGENCE_HTTP_HEADER": "token"]) }
    }

    @Test func bitrates() {
        #expect(parseBitrate("20") == 20_000_000)
        #expect(parseBitrate("20m") == 20_000_000)
        #expect(parseBitrate("2.5M") == 2_500_000)
        #expect(parseBitrate("750K") == 750_000)
        #expect(parseBitrate("20000000") == 20_000_000)
        #expect(parseBitrate("0") == nil)
        #expect(parseBitrate("-5") == nil)
        #expect(parseBitrate("M") == nil)
        #expect(parseBitrate("1e300") == nil)
    }
}

/// Where `make`, `sbs` and `sbs --segments` write when not told exactly.
@Suite struct OutputNameTests {
    static func options(_ args: String...) throws -> Options { try parse(["make"] + args) }

    let input = URL(fileURLWithPath: "/videos/Holiday 2024.mp4")

    @Test func makeWritesBesideTheInput() throws {
        #expect(OutputNames.spatialVideo(for: input, try Self.options(), single: true).path == "/videos/Holiday 2024.spatial.mov")
        #expect(OutputNames.spatialVideo(for: URL(fileURLWithPath: "/v/clip.3D.HSBS.mkv"), try Self.options(), single: false).path
            == "/v/clip.3D.HSBS.spatial.mov")
    }

    @Test func outputIsAFileForOneInputAndAFolderOtherwise() throws {
        let file = try Self.options("-o", "/out/mine.mov")
        #expect(OutputNames.spatialVideo(for: input, file, single: true).path == "/out/mine.mov")
        #expect(OutputNames.spatialVideo(for: input, file, single: false).path == "/out/mine.mov/Holiday 2024.spatial.mov")
        // A trailing slash makes it a folder even for one input.
        let folder = try Self.options("-o", "/out/")
        #expect(OutputNames.spatialVideo(for: input, folder, single: true).path == "/out/Holiday 2024.spatial.mov")
    }

    @Test func anExistingFolderIsAFolder() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("vergence-names-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let o = try Self.options("-o", directory.path)
        #expect(OutputNames.spatialVideo(for: input, o, single: true) == directory.appendingPathComponent("Holiday 2024.spatial.mov"))
        #expect(OutputNames.sideBySide(for: input, o, single: true) == directory.appendingPathComponent("Holiday 2024.3D.FSBS.mp4"))
    }

    @Test func sideBySideNamesSayFullOrHalf() throws {
        let spatial = URL(fileURLWithPath: "/v/IMG_0042.MOV")
        #expect(OutputNames.sideBySide(for: spatial, try Self.options(), single: true).path == "/v/IMG_0042.3D.FSBS.mp4")
        #expect(OutputNames.sideBySide(for: spatial, try Self.options("--half"), single: true).path == "/v/IMG_0042.3D.HSBS.mp4")
        #expect(OutputNames.sideBySide(for: spatial, try Self.options("--half", "-o", "/q/"), single: true).path == "/q/IMG_0042.3D.HSBS.mp4")
        #expect(OutputNames.sideBySide(for: spatial, try Self.options("-o", "/q/x.mp4"), single: true).path == "/q/x.mp4")
    }

    @Test func segmentsGoInTheirFolder() throws {
        let o = try parse(["sbs", "a.mov", "--segments", "/srv/seg"])
        #expect(OutputNames.segmentDirectory(o).path == "/srv/seg")
        #expect(OutputNames.segmentDirectory(o).appendingPathComponent(OutputNames.segmentPlaylist).path == "/srv/seg/index.m3u8")
        // --info alone writes nothing, but the request still has a folder.
        let info = try parse(["sbs", "a.mov", "--info"])
        #expect(OutputNames.segmentDirectory(info).standardizedFileURL.path
            == URL(fileURLWithPath: FileManager.default.currentDirectoryPath).standardizedFileURL.path)
    }
}
