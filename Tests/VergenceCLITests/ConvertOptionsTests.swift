import Foundation
import Testing
import VergenceCommandLine

/// `vergence convert`'s command line, before anything is opened.
@Suite struct ConvertOptionsTests {
    @Test func convertTakesEveryOption() throws {
        let o = try parse(["convert", "film.mkv", "--layout", "HSBS", "--strength", "0.012", "--convergence=0.9",
                           "--sample", "10", "--start", "62.5", "--frames", "240", "--model", "~/m.mlpackage",
                           "--download-model", "--depth-track", "film.vgat", "--quality", "high", "--bitrate", "30",
                           "--max-eye", "1920", "--no-audio", "--dry-run", "--force", "-o", "out/"])
        var expected = Options()
        expected.command = "convert"
        expected.positional = ["film.mkv"]
        expected.layout = "hsbs"
        expected.strength = 0.012
        expected.convergence = 0.9
        expected.sample = 10
        // convert's --start is seconds, kept where --seek goes.
        expected.seek = 62.5
        expected.frames = 240
        expected.model = "~/m.mlpackage"
        expected.downloadModel = true
        expected.depthTrack = "film.vgat"
        expected.quality = .high
        expected.bitrate = 30_000_000
        expected.bitrateText = "30"
        expected.maxEye = 1920
        expected.includeAudio = false
        expected.dryRun = true
        expected.force = true
        expected.output = "out/"
        #expect(o == expected)
        #expect(o.convertLayout == .halfSideBySide)
    }

    @Test func defaultsAreFullSideBySideAndTheSubtleSettings() throws {
        let o = try parse(["convert", "a.mp4"])
        #expect(o.convertLayout == .fullSideBySide)
        #expect(o.strength == nil && o.convergence == nil && o.sample == nil && o.seek == nil && o.frames == Int.max)
        #expect(o.model == nil && o.depthTrack == nil && !o.downloadModel && o.includeAudio)
        // The disc commands still count --start in frames.
        #expect(try parse(["decode", "d", "--start", "24"]).start == 24)
        #expect(try parse(["decode", "d", "--start", "24"]).seek == nil)
    }

    @Test(arguments: [
        ("fsbs", ConvertLayout.fullSideBySide), ("sbs", .fullSideBySide), ("auto", .fullSideBySide),
        ("hsbs", .halfSideBySide), ("tab", .fullTopAndBottom), ("ou", .fullTopAndBottom), ("ftab", .fullTopAndBottom),
        ("htab", .halfTopAndBottom), ("hou", .halfTopAndBottom), ("spatial", .spatial), ("MVHEVC", .spatial),
    ])
    func layouts(_ option: String, _ layout: ConvertLayout) throws {
        #expect(try parse(["convert", "a.mp4", "--layout", option]).convertLayout == layout)
    }

    @Test func outputNamesSayTheLayout() throws {
        let input = URL(fileURLWithPath: "/films/The Film (1999).mkv")
        func name(_ args: String...) throws -> String {
            OutputNames.converted(for: input, try parse(["convert", input.path] + args), single: true).path
        }
        #expect(try name() == "/films/The Film (1999).3D.FSBS.mp4")
        #expect(try name("--layout", "hsbs") == "/films/The Film (1999).3D.HSBS.mp4")
        #expect(try name("--layout", "tab") == "/films/The Film (1999).3D.FTAB.mp4")
        #expect(try name("--layout", "htab") == "/films/The Film (1999).3D.HTAB.mp4")
        #expect(try name("--layout", "spatial") == "/films/The Film (1999).spatial.mov")
        #expect(try name("--sample", "10") == "/films/The Film (1999).sample.3D.FSBS.mp4")
        #expect(try name("-o", "/q/") == "/q/The Film (1999).3D.FSBS.mp4")
        #expect(try name("-o", "/q/x.mp4") == "/q/x.mp4")
    }

    static let refused: [(args: [String], message: String)] = [
        (["convert", "a", "--strength", "0.2"], "--strength takes a share of the frame's width, 0...0.05"),
        (["convert", "a", "--strength", "-0.01"], "--strength takes a share"),
        (["convert", "a", "--convergence", "1.5"], "--convergence takes 0...1"),
        (["convert", "a", "--sample", "0"], "--sample takes seconds, more than 0"),
        (["convert", "a", "--start", "-3"], "--start must be 0 or more"),
        (["convert", "a", "--frames", "0"], "--frames must be 1 or more"),
        (["convert", "a", "--layout", "anaglyph"], "unknown --layout anaglyph"),
        (["convert", "a", "--download-model=yes"], "--download-model takes no value"),
        (["convert", "a", "--model"], "--model needs a value"),
        (["make", "a", "--layout", "spatial"], "--layout spatial is for convert"),
        (["make", "a", "--layout", "fsbs"], "--layout fsbs is for convert"),
    ]

    @Test(arguments: refused)
    func refuses(_ args: [String], _ message: String) {
        #expect {
            _ = try parse(args)
        } throws: { error in
            (error as? UsageError)?.problem?.contains(message) == true
        }
    }
}
