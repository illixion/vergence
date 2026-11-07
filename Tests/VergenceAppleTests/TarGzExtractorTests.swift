#if canImport(Compression) && os(macOS)
import Foundation
import Testing
@testable import VergenceApple

/// A model published as a `.tar.gz` of its package unpacks without `tar`:
/// the files come out as they went in; macOS's `._` entries are left out and
/// a path leaving the folder is refused.
@Suite("Tar.gz extractor")
struct TarGzExtractorTests {
    @Test func aPackageComesOutAsItWentIn() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let package = root.appendingPathComponent("in/Model.mlpackage/Data/com.apple.CoreML")
        try FileManager.default.createDirectory(at: package.appendingPathComponent("weights"), withIntermediateDirectories: true)
        let weights = Data((0 ..< 1_300_000).map { UInt8(truncatingIfNeeded: $0 &* 2_654_435_761 >> 7) })
        try weights.write(to: package.appendingPathComponent("weights/weight.bin"))
        try Data("model".utf8).write(to: package.appendingPathComponent("model.mlmodel"))
        try Data("{}".utf8).write(to: root.appendingPathComponent("in/Model.mlpackage/Manifest.json"))
        let archive = root.appendingPathComponent("model.tar.gz")
        let tar = Process()
        tar.executableURL = URL(fileURLWithPath: "/usr/bin/tar")
        tar.arguments = ["-czf", archive.path, "-C", root.appendingPathComponent("in").path, "Model.mlpackage"]
        try tar.run()
        tar.waitUntilExit()
        #expect(tar.terminationStatus == 0)

        let out = root.appendingPathComponent("out")
        let top = try TarGzExtractor.extract(archive, into: out)
        #expect(top == ["Model.mlpackage"])
        #expect(try Data(contentsOf: out.appendingPathComponent("Model.mlpackage/Data/com.apple.CoreML/weights/weight.bin")) == weights)
        #expect(try String(contentsOf: out.appendingPathComponent("Model.mlpackage/Manifest.json"), encoding: .utf8) == "{}")
    }

    @Test func somethingElseIsRefused() throws {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try Data("not gzip".utf8).write(to: file)
        defer { try? FileManager.default.removeItem(at: file) }
        #expect(throws: TarGzExtractor.Failure.self) { try TarGzExtractor.extract(file, into: file.appendingPathExtension("out")) }
    }
}
#endif
