/*
 Vergence - the upscaling network's weights for `convert --upscale`

 Real-ESRGAN's compact general network, `realesr-general-x4v3.pth`, as its
 authors publish it on GitHub (xinntao/Real-ESRGAN, BSD-3-Clause, 4.9 MB),
 checked against its SHA-256. Vergence ships no weights: `--download-model`
 fetches it into ~/Library/Application Support/Vergence/UpscaleModels, the
 folder `vergence-convert` uses on a Mac too; `--upscale-model PATH` or
 VERGENCE_UPSCALE_MODEL names another compact Real-ESRGAN `.pth`
 (realesr-animevideov3 is about twice as fast and smoother).
 */

#if canImport(CoreML) && os(macOS)
import CryptoKit
import Foundation
import VergenceApple
import VergenceCommandLine

struct UpscaleWeights {
    let name: String
    let url: URL
    let sha256: String
    let bytes: Int64
    let licence: String

    static let general = UpscaleWeights(
        name: "realesr-general-x4v3",
        url: URL(string: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth")!,
        sha256: "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292",
        bytes: 4_885_111, licence: "BSD-3-Clause")

    static var folder: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Vergence/UpscaleModels", isDirectory: true)
    }

    var installed: URL { Self.folder.appendingPathComponent(name + ".pth") }
}

/// --upscale-model, VERGENCE_UPSCALE_MODEL, else the installed general
/// network (downloaded first with --download-model).
func upscaleNetwork(_ o: Options) async throws -> UpscaleNetwork {
    if let path = o.upscaleModel ?? ProcessInfo.processInfo.environment["VERGENCE_UPSCALE_MODEL"] {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        guard FileManager.default.fileExists(atPath: url.path) else { throw FormatError("no upscaling model at \(path)") }
        return try UpscaleNetwork(checkpoint: url)
    }
    let weights = UpscaleWeights.general
    if FileManager.default.fileExists(atPath: weights.installed.path) { return try UpscaleNetwork(checkpoint: weights.installed) }
    guard o.downloadModel else {
        throw FormatError("no upscaling model: pass --upscale-model PATH (Real-ESRGAN's \(weights.name).pth), set "
            + "VERGENCE_UPSCALE_MODEL, or add --download-model to fetch it from github.com/xinntao/Real-ESRGAN "
            + "(\(weights.licence), about 5 MB)")
    }
    log("vergence: downloading \(weights.name) from github.com/xinntao/Real-ESRGAN (\(weights.licence))")
    let downloaded = try await ModelFileDownloader(expectedBytes: weights.bytes, onFraction: { _ in }).download(from: weights.url)
    defer { try? FileManager.default.removeItem(at: downloaded) }
    let digest = SHA256.hash(data: try Data(contentsOf: downloaded, options: .mappedIfSafe)).map { String(format: "%02x", $0) }.joined()
    guard digest == weights.sha256 else {
        throw FormatError("the downloaded \(weights.name).pth isn't the published one (SHA-256 \(digest))")
    }
    try FileManager.default.createDirectory(at: UpscaleWeights.folder, withIntermediateDirectories: true)
    try? FileManager.default.removeItem(at: weights.installed)
    try FileManager.default.moveItem(at: downloaded, to: weights.installed)
    return try UpscaleNetwork(checkpoint: weights.installed)
}
#endif
