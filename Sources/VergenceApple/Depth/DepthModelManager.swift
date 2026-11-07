/*
 Spatial Stash - Depth Model Manager

 In-app download + management of the Core ML monocular-depth models that power
 higher-quality fake-3D video. Downloads straight from Apple's canonical
 Hugging Face repo (apple/coreml-depth-anything-v2-small) — nothing is re-hosted.

 Each model is a `.mlpackage` (a directory of exactly three files with a fixed
 layout), so we fetch those files via HF's `resolve/main/…` URLs, rebuild the
 package in the managed store (DepthModelStore), and let Core ML compile it on
 first use. File sizes and SHA-256 checksums come from HF's tree API, so a
 truncated/corrupt download fails loudly rather than installing a broken model.

 Without any model the warp falls back to a lightweight heuristic, so this is
 purely an optional quality upgrade. A newly-installed model takes effect the
 next time a fake-3D video is opened (the running StereoPump keeps its provider).
 */

#if canImport(Metal)
import CryptoKit
import Foundation
import os
import VergenceCore

@MainActor
@Observable
public final class DepthModelManager {
    public static let shared = DepthModelManager()

    /// A downloadable model offered in Settings and the video ViewMode menu. The
    /// F32 variants are intentionally omitted — negligible on-device quality gain
    /// for ~2× the bytes.
    public struct Variant: Identifiable, Hashable, Sendable {
        /// Package base name — also the installed name / picker tag,
        /// e.g. `DepthAnythingV2SmallF16`.
        public let name: String
        public let displayName: String
        public let subtitle: String
        public let approxBytes: Int64
        /// Where a model other than Apple's Small comes from: a gzipped tar of
        /// `<name>.mlpackage` in a Hugging Face repo. Nil: Apple's repo.
        public let archive: Archive?
        /// The weights' licence, shown before downloading.
        public let licence: Licence?
        public var id: String { name }

        public init(name: String, displayName: String, subtitle: String, approxBytes: Int64,
                    archive: Archive? = nil, licence: Licence? = nil) {
            self.name = name
            self.displayName = displayName
            self.subtitle = subtitle
            self.approxBytes = approxBytes
            self.archive = archive
            self.licence = licence
        }
    }

    /// A Hugging Face repo and the `.tar.gz` in it that holds the package.
    public struct Archive: Hashable, Sendable {
        public let repo: String
        public let file: String
        public init(repo: String, file: String) {
            self.repo = repo
            self.file = file
        }
    }

    public struct Licence: Hashable, Sendable {
        /// "CC-BY-NC-4.0".
        public let name: String
        public let url: URL
        /// Use is limited to non-commercial purposes.
        public let nonCommercial: Bool
        /// Who to credit, as the licence asks.
        public let attribution: String
        public init(name: String, url: URL, nonCommercial: Bool, attribution: String) {
            self.name = name
            self.url = url
            self.nonCommercial = nonCommercial
            self.attribution = attribution
        }
    }

    /// Models a host app offers besides `variants`, set once at launch. Only a
    /// build that is allowed to offer them sets any: a model whose licence
    /// restricts its use is named only by the app that chose to offer it.
    public static var additionalVariants: [Variant] = []

    /// Every model on offer.
    public static var allVariants: [Variant] { variants + additionalVariants }

    public static let variants: [Variant] = [
        Variant(name: "DepthAnythingV2SmallF16",
                displayName: "DepthAnything V2 (F16)",
                subtitle: "Best quality · recommended",
                approxBytes: 49_820_000),
        Variant(name: "DepthAnythingV2SmallF16INT8",
                displayName: "DepthAnything V2 (INT8)",
                subtitle: "Quantized · smaller, a touch faster",
                approxBytes: 25_400_000),
        Variant(name: "DepthAnythingV2SmallF16P6",
                displayName: "DepthAnything V2 (6-bit)",
                subtitle: "Palettized · smallest",
                approxBytes: 19_050_000),
        Variant(name: "DepthAnythingV2SmallF16P8",
                displayName: "DepthAnything V2 (8-bit)",
                subtitle: "Palettized · small",
                approxBytes: 25_260_000)
    ]

    private static let repoBase = "https://huggingface.co/apple/coreml-depth-anything-v2-small"
    private static let treeAPI = "https://huggingface.co/api/models/apple/coreml-depth-anything-v2-small/tree/main?recursive=true"
    /// The three files inside every `…Small….mlpackage`, in a fixed layout.
    private static let packageFiles = [
        "Manifest.json",
        "Data/com.apple.CoreML/model.mlmodel",
        "Data/com.apple.CoreML/weights/weight.bin"
    ]

    /// Base names of installed models (managed store). Observable.
    public private(set) var installedNames: [String] = []
    /// Download progress (0…1) keyed by variant name; a key is present only while
    /// that variant is downloading. Observable.
    public private(set) var progress: [String: Double] = [:]
    /// Last download error, surfaced in Settings. Observable.
    public private(set) var errorMessage: String?

    private init() {
        importInboxIfNeeded()
    }

    // MARK: State

    /// Drain the Documents inbox into the managed store, then refresh the list.
    public func importInboxIfNeeded() {
        DepthModelStore.importInboxModels()
        refresh()
    }

    func refresh() {
        installedNames = DepthModelStore.installedModelNames()
    }

    public func isInstalled(_ variant: Variant) -> Bool { installedNames.contains(variant.name) }
    public func isDownloading(_ variant: Variant) -> Bool { progress[variant.name] != nil }
    var isAnyDownloading: Bool { !progress.isEmpty }

    public func delete(_ name: String) {
        DepthModelStore.deleteModel(named: name)
        refresh()
    }

    /// Friendly label for an arbitrary installed model name (falls back to raw).
    public static func displayName(for name: String) -> String {
        allVariants.first { $0.name == name }?.displayName ?? name
    }

    // MARK: Download

    public func download(_ variant: Variant) async {
        guard progress[variant.name] == nil else { return }
        errorMessage = nil
        progress[variant.name] = 0
        do {
            try await performDownload(variant)
            refresh()
            VergenceLog.pipeline.info("Depth model installed: \(variant.name, privacy: .public)")
        } catch {
            errorMessage = "Couldn't download \(variant.displayName): \(error.localizedDescription)"
            VergenceLog.pipeline.error("Depth model download failed (\(variant.name, privacy: .public)): \(error.localizedDescription, privacy: .public)")
        }
        progress[variant.name] = nil
    }

    private func setProgress(_ name: String, _ value: Double) {
        // Only update while still downloading (guards against late callbacks).
        guard progress[name] != nil else { return }
        progress[name] = value
    }

    private func performDownload(_ variant: Variant) async throws {
        if let archive = variant.archive {
            try await performArchiveDownload(variant, archive)
            return
        }
        let fm = FileManager.default
        let pkg = "\(variant.name).mlpackage"

        // Best-effort metadata (per-file size + SHA-256) from HF's tree API.
        var sizes: [String: Int64] = [:]
        var hashes: [String: String] = [:]
        if let entries = try? await fetchTree() {
            for e in entries where e.type == "file" && e.path.hasPrefix(pkg + "/") {
                let rel = String(e.path.dropFirst(pkg.count + 1))
                sizes[rel] = e.lfs?.size ?? e.size
                if let oid = e.lfs?.oid { hashes[rel] = oid }
            }
        }
        let totalBytes = Self.packageFiles.reduce(Int64(0)) { $0 + (sizes[$1] ?? 0) }
        let effectiveTotal = totalBytes > 0 ? totalBytes : variant.approxBytes

        // Assemble into a temp package dir, then move into the store atomically.
        let tmpRoot = fm.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        let tmpPkg = tmpRoot.appendingPathComponent(pkg, isDirectory: true)
        try fm.createDirectory(at: tmpPkg, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: tmpRoot) }

        var completedBytes: Int64 = 0
        for rel in Self.packageFiles {
            guard let fileURL = URL(string: "\(Self.repoBase)/resolve/main/\(pkg)/\(rel)") else {
                throw DownloadError.badURL
            }
            // weight.bin dominates; if sizes are unknown, drive the bar off it.
            let expected = sizes[rel] ?? (rel.hasSuffix("weight.bin") ? effectiveTotal : 0)
            let base = completedBytes
            let denom = Double(effectiveTotal)
            let downloader = ModelFileDownloader(expectedBytes: expected) { [weak self] fraction in
                let done = Double(base) + fraction * Double(max(expected, 0))
                let overall = denom > 0 ? min(1.0, done / denom) : 0
                Task { @MainActor in self?.setProgress(variant.name, overall) }
            }

            let tempFile = try await downloader.download(from: fileURL)
            if let want = hashes[rel] {
                guard let got = Self.sha256Hex(of: tempFile), got == want.lowercased() else {
                    try? fm.removeItem(at: tempFile)
                    throw DownloadError.checksumMismatch(rel)
                }
            }
            let dest = tmpPkg.appendingPathComponent(rel)
            try fm.createDirectory(at: dest.deletingLastPathComponent(), withIntermediateDirectories: true)
            try? fm.removeItem(at: dest)
            try fm.moveItem(at: tempFile, to: dest)
            completedBytes += (sizes[rel] ?? 0)
        }

        let finalURL = DepthModelStore.modelsDirectory.appendingPathComponent(pkg)
        try? fm.removeItem(at: finalURL)
        try fm.moveItem(at: tmpPkg, to: finalURL)
    }

    /// A packaged model as a `.tar.gz`: downloaded (checked against the
    /// repo's SHA-256 when it says one), unpacked in a temporary folder, then
    /// moved into the store.
    private func performArchiveDownload(_ variant: Variant, _ archive: Archive) async throws {
        let fm = FileManager.default
        var expected: Int64 = variant.approxBytes
        var hash: String?
        if let entries = try? await fetchTree(repo: archive.repo),
           let entry = entries.first(where: { $0.type == "file" && $0.path == archive.file }) {
            expected = entry.lfs?.size ?? entry.size ?? expected
            hash = entry.lfs?.oid
        }
        guard let url = URL(string: "https://huggingface.co/\(archive.repo)/resolve/main/\(archive.file)") else {
            throw DownloadError.badURL
        }
        let downloader = ModelFileDownloader(expectedBytes: expected) { [weak self] fraction in
            // The last tenth is the unpacking.
            Task { @MainActor in self?.setProgress(variant.name, fraction * 0.9) }
        }
        let file = try await downloader.download(from: url)
        defer { try? fm.removeItem(at: file) }
        if let hash {
            guard let got = Self.sha256Hex(of: file), got == hash.lowercased() else {
                throw DownloadError.checksumMismatch(archive.file)
            }
        }
        let tmpRoot = fm.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? fm.removeItem(at: tmpRoot) }
        try await Task.detached(priority: .utility) { try TarGzExtractor.extract(file, into: tmpRoot) }.value
        let package = tmpRoot.appendingPathComponent("\(variant.name).mlpackage")
        guard fm.fileExists(atPath: package.appendingPathComponent("Manifest.json").path) else {
            throw DownloadError.missingPackage(variant.name)
        }
        let finalURL = DepthModelStore.modelsDirectory.appendingPathComponent("\(variant.name).mlpackage")
        try? fm.removeItem(at: finalURL)
        try fm.moveItem(at: package, to: finalURL)
    }

    // MARK: HF tree metadata

    private struct TreeEntry: Decodable {
        let type: String
        let path: String
        let size: Int64?
        let lfs: LFS?
        struct LFS: Decodable { let oid: String; let size: Int64? }
    }

    private func fetchTree(repo: String? = nil) async throws -> [TreeEntry] {
        let api = repo.map { "https://huggingface.co/api/models/\($0)/tree/main?recursive=true" } ?? Self.treeAPI
        guard let url = URL(string: api) else { throw DownloadError.badURL }
        let (data, _) = try await URLSession.shared.data(from: url)
        return try JSONDecoder().decode([TreeEntry].self, from: data)
    }

    // MARK: Helpers

    private static func sha256Hex(of url: URL) -> String? {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        var hasher = SHA256()
        while true {
            let chunk = autoreleasepool { handle.readData(ofLength: 1 << 20) }
            if chunk.isEmpty { break }
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    enum DownloadError: LocalizedError {
        case badURL
        case checksumMismatch(String)
        case missingPackage(String)

        var errorDescription: String? {
            switch self {
            case .badURL: return "Invalid download URL."
            case .checksumMismatch(let file): return "Checksum mismatch (\(file))."
            case .missingPackage(let name): return "The archive has no \(name).mlpackage."
            }
        }
    }
}

#endif
