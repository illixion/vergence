/*
 Vergence - offline depth for a file

 The pre-processed AI 3D path (`DepthConverter`: inference, joint-bilateral
 refinement on the frame's own edges, the cut-aware ±2-frame window, Stable
 Depth anchoring) run for one local file into a folder the caller chooses,
 with a model the caller names, rather than into the app's depth cache with
 its preferred model. `vergence convert` warps each eye from the result and
 writes a side-by-side file; `writeAnalysisTrack` turns the same depth into
 analysis-track `depth` records, so the depth made for one converted copy can
 serve a live player too.
 */

#if canImport(Metal) && canImport(AVFoundation)
import AVFoundation
import CoreMedia
import CoreVideo
import Foundation
import VergenceCore

public enum OfflineDepthConversion {
    public enum Failure: Error, LocalizedError {
        case noModel(URL)
        case incomplete
        case unreadable(String)

        public var errorDescription: String? {
            switch self {
            case .noModel(let url): return "can't load the depth model at \(url.path)"
            case .incomplete: return "the depth pass finished without a complete entry"
            case .unreadable(let detail): return "can't read the depth: \(detail)"
            }
        }
    }

    /// Converts `video` (a local file AVFoundation reads) into a depth entry in
    /// `directory`, replacing anything there. `progress` is 0…1.
    public static func convert(video: URL, into directory: URL, model: URL, timeRange: CMTimeRange? = nil,
                               progress: @escaping @Sendable (Double) -> Void = { _ in }) async throws -> DepthCacheStore.Entry {
        let request = DepthConverter.Request(
            videoIdentity: video.path, title: nil, localFileURL: video,
            directory: directory, modelURL: model, timeRange: timeRange)
        let output = try await DepthConverter().convert(request: request) { fraction, _, _ in progress(fraction) }
        guard output.meta.completed else { throw Failure.incomplete }
        return DepthCacheStore.Entry(directory: output.directory, meta: output.meta)
    }

    /// The model's base name and whether it loads, without converting anything.
    public static func canLoad(model: URL) -> Bool {
        guard let device = VergenceMetal.shared?.device else { return false }
        return CoreMLDepthProvider(device: device, modelURL: model) != nil
    }

    /// The depth in `entry` as analysis-track `depth` records (mono, one per
    /// frame), written to `url` after a header. Each record covers the
    /// picture only (the model's letterbox bars cut away) at the model's own
    /// resolution, as bytes whose scale and offset give the depth the warp
    /// reads: the frame's shot-space inverse depth, near high, before
    /// `DepthBudget.studio`'s soft knee. Returns how many records it wrote.
    @discardableResult
    public static func writeAnalysisTrack(_ entry: DepthCacheStore.Entry, to url: URL, mediaID: String,
                                          producer: String, sourceWidth: Int, sourceHeight: Int) async throws -> Int {
        let meta = entry.meta
        var files: [(AVURLAsset, AVAssetTrack)] = []
        for file in [entry.depthVideoURL, entry.secondaryDepthVideoURL] where FileManager.default.fileExists(atPath: file.path) {
            let asset = AVURLAsset(url: file)
            guard let track = try await asset.loadTracks(withMediaType: .video).first else {
                throw Failure.unreadable("no depth track in \(file.lastPathComponent)")
            }
            files.append((asset, track))
        }
        let header = AnalysisTrackHeader(
            mediaID: mediaID, producer: producer,
            models: ["depth": "\(meta.modelName) (Vergence offline depth v\(meta.version), shot space before the depth budget)"],
            sourceWidth: sourceWidth, sourceHeight: sourceHeight)
        let partial = url.deletingPathExtension().appendingPathExtension("partial.vgat")
        FileManager.default.createFile(atPath: partial.path, contents: nil)
        let handle = try FileHandle(forWritingTo: partial)
        var count = 0
        do {
            try handle.write(contentsOf: AnalysisTrack.header(header))
            for (asset, track) in files {
                try forEachDepthFrame(asset, track) { pts, plane in
                    guard let record = try depthRecord(plane, at: pts, meta: meta) else { return }
                    try handle.write(contentsOf: AnalysisTrack.record(record.deflated()))
                    count += 1
                }
            }
            try handle.close()
        } catch {
            try? handle.close()
            try? FileManager.default.removeItem(at: partial)
            throw error
        }
        try? FileManager.default.removeItem(at: url)
        try FileManager.default.moveItem(at: partial, to: url)
        return count
    }

    /// One depth frame's luma, valid inside the callback.
    struct LumaPlane {
        let base: UnsafePointer<UInt8>
        let width: Int
        let height: Int
        let stride: Int
    }

    private static func forEachDepthFrame(_ asset: AVURLAsset, _ track: AVAssetTrack,
                                          _ body: (CMTime, LumaPlane) throws -> Void) throws {
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
        ])
        output.alwaysCopiesSampleData = false
        reader.add(output)
        guard reader.startReading() else { throw Failure.unreadable(reader.error?.localizedDescription ?? "reader") }
        while let sample = output.copyNextSampleBuffer() {
            guard let buffer = CMSampleBufferGetImageBuffer(sample) else { continue }
            CVPixelBufferLockBaseAddress(buffer, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
            guard let base = CVPixelBufferGetBaseAddressOfPlane(buffer, 0) else { continue }
            try body(CMSampleBufferGetPresentationTimeStamp(sample), LumaPlane(
                base: UnsafePointer(base.assumingMemoryBound(to: UInt8.self)),
                width: CVPixelBufferGetWidthOfPlane(buffer, 0), height: CVPixelBufferGetHeightOfPlane(buffer, 0),
                stride: CVPixelBufferGetBytesPerRowOfPlane(buffer, 0)))
        }
        if reader.status == .failed { throw Failure.unreadable(reader.error?.localizedDescription ?? "reader") }
    }

    /// The picture's part of a stored depth frame, halved back to the model's
    /// resolution (the cache stores it at twice that), as an analysis record.
    static func depthRecord(_ plane: LumaPlane, at pts: CMTime, meta: DepthCacheStore.Meta) throws -> AnalysisRecord? {
        guard !meta.framePTS.isEmpty else { return nil }
        let index = nearest(meta.framePTS, to: pts.seconds)
        let scale = index < meta.displayScale.count ? meta.displayScale[index] : 1
        let bias = index < meta.displayBias.count ? meta.displayBias[index] : 0
        // The content rectangle in the stored frame (depth UV = frame UV × scale + offset).
        let x0 = Int((meta.uvOffsetX * Float(plane.width)).rounded())
        let y0 = Int((meta.uvOffsetY * Float(plane.height)).rounded())
        let w = min(plane.width - x0, Int((meta.uvScaleX * Float(plane.width)).rounded()))
        let h = min(plane.height - y0, Int((meta.uvScaleY * Float(plane.height)).rounded()))
        let outWidth = w / 2, outHeight = h / 2
        guard x0 >= 0, y0 >= 0, outWidth > 0, outHeight > 0 else { return nil }
        var payload = Data(count: outWidth * outHeight)
        payload.withUnsafeMutableBytes { raw in
            let out = raw.bindMemory(to: UInt8.self)
            for y in 0 ..< outHeight {
                let row0 = plane.base + (y0 + 2 * y) * plane.stride + x0
                let row1 = row0 + plane.stride
                for x in 0 ..< outWidth {
                    let sum = Int(row0[2 * x]) + Int(row0[2 * x + 1]) + Int(row1[2 * x]) + Int(row1[2 * x + 1])
                    out[y * outWidth + x] = UInt8((sum + 2) / 4)
                }
            }
        }
        // value = byte / 255 × displayScale + displayBias.
        return AnalysisRecord(kind: .depth, encoding: .uint8, channels: 1, width: outWidth, height: outHeight,
                              time: AnalysisTime(value: pts.value, timescale: pts.timescale),
                              scale: scale / 255, offset: bias, payload: payload)
    }

    private static func nearest(_ sorted: [Double], to value: Double) -> Int {
        var lo = 0, hi = sorted.count - 1
        while lo < hi {
            let mid = (lo + hi) / 2
            if sorted[mid] < value { lo = mid + 1 } else { hi = mid }
        }
        if lo > 0, abs(sorted[lo - 1] - value) < abs(sorted[lo] - value) { return lo - 1 }
        return lo
    }
}

extension DepthCacheStore {
    /// The completed entry in `directory` (anywhere, not only the cache
    /// store), made by this pipeline version; nil otherwise.
    public static func entry(at directory: URL) -> Entry? {
        guard let meta = readMeta(in: directory), meta.completed, meta.version == pipelineVersion,
              FileManager.default.fileExists(atPath: directory.appendingPathComponent(depthVideoFilename).path)
        else { return nil }
        return Entry(directory: directory, meta: meta)
    }
}
#endif
