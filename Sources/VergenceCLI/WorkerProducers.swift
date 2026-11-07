/*
 vergence worker: what each workload makes of a run of decoded frames

 A piece is decoded once by ffmpeg (`PieceDecoder`), from the media server
 through the worker's loopback forwarder, or from a local file for a
 reference run. Each workload's `PieceProducer` sees the frames in order and
 returns the records for the frames inside the piece; frames decoded around
 it are context only (the next frame a flow pair reaches, a lookahead).
 Times are frame index / the title's frame rate, the coordinator's grid.
 */

#if canImport(CoreML) && canImport(Network)
import CoreML
import CoreVideo
import Foundation
import VergenceApple
import VergenceCore

/// One decoded frame: the left eye (or the whole picture), and the right
/// eye of a stereo title. Borrowed planes, valid only during the call.
struct DecodedFrame {
    var left: [AnalysisPlane]
    var right: [AnalysisPlane]?
}

protocol PieceProducer: AnyObject {
    /// The record kinds it makes, as the coordinator names them.
    var kinds: Set<String> { get }
    /// The model behind each kind, as track headers and claims name them.
    var models: [String: String] { get }
    /// Stereo titles only (both eyes); otherwise flat titles only.
    var needsStereo: Bool { get }
    /// Frames decoded before and after the piece, for context.
    var framesBefore: Int64 { get }
    var framesAfter: Int64 { get }
    /// A new run starts. `continuing`: it carries on from the producer's own
    /// state (the frame it asked to resume from), not from context.
    func begin(continuing: Bool)
    /// Sticky runs: when the producer holds this title's state from its last
    /// piece and `piece` follows it, the frame to start decoding at instead of
    /// the piece's context (nil: decode the context).
    func resumeFrame(mediaID: String, piece: Range<Int64>) -> Int64?
    /// Whether a claim may ask to carry on into the next piece (the
    /// producer keeps state worth keeping).
    var keepsRuns: Bool { get }
    /// The next frame, `index` on the title's grid. Returns records for frames in `piece` only.
    func frame(_ frame: DecodedFrame, index: Int64, time: AnalysisTime, piece: Range<Int64>) throws -> [AnalysisRecord]
    /// Records held back for context, once every frame was seen.
    func finish(piece: Range<Int64>) throws -> [AnalysisRecord]
}

extension PieceProducer {
    func begin(continuing: Bool) {}
    func resumeFrame(mediaID: String, piece: Range<Int64>) -> Int64? { nil }
    var keepsRuns: Bool { false }
    func finish(piece: Range<Int64>) throws -> [AnalysisRecord] { [] }
}

/// A workload that decodes a piece its own way (to a file a pipeline reads
/// whole, such as the offline depth converter) instead of frame by frame.
protocol PieceFileProducer: PieceProducer {
    func track(of input: PieceInput, frames piece: Range<Int64>, frameCount: Int64, ffmpeg: String?,
               mediaID: String, producerName: String) async throws -> Data
}

extension PieceFileProducer {
    var framesBefore: Int64 { 0 }
    var framesAfter: Int64 { 0 }
    func frame(_ frame: DecodedFrame, index: Int64, time: AnalysisTime, piece: Range<Int64>) throws -> [AnalysisRecord] {
        throw WorkerFailure("\(Self.self) decodes its pieces itself")
    }
}

/// What a decode reads: a URL or path, and the title as the coordinator described it.
struct PieceInput {
    var url: String
    var width: Int
    var height: Int
    var codec: String
    /// The coordinator's layout name: mono, halfSideBySide, fullTopAndBottom, ...
    var stereo: String
    var rate: (num: Int, den: Int)

    /// A local file, as ffprobe sees it.
    static func probe(_ path: String, stereo: String) throws -> PieceInput {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                             "stream=codec_name,width,height,r_frame_rate", "-of", "json", path]
        let output = Pipe()
        process.standardOutput = output
        try process.run()
        let data = output.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        struct Probe: Decodable {
            struct Stream: Decodable {
                var codec_name: String
                var width: Int
                var height: Int
                var r_frame_rate: String
            }
            var streams: [Stream]
        }
        guard let stream = try JSONDecoder().decode(Probe.self, from: data).streams.first else {
            throw WorkerFailure("ffprobe found no video in \(path)")
        }
        let rate = stream.r_frame_rate.split(separator: "/").compactMap { Int($0) }
        guard rate.count == 2, rate[0] > 0, rate[1] > 0 else { throw WorkerFailure("no frame rate in \(path)") }
        return PieceInput(url: path, width: stream.width, height: stream.height, codec: stream.codec_name,
                          stereo: stereo, rate: (rate[0], rate[1]))
    }
}

struct PieceDecoder {
    /// ffmpeg's path; nil finds it on PATH.
    var ffmpeg: String?

    /// Decodes `frames` (with the producer's context around them) and returns
    /// the records as a track. `frameCount` is the title's (`Int64.max` for a
    /// reference run, which decodes to the end).
    func track(of input: PieceInput, frames piece: Range<Int64>, frameCount: Int64, producer: PieceProducer,
               mediaID: String, producerName: String) async throws -> Data {
        if let producer = producer as? PieceFileProducer {
            return try await producer.track(of: input, frames: piece, frameCount: frameCount, ffmpeg: ffmpeg,
                                            mediaID: mediaID, producerName: producerName)
        }
        let w = input.width, h = input.height
        let packing: (sideBySide: Bool, stereo: Bool)
        switch input.stereo {
        case "halfSideBySide", "fullSideBySide": packing = (true, true)
        case "halfTopAndBottom", "fullTopAndBottom": packing = (false, true)
        case "mono": packing = (false, false)
        default: throw WorkerFailure("the title's layout (\(input.stereo)) isn't one a worker reads")
        }
        guard packing.stereo == producer.needsStereo else {
            throw WorkerFailure(producer.needsStereo ? "the title isn't frame-packed stereo" : "the title is stereo; this workload is for 2D")
        }
        guard w > 0, h > 0, w % 4 == 0, h % 4 == 0 else { throw WorkerFailure("the picture's size (\(w)x\(h)) is unusable") }
        let eyeW = packing.stereo && packing.sideBySide ? w / 2 : w
        let eyeH = packing.stereo && !packing.sideBySide ? h / 2 : h

        if let producer = producer as? GainGridProducer {
            producer.prepare(frameRate: Double(input.rate.num) / Double(input.rate.den))
            producer.timeScaleNum = Int32(input.rate.num)
            producer.timeScaleDen = Int64(input.rate.den)
        }
        let resume = producer.resumeFrame(mediaID: mediaID, piece: piece)
        let first = resume ?? max(0, piece.lowerBound - producer.framesBefore)
        let (after, overflow) = piece.upperBound.addingReportingOverflow(producer.framesAfter)
        let last = overflow ? frameCount : min(frameCount, after)
        let wanted = last - first
        // Half a frame early, so the first frame out is exactly `first`.
        let seek = (Double(first) - 0.5) * Double(input.rate.den) / Double(input.rate.num)
        var args = ["-nostdin", "-v", "error"]
        // A frame-packing SEI makes ffmpeg 8 and 9 fail filter graphs; drop SEI before decoding.
        if input.codec.lowercased() == "h264" { args += ["-bsf:v", "filter_units=remove_types=6"] }
        if seek > 0 { args += ["-ss", String(format: "%.6f", seek)] }
        // Passthrough: rawvideo's default constant-rate sync would repeat the first frame after a seek.
        args += ["-i", input.url, "-map", "0:v:0", "-fps_mode", "passthrough"]
        if wanted < Int64(Int32.max) { args += ["-frames:v", String(wanted)] }
        args += ["-f", "rawvideo", "-pix_fmt", "yuv420p", "-"]
        let process = Process()
        process.executableURL = URL(fileURLWithPath: ffmpeg ?? "/usr/bin/env")
        process.arguments = ffmpeg == nil ? ["ffmpeg"] + args : args
        process.qualityOfService = .utility
        let output = Pipe(), errors = Pipe()
        process.standardOutput = output
        process.standardError = errors
        // ffmpeg's messages are read as they come, so a chatty decode can't fill the pipe and stall.
        let messages = MessageSink()
        errors.fileHandleForReading.readabilityHandler = { messages.append($0.availableData) }
        try process.run()
        defer { if process.isRunning { process.terminate() } }

        var header = AnalysisTrackHeader(mediaID: mediaID, producer: producerName, models: producer.models,
                                         sourceWidth: eyeW, sourceHeight: eyeH)
        // The records; the header goes in front once it is known whether the video ended early.
        var track = Data()
        func append(_ records: [AnalysisRecord]) throws {
            for record in records { track.append(AnalysisTrack.record(try record.deflated())) }
        }
        producer.begin(continuing: resume != nil)
        let frameBytes = w * h * 3 / 2
        let reader = output.fileHandleForReading
        var index = first
        while index < last {
            var data = Data()
            while data.count < frameBytes {
                guard let chunk = try reader.read(upToCount: frameBytes - data.count), !chunk.isEmpty else { break }
                data.append(chunk)
            }
            guard data.count == frameBytes else { break }
            let time = AnalysisTime(value: index * Int64(input.rate.den), timescale: Int32(input.rate.num))
            try autoreleasepool {
                try data.withUnsafeBytes { raw in
                    let base = raw.bindMemory(to: UInt8.self).baseAddress!
                    let frame = packing.stereo ? Self.eyes(base: base, width: w, height: h, sideBySide: packing.sideBySide)
                        : DecodedFrame(left: Self.whole(base: base, width: w, height: h), right: nil)
                    try append(producer.frame(frame, index: index, time: time, piece: piece))
                }
            }
            index += 1
        }
        process.waitUntilExit()
        errors.fileHandleForReading.readabilityHandler = nil
        // A resumed run may start inside the piece: its earlier frames were seen in the last one.
        let decodedPiece = min(index, piece.upperBound) - max(piece.lowerBound, first)
        let pieceToDecode = piece.upperBound - max(piece.lowerBound, first)
        let reference = frameCount == Int64.max
        // ffmpeg finished cleanly short of the piece: the video stream ends here (the coordinator
        // takes that only near the title's end, where a container's audio often outlasts its picture).
        let endedEarly = !reference && decodedPiece < pieceToDecode && process.terminationStatus == 0 && index > piece.lowerBound
        guard reference ? decodedPiece > 0 : decodedPiece >= pieceToDecode || endedEarly else {
            throw WorkerFailure("ffmpeg gave \(index - first) of \(wanted) frames (\(process.terminationStatus)): \(messages.text.prefix(300))")
        }
        if endedEarly { header.videoEndFrame = index }
        try append(producer.finish(piece: piece.lowerBound ..< min(piece.upperBound, index)))
        return try AnalysisTrack.header(header) + track
    }

    static func whole(base: UnsafePointer<UInt8>, width w: Int, height h: Int) -> [AnalysisPlane] {
        let cb = base + w * h, cr = cb + (w / 2) * (h / 2)
        return [AnalysisPlane(base: base, width: w, height: h, stride: w),
                AnalysisPlane(base: cb, width: w / 2, height: h / 2, stride: w / 2),
                AnalysisPlane(base: cr, width: w / 2, height: h / 2, stride: w / 2)]
    }

    /// The two eyes of a packed I420 picture, read in place.
    static func eyes(base: UnsafePointer<UInt8>, width w: Int, height h: Int, sideBySide: Bool) -> DecodedFrame {
        let cb = base + w * h, cr = cb + (w / 2) * (h / 2)
        if sideBySide {
            func eye(_ e: Int) -> [AnalysisPlane] {
                [AnalysisPlane(base: base + e * w / 2, width: w / 2, height: h, stride: w),
                 AnalysisPlane(base: cb + e * w / 4, width: w / 4, height: h / 2, stride: w / 2),
                 AnalysisPlane(base: cr + e * w / 4, width: w / 4, height: h / 2, stride: w / 2)]
            }
            return DecodedFrame(left: eye(0), right: eye(1))
        }
        func eye(_ e: Int) -> [AnalysisPlane] {
            [AnalysisPlane(base: base + e * w * h / 2, width: w, height: h / 2, stride: w),
             AnalysisPlane(base: cb + e * (w / 2) * (h / 4), width: w / 2, height: h / 4, stride: w / 2),
             AnalysisPlane(base: cr + e * (w / 2) * (h / 4), width: w / 2, height: h / 4, stride: w / 2)]
        }
        return DecodedFrame(left: eye(0), right: eye(1))
    }
}

/// ffmpeg's stderr, kept for an error message.
private final class MessageSink: @unchecked Sendable {
    private let lock = NSLock()
    private var data = Data()
    func append(_ more: Data) {
        lock.withLock { if data.count < 65536 { data.append(more) } }
    }
    var text: String { lock.withLock { String(decoding: data, as: UTF8.self) } }
}

// MARK: - Smooth Motion

/// smooth-motion-3d: the left eye's flow between frames and each frame's
/// right→left disparity, exactly as `vergence analyze` makes them.
final class StereoFlowProducer: PieceProducer, @unchecked Sendable {
    static let workload = "smooth-motion-3d"
    let analyzer: StereoFlowAnalyzer
    let kinds: Set<String> = ["flowPair", "disparity"]
    let models: [String: String]
    let needsStereo = true
    let framesBefore: Int64 = 0
    /// The last flow pair reaches the frame after the piece.
    let framesAfter: Int64 = 1
    private var previous: (left: [Float], index: Int64, time: AnalysisTime)?

    init(analyzer: StereoFlowAnalyzer) {
        self.analyzer = analyzer
        let f = SharedFieldAnalysis.flowModelSize, d = SharedFieldAnalysis.disparityModelSize
        // Exactly what `vergence analyze` writes, so tracks from either agree on the profile.
        models = ["flowPair": "\(analyzer.flowModelName) \(f.width)x\(f.height)",
                  "disparity": "\(analyzer.disparityModelName) \(d.width)x\(d.height)"]
    }

    func begin(continuing: Bool) { previous = nil }

    func frame(_ frame: DecodedFrame, index: Int64, time: AnalysisTime, piece: Range<Int64>) throws -> [AnalysisRecord] {
        guard let right = frame.right else { throw WorkerFailure("a 3D frame without a right eye") }
        let left = frame.left
        let f = SharedFieldAnalysis.flowModelSize, d = SharedFieldAnalysis.disparityModelSize
        let leftFlow = SharedFieldAnalysis.rgbPlanar(y: left[0], cb: left[1], cr: left[2], width: f.width, height: f.height)
        let pairStart = previous.flatMap { piece.contains($0.index) ? $0 : nil }
        let inPiece = piece.contains(index)
        let runs = PieceRuns()
        // The two runs overlap: RIFE usually lands on the Neural Engine, the fp32 NeuFlow on the GPU.
        DispatchQueue.concurrentPerform(iterations: 2) { job in
            do {
                if job == 0, let pairStart {
                    runs.flow = try analyzer.flow(from: pairStart.left, to: leftFlow)
                } else if job == 1, inPiece {
                    let l = SharedFieldAnalysis.rgbPlanar(y: left[0], cb: left[1], cr: left[2], width: d.width, height: d.height)
                    let r = SharedFieldAnalysis.rgbPlanar(y: right[0], cb: right[1], cr: right[2], width: d.width, height: d.height)
                    runs.disparity = try analyzer.disparity(left: l, right: r)
                }
            } catch {
                runs.error = error
            }
        }
        if let error = runs.error { throw error }
        var records: [AnalysisRecord] = []
        if let pairStart, let (flow, mask) = runs.flow {
            records.append(SharedFieldAnalysis.flowPairRecord(flow: flow, mask: mask, width: f.width, height: f.height,
                                                              time: pairStart.time, endTime: time))
        }
        if let disparity = runs.disparity {
            records.append(SharedFieldAnalysis.disparityRecord(rightToLeft: disparity, width: d.width, height: d.height,
                                                               time: time))
        }
        previous = (leftFlow, index, time)
        return records
    }
}

/// What one frame's two concurrent model runs produced.
private final class PieceRuns: @unchecked Sendable {
    var flow: (flow: [Float], mask: [Float])?
    var disparity: [Float]?
    var error: Error?
}

/// smooth-motion-2d: the whole picture's flow between frames, for 2D titles
/// and AI 3D, at RIFE 960×576 (`MonoFlowAnalyzer`), stored at 240×144.
final class MonoFlowProducer: PieceProducer {
    static let workload = "smooth-motion-2d"
    static let modelSize = "960x576"
    let analyzer: MonoFlowAnalyzer
    let kinds: Set<String> = ["flowPair"]
    let models: [String: String]
    let needsStereo = false
    let framesBefore: Int64 = 0
    let framesAfter: Int64 = 1
    private var previous: (picture: MLMultiArray, index: Int64, time: AnalysisTime)?

    init(analyzer: MonoFlowAnalyzer) {
        self.analyzer = analyzer
        models = ["flowPair": analyzer.profileName]
    }

    func begin(continuing: Bool) { previous = nil }

    func frame(_ frame: DecodedFrame, index: Int64, time: AnalysisTime, piece: Range<Int64>) throws -> [AnalysisRecord] {
        let p = frame.left
        let picture = try analyzer.input(SharedFieldAnalysis.rgbPlanar(y: p[0], cb: p[1], cr: p[2],
                                                                       width: analyzer.width, height: analyzer.height))
        defer { previous = (picture, index, time) }
        guard let previous, piece.contains(previous.index) else { return [] }
        let (flow, mask) = try analyzer.flow(from: previous.picture, to: picture)
        return [SharedFieldAnalysis.flowPairRecord(flow: flow, mask: mask, width: analyzer.width, height: analyzer.height,
                                                   time: previous.time, endTime: time,
                                                   step: SharedFieldAnalysis.monoFlowStep)]
    }
}
// MARK: - AI 3D depth

/// ai-3d-depth: the pre-processed AI 3D depth of a 2D title (Vergence's
/// offline depth pipeline, `OfflineDepthConversion`: inference, the
/// joint-bilateral refinement on the frame's own edges, the cut-aware ±2
/// frame window and Stable Depth's shot-space anchoring) with a larger model
/// than a headset runs live, as `depth` records a player warps from.
///
/// The pipeline reads a whole file, so each piece is decoded to a short
/// intermediate (ProRes, at most 1036 px, the size the converter decodes to)
/// with a second of frames before it, which the anchoring settles over, and
/// the two the temporal window needs after it; only the piece's records go
/// up. Time constants in the anchoring run to a few seconds, so a long shot
/// can still differ slightly from a whole-file conversion around a piece's
/// start (`scripts/check-worker.sh … ai-3d-depth` measures it).
final class DepthProducer: PieceFileProducer {
    static let workload = "ai-3d-depth"
    /// Frames of context before a piece, in seconds of the title.
    static let contextSeconds = 1.0
    let model: URL
    let kinds: Set<String> = ["depth"]
    let models: [String: String]
    let needsStereo = false
    let keepsRuns = true
    /// Frames each piece decodes before its own when carrying on a run: the
    /// temporal window's reach. The anchoring's state comes from the last
    /// piece (`Meta.shotLock`), so its second of context isn't needed.
    static let runOverlap: Int64 = 2
    /// Sticky runs: the title, where its next piece starts, and the locked
    /// shot mapping of the last piece's final frames (the anchoring's state).
    private var run: (title: String, next: Int64, locks: [Int64: DepthShotRange])?
    /// How the last piece started, for the log.
    private(set) var lastPieceCarriedOn = false

    init(model: URL) async throws {
        guard OfflineDepthConversion.canLoad(model: model) else { throw WorkerFailure("can't load the depth model at \(model.path)") }
        self.model = model
        // Model and pipeline version: a title keeps one, and the pipeline's own version changes its records.
        // A model under a restrictive licence says so in the profile too, so its titles stay apart.
        let licence = DepthModelLicence.restriction(forModelAt: model).map { " \($0.name)" } ?? ""
        let name = model.deletingPathExtension().lastPathComponent
        let compiled = try await InterpolationModelBenchmark.compiledURL(for: model)
        let image = try MLModel(contentsOf: compiled).modelDescription.inputDescriptionsByName.values
            .compactMap(\.imageConstraint).first
        models = ["depth": Self.profile(modelName: name, inputSize: image.map { ($0.pixelsWide, $0.pixelsHigh) },
                                        licence: licence)]
    }

    /// The depth profile. Depth Anything V2 Small and Base get one name whatever
    /// runs them (family, input size, licence, pipeline, context), so a Mac and a
    /// PC (`vergence-worker`, ONNX Runtime) can split a title: their records
    /// agree within the tolerance `scripts/check-onnx-workload.sh` measured (Hypnos
    /// `docs/plans/analysis-workers.md`, question 3). Any other model keeps its
    /// file's name, as before. The ONNX worker names them the same way
    /// (`worker/src/workloads.rs`).
    static func profile(modelName: String, inputSize: (Int, Int)?, licence: String) -> String {
        let tail = "\(licence) offline-v\(DepthCacheStore.pipelineVersion) context-1s"
        if let family = sharedDepthFamily(modelName), let (w, h) = inputSize {
            return "\(family) \(w)x\(h)\(tail)"
        }
        return "\(modelName)\(tail)"
    }

    /// The model families whose depth is shared across engines.
    static func sharedDepthFamily(_ modelName: String) -> String? {
        let name = modelName.lowercased().replacingOccurrences(of: "-", with: "").replacingOccurrences(of: "_", with: "")
        guard name.contains("depthanything"), name.contains("v2"), !name.contains("video") else { return nil }
        if name.contains("small") { return "Depth Anything V2 Small" }
        if name.contains("base") { return "Depth Anything V2 Base" }
        return nil
    }

    /// ffmpeg's arguments for a piece's intermediate: `wanted` frames from
    /// frame `first`, at most 1036 px, ProRes, frame k at exactly k / rate.
    static func intermediateArguments(input: PieceInput, first: Int64, wanted: Int64, into intermediate: URL) -> [String] {
        let rate = Double(input.rate.num) / Double(input.rate.den)
        let seek = (Double(first) - 0.5) / rate
        var args = ["-nostdin", "-v", "error"]
        if input.codec.lowercased() == "h264" { args += ["-bsf:v", "filter_units=remove_types=6"] }
        if seek > 0 { args += ["-ss", String(format: "%.6f", seek)] }
        args += ["-i", input.url, "-map", "0:v:0", "-fps_mode", "passthrough"]
        if wanted < Int64(Int32.max) { args += ["-frames:v", String(wanted)] }
        // Frame k of the intermediate is shown at k / rate, exactly: a Matroska source's millisecond
        // timestamps, passed through, came out uneven (two frames 0.08 ms apart), and the depth pass,
        // which reads the file by time, then lost a quarter of a piece's frames.
        let timing = "setpts=N*\(input.rate.den)/\(input.rate.num)/TB"
        args += ["-vf", "\(timing),scale='min(iw,1036)':'min(ih,1036)':force_original_aspect_ratio=decrease:force_divisible_by=2",
                 "-video_track_timescale", "\(input.rate.num)",
                 "-c:v", "prores_ks", "-profile:v", "2", "-pix_fmt", "yuv422p10le", "-an", "-sn", "-dn",
                 "-y", intermediate.path]
        return args
    }

    func track(of input: PieceInput, frames piece: Range<Int64>, frameCount: Int64, ffmpeg: String?,
               mediaID: String, producerName: String) async throws -> Data {
        guard input.stereo == "mono" else { throw WorkerFailure("the title is stereo; AI 3D depth is for 2D") }
        let rate = Double(input.rate.num) / Double(input.rate.den)
        let before = Int64((Self.contextSeconds * rate).rounded())
        // Carrying on a run: the anchoring goes on from the lock the last piece gave this
        // piece's first overlap frame, as one decode of the whole title would have it.
        let previous = run
        run = nil
        let seed = previous.flatMap { run -> DepthShotRange? in
            guard run.title == mediaID, run.next == piece.lowerBound, rate < 48 else { return nil }
            return run.locks[piece.lowerBound - Self.runOverlap]
        }
        lastPieceCarriedOn = seed != nil
        if let previous, seed == nil, previous.title == mediaID {
            log("vergence worker: depth starts piece at \(piece.lowerBound) afresh (run ended at \(previous.next), \(previous.locks.count) locks)")
        }
        let first = seed != nil ? piece.lowerBound - Self.runOverlap : max(0, piece.lowerBound - before)
        let (after, overflow) = piece.upperBound.addingReportingOverflow(2)
        let last = overflow ? frameCount : min(frameCount, after)
        let wanted = last - first
        let work = FileManager.default.temporaryDirectory.appendingPathComponent("vergence-depth-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: work, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: work) }
        let intermediate = work.appendingPathComponent("piece.mov")

        let args = Self.intermediateArguments(input: input, first: first, wanted: wanted, into: intermediate)
        let process = Process()
        process.executableURL = URL(fileURLWithPath: ffmpeg ?? "/usr/bin/env")
        process.arguments = ffmpeg == nil ? ["ffmpeg"] + args : args
        process.qualityOfService = .utility
        let errors = Pipe()
        process.standardError = errors
        process.standardOutput = FileHandle.nullDevice
        let messages = MessageSink()
        errors.fileHandleForReading.readabilityHandler = { messages.append($0.availableData) }
        try process.run()
        process.waitUntilExit()
        errors.fileHandleForReading.readabilityHandler = nil
        guard process.terminationStatus == 0 else {
            throw WorkerFailure("ffmpeg couldn't make the piece (\(process.terminationStatus)): \(messages.text.prefix(300))")
        }

        let entry = try await OfflineDepthConversion.convert(video: intermediate, into: work.appendingPathComponent("depth"),
                                                             model: model, seed: seed, reusesStaticDepth: false)
        let records = try await OfflineDepthConversion.records(entry)
        // The intermediate's frame at k / rate is the title's frame first + k.
        var byIndex: [Int64: AnalysisRecord] = [:]
        for record in records {
            let k = Int64((record.time.seconds * rate).rounded())
            byIndex[first + k] = byIndex[first + k] ?? record
        }
        let inPiece = byIndex.keys.filter { piece.contains($0) }.count
        let reference = frameCount == Int64.max
        let expected = min(piece.upperBound, last) - piece.lowerBound
        // The intermediate (written by an ffmpeg that finished cleanly) ran out inside the piece:
        // the video stream ends there.
        let decodedEnd = (byIndex.keys.max() ?? first - 1) + 1
        let endedEarly = !reference && Int64(inPiece) < expected && decodedEnd > piece.lowerBound
            && Int64(byIndex.count) == decodedEnd - first
        guard reference || Int64(inPiece) >= expected || endedEarly else {
            throw WorkerFailure("the depth pass gave \(inPiece) of the piece's \(expected) frames (\(records.count) of \(wanted) decoded)")
        }
        let header = AnalysisTrackHeader(mediaID: mediaID, producer: producerName, models: models,
                                         sourceWidth: input.width, sourceHeight: input.height,
                                         videoEndFrame: endedEarly ? decodedEnd : nil)
        var track = try AnalysisTrack.header(header)
        for index in byIndex.keys.sorted() {
            guard piece.contains(index), let found = byIndex[index] else { continue }
            var record = found
            record.time = AnalysisTime(value: index * Int64(input.rate.den), timescale: Int32(input.rate.num))
            record.endTime = record.time
            track.append(AnalysisTrack.record(try record.deflated()))
        }
        // Keep the anchoring's state at the end of this piece for the next one.
        if !endedEarly, let lock = entry.meta.shotLock, lock.count == 2 * entry.meta.framePTS.count {
            var locks: [Int64: DepthShotRange] = [:]
            for (k, seconds) in entry.meta.framePTS.enumerated() {
                let index = first + Int64((seconds * rate).rounded())
                if index >= piece.upperBound - Self.runOverlap {
                    locks[index] = DepthShotRange(lo: lock[2 * k], hi: lock[2 * k + 1])
                }
            }
            run = (mediaID, piece.upperBound, locks)
        } else {
            log("vergence worker: no anchoring state kept (shotLock \(entry.meta.shotLock?.count ?? -1), frames \(entry.meta.framePTS.count))")
        }
        return track
    }
}
// MARK: - AI HDR

/// ai-hdr: a 2D title's AI HDR as one gain grid per frame (`GainGrid`, the
/// layout pinned in VergenceCore). GMNet runs on keyframes, every half second
/// on the title's frame grid and at every cut (the depth pipeline's luma-grid
/// detector); each keyframe's gain map is fitted into a grid, and the frames
/// between ease from one grid to the next within a shot and hold before a
/// cut (`GainGridSchedule`). A piece decodes one keyframe interval either
/// side, so its frames see the same keyframes a whole-file run does.
///
/// The model is Convolution's photo conversion (`GainMapGMNet`, from Hypnos's
/// `scripts/convert-gainmap-model.py`; GMNet is MIT), named by `gainModel`:
/// Vergence ships no weights, so the worker is told where they are, as with
/// depth. Its inputs are an image at one of the enumerated sizes (squeezed
/// from the frame, as the photo path does) and a 256×256 thumbnail; it gives
/// the linear gain for each pixel.
final class GainGridProducer: PieceProducer {
    static let workload = "ai-hdr"
    /// The model's enumerated input sizes (w, h), as in the conversion script.
    static let inputSizes: [(Int, Int)] = [(1024, 768), (768, 1024), (1024, 576), (576, 1024), (896, 896)]
    let model: MLModel
    let kinds: Set<String> = ["gainMap"]
    let models: [String: String]
    let needsStereo = false
    private(set) var framesBefore: Int64 = 12
    private(set) var framesAfter: Int64 = 12
    private var interval: Int64 = 12
    private var keyframes: [GainGridSchedule.Keyframe] = []
    private var previousSignature: [Float]?
    /// Sticky runs: the title being decoded, the last frame seen (its
    /// signature is `previousSignature`), and where the next piece starts.
    /// Nothing carries over from a piece that didn't finish.
    private var title: String?
    private var lastDecoded: Int64 = -1
    private var runNext: Int64?
    private var inProgress = false
    let keepsRuns = true

    init(modelURL: URL, computeUnits: MLComputeUnits) async throws {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        model = try MLModel(contentsOf: try await InterpolationModelBenchmark.compiledURL(for: modelURL),
                            configuration: configuration)
        let keyframes = Int((GainGridSchedule.keyframeSeconds * 1000).rounded())
        // "hl2": highlights expanded from the mid-tones (`GainGrid.expandingHighlights`); older tracks lack it.
        models = ["gainMap": "\(modelURL.deletingPathExtension().lastPathComponent) grid16x9x8 key\(keyframes)ms hl2"]
    }

    /// Set from the title's rate before a run.
    func prepare(frameRate: Double) {
        interval = GainGridSchedule.keyframeInterval(frameRate: frameRate)
        framesBefore = interval
        framesAfter = interval
    }

    /// The next piece of the same title carries on after the last frame seen:
    /// the keyframes up to there are known, and so is that frame's signature
    /// for the cut detector, so its context isn't decoded again.
    func resumeFrame(mediaID: String, piece: Range<Int64>) -> Int64? {
        defer { title = mediaID }
        guard !inProgress, title == mediaID, runNext == piece.lowerBound, previousSignature != nil,
              lastDecoded + 1 >= piece.lowerBound, lastDecoded + 1 < piece.upperBound else { return nil }
        return lastDecoded + 1
    }

    func begin(continuing: Bool) {
        if continuing {
            // Keep what the new piece's frames may still ease from.
            if let start = runNext { keyframes.removeAll { $0.index < start - 2 * interval } }
        } else {
            keyframes = []
            previousSignature = nil
        }
        runNext = nil
        inProgress = true
    }

    func frame(_ frame: DecodedFrame, index: Int64, time: AnalysisTime, piece: Range<Int64>) throws -> [AnalysisRecord] {
        lastDecoded = index
        let p = frame.left
        let signature = GainGridSchedule.signature(p[0])
        let cut = previousSignature.map { GainGridSchedule.isCut($0, signature) } ?? false
        previousSignature = signature
        guard GainGridSchedule.isKeyframe(index, interval: interval, cut: cut) else { return [] }
        keyframes.append(GainGridSchedule.Keyframe(index: index, grid: try grid(p), cut: cut))
        return []
    }

    func finish(piece: Range<Int64>) throws -> [AnalysisRecord] {
        inProgress = false
        runNext = piece.upperBound
        return piece.compactMap { index in
            GainGridSchedule.grid(at: index, keyframes: keyframes)?
                .record(time: AnalysisTime(value: index * timeScaleDen, timescale: timeScaleNum))
        }
    }

    /// The title's rate as a time base, for `finish`.
    var timeScaleNum: Int32 = 24
    var timeScaleDen: Int64 = 1

    /// GMNet on one frame, fitted into a grid, its highlights spread from the mid-tones.
    private func grid(_ p: [AnalysisPlane]) throws -> GainGrid {
        let aspect = Double(p[0].width) / Double(max(p[0].height, 1))
        let (w, h) = Self.inputSizes.min {
            abs(log(Double($0.0) / Double($0.1) / aspect)) < abs(log(Double($1.0) / Double($1.1) / aspect))
        }!
        let image = SharedFieldAnalysis.rgbPlanar(y: p[0], cb: p[1], cr: p[2], width: w, height: h)
        let thumbnail = SharedFieldAnalysis.rgbPlanar(y: p[0], cb: p[1], cr: p[2], width: 256, height: 256)
        guard let imageBuffer = Self.bgra(image, width: w, height: h),
              let thumbnailBuffer = Self.bgra(thumbnail, width: 256, height: 256) else {
            throw WorkerFailure("couldn't make GMNet's input buffers")
        }
        let inputs = try MLDictionaryFeatureProvider(dictionary: [
            "image": MLFeatureValue(pixelBuffer: imageBuffer), "thumbnail": MLFeatureValue(pixelBuffer: thumbnailBuffer),
        ])
        let output = try model.prediction(from: inputs)
        guard let gain = output.featureValue(for: "gain")?.multiArrayValue, gain.shape.count == 4 else {
            throw WorkerFailure("GMNet gave no gain map")
        }
        let gh = gain.shape[2].intValue, gw = gain.shape[3].intValue
        let values = StereoFlowAnalyzer.floats(gain)
        let rgb = SharedFieldAnalysis.rgbPlanar(y: p[0], cb: p[1], cr: p[2], width: gw, height: gh)
        let n = gw * gh
        let luma = (0 ..< n).map { GainGrid.luma(r: rgb[$0], g: rgb[n + $0], b: rgb[2 * n + $0]) }
        return GainGrid.fit(gain: values, luma: luma, width: gw, height: gh).expandingHighlights()
    }

    /// Planar RGB 0…1 as a BGRA pixel buffer.
    static func bgra(_ planar: [Float], width: Int, height: Int) -> CVPixelBuffer? {
        var buffer: CVPixelBuffer?
        CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA,
                            [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &buffer)
        guard let buffer else { return nil }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }
        let stride = CVPixelBufferGetBytesPerRow(buffer), n = width * height
        for y in 0 ..< height {
            let row = base.advanced(by: y * stride).assumingMemoryBound(to: UInt8.self)
            for x in 0 ..< width {
                let i = y * width + x
                func byte(_ v: Float) -> UInt8 { UInt8((min(max(v, 0), 1) * 255).rounded()) }
                row[4 * x] = byte(planar[2 * n + i])
                row[4 * x + 1] = byte(planar[n + i])
                row[4 * x + 2] = byte(planar[i])
                row[4 * x + 3] = 255
            }
        }
        return buffer
    }
}
#endif
