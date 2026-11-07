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
    /// A new run starts.
    func begin()
    /// The next frame, `index` on the title's grid. Returns records for frames in `piece` only.
    func frame(_ frame: DecodedFrame, index: Int64, time: AnalysisTime, piece: Range<Int64>) throws -> [AnalysisRecord]
    /// Records held back for context, once every frame was seen.
    func finish(piece: Range<Int64>) throws -> [AnalysisRecord]
}

extension PieceProducer {
    func begin() {}
    func finish(piece: Range<Int64>) throws -> [AnalysisRecord] { [] }
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
               mediaID: String, producerName: String) throws -> Data {
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

        let first = max(0, piece.lowerBound - producer.framesBefore)
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

        let header = AnalysisTrackHeader(mediaID: mediaID, producer: producerName, models: producer.models,
                                         sourceWidth: eyeW, sourceHeight: eyeH)
        var track = try AnalysisTrack.header(header)
        func append(_ records: [AnalysisRecord]) throws {
            for record in records { track.append(AnalysisTrack.record(try record.deflated())) }
        }
        producer.begin()
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
        let decodedPiece = min(index, piece.upperBound) - piece.lowerBound
        let reference = frameCount == Int64.max
        guard reference ? decodedPiece > 0 : decodedPiece >= piece.count else {
            throw WorkerFailure("ffmpeg gave \(index - first) of \(wanted) frames (\(process.terminationStatus)): \(messages.text.prefix(300))")
        }
        try append(producer.finish(piece: piece.lowerBound ..< min(piece.upperBound, index)))
        return track
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
final class StereoFlowProducer: PieceProducer {
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

    func begin() { previous = nil }

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

    func begin() { previous = nil }

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
#endif
