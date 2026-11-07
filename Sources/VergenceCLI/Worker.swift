/*
 vergence worker: analysis for a coordinator (`vergenced`)

   vergence worker --config FILE [--once]

 Claims 2 s pieces of titles from the coordinator, reads each piece from the
 media server itself (with its own credentials; the coordinator hands out no
 media URLs), runs the same models as `vergence analyze`, and uploads the
 records as a small track. Smooth Motion for 3D titles first: the left eye's
 flow between frames and each frame's right→left disparity. Design: Hypnos
 `docs/plans/analysis-workers.md`.

 FILE is JSON:

   {
     "coordinator": "http://home-server:8420",
     "tokenFile": "~/.config/vergence/worker.token",      (or "token")
     "name": "m2max",                                     (default: the host name)
     "remote": false,                                     (true: never takes live pieces)
     "models": "~/Models/interp",                         (RIFE*480x288 and NeuFlow*512x288, as for analyze)
     "computeUnits": "all",                               (ane | gpu | cpu | all)
     "ffmpeg": "/opt/homebrew/bin/ffmpeg",                (default: ffmpeg on PATH)
     "jellyfin": { "url": "http://home-server:8096", "apiKeyFile": "~/.config/vergence/jellyfin.key" },
     "stash": { "url": "http://home-server:9999", "apiKeyFile": "~/.config/vergence/stash.key" }
   }

 ffmpeg reads the media through a loopback forwarder that adds the server's
 credentials, so no key appears in a process list. It decodes the whole
 frame-packed picture; the eyes are read in place (half the width, or half
 the height), with no filter graph, so H.264 frame-packing messages can't
 break one. Times are frame index / the title's frame rate, the coordinator's
 frame grid. --once takes one piece and exits (for tests); otherwise it
 polls until stopped, backing off while there is nothing to do.
 */

#if canImport(CoreML) && canImport(Network)
import CoreML
import Foundation
import Network
import VergenceApple
import VergenceCore

struct WorkerConfig: Decodable {
    struct Server: Decodable {
        var url: String
        var apiKey: String?
        var apiKeyFile: String?

        func key() throws -> String? {
            if let apiKey { return apiKey }
            guard let apiKeyFile else { return nil }
            return try String(contentsOfFile: expandTilde(apiKeyFile), encoding: .utf8)
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }
    }

    var coordinator: String
    var token: String?
    var tokenFile: String?
    var name: String?
    var remote: Bool?
    var models: String
    var computeUnits: String?
    var ffmpeg: String?
    var jellyfin: Server?
    var stash: Server?
}

func expandTilde(_ path: String) -> String {
    (path as NSString).expandingTildeInPath
}

/// What the coordinator hands out (its `Lease`), as far as the worker needs it.
struct WorkLease: Decodable {
    struct Source: Decodable {
        var server: String
        var id: String
    }

    struct Rate: Decodable {
        var num: Int
        var den: Int
    }

    struct Info: Decodable {
        var durationSeconds: Double
        var frameRate: Rate
        var width: Int
        var height: Int
        var codec: String
        var stereo: String
        var mediaSourceId: String?
    }

    var lease: UInt64
    var title: String
    var source: Source
    var kinds: [String]
    var piece: Int
    var firstFrame: Int64
    var endFrame: Int64
    var frameCount: Int64
    var info: Info
    var leaseSeconds: Double
    var live: Bool
}

struct WorkerFailure: Error, CustomStringConvertible {
    var description: String
    init(_ description: String) { self.description = description }
}

func runWorker(_ args: [String]) async throws {
    var configPath: String?
    var once = false
    var i = 0
    while i < args.count {
        switch args[i] {
        case "--config":
            guard i + 1 < args.count else { usage("--config needs a value") }
            configPath = args[i + 1]
            i += 1
        case "--once": once = true
        default: usage("worker doesn't take \(args[i])")
        }
        i += 1
    }
    guard let configPath else { usage("worker needs --config FILE") }
    let config = try JSONDecoder().decode(WorkerConfig.self, from: Data(contentsOf: URL(fileURLWithPath: expandTilde(configPath))))
    let worker = try await AnalysisWorker(config: config)
    try await worker.run(once: once)
}

final class AnalysisWorker: @unchecked Sendable {
    let config: WorkerConfig
    let coordinator: URL
    let token: String
    let name: String
    let analyzer: StereoFlowAnalyzer
    let models: [String: String]
    let session = URLSession(configuration: .ephemeral)

    init(config: WorkerConfig) async throws {
        self.config = config
        guard let url = URL(string: config.coordinator) else { throw WorkerFailure("coordinator isn't a URL") }
        coordinator = url
        if let token = config.token {
            self.token = token
        } else if let file = config.tokenFile {
            token = try String(contentsOfFile: expandTilde(file), encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines)
        } else {
            throw WorkerFailure("the config needs token or tokenFile")
        }
        name = config.name ?? ProcessInfo.processInfo.hostName
        let folder = URL(fileURLWithPath: expandTilde(config.models))
        let flowURL = try findModel(in: folder, prefix: "RIFE", size: "480x288")
        let disparityURL = try findModel(in: folder, prefix: "NeuFlow", size: "512x288")
        let units: MLComputeUnits = switch config.computeUnits {
        case "ane": .cpuAndNeuralEngine
        case "gpu": .cpuAndGPU
        case "cpu": .cpuOnly
        default: .all
        }
        analyzer = try await StereoFlowAnalyzer(flowModel: flowURL, disparityModel: disparityURL, computeUnits: units)
        let flowSize = SharedFieldAnalysis.flowModelSize, disparitySize = SharedFieldAnalysis.disparityModelSize
        // Exactly what `vergence analyze` writes, so tracks from either agree on the profile.
        models = ["flowPair": "\(analyzer.flowModelName) \(flowSize.width)x\(flowSize.height)",
                  "disparity": "\(analyzer.disparityModelName) \(disparitySize.width)x\(disparitySize.height)"]
        // Below the user's own work.
        setpriority(PRIO_PROCESS, 0, 10)
    }

    func run(once: Bool) async throws {
        log("vergence worker: \(name), \(models["flowPair"]!), \(models["disparity"]!)")
        var idle: UInt64 = 2
        while true {
            let lease: WorkLease?
            do {
                lease = try await claim()
            } catch {
                log("vergence worker: claim failed: \(error)")
                if once { throw error }
                try await Task.sleep(nanoseconds: 10_000_000_000)
                continue
            }
            guard let lease else {
                if once { log("vergence worker: nothing to do"); return }
                try await Task.sleep(nanoseconds: idle * 1_000_000_000)
                idle = min(idle * 2, 30)
                continue
            }
            idle = 2
            let clock = ContinuousClock()
            let started = clock.now
            let heartbeat = Task { [self] in
                while !Task.isCancelled {
                    try? await Task.sleep(nanoseconds: UInt64(max(lease.leaseSeconds / 3, 5) * 1e9))
                    if Task.isCancelled { break }
                    _ = try? await post("v1/work/leases/\(lease.lease)/heartbeat")
                }
            }
            do {
                let track = try await analyze(lease)
                heartbeat.cancel()
                try await upload(track, for: lease)
                let seconds = (clock.now - started) / .seconds(1)
                log(String(format: "vergence worker: piece %d of %@ (%@) in %.1f s, %.0f KB", lease.piece,
                           lease.title, lease.live ? "live" : "ahead/backfill", seconds, Double(track.count) / 1024))
            } catch {
                heartbeat.cancel()
                log("vergence worker: piece \(lease.piece) of \(lease.title) failed: \(error)")
                _ = try? await post("v1/work/leases/\(lease.lease)/release")
                if once { throw error }
                try await Task.sleep(nanoseconds: 5_000_000_000)
            }
            if once { return }
        }
    }

    // MARK: Coordinator

    private func request(_ path: String, method: String) -> URLRequest {
        var components = URLComponents(url: coordinator.appending(path: path), resolvingAgainstBaseURL: false)!
        if path.hasPrefix("v1/work/leases") { components.queryItems = [URLQueryItem(name: "worker", value: name)] }
        var request = URLRequest(url: components.url!)
        request.httpMethod = method
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        request.timeoutInterval = 30
        return request
    }

    private func post(_ path: String) async throws -> Int {
        let (_, response) = try await session.data(for: request(path, method: "POST"))
        return (response as? HTTPURLResponse)?.statusCode ?? -1
    }

    func claim() async throws -> WorkLease? {
        var request = request("v1/work/claim", method: "POST")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: [
            "worker": name, "remote": config.remote ?? false, "models": models,
        ])
        let (data, response) = try await session.data(for: request)
        switch (response as? HTTPURLResponse)?.statusCode {
        case 200: return try JSONDecoder().decode(WorkLease.self, from: data)
        case 204: return nil
        case let status: throw WorkerFailure("the coordinator answered \(status ?? -1): \(String(decoding: data, as: UTF8.self))")
        }
    }

    func upload(_ track: Data, for lease: WorkLease) async throws {
        var request = request("v1/work/leases/\(lease.lease)/result", method: "PUT")
        request.setValue("application/octet-stream", forHTTPHeaderField: "Content-Type")
        let (data, response) = try await session.upload(for: request, from: track)
        let status = (response as? HTTPURLResponse)?.statusCode ?? -1
        guard status == 200 else {
            throw WorkerFailure("the result was refused (\(status)): \(String(decoding: data, as: UTF8.self))")
        }
    }

    // MARK: Media

    /// The piece's URL on its server and the header that authorizes it.
    func mediaRequest(_ lease: WorkLease) throws -> (URL, [String: String]) {
        switch lease.source.server {
        case "jellyfin":
            guard let server = config.jellyfin else { throw WorkerFailure("no Jellyfin in the worker's config") }
            var components = URLComponents(string: server.url.trimmingCharacters(in: CharacterSet(charactersIn: "/")) + "/Videos/\(lease.source.id)/stream")!
            components.queryItems = [URLQueryItem(name: "static", value: "true")]
            if let id = lease.info.mediaSourceId { components.queryItems?.append(URLQueryItem(name: "mediaSourceId", value: id)) }
            let headers = try server.key().map { ["Authorization": "MediaBrowser Client=\"vergence\", Token=\"\($0)\""] } ?? [:]
            return (components.url!, headers)
        case "stash":
            guard let server = config.stash else { throw WorkerFailure("no Stash in the worker's config") }
            let url = URL(string: server.url.trimmingCharacters(in: CharacterSet(charactersIn: "/")) + "/scene/\(lease.source.id)/stream")!
            return (url, try server.key().map { ["ApiKey": $0] } ?? [:])
        default:
            throw WorkerFailure("unknown server \(lease.source.server)")
        }
    }

    /// Runs the models over the piece's frames; returns the piece as a track.
    func analyze(_ lease: WorkLease) async throws -> Data {
        guard Set(lease.kinds) == ["flowPair", "disparity"] else {
            throw WorkerFailure("this worker makes Smooth Motion fields only, not \(lease.kinds)")
        }
        let w = lease.info.width, h = lease.info.height
        let sideBySide: Bool
        switch lease.info.stereo {
        case "halfSideBySide", "fullSideBySide": sideBySide = true
        case "halfTopAndBottom", "fullTopAndBottom": sideBySide = false
        default: throw WorkerFailure("the title's layout (\(lease.info.stereo)) isn't frame-packed stereo")
        }
        guard w > 0, h > 0, w % 4 == 0, h % 4 == 0 else { throw WorkerFailure("the picture's size (\(w)x\(h)) is unusable") }
        let eyeW = sideBySide ? w / 2 : w, eyeH = sideBySide ? h : h / 2

        let (url, headers) = try mediaRequest(lease)
        let forwarder = try LoopbackForwarder(target: url, headers: headers)
        defer { forwarder.stop() }
        let rate = lease.info.frameRate
        // One frame past the piece when there is one: the last flow pair reaches it.
        let wanted = Int(lease.endFrame - lease.firstFrame) + (lease.endFrame < lease.frameCount ? 1 : 0)
        // Half a frame early, so the first frame out is exactly firstFrame.
        let seek = (Double(lease.firstFrame) - 0.5) * Double(rate.den) / Double(rate.num)
        var args = ["-nostdin", "-v", "error"]
        // A frame-packing SEI makes ffmpeg 8 and 9 fail filter graphs; drop SEI before decoding.
        if lease.info.codec.lowercased() == "h264" { args += ["-bsf:v", "filter_units=remove_types=6"] }
        if seek > 0 { args += ["-ss", String(format: "%.6f", seek)] }
        // Passthrough: rawvideo's default constant-rate sync would repeat the first frame after a seek.
        args += ["-i", forwarder.url.absoluteString, "-map", "0:v:0", "-fps_mode", "passthrough",
                 "-frames:v", String(wanted), "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"]
        let process = Process()
        process.executableURL = URL(fileURLWithPath: config.ffmpeg.map(expandTilde) ?? "/usr/bin/env")
        process.arguments = config.ffmpeg == nil ? ["ffmpeg"] + args : args
        process.qualityOfService = .utility
        let output = Pipe(), errors = Pipe()
        process.standardOutput = output
        process.standardError = errors
        try process.run()
        defer { if process.isRunning { process.terminate() } }

        let header = AnalysisTrackHeader(mediaID: lease.title, producer: "vergence worker",
                                         models: models, sourceWidth: eyeW, sourceHeight: eyeH)
        var track = try AnalysisTrack.header(header)
        let flowSize = SharedFieldAnalysis.flowModelSize, disparitySize = SharedFieldAnalysis.disparityModelSize
        let frameBytes = w * h * 3 / 2
        let reader = output.fileHandleForReading
        var previous: (left: [Float], time: AnalysisTime)?
        var index = lease.firstFrame
        while index < lease.firstFrame + Int64(wanted) {
            var data = Data()
            while data.count < frameBytes {
                guard let chunk = try reader.read(upToCount: frameBytes - data.count), !chunk.isEmpty else { break }
                data.append(chunk)
            }
            guard data.count == frameBytes else { break }
            let time = AnalysisTime(value: index * Int64(rate.den), timescale: Int32(rate.num))
            let inPiece = index < lease.endFrame
            try autoreleasepool {
                try data.withUnsafeBytes { raw in
                    let base = raw.bindMemory(to: UInt8.self).baseAddress!
                    let (left, right) = Self.eyes(base: base, width: w, height: h, sideBySide: sideBySide)
                    let leftFlow = SharedFieldAnalysis.rgbPlanar(y: left[0], cb: left[1], cr: left[2],
                                                                 width: flowSize.width, height: flowSize.height)
                    let runs = PieceRuns()
                    DispatchQueue.concurrentPerform(iterations: 2) { job in
                        do {
                            if job == 0, let previous {
                                runs.flow = try analyzer.flow(from: previous.left, to: leftFlow)
                            } else if job == 1, inPiece {
                                let l = SharedFieldAnalysis.rgbPlanar(y: left[0], cb: left[1], cr: left[2],
                                                                      width: disparitySize.width, height: disparitySize.height)
                                let r = SharedFieldAnalysis.rgbPlanar(y: right[0], cb: right[1], cr: right[2],
                                                                      width: disparitySize.width, height: disparitySize.height)
                                runs.disparity = try analyzer.disparity(left: l, right: r)
                            }
                        } catch {
                            runs.error = error
                        }
                    }
                    if let error = runs.error { throw error }
                    if let previous, let (flow, mask) = runs.flow {
                        track.append(AnalysisTrack.record(try SharedFieldAnalysis.flowPairRecord(
                            flow: flow, mask: mask, width: flowSize.width, height: flowSize.height,
                            time: previous.time, endTime: time).deflated()))
                    }
                    if let disparity = runs.disparity {
                        track.append(AnalysisTrack.record(try SharedFieldAnalysis.disparityRecord(
                            rightToLeft: disparity, width: disparitySize.width, height: disparitySize.height,
                            time: time).deflated()))
                    }
                    previous = (leftFlow, time)
                }
            }
            index += 1
        }
        process.waitUntilExit()
        let decoded = index - lease.firstFrame
        guard decoded >= Int64(lease.endFrame - lease.firstFrame) else {
            let message = String(decoding: errors.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
            throw WorkerFailure("ffmpeg gave \(decoded) of \(wanted) frames (\(process.terminationStatus)): \(message.prefix(300))")
        }
        return track
    }

    /// The two eyes of a packed I420 picture, read in place.
    static func eyes(base: UnsafePointer<UInt8>, width w: Int, height h: Int,
                     sideBySide: Bool) -> (left: [AnalysisPlane], right: [AnalysisPlane]) {
        let cb = base + w * h, cr = cb + (w / 2) * (h / 2)
        if sideBySide {
            func eye(_ e: Int) -> [AnalysisPlane] {
                [AnalysisPlane(base: base + e * w / 2, width: w / 2, height: h, stride: w),
                 AnalysisPlane(base: cb + e * w / 4, width: w / 4, height: h / 2, stride: w / 2),
                 AnalysisPlane(base: cr + e * w / 4, width: w / 4, height: h / 2, stride: w / 2)]
            }
            return (eye(0), eye(1))
        }
        func eye(_ e: Int) -> [AnalysisPlane] {
            [AnalysisPlane(base: base + e * w * h / 2, width: w, height: h / 2, stride: w),
             AnalysisPlane(base: cb + e * (w / 2) * (h / 4), width: w / 2, height: h / 4, stride: w / 2),
             AnalysisPlane(base: cr + e * (w / 2) * (h / 4), width: w / 2, height: h / 4, stride: w / 2)]
        }
        return (eye(0), eye(1))
    }
}

/// What one frame's two concurrent model runs produced.
private final class PieceRuns: @unchecked Sendable {
    var flow: (flow: [Float], mask: [Float])?
    var disparity: [Float]?
    var error: Error?
}

/// An HTTP endpoint on 127.0.0.1, at a random path, that passes each
/// request (with its Range) on to one URL with headers added: ffmpeg reads
/// the media through it, and the credentials never reach its command line.
final class LoopbackForwarder: @unchecked Sendable {
    private let listener: NWListener
    private let target: URL
    private let headers: [String: String]
    private let secret = UUID().uuidString
    private let queue = DispatchQueue(label: "vergence.worker.forwarder")
    private let session: URLSession
    private var port: UInt16 = 0

    init(target: URL, headers: [String: String]) throws {
        self.target = target
        self.headers = headers
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 60
        session = URLSession(configuration: configuration)
        let parameters = NWParameters.tcp
        parameters.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: .any)
        listener = try NWListener(using: parameters)
        let ready = DispatchSemaphore(value: 0)
        listener.stateUpdateHandler = { state in
            switch state {
            case .ready, .failed, .cancelled: ready.signal()
            default: break
            }
        }
        listener.newConnectionHandler = { [weak self] connection in self?.serve(connection) }
        listener.start(queue: queue)
        _ = ready.wait(timeout: .now() + 5)
        guard let port = listener.port?.rawValue, port != 0 else { throw WorkerFailure("the loopback forwarder didn't start") }
        self.port = port
    }

    var url: URL { URL(string: "http://127.0.0.1:\(port)/\(secret)")! }

    func stop() {
        listener.cancel()
        session.invalidateAndCancel()
    }

    private func serve(_ connection: NWConnection) {
        connection.start(queue: queue)
        readHead(connection, Data())
    }

    private func readHead(_ connection: NWConnection, _ buffer: Data) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 16384) { [self] data, _, done, error in
            var buffer = buffer
            if let data { buffer.append(data) }
            if let end = buffer.range(of: Data("\r\n\r\n".utf8)) {
                respond(connection, head: String(decoding: buffer[..<end.lowerBound], as: UTF8.self))
            } else if error != nil || done || buffer.count > 65536 {
                connection.cancel()
            } else {
                readHead(connection, buffer)
            }
        }
    }

    private func respond(_ connection: NWConnection, head: String) {
        let lines = head.components(separatedBy: "\r\n")
        let parts = lines.first?.split(separator: " ") ?? []
        guard parts.count >= 2, ["GET", "HEAD"].contains(parts[0]), parts[1] == "/\(secret)" else {
            connection.send(content: Data("HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".utf8),
                            isComplete: true, completion: .contentProcessed { _ in connection.cancel() })
            return
        }
        var request = URLRequest(url: target)
        request.httpMethod = String(parts[0])
        for (name, value) in headers { request.setValue(value, forHTTPHeaderField: name) }
        for line in lines.dropFirst() {
            guard let colon = line.firstIndex(of: ":") else { continue }
            let name = line[..<colon].trimmingCharacters(in: .whitespaces)
            if name.caseInsensitiveCompare("Range") == .orderedSame {
                request.setValue(line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces), forHTTPHeaderField: "Range")
            }
        }
        let relay = ForwardRelay(connection: connection, isHead: parts[0] == "HEAD")
        let task = session.dataTask(with: request)
        task.delegate = relay
        connection.stateUpdateHandler = { state in
            if case .cancelled = state { task.cancel() }
            if case .failed = state { task.cancel() }
        }
        task.resume()
    }
}

/// Streams one upstream response back down a loopback connection.
private final class ForwardRelay: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    let connection: NWConnection
    let isHead: Bool

    init(connection: NWConnection, isHead: Bool) {
        self.connection = connection
        self.isHead = isHead
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
                    completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
        let http = response as? HTTPURLResponse
        var head = "HTTP/1.1 \(http?.statusCode ?? 502) Relay\r\nConnection: close\r\nAccept-Ranges: bytes\r\n"
        for name in ["Content-Length", "Content-Range", "Content-Type"] {
            if let value = http?.value(forHTTPHeaderField: name) { head += "\(name): \(value)\r\n" }
        }
        connection.send(content: Data((head + "\r\n").utf8), completion: .idempotent)
        completionHandler(isHead ? .cancel : .allow)
        if isHead { connection.send(content: nil, isComplete: true, completion: .contentProcessed { [connection] _ in connection.cancel() }) }
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        connection.send(content: data, completion: .idempotent)
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        guard !isHead else { return }
        connection.send(content: nil, isComplete: true, completion: .contentProcessed { [connection] _ in connection.cancel() })
    }
}
#else
func runWorker(_ args: [String]) async throws {
    throw FormatError("worker needs Core ML (Apple platforms only)")
}
#endif
