/*
 vergence worker: analysis for a coordinator (`vergenced`)

   vergence worker --config FILE [--once]
   vergence worker --config FILE --reference INPUT --workload NAME [--stereo LAYOUT] [--frames N] --output FILE.vgat

 Claims 2 s pieces of titles from the coordinator, reads each piece from the
 media server itself (with its own credentials; the coordinator hands out no
 media URLs), runs the workload's models, and uploads the records as a small
 track. Workloads (`WorkerProducers.swift`):
 - smooth-motion-3d: the left eye's flow between frames (RIFE 480×288) and
   each frame's right→left disparity (NeuFlow 512×288), as `vergence analyze`.
 - smooth-motion-2d: the whole picture's flow (RIFE 960×576), for 2D titles
   and AI 3D; offered when the models folder has a RIFE*960x576.
 - ai-hdr: a 2D title's AI HDR as a gain grid per frame (GMNet on keyframes
   with `gainModel`, `GainGrid`); offered when the config names one.
 - ai-3d-depth: a 2D title's AI 3D depth through the offline pipeline with
   `depthModel` (Depth Anything V2 Small at 770×574 rather than the headset's
   518×392); offered when the config names one.
 Design: Hypnos `docs/plans/analysis-workers.md`.

 --reference runs one workload over a local file in a single decode and
 writes the track, the reference `scripts/check-worker.sh` compares a
 coordinator's chunks with (LAYOUT as the coordinator names it, mono by
 default).

 FILE is JSON:

   {
     "coordinator": "http://home-server:8420",
     "tokenFile": "~/.config/vergence/worker.token",      (or "token")
     "name": "m2max",                                     (default: the host name)
     "remote": false,                                     (true: never takes live pieces)
     "models": "~/Models/interp",                         (RIFE*480x288 and NeuFlow*512x288, as for analyze; RIFE*960x576 for 2D)
     "workloads": ["smooth-motion-3d"],                   (default: every one the models allow)
     "stickyRuns": true,                                  (carry depth and AI HDR state on into a title's next piece)
     "depthModel": "~/Models/DepthAnythingV2Small770x574.mlpackage",   (ai-3d-depth; Apache-2.0 weights only)
     "gainModel": "~/Models/GainMapGMNet.mlpackage",      (ai-hdr; Convolution's GMNet conversion)
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
    var workloads: [String]?
    /// Ask to carry on into the next piece of a title (default true): depth
    /// keeps its anchoring and AI HDR its keyframes instead of decoding context.
    var stickyRuns: Bool?
    /// The AI 3D depth model (`ai-3d-depth`), a Core ML Depth Anything V2 Small conversion.
    var depthModel: String?
    /// AI HDR's GMNet (`ai-hdr`): Convolution's GainMapGMNet conversion (MIT).
    var gainModel: String?
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
    var workload: String
    var kinds: [String]
    var piece: Int
    var firstFrame: Int64
    var endFrame: Int64
    var frameCount: Int64
    var info: Info
    var leaseSeconds: Double
    var live: Bool
    /// The piece's server credential a player handed to the coordinator, for a worker on
    /// loopback or the paired channel; read with it instead of the config's own.
    var credential: Credential?

    struct Credential: Decodable, CustomStringConvertible {
        var url: String
        var key: String
        var description: String { "Credential(url: \(url), key: <redacted>)" }
    }
}

struct WorkerFailure: Error, CustomStringConvertible {
    var description: String
    init(_ description: String) { self.description = description }
}

func runWorker(_ args: [String]) async throws {
    var configPath: String?
    var once = false
    var reference: String?, workload: String?, stereo = "mono", output: String?
    var frames = Int64.max
    var i = 0
    func value() -> String {
        guard i + 1 < args.count else { usage("\(args[i]) needs a value") }
        i += 1
        return args[i]
    }
    while i < args.count {
        switch args[i] {
        case "--config": configPath = value()
        case "--once": once = true
        case "--reference": reference = value()
        case "--workload": workload = value()
        case "--stereo": stereo = value()
        case "--output", "-o": output = value()
        case "--frames":
            guard let n = Int64(value()) else { usage("--frames needs a number") }
            frames = n
        default: usage("worker doesn't take \(args[i])")
        }
        i += 1
    }
    guard let configPath else { usage("worker needs --config FILE") }
    // `-` reads it from standard input: how the desktop app starts its worker, with the token
    // inline, so nothing secret is written to a file.
    let configData = configPath == "-"
        ? FileHandle.standardInput.readDataToEndOfFile()
        : try Data(contentsOf: URL(fileURLWithPath: expandTilde(configPath)))
    let config = try JSONDecoder().decode(WorkerConfig.self, from: configData)
    if let reference {
        guard let workload, let output else { usage("--reference needs --workload and --output") }
        let producers = try await makeProducers(config)
        guard let producer = producers[workload] else { usage("no \(workload) with these models") }
        let input = try PieceInput.probe(reference, stereo: stereo)
        let track = try await PieceDecoder(ffmpeg: config.ffmpeg.map(expandTilde)).track(
            of: input, frames: 0 ..< frames, frameCount: frames, producer: producer,
            mediaID: "reference", producerName: "vergence worker --reference")
        try track.write(to: URL(fileURLWithPath: output))
        log("vergence worker: \(workload) reference, \(track.count / 1024) KB -> \(output)")
        return
    }
    let worker = try await AnalysisWorker(config: config)
    try await worker.run(once: once)
}

/// The workloads this config's models make, by name.
func makeProducers(_ config: WorkerConfig) async throws -> [String: PieceProducer] {
    let folder = URL(fileURLWithPath: expandTilde(config.models))
    let units: MLComputeUnits = switch config.computeUnits {
    case "ane": .cpuAndNeuralEngine
    case "gpu": .cpuAndGPU
    case "cpu": .cpuOnly
    default: .all
    }
    let wanted = config.workloads.map(Set.init)
    var producers: [String: PieceProducer] = [:]
    if wanted?.contains(StereoFlowProducer.workload) ?? true {
        let flowURL = try findModel(in: folder, prefix: "RIFE", size: "480x288")
        let disparityURL = try findModel(in: folder, prefix: "NeuFlow", size: "512x288")
        let analyzer = try await StereoFlowAnalyzer(flowModel: flowURL, disparityModel: disparityURL, computeUnits: units)
        producers[StereoFlowProducer.workload] = StereoFlowProducer(analyzer: analyzer)
    }
    if wanted?.contains(MonoFlowProducer.workload) ?? true {
        // Optional unless asked for by name: without the larger RIFE, 2D titles are left to others.
        // RIFE 4.25 lite, the family the 3D workload and the ONNX export use, before any other RIFE.
        if let url = (try? findModel(in: folder, prefix: "RIFE425lite", size: MonoFlowProducer.modelSize))
            ?? (try? findModel(in: folder, prefix: "RIFE", size: MonoFlowProducer.modelSize)) {
            // At 960×576 the GPU is the fast place on an M2 Max (40 fps a title, against 16 on the
            // Neural Engine or with Core ML choosing), so it goes there unless the config says otherwise.
            producers[MonoFlowProducer.workload] = MonoFlowProducer(
                analyzer: try await MonoFlowAnalyzer(flowModel: url, computeUnits: config.computeUnits == nil ? .cpuAndGPU : units))
        } else if wanted != nil {
            throw WorkerFailure("smooth-motion-2d needs RIFE*\(MonoFlowProducer.modelSize) in \(folder.path)")
        }
    }
    if wanted?.contains(GainGridProducer.workload) ?? (config.gainModel != nil) {
        guard let path = config.gainModel else { throw WorkerFailure("ai-hdr needs gainModel in the config") }
        producers[GainGridProducer.workload] = try await GainGridProducer(
            modelURL: URL(fileURLWithPath: expandTilde(path)), computeUnits: units)
    }
    if wanted?.contains(DepthProducer.workload) ?? (config.depthModel != nil) {
        guard let path = config.depthModel else { throw WorkerFailure("ai-3d-depth needs depthModel in the config") }
        producers[DepthProducer.workload] = try await DepthProducer(
            model: checkDepthModelLicence(URL(fileURLWithPath: expandTilde(path))))
    }
    guard !producers.isEmpty else { throw WorkerFailure("the config asks for no workload this worker makes") }
    return producers
}

final class AnalysisWorker: @unchecked Sendable {
    let config: WorkerConfig
    let coordinator: URL
    let token: String
    let name: String
    let producers: [String: PieceProducer]
    /// The title and frame this worker's last piece ended at, when its producer
    /// keeps state for the next one (a sticky run; the coordinator decides).
    var lastRun: (title: String, next: Int64)?
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
        producers = try await makeProducers(config)
        // Below the user's own work.
        setpriority(PRIO_PROCESS, 0, 10)
    }

    func run(once: Bool) async throws {
        for (workload, producer) in producers.sorted(by: { $0.key < $1.key }) {
            log("vergence worker: \(name), \(workload): \(producer.models.sorted(by: { $0.key < $1.key }).map(\.value).joined(separator: ", "))")
        }
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
                let carried = lastRun.map { $0.title == lease.title && $0.next == lease.firstFrame } ?? false
                log(String(format: "vergence worker: piece %d of %@ (%@%@) in %.1f s, %.0f KB", lease.piece,
                           lease.title, lease.live ? "live" : "ahead/backfill", carried ? ", carried on" : "", seconds,
                           Double(track.count) / 1024))
                // Sticky runs: ask for the next piece of this title, which the producer can carry on into.
                lastRun = config.stickyRuns != false && producers[lease.workload]?.keepsRuns == true
                    && lease.endFrame < lease.frameCount
                    ? (lease.title, lease.endFrame) : nil
            } catch {
                lastRun = nil
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
        // Each workload with its own models (RIFE runs at two sizes); `models` for an older coordinator.
        let byWorkload = producers.mapValues(\.models)
        request.httpBody = try JSONSerialization.data(withJSONObject: [
            "worker": name, "remote": config.remote ?? false,
            "models": byWorkload[StereoFlowProducer.workload] ?? byWorkload.values.first ?? [:],
            "workloads": producers.keys.sorted(), "workloadModels": byWorkload,
        ].merging(lastRun.map { ["continues": ["title": $0.title, "frame": $0.next]] } ?? [:]) { a, _ in a } as [String: Any])
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

    /// The piece's URL on its server and the header that authorizes it: with the credential the
    /// lease carries (a player's, held in the coordinator's memory), else the config's own.
    func mediaRequest(_ lease: WorkLease) throws -> (URL, [String: String]) {
        func server(_ configured: WorkerConfig.Server?, _ name: String) throws -> (url: String, key: String?) {
            if let credential = lease.credential { return (credential.url, credential.key) }
            guard let configured else { throw WorkerFailure("no \(name) in the worker's config, and the lease carries no credential") }
            return (configured.url, try configured.key())
        }
        switch lease.source.server {
        case "jellyfin":
            let (base, key) = try server(config.jellyfin, "Jellyfin")
            guard var components = URLComponents(string: base.trimmingCharacters(in: CharacterSet(charactersIn: "/")) + "/Videos/\(lease.source.id)/stream") else {
                throw WorkerFailure("the Jellyfin address isn't a URL")
            }
            components.queryItems = [URLQueryItem(name: "static", value: "true")]
            if let id = lease.info.mediaSourceId { components.queryItems?.append(URLQueryItem(name: "mediaSourceId", value: id)) }
            let headers = key.map { ["Authorization": "MediaBrowser Client=\"vergence\", Token=\"\($0)\""] } ?? [:]
            guard let url = components.url else { throw WorkerFailure("the Jellyfin address isn't a URL") }
            return (url, headers)
        case "stash":
            let (base, key) = try server(config.stash, "Stash")
            guard let url = URL(string: base.trimmingCharacters(in: CharacterSet(charactersIn: "/")) + "/scene/\(lease.source.id)/stream") else {
                throw WorkerFailure("the Stash address isn't a URL")
            }
            return (url, key.map { ["ApiKey": $0] } ?? [:])
        default:
            throw WorkerFailure("unknown server \(lease.source.server)")
        }
    }

    /// Runs the workload's models over the piece's frames; returns the piece as a track.
    func analyze(_ lease: WorkLease) async throws -> Data {
        guard let producer = producers[lease.workload] else {
            throw WorkerFailure("this worker doesn't make \(lease.workload)")
        }
        guard Set(lease.kinds) == producer.kinds else {
            throw WorkerFailure("\(lease.workload) wants \(lease.kinds), this worker makes \(producer.kinds.sorted())")
        }
        let (url, headers) = try mediaRequest(lease)
        let forwarder = try LoopbackForwarder(target: url, headers: headers)
        defer { forwarder.stop() }
        let input = PieceInput(url: forwarder.url.absoluteString, width: lease.info.width, height: lease.info.height,
                               codec: lease.info.codec, stereo: lease.info.stereo,
                               rate: (lease.info.frameRate.num, lease.info.frameRate.den))
        return try await PieceDecoder(ffmpeg: config.ffmpeg.map(expandTilde)).track(
            of: input, frames: lease.firstFrame ..< lease.endFrame, frameCount: lease.frameCount,
            producer: producer, mediaID: lease.title, producerName: "vergence worker")
    }
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
