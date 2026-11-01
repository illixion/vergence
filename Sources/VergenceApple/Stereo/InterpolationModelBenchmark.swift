/*
 Vergence - On-device timing for frame-interpolation flow networks.

 Phase E0 of the stereo Smooth Motion plan (Hypnos
 `docs/plans/blu-ray-3d-smooth-motion.md`): before building on a neural flow
 network, learn what one costs on the headset. For every Core ML model in a
 folder (RIFE and NeuFlow conversions from Hypnos's
 `scripts/convert-interp-model.py`), on each compute-unit choice, it reports:

 - **Load time**, compile included the first time (a `.mlpackage` is compiled
   once and cached beside the benchmark's other files).
 - **Steady prediction time**: median, p90 and best of `iterations` runs after
   a warm-up, inputs filled with noise (the networks' cost does not depend on
   the picture).
 - **Where Core ML put the ops** (`MLComputePlan`): a warp (`resample`) that
   falls off the Neural Engine splits the graph and costs transfers, which is
   the first thing to know when the ANE number looks wrong.

 The models take fixed-size float16 tensors, so the inputs are made from each
 model's own description; a `timestep` input gets 0.5. Runs off the main
 thread, one model at a time.
 */

#if canImport(CoreML)
import CoreML
import Foundation
import VergenceCore

public enum InterpolationModelBenchmark {
    public struct Result: Codable, Sendable {
        public var model: String
        public var computeUnits: String
        public var loadMs: Double
        public var medianMs: Double
        public var p90Ms: Double
        public var bestMs: Double
        /// Ops per preferred device (`neuralEngine`, `gpu`, `cpu`) under this
        /// compute-unit choice. Empty when the plan couldn't be read.
        public var opsByDevice: [String: Int]
        /// The off-ANE op types, most frequent first, when any are off it.
        public var offNeuralEngineOps: [String]
        public var error: String?
    }

    public static let unitChoices: [(name: String, units: MLComputeUnits)] = [
        ("cpuAndNeuralEngine", .cpuAndNeuralEngine),
        ("cpuAndGPU", .cpuAndGPU),
        ("all", .all),
    ]

    /// Times every `.mlpackage` / `.mlmodelc` in `folder`. `units` filters
    /// `unitChoices` by name; empty means all of them.
    public static func run(folder: URL, units: [String] = [], iterations: Int = 30,
                           only: String? = nil) async -> [Result] {
        await Task.detached(priority: .userInitiated) {
            let fm = FileManager.default
            let entries = ((try? fm.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)) ?? [])
                .filter { ["mlpackage", "mlmodelc"].contains($0.pathExtension) }
                .filter { only == nil || $0.lastPathComponent.localizedCaseInsensitiveContains(only!) }
                .sorted { $0.lastPathComponent < $1.lastPathComponent }
            let choices = unitChoices.filter { units.isEmpty || units.contains($0.name) }
            var results: [Result] = []
            for entry in entries {
                let name = entry.deletingPathExtension().lastPathComponent
                let compiled: URL
                let compileStart = ContinuousClock.now
                do { compiled = try await compiledURL(for: entry) } catch {
                    results.append(Result(model: name, computeUnits: "-", loadMs: 0, medianMs: 0, p90Ms: 0, bestMs: 0,
                                          opsByDevice: [:], offNeuralEngineOps: [],
                                          error: "compile: \(error.localizedDescription)"))
                    continue
                }
                let compileMs = compileStart.duration(to: .now).ms
                for (index, choice) in choices.enumerated() {
                    var result = await measure(compiled: compiled, name: name, unitsName: choice.name,
                                               units: choice.units, iterations: iterations)
                    if index == 0 { result.loadMs += compileMs }
                    VergenceLog.pipeline.info(
                        "[interp-bench] \(name, privacy: .private) \(choice.name, privacy: .public): median \(result.medianMs, format: .fixed(precision: 1), privacy: .public) ms, p90 \(result.p90Ms, format: .fixed(precision: 1), privacy: .public) ms, load \(result.loadMs, format: .fixed(precision: 0), privacy: .public) ms, ops \(result.opsByDevice.description, privacy: .public)")
                    results.append(result)
                }
            }
            return results
        }.value
    }

    private static func compiledURL(for url: URL) async throws -> URL {
        guard url.pathExtension == "mlpackage" else { return url }
        let fm = FileManager.default
        let cache = fm.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("InterpModelBenchmark", isDirectory: true)
        try fm.createDirectory(at: cache, withIntermediateDirectories: true)
        let target = cache.appendingPathComponent(url.deletingPathExtension().lastPathComponent + ".mlmodelc")
        let sourceDate = (try? fm.attributesOfItem(atPath: url.appendingPathComponent("Manifest.json").path)[.modificationDate]
                          as? Date) ?? .distantPast
        if let cachedDate = try? fm.attributesOfItem(atPath: target.path)[.modificationDate] as? Date,
           cachedDate >= sourceDate {
            return target
        }
        let temp = try await MLModel.compileModel(at: url)
        try? fm.removeItem(at: target)
        try fm.moveItem(at: temp, to: target)
        return target
    }

    private static func measure(compiled: URL, name: String, unitsName: String, units: MLComputeUnits,
                                iterations: Int) async -> Result {
        var result = Result(model: name, computeUnits: unitsName, loadMs: 0, medianMs: 0, p90Ms: 0, bestMs: 0,
                            opsByDevice: [:], offNeuralEngineOps: [], error: nil)
        let config = MLModelConfiguration()
        config.computeUnits = units
        do {
            let loadStart = ContinuousClock.now
            let model = try MLModel(contentsOf: compiled, configuration: config)
            result.loadMs = loadStart.duration(to: .now).ms
            let inputs = try makeInputs(for: model.modelDescription)
            for _ in 0 ..< 5 { _ = try await model.prediction(from: inputs) }
            var times: [Double] = []
            times.reserveCapacity(iterations)
            for _ in 0 ..< max(iterations, 1) {
                let start = ContinuousClock.now
                _ = try await model.prediction(from: inputs)
                times.append(start.duration(to: .now).ms)
            }
            times.sort()
            result.medianMs = times[times.count / 2]
            result.p90Ms = times[min(times.count - 1, Int(Double(times.count) * 0.9))]
            result.bestMs = times[0]
        } catch {
            result.error = error.localizedDescription
        }
        if #available(macOS 14.4, iOS 17.4, tvOS 17.4, visionOS 1.1, *) {
            (result.opsByDevice, result.offNeuralEngineOps) = await placement(compiled: compiled, config: config)
        }
        return result
    }

    @available(macOS 14.4, iOS 17.4, tvOS 17.4, visionOS 1.1, *)
    private static func placement(compiled: URL, config: MLModelConfiguration) async -> ([String: Int], [String]) {
        if let plan = try? await MLComputePlan.load(contentsOf: compiled, configuration: config),
           case let .program(program) = plan.modelStructure,
           let main = program.functions["main"] {
            var counts: [String: Int] = [:]
            var off: [String: Int] = [:]
            for op in main.block.operations {
                guard let usage = plan.deviceUsage(for: op) else { continue }
                let device: String
                switch usage.preferred {
                case .neuralEngine: device = "neuralEngine"
                case .gpu: device = "gpu"
                case .cpu: device = "cpu"
                @unknown default: device = "other"
                }
                counts[device, default: 0] += 1
                if device != "neuralEngine" { off[op.operatorName, default: 0] += 1 }
            }
            return (counts, off.sorted { $0.value > $1.value }.prefix(8).map { "\($0.key)×\($0.value)" })
        }
        return ([:], [])
    }

    private static func makeInputs(for description: MLModelDescription) throws -> MLFeatureProvider {
        var features: [String: MLFeatureValue] = [:]
        for (name, input) in description.inputDescriptionsByName {
            guard let constraint = input.multiArrayConstraint else { continue }
            let array = try MLMultiArray(shape: constraint.shape, dataType: .float16)
            let isTimestep = name == "timestep"
            // Half-float bit patterns, since `Float16` doesn't exist on an
            // Intel Mac: 0x3800 is 0.5, and 0…0x3BFF covers [0, 1).
            array.withUnsafeMutableBytes { raw, _ in
                let values = raw.bindMemory(to: UInt16.self)
                for i in values.indices { values[i] = isTimestep ? 0x3800 : UInt16.random(in: 0 ... 0x3BFF) }
            }
            features[name] = MLFeatureValue(multiArray: array)
        }
        return try MLDictionaryFeatureProvider(dictionary: features)
    }
}

private extension Duration {
    var ms: Double { Double(components.seconds) * 1000 + Double(components.attoseconds) / 1e15 }
}
#endif
