#if os(macOS)
import Foundation
import Metal
import Testing
@testable import VergenceApple

/// Opt-in replay of recorded captures through the real shaders, for measuring the layered
/// silhouettes against the connected mesh (docs/investigations/ai-3d-coat-stairstepping-2026-10-29.md).
///
/// `VERGENCE_LAYER_LAB` is one capture directory (`source.png`, `depth.f32`, `depth.json`) or a
/// directory of them; `VERGENCE_LAYER_OUTPUT` a new directory that gets one folder per capture
/// holding `left.png` and `right.png`, the layout `scripts/measure-stereo-matte.py` reads.
/// Optional: `VERGENCE_LAYER_MESH=1` also writes `mesh-left.png` and `mesh-right.png`, to check
/// that this harness reproduces the recorded `connected-*.png`; `VERGENCE_LAYER_GRID=moderate`
/// renders on the 193×109 grid instead of the dense one; `VERGENCE_LAYER_DEBUG=1` writes
/// the decomposition (trust, reach, alpha, layer depths, both layers); and
/// `VERGENCE_LAYER_TUNING` overrides thresholds without a rebuild, as `name=lo,hi` pairs joined
/// by `;` (`seed`, `support`, `contrast`, `spread`, `separation`; `radius`, `sigma` and `kernel`
/// take one value each).
@Suite("Layered silhouette lab", .serialized)
struct StereoLayeredLabTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["VERGENCE_LAYER_LAB"] != nil))
    func replayCaptures() throws {
        let env = ProcessInfo.processInfo.environment
        let input = URL(fileURLWithPath: try #require(env["VERGENCE_LAYER_LAB"]))
        let output = URL(fileURLWithPath: try #require(env["VERGENCE_LAYER_OUTPUT"]))
        try #require(!FileManager.default.fileExists(atPath: output.path), "Choose a new output directory")
        let manager = FileManager.default
        let captures = manager.fileExists(atPath: input.appendingPathComponent("source.png").path)
            ? [input]
            : try manager.contentsOfDirectory(at: input, includingPropertiesForKeys: nil)
                .filter {
                    manager.fileExists(atPath: $0.appendingPathComponent("source.png").path)
                        && manager.fileExists(atPath: $0.appendingPathComponent("depth.json").path)
                }
                .sorted { $0.lastPathComponent < $1.lastPathComponent }
        try #require(!captures.isEmpty, "No captures with a source.png and depth.json under \(input.path)")

        let bench = try LayeredBench()
        if let overrides = env["VERGENCE_LAYER_TUNING"] {
            try Self.apply(overrides, to: bench.layered)
        }
        let writesMesh = env["VERGENCE_LAYER_MESH"] == "1", writesDebug = env["VERGENCE_LAYER_DEBUG"] == "1"
        // Refined and cached depth both ask the pump for the dense grid, so that is what a capture shows.
        let dense = env["VERGENCE_LAYER_GRID"] != "moderate"
        try manager.createDirectory(at: output, withIntermediateDirectories: true)

        for capture in captures {
            let picture = try LayeredFiles.loadCapture(capture)
            let info = try JSONSerialization.jsonObject(
                with: Data(contentsOf: capture.appendingPathComponent("depth.json"))) as? [String: Any] ?? [:]
            let strength = try #require((info["strength"] as? NSNumber)?.floatValue)
            let convergence = try #require((info["convergence"] as? NSNumber)?.floatValue)
            let mirror = info["mirror"] as? Bool ?? false
            let levels = info["tone"] as? [NSNumber]
            let tone = levels?.count == 3
                ? SIMD3(levels![0].floatValue, levels![1].floatValue, levels![2].floatValue) : SIMD3<Float>(0, 1, 1)
            let frameDense = env["VERGENCE_LAYER_GRID"] == nil ? info["denseGrid"] as? Bool ?? dense : dense
            let folder = captures.count == 1 && input == capture ? output : output.appendingPathComponent(capture.lastPathComponent)
            try manager.createDirectory(at: folder, withIntermediateDirectories: true)

            let layered = try bench.render(picture, strength: strength, convergence: convergence,
                                           mirror: mirror, layers: true, dense: frameDense, tone: tone)
            try LayeredFiles.writePNG(bgra: layered.left, width: layered.width, height: layered.height,
                                      to: folder.appendingPathComponent("left.png"))
            try LayeredFiles.writePNG(bgra: layered.right, width: layered.width, height: layered.height,
                                      to: folder.appendingPathComponent("right.png"))
            var meshMilliseconds: Double?
            if writesMesh {
                let mesh = try bench.render(picture, strength: strength, convergence: convergence,
                                            mirror: mirror, layers: false, dense: frameDense, tone: tone)
                meshMilliseconds = mesh.gpuMilliseconds
                try LayeredFiles.writePNG(bgra: mesh.left, width: mesh.width, height: mesh.height,
                                          to: folder.appendingPathComponent("mesh-left.png"))
                try LayeredFiles.writePNG(bgra: mesh.right, width: mesh.width, height: mesh.height,
                                          to: folder.appendingPathComponent("mesh-right.png"))
            }
            if writesDebug, let layers = bench.layered.layers {
                try Self.writeDebug(layers, bench: bench, to: folder)
            }
            // Both eyes and the split on this Mac's GPU, the first capture including warm-up: an
            // indication of cost, not a headset figure.
            let timing = String(format: "gpu %.2f ms layered", layered.gpuMilliseconds)
                + (meshMilliseconds.map { String(format: ", %.2f ms mesh alone", $0) } ?? "")
            print("layer lab: \(capture.lastPathComponent) strength \(strength) convergence \(convergence), \(timing)")
        }
    }

    private static func apply(_ overrides: String, to warp: StereoLayeredWarp) throws {
        for entry in overrides.split(separator: ";") {
            let parts = entry.split(separator: "=")
            let values = parts.count == 2 ? parts[1].split(separator: ",").compactMap { Float($0) } : []
            guard !values.isEmpty, values.count <= 2 else {
                throw LayeredBenchError(description: "VERGENCE_LAYER_TUNING entry '\(entry)' is not name=value or name=lo,hi")
            }
            let pair = SIMD2<Float>(values[0], values.last ?? values[0])
            switch parts[0] {
            case "seed": warp.tuning.seedSignificance = pair
            case "support": warp.tuning.support = pair
            case "contrast": warp.tuning.contrast = pair
            case "spread": warp.tuning.spread = pair
            case "separation": warp.tuning.separation = pair
            case "radius": warp.tuning.seedRadius = Int(values[0])
            case "sigma": warp.tuning.sigmaRatio = values[0]
            case "kernel": warp.tuning.kernelSigmas = values[0]
            default: throw LayeredBenchError(description: "unknown tuning '\(parts[0])'")
            }
        }
    }

    private static func writeDebug(_ layers: StereoLayeredWarp.Layers, bench: LayeredBench,
                                   to folder: URL) throws {
        func plane(_ texture: MTLTexture, _ channel: Int, _ name: String) throws {
            try LayeredFiles.writePlane(try bench.plane(of: texture, channel: channel),
                                        width: texture.width, height: texture.height,
                                        to: folder.appendingPathComponent("debug-\(name).png"))
        }
        func color(_ texture: MTLTexture, _ name: String) throws {
            let (r, g, b) = (try bench.plane(of: texture, channel: 0), try bench.plane(of: texture, channel: 1),
                             try bench.plane(of: texture, channel: 2))
            var bytes = [UInt8](repeating: 255, count: texture.width * texture.height * 4)
            func level(_ v: Float) -> UInt8 { UInt8(max(0, min(1, v)) * 255 + 0.5) }
            for i in 0..<(texture.width * texture.height) {
                bytes[i * 4] = level(b[i])
                bytes[i * 4 + 1] = level(g[i])
                bytes[i * 4 + 2] = level(r[i])
            }
            try LayeredFiles.writePNG(bgra: bytes, width: texture.width, height: texture.height,
                                      to: folder.appendingPathComponent("debug-\(name).png"))
        }
        try plane(layers.trust, 0, "trust")
        try plane(layers.reach, 0, "reach")
        try plane(layers.seeds, 0, "seed-near")
        try plane(layers.seeds, 1, "seed-far")
        try plane(layers.nearLayer, 3, "alpha")
        try plane(layers.farLayer, 3, "trust-full")
        try plane(layers.depth, 0, "depth-near")
        try plane(layers.depth, 1, "depth-far")
        try color(layers.nearLayer, "near-premultiplied")
        try color(layers.farLayer, "far")
    }
}
#endif
