/*
 Vergence - the upscaling network, built from its published weights

 Real-ESRGAN's compact video network (SRVGGNetCompact: xinntao/Real-ESRGAN,
 BSD-3-Clause), as `realesr-general-x4v3.pth` and `realesr-animevideov3.pth`
 hold it: a 3×3 convolution and a per-channel PReLU, `n` more of each, a last
 convolution to 3·s² channels, a pixel shuffle by `s` (4), plus the input
 enlarged by nearest neighbour.

 Vergence ships no weights and no converted model. The weights are read from
 the authors' own `.pth` (`TorchCheckpoint`), and the Core ML model is written
 here as a NeuralNetwork specification (the protobuf `coremltools` would
 write for the same network, layer for layer) and compiled on this Mac, once
 per input size. The output is 2×: the 4× network's picture averaged over 2×2
 pixels, which is cleaner than the 2× RRDB network and far cheaper.
 */

#if canImport(CoreML)
import CoreML
import CryptoKit
import Foundation
import VergenceCore

public struct UpscaleNetwork: Sendable {
    public struct Failure: Error, CustomStringConvertible {
        public let description: String
        init(_ text: String) { description = text }
    }

    struct Layer: Sendable {
        var weight: [Float]
        var bias: [Float]
        var outputs: Int
        var inputs: Int
        /// The PReLU after it (none after the last).
        var slopes: [Float]?
    }

    let layers: [Layer]
    /// The network's own enlargement (4 for the x4 models).
    public let scale: Int
    /// A digest of the weights, naming the compiled models.
    public let digest: String
    /// The checkpoint's file name without its extension.
    public let name: String

    /// The convolutions it runs, for logging.
    public var convolutions: Int { layers.count }

    public init(checkpoint url: URL) throws {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        let checkpoint = try TorchCheckpoint(data: data)
        let tensors = checkpoint.tensors()
        // body.N.weight: 4-D for a convolution, 1-D for the PReLU after it.
        var indices = tensors.keys.compactMap { key -> Int? in
            let parts = key.split(separator: ".")
            guard parts.count == 3, parts[0] == "body", parts[2] == "weight" else { return nil }
            return Int(parts[1])
        }
        indices.sort()
        guard !indices.isEmpty, indices == Array(0 ..< indices.count) else {
            throw Failure("\(url.lastPathComponent) isn't a Real-ESRGAN compact network (SRVGGNetCompact): "
                + "realesr-general-x4v3 or realesr-animevideov3 are; the RRDB networks (RealESRGAN_x4plus) are too slow for video")
        }
        var layers: [Layer] = []
        for i in indices {
            guard let w = tensors["body.\(i).weight"] else { continue }
            if w.shape.count == 4 {
                guard w.shape[2] == 3, w.shape[3] == 3, let b = tensors["body.\(i).bias"], b.count == w.shape[0] else {
                    throw Failure("body.\(i) isn't a 3×3 convolution with a bias")
                }
                let inputs = layers.last?.outputs ?? 3
                guard w.shape[1] == inputs else { throw Failure("body.\(i) takes \(w.shape[1]) channels, not \(inputs)") }
                layers.append(Layer(weight: w.values, bias: b.values, outputs: w.shape[0], inputs: inputs, slopes: nil))
            } else if w.shape.count == 1, !layers.isEmpty, layers[layers.count - 1].slopes == nil,
                      w.count == layers[layers.count - 1].outputs || w.count == 1 {
                let channels = layers[layers.count - 1].outputs
                layers[layers.count - 1].slopes = w.count == 1 ? Array(repeating: w.values[0], count: channels) : w.values
            } else {
                throw Failure("body.\(i) is neither a convolution nor a PReLU")
            }
        }
        guard layers.count >= 2, let last = layers.last, last.slopes == nil,
              layers.dropLast().allSatisfy({ $0.slopes != nil }) else {
            throw Failure("\(url.lastPathComponent): the layers aren't convolution-PReLU pairs ending in a convolution")
        }
        let scale = Int(Double(last.outputs / 3).squareRoot().rounded())
        guard scale >= 2, scale * scale * 3 == last.outputs, layers[0].inputs == 3 else {
            throw Failure("\(url.lastPathComponent): the last convolution's \(last.outputs) channels aren't 3·s²")
        }
        self.layers = layers
        self.scale = scale
        self.digest = SHA256.hash(data: data).prefix(8).map { String(format: "%02x", $0) }.joined()
        self.name = url.deletingPathExtension().lastPathComponent
    }

    /// The Core ML model for `batch` pictures of `width`×`height`: input
    /// "image" [batch, 3, height, width], output "upscaled" [batch, 3,
    /// 2·height, 2·width], both float16, RGB 0…1.
    func specification(batch: Int, width: Int, height: Int) -> Data {
        var model = ProtoWriter()
        model.uint(1, 7) // specificationVersion: float16 multiarrays
        model.message(2) { description in
            for (field, name, w, h) in [(1, "image", width, height), (10, "upscaled", width * 2, height * 2)] {
                description.message(field) { feature in
                    feature.string(1, name)
                    feature.message(3) { type in
                        type.message(5) { array in
                            array.packed(1, [batch, 3, h, w].map(UInt64.init))
                            array.uint(2, 65552) // FLOAT16
                        }
                    }
                }
            }
        }
        model.message(500) { network in
            var input = "image"
            for (i, layer) in layers.enumerated() {
                let conv = "conv\(i)"
                network.message(1) { l in
                    l.string(1, conv)
                    l.string(2, input)
                    l.string(3, conv)
                    l.message(100) { c in
                        c.uint(1, UInt64(layer.outputs))
                        c.uint(2, UInt64(layer.inputs))
                        c.uint(10, 1)
                        c.packed(20, [3, 3])
                        c.packed(30, [1, 1])
                        c.packed(40, [1, 1])
                        c.message(51) { _ in } // same padding
                        c.bool(70, true)
                        c.message(90) { $0.bytes(2, halves(layer.weight)) }
                        c.message(91) { $0.bytes(2, halves(layer.bias)) }
                    }
                }
                input = conv
                if let slopes = layer.slopes {
                    let act = "prelu\(i)"
                    network.message(1) { l in
                        l.string(1, act)
                        l.string(2, input)
                        l.string(3, act)
                        l.message(130) { a in a.message(25) { p in p.message(1) { $0.bytes(2, halves(slopes)) } } }
                    }
                    input = act
                }
            }
            network.message(1) { l in // pixel shuffle
                l.string(1, "shuffle"); l.string(2, input); l.string(3, "shuffle")
                l.message(345) { r in r.uint(1, 2); r.uint(2, UInt64(scale)) }
            }
            network.message(1) { l in // the input, enlarged
                l.string(1, "base"); l.string(2, "image"); l.string(3, "base")
                l.message(210) { u in u.packed(1, [UInt64(scale), UInt64(scale)]); u.uint(5, 0) }
            }
            network.message(1) { l in
                l.string(1, "sum"); l.string(2, "shuffle"); l.string(2, "base"); l.string(3, "sum")
                l.message(230) { _ in }
            }
            let pool = scale / 2
            network.message(1) { l in // to 2×
                l.string(1, "upscaled"); l.string(2, "sum"); l.string(3, "upscaled")
                l.message(120) { p in
                    p.uint(1, 1) // AVERAGE
                    p.packed(10, [UInt64(pool), UInt64(pool)])
                    p.packed(20, [UInt64(pool), UInt64(pool)])
                    p.message(30) { _ in } // valid
                }
            }
            network.uint(5, 1) // EXACT_ARRAY_MAPPING
        }
        return model.data
    }

    /// The compiled model for that size, from the cache or compiled now.
    public func compiledModel(batch: Int, width: Int, height: Int) async throws -> URL {
        let cache = Self.cacheDirectory
        let compiled = cache.appendingPathComponent("\(name)-\(digest)-v1-\(batch)x\(width)x\(height).mlmodelc")
        if FileManager.default.fileExists(atPath: compiled.path) { return compiled }
        try FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
        let spec = FileManager.default.temporaryDirectory.appendingPathComponent("vergence-upscale-\(UUID().uuidString).mlmodel")
        try specification(batch: batch, width: width, height: height).write(to: spec)
        defer { try? FileManager.default.removeItem(at: spec) }
        let made = try await MLModel.compileModel(at: spec)
        do {
            try FileManager.default.moveItem(at: made, to: compiled)
        } catch {
            // Made meanwhile by another run (or test) compiling the same size.
            try? FileManager.default.removeItem(at: made)
            guard FileManager.default.fileExists(atPath: compiled.path) else { throw error }
        }
        return compiled
    }

    static var cacheDirectory: URL {
        FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Vergence/UpscaleModels", isDirectory: true)
    }
}

private func halves(_ values: [Float]) -> Data {
    var data = Data(capacity: values.count * 2)
    for v in values {
        let bits = HalfFloat.bits(from: v)
        data.append(UInt8(bits & 0xFF))
        data.append(UInt8(bits >> 8))
    }
    return data
}

/// Protocol buffers, written: just what a Core ML specification needs.
struct ProtoWriter {
    var data = Data()

    mutating func varint(_ value: UInt64) {
        var v = value
        while v >= 0x80 {
            data.append(UInt8(v & 0x7F) | 0x80)
            v >>= 7
        }
        data.append(UInt8(v))
    }

    mutating func key(_ field: Int, _ wire: UInt64) { varint(UInt64(field) << 3 | wire) }

    mutating func uint(_ field: Int, _ value: UInt64) {
        key(field, 0)
        varint(value)
    }

    mutating func bool(_ field: Int, _ value: Bool) { uint(field, value ? 1 : 0) }

    mutating func bytes(_ field: Int, _ value: Data) {
        key(field, 2)
        varint(UInt64(value.count))
        data.append(value)
    }

    mutating func string(_ field: Int, _ value: String) { bytes(field, Data(value.utf8)) }

    mutating func packed(_ field: Int, _ values: [UInt64]) {
        var inner = ProtoWriter()
        for v in values { inner.varint(v) }
        bytes(field, inner.data)
    }

    mutating func message(_ field: Int, _ build: (inout ProtoWriter) -> Void) {
        var inner = ProtoWriter()
        build(&inner)
        bytes(field, inner.data)
    }
}
#endif
