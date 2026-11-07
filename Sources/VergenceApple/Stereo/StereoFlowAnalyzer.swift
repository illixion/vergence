/*
 Vergence - runs Smooth Motion's flow and disparity models with Core ML

 The model half of the producer (`SharedFieldAnalysis` in VergenceCore is the
 rest): RIFE 4.25 lite at 480×288 for the left eye's motion between two real
 frames, NeuFlow v2 at 512×288 for each real pair's right→left disparity. Both
 are conversions by Hypnos's `scripts/convert-interp-model.py`; their inputs
 are fixed-size float16 tensors, read from each model's own description, so a
 model of a different size is refused rather than fed the wrong shape.

 `flow` and `disparity` may run at the same time (different models, often on
 different units: RIFE on the Neural Engine, an fp32 NeuFlow on the GPU);
 neither may be called from two threads at once.
 */

#if canImport(CoreML) && canImport(Compression)
import Compression
import CoreML
import Foundation
import VergenceCore

public final class StereoFlowAnalyzer: @unchecked Sendable {
    public enum Error: Swift.Error, CustomStringConvertible {
        case unexpectedModel(String)

        public var description: String {
            switch self {
            case .unexpectedModel(let detail): "unexpected model: \(detail)"
            }
        }
    }

    private let flowModel: MLModel
    private let disparityModel: MLModel
    public let flowModelName: String
    public let disparityModelName: String

    /// - Parameters:
    ///   - flowModel: RIFE at 480×288 (`img0`, `img1` 1×3×288×480, `timestep`; `flow`, `mask`).
    ///   - disparityModel: NeuFlow at 512×288, both directions (`img0`, `img1` 1×3×288×512; `flow` 2×2×288×512).
    public init(flowModel: URL, disparityModel: URL, computeUnits: MLComputeUnits = .all) async throws {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        self.flowModel = try MLModel(contentsOf: try await InterpolationModelBenchmark.compiledURL(for: flowModel),
                                     configuration: config)
        self.disparityModel = try MLModel(
            contentsOf: try await InterpolationModelBenchmark.compiledURL(for: disparityModel), configuration: config)
        flowModelName = flowModel.deletingPathExtension().lastPathComponent
        disparityModelName = disparityModel.deletingPathExtension().lastPathComponent
        let f = SharedFieldAnalysis.flowModelSize, d = SharedFieldAnalysis.disparityModelSize
        try Self.expect(self.flowModel, input: "img0", shape: [1, 3, f.height, f.width])
        try Self.expect(self.disparityModel, input: "img0", shape: [1, 3, d.height, d.width])
        // The first prediction specialises the model for the device (seconds on the Neural Engine);
        // pay it here, not on the first frame.
        let grey = [Float](repeating: 0.5, count: 3 * f.width * f.height)
        _ = try flow(from: grey, to: grey)
        let greyD = [Float](repeating: 0.5, count: 3 * d.width * d.height)
        _ = try disparity(left: greyD, right: greyD)
    }

    private static func expect(_ model: MLModel, input: String, shape: [Int]) throws {
        let actual = model.modelDescription.inputDescriptionsByName[input]?.multiArrayConstraint?.shape.map(\.intValue)
        guard actual == shape else {
            throw Error.unexpectedModel("\(input) is \(actual.map { "\($0)" } ?? "missing"), wanted \(shape)")
        }
    }

    /// The left eye's motion between two real frames, at t = ½.
    /// - Parameters: a, b: planar RGB 0…1 at 480×288 (`SharedFieldAnalysis.rgbPlanar`).
    /// - Returns: F½→0 (x, y), F½→1 (x, y) planar, and frame 0's blend weight, at 480×288.
    public func flow(from a: [Float], to b: [Float]) throws -> (flow: [Float], mask: [Float]) {
        let f = SharedFieldAnalysis.flowModelSize
        let shape = [1, 3, f.height, f.width]
        let timestep = try Self.halfArray([1, 1, 1, 1], values: [0.5])
        let inputs = try MLDictionaryFeatureProvider(dictionary: [
            "img0": MLFeatureValue(multiArray: try Self.halfArray(shape, values: a)),
            "img1": MLFeatureValue(multiArray: try Self.halfArray(shape, values: b)),
            "timestep": MLFeatureValue(multiArray: timestep),
        ])
        let out = try flowModel.prediction(from: inputs)
        guard let flow = out.featureValue(for: "flow")?.multiArrayValue,
              let mask = out.featureValue(for: "mask")?.multiArrayValue else {
            throw Error.unexpectedModel("flow model has no flow/mask output")
        }
        return (Self.floats(flow), Self.floats(mask))
    }

    /// Right→left disparity of one real pair (x, y planar at 512×288), the
    /// horizontal channel first.
    /// - Parameters: left, right: planar RGB 0…1 at 512×288.
    public func disparity(left: [Float], right: [Float]) throws -> [Float] {
        let d = SharedFieldAnalysis.disparityModelSize
        let shape = [1, 3, d.height, d.width]
        // NeuFlow takes 0…255. img0 → img1 is batch 0: right → left.
        let inputs = try MLDictionaryFeatureProvider(dictionary: [
            "img0": MLFeatureValue(multiArray: try Self.halfArray(shape, values: right, scale: 255)),
            "img1": MLFeatureValue(multiArray: try Self.halfArray(shape, values: left, scale: 255)),
        ])
        let out = try disparityModel.prediction(from: inputs)
        guard let flow = out.featureValue(for: "flow")?.multiArrayValue else {
            throw Error.unexpectedModel("disparity model has no flow output")
        }
        let all = Self.floats(flow)
        return Array(all[0 ..< 2 * d.width * d.height])
    }

    // MARK: - Half-float tensors (bit patterns: `Float16` doesn't exist on Intel Macs)

    static func halfArray(_ shape: [Int], values: [Float], scale: Float = 1) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float16)
        precondition(values.count == array.count, "tensor size")
        let strides = array.strides.map(\.intValue)
        array.withUnsafeMutableBytes { raw, _ in
            let out = raw.bindMemory(to: UInt16.self)
            if strides == contiguousStrides(shape) {
                for i in values.indices { out[i] = HalfFloat.bits(from: values[i] * scale) }
            } else {
                forEachIndex(shape) { linear, index in
                    out[zip(index, strides).reduce(0) { $0 + $1.0 * $1.1 }] = HalfFloat.bits(from: values[linear] * scale)
                }
            }
        }
        return array
    }

    /// Any output as floats in logical (row-major) order, whatever its strides and type.
    public static func floats(_ array: MLMultiArray) -> [Float] {
        let shape = array.shape.map(\.intValue)
        let strides = array.strides.map(\.intValue)
        var out = [Float](repeating: 0, count: array.count)
        array.withUnsafeBytes { raw in
            if strides == contiguousStrides(shape) {
                switch array.dataType {
                case .float16:
                    let p = raw.bindMemory(to: UInt16.self)
                    for i in 0 ..< out.count { out[i] = HalfFloat.value(from: p[i]) }
                    return
                case .float32:
                    let p = raw.bindMemory(to: Float.self)
                    for i in 0 ..< out.count { out[i] = p[i] }
                    return
                default: break
                }
            }
            let read: (Int) -> Float
            switch array.dataType {
            case .float16:
                let p = raw.bindMemory(to: UInt16.self)
                read = { HalfFloat.value(from: p[$0]) }
            case .float32:
                let p = raw.bindMemory(to: Float.self)
                read = { p[$0] }
            default:
                read = { _ in .nan }
            }
            forEachIndex(shape) { linear, index in
                out[linear] = read(zip(index, strides).reduce(0) { $0 + $1.0 * $1.1 })
            }
        }
        return out
    }

    private static func contiguousStrides(_ shape: [Int]) -> [Int] {
        var strides = [Int](repeating: 1, count: shape.count)
        for i in stride(from: shape.count - 2, through: 0, by: -1) { strides[i] = strides[i + 1] * shape[i + 1] }
        return strides
    }

    private static func forEachIndex(_ shape: [Int], _ body: (Int, [Int]) -> Void) {
        var index = [Int](repeating: 0, count: shape.count)
        let count = shape.reduce(1, *)
        for linear in 0 ..< count {
            body(linear, index)
            var d = shape.count - 1
            while d >= 0 {
                index[d] += 1
                if index[d] < shape[d] { break }
                index[d] = 0
                d -= 1
            }
        }
    }
}

// MARK: - DEFLATE for analysis records

public enum AnalysisDeflate {
    /// Raw DEFLATE (RFC 1951), Apple's `COMPRESSION_ZLIB`.
    public static func deflate(_ data: Data) throws -> Data {
        let capacity = data.count + data.count / 2 + 1024
        var out = Data(count: capacity)
        let written = out.withUnsafeMutableBytes { dst in
            data.withUnsafeBytes { src in
                compression_encode_buffer(dst.bindMemory(to: UInt8.self).baseAddress!, capacity,
                                          src.bindMemory(to: UInt8.self).baseAddress!, data.count, nil,
                                          COMPRESSION_ZLIB)
            }
        }
        guard written > 0 else { throw AnalysisTrackError.payloadSize }
        out.count = written
        return out
    }

    public static func inflate(_ data: Data, size: Int) throws -> Data {
        var out = Data(count: size)
        let written = out.withUnsafeMutableBytes { dst in
            data.withUnsafeBytes { src in
                compression_decode_buffer(dst.bindMemory(to: UInt8.self).baseAddress!, size,
                                          src.bindMemory(to: UInt8.self).baseAddress!, data.count, nil,
                                          COMPRESSION_ZLIB)
            }
        }
        guard written == size else { throw AnalysisTrackError.payloadSize }
        return out
    }
}

extension AnalysisRecord {
    public func deflated() throws -> AnalysisRecord { try deflated(using: AnalysisDeflate.deflate) }
    public func inflated() throws -> AnalysisRecord { try inflated(using: AnalysisDeflate.inflate) }
}
#endif
