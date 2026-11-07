/*
 Vergence - runs Smooth Motion's flow model on a whole 2D picture

 The 2D half of the producer: a flat title (or one shown as AI 3D) has no
 second eye, so a worker spends its time on a larger flow network instead,
 RIFE 4.25 lite at 960×576 rather than the 3D workload's 480×288 (13.5 ms on
 an M2 Max's GPU, where a Vision Pro's Neural Engine needs 48). The records
 are the same `flowPair` kind, eye mono, stored a quarter of the model's
 size (240×144), so the player's shared-field pass reads them the way it
 reads a 3D track's.

 The model's size is read from its own description, so any RIFE conversion
 by Hypnos's `scripts/convert-interp-model.py` works; one call at a time.
 */

#if canImport(CoreML) && canImport(Compression) && canImport(Accelerate)
import Accelerate
import CoreML
import Foundation
import VergenceCore

public final class MonoFlowAnalyzer: @unchecked Sendable {
    private let model: MLModel
    public let modelName: String
    /// The model's working size.
    public let width: Int
    public let height: Int

    /// - Parameter flowModel: RIFE (`img0`, `img1` 1×3×H×W, `timestep`; `flow`, `mask`).
    public init(flowModel: URL, computeUnits: MLComputeUnits = .all) async throws {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        model = try MLModel(contentsOf: try await InterpolationModelBenchmark.compiledURL(for: flowModel),
                            configuration: config)
        modelName = flowModel.deletingPathExtension().lastPathComponent
        guard let shape = model.modelDescription.inputDescriptionsByName["img0"]?.multiArrayConstraint?.shape
            .map(\.intValue), shape.count == 4, shape[0] == 1, shape[1] == 3 else {
            throw StereoFlowAnalyzer.Error.unexpectedModel("img0 isn't 1×3×H×W")
        }
        height = shape[2]
        width = shape[3]
        // The first prediction specialises the model for the device; pay it here.
        let grey = [Float](repeating: 0.5, count: 3 * width * height)
        _ = try flow(from: grey, to: grey)
    }

    /// The profile name a track's header gives this model ("RIFE425lite_960x576 960x576").
    public var profileName: String { "\(modelName) \(width)x\(height)" }

    /// The motion between two frames at t = ½.
    /// - Parameters: a, b: planar RGB 0…1 at the model's size (`SharedFieldAnalysis.rgbPlanar`).
    /// - Returns: F½→0 (x, y), F½→1 (x, y) planar, and frame 0's blend weight, at the model's size.
    public func flow(from a: [Float], to b: [Float]) throws -> (flow: [Float], mask: [Float]) {
        try flow(from: try input(a), to: try input(b))
    }

    /// A picture as the model's input tensor, once: a worker hands each
    /// frame's to two pairs (as `b`, then as `a`).
    public func input(_ planar: [Float]) throws -> MLMultiArray {
        let shape = [1, 3, height, width]
        let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float16)
        precondition(planar.count == array.count, "tensor size")
        let contiguous = array.strides.map(\.intValue) == [3 * width * height, width * height, width, 1]
        guard contiguous else { return try StereoFlowAnalyzer.halfArray(shape, values: planar) }
        // Accelerate converts 1.6 M samples in well under a millisecond; one at a time took ~10.
        array.withUnsafeMutableBytes { raw, _ in
            planar.withUnsafeBufferPointer { source in
                var from = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: source.baseAddress!), height: 1,
                                         width: vImagePixelCount(planar.count), rowBytes: planar.count * 4)
                var to = vImage_Buffer(data: raw.baseAddress!, height: 1, width: vImagePixelCount(planar.count),
                                       rowBytes: planar.count * 2)
                vImageConvert_PlanarFtoPlanar16F(&from, &to, vImage_Flags(kvImageNoFlags))
            }
        }
        return array
    }

    public func flow(from a: MLMultiArray, to b: MLMultiArray) throws -> (flow: [Float], mask: [Float]) {
        let inputs = try MLDictionaryFeatureProvider(dictionary: [
            "img0": MLFeatureValue(multiArray: a),
            "img1": MLFeatureValue(multiArray: b),
            "timestep": MLFeatureValue(multiArray: timestep),
        ])
        let out = try model.prediction(from: inputs)
        guard let flow = out.featureValue(for: "flow")?.multiArrayValue,
              let mask = out.featureValue(for: "mask")?.multiArrayValue else {
            throw StereoFlowAnalyzer.Error.unexpectedModel("flow model has no flow/mask output")
        }
        return (Self.floats(flow), Self.floats(mask))
    }

    private lazy var timestep: MLMultiArray = try! StereoFlowAnalyzer.halfArray([1, 1, 1, 1], values: [0.5])

    /// An output as floats; half-float contiguous ones through Accelerate.
    private static func floats(_ array: MLMultiArray) -> [Float] {
        let shape = array.shape.map(\.intValue)
        var strides = [Int](repeating: 1, count: shape.count)
        for i in stride(from: shape.count - 2, through: 0, by: -1) { strides[i] = strides[i + 1] * shape[i + 1] }
        guard array.dataType == .float16, array.strides.map(\.intValue) == strides else {
            return StereoFlowAnalyzer.floats(array)
        }
        var out = [Float](repeating: 0, count: array.count)
        array.withUnsafeBytes { raw in
            out.withUnsafeMutableBufferPointer { target in
                var from = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: raw.baseAddress!), height: 1,
                                         width: vImagePixelCount(target.count), rowBytes: target.count * 2)
                var to = vImage_Buffer(data: target.baseAddress!, height: 1, width: vImagePixelCount(target.count),
                                       rowBytes: target.count * 4)
                vImageConvert_Planar16FtoPlanarF(&from, &to, vImage_Flags(kvImageNoFlags))
            }
        }
        return out
    }
}
#endif
