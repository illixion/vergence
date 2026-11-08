#if canImport(Metal) && canImport(VideoToolbox) && !targetEnvironment(simulator)
import CoreVideo
import Foundation
import QuartzCore
import VideoToolbox

/// How VideoToolbox's low-latency super resolution (`VTRealtimeUpscaler`)
/// does on this device, away from playback: what it takes (factors and their
/// input limits), how long each size's model takes to load, and the wall time
/// a frame takes with both format conversions (what `UpscalingFrameSource`'s
/// budget counts), over textured synthetic frames. Its cost doesn't depend
/// on the picture.
///
/// On an M2 Max (2026-11-08): 640×360 → 1280×720 0.7 ms, 960×540 → 1080p
/// 1.5 ms, 1280×720 → 1080p 2.5 ms, after a 0.5-2.3 s model load per size.
public enum RealtimeUpscalerBenchmark {
    public struct Limit: Codable, Sendable {
        public var factor: Float
        public var maxDimension: Int?
        public var maxPixels: Int?
    }

    public struct Result: Codable, Sendable {
        public var width: Int
        public var height: Int
        public var factor: Float?
        public var outputWidth: Int?
        public var outputHeight: Int?
        public var loadMs: Double?
        public var iterations: Int
        public var medianMs: Double?
        public var p90Ms: Double?
        public var bestMs: Double?
        public var error: String?
    }

    public struct Report: Codable, Sendable {
        public var supported: Bool
        public var scaleFactors: [Float]
        public var limits: [Limit]
        public var results: [Result]
    }

    public static let defaultSizes: [(width: Int, height: Int)] = [(640, 360), (854, 480), (960, 540), (1280, 720)]

    /// Each size in `sizes` at the factor playback would use, after a warm-up.
    public static func run(sizes: [(width: Int, height: Int)] = defaultSizes, iterations: Int = 120) async -> Report {
        await Task.detached(priority: .userInitiated) {
            guard #available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *) else {
                return Report(supported: false, scaleFactors: [], limits: [], results: [])
            }
            let supported = VTRealtimeUpscaler.isSupported
            // The list and the limits exist only in the version 27 SDK
            // (Swift 6.4) and OS, as for the interpolator.
            var factors: [Float] = []
            #if compiler(>=6.4)
            if supported, #available(macOS 27.0, iOS 27.0, tvOS 27.0, visionOS 27.0, *) {
                factors = VTLowLatencySuperResolutionScalerConfiguration.supportedScaleFactors
            }
            #endif
            var report = Report(supported: supported, scaleFactors: factors, limits: limits(factors), results: [])
            guard supported else { return report }
            for size in sizes {
                report.results.append(measure(width: size.width, height: size.height, iterations: iterations))
            }
            return report
        }.value
    }

    @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *)
    private static func limits(_ factors: [Float]) -> [Limit] {
        factors.map { factor in
            var limit = Limit(factor: factor)
            #if compiler(>=6.4)
            if #available(macOS 27.0, iOS 27.0, tvOS 27.0, visionOS 27.0, *) {
                limit.maxDimension = VTLowLatencySuperResolutionScalerConfiguration.maximumDimension(forSpatialScaleFactor: factor)
                limit.maxPixels = VTLowLatencySuperResolutionScalerConfiguration.maximumPixelCount(forSpatialScaleFactor: factor)
            }
            #endif
            return limit
        }
    }

    @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *)
    static func measure(width: Int, height: Int, iterations: Int) -> Result {
        var result = Result(width: width, height: height, iterations: iterations)
        let upscaler = VTRealtimeUpscaler()
        guard let factor = VTRealtimeUpscaler.factor(width: width, height: height),
              let output = upscaler.outputSize(width: width, height: height) else {
            result.error = "no factor fits 1080p at this size"
            return result
        }
        result.factor = factor
        result.outputWidth = output.width
        result.outputHeight = output.height
        let loadStarted = CACurrentMediaTime()
        guard upscaler.prepare(width: width, height: height) else {
            result.error = "the processor didn't start"
            return result
        }
        result.loadMs = (CACurrentMediaTime() - loadStarted) * 1000
        guard let frame = FlowInterpolationBenchmark.pattern(width: width, height: height, shift: (0, 0)) else {
            result.error = "couldn't make a \(width)x\(height) frame"
            return result
        }
        var times: [Double] = []
        for index in 0 ..< iterations + 10 {
            let done = DispatchSemaphore(value: 0)
            nonisolated(unsafe) var ms = Double.nan
            nonisolated(unsafe) var ok = false
            upscaler.upscale(frame) { out, taken in
                ok = out != nil
                ms = taken
                done.signal()
            }
            done.wait()
            guard ok else {
                result.error = "upscaling failed"
                return result
            }
            if index >= 10 { times.append(ms) }
        }
        let sorted = times.sorted()
        result.medianMs = sorted[sorted.count / 2]
        result.p90Ms = sorted[min(sorted.count - 1, Int(Double(sorted.count) * 0.9))]
        result.bestMs = sorted.first
        VergenceLog.pipeline.info(
            "[upscale-bench] \(width)×\(height) → \(output.width)×\(output.height): load \(result.loadMs ?? 0, format: .fixed(precision: 0), privacy: .public) ms, median \(result.medianMs ?? 0, format: .fixed(precision: 2), privacy: .public) ms, p90 \(result.p90Ms ?? 0, format: .fixed(precision: 2), privacy: .public)"
        )
        return result
    }
}
#endif
