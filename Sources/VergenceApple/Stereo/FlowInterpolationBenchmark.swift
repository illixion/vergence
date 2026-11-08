#if canImport(Metal)
import CoreVideo
import Foundation
import Metal
import QuartzCore
import VergenceCore

/// How long `FlowMidFrameInterpolator` takes on this device, away from
/// playback: GPU time (the command buffer's own start and end) and wall time
/// (commit to completion, so queueing behind other GPU work counts), over
/// textured synthetic frames moved a few pixels apart. Its cost doesn't
/// depend on the picture: every block tries the same shifts.
///
/// A playing video's log gives one number per window that mixes in the
/// compositor and the depth model; this gives the interpolator's own.
public enum FlowInterpolationBenchmark {
    public struct Result: Codable, Sendable {
        public var width: Int
        public var height: Int
        public var withMotion: Bool
        public var iterations: Int
        public var gpuMedianMs: Double
        public var gpuP90Ms: Double
        public var gpuBestMs: Double
        public var wallMedianMs: Double
        public var wallP90Ms: Double
        public var error: String?
    }

    /// Each size in `sizes`, with and without the motion output, after a warm-up.
    public static func run(sizes: [(width: Int, height: Int)], iterations: Int = 120) async -> [Result] {
        await Task.detached(priority: .userInitiated) {
            var results: [Result] = []
            for size in sizes {
                for withMotion in [false, true] {
                    results.append(measure(width: size.width, height: size.height, withMotion: withMotion,
                                           iterations: iterations))
                }
            }
            return results
        }.value
    }

    static func measure(width: Int, height: Int, withMotion: Bool, iterations: Int) -> Result {
        var result = Result(width: width, height: height, withMotion: withMotion, iterations: iterations,
                            gpuMedianMs: 0, gpuP90Ms: 0, gpuBestMs: 0, wallMedianMs: 0, wallP90Ms: 0)
        guard let flow = FlowMidFrameInterpolator() else {
            result.error = "no Metal device, or the kernels didn't compile"
            return result
        }
        guard flow.prepare(width: width, height: height, scale: 1),
              let a = pattern(width: width, height: height, shift: (0, 0)),
              let b = pattern(width: width, height: height, shift: (6, 3)) else {
            result.error = "couldn't prepare \(width)x\(height)"
            return result
        }
        var gpu: [Double] = []
        var wall: [Double] = []
        for index in 0 ..< iterations + 10 {
            let done = DispatchSemaphore(value: 0)
            nonisolated(unsafe) var ms = Double.nan
            nonisolated(unsafe) var ok = false
            flow.midFrame(between: a, and: b, wantsMotion: withMotion) { frame, _, processMs in
                ok = frame != nil
                ms = processMs
                done.signal()
            }
            done.wait()
            guard ok else {
                result.error = "interpolation failed"
                return result
            }
            if index >= 10 {
                wall.append(ms)
                gpu.append(flow.lastGPUMilliseconds)
            }
        }
        func percentile(_ values: [Double], _ p: Double) -> Double {
            let sorted = values.sorted()
            return sorted[min(sorted.count - 1, Int(Double(sorted.count) * p))]
        }
        result.gpuMedianMs = percentile(gpu, 0.5)
        result.gpuP90Ms = percentile(gpu, 0.9)
        result.gpuBestMs = gpu.min() ?? 0
        result.wallMedianMs = percentile(wall, 0.5)
        result.wallP90Ms = percentile(wall, 0.9)
        VergenceLog.pipeline.info(
            "[flow-bench] \(width)×\(height)\(withMotion ? " +motion" : "", privacy: .public): GPU median \(result.gpuMedianMs, format: .fixed(precision: 2), privacy: .public) ms, p90 \(result.gpuP90Ms, format: .fixed(precision: 2), privacy: .public); wall median \(result.wallMedianMs, format: .fixed(precision: 2), privacy: .public), p90 \(result.wallP90Ms, format: .fixed(precision: 2), privacy: .public)")
        return result
    }

    /// Smooth value noise with some fine detail, moved by `shift` pixels.
    private static func pattern(width: Int, height: Int, shift: (Int, Int)) -> CVPixelBuffer? {
        var buffer: CVPixelBuffer?
        let attributes: [String: Any] = [kCVPixelBufferMetalCompatibilityKey as String: true,
                                         kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any]()]
        CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA, attributes as CFDictionary, &buffer)
        guard let buffer else { return nil }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer)?.assumingMemoryBound(to: UInt8.self) else { return nil }
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        func hash(_ x: Int, _ y: Int) -> Double {
            var h = UInt64(bitPattern: Int64(x &* 374_761_393 &+ y &* 668_265_263))
            h = (h ^ (h >> 13)) &* 1_274_126_177
            return Double(h & 0xFFFF) / 65535
        }
        func noise(_ x: Double, _ y: Double) -> Double {
            let x0 = Int(floor(x)), y0 = Int(floor(y))
            let fx = x - Double(x0), fy = y - Double(y0)
            let top = hash(x0, y0) * (1 - fx) + hash(x0 + 1, y0) * fx
            let bottom = hash(x0, y0 + 1) * (1 - fx) + hash(x0 + 1, y0 + 1) * fx
            return top * (1 - fy) + bottom * fy
        }
        for y in 0 ..< height {
            let row = base + y * stride
            for x in 0 ..< width {
                let sx = Double(x - shift.0), sy = Double(y - shift.1)
                let v = 0.6 * noise(sx / 24, sy / 24) + 0.3 * noise(sx / 6, sy / 6) + 0.1 * hash(Int(sx), Int(sy))
                let c = UInt8(max(0, min(255, v * 255)))
                row[x * 4] = c
                row[x * 4 + 1] = UInt8(max(0, min(255, (1 - v) * 200)))
                row[x * 4 + 2] = c / 2 &+ 60
                row[x * 4 + 3] = 255
            }
        }
        return buffer
    }
}
#endif
