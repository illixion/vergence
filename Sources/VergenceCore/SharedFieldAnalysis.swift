/*
 Vergence - the producer's half of Smooth Motion for true stereo

 What a Mac (or a server) computes for each real frame pair so the headset
 only has to warp: the left eye's motion between two real frames, and each real
 frame's right→left disparity. The design and the measurements behind every
 number here are in Hypnos `docs/plans/blu-ray-3d-smooth-motion.md` (E1, E2):

 - Flow: RIFE 4.25 lite at 480×288 on the left eye, once per pair, at t = ½.
 - Disparity: NeuFlow v2 at 512×288 on (right, left) of each real pair, the
   horizontal part only (a 3D Blu-ray is rectified).
 - Both are stored at a quarter of the model's size (120×72 and 128×72): the
   fields are smooth, and the in-between frames came out no worse (+0.05 dB)
   while the record shrank 16-fold. Row-delta coded and deflated at a half
   pixel step, a frame's flow and disparity come to about 25 KB: 6 Mbit/s
   live, 4 GB for a two-hour film.

 Everything here is Foundation only, so a Linux server producing for a Quest
 runs the same code; only the model runs are platform-specific.
 */

import Foundation

/// A buffer handed to `concurrentPerform` workers that write disjoint parts of it.
private struct UnsafeSendableBuffer<T>: @unchecked Sendable {
    let buffer: UnsafeMutableBufferPointer<T>
    init(_ buffer: UnsafeMutableBufferPointer<T>) { self.buffer = buffer }
}

/// One borrowed 8-bit plane.
public struct AnalysisPlane: @unchecked Sendable {
    public var base: UnsafePointer<UInt8>
    public var width: Int
    public var height: Int
    public var stride: Int

    public init(base: UnsafePointer<UInt8>, width: Int, height: Int, stride: Int) {
        self.base = base
        self.width = width
        self.height = height
        self.stride = stride
    }

    @inline(__always) func sample(_ x: Float, _ y: Float) -> Float {
        let cx = min(max(x, 0), Float(width - 1)), cy = min(max(y, 0), Float(height - 1))
        let x0 = Int(cx), y0 = Int(cy)
        let x1 = min(x0 + 1, width - 1), y1 = min(y0 + 1, height - 1)
        let fx = cx - Float(x0), fy = cy - Float(y0)
        let r0 = base + y0 * stride, r1 = base + y1 * stride
        let top = Float(r0[x0]) + (Float(r0[x1]) - Float(r0[x0])) * fx
        let bottom = Float(r1[x0]) + (Float(r1[x1]) - Float(r1[x0])) * fx
        return top + (bottom - top) * fy
    }
}

public enum SharedFieldAnalysis {
    /// RIFE's working size.
    public static let flowModelSize = (width: 480, height: 288)
    /// NeuFlow's working size.
    public static let disparityModelSize = (width: 512, height: 288)
    /// How much smaller than the model's output the fields are stored.
    public static let storageFactor = 4
    /// Stored fields are int16 in steps of 1/32 of a stored pixel (half a pixel
    /// at 1080p; ±1024 stored px of range, far past any real motion). Measured
    /// against the PyTorch reference on 1080p Blu-ray (Hypnos
    /// `scripts/check-analysis-track.py`): 1/64 costs nothing and ~31 KB a
    /// frame, 1/16 0.15 dB on a slow pan and ~20 KB, 1/4 0.9 dB and ~12 KB.
    public static let step: Float = 1.0 / 32

    /// A 4:2:0 BT.709 limited-range picture as planar RGB, 0…1 (or 0…`scale`),
    /// resized to `width`×`height` bilinearly the way PyTorch's
    /// `F.interpolate(align_corners=False)` does (no prefilter), which is how
    /// the models were measured. Chroma is centre-sited.
    public static func rgbPlanar(y: AnalysisPlane, cb: AnalysisPlane, cr: AnalysisPlane,
                                 width: Int, height: Int, scale: Float = 1) -> [Float] {
        let plane = width * height
        var out = [Float](repeating: 0, count: 3 * plane)
        let sx = Float(y.width) / Float(width), sy = Float(y.height) / Float(height)
        let csx = Float(cb.width) / Float(y.width), csy = Float(cb.height) / Float(y.height)
        out.withUnsafeMutableBufferPointer { rgb in
            // Rows in bands across cores; each band writes only its own rows.
            let bands = 8
            let rgb = UnsafeSendableBuffer(rgb)
            DispatchQueue.concurrentPerform(iterations: bands) { band in
            let rgb = rgb.buffer
            for oy in stride(from: band, to: height, by: bands) {
                let fy = (Float(oy) + 0.5) * sy - 0.5
                let cfy = (fy + 0.5) * csy - 0.5
                for ox in 0 ..< width {
                    let fx = (Float(ox) + 0.5) * sx - 0.5
                    let cfx = (fx + 0.5) * csx - 0.5
                    let luma = (y.sample(fx, fy) - 16) / 219
                    let u = (cb.sample(cfx, cfy) - 128) / 224
                    let v = (cr.sample(cfx, cfy) - 128) / 224
                    let i = oy * width + ox
                    rgb[i] = min(max(luma + 1.5748 * v, 0), 1) * scale
                    rgb[plane + i] = min(max(luma - 0.18733 * u - 0.46812 * v, 0), 1) * scale
                    rgb[2 * plane + i] = min(max(luma + 1.8556 * u, 0), 1) * scale
                }
            }
            }
        }
        return out
    }

    /// Box average over `factor`×`factor` blocks, planar. Dimensions must divide.
    public static func areaDownsample(_ planar: [Float], channels: Int, width: Int, height: Int,
                                      factor: Int) -> [Float] {
        precondition(width % factor == 0 && height % factor == 0 && planar.count == channels * width * height)
        let w = width / factor, h = height / factor
        var out = [Float](repeating: 0, count: channels * w * h)
        let norm = 1 / Float(factor * factor)
        for c in 0 ..< channels {
            for oy in 0 ..< h {
                for ox in 0 ..< w {
                    var sum: Float = 0
                    for dy in 0 ..< factor {
                        let row = c * width * height + (oy * factor + dy) * width + ox * factor
                        for dx in 0 ..< factor { sum += planar[row + dx] }
                    }
                    out[c * w * h + oy * w + ox] = sum * norm
                }
            }
        }
        return out
    }

    /// RIFE's output at t = ½ as a `flowPair` record.
    /// - Parameters:
    ///   - flow: 4 planar channels at the model's size, F½→0 (x, y) then F½→1 (x, y), model pixels.
    ///   - mask: frame 0's blend weight at ½, 0…1.
    public static func flowPairRecord(flow: [Float], mask: [Float], width: Int, height: Int,
                                      time: AnalysisTime, endTime: AnalysisTime) -> AnalysisRecord {
        let f = storageFactor
        let small = areaDownsample(flow, channels: 4, width: width, height: height, factor: f)
        let m = areaDownsample(mask, channels: 1, width: width, height: height, factor: f)
        let n = (width / f) * (height / f)
        // Stored as the record kind defines: F0→1 ≈ 2·F½→1, F1→0 ≈ 2·F½→0, in stored pixels.
        let k = 2 / Float(f)
        var values = [Float](repeating: 0, count: 5 * n)
        for i in 0 ..< n {
            values[i] = small[2 * n + i] * k
            values[n + i] = small[3 * n + i] * k
            values[2 * n + i] = small[i] * k
            values[3 * n + i] = small[n + i] * k
            values[4 * n + i] = m[i]
        }
        return AnalysisRecord.encoding(values, kind: .flowPair, as: .int16, channels: 5, width: width / f,
                                       height: height / f, time: time, endTime: endTime,
                                       maxMagnitude: step * Float(Int16.max))
    }

    /// NeuFlow's right→left flow for one real pair as a `disparity` record on
    /// the right eye, horizontal only.
    /// - Parameter rightToLeft: 2 planar channels (x, y) at the model's size, model pixels.
    public static func disparityRecord(rightToLeft: [Float], width: Int, height: Int,
                                       time: AnalysisTime) -> AnalysisRecord {
        let f = storageFactor
        let x = areaDownsample(Array(rightToLeft[0 ..< width * height]), channels: 1, width: width,
                               height: height, factor: f).map { $0 / Float(f) }
        return AnalysisRecord.encoding(x, kind: .disparity, eye: .right, as: .int16, channels: 1,
                                       width: width / f, height: height / f, time: time,
                                       maxMagnitude: step * Float(Int16.max))
    }
}
