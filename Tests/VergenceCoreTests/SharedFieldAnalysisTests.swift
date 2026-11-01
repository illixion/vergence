import Foundation
import Testing
@testable import VergenceCore

/// The producer's half of stereo Smooth Motion: the resize matches PyTorch's
/// (the models were measured that way), and records come out in the order and
/// units the record kinds define.
@Suite("Shared field analysis")
struct SharedFieldAnalysisTests {
    /// A 4:2:0 picture whose luma is a horizontal ramp, with flat chroma.
    static func picture(width: Int, height: Int, luma: (Int, Int) -> UInt8, cb: UInt8 = 128, cr: UInt8 = 128)
        -> (y: [UInt8], cb: [UInt8], cr: [UInt8]) {
        var y = [UInt8](repeating: 0, count: width * height)
        for row in 0 ..< height { for x in 0 ..< width { y[row * width + x] = luma(x, row) } }
        let c = (width / 2) * (height / 2)
        return (y, [UInt8](repeating: cb, count: c), [UInt8](repeating: cr, count: c))
    }

    static func planar(_ p: (y: [UInt8], cb: [UInt8], cr: [UInt8]), width: Int, height: Int, to w: Int, _ h: Int,
                       scale: Float = 1) -> [Float] {
        p.y.withUnsafeBufferPointer { y in
            p.cb.withUnsafeBufferPointer { cb in
                p.cr.withUnsafeBufferPointer { cr in
                    SharedFieldAnalysis.rgbPlanar(
                        y: AnalysisPlane(base: y.baseAddress!, width: width, height: height, stride: width),
                        cb: AnalysisPlane(base: cb.baseAddress!, width: width / 2, height: height / 2, stride: width / 2),
                        cr: AnalysisPlane(base: cr.baseAddress!, width: width / 2, height: height / 2, stride: width / 2),
                        width: w, height: h, scale: scale)
                }
            }
        }
    }

    @Test func limitedRangeGreyBecomesFullRange() {
        for (code, expected): (UInt8, Float) in [(16, 0), (235, 1), (126, 0.5023), (0, 0), (255, 1)] {
            let p = Self.picture(width: 8, height: 4, luma: { _, _ in code })
            let rgb = Self.planar(p, width: 8, height: 4, to: 4, 2)
            #expect(rgb.allSatisfy { abs($0 - expected) < 1e-3 }, "Y=\(code)")
        }
        // Pure BT.709 red: Y 63, Cb 102, Cr 240.
        let red = Self.picture(width: 8, height: 4, luma: { _, _ in 63 }, cb: 102, cr: 240)
        let rgb = Self.planar(red, width: 8, height: 4, to: 4, 2)
        #expect(abs(rgb[0] - 1) < 0.01 && abs(rgb[8]) < 0.01 && abs(rgb[16]) < 0.01)
    }

    @Test func downscalingSamplesLikePyTorch() {
        // F.interpolate(bilinear, align_corners=False) 8 → 2 wide samples source x = 1.5 and 5.5.
        let p = Self.picture(width: 8, height: 2, luma: { x, _ in UInt8(16 + 20 * x) })
        let rgb = Self.planar(p, width: 8, height: 2, to: 2, 1, scale: 219)
        #expect(abs(rgb[0] - 30) < 0.01 && abs(rgb[1] - 110) < 0.01)
    }

    @Test func areaDownsampleAverages() {
        let values: [Float] = (0 ..< 32).map(Float.init) // 2 channels of 4×4
        let out = SharedFieldAnalysis.areaDownsample(values, channels: 2, width: 4, height: 4, factor: 2)
        #expect(out == [2.5, 4.5, 10.5, 12.5, 18.5, 20.5, 26.5, 28.5])
    }

    @Test func flowPairRecordsHoldTheKindsChannelsInStoredPixels() {
        let w = 8, h = 4, n = w * h
        // F½→0 = (3, −1), F½→1 = (−3, 1) everywhere at the model's size; mask 0.25.
        let flow = [Float](repeating: 3, count: n) + [Float](repeating: -1, count: n)
            + [Float](repeating: -3, count: n) + [Float](repeating: 1, count: n)
        let record = SharedFieldAnalysis.flowPairRecord(
            flow: flow, mask: [Float](repeating: 0.25, count: n), width: w, height: h,
            time: AnalysisTime(value: 0, timescale: 24000), endTime: AnalysisTime(value: 1001, timescale: 24000))
        #expect(record.kind == .flowPair && record.channels == 5 && record.width == 2 && record.height == 1)
        let v = record.values()
        // F0→1 = 2·F½→1 = (−6, 2), F1→0 = (6, −2), divided by the storage factor 4.
        let expected: [Float] = [-1.5, -1.5, 0.5, 0.5, 1.5, 1.5, -0.5, -0.5, 0.25, 0.25]
        #expect(zip(v, expected).allSatisfy { abs($0 - $1) <= SharedFieldAnalysis.step / 2 + 1e-6 }, "\(v)")
        #expect(record.scale == SharedFieldAnalysis.step)
    }

    @Test func disparityRecordsAreHorizontalOnTheRightEye() {
        let w = 8, h = 4, n = w * h
        let rightToLeft = [Float](repeating: -20, count: n) + [Float](repeating: 3, count: n)
        let record = SharedFieldAnalysis.disparityRecord(rightToLeft: rightToLeft, width: w, height: h,
                                                         time: AnalysisTime(seconds: 1))
        #expect(record.kind == .disparity && record.eye == .right && record.channels == 1)
        #expect(record.values().allSatisfy { abs($0 + 5) < 1e-6 })
    }
}
