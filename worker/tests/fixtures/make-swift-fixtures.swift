// Writes the Swift fixtures the worker's tests compare against: rgbPlanar on a
// fixed 16x8 picture (bit patterns) and a deflated flowPair + disparity track.
//
//   swiftc -O -module-name VergenceCore -o /tmp/fixtures Sources/VergenceCore/SharedFieldAnalysis.swift \
//       Sources/VergenceCore/AnalysisTrack.swift worker/tests/fixtures/make-swift-fixtures.swift
//   /tmp/fixtures worker/tests/fixtures

import Compression
import Foundation

func rawDeflate(_ data: Data) throws -> Data {
    let capacity = data.count + data.count / 2 + 1024
    var out = Data(count: capacity)
    let written = out.withUnsafeMutableBytes { dst in
        data.withUnsafeBytes { src in
            compression_encode_buffer(dst.bindMemory(to: UInt8.self).baseAddress!, capacity,
                                      src.bindMemory(to: UInt8.self).baseAddress!, data.count, nil, COMPRESSION_ZLIB)
        }
    }
    out.count = written
    return out
}

// The same pattern as the Rust test's `pattern()`.
let w = 16, h = 8
let y = (0 ..< w * h).map { UInt8(16 + ($0 * 37 + ($0 / w) * 11) % 220) }
let cb = (0 ..< (w / 2) * (h / 2)).map { UInt8(40 + ($0 * 53) % 180) }
let cr = (0 ..< (w / 2) * (h / 2)).map { UInt8(200 - ($0 * 29) % 170) }
let rgb = y.withUnsafeBufferPointer { yp in cb.withUnsafeBufferPointer { cbp in cr.withUnsafeBufferPointer { crp in
    SharedFieldAnalysis.rgbPlanar(y: AnalysisPlane(base: yp.baseAddress!, width: w, height: h, stride: w),
                                  cb: AnalysisPlane(base: cbp.baseAddress!, width: w / 2, height: h / 2, stride: w / 2),
                                  cr: AnalysisPlane(base: crp.baseAddress!, width: w / 2, height: h / 2, stride: w / 2),
                                  width: 6, height: 3)
}}}
try ("[" + rgb.map { String(format: "0x%08x", $0.bitPattern) }.joined(separator: ", ") + "]\n")
    .write(toFile: CommandLine.arguments[1] + "/swift_rgb_planar.txt", atomically: true, encoding: .utf8)

// A flow pair and a disparity record from synthetic model outputs (48×24 → 12×6 stored), deflated.
let fw = 48, fh = 24, n = fw * fh
let flow = (0 ..< 4 * n).map { i -> Float in let c = i / n, p = i % n; return Float(c + 1) * sin(Float(p) * 0.05) * 7.3 + Float(p % fw) * 0.11 }
let mask = (0 ..< n).map { 0.5 + 0.5 * cos(Float($0) * 0.03) }
let disparity = (0 ..< 2 * n).map { i -> Float in -3.7 + Float(i % fw) * 0.21 - Float(i / fw) * 0.05 }
var track = try AnalysisTrack.header(AnalysisTrackHeader(mediaID: "fixture", producer: "swift fixture",
    models: ["flowPair": "RIFE 48x24", "disparity": "NeuFlow 48x24"], sourceWidth: 1920, sourceHeight: 1080))
let t0 = AnalysisTime(value: 1001 * 5, timescale: 24000), t1 = AnalysisTime(value: 1001 * 6, timescale: 24000)
track.append(AnalysisTrack.record(try SharedFieldAnalysis.flowPairRecord(flow: flow, mask: mask, width: fw, height: fh,
                                                                          time: t0, endTime: t1).deflated(using: rawDeflate)))
track.append(AnalysisTrack.record(try SharedFieldAnalysis.disparityRecord(rightToLeft: disparity, width: fw, height: fh,
                                                                           time: t1).deflated(using: rawDeflate)))
try track.write(to: URL(fileURLWithPath: CommandLine.arguments[1] + "/swift_records.vgat"))
print("ok", rgb.prefix(3), track.count)
