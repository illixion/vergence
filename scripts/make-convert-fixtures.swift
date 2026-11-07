// Writes convert/tests/fixtures/core.json: VergenceCore's depth maths
// (DepthRefinement.swift, DepthAnchor.swift, DepthHistogram.swift) run on
// pseudo-random inputs, inputs and outputs together, so `vergence-convert`'s
// Rust port is tested against the Swift it ports.
//
//   scripts/make-convert-fixtures.sh
//
// (compiles this file with those three sources; Foundation only, so it runs
// on Linux too).

import Foundation

struct Random {
    var state: UInt64
    mutating func next() -> Double {
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return Double(state >> 11) / Double(1 << 53)
    }
    mutating func float(_ lo: Float, _ hi: Float) -> Float { lo + Float(next()) * (hi - lo) }
}

var random = Random(state: 0x5EED)
var out: [String: Any] = [:]

func affine(_ a: DepthAffine?) -> Any { a.map { [Double($0.a), Double($0.b)] } ?? NSNull() }

// The depth budget.
let ys: [Float] = (0..<64).map { _ in random.float(-0.6, 1.8) } + [0, 0.08, 0.7, 1, -5, 5]
out["budget"] = ["y": ys.map { Double($0) }, "studio": ys.map { Double(DepthBudget.studio.apply($0)) }]

// Anchor fits from sums of synthetic anchor pixels.
var fits: [[String: Any]] = []
for i in 0..<40 {
    var sums = DepthAlignmentSums()
    let a = random.float(0.2, 3.5), b = random.float(-0.5, 0.5), noise = random.float(0, i % 5 == 0 ? 0.5 : 0.03)
    let n = 200
    let anchored = i % 7 == 0 ? 20 : n
    for _ in 0..<anchored {
        let x = random.float(0, 1)
        sums.add(x: x, y: a * x + b + random.float(-noise, noise), weight: random.float(0.2, 1))
    }
    let span = random.float(0.01, 1.2)
    fits.append(["sums": [sums.w, sums.wx, sums.wy, sums.wxx, sums.wyy, sums.wxy], "pixels": n, "span": Double(span),
                 "fit": affine(sums.fit(pixelCount: n, frameSpan: span))])
}
out["fits"] = fits

// The tracker over a run of frames.
var tracker = DepthScaleShiftTracker()
var trackerCases: [[String: Any]] = []
for i in 0..<120 {
    let lo = random.float(0, 0.3), hi = lo + random.float(0.05, 1.5)
    let relative: DepthAffine? = i % 9 == 4 ? nil : DepthAffine(a: random.float(0.7, 1.4), b: random.float(-0.1, 0.1))
    let cut = i % 37 == 0
    let dt = Double(random.float(0.01, 0.1))
    let mapping = tracker.update(lo: lo, hi: hi, relative: relative, cut: cut, dt: dt)
    trackerCases.append(["lo": Double(lo), "hi": Double(hi), "relative": affine(relative), "cut": cut, "dt": dt,
                         "mapping": affine(mapping)])
}
out["tracker"] = trackerCases

// Cut detection over histogram shapes.
var detector = DepthCutDetector()
var cutCases: [[String: Any]] = []
var base: [UInt32] = (0..<256).map { _ in UInt32(random.next() * 50) }
for i in 0..<60 {
    if i % 13 == 6 { base = (0..<256).map { _ in UInt32(random.next() * 50) } }
    let bins = base.map { $0 + UInt32(random.next() * 4) }
    let lo = random.float(0, 0.05), hi = lo + random.float(0.8, 1.1) * (i % 17 == 9 ? 2 : 1)
    let jump: Float? = i % 11 == 3 ? random.float(0, 0.2) : nil
    let cut = detector.isCut(shape: DepthCutDetector.shape(of: bins), lo: lo, hi: hi, lumaJump: jump)
    cutCases.append(["bins": bins.map(Int.init), "lo": Double(lo), "hi": Double(hi), "jump": jump.map { Double($0) as Any } ?? NSNull(),
                     "cut": cut])
}
out["cuts"] = cutCases

// Percentiles.
var percentiles: [[String: Any]] = []
for i in 0..<20 {
    let bins: [UInt32] = (0..<256).map { k in (k + i) % 3 == 0 ? 0 : UInt32(random.next() * 30) }
    let total = bins.reduce(0) { $0 + Int($1) } + (i % 4 == 0 ? 100 : 0)
    let q = Float(random.next())
    let lo = random.float(-1, 0), hi = random.float(0.5, 2)
    percentiles.append(["bins": bins.map(Int.init), "total": total, "q": Double(q), "lo": Double(lo), "hi": Double(hi),
                        "value": Double(DepthHistogram.percentile(bins, total: total, q: q, lo: lo, hi: hi))])
}
out["percentiles"] = percentiles

// The converter's whole-timeline mapping.
var frames: [DepthShotMapping.Frame] = []
for i in 0..<150 {
    let lo = random.float(0, 0.2), hi = lo + random.float(0.3, 1.2)
    let cut = i > 0 && i % 41 == 0
    let relative: DepthAffine? = i % 6 == 2 ? nil : DepthAffine(a: random.float(0.8, 1.25), b: random.float(-0.05, 0.05))
    frames.append(.init(seconds: Double(i) / 24, lo: lo, hi: hi, cutBefore: cut, relative: relative, lattice: 0))
}
let shot = DepthShotMapping.map(frames)
out["shotMapping"] = frames.indices.map { i -> [String: Any] in
    let f = frames[i]
    return ["seconds": f.seconds, "lo": Double(f.lo), "hi": Double(f.hi), "cut": f.cutBefore, "relative": affine(f.relative),
            "mapping": affine(shot[i])]
}

// The anchor ceiling.
out["ceilings"] = (0..<10).map { _ -> [String: Any] in
    let m = DepthAffine(a: random.float(0.3, 4), b: random.float(-1, 1))
    let c = DepthAnchorCeiling(mapping: m)
    return ["mapping": affine(m), "ceiling": Double(c.ceiling), "softness": Double(c.softness)]
}

let data = try JSONSerialization.data(withJSONObject: out, options: [.sortedKeys])
let path = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "convert/tests/fixtures/core.json"
try data.write(to: URL(fileURLWithPath: path))
print("wrote \(path)")
