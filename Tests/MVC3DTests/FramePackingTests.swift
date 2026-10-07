import Foundation
import Testing
@testable import MVC3D

/// A 4x2 picture pair whose every sample names its view and plane.
private func withFrame(_ body: (StereoFrame) -> Void) {
    // Stride wider than width, as decoders allocate.
    let stride = 8, cstride = 4
    var planes: [[UInt8]] = []
    for view in 0 ..< 2 {
        planes.append((0 ..< stride * 2).map { UInt8(view * 100 + $0) })     // Y 4x2
        planes.append((0 ..< cstride).map { UInt8(view * 100 + 50 + $0) })   // Cb 2x1
        planes.append((0 ..< cstride).map { UInt8(view * 100 + 70 + $0) })   // Cr 2x1
    }
    planes[0].withUnsafeBufferPointer { ly in planes[1].withUnsafeBufferPointer { lb in planes[2].withUnsafeBufferPointer { lr in
    planes[3].withUnsafeBufferPointer { ry in planes[4].withUnsafeBufferPointer { rb in planes[5].withUnsafeBufferPointer { rr in
        func views(_ y: UnsafeBufferPointer<UInt8>, _ b: UnsafeBufferPointer<UInt8>, _ r: UnsafeBufferPointer<UInt8>) -> [PlaneView] {
            [PlaneView(base: y.baseAddress!, width: 4, height: 2, stride: stride),
             PlaneView(base: b.baseAddress!, width: 2, height: 1, stride: cstride),
             PlaneView(base: r.baseAddress!, width: 2, height: 1, stride: cstride)]
        }
        body(StereoFrame(pts: 0, left: views(ly, lb, lr), right: views(ry, rb, rr)))
    }}}}}}
}

private func packed(_ packing: StereoPacking, _ frame: StereoFrame) -> [UInt8] {
    var out = [UInt8](repeating: 0, count: frame.packedByteCount(packing))
    out.withUnsafeMutableBytes { frame.pack(packing, into: $0) }
    return out
}

@Test func topAndBottomStacksLeftAboveRightPerPlane() {
    withFrame { frame in
        #expect(frame.packedByteCount(.topAndBottom) == 4 * 4 + 2 * 2 * 2)
        #expect(packed(.topAndBottom, frame) == [
            0, 1, 2, 3, 8, 9, 10, 11, 100, 101, 102, 103, 108, 109, 110, 111, // Y: left rows, right rows
            50, 51, 150, 151, // Cb
            70, 71, 170, 171, // Cr
        ])
    }
}

@Test func sideBySidePutsLeftThenRightOnEachRow() {
    withFrame { frame in
        #expect(packed(.sideBySide, frame) == [
            0, 1, 2, 3, 100, 101, 102, 103, 8, 9, 10, 11, 108, 109, 110, 111,
            50, 51, 150, 151,
            70, 71, 170, 171,
        ])
    }
}
