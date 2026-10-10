#if canImport(CoreVideo)
import CoreVideo
import Foundation
import Testing
@testable import VergenceApple
@testable import VergenceDisc

/// An eye whose every sample encodes its plane, position and eye.
private func eyePlanes(_ eye: UInt8, width: Int, height: Int, storage: inout [[UInt8]]) -> [PlaneView] {
    let sizes = [(width, height), (width / 2, height / 2), (width / 2, height / 2)]
    var planes: [PlaneView] = []
    for (p, (w, h)) in sizes.enumerated() {
        let stride = w + 16 // padded rows, as edge264's are
        storage.append((0 ..< stride * h).map { UInt8(truncatingIfNeeded: Int(eye) * 100 + p * 40 + $0 % stride + $0 / stride * 3) })
        storage[storage.count - 1].withUnsafeBufferPointer {
            planes.append(PlaneView(base: $0.baseAddress!, width: w, height: h, stride: stride))
        }
    }
    return planes
}

private func sample(_ buffer: CVPixelBuffer, plane: Int, x: Int, y: Int) -> UInt8 {
    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
    let base = CVPixelBufferGetBaseAddressOfPlane(buffer, plane)!.assumingMemoryBound(to: UInt8.self)
    return base[y * CVPixelBufferGetBytesPerRowOfPlane(buffer, plane) + x]
}

@Test func eyesLandWhereTheLayoutPutsThem() throws {
    let w = 64, h = 32
    var storage: [[UInt8]] = []
    let left = eyePlanes(0, width: w, height: h, storage: &storage)
    let right = eyePlanes(1, width: w, height: h, storage: &storage)
    let frame = StereoFrame(pts: 0, left: left, right: right)
    func expected(_ eye: [PlaneView], _ p: Int, _ x: Int, _ y: Int) -> UInt8 { eye[p].base[y * eye[p].stride + x] }

    let tab = try StereoPixelBufferMaker(layout: .packed(.topAndBottom)).makeBuffers(frame).left
    #expect(CVPixelBufferGetWidth(tab) == w && CVPixelBufferGetHeight(tab) == 2 * h)
    #expect(CVPixelBufferGetPixelFormatType(tab) == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange)
    #expect(sample(tab, plane: 0, x: 5, y: 7) == expected(left, 0, 5, 7))
    #expect(sample(tab, plane: 0, x: 5, y: h + 7) == expected(right, 0, 5, 7))
    #expect(sample(tab, plane: 1, x: 2 * 3, y: 4) == expected(left, 1, 3, 4))      // Cb
    #expect(sample(tab, plane: 1, x: 2 * 3 + 1, y: 4) == expected(left, 2, 3, 4))  // Cr
    #expect(sample(tab, plane: 1, x: 2 * 3 + 1, y: h / 2 + 4) == expected(right, 2, 3, 4))

    let sbs = try StereoPixelBufferMaker(layout: .packed(.sideBySide)).makeBuffers(frame).left
    #expect(CVPixelBufferGetWidth(sbs) == 2 * w && CVPixelBufferGetHeight(sbs) == h)
    #expect(sample(sbs, plane: 0, x: w + 5, y: 7) == expected(right, 0, 5, 7))
    #expect(sample(sbs, plane: 1, x: w + 2 * 3, y: 4) == expected(right, 1, 3, 4))

    let separate = try StereoPixelBufferMaker(layout: .separate).makeBuffers(frame)
    #expect(sample(separate.left, plane: 0, x: 9, y: 9) == expected(left, 0, 9, 9))
    #expect(sample(try #require(separate.right), plane: 1, x: 1, y: 2) == expected(right, 2, 0, 2))
    #expect(CVBufferCopyAttachment(separate.left, kCVImageBufferYCbCrMatrixKey, nil) as? String
            == kCVImageBufferYCbCrMatrix_ITU_R_709_2 as String)

    let mono = try StereoPixelBufferMaker(layout: .separate).makeBuffers(StereoFrame(pts: 0, left: left, right: nil))
    #expect(mono.right == nil)
}
#endif
