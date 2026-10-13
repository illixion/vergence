#if canImport(Metal)
import CoreML
import Foundation
import Testing
@testable import VergenceApple

@Suite("Depth array storage layout")
struct DepthArrayStrideTests {
    @Test(arguments: [MLMultiArrayDataType.float16, .float32, .double])
    func paddedAndInterleavedPlanesExcludePadding(type: MLMultiArrayDataType) throws {
        let bytes = type == .float16 ? 2 : (type == .float32 ? 4 : 8)
        let pointer = UnsafeMutableRawPointer.allocate(byteCount: 64 * bytes, alignment: 8)
        pointer.initializeMemory(as: UInt8.self, repeating: 0xff, count: 64 * bytes)
        defer { pointer.deallocate() }
        let array = try MLMultiArray(dataPointer: pointer, shape: [1, 1, 3, 5], dataType: type,
                                     strides: [64, 64, 16, 2], deallocator: nil)
        var expected: [Float] = []
        for y in 0..<3 {
            for x in 0..<5 {
                let value = Float(y * 10 + x)
                array[[0, 0, NSNumber(value: y), NSNumber(value: x)]] = NSNumber(value: value)
                expected.append(value)
            }
        }
        let plane = try #require(StillDepthExtraction.floats(from: array))
        #expect(plane.width == 5 && plane.height == 3)
        #expect(plane.values == expected)
    }

    @Test func widescreenModelPaddingDoesNotShiftRows() throws {
        // Actual Vision output: 686 columns, stride 704 (not 686).
        let pointer = UnsafeMutablePointer<Float>.allocate(capacity: 704 * 3)
        pointer.initialize(repeating: .nan, count: 704 * 3)
        defer { pointer.deallocate() }
        let array = try MLMultiArray(dataPointer: pointer, shape: [1, 1, 3, 686], dataType: .float32,
                                     strides: [2112, 2112, 704, 1], deallocator: nil)
        for y in 0..<3 {
            for x in 0..<686 { pointer[y * 704 + x] = Float(y * 1000 + x) }
        }
        let plane = try #require(StillDepthExtraction.floats(from: array))
        #expect(plane.values[686] == 1000)
        #expect(plane.values[1372] == 2000)
        #expect(plane.values.allSatisfy { $0.isFinite })
    }
}
#endif
