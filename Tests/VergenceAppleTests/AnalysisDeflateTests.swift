#if canImport(Compression)
import Foundation
import Testing
import VergenceApple
import VergenceCore

@Suite("Analysis record DEFLATE")
struct AnalysisDeflateTests {
    @Test func aSmoothFieldShrinksAndComesBackExactly() throws {
        let w = 120, h = 72
        let values = (0 ..< 5 * w * h).map { i -> Float in
            let x = Float(i % w), y = Float(i / w % h), c = Float(i / (w * h))
            return c == 4 ? 0.5 + 0.4 * sin(x / 9) : 6 * sin(x / 40 + c) + 2 * cos(y / 25)
        }
        let record = AnalysisRecord.encoding(values, kind: .flowPair, as: .int16, channels: 5, width: w, height: h,
                                             time: AnalysisTime(seconds: 0), endTime: AnalysisTime(seconds: 1.0 / 24),
                                             maxMagnitude: SharedFieldAnalysis.step * Float(Int16.max))
        let packed = try record.deflated()
        #expect(packed.payload.count < record.payload.count / 3, "\(packed.payload.count) of \(record.payload.count)")
        #expect(try packed.inflated() == record)
    }

    @Test func garbageDoesNotInflate() {
        let record = AnalysisRecord(kind: .depth, encoding: .uint8, channels: 1, width: 16, height: 16,
                                    time: AnalysisTime(seconds: 0), payload: Data([1, 2, 3]), isDeflated: true)
        #expect(throws: (any Error).self) { _ = try record.inflated() }
    }
}
#endif
