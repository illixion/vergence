import Foundation
import Testing
@testable import VergenceDisc

/// ITU-T MVC conformance bitstreams, when fetched with
/// `scripts/fetch-conformance.sh` and named by `VERGENCE_CONFORMANCE`. Skipped
/// otherwise: the streams are not ours to redistribute.
private let conformanceDirectory = ProcessInfo.processInfo.environment["VERGENCE_CONFORMANCE"].map { URL(fileURLWithPath: $0) }

/// Streams edge264 decodes in stereo, with their frame counts.
private let stereoStreams: [(String, Int)] = [
    ("MVCDS1", 290), ("MVCDS2", 290), ("MVCDS3", 290), ("MVCDS-4", 9), ("MVCDS-5", 9), ("MVCDS-6", 9),
    ("MVCRP_1", 200), ("MVCRP_2", 250), ("MVCRP_3", 250), ("MVCRP_4", 250), ("MVCRP_5", 250), ("MVCRP_6", 250),
    ("MVCSPS-1", 9),
]

private func decodeAll(_ data: Data) throws -> (frames: Int, stereo: Int) {
    let decoder = try MVCDecoder()
    var frames = 0, stereo = 0
    let count: (StereoFrame) -> Void = { frame in
        frames += 1
        if frame.right != nil { stereo += 1 }
    }
    try decoder.decode(StereoAccessUnit(pts: nil, dts: nil, data: data, hasDependentView: true), output: count)
    try decoder.finish(output: count)
    return (frames, stereo)
}

@Test(.enabled(if: conformanceDirectory != nil), arguments: stereoStreams)
func conformanceStreamDecodesInStereo(name: String, frames: Int) throws {
    let data = try Data(contentsOf: conformanceDirectory!.appendingPathComponent("\(name).264"))
    let result = try decodeAll(data)
    #expect(result.frames == frames)
    #expect(result.stereo == frames)
}

/// A stream cut inside a slice used to hang `finish`: edge264 answers the
/// end-of-stream call with ENOBUFS forever when the last picture is
/// incomplete.
@Test(.enabled(if: conformanceDirectory != nil), .timeLimit(.minutes(1)))
func aStreamCutMidSliceFinishes() throws {
    let data = try Data(contentsOf: conformanceDirectory!.appendingPathComponent("MVCDS1.264"))
    for cut in [100_000, 250_001, 777_777] {
        let result = try decodeAll(data.prefix(cut))
        #expect(result.frames > 0)
    }
}
