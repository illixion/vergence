#if canImport(CoreML) && canImport(Network)
import Testing
@testable import VergenceCLI

/// The depth profile a Mac worker claims with: shared with the ONNX worker for
/// Depth Anything V2 Small and Base (`worker/src/workloads.rs` names them the same).
@Suite("Depth profiles")
struct DepthProfileTests {
    @Test func smallAndBaseShareOneNameAcrossEngines() {
        #expect(DepthProducer.profile(modelName: "DepthAnythingV2Small770x574", inputSize: (770, 574), licence: "")
            == "Depth Anything V2 Small 770x574 offline-v5 context-1s")
        #expect(DepthProducer.profile(modelName: "DepthAnythingV2BaseF16", inputSize: (518, 392), licence: " CC-BY-NC-4.0")
            == "Depth Anything V2 Base 518x392 CC-BY-NC-4.0 offline-v5 context-1s")
        // Not checked across engines: named by their file as before.
        #expect(DepthProducer.profile(modelName: "DepthAnythingV2LargeF16", inputSize: (518, 392), licence: " CC-BY-NC-4.0")
            == "DepthAnythingV2LargeF16 CC-BY-NC-4.0 offline-v5 context-1s")
        #expect(DepthProducer.sharedDepthFamily("VideoDepthAnythingV2Small") == nil)
    }
}
#endif
