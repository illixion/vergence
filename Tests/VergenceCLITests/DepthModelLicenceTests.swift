import Foundation
import Testing
@testable import VergenceCLI

/// Which depth models the command line refuses without
/// VERGENCE_NONCOMMERCIAL_MODELS: by name, whatever a conversion's metadata says.
@Suite("Depth model licences")
struct DepthModelLicenceTests {
    @Test func smallIsFreeAndTheLargerOnesAreNot() {
        #expect(DepthModelLicence.restriction(modelName: "DepthAnythingV2SmallF16") == nil)
        #expect(DepthModelLicence.restriction(modelName: "DepthAnythingV2Small770x574") == nil)
        #expect(DepthModelLicence.restriction(modelName: "DepthAnythingV2BaseF16", metadataLicence: "Apache 2")?.name == "CC-BY-NC-4.0")
        #expect(DepthModelLicence.restriction(modelName: "DepthAnything_v2-Large")?.name == "CC-BY-NC-4.0")
        #expect(DepthModelLicence.restriction(modelName: "depth-anything-v2-large")?.name == "CC-BY-NC-4.0")
        #expect(DepthModelLicence.restriction(modelName: "DepthProNormalizedInverseDepth") != nil)
        #expect(DepthModelLicence.restriction(modelName: "Custom", metadataLicence: "CC-BY-NC-4.0") != nil)
    }

    @Test func withoutTheBuildConditionALargerModelIsRefused() throws {
        let url = URL(fileURLWithPath: "/tmp/DepthAnythingV2BaseF16.mlmodelc")
        #if VERGENCE_NONCOMMERCIAL_MODELS
        #expect(try checkDepthModelLicence(url) == url)
        #else
        #expect(throws: (any Error).self) { try checkDepthModelLicence(url) }
        #endif
        let small = URL(fileURLWithPath: "/tmp/DepthAnythingV2SmallF16.mlmodelc")
        #expect(try checkDepthModelLicence(small) == small)
    }
}
