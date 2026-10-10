#if canImport(CoreGraphics)
import CoreGraphics
#endif
import Foundation
import Testing
@testable import VergenceCore

@Test func halfFormatsUnsqueezeTheirEyes() {
    #expect(StereoscopicFormat.halfSideBySide.perEyeDimensions(sourceWidth: 1920, sourceHeight: 1080) == (1920, 1080))
    #expect(StereoscopicFormat.sideBySide.perEyeDimensions(sourceWidth: 3840, sourceHeight: 1080) == (1920, 1080))
    #expect(StereoscopicFormat.halfOverUnder.perEyeDimensions(sourceWidth: 1920, sourceHeight: 1080) == (1920, 1080))
    #expect(StereoscopicFormat.overUnder.perEyeDimensions(sourceWidth: 1920, sourceHeight: 2160) == (1920, 1080))
}

@Test func tagsNameTheFormat() {
    #expect(StereoscopicFormat.detect(from: ["Movie", "HSBS"]) == (true, .halfSideBySide))
    #expect(StereoscopicFormat.detect(from: ["3D", "Over-Under"]) == (true, .overUnder))
    #expect(StereoscopicFormat.detect(from: [" VR "]) == (true, .sideBySide))
    #expect(StereoscopicFormat.detect(from: ["Documentary"]) == (false, nil))
}

@Test func rawValuesStayStableForSavedSettings() {
    #expect(StereoscopicFormat.allCases.map(\.rawValue) == ["sbs", "ou", "hsbs", "hou"])
}
