import Testing
@testable import VergenceCore

struct StillStereoResolutionTests {
    @Test func obeysViewerLimitAndNativeDimensions() {
        #expect(StillStereoResolution.longSide(width: 8192, height: 4608, requested: 2048) == 2048)
        #expect(StillStereoResolution.longSide(width: 640, height: 480, requested: 4096) == 640)
    }
    @Test func boundsEightKAndPortraitWorkingSets() {
        let wide = StillStereoResolution.longSide(width: 8192, height: 4608, requested: 8192)
        let tall = StillStereoResolution.longSide(width: 4608, height: 8192, requested: 8192)
        #expect(wide == tall)
        #expect(wide <= 4096)
        #expect(Double(wide * wide) * 4608 / 8192 * 40 <= Double(StillStereoResolution.defaultWorkingBytes))
        #expect(StillStereoResolution.longSide(width: 8192, height: 8192, requested: 8192) < wide)
    }
    @Test func invalidDimensionsDoNotTrap() {
        #expect(StillStereoResolution.longSide(width: 0, height: 1, requested: .infinity) == 1)
        #expect(StillStereoResolution.longSide(width: 1, height: 1, requested: .nan) == 1)
    }
}
