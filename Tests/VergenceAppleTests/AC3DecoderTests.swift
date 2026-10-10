#if canImport(AVFoundation)
import AVFoundation
import Foundation
import Testing
@testable import VergenceApple

@Test func ac3LayoutsNameTheirChannels() {
    #expect(AC3Decoder.labels(of: AC3Decoder.layoutTag(mode: 7, lfe: true)) ==
            [kAudioChannelLabel_Left, kAudioChannelLabel_Center, kAudioChannelLabel_Right,
             kAudioChannelLabel_LeftSurround, kAudioChannelLabel_RightSurround, kAudioChannelLabel_LFEScreen])
}
#endif
