import Foundation
import Testing
@testable import VergenceDisc

@Test func ac3HeaderGivesSizeAndChannels() throws {
    // 48 kHz, frmsizecod 30 (448 kb/s), bsid 8, acmod 7 + LFE.
    let header = try #require(AC3FrameHeader([0x0B, 0x77, 0, 0, 0x1E, 0x40, 0xEB, 0]))
    #expect(!header.isEnhanced)
    #expect(header.byteCount == 1792)
    #expect(header.sampleCount == 1536)
    #expect(header.channelMode == 7 && header.hasLFE)
    #expect(header.channelCount == 6)
}

@Test func eac3HeaderGivesSizeAndBlocks() throws {
    // Independent substream, frmsiz 767, 48 kHz, 6 blocks, acmod 7 + LFE, bsid 16.
    let header = try #require(AC3FrameHeader([0x0B, 0x77, 0x02, 0xFF, 0x3F, 0x80, 0, 0]))
    #expect(header.isEnhanced)
    #expect(header.byteCount == 1536)
    #expect(header.sampleCount == 1536)
    #expect(header.channelCount == 6)
}

@Test func trueHDAccessUnitsAreNotAC3() {
    // A TrueHD access unit: check nibble + length, timing, then the major sync.
    #expect(AC3FrameHeader([0xF1, 0x2C, 0x00, 0x00, 0xF8, 0x72, 0x6F, 0xBA]) == nil)
}
