/*
 Vergence - AC-3 and E-AC-3 decoding through Apple's AudioToolbox

 The Apple decoder behind `PlaylistAudioReader` (VergenceDisc), which reads
 a disc's AC-3 frames but leaves decoding to the platform.
 */

#if canImport(AVFoundation)
import AVFoundation
import Foundation

/// AC-3 / E-AC-3 frames to planar Float32 5.1 (L R C LFE Ls Rs) through
/// AudioToolbox. One decoder per channel configuration; a change rebuilds it.
public final class AC3Decoder: AudioFrameDecoder {
    public static let outputChannels = audioOutputChannels
    private var converter: AVAudioConverter?
    private var configuration: (enhanced: Bool, mode: Int, lfe: Bool)?
    /// For each decoder output channel, where it goes in the 5.1 output
    /// (a centre surround goes to both surrounds).
    private var routing: [[(slot: Int, gain: Float)]] = []
    private var output: AVAudioPCMBuffer?

    public init() {}

    /// Forgets decoder state, for a jump in the stream.
    public func reset() {
        converter?.reset()
    }

    /// Decodes one frame. Returns `sampleCount` frames per output channel
    /// (6 planes), or nil when the frame doesn't decode.
    public func decode(_ frame: Data) -> [[Float]]? {
        guard let header = AC3FrameHeader(frame), frame.count >= header.byteCount else { return nil }
        if configuration.map({ $0 != (header.isEnhanced, header.channelMode, header.hasLFE) }) ?? true {
            guard build(for: header) else { return nil }
        }
        guard let converter, let output else { return nil }
        let input = AVAudioCompressedBuffer(format: converter.inputFormat, packetCapacity: 1, maximumPacketSize: frame.count)
        frame.withUnsafeBytes { input.data.copyMemory(from: $0.baseAddress!, byteCount: frame.count) }
        input.byteLength = UInt32(frame.count)
        input.packetCount = 1
        input.packetDescriptions?.pointee = AudioStreamPacketDescription(mStartOffset: 0, mVariableFramesInPacket: 0,
                                                                          mDataByteSize: UInt32(frame.count))
        output.frameLength = 0
        var supplied = false
        var error: NSError?
        let status = converter.convert(to: output, error: &error) { _, inputStatus in
            if supplied {
                inputStatus.pointee = .noDataNow
                return nil
            }
            supplied = true
            inputStatus.pointee = .haveData
            return input
        }
        guard status != .error, let planes = output.floatChannelData else { return nil }
        let count = Int(output.frameLength)
        var result = [[Float]](repeating: [Float](repeating: 0, count: count), count: Self.outputChannels)
        for (channel, routes) in routing.enumerated() where channel < Int(output.format.channelCount) {
            let source = planes[channel]
            for route in routes {
                for i in 0 ..< count { result[route.slot][i] += source[i] * route.gain }
            }
        }
        return result
    }

    private func build(for header: AC3FrameHeader) -> Bool {
        let tag = Self.layoutTag(mode: header.channelMode, lfe: header.hasLFE)
        guard let layout = AVAudioChannelLayout(layoutTag: tag) else { return false }
        var description = AudioStreamBasicDescription(
            mSampleRate: Double(header.sampleRate),
            mFormatID: header.isEnhanced ? kAudioFormatEnhancedAC3 : kAudioFormatAC3,
            mFormatFlags: 0, mBytesPerPacket: 0, mFramesPerPacket: UInt32(header.sampleCount),
            mBytesPerFrame: 0, mChannelsPerFrame: UInt32(header.channelCount), mBitsPerChannel: 0, mReserved: 0)
        let inputFormat = AVAudioFormat(streamDescription: &description, channelLayout: layout)
        guard let inputFormat else { return false }
        let outputFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(header.sampleRate),
                                         interleaved: false, channelLayout: layout)
        guard let converter = AVAudioConverter(from: inputFormat, to: outputFormat),
              let output = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: 1536 * 2) else { return false }
        self.converter = converter
        self.output = output
        routing = Self.labels(of: tag).map { label in
            switch label {
            case kAudioChannelLabel_Left: [(0, 1)]
            case kAudioChannelLabel_Right: [(1, 1)]
            case kAudioChannelLabel_Center, kAudioChannelLabel_Mono: [(2, 1)]
            case kAudioChannelLabel_LFEScreen: [(3, 1)]
            case kAudioChannelLabel_LeftSurround: [(4, 1)]
            case kAudioChannelLabel_RightSurround: [(5, 1)]
            case kAudioChannelLabel_CenterSurround: [(4, 0.7071), (5, 0.7071)]
            default: []
            }
        }
        configuration = (header.isEnhanced, header.channelMode, header.hasLFE)
        return true
    }

    /// The layout a frame's channel mode decodes to, in AC-3's own channel order.
    static func layoutTag(mode: Int, lfe: Bool) -> AudioChannelLayoutTag {
        switch (mode, lfe) {
        case (1, false): kAudioChannelLayoutTag_Mono
        case (1, true): kAudioChannelLayoutTag_AC3_1_0_1
        case (3, false): kAudioChannelLayoutTag_AC3_3_0
        case (3, true): kAudioChannelLayoutTag_AC3_3_0_1
        case (4, false): kAudioChannelLayoutTag_ITU_2_1
        case (4, true): kAudioChannelLayoutTag_AC3_2_1_1
        case (5, false): kAudioChannelLayoutTag_AC3_3_1
        case (5, true): kAudioChannelLayoutTag_AC3_3_1_1
        case (6, false): kAudioChannelLayoutTag_ITU_2_2
        case (6, true): kAudioChannelLayoutTag_DVD_18
        case (7, false): kAudioChannelLayoutTag_MPEG_5_0_C
        case (7, true): kAudioChannelLayoutTag_MPEG_5_1_C
        case (_, true): kAudioChannelLayoutTag_DVD_4
        default: kAudioChannelLayoutTag_Stereo
        }
    }

    /// A layout tag's channel labels, in order.
    static func labels(of tag: AudioChannelLayoutTag) -> [AudioChannelLabel] {
        var tag = tag
        var size: UInt32 = 0
        guard AudioFormatGetPropertyInfo(kAudioFormatProperty_ChannelLayoutForTag, UInt32(MemoryLayout<AudioChannelLayoutTag>.size),
                                         &tag, &size) == noErr, size > 0 else { return [] }
        let raw = UnsafeMutableRawPointer.allocate(byteCount: Int(size), alignment: MemoryLayout<AudioChannelLayout>.alignment)
        defer { raw.deallocate() }
        guard AudioFormatGetProperty(kAudioFormatProperty_ChannelLayoutForTag, UInt32(MemoryLayout<AudioChannelLayoutTag>.size),
                                     &tag, &size, raw) == noErr else { return [] }
        let layout = raw.assumingMemoryBound(to: AudioChannelLayout.self)
        let count = Int(layout.pointee.mNumberChannelDescriptions)
        return withUnsafeMutablePointer(to: &layout.pointee.mChannelDescriptions) { first in
            first.withMemoryRebound(to: AudioChannelDescription.self, capacity: count) { descriptions in
                (0 ..< count).map { descriptions[$0].mChannelLabel }
            }
        }
    }
}

extension PlaylistAudioReader {
    /// A reader decoding through Apple's AC-3 decoder.
    public convenience init(disc: BlurayDisc, playlist: BlurayPlaylist, pid: UInt16) {
        self.init(disc: disc, playlist: playlist, pid: pid, decoder: AC3Decoder())
    }
}
#endif
