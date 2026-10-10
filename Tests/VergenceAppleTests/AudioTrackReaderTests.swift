#if os(macOS)
import AVFoundation
import CoreMedia
import Testing
@testable import VergenceApple

@Suite struct AudioTrackReaderTests {
    /// A 4-channel track tagged first-order ambisonics with `tag`, every
    /// channel a constant 0.5.
    static func writeAmbisonicTrack(to url: URL, tag: AudioChannelLayoutTag) async throws {
        var layout = AudioChannelLayout()
        layout.mChannelLayoutTag = tag | 4
        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: 48_000, AVNumberOfChannelsKey: 4,
            AVLinearPCMBitDepthKey: 32, AVLinearPCMIsFloatKey: true, AVLinearPCMIsNonInterleaved: false,
            AVLinearPCMIsBigEndianKey: false,
            AVChannelLayoutKey: Data(bytes: &layout, count: MemoryLayout<AudioChannelLayout>.size),
        ]
        let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        let input = AVAssetWriterInput(mediaType: .audio, outputSettings: settings)
        writer.add(input)
        #expect(writer.startWriting())
        writer.startSession(atSourceTime: .zero)

        var description = AudioStreamBasicDescription(
            mSampleRate: 48_000, mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked, mBytesPerPacket: 16, mFramesPerPacket: 1,
            mBytesPerFrame: 16, mChannelsPerFrame: 4, mBitsPerChannel: 32, mReserved: 0)
        var format: CMAudioFormatDescription?
        CMAudioFormatDescriptionCreate(allocator: nil, asbd: &description, layoutSize: MemoryLayout<AudioChannelLayout>.size,
                                       layout: &layout, magicCookieSize: 0, magicCookie: nil, extensions: nil,
                                       formatDescriptionOut: &format)
        let frames = 4800
        var samples = [Float](repeating: 0.5, count: frames * 4)
        var block: CMBlockBuffer?
        CMBlockBufferCreateWithMemoryBlock(allocator: nil, memoryBlock: nil, blockLength: samples.count * 4, blockAllocator: nil,
                                           customBlockSource: nil, offsetToData: 0, dataLength: samples.count * 4, flags: 0,
                                           blockBufferOut: &block)
        samples.withUnsafeMutableBytes { _ = CMBlockBufferReplaceDataBytes(with: $0.baseAddress!, blockBuffer: block!, offsetIntoDestination: 0, dataLength: $0.count) }
        var sample: CMSampleBuffer?
        CMAudioSampleBufferCreateReadyWithPacketDescriptions(allocator: nil, dataBuffer: block!, formatDescription: try #require(format),
                                                             sampleCount: frames, presentationTimeStamp: .zero,
                                                             packetDescriptions: nil, sampleBufferOut: &sample)
        while !input.isReadyForMoreMediaData { try await Task.sleep(for: .milliseconds(2)) }
        #expect(input.append(try #require(sample)))
        input.markAsFinished()
        await writer.finishWriting()
        #expect(writer.status == .completed, "\(String(describing: writer.error))")
    }

    @Test func ambisonicsAreRecognisedAndComeOutAsAmbiX() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("vergence-audio-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        for (tag, normalization) in [(kAudioChannelLayoutTag_HOA_ACN_SN3D, AudioTrackInfo.Normalization.sn3d),
                                     (kAudioChannelLayoutTag_HOA_ACN_N3D, .n3d)] {
            let url = directory.appendingPathComponent("\(normalization.rawValue).mov")
            try await Self.writeAmbisonicTrack(to: url, tag: tag)
            let track = try #require(try await AudioTrackReader.tracks(url).first)
            #expect(track.ambisonicOrder == 1)
            #expect(track.normalization == normalization)

            var firstFrame: [Float] = []
            let frames = try await AudioTrackReader.decode(url, trackID: track.trackID) { samples, channels in
                if firstFrame.isEmpty { firstFrame = Array(samples.prefix(channels)) }
            }
            #expect(frames == 4800)
            // SN3D passes through; N3D's first-order channels shrink by √3.
            let expected: [Float] = normalization == .sn3d ? [0.5, 0.5, 0.5, 0.5] : [0.5, 0.5 / Float(3).squareRoot(), 0.5 / Float(3).squareRoot(), 0.5 / Float(3).squareRoot()]
            #expect(zip(firstFrame, expected).allSatisfy { abs($0 - $1) < 1e-5 }, "\(firstFrame)")
        }
    }
}
#endif
