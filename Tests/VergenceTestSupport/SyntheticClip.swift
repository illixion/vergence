#if os(macOS)
import AVFoundation
import CoreMedia
import CoreVideo
import Foundation
import VideoToolbox

/// Synthetic side-by-side clips for the tests that need a real video file:
/// the Apple tests and the command line's end-to-end ones.
public enum SyntheticClip {
    public static let eye = (width: 256, height: 144)
    public static let disparity = 8
    public static let frames = 12
    public static let audioRate = 48_000

    public struct Failure: Error, CustomStringConvertible {
        public var description: String
        init(_ description: String) { self.description = description }
    }

    /// A noise scene seen by two eyes `disparity` pixels apart, side by side,
    /// with the left eye's top-left pixel marked bright and the right's dark.
    /// `audio` adds a stereo AAC track (48 kHz) as long as the video; `hdr`
    /// encodes 10-bit HEVC tagged BT.2020 PQ instead of 8-bit H.264.
    public static func writeSideBySide(to url: URL, frames: Int = frames, audio: Bool = false, hdr: Bool = false) async throws {
        let width = eye.width * 2, height = eye.height
        let writer = try AVAssetWriter(outputURL: url, fileType: hdr ? .mov : .mp4)
        var settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: width, AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: [AVVideoAverageBitRateKey: 8_000_000],
        ]
        if hdr {
            settings[AVVideoCodecKey] = AVVideoCodecType.hevc
            settings[AVVideoCompressionPropertiesKey] = [
                AVVideoAverageBitRateKey: 8_000_000,
                AVVideoProfileLevelKey: kVTProfileLevel_HEVC_Main10_AutoLevel as String,
            ]
            settings[AVVideoColorPropertiesKey] = [
                AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_2020,
                AVVideoTransferFunctionKey: AVVideoTransferFunction_SMPTE_ST_2084_PQ,
                AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_2020,
            ]
        }
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        input.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width, kCVPixelBufferHeightKey as String: height,
        ])
        writer.add(input)
        var audioInput: AVAssetWriterInput?
        if audio {
            let made = AVAssetWriterInput(mediaType: .audio, outputSettings: [
                AVFormatIDKey: kAudioFormatMPEG4AAC, AVSampleRateKey: audioRate, AVNumberOfChannelsKey: 2,
                AVEncoderBitRateKey: 128_000,
            ])
            made.expectsMediaDataInRealTime = false
            writer.add(made)
            audioInput = made
        }
        guard writer.startWriting() else { throw Failure("can't start writing: \(String(describing: writer.error))") }
        writer.startSession(atSourceTime: .zero)
        let sceneWidth = eye.width + disparity
        func picture(_ frame: Int) throws -> CVPixelBuffer {
            var state = UInt64(frame + 1) &* 0x9E37_79B9_7F4A_7C15 | 1
            let scene = (0 ..< sceneWidth * height).map { _ -> UInt8 in
                state ^= state << 13; state ^= state >> 7; state ^= state << 17
                return UInt8(truncatingIfNeeded: state >> 24)
            }
            guard let pool = adaptor.pixelBufferPool else { throw Failure("no pixel buffer pool") }
            var buffer: CVPixelBuffer?
            CVPixelBufferPoolCreatePixelBuffer(nil, pool, &buffer)
            guard let picture = buffer else { throw Failure("no pixel buffer") }
            CVPixelBufferLockBaseAddress(picture, [])
            defer { CVPixelBufferUnlockBaseAddress(picture, []) }
            guard let base = CVPixelBufferGetBaseAddress(picture)?.assumingMemoryBound(to: UInt8.self) else {
                throw Failure("no pixel buffer memory")
            }
            let stride = CVPixelBufferGetBytesPerRow(picture)
            for y in 0 ..< height {
                for x in 0 ..< width {
                    let isRight = x >= eye.width
                    let v = scene[y * sceneWidth + (isRight ? x - eye.width + disparity : x)]
                    let p = base + y * stride + x * 4
                    p[0] = v; p[1] = v; p[2] = v; p[3] = 255
                }
            }
            // A bright left eye band and a dark right eye band at the top.
            for y in 0 ..< 16 {
                for x in 0 ..< width {
                    let p = base + y * stride + x * 4
                    let v: UInt8 = x < eye.width ? 235 : 16
                    p[0] = v; p[1] = v; p[2] = v
                }
            }
            return picture
        }
        // As long as the video, in tenth-of-a-second chunks of a 440 Hz tone.
        let audioSamples = audio ? frames * audioRate / 24 : 0
        var frame = 0, written = 0
        while frame < frames || written < audioSamples {
            var moved = false
            if frame < frames, input.isReadyForMoreMediaData {
                guard adaptor.append(try picture(frame), withPresentationTime: CMTime(value: CMTimeValue(frame), timescale: 24)) else {
                    throw Failure("frame \(frame) not taken: \(String(describing: writer.error))")
                }
                frame += 1
                if frame == frames { input.markAsFinished() }
                moved = true
            }
            if written < audioSamples, let audioInput, audioInput.isReadyForMoreMediaData {
                let count = min(audioRate / 10, audioSamples - written)
                guard audioInput.append(try tone(at: written, count: count)) else {
                    throw Failure("audio at \(written) not taken: \(String(describing: writer.error))")
                }
                written += count
                if written == audioSamples { audioInput.markAsFinished() }
                moved = true
            }
            if !moved { try await Task.sleep(for: .milliseconds(2)) }
        }
        await writer.finishWriting()
        guard writer.status == .completed else { throw Failure("writing failed: \(String(describing: writer.error))") }
    }

    /// `count` frames of a stereo 16-bit 440 Hz tone starting at frame `start`.
    public static func tone(at start: Int, count: Int) throws -> CMSampleBuffer {
        var description = AudioStreamBasicDescription(
            mSampleRate: Double(audioRate), mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: kAudioFormatFlagIsSignedInteger | kAudioFormatFlagIsPacked, mBytesPerPacket: 4, mFramesPerPacket: 1,
            mBytesPerFrame: 4, mChannelsPerFrame: 2, mBitsPerChannel: 16, mReserved: 0)
        var format: CMAudioFormatDescription?
        CMAudioFormatDescriptionCreate(allocator: nil, asbd: &description, layoutSize: 0, layout: nil, magicCookieSize: 0,
                                       magicCookie: nil, extensions: nil, formatDescriptionOut: &format)
        var samples = [Int16](repeating: 0, count: count * 2)
        for i in 0 ..< count {
            let v = Int16(8000 * sin(2 * Double.pi * 440 * Double(start + i) / Double(audioRate)))
            samples[2 * i] = v; samples[2 * i + 1] = v
        }
        var block: CMBlockBuffer?
        CMBlockBufferCreateWithMemoryBlock(allocator: nil, memoryBlock: nil, blockLength: count * 4, blockAllocator: nil,
                                           customBlockSource: nil, offsetToData: 0, dataLength: count * 4, flags: 0,
                                           blockBufferOut: &block)
        guard let data = block, let format else { throw Failure("no audio buffer") }
        samples.withUnsafeBytes { _ = CMBlockBufferReplaceDataBytes(with: $0.baseAddress!, blockBuffer: data, offsetIntoDestination: 0, dataLength: $0.count) }
        var sample: CMSampleBuffer?
        CMAudioSampleBufferCreateReadyWithPacketDescriptions(allocator: nil, dataBuffer: data, formatDescription: format,
                                                             sampleCount: count,
                                                             presentationTimeStamp: CMTime(value: CMTimeValue(start), timescale: CMTimeScale(audioRate)),
                                                             packetDescriptions: nil, sampleBufferOut: &sample)
        guard let sample else { throw Failure("no audio sample") }
        return sample
    }
}
#endif
