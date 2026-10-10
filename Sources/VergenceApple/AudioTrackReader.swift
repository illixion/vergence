/*
 Vergence - reading a video's audio tracks

 Decodes any audio track AVFoundation can, APAC included (Apple's spatial
 audio codec, which a Vision Pro or iPhone records next to a stereo AAC
 track and nothing outside Apple's frameworks decodes), to interleaved
 32-bit float.

 An ambisonic track (channel layout HOA ACN SN3D or N3D) always comes out as
 AmbiX: ACN channel order, SN3D normalisation, which is what YouTube, Opus
 (channel mapping family 2) and most ambisonic tools expect. A Vision Pro's
 APAC track decodes to first-order AmbiX already (4 channels, W Y Z X).
 */

// SpatialVideoError lives with the spatial video code, which tvOS lacks.
#if canImport(AVFoundation) && !os(tvOS)

import AVFoundation
import CoreMedia

public struct AudioTrackInfo: Sendable {
    public enum Normalization: String, Sendable {
        case sn3d = "SN3D", n3d = "N3D"
    }

    public let trackID: CMPersistentTrackID
    public let codec: String
    public let channels: Int
    public let sampleRate: Double
    /// Set for ambisonics: 1 for first order (4 channels), 2 for second (9)...
    public let ambisonicOrder: Int?
    public let normalization: Normalization?
    public let language: String?

    public var isAmbisonic: Bool { ambisonicOrder != nil }

    public var summary: String {
        var parts = ["\(codec)"]
        if let order = ambisonicOrder {
            let name = order == 1 ? "first" : order == 2 ? "second" : order == 3 ? "third" : "order \(order)"
            parts.append("\(name)-order ambisonics (\(channels) ch, ACN \(normalization?.rawValue ?? "?"))")
        } else {
            parts.append("\(channels) ch")
        }
        parts.append("\(Int(sampleRate)) Hz")
        if let language { parts.append(language) }
        return parts.joined(separator: " ")
    }
}

public enum AudioTrackReader {
    public static func tracks(_ url: URL) async throws -> [AudioTrackInfo] {
        let asset = AVURLAsset(url: url)
        var found: [AudioTrackInfo] = []
        for track in try await asset.loadTracks(withMediaType: .audio) {
            guard let format = try await track.load(.formatDescriptions).first,
                  let basic = CMAudioFormatDescriptionGetStreamBasicDescription(format)?.pointee else { continue }
            let channels = Int(basic.mChannelsPerFrame)
            let (order, normalization) = ambisonics(format, channels: channels)
            let language = try await track.load(.languageCode)
            found.append(AudioTrackInfo(trackID: track.trackID, codec: codecName(basic.mFormatID), channels: channels,
                                        sampleRate: basic.mSampleRate, ambisonicOrder: order, normalization: normalization,
                                        language: language == "und" ? nil : language))
        }
        return found
    }

    /// Decodes `trackID` from the start, handing interleaved float frames to
    /// `chunk` (`channels` values per frame). Returns the frame count.
    @discardableResult
    public static func decode(_ url: URL, trackID: CMPersistentTrackID,
                              _ chunk: (_ samples: UnsafeBufferPointer<Float>, _ channels: Int) throws -> Void) async throws -> Int {
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTrack(withTrackID: trackID) else { throw SpatialVideoError.unreadable("no track \(trackID)") }
        let info = try await tracks(url).first { $0.trackID == trackID }
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVLinearPCMBitDepthKey: 32,
            AVLinearPCMIsFloatKey: true,
            AVLinearPCMIsNonInterleaved: false,
            AVLinearPCMIsBigEndianKey: false,
        ])
        reader.add(output)
        guard reader.startReading() else { throw SpatialVideoError.unreadable(reader.error?.localizedDescription ?? "unknown") }
        // N3D to SN3D: each degree-l channel divided by sqrt(2l + 1).
        let gains: [Float]? = info.flatMap { info in
            guard info.normalization == .n3d else { return nil }
            return (0 ..< info.channels).map { acn in 1 / Float(2 * Int(Double(acn).squareRoot()) + 1).squareRoot() }
        }
        var frames = 0
        var scratch: [Float] = []
        while let sample = output.copyNextSampleBuffer() {
            guard let block = CMSampleBufferGetDataBuffer(sample),
                  let description = CMSampleBufferGetFormatDescription(sample),
                  let basic = CMAudioFormatDescriptionGetStreamBasicDescription(description)?.pointee else { continue }
            let channels = Int(basic.mChannelsPerFrame)
            let length = CMBlockBufferGetDataLength(block)
            scratch = [Float](repeating: 0, count: length / 4)
            let status = scratch.withUnsafeMutableBytes { CMBlockBufferCopyDataBytes(block, atOffset: 0, dataLength: length, destination: $0.baseAddress!) }
            guard status == noErr else { throw SpatialVideoError.unreadable("audio block \(status)") }
            if let gains {
                for i in scratch.indices { scratch[i] *= gains[i % channels] }
            }
            try scratch.withUnsafeBufferPointer { try chunk($0, channels) }
            frames += scratch.count / max(channels, 1)
        }
        if reader.status == .failed { throw SpatialVideoError.unreadable(reader.error?.localizedDescription ?? "unknown") }
        return frames
    }

    static func ambisonics(_ format: CMAudioFormatDescription, channels: Int) -> (Int?, AudioTrackInfo.Normalization?) {
        var size = 0
        guard let layout = CMAudioFormatDescriptionGetChannelLayout(format, sizeOut: &size)?.pointee else { return (nil, nil) }
        let tag = layout.mChannelLayoutTag
        let normalization: AudioTrackInfo.Normalization
        switch tag & 0xFFFF_0000 {
        case kAudioChannelLayoutTag_HOA_ACN_SN3D: normalization = .sn3d
        case kAudioChannelLayoutTag_HOA_ACN_N3D: normalization = .n3d
        default: return (nil, nil)
        }
        // (order + 1)² channels.
        let order = Int(Double(channels).squareRoot().rounded()) - 1
        guard order >= 1, (order + 1) * (order + 1) == channels else { return (nil, nil) }
        return (order, normalization)
    }

    static func codecName(_ id: AudioFormatID) -> String {
        switch id {
        case kAudioFormatMPEG4AAC, kAudioFormatMPEG4AAC_HE, kAudioFormatMPEG4AAC_HE_V2: "aac"
        case kAudioFormatAppleLossless: "alac"
        case kAudioFormatAC3: "ac3"
        case kAudioFormatEnhancedAC3: "eac3"
        case kAudioFormatLinearPCM: "pcm"
        case kAudioFormatOpus: "opus"
        case kAudioFormatFLAC: "flac"
        default:
            String(decoding: [24, 16, 8, 0].map { UInt8((id >> $0) & 0xFF) }, as: UTF8.self).trimmingCharacters(in: .whitespaces)
        }
    }
}
#endif
