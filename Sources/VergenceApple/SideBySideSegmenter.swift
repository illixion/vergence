/*
 Vergence - spatial video to side by side, as stream segments

 `SideBySideExporter` makes one file of the whole video. A server streaming
 to a Quest wants a few seconds at a time instead, starting wherever the
 viewer seeks, without converting (or storing) the whole thing first. This
 writes the same picture as HLS fragmented-MP4 segments on a `SegmentGrid`:
 `init.mp4` and `<n>.m4s` for segments `first...last`, each written to a
 temporary name and renamed once complete, so a server can serve each one as
 soon as it appears, while the run goes on.

 - The encoder is driven directly (VideoToolbox) so that every segment
   starts with a forced keyframe at its grid time, and without B-frames, so
   a keyframe is never preceded in presentation by frames that follow it in
   decoding. Each segment decodes on its own; segments from different runs
   join.
 - Fragment times are the source's own presentation times plus
   `SideBySideSegments.timeOffset`, so a segment plays at its place in the
   video whichever run made it. The offset leaves room for audio that
   starts before the first frame (AAC priming), which a fragment can't
   place before zero; a stream player times segments by the playlist, so
   it never shows.
 - Audio: the first plain AAC track, copied packet for packet, each packet in
   the segment its time falls in, and placed by its own source time (less
   the priming the reader reports), not by the writer's placing against the
   video: the writer rounds that to a video tick, which moved a run's audio
   by up to one, so it jumped where two runs' segments joined. An ambisonic
   track is left out: it would have to be re-encoded, and an encoder
   restarted at every seek primes differently, which shows as clicks where
   runs join.
 - HDR keeps the source's colour tags, set on the encoder: it writes only
   what it's told into the stream, whatever the pictures carry.
 */

#if canImport(AVFoundation) && !os(tvOS)

import AVFoundation
import CoreMedia
import CoreVideo
import UniformTypeIdentifiers
import VideoToolbox
import VergenceCore

public struct SideBySideSegments: Sendable {
    /// The picture: size, quality, bitrate. Its output and ambisonics are ignored.
    public var job: SideBySideJob
    public var directory: URL
    public var segmentSeconds: Double
    public var first: Int
    /// The last segment to make (inclusive); nil runs to the end.
    public var last: Int?
    public var includeAudio: Bool

    public init(job: SideBySideJob, directory: URL, segmentSeconds: Double = 4, first: Int = 0, last: Int? = nil,
                includeAudio: Bool = true) {
        self.job = job
        self.directory = directory
        self.segmentSeconds = segmentSeconds
        self.first = first
        self.last = last
        self.includeAudio = includeAudio
    }

    public var grid: SegmentGrid {
        SegmentGrid(segmentSeconds: segmentSeconds, durationSeconds: job.source.duration, frameRate: job.source.frameRate,
                    lastFrameSeconds: job.source.lastFrameTime)
    }

    /// Added to every time in the segments (see the top of the file).
    public static let timeOffset = CMTime(value: 1, timescale: 1)

    public static let initName = "init.mp4"
    public static func segmentName(_ n: Int) -> String { "\(n).m4s" }
}

public struct SideBySideSegmentsResult: Sendable {
    public let segments: [Int]
    public let frames: Int
    public let bitsPerSecond: Int
    public let audio: Bool
}

public enum SideBySideSegmenter {
    public static func run(_ request: SideBySideSegments, segmentDone: @escaping @Sendable (Int) -> Void = { _ in }) async throws -> SideBySideSegmentsResult {
        let job = request.job
        guard job.source.isMultiviewStereo else { throw SpatialVideoError.unreadable("not MV-HEVC stereo") }
        let grid = request.grid
        guard request.segmentSeconds >= 0.5 else { throw SpatialVideoError.writer("segments must be at least half a second") }
        let last = min(request.last ?? grid.count - 1, grid.count - 1)
        guard request.first >= 0, request.first <= last else {
            throw SpatialVideoError.writer("segment \(request.first) is outside 0...\(grid.count - 1)")
        }
        try FileManager.default.createDirectory(at: request.directory, withIntermediateDirectories: true)

        let asset = AVURLAsset(url: job.source.url)
        guard let videoTrack = try await asset.loadTracks(withMediaType: .video).first else { throw SpatialVideoError.noVideoTrack }
        let (range, naturalTimeScale) = try await videoTrack.load(.timeRange, .naturalTimeScale)
        let origin = range.start
        let timescale = CMTimeScale(max(600, origin.timescale))
        func gridTime(_ segment: Int) -> CMTime {
            CMTimeAdd(origin, CMTime(seconds: grid.start(segment), preferredTimescale: timescale))
        }
        let start = gridTime(request.first)
        let end = last == grid.count - 1 ? CMTime.positiveInfinity : gridTime(last + 1)

        let hdr = job.source.isHDR
        let tenBit = job.source.needsTenBit
        let pixelFormat = tenBit ? kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange : kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
        let eye = job.storedEye, picture = job.pictureSize
        let bitsPerSecond = job.resolvedBitsPerSecond

        let reader = try AVAssetReader(asset: asset)
        // Read from a little before: a reader clamps the first sample of its
        // range to the range's start, which would move a frame from the
        // segment before into this one. Earlier frames are dropped below.
        let lead = CMTimeMaximum(origin, CMTimeSubtract(start, CMTime(seconds: 1, preferredTimescale: timescale)))
        reader.timeRange = CMTimeRange(start: lead, end: end)
        let videoOutput = AVAssetReaderTrackOutput(track: videoTrack, outputSettings: [
            AVVideoDecompressionPropertiesKey: [kVTDecompressionPropertyKey_RequestedMVHEVCVideoLayerIDs: [0, 1]],
            kCVPixelBufferPixelFormatTypeKey as String: pixelFormat,
            kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
        ])
        videoOutput.alwaysCopiesSampleData = false
        reader.add(videoOutput)

        var audioOutput: AVAssetReaderTrackOutput?
        var audioFormat: CMFormatDescription?
        var audioLead = CMTime.zero
        if request.includeAudio {
            let infos = try await AudioTrackReader.tracks(job.source.url)
            for (track, info) in zip(try await asset.loadTracks(withMediaType: .audio), infos)
            where info.ambisonicOrder == nil && info.codec == "aac" {
                guard let format = try await track.load(.formatDescriptions).first else { continue }
                let output = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
                guard reader.canAdd(output) else { continue }
                reader.add(output)
                audioOutput = output
                audioFormat = format
                audioLead = try primingLead(asset, track: track)
                break
            }
        }

        // The encoder tags its stream only with what it's told, whatever the
        // pictures carry: HDR keeps the source's tags, the rest is BT.709.
        let color: SegmentEncoder.Color = hdr
            ? ((job.source.colorPrimaries ?? AVVideoColorPrimaries_ITU_R_2020) as CFString,
               (job.source.transferFunction ?? AVVideoTransferFunction_SMPTE_ST_2084_PQ) as CFString,
               (job.source.yCbCrMatrix ?? AVVideoYCbCrMatrix_ITU_R_2020) as CFString)
            : (kCVImageBufferColorPrimaries_ITU_R_709_2, kCVImageBufferTransferFunction_ITU_R_709_2, kCVImageBufferYCbCrMatrix_ITU_R_709_2)
        let encoder = try SegmentEncoder(width: picture.width, height: picture.height, tenBit: tenBit, color: color,
                                         bitsPerSecond: bitsPerSecond, frameRate: job.source.frameRate,
                                         keyFrameSeconds: request.segmentSeconds)
        defer { encoder.invalidate() }

        var transfer: VTPixelTransferSession?
        VTPixelTransferSessionCreate(allocator: kCFAllocatorDefault, pixelTransferSessionOut: &transfer)
        guard let transfer else { throw SpatialVideoError.writer("can't create a pixel transfer session") }
        defer { VTPixelTransferSessionInvalidate(transfer) }
        if !hdr {
            VTSessionSetProperty(transfer, key: kVTPixelTransferPropertyKey_DestinationColorPrimaries, value: kCVImageBufferColorPrimaries_ITU_R_709_2)
            VTSessionSetProperty(transfer, key: kVTPixelTransferPropertyKey_DestinationTransferFunction, value: kCVImageBufferTransferFunction_ITU_R_709_2)
            VTSessionSetProperty(transfer, key: kVTPixelTransferPropertyKey_DestinationYCbCrMatrix, value: kCVImageBufferYCbCrMatrix_ITU_R_709_2)
        }
        func pool(_ width: Int, _ height: Int) -> CVPixelBufferPool? {
            var pool: CVPixelBufferPool?
            CVPixelBufferPoolCreate(nil, nil, [
                kCVPixelBufferPixelFormatTypeKey as String: pixelFormat,
                kCVPixelBufferWidthKey as String: width, kCVPixelBufferHeightKey as String: height,
                kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
            ] as CFDictionary, &pool)
            return pool
        }
        guard let eyePool = pool(eye.width, eye.height), let picturePool = pool(picture.width, picture.height) else {
            throw SpatialVideoError.writer("can't make picture buffers")
        }

        guard reader.startReading() else { throw SpatialVideoError.unreadable(reader.error?.localizedDescription ?? "unknown") }
        defer { if reader.status == .reading { reader.cancelReading() } }

        let sink = SegmentSink(directory: request.directory, first: request.first, done: segmentDone)
        var writer: AVAssetWriter?
        var videoInput: AVAssetWriterInput?
        var audioInput: AVAssetWriterInput?
        var pendingVideo: CMSampleBuffer?
        var pendingAudio: CMSampleBuffer?
        var videoFinished = false, audioFinished = audioOutput == nil
        var frames = 0

        for segment in request.first ... last {
            let segmentEnd = segment == last ? end : gridTime(segment + 1)
            // Video: decode, pack and encode this segment's frames.
            var firstInSegment = true
            while !videoFinished {
                let sample: CMSampleBuffer
                if let held = pendingVideo { sample = held; pendingVideo = nil }
                else if let next = videoOutput.copyNextSampleBuffer() { sample = next }
                else {
                    if reader.status == .failed { throw SpatialVideoError.unreadable(reader.error?.localizedDescription ?? "unknown") }
                    videoFinished = true
                    break
                }
                let time = CMSampleBufferGetPresentationTimeStamp(sample)
                if CMTimeCompare(time, start) < 0 { continue }
                if CMTimeCompare(time, segmentEnd) >= 0 { pendingVideo = sample; break }
                guard let views = sample.taggedBuffers else { continue }
                let packed = try SideBySideExporter.pack(views, transfer: transfer, eyePool: eyePool, pool: picturePool)
                try encoder.encode(packed, time: time, duration: CMSampleBufferGetDuration(sample), keyFrame: firstInSegment)
                firstInSegment = false
                frames += 1
            }
            let encoded = try encoder.flush()
            if encoded.isEmpty { break }

            if writer == nil {
                // The writer needs the encoder's format before it starts.
                guard let format = CMSampleBufferGetFormatDescription(encoded[0]) else { throw SpatialVideoError.writer("the encoder gave no format") }
                let made = AVAssetWriter(contentType: UTType(AVFileType.mp4.rawValue) ?? .mpeg4Movie)
                made.outputFileTypeProfile = .mpeg4AppleHLS
                made.preferredOutputSegmentInterval = .indefinite
                made.delegate = sink
                let video = AVAssetWriterInput(mediaType: .video, outputSettings: nil, sourceFormatHint: format)
                video.expectsMediaDataInRealTime = false
                // The source's own clock, so frame times come through exactly.
                video.mediaTimeScale = naturalTimeScale
                made.add(video)
                if let audioFormat {
                    let audio = AVAssetWriterInput(mediaType: .audio, outputSettings: nil, sourceFormatHint: audioFormat)
                    audio.expectsMediaDataInRealTime = false
                    if made.canAdd(audio) { made.add(audio); audioInput = audio } else { audioFinished = true }
                }
                guard made.startWriting() else { throw SpatialVideoError.writer(made.error?.localizedDescription ?? "unknown") }
                made.startSession(atSourceTime: CMSampleBufferGetPresentationTimeStamp(encoded[0]))
                writer = made
                videoInput = video
            }
            guard let writer, let videoInput else { break }

            // Audio packets presented before the segment's end, the buffer
            // that straddles it split there.
            var audio: [CMSampleBuffer] = []
            while !audioFinished, let output = audioOutput {
                let sample: CMSampleBuffer
                if let held = pendingAudio { sample = held; pendingAudio = nil }
                else if let next = output.copyNextSampleBuffer() { sample = next }
                else { audioFinished = true; break }
                let (before, after) = try split(sample, from: start, at: segmentEnd)
                if let before { audio.append(before) }
                if let after { pendingAudio = after; break }
            }

            // Where this segment's first frame and first audio packet play,
            // which the sink writes into it.
            sink.expect(segment, video: CMTimeAdd(CMSampleBufferGetPresentationTimeStamp(encoded[0]), SideBySideSegments.timeOffset),
                        audio: audio.first.map {
                            CMTimeAdd(CMTimeSubtract(CMSampleBufferGetPresentationTimeStamp($0), audioLead), SideBySideSegments.timeOffset)
                        })

            // Interleaved, as the writer asks for each.
            var v = encoded[...], a = audio[...]
            while !v.isEmpty || !a.isEmpty {
                var moved = false
                if let next = v.first, videoInput.isReadyForMoreMediaData {
                    guard videoInput.append(next) else { throw SpatialVideoError.writer(writer.error?.localizedDescription ?? "frame refused") }
                    v = v.dropFirst()
                    moved = true
                }
                if let next = a.first, let audioInput, audioInput.isReadyForMoreMediaData {
                    guard audioInput.append(next) else { throw SpatialVideoError.writer(writer.error?.localizedDescription ?? "audio refused") }
                    a = a.dropFirst()
                    moved = true
                }
                if audioInput == nil { a = [] }
                if writer.status == .failed { throw SpatialVideoError.writer(writer.error?.localizedDescription ?? "unknown") }
                if !moved { try await Task.sleep(for: .milliseconds(1)) }
            }
            if segment < last && !videoFinished { writer.flushSegment() }
            try sink.check()
            if videoFinished { break }
        }

        if let writer {
            videoInput?.markAsFinished()
            audioInput?.markAsFinished()
            await writer.finishWriting()
            guard writer.status == .completed else { throw SpatialVideoError.writer(writer.error?.localizedDescription ?? "unknown") }
        }
        try sink.check()
        return SideBySideSegmentsResult(segments: sink.written, frames: frames, bitsPerSecond: bitsPerSecond, audio: audioInput != nil)
    }

    /// How far an audio track's packet times, as a reader gives them, run
    /// ahead of where the packets play: an AAC track's first buffer carries
    /// its priming as a trim, and its output time is where the first sample
    /// after the trim plays, so its first packet plays a trim before that.
    static func primingLead(_ asset: AVAsset, track: AVAssetTrack) throws -> CMTime {
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
        reader.add(output)
        guard reader.startReading() else { return .zero }
        defer { reader.cancelReading() }
        while let sample = output.copyNextSampleBuffer() {
            guard CMSampleBufferGetNumSamples(sample) > 0 else { continue }
            var trim = CMTime.zero
            if let value = CMGetAttachment(sample, key: kCMSampleBufferAttachmentKey_TrimDurationAtStart, attachmentModeOut: nil) {
                let time = CMTimeMakeFromDictionary((value as! CFDictionary))
                if time.isValid { trim = time }
            }
            let plays = CMTimeSubtract(CMSampleBufferGetOutputPresentationTimeStamp(sample), trim)
            return CMTimeSubtract(CMSampleBufferGetPresentationTimeStamp(sample), plays)
        }
        return .zero
    }

    /// The packets of `sample` presented in [`from`, `at`), and the rest.
    static func split(_ sample: CMSampleBuffer, from: CMTime, at: CMTime) throws -> (CMSampleBuffer?, CMSampleBuffer?) {
        let count = CMSampleBufferGetNumSamples(sample)
        guard count > 0 else { return (nil, nil) }
        var timing = [CMSampleTimingInfo](repeating: CMSampleTimingInfo(), count: count)
        var got = 0
        CMSampleBufferGetSampleTimingInfoArray(sample, entryCount: count, arrayToFill: &timing, entriesNeededOut: &got)
        let times: [CMTime]
        if got == count {
            times = timing.map(\.presentationTimeStamp)
        } else {
            // One entry for every packet: the first's time, a packet apart.
            let first = got > 0 ? timing[0] : CMSampleTimingInfo(duration: .invalid, presentationTimeStamp: CMSampleBufferGetPresentationTimeStamp(sample), decodeTimeStamp: .invalid)
            let step = first.duration.isValid ? first.duration : CMTimeMultiplyByRatio(CMSampleBufferGetDuration(sample), multiplier: 1, divisor: Int32(count))
            times = (0 ..< count).map { CMTimeAdd(first.presentationTimeStamp, CMTimeMultiply(step, multiplier: Int32($0))) }
        }
        let lower = times.firstIndex { CMTimeCompare($0, from) >= 0 } ?? count
        let upper = times.firstIndex { CMTimeCompare($0, at) >= 0 } ?? count
        func range(_ r: Range<Int>) throws -> CMSampleBuffer? {
            guard !r.isEmpty else { return nil }
            if r.count == count { return sample }
            var out: CMSampleBuffer?
            let status = CMSampleBufferCopySampleBufferForRange(allocator: nil, sampleBuffer: sample,
                                                                sampleRange: CFRange(location: r.lowerBound, length: r.count), sampleBufferOut: &out)
            guard status == noErr, let out else { throw SpatialVideoError.writer("splitting audio failed (\(status))") }
            // A trim (priming) belongs to the buffer's first packets only.
            if r.lowerBound > 0 { CMRemoveAttachment(out, key: kCMSampleBufferAttachmentKey_TrimDurationAtStart) }
            return out
        }
        return (try range(min(lower, upper) ..< upper), try range(max(lower, upper) ..< count))
    }
}

/// Writes what the asset writer hands over: the init segment once, then each
/// media segment under its number, by way of a temporary name.
private final class SegmentSink: NSObject, AVAssetWriterDelegate, @unchecked Sendable {
    let directory: URL
    var next: Int
    var written: [Int] = []
    var failure: Error?
    let done: @Sendable (Int) -> Void
    private let lock = NSLock()
    /// Per segment, where its first frame plays (track 1, the video, as it
    /// was added first) and its first audio packet (track 2), if it has one.
    private var expected: [Int: (video: CMTime, audio: CMTime?)] = [:]
    private var timescales: [UInt32: UInt32] = [:]

    func expect(_ segment: Int, video: CMTime, audio: CMTime?) {
        lock.lock()
        expected[segment] = (video, audio)
        lock.unlock()
    }

    init(directory: URL, first: Int, done: @escaping @Sendable (Int) -> Void) {
        self.directory = directory
        self.next = first
        self.done = done
    }

    func check() throws { if let failure { throw failure } }

    func assetWriter(_ writer: AVAssetWriter, didOutputSegmentData data: Data, segmentType: AVAssetSegmentType) {
        do {
            switch segmentType {
            case .initialization:
                timescales = FragmentTimes.timescales(initSegment: Data(data))
                let url = directory.appendingPathComponent(SideBySideSegments.initName)
                if !FileManager.default.fileExists(atPath: url.path) { try place(data, at: url) }
            case .separable:
                // The writer's times are its session's; put back the source's.
                var segment = Data(data)
                lock.lock()
                let time = expected.removeValue(forKey: next)
                lock.unlock()
                guard let time, let scale = timescales[1] else { throw SpatialVideoError.writer("segment \(next) came out of nowhere") }
                func ticks(_ time: CMTime, _ scale: UInt32) -> Int64 {
                    CMTimeConvertScale(time, timescale: Int32(scale), method: .roundHalfAwayFromZero).value
                }
                // Each track to its own time, the other left where it is.
                try FragmentTimes.retime(&segment, track: 1, to: ticks(time.video, scale), timescales: timescales, othersBy: 0)
                if let audio = time.audio, let audioScale = timescales[2], FragmentTimes.firstPresentation(segment)[2] != nil {
                    try FragmentTimes.retime(&segment, track: 2, to: ticks(audio, audioScale), timescales: timescales, othersBy: 0)
                }
                FragmentTimes.renumber(&segment, segment: next)
                try place(segment, at: directory.appendingPathComponent(SideBySideSegments.segmentName(next)))
                written.append(next)
                done(next)
                next += 1
            @unknown default:
                break
            }
        } catch {
            failure = error
        }
    }

    private func place(_ data: Data, at url: URL) throws {
        let temporary = url.deletingLastPathComponent().appendingPathComponent(".\(url.lastPathComponent).\(UUID().uuidString.prefix(8))")
        try data.write(to: temporary)
        if FileManager.default.fileExists(atPath: url.path) { try FileManager.default.removeItem(at: url) }
        try FileManager.default.moveItem(at: temporary, to: url)
    }
}

/// An HEVC encoder that makes a keyframe when told to, and never reorders.
private final class SegmentEncoder: @unchecked Sendable {
    private var session: VTCompressionSession?
    private let lock = NSLock()
    private var output: [CMSampleBuffer] = []
    private var failure: OSStatus = noErr

    typealias Color = (primaries: CFString, transfer: CFString, matrix: CFString)

    init(width: Int, height: Int, tenBit: Bool, color: Color, bitsPerSecond: Int, frameRate: Double, keyFrameSeconds: Double) throws {
        var made: VTCompressionSession?
        let status = VTCompressionSessionCreate(
            allocator: nil, width: Int32(width), height: Int32(height), codecType: kCMVideoCodecType_HEVC,
            encoderSpecification: [kVTVideoEncoderSpecification_EnableHardwareAcceleratedVideoEncoder: true] as CFDictionary,
            imageBufferAttributes: nil, compressedDataAllocator: nil, outputCallback: nil, refcon: nil, compressionSessionOut: &made)
        guard status == noErr, let made else { throw SpatialVideoError.writer("the encoder refused \(width)×\(height) HEVC (\(status))") }
        session = made
        func set(_ key: CFString, _ value: CFTypeRef) { VTSessionSetProperty(made, key: key, value: value) }
        set(kVTCompressionPropertyKey_ProfileLevel, tenBit ? kVTProfileLevel_HEVC_Main10_AutoLevel : kVTProfileLevel_HEVC_Main_AutoLevel)
        set(kVTCompressionPropertyKey_AverageBitRate, bitsPerSecond as CFNumber)
        set(kVTCompressionPropertyKey_RealTime, kCFBooleanFalse)
        set(kVTCompressionPropertyKey_AllowFrameReordering, kCFBooleanFalse)
        set(kVTCompressionPropertyKey_ExpectedFrameRate, (frameRate > 0 ? frameRate : 30) as CFNumber)
        // Keyframes are forced at every segment; this only bounds a long one.
        set(kVTCompressionPropertyKey_MaxKeyFrameIntervalDuration, keyFrameSeconds * 2 as CFNumber)
        set(kVTCompressionPropertyKey_ColorPrimaries, color.primaries)
        set(kVTCompressionPropertyKey_TransferFunction, color.transfer)
        set(kVTCompressionPropertyKey_YCbCrMatrix, color.matrix)
        VTCompressionSessionPrepareToEncodeFrames(made)
    }

    func encode(_ buffer: CVPixelBuffer, time: CMTime, duration: CMTime, keyFrame: Bool) throws {
        guard let session else { return }
        let options = keyFrame ? [kVTEncodeFrameOptionKey_ForceKeyFrame: true] as CFDictionary : nil
        let status = VTCompressionSessionEncodeFrame(session, imageBuffer: buffer, presentationTimeStamp: time, duration: duration,
                                                     frameProperties: options, infoFlagsOut: nil) { [self] status, _, sample in
            lock.lock()
            defer { lock.unlock() }
            if status != noErr { failure = status } else if let sample { output.append(sample) }
        }
        guard status == noErr else { throw SpatialVideoError.writer("encoding a frame failed (\(status))") }
    }

    /// Everything encoded so far, in order.
    func flush() throws -> [CMSampleBuffer] {
        guard let session else { return [] }
        VTCompressionSessionCompleteFrames(session, untilPresentationTimeStamp: .invalid)
        lock.lock()
        defer { lock.unlock() }
        guard failure == noErr else { throw SpatialVideoError.writer("encoding failed (\(failure))") }
        defer { output.removeAll() }
        return output
    }

    func invalidate() {
        if let session { VTCompressionSessionInvalidate(session) }
        session = nil
    }
}
#endif
