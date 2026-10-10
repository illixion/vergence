#if os(macOS)
import AVFoundation
import CoreMedia
import CoreVideo
import Testing
@testable import VergenceApple
@testable import VergenceCore

/// Stream segments beyond the picture: audio across segment and run joins,
/// HDR, and the smaller picture sizes. Sources are synthetic spatial videos
/// made from `SpatialVideoMakerTests.writeSideBySideClip`.
@Suite struct SideBySideSegmenterTests {
    typealias Clip = SpatialVideoMakerTests

    /// A temporary directory holding a spatial video made from a synthetic
    /// side-by-side clip.
    static func spatialVideo(frames: Int, audio: Bool = false, hdr: Bool = false) async throws -> (directory: URL, spatial: URL) {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("vergence-sbs-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let clip = directory.appendingPathComponent(hdr ? "clip.mov" : "clip.mp4")
        let spatial = directory.appendingPathComponent("clip.spatial.mov")
        try await Clip.writeSideBySideClip(to: clip, frames: frames, audio: audio, hdr: hdr)
        let made = try await SpatialVideoMaker.make(SpatialVideoJob(source: try await SpatialVideoSource.open(clip), output: spatial,
                                                                     layout: PackedStereoLayout(arrangement: .sideBySide, isHalfResolution: false),
                                                                     camera: SpatialPreset.generic.metadata))
        #expect(made.audioTracksCopied == (audio ? 1 : 0))
        return (directory, spatial)
    }

    /// The init segment and segments `0..<count`, joined into one file.
    static func join(_ segments: URL, count: Int, into file: URL) throws {
        var joined = try Data(contentsOf: segments.appendingPathComponent(SideBySideSegments.initName))
        for n in 0 ..< count { joined += try Data(contentsOf: segments.appendingPathComponent(SideBySideSegments.segmentName(n))) }
        try joined.write(to: file)
    }

    /// Every audio packet of a file's first audio track, as the reader passes them through.
    static func audioPackets(_ url: URL) async throws -> [Data] {
        let asset = AVURLAsset(url: url)
        let track = try #require(try await asset.loadTracks(withMediaType: .audio).first)
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
        reader.add(output)
        #expect(reader.startReading())
        var packets: [Data] = []
        while let sample = output.copyNextSampleBuffer() {
            let count = CMSampleBufferGetNumSamples(sample)
            guard count > 0, let block = CMSampleBufferGetDataBuffer(sample) else { continue }
            var bytes = Data(count: CMBlockBufferGetDataLength(block))
            bytes.withUnsafeMutableBytes { _ = CMBlockBufferCopyDataBytes(block, atOffset: 0, dataLength: $0.count, destination: $0.baseAddress!) }
            var at = 0
            for i in 0 ..< count {
                let size = CMSampleBufferGetSampleSize(sample, at: i)
                packets.append(bytes.subdata(in: at ..< at + size))
                at += size
            }
            #expect(at == bytes.count)
        }
        return packets
    }

    /// One track's fragments in a segment: base decode time, sample count
    /// and the samples' durations, read from tfdt, tfhd and trun.
    static func fragments(_ data: Data, track: UInt32) -> [(decodeTime: UInt64, durations: [UInt32])] {
        typealias F = FragmentTimes
        var result: [(UInt64, [UInt32])] = []
        for (type, moof) in F.boxes(data, data.startIndex ..< data.endIndex) where type == "moof" {
            for (type, traf) in F.boxes(data, moof) where type == "traf" {
                var id: UInt32?, time: UInt64?, defaultDuration: UInt32 = 0, durations: [UInt32] = []
                for (type, box) in F.boxes(data, traf) {
                    let flags = F.u32(data, box.lowerBound) & 0xFFFFFF
                    switch type {
                    case "tfhd":
                        id = F.u32(data, box.lowerBound + 4)
                        var at = box.lowerBound + 8
                        if flags & 0x1 != 0 { at += 8 }
                        if flags & 0x2 != 0 { at += 4 }
                        if flags & 0x8 != 0 { defaultDuration = F.u32(data, at) }
                    case "tfdt":
                        time = data[box.lowerBound] == 1 ? F.u64(data, box.lowerBound + 4) : UInt64(F.u32(data, box.lowerBound + 4))
                    case "trun":
                        let count = Int(F.u32(data, box.lowerBound + 4))
                        var at = box.lowerBound + 8
                        if flags & 0x1 != 0 { at += 4 }
                        if flags & 0x4 != 0 { at += 4 }
                        let fields = [0x100, 0x200, 0x400, 0x800].filter { flags & UInt32($0) != 0 }.count
                        for i in 0 ..< count {
                            durations.append(flags & 0x100 != 0 ? F.u32(data, at + i * fields * 4) : defaultDuration)
                        }
                    default: break
                    }
                }
                if id == track, let time { result.append((time, durations)) }
            }
        }
        return result
    }

    /// Two runs, joined out of order like a server seeking: every AAC packet
    /// once and in order, packets exactly 1024 samples apart in the
    /// segments' own times across segment and run joins, and the audio
    /// placed against the video as the source has it, priming included (an
    /// AAC encoder's first packet plays before the first frame, which is
    /// what `timeOffset` makes room for).
    @Test func audioComesThroughEveryPacketOnceContiguousAndInPlace() async throws {
        let (directory, spatial) = try await Self.spatialVideo(frames: 48, audio: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let job = SideBySideJob(source: try await SpatialVideoSource.open(spatial), output: spatial)
        let segments = directory.appendingPathComponent("segments")
        let late = try await SideBySideSegmenter.run(SideBySideSegments(job: job, directory: segments, segmentSeconds: 0.5, first: 2))
        let early = try await SideBySideSegmenter.run(SideBySideSegments(job: job, directory: segments, segmentSeconds: 0.5, first: 0, last: 1))
        #expect(late.audio && early.audio)
        let file = directory.appendingPathComponent("joined.mp4")
        try Self.join(segments, count: 4, into: file)

        let source = try await Self.audioPackets(spatial)
        // 2 s at 48 kHz plus the encoder's priming, 1024 samples a packet.
        #expect(source.count == 96)
        #expect(try await Self.audioPackets(file) == source)

        // Where the source's first packet plays against its first frame (at
        // 0): the reader gives the buffer's output time, where the sample
        // after the priming trim plays, so the packet starts a trim earlier.
        let asset = AVURLAsset(url: spatial)
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: try #require(try await asset.loadTracks(withMediaType: .audio).first), outputSettings: nil)
        reader.add(output)
        #expect(reader.startReading())
        var first = try #require(output.copyNextSampleBuffer())
        while CMSampleBufferGetNumSamples(first) == 0 { first = try #require(output.copyNextSampleBuffer()) }
        reader.cancelReading()
        let trim = CMTimeMakeFromDictionary(try #require(CMGetAttachment(first, key: kCMSampleBufferAttachmentKey_TrimDurationAtStart,
                                                                          attachmentModeOut: nil) as! CFDictionary?))
        #expect(trim.seconds > 0.02, "an AAC encoder primes, so the test means something")
        let firstPacketPlays = CMTimeSubtract(CMSampleBufferGetOutputPresentationTimeStamp(first), trim)

        let scales = FragmentTimes.timescales(initSegment: try Data(contentsOf: segments.appendingPathComponent(SideBySideSegments.initName)))
        let videoScale = Int64(try #require(scales[1])), audioScale = Int64(try #require(scales[2]))
        #expect(audioScale == 48_000)
        let offset = Int64(SideBySideSegments.timeOffset.seconds * 48_000)
        let firstPacket = firstPacketPlays.convertScale(48_000, method: .roundHalfAwayFromZero).value
        var expected = firstPacket + offset
        var packets = 0
        for n in 0 ..< 4 {
            let data = try Data(contentsOf: segments.appendingPathComponent(SideBySideSegments.segmentName(n)))
            let audio = Self.fragments(data, track: 2)
            #expect(audio.count == 1, "segment \(n): one audio fragment")
            let fragment = try #require(audio.first)
            #expect(fragment.durations.allSatisfy { $0 == 1024 }, "segment \(n): \(fragment.durations)")
            // Contiguous: each segment's audio starts where the last one's ended.
            #expect(Int64(fragment.decodeTime) == expected, "segment \(n) audio at \(fragment.decodeTime), not \(expected)")
            #expect(FragmentTimes.firstPresentation(data)[2] == Int64(fragment.decodeTime))
            // And against the frames: packet `packets` of the source plays
            // `packets` × 1024 after the first, the segment's first frame at its grid time.
            let video = try #require(FragmentTimes.firstPresentation(data)[1])
            #expect(video * audioScale / videoScale == Int64(n) * 24_000 + offset)
            #expect(Int64(fragment.decodeTime) - video * audioScale / videoScale
                    == firstPacket + Int64(packets * 1024) - Int64(n) * 24_000)
            packets += fragment.durations.count
            expected += Int64(fragment.durations.count * 1024)
        }
        #expect(packets == source.count)
    }

    /// Splitting a buffer of packets at a segment's end: each side gets its
    /// own packets at their own times, and only the first part keeps the
    /// priming trim (a later part trimmed too would drop real samples).
    @Test func splitsAudioBuffersByPacketTime() async throws {
        let (directory, spatial) = try await Self.spatialVideo(frames: 12, audio: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let asset = AVURLAsset(url: spatial)
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: try #require(try await asset.loadTracks(withMediaType: .audio).first), outputSettings: nil)
        reader.add(output)
        #expect(reader.startReading())
        var sample = try #require(output.copyNextSampleBuffer())
        while CMSampleBufferGetNumSamples(sample) == 0 { sample = try #require(output.copyNextSampleBuffer()) }
        reader.cancelReading()
        let count = CMSampleBufferGetNumSamples(sample)
        #expect(count > 20)
        func trimmed(_ buffer: CMSampleBuffer) -> Bool {
            CMGetAttachment(buffer, key: kCMSampleBufferAttachmentKey_TrimDurationAtStart, attachmentModeOut: nil) != nil
        }
        #expect(trimmed(sample))
        let start = CMSampleBufferGetPresentationTimeStamp(sample)
        func packetTime(_ k: Int) -> CMTime { CMTimeAdd(start, CMTime(value: CMTimeValue(k * 1024), timescale: 48_000)) }

        // At a time between packets 9 and 10.
        let (before, after) = try SideBySideSegmenter.split(sample, from: start, at: CMTimeAdd(packetTime(9), CMTime(value: 1, timescale: 48_000)))
        let head = try #require(before), tail = try #require(after)
        #expect(CMSampleBufferGetNumSamples(head) == 10 && CMSampleBufferGetNumSamples(tail) == count - 10)
        #expect(CMSampleBufferGetPresentationTimeStamp(head) == start)
        #expect(CMSampleBufferGetPresentationTimeStamp(tail) == packetTime(10))
        #expect(trimmed(head) && !trimmed(tail))

        // Exactly on a packet: that packet goes after. Packets before `from`
        // (a run starting later) are dropped.
        let (on, rest) = try SideBySideSegmenter.split(sample, from: packetTime(4), at: packetTime(10))
        #expect(CMSampleBufferGetNumSamples(try #require(on)) == 6)
        #expect(CMSampleBufferGetPresentationTimeStamp(try #require(on)) == packetTime(4))
        #expect(!trimmed(try #require(on)))
        #expect(CMSampleBufferGetPresentationTimeStamp(try #require(rest)) == packetTime(10))

        // All before the end: the buffer itself; all after: nothing before.
        let (whole, none) = try SideBySideSegmenter.split(sample, from: start, at: packetTime(count))
        #expect(whole === sample && none == nil)
        let (nothing, later) = try SideBySideSegmenter.split(sample, from: start, at: start)
        #expect(nothing == nil && later === sample)
    }

    /// A 10-bit PQ source streams as 10-bit PQ: Main 10 HEVC whose init
    /// segment says so, not quietly re-tagged BT.709.
    @Test func hdrSegmentsStayTenBitPQ() async throws {
        let (directory, spatial) = try await Self.spatialVideo(frames: 24, hdr: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let source = try await SpatialVideoSource.open(spatial)
        #expect(source.isHDR && source.needsTenBit)
        let segments = directory.appendingPathComponent("segments")
        let result = try await SideBySideSegmenter.run(SideBySideSegments(job: SideBySideJob(source: source, output: spatial),
                                                                          directory: segments, segmentSeconds: 0.5))
        #expect(result.segments == [0, 1] && result.frames == 24)
        let file = directory.appendingPathComponent("joined.mp4")
        try Self.join(segments, count: 2, into: file)

        let track = try #require(try await AVURLAsset(url: file).loadTracks(withMediaType: .video).first)
        let format = try #require(try await track.load(.formatDescriptions).first)
        #expect(CMFormatDescriptionGetMediaSubType(format) == kCMVideoCodecType_HEVC)
        let extensions = CMFormatDescriptionGetExtensions(format) as? [CFString: Any] ?? [:]
        #expect(extensions[kCMFormatDescriptionExtension_TransferFunction] as? String == kCMFormatDescriptionTransferFunction_SMPTE_ST_2084_PQ as String)
        #expect(extensions[kCMFormatDescriptionExtension_ColorPrimaries] as? String == kCMFormatDescriptionColorPrimaries_ITU_R_2020 as String)
        // hvcC: general_profile_idc in byte 1 (2 is Main 10), bit depths − 8 in bytes 17 and 18.
        let atoms = try #require(extensions[kCMFormatDescriptionExtension_SampleDescriptionExtensionAtoms] as? [String: Any])
        let hvcC = [UInt8](try #require(atoms["hvcC"] as? Data))
        #expect(hvcC[1] & 0x1F == 2, "profile \(hvcC[1] & 0x1F)")
        #expect(hvcC[17] & 0x07 == 2 && hvcC[18] & 0x07 == 2, "luma/chroma bit depth − 8")
        #expect(try await SpatialVideoSource.open(file).isHDR)
    }

    /// `halfWidth` and `maxEyeSide` shrink the streamed picture as they do
    /// the exported file: what the init segment declares and what decodes.
    @Test func halfWidthAndMaxEyeSideSetThePictureSize() async throws {
        let (directory, spatial) = try await Self.spatialVideo(frames: 12)
        defer { try? FileManager.default.removeItem(at: directory) }
        let source = try await SpatialVideoSource.open(spatial)
        // A 256×144 eye: squeezed to 128×144, or capped to 128×72.
        for (name, job, size) in [
            ("half", SideBySideJob(source: source, output: spatial, halfWidth: true), CGSize(width: 256, height: 144)),
            ("capped", SideBySideJob(source: source, output: spatial, maxEyeSide: 128), CGSize(width: 256, height: 72)),
            ("both", SideBySideJob(source: source, output: spatial, halfWidth: true, maxEyeSide: 128), CGSize(width: 128, height: 72)),
        ] {
            let segments = directory.appendingPathComponent(name)
            let result = try await SideBySideSegmenter.run(SideBySideSegments(job: job, directory: segments, segmentSeconds: 0.5))
            #expect(result.segments == [0] && result.frames == 12)
            let file = directory.appendingPathComponent("\(name).mp4")
            try Self.join(segments, count: 1, into: file)
            #expect(try await SpatialVideoSource.open(file).frameSize == size, "\(name)")

            let asset = AVURLAsset(url: file)
            let reader = try AVAssetReader(asset: asset)
            let output = AVAssetReaderTrackOutput(track: try #require(try await asset.loadTracks(withMediaType: .video).first), outputSettings: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
            ])
            reader.add(output)
            #expect(reader.startReading())
            let picture = try #require(output.copyNextSampleBuffer().flatMap(CMSampleBufferGetImageBuffer))
            #expect(CVPixelBufferGetWidth(picture) == Int(size.width) && CVPixelBufferGetHeight(picture) == Int(size.height), "\(name)")
            // Still the left eye on the left: the marker band, bright then dark.
            CVPixelBufferLockBaseAddress(picture, .readOnly)
            let base = try #require(CVPixelBufferGetBaseAddressOfPlane(picture, 0)).assumingMemoryBound(to: UInt8.self)
            let stride = CVPixelBufferGetBytesPerRowOfPlane(picture, 0), half = Int(size.width) / 2
            func mean(_ x0: Int) -> Double {
                var sum = 0.0
                for y in 1 ..< 5 { for x in x0 + 4 ..< x0 + half - 4 { sum += Double(base[y * stride + x]) } }
                return sum / Double(4 * (half - 8))
            }
            let left = mean(0), right = mean(half)
            CVPixelBufferUnlockBaseAddress(picture, .readOnly)
            reader.cancelReading()
            #expect(left > 200 && right < 50, "\(name): left \(left), right \(right)")
        }
    }
}
#endif
