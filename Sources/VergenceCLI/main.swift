/*
 Vergence - command line

   vergence info DISC
   vergence decode DISC [--playlist NNNNN] [options]
   vergence decode BASE.m2ts [DEPENDENT.m2ts] [options]
   vergence stream DISC [--playlist NNNNN] [--seek SECONDS]
   vergence streams DISC [--playlist NNNNN]
   vergence subtitles DISC --pid PID [--playlist NNNNN] [--seek SECONDS] [--frames N] [--output DIR]
   vergence audio DISC --pid PID [--playlist NNNNN] [--seek SECONDS] [--frames SECONDS] --output FILE.wav

   options: --packing tab|sbs  --seek SECONDS  --start N  --frames N  --threads N
            --output FILE|-

 --seek starts at a time into the playlist (from the nearest entry point
 before it, then frame-exact); --start then skips N more frames.

 DISC is a folder holding BDMV (a rip or a mounted disc), a disc image (.iso)
 read in place without mounting, or an http(s) URL of a disc image (read with
 range requests; --header "Name: value" adds a header, as does
 VERGENCE_HTTP_HEADER, which keeps a token out of the process list). Without
 --playlist, decode picks the feature: the longest 3D playlist.

 decode writes packed I420 frames (rawvideo, yuv420p) for ffmpeg, e.g.

   vergence decode /Volumes/DISC --threads -1 | ffmpeg -f rawvideo -pix_fmt yuv420p \
     -s 1920x2160 -r 24000/1001 -i - -c:v hevc_videotoolbox out.mov

 stream writes a playlist's base clips (base view, audio, subtitles) as
 BDAV transport stream (.m2ts, 192-byte packets) from the entry point at or
 before --seek, for tools that read a stream but can't seek in a disc image:

   vergence stream disc.iso --seek 3600 | ffmpeg -i - -map 0:a:0 -c copy audio.mka

 Its timestamps are the clips' own; stderr says which playlist time the
 output starts at.

 streams lists the feature's audio and subtitle streams as the disc labels
 them, one per line: kind, PID, codec, language ("audio 0x1100 truehd eng").

 subtitles decodes a PGS subtitle stream (PID as `streams` lists it, e.g.
 0x1203) without decoding the video: one line per caption (playlist time,
 position and size in the frame), at most --frames captions, and with
 --output each caption as a PNG in that folder.

 audio decodes an AC-3 or E-AC-3 track (or a TrueHD track's AC-3 core) the
 way a player on the device does, in windows of a few seconds, to a 5.1
 WAV (L R C LFE Ls Rs) starting exactly at --seek, --frames seconds long.

 Progress and a summary go to stderr.
 */

import Foundation
import VergenceApple
import VergenceDisc
#if canImport(ImageIO)
import CoreGraphics
import ImageIO
#endif

struct Options {
    var command = ""
    var positional: [String] = []
    var playlist: String?
    var packing = StereoPacking.topAndBottom
    var seek: Double?
    var start = 0
    var frames = Int.max
    var threads = 0
    var output = "-"
    var pid: UInt16?
    var headers: [String: String] = [:]
}

func usage() -> Never {
    FileHandle.standardError.write(Data("""
    usage: vergence info DISC
           vergence decode DISC [--playlist NNNNN] [options]
           vergence decode BASE.m2ts [DEPENDENT.m2ts] [options]
           vergence stream DISC [--playlist NNNNN] [--seek SECONDS]
           vergence streams DISC [--playlist NNNNN]
           vergence subtitles DISC --pid PID [--playlist NNNNN] [--seek SECONDS] [--frames N] [--output DIR]
           vergence audio DISC --pid PID [--playlist NNNNN] [--seek SECONDS] [--frames SECONDS] --output FILE.wav
           vergence cat DISC PATH [--start BYTE] [--frames BYTES]
    options: --packing tab|sbs  --seek SECONDS  --start N  --frames N  --threads N  --output FILE|-
             --header "Name: value" (for an http(s) DISC)

    """.utf8))
    exit(64)
}

func parse(_ args: [String]) -> Options {
    var o = Options()
    guard let command = args.first else { usage() }
    o.command = command
    var i = 1
    func value() -> String {
        i += 1
        guard i < args.count else { usage() }
        return args[i]
    }
    while i < args.count {
        switch args[i] {
        case "--playlist": o.playlist = value()
        case "--packing": o.packing = StereoPacking(rawValue: value()) ?? { usage() }()
        case "--seek": o.seek = Double(value()) ?? { usage() }()
        case "--start": o.start = Int(value()) ?? 0
        case "--frames": o.frames = Int(value()) ?? Int.max
        case "--threads": o.threads = Int(value()) ?? 0
        case "--output", "-o": o.output = value()
        case "--pid": o.pid = UInt16(value().replacingOccurrences(of: "0x", with: ""), radix: 16) ?? { usage() }()
        case "--header", "-H": addHeader(value(), to: &o)
        case let a where a.hasPrefix("-") && a != "-": usage()
        default: o.positional.append(args[i])
        }
        i += 1
    }
    if let header = ProcessInfo.processInfo.environment["VERGENCE_HTTP_HEADER"] { addHeader(header, to: &o) }
    return o
}

func addHeader(_ line: String, to o: inout Options) {
    guard let colon = line.firstIndex(of: ":") else { usage() }
    o.headers[String(line[..<colon]).trimmingCharacters(in: .whitespaces)] =
        String(line[line.index(after: colon)...]).trimmingCharacters(in: .whitespaces)
}

func log(_ s: String) { FileHandle.standardError.write(Data((s + "\n").utf8)) }

func clockTime(_ seconds: Double) -> String {
    let s = Int(seconds.rounded())
    return String(format: "%d:%02d:%02d", s / 3600, s / 60 % 60, s % 60)
}

/// A folder holding BDMV, or a disc image read in place, local or remote.
func openDisc(_ path: String) throws -> BlurayDisc? {
    if let url = URL(string: path), ["http", "https"].contains(url.scheme?.lowercased()) {
        return BlurayDisc(fileSystem: try UDFFileSystem(image: HTTPByteSource(url: url, headers: options.headers)))
    }
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory) else { return nil }
    if isDirectory.boolValue {
        return BlurayDisc(fileSystem: DirectoryFileSystem(root: URL(fileURLWithPath: path)))
    }
    guard path.lowercased().hasSuffix(".iso") else { return nil }
    return BlurayDisc(fileSystem: try UDFFileSystem(image: FileByteSource(url: URL(fileURLWithPath: path))))
}

func info(_ o: Options) throws {
    guard o.positional.count == 1, let disc = try openDisc(o.positional[0]) else { usage() }
    let main = try disc.mainPlaylist()?.name
    for (name, playlist) in try disc.playlists() where playlist.duration >= 60 {
        let clips = playlist.items.map { $0.clip + ($0.dependentClip.map { "+" + $0 } ?? "") }
        let shown = clips.count > 4 ? clips.prefix(4).joined(separator: " ") + " … (\(clips.count))" : clips.joined(separator: " ")
        print("\(name == main ? "*" : " ") \(name)  \(clockTime(playlist.duration))  \(playlist.isStereo ? "3D" : "2D")  \(shown)")
    }
}

/// Where frames come from: a disc playlist or a clip pair.
struct FrameSource {
    /// Decodes, calling the closure per frame until it returns false.
    var run: (_ output: (StereoFrame) throws -> Bool) throws -> Void
    var summary: () -> String
}

func frameSource(_ o: Options) throws -> FrameSource {
    if o.positional.count == 1, let disc = try openDisc(o.positional[0]) {
        let (name, playlist) = try chosenPlaylist(disc, o)
        log("vergence: playlist \(name), \(clockTime(playlist.duration)), \(playlist.isStereo ? "3D" : "2D only"), clips \(playlist.items.map(\.clip).joined(separator: " "))")
        let decoder = try PlaylistDecoder(disc: disc, playlist: playlist, threads: o.threads)
        if let seek = o.seek { try decoder.seek(to: seek) }
        return FrameSource(run: { try decoder.decode(output: $0) }, summary: {
            "\(decoder.unpairedBaseCount) unpaired AUs, \(decoder.resyncCount) damaged spots skipped, \(decoder.corruptNALCount) corrupt NALs, \(decoder.unsupportedNALCount) unsupported NALs"
        })
    }
    guard (1 ... 2).contains(o.positional.count), o.seek == nil else { usage() }
    let reader = StereoStreamReader(
        base: try FileByteSource(url: URL(fileURLWithPath: o.positional[0])),
        dependent: try o.positional.dropFirst().first.map { try FileByteSource(url: URL(fileURLWithPath: $0)) })
    let decoder = try MVCDecoder(threads: o.threads)
    return FrameSource(run: { output in
        struct Stop: Error {}
        let emit: (StereoFrame) throws -> Void = { if try !output($0) { throw Stop() } }
        do {
            while let unit = try reader.next() { try decoder.decode(unit, output: emit) }
            try decoder.finish(output: emit)
        } catch is Stop {}
    }, summary: {
        "\(reader.unpairedBaseCount) unpaired AUs, \(reader.resyncCount) damaged spots skipped, \(decoder.corruptNALCount) corrupt NALs, \(decoder.unsupportedNALCount) unsupported NALs"
    })
}

func decode(_ o: Options) throws {
    let source = try frameSource(o)
    let out: FileHandle
    if o.output == "-" {
        out = .standardOutput
    } else {
        FileManager.default.createFile(atPath: o.output, contents: nil)
        out = try FileHandle(forWritingTo: URL(fileURLWithPath: o.output))
    }

    let clock = ContinuousClock()
    let began = clock.now
    var index = 0, written = 0, mono = 0
    var buffer = Data()
    var announced = false
    // Timestamps should step by one frame; a jump means frames went missing
    // or came out of order.
    var previousPTS: Int64?, frameTicks: Int64?, irregular = 0
    let last = o.frames == Int.max ? Int.max : o.start + o.frames

    try source.run { frame in
        defer { index += 1 }
        guard index >= o.start else { return true }
        guard index < last else { return false }
        if frame.right == nil { mono += 1 }
        if let pts = frame.pts {
            if let previousPTS {
                let step = pts - previousPTS
                if frameTicks == nil { frameTicks = step }
                if let frameTicks, abs(step - frameTicks) > 2 { irregular += 1 }
            }
            previousPTS = pts
        }
        if !announced {
            let (w, h) = o.packing.packedSize(width: frame.width, height: frame.height)
            log("vergence: \(frame.width)x\(frame.height) per eye -> \(w)x\(h) \(o.packing.rawValue), yuv420p, first frame at \(String(format: "%.3f", Double(frame.pts ?? 0) / 90_000)) s")
            announced = true
        }
        let count = frame.packedByteCount(o.packing)
        if buffer.count != count { buffer = Data(count: count) }
        buffer.withUnsafeMutableBytes { frame.pack(o.packing, into: $0) }
        try out.write(contentsOf: buffer)
        written += 1
        if written % 240 == 0 {
            let seconds = (clock.now - began) / .seconds(1)
            log(String(format: "vergence: %d frames (%@), %.1f fps", written, clockTime(Double(frame.pts ?? 0) / 90_000), Double(written) / seconds))
        }
        return written < last - o.start
    }

    let seconds = (clock.now - began) / .seconds(1)
    log(String(format: "vergence: wrote %d frames in %.1f s (%.1f fps); %d without a second view, %d irregular timestamp steps, %@",
               written, seconds, Double(written) / seconds, mono, irregular, source.summary()))
}

/// The playlist named by --playlist, else the feature.
func chosenPlaylist(_ disc: BlurayDisc, _ o: Options) throws -> (name: String, playlist: BlurayPlaylist) {
    let chosen = try o.playlist.map { wanted in try disc.playlists().first { $0.name == wanted } } ?? disc.mainPlaylist()
    guard let chosen else { throw FormatError("no such playlist") }
    return chosen
}

/// vergence stream DISC: the playlist's base clips as one .m2ts byte stream.
func stream(_ o: Options) throws {
    guard o.positional.count == 1, let disc = try openDisc(o.positional[0]) else { usage() }
    let (name, playlist) = try chosenPlaylist(disc, o)
    let spans = try disc.clipSpans(of: playlist, from: o.seek ?? 0)
    if let first = spans.first {
        log(String(format: "vergence: playlist %@, streaming from %.3f s (clip %@, byte %lld)", name, Double(first.playlistTime) / 90_000, first.clip, first.start))
    }
    for (i, span) in spans.enumerated() {
        let item = playlist.items[playlist.items.count - spans.count + i]
        log("vergence: clip \(span.clip) bytes \(span.start)-\(span.end), IN at PTS \(item.inPTS), playlist time \(span.playlistTime)")
        let file = try disc.clip(span.clip)
        var offset = span.start
        while offset < span.end {
            let chunk = try file.read(at: offset, count: Int(min(4 << 20, span.end - offset)))
            if chunk.isEmpty { break }
            try FileHandle.standardOutput.write(contentsOf: chunk)
            offset += Int64(chunk.count)
        }
    }
}

/// vergence streams DISC: the first PlayItem's audio and subtitle streams.
func streams(_ o: Options) throws {
    guard o.positional.count == 1, let disc = try openDisc(o.positional[0]) else { usage() }
    let (_, playlist) = try chosenPlaylist(disc, o)
    guard let item = playlist.items.first else { return }
    for (kind, list) in [("audio", item.audioStreams), ("subtitle", item.subtitleStreams)] {
        for stream in list {
            print(String(format: "%@ 0x%04x %@ %@", kind, stream.pid, stream.codecName, stream.language ?? "und"))
        }
    }
}

/// vergence subtitles DISC --pid PID: a PGS stream's captions, read straight
/// from the base clips.
func subtitles(_ o: Options) throws {
    guard o.positional.count == 1, let pid = o.pid, let disc = try openDisc(o.positional[0]) else { usage() }
    let (_, playlist) = try chosenPlaylist(disc, o)
    let spans = try disc.clipSpans(of: playlist, from: o.seek ?? 0)
    let directory = o.output == "-" ? nil : URL(fileURLWithPath: o.output)
    if let directory { try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true) }
    var itemStart: Int64 = 0
    var starts: [String: (start: Int64, inPTS: Int64)] = [:]
    for item in playlist.items {
        starts[item.clip] = starts[item.clip] ?? (itemStart, item.inPTS)
        itemStart += item.outPTS - item.inPTS
    }
    var count = 0
    var decoder = PGSDecoder()
    for span in spans {
        guard let item = starts[span.clip] else { continue }
        var demuxer = TransportStreamDemuxer(pids: [pid])
        let file = try disc.clip(span.clip)
        var offset = span.start
        func handle(_ packets: [PESPacket]) throws -> Bool {
            for packet in packets {
                for caption in decoder.feed(pes: packet.payload, pts: packet.pts) {
                    let time = Double(item.start + PlaylistStereoReader.offset(caption.pts, from: item.inPTS)) / 90_000
                    let place = caption.bounds.map { "\($0.width)x\($0.height) at \($0.x),\($0.y)" } ?? "clear"
                    print(String(format: "%@ %.3f %@ in %dx%d%@", clockTime(time), time, place, caption.videoWidth,
                                 caption.videoHeight, caption.isRefresh ? " (refresh)" : ""))
                    if let directory, let image = caption.renderRGBA() {
                        try writePNG(image.pixels, width: image.width, height: image.height,
                                     to: directory.appendingPathComponent(String(format: "%06d-%.3f.png", count, time)))
                    }
                    count += 1
                    if count >= o.frames { return false }
                }
            }
            return true
        }
        while offset < span.end {
            let chunk = try file.read(at: offset, count: Int(min(4 << 20, span.end - offset)))
            if chunk.isEmpty { break }
            offset += Int64(chunk.count)
            guard try handle(demuxer.feed(chunk)) else { return }
        }
        guard try handle(demuxer.flush()) else { return }
    }
    log("vergence: \(count) captions")
}

func writePNG(_ pixels: [UInt32], width: Int, height: Int, to url: URL) throws {
    #if canImport(ImageIO)
    var pixels = pixels
    let image = pixels.withUnsafeMutableBytes { raw -> CGImage? in
        guard let context = CGContext(data: raw.baseAddress, width: width, height: height, bitsPerComponent: 8,
                                      bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        return context.makeImage()
    }
    guard let image, let destination = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil)
    else { throw FormatError("can't write \(url.path)") }
    CGImageDestinationAddImage(destination, image, nil)
    guard CGImageDestinationFinalize(destination) else { throw FormatError("can't write \(url.path)") }
    #else
    throw FormatError("writing PNGs needs ImageIO (Apple platforms only)")
    #endif
}

/// vergence audio DISC --pid PID: decoded 5.1 PCM to a WAV file.
func audio(_ o: Options) throws {
    #if !canImport(AVFoundation)
    throw FormatError("decoding AC-3 needs Apple's decoder (Apple platforms only)")
    #else
    guard o.positional.count == 1, let pid = o.pid, o.output != "-", let disc = try openDisc(o.positional[0]) else { usage() }
    let (_, playlist) = try chosenPlaylist(disc, o)
    let reader = PlaylistAudioReader(disc: disc, playlist: playlist, pid: pid)
    let rate = PlaylistAudioReader.sampleRate
    let total = (o.frames == Int.max ? 30 : o.frames) * rate
    let window = 4 * rate
    var start = Int64(((o.seek ?? 0) * 90_000).rounded())
    var pcm = Data()
    let began = Date()
    var written = 0
    while written < total {
        let count = min(window, total - written)
        let planes = try reader.read(fromPTS: start, frames: count)
        for i in 0 ..< count {
            for channel in 0 ..< planes.count {
                let v = Int16(max(-1, min(1, planes[channel][i])) * 32767)
                withUnsafeBytes(of: v.littleEndian) { pcm.append(contentsOf: $0) }
            }
        }
        written += count
        start += Int64(count) * 90_000 / Int64(rate)
    }
    var header = Data()
    func u32(_ v: Int) { withUnsafeBytes(of: UInt32(v).littleEndian) { header.append(contentsOf: $0) } }
    func u16(_ v: Int) { withUnsafeBytes(of: UInt16(v).littleEndian) { header.append(contentsOf: $0) } }
    header.append(contentsOf: Array("RIFF".utf8)); u32(36 + pcm.count)
    header.append(contentsOf: Array("WAVEfmt ".utf8)); u32(16); u16(1); u16(6); u32(rate); u32(rate * 12); u16(12); u16(16)
    header.append(contentsOf: Array("data".utf8)); u32(pcm.count)
    try (header + pcm).write(to: URL(fileURLWithPath: o.output))
    log(String(format: "vergence: %.1f s of audio in %.1f s", Double(total) / Double(rate), Date().timeIntervalSince(began)))
#endif
}

/// vergence cat DISC PATH [--start BYTE] [--frames BYTES]: a file's bytes out
/// of a disc (mainly for checking the image reader against a mounted copy).
func cat(_ o: Options) throws {
    guard o.positional.count == 2, let disc = try openDisc(o.positional[0]) else { usage() }
    let file = try disc.fileSystem.open(o.positional[1])
    var offset = Int64(o.start)
    let end = o.frames == Int.max ? file.size : min(file.size, offset + Int64(o.frames))
    while offset < end {
        let chunk = try file.read(at: offset, count: Int(min(1 << 20, end - offset)))
        if chunk.isEmpty { break }
        FileHandle.standardOutput.write(chunk)
        offset += Int64(chunk.count)
    }
}

let options = parse(Array(CommandLine.arguments.dropFirst()))
do {
    switch options.command {
    case "info": try info(options)
    case "decode": try decode(options)
    case "stream": try stream(options)
    case "streams": try streams(options)
    case "subtitles": try subtitles(options)
    case "audio": try audio(options)
    case "cat": try cat(options)
    default: usage()
    }
} catch {
    log("vergence: \(error)")
    exit(1)
}
