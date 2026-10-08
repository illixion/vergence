/*
 mvc3d - command line

   mvc3d info DISC
   mvc3d decode DISC [--playlist NNNNN] [options]
   mvc3d decode BASE.m2ts [DEPENDENT.m2ts] [options]
   mvc3d stream DISC [--playlist NNNNN] [--seek SECONDS]

   options: --packing tab|sbs  --seek SECONDS  --start N  --frames N  --threads N
            --output FILE|-

 --seek starts at a time into the playlist (from the nearest entry point
 before it, then frame-exact); --start then skips N more frames.

 DISC is a folder holding BDMV (a rip or a mounted disc), a disc image (.iso)
 read in place without mounting, or an http(s) URL of a disc image (read with
 range requests; --header "Name: value" adds a header, as does
 MVC3D_HTTP_HEADER, which keeps a token out of the process list). Without
 --playlist, decode picks the feature: the longest 3D playlist.

 decode writes packed I420 frames (rawvideo, yuv420p) for ffmpeg, e.g.

   mvc3d decode /Volumes/DISC --threads -1 | ffmpeg -f rawvideo -pix_fmt yuv420p \
     -s 1920x2160 -r 24000/1001 -i - -c:v hevc_videotoolbox out.mov

 stream writes a playlist's base clips (base view, audio, subtitles) as
 BDAV transport stream (.m2ts, 192-byte packets) from the entry point at or
 before --seek, for tools that read a stream but can't seek in a disc image:

   mvc3d stream disc.iso --seek 3600 | ffmpeg -i - -map 0:a:0 -c copy audio.mka

 Its timestamps are the clips' own; stderr says which playlist time the
 output starts at.

 Progress and a summary go to stderr.
 */

import Foundation
import MVC3D

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
    var headers: [String: String] = [:]
}

func usage() -> Never {
    FileHandle.standardError.write(Data("""
    usage: mvc3d info DISC
           mvc3d decode DISC [--playlist NNNNN] [options]
           mvc3d decode BASE.m2ts [DEPENDENT.m2ts] [options]
           mvc3d stream DISC [--playlist NNNNN] [--seek SECONDS]
           mvc3d cat DISC PATH [--start BYTE] [--frames BYTES]
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
        case "--header", "-H": addHeader(value(), to: &o)
        case let a where a.hasPrefix("-") && a != "-": usage()
        default: o.positional.append(args[i])
        }
        i += 1
    }
    if let header = ProcessInfo.processInfo.environment["MVC3D_HTTP_HEADER"] { addHeader(header, to: &o) }
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
        log("mvc3d: playlist \(name), \(clockTime(playlist.duration)), \(playlist.isStereo ? "3D" : "2D only"), clips \(playlist.items.map(\.clip).joined(separator: " "))")
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
            log("mvc3d: \(frame.width)x\(frame.height) per eye -> \(w)x\(h) \(o.packing.rawValue), yuv420p, first frame at \(String(format: "%.3f", Double(frame.pts ?? 0) / 90_000)) s")
            announced = true
        }
        let count = frame.packedByteCount(o.packing)
        if buffer.count != count { buffer = Data(count: count) }
        buffer.withUnsafeMutableBytes { frame.pack(o.packing, into: $0) }
        try out.write(contentsOf: buffer)
        written += 1
        if written % 240 == 0 {
            let seconds = (clock.now - began) / .seconds(1)
            log(String(format: "mvc3d: %d frames (%@), %.1f fps", written, clockTime(Double(frame.pts ?? 0) / 90_000), Double(written) / seconds))
        }
        return written < last - o.start
    }

    let seconds = (clock.now - began) / .seconds(1)
    log(String(format: "mvc3d: wrote %d frames in %.1f s (%.1f fps); %d without a second view, %d irregular timestamp steps, %@",
               written, seconds, Double(written) / seconds, mono, irregular, source.summary()))
}

/// The playlist named by --playlist, else the feature.
func chosenPlaylist(_ disc: BlurayDisc, _ o: Options) throws -> (name: String, playlist: BlurayPlaylist) {
    let chosen = try o.playlist.map { wanted in try disc.playlists().first { $0.name == wanted } } ?? disc.mainPlaylist()
    guard let chosen else { throw FormatError("no such playlist") }
    return chosen
}

/// mvc3d stream DISC: the playlist's base clips as one .m2ts byte stream.
func stream(_ o: Options) throws {
    guard o.positional.count == 1, let disc = try openDisc(o.positional[0]) else { usage() }
    let (name, playlist) = try chosenPlaylist(disc, o)
    let spans = try disc.clipSpans(of: playlist, from: o.seek ?? 0)
    if let first = spans.first {
        log(String(format: "mvc3d: playlist %@, streaming from %.3f s (clip %@, byte %lld)", name, Double(first.playlistTime) / 90_000, first.clip, first.start))
    }
    for (i, span) in spans.enumerated() {
        let item = playlist.items[playlist.items.count - spans.count + i]
        log("mvc3d: clip \(span.clip) bytes \(span.start)-\(span.end), IN at PTS \(item.inPTS), playlist time \(span.playlistTime)")
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

/// mvc3d cat DISC PATH [--start BYTE] [--frames BYTES]: a file's bytes out
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
    case "cat": try cat(options)
    default: usage()
    }
} catch {
    log("mvc3d: \(error)")
    exit(1)
}
