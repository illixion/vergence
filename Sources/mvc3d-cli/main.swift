/*
 mvc3d - command line

   mvc3d info DISC
   mvc3d decode DISC [--playlist NNNNN] [options]
   mvc3d decode BASE.m2ts [DEPENDENT.m2ts] [options]

   options: --packing tab|sbs  --start N  --frames N  --threads N  --output FILE|-

 DISC is a folder holding BDMV (a rip or a mounted disc) or a disc image
 (.iso), read in place without mounting. Without
 --playlist, decode picks the feature: the longest 3D playlist.

 decode writes packed I420 frames (rawvideo, yuv420p) for ffmpeg, e.g.

   mvc3d decode /Volumes/DISC --threads -1 | ffmpeg -f rawvideo -pix_fmt yuv420p \
     -s 1920x2160 -r 24000/1001 -i - -c:v hevc_videotoolbox out.mov

 Progress and a summary go to stderr.
 */

import Foundation
import MVC3D

struct Options {
    var command = ""
    var positional: [String] = []
    var playlist: String?
    var packing = StereoPacking.topAndBottom
    var start = 0
    var frames = Int.max
    var threads = 0
    var output = "-"
}

func usage() -> Never {
    FileHandle.standardError.write(Data("""
    usage: mvc3d info DISC
           mvc3d decode DISC [--playlist NNNNN] [options]
           mvc3d decode BASE.m2ts [DEPENDENT.m2ts] [options]
           mvc3d cat DISC PATH [--start BYTE] [--frames BYTES]
    options: --packing tab|sbs  --start N  --frames N  --threads N  --output FILE|-

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
        case "--start": o.start = Int(value()) ?? 0
        case "--frames": o.frames = Int(value()) ?? Int.max
        case "--threads": o.threads = Int(value()) ?? 0
        case "--output", "-o": o.output = value()
        case let a where a.hasPrefix("-") && a != "-": usage()
        default: o.positional.append(args[i])
        }
        i += 1
    }
    return o
}

func log(_ s: String) { FileHandle.standardError.write(Data((s + "\n").utf8)) }

func clockTime(_ seconds: Double) -> String {
    let s = Int(seconds.rounded())
    return String(format: "%d:%02d:%02d", s / 3600, s / 60 % 60, s % 60)
}

/// A folder holding BDMV, or a disc image read in place.
func openDisc(_ path: String) throws -> BlurayDisc? {
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

/// Where access units come from: a disc playlist or a clip pair.
struct UnitSource {
    var next: () throws -> StereoAccessUnit?
    var unpaired: () -> Int
}

func unitSource(_ o: Options) throws -> UnitSource {
    if o.positional.count == 1, let disc = try openDisc(o.positional[0]) {
        let chosen: (name: String, playlist: BlurayPlaylist)?
        if let wanted = o.playlist {
            chosen = try disc.playlists().first { $0.name == wanted }
        } else {
            chosen = try disc.mainPlaylist()
        }
        guard let (name, playlist) = chosen else { throw FormatError("no such playlist") }
        log("mvc3d: playlist \(name), \(clockTime(playlist.duration)), \(playlist.isStereo ? "3D" : "2D only"), clips \(playlist.items.map(\.clip).joined(separator: " "))")
        let reader = disc.stereoReader(for: playlist)
        return UnitSource(next: reader.next, unpaired: { reader.unpairedBaseCount })
    }
    guard (1 ... 2).contains(o.positional.count) else { usage() }
    let reader = StereoStreamReader(
        base: try FileByteSource(url: URL(fileURLWithPath: o.positional[0])),
        dependent: try o.positional.dropFirst().first.map { try FileByteSource(url: URL(fileURLWithPath: $0)) })
    return UnitSource(next: reader.next, unpaired: { reader.unpairedBaseCount })
}

func decode(_ o: Options) throws {
    let source = try unitSource(o)
    let decoder = try MVCDecoder(threads: o.threads)
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
    let last = o.frames == Int.max ? Int.max : o.start + o.frames
    struct Done: Error {}

    func emit(_ frame: StereoFrame) throws {
        defer { index += 1 }
        guard index >= o.start else { return }
        guard index < last else { throw Done() }
        if frame.right == nil { mono += 1 }
        if !announced {
            let (w, h) = o.packing.packedSize(width: frame.width, height: frame.height)
            log("mvc3d: \(frame.width)x\(frame.height) per eye -> \(w)x\(h) \(o.packing.rawValue), yuv420p")
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
    }

    do {
        while let unit = try source.next() {
            try decoder.decode(unit, output: emit)
        }
        try decoder.finish(output: emit)
    } catch is Done {}

    let seconds = (clock.now - began) / .seconds(1)
    log(String(format: "mvc3d: wrote %d frames in %.1f s (%.1f fps); %d without a second view, %d unpaired AUs, %d corrupt NALs, %d unsupported NALs",
               written, seconds, Double(written) / seconds, mono, source.unpaired(),
               decoder.corruptNALCount, decoder.unsupportedNALCount))
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
    case "cat": try cat(options)
    default: usage()
    }
} catch {
    log("mvc3d: \(error)")
    exit(1)
}
