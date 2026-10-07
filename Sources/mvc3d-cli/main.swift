/*
 mvc3d - command line

   mvc3d decode BASE.m2ts DEPENDENT.m2ts [--packing tab|sbs] [--start N]
                [--frames N] [--threads N] [--output FILE|-]

 Writes packed I420 frames (rawvideo, yuv420p) for ffmpeg, e.g.

   mvc3d decode 00007.m2ts 00008.m2ts | ffmpeg -f rawvideo -pix_fmt yuv420p \
     -s 1920x2160 -r 24000/1001 -i - -c:v hevc_videotoolbox out.mkv

 Progress and a summary go to stderr.
 */

import Foundation
import MVC3D

struct Options {
    var base = ""
    var dependent: String?
    var packing = StereoPacking.topAndBottom
    var start = 0
    var frames = Int.max
    var threads = 0
    var output = "-"
}

func usage() -> Never {
    FileHandle.standardError.write(Data("""
    usage: mvc3d decode BASE.m2ts [DEPENDENT.m2ts] [--packing tab|sbs] [--start N] [--frames N]
                        [--threads N] [--output FILE|-]

    """.utf8))
    exit(64)
}

func parse(_ args: [String]) -> Options {
    var o = Options()
    var positional: [String] = []
    var i = 0
    func value() -> String {
        i += 1
        guard i < args.count else { usage() }
        return args[i]
    }
    while i < args.count {
        switch args[i] {
        case "--packing": o.packing = StereoPacking(rawValue: value()) ?? { usage() }()
        case "--start": o.start = Int(value()) ?? 0
        case "--frames": o.frames = Int(value()) ?? Int.max
        case "--threads": o.threads = Int(value()) ?? 0
        case "--output", "-o": o.output = value()
        case let a where a.hasPrefix("-") && a != "-": usage()
        default: positional.append(args[i])
        }
        i += 1
    }
    guard (1 ... 2).contains(positional.count) else { usage() }
    o.base = positional[0]
    o.dependent = positional.count > 1 ? positional[1] : nil
    return o
}

func log(_ s: String) { FileHandle.standardError.write(Data((s + "\n").utf8)) }

let arguments = Array(CommandLine.arguments.dropFirst())
guard arguments.first == "decode" else { usage() }
let options = parse(Array(arguments.dropFirst()))

do {
    let reader = StereoStreamReader(
        base: try FileByteSource(url: URL(fileURLWithPath: options.base)),
        dependent: try options.dependent.map { try FileByteSource(url: URL(fileURLWithPath: $0)) })
    let decoder = try MVCDecoder(threads: options.threads)
    let out: FileHandle
    if options.output == "-" {
        out = .standardOutput
    } else {
        FileManager.default.createFile(atPath: options.output, contents: nil)
        out = try FileHandle(forWritingTo: URL(fileURLWithPath: options.output))
    }

    let clock = ContinuousClock()
    let began = clock.now
    var index = 0, written = 0, mono = 0
    var buffer = Data()
    var announced = false
    let last = options.frames == Int.max ? Int.max : options.start + options.frames
    struct Done: Error {}

    func emit(_ frame: StereoFrame) throws {
        defer { index += 1 }
        guard index >= options.start else { return }
        guard index < last else { throw Done() }
        if frame.right == nil { mono += 1 }
        if !announced {
            let (w, h) = options.packing.packedSize(width: frame.width, height: frame.height)
            log("mvc3d: \(frame.width)x\(frame.height) per eye -> \(w)x\(h) \(options.packing.rawValue), yuv420p")
            announced = true
        }
        let count = frame.packedByteCount(options.packing)
        if buffer.count != count { buffer = Data(count: count) }
        buffer.withUnsafeMutableBytes { frame.pack(options.packing, into: $0) }
        try out.write(contentsOf: buffer)
        written += 1
        if written % 240 == 0 {
            let seconds = (clock.now - began) / .seconds(1)
            log(String(format: "mvc3d: %d frames, %.1f fps", written, Double(written) / seconds))
        }
    }

    do {
        while let unit = try reader.next() {
            try decoder.decode(unit, output: emit)
        }
        try decoder.finish(output: emit)
    } catch is Done {}

    let seconds = (clock.now - began) / .seconds(1)
    log(String(format: "mvc3d: wrote %d frames in %.1f s (%.1f fps); %d without a second view, %d unpaired AUs, %d corrupt NALs, %d unsupported NALs",
               written, seconds, Double(written) / seconds, mono, reader.unpairedBaseCount,
               decoder.corruptNALCount, decoder.unsupportedNALCount))
} catch {
    log("mvc3d: \(error)")
    exit(1)
}
