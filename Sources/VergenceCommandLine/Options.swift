/*
 Vergence - the command line's options and output names

 The parsing and naming half of the `vergence` tool, apart from the
 executable so tests can reach it. Foundation only: it builds wherever the
 tool does. Nothing here prints or exits; a bad command line throws
 `UsageError`, which the tool turns into its usage text and exit code 64.
 */

import Foundation
import VergenceCore
import VergenceDisc

public struct Options: Equatable, Sendable {
    public var command = ""
    public var positional: [String] = []
    public var playlist: String?
    public var packing = StereoPacking.topAndBottom
    public var seek: Double?
    public var start = 0
    public var frames = Int.max
    public var threads = 0
    public var output = "-"
    public var pid: UInt16?
    public var headers: [String: String] = [:]
    // make
    public var layout: String = "auto"
    public var swapEyes = false
    public var preset = SpatialPreset.generic
    public var baseline: Double?
    public var hfov: Double?
    public var adjust: Double?
    public var quality = SpatialQuality.standard
    public var bitrate: Int?
    public var bitrateText: String?
    public var track: Int?
    // sbs
    public var half = false
    public var maxEye: Int?
    public var ambisonics = true
    public var includeAudio = true
    public var dryRun = false
    public var force = false
    // sbs as stream segments
    public var segments: String?
    public var segmentSeconds = 4.0
    public var firstSegment = 0
    public var lastSegment: Int?
    public var info = false
    /// `analyze --raw WxH`: packed top/bottom I420 frames, each eye W×H.
    public var raw: EyeSize?
    /// `analyze --rate N/D`: frames per second of raw input.
    public var rate = FrameRate(numerator: 24000, denominator: 1001)
    /// `analyze --models DIR`: the Core ML flow and disparity models.
    public var models: String?
    /// `interpolate --fields FILE.vgat`: the analysis track to make in-between frames from.
    public var fields: String?
    /// `interpolate --no-guard` (mono): VideoToolbox's frames as they come, without `InterpolationGuard`.
    public var noGuard = false
    /// `interpolate --gpu` (mono): `FlowMidFrameInterpolator`'s frames (block-matched motion on the GPU,
    /// Smooth Motion's 60 → 120 fps path) instead of VideoToolbox's.
    public var gpuFlow = false
    /// `interpolate --scale S` (mono): the share of the frame's size the interpolator works at, as the
    /// governor's smaller step (0.67) runs it.
    public var scale: Double?
    // convert
    /// `--strength`: the warp's largest per-eye shift, a share of the frame's
    /// width (nil: `Pseudo3DSettings`' default).
    public var strength: Double?
    /// `--convergence`: the depth on the screen plane, 0…1 (1: the nearest).
    public var convergence: Double?
    /// `--sample SECONDS`: convert only that long a piece (from the middle
    /// unless `--start` says where).
    public var sample: Double?
    /// `--model PATH`: the Core ML depth model (.mlpackage or .mlmodelc).
    public var model: String?
    /// `--depth-track FILE.vgat`: also write the depth as an analysis track.
    public var depthTrack: String?
    /// `--download-model`: fetch Depth Anything V2 Small first if none is installed.
    public var downloadModel = false

    public init() {}

    /// `convert`'s output layout from `--layout` (full side by side unless given).
    public var convertLayout: ConvertLayout { ConvertLayout(option: layout) ?? .fullSideBySide }
}

/// What `vergence convert` writes.
public enum ConvertLayout: String, Sendable, CaseIterable {
    case fullSideBySide, halfSideBySide, fullTopAndBottom, halfTopAndBottom, spatial

    /// From a `--layout` value; nil for one convert doesn't take.
    public init?(option: String) {
        switch option.lowercased() {
        case "auto", "sbs", "fsbs": self = .fullSideBySide
        case "hsbs": self = .halfSideBySide
        case "tab", "ftab", "ou": self = .fullTopAndBottom
        case "htab", "hou": self = .halfTopAndBottom
        case "spatial", "mvhevc": self = .spatial
        default: return nil
        }
    }

    public var isSideBySide: Bool { self == .fullSideBySide || self == .halfSideBySide || self == .spatial }
    /// Each eye squeezed to half the frame's width (side by side) or height.
    public var isHalf: Bool { self == .halfSideBySide || self == .halfTopAndBottom }

    /// The name's ending, which players and media servers read the layout from.
    public var nameSuffix: String {
        switch self {
        case .fullSideBySide: ".3D.FSBS.mp4"
        case .halfSideBySide: ".3D.HSBS.mp4"
        case .fullTopAndBottom: ".3D.FTAB.mp4"
        case .halfTopAndBottom: ".3D.HTAB.mp4"
        case .spatial: ".spatial.mov"
        }
    }
}

public struct EyeSize: Equatable, Sendable {
    public var width: Int
    public var height: Int
    public init(width: Int, height: Int) {
        self.width = width
        self.height = height
    }
}

public struct FrameRate: Equatable, Sendable {
    public var numerator: Int
    public var denominator: Int
    public init(numerator: Int, denominator: Int) {
        self.numerator = numerator
        self.denominator = denominator
    }
}

/// A command line that can't run: what's wrong with it, if there is more to
/// say than the usage text.
public struct UsageError: Error, Equatable, CustomStringConvertible {
    public var problem: String?
    public init(_ problem: String? = nil) { self.problem = problem }
    public var description: String { problem ?? "usage" }
}

/// Options take their value as the next argument or after "=". `environment`
/// is where VERGENCE_HTTP_HEADER is read from.
public func parse(_ args: [String], environment: [String: String] = [:]) throws(UsageError) -> Options {
    var o = Options()
    guard let command = args.first else { throw UsageError() }
    o.command = command
    var i = 1
    while i < args.count {
        let arg = args[i]
        i += 1
        guard arg.hasPrefix("-"), arg != "-", Double(arg) == nil else {
            o.positional.append(arg)
            continue
        }
        // --name=value, or --name followed by its value (which may itself
        // start with "-", like a negative number).
        var name = arg, inline: String?
        if let equals = arg.firstIndex(of: "="), arg.hasPrefix("--") {
            name = String(arg[..<equals])
            inline = String(arg[arg.index(after: equals)...])
        }
        func value() throws(UsageError) -> String {
            if let inline { return inline }
            guard i < args.count else { throw UsageError("\(name) needs a value") }
            defer { i += 1 }
            return args[i]
        }
        func number(_ text: String) throws(UsageError) -> Double {
            guard let v = Double(text), v.isFinite else { throw UsageError("\(name) takes a number, not \"\(text)\"") }
            return v
        }
        // Whole numbers only: Int(Double) traps past Int's range, and a
        // fraction would be dropped without a word.
        func integer(_ text: String) throws(UsageError) -> Int {
            guard let v = Int(text) ?? Double(text).flatMap({ $0.rounded() == $0 ? Int(exactly: $0) : nil }) else {
                throw UsageError("\(name) takes a whole number, not \"\(text)\"")
            }
            return v
        }
        func flag() throws(UsageError) {
            if inline != nil { throw UsageError("\(name) takes no value") }
        }
        switch name {
        case "--playlist": o.playlist = try value()
        case "--packing":
            let v = try value()
            guard let packing = StereoPacking(rawValue: v) else { throw UsageError("unknown --packing \(v)") }
            o.packing = packing
        case "--seek": o.seek = try number(value())
        // convert counts in seconds, like --seek; the disc commands in frames.
        case "--start" where o.command == "convert": o.seek = try number(value())
        case "--start": o.start = try integer(value())
        case "--frames": o.frames = try integer(value())
        case "--threads": o.threads = try integer(value())
        case "--output", "-o": o.output = try value()
        case "--pid":
            let v = try value()
            guard let pid = UInt16(v.replacingOccurrences(of: "0x", with: ""), radix: 16) else { throw UsageError("bad --pid \(v)") }
            o.pid = pid
        case "--header", "-H": try addHeader(value(), to: &o)
        case "--layout", "-f", "--format":
            let v = try value().lowercased()
            guard ["auto", "sbs", "hsbs", "ou", "hou", "tab", "htab", "fsbs", "ftab", "spatial", "mvhevc"].contains(v) else {
                throw UsageError("unknown --layout \(v)")
            }
            o.layout = v
        case "--swap-eyes", "--reverse", "-r": try flag(); o.swapEyes = true
        case "--preset":
            let v = try value().lowercased()
            guard let preset = SpatialPreset(rawValue: v) else {
                throw UsageError("unknown --preset \(v) (\(SpatialPreset.allCases.map(\.rawValue).joined(separator: ", ")))")
            }
            o.preset = preset
        case "--baseline", "--cdist": o.baseline = try number(value())
        case "--hfov", "--fov": o.hfov = try number(value())
        case "--adjust", "--hadjust", "--disparity": o.adjust = try number(value())
        case "--quality", "-q":
            let v = try value().lowercased()
            guard let quality = SpatialQuality(rawValue: v) else { throw UsageError("unknown --quality \(v)") }
            o.quality = quality
        case "--bitrate", "-b":
            let v = try value()
            o.bitrateText = v
            guard let bitrate = parseBitrate(v) else { throw UsageError("bad --bitrate (try 20, 20M or 750K)") }
            o.bitrate = bitrate
        case "--track": o.track = try integer(value())
        case "--half": try flag(); o.half = true
        case "--max-eye":
            let v = try integer(value())
            guard v > 0 else { throw UsageError("--max-eye takes a size in pixels, not \(v)") }
            o.maxEye = v
        case "--no-ambisonics": try flag(); o.ambisonics = false
        case "--no-audio": try flag(); o.includeAudio = false
        case "--dry-run", "-n": try flag(); o.dryRun = true
        case "--force", "-y": try flag(); o.force = true
        case "--segments": o.segments = try value()
        case "--segment-seconds": o.segmentSeconds = try number(value())
        case "--first": o.firstSegment = try integer(value())
        case "--last": o.lastSegment = try integer(value())
        case "--info": try flag(); o.info = true
        case "--raw":
            let parts = try value().lowercased().split(separator: "x").compactMap { Int($0) }
            guard parts.count == 2, parts.allSatisfy({ $0 > 0 && $0 % 2 == 0 }) else {
                throw UsageError("--raw takes the size of one eye, e.g. 1920x1080")
            }
            o.raw = EyeSize(width: parts[0], height: parts[1])
        case "--rate":
            let text = try value()
            let parts = text.split(separator: "/").compactMap { Int($0) }
            guard (1 ... 2).contains(parts.count), parts.allSatisfy({ $0 > 0 }) else {
                throw UsageError("--rate takes frames per second as N or N/D, not \"\(text)\"")
            }
            o.rate = FrameRate(numerator: parts[0], denominator: parts.count == 2 ? parts[1] : 1)
        case "--models": o.models = try value()
        case "--fields": o.fields = try value()
        case "--no-guard": try flag(); o.noGuard = true
        case "--gpu": try flag(); o.gpuFlow = true
        case "--scale":
            let text = try value()
            guard let scale = Double(text), scale > 0, scale <= 1 else {
                throw UsageError("--scale takes a share of the frame's size, 0...1, not \"\(text)\"")
            }
            o.scale = scale
        case "--strength":
            let v = try number(value())
            guard v >= 0, v <= 0.05 else { throw UsageError("--strength takes a share of the frame's width, 0...0.05, not \(v)") }
            o.strength = v
        case "--convergence":
            let v = try number(value())
            guard (0 ... 1).contains(v) else { throw UsageError("--convergence takes 0...1, not \(v)") }
            o.convergence = v
        case "--sample":
            let v = try number(value())
            guard v > 0 else { throw UsageError("--sample takes seconds, more than 0") }
            o.sample = v
        case "--model": o.model = try value()
        case "--depth-track": o.depthTrack = try value()
        case "--download-model": try flag(); o.downloadModel = true
        default: throw UsageError("unknown option \(name)")
        }
    }
    guard o.segmentSeconds >= 0.5, o.segmentSeconds <= 60 else { throw UsageError("--segment-seconds must be 0.5...60") }
    guard o.firstSegment >= 0 else { throw UsageError("--first must be 0 or more") }
    if let last = o.lastSegment, last < o.firstSegment {
        throw UsageError("--last \(last) comes before --first \(o.firstSegment)")
    }
    if o.command != "convert", ["fsbs", "ftab", "spatial", "mvhevc"].contains(o.layout) {
        throw UsageError("--layout \(o.layout) is for convert")
    }
    if o.command == "convert" {
        if let seek = o.seek, seek < 0 { throw UsageError("--start must be 0 or more") }
        if o.frames != Int.max, o.frames < 1 { throw UsageError("--frames must be 1 or more") }
    }
    if let header = environment["VERGENCE_HTTP_HEADER"] { try addHeader(header, to: &o) }
    return o
}

/// "20" and "20M" are megabits per second, "750K" kilobits, "20000000" bits.
public func parseBitrate(_ text: String) -> Int? {
    let t = text.uppercased()
    let (digits, scale): (Substring, Double) = t.hasSuffix("M") ? (t.dropLast(), 1e6) : t.hasSuffix("K") ? (t.dropLast(), 1e3) : (Substring(t), 0)
    guard let v = Double(digits), v > 0,
          let bits = Int(exactly: (v * (scale > 0 ? scale : v < 10_000 ? 1e6 : 1)).rounded()), bits > 0 else { return nil }
    return bits
}

/// "Name: value", as curl takes it.
public func addHeader(_ line: String, to o: inout Options) throws(UsageError) {
    guard let colon = line.firstIndex(of: ":") else { throw UsageError("a header is \"Name: value\", not \"\(line)\"") }
    o.headers[String(line[..<colon]).trimmingCharacters(in: .whitespaces)] =
        String(line[line.index(after: colon)...]).trimmingCharacters(in: .whitespaces)
}

/// Where each command writes, from its input's name and --output.
public enum OutputNames {
    /// `make`: VIDEO.ext becomes VIDEO.spatial.mov.
    public static func spatialVideo(for input: URL, _ o: Options, single: Bool) -> URL {
        placed(input.deletingPathExtension().lastPathComponent + ".spatial.mov", beside: input, o, single: single)
    }

    /// `sbs`: VIDEO.ext becomes VIDEO.3D.FSBS.mp4, or .3D.HSBS.mp4 with --half.
    public static func sideBySide(for input: URL, _ o: Options, single: Bool) -> URL {
        placed(input.deletingPathExtension().lastPathComponent + (o.half ? ".3D.HSBS.mp4" : ".3D.FSBS.mp4"),
               beside: input, o, single: single)
    }

    /// `convert`: VIDEO.ext becomes VIDEO.3D.FSBS.mp4 (.3D.HSBS.mp4, .3D.FTAB.mp4,
    /// .3D.HTAB.mp4 or .spatial.mov by --layout); a --sample piece is
    /// VIDEO.sample.3D.FSBS.mp4.
    public static func converted(for input: URL, _ o: Options, single: Bool) -> URL {
        placed(input.deletingPathExtension().lastPathComponent + (o.sample != nil ? ".sample" : "")
                   + o.convertLayout.nameSuffix, beside: input, o, single: single)
    }

    /// `sbs --segments DIR`: the folder (the current one for `--info` alone)
    /// and the media playlist in it, beside the init segment and N.m4s.
    public static func segmentDirectory(_ o: Options) -> URL { URL(fileURLWithPath: o.segments ?? ".") }
    public static let segmentPlaylist = "index.m3u8"

    /// Beside the input without --output; with it, that file for a single
    /// input, else `name` in that folder (also when it is an existing folder
    /// or ends in "/").
    static func placed(_ name: String, beside input: URL, _ o: Options, single: Bool) -> URL {
        guard o.output != "-" else { return input.deletingLastPathComponent().appendingPathComponent(name) }
        let target = URL(fileURLWithPath: o.output)
        var isDirectory: ObjCBool = false
        let exists = FileManager.default.fileExists(atPath: target.path, isDirectory: &isDirectory)
        if single, !(exists && isDirectory.boolValue), !o.output.hasSuffix("/") { return target }
        return target.appendingPathComponent(name)
    }
}
