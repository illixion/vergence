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

    public init() {}
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
            guard ["auto", "sbs", "hsbs", "ou", "hou", "tab", "htab"].contains(v) else { throw UsageError("unknown --layout \(v)") }
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
        default: throw UsageError("unknown option \(name)")
        }
    }
    guard o.segmentSeconds >= 0.5, o.segmentSeconds <= 60 else { throw UsageError("--segment-seconds must be 0.5...60") }
    guard o.firstSegment >= 0 else { throw UsageError("--first must be 0 or more") }
    if let last = o.lastSegment, last < o.firstSegment {
        throw UsageError("--last \(last) comes before --first \(o.firstSegment)")
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
