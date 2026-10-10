/*
 Vergence - ffmpeg, for what Apple's frameworks can't read

 AVFoundation reads QuickTime and MP4 only, and many stereo files are
 Matroska (`.mkv`), WebM, AVI or transport streams. With ffmpeg on PATH (or
 VERGENCE_FFMPEG pointing at it), those are remuxed into a temporary
 QuickTime file first: H.264 and HEVC video copied as it is (anything else
 becomes ProRes 422 HQ, 10-bit, so nothing is lost before the real encode),
 AAC, AC-3, E-AC-3, ALAC and MP3 audio copied, lossless audio (TrueHD,
 DTS-HD MA, FLAC, PCM) to ALAC, other lossy audio to AAC. Subtitles and
 attachments are left out.

 It also grabs small greyscale pictures for the layout detector and reads a
 Matroska StereoMode, which a remux would lose.
 */

import Foundation
import VergenceCore
import VergenceDisc // FormatError

struct FFmpeg {
    let ffmpeg: URL
    let ffprobe: URL

    /// ffmpeg and ffprobe from VERGENCE_FFMPEG, PATH, or Homebrew's places.
    static func locate() -> FFmpeg? {
        let env = ProcessInfo.processInfo.environment
        var directories: [String] = []
        if let explicit = env["VERGENCE_FFMPEG"] { directories.append((explicit as NSString).deletingLastPathComponent) }
        directories += (env["PATH"] ?? "").split(separator: ":").map(String.init)
        directories += ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"]
        func find(_ name: String) -> URL? {
            if name == "ffmpeg", let explicit = env["VERGENCE_FFMPEG"], FileManager.default.isExecutableFile(atPath: explicit) {
                return URL(fileURLWithPath: explicit)
            }
            return directories.lazy.map { URL(fileURLWithPath: $0).appendingPathComponent(name) }
                .first { FileManager.default.isExecutableFile(atPath: $0.path) }
        }
        guard let ffmpeg = find("ffmpeg"), let ffprobe = find("ffprobe") else { return nil }
        return FFmpeg(ffmpeg: ffmpeg, ffprobe: ffprobe)
    }

    static func required(for what: String) throws -> FFmpeg {
        guard let found = locate() else {
            throw FormatError("\(what) needs ffmpeg: install it (brew install ffmpeg) or set VERGENCE_FFMPEG to its path")
        }
        return found
    }

    /// Files AVFoundation reads itself.
    static func isNative(_ url: URL) -> Bool { ["mov", "mp4", "m4v", "m4a"].contains(url.pathExtension.lowercased()) }

    // MARK: Probing

    struct Stream: Decodable {
        struct SideData: Decodable {
            let side_data_type: String?
            let type: String?
            let inverted: Int?
        }
        struct Tags: Decodable {
            let language: String?
        }
        let index: Int
        let codec_type: String?
        let codec_name: String?
        let profile: String?
        let width: Int?
        let height: Int?
        let pix_fmt: String?
        let avg_frame_rate: String?
        let channels: Int?
        let channel_layout: String?
        let sample_rate: String?
        let bit_rate: String?
        let color_primaries: String?
        let color_transfer: String?
        let tags: Tags?
        let side_data_list: [SideData]?

        var frameRate: Double {
            let parts = (avg_frame_rate ?? "").split(separator: "/").compactMap { Double($0) }
            return parts.count == 2 && parts[1] > 0 ? parts[0] / parts[1] : 0
        }

        /// A stereo layout the container declares (Matroska StereoMode,
        /// MP4 `st3d`).
        var declaredLayout: PackedStereoLayout? {
            guard let stereo = side_data_list?.first(where: { $0.side_data_type == "Stereo 3D" }) else { return nil }
            let swapped = stereo.inverted == 1
            switch stereo.type {
            case "side by side": return PackedStereoLayout(arrangement: .sideBySide, isHalfResolution: false, eyesReversed: swapped)
            case "top and bottom": return PackedStereoLayout(arrangement: .overUnder, isHalfResolution: false, eyesReversed: swapped)
            default: return nil
            }
        }
    }

    struct Probe: Decodable {
        struct Format: Decodable {
            let duration: String?
            let format_name: String?
        }
        let streams: [Stream]
        let format: Format?

        var video: Stream? { streams.first { $0.codec_type == "video" && !["mjpeg", "png"].contains($0.codec_name ?? "") } }
        var audio: [Stream] { streams.filter { $0.codec_type == "audio" } }
        var duration: Double { Double(format?.duration ?? "") ?? 0 }
    }

    func probe(_ url: URL) throws -> Probe {
        let data = try run(ffprobe, ["-v", "error", "-show_format", "-show_streams", "-of", "json", url.path], capture: true)
        do {
            return try JSONDecoder().decode(Probe.self, from: data)
        } catch {
            throw FormatError("ffprobe's answer for \(url.lastPathComponent) didn't parse: \(error)")
        }
    }

    // MARK: Remux

    /// What a remux does with each stream, for the log.
    struct RemuxPlan {
        var arguments: [String]
        var notes: [String]
    }

    func remuxPlan(_ probe: Probe, to output: URL, input: URL) throws -> RemuxPlan {
        guard let video = probe.video else { throw FormatError("no video stream") }
        var args = ["-nostdin", "-hide_banner", "-loglevel", "error", "-stats", "-y", "-i", input.path,
                    "-map", "0:\(video.index)", "-map_metadata", "-1", "-sn", "-dn"]
        var notes: [String] = []
        if ["h264", "hevc"].contains(video.codec_name ?? "") {
            args += ["-c:v", "copy"]
            if video.codec_name == "hevc" { args += ["-tag:v", "hvc1"] }
            notes.append("video \(video.codec_name ?? "?") copied")
        } else {
            args += ["-c:v", "prores_videotoolbox", "-profile:v", "hq"]
            notes.append("video \(video.codec_name ?? "?") to ProRes 422 HQ (AVFoundation can't decode it)")
        }
        for (n, audio) in probe.audio.enumerated() {
            args += ["-map", "0:\(audio.index)"]
            let codec = audio.codec_name ?? "?"
            let lossless = ["truehd", "flac", "mlp"].contains(codec) || codec.hasPrefix("pcm_")
                || (codec == "dts" && (audio.profile ?? "").contains("MA"))
            if ["aac", "ac3", "eac3", "alac", "mp3"].contains(codec) {
                args += ["-c:a:\(n)", "copy"]
                notes.append("audio \(n + 1) \(codec) copied")
            } else if lossless {
                args += ["-c:a:\(n)", "alac"]
                notes.append("audio \(n + 1) \(codec) to ALAC (lossless)")
            } else {
                let channels = max(audio.channels ?? 2, 1)
                args += ["-c:a:\(n)", "aac_at", "-b:a:\(n)", "\(min(channels, 8) * 96)k"]
                notes.append("audio \(n + 1) \(codec) to AAC")
            }
        }
        args += ["-f", "mov", output.path]
        return RemuxPlan(arguments: args, notes: notes)
    }

    func remux(_ plan: RemuxPlan) throws {
        _ = try run(ffmpeg, plan.arguments, capture: false)
    }

    // MARK: Pictures

    /// `count` greyscale pictures from across the video, `width` wide.
    func pictures(_ url: URL, probe: Probe, count: Int = 8, width: Int = 480) throws -> [LumaPlane] {
        guard let video = probe.video, let w = video.width, let h = video.height, w > 0, h > 0 else { return [] }
        let height = max(2, Int((Double(h) * Double(width) / Double(w)).rounded()) & ~1)
        var planes: [LumaPlane] = []
        for i in 0 ..< count {
            let time = probe.duration * (0.1 + 0.8 * Double(i) / Double(max(count - 1, 1)))
            let data = try run(ffmpeg, ["-nostdin", "-hide_banner", "-loglevel", "error", "-ss", String(format: "%.3f", time),
                                        "-i", url.path, "-frames:v", "1", "-vf", "scale=\(width):\(height),format=gray",
                                        "-f", "rawvideo", "-"], capture: true)
            guard data.count == width * height else { continue }
            planes.append(LumaPlane(width: width, height: height, pixels: [UInt8](data)))
        }
        return planes
    }

    // MARK: Running

    /// Runs `tool`, returning its stdout when `capture` (else it goes to the
    /// terminal). Throws with its stderr when it fails.
    @discardableResult
    func run(_ tool: URL, _ arguments: [String], capture: Bool, input: ((FileHandle) throws -> Void)? = nil) throws -> Data {
        let process = Process()
        process.executableURL = tool
        process.arguments = arguments
        let out = Pipe(), err = Pipe(), stdin = Pipe()
        if capture { process.standardOutput = out }
        process.standardError = capture ? err : FileHandle.standardError
        if input != nil { process.standardInput = stdin }
        try process.run()
        var feedError: Error?
        if let input {
            do { try input(stdin.fileHandleForWriting) } catch { feedError = error }
            try? stdin.fileHandleForWriting.close()
        }
        // Read before waiting, or a full pipe stalls the tool.
        let data = capture ? out.fileHandleForReading.readDataToEndOfFile() : Data()
        let errors = capture ? err.fileHandleForReading.readDataToEndOfFile() : Data()
        process.waitUntilExit()
        if let feedError { throw feedError }
        guard process.terminationStatus == 0 else {
            let message = String(decoding: errors, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
            throw FormatError("\(tool.lastPathComponent) failed (\(process.terminationStatus))\(message.isEmpty ? "" : ": " + message)")
        }
        return data
    }
}
