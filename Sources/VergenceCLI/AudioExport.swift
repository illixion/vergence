/*
 Vergence - `audio VIDEO`: one audio track out of a video file

 Picks the spatial track unless --track says otherwise, decodes it through
 AVFoundation (the only decoder for APAC) and writes:
 - .wav: 32-bit float, the channels as decoded (ambisonics as AmbiX);
 - .opus .ogg .mka .webm .mp4 .m4a: Opus through ffmpeg, ambisonics with
   channel mapping family 2 (RFC 8486), which marks them as ambisonic in the
   stream itself.
 */

import Foundation
import VergenceApple
import VergenceDisc

func exportAudio(_ o: Options) async throws {
    #if !canImport(AVFoundation) || os(tvOS)
    throw FormatError("decoding a video's audio needs AVFoundation (macOS)")
    #else
    guard o.positional.count == 1, o.output != "-" else { usage("audio VIDEO needs --output FILE.opus or FILE.wav") }
    let input = URL(fileURLWithPath: o.positional[0])
    let output = URL(fileURLWithPath: o.output)
    let tracks = try await AudioTrackReader.tracks(input)
    guard !tracks.isEmpty else { throw FormatError("\(input.lastPathComponent) has no audio track AVFoundation can read") }
    let chosen: AudioTrackInfo
    if let n = o.track {
        guard (1 ... tracks.count).contains(n) else { usage("--track goes from 1 to \(tracks.count)") }
        chosen = tracks[n - 1]
    } else {
        chosen = tracks.first(where: \.isAmbisonic) ?? tracks[0]
    }
    for (i, track) in tracks.enumerated() {
        log("vergence: track \(i + 1): \(track.summary)\(track.trackID == chosen.trackID ? "  <- exporting" : "")")
    }

    let began = Date()
    let frames: Int
    switch output.pathExtension.lowercased() {
    case "wav":
        frames = try await writeFloatWAV(input, chosen, to: output)
    case "opus", "ogg", "mka", "webm", "mp4", "m4a":
        frames = try await encodeOpus(input, chosen, to: output, bitrate: opusBitrate(o.bitrateText, chosen))
    default:
        usage("audio VIDEO writes .opus, .ogg, .mka, .webm, .mp4, .m4a or .wav")
    }
    log(String(format: "vergence: %.1f s of audio in %.1f s -> %@%@", Double(frames) / chosen.sampleRate,
               Date().timeIntervalSince(began), output.lastPathComponent,
               chosen.isAmbisonic ? " (AmbiX: ACN, SN3D)" : ""))
    #endif
}

#if canImport(AVFoundation) && !os(tvOS)
/// Opus bits per second: --bitrate in kbps (or with K/M), else 64 kbps a
/// channel for ambisonics and 96 for the rest, at most 512.
func opusBitrate(_ text: String?, _ track: AudioTrackInfo) -> Int {
    if let text {
        let t = text.uppercased()
        if t.hasSuffix("K") || t.hasSuffix("M"), let v = parseBitrate(t) { return v }
        if let kbps = Double(t), kbps > 0 { return Int(kbps * 1000) }
        usage("bad --bitrate \(text) (kbps, e.g. 256)")
    }
    return min(track.channels * (track.isAmbisonic ? 64 : 96), 512) * 1000
}

func encodeOpus(_ input: URL, _ track: AudioTrackInfo, to output: URL, bitrate: Int) async throws -> Int {
    let ffmpeg = try FFmpeg.required(for: "writing Opus")
    // Decode everything first: the reader is async, ffmpeg's stdin isn't.
    let raw = output.deletingLastPathComponent().appendingPathComponent(".\(output.lastPathComponent).f32")
    FileManager.default.createFile(atPath: raw.path, contents: nil)
    defer { try? FileManager.default.removeItem(at: raw) }
    let handle = try FileHandle(forWritingTo: raw)
    let frames = try await AudioTrackReader.decode(input, trackID: track.trackID) { samples, _ in
        try handle.write(contentsOf: Data(buffer: samples))
    }
    try handle.close()

    var args = ["-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "f32le", "-ar", String(Int(track.sampleRate))]
    if let order = track.ambisonicOrder {
        args += ["-ch_layout", "ambisonic \(order)"]
    } else {
        args += ["-ac", String(track.channels)]
    }
    args += ["-i", raw.path, "-c:a", "libopus", "-b:a", String(bitrate)]
    if track.isAmbisonic { args += ["-mapping_family", "2"] }
    if let language = track.language { args += ["-metadata:s:a:0", "language=\(language)"] }
    args.append(output.path)
    try ffmpeg.run(ffmpeg.ffmpeg, args, capture: false)
    return frames
}

/// 32-bit float WAV (WAVE_FORMAT_EXTENSIBLE past two channels, with no
/// speaker positions for ambisonics).
func writeFloatWAV(_ input: URL, _ track: AudioTrackInfo, to output: URL) async throws -> Int {
    FileManager.default.createFile(atPath: output.path, contents: nil)
    let handle = try FileHandle(forWritingTo: output)
    defer { try? handle.close() }
    let channels = track.channels, rate = Int(track.sampleRate)
    let extensible = channels > 2
    let headerSize = extensible ? 68 : 44
    try handle.write(contentsOf: Data(count: headerSize))
    var bytes = 0
    let frames = try await AudioTrackReader.decode(input, trackID: track.trackID) { samples, _ in
        try handle.write(contentsOf: Data(buffer: samples))
        bytes += samples.count * 4
    }
    var header = Data()
    func u32(_ v: Int) { withUnsafeBytes(of: UInt32(v).littleEndian) { header.append(contentsOf: $0) } }
    func u16(_ v: Int) { withUnsafeBytes(of: UInt16(v).littleEndian) { header.append(contentsOf: $0) } }
    header.append(contentsOf: Array("RIFF".utf8)); u32(headerSize - 8 + bytes)
    header.append(contentsOf: Array("WAVEfmt ".utf8)); u32(extensible ? 40 : 16)
    u16(extensible ? 0xFFFE : 3); u16(channels); u32(rate); u32(rate * channels * 4); u16(channels * 4); u16(32)
    if extensible {
        u16(22); u16(32)
        u32(track.isAmbisonic ? 0 : (1 << channels) - 1)
        // KSDATAFORMAT_SUBTYPE_IEEE_FLOAT
        header.append(contentsOf: [0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71])
    }
    header.append(contentsOf: Array("data".utf8)); u32(bytes)
    try handle.seek(toOffset: 0)
    try handle.write(contentsOf: header)
    return frames
}
#endif
