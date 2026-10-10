# Vergence

Stereo 3D video tools in Swift. Today: decode 3D Blu-ray video (H.264 MVC, "Stereo High") into
ordinary left/right pictures, on any platform Swift runs on, Apple's included. Planned: making
Apple spatial and immersive video (MV-HEVC, Apple Projected Media Profile) from any stereo source
without typing camera parameters by hand, and 2D-to-3D conversion from estimated depth.

Vergence works with decrypted discs only: a disc folder or image you have already made readable.
It contains no AACS or BD+ code.

## Layout

| Module | What | Builds on |
| --- | --- | --- |
| `VergenceDisc` | UDF images, BDAV demux, playlists and clip info, MVC decoding (edge264), frame packing, PGS subtitles, AC-3 frame parsing. Foundation only. | Apple platforms, Linux (tested), Android arm64 (builds; `scripts/build-android.sh`), Windows (planned) |
| `VergenceApple` | The Apple-framework half: `CVPixelBuffer` output, AC-3 decoding through AudioToolbox. Re-exports `VergenceDisc`. Builds empty elsewhere. | Apple platforms |
| `vergence` | The command line tool. | macOS, Linux (no PNG or audio output there) |

No consumer platform ships an MVC decoder: Apple's VideoToolbox, Android's MediaCodec and FFmpeg
all decode only the base view of a 3D Blu-ray and silently drop the second eye. Vergence reads both
views off the disc's clip files, joins them into one MVC stream, decodes it with
[edge264](https://github.com/tvlabs/edge264) and hands back both eyes, either as planes in memory
or frame-packed (top-and-bottom or side by side) for an encoder.

## Status

Early. Working: BDAV/TS demux, playlist (MPLS) parsing including the 3D extensions (the
dependent-view SubPath and STN_table_SS), picking the feature, pairing the base (`0x1011`) and
dependent (`0x1012`) views, decoding both with edge264, frame packing, a command line tool. Verified on a commercial 3D
Blu-ray (1080p23.976, CABAC): the whole 1:53:53 feature, 163,820 stereo frames in display order
with every timestamp one frame after the last and no errors, read from the disc image over SMB at
175 fps; the base view matches ffmpeg's decode frame for frame.

## Tests

`swift test` runs on synthetic data only, on macOS and on Linux
(`docker run --rm -v "$PWD":/src:ro swift:6.2.4-noble bash -c 'cp -r /src /work && cd /work && swift test'`;
the copy keeps the container's build out of the checkout). `scripts/build-android.sh` cross-compiles
for Android arm64 with the open-source toolchain and its Android SDK bundle; nothing runs on a device
yet. `scripts/fetch-conformance.sh` downloads the ITU-T MVC
conformance bitstreams (36 MB, not redistributed here);
`VERGENCE_CONFORMANCE=.build/conformance/mvc swift test` then decodes them too.

Disc images (`.iso`) are read in place, without mounting, by a read-only UDF 2.50 reader
(metadata partition included) over any `ByteSource`. `HTTPByteSource` reads one from a web server
with range requests (a media server's download URL, any static file server), through a cache of
4 MB blocks with read-ahead. Jellyfin serves a disc image item's raw bytes with ranges at
`/Items/{id}/Download` (authorised with `Authorization: MediaBrowser Token="…"`); decoding from it
over HTTPS on the LAN ran at 278 fps (11× realtime), byte-identical to reading the file. Damaged bytes in a clip don't stop decoding:
the demuxer skips to where packets resume and counts the spot.

Seeking uses the clips' entry point maps (CLPI EP_map; the dependent view's is in the CLPI's
ExtensionData): decoding restarts at the random access picture before the target and drops the
pictures in between, so a seek costs at most a GOP of decoding (0.4 s from an image over SMB) and
lands on the exact frame a decode from the start would give. PlayItems start at their IN time and
end at their OUT time.

On Apple platforms `StereoPixelBufferMaker` copies each frame into pooled, IOSurface-backed
`CVPixelBuffer`s ('420v', tagged BT.709): one per eye, or both eyes packed top and bottom or side
by side, for players that already crop eyes from frame-packed video. About 0.25 ms per 1080p
stereo frame.

Planned: encoder integration.

## Speed

Apple M-series Mac, both eyes of a 1080p Blu-ray:

| threads | stereo frames/s | × realtime (23.976 fps) |
| --- | --- | --- |
| 1 | ~61 | 2.5× |
| all cores | ~426 | 17.8× |

## Command line

```sh
swift build -c release
.build/release/vergence info /Volumes/DISC          # playlists; * marks the feature
.build/release/vergence decode /Volumes/DISC --threads -1 \
  | ffmpeg -f rawvideo -pix_fmt yuv420p -s 1920x2160 -r 24000/1001 -i - \
           -c:v hevc_videotoolbox -q:v 65 -tag:v hvc1 film.3D.FTAB.mov
```

`DISC` is a folder holding `BDMV`, a disc image (`.iso`), or the `http(s)` URL of one
(`--header "Name: value"` or `VERGENCE_HTTP_HEADER` for authentication). `--playlist NNNNN` picks another title; a clip pair
(`decode 00007.m2ts 00008.m2ts`) works too. `--packing sbs` for side by side,
`--seek SECONDS` to start part way in, `--start N --frames N` for a range of frames,
`--output FILE`.

`vergence stream DISC [--seek SECONDS]` writes the feature's main clips (base view,
audio, subtitles) as a `.m2ts` byte stream from the entry point at or before the
seek, for tools that can read a stream but not seek in a disc image. ffmpeg's
`bluray:` input took 75 s to start a cut an hour into a film on a home server,
this about 2 s:

```sh
.build/release/vergence stream disc.iso --seek 3600 | ffmpeg -i - -map 0:1 -c copy -f truehd audio.thd
```

`vergence subtitles DISC --pid 0x1203 [--output DIR]` decodes a PGS subtitle track
(PIDs as `vergence streams` lists them) without the video: a line per caption with
its time and place in the frame, and with `--output` each caption as a PNG.

`vergence audio DISC --pid 0x1101 --seek 600 --frames 20 -o out.wav` decodes an AC-3
or E-AC-3 track, or the AC-3 core of a TrueHD track, through Apple's decoder to a
5.1 WAV (L R C LFE Ls Rs). It reads in windows of a few seconds, the way a player
does. Against ffmpeg's decode of the same core it lines up within 32 samples, and
the channels correlate at 0.996 or better. ffmpeg comes out about 2 dB quieter
because it applies the stream's dynamic range compression.

## Library

```swift
import VergenceApple // or VergenceDisc off Apple platforms

let disc = BlurayDisc(fileSystem: DirectoryFileSystem(root: discFolder))
let decoder = try PlaylistDecoder(disc: disc, playlist: try disc.mainPlaylist()!.playlist, threads: -1)
try decoder.seek(to: 600)
try decoder.decode { frame in
    // frame.pts: 90 kHz on the playlist's timeline
    // frame.left / frame.right: Y, Cb, Cr planes, valid inside this closure
    return true // false stops; calling decode again continues
}
```

`StereoPixelBufferMaker(layout: .packed(.topAndBottom)).makeBuffers(frame)` turns a frame into
`CVPixelBuffer`s inside that closure. Lower down, `PlaylistStereoReader` gives the stereo access units and `MVCDecoder` decodes any
stream of them.

The other streams of a clip come from the same reads: set `PlaylistDecoder.sidePIDs`
and `onSidePackets` hands over their PES, on the playlist timeline, as the video is
read. `PGSDecoder` turns subtitle PES (or a `.sup` file, through `SupFile`) into
captions. They keep their bitmaps compressed until `renderRGBA()`.
`PlaylistAudioReader` reads any window of an audio track as 5.1 PCM. It uses its
own reads, so sound can be ready ahead of the picture. It takes an `AudioFrameDecoder`;
`VergenceApple` supplies `AC3Decoder` and a convenience initializer that uses it.

## License

MIT, see `LICENSE`. edge264 (vendored under `Sources/CEdge264`) is BSD-3-Clause; see
`Vendor/README.md` for the local patches.
