# mvc3d

Decode 3D Blu-ray video (H.264 MVC, "Stereo High") into ordinary left/right pictures, in Swift,
on any platform Swift runs on, Apple's included.

No consumer platform ships an MVC decoder: Apple's VideoToolbox, Android's MediaCodec and FFmpeg
all decode only the base view of a 3D Blu-ray and silently drop the second eye. mvc3d reads both
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

`swift test` runs on synthetic data only. `scripts/fetch-conformance.sh` downloads the ITU-T MVC
conformance bitstreams (36 MB, not redistributed here);
`MVC3D_CONFORMANCE=.build/conformance/mvc swift test` then decodes them too.

Disc images (`.iso`) are read in place, without mounting, by a read-only UDF 2.50 reader
(metadata partition included) over any `ByteSource`. `HTTPByteSource` reads one from a web server
with range requests (a media server's download URL, any static file server), through a cache of
4 MB blocks with read-ahead; decoding from an image served over HTTP on the LAN ran at 309 fps
(13× realtime), byte-identical to reading the file. Damaged bytes in a clip don't stop decoding:
the demuxer skips to where packets resume and counts the spot.

Seeking uses the clips' entry point maps (CLPI EP_map; the dependent view's is in the CLPI's
ExtensionData): decoding restarts at the random access picture before the target and drops the
pictures in between, so a seek costs at most a GOP of decoding (0.4 s from an image over SMB) and
lands on the exact frame a decode from the start would give. PlayItems start at their IN time and
end at their OUT time.

Planned: `CVPixelBuffer` output on Apple platforms, encoder integration.

## Speed

Apple M-series Mac, both eyes of a 1080p Blu-ray:

| threads | stereo frames/s | × realtime (23.976 fps) |
| --- | --- | --- |
| 1 | ~61 | 2.5× |
| all cores | ~426 | 17.8× |

## Command line

```sh
swift build -c release
.build/release/mvc3d info /Volumes/DISC          # playlists; * marks the feature
.build/release/mvc3d decode /Volumes/DISC --threads -1 \
  | ffmpeg -f rawvideo -pix_fmt yuv420p -s 1920x2160 -r 24000/1001 -i - \
           -c:v hevc_videotoolbox -q:v 65 -tag:v hvc1 film.3D.FTAB.mov
```

`DISC` is a folder holding `BDMV`, a disc image (`.iso`), or the `http(s)` URL of one
(`--header "Name: value"` or `MVC3D_HTTP_HEADER` for authentication). `--playlist NNNNN` picks another title; a clip pair
(`decode 00007.m2ts 00008.m2ts`) works too. `--packing sbs` for side by side,
`--seek SECONDS` to start part way in, `--start N --frames N` for a range of frames,
`--output FILE`.

## Library

```swift
import MVC3D

let disc = BlurayDisc(fileSystem: DirectoryFileSystem(root: discFolder))
let decoder = try PlaylistDecoder(disc: disc, playlist: try disc.mainPlaylist()!.playlist, threads: -1)
try decoder.seek(to: 600)
try decoder.decode { frame in
    // frame.pts: 90 kHz on the playlist's timeline
    // frame.left / frame.right: Y, Cb, Cr planes, valid inside this closure
    return true // false stops; calling decode again continues
}
```

Lower down, `PlaylistStereoReader` gives the stereo access units and `MVCDecoder` decodes any
stream of them.

## License

MIT, see `LICENSE`. edge264 (vendored under `Sources/CEdge264`) is BSD-3-Clause; see
`Vendor/README.md` for the local patches.
