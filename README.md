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
Blu-ray (1080p23.976, CABAC): the whole 1:53:53 feature, 163,820 stereo frames, with no errors,
read from the disc image over SMB at 194 fps; output identical single- and multi-threaded.

## Tests

`swift test` runs on synthetic data only. `scripts/fetch-conformance.sh` downloads the ITU-T MVC
conformance bitstreams (36 MB, not redistributed here);
`MVC3D_CONFORMANCE=.build/conformance/mvc swift test` then decodes them too.

Disc images (`.iso`) are read in place, without mounting, by a read-only UDF 2.50 reader
(metadata partition included) over any `ByteSource`, so a remote image can be read the same way.

Planned: an HTTP range `ByteSource`, seeking (CLPI EP maps), `CVPixelBuffer` output on Apple
platforms, encoder integration.

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

`DISC` is a folder holding `BDMV`, or a disc image (`.iso`). `--playlist NNNNN` picks another title; a clip pair
(`decode 00007.m2ts 00008.m2ts`) works too. `--packing sbs` for side by side,
`--start N --frames N` for a range, `--output FILE`.

## Library

```swift
import MVC3D

let disc = BlurayDisc(fileSystem: DirectoryFileSystem(root: discFolder))
let reader = disc.stereoReader(for: try disc.mainPlaylist()!.playlist)
let decoder = try MVCDecoder(threads: -1)
while let unit = try reader.next() {
    try decoder.decode(unit) { frame in
        // frame.left / frame.right: Y, Cb, Cr planes, valid inside this closure
    }
}
try decoder.finish { frame in /* frames held for reordering */ }
```

## License

MIT, see `LICENSE`. edge264 (vendored under `Sources/CEdge264`) is BSD-3-Clause; see
`Vendor/README.md` for the local patches.
