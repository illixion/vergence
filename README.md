# mvc3d

Decode 3D Blu-ray video (H.264 MVC, "Stereo High") into ordinary left/right pictures, in Swift,
on any platform Swift runs on, Apple's included.

No consumer platform ships an MVC decoder: Apple's VideoToolbox, Android's MediaCodec and FFmpeg
all decode only the base view of a 3D Blu-ray and silently drop the second eye. mvc3d reads both
views off the disc's clip files, joins them into one MVC stream, decodes it with
[edge264](https://github.com/tvlabs/edge264) and hands back both eyes, either as planes in memory
or frame-packed (top-and-bottom or side by side) for an encoder.

## Status

Early. Working: BDAV/TS demux, pairing the base (`0x1011`) and dependent (`0x1012`) views,
decoding both with edge264, frame packing, a command line tool. Verified on a commercial 3D
Blu-ray (1080p23.976, CABAC): 8,486 consecutive stereo frames with no errors, output identical
single- and multi-threaded.

Planned: playlist (MPLS/CLPI) parsing so a disc folder or ISO is enough, reading ISO images
(UDF 2.50) directly, `CVPixelBuffer` output on Apple platforms, encoder integration.

## Speed

Apple M-series Mac, both eyes of a 1080p Blu-ray:

| threads | stereo frames/s | × realtime (23.976 fps) |
| --- | --- | --- |
| 1 | ~61 | 2.5× |
| all cores | ~426 | 17.8× |

## Command line

```sh
swift build -c release
# Clips from BDMV/STREAM: base view first, then the dependent view (usually the next number).
.build/release/mvc3d decode 00007.m2ts 00008.m2ts --threads -1 \
  | ffmpeg -f rawvideo -pix_fmt yuv420p -s 1920x2160 -r 24000/1001 -i - \
           -c:v hevc_videotoolbox -q:v 65 -tag:v hvc1 film.3D.FTAB.mov
```

`--packing sbs` for side by side, `--start N --frames N` for a range, `--output FILE`.

## Library

```swift
import MVC3D

let reader = StereoStreamReader(base: try FileByteSource(url: base),
                                dependent: try FileByteSource(url: dependent))
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
