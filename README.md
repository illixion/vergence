# Vergence

Stereo 3D video tools in Swift:
- **Spatial video** for Apple Vision Pro from side-by-side and over-under files, with no flags:
  `vergence make` works out the layout and writes the complete metadata Photos and Files need.
- **And back:** `vergence sbs` turns a Vision Pro or iPhone recording into side-by-side video
  for a Quest, Android or YouTube, with its spatial audio.
- **3D Blu-ray**: decodes H.264 MVC ("Stereo High") into ordinary left/right pictures, on any
  platform Swift runs on, Apple's included.
- **2D to 3D** from estimated depth: `vergence convert` turns any 2D video into an AI 3D
  side-by-side copy (or spatial video) that 3D players read.

Planned: immersive video (Apple Projected Media Profile: 180°/360°, fisheye), spatial audio, and
spatial video straight from a disc.

Vergence works with decrypted discs only: a disc folder or image you have already made readable.
It contains no AACS or BD+ code.

## Layout

| Module | What | Builds on |
| --- | --- | --- |
| `VergenceCore` | Stereo layouts (`PackedStereoLayout`) and telling them from a file name or picture (`StereoLayoutDetector`), spatial-video camera metadata and presets (`SpatialCameraMetadata`, `SpatialPreset`), bitrates (`SpatialBitrate`), depth-warp settings, and the depth maths behind stable 2D-to-3D: anchoring each frame's depth to the last (`DepthAlignmentSums`, `DepthScaleShiftTracker`), the soft-knee depth budget, cut detection, shot mapping, histogram percentiles. Analysis tracks (`AnalysisTrack`: flow, disparity and depth fields keyed by media time, made on one machine for another) and the producer's half of stereo Smooth Motion (`SharedFieldAnalysis`). Reading PyTorch checkpoints without Python (`TorchCheckpoint`) and finding a pulldown's repeated frames (`RepeatedFrames`). Foundation only, no dependencies. | Apple platforms, Linux (tested), Android arm64 (builds) |
| `VergenceDisc` | UDF images, BDAV demux, playlists and clip info, MVC decoding (edge264), frame packing, PGS subtitles, AC-3 frame parsing. Foundation only. | Apple platforms, Linux (tested), Android arm64 (builds; `scripts/build-android.sh`), Windows (planned) |
| `VergenceApple` | The Apple-framework half: spatial video making and inspection (`SpatialVideoMaker`, `SpatialVideoInspector`), realtime and offline 2D-to-3D (Core ML depth, the depth converter and its cache, model management, the stereo pump and Metal warp, stereo subtitles), decoder-loss recovery, `CVPixelBuffer` output, AC-3 decoding through AudioToolbox, Smooth Motion (frame interpolation, and `StereoFlowAnalyzer`: the flow and disparity models with Core ML), 2× upscaling with Real-ESRGAN's compact network built from its weights (`FrameUpscaler`, with `StereoDetailFusion` for a stereo pair). Re-exports `VergenceCore` and `VergenceDisc`. Builds empty elsewhere. | Apple platforms (iOS/tvOS/visionOS 26, macOS 14) |
| `vergence` | The command line tool. | macOS, Linux (disc commands only, without PNG or audio output) |
| `VergenceCommandLine` | The tool's option parsing and output names, apart from the executable so its tests can import them (not a product). Foundation only. | Apple platforms, Linux (tested) |
| `coordinator/` (Rust) | `vergenced`, the analysis coordinator: queues work for Jellyfin and Stash titles in 2 s pieces, nearest a player's playhead first, checks and keeps workers' results, serves them in 10 s chunks. | macOS, Linux, Windows |
| `companion/` (Rust) | RAVECompanion pairing (six-digit code, per-device keys, TLS-PSK), byte-compatible with the Swift and C# implementations. | macOS, Linux, Windows |
| `worker/` (Rust) | `vergence-worker`, the analysis worker on ONNX Runtime (CUDA, TensorRT, DirectML, Core ML, CPU) for PCs. | Windows, Linux, macOS |
| `desktop/` (Rust, Tauri) | The Vergence desktop app: the coordinator, pairing, a worker and file conversion in one tray app. | macOS (Windows planned) |

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

## Logging

Everything logs through `VergenceLog`, written like `os.Logger` with privacy labels
(`"\(path)"` is private, `"\(count, privacy: .public)"` public). With nothing attached,
lines go to the unified log; a host can set `VergenceLog.sink` to receive each message with
its values and their privacy kept apart, to apply its own redaction.

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


## Speed

Apple M-series Mac, both eyes of a 1080p Blu-ray:

| threads | stereo frames/s | × realtime (23.976 fps) |
| --- | --- | --- |
| 1 | ~61 | 2.5× |
| all cores | ~426 | 17.8× |

## Command line

### Spatial video

```sh
swift build -c release
.build/release/vergence make "Film.3D.HSBS.mp4"     # writes Film.3D.HSBS.spatial.mov beside it
.build/release/vergence make ~/Videos/3D --output ~/Videos/Spatial   # a whole folder
.build/release/vergence inspect Film.3D.HSBS.spatial.mov
```

`make` needs no options. For each file it:
- **finds the layout** from the file's stereo metadata, else from words in its name (`SBS`,
  `HSBS`, `OU`, `TAB`, `Half-OU`, ...), else from the picture itself: the two halves of a
  stereo frame show the same scene a few pixels apart, which ordinary video doesn't. Full or
  half resolution per eye comes from the frame's shape (a 1920×1080 side-by-side frame can
  only be two squeezed eyes). It says what it found and how sure it is, and skips files that
  look 2D;
- **writes complete metadata** from a preset (`--preset generic`, the default; `iphone`;
  `avp` for Apple Vision Pro; `film`): camera baseline, field of view, disparity adjustment, projection and eyes.
  `--baseline MM`, `--hfov DEG` and `--adjust -1...1` change single values;
- **stretches half-resolution eyes** back to their display size;
- **picks the bitrate** from the eye size and frame rate (`--quality low|standard|high|max`,
  about 10 Mbps for two 1080p24 eyes at `standard`), never far above the source's own;
  `--bitrate 20` sets it;
- **keeps 10-bit and HDR** (PQ, HLG) sources 10-bit with their colour tags, keeps the source's
  timestamps, and copies the audio as it is (`--no-audio` leaves it out).

A folder is converted file by file. A finished output is skipped, so running it again carries
on where it stopped; `--force` redoes it. `--dry-run` only reports what it would do,
`--swap-eyes` fixes a file with its eyes the wrong way round, and `--layout sbs|hsbs|ou|hou`
overrides the detection. Option values can follow an `=` or the option itself, so negative
numbers work either way (`--adjust -0.02`, `--adjust=-0.02`).

`inspect` lists a file's video, colour and audio, and for spatial video its stereo metadata
and anything a player would miss. For a frame-packed file it says which layout it looks like.
Making needs macOS (Apple's MV-HEVC encoder). QuickTime and MP4 files are read directly; anything
else (`.mkv`, `.webm`, `.avi`, `.ts`, `.m2ts`, ...) needs [ffmpeg](https://ffmpeg.org) on `PATH`
(`brew install ffmpeg`, or `VERGENCE_FFMPEG=/path/to/ffmpeg`). Such a file is checked for 3D first,
reading Matroska's StereoMode and a few frames, so 2D files are skipped without copying anything.
A stereo one is remuxed into a temporary QuickTime file beside the output:
- H.264 and HEVC video copied as it is, anything else (VP9, AV1, ...) to ProRes 422 HQ;
- AAC, AC-3, E-AC-3, ALAC and MP3 audio copied, lossless audio (TrueHD, DTS-HD MA, FLAC, PCM)
  to ALAC, other lossy audio (DTS, Opus, Vorbis) to AAC;
- subtitles left out.

`inspect` reads those through ffmpeg too, on Linux included, and shows an MP4's `st3d` and
`SA3D` boxes.

### Vision Pro and iPhone recordings on a Quest

```sh
vergence sbs IMG_0664.MOV                 # writes IMG_0664.3D.FSBS.mp4 beside it
vergence sbs ~/Downloads --output ~/Quest # every spatial video in a folder
```

Players outside Apple's can't decode MV-HEVC's second view; a Quest, an Android phone or
YouTube play one picture with both eyes in it. `sbs` decodes both views and writes HEVC in MP4,
the left eye on the left:
- each eye at full resolution (a Vision Pro's 2200×2200 eyes make a 4400×2200 picture, which a
  Quest 3 decodes); `--half` squeezes each eye to half width, `--max-eye 1920` caps their size
  for older decoders;
- colour converted from Display P3 to BT.709, which Android players assume (HDR stays as it is);
- the stereo AAC track copied, and the ambisonic track (a Vision Pro's APAC, which only Apple
  decodes) re-encoded as 4-channel AAC, the form YouTube's spatial audio takes;
- Google's Spherical Video V2 `st3d` box (left-right) and Spatial Audio `SA3D` box, which
  non-Apple players read, and a name most players and media servers recognise
  (`.3D.FSBS.mp4`, `.3D.HSBS.mp4` with `--half`).

`vergence make` turns such a file back into spatial video, reading the layout from its `st3d`.

A server can stream it instead of converting whole files first:

```sh
vergence sbs IMG_0664.MOV --info                          # JSON: spatial or not, size, segment count
vergence sbs IMG_0664.MOV --segments cache/ --first 30 --last 59
```

`--segments` writes the same picture as HLS fragmented-MP4 segments, `init.mp4` and `N.m4s`, on
a fixed grid (`--segment-seconds`, default 4) from the start, each renamed into place the moment
it is complete. Every segment opens with a keyframe and carries the video's own times, so a
server can make any stretch on demand (where the viewer seeks) and segments from different runs
play as one stream; `index.m3u8` lists the whole grid. The stereo AAC is copied; the ambisonic
track is left out, since an encoder restarted at every seek would click where runs join. The
Convolution Jellyfin plugin streams spatial video to its Quest app this way.

### 2D to 3D

```sh
vergence convert Film.mkv                         # writes Film.3D.FSBS.mp4 beside it
vergence convert Film.mkv --layout hsbs           # Film.3D.HSBS.mp4: half-width eyes, for TVs
vergence convert Film.mkv --sample 10             # a 10 s piece from the middle: Film.sample.3D.FSBS.mp4
vergence convert ~/Videos --output ~/Videos/3D    # a whole folder
```

`convert` runs the offline AI 3D path Convolution's Pre-Process uses, to a file instead of a
cache, in two passes:
1. **Depth** for every frame with Depth Anything V2 Small (Core ML), then the pre-processed
   refinement: a joint-bilateral filter on the frame's own edges, a ±2-frame window that stops at
   scene cuts, and each shot's depth scale locked by anchoring on the pixels that didn't change
   (`DepthConverter`).
2. **Picture:** each frame decoded again with its depth, both eyes drawn by the depth-displaced
   mesh as cached playback draws them (with the soft-knee depth budget), packed, and encoded as
   HEVC.

The output:
- **Layout** by `--layout`: `fsbs` (the default: full side by side, `.3D.FSBS.mp4`), `hsbs`
  (`.3D.HSBS.mp4`), `tab` (full top and bottom, `.3D.FTAB.mp4`), `htab` (`.3D.HTAB.mp4`), or
  `spatial`: MV-HEVC spatial video made from a full side-by-side intermediate the way `make`
  makes it (`.spatial.mov`; `--preset`, `--baseline`, `--hfov`, `--adjust` apply). Google's `st3d`
  box says the layout to players that read it.
- **Depth:** Convolution's Subtle settings unless changed: `--strength` (the largest per-eye shift,
  a share of the frame's width, 0.008) and `--convergence` (the depth on the screen plane, 1: the
  nearest thing in each shot, so nothing comes out through the frame).
- **Colour:** 10-bit sources and HDR (PQ, HLG) stay 10-bit HEVC with their colour tags, mastering
  display and light levels. Depth runs on an 8-bit copy converted to BT.709 for the model, as
  in Convolution's Pre-Process (not checked on real HDR films yet).
- **Audio:** AAC, AC-3, E-AC-3, ALAC and MP3 tracks copied as they are. Other files are remuxed
  with ffmpeg first, as `make` does (TrueHD and DTS-HD MA to ALAC, DTS to AAC). Subtitles are left
  out; nothing is burned in.
- **A part of it:** `--start S` (seconds) and `--frames N`, or `--sample SECONDS`.
- `--quality`, `--bitrate`, `--max-eye`, `--no-audio`, `--dry-run` and `--force` work as for `make`.
  A finished output is skipped. The depth goes into a hidden folder beside the output while
  converting, so a run that stops starts again from its finished depth.
- `--depth-track FILE.vgat` also writes the depth as an analysis track (`depth` records, one per
  frame, at the model's resolution), so the same depth can serve a live player.

The model is the only one whose licence has been checked (Apache-2.0): `--model PATH` (an
`.mlpackage` or `.mlmodelc`) or `VERGENCE_DEPTH_MODEL` names it, else the one the apps install in
`~/Library/Application Support/DepthModels`; `--download-model` fetches it there from
[Apple's Core ML repository](https://huggingface.co/apple/coreml-depth-anything-v2-small) (about
50 MB). A tool built with `swift build` compiles the warp's Metal shaders from
`Sources/VergenceApple/Stereo/StereoShaders.metal` on its first run (SwiftPM builds no Metal
library); `VERGENCE_SHADERS` points elsewhere.

On an M2 Max a 1080p24 video converts at about 30 frames a second, 1.2× realtime, so a two-hour
film takes about 100 minutes: depth runs at 30-35 frames a second (the Neural Engine's model is
nearly all of it) and the picture at about 180.

`VERGENCE_PROGRESS=1` makes `convert` print its progress as lines (`  depth  42%`) when stderr
isn't a terminal, for a caller reading a pipe.

#### On Windows and Linux: `vergence-convert`

`convert/` is the same command for a PC: ONNX Runtime runs the depth model (CUDA, TensorRT or
DirectML on Windows, Core ML or CPU on a Mac), wgpu draws the eyes (Direct3D 12 or Vulkan), and an
ffmpeg executable decodes and encodes (NVENC, then AMF, Quick Sync or x265). It takes the same flags
and writes the same files and stderr lines as `vergence convert`, so the desktop app's convert queue
runs either; `--layout spatial` is Mac only (no MV-HEVC encoder exists on Windows yet). The contract,
including the environment it reads, is at the top of `convert/src/main.rs`.

```powershell
cargo build --release -p vergence-convert --features cuda     # ORT_CUDA_VERSION=13: CUDA 13 + cuDNN 9 DLLs on PATH
$env:VERGENCE_FFMPEG = "C:\ffmpeg\bin\ffmpeg.exe"              # an LGPL build with hevc_nvenc; ffprobe beside it
target\release\vergence-convert.exe Film.mkv --download-model  # Film.3D.FSBS.mp4 beside it
```

- **Model:** Depth Anything V2 Small as ONNX. `--download-model` fetches the Apache-2.0 export from
  [onnx-community](https://huggingface.co/onnx-community/depth-anything-v2-small) (pinned and
  checksummed) into `%LOCALAPPDATA%\Vergence\DepthModels`; `scripts/export-depth-onnx.py` makes the
  same file from the original weights and checks it against the Core ML model (correlation 0.99997
  on a film frame, the largest difference under 1 % of the range).
- **Depth:** the pre-processed pipeline ported from VergenceCore and VergenceApple: the same
  letterboxed model input, robust ranges, joint-bilateral refinement and guided ×2 upsample (as
  WGSL compute), Stable Depth anchoring, cut detection, the ±2-frame window and the shot-locked
  display mapping. The maths is Rust (`convert/src/maths.rs`), tested against fixtures VergenceCore
  itself writes (`scripts/make-convert-fixtures.sh`). One deliberate difference: depth is made in one
  sweep at every frame rate, where the Mac sweeps 48 fps and faster sources twice to make its cache
  playable sooner.
- **Picture:** the dense depth-displaced mesh with the soft-knee budget and edge feather, 4×
  multisampled, drawn on the YCbCr code values (HDR stays PQ or HLG; 10-bit sources are encoded
  10-bit with their colour tags). HDR10 mastering metadata is copied with x265 only; NVENC output
  keeps the tags without it.
- **Audio:** MP4's own codecs copied; TrueHD, DTS-HD MA, FLAC and PCM to ALAC; other lossy audio to
  AAC, as the Mac's remux does. The output is always MP4.
- **Parity:** `scripts/check-convert.sh CLIP MODEL.onnx` runs a clip through both and compares each
  eye. On a 10 s 1080p film clip the copies agree at 48 dB PSNR (luma) and SSIM 0.984 per eye (SDR
  full side by side), 50.7 dB and 0.991 (HDR10), 43.7–45.4 dB and 0.982–0.988 (half side by side);
  each tool's own copy is about 44 dB from its source, so the two differ by less than either loses to
  its encoder. The tolerance is 40 dB and SSIM 0.97: the depth model's output is the same within
  float16, but anchoring and cut decisions sit on thresholds (a fit needs 15 % of the pixels to hold
  still), and a frame that lands on the other side of one gives its shot a different depth scale
  until the next cut, which moves pixels by up to the strength (0.8 % of the width) but never
  reshapes the depth. Built on Windows and run on CUDA, the same clips agree with the Mac's copies
  at 49.1 dB and SSIM 0.985 (SDR), 52.2 dB and 0.992 (HDR10).
- **Speed** on an RTX 3080 (CUDA, NVENC; 10 s clips, start-up included):

  | Source | Depth | Picture | Overall | VRAM |
  | --- | --- | --- | --- | --- |
  | 1080p24 SDR | 68 fps | 110 fps | 33 fps | 1.0 GB |
  | 1080p24 HDR10 | 61 fps | 74 fps | 28 fps | 1.2 GB |
  | 4K24 SDR | 54 fps | 31 fps | 18 fps | 1.8 GB |
  | 4K24 HDR10 | 41 fps | 20 fps | 12 fps | 2.7 GB |

  On a long film the start-up stops counting: a two-hour 1080p film takes a little over an hour,
  a 4K HDR one about three and a half. DirectML on the same card runs the depth at 38 fps.
- **Spatial video:** no Windows encoder makes MV-HEVC that this can use yet. NVIDIA's Video Codec
  SDK 13 encodes two-view MV-HEVC through its own API, but ffmpeg's `hevc_nvenc` exposes none of it,
  and x265 4's `--multiview-config` needs its own (GPL) executable and an Apple spatial-metadata
  muxer on top; either would be new work.

#### Upscaling: `--upscale 2x`

```sh
vergence convert "Film sbs.mkv" --upscale 2x --download-model   # a 3D film: each eye 2×, Film sbs.2x.3D.FSBS.mp4
vergence convert Film.mkv --upscale 2x                          # 2D: upscaled, then AI 3D as above
vergence convert Film.mkv --upscale 2x --layout 2d              # 2D: upscaled only, Film.2x.mp4
```

Both converters take the same flags (`vergence-convert` on Windows and Linux too). The network is
Real-ESRGAN's compact general model, `realesr-general-x4v3` (BSD-3-Clause, 4.9 MB), run at 4× and
averaged to 2×. Vergence ships no weights and no converted model: `--download-model` fetches the
authors' own `.pth` from [github.com/xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
(checked by SHA-256) into `Vergence/UpscaleModels` beside the depth models, and each converter builds
the network from it, without Python: the Mac writes the Core ML model itself and compiles it once per
size, splitting each batch between the Neural Engine and the GPU; the PC writes the ONNX graph for
ONNX Runtime. `--upscale-model PATH` (or `VERGENCE_UPSCALE_MODEL`) takes another compact Real-ESRGAN
`.pth`; `realesr-animevideov3.pth` from the same release is about twice as fast and smoother.

- **A 2D source** is upscaled in the picture pass and then warped from depth as before (depth is
  still made at the source's size), or with `--layout 2d` only upscaled.
- **A side-by-side or top-and-bottom source** (declared by the file, named so, or seen in its picture;
  `--input-layout sbs|hsbs|tab|htab|2d` to say) has each eye upscaled, with no depth,
  and packed as `--layout` asks (full or half side by side, top and bottom, or spatial video on the
  Mac). A half-width eye is stretched back to its real shape first. Then the eyes' fine detail is
  brought into agreement: an upscaler run on each eye alone invents texture in each on its own, so
  where block matching on the source finds the same point in both eyes (SAD on luma, a left-right
  check), each eye's detail band is moved halfway towards the other's. The coarse picture, and so
  the depth, is untouched.
- **Names:** `VIDEO.2x.3D.FSBS.mp4` (and the other layouts), `VIDEO.2x.mp4` for `--layout 2d`.
- HDR, audio, `--quality`, `--max-eye` (applied to the upscaled eye), `--sample` and the rest work as
  without it. The bitrate is worked out for the bigger picture; `--quality high` keeps more of the
  new detail.

**Old SD sources: `--antialias` and `--ai-hdr`.**

```sh
vergence convert "Film sbs.mkv" --upscale 2x --antialias --ai-hdr --gain-model ~/Models/GainMapGMNet.mlpackage
```

- `--antialias` (both converters) turns stair-stepped edges back into lines. A 2003 SD CG film
  after NTSC deinterlacing has diagonals in steps a line or two tall, which an upscaler sharpens into
  crisp sawtooth edges. The finished picture's structure tensor (luma blurred to σ 2 pixels, so a
  stair's tread and riser read as the slope they stand for; products blurred to σ 8) says at every
  pixel which way an edge runs and how straight it is; along strong, straight edges the picture is
  averaged along the edge (σ 12 over ±24 pixels), never across it, and texture, which isn't straight
  over that length, stays. It runs on each eye after upscaling, last before AI HDR and the encoder
  (`EdgeAntialias` on the Mac's GPU, about 5 ms a 2880×800 frame; `antialias.rs` on the PC's CPU,
  beside the network, without `--upscale` on the picture before the warp). A line drawn in 4-pixel
  stairs ends 0.12 px RMS off its line (1.14 before); on five aliased spots of that film the sawtooth
  is gone, a staircase measure (luma's second difference along the edge, through the encoder) drops
  22–41 %, texture elsewhere within 1–9 %. FXAA-style passes, made for steps a pixel tall, took 3–5 %
  off; the earlier ffmpeg pre-pass (both fields re-made by `estdif`) at most 30 % for 4–15 % of the texture;
  Real-ESRGAN's anime model instead of the general one, −21 to +12 %, for 17–32 % of the texture.
- `--ai-hdr` (Mac for now) makes an SDR source HDR the way the `ai-hdr` workload and the players
  do: the worker's own producer makes GMNet's gain grids (keyframes every half second and at cuts,
  eased within shots, the 16×9×8 grid, highlights spread from the mid-tones), from the left eye of
  a 3D source, and `GainGridPass`, the players' shader, applies them, holding the frame's average gain
  to 2.5×. On film GMNet's gain is close to one curve over luma (mid-tones 1.3–1.8×, the top 2–3×),
  so a light rose only about 1.5× over a lit wall; a grid therefore keeps only the gain above each
  cell's mid-tone gain, doubled up to GMNet's 5× (`GainGrid.expandingHighlights`): mid-tones stay as
  SDR, highlights reach 800–1,000 nits (per-frame p99.9 685 → 960 nits median over two minutes of the
  3D film, the frame's mean unchanged). The output is 10-bit
  HEVC in BT.2020 PQ with SDR white at 203 nits, HDR10 mastering display (P3 D65, 1,000 nits) and
  light levels (MaxCLL 1,015, MaxFALL 507), both eyes with one grid and one scale, named
  `VIDEO.hdr.3D.FSBS.mp4`. GMNet is Convolution's Core ML conversion, a worker's `gainModel`:
  `--gain-model PATH` or `VERGENCE_GAIN_MODEL`. The grids take a pass of their own (about 35 frames
  a second); applying them costs the picture pass nothing measurable. A 10-minute piece of the 3D
  film with `--upscale 2x --ai-hdr` and the earlier antialiasing took 40.6 minutes on the M2 Max:
  that antialiased intermediate 4.3 min, the grids 9.2 min (34 fps), the upscaled HDR picture 26 min
  (9.6 fps, as fast as without HDR); the antialiasing now costs about 5 % of the picture pass instead.

**Repeated frames.** Every `convert` now checks whether a video repeats one frame in five: film made
30 fps by doubling a frame in four and then retimed, often to an odd rate after a remux (a 3D film
read as 31.24 fps: NTSC's 29.97 sped up 25/23.976 to fit PAL audio). Those repeats judder in a headset
and cost a fifth more work. It drops one frame in five, the one most like its predecessor (ffmpeg's
`decimate` on the PC), so the rest keep their place against the audio, and writes four fifths of the
rate (31.24 → 24.99 fps). `--keep-repeated-frames` keeps them.

**What was measured** on that film (720×400 per eye, H.264 at 7.8 Mbps), two 10 s clips (a dialogue
scene and a dark, fast one), each eye 2×, against Lanczos: the detail added (mean high-pass energy),
the left/right disagreement of that detail after warping the right eye onto the left by optical flow
(DIS, independent of the converter's matcher), and the frame-to-frame flicker of it along the flow.

| Method (licence) | Detail | L/R disagreement | Flicker | Per eye (M2 Max GPU, PyTorch) |
| --- | --- | --- | --- | --- |
| Lanczos | 1 | 1 | 1 | |
| realesr-animevideov3 (BSD-3) | 1.13–1.26 | 0.89–0.98 | 0.95–1.00 | 49 ms |
| **realesr-general-x4v3 (BSD-3)** | 1.36–1.48 | 1.23–1.29 | 1.23–1.36 | 83 ms |
| **the same, eyes' detail fused** (what `convert` does) | 1.31–1.44 | 0.92–1.20 | 1.16–1.34 | a few ms |
| RealESRGAN_x2plus, RRDB (BSD-3) | 1.23–1.28 | 1.20–1.28 | 1.39–1.71 | 323 ms |
| RealESRGAN_x4plus, RRDB (BSD-3) | 1.30–1.42 | 1.34–1.39 | 1.64–1.69 | 1.0 s |
| RealBasicVSR, a video model (Apache-2.0) | 1.54–1.84 | 1.51–2.10 | 1.46–1.99 | 0.8 s |

The RRDB networks are slower and flicker more than the compact one; RealBasicVSR invents the most
texture but differently in each eye and from frame to frame on this compressed source. The compact
general network adds the most detail for its cost, and relative to the detail it adds it disagrees
less between the eyes than Lanczos does; the fusion takes that down to Lanczos's level or below
for a few percent of the detail. A stereo network (NAFSSR: MIT code, weights trained on Flickr1024
with clean bicubic downscaling) wasn't tried: it isn't trained for compressed video. Not built yet:
the same detail blend along motion (flow-warped from the frame before) cut the flicker by a third in
the experiment, but needs a flow model per frame.

**Speed.** In `convert` on the M2 Max both eyes of a 720×400 pair take about 100 ms (one eye on the
Neural Engine, the other on the GPU, at once): 9.3 frames a second for 3D, 14 for a 2D picture
(cut in two between them) with the warp and HEVC encode included. The whole 81-minute 3D film
(121,493 frames once its repeats were dropped) took 3 h 36 min, 9.4 frames a second. On a Windows PC with an
RTX 3080 and a 6-core CPU the same 3D clips go at 6.9 frames a second with DirectML and 6.5 with CUDA
(the network in float16 about 85–120 ms a pair, the eyes' fusion on the CPU about 140 ms beside it,
which sets the pace), so a feature takes about five hours there. The two converters' copies agree at
about 40–43 dB PSNR (luma) frame for frame; where the Mac and ffmpeg's `decimate` drop a different
frame of a five, the frames there differ.

### Spatial audio

```sh
vergence audio IMG_0664.MOV -o room.opus     # Vision Pro or iPhone recording -> ambisonic Opus
vergence audio IMG_0664.MOV -o room.wav      # or 32-bit float AmbiX WAV
vergence audio IMG_0664.MOV --track 1 -o stereo.mka
```

A Vision Pro (and an iPhone recording spatial audio) stores two audio tracks: stereo AAC and
APAC, Apple's spatial audio codec, which nothing outside Apple's frameworks decodes. The APAC
track holds first-order ambisonics (four channels, ACN order, SN3D: AmbiX). `vergence audio`
decodes it and writes Opus with ambisonic channel mapping (RFC 8486, the format YouTube's VR
players use; needs ffmpeg) in `.opus`, `.ogg`, `.mka`, `.webm` or `.mp4`, or a float WAV. N3D
ambisonics are converted to SN3D. `--bitrate` is in kbps, 64 per channel by default (256 for
first order). Decoding a 7 s Vision Pro clip and re-encoding it to Opus kept every channel at 0.99
correlation with the decoded APAC. MP4 output carries no `SA3D` box (ffmpeg doesn't write one);
players that need it should get `.mka`, `.webm` or `.ogg`.

### 3D Blu-ray

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

### Smooth Motion analysis tracks

`vergence analyze` computes what a headset needs to make in-between frames for both eyes of a
3D film without running a network itself: for every pair of real frames the left eye's motion
(RIFE 4.25 lite at 480×288), and for every frame the right eye's disparity to the left
(NeuFlow v2 at 512×288), stored at a quarter of that size, row-delta coded and deflated, about
25 KB a frame. The design and its measurements are in Convolution's
`docs/plans/blu-ray-3d-smooth-motion.md`, and the models are converted by its
`scripts/convert-interp-model.py` (RIFE `--size 480x288 --scales 8 4 2 1 1`; NeuFlow
`--size 512x288 --iters 4 --one-way --precision fp32`, since the fp16 conversion overflows to
NaN on a Mac GPU and runs 100 ms on its Neural Engine).

```sh
.build/release/vergence analyze /Volumes/DISC --models MODELS --output film.vgat
ffmpeg -i clip.mkv -f rawvideo -pix_fmt yuv420p - \
  | .build/release/vergence analyze --raw 1920x1080 - --models MODELS -o clip.vgat
```

`--raw WxH` reads packed top/bottom I420 (left on top, one eye's size given) at `--rate`
(default 24000/1001), which is what `decode` writes and what ffmpeg makes of any stereo file.
On an M2 Max it runs at about 35 frames a second (RIFE on the Neural Engine and NeuFlow on the
GPU, side by side), faster than the film plays. `VERGENCE_COMPUTE_UNITS=ane|gpu|cpu|all` overrides
where Core ML runs them. A disc's track is timed on the playlist's own clock (the first clip's
in-time plus each frame's PTS), the clock players show.

A side-by-side file is stacked first, at the size its eyes are stored (a half-SBS 1080p film's
eyes are 960×1080; the player scales fields to whatever its eyes are). If ffmpeg fails every
filter with "File exists" (-17) on such a file, its H.264 frame-packing SEI is the cause (seen
with ffmpeg 8.1.2): copy the video without SEI first.

```sh
ffmpeg -i film.mkv -map 0:v:0 -bsf:v filter_units=remove_types=6 -c:v copy -f matroska - \
  | ffmpeg -i - -filter_complex "[0:v:0]split[a][b];[a]crop=iw/2:ih:0:0[l];[b]crop=iw/2:ih:iw/2:0[r];[l][r]vstack[out]" \
      -map "[out]" -f rawvideo -pix_fmt yuv420p - \
  | .build/release/vergence analyze --raw 960x1080 - --models MODELS -o film.vgat
```

The player's half is `SharedFieldInterpolator` (VergenceApple): one Metal pass that makes both
in-between eyes of a packed pair from a pair's records, plugged into `InterpolatingFrameSource`
with a `SharedFieldProvider` (`AnalysisTrackFieldProvider` reads a track file). `vergence
interpolate` runs that same pass on a Mac, packed top/bottom BGRA in and in-between frames out,
for checking it against the PyTorch reference; about 2 ms a 1080p pair on an M2 Max.

```sh
ffmpeg -i clip.mkv -f rawvideo -pix_fmt bgra - \
  | .build/release/vergence interpolate --raw 1920x1080 - --fields clip.vgat -o clip-mid.bgra
```

### Analysis workers

Other machines can make a title's analysis while it plays, so a headset only presents (the design
is Convolution's `docs/plans/analysis-workers.md`). `vergenced` (`coordinator/`) knows each
title from Jellyfin or Stash and hands it out in 2 s pieces: first the 30 s ahead of whoever is
watching, then the rest of that title, then the library. Workers read the media from the server
themselves, with their own credentials, and upload only the records. Players read 10 s chunks
while they fill; the Jellyfin plugin's `SmoothMotion/` routes are served too.

```sh
cargo build --release -p vergence-coordinator
target/release/vergenced token add vision-pro --grants play --config vergenced.json   # prints a token once
target/release/vergenced token add mac --grants work --config vergenced.json
target/release/vergenced serve --config vergenced.json       # config format: coordinator/src/config.rs
target/release/vergenced pair --config vergenced.json        # pair a headset: compare six digits, answer y
.build/release/vergence worker --config worker.json          # a Mac (Core ML); format: Sources/VergenceCLI/Worker.swift
target/release/vergence-worker --config worker.json          # a PC (ONNX Runtime), same format
```

Two ways in, to the same API. **Paired devices** (a headset paired with `vergenced pair` or the
desktop app's Headsets tab) use TLS 1.2 with an external PSK on port 8421, keyed from their
pairing, with nothing secret on the wire and no token to copy; the spec is in
`coordinator/src/paired.rs` (identity `VergenceAPI/v1/<device id>`, PSK
HKDF-SHA256(pairing token, `Vergence-API-PSK-v1`, `psk`), then plain HTTP/1.1). **Tokens** over
plain HTTP on port 8420 are for scripts, workers set up by hand and the Jellyfin plugin
(`vergenced token add`); a paired device may use a token derived from its pairing there too.

Workloads (`coordinator/src/queue.rs`; a player names one when it reports where it plays):

| Workload | Titles | Records | Mac worker's model |
| --- | --- | --- | --- |
| `smooth-motion-3d` | frame-packed 3D | `flowPair` (left eye), `disparity` (right) | RIFE 4.25 lite 480×288, NeuFlow v2 512×288 |
| `smooth-motion-2d` | 2D, flat or AI 3D | `flowPair` (whole picture, 240×144 stored) | RIFE 4.25 lite 960×576 on the GPU, about 40 fps for 720p |
| `ai-3d-depth` | 2D, played as AI 3D | `depth` (the picture's shot-space depth, model resolution) | the offline depth pipeline with `depthModel`, e.g. Depth Anything V2 Small at 770×574: about 9 fps |
| `ai-hdr` | 2D | `gainMap` (a 16×9×8 grid of log2 gain, `GainGrid`, about 1 KB) | GMNet (`gainModel`, Convolution's conversion) on keyframes every 0.5 s and at cuts, eased between: about 30 fps |

`scripts/check-worker.sh … smooth-motion-2d` checks any workload but the first against the
worker's own `--reference` run over the whole file in one decode. A depth piece is converted with
a second of the title before it, so the shot anchoring has settled; against a whole-file
conversion its records differ by 0.005 on average (0.05 at most) in shot-space units. A player
warps a depth track's maps (`AnalysisDepthSource`) and runs its own model only for frames the
track doesn't hold yet. Vergence ships no weights and is built around Depth Anything V2 Small
(Apache-2.0). `vergence convert` and `vergence worker` refuse the larger models whose weights
restrict their use (Depth Anything V2 Base and Large, Video Depth Anything Base and Large:
CC-BY-NC-4.0; Depth Pro: Apple's research licence) unless built with
`swift build -c release -Xswiftc -DVERGENCE_NONCOMMERCIAL_MODELS`, and then say the licence; a
worker's profile names it, so such titles never mix with Small's. On an M2 Max, Base resolved
fine structure Small missed (fibres, thin stems) at about a third of Small's speed through the
offline pipeline. The Rust tools follow the same rule behind a cargo feature, off by default:
`cargo build --release -p vergence-convert -p vergence-worker --features
vergence-convert/noncommercial-models,vergence-worker/noncommercial-models`; such a build of
`vergence-convert` also downloads them (`--download-model --variant base|large`, the
onnx-community exports, pinned and checked), and the desktop app built with its own
`noncommercial-models` feature offers them on Windows only after showing the licence. An AI HDR piece decodes one keyframe interval either side and matches a whole-file run
exactly. A player shows each frame's grid (`GainGridPlayback`: ramped in over a second after it
starts and after a seek, held rather than dropped while a piece is missing, refreshed every half
second but still snapped at cuts while the device is short of time), its `GainGridPass` holding the
frame's average gain to a cap: AI 3D's pump warps in SDR and writes each eye as BT.2020 PQ into its
10-bit eyes; Convolution's flat player does the lookup in its final pass into an EDR drawable.

A Mac worker gives the same records as `vergence analyze` over the whole file (disparity
identical, flow within Core ML's run-to-run noise; `scripts/check-worker.sh`), at about 37 fps
for 1080p 3D on an M2 Max. The ONNX worker agrees on disparity; its RIFE flow differs from Core
ML's float16 run, so its titles use their own profile (`scripts/check-onnx-worker.sh`). The
desktop app (`desktop/`, `cargo run -p vergence-desktop`; for Windows `desktop/scripts/package-windows.sh`) runs the coordinator, pairs headsets,
runs a worker and converts files to AI 3D side by side, from one tray icon.

#### The ONNX worker on Windows

Built and run on Windows 10 with an RTX 3080 (rustup's `stable-x86_64-pc-windows-msvc` and the Visual
Studio 2022 Build Tools):

```powershell
$env:ORT_CUDA_VERSION = "13"
cargo build --release -p vergence-worker --features cuda,tensorrt
target\release\vergence-worker.exe bench --models onnx-models --ep cuda
target\release\vergence-worker.exe --config worker.json
```

`ort` downloads ONNX Runtime 1.31 built for CUDA 13 with TensorRT and DirectML, and puts its DLLs
(DirectML's included) beside the executable. CUDA and TensorRT need NVIDIA's runtime DLLs on `PATH`
or beside it, nothing installed system-wide; the driver must support CUDA 13. NVIDIA's
redistributable wheels hold them (a wheel is a zip): `nvidia-cuda-runtime`, `nvidia-cublas` (13.x),
`nvidia-cudnn-cu13`, `nvidia-cufft`, `nvidia-curand` for CUDA, plus `tensorrt-cu13-libs` 10.x (from
`pypi.nvidia.com`; `nvinfer_10.dll`, `nvonnxparser_10.dll` and the builder resource for the GPU's
architecture) for TensorRT. DirectML needs nothing more. `--ep` (or `executionProvider`) picks one;
`auto` tries CUDA, then DirectML.

1080p 3D Smooth Motion (RIFE 480×288 and NeuFlow 512×288) on an RTX 3080: the models alone
(`bench`, both at once), and whole pieces from a coordinator, ffmpeg's seek and decode included:

| Provider | Models | Pieces | GPU memory | First load |
| --- | --- | --- | --- | --- |
| TensorRT (fp32) | 103–111 fps | 55 fps | about 0.45–0.9 GB | about 100 s to build its engines, then cached in `model-cache` beside the config |
| DirectML | 94 fps | 52 fps | about 0.5 GB | 1.4 s; NeuFlow runs without DirectML's graph fusion, which rejects it |
| CUDA | 81–83 fps | 47 fps | about 0.75 GB | 3 s |
| CPU (Ryzen 5 5600X) | 3.7 fps | 3.8 fps | | |

All three GPU providers give the same records within one or two storage steps (0.06 px) of each
other, and the Core ML worker's disparity within one step (0.03 px); RIFE's flow differs from
Core ML's float16 run by up to 1 px (p99 0.06 px), as the Mac's ONNX CPU run does
(`scripts/check-onnx-worker.sh` on a Mac, then the PC's chunks against its `KEEP_COREML` track with
`scripts/compare-analysis-tracks.py`). So PCs share one profile, and Core ML titles keep theirs.

The 2D workloads run on the PC worker too (`workloads`,
`depthModel`, `depthSize` and `gainModel` in its config; `worker/src/config.rs`):

| Workload | Model (export) | RTX 3080, a 720p title's pieces | Against the Core ML worker |
| --- | --- | --- | --- |
| `smooth-motion-2d` | RIFE 4.25 lite 960×576 (`export-onnx-models.py --only rife-2d`); 89 fps alone on TensorRT, 71 DirectML, 60 CUDA | 58 fps (TensorRT) | flow p99 0.06 px, mean 0.003; up to 3 px where Core ML's float16 run drifts |
| `ai-3d-depth` | Depth Anything V2 Small at 770×574 (any free-size export, e.g. `vergence-convert --download-model`'s), through `vergence-convert`'s depth pipeline | 13 fps of title (CUDA), each piece with its second of context; DirectML about a third of that | mean 0.012, max 0.09 (shot-space depth) |
| `ai-hdr` | GMNet (`export-gmnet-onnx.py`, MIT) on keyframes | 59 fps (CUDA) | log2 gain within 0.01 |

2D flow and AI HDR keep a profile per engine family. Depth is shared: Depth Anything V2 Small and
Base have one profile name on both workers (`Depth Anything V2 Small 770x574 offline-v5
context-1s`), so a Mac and a PC split a title's pieces. Their records differ by about 0.012 on
average and 0.04 at p99 (Base 0.006 and 0.025), against 0.005 and 0.023 between a Core ML piece
and a whole-file run: a quarter of a pixel of parallax on average at Convolution's Subtle strength.
TensorRT builds an engine per input size, so the two free-size models (Depth Anything, GMNet) run on
CUDA when TensorRT is chosen. `scripts/check-onnx-workload.sh` runs both workers on one title and
prints the differences.

A shared PC stays the owner's: the worker runs at below-normal priority, and stands aside (no
claims, its piece given back, its models unloaded so the GPU's memory goes back) while a game, an
app launched through a Sunshine stream, SteamVR or another program keeps the GPU busy, or while its
memory runs short; it goes back to work after 30 s of quiet. A stream of the desktop alone doesn't
stop it. What counts is in `worker/src/yielding.rs`.

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

For a player, `VergenceApple` has frame sources that sit between a decoder's output
(`AVPlayerFrameSource`, or any `LookaheadFrameSource`) and the screen, shown by
`InterpolatedFramePresenter` in a flat player or by `StereoPump` in a stereo one:
`InterpolatingFrameSource` doubles 24-30 and 50-60 fps (VideoToolbox's low-latency
interpolator, or the GPU one, `FlowMidFrameInterpolator`), and `UpscalingFrameSource`
plays a picture of 720p or less sharper through VideoToolbox's low-latency super
resolution (`VTRealtimeUpscaler`), up to 1080p: 1.5 ms a 540p frame on an M2 Max,
where `convert --upscale`'s Real-ESRGAN takes tens of milliseconds an eye and adds
more detail.
`RealtimeUpscalerBenchmark` times it on a device. Put the upscaler under the
interpolator, so the in-between frames come from the sharper frames.

## Stereo boundary experiment

`DepthRefinement.separatesBoundaries` opts into a disconnected mesh and explicit
hole fill; it defaults to false. Triangles spanning a normalized depth jump
greater than 0.08 and more than one output pixel of differential disparity are
drawn as separate near/far surfaces, each clipped to its part of the depth map.
Continuous triangles are drawn normally. Covered surfaces retain depth testing. Holes reflect a nearby covered
background region, checking its depth to avoid copying a nearer subject. A
vertical fallback handles horizontal rails whose entire row is uncovered.
Searches are bounded by disparity plus grid pitch and read only the original
surface pass, so filling is deterministic and uses no stale frame history.

`showsBoundaryHoles` highlights uncovered pixels in magenta. The prototype uses 4× MSAA where supported, resolving surface colour/depth and
coverage separately before background fill. This avoids averaging depth into
coverage, preserves partial foreground coverage, and reflects only fully covered
background samples. It still needs temporal/background reconstruction and device
quality/performance validation before becoming the default. Sub-grid hair,
soft/incorrect depth boundaries and transparent pixels remain limitations.

On-device comparison on the paused Movie A corral shot (2026-10-13) found noisy
outlines, particularly the vertical railing. Keep it opt-in; passing the GPU
tests does not establish movie quality. A separate Core ML row-stride fix
removed broad distortion with the custom 686×308 model: its logical rows have
686 elements but Vision returns a physical stride of 704. All multi-array depth
reads honor the reported strides, including still images and raw cache output.
The original mesh's heuristic background edge fill also defaults to off: an
exact paused-frame check and headset comparison showed it damaging the horse
rider's jacket outline. It remains available through the debug switch.
Luma-based outline snapping defaults to off too: Movie B headset comparisons
showed improved face and subject edges without it. Joint-bilateral refinement
and shot stabilization still run.

`swift test --filter StereoBoundaryWarpTests` compiles the actual Metal source at
runtime and executes the GPU passes, including BGRA output. It checks both eyes,
vertical and horizontal rails, smooth slopes, zero disparity, mirror and depth
UV remapping. These tests do not silently skip because a SwiftPM resource bundle
is missing. A host app can capture the same movie frames before and after a change to compare them.

## Layered silhouettes (experiment)

`DepthRefinement.layersSilhouettes` opts the visionOS video pump into a local
near/far decomposition around trustworthy depth discontinuities. It defaults to
false. Interior seeds extend each side's colour and depth; colour-line coverage
and a premultiplied residual preserve the soft source outline. Each eye ray-casts
the layers and blends them over the connected mesh in its existing MSAA pass.
Support, contrast, depth variance and separation gate the patch; ambiguous
regions retain the mesh. The disconnected-boundary experiment takes precedence.

Composition is capped at 2048 pixels wide. This is a local two-layer estimate,
not general scene reconstruction: narrow background strips can take the next
surface's colour, thin structures fall back to the mesh, and twice the normal
strength can exceed the trusted band's reach. Only SDR captures at the normal
strength have been reviewed. Headset appearance, motion, HDR and frame cost
remain unvalidated; keep this opt-in.

`swift test --filter StereoLayeredWarpTests` checks silhouette width, position,
both orientations, the composition cap, inert cases and mirror symmetry using
the real Metal shaders. The macOS replay lab accepts lossless depth captures:

```sh
VERGENCE_LAYER_LAB=/path/to/captures VERGENCE_LAYER_OUTPUT=/path/to/new-output \
VERGENCE_LAYER_MESH=1 swift test --filter StereoLayeredLabTests/replayCaptures
```

Each capture contains `source.png`, `depth.f32` and `depth.json`. The Hypnos
investigation `docs/investigations/ai-3d-coat-stairstepping-2026-10-29.md` records
the coat sequence measurements, broader scene checks and remaining limits.

## AI stereo stills

`StillStereoRenderer` reuses the video's occlusion-tested mesh pass and dense grid,
with nearest depth at zero parallax. It accepts full-precision `StillDepthMap`s,
refines them with the shared joint-bilateral / guided upsample kernels (no temporal
state), and returns linear extended-P3 half-float eye textures. The experimental
separated-boundary mode also works on stills and retains extended colour range.
Calls on one renderer must be serialized and run off the main thread.

`StillStereoResolution` applies an additional 4096-pixel long-edge ceiling and a
256 MiB estimated stereo working-set budget **after** the app's normal resolution
policy. It is not a codec memory guarantee: ImageIO may fully decode a JXL before
returning a small image. For an 8192×4608 input the working-set estimate currently
selects at most 3454 pixels on the long edge. Keep originals; downscale only the
presentation rendition.

A repeatable Mac diagnostic (personal inputs/outputs stay outside the repo):

```bash
scripts/still-stereo-lab.sh /path/photo.jxl /path/model.mlmodelc /tmp/still-comparison 4096
swift test --filter 'StillStereo|StereoBoundaryWarp|DepthArrayStride'
```

The lab runs the actual model and shaders, writes source, raw/refined depth, both
eyes and anaglyphs for continuous and separated boundaries, and a JSON report of
sizes, timing and whole-process peak RSS. It compiles the real Metal source even
when SwiftPM doesn't build a metallib. Synthetic GPU tests always run; the personal
image test runs only when the script supplies its environment. Passing these tests
does not establish headset comfort or correct recognition of sub-model-resolution
hair, transparent steam or low-contrast silhouettes.

## License

MIT, see `LICENSE`. edge264 (vendored under `Sources/CEdge264`) is BSD-3-Clause; see
`THIRD_PARTY_NOTICES.md` and
`Vendor/README.md` for the local patches.
