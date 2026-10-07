# Vendored edge264

`Sources/CEdge264` is [edge264](https://github.com/tvlabs/edge264) (BSD-3-Clause), the H.264
decoder doing the actual picture decoding, MVC (Stereo High) included.

- **Upstream commit:** `2c2ab95d63c1ad89c5687f9e50b57cad772c871b` (2026-07-28).
- **Files:** `edge264.h` → `include/`, `src/*.c` and `src/edge264_internal.h` as-is, minus the
  test programs (`edge264_test.c`, `edge264_check.c`). SwiftPM compiles only `src/edge264.c`,
  which includes the other sources (edge264 is a unity build); the rest are listed in `exclude`
  so SwiftPM does not compile them a second time.
- **Build flags:** none of upstream's Makefile flags are needed. `-flax-vector-conversions`
  (upstream) is clang's default for integer vectors already, and SwiftPM forbids `unsafeFlags`
  in a package other packages depend on. edge264's `logs` variant (`-DLOGS`) is not built.

## Local changes

1. `src/edge264_internal.h`: `#include "../edge264.h"` → `#include "edge264.h"` (the header
   moved to `include/`).
2. `src/edge264_headers.c` and `src/edge264_internal.h`: the DPB-fullness test before storing
   a picture (C.4.5) counts the DPB *after* the current picture's reference marking. edge264
   counted the reference the sliding window was about to drop, so in a stream whose
   `max_num_ref_frames` equals its `max_dec_frame_buffering` (both 4 on the 3D disc tested, and
   in x264 output with `ref=4`) the DPB always looked full, bumping found nothing to output, and
   each new reference picture was output at once, ahead of the B-frames shown before it. Same
   pixels, wrong order: 76 of 96 frames of an x264 test stream. Patch:
   `edge264-dpb-after-marking.patch`; regression test `DecodeOrderTests`. To be offered upstream.
   No change on the 27 ITU MVC conformance streams or `make check`. The same miscount made
   edge264 assert (`dpb_frames <= max_dec_frame_buffering`) at the second GOP of the 3D disc's
   dependent view; an earlier local patch skipping the frame_num gap process on IDR pictures
   only hid that and has been dropped.
3. `edge264.c` / `edge264.h`: `edge264_last_frame_id()`, the FrameId the last slice started.
   mvc3d files each access unit's PTS under it and reads it back from the output frame's
   `FrameId`, since edge264 carries no timestamps. mvc3d-only (upstream might prefer a per-picture
   user value).

## Known upstream issues

- When the last picture is missing slices (a stream cut inside a slice), the end-of-stream call
  (`edge264_decode_NAL(dec, end, end, ...)`) returns `ENOBUFS` forever while `edge264_get_frame`
  has no frame to give, so a drain loop never sees `ENODATA` (`edge264_test` hangs the same way).
  Reproduced with `MVCDS1.264` cut at any of three offsets. `MVCDecoder.finish` stops when a
  drain call returns `ENOBUFS` and yields nothing; `StereoStreamReader` also never passes a
  partial final PES.
