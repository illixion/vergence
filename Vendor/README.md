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
2. `src/edge264_headers.c`, `parse_slice_layer_without_partitioning`: the frame_num gap
   process (H.264 8.2.5.2) is skipped for IDR pictures. edge264 counts `FrameNum` on across
   IDRs, so every IDR looked like a gap and got up to `max_num_ref_frames` "non-existing"
   frames. On the base view the IDR's marking discarded them again; on an MVC dependent view
   (a NAL 20 slice with `non_idr_flag = 0`, which every 3D Blu-ray GOP opens with) they
   stayed, and the next picture overflowed the DPB (`assert` in
   `parse_slice_layer_without_partitioning`). Patch: `edge264-idr-gap.patch`. To be offered
   upstream.

## Known upstream issues

- A slice cut short by the end of the input makes the decoder loop forever (seen with
  `edge264_test` on a truncated file). `StereoStreamReader` never passes a partial final PES.
