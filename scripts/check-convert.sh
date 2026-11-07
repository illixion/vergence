#!/bin/bash
# Checks vergence-convert (ONNX Runtime, wgpu, ffmpeg) against the Mac's
# `vergence convert` (Core ML, Metal, AVFoundation): the same clip through
# both, then each eye's PSNR (luma) and SSIM of the converter's copy against
# the Mac's, and the depth tracks record by record. Prints the numbers and
# PASS/FAIL against the tolerance (default 40 dB and 0.97 per eye; see the
# README's "On Windows and Linux" for why).
#
#   scripts/check-convert.sh CLIP MODEL.onnx [--layout L] [--frames N] [--min-psnr DB] [--min-ssim S]
#
# Needs a Mac with the Core ML model installed (or VERGENCE_DEPTH_MODEL), ffmpeg
# and both tools built (swift build -c release; cargo build --release -p vergence-convert).
set -euo pipefail
if [ "$#" -lt 2 ]; then
    sed -n '2,13p' "$0" >&2
    exit 2
fi
clip="$1"; model="$2"; shift 2
layout=fsbs; frames=""; min_psnr=40; min_ssim=0.97
while [ "$#" -gt 0 ]; do
    case "$1" in
        --layout) layout="$2"; shift 2 ;;
        --frames) frames="$2"; shift 2 ;;
        --min-psnr) min_psnr="$2"; shift 2 ;;
        --min-ssim) min_ssim="$2"; shift 2 ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
vergence="${VERGENCE_CLI:-$root/.build/release/vergence}"
convert="${VERGENCE_CONVERT:-${CARGO_TARGET_DIR:-$root/target}/release/vergence-convert}"
work=$(mktemp -d)
extra=()
[ -n "$frames" ] && extra=(--frames "$frames")
"$vergence" convert "$clip" --layout "$layout" "${extra[@]}" --output "$work/mac.mp4" --depth-track "$work/mac.vgat" 2>&1 | grep -E "depth:|picture:|done:"
"$convert" convert "$clip" --layout "$layout" "${extra[@]}" --output "$work/pc.mp4" --depth-track "$work/pc.vgat" \
    --model "$model" 2>&1 | grep -E "engines|depth:|picture:|done:"

case "$layout" in
    fsbs | hsbs) crops=("crop=iw/2:ih:0:0" "crop=iw/2:ih:iw/2:0") ;;
    *) crops=("crop=iw:ih/2:0:0" "crop=iw:ih/2:0:ih/2") ;;
esac
status=0
for i in 0 1; do
    eye=$([ "$i" = 0 ] && echo left || echo right)
    graph="[0:v]${crops[$i]},format=yuv420p10le[a];[1:v]${crops[$i]},format=yuv420p10le[b];[a][b]"
    psnr=$(ffmpeg -nostdin -v error -i "$work/mac.mp4" -i "$work/pc.mp4" -lavfi "${graph}psnr=stats_file=-" -f null - |
        awk '{for (i = 1; i <= NF; i++) if ($i ~ /^psnr_y:/) { split($i, v, ":"); if (v[2] != "inf") { s += v[2]; n++ } }}
             END { printf "%.2f", n ? s / n : 99 }')
    ssim=$(ffmpeg -nostdin -i "$work/mac.mp4" -i "$work/pc.mp4" -lavfi "${graph}ssim" -f null - 2>&1 |
        grep -o "All:[0-9.]*" | tail -1 | cut -d: -f2)
    verdict=$(awk -v p="$psnr" -v s="$ssim" -v mp="$min_psnr" -v ms="$min_ssim" 'BEGIN { print (p >= mp && s >= ms) ? "PASS" : "FAIL" }')
    echo "check-convert: $eye eye: PSNR-Y $psnr dB, SSIM $ssim: $verdict"
    [ "$verdict" = PASS ] || status=1
done
# The depth itself: the values the warp reads, record by record (informational).
python3 -I "$root/scripts/compare-analysis-tracks.py" "$work/mac.vgat" "$work/pc.vgat" --tolerance 1 | sed 's/^/check-convert: depth: /' || true
echo "check-convert: outputs in $work"
exit "$status"
