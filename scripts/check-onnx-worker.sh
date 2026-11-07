#!/bin/bash
# Checks `vergence-worker` (ONNX Runtime) end to end and against Core ML: for
# each execution provider, starts a throwaway `vergenced` on a Jellyfin (use a
# disposable dev instance, never a real server), lets the worker analyse a
# frame-packed 3D title piece by piece through it, then compares the chunks
# with `vergence analyze` (Core ML) run over the same file in one go. Prints
# the differences per kind (max, p99, mean) and how long each provider took:
# the answer to whether ONNX and Core ML may share a profile.
#
#   scripts/check-onnx-worker.sh JELLYFIN_URL KEY_FILE ITEM_ID LOCAL_FILE COREML_DIR ONNX_DIR [EP...]
#
# COREML_DIR holds RIFE*480x288 and NeuFlow*512x288 .mlpackages (for the
# reference), ONNX_DIR the exports of scripts/export-onnx-models.py. EPs
# default to "cpu coreml" (what a Mac can run). Half or full side by side only.
set -euo pipefail
if [ "$#" -lt 6 ]; then
    sed -n '2,16p' "$0" >&2
    exit 2
fi
jellyfin="$1"; key_file="$2"; item="$3"; file="$4"; coreml="$5"; onnx="$6"
shift 6
providers=("$@")
[ "$#" -eq 0 ] && providers=(cpu coreml)
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
work=$(mktemp -d)
echo "check-onnx-worker: building (work in $work)"
(cd "$root" && swift build -c release --product vergence >/dev/null \
    && cargo build -q --release -p vergence-coordinator -p vergence-worker)
vergence="$root/.build/release/vergence"
vergenced="$root/target/release/vergenced"
worker="$root/target/release/vergence-worker"

coordinator=""
trap '[ -n "$coordinator" ] && kill "$coordinator" 2>/dev/null || true' EXIT

for ep in "${providers[@]}"; do
    dir="$work/$ep"
    mkdir -p "$dir"
    port=$((20000 + RANDOM % 20000))
    cat > "$dir/vergenced.json" <<JSON
{"listen": "127.0.0.1:$port", "store": "$dir/store",
 "jellyfin": {"url": "$jellyfin", "apiKeyFile": "$key_file"}}
JSON
    "$vergenced" token add worker --grants work --config "$dir/vergenced.json" > "$dir/work.token"
    "$vergenced" token add player --grants play --config "$dir/vergenced.json" > "$dir/play.token"
    cat > "$dir/worker.json" <<JSON
{"coordinator": "http://127.0.0.1:$port", "tokenFile": "$dir/work.token", "name": "check-$ep",
 "models": "$onnx", "executionProvider": "$ep", "jellyfin": {"url": "$jellyfin", "apiKeyFile": "$key_file"}}
JSON
    "$vergenced" serve --config "$dir/vergenced.json" > "$dir/vergenced.log" 2>&1 &
    coordinator=$!
    for _ in $(seq 50); do curl -sf "http://127.0.0.1:$port/v1/health" >/dev/null && break; sleep 0.1; done
    auth="Authorization: Bearer $(cat "$dir/play.token")"
    curl -sf -X POST "http://127.0.0.1:$port/SmoothMotion/$item/Analyze" -H "$auth" >/dev/null
    echo "check-onnx-worker: $ep: analysing"
    start=$(date +%s)
    runs=0
    while [ $((runs += 1)) -le 100 ] && "$worker" --config "$dir/worker.json" --once 2>&1 | tee -a "$dir/worker.log" | grep -v "nothing to do" >/dev/null; do
        state=$(curl -sf "http://127.0.0.1:$port/SmoothMotion/$item" -H "$auth")
        case "$state" in *'"ready"'*) break ;; esac
    done
    seconds=$(($(date +%s) - start))
    grep -E "piece|on " "$dir/worker.log" | head -2
    chunks=$(curl -sf "http://127.0.0.1:$port/SmoothMotion/$item" -H "$auth" | python3 -c 'import json,sys; print(json.load(sys.stdin)["chunks"])')
    for ((n = 0; n < chunks; n++)); do
        curl -sf -o "$dir/chunk$n.vgat" "http://127.0.0.1:$port/SmoothMotion/$item/Chunks/$n" -H "$auth"
    done
    pieces=$(grep -c "piece .* in " "$dir/worker.log" || true)
    echo "check-onnx-worker: $ep: $pieces pieces in $seconds s (each run loads the models again)"
    kill "$coordinator" 2>/dev/null || true
    wait "$coordinator" 2>/dev/null || true
    coordinator=""
done

echo "check-onnx-worker: the Core ML reference, analysed whole"
size=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate -of csv=p=0 "$file")
IFS=, read -r width height rate <<<"$size"
ffmpeg -nostdin -v error -i "$file" -filter_complex \
    "[0:v:0]split[a][b];[a]crop=iw/2:ih:0:0[l];[b]crop=iw/2:ih:iw/2:0[r];[l][r]vstack[out]" -map "[out]" \
    -f rawvideo -pix_fmt yuv420p - |
    "$vergence" analyze --raw "$((width / 2))x$height" --rate "$rate" - --models "$coreml" --output "$work/coreml.vgat"

# Differences per kind and channel group, with the record reader of compare-analysis-tracks.py.
stats() {
    python3 -I - "$root/scripts/compare-analysis-tracks.py" "$@" <<'PY'
import sys
source = open(sys.argv[1]).read().replace("\nmain()\n", "\n")
scope = {"__name__": "compare"}
exec(compile(source, sys.argv[1], "exec"), scope)
label, reference_path, others = sys.argv[2], sys.argv[3], sys.argv[4:]
_, reference = scope["records"](reference_path)
other = {}
for path in others:
    other.update(scope["records"](path)[1])
groups = {}
for key in set(reference) & set(other):
    a, b = reference[key], other[key]
    if key[0] == 1:  # flowPair: 4 flow channels then the mask
        n = len(a) // 5
        parts = {"flowPair flow (stored px)": (a[:4 * n], b[:4 * n]), "flowPair mask": (a[4 * n:], b[4 * n:])}
    else:
        parts = {"disparity (stored px)": (a, b)}
    for name, (x, y) in parts.items():
        groups.setdefault(name, []).extend(abs(p - q) for p, q in zip(x, y))
missing, extra = len(set(reference) - set(other)), len(set(other) - set(reference))
print(f"{label}: {len(reference)} reference records, {missing} missing, {extra} extra")
for name, d in sorted(groups.items()):
    d.sort()
    print(f"  {name:28} max {d[-1]:.3f}  p99 {d[int(len(d) * 0.99)]:.3f}  mean {sum(d) / len(d):.4f}")
PY
}

for ep in "${providers[@]}"; do
    stats "ONNX $ep vs Core ML analyze" "$work/coreml.vgat" "$work/$ep"/chunk*.vgat
done
if [ "${#providers[@]}" -ge 2 ]; then
    first="${providers[0]}"
    for ep in "${providers[@]:1}"; do
        # The two providers' chunk sets, record by record.
        python3 -I - "$work/$first" "$work/$ep" "$root/scripts/compare-analysis-tracks.py" <<'PY'
import glob, sys
source = open(sys.argv[3]).read().replace("\nmain()\n", "\n")
scope = {"__name__": "compare"}
exec(compile(source, sys.argv[3], "exec"), scope)
def load(d):
    out = {}
    for p in sorted(glob.glob(d + "/chunk*.vgat")):
        out.update(scope["records"](p)[1])
    return out
a, b = load(sys.argv[1]), load(sys.argv[2])
worst = {}
for k in set(a) & set(b):
    kind = {1: "flowPair", 2: "disparity"}.get(k[0], k[0])
    worst[kind] = max(worst.get(kind, 0), max(abs(x - y) for x, y in zip(a[k], b[k])))
print(f"ONNX {sys.argv[2].rsplit('/', 1)[1]} vs ONNX {sys.argv[1].rsplit('/', 1)[1]}: max",
      ", ".join(f"{k} {v:.3f}" for k, v in sorted(worst.items())))
PY
    done
fi
echo "check-onnx-worker: chunks and logs are in $work"
