#!/bin/bash
# Checks one of the 2D workloads (smooth-motion-2d, ai-3d-depth, ai-hdr) of
# `vergence-worker` (ONNX Runtime) against `vergence worker` (Core ML): each
# analyses the same title of a Jellyfin (use a disposable dev instance, never a
# real server) through its own throwaway `vergenced`, and the two chunk sets
# are compared record by record (max and mean difference per kind; 3D Smooth
# Motion has scripts/check-onnx-worker.sh).
#
#   scripts/check-onnx-workload.sh JELLYFIN_URL KEY_FILE ITEM_ID WORKLOAD [EP]
#
# Models, by environment (defaults in brackets):
#   COREML_MODELS  RIFE*960x576.mlpackage   [../Hypnos/build/interp-models]
#   ONNX_MODELS    RIFE*960x576.onnx        [onnx-models]
#   DEPTH_MODEL    Core ML Depth Anything V2 Small 770x574  [../Hypnos/DepthAnythingV2Small770x574.mlpackage]
#   DEPTH_ONNX     its ONNX export, free input size         [onnx-models/DepthAnythingV2Small.onnx]
#   GAIN_MODEL     GainMapGMNet.mlpackage   [../Hypnos/Hypnos/Hypnos/Resources/GainMapGMNet.mlpackage]
#   GAIN_ONNX      GainMapGMNet.onnx (scripts/export-gmnet-onnx.py) [onnx-models/GainMapGMNet.onnx]
# EP is the ONNX worker's execution provider (cpu by default, as a Mac runs
# it). Builds go to CARGO_TARGET_DIR and SWIFT_SCRATCH_PATH when set. Chunks
# and logs stay in a temporary folder, printed at the end.
set -euo pipefail
if [ "$#" -lt 4 ]; then
    sed -n '2,22p' "$0" >&2
    exit 2
fi
jellyfin="$1"; key_file="$2"; item="$3"; workload="$4"; ep="${5:-cpu}"
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
hypnos="$root/../Hypnos"
coreml_models="${COREML_MODELS:-$hypnos/build/interp-models}"
onnx_models="${ONNX_MODELS:-$root/onnx-models}"
depth_model="${DEPTH_MODEL:-$hypnos/DepthAnythingV2Small770x574.mlpackage}"
depth_onnx="${DEPTH_ONNX:-$root/onnx-models/DepthAnythingV2Small.onnx}"
gain_model="${GAIN_MODEL:-$hypnos/Hypnos/Hypnos/Resources/GainMapGMNet.mlpackage}"
gain_onnx="${GAIN_ONNX:-$root/onnx-models/GainMapGMNet.onnx}"
work=$(mktemp -d)
echo "check-onnx-workload: building (work in $work)"
swift_build="${SWIFT_SCRATCH_PATH:-$root/.build}"
cargo_target="${CARGO_TARGET_DIR:-$root/target}"
(cd "$root" && swift build -c release --product vergence --scratch-path "$swift_build" >/dev/null \
    && cargo build -q --release -p vergence-coordinator -p vergence-worker --target-dir "$cargo_target")
vergenced="$cargo_target/release/vergenced"

coordinator=""; worker=""
trap '[ -n "$worker" ] && kill "$worker" 2>/dev/null; [ -n "$coordinator" ] && kill "$coordinator" 2>/dev/null; true' EXIT

# One run: a coordinator, the worker until the title is ready, the chunks.
run() {
    local who="$1" dir="$work/$1" port=$((20000 + RANDOM % 20000))
    mkdir -p "$dir"
    cat > "$dir/vergenced.json" <<JSON
{"listen": "127.0.0.1:$port", "store": "$dir/store", "pairedListen": null,
 "jellyfin": {"url": "$jellyfin", "apiKeyFile": "$key_file"}}
JSON
    "$vergenced" token add worker --grants work --config "$dir/vergenced.json" > "$dir/work.token"
    "$vergenced" token add player --grants play --config "$dir/vergenced.json" > "$dir/play.token"
    "$vergenced" serve --config "$dir/vergenced.json" > "$dir/vergenced.log" 2>&1 &
    coordinator=$!
    for _ in $(seq 50); do curl -sf "http://127.0.0.1:$port/v1/health" >/dev/null && break; sleep 0.1; done
    local auth="Authorization: Bearer $(cat "$dir/play.token")"
    local title="http://127.0.0.1:$port/v1/titles/jellyfin/$item/$workload"
    curl -sf -X POST "$title" -H "$auth" >/dev/null
    if [ "$who" = coreml ]; then
        cat > "$dir/worker.json" <<JSON
{"coordinator": "http://127.0.0.1:$port", "tokenFile": "$dir/work.token", "name": "check-coreml",
 "models": "$coreml_models", "workloads": ["$workload"], "depthModel": "$depth_model", "gainModel": "$gain_model",
 "jellyfin": {"url": "$jellyfin", "apiKeyFile": "$key_file"}}
JSON
        "$swift_build/release/vergence" worker --config "$dir/worker.json" > "$dir/worker.log" 2>&1 &
    else
        cat > "$dir/worker.json" <<JSON
{"coordinator": "http://127.0.0.1:$port", "tokenFile": "$dir/work.token", "name": "check-onnx",
 "models": "$onnx_models", "workloads": ["$workload"], "executionProvider": "$ep",
 "depthModel": "$depth_onnx", "gainModel": "$gain_onnx",
 "jellyfin": {"url": "$jellyfin", "apiKeyFile": "$key_file"}}
JSON
        "$cargo_target/release/vergence-worker" --config "$dir/worker.json" > "$dir/worker.log" 2>&1 &
    fi
    worker=$!
    local started=$(date +%s)
    until curl -sf "$title" -H "$auth" | grep -q '"ready"'; do
        kill -0 "$worker" 2>/dev/null || { echo "check-onnx-workload: the $who worker stopped:"; tail -3 "$dir/worker.log"; exit 1; }
        sleep 1
    done
    echo "check-onnx-workload: $who: $(grep -c "piece .* in " "$dir/worker.log") pieces in $(($(date +%s) - started)) s"
    kill "$worker" 2>/dev/null; wait "$worker" 2>/dev/null || true; worker=""
    local chunks n
    chunks=$(curl -sf "$title" -H "$auth" | python3 -c 'import json,sys; print(json.load(sys.stdin)["chunks"])')
    for ((n = 0; n < chunks; n++)); do curl -sf -o "$dir/chunk$n.vgat" "$title/chunks/$n" -H "$auth"; done
    kill "$coordinator" 2>/dev/null; wait "$coordinator" 2>/dev/null || true; coordinator=""
}
run coreml
run onnx

# Differences per kind, with the record reader of compare-analysis-tracks.py.
python3 -I - "$root/scripts/compare-analysis-tracks.py" "$work/coreml" "$work/onnx" <<'PY'
import glob, sys
source = open(sys.argv[1]).read().replace("\nmain()\n", "\n")
scope = {"__name__": "compare"}
exec(compile(source, sys.argv[1], "exec"), scope)
def load(d):
    out = {}
    for path in sorted(glob.glob(d + "/chunk*.vgat")):
        out.update(scope["records"](path)[1])
    return out
a, b = load(sys.argv[2]), load(sys.argv[3])
print(f"ONNX vs Core ML: {len(a)} records, {len(set(a) - set(b))} missing, {len(set(b) - set(a))} extra")
worst, total, count = {}, {}, {}
for key in set(a) & set(b):
    kind = scope["KINDS"].get(key[0], key[0])
    if len(a[key]) != len(b[key]):
        print(f"  {kind} at {key[2]:.3f} s: {len(a[key])} values against {len(b[key])}")
        continue
    d = [abs(x - y) for x, y in zip(a[key], b[key])]
    worst[kind] = max(worst.get(kind, 0), max(d))
    total[kind] = total.get(kind, 0) + sum(d)
    count[kind] = count.get(kind, 0) + len(d)
for kind in sorted(worst):
    print(f"  {kind}: max {worst[kind]:.4f}, mean {total[kind] / count[kind]:.5f} (the record's own units)")
PY
echo "check-onnx-workload: chunks and logs are in $work"
