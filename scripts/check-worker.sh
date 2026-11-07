#!/bin/bash
# Checks `vergence worker` end to end against a coordinator: starts a
# throwaway `vergenced` on a Jellyfin (use a disposable dev instance, never a
# real server), lets the worker analyse a frame-packed 3D title piece by piece
# through it, then compares the coordinator's chunks with `vergence analyze`
# run over the same file in one go. The two must hold the same records with
# the same values (within Core ML's run-to-run noise).
#
#   scripts/check-worker.sh JELLYFIN_URL KEY_FILE ITEM_ID LOCAL_FILE MODELS_DIR
#
# LOCAL_FILE is the item's file as this Mac sees it (for the reference run);
# MODELS_DIR holds RIFE*480x288 and NeuFlow*512x288 (as for `analyze`).
# Half or full side by side only (the reference crops the halves).
set -euo pipefail
if [ "$#" -ne 5 ]; then
    sed -n '2,15p' "$0" >&2
    exit 2
fi
jellyfin="$1"; key_file="$2"; item="$3"; file="$4"; models="$5"
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
work=$(mktemp -d)
port=$((20000 + RANDOM % 20000))
echo "check-worker: building"
(cd "$root" && swift build -c release --product vergence >/dev/null && cargo build -q --release -p vergence-coordinator)
vergence="$root/.build/release/vergence"
vergenced="$root/target/release/vergenced"

cat > "$work/vergenced.json" <<JSON
{"listen": "127.0.0.1:$port", "store": "$work/store",
 "jellyfin": {"url": "$jellyfin", "apiKeyFile": "$key_file"}}
JSON
"$vergenced" token add worker --grants work --config "$work/vergenced.json" > "$work/work.token"
"$vergenced" token add player --grants play --config "$work/vergenced.json" > "$work/play.token"
cat > "$work/worker.json" <<JSON
{"coordinator": "http://127.0.0.1:$port", "tokenFile": "$work/work.token", "name": "check",
 "models": "$models", "jellyfin": {"url": "$jellyfin", "apiKeyFile": "$key_file"}}
JSON

"$vergenced" serve --config "$work/vergenced.json" > "$work/vergenced.log" 2>&1 &
coordinator=$!
trap 'kill "$coordinator" 2>/dev/null || true' EXIT
for _ in $(seq 50); do curl -sf "http://127.0.0.1:$port/v1/health" >/dev/null && break; sleep 0.1; done

auth="Authorization: Bearer $(cat "$work/play.token")"
status=$(curl -sf -X POST "http://127.0.0.1:$port/SmoothMotion/$item/Analyze" -H "$auth")
echo "check-worker: queued: $status"
while "$vergence" worker --config "$work/worker.json" --once 2>&1 | grep -v "nothing to do"; do
    state=$(curl -sf "http://127.0.0.1:$port/SmoothMotion/$item" -H "$auth")
    case "$state" in *'"ready"'*) break ;; esac
done
chunks=$(curl -sf "http://127.0.0.1:$port/SmoothMotion/$item" -H "$auth" | python3 -c 'import json,sys; print(json.load(sys.stdin)["chunks"])')
for ((n = 0; n < chunks; n++)); do
    curl -sf -o "$work/chunk$n.vgat" "http://127.0.0.1:$port/SmoothMotion/$item/Chunks/$n" -H "$auth"
done

echo "check-worker: the reference, analysed whole"
size=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate -of csv=p=0 "$file")
IFS=, read -r width height rate <<<"$size"
ffmpeg -nostdin -v error -i "$file" -filter_complex \
    "[0:v:0]split[a][b];[a]crop=iw/2:ih:0:0[l];[b]crop=iw/2:ih:iw/2:0[r];[l][r]vstack[out]" -map "[out]" \
    -f rawvideo -pix_fmt yuv420p - |
    "$vergence" analyze --raw "$((width / 2))x$height" --rate "$rate" - --models "$models" --output "$work/whole.vgat"
python3 -I "$root/scripts/compare-analysis-tracks.py" "$work/whole.vgat" "$work"/chunk*.vgat
