#!/usr/bin/env bash
# Interop check: RAVESDK's RAVECompanion (Swift, Network.framework) against the
# Rust port, both directions, on loopback. macOS only; needs RAVESDK beside
# vergence (~/Projects/RAVESDK).
#
#   1. Swift device -> Rust host: six-digit pairing over the direct pairing
#      port, then the Swift client's TLS 1.2 PSK channel to the Rust OpenSSL
#      server and its identify proof.
#   2. Rust device -> Swift host: the same, the other way round.
#   3. Swift device -> vergenced: `vergenced pair` (answered "y"), then the coordinator's HTTP API
#      on its paired TLS-PSK port through RAVECompanionHTTPClient, as Convolution uses it.
#
# Each direction passes when both sides print the same code and both end OK.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf -- "$work"' EXIT

# Built under target/ (ignored), so a re-run doesn't rebuild RAVESDK.
swift_build="$root/target/companion-interop-swift"
swift build --package-path "$here/swift" --scratch-path "$swift_build" >"$work/swift-build.log" 2>&1 ||
    { cat "$work/swift-build.log"; exit 1; }
swift_bin="$swift_build/debug/CompanionInterop"
cargo build --quiet --manifest-path "$root/Cargo.toml" -p vergence-companion --examples
cargo build --quiet --manifest-path "$root/Cargo.toml" -p vergence-coordinator --bin vergenced
target_dir="${CARGO_TARGET_DIR:-$root/target}"
rust_dir="$target_dir/debug/examples"

pair_port=$((40000 + RANDOM % 10000))
tls_port=$((pair_port + 1))

wait_for() { # file pattern
    for _ in $(seq 1 100); do grep -q "$2" "$1" 2>/dev/null && return 0; sleep 0.1; done
    return 1
}

check() { # name host-log device-log
    local host_code device_code
    host_code="$(grep '^CODE ' "$2" | head -1 | cut -d' ' -f2)"
    device_code="$(grep '^CODE ' "$3" | head -1 | cut -d' ' -f2)"
    echo "== $1"
    sed 's/^/  host:   /' "$2"
    sed 's/^/  device: /' "$3"
    if [[ -n "$host_code" && "$host_code" == "$device_code" ]] && grep -q '^OK' "$2" && grep -q '^OK' "$3"; then
        echo "  PASS (both showed $host_code)"
    else
        echo "  FAIL"
        return 1
    fi
}

status=0

# 1. Swift device, Rust host.
timeout 90 "$rust_dir/interop_host" "$pair_port" "$tls_port" "$work/store.json" >"$work/h1.log" 2>&1 &
host_pid=$!
wait_for "$work/h1.log" LISTENING
timeout 60 "$swift_bin" device "$pair_port" "$tls_port" >"$work/d1.log" 2>&1 || true
wait "$host_pid" || true
check "Swift device -> Rust host" "$work/h1.log" "$work/d1.log" || status=1

# 2. Rust device, Swift host.
pair_port=$((pair_port + 2))
tls_port=$((pair_port + 1))
timeout 90 "$swift_bin" host "$pair_port" "$tls_port" >"$work/h2.log" 2>&1 &
host_pid=$!
wait_for "$work/h2.log" LISTENING
timeout 60 "$rust_dir/interop_device" "$pair_port" "$tls_port" >"$work/d2.log" 2>&1 || true
wait "$host_pid" || true
check "Rust device -> Swift host" "$work/h2.log" "$work/d2.log" || status=1

# 3. Swift device, vergenced (pairing, then the API over the paired channel).
pair_port=$((pair_port + 2))
api_port=$((pair_port + 1))
http_port=$((pair_port + 2))
mkdir -p "$work/store"
printf '{ "listen": "127.0.0.1:%s", "store": "%s", "pairedListen": "127.0.0.1:%s" }\n' \
    "$http_port" "$work/store" "$api_port" >"$work/vergenced.json"
timeout 90 "$target_dir/debug/vergenced" serve --config "$work/vergenced.json" >"$work/serve.log" 2>&1 &
serve_pid=$!
(sleep 3; printf 'y\n') | timeout 90 "$target_dir/debug/vergenced" pair --port "$pair_port" --config "$work/vergenced.json" >"$work/h3.log" 2>&1 &
host_pid=$!
wait_for "$work/h3.log" "Pairing is open"
wait_for "$work/serve.log" "serving paired devices"
timeout 60 "$swift_bin" api "$pair_port" "$api_port" >"$work/d3.log" 2>&1 || true
wait "$host_pid" || true
kill "$serve_pid" 2>/dev/null || true
wait "$serve_pid" 2>/dev/null || true
host_code="$(grep -o 'shows [0-9]* [0-9]*' "$work/h3.log" | head -1 | tr -dc '0-9')"
device_code="$(grep '^CODE ' "$work/d3.log" | head -1 | cut -d' ' -f2)"
echo "== Swift device -> vergenced (pair, then the API over TLS-PSK)"
sed 's/^/  host:   /' "$work/h3.log"
sed 's/^/  device: /' "$work/d3.log"
if [[ -n "$host_code" && "$host_code" == "$device_code" ]] && grep -q 'Paired with' "$work/h3.log" && grep -q '^OK' "$work/d3.log"; then
    echo "  PASS (both showed $host_code)"
else
    echo "  FAIL"
    status=1
fi

exit "$status"
