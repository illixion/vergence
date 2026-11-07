#!/usr/bin/env bash
#
# Builds the Vergence desktop app for Windows (x86_64) on a Mac and packs it the way Apparition
# packs its Windows installer: one zip, Vergence-<version>-windows-x86_64.zip (the name the
# updater looks for, `updater::installer_name`), holding
#
#   Vergence.exe          the app (tray, coordinator, pairing, worker and convert sidecars)
#   WebView2Loader.dll    keep it beside Vergence.exe (the window is Microsoft Edge WebView2,
#                         which Windows 10 and 11 have)
#   README.txt
#   + with --tools DIR:   every .exe and .dll in DIR, e.g. vergence-worker.exe,
#                         vergence-convert.exe, onnxruntime*.dll, ffmpeg.exe (built on Windows;
#                         see worker/ and convert/). The app finds them beside itself.
#
#   desktop/scripts/package-windows.sh [--tools DIR] [--out DIR]
#
# Cross-compiles with the x86_64-pc-windows-gnu target and mingw-w64 (Homebrew:
# `brew install mingw-w64`; `rustup target add x86_64-pc-windows-gnu`), as Apparition's
# release script does. OpenSSL (the paired channel's TLS-PSK) is built from source into the
# executable (`vendored-openssl` on Windows), so nothing else needs installing there. Unsigned:
# Windows SmartScreen warns on first open until releases are code-signed.
#
# The zip holds no secret and no machine-specific path; the release manifest that lists it is
# signed separately (desktop/scripts/bless-release.sh).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TOOLS=""
OUT="$ROOT/target/package"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tools) TOOLS="${2:?--tools needs a folder}"; shift ;;
    --out)   OUT="${2:?--out needs a folder}"; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

TARGET=x86_64-pc-windows-gnu
command -v x86_64-w64-mingw32-gcc >/dev/null || { echo "error: mingw-w64 isn't installed (brew install mingw-w64)" >&2; exit 1; }
rustup target list --installed | grep -qx "$TARGET" || { echo "error: rustup target add $TARGET" >&2; exit 1; }

VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' "$ROOT/Cargo.toml" | head -1)"
TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"

echo "==> building Vergence $VERSION for Windows ($TARGET)"
# custom-protocol: serve the bundled ui/ (what `cargo tauri build` turns on).
cargo build --release --locked --manifest-path "$ROOT/Cargo.toml" --target "$TARGET" \
  -p vergence-desktop --features vergence-desktop/custom-protocol

built="$TARGET_DIR/$TARGET/release"
stage="$(mktemp -d)"
trap 'rm -rf -- "$stage"' EXIT
cp "$built/vergence-app.exe" "$stage/Vergence.exe"
cp "$built/WebView2Loader.dll" "$stage/"
files=(Vergence.exe WebView2Loader.dll README.txt)
if [[ -n "$TOOLS" ]]; then
  shopt -s nullglob
  for f in "$TOOLS"/*.exe "$TOOLS"/*.dll; do
    cp "$f" "$stage/"
    files+=("$(basename "$f")")
  done
  shopt -u nullglob
fi

cat >"$stage/README.txt" <<EOF
Vergence $VERSION for Windows (x86_64)

Open Vergence.exe. It stays in the tray (closing the window keeps it running): the analysis
coordinator for Jellyfin and Stash, headset pairing, this computer's worker and AI 3D
conversion of video files. Keep WebView2Loader.dll beside it; the window uses Microsoft Edge
WebView2, which Windows 10 and 11 normally have.

The worker and the converter are vergence-worker.exe and vergence-convert.exe (ONNX Runtime):
put them, with their onnxruntime DLLs and ffmpeg.exe, beside Vergence.exe, or give their folder
in Settings. Windows asks once whether Vergence may use your network: allow private networks,
so headsets can pair (port 4869) and reach the coordinator (port 8421).

MIT licensed. https://github.com/illixion/vergence
EOF

mkdir -p "$OUT"
zip_path="$OUT/Vergence-$VERSION-windows-x86_64.zip"
rm -f -- "$zip_path"
(cd "$stage" && TZ=UTC zip -qX "$zip_path" "${files[@]}")
echo "==> $zip_path"
(cd "$OUT" && shasum -a 256 "$(basename "$zip_path")")
