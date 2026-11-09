#!/usr/bin/env bash
#
# Builds the Vergence desktop app for Windows and packs it the way Apparition packs its Windows
# installer: one zip per architecture, Vergence-<version>-windows-<arch>.zip (the name the
# updater looks for, `updater::installer_pattern`), holding
#
#   Vergence.exe          the app (tray, coordinator, pairing, worker and convert sidecars)
#   WebView2Loader.dll    keep it beside Vergence.exe (the window is Microsoft Edge WebView2,
#                         which Windows 10 and 11 have)
#   README.txt
#   + with --tools DIR:   every .exe and .dll in DIR, e.g. vergence-worker.exe,
#                         vergence-convert.exe, onnxruntime*.dll, ffmpeg.exe (built on Windows;
#                         see worker/ and convert/). The app finds them beside itself.
#
#   desktop/scripts/package-windows.sh [--arch x86_64|arm64] [--tools DIR] [--out DIR]
#
# x86_64 (the default) cross-compiles on a Mac with the x86_64-pc-windows-gnu target and mingw-w64
# (Homebrew: `brew install mingw-w64`; `rustup target add x86_64-pc-windows-gnu`), as Apparition's
# release script does. arm64 builds aarch64-pc-windows-msvc natively, in Git Bash on a Windows ARM
# PC (or the windows-11-arm runner) with the Visual Studio C++ tools, LLVM's clang on PATH (ring
# and aws-lc, the update check's TLS, compile C and assembly with it there) and a Windows Perl for
# OpenSSL; mingw has no ARM64 toolchain, and the worker and converter need MSVC anyway. OpenSSL (the paired channel's TLS-PSK) is built from source into the
# executable (`vendored-openssl` on Windows), so nothing else needs installing there. Unsigned:
# Windows SmartScreen warns on first open until releases are code-signed.
#
# The zip holds no secret and no machine-specific path; the release manifest that lists it is
# signed separately (desktop/scripts/bless-release.sh).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARCH=x86_64
TOOLS=""
OUT="$ROOT/target/package"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch)  ARCH="${2:?--arch needs x86_64 or arm64}"; shift ;;
    --tools) TOOLS="${2:?--tools needs a folder}"; shift ;;
    --out)   OUT="${2:?--out needs a folder}"; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

case "$ARCH" in
  x86_64)
    TARGET=x86_64-pc-windows-gnu
    command -v x86_64-w64-mingw32-gcc >/dev/null || { echo "error: mingw-w64 isn't installed (brew install mingw-w64)" >&2; exit 1; }
    ;;
  arm64)
    TARGET=aarch64-pc-windows-msvc
    command -v clang >/dev/null || { echo "error: clang isn't on PATH (LLVM, e.g. C:\\Program Files\\LLVM\\bin)" >&2; exit 1; }
    # MSVC's ARM64 compiler, with the /Gs0 OpenSSL's VC-WIN64-ARM config passes, puts a __chkstk
    # call before the link register is saved in tls_parse_all_extensions: the server crashes on
    # the first ClientHello (the paired channel, every connection). /Gs (the default threshold)
    # overrides it. In CPPFLAGS because OpenSSL's makefile puts those after its own cflags, and
    # no other build reads them (cc-rs adds a plain CFLAGS to every crate, ring's clang included).
    # openssl/openssl#26239; MeshInspector/MeshLib#7121 has the analysis.
    # Written -Gs, not /Gs: Git Bash rewrites a value starting with / into a Windows path.
    export CPPFLAGS="${CPPFLAGS:--Gs}"
    ;;
  *) echo "error: --arch is x86_64 or arm64, not $ARCH" >&2; exit 2 ;;
esac
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
files=(Vergence.exe README.txt)
# The gnu build loads WebView2 through this DLL; the MSVC build links the loader in.
if [[ -f "$built/WebView2Loader.dll" ]]; then
  cp "$built/WebView2Loader.dll" "$stage/"
  files+=(WebView2Loader.dll)
fi
if [[ -n "$TOOLS" ]]; then
  shopt -s nullglob
  for f in "$TOOLS"/*.exe "$TOOLS"/*.dll; do
    cp "$f" "$stage/"
    files+=("$(basename "$f")")
  done
  shopt -u nullglob
fi

cat >"$stage/README.txt" <<EOF
Vergence $VERSION for Windows ($ARCH)

Open Vergence.exe. It stays in the tray (closing the window keeps it running): the analysis
coordinator for Jellyfin and Stash, headset pairing, this computer's worker and AI 3D
conversion of video files. Keep the DLLs beside it; the window uses Microsoft Edge WebView2,
which Windows 10 and 11 normally have.

The worker and the converter are vergence-worker.exe and vergence-convert.exe (ONNX Runtime):
put them, with their onnxruntime DLLs and ffmpeg.exe, beside Vergence.exe, or give their folder
in Settings. Windows asks once whether Vergence may use your network: allow private networks,
so headsets can pair (port 4869) and reach the coordinator (port 8421).

MIT licensed. https://github.com/illixion/vergence
EOF

mkdir -p "$OUT"
zip_path="$OUT/Vergence-$VERSION-windows-$ARCH.zip"
rm -f -- "$zip_path"
if command -v zip >/dev/null; then
  (cd "$stage" && TZ=UTC zip -qX "$zip_path" "${files[@]}")
elif command -v 7z >/dev/null; then
  # Git Bash on Windows has no zip; 7-Zip is on the GitHub images. Flat files, so the same layout.
  (cd "$stage" && 7z a -tzip -bso0 -bsp0 "$(cygpath -w "$zip_path")" "${files[@]}")
else
  # Windows' own, for a PC with neither.
  (cd "$stage" && powershell.exe -NoProfile -Command "Compress-Archive -Path $(printf "'%s'," "${files[@]}" | sed 's/,$//') -DestinationPath '$(cygpath -w "$zip_path")'")
fi
echo "==> $zip_path"
(cd "$OUT" && { command -v shasum >/dev/null && shasum -a 256 "$(basename "$zip_path")" || sha256sum "$(basename "$zip_path")"; })
