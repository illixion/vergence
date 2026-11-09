#!/usr/bin/env bash
#
# Builds the Vergence desktop app for macOS (arm64 + x86_64, one universal bundle) and packs it
# the way the updater unpacks it (`updater::install_mac`): Vergence-<version>-macos-universal.zip
# with Vergence.app at its root, made by `ditto`, ad-hoc sealed so `codesign --verify --deep
# --strict` passes. The `vergence` command line (the Swift package's executable, also universal)
# rides in Contents/Resources/, where the app looks for it (`sidecar::find_cli`); it is not in
# MacOS/, because a case-insensitive volume would merge `vergence` with the app's `Vergence`.
#
#   desktop/scripts/package-macos.sh [--out DIR] [--skip-cli]
#
# Needs: Xcode (for swift, lipo, codesign), rustup targets aarch64-apple-darwin and
# x86_64-apple-darwin, and the Tauri CLI (`cargo install tauri-cli --version '^2' --locked`, or
# Tauri's prebuilt release; the release workflow takes the latter).
#
# Not notarized: macOS blocks the first open of a downloaded copy until its quarantine flag is
# cleared (the README in the zip says how). The zip holds no secret and no machine-specific path;
# the release manifest that lists it is signed separately (desktop/scripts/bless-release.sh).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
OUT="$ROOT/target/package"
CLI=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)      OUT="${2:?--out needs a folder}"; shift ;;
    --skip-cli) CLI=0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

[[ "$(uname -s)" == Darwin ]] || { echo "error: the macOS app is built on a Mac" >&2; exit 1; }
for target in aarch64-apple-darwin x86_64-apple-darwin; do
  rustup target list --installed | grep -qx "$target" || { echo "error: rustup target add $target" >&2; exit 1; }
done
tauri_cli="$(command -v cargo-tauri || true)"
[[ -n "$tauri_cli" ]] || { echo "error: the Tauri CLI isn't installed (cargo install tauri-cli --version '^2' --locked)" >&2; exit 1; }

VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' "$ROOT/Cargo.toml" | head -1)"
TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
app_name=Vergence
cd "$ROOT"

work="$(mktemp -d)"
trap 'rm -rf -- "$work"' EXIT

echo "==> building Vergence $VERSION for macOS (arm64 + x86_64)"
# custom-protocol: serve the bundled ui/ (what `cargo tauri build` turns on). No
# MACOSX_DEPLOYMENT_TARGET: it reaches the proc-macro dylibs built for this Mac too, and Xcode 27's
# linker then writes ones dlopen refuses ("mis-aligned LINKEDIT"). The 14.0 floor is the bundle's
# (tauri.conf.json, minimumSystemVersion).
# vendored-openssl: the paired channel's TLS-PSK builds OpenSSL from source into the executable, as
# on Windows. Linking Homebrew's would leave /opt/homebrew paths in the app (it would not start on
# a Mac without them), and the x86_64 half would find only Homebrew's arm64 build.
for target in aarch64-apple-darwin x86_64-apple-darwin; do
  cargo build --release --locked --target "$target" \
    -p vergence-desktop --features vergence-desktop/custom-protocol,vergence-companion/vendored-openssl
done
# The universal binary is put where `tauri bundle` looks for it, under tauri.conf.json's
# mainBinaryName (not the cargo name): `tauri build` would compile the app again on its own.
universal="$TARGET_DIR/universal-apple-darwin/release"
mkdir -p "$universal"
xcrun lipo -create -output "$universal/$app_name" \
  "$TARGET_DIR/aarch64-apple-darwin/release/vergence-app" "$TARGET_DIR/x86_64-apple-darwin/release/vergence-app"
rm -rf "$universal/bundle"
(cd desktop && "$tauri_cli" tauri bundle --target universal-apple-darwin --bundles app)
ditto "$universal/bundle/macos/$app_name.app" "$work/$app_name.app"

if [[ "$CLI" == 1 ]]; then
  echo "==> the vergence command line (universal)"
  swift build -c release --product vergence --arch arm64 --arch x86_64
  cli="$(swift build -c release --arch arm64 --arch x86_64 --show-bin-path)/vergence"
  lipo_archs="$(xcrun lipo -archs "$cli")"
  [[ "$lipo_archs" == *arm64* && "$lipo_archs" == *x86_64* ]] || { echo "error: $cli is $lipo_archs, not universal" >&2; exit 1; }
  mkdir -p "$work/$app_name.app/Contents/Resources"
  cp "$cli" "$work/$app_name.app/Contents/Resources/vergence"
  codesign -f -s - "$work/$app_name.app/Contents/Resources/vergence"
fi
# Sealed again around what was added (the bundle's seal covers Resources/).
codesign -f -s - --entitlements "$ROOT/desktop/Entitlements.plist" "$work/$app_name.app"
codesign --verify --deep --strict "$work/$app_name.app"

mkdir -p "$OUT"
zip_path="$OUT/Vergence-$VERSION-macos-universal.zip"
rm -f -- "$zip_path"
# Vergence.app at the zip's root, resource forks kept (`ditto -x -k` is what unpacks it).
ditto -c -k --keepParent --sequesterRsrc "$work/$app_name.app" "$zip_path"
echo "==> $zip_path"
(cd "$OUT" && shasum -a 256 "$(basename "$zip_path")")
