#!/usr/bin/env bash
#
# Builds the Vergence desktop app (Compute Assist, conversion) for this Mac, signs it with a stable
# identity, installs it to /Applications and starts it: quits the running copy first. For using
# the app on this Mac and for a rebuild-test loop. package-macos.sh makes the universal release.
#
# Why signing matters: macOS ties what it lets the app do to the app's code-signing identity.
# - The Keychain: paired headsets are kept in the login Keychain (vergence-companion's
#   KeychainStore), and an item's access list trusts the app by its designated requirement. An
#   ad-hoc signature ("-") has none that survives a rebuild, so every new build would be asked
#   "Vergence wants to use your confidential information" again.
# - TCC: the Local Network grant (reaching Jellyfin or Stash on this network, serving headsets)
#   is kept per signing identity too, and an ad-hoc build looks like a new app each time.
# Signing the app and its `vergence` command line with your Apple Development identity keeps
# both across rebuilds. With none found it signs ad-hoc and says so.
#
#   desktop/scripts/install-desktop.sh
#
# Options (env):
#   SIGN_IDENTITY="…"   codesign identity. Default: the first Apple Development identity in the
#                       login keychain, else "-" (ad-hoc). A name or a 40-character SHA-1 pins one.
#   DEST_DIR=…          where the app goes (default /Applications)
#   LAUNCH=0            install without opening the app
#   VERBOSE=1           stream the build instead of logging it (build/install-desktop.log)

set -euo pipefail

APP_NAME="Vergence"
BUNDLE_ID="com.illixion.vergence.desktop"

if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
  sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
fi
[[ "$(uname -s)" == Darwin ]] || { echo "✗ This installs the macOS app; run it on a Mac." >&2; exit 1; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
DEST_DIR="${DEST_DIR:-/Applications}"
DEST="$DEST_DIR/$APP_NAME.app"
BUILD_LOG="$ROOT/build/install-desktop.log"
TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
cd "$ROOT"

if [[ -z "${SIGN_IDENTITY:-}" ]]; then
  SIGN_IDENTITY="$(security find-identity -v -p codesigning 2>/dev/null \
    | awk -F'"' '/Apple Development/ { print $2; exit }')"
  if [[ -n "$SIGN_IDENTITY" ]]; then
    echo "▶ Signing identity: $SIGN_IDENTITY"
  else
    SIGN_IDENTITY="-"
    echo "⚠ No 'Apple Development' identity found: signing ad-hoc (\"-\")." >&2
    echo "  The Keychain and Local Network will ask again after every rebuild. Open Xcode once" >&2
    echo "  to provision an identity, or pass SIGN_IDENTITY." >&2
  fi
fi

command -v cargo >/dev/null || { echo "✗ cargo not found; install Rust (rustup.rs)." >&2; exit 1; }
command -v cargo-tauri >/dev/null || { echo "✗ The Tauri CLI isn't installed (cargo install tauri-cli --version '^2' --locked)." >&2; exit 1; }

mkdir -p "$(dirname "$BUILD_LOG")"
: >"$BUILD_LOG"
run() {
  if [[ "${VERBOSE:-0}" == 1 ]]; then "$@"; else "$@" >>"$BUILD_LOG" 2>&1; fi
}
fail() {
  if [[ "${VERBOSE:-0}" != 1 ]]; then
    grep -E 'error\[|error:|^error' "$BUILD_LOG" | sort -u | head -30 >&2 || true
  fi
  echo "✗ Build failed. Full log: $BUILD_LOG" >&2
  exit 1
}

# A deployment target reaches the proc-macro dylibs built for this Mac too, and Xcode 27's linker
# then writes ones dlopen refuses ("mis-aligned LINKEDIT string pool"); the bundle sets its own
# floor (tauri.conf.json). If such a dylib is already in target/ from an earlier build, run
# `cargo clean --release -p <crate>` for each one the error names.
unset MACOSX_DEPLOYMENT_TARGET

echo "▶ Building the app (release, $(uname -m))…"
# custom-protocol serves the bundled ui/; vendored-openssl builds OpenSSL into the app, so it
# doesn't depend on Homebrew's (package-macos.sh explains both).
run cargo build --release -p vergence-desktop \
  --features vergence-desktop/custom-protocol,vergence-companion/vendored-openssl || fail

echo "▶ Building the vergence command line (release)…"
run swift build -c release --product vergence || fail
CLI="$(swift build -c release --show-bin-path)/vergence"
[[ -x "$CLI" ]] || { echo "✗ No command line at $CLI" >&2; exit 1; }

# tauri bundle looks for the binary under tauri.conf.json's mainBinaryName (Vergence), not the
# cargo name; it is removed again after bundling, so it never shadows anything in target/.
cp "$TARGET_DIR/release/vergence-app" "$TARGET_DIR/release/$APP_NAME"
rm -rf "$TARGET_DIR/release/bundle/macos"
echo "▶ Bundling $APP_NAME.app…"
(cd desktop && run cargo tauri bundle --bundles app) || { rm -f -- "$TARGET_DIR/release/$APP_NAME"; fail; }
rm -f -- "$TARGET_DIR/release/$APP_NAME"
BUNDLED="$TARGET_DIR/release/bundle/macos/$APP_NAME.app"
[[ -d "$BUNDLED" ]] || { echo "✗ Bundled app not found at: $BUNDLED" >&2; exit 1; }

# Staged, so a failure leaves the installed app alone.
STAGE="$ROOT/build/install-desktop-stage"
rm -rf -- "$STAGE"
mkdir -p "$STAGE"
ditto "$BUNDLED" "$STAGE/$APP_NAME.app"
# In Resources/, where the app looks for it (sidecar::find_cli): not MacOS/, where a
# case-insensitive volume would merge `vergence` with `Vergence`.
mkdir -p "$STAGE/$APP_NAME.app/Contents/Resources"
cp "$CLI" "$STAGE/$APP_NAME.app/Contents/Resources/vergence"

echo "▶ Signing with $SIGN_IDENTITY…"
# The command line first, with the same identity: it runs as the app's worker and converter,
# and reads the media servers on this network under the app's grant.
codesign -f -s "$SIGN_IDENTITY" "$STAGE/$APP_NAME.app/Contents/Resources/vergence"
codesign -f -s "$SIGN_IDENTITY" --entitlements desktop/Entitlements.plist "$STAGE/$APP_NAME.app"
codesign --verify --deep --strict "$STAGE/$APP_NAME.app" || { echo "✗ The signed app doesn't verify." >&2; exit 1; }

echo "▶ Quitting the running app (if any)…"
# By bundle id, only when it runs (so AppleScript doesn't launch it to quit it); nothing is
# killed by name. If it ignores the quit, say so.
running() { pgrep -f "^$DEST/Contents/MacOS/$APP_NAME( |\$)" >/dev/null 2>&1; }
if running; then
  osascript -e "tell application id \"$BUNDLE_ID\" to quit" 2>/dev/null || true
  for _ in $(seq 1 20); do
    running || break
    sleep 0.3
  done
  if running; then
    echo "✗ $APP_NAME is still running after a quit request. Quit it from its menu-bar icon and run this again." >&2
    exit 1
  fi
fi

echo "▶ Installing to ${DEST}…"
NEW="$DEST.new"
rm -rf -- "$NEW"
ditto "$STAGE/$APP_NAME.app" "$NEW"
if [[ -e "$DEST" ]]; then
  # renamex_np(RENAME_SWAP) exchanges both at once; where it isn't there, two renames.
  if /usr/bin/python3 -c 'import ctypes,sys; l=ctypes.CDLL(None, use_errno=True); sys.exit(0 if l.renamex_np(sys.argv[1].encode(), sys.argv[2].encode(), 2)==0 else 1)' "$NEW" "$DEST" 2>/dev/null; then
    rm -rf -- "$NEW"   # now holds the previous app
  else
    mv -- "$DEST" "$DEST.old" && mv -- "$NEW" "$DEST" && rm -rf -- "$DEST.old"
  fi
else
  mv -- "$NEW" "$DEST"
fi
rm -rf -- "$STAGE"

if [[ "${LAUNCH:-1}" == 0 ]]; then
  echo "✓ $APP_NAME installed to $DEST_DIR (not launched)."
else
  echo "▶ Launching…"
  open "$DEST"
  echo "✓ $APP_NAME installed to $DEST_DIR and started."
fi
