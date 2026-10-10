#!/usr/bin/env bash
# Cross-compiles Vergence's portable layer for Android arm64 (Quest, phones).
#
#   SWIFT_ANDROID_BIN=/path/to/swift-6.3.3/usr/bin/swift \
#   SWIFT_ANDROID_BUNDLE=/path/to/swift-6.3.3-RELEASE_android.artifactbundle \
#   scripts/build-android.sh [debug|release]
#
# SWIFT_ANDROID_BIN is the open-source toolchain matching the Android SDK
# bundle (Xcode's swift can't use it); the bundle needs its NDK sysroot set
# up once with its own swift-android/scripts/setup-android-sdk.sh.
# SwiftPM 6.3.3 can pick any architecture from the bundle's several triples,
# so this describes exactly one in a bundle of its own under .build and
# leaves the installed one untouched.
set -euo pipefail
: "${SWIFT_ANDROID_BIN:?Set SWIFT_ANDROID_BIN to the open-source swift matching the Android SDK}"
: "${SWIFT_ANDROID_BUNDLE:?Set SWIFT_ANDROID_BUNDLE to the Android artifactbundle directory}"
configuration="${1:-debug}"
package="$(cd "$(dirname "$0")/.." && pwd)"
triple=aarch64-unknown-linux-android29
sdks="$package/.build/android-sdk"

/usr/bin/python3 - "$SWIFT_ANDROID_BUNDLE" "$sdks" "$triple" <<'PY'
import json, sys
from pathlib import Path
source, destination, triple = Path(sys.argv[1]).resolve() / "swift-android", Path(sys.argv[2]), sys.argv[3]
bundle = destination / "vergence-android-arm64.artifactbundle"
bundle.mkdir(parents=True, exist_ok=True)
target = json.loads((source / "swift-sdk.json").read_text())["targetTriples"][triple]
for key in ("sdkRootPath", "swiftResourcesPath", "swiftStaticResourcesPath"):
    target[key] = str(source / target[key])
target["toolsetPaths"] = [str(source / p) for p in target["toolsetPaths"]]
(bundle / "swift-sdk.json").write_text(json.dumps({"schemaVersion": "4.0", "targetTriples": {triple: target}}, indent=2))
(bundle / "info.json").write_text(json.dumps({"schemaVersion": "1.0", "artifacts": {
    "vergence-android-arm64": {"type": "swiftSDK", "version": "0.1", "variants": [{"path": "."}]}}}, indent=2))
PY

"$SWIFT_ANDROID_BIN" build --package-path "$package" \
  --swift-sdks-path "$sdks" --swift-sdk vergence-android-arm64 --triple "$triple" \
  --scratch-path "$package/.build/android" -c "$configuration"
