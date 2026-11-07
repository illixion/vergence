#!/bin/bash
# Regenerates convert/tests/fixtures/core.json from VergenceCore's own depth
# maths (see make-convert-fixtures.swift), for vergence-convert's Rust port.
set -euo pipefail
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
work=$(mktemp -d)
trap 'rm -rf -- "$work"' EXIT
mkdir -p "$root/convert/tests/fixtures"
# Top-level code has to be main.swift when several files are compiled.
cp "$root/scripts/make-convert-fixtures.swift" "$work/main.swift"
swiftc -O -o "$work/fixtures" -module-name Fixtures \
    "$root/Sources/VergenceCore/DepthRefinement.swift" \
    "$root/Sources/VergenceCore/DepthAnchor.swift" \
    "$root/Sources/VergenceCore/DepthHistogram.swift" \
    "$work/main.swift"
"$work/fixtures" "$root/convert/tests/fixtures/core.json"
