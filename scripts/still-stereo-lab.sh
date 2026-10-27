#!/bin/bash
set -euo pipefail
if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    echo 'Usage: still-stereo-lab.sh IMAGE MODEL OUTPUT_DIRECTORY [LONG_SIDE=2048]' >&2
    exit 2
fi
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export VERGENCE_STILL_IMAGE="$1"
export VERGENCE_STILL_MODEL="$2"
export VERGENCE_STILL_OUTPUT="$3"
export VERGENCE_STILL_SIZE="${4:-2048}"
swift test --package-path "$script_dir/.." --filter StillStereoLabTests/exportComparison
