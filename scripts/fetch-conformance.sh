#!/bin/sh
# Downloads the ITU-T H.264 MVC conformance bitstreams (36 MB) used by the
# conformance tests. They are not redistributed here.
#   scripts/fetch-conformance.sh [DIR]     (default: .build/conformance/mvc)
#   MVC3D_CONFORMANCE=.build/conformance/mvc swift test
set -eu
dir=${1:-.build/conformance/mvc}
base=https://www.itu.int/wftp3/av-arch/jvt-site/draft_conformance/MVC/
mkdir -p "$dir"
for f in MVCDS1 MVCDS2 MVCDS3 MVCDS-4 MVCDS-5 MVCDS-6 MVCRP_1 MVCRP_2 MVCRP_3 MVCRP_4 MVCRP_5 MVCRP_6 MVCSPS-1; do
  [ -s "$dir/$f.264" ] || curl -fsSL -o "$dir/$f.264" "$base$f.264"
done
echo "MVC3D_CONFORMANCE=$dir"
