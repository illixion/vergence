#!/usr/bin/env bash
#
# Signs a Vergence release with a pinned release key. A thin wrapper: the script is the
# signed-update crate's (../signed-update/scripts/bless-release.sh, shared with Apparition), given
# this app's namespace, signers and repository.
#
#   desktop/scripts/bless-release.sh [--tag v0.2.0] [--publish]       # sign the GitHub draft release, upload, publish
#   desktop/scripts/bless-release.sh --key-file KEY                   # the offline backup key, no agent
#   desktop/scripts/bless-release.sh --manifest SHA256SUMS            # sign a manifest you already have
#
# CI (release.yml) builds the files into a draft release; this hashes them into one SHA256SUMS,
# signs that with SSHSIG under the namespace "vergence-release" (the hardware key: a PIN and a
# touch, so run it in the foreground), checks the signature against desktop/release-signers, and
# uploads SHA256SUMS and SHA256SUMS.sig. --publish then publishes the draft, which is when apps
# find it: apps.illixion.com redirects to GitHub's `releases/latest`, which never names a draft.
# --help lists every option.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED="${SIGNED_UPDATE:-$HERE/../../../signed-update}/scripts/bless-release.sh"
[[ -x "$SHARED" ]] || { echo "error: signed-update isn't beside this repository (set SIGNED_UPDATE to its folder): $SHARED" >&2; exit 1; }
exec "$SHARED" --repo illixion/vergence --namespace vergence-release --signers "$HERE/../release-signers" "$@"
