# vergence-desktop

The Vergence desktop app (Tauri 2, binary `vergence-app`), Mac first, then Windows. One tray
(menu-bar) app that:

- **runs the analysis coordinator** in-process: the same `vergence-coordinator` axum server
  `vergenced` runs headless, on 127.0.0.1:8420 by default (Settings can move the port, or listen on
  the network), with its store under the app-data folder;
- **signs in to Jellyfin and Stash**: addresses in `settings.json`, API keys in `secrets/*.key`
  (0600). The page sends a key once and only ever sees "set, ends in …abcd";
- **shows the queue** (titles with progress, leases, workers, playheads) and manages `play` and
  `work` tokens, each shown once;
- **pairs headsets** with RAVECompanion's six-digit comparison (`vergence-companion`, port 4869,
  open only while pairing). A paired device gets a coordinator `play` token: see below;
- **runs this Mac's worker** (`vergence worker --config …`) as a child process, with a `work`
  token it mints for it, and shows its output;
- **converts files to AI 3D**: drop videos or folders on the window, pick them, or "Open With"
  from Finder. One `vergence convert` runs at a time, its progress read from its output; the
  queue (`convert-queue.json`) survives a quit and carries on at the next start.

Design: Hypnos `docs/plans/analysis-workers.md`, "The Vergence desktop app". The shell (tray,
autostart with `--hidden`, single instance, the window closing to the tray, the bounded command
set, the SSHSIG-verified updater) is adapted from Apparition's desktop app (MIT, same author).

## Running it

```sh
swift build -c release                    # the vergence command line, found from the checkout
cargo run -p vergence-desktop
```

The `vergence` executable is looked for in Settings, `VERGENCE_CLI`, beside the app (a bundle's
`MacOS/` or `Resources/`), then this checkout's `.build/release/`. ffmpeg comes from `PATH`,
`/opt/homebrew/bin` or `/usr/local/bin` (an app started from Finder has no Homebrew on its PATH).

`VERGENCE_APP_DATA=DIR` moves the app's data (for a second copy, or a test run). In a debug build
`VERGENCE_APP_CONVERT=a.mp4:b.mkv` queues files to convert at launch.

## Tests

`cargo test -p vergence-desktop` covers settings and keys, the coordinator host (health, tokens
across restarts, a taken port), pairing a device over TCP and its token, the worker's config,
the convert command line, progress parsing and queue persistence, the sidecar's terminal and
stop, and the updater's signature checks. `cargo test -p vergence-desktop -- --ignored
converts_a_clip` runs a real conversion of a generated clip (needs the release CLI, ffmpeg and a
depth model).

## A paired headset's token (interim)

Until the chunk API runs over the paired TLS-PSK channel, a paired device reaches the coordinator
with a bearer token over HTTP on the LAN. It is derived from its pairing token, never the pairing
token itself (which keys the PSK channel and mustn't cross the network):

```text
token = hex(HKDF-SHA256(ikm: pairing token (UTF-8), salt: "Vergence-HTTP-Token-v1",
                        info: device id (UTF-8), length 32))
```

The coordinator keeps its SHA-256 as `device-<device id>` with the `play` grant; forgetting the
device removes it.

## Signing a release

The updater trusts a release only through its `SHA256SUMS`, signed with SSHSIG under the
namespace `vergence-release` by a key in `release-signers`, which is compiled in. Those are the
same two keys that sign Apparition's releases: the everyday hardware key (PIN and touch, never on
disk or in CI) and an offline backup, both pinned from the start so the backup is trusted by every
copy installed. To sign a release's manifest, in the foreground:

```sh
desktop/scripts/bless-release.sh build/release/0.2.0/SHA256SUMS                 # the hardware key, via ssh-agent or gpg-agent
desktop/scripts/bless-release.sh build/release/0.2.0/SHA256SUMS --key-file KEY  # the offline backup
```

It writes `SHA256SUMS.sig` beside the manifest and checks it with `ssh-keygen -Y verify` against
`release-signers`. A release lists `Vergence-<version>-macos-universal.zip` and
`Vergence-<version>-windows-x86_64.zip` (`updater::installer_name`); the version is in the
signed names, so an old release can't be relabelled as new.

## Not done yet

- **The TLS-PSK channel** for the chunk and worker API (pairing and the derived HTTP token work).
- **Bonjour**: devices pair from the address link the Headsets tab shows.
- **Windows:** the worker and conversion arrive with the ONNX Runtime engine; the app shows so.
  The `cfg`s are in place, but no Windows build has been made.
- **Updates:** there is no release site yet (`updater::SITE` is `None`), so checking is off by
  default, and the Windows install step isn't ported. The signing keys are pinned (below).
- **Titles by name** in the queue (it shows server and id), a preview frame before converting,
  model and GPU choice, yielding to games, and a signed bundle with the CLI as a sidecar.
- `vergence convert` prints progress only to a terminal, so the app gives its stderr a
  pseudo-terminal; a machine-readable progress option in the CLI would be cleaner.
