# vergence-desktop

The Vergence desktop app (Tauri 2, binary `vergence-app`) for Mac and Windows. One tray
(menu-bar) app that:

- **runs the analysis coordinator** in-process: the same `vergence-coordinator` axum server
  `vergenced` runs headless, on 127.0.0.1:8420 by default (Settings can move the port, or listen on
  the network), with its store under the app-data folder;
- **signs in to Jellyfin and Stash**: addresses in `settings.json`, API keys in `secrets/*.key`
  (0600). The page sends a key once and only ever sees "set, ends in …abcd";
- **shows the queue** (titles with progress, leases, workers, playheads) and manages `play` and
  `work` tokens, each shown once;
- **pairs headsets** with RAVECompanion's six-digit comparison (`vergence-companion`, port 4869,
  open only while pairing). A paired headset then reaches the coordinator over TLS-PSK on port
  8421 (every interface, `pairedPort` in `settings.json`), keyed from its pairing: see below;
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

## macOS bundle

```sh
desktop/scripts/package-macos.sh        # -> target/package/Vergence-<version>-macos-universal.zip
```

One universal (arm64 + x86_64) `Vergence.app` with the `vergence` command line in
`Contents/Resources/` (where `find_cli` looks), ad-hoc sealed, zipped with `ditto` so the updater's
`ditto -x -k` and `codesign --verify --deep --strict` accept it. OpenSSL is built in
(`vendored-openssl`): linking Homebrew's would leave `/opt/homebrew` paths in the app. Needs the
Tauri CLI and both Apple Rust targets. Not notarized; a downloaded copy needs its quarantine flag
cleared (`xattr -dr com.apple.quarantine Vergence.app`) or Open Anyway in System Settings.

Don't set `MACOSX_DEPLOYMENT_TARGET` for the build: it reaches the proc-macro dylibs built for the
Mac itself, and Xcode 27's linker writes ones `dlopen` refuses ("mis-aligned LINKEDIT string
pool", shown as "can't find crate"). The 14.0 floor is the bundle's (`minimumSystemVersion`).

## Windows

The same app, cross-compiled from a Mac and packed the way Apparition packs its Windows
installer: one zip, `Vergence-<version>-windows-x86_64.zip` (the name the updater looks for).

```sh
brew install mingw-w64 && rustup target add x86_64-pc-windows-gnu    # once
desktop/scripts/package-windows.sh                                    # -> target/package/
desktop/scripts/package-windows.sh --tools DIR    # also packs every .exe and .dll in DIR
```

It holds `Vergence.exe`, `WebView2Loader.dll` (keep it beside the app) and a README. OpenSSL is
built into the executable (`vendored-openssl` on Windows), since Windows has none to link. Unpack
it anywhere and open `Vergence.exe`; the window is Microsoft Edge WebView2, which Windows 10 and
11 have.

On Windows the app runs the Rust tools instead of the Swift command line, with the same config
file and flags (`src/sidecar.rs`, `find_tool`):

- the worker is `vergence-worker.exe --config FILE` (ONNX Runtime, `worker/`);
- conversion is `vergence-convert.exe <file> --layout … ` (`convert/`), given
  `VERGENCE_PROGRESS=1` because its stderr is a pipe, not a terminal, there.

Both are looked for in Settings (the file, or the folder holding it), `VERGENCE_WORKER` /
`VERGENCE_CONVERT`, beside `Vergence.exe`, then this checkout's `target/release/`; ffmpeg beside
the app, then `PATH`. Build them on Windows (they need ONNX Runtime's Windows binaries) and pack
them with `--tools`.

While pairing is open the app announces `_vergence-pair._tcp` through Windows' own mDNS responder
(`DnsServiceRegister`, Windows 10 1809 and later; nothing to install), as it does through
`dns-sd` on a Mac, so a Quest finds it without typing an address.

Windows Defender Firewall asks on first start whether Vergence may use the network: private
networks are enough for headsets to pair (4869) and reach the paired channel (8421).

## Tests

`cargo test -p vergence-desktop` covers settings and keys, the coordinator host (health, tokens
across restarts, a taken port), pairing a device over TCP and its token, the worker's config,
the convert command line, progress parsing and queue persistence, the sidecar's terminal and
stop, and the updater's signature checks. `cargo test -p vergence-desktop -- --ignored
converts_a_clip` runs a real conversion of a generated clip (needs the release CLI, ffmpeg and a
depth model).

## How a paired headset reaches the coordinator

Over the paired channel: TLS 1.2 with an external PSK on port 8421, keyed from the pairing token
the headset got, the PSK identity naming the device (`VergenceAPI/v1/<device id>`), then plain
HTTP/1.1 to the same routes. The coordinator reads `devices.json` itself, so a new pairing works
at once and Forget locks the device out at once, open connections included. The exact wire is in
`coordinator/src/paired.rs` and Convolution's `docs/plans/analysis-workers.md` ("The paired
channel"). It needs no *Listen on the network*: that switch is for the plain HTTP port.

A paired device may instead use the plain HTTP port with a bearer token derived from its pairing
token, never the pairing token itself (which keys the PSK channel and mustn't cross the network):

```text
token = hex(HKDF-SHA256(ikm: pairing token (UTF-8), salt: "Vergence-HTTP-Token-v1",
                        info: device id (UTF-8), length 32))
```

The coordinator recognises it from `devices.json`, and also keeps its SHA-256 as `device-<device
id>` with the `play` grant; forgetting the device removes both.

## CI and release builds

`.github/workflows/ci.yml` runs the desktop, coordinator and companion tests and clippy on macOS and
Windows for every push to `main` and pull request. `.github/workflows/release.yml` (run by hand, or
by pushing a `v<version>` tag) builds both release zips with the scripts above, the Windows worker
and converter natively on Windows, and attests them (`gh attestation verify FILE -R
illixion/vergence`). It signs nothing: download the `release-<version>` artifact and run
`bless-release.sh SHA256SUMS` below. A manual run with no version builds without publishing, which
also warms the caches (only runs on `main` save them). The repository variable `MACOS_RUNNER` picks a
runner with Xcode 27 for the Swift command line.

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

- **Bonjour** is only the pairing announcement (macOS, Linux, Windows); the coordinator's own ports
  aren't announced, so a paired device remembers the address it paired over.
- **Windows:** no installer that writes Start menu entries or registers "Open with" (the zip is
  unpacked anywhere, as Apparition's is), no code signing (SmartScreen warns once), and the
  firewall asks on first start; the tray and a conversion haven't been tried there yet.
- **Updates:** there is no release site yet (`updater::SITE` is `None`), so checking is off by
  default, and the Windows install step isn't ported. The signing keys are pinned (below).
- **Titles by name** in the queue (it shows server and id), a preview frame before converting,
  model and GPU choice, yielding to games, and a signed bundle with the CLI as a sidecar.
- `vergence convert` prints progress only to a terminal, so the app gives its stderr a
  pseudo-terminal; a machine-readable progress option in the CLI would be cleaner.
