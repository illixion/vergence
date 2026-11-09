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

The same app, packed the way Apparition packs its Windows installer: one zip per architecture,
`Vergence-<version>-windows-x86_64.zip` and `Vergence-<version>-windows-arm64.zip` (the names the
updater looks for; an x86_64 copy running emulated on an ARM PC keeps updating to x86_64). The
x86_64 one is cross-compiled from a Mac:

```sh
brew install mingw-w64 && rustup target add x86_64-pc-windows-gnu    # once
desktop/scripts/package-windows.sh                                    # -> target/package/
desktop/scripts/package-windows.sh --tools DIR    # also packs every .exe and .dll in DIR
```

The ARM64 one is built natively, in Git Bash on a Windows ARM PC (`aarch64-pc-windows-msvc`; mingw
has no ARM64 toolchain), with the Visual Studio C++ tools, LLVM's `clang` on `PATH` (ring and aws-lc
compile their C and assembly with it there) and a Windows Perl for OpenSSL (Strawberry Perl; Git
Bash's own can't configure it, so point `OPENSSL_SRC_PERL` at it). The script also sets
`CPPFLAGS=-Gs` for OpenSSL: MSVC's ARM64 compiler miscompiles it under the `/Gs0` its config passes,
and the paired channel then crashes on every connection (any other build of OpenSSL for
`aarch64-pc-windows-msvc` needs the same, `cargo test` included):

```sh
desktop/scripts/package-windows.sh --arch arm64 --tools DIR
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

`.github/workflows/ci.yml` runs the desktop, coordinator and companion tests and clippy on macOS,
Windows and Windows on ARM (`windows-11-arm`) for every push to `main` and pull request. `.github/workflows/release.yml` (run by hand, or
by pushing a `v<version>` tag) builds the three release zips with the scripts above (the Windows
worker and converter natively on Windows, and all of the ARM64 zip natively on `windows-11-arm`),
attests them (`gh attestation verify FILE -R
illixion/vergence`) and puts them on a **draft** GitHub release. It signs and publishes nothing: see
"Releasing" below. A manual run with no version builds without a release, which also warms the caches
(only runs on `main` save them). The repository variable `MACOS_RUNNER` picks a runner with Xcode 27
for the Swift command line. The workspace depends on `../signed-update` by path (below), so both
workflows check that repository out beside this one.

## Releasing

1. Bump the version in `Cargo.toml`, commit, push a `v<version>` tag (or run the workflow by hand).
   CI builds the files into a draft release.
2. `desktop/scripts/bless-release.sh --tag v<version> --publish`, in the foreground: it checks each
   file's build attestation and that every file is a `Vergence-<version>-…` one, hashes the
   draft's files, signs them with the release key (PIN and touch), checks the signature against
   `release-signers`, uploads `SHA256SUMS` and `SHA256SUMS.sig` and publishes. `--key-file KEY` signs
   with the offline backup instead.
3. Installed copies find it at their next check.

## Updates

The updater is the [`signed-update`](../../signed-update) crate (`docs/PROTOCOL.md` there), shared
with Apparition; `src/updater.rs` is only the status the window shows and when to install. A copy
asks `https://apps.illixion.com/vergence/latest/SHA256SUMS` (and `.sig`) first, then GitHub's
`releases/latest/download/` directly, and believes the first signed manifest that verifies against
the pinned keys. apps.illixion.com holds no files: its `_redirects` point at GitHub Releases, and at
a self-hosted copy while GitHub is down, so moving every copy is a two-line edit there and needs no
app update. Checking is off until the person turns it on (Settings).

"Hide the update check from the network" (on by default) keeps apps.illixion.com's name off a network
you travel through: its address and ECH key are read over DNS-over-HTTPS from 1.1.1.1 by IP, and the
connection uses Encrypted Client Hello, so a log shows a connection to 1.1.1.1 and a handshake naming
Cloudflare's shared `cloudflare-ech.com`, then github.com (which names no one). Where DoH or ECH is
blocked the check goes to GitHub directly: never to apps.illixion.com in the clear. Verified on a wire
capture and on Windows 11 (`signed-update/docs/PROTOCOL.md`, "Privacy"). macOS installs the update in
place (unzip beside the app, check its code-signing seal, swap); Windows says where to download it.

## Signing a release

`desktop/scripts/bless-release.sh` (a wrapper over the shared script in `signed-update`) signs a
release with SSHSIG under the namespace `vergence-release`, with a key in `release-signers`, which is
compiled in. Those are the same two keys that sign Apparition's releases: the everyday hardware key
(PIN and touch, never on disk or in CI) and an offline backup, both pinned from the start so the
backup is trusted by every copy installed. A release lists `Vergence-<version>-macos-universal.zip`
and `Vergence-<version>-windows-{x86_64,arm64}.zip`; the version is in the signed names, so an old release
can't be relabelled as new.

## Not done yet

- **Bonjour** is only the pairing announcement (macOS, Linux, Windows); the coordinator's own ports
  aren't announced, so a paired device remembers the address it paired over.
- **Windows:** no installer that writes Start menu entries or registers "Open with" (the zip is
  unpacked anywhere, as Apparition's is), no code signing (SmartScreen warns once), and the
  firewall asks on first start; the tray and a conversion haven't been tried there yet.
- **Updates:** the Windows install step isn't ported (Apparition's `update-windows.ps1` swaps a
  running executable); the macOS one is in `signed-update`. The first signed release is yet to be made.
- **Titles by name** in the queue (it shows server and id), a preview frame before converting,
  model and GPU choice, yielding to games, and a signed bundle with the CLI as a sidecar.
- `vergence convert` prints progress only to a terminal, so the app gives its stderr a
  pseudo-terminal; a machine-readable progress option in the CLI would be cleaner.
