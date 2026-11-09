//! Running the `vergence` command line as a child process: finding it (and ffmpeg), reading its
//! output line by line, and stopping it, only ever by that child's own PID.
//!
//! Which program does what ([`find_tool`]):
//! - **Mac:** `vergence worker` (Core ML) and `vergence convert`, both the Swift command line.
//! - **Windows:** `vergence-worker.exe` (ONNX Runtime, `worker/`) and `vergence-convert.exe`
//!   (ONNX Runtime depth, a wgpu warp, ffmpeg; `convert/`), which take the same config file and
//!   the same flags as the Mac's, with no subcommand word. They are looked for in Settings (the
//!   file, or the folder holding it), `VERGENCE_WORKER` / `VERGENCE_CONVERT`, beside the app, then
//!   this checkout's `target/release/`.
//!
//! `vergence convert` prints its progress (`\r  depth  42%`) only when stderr is a terminal, so
//! a conversion's stderr is a pseudo-terminal on Unix, read here like any other output.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::{ExitStatus, Stdio};
#[cfg(unix)]
use std::time::Duration;

use tokio::io::AsyncReadExt;
use tokio::sync::{mpsc, oneshot};

use crate::settings::Settings;

/// Where Homebrew and friends put tools: an app started from Finder gets only the system's
/// PATH, so the sidecar would find no ffmpeg without these.
#[cfg(target_os = "macos")]
const EXTRA_PATH: &[&str] = &["/opt/homebrew/bin", "/usr/local/bin"];
#[cfg(not(target_os = "macos"))]
const EXTRA_PATH: &[&str] = &[];

/// The `vergence` executable: the one set in Settings, `VERGENCE_CLI`, one bundled beside the
/// app, else the repository's release build (development).
pub fn find_cli(settings: &Settings) -> Option<PathBuf> {
    let name = if cfg!(windows) { "vergence.exe" } else { "vergence" };
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(path) = &settings.vergence_cli {
        candidates.push(expand_tilde(path));
    }
    if let Some(path) = std::env::var_os("VERGENCE_CLI") {
        candidates.push(PathBuf::from(path));
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        candidates.push(dir.join(name));
        // A macOS bundle's Resources folder.
        candidates.push(dir.join("../Resources").join(name));
    }
    // This checkout's `swift build -c release`, then the usual place of one.
    candidates.push(Path::new(env!("CARGO_MANIFEST_DIR")).join("../.build/release").join(name));
    if let Some(home) = home_dir() {
        candidates.push(home.join("Projects/vergence/.build/release").join(name));
    }
    candidates.into_iter().find(|p| p.is_file()).map(|p| p.canonicalize().unwrap_or(p))
}

/// What the app runs a tool as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolKind {
    Worker,
    Convert,
}

/// A tool to run: the program, and the words before its own arguments (`worker`, `convert` on
/// a Mac; none for the Windows executables).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tool {
    pub program: PathBuf,
    pub prefix: Vec<OsString>,
}

impl Tool {
    /// The full argument list for `args`.
    pub fn arguments(&self, args: impl IntoIterator<Item = OsString>) -> Vec<OsString> {
        self.prefix.iter().cloned().chain(args).collect()
    }
}

/// The program that runs `kind` on this platform (see the module docs), or None when it isn't
/// there.
pub fn find_tool(settings: &Settings, kind: ToolKind) -> Option<Tool> {
    if cfg!(windows) {
        let (name, var) = match kind {
            ToolKind::Worker => ("vergence-worker.exe", "VERGENCE_WORKER"),
            ToolKind::Convert => ("vergence-convert.exe", "VERGENCE_CONVERT"),
        };
        find_executable(settings, name, var).map(|program| Tool { program, prefix: Vec::new() })
    } else {
        let word = match kind {
            ToolKind::Worker => "worker",
            ToolKind::Convert => "convert",
        };
        find_cli(settings).map(|program| Tool { program, prefix: vec![word.into()] })
    }
}

/// What to tell the person when a tool isn't found.
pub fn missing_tool(kind: ToolKind) -> &'static str {
    match (cfg!(windows), kind) {
        (true, ToolKind::Worker) => "vergence-worker.exe wasn't found: put it beside the app, or give its folder in Settings.",
        (true, ToolKind::Convert) => "vergence-convert.exe wasn't found: put it beside the app, or give its folder in Settings.",
        (false, _) => "No vergence command line found: build it (swift build -c release) or set its path in Settings.",
    }
}

/// A Windows tool by name: the Settings path (a file, or the folder holding it), its
/// environment variable, beside the app, then this checkout's Rust release build.
fn find_executable(settings: &Settings, name: &str, var: &str) -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(path) = &settings.vergence_cli {
        let path = expand_tilde(path);
        if path.is_dir() {
            candidates.push(path.join(name));
        } else if path.file_name().is_some_and(|n| n.eq_ignore_ascii_case(name)) {
            candidates.push(path.clone());
        } else if let Some(dir) = path.parent() {
            candidates.push(dir.join(name));
        }
    }
    if let Some(path) = std::env::var_os(var) {
        candidates.push(PathBuf::from(path));
    }
    if let Some(dir) = std::env::current_exe().ok().and_then(|exe| exe.parent().map(Path::to_path_buf)) {
        candidates.push(dir.join(name));
    }
    candidates.push(Path::new(env!("CARGO_MANIFEST_DIR")).join("../target/release").join(name));
    candidates.into_iter().find(|p| p.is_file())
}

pub fn find_ffmpeg() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("VERGENCE_FFMPEG").map(PathBuf::from).filter(|p| p.is_file()) {
        return Some(path);
    }
    let name = if cfg!(windows) { "ffmpeg.exe" } else { "ffmpeg" };
    // One shipped beside the app (Windows has no Homebrew), then PATH.
    let beside = std::env::current_exe().ok().and_then(|exe| exe.parent().map(|d| d.join(name)));
    beside.into_iter().chain(search_path().into_iter().map(|dir| dir.join(name))).find(|p| p.is_file())
}

fn search_path() -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = std::env::var_os("PATH").map(|p| std::env::split_paths(&p).collect()).unwrap_or_default();
    for extra in EXTRA_PATH {
        let extra = PathBuf::from(extra);
        if !dirs.contains(&extra) {
            dirs.push(extra);
        }
    }
    dirs
}

pub fn home_dir() -> Option<PathBuf> {
    std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" }).map(PathBuf::from)
}

pub fn expand_tilde(path: &str) -> PathBuf {
    match (path.strip_prefix("~/"), home_dir()) {
        (Some(rest), Some(home)) => home.join(rest),
        _ => PathBuf::from(path),
    }
}

/// Splits output into lines at `\n` and at `\r` (a progress line redrawn in place), dropping
/// empty ones.
#[derive(Default)]
pub struct LineSplitter {
    pending: Vec<u8>,
}

impl LineSplitter {
    pub fn push(&mut self, bytes: &[u8]) -> Vec<String> {
        let mut lines = Vec::new();
        for &b in bytes {
            if b == b'\n' || b == b'\r' {
                self.flush_into(&mut lines);
            } else {
                self.pending.push(b);
            }
        }
        // A progress line is written without an ending; don't hold it back.
        if self.pending.ends_with(b"%") {
            self.flush_into(&mut lines);
        }
        lines
    }

    pub fn finish(&mut self) -> Vec<String> {
        let mut lines = Vec::new();
        self.flush_into(&mut lines);
        lines
    }

    fn flush_into(&mut self, lines: &mut Vec<String>) {
        let line = String::from_utf8_lossy(&self.pending).trim().to_string();
        self.pending.clear();
        if !line.is_empty() {
            lines.push(line);
        }
    }
}

/// A running sidecar: its output lines, a way to stop it, and its exit.
pub struct Sidecar {
    pub pid: Option<u32>,
    pub lines: mpsc::UnboundedReceiver<String>,
    stop: Option<oneshot::Sender<()>>,
    pub exit: tokio::task::JoinHandle<std::io::Result<ExitStatus>>,
}

impl Sidecar {
    /// What stops it: SIGTERM to its PID, SIGKILL if it is still there 3 s later. Kept apart
    /// from the lines and the exit, so one task can read while another stops it.
    pub fn take_stopper(&mut self) -> Option<oneshot::Sender<()>> {
        self.stop.take()
    }
}

/// Starts `program args` with stdout piped, and stderr on a pseudo-terminal when `tty` (Unix).
pub fn spawn(program: &Path, args: &[OsString], tty: bool) -> std::io::Result<Sidecar> {
    spawn_with_input(program, args, tty, None)
}

/// As [`spawn`], writing `input` to the child's standard input and then closing it: how a secret
/// reaches a child without a file or its command line (which other processes can read).
pub fn spawn_with_input(program: &Path, args: &[OsString], tty: bool, input: Option<Vec<u8>>) -> std::io::Result<Sidecar> {
    let (tx, lines) = mpsc::unbounded_channel();
    let mut command = tokio::process::Command::new(program);
    let stdin = if input.is_some() { Stdio::piped() } else { Stdio::null() };
    command.args(args).stdin(stdin).stdout(Stdio::piped()).kill_on_drop(true);
    let path = std::env::join_paths(search_path()).map_err(std::io::Error::other)?;
    command.env("PATH", path);
    if std::env::var_os("VERGENCE_FFMPEG").is_none()
        && let Some(ffmpeg) = find_ffmpeg()
    {
        command.env("VERGENCE_FFMPEG", ffmpeg);
    }
    #[cfg(windows)]
    {
        // No console window flashing up for the sidecar.
        command.creation_flags(0x08000000);
    }
    #[cfg(unix)]
    let master = if tty {
        let (master, slave) = open_pty()?;
        command.stderr(Stdio::from(slave));
        Some(master)
    } else {
        command.stderr(Stdio::piped());
        None
    };
    #[cfg(not(unix))]
    {
        // No terminal to give it here: a tool that prints progress only to a terminal is asked
        // for it anyway.
        if tty {
            command.env("VERGENCE_PROGRESS", "1");
        }
        command.stderr(Stdio::piped());
    }
    let mut child = command.spawn()?;
    // The command holds the terminal's other end; drop it, so the terminal reads end when the child does.
    drop(command);
    let pid = child.id();
    if let (Some(input), Some(mut stdin)) = (input, child.stdin.take()) {
        tokio::spawn(async move {
            use tokio::io::AsyncWriteExt;
            if let Err(e) = stdin.write_all(&input).await {
                tracing::warn!(error = %e, "couldn't give the sidecar its input");
            }
            // Dropping it closes the pipe: the child reads to its end.
        });
    }
    if let Some(stdout) = child.stdout.take() {
        pump_async(stdout, tx.clone());
    }
    if let Some(stderr) = child.stderr.take() {
        pump_async(stderr, tx.clone());
    }
    #[cfg(unix)]
    if let Some(master) = master {
        pump_blocking(std::fs::File::from(master), tx.clone());
    }
    drop(tx);
    let (stop, stopped) = oneshot::channel::<()>();
    let exit = tokio::spawn(async move {
        tokio::select! {
            status = child.wait() => status,
            Ok(()) = stopped => terminate(&mut child).await,
        }
    });
    Ok(Sidecar { pid, lines, stop: Some(stop), exit })
}

async fn terminate(child: &mut tokio::process::Child) -> std::io::Result<ExitStatus> {
    #[cfg(unix)]
    if let Some(pid) = child.id() {
        // Only this child: the PID came from its own handle, which hasn't been waited on yet.
        unsafe { libc::kill(pid as libc::pid_t, libc::SIGTERM) };
        if let Ok(status) = tokio::time::timeout(Duration::from_secs(3), child.wait()).await {
            return status;
        }
    }
    child.kill().await?;
    child.wait().await
}

fn pump_async(mut reader: impl tokio::io::AsyncRead + Unpin + Send + 'static, tx: mpsc::UnboundedSender<String>) {
    tokio::spawn(async move {
        let mut splitter = LineSplitter::default();
        let mut buf = [0u8; 4096];
        loop {
            match reader.read(&mut buf).await {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    for line in splitter.push(&buf[..n]) {
                        let _ = tx.send(line);
                    }
                }
            }
        }
        for line in splitter.finish() {
            let _ = tx.send(line);
        }
    });
}

#[cfg(unix)]
fn pump_blocking(mut reader: std::fs::File, tx: mpsc::UnboundedSender<String>) {
    use std::io::Read;
    std::thread::spawn(move || {
        let mut splitter = LineSplitter::default();
        let mut buf = [0u8; 4096];
        loop {
            // A terminal's reads end in EIO, not EOF, once the child side is closed.
            match reader.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    for line in splitter.push(&buf[..n]) {
                        let _ = tx.send(line);
                    }
                }
            }
        }
        for line in splitter.finish() {
            let _ = tx.send(line);
        }
    });
}

#[cfg(unix)]
fn open_pty() -> std::io::Result<(std::os::fd::OwnedFd, std::os::fd::OwnedFd)> {
    use std::os::fd::FromRawFd;
    let (mut master, mut slave) = (-1, -1);
    let rc = unsafe { libc::openpty(&mut master, &mut slave, std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut()) };
    if rc != 0 {
        return Err(std::io::Error::last_os_error());
    }
    let (master, slave) = unsafe { (std::os::fd::OwnedFd::from_raw_fd(master), std::os::fd::OwnedFd::from_raw_fd(slave)) };
    // Neither end leaks into other children; the slave is dup'ed onto stderr, which clears it there.
    for fd in [&master, &slave] {
        use std::os::fd::AsRawFd;
        unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_SETFD, libc::FD_CLOEXEC) };
    }
    Ok((master, slave))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tools_run_as_the_platform_has_them() {
        let dir = tempfile::tempdir().unwrap();
        let name = if cfg!(windows) { "vergence-worker.exe" } else { "vergence" };
        std::fs::write(dir.path().join(name), b"").unwrap();
        let settings = Settings { vergence_cli: Some(dir.path().join(name).to_string_lossy().into_owned()), ..Settings::default() };
        let tool = find_tool(&settings, ToolKind::Worker).unwrap();
        let args = tool.arguments(["--config".into(), "w.json".into()]);
        if cfg!(windows) {
            assert!(tool.program.ends_with("vergence-worker.exe"));
            assert_eq!(args, ["--config", "w.json"]);
            // The folder works too.
            let folder = Settings { vergence_cli: Some(dir.path().to_string_lossy().into_owned()), ..Settings::default() };
            assert!(find_tool(&folder, ToolKind::Worker).is_some());
        } else {
            assert_eq!(args, ["worker", "--config", "w.json"]);
            let convert = find_tool(&settings, ToolKind::Convert).unwrap();
            assert_eq!(convert.prefix, ["convert"]);
        }
    }

    #[test]
    fn splits_at_returns_and_newlines() {
        let mut s = LineSplitter::default();
        assert_eq!(s.push(b"vergence: a -> b\n  depth"), ["vergence: a -> b"]);
        assert_eq!(s.push(b"  42%"), ["depth  42%"]);
        assert_eq!(s.push(b"\r  depth 100%\r\r\n  depth: 10 frames in 1.0 s\r\n"), ["depth 100%", "depth: 10 frames in 1.0 s"]);
        assert_eq!(s.push(b"tail"), Vec::<String>::new());
        assert_eq!(s.finish(), ["tail"]);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn reads_a_terminal_stderr_and_stops_by_pid() {
        // stderr is a terminal for the child.
        let mut sidecar = spawn(
            Path::new("/bin/sh"),
            &["-c".into(), "[ -t 2 ] && echo tty >&2; echo out; printf '  depth  50%%' >&2".into()],
            true,
        )
        .unwrap();
        let mut lines = Vec::new();
        while let Some(line) = sidecar.lines.recv().await {
            lines.push(line);
        }
        lines.sort();
        assert_eq!(lines, ["depth  50%", "out", "tty"]);
        assert!(sidecar.exit.await.unwrap().unwrap().success());

        let mut sleeper = spawn(Path::new("/bin/sleep"), &["30".into()], false).unwrap();
        let _ = sleeper.take_stopper().unwrap().send(());
        let status = tokio::time::timeout(Duration::from_secs(5), sleeper.exit).await.unwrap().unwrap().unwrap();
        assert!(!status.success());
    }
}
