//! When a shared machine must stand aside. The worker asks before every claim
//! and every few seconds while it works; any reason stops claiming, and a
//! reason that turns up mid-piece abandons the piece and releases its lease,
//! so the coordinator hands it to someone else straight away.
//!
//! - **Everywhere:** a `pause` file next to the config (`touch` it to pause,
//!   delete it to resume). On an NVIDIA GPU, from `nvidia-smi`:
//!   - elsewhere than Windows, another program computing on it (another
//!     model, a CUDA renderer): a process of type `C`, but a streaming host
//!     (Sunshine converts colours for its encoder with CUDA). On Windows,
//!     where the desktop is `C+G` and an idle Sunshine shows up as `C`, the
//!     per-process engine counters below say how busy each one is instead;
//!   - less than 1 GiB of the GPU's memory free (a game loading), until
//!     2.5 GiB are free again.
//!
//!   Vergence's own programs (`vergence*` and what they start) never count:
//!   a conversion runs below live analysis, not the other way round.
//! - **Windows:** a game or a stream, detected the way Longwave's Windows
//!   Companion knows them (`CompanionWindows/backend`):
//!   - a running executable inside an installed Steam title's folder, or a
//!     custom title's executable. The titles come from the Companion itself
//!     over its pipe (`LibraryList`), so its custom titles and kind overrides
//!     count; without it, from Steam's own `libraryfolders.vdf` (any
//!     `steamapps\common` folder is a game);
//!   - SteamVR (`vrserver.exe`), which runs only while a PCVR session
//!     (Longwave's foveated streaming included) is up;
//!   - the shell saying a full-screen Direct3D program or presentation mode is
//!     up (`SHQueryUserNotificationState`), which catches games outside any
//!     library. Only for a worker in the user's session: a service in
//!     session 0 sees its own desktop;
//!   - another process keeping the GPU's 3D or compute engines at least 30%
//!     busy in two samples in a row (the `GPU Engine` performance counters,
//!     per process, which catch any GPU vendor and a game in a window); 15%
//!     while something encodes video, i.e. a stream or a recording is on;
//!   - a program launched through a Sunshine stream (an app of its list, not
//!     the desktop);
//!   - any executable named in the config's `yieldProcesses`.
//!
//! A stream of the desktop alone (Sunshine, Moonlight, Steam Link,
//! Longwave's screen stream) doesn't make it stand aside: it keeps the GPU a
//! few percent busy, and the worker's below-normal priority leaves it be. A
//! game in that stream does, by any of the rules above.
//!
//! While it stands aside, the worker also unloads its models, so the GPU's
//! memory goes back too (`main.rs`).

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Something that can ask the worker to stand aside.
pub trait Yield: Send + Sync {
    /// Why the worker should stand aside right now, or `None`.
    fn reason(&self) -> Option<String>;
}

/// A file whose existence pauses the worker.
pub struct PauseFile(pub PathBuf);

impl Yield for PauseFile {
    fn reason(&self) -> Option<String> {
        self.0.exists().then(|| format!("{} exists", self.0.display()))
    }
}

/// A process, as far as telling Vergence's own from others' needs.
#[derive(Clone, Debug)]
pub struct Process {
    pub pid: u32,
    pub parent: u32,
    pub name: String,
}

/// Every process with its parent (Windows and Linux; empty elsewhere).
pub fn process_table() -> Vec<Process> {
    #[cfg(windows)]
    {
        windows::processes().into_iter().map(|p| Process { pid: p.pid, parent: p.parent, name: p.name }).collect()
    }
    #[cfg(target_os = "linux")]
    {
        let mut out = Vec::new();
        for entry in std::fs::read_dir("/proc").into_iter().flatten().flatten() {
            let Ok(pid) = entry.file_name().to_string_lossy().parse::<u32>() else { continue };
            let Ok(stat) = std::fs::read_to_string(entry.path().join("stat")) else { continue };
            // `pid (name) state ppid ...`; the name may hold spaces and parentheses.
            let (Some(open), Some(close)) = (stat.find('('), stat.rfind(')')) else { continue };
            let parent = stat[close + 1..].split_whitespace().nth(1).and_then(|p| p.parse().ok()).unwrap_or(0);
            out.push(Process { pid, parent, name: stat[open + 1..close].to_string() });
        }
        out
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    {
        Vec::new()
    }
}

/// Whether `pid` is this worker, another Vergence program (`vergence*`), or
/// something one of them started (ffmpeg under a converter).
pub fn is_vergence(pid: u32, table: &[Process], own_pid: u32) -> bool {
    let mut current = pid;
    for _ in 0..16 {
        if current == own_pid {
            return true;
        }
        let Some(process) = table.iter().find(|p| p.pid == current) else { return false };
        if process.name.to_lowercase().starts_with("vergence") {
            return true;
        }
        if process.parent == 0 || process.parent == current {
            return false;
        }
        current = process.parent;
    }
    false
}

/// The first process under one named `parent` (at any depth), other than
/// `ignored` names and the parent's own kind: what a streaming host launched.
#[cfg_attr(not(windows), allow(dead_code))]
pub fn launched_through(table: &[Process], parent: &str, ignored: &[&str]) -> Option<String> {
    let hosts: Vec<u32> = table.iter().filter(|p| p.name.eq_ignore_ascii_case(parent)).map(|p| p.pid).collect();
    if hosts.is_empty() {
        return None;
    }
    let stem = parent.trim_end_matches(".exe").to_lowercase();
    table
        .iter()
        .filter(|p| {
            let name = p.name.to_lowercase();
            !name.starts_with(&stem) && !ignored.iter().any(|i| name == i.to_lowercase())
        })
        .find(|p| {
            let mut current = p.parent;
            for _ in 0..16 {
                if hosts.contains(&current) {
                    return true;
                }
                match table.iter().find(|q| q.pid == current) {
                    Some(q) if q.parent != 0 && q.parent != current => current = q.parent,
                    _ => return false,
                }
            }
            false
        })
        .map(|p| p.name.clone())
}

/// `(pid, type, name)` of each process in `nvidia-smi -q -d PIDS`.
pub fn gpu_processes(query: &str) -> Vec<(u32, String, String)> {
    let mut out = Vec::new();
    let mut current: Option<(u32, String, String)> = None;
    for line in query.lines() {
        let Some((key, value)) = line.split_once(':') else { continue };
        let (key, value) = (key.trim(), value.trim());
        match key {
            "Process ID" => {
                out.extend(current.take());
                current = value.parse().ok().map(|pid| (pid, String::new(), String::new()));
            }
            "Type" => {
                if let Some(c) = current.as_mut() {
                    c.1 = value.to_string();
                }
            }
            "Name" => {
                if let Some(c) = current.as_mut() {
                    c.2 = value.rsplit(['/', '\\']).next().unwrap_or(value).to_string();
                }
            }
            _ => {}
        }
    }
    out.extend(current);
    out
}

/// Free memory in MiB and encoder sessions (when asked for too), from
/// `--query-gpu=memory.used,memory.total[,encoder.stats.sessionCount]`.
pub fn gpu_state(csv: &str) -> Option<(u64, u64)> {
    let mut parts = csv.lines().next()?.split(',').map(|p| p.trim().parse::<u64>().ok());
    let (used, total, sessions) = (parts.next()??, parts.next()??, parts.next().flatten().unwrap_or(0));
    Some((total.saturating_sub(used), sessions))
}

/// Below this much free memory the worker stands aside, and it comes back
/// only once this much more is free (so unloading its own models doesn't
/// count as the memory coming back).
const LOW_MEMORY_MIB: u64 = 1024;
const RESUME_MEMORY_MIB: u64 = 2560;

/// What `nvidia-smi` says about others on the GPU, at most once every 10 s.
/// Silent where there is no `nvidia-smi`.
pub struct NvidiaPeers {
    last: Mutex<Option<(Instant, Option<String>)>>,
    short_of_memory: Mutex<bool>,
}

impl NvidiaPeers {
    pub fn new() -> NvidiaPeers {
        NvidiaPeers { last: Mutex::new(None), short_of_memory: Mutex::new(false) }
    }

    fn check(&self) -> Option<String> {
        if !cfg!(windows) {
            let query = nvidia_smi(&["-q", "-d", "PIDS"])?;
            let own = std::process::id();
            let table = process_table();
            let computing: Vec<String> = gpu_processes(&query)
                .into_iter()
                .filter(|(pid, kind, name)| {
                    kind == "C" && !name.to_lowercase().contains("sunshine") && !is_vergence(*pid, &table, own)
                })
                .map(|(_, _, name)| name)
                .collect();
            if !computing.is_empty() {
                return Some(format!("the GPU is computing for {}", computing.join(", ")));
            }
        }
        let (free, _) =
            nvidia_smi(&["--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"])
                .and_then(|csv| gpu_state(&csv))?;
        let mut short = self.short_of_memory.lock().unwrap();
        *short = free < if *short { RESUME_MEMORY_MIB } else { LOW_MEMORY_MIB };
        short.then(|| format!("only {free} MiB of the GPU's memory is free"))
    }
}

impl Default for NvidiaPeers {
    fn default() -> Self {
        Self::new()
    }
}

fn nvidia_smi(args: &[&str]) -> Option<String> {
    let mut command = Command::new("nvidia-smi");
    command.args(args);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // No console window flashing up in the user's session.
        command.creation_flags(0x08000000);
    }
    command.output().ok().filter(|o| o.status.success()).map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
}

impl Yield for NvidiaPeers {
    fn reason(&self) -> Option<String> {
        let mut last = self.last.lock().unwrap();
        if let Some((at, reason)) = &*last
            && at.elapsed() < Duration::from_secs(10)
        {
            return reason.clone();
        }
        let reason = self.check();
        *last = Some((Instant::now(), reason.clone()));
        reason
    }
}

/// `(pid, engine type)` out of a `GPU Engine` counter instance,
/// `pid_1234_luid_0x0_0xD69A_phys_0_eng_10_engtype_Cuda`.
#[cfg_attr(not(windows), allow(dead_code))]
pub fn engine_instance(name: &str) -> Option<(u32, String)> {
    let pid = name.strip_prefix("pid_")?.split('_').next()?.parse().ok()?;
    let kind = name.rsplit_once("_engtype_")?.1.to_lowercase();
    Some((pid, kind))
}

/// How busy another process's 3D or compute engine must be, in percent, for
/// the worker to stand aside; less while a stream or recording is on.
#[cfg_attr(not(windows), allow(dead_code))]
pub const BUSY_PERCENT: f64 = 30.0;
#[cfg_attr(not(windows), allow(dead_code))]
pub const BUSY_WHILE_STREAMING_PERCENT: f64 = 15.0;

/// What a process does on the GPU, in percent.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(not(windows), allow(dead_code))]
pub struct GpuUse {
    /// Its busiest 3D or compute engine (CUDA, DirectML).
    pub busy: f64,
    /// Its video encoders.
    pub encoding: f64,
}

/// Per process, what it does on the GPU, from `(instance, percent)` samples of
/// `GPU Engine(*)\Utilization Percentage` (several engines of one type add
/// up). Copy and video decode (a video playing) don't count.
#[cfg_attr(not(windows), allow(dead_code))]
pub fn gpu_users(samples: &[(String, f64)]) -> std::collections::BTreeMap<u32, GpuUse> {
    let mut per_engine: std::collections::BTreeMap<(u32, String), f64> = std::collections::BTreeMap::new();
    for (name, value) in samples {
        if let Some((pid, kind)) = engine_instance(name) {
            *per_engine.entry((pid, kind)).or_default() += value;
        }
    }
    let mut out: std::collections::BTreeMap<u32, GpuUse> = std::collections::BTreeMap::new();
    for ((pid, kind), value) in per_engine {
        let usage = out.entry(pid).or_default();
        if kind == "videoencode" {
            usage.encoding += value;
        } else if kind == "3d" || kind == "cuda" || kind.starts_with("compute") {
            usage.busy = usage.busy.max(value);
        }
    }
    out
}

/// The processes busy enough to stand aside for: `(pid, busiest engine)`,
/// with the lower bar while any of them encodes video.
#[cfg_attr(not(windows), allow(dead_code))]
pub fn heavy_users(users: &std::collections::BTreeMap<u32, GpuUse>) -> Vec<(u32, f64)> {
    let streaming = users.values().any(|u| u.encoding >= 0.5);
    let bar = if streaming { BUSY_WHILE_STREAMING_PERCENT } else { BUSY_PERCENT };
    users.iter().filter(|(_, u)| u.busy >= bar).map(|(&pid, u)| (pid, u.busy)).collect()
}

/// The first reason any of them gives.
pub struct AnyOf(pub Vec<Box<dyn Yield>>);

impl Yield for AnyOf {
    fn reason(&self) -> Option<String> {
        self.0.iter().find_map(|y| y.reason())
    }
}

/// What this machine yields to: the pause file in `config_dir`, others on an
/// NVIDIA GPU, and on Windows games, games launched through a stream and anything keeping the GPU
/// busy (with `extra_processes` by name).
pub fn for_this_machine(config_dir: &Path, extra_processes: Vec<String>) -> AnyOf {
    #[allow(unused_mut)]
    let mut all: Vec<Box<dyn Yield>> = vec![Box::new(PauseFile(config_dir.join("pause"))), Box::new(NvidiaPeers::new())];
    #[cfg(windows)]
    {
        all.push(Box::new(windows::GamesAndStreams::new(extra_processes)));
        all.push(Box::new(windows::BusyGpu::new()));
    }
    #[cfg(not(windows))]
    let _ = extra_processes;
    AnyOf(all)
}

// MARK: Game titles, as the Companion's library describes them (portable, so it is tested here)

/// A title from the Companion's `LibraryList`, as far as matching needs it.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
#[cfg_attr(not(windows), allow(dead_code))]
pub struct Title {
    pub name: String,
    pub source: String,
    #[serde(default)]
    pub executable: Option<String>,
    #[serde(default)]
    pub install_dir: Option<String>,
}

#[cfg_attr(not(windows), allow(dead_code))]
fn normalize(path: &str) -> String {
    path.replace('/', "\\").trim_end_matches('\\').to_lowercase()
}

/// `GameLibrary.MatchPath`'s identity half: the custom title with exactly
/// that executable, else the Steam title whose folder holds it (the deepest).
#[cfg_attr(not(windows), allow(dead_code))]
pub fn match_title<'a>(titles: &'a [Title], image: &str) -> Option<&'a Title> {
    let full = normalize(image);
    if let Some(custom) = titles.iter().find(|t| t.executable.as_deref().is_some_and(|e| normalize(e) == full)) {
        return Some(custom);
    }
    titles
        .iter()
        .filter(|t| t.source == "steam")
        .filter_map(|t| t.install_dir.as_deref().map(|d| (t, normalize(d) + "\\")))
        .filter(|(_, dir)| full.starts_with(dir.as_str()))
        .max_by_key(|(_, dir)| dir.len())
        .map(|(t, _)| t)
}

/// The library roots in Steam's `libraryfolders.vdf`, current and pre-2021
/// layouts (`SteamLibrary.ParseLibraryFolders`).
#[cfg_attr(not(windows), allow(dead_code))]
pub fn library_folders(vdf: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in vdf.lines() {
        let quoted: Vec<&str> = line.split('"').skip(1).step_by(2).collect();
        if let [key, value] = quoted[..] {
            let value = value.replace("\\\\", "\\");
            let old_layout = key.parse::<u32>().is_ok() && value.as_bytes().get(1) == Some(&b':');
            if key.eq_ignore_ascii_case("path") || old_layout {
                out.push(value);
            }
        }
    }
    out
}

/// The titles out of a `LibraryList` response line, whether the result is the
/// list itself or an object holding it.
#[cfg_attr(not(windows), allow(dead_code))]
pub fn titles_from_response(line: &str, id: u64) -> Option<Vec<Title>> {
    let value: serde_json::Value = serde_json::from_str(line).ok()?;
    if value.get("id")?.as_u64()? != id {
        return None;
    }
    let result = value.get("result")?;
    let list = if result.is_array() { result } else { result.get("titles")? };
    serde_json::from_value(list.clone()).ok()
}

#[cfg(windows)]
mod windows {
    //! The Windows half: processes, the shell's notification state, the
    //! registry and the Companion's pipe.

    use std::io::{BufRead, BufReader, Write};
    use std::sync::Mutex;
    use std::time::{Duration, Instant};

    use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{
        CreateToolhelp32Snapshot, PROCESSENTRY32W, Process32FirstW, Process32NextW, TH32CS_SNAPPROCESS,
    };
    use windows_sys::Win32::System::Registry::{HKEY_CURRENT_USER, RRF_RT_REG_SZ, RegGetValueW};
    use windows_sys::Win32::System::Threading::{
        OpenProcess, PROCESS_NAME_WIN32, PROCESS_QUERY_LIMITED_INFORMATION, QueryFullProcessImageNameW,
    };
    use windows_sys::Win32::UI::Shell::{
        QUNS_PRESENTATION_MODE, QUNS_RUNNING_D3D_FULL_SCREEN, SHQueryUserNotificationState,
    };

    use windows_sys::Win32::System::Performance::{
        PDH_FMT_COUNTERVALUE_ITEM_W, PDH_FMT_DOUBLE, PDH_HCOUNTER, PDH_HQUERY, PDH_MORE_DATA, PdhAddEnglishCounterW,
        PdhCloseQuery, PdhCollectQueryData, PdhGetFormattedCounterArrayW, PdhOpenQueryW,
    };

    use super::{
        Process, Title, Yield, gpu_users, heavy_users, is_vergence, library_folders, match_title, process_table,
        titles_from_response,
    };

    const COMPANION_PIPE: &str = r"\\.\pipe\longwave-hotspot";
    /// Running only during a PCVR session.
    const STREAM_PROCESSES: &[&str] = &["vrserver.exe"];

    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(std::iter::once(0)).collect()
    }

    fn from_wide(buffer: &[u16]) -> String {
        let end = buffer.iter().position(|&c| c == 0).unwrap_or(buffer.len());
        String::from_utf16_lossy(&buffer[..end])
    }

    pub struct WindowsProcess {
        pub pid: u32,
        pub parent: u32,
        pub name: String,
        /// The executable's full path, where readable.
        pub path: Option<String>,
    }

    /// Every process.
    pub fn processes() -> Vec<WindowsProcess> {
        let mut out = Vec::new();
        unsafe {
            let snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
            if snapshot == INVALID_HANDLE_VALUE {
                return out;
            }
            let mut entry: PROCESSENTRY32W = std::mem::zeroed();
            entry.dwSize = size_of::<PROCESSENTRY32W>() as u32;
            let mut more = Process32FirstW(snapshot, &mut entry) != 0;
            while more {
                let name = from_wide(&entry.szExeFile);
                let mut path = None;
                let process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, entry.th32ProcessID);
                if !process.is_null() {
                    let mut buffer = [0u16; 1024];
                    let mut length = buffer.len() as u32;
                    if QueryFullProcessImageNameW(process, PROCESS_NAME_WIN32, buffer.as_mut_ptr(), &mut length) != 0 {
                        path = Some(String::from_utf16_lossy(&buffer[..length as usize]));
                    }
                    CloseHandle(process);
                }
                out.push(WindowsProcess { pid: entry.th32ProcessID, parent: entry.th32ParentProcessID, name, path });
                more = Process32NextW(snapshot, &mut entry) != 0;
            }
            CloseHandle(snapshot);
        }
        out
    }

    fn full_screen() -> Option<&'static str> {
        let mut state = 0;
        let ok = unsafe { SHQueryUserNotificationState(&mut state) } >= 0;
        match state {
            _ if !ok => None,
            QUNS_RUNNING_D3D_FULL_SCREEN => Some("a full-screen Direct3D program is running"),
            QUNS_PRESENTATION_MODE => Some("presentation mode is on"),
            _ => None,
        }
    }

    fn steam_root() -> Option<String> {
        let mut buffer = [0u16; 1024];
        let mut size = (buffer.len() * 2) as u32;
        let status = unsafe {
            RegGetValueW(HKEY_CURRENT_USER, wide(r"Software\Valve\Steam").as_ptr(), wide("SteamPath").as_ptr(),
                         RRF_RT_REG_SZ, std::ptr::null_mut(), buffer.as_mut_ptr().cast(), &mut size)
        };
        (status == 0).then(|| from_wide(&buffer).replace('/', "\\"))
    }

    /// Without the Companion: every Steam library's `steamapps\common` as one "title".
    fn steam_titles() -> Vec<Title> {
        let Some(root) = steam_root() else { return Vec::new() };
        let vdf = std::fs::read_to_string(format!(r"{root}\steamapps\libraryfolders.vdf")).unwrap_or_default();
        let mut libraries = library_folders(&vdf);
        libraries.push(root);
        libraries
            .into_iter()
            .map(|library| Title {
                name: "a Steam game".into(),
                source: "steam".into(),
                executable: None,
                install_dir: Some(format!(r"{library}\steamapps\common")),
            })
            .collect()
    }

    /// One request on the Companion's pipe; the matching response line, or None.
    fn companion(method: &str, timeout: Duration) -> Option<String> {
        let method = method.to_string();
        let (send, receive) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let answer = (|| {
                let mut pipe = std::fs::OpenOptions::new().read(true).write(true).open(COMPANION_PIPE).ok()?;
                writeln!(pipe, r#"{{"id":4242,"method":"{method}"}}"#).ok()?;
                // Events are interleaved; the response echoes the id.
                BufReader::new(pipe).lines().map_while(Result::ok).find(|l| l.contains(r#""id":4242"#))
            })();
            let _ = send.send(answer);
        });
        receive.recv_timeout(timeout).ok().flatten()
    }

    /// A program Sunshine started for its client (an app of its list): any
    /// process under `sunshine.exe` but its own console and prep commands.
    fn launched_through_sunshine(table: &[Process]) -> Option<String> {
        super::launched_through(table, "sunshine.exe", &["conhost.exe", "cmd.exe"])
            .map(|name| format!("{name} was launched through a Sunshine stream"))
    }

    pub struct GamesAndStreams {
        extra: Vec<String>,
        titles: Mutex<Option<(Instant, Vec<Title>)>>,
    }

    impl GamesAndStreams {
        pub fn new(extra: Vec<String>) -> GamesAndStreams {
            GamesAndStreams { extra: extra.into_iter().map(|e| e.to_lowercase()).collect(), titles: Mutex::new(None) }
        }

        /// The Companion's library when it answers, else Steam's folders; kept 5 minutes.
        fn titles(&self) -> Vec<Title> {
            let mut cached = self.titles.lock().unwrap();
            if let Some((at, titles)) = &*cached
                && at.elapsed() < Duration::from_secs(300)
            {
                return titles.clone();
            }
            let titles = companion("LibraryList", Duration::from_secs(5))
                .and_then(|line| titles_from_response(&line, 4242))
                .unwrap_or_else(steam_titles);
            *cached = Some((Instant::now(), titles.clone()));
            titles
        }
    }

    impl Yield for GamesAndStreams {
        fn reason(&self) -> Option<String> {
            if let Some(reason) = full_screen() {
                return Some(reason.into());
            }
            let titles = self.titles();
            for WindowsProcess { name, path, .. } in processes() {
                let lower = name.to_lowercase();
                if STREAM_PROCESSES.contains(&lower.as_str()) {
                    return Some(format!("{name} is running (a PCVR session)"));
                }
                if self.extra.contains(&lower) {
                    return Some(format!("{name} is running"));
                }
                if let Some(title) = path.as_deref().and_then(|p| match_title(&titles, p)) {
                    return Some(format!("{} is running ({name})", title.name));
                }
            }
            launched_through_sunshine(&process_table())
        }
    }

    struct Query {
        query: PDH_HQUERY,
        counter: PDH_HCOUNTER,
    }

    // The handles are used under the mutex only.
    unsafe impl Send for Query {}

    impl Drop for Query {
        fn drop(&mut self) {
            unsafe { PdhCloseQuery(self.query) };
        }
    }

    impl Query {
        fn open() -> Option<Query> {
            let mut query: PDH_HQUERY = std::ptr::null_mut();
            let mut counter: PDH_HCOUNTER = std::ptr::null_mut();
            unsafe {
                if PdhOpenQueryW(std::ptr::null(), 0, &mut query) != 0 {
                    return None;
                }
                let path = wide(r"\GPU Engine(*)\Utilization Percentage");
                if PdhAddEnglishCounterW(query, path.as_ptr(), 0, &mut counter) != 0 {
                    PdhCloseQuery(query);
                    return None;
                }
                // A rate needs two collections; this is the first.
                PdhCollectQueryData(query);
                Some(Query { query, counter })
            }
        }

        /// Each engine instance's utilization since the last collection.
        fn sample(&self) -> Vec<(String, f64)> {
            let mut out = Vec::new();
            unsafe {
                if PdhCollectQueryData(self.query) != 0 {
                    return out;
                }
                let (mut size, mut count) = (0u32, 0u32);
                let status = PdhGetFormattedCounterArrayW(self.counter, PDH_FMT_DOUBLE, &mut size, &mut count, std::ptr::null_mut());
                if status != PDH_MORE_DATA || size == 0 {
                    return out;
                }
                // Items first, their names after them in the same buffer.
                let mut buffer = vec![0u64; (size as usize).div_ceil(8)];
                let items = buffer.as_mut_ptr().cast::<PDH_FMT_COUNTERVALUE_ITEM_W>();
                if PdhGetFormattedCounterArrayW(self.counter, PDH_FMT_DOUBLE, &mut size, &mut count, items) != 0 {
                    return out;
                }
                for item in std::slice::from_raw_parts(items, count as usize) {
                    if item.FmtValue.CStatus != 0 || item.szName.is_null() {
                        continue;
                    }
                    let length = (0..).take_while(|&i| *item.szName.add(i) != 0).count();
                    let name = String::from_utf16_lossy(std::slice::from_raw_parts(item.szName, length));
                    out.push((name, item.FmtValue.Anonymous.doubleValue));
                }
            }
            out
        }
    }

    /// Another process keeping the GPU's 3D or compute engines busy, or
    /// encoding video, from Windows' per-process `GPU Engine` counters: any
    /// GPU, any program, a game in a window included.
    pub struct BusyGpu {
        state: Mutex<BusyState>,
    }

    struct BusyState {
        query: Option<Query>,
        /// Who was busy in the last sample.
        before: Vec<u32>,
        /// The last sample's time and answer: a rate needs some time between samples.
        last: Option<(Instant, Option<String>)>,
    }

    impl BusyGpu {
        pub fn new() -> BusyGpu {
            BusyGpu { state: Mutex::new(BusyState { query: Query::open(), before: Vec::new(), last: None }) }
        }
    }

    impl Yield for BusyGpu {
        fn reason(&self) -> Option<String> {
            let mut state = self.state.lock().unwrap();
            if let Some((at, reason)) = &state.last
                && at.elapsed() < Duration::from_secs(1)
            {
                return reason.clone();
            }
            let BusyState { query, before: busy_before, last } = &mut *state;
            let samples = query.as_ref()?.sample();
            let own = std::process::id();
            let table = process_table();
            let mut users = gpu_users(&samples);
            users.retain(|&pid, _| pid != 0 && !is_vergence(pid, &table, own));
            let busy = heavy_users(&users);
            // In this sample and the one before: not a moment's spike.
            let reason = busy.iter().find(|(pid, _)| busy_before.contains(pid)).map(|&(pid, value)| {
                let name = table.iter().find(|p| p.pid == pid).map(|p| p.name.clone()).unwrap_or_else(|| format!("process {pid}"));
                format!("{name} keeps the GPU {value:.0}% busy")
            });
            *busy_before = busy.into_iter().map(|(pid, _)| pid).collect();
            *last = Some((Instant::now(), reason.clone()));
            reason
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn steam(name: &str, dir: &str) -> Title {
        Title { name: name.into(), source: "steam".into(), executable: None, install_dir: Some(dir.into()) }
    }

    #[test]
    fn matches_games_like_the_companion() {
        let titles = vec![
            steam("Half-Life", r"D:\SteamLibrary\steamapps\common\Half-Life"),
            steam("Library", r"D:\SteamLibrary\steamapps\common"),
            Title {
                name: "Emulator".into(),
                source: "custom".into(),
                executable: Some(r"C:\Games\emu\emu.exe".into()),
                install_dir: None,
            },
        ];
        let hit = |p: &str| match_title(&titles, p).map(|t| t.name.as_str());
        assert_eq!(hit(r"d:\steamlibrary\steamapps\common\Half-Life\hl.exe"), Some("Half-Life"));
        assert_eq!(hit(r"D:/SteamLibrary/steamapps/common/Other/x.exe"), Some("Library"));
        assert_eq!(hit(r"C:\Games\emu\emu.exe"), Some("Emulator"));
        assert_eq!(hit(r"C:\Games\emu\tool.exe"), None);
        assert_eq!(hit(r"D:\SteamLibrary\steamapps\commonplace\x.exe"), None);
    }

    #[test]
    fn reads_steam_library_folders() {
        let vdf = "\"libraryfolders\"\n{\n\t\"0\"\n\t{\n\t\t\"path\"\t\t\"C:\\\\Program Files (x86)\\\\Steam\"\n\t}\n\
                   \t\"1\"\t\t\"E:\\\\Old\"\n}\n";
        assert_eq!(library_folders(vdf), vec![r"C:\Program Files (x86)\Steam".to_string(), r"E:\Old".to_string()]);
    }

    #[test]
    fn reads_the_companions_library_list() {
        let line = r#"{"id":4242,"result":[{"id":"steam:70","name":"Half-Life","source":"steam","installDir":"D:\\S\\common\\HL","exposed":true}]}"#;
        let titles = titles_from_response(line, 4242).unwrap();
        assert_eq!(titles[0].install_dir.as_deref(), Some(r"D:\S\common\HL"));
        assert!(titles_from_response(r#"{"event":"state","data":{}}"#, 4242).is_none());
    }

    #[test]
    fn pause_file_and_gpu_peers() {
        let dir = tempfile::tempdir().unwrap();
        let pause = PauseFile(dir.path().join("pause"));
        assert_eq!(pause.reason(), None);
        std::fs::write(dir.path().join("pause"), b"").unwrap();
        assert!(pause.reason().unwrap().contains("pause"));
    }

    #[test]
    fn reads_nvidia_smi() {
        // Windows lists the desktop as C+G; only a pure compute process counts.
        let query = "GPU 00000000:2B:00.0\n    Processes\n        Process ID                        : 772\n\
                     \x20           Type                          : C+G\n\
                     \x20           Name                          : C:\\Windows\\System32\\dwm.exe\n\
                     \x20       Process ID                        : 17684\n\
                     \x20           Type                          : C\n\
                     \x20           Name                          : /usr/bin/python3\n";
        assert_eq!(
            gpu_processes(query),
            vec![(772, "C+G".to_string(), "dwm.exe".to_string()), (17684, "C".to_string(), "python3".to_string())]
        );
        assert_eq!(gpu_state("9000, 10240, 1\n"), Some((1240, 1)));
        assert_eq!(gpu_state("9000, 10240, [N/A]"), Some((1240, 0)));
        assert_eq!(gpu_state("[N/A], 10240, 0"), None);
    }

    #[test]
    fn knows_its_own_family() {
        let table = vec![
            Process { pid: 10, parent: 1, name: "vergence-convert.exe".into() },
            Process { pid: 11, parent: 10, name: "ffmpeg.exe".into() },
            Process { pid: 20, parent: 1, name: "sunshine.exe".into() },
            Process { pid: 21, parent: 20, name: "ffmpeg.exe".into() },
            Process { pid: 30, parent: 5, name: "x.exe".into() },
        ];
        assert!(is_vergence(11, &table, 99));
        assert!(is_vergence(30, &table, 30));
        assert!(!is_vergence(21, &table, 99));
        assert!(!is_vergence(404, &table, 99));
        assert_eq!(launched_through(&table, "sunshine.exe", &["conhost.exe"]), Some("ffmpeg.exe".to_string()));
        let desktop_only = vec![
            Process { pid: 20, parent: 1, name: "sunshine.exe".into() },
            Process { pid: 22, parent: 20, name: "conhost.exe".into() },
        ];
        assert_eq!(launched_through(&desktop_only, "sunshine.exe", &["conhost.exe"]), None);
    }

    #[test]
    fn finds_busy_engines() {
        let samples = vec![
            ("pid_17684_luid_0x00000000_0x0000D69A_phys_0_eng_10_engtype_Cuda".to_string(), 75.2),
            ("pid_17684_luid_0x00000000_0x0000D69A_phys_0_eng_4_engtype_Copy".to_string(), 90.0),
            ("pid_20532_luid_0x00000000_0x0000D69A_phys_0_eng_0_engtype_3D".to_string(), 20.0),
            ("pid_20532_luid_0x00000000_0x0000D69A_phys_0_eng_1_engtype_3D".to_string(), 15.0),
            ("pid_5_luid_0x00000000_0x0000D69A_phys_0_eng_7_engtype_VideoDecode".to_string(), 80.0),
            ("pid_6_luid_0x00000000_0x0000D69A_phys_0_eng_0_engtype_3D".to_string(), 12.0),
            ("pid_7_luid_0x00000000_0x0000D69A_phys_0_eng_0_engtype_3D".to_string(), 60.0),
            ("pid_7_luid_0x00000000_0x0000D69A_phys_0_eng_8_engtype_VideoEncode".to_string(), 4.0),
        ];
        let users = gpu_users(&samples);
        assert_eq!(users[&17684], GpuUse { busy: 75.2, encoding: 0.0 });
        assert_eq!(users[&20532].busy, 35.0);
        assert_eq!(users[&5], GpuUse::default());
        assert_eq!(users[&7], GpuUse { busy: 60.0, encoding: 4.0 });
        // A stream (pid 7 encodes) lowers the bar: pid 6 at 12% stays below it, pid 20532 at 35% not.
        let mut heavy: Vec<u32> = heavy_users(&users).into_iter().map(|(pid, _)| pid).collect();
        heavy.sort();
        assert_eq!(heavy, vec![7, 17684, 20532]);
        // A desktop stream alone: the encoder busy, its 3D engine nearly idle.
        let desktop = vec![
            ("pid_9_luid_0x0_0x1_phys_0_eng_7_engtype_VideoEncode".to_string(), 28.0),
            ("pid_9_luid_0x0_0x1_phys_0_eng_0_engtype_3D".to_string(), 1.0),
            ("pid_10_luid_0x0_0x1_phys_0_eng_0_engtype_3D".to_string(), 16.0),
        ];
        let users = gpu_users(&desktop);
        assert_eq!(heavy_users(&users), vec![(10, 16.0)]);
        let quiet = gpu_users(&desktop[..2]);
        assert!(heavy_users(&quiet).is_empty());
        assert_eq!(engine_instance("pid_x_engtype_3D"), None);
    }
}
