//! When a shared machine must stand aside. The worker asks before every claim
//! and every few seconds while it works; any reason stops claiming, and a
//! reason that turns up mid-piece abandons the piece and releases its lease,
//! so the coordinator hands it to someone else straight away.
//!
//! - **Everywhere:** a `pause` file next to the config (`touch` it to pause,
//!   delete it to resume), and other programs computing on an NVIDIA GPU
//!   (another model, a CUDA renderer), from `nvidia-smi`.
//! - **Windows:** a game or a stream, detected the way Longwave's Windows
//!   Companion knows them (`CompanionWindows/backend`):
//!   - a running executable inside an installed Steam title's folder, or a
//!     custom title's executable. The titles come from the Companion itself
//!     over its pipe (`LibraryList`), so its custom titles and kind overrides
//!     count; without it, from Steam's own `libraryfolders.vdf` (any
//!     `steamapps\common` folder is a game);
//!   - the Companion's native screen stream with a device connected
//!     (`NativeStreamStatus.connectedDevice`), and SteamVR (`vrserver.exe`),
//!     which runs only while a PCVR session (Longwave's foveated streaming
//!     included) is up;
//!   - the shell saying a full-screen Direct3D program or presentation mode is
//!     up (`SHQueryUserNotificationState`), which catches games outside any
//!     library. Only for a worker in the user's session: a service in
//!     session 0 sees its own desktop;
//!   - any executable named in the config's `yieldProcesses`.
//!
//! Untested on Windows so far (written on a Mac; it compiles for
//! `x86_64-pc-windows-msvc`).

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

/// Other processes with a CUDA context on an NVIDIA GPU (`nvidia-smi`), at most
/// once every 10 s. Silent where there is no `nvidia-smi`.
pub struct GpuComputePeers {
    last: Mutex<Option<(Instant, Option<String>)>>,
}

impl GpuComputePeers {
    pub fn new() -> GpuComputePeers {
        GpuComputePeers { last: Mutex::new(None) }
    }
}

impl Default for GpuComputePeers {
    fn default() -> Self {
        Self::new()
    }
}

/// `pid, name` lines from `nvidia-smi --query-compute-apps`, ours left out.
pub fn compute_peers(csv: &str, own_pid: u32) -> Vec<String> {
    csv.lines()
        .filter_map(|line| {
            let (pid, name) = line.split_once(',')?;
            let pid: u32 = pid.trim().parse().ok()?;
            (pid != own_pid).then(|| name.trim().rsplit(['/', '\\']).next().unwrap_or("").to_string())
        })
        .collect()
}

impl Yield for GpuComputePeers {
    fn reason(&self) -> Option<String> {
        let mut last = self.last.lock().unwrap();
        if let Some((at, reason)) = &*last
            && at.elapsed() < Duration::from_secs(10)
        {
            return reason.clone();
        }
        let reason = Command::new("nvidia-smi")
            .args(["--query-compute-apps=pid,process_name", "--format=csv,noheader"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .and_then(|o| {
                let peers = compute_peers(&String::from_utf8_lossy(&o.stdout), std::process::id());
                (!peers.is_empty()).then(|| format!("the GPU is computing for {}", peers.join(", ")))
            });
        *last = Some((Instant::now(), reason.clone()));
        reason
    }
}

/// The first reason any of them gives.
pub struct AnyOf(pub Vec<Box<dyn Yield>>);

impl Yield for AnyOf {
    fn reason(&self) -> Option<String> {
        self.0.iter().find_map(|y| y.reason())
    }
}

/// What this machine yields to: the pause file in `config_dir`, GPU peers, and
/// on Windows games and streams (with `extra_processes` by name).
pub fn for_this_machine(config_dir: &Path, extra_processes: Vec<String>) -> AnyOf {
    #[allow(unused_mut)]
    let mut all: Vec<Box<dyn Yield>> = vec![Box::new(PauseFile(config_dir.join("pause"))), Box::new(GpuComputePeers::new())];
    #[cfg(windows)]
    all.push(Box::new(windows::GamesAndStreams::new(extra_processes)));
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

    use super::{Title, Yield, library_folders, match_title, titles_from_response};

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

    /// Every process: its executable's name and, where readable, full path.
    fn processes() -> Vec<(String, Option<String>)> {
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
                out.push((name, path));
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

    fn native_stream_connected() -> Option<String> {
        let line = companion("NativeStreamStatus", Duration::from_secs(2))?;
        let value: serde_json::Value = serde_json::from_str(&line).ok()?;
        // Only this field is read: the status also carries a token.
        let device = value.get("result")?.get("connectedDevice")?.as_str()?;
        Some(format!("Longwave is streaming the screen to {device}"))
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
            for (name, path) in processes() {
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
            native_stream_connected()
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
        assert_eq!(compute_peers("123, /usr/bin/python3\n77, C:\\x\\me.exe\n", 77), vec!["python3".to_string()]);
    }
}
