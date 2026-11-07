//! Converting files to AI 3D: a queue that runs one `vergence convert` at a time as a child
//! process, reading its progress from its output, with cancel. The queue is kept in
//! `convert-queue.json`, so it carries on after a quit; `vergence convert` itself picks up from
//! the depth it had finished (its hidden work folder beside the output).
//!
//! On a Mac it runs `vergence convert` (Core ML, Metal); on Windows `vergence-convert.exe` (ONNX
//! Runtime, wgpu, ffmpeg), which takes the same flags and prints the same progress lines.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::Notify;

use crate::settings::{Settings, write_atomic};
use crate::sidecar::{self, ToolKind};

pub const AVAILABLE: bool = cfg!(any(target_os = "macos", windows));
pub const UNAVAILABLE_REASON: &str = "Converting needs a Mac or Windows for now.";
/// What `--sample` renders, for a quick look before the whole file.
pub const SAMPLE_SECONDS: u32 = 10;

const VIDEO_EXTENSIONS: &[&str] = &["mp4", "m4v", "mov", "mkv", "webm", "avi", "ts", "m2ts", "mts", "wmv", "mpg", "mpeg", "flv"];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Layout {
    /// Full side by side.
    #[default]
    Fsbs,
    /// Half side by side, for players and TVs that want the original width.
    Hsbs,
    /// Top and bottom.
    Tab,
    Htab,
    /// MV-HEVC spatial video, for a Vision Pro or iPhone.
    Spatial,
    /// No 3D: upscaled only (with `upscale`).
    #[serde(rename = "2d")]
    Flat,
}

impl Layout {
    fn option(self) -> &'static str {
        match self {
            Layout::Fsbs => "fsbs",
            Layout::Hsbs => "hsbs",
            Layout::Tab => "tab",
            Layout::Htab => "htab",
            Layout::Spatial => "spatial",
            Layout::Flat => "2d",
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ConvertOptions {
    pub layout: Layout,
    /// The warp's largest per-eye shift as a share of the width, 0…0.05 (None: Convolution's default).
    pub strength: Option<f64>,
    /// The depth on the screen plane, 0…1 (None: the default).
    pub convergence: Option<f64>,
    /// A 10 s sample from the middle instead of the whole file.
    pub sample: bool,
    /// Redo a file whose output exists.
    pub force: bool,
    /// Make each eye twice the size first (Real-ESRGAN's compact network); a side-by-side or
    /// top-and-bottom source is upscaled eye by eye with no depth.
    pub upscale: bool,
}

impl ConvertOptions {
    pub fn checked(self) -> Result<ConvertOptions, String> {
        if let Some(s) = self.strength
            && !(0.0..=0.05).contains(&s)
        {
            return Err("Strength is a share of the width, 0 to 0.05.".into());
        }
        if let Some(c) = self.convergence
            && !(0.0..=1.0).contains(&c)
        {
            return Err("Convergence is 0 to 1.".into());
        }
        if self.layout == Layout::Flat && !self.upscale {
            return Err("2D (no 3D) only upscales: turn on Upscale 2×.".into());
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JobState {
    Queued,
    Running,
    Done,
    /// Nothing to do: the output exists, or the file is already 3D.
    Skipped,
    Failed,
    Cancelled,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConvertJob {
    pub id: u64,
    pub input: PathBuf,
    pub options: ConvertOptions,
    /// The folder the copy goes into (None: beside the input).
    #[serde(default)]
    pub output_dir: Option<PathBuf>,
    pub state: JobState,
    #[serde(default)]
    pub phase: Option<String>,
    /// The whole job, 0…1.
    #[serde(default)]
    pub progress: f64,
    #[serde(default)]
    pub output: Option<PathBuf>,
    #[serde(default)]
    pub message: Option<String>,
}

impl ConvertJob {
    /// The converter's arguments for this job (after `convert`, on a Mac).
    #[cfg(test)]
    pub fn arguments(&self, model: Option<&str>) -> Vec<OsString> {
        self.arguments_with(model, crate::models::SMALL.variant)
    }

    /// The same, downloading `variant` when no model is given (`--variant`: `vergence-convert` only).
    pub fn arguments_with(&self, model: Option<&str>, variant: &str) -> Vec<OsString> {
        let mut args: Vec<OsString> = vec![self.input.clone().into(), "--layout".into(), self.options.layout.option().into()];
        if let Some(s) = self.options.strength {
            args.extend(["--strength".into(), format!("{s}").into()]);
        }
        if let Some(c) = self.options.convergence {
            args.extend(["--convergence".into(), format!("{c}").into()]);
        }
        if self.options.sample {
            args.extend(["--sample".into(), SAMPLE_SECONDS.to_string().into()]);
        }
        if self.options.upscale {
            args.extend(["--upscale".into(), "2x".into()]);
        }
        if let Some(dir) = &self.output_dir {
            // A trailing separator makes it a folder to `convert`, even before it exists.
            let mut dir = dir.clone().into_os_string();
            dir.push(std::path::MAIN_SEPARATOR_STR);
            args.extend(["--output".into(), dir]);
        }
        match model {
            Some(model) => {
                args.extend(["--model".into(), model.into()]);
                // The upscaling network's weights (BSD-3-Clause) are fetched once, if missing.
                if self.options.upscale {
                    args.push("--download-model".into());
                }
            }
            // Fetches Depth Anything V2 Small (Apache-2.0) once, if no model is installed; or
            // the larger model the person chose and whose licence they accepted (`models`).
            None => {
                if variant != crate::models::SMALL.variant {
                    args.extend(["--variant".into(), variant.into()]);
                }
                args.push("--download-model".into());
            }
        }
        if self.options.force {
            args.push("--force".into());
        }
        args
    }
}

/// What a line of `vergence convert`'s output says.
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    /// `depth  42%`, `picture   7%`, `spatial 100%`.
    Progress(String, f64),
    /// A stage finished (`depth: 120 frames in 3.0 s`, `... from an earlier run`).
    StageDone(String),
    /// `vergence: NAME -> PATH`.
    Output(PathBuf),
    /// `done: ...`.
    Done,
    /// Nothing to do for this file.
    Skipped(String),
    Downloading,
    /// Any other `vergence: NAME: ...` line: the reason, if the run fails.
    Note(String),
}

pub fn parse_line(line: &str) -> Option<Event> {
    let line = line.trim();
    let words: Vec<&str> = line.split_whitespace().collect();
    if words.len() == 2
        && ["depth", "picture", "spatial"].contains(&words[0])
        && let Some(percent) = words[1].strip_suffix('%').and_then(|p| p.parse::<f64>().ok())
    {
        return Some(Event::Progress(words[0].to_string(), (percent / 100.0).clamp(0.0, 1.0)));
    }
    for (prefix, stage) in [("depth: ", "depth"), ("picture: ", "picture"), ("spatial video: ", "spatial")] {
        if line.starts_with(prefix) {
            return Some(Event::StageDone(stage.into()));
        }
    }
    if line.starts_with("done: ") {
        return Some(Event::Done);
    }
    let rest = line.strip_prefix("vergence: ")?;
    if rest.starts_with("downloading ") {
        return Some(Event::Downloading);
    }
    if let Some((_, path)) = rest.split_once(" -> ") {
        return Some(Event::Output(PathBuf::from(path)));
    }
    if rest.ends_with("(--force to redo)") || rest.ends_with(", skipped") {
        return Some(Event::Skipped(after_name(rest).to_string()));
    }
    Some(Event::Note(after_name(rest).to_string()))
}

/// `NAME: message` → `message` (a name may hold ": " itself, so the last split wins only if
/// the first doesn't look like a file name).
fn after_name(rest: &str) -> &str {
    rest.split_once(": ").map(|(_, message)| message).unwrap_or(rest)
}

/// The stages a layout goes through, and each one's share of the time.
fn stages(layout: Layout) -> &'static [(&'static str, f64)] {
    if layout == Layout::Flat {
        &[("picture", 1.0)]
    } else if layout == Layout::Spatial {
        &[("depth", 0.45), ("picture", 0.4), ("spatial", 0.15)]
    } else {
        &[("depth", 0.5), ("picture", 0.5)]
    }
}

/// The whole job's progress, at `fraction` through `stage`.
pub fn overall(layout: Layout, stage: &str, fraction: f64) -> f64 {
    let mut done = 0.0;
    for (name, share) in stages(layout) {
        if *name == stage {
            return (done + share * fraction).clamp(0.0, 1.0);
        }
        done += share;
    }
    done.min(1.0)
}

impl ConvertJob {
    /// Applies a line's news; true when the page should hear about it.
    pub fn apply(&mut self, event: &Event) -> bool {
        match event {
            Event::Progress(stage, fraction) => {
                let progress = overall(self.options.layout, stage, *fraction);
                let changed = self.phase.as_deref() != Some(stage) || (progress - self.progress).abs() >= 0.005;
                self.phase = Some(stage.clone());
                self.progress = self.progress.max(progress);
                changed
            }
            Event::StageDone(stage) => {
                self.progress = self.progress.max(overall(self.options.layout, stage, 1.0));
                true
            }
            Event::Output(path) => {
                self.output = Some(path.clone());
                true
            }
            Event::Done => {
                self.progress = 1.0;
                true
            }
            Event::Skipped(message) => {
                self.state = JobState::Skipped;
                self.message = Some(message.clone());
                true
            }
            Event::Downloading => {
                self.phase = Some("downloading a model".into());
                true
            }
            Event::Note(message) => {
                self.message = Some(message.clone());
                false
            }
        }
    }
}

/// Video files under the given paths (folders searched through), skipping copies already 3D.
pub fn expand(paths: &[PathBuf]) -> Vec<PathBuf> {
    fn visit(path: &Path, out: &mut Vec<PathBuf>, depth: usize) {
        let name = path.file_name().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
        if name.starts_with('.') {
            return;
        }
        if path.is_dir() {
            if depth > 8 {
                return;
            }
            let mut entries: Vec<PathBuf> = std::fs::read_dir(path).map(|d| d.filter_map(|e| e.ok().map(|e| e.path())).collect()).unwrap_or_default();
            entries.sort();
            for entry in entries {
                visit(&entry, out, depth + 1);
            }
        } else if path.is_file() {
            let extension = path.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
            let made = name.contains(".3d.") || name.ends_with(".spatial.mov") || name.contains(".sample.");
            if VIDEO_EXTENSIONS.contains(&extension.as_str()) && !made {
                out.push(path.to_path_buf());
            }
        }
    }
    let mut out = Vec::new();
    for path in paths {
        visit(path, &mut out, 0);
    }
    out
}

#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct QueueFile {
    pub next_id: u64,
    /// What a dropped or picked file is converted with.
    pub defaults: ConvertOptions,
    pub output_dir: Option<PathBuf>,
    pub jobs: Vec<ConvertJob>,
}

impl QueueFile {
    /// The queue as saved; a job that was running when the app quit is queued again.
    pub fn load(path: &Path) -> QueueFile {
        let mut file: QueueFile = std::fs::read(path).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
        for job in &mut file.jobs {
            if job.state == JobState::Running {
                job.state = JobState::Queued;
                job.message = Some("Carrying on from where it stopped.".into());
            }
        }
        file.next_id = file.next_id.max(file.jobs.iter().map(|j| j.id + 1).max().unwrap_or(1));
        file
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        write_atomic(path, &serde_json::to_vec_pretty(self)?, false)
    }

    /// Adds every video under `paths` with the current defaults; returns how many.
    pub fn add(&mut self, paths: &[PathBuf]) -> usize {
        let files = expand(paths);
        for input in &files {
            // The same file with the same options already waiting isn't added twice.
            let waiting = self.jobs.iter().any(|j| {
                &j.input == input && j.options == self.defaults && j.output_dir == self.output_dir && matches!(j.state, JobState::Queued | JobState::Running)
            });
            if waiting {
                continue;
            }
            self.jobs.push(ConvertJob {
                id: self.next_id.max(1),
                input: input.clone(),
                options: self.defaults.clone(),
                output_dir: self.output_dir.clone(),
                state: JobState::Queued,
                phase: None,
                progress: 0.0,
                output: None,
                message: None,
            });
            self.next_id = self.next_id.max(1) + 1;
        }
        files.len()
    }

    fn job(&mut self, id: u64) -> Result<&mut ConvertJob, String> {
        self.jobs.iter_mut().find(|j| j.id == id).ok_or_else(|| "That conversion is gone.".to_string())
    }
}

pub struct ConvertQueue {
    path: PathBuf,
    file: Mutex<QueueFile>,
    /// The running job, and its stopper.
    current: Mutex<Option<(u64, tokio::sync::oneshot::Sender<()>)>>,
    wake: Notify,
    quitting: AtomicBool,
    settings: Arc<Mutex<Settings>>,
    on_change: Mutex<Option<Arc<dyn Fn() + Send + Sync>>>,
}

impl ConvertQueue {
    pub fn new(path: PathBuf, settings: Arc<Mutex<Settings>>) -> ConvertQueue {
        ConvertQueue {
            file: Mutex::new(QueueFile::load(&path)),
            path,
            current: Default::default(),
            wake: Notify::new(),
            quitting: AtomicBool::new(false),
            settings,
            on_change: Default::default(),
        }
    }

    pub fn on_change(&self, f: Arc<dyn Fn() + Send + Sync>) {
        *self.on_change.lock().unwrap() = Some(f);
    }

    fn changed(&self) {
        let f = self.on_change.lock().unwrap().clone();
        if let Some(f) = f {
            f();
        }
    }

    fn save(&self) {
        if let Err(e) = self.file.lock().unwrap().save(&self.path) {
            tracing::warn!(error = %e, "couldn't save the conversion queue");
        }
    }

    fn edit<T>(&self, f: impl FnOnce(&mut QueueFile) -> Result<T, String>) -> Result<T, String> {
        let result = f(&mut self.file.lock().unwrap())?;
        self.save();
        self.changed();
        self.wake.notify_one();
        Ok(result)
    }

    pub fn add(&self, paths: &[PathBuf]) -> Result<usize, String> {
        if !AVAILABLE {
            return Err(UNAVAILABLE_REASON.into());
        }
        let added = self.edit(|q| Ok(q.add(paths)))?;
        if added == 0 {
            return Err("No video files there.".into());
        }
        Ok(added)
    }

    pub fn set_defaults(&self, options: ConvertOptions) -> Result<(), String> {
        let options = options.checked()?;
        self.edit(|q| {
            q.defaults = options;
            Ok(())
        })
    }

    pub fn set_output_dir(&self, dir: Option<PathBuf>) -> Result<(), String> {
        self.edit(|q| {
            q.output_dir = dir;
            Ok(())
        })
    }

    pub fn cancel(&self, id: u64) -> Result<(), String> {
        let mut current = self.current.lock().unwrap();
        if current.as_ref().is_some_and(|(running, _)| *running == id) {
            if let Some((_, stop)) = current.take() {
                let _ = stop.send(());
            }
            return Ok(());
        }
        drop(current);
        self.edit(|q| {
            let job = q.job(id)?;
            if job.state == JobState::Queued {
                job.state = JobState::Cancelled;
            }
            Ok(())
        })
    }

    /// Queues a finished, failed or cancelled job again (a cancelled one resumes from its depth).
    pub fn retry(&self, id: u64) -> Result<(), String> {
        self.edit(|q| {
            let job = q.job(id)?;
            if job.state != JobState::Running {
                job.state = JobState::Queued;
                job.message = None;
                job.progress = 0.0;
                job.phase = None;
            }
            Ok(())
        })
    }

    pub fn remove(&self, id: u64) -> Result<(), String> {
        self.edit(|q| {
            if q.job(id)?.state == JobState::Running {
                return Err("Cancel it first.".into());
            }
            q.jobs.retain(|j| j.id != id);
            Ok(())
        })
    }

    pub fn clear_finished(&self) -> Result<(), String> {
        self.edit(|q| {
            q.jobs.retain(|j| matches!(j.state, JobState::Queued | JobState::Running));
            Ok(())
        })
    }

    pub fn output_of(&self, id: u64) -> Option<PathBuf> {
        let file = self.file.lock().unwrap();
        file.jobs.iter().find(|j| j.id == id).and_then(|j| j.output.clone())
    }

    pub fn status(&self) -> Value {
        let file = self.file.lock().unwrap();
        let settings = self.settings.lock().unwrap().clone();
        json!({
            "available": AVAILABLE,
            "unavailableReason": (!AVAILABLE).then_some(UNAVAILABLE_REASON),
            "cli": sidecar::find_tool(&settings, ToolKind::Convert).map(|t| t.program.to_string_lossy().into_owned()),
            "defaults": file.defaults,
            "outputDir": file.output_dir,
            "sampleSeconds": SAMPLE_SECONDS,
            "jobs": file.jobs,
        })
    }

    pub fn jobs(&self) -> Vec<ConvertJob> {
        self.file.lock().unwrap().jobs.clone()
    }

    /// Stops the running conversion for a quit: it stays queued, to carry on next time.
    pub async fn shutdown(&self) {
        self.quitting.store(true, Ordering::SeqCst);
        let stop = self.current.lock().unwrap().take();
        if let Some((_, stop)) = stop {
            let _ = stop.send(());
            for _ in 0..50 {
                let running = self.file.lock().unwrap().jobs.iter().any(|j| j.state == JobState::Running);
                if !running {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }
    }

    /// Runs queued jobs one at a time, forever.
    pub async fn run(self: Arc<Self>) {
        if !AVAILABLE {
            return;
        }
        loop {
            if self.quitting.load(Ordering::SeqCst) {
                return;
            }
            let next = {
                let mut file = self.file.lock().unwrap();
                file.jobs.iter_mut().find(|j| j.state == JobState::Queued).map(|job| {
                    job.state = JobState::Running;
                    job.message = None;
                    job.clone()
                })
            };
            match next {
                Some(job) => {
                    self.save();
                    self.changed();
                    self.run_one(job).await;
                }
                None => {
                    let _ = tokio::time::timeout(Duration::from_secs(5), self.wake.notified()).await;
                }
            }
        }
    }

    fn update(&self, id: u64, f: impl FnOnce(&mut ConvertJob)) {
        let mut file = self.file.lock().unwrap();
        if let Some(job) = file.jobs.iter_mut().find(|j| j.id == id) {
            f(job);
        }
    }

    async fn run_one(&self, job: ConvertJob) {
        let settings = self.settings.lock().unwrap().clone();
        let finish = |state: JobState, message: String| {
            self.update(job.id, |j| {
                j.state = state;
                j.message = Some(message);
            });
            self.save();
            self.changed();
        };
        let Some(tool) = sidecar::find_tool(&settings, ToolKind::Convert) else {
            finish(JobState::Failed, sidecar::missing_tool(ToolKind::Convert).into());
            return;
        };
        let cli = tool.program.clone();
        let variant = crate::models::effective(settings.depth_variant.as_deref(), &settings.licences_accepted);
        let args = tool.arguments(job.arguments_with(settings.depth_model.as_deref(), variant.variant));
        tracing::info!(id = job.id, "converting");
        let mut child = match sidecar::spawn(&cli, &args, true) {
            Ok(child) => child,
            Err(e) => {
                finish(JobState::Failed, format!("Couldn't start {}: {e}", cli.display()));
                return;
            }
        };
        let stop = child.take_stopper().expect("a new sidecar has its stopper");
        *self.current.lock().unwrap() = Some((job.id, stop));
        let mut last_note: Option<String> = None;
        let mut skipped = false;
        let mut last_saved = Instant::now();
        let mut last_told = Instant::now();
        while let Some(line) = child.lines.recv().await {
            tracing::debug!(line = %line, "convert");
            let Some(event) = parse_line(&line) else { continue };
            if let Event::Note(note) = &event {
                last_note = Some(note.clone());
            }
            skipped |= matches!(event, Event::Skipped(_));
            let mut tell = false;
            self.update(job.id, |j| {
                tell = j.apply(&event);
                // The job is still running until the process is gone.
                if j.state == JobState::Skipped {
                    j.state = JobState::Running;
                }
            });
            if last_saved.elapsed() > Duration::from_secs(2) {
                self.save();
                last_saved = Instant::now();
            }
            if tell && last_told.elapsed() > Duration::from_millis(250) {
                self.changed();
                last_told = Instant::now();
            }
        }
        let status = child.exit.await;
        let stopped = {
            let mut current = self.current.lock().unwrap();
            let stopped = current.as_ref().is_none_or(|(id, _)| *id != job.id);
            if !stopped {
                *current = None;
            }
            stopped
        };
        // (state, message): None keeps what the run said last (a skip's reason).
        let (state, message) = match status {
            _ if stopped && self.quitting.load(Ordering::SeqCst) => (JobState::Queued, Some("Carrying on from where it stopped.".into())),
            _ if stopped => (JobState::Cancelled, Some("Cancelled. Retry carries on from the finished depth.".into())),
            Ok(Ok(status)) if status.success() && skipped => (JobState::Skipped, None),
            Ok(Ok(status)) if status.success() => (JobState::Done, Some(String::new())),
            Ok(Ok(status)) => (JobState::Failed, Some(last_note.unwrap_or_else(|| format!("vergence convert stopped ({status})")))),
            Ok(Err(e)) => (JobState::Failed, Some(e.to_string())),
            Err(e) => (JobState::Failed, Some(e.to_string())),
        };
        self.update(job.id, |j| {
            j.state = state;
            if state == JobState::Done {
                j.progress = 1.0;
            }
            if let Some(message) = message {
                j.message = Some(message).filter(|m| !m.is_empty());
            }
        });
        self.save();
        self.changed();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job(options: ConvertOptions) -> ConvertJob {
        ConvertJob {
            id: 1,
            input: PathBuf::from("/films/Film.mkv"),
            options,
            output_dir: None,
            state: JobState::Queued,
            phase: None,
            progress: 0.0,
            output: None,
            message: None,
        }
    }

    fn strings(args: Vec<OsString>) -> Vec<String> {
        args.into_iter().map(|a| a.to_string_lossy().into_owned()).collect()
    }

    #[test]
    fn builds_the_command_line() {
        assert_eq!(
            strings(job(ConvertOptions::default()).arguments(None)),
            ["/films/Film.mkv", "--layout", "fsbs", "--download-model"]
        );
        // A larger model the person chose (and accepted) is downloaded by name; a given one wins.
        assert_eq!(
            strings(job(ConvertOptions::default()).arguments_with(None, "base")),
            ["/films/Film.mkv", "--layout", "fsbs", "--variant", "base", "--download-model"]
        );
        assert_eq!(strings(job(ConvertOptions::default()).arguments_with(Some("/m/x.onnx"), "base")).last().unwrap(), "/m/x.onnx");
        let mut j = job(ConvertOptions { layout: Layout::Spatial, strength: Some(0.012), convergence: Some(0.5), sample: true, force: true, upscale: false });
        j.output_dir = Some(PathBuf::from("/out/3D"));
        assert_eq!(
            strings(j.arguments(Some("/models/depth.mlpackage"))),
            [
                "/films/Film.mkv", "--layout", "spatial", "--strength", "0.012", "--convergence", "0.5", "--sample", "10",
                "--output", "/out/3D/", "--model", "/models/depth.mlpackage", "--force",
            ]
        );
        let upscaled = job(ConvertOptions { layout: Layout::Flat, upscale: true, ..Default::default() });
        assert_eq!(
            strings(upscaled.arguments(Some("/m/d.mlpackage"))),
            ["/films/Film.mkv", "--layout", "2d", "--upscale", "2x", "--model", "/m/d.mlpackage", "--download-model"]
        );
        assert!(ConvertOptions { layout: Layout::Flat, ..Default::default() }.checked().is_err());
        assert_eq!(overall(Layout::Flat, "picture", 0.5), 0.5);
        assert!(ConvertOptions { strength: Some(0.2), ..Default::default() }.checked().is_err());
        assert!(ConvertOptions { convergence: Some(-1.0), ..Default::default() }.checked().is_err());
    }

    #[test]
    fn reads_converts_output() {
        assert_eq!(parse_line("depth  42%"), Some(Event::Progress("depth".into(), 0.42)));
        assert_eq!(parse_line("  picture 100%"), Some(Event::Progress("picture".into(), 1.0)));
        assert_eq!(parse_line("depth: 240 frames in 4.1 s (58.5 fps)"), Some(Event::StageDone("depth".into())));
        assert_eq!(parse_line("depth: 240 frames from an earlier run"), Some(Event::StageDone("depth".into())));
        assert_eq!(parse_line("depth track: 4 records -> /x.vgat"), None);
        assert_eq!(parse_line("spatial video: 240 frames, 20.0 Mbps"), Some(Event::StageDone("spatial".into())));
        assert_eq!(parse_line("done: 240 frames in 9.0 s, 26.7 fps overall"), Some(Event::Done));
        assert_eq!(
            parse_line("vergence: Film.mkv -> /films/Film.3D.FSBS.mp4"),
            Some(Event::Output(PathBuf::from("/films/Film.3D.FSBS.mp4")))
        );
        assert_eq!(
            parse_line("vergence: Film.mkv: Film.3D.FSBS.mp4 is already made (--force to redo)"),
            Some(Event::Skipped("Film.3D.FSBS.mp4 is already made (--force to redo)".into()))
        );
        assert_eq!(
            parse_line("vergence: Film.mkv: already 3D (side by side), skipped"),
            Some(Event::Skipped("already 3D (side by side), skipped".into()))
        );
        assert_eq!(parse_line("vergence: Film.mkv: the file is damaged"), Some(Event::Note("the file is damaged".into())));
        assert_eq!(parse_line("vergence: downloading Depth Anything V2 Small from x"), Some(Event::Downloading));
        assert_eq!(parse_line("  full side by side, 1920×1080 (eyes 1920×1080), HEVC 20.0 Mbps"), None);
    }

    #[test]
    fn progress_spans_the_stages() {
        let mut j = job(ConvertOptions::default());
        j.apply(&Event::Progress("depth".into(), 0.5));
        assert!((j.progress - 0.25).abs() < 1e-9);
        j.apply(&Event::StageDone("depth".into()));
        assert!((j.progress - 0.5).abs() < 1e-9);
        j.apply(&Event::Progress("picture".into(), 0.5));
        assert!((j.progress - 0.75).abs() < 1e-9);
        // A redraw that goes back never moves the bar back.
        j.apply(&Event::Progress("picture".into(), 0.1));
        assert!((j.progress - 0.75).abs() < 1e-9);
        assert!((overall(Layout::Spatial, "spatial", 0.0) - 0.85).abs() < 1e-9);
    }

    #[test]
    fn the_queue_persists_and_resumes() {
        let dir = tempfile::tempdir().unwrap();
        let media = dir.path().join("media");
        std::fs::create_dir_all(media.join("season")).unwrap();
        for name in ["a.mp4", "season/b.MKV", "notes.txt", "a.3D.FSBS.mp4", ".hidden.mp4", "a.sample.3D.FSBS.mp4"] {
            std::fs::write(media.join(name), b"x").unwrap();
        }
        let path = dir.path().join("queue.json");
        let mut file = QueueFile::load(&path);
        file.defaults = ConvertOptions { layout: Layout::Hsbs, ..Default::default() };
        assert_eq!(file.add(std::slice::from_ref(&media)), 2);
        // Added again while waiting: not twice.
        assert_eq!(file.add(&[media.join("a.mp4")]), 1);
        assert_eq!(file.jobs.len(), 2);
        assert_eq!(file.jobs[0].input, media.join("a.mp4"));
        assert_eq!(file.jobs[1].input, media.join("season/b.MKV"));
        assert_eq!(file.jobs[1].options.layout, Layout::Hsbs);
        file.jobs[0].state = JobState::Running;
        file.jobs[0].progress = 0.4;
        file.jobs[1].state = JobState::Done;
        file.save(&path).unwrap();

        // After a quit, the running one is queued again and new ids don't collide.
        let mut reloaded = QueueFile::load(&path);
        assert_eq!(reloaded.jobs[0].state, JobState::Queued);
        assert_eq!(reloaded.jobs[1].state, JobState::Done);
        assert_eq!(reloaded.defaults.layout, Layout::Hsbs);
        std::fs::write(media.join("c.mov"), b"x").unwrap();
        reloaded.add(&[media.join("c.mov")]);
        let ids: Vec<u64> = reloaded.jobs.iter().map(|j| j.id).collect();
        assert_eq!(ids, [1, 2, 3]);
    }

    /// The real thing: a short generated clip through `vergence convert`, as the app runs it.
    /// Needs the release CLI, ffmpeg and a depth model; run with
    /// `cargo test -p vergence-desktop -- --ignored converts_a_clip`.
    #[cfg(target_os = "macos")]
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn converts_a_clip() {
        let settings = Settings::default();
        let Some(_) = sidecar::find_cli(&settings) else { panic!("no vergence CLI") };
        let ffmpeg = sidecar::find_ffmpeg().expect("ffmpeg");
        let dir = tempfile::tempdir().unwrap();
        let clip = dir.path().join("clip.mp4");
        let made = std::process::Command::new(ffmpeg)
            .args(["-v", "error", "-f", "lavfi", "-i", "testsrc2=size=480x270:rate=24:duration=2", "-pix_fmt", "yuv420p", "-c:v", "libx264"])
            .arg(&clip)
            .status()
            .unwrap();
        assert!(made.success());
        let queue = Arc::new(ConvertQueue::new(dir.path().join("queue.json"), Arc::new(Mutex::new(settings))));
        let seen = Arc::new(Mutex::new(Vec::<f64>::new()));
        {
            let (queue2, seen) = (Arc::downgrade(&queue), seen.clone());
            queue.on_change(Arc::new(move || {
                if let Some(q) = queue2.upgrade() {
                    let progress = q.jobs().first().map(|j| j.progress).unwrap_or(0.0);
                    seen.lock().unwrap().push(progress);
                }
            }));
        }
        queue.add(std::slice::from_ref(&clip)).unwrap();
        let runner = tokio::spawn(queue.clone().run());
        let deadline = Instant::now() + Duration::from_secs(300);
        let job = loop {
            let job = queue.jobs()[0].clone();
            if !matches!(job.state, JobState::Queued | JobState::Running) {
                break job;
            }
            assert!(Instant::now() < deadline, "the conversion took too long");
            tokio::time::sleep(Duration::from_millis(200)).await;
        };
        runner.abort();
        assert_eq!(job.state, JobState::Done, "{job:?}");
        let output = job.output.clone().expect("the output's path");
        assert_eq!(output, dir.path().join("clip.3D.FSBS.mp4"));
        assert!(std::fs::metadata(&output).unwrap().len() > 0);
        assert_eq!(job.progress, 1.0);
        // Progress came through on the way, not only at the end.
        let seen = seen.lock().unwrap();
        assert!(seen.iter().any(|p| *p > 0.0 && *p < 1.0), "{seen:?}");
        // The saved queue says the same.
        let saved = QueueFile::load(&dir.path().join("queue.json"));
        assert_eq!(saved.jobs[0].state, JobState::Done);
    }
}
