//! The command line, parsed as VergenceCommandLine's `Options.swift` parses
//! `vergence convert`'s (the same flags, values and errors), and the output
//! names (`OutputNames.converted`).

use std::path::{Path, PathBuf};

use vergence_convert::maths::Quality;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    FullSideBySide,
    HalfSideBySide,
    FullTopAndBottom,
    HalfTopAndBottom,
    Spatial,
}

impl Layout {
    fn parse(option: &str) -> Option<Layout> {
        Some(match option {
            "auto" | "sbs" | "fsbs" => Layout::FullSideBySide,
            "hsbs" => Layout::HalfSideBySide,
            "tab" | "ftab" | "ou" => Layout::FullTopAndBottom,
            "htab" | "hou" => Layout::HalfTopAndBottom,
            "spatial" | "mvhevc" => Layout::Spatial,
            _ => return None,
        })
    }

    pub fn is_side_by_side(self) -> bool {
        matches!(self, Layout::FullSideBySide | Layout::HalfSideBySide | Layout::Spatial)
    }

    pub fn is_half(self) -> bool {
        matches!(self, Layout::HalfSideBySide | Layout::HalfTopAndBottom)
    }

    pub fn name_suffix(self) -> &'static str {
        match self {
            Layout::FullSideBySide => ".3D.FSBS.mp4",
            Layout::HalfSideBySide => ".3D.HSBS.mp4",
            Layout::FullTopAndBottom => ".3D.FTAB.mp4",
            Layout::HalfTopAndBottom => ".3D.HTAB.mp4",
            Layout::Spatial => ".spatial.mov",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Layout::FullSideBySide => "full side by side",
            Layout::HalfSideBySide => "half side by side",
            Layout::FullTopAndBottom => "full top and bottom",
            Layout::HalfTopAndBottom => "half top and bottom",
            Layout::Spatial => "spatial video (MV-HEVC)",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    pub positional: Vec<String>,
    pub output: String,
    pub layout: Layout,
    pub strength: Option<f64>,
    pub convergence: Option<f64>,
    pub sample: Option<f64>,
    pub start: Option<f64>,
    pub frames: Option<usize>,
    pub model: Option<String>,
    pub download_model: bool,
    /// Which model `--download-model` fetches and an installed one is looked up by.
    pub variant: Option<String>,
    pub depth_track: Option<String>,
    pub quality: Quality,
    pub bitrate: Option<u64>,
    pub max_eye: Option<usize>,
    pub include_audio: bool,
    pub dry_run: bool,
    pub force: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            positional: Vec::new(),
            output: "-".into(),
            layout: Layout::FullSideBySide,
            strength: None,
            convergence: None,
            sample: None,
            start: None,
            frames: None,
            model: None,
            download_model: false,
            variant: None,
            depth_track: None,
            quality: Quality::Standard,
            bitrate: None,
            max_eye: None,
            include_audio: true,
            dry_run: false,
            force: false,
        }
    }
}

/// A command line that can't run: the problem, said after "vergence: ".
#[derive(Debug, PartialEq)]
pub struct UsageError(pub String);

/// `vergence convert`'s arguments (a leading `convert` is optional).
pub fn parse(args: &[String]) -> Result<Options, UsageError> {
    let mut o = Options::default();
    let mut i = usize::from(args.first().map(String::as_str) == Some("convert"));
    while i < args.len() {
        let arg = &args[i];
        i += 1;
        if !arg.starts_with('-') || arg == "-" || arg.parse::<f64>().is_ok() {
            o.positional.push(arg.clone());
            continue;
        }
        let (name, inline) = match arg.split_once('=') {
            Some((n, v)) if arg.starts_with("--") => (n.to_string(), Some(v.to_string())),
            _ => (arg.clone(), None),
        };
        let mut value = || -> Result<String, UsageError> {
            if let Some(v) = &inline {
                return Ok(v.clone());
            }
            let v = args.get(i).cloned().ok_or_else(|| UsageError(format!("{name} needs a value")))?;
            i += 1;
            Ok(v)
        };
        let number = |text: String, name: &str| -> Result<f64, UsageError> {
            text.parse::<f64>()
                .ok()
                .filter(|v| v.is_finite())
                .ok_or_else(|| UsageError(format!("{name} takes a number, not \"{text}\"")))
        };
        let flag = |inline: &Option<String>, name: &str| -> Result<(), UsageError> {
            if inline.is_some() { Err(UsageError(format!("{name} takes no value"))) } else { Ok(()) }
        };
        match name.as_str() {
            "--output" | "-o" => o.output = value()?,
            "--layout" | "-f" | "--format" => {
                let v = value()?.to_lowercase();
                o.layout = Layout::parse(&v).ok_or_else(|| UsageError(format!("unknown --layout {v}")))?;
            }
            "--strength" => {
                let v = number(value()?, &name)?;
                if !(0.0..=0.05).contains(&v) {
                    return Err(UsageError(format!("--strength takes a share of the frame's width, 0...0.05, not {v}")));
                }
                o.strength = Some(v);
            }
            "--convergence" => {
                let v = number(value()?, &name)?;
                if !(0.0..=1.0).contains(&v) {
                    return Err(UsageError(format!("--convergence takes 0...1, not {v}")));
                }
                o.convergence = Some(v);
            }
            "--sample" => {
                let v = number(value()?, &name)?;
                if v <= 0.0 {
                    return Err(UsageError("--sample takes seconds, more than 0".into()));
                }
                o.sample = Some(v);
            }
            "--start" | "--seek" => {
                let v = number(value()?, &name)?;
                if v < 0.0 {
                    return Err(UsageError("--start must be 0 or more".into()));
                }
                o.start = Some(v);
            }
            "--frames" => {
                let text = value()?;
                let v = whole(&text).ok_or_else(|| UsageError(format!("--frames takes a whole number, not \"{text}\"")))?;
                if v < 1 {
                    return Err(UsageError("--frames must be 1 or more".into()));
                }
                o.frames = Some(v as usize);
            }
            "--model" => o.model = Some(value()?),
            "--download-model" => {
                flag(&inline, &name)?;
                o.download_model = true;
            }
            "--variant" => o.variant = Some(value()?),
            "--depth-track" => o.depth_track = Some(value()?),
            "--quality" | "-q" => {
                let v = value()?.to_lowercase();
                o.quality = Quality::parse(&v).ok_or_else(|| UsageError(format!("unknown --quality {v}")))?;
            }
            "--bitrate" | "-b" => {
                let v = value()?;
                o.bitrate = Some(parse_bitrate(&v).ok_or_else(|| UsageError("bad --bitrate (try 20, 20M or 750K)".into()))?);
            }
            "--max-eye" => {
                let text = value()?;
                let v = whole(&text).ok_or_else(|| UsageError(format!("--max-eye takes a whole number, not \"{text}\"")))?;
                if v <= 0 {
                    return Err(UsageError(format!("--max-eye takes a size in pixels, not {v}")));
                }
                o.max_eye = Some(v as usize);
            }
            "--no-audio" => {
                flag(&inline, &name)?;
                o.include_audio = false;
            }
            "--dry-run" | "-n" => {
                flag(&inline, &name)?;
                o.dry_run = true;
            }
            "--force" | "-y" => {
                flag(&inline, &name)?;
                o.force = true;
            }
            _ => return Err(UsageError(format!("unknown option {name}"))),
        }
    }
    Ok(o)
}

fn whole(text: &str) -> Option<i64> {
    text.parse::<i64>().ok().or_else(|| text.parse::<f64>().ok().filter(|v| v.fract() == 0.0 && v.abs() < 9e15).map(|v| v as i64))
}

/// "20" and "20M" are megabits per second, "750K" kilobits, "20000000" bits.
pub fn parse_bitrate(text: &str) -> Option<u64> {
    let t = text.to_uppercase();
    let (digits, scale) = if let Some(d) = t.strip_suffix('M') {
        (d, 1e6)
    } else if let Some(d) = t.strip_suffix('K') {
        (d, 1e3)
    } else {
        (t.as_str(), 0.0)
    };
    let v: f64 = digits.parse().ok().filter(|v: &f64| *v > 0.0 && v.is_finite())?;
    let bits = (v * if scale > 0.0 { scale } else if v < 10_000.0 { 1e6 } else { 1.0 }).round();
    (bits > 0.0 && bits < 1e15).then_some(bits as u64)
}

/// VIDEO.ext becomes VIDEO.3D.FSBS.mp4 (by layout), VIDEO.sample.3D.FSBS.mp4
/// for a sample; beside the input, or in/at --output.
pub fn converted(input: &Path, o: &Options, single: bool) -> PathBuf {
    let stem = input.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let name = format!("{stem}{}{}", if o.sample.is_some() { ".sample" } else { "" }, o.layout.name_suffix());
    if o.output == "-" {
        return input.parent().unwrap_or(Path::new("")).join(name);
    }
    let target = PathBuf::from(&o.output);
    let names_folder = o.output.ends_with('/') || (cfg!(windows) && o.output.ends_with('\\'));
    if single && !target.is_dir() && !names_folder {
        return target;
    }
    target.join(name)
}

/// The videos among `paths` (folders searched), sorted, leaving out hidden
/// files, partial outputs and anything already converted.
pub fn video_inputs(paths: &[String], note: &mut dyn FnMut(String)) -> Vec<PathBuf> {
    const EXTENSIONS: [&str; 12] = ["mov", "mp4", "m4v", "mkv", "webm", "avi", "ts", "m2ts", "mts", "mpg", "mpeg", "wmv"];
    fn is_input(path: &Path) -> bool {
        let name = path.file_name().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
        let ext = path.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
        !name.starts_with('.') && EXTENSIONS.contains(&ext.as_str()) && !name.contains(".partial.")
    }
    fn walk(dir: &Path, out: &mut Vec<PathBuf>, depth: usize) {
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.file_name().is_some_and(|n| n.to_string_lossy().starts_with('.')) {
                continue;
            }
            if path.is_dir() {
                if depth < 32 {
                    walk(&path, out, depth + 1);
                }
            } else if is_input(&path) {
                out.push(path);
            }
        }
    }
    let mut found = Vec::new();
    for path in paths {
        let p = PathBuf::from(path);
        if !p.exists() {
            note(format!("vergence: {path}: no such file"));
        } else if p.is_dir() {
            walk(&p, &mut found, 0);
        } else {
            found.push(p);
        }
    }
    found.retain(|p| {
        let name = p.file_name().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
        !name.contains(".3d.") && !name.ends_with(".spatial.mov")
    });
    found.sort_by(|a, b| natural_cmp(&a.to_string_lossy(), &b.to_string_lossy()));
    found
}

/// Finder's order: case-insensitive, runs of digits by value.
fn natural_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    let (mut a, mut b) = (a.chars().peekable(), b.chars().peekable());
    loop {
        match (a.peek().copied(), b.peek().copied()) {
            (None, None) => return std::cmp::Ordering::Equal,
            (None, Some(_)) => return std::cmp::Ordering::Less,
            (Some(_), None) => return std::cmp::Ordering::Greater,
            (Some(x), Some(y)) if x.is_ascii_digit() && y.is_ascii_digit() => {
                let n = |it: &mut std::iter::Peekable<std::str::Chars>| {
                    let mut s = String::new();
                    while let Some(c) = it.peek().copied().filter(char::is_ascii_digit) {
                        s.push(c);
                        it.next();
                    }
                    s.trim_start_matches('0').to_string()
                };
                let (x, y) = (n(&mut a), n(&mut b));
                let order = x.len().cmp(&y.len()).then(x.cmp(&y));
                if order.is_ne() {
                    return order;
                }
            }
            (Some(x), Some(y)) => {
                let order = x.to_lowercase().cmp(y.to_lowercase());
                if order.is_ne() {
                    return order;
                }
                a.next();
                b.next();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
    }

    #[test]
    fn parses_what_the_desktop_app_passes() {
        let o = parse(&args("convert /films/Film.mkv --layout spatial --strength 0.012 --convergence 0.5 --sample 10 \
                             --output /out/3D/ --model /m/d.onnx --force"))
        .unwrap();
        assert_eq!(o.positional, vec!["/films/Film.mkv"]);
        assert_eq!(o.layout, Layout::Spatial);
        assert_eq!((o.strength, o.convergence, o.sample), (Some(0.012), Some(0.5), Some(10.0)));
        assert!(o.force && !o.download_model);
        let o = parse(&args("Film.mkv --layout=hsbs --download-model --start 30 --frames 48 -q high -b 20 --no-audio")).unwrap();
        assert_eq!(o.layout, Layout::HalfSideBySide);
        assert_eq!((o.start, o.frames, o.bitrate), (Some(30.0), Some(48), Some(20_000_000)));
        assert!(o.download_model && !o.include_audio && o.quality == Quality::High);
    }

    #[test]
    fn rejects_what_swift_rejects() {
        assert_eq!(parse(&args("a --strength 0.2")).unwrap_err().0, "--strength takes a share of the frame's width, 0...0.05, not 0.2");
        assert!(parse(&args("a --layout cube")).is_err());
        assert!(parse(&args("a --frames 0")).is_err());
        assert!(parse(&args("a --force=1")).is_err());
        assert!(parse(&args("a --wat")).is_err());
        assert!(parse(&args("a --sample")).is_err());
    }

    #[test]
    fn names_outputs_like_the_mac() {
        let o = Options::default();
        assert_eq!(converted(Path::new("/f/Film.mkv"), &o, true), PathBuf::from("/f/Film.3D.FSBS.mp4"));
        let o = Options { layout: Layout::HalfTopAndBottom, sample: Some(10.0), output: "/out/".into(), ..Options::default() };
        assert_eq!(converted(Path::new("/f/Film.mkv"), &o, true), PathBuf::from("/out/Film.sample.3D.HTAB.mp4"));
        let o = Options { output: "/out/x.mp4".into(), ..Options::default() };
        assert_eq!(converted(Path::new("/f/Film.mkv"), &o, true), PathBuf::from("/out/x.mp4"));
        assert_eq!(converted(Path::new("/f/Film.mkv"), &o, false), PathBuf::from("/out/x.mp4/Film.3D.FSBS.mp4"));
    }

    #[test]
    fn finds_inputs_in_folders() {
        let dir = tempfile::tempdir().unwrap();
        for name in ["b10.mp4", "b9.mkv", ".hidden.mp4", "x.3D.FSBS.mp4", "y.partial.mp4", "notes.txt", "s.spatial.mov"] {
            std::fs::write(dir.path().join(name), b"").unwrap();
        }
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("sub/a.MOV"), b"").unwrap();
        let mut notes = Vec::new();
        let found = video_inputs(&[dir.path().to_string_lossy().into_owned(), "/nope/x.mp4".into()], &mut |n| notes.push(n));
        let names: Vec<String> = found.iter().map(|p| p.strip_prefix(dir.path()).unwrap().to_string_lossy().into_owned()).collect();
        assert_eq!(names, vec!["b9.mkv", "b10.mp4", "sub/a.MOV"].into_iter().map(|s| s.replace('/', std::path::MAIN_SEPARATOR_STR)).collect::<Vec<_>>());
        assert_eq!(notes, vec!["vergence: /nope/x.mp4: no such file"]);
    }
}
