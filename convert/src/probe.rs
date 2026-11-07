//! What ffprobe says about a source, and where ffmpeg and ffprobe are.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde::Deserialize;

/// ffmpeg and ffprobe: VERGENCE_FFMPEG (ffmpeg's path; ffprobe beside it),
/// else PATH.
#[derive(Clone, Debug)]
pub struct Tools {
    pub ffmpeg: PathBuf,
    pub ffprobe: PathBuf,
}

impl Tools {
    pub fn locate() -> Result<Tools, String> {
        let exe = |name: &str| if cfg!(windows) { format!("{name}.exe") } else { name.to_string() };
        let mut dirs: Vec<PathBuf> = Vec::new();
        let explicit = std::env::var_os("VERGENCE_FFMPEG").map(PathBuf::from).filter(|p| p.is_file());
        if let Some(dir) = explicit.as_ref().and_then(|p| p.parent()) {
            dirs.push(dir.to_path_buf());
        }
        if let Some(path) = std::env::var_os("PATH") {
            dirs.extend(std::env::split_paths(&path));
        }
        if cfg!(target_os = "macos") {
            dirs.extend(["/opt/homebrew/bin", "/usr/local/bin"].map(PathBuf::from));
        }
        let find = |name: &str| dirs.iter().map(|d| d.join(exe(name))).find(|p| p.is_file());
        let ffmpeg = explicit.or_else(|| find("ffmpeg"));
        match (ffmpeg, find("ffprobe")) {
            (Some(ffmpeg), Some(ffprobe)) => Ok(Tools { ffmpeg, ffprobe }),
            _ => Err("converting needs ffmpeg and ffprobe: put them on PATH or set VERGENCE_FFMPEG to ffmpeg's path".into()),
        }
    }

    /// A command that never shows a console window on Windows.
    pub fn command(program: &Path) -> Command {
        #[allow(unused_mut)]
        let mut command = Command::new(program);
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x08000000);
        }
        command
    }

    pub fn probe(&self, input: &Path) -> Result<Probe, String> {
        let output = Self::command(&self.ffprobe)
            .args(["-v", "error", "-show_streams", "-show_format", "-of", "json"])
            .arg(input)
            .stdin(Stdio::null())
            .output()
            .map_err(|e| format!("ffprobe didn't start: {e}"))?;
        if !output.status.success() {
            let message = String::from_utf8_lossy(&output.stderr);
            return Err(format!("ffprobe can't read it: {}", message.trim().chars().take(300).collect::<String>()));
        }
        serde_json::from_slice(&output.stdout).map_err(|e| format!("ffprobe's answer: {e}"))
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct SideData {
    pub side_data_type: Option<String>,
    #[serde(rename = "type")]
    pub kind: Option<String>,
    pub max_luminance: Option<String>,
    pub min_luminance: Option<String>,
    pub red_x: Option<String>,
    pub red_y: Option<String>,
    pub green_x: Option<String>,
    pub green_y: Option<String>,
    pub blue_x: Option<String>,
    pub blue_y: Option<String>,
    pub white_point_x: Option<String>,
    pub white_point_y: Option<String>,
    pub max_content: Option<u32>,
    pub max_average: Option<u32>,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct Tags {
    #[serde(rename = "BPS")]
    pub bps: Option<String>,
    #[serde(rename = "BPS-eng")]
    pub bps_eng: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Stream {
    pub index: usize,
    pub codec_type: Option<String>,
    pub codec_name: Option<String>,
    pub profile: Option<String>,
    pub width: Option<usize>,
    pub height: Option<usize>,
    pub pix_fmt: Option<String>,
    pub avg_frame_rate: Option<String>,
    pub r_frame_rate: Option<String>,
    pub channels: Option<u32>,
    pub bit_rate: Option<String>,
    pub start_time: Option<String>,
    pub duration: Option<String>,
    pub color_range: Option<String>,
    pub color_space: Option<String>,
    pub color_primaries: Option<String>,
    pub color_transfer: Option<String>,
    pub view_ids_available: Option<String>,
    #[serde(default)]
    pub tags: Tags,
    #[serde(default)]
    pub side_data_list: Vec<SideData>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Format {
    pub duration: Option<String>,
    pub bit_rate: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Probe {
    #[serde(default)]
    pub streams: Vec<Stream>,
    pub format: Option<Format>,
}

/// A frame rate as a fraction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rate {
    pub num: u64,
    pub den: u64,
}

impl Rate {
    pub fn fps(self) -> f64 {
        self.num as f64 / self.den as f64
    }

    fn parse(text: &str) -> Option<Rate> {
        let (n, d) = text.split_once('/').unwrap_or((text, "1"));
        let (num, den) = (n.trim().parse::<u64>().ok()?, d.trim().parse::<u64>().ok()?);
        (num > 0 && den > 0).then_some(Rate { num, den })
    }
}

fn number(text: &Option<String>) -> Option<f64> {
    text.as_deref().and_then(|t| t.parse::<f64>().ok()).filter(|v| v.is_finite())
}

/// The colour of the picture, as ffmpeg names it.
#[derive(Clone, Debug, PartialEq)]
pub struct Colour {
    pub primaries: String,
    pub transfer: String,
    /// The YCbCr matrix the picture is decoded and encoded with.
    pub matrix: String,
    pub full_range: bool,
}

impl Probe {
    pub fn video(&self) -> Option<&Stream> {
        self.streams.iter().find(|s| s.codec_type.as_deref() == Some("video") && s.codec_name.as_deref() != Some("mjpeg")
            && s.codec_name.as_deref() != Some("png"))
    }

    pub fn audio(&self) -> Vec<&Stream> {
        self.streams.iter().filter(|s| s.codec_type.as_deref() == Some("audio")).collect()
    }

    pub fn duration(&self) -> f64 {
        let video = self.video();
        video
            .and_then(|v| number(&v.duration))
            .or_else(|| self.format.as_ref().and_then(|f| number(&f.duration)))
            .unwrap_or(0.0)
    }

    pub fn rate(&self) -> Rate {
        let video = self.video();
        video
            .and_then(|v| v.avg_frame_rate.as_deref().and_then(Rate::parse))
            .or_else(|| video.and_then(|v| v.r_frame_rate.as_deref().and_then(Rate::parse)))
            .unwrap_or(Rate { num: 30, den: 1 })
    }

    /// The video's own bit rate: its stream's, a Matroska statistics tag, or
    /// the file's less its audio.
    pub fn video_bits_per_second(&self) -> Option<f64> {
        let video = self.video()?;
        number(&video.bit_rate)
            .or_else(|| number(&video.tags.bps).or_else(|| number(&video.tags.bps_eng)))
            .or_else(|| {
                let total = number(&self.format.as_ref()?.bit_rate)?;
                let audio: f64 = self.audio().iter().filter_map(|a| number(&a.bit_rate).or_else(|| number(&a.tags.bps))).sum();
                Some(total - audio).filter(|v| *v > 0.0)
            })
    }

    pub fn video_start(&self) -> f64 {
        self.video().and_then(|v| number(&v.start_time)).unwrap_or(0.0)
    }

    /// PQ or HLG.
    pub fn is_hdr(&self) -> bool {
        matches!(self.video().and_then(|v| v.color_transfer.as_deref()), Some("smpte2084" | "arib-std-b67"))
    }

    /// More than 8 bits a component.
    pub fn bit_depth(&self) -> u32 {
        let format = self.video().and_then(|v| v.pix_fmt.as_deref()).unwrap_or("");
        ["16", "14", "12", "10", "9"].iter().find(|b| format.contains(*b)).and_then(|b| b.parse().ok()).unwrap_or(8)
    }

    /// The colour to decode with and to tag the output with: the source's,
    /// with Convolution's defaults for what it leaves out (BT.709 for HD and
    /// up, BT.601 below; BT.2020 and PQ for an untagged HDR picture).
    pub fn colour(&self) -> Colour {
        let video = self.video();
        let known = |v: Option<&String>| v.filter(|s| !s.is_empty() && s.as_str() != "unknown" && s.as_str() != "reserved").cloned();
        let hdr = self.is_hdr();
        let height = video.and_then(|v| v.height).unwrap_or(1080);
        let fallback_matrix = if hdr { "bt2020nc" } else if height >= 720 { "bt709" } else { "smpte170m" };
        let matrix = known(video.and_then(|v| v.color_space.as_ref())).unwrap_or_else(|| fallback_matrix.into());
        Colour {
            primaries: known(video.and_then(|v| v.color_primaries.as_ref()))
                .unwrap_or_else(|| if hdr { "bt2020" } else if height >= 720 { "bt709" } else { "smpte170m" }.into()),
            transfer: known(video.and_then(|v| v.color_transfer.as_ref())).unwrap_or_else(|| "bt709".into()),
            matrix,
            full_range: video.and_then(|v| v.color_range.as_deref()) == Some("pc")
                || video.and_then(|v| v.pix_fmt.as_deref()).is_some_and(|f| f.starts_with("yuvj")),
        }
    }

    /// A frame-packed or multiview layout the file already declares.
    pub fn already_3d(&self) -> Option<&'static str> {
        let video = self.video()?;
        if video.view_ids_available.as_deref().is_some_and(|ids| ids.contains(',')) {
            return Some("MV-HEVC spatial video");
        }
        let stereo = video.side_data_list.iter().find(|s| s.side_data_type.as_deref() == Some("Stereo 3D"))?;
        match stereo.kind.as_deref() {
            Some("side by side") => Some("side by side"),
            Some("top and bottom") => Some("over-under"),
            _ => None,
        }
    }

    /// x265's master-display and max-cll, when the stream carries them.
    pub fn hdr10_metadata(&self) -> Option<(String, Option<String>)> {
        let video = self.video()?;
        let mastering = video.side_data_list.iter().find(|s| s.side_data_type.as_deref() == Some("Mastering display metadata"))?;
        let ratio = |v: &Option<String>, scale: f64| -> Option<u64> {
            let text = v.as_deref()?;
            let (n, d) = text.split_once('/').unwrap_or((text, "1"));
            let (n, d) = (n.parse::<f64>().ok()?, d.parse::<f64>().ok()?);
            (d != 0.0).then(|| (n / d * scale).round() as u64)
        };
        let p = |v: &Option<String>| ratio(v, 50000.0);
        let master = format!(
            "G({},{})B({},{})R({},{})WP({},{})L({},{})",
            p(&mastering.green_x)?, p(&mastering.green_y)?, p(&mastering.blue_x)?, p(&mastering.blue_y)?,
            p(&mastering.red_x)?, p(&mastering.red_y)?, p(&mastering.white_point_x)?, p(&mastering.white_point_y)?,
            ratio(&mastering.max_luminance, 10000.0)?, ratio(&mastering.min_luminance, 10000.0)?
        );
        let cll = video
            .side_data_list
            .iter()
            .find(|s| s.side_data_type.as_deref() == Some("Content light level metadata"))
            .and_then(|c| Some(format!("{},{}", c.max_content?, c.max_average?)));
        Some((master, cll))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_ffprobes_answer() {
        let json = r#"{"streams":[{"index":0,"codec_type":"video","codec_name":"hevc","width":3840,"height":2160,
            "pix_fmt":"yuv420p10le","avg_frame_rate":"24000/1001","color_space":"bt2020nc","color_transfer":"smpte2084",
            "color_primaries":"bt2020","tags":{"BPS":"40000000"},"side_data_list":[{"side_data_type":"Mastering display metadata",
            "red_x":"34000/50000","red_y":"16000/50000","green_x":"13250/50000","green_y":"34500/50000","blue_x":"7500/50000",
            "blue_y":"3000/50000","white_point_x":"15635/50000","white_point_y":"16450/50000","min_luminance":"50/10000",
            "max_luminance":"10000000/10000"},{"side_data_type":"Content light level metadata","max_content":1000,"max_average":400}]},
            {"index":1,"codec_type":"audio","codec_name":"truehd","channels":8}],"format":{"duration":"60.0","bit_rate":"45000000"}}"#;
        let probe: Probe = serde_json::from_str(json).unwrap();
        assert!(probe.is_hdr() && probe.bit_depth() == 10);
        assert_eq!(probe.rate(), Rate { num: 24000, den: 1001 });
        assert_eq!(probe.video_bits_per_second(), Some(40_000_000.0));
        assert_eq!(probe.colour().matrix, "bt2020nc");
        let (master, cll) = probe.hdr10_metadata().unwrap();
        assert_eq!(master, "G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,50)");
        assert_eq!(cll.as_deref(), Some("1000,400"));
        assert_eq!(probe.already_3d(), None);
        let untagged: Probe = serde_json::from_str(r#"{"streams":[{"index":0,"codec_type":"video","height":480,"pix_fmt":"yuv420p"}]}"#).unwrap();
        assert_eq!(untagged.colour().matrix, "smpte170m");
        assert_eq!(untagged.bit_depth(), 8);
    }
}
