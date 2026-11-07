//! One piece: decode its frames with ffmpeg, run the models, return the track.
//! The same steps, arguments and records as `vergence worker`'s `analyze`.

use std::collections::BTreeMap;
use std::io::Read;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::client::Lease;
use crate::engine::Engine;
use crate::preprocess::{DISPARITY_SIZE, FLOW_SIZE, eyes, rgb_planar};
use crate::record::{Time, disparity_record, flow_pair_record, track_header};

pub enum Failure {
    /// The machine is wanted for something else: give the lease back.
    Yielded(String),
    Error(String),
}

impl From<String> for Failure {
    fn from(e: String) -> Failure {
        Failure::Error(e)
    }
}

pub struct Piece<'a> {
    pub lease: &'a Lease,
    /// Where ffmpeg reads the media (the loopback forwarder).
    pub url: &'a str,
    pub ffmpeg: Option<&'a str>,
    pub models: &'a BTreeMap<String, String>,
    pub producer: &'a str,
}

/// ffmpeg's arguments for a piece: half a frame early so the first frame out
/// is exactly `firstFrame`, passthrough so rawvideo's constant-rate sync
/// doesn't repeat the first frame after a seek, and (H.264) no SEI, whose
/// frame-packing messages make ffmpeg 8 and 9 fail.
pub fn ffmpeg_args(lease: &Lease, url: &str) -> Vec<String> {
    let rate = lease.info.frame_rate;
    let wanted = frames_wanted(lease);
    let seek = (lease.first_frame as f64 - 0.5) * rate.den as f64 / rate.num as f64;
    let mut args: Vec<String> = ["-nostdin", "-v", "error"].map(String::from).to_vec();
    if lease.info.codec.eq_ignore_ascii_case("h264") {
        args.extend(["-bsf:v", "filter_units=remove_types=6"].map(String::from));
    }
    if seek > 0.0 {
        args.extend(["-ss".to_string(), format!("{seek:.6}")]);
    }
    args.extend(
        ["-i", url, "-map", "0:v:0", "-fps_mode", "passthrough", "-frames:v", &wanted.to_string(), "-f", "rawvideo",
         "-pix_fmt", "yuv420p", "-"]
            .map(String::from),
    );
    args
}

/// The piece's frames, and one past it when there is one: the last flow pair reaches it.
pub fn frames_wanted(lease: &Lease) -> i64 {
    lease.end_frame - lease.first_frame + i64::from(lease.end_frame < lease.frame_count)
}

pub fn run(piece: &Piece, engine: &mut Engine, stop: &AtomicBool) -> Result<Vec<u8>, Failure> {
    let lease = piece.lease;
    let mut kinds = lease.kinds.clone();
    kinds.sort();
    if kinds != ["disparity", "flowPair"] {
        return Err(format!("this worker makes Smooth Motion fields only, not {:?}", lease.kinds).into());
    }
    let (w, h) = (lease.info.width, lease.info.height);
    let side_by_side = match lease.info.stereo.as_str() {
        "halfSideBySide" | "fullSideBySide" => true,
        "halfTopAndBottom" | "fullTopAndBottom" => false,
        other => return Err(format!("the title's layout ({other}) isn't frame-packed stereo").into()),
    };
    if w == 0 || h == 0 || w % 4 != 0 || h % 4 != 0 {
        return Err(format!("the picture's size ({w}x{h}) is unusable").into());
    }
    let (eye_w, eye_h) = if side_by_side { (w / 2, h) } else { (w, h / 2) };

    let mut child = Command::new(piece.ffmpeg.unwrap_or("ffmpeg"))
        .args(ffmpeg_args(lease, piece.url))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("ffmpeg didn't start: {e}"))?;
    let mut stderr = child.stderr.take().expect("piped");
    let errors = std::thread::spawn(move || {
        let mut text = String::new();
        let _ = stderr.read_to_string(&mut text);
        text
    });
    let mut output = child.stdout.take().expect("piped");
    let result = frames(piece, engine, stop, &mut output, (w, h, side_by_side), (eye_w, eye_h));
    drop(output);
    let _ = child.kill();
    let status = child.wait().ok();
    let message = errors.join().unwrap_or_default();
    let (track, decoded) = result?;
    if decoded < lease.end_frame - lease.first_frame {
        let code = status.and_then(|s| s.code()).unwrap_or(-1);
        let message: String = message.chars().take(300).collect();
        return Err(format!("ffmpeg gave {decoded} of {} frames ({code}): {message}", frames_wanted(lease)).into());
    }
    Ok(track)
}

fn frames(piece: &Piece, engine: &mut Engine, stop: &AtomicBool, output: &mut impl Read,
          (w, h, side_by_side): (usize, usize, bool), (eye_w, eye_h): (usize, usize)) -> Result<(Vec<u8>, i64), Failure> {
    let lease = piece.lease;
    let rate = lease.info.frame_rate;
    let mut track = track_header(&lease.title, piece.producer, piece.models, eye_w as u32, eye_h as u32);
    let (fw, fh) = FLOW_SIZE;
    let (dw, dh) = DISPARITY_SIZE;
    let frame_bytes = w * h * 3 / 2;
    let mut frame = vec![0u8; frame_bytes];
    let mut previous: Option<(Vec<f32>, Time)> = None;
    let mut index = lease.first_frame;
    let end = lease.first_frame + frames_wanted(lease);
    while index < end {
        if stop.load(Ordering::Relaxed) {
            return Err(Failure::Yielded("asked to stop".into()));
        }
        if output.read_exact(&mut frame).is_err() {
            break;
        }
        let time = Time { value: index * rate.den, timescale: rate.num as i32 };
        let in_piece = index < lease.end_frame;
        let (left, right) = eyes(&frame, w, h, side_by_side);
        let left_flow = rgb_planar(left[0], left[1], left[2], fw, fh, 1.0);
        let Engine { flow, disparity, .. } = engine;
        // The two models run at once, as the Swift worker runs them.
        let (flow_out, disparity_out) = std::thread::scope(|scope| {
            let flow_job = previous.as_ref().map(|(prev, _)| scope.spawn(|| flow.run(prev, &left_flow)));
            let disparity_out = in_piece.then(|| {
                let l = rgb_planar(left[0], left[1], left[2], dw, dh, 1.0);
                let r = rgb_planar(right[0], right[1], right[2], dw, dh, 1.0);
                disparity.run(&l, &r)
            });
            (flow_job.map(|j| j.join().expect("the flow model panicked")), disparity_out)
        });
        if let (Some((_, prev_time)), Some(result)) = (&previous, flow_out) {
            let (f, m) = result?;
            track.extend(flow_pair_record(&f, &m, fw, fh, *prev_time, time).deflated().framed());
        }
        if let Some(result) = disparity_out {
            track.extend(disparity_record(&result?, dw, dh, time).deflated().framed());
        }
        previous = Some((left_flow, time));
        index += 1;
    }
    Ok((track, index - lease.first_frame))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::{Info, Rate, Source};

    fn lease(first: i64, end: i64, count: i64, codec: &str) -> Lease {
        Lease {
            lease: 1,
            title: "t".into(),
            source: Source { server: "jellyfin".into(), id: "x".into() },
            kinds: vec!["flowPair".into(), "disparity".into()],
            piece: 0,
            first_frame: first,
            end_frame: end,
            frame_count: count,
            info: Info {
                frame_rate: Rate { num: 24000, den: 1001 },
                width: 1920,
                height: 1080,
                codec: codec.into(),
                stereo: "halfSideBySide".into(),
                media_source_id: None,
            },
            lease_seconds: 60.0,
            live: false,
        }
    }

    #[test]
    fn asks_ffmpeg_for_what_the_swift_worker_asks_for() {
        let args = ffmpeg_args(&lease(48, 96, 240, "h264"), "http://127.0.0.1:1/s");
        assert_eq!(
            args.join(" "),
            "-nostdin -v error -bsf:v filter_units=remove_types=6 -ss 1.981146 -i http://127.0.0.1:1/s -map 0:v:0 \
             -fps_mode passthrough -frames:v 49 -f rawvideo -pix_fmt yuv420p -"
        );
        // The first piece doesn't seek; the last has no frame past it; HEVC keeps its SEI.
        let args = ffmpeg_args(&lease(0, 48, 48, "hevc"), "u").join(" ");
        assert!(!args.contains("-ss") && !args.contains("-bsf") && args.contains("-frames:v 48"));
    }
}
