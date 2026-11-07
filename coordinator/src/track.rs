//! Analysis tracks (`VGAT`), as far as the coordinator needs them: read a
//! track's header and each record's fixed header, and write a track back out
//! from records it already holds. Payloads are never decoded here; the
//! format's full definition is VergenceCore's `AnalysisTrack.swift`.
//!
//! ```text
//! track   = "VGAT" u8(version) u32(header JSON length) header-JSON  record*
//! record  = u32(length of what follows)  u8(kind) u8(eye) u8(encoding) u8(channels)
//!           u16(width) u16(height)  i64(time value) i64(end value) i32(timescale)
//!           f32(scale) f32(offset)  payload
//! ```

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

pub const MAGIC: &[u8; 4] = b"VGAT";
pub const VERSION: u8 = 1;
/// The fixed part of a record body, before its payload.
pub const RECORD_HEADER_BYTES: usize = 4 + 4 + 8 + 8 + 4 + 4 + 4;
/// No real record comes near this; a length past it means a corrupt stream.
pub const MAX_RECORD_BYTES: usize = 64 << 20;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum TrackError {
    #[error("not an analysis track")]
    NotATrack,
    #[error("unsupported track version {0}")]
    UnsupportedVersion(u8),
    #[error("the track is truncated")]
    Truncated,
    #[error("a record has an unknown kind, eye or encoding")]
    UnknownValue,
    #[error("a record's payload has the wrong size")]
    PayloadSize,
    #[error("a record or header is too large")]
    TooLarge,
    #[error("the header is not valid JSON: {0}")]
    Header(String),
}

/// The record kinds, with VergenceCore's raw values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Kind {
    FlowPair = 1,
    Disparity = 2,
    Depth = 3,
    GainMap = 4,
}

impl Kind {
    pub fn from_raw(raw: u8) -> Option<Kind> {
        match raw {
            1 => Some(Kind::FlowPair),
            2 => Some(Kind::Disparity),
            3 => Some(Kind::Depth),
            4 => Some(Kind::GainMap),
            _ => None,
        }
    }

    /// The name used in a header's `models` map and in the API.
    pub fn name(self) -> &'static str {
        match self {
            Kind::FlowPair => "flowPair",
            Kind::Disparity => "disparity",
            Kind::Depth => "depth",
            Kind::GainMap => "gainMap",
        }
    }

    pub fn from_name(name: &str) -> Option<Kind> {
        [Kind::FlowPair, Kind::Disparity, Kind::Depth, Kind::GainMap].into_iter().find(|k| k.name() == name)
    }

    /// Pair kinds span two real frames (`time` to `end`).
    pub fn is_pair(self) -> bool {
        self == Kind::FlowPair
    }

    /// The eye a workload's records of this kind are laid out on (VergenceCore's
    /// `AnalysisEye`): disparity on the right eye (right→left), everything
    /// else on the left eye or the whole picture, written as mono (0).
    pub fn eye(self) -> u8 {
        match self {
            Kind::Disparity => 2,
            _ => 0,
        }
    }
}

/// A track's opening JSON, as VergenceCore's `AnalysisTrackHeader`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Header {
    #[serde(rename = "mediaID")]
    pub media_id: String,
    pub producer: String,
    #[serde(default)]
    pub models: BTreeMap<String, String>,
    pub source_width: u32,
    pub source_height: u32,
    /// The video stream ended at this frame (exclusive), inside the piece: a
    /// container whose audio runs past its picture. Only a worker that hit
    /// the end of the stream early says it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub video_end_frame: Option<i64>,
}

/// Media time as a rational.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Time {
    pub value: i64,
    pub timescale: i32,
}

impl Time {
    pub fn seconds(self) -> f64 {
        self.value as f64 / self.timescale as f64
    }
}

/// One record's fixed header, and where its whole frame (length prefix
/// included) sits in the bytes it was read from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordInfo {
    pub kind: Kind,
    pub eye: u8,
    pub encoding: u8,
    pub channels: u8,
    pub width: u16,
    pub height: u16,
    pub time: Time,
    pub end: Time,
    pub frame: std::ops::Range<usize>,
}

/// A whole track read from bytes: its header and every record's position.
#[derive(Clone, Debug)]
pub struct ParsedTrack {
    pub header: Header,
    pub records: Vec<RecordInfo>,
}

fn u16_at(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}

fn i64_at(b: &[u8], at: usize) -> i64 {
    i64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}

/// Reads the header; returns it and how many bytes it took.
pub fn parse_header(data: &[u8]) -> Result<(Header, usize), TrackError> {
    if data.len() < 9 {
        return Err(if MAGIC.starts_with(&data[..data.len().min(4)]) { TrackError::Truncated } else { TrackError::NotATrack });
    }
    if &data[..4] != MAGIC {
        return Err(TrackError::NotATrack);
    }
    if data[4] != VERSION {
        return Err(TrackError::UnsupportedVersion(data[4]));
    }
    let length = u32_at(data, 5) as usize;
    if length >= 1 << 20 {
        return Err(TrackError::TooLarge);
    }
    if data.len() < 9 + length {
        return Err(TrackError::Truncated);
    }
    let header: Header = serde_json::from_slice(&data[9..9 + length]).map_err(|e| TrackError::Header(e.to_string()))?;
    Ok((header, 9 + length))
}

/// Reads a record's fixed header from its body (after the length prefix).
fn parse_record(body: &[u8], frame: std::ops::Range<usize>) -> Result<RecordInfo, TrackError> {
    if body.len() < RECORD_HEADER_BYTES {
        return Err(TrackError::Truncated);
    }
    let kind = Kind::from_raw(body[0]).ok_or(TrackError::UnknownValue)?;
    let eye = body[1];
    if eye > 2 {
        return Err(TrackError::UnknownValue);
    }
    let encoding = body[2];
    let bytes_per_sample = match encoding & 0x7F {
        1 | 2 => 2,
        3 => 1,
        _ => return Err(TrackError::UnknownValue),
    };
    let deflated = encoding & 0x80 != 0;
    let channels = body[3];
    let width = u16_at(body, 4);
    let height = u16_at(body, 6);
    let time = i64_at(body, 8);
    let end = i64_at(body, 16);
    let timescale = u32_at(body, 24) as i32;
    if timescale <= 0 {
        return Err(TrackError::UnknownValue);
    }
    let expected = channels as usize * width as usize * height as usize * bytes_per_sample;
    let payload = body.len() - RECORD_HEADER_BYTES;
    let fits = if deflated { payload > 0 } else { payload == expected };
    if !fits {
        return Err(TrackError::PayloadSize);
    }
    Ok(RecordInfo {
        kind,
        eye,
        encoding,
        channels,
        width,
        height,
        time: Time { value: time, timescale },
        end: Time { value: end, timescale },
        frame,
    })
}

/// Reads a whole track: the header and every record's fixed header.
pub fn parse(data: &[u8]) -> Result<ParsedTrack, TrackError> {
    let (header, mut position) = parse_header(data)?;
    let mut records = Vec::new();
    while position < data.len() {
        if data.len() - position < 4 {
            return Err(TrackError::Truncated);
        }
        let length = u32_at(data, position) as usize;
        if length > MAX_RECORD_BYTES {
            return Err(TrackError::TooLarge);
        }
        let start = position + 4;
        if data.len() - start < length {
            return Err(TrackError::Truncated);
        }
        records.push(parse_record(&data[start..start + length], position..start + length)?);
        position = start + length;
    }
    Ok(ParsedTrack { header, records })
}

/// A track's opening bytes for `header`.
pub fn write_header(header: &Header) -> Vec<u8> {
    let json = serde_json::to_vec(header).expect("a header always encodes");
    let mut out = Vec::with_capacity(9 + json.len());
    out.extend_from_slice(MAGIC);
    out.push(VERSION);
    out.extend_from_slice(&(json.len() as u32).to_le_bytes());
    out.extend_from_slice(&json);
    out
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A record frame as VergenceCore writes one: uint8, one channel.
    pub(crate) fn record(kind: Kind, eye: u8, time: i64, end: i64, timescale: i32, w: u16, h: u16) -> Vec<u8> {
        let mut body = vec![kind as u8, eye, 3, 1];
        body.extend_from_slice(&w.to_le_bytes());
        body.extend_from_slice(&h.to_le_bytes());
        body.extend_from_slice(&time.to_le_bytes());
        body.extend_from_slice(&end.to_le_bytes());
        body.extend_from_slice(&(timescale as u32).to_le_bytes());
        body.extend_from_slice(&1f32.to_le_bytes());
        body.extend_from_slice(&0f32.to_le_bytes());
        body.extend(std::iter::repeat_n(7u8, w as usize * h as usize));
        let mut frame = (body.len() as u32).to_le_bytes().to_vec();
        frame.extend(body);
        frame
    }

    pub(crate) fn header(models: &[(&str, &str)]) -> Header {
        Header {
            media_id: "m".into(),
            producer: "test".into(),
            models: models.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
            source_width: 1920,
            source_height: 1080,
            video_end_frame: None,
        }
    }

    #[test]
    fn reads_back_what_it_writes() {
        let mut data = write_header(&header(&[("flowPair", "RIFE 480x288")]));
        data.extend(record(Kind::FlowPair, 0, 0, 1001, 24000, 4, 2));
        data.extend(record(Kind::Disparity, 2, 1001, 1001, 24000, 4, 2));
        let track = parse(&data).unwrap();
        assert_eq!(track.header.models["flowPair"], "RIFE 480x288");
        assert_eq!(track.records.len(), 2);
        assert_eq!(track.records[0].kind, Kind::FlowPair);
        assert_eq!(track.records[1].time, Time { value: 1001, timescale: 24000 });
        let second = &track.records[1].frame;
        assert_eq!(second.end, data.len());
    }

    #[test]
    fn reads_the_swift_header_keys() {
        let json = br#"{"mediaID":"ab","producer":"vergence analyze","models":{"flowPair":"x"},"sourceWidth":1920,"sourceHeight":1080}"#;
        let mut data = b"VGAT\x01".to_vec();
        data.extend_from_slice(&(json.len() as u32).to_le_bytes());
        data.extend_from_slice(json);
        let (header, used) = parse_header(&data).unwrap();
        assert_eq!(header.media_id, "ab");
        assert_eq!(header.source_height, 1080);
        assert_eq!(used, data.len());
    }

    #[test]
    fn rejects_damage() {
        assert_eq!(parse(b"NOPE\x01\0\0\0\0").unwrap_err(), TrackError::NotATrack);
        let mut data = write_header(&header(&[]));
        let mut rec = record(Kind::Depth, 0, 0, 0, 24, 4, 2);
        rec.truncate(rec.len() - 1);
        data.extend(rec);
        assert_eq!(parse(&data).unwrap_err(), TrackError::Truncated);
    }
}
