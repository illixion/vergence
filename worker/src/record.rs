//! Analysis records and tracks (`VGAT`), written as VergenceCore's
//! `AnalysisTrack.swift` writes them: the same fixed header, int16 samples at
//! the same scale, row deltas and raw DEFLATE (RFC 1951). The deflate stream
//! itself comes from flate2 rather than Apple's Compression, so the
//! compressed bytes differ from a Swift worker's; what they inflate to does not.
//!
//! ```text
//! track   = "VGAT" u8(version) u32(header JSON length) header-JSON  record*
//! record  = u32(length of what follows)  u8(kind) u8(eye) u8(encoding) u8(channels)
//!           u16(width) u16(height)  i64(time value) i64(end value) i32(timescale)
//!           f32(scale) f32(offset)  payload
//! ```

use std::collections::BTreeMap;
use std::io::Write;

use serde::Serialize;

use crate::preprocess::{STEP, STORAGE_FACTOR, area_downsample};

pub const KIND_FLOW_PAIR: u8 = 1;
pub const KIND_DISPARITY: u8 = 2;
pub const EYE_MONO: u8 = 0;
pub const EYE_RIGHT: u8 = 2;
pub const ENCODING_INT16: u8 = 2;
const DEFLATED: u8 = 0x80;

/// Media time as a rational.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Time {
    pub value: i64,
    pub timescale: i32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Record {
    pub kind: u8,
    pub eye: u8,
    /// The base encoding (`ENCODING_INT16`); `deflated` says whether the payload is compressed.
    pub encoding: u8,
    pub channels: u8,
    pub width: u16,
    pub height: u16,
    pub time: Time,
    pub end: Time,
    pub scale: f32,
    pub offset: f32,
    pub payload: Vec<u8>,
    pub deflated: bool,
}

impl Record {
    /// Planar float samples as int16, full range at `max_magnitude`
    /// (`AnalysisRecord.encoding(_:as: .int16, maxMagnitude:)`).
    #[allow(clippy::too_many_arguments)]
    pub fn int16(values: &[f32], kind: u8, eye: u8, channels: usize, width: usize, height: usize, time: Time, end: Time,
                 max_magnitude: f32) -> Record {
        assert_eq!(values.len(), channels * width * height, "sample count");
        let scale = max_magnitude / i16::MAX as f32;
        let mut payload = Vec::with_capacity(values.len() * 2);
        for &value in values {
            // A model's NaN must not take the producer down: store 0.
            let raw = if value.is_nan() { 0.0 } else { (value / scale).round().clamp(-(i16::MAX as f32), i16::MAX as f32) };
            payload.extend_from_slice(&(raw as i16).to_le_bytes());
        }
        Record {
            kind,
            eye,
            encoding: ENCODING_INT16,
            channels: channels as u8,
            width: width as u16,
            height: height as u16,
            time,
            end,
            scale,
            offset: 0.0,
            payload,
            deflated: false,
        }
    }

    fn bytes_per_sample(&self) -> usize {
        if self.encoding == 3 { 1 } else { 2 }
    }

    /// Row deltas (encode) or their running sums (decode), wrapping in the
    /// raw integer type.
    fn row_deltas(&self, data: &mut [u8], encode: bool) {
        let (rows, width) = (self.channels as usize * self.height as usize, self.width as usize);
        if self.bytes_per_sample() == 2 {
            for row in 0..rows {
                let mut previous = 0u16;
                for x in 0..width {
                    let at = 2 * (row * width + x);
                    let value = u16::from_le_bytes([data[at], data[at + 1]]);
                    let stored = if encode {
                        let delta = value.wrapping_sub(previous);
                        previous = value;
                        delta
                    } else {
                        previous = value.wrapping_add(previous);
                        previous
                    };
                    data[at..at + 2].copy_from_slice(&stored.to_le_bytes());
                }
            }
        } else {
            for row in 0..rows {
                let mut previous = 0u8;
                for x in 0..width {
                    let i = row * width + x;
                    if encode {
                        let value = data[i];
                        data[i] = value.wrapping_sub(previous);
                        previous = value;
                    } else {
                        previous = data[i].wrapping_add(previous);
                        data[i] = previous;
                    }
                }
            }
        }
    }

    /// Row-delta coded and raw-DEFLATE compressed.
    pub fn deflated(mut self) -> Record {
        if self.deflated {
            return self;
        }
        let mut plain = std::mem::take(&mut self.payload);
        self.row_deltas(&mut plain, true);
        let mut encoder = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(&plain).expect("deflating into memory");
        self.payload = encoder.finish().expect("deflating into memory");
        self.deflated = true;
        self
    }

    /// The plain payload back.
    #[cfg(test)]
    pub fn inflated(mut self) -> std::io::Result<Record> {
        if !self.deflated {
            return Ok(self);
        }
        let expected = self.channels as usize * self.width as usize * self.height as usize * self.bytes_per_sample();
        let mut plain = Vec::with_capacity(expected);
        std::io::Read::read_to_end(&mut flate2::read::DeflateDecoder::new(&self.payload[..]), &mut plain)?;
        if plain.len() != expected {
            return Err(std::io::Error::other("the payload inflates to the wrong size"));
        }
        self.row_deltas(&mut plain, false);
        self.payload = plain;
        self.deflated = false;
        Ok(self)
    }

    /// The samples as floats; the record must be plain.
    #[cfg(test)]
    pub fn values(&self) -> Vec<f32> {
        assert!(!self.deflated, "inflate the record first");
        self.payload.as_chunks::<2>().0.iter().map(|&b| i16::from_le_bytes(b) as f32 * self.scale + self.offset).collect()
    }

    /// The record with its length prefix.
    pub fn framed(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(4 + 36 + self.payload.len());
        out.extend_from_slice(&((36 + self.payload.len()) as u32).to_le_bytes());
        out.extend_from_slice(&[self.kind, self.eye, self.encoding | if self.deflated { DEFLATED } else { 0 }, self.channels]);
        out.extend_from_slice(&self.width.to_le_bytes());
        out.extend_from_slice(&self.height.to_le_bytes());
        out.extend_from_slice(&self.time.value.to_le_bytes());
        out.extend_from_slice(&self.end.value.to_le_bytes());
        out.extend_from_slice(&self.time.timescale.to_le_bytes());
        out.extend_from_slice(&self.scale.to_le_bytes());
        out.extend_from_slice(&self.offset.to_le_bytes());
        out.extend_from_slice(&self.payload);
        out
    }

    /// Parses one record body (after its length prefix).
    #[cfg(test)]
    pub fn parse(body: &[u8]) -> Option<Record> {
        if body.len() < 36 {
            return None;
        }
        let i64_at = |at: usize| i64::from_le_bytes(body[at..at + 8].try_into().unwrap());
        let f32_at = |at: usize| f32::from_le_bytes(body[at..at + 4].try_into().unwrap());
        let timescale = i32::from_le_bytes(body[24..28].try_into().unwrap());
        Some(Record {
            kind: body[0],
            eye: body[1],
            encoding: body[2] & 0x7F,
            channels: body[3],
            width: u16::from_le_bytes([body[4], body[5]]),
            height: u16::from_le_bytes([body[6], body[7]]),
            time: Time { value: i64_at(8), timescale },
            end: Time { value: i64_at(16), timescale },
            scale: f32_at(28),
            offset: f32_at(32),
            payload: body[36..].to_vec(),
            deflated: body[2] & DEFLATED != 0,
        })
    }
}

/// A track's opening JSON, VergenceCore's `AnalysisTrackHeader`.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Header<'a> {
    #[serde(rename = "mediaID")]
    media_id: &'a str,
    producer: &'a str,
    models: &'a BTreeMap<String, String>,
    source_width: u32,
    source_height: u32,
}

/// A track's opening bytes.
pub fn track_header(media_id: &str, producer: &str, models: &BTreeMap<String, String>, source_width: u32,
                    source_height: u32) -> Vec<u8> {
    let json = serde_json::to_vec(&Header { media_id, producer, models, source_width, source_height })
        .expect("a header always encodes");
    let mut out = b"VGAT\x01".to_vec();
    out.extend_from_slice(&(json.len() as u32).to_le_bytes());
    out.extend_from_slice(&json);
    out
}

/// RIFE's output at t = ½ as a `flowPair` record (`SharedFieldAnalysis.flowPairRecord`).
/// `flow`: 4 planar channels at the model's size, F½→0 (x, y) then F½→1 (x, y);
/// `mask`: frame 0's blend weight at ½.
pub fn flow_pair_record(flow: &[f32], mask: &[f32], width: usize, height: usize, time: Time, end: Time) -> Record {
    let f = STORAGE_FACTOR;
    let small = area_downsample(flow, 4, width, height, f);
    let m = area_downsample(mask, 1, width, height, f);
    let n = (width / f) * (height / f);
    // F0→1 ≈ 2·F½→1, F1→0 ≈ 2·F½→0, in stored pixels.
    let k = 2.0 / f as f32;
    let mut values = vec![0f32; 5 * n];
    for i in 0..n {
        values[i] = small[2 * n + i] * k;
        values[n + i] = small[3 * n + i] * k;
        values[2 * n + i] = small[i] * k;
        values[3 * n + i] = small[n + i] * k;
        values[4 * n + i] = m[i];
    }
    Record::int16(&values, KIND_FLOW_PAIR, EYE_MONO, 5, width / f, height / f, time, end, STEP * i16::MAX as f32)
}

/// NeuFlow's right→left flow as a `disparity` record on the right eye,
/// horizontal only (`SharedFieldAnalysis.disparityRecord`).
pub fn disparity_record(right_to_left: &[f32], width: usize, height: usize, time: Time) -> Record {
    let f = STORAGE_FACTOR;
    let x: Vec<f32> = area_downsample(&right_to_left[..width * height], 1, width, height, f)
        .into_iter()
        .map(|v| v / f as f32)
        .collect();
    Record::int16(&x, KIND_DISPARITY, EYE_RIGHT, 1, width / f, height / f, time, time, STEP * i16::MAX as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The synthetic model outputs `tests/fixtures/make-swift-fixtures.swift` uses.
    fn fixture_outputs() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let (fw, n) = (48usize, 48 * 24);
        let flow = (0..4 * n)
            .map(|i| {
                let (c, p) = (i / n, i % n);
                (c + 1) as f32 * (p as f32 * 0.05).sin() * 7.3 + (p % fw) as f32 * 0.11
            })
            .collect();
        let mask = (0..n).map(|i| 0.5 + 0.5 * (i as f32 * 0.03).cos()).collect();
        let disparity = (0..2 * n).map(|i| -3.7 + (i % fw) as f32 * 0.21 - (i / fw) as f32 * 0.05).collect();
        (flow, mask, disparity)
    }

    fn swift_records() -> Vec<Record> {
        let data = include_bytes!("../tests/fixtures/swift_records.vgat");
        let track = vergence_coordinator::track::parse(data).unwrap();
        assert_eq!(track.header.producer, "swift fixture");
        track.records.iter().map(|r| Record::parse(&data[r.frame.start + 4..r.frame.end]).unwrap()).collect()
    }

    #[test]
    fn decodes_what_the_swift_worker_writes() {
        let swift = swift_records();
        assert_eq!(swift.len(), 2);
        assert!(swift.iter().all(|r| r.deflated));
        let flow = swift[0].clone().inflated().unwrap();
        assert_eq!((flow.kind, flow.channels, flow.width, flow.height), (KIND_FLOW_PAIR, 5, 12, 6));
        assert_eq!(flow.scale, 1.0 / 32.0);
    }

    #[test]
    fn encodes_the_same_samples_as_swift() {
        // sin/cos may differ in the last bit between Swift's and Rust's libm,
        // so the samples are allowed one int16 step; the headers must match.
        let (flow, mask, disparity) = fixture_outputs();
        let t0 = Time { value: 1001 * 5, timescale: 24000 };
        let t1 = Time { value: 1001 * 6, timescale: 24000 };
        let ours = [flow_pair_record(&flow, &mask, 48, 24, t0, t1), disparity_record(&disparity, 48, 24, t1)];
        for (mine, swift) in ours.iter().zip(swift_records()) {
            let swift = swift.inflated().unwrap();
            let (mut a, mut b) = (mine.clone(), swift.clone());
            a.payload.clear();
            b.payload.clear();
            assert_eq!(a, b, "record headers");
            let worst = mine
                .payload
                .as_chunks::<2>()
                .0
                .iter()
                .zip(swift.payload.as_chunks::<2>().0)
                .map(|(&x, &y)| (i16::from_le_bytes(x) as i32 - i16::from_le_bytes(y) as i32).abs())
                .max()
                .unwrap();
            assert!(worst <= 1, "kind {} differs by {worst} steps", mine.kind);
        }
    }

    #[test]
    fn round_trips_through_the_coordinators_parser() {
        let (flow, mask, disparity) = fixture_outputs();
        let t0 = Time { value: 0, timescale: 24000 };
        let t1 = Time { value: 1001, timescale: 24000 };
        let models: BTreeMap<String, String> =
            [("flowPair".to_string(), "RIFE x onnx".to_string())].into_iter().collect();
        let mut track = track_header("title", "test", &models, 960, 1080);
        let records = [flow_pair_record(&flow, &mask, 48, 24, t0, t1), disparity_record(&disparity, 48, 24, t1)];
        for r in &records {
            track.extend(r.clone().deflated().framed());
        }
        let parsed = vergence_coordinator::track::parse(&track).unwrap();
        assert_eq!(parsed.header.models["flowPair"], "RIFE x onnx");
        assert_eq!((parsed.header.source_width, parsed.header.source_height), (960, 1080));
        assert_eq!(parsed.records.len(), 2);
        assert_eq!(parsed.records[0].kind, vergence_coordinator::track::Kind::FlowPair);
        assert_eq!(parsed.records[0].end.value, 1001);
        assert_eq!(parsed.records[1].eye, EYE_RIGHT);
        for (info, original) in parsed.records.iter().zip(&records) {
            let back = Record::parse(&track[info.frame.start + 4..info.frame.end]).unwrap().inflated().unwrap();
            assert_eq!(&back, original);
            let values = back.values();
            assert_eq!(values.len(), original.payload.len() / 2);
        }
    }

    #[test]
    fn stores_nan_as_zero_and_clamps() {
        let r = Record::int16(&[f32::NAN, 1e9, -1e9, 0.5], 1, 0, 1, 4, 1, Time { value: 0, timescale: 1 },
                              Time { value: 0, timescale: 1 }, STEP * i16::MAX as f32);
        assert_eq!(r.values(), vec![0.0, 1023.96875, -1023.96875, 0.5]);
    }
}
