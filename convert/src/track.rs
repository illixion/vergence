//! Analysis tracks (`VGAT`, VergenceCore's `AnalysisTrack.swift`), as far as
//! depth records need them: the header and uint8 records, row-delta coded and
//! raw-DEFLATE compressed.

use std::collections::BTreeMap;
use std::io::Write;

use serde::Serialize;

const KIND_DEPTH: u8 = 3;
const EYE_MONO: u8 = 0;
const ENCODING_UINT8: u8 = 3;
const DEFLATED: u8 = 0x80;

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

pub fn header(media_id: &str, producer: &str, models: &BTreeMap<String, String>, source_width: u32, source_height: u32) -> Vec<u8> {
    let json = serde_json::to_vec(&Header { media_id, producer, models, source_width, source_height }).expect("a header encodes");
    let mut out = b"VGAT\x01".to_vec();
    out.extend_from_slice(&(json.len() as u32).to_le_bytes());
    out.extend_from_slice(&json);
    out
}

pub struct Record {
    width: u16,
    height: u16,
    time: i64,
    timescale: i32,
    scale: f32,
    offset: f32,
    payload: Vec<u8>,
}

impl Record {
    /// One mono depth map, a byte a sample: value = byte · scale + offset.
    pub fn depth(payload: Vec<u8>, width: usize, height: usize, time: i64, timescale: i32, scale: f32, offset: f32) -> Record {
        Record { width: width as u16, height: height as u16, time, timescale, scale, offset, payload }
    }

    /// Row-delta coded, raw DEFLATE, with its length prefix.
    pub fn deflated_framed(&self) -> Vec<u8> {
        let mut plain = self.payload.clone();
        let w = self.width as usize;
        for row in plain.chunks_mut(w.max(1)) {
            let mut previous = 0u8;
            for v in row.iter_mut() {
                let value = *v;
                *v = value.wrapping_sub(previous);
                previous = value;
            }
        }
        let mut encoder = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(&plain).expect("deflating into memory");
        let payload = encoder.finish().expect("deflating into memory");
        let mut out = Vec::with_capacity(40 + payload.len());
        out.extend_from_slice(&((36 + payload.len()) as u32).to_le_bytes());
        out.extend_from_slice(&[KIND_DEPTH, EYE_MONO, ENCODING_UINT8 | DEFLATED, 1]);
        out.extend_from_slice(&self.width.to_le_bytes());
        out.extend_from_slice(&self.height.to_le_bytes());
        out.extend_from_slice(&self.time.to_le_bytes());
        out.extend_from_slice(&self.time.to_le_bytes());
        out.extend_from_slice(&self.timescale.to_le_bytes());
        out.extend_from_slice(&self.scale.to_le_bytes());
        out.extend_from_slice(&self.offset.to_le_bytes());
        out.extend_from_slice(&payload);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_a_record_compare_analysis_tracks_reads() {
        let r = Record::depth(vec![10, 12, 9, 200, 201, 202], 3, 2, 1001, 24000, 0.5 / 255.0, 0.1);
        let bytes = r.deflated_framed();
        assert_eq!(u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize, bytes.len() - 4);
        assert_eq!(&bytes[4..8], &[3, 0, 0x83, 1]);
        let mut plain = Vec::new();
        std::io::Read::read_to_end(&mut flate2::read::DeflateDecoder::new(&bytes[40..]), &mut plain).unwrap();
        assert_eq!(plain, vec![10, 2, 253, 200, 1, 1]);
        assert_eq!(&header("m", "p", &BTreeMap::new(), 4, 2)[..5], b"VGAT\x01");
    }
}
