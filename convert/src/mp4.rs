//! Google's Spherical Video V2 `st3d` box in the video's sample entry, which
//! tells players outside Apple's that the file is side by side or top and
//! bottom: VergenceCore's `SphericalMetadata.write`, stereo half. `moov` is
//! rewritten with the box added; when it sits before the media, every chunk
//! offset past it moves.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

/// `st3d` stereo_mode: 1 top and bottom, 2 left and right.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StereoMode {
    TopBottom = 1,
    LeftRight = 2,
}

fn be32(d: &[u8], i: usize) -> u32 {
    u32::from_be_bytes(d[i..i + 4].try_into().unwrap())
}

fn be64(d: &[u8], i: usize) -> u64 {
    u64::from_be_bytes(d[i..i + 8].try_into().unwrap())
}

struct Node {
    kind: [u8; 4],
    /// Bytes after the header and before the children (all of them for a leaf).
    payload: Vec<u8>,
    children: Option<Vec<Node>>,
    large: bool,
}

impl Node {
    fn child_offset(kind: &[u8; 4], parent: Option<&[u8; 4]>) -> Option<usize> {
        match kind {
            b"moov" | b"trak" | b"mdia" | b"minf" | b"stbl" | b"edts" | b"dinf" => Some(0),
            b"stsd" => Some(8),
            b"hvc1" | b"hev1" | b"avc1" | b"avc3" | b"mp4v" | b"dvh1" | b"dvhe" if parent == Some(b"stsd") => Some(78),
            _ => None,
        }
    }

    fn parse(data: &[u8], parent: Option<&[u8; 4]>) -> Result<(Node, usize), String> {
        if data.len() < 8 {
            return Err("a box is cut short".into());
        }
        let mut size = be32(data, 0) as usize;
        let kind: [u8; 4] = data[4..8].try_into().unwrap();
        let mut header = 8;
        if size == 1 {
            size = be64(data, 8) as usize;
            header = 16;
        }
        if size < header || size > data.len() {
            return Err(format!("bad size for {}", String::from_utf8_lossy(&kind)));
        }
        let body = &data[header..size];
        let fields = if header == 8 { Self::child_offset(&kind, parent).filter(|&f| f <= body.len()) } else { None };
        let Some(fields) = fields else {
            return Ok((Node { kind, payload: body.to_vec(), children: None, large: header == 16 }, size));
        };
        let mut children = Vec::new();
        let mut offset = fields;
        while offset + 8 <= body.len() {
            let (child, used) = Self::parse(&body[offset..], Some(&kind))?;
            children.push(child);
            offset += used;
        }
        Ok((Node { kind, payload: body[..fields].to_vec(), children: Some(children), large: false }, size))
    }

    fn serialized(&self) -> Vec<u8> {
        let mut body = self.payload.clone();
        for child in self.children.iter().flatten() {
            body.extend(child.serialized());
        }
        let mut out = Vec::with_capacity(body.len() + 16);
        if self.large || body.len() + 8 > u32::MAX as usize {
            out.extend(1u32.to_be_bytes());
            out.extend(self.kind);
            out.extend(((body.len() + 16) as u64).to_be_bytes());
        } else {
            out.extend(((body.len() + 8) as u32).to_be_bytes());
            out.extend(self.kind);
        }
        out.extend(body);
        out
    }

    fn first(&self, kind: &[u8; 4]) -> Option<&Node> {
        if &self.kind == kind {
            return Some(self);
        }
        self.children.iter().flatten().find_map(|c| c.first(kind))
    }

    fn sample_entries(&mut self, body: &mut dyn FnMut(&mut Node)) {
        let Some(children) = self.children.as_mut() else { return };
        if &self.kind == b"stsd" {
            children.iter_mut().filter(|c| c.children.is_some()).for_each(&mut *body);
            return;
        }
        children.iter_mut().for_each(|c| c.sample_entries(body));
    }

    fn add_st3d(&mut self, mode: StereoMode) {
        for track in self.children.iter_mut().flatten().filter(|c| &c.kind == b"trak") {
            let is_video = track.first(b"hdlr").is_some_and(|h| h.payload.len() >= 12 && &h.payload[8..12] == b"vide");
            if !is_video {
                continue;
            }
            track.sample_entries(&mut |entry| {
                let children = entry.children.as_mut().expect("a sample entry with children");
                children.retain(|c| &c.kind != b"st3d");
                children.push(Node { kind: *b"st3d", payload: vec![0, 0, 0, 0, mode as u8], children: None, large: false });
            });
        }
    }

    fn shift_chunk_offsets(&mut self, after: u64, delta: i64) {
        if &self.kind == b"stco" || &self.kind == b"co64" {
            let wide = &self.kind == b"co64";
            let count = be32(&self.payload, 4) as usize;
            for i in 0..count {
                let at = 8 + i * if wide { 8 } else { 4 };
                let value = if wide { be64(&self.payload, at) } else { be32(&self.payload, at) as u64 };
                if value < after {
                    continue;
                }
                let moved = (value as i64 + delta) as u64;
                if wide {
                    self.payload[at..at + 8].copy_from_slice(&moved.to_be_bytes());
                } else {
                    self.payload[at..at + 4].copy_from_slice(&(moved as u32).to_be_bytes());
                }
            }
            return;
        }
        self.children.iter_mut().flatten().for_each(|c| c.shift_chunk_offsets(after, delta));
    }
}

/// Top-level boxes: (type, offset, size).
fn top_level(file: &mut File) -> Result<Vec<([u8; 4], u64, u64)>, String> {
    let end = file.seek(SeekFrom::End(0)).map_err(|e| e.to_string())?;
    let mut offset = 0u64;
    let mut boxes = Vec::new();
    while offset + 8 <= end {
        file.seek(SeekFrom::Start(offset)).map_err(|e| e.to_string())?;
        let mut header = [0u8; 16];
        let n = file.read(&mut header).map_err(|e| e.to_string())?;
        let mut size = be32(&header, 0) as u64;
        let kind: [u8; 4] = header[4..8].try_into().unwrap();
        if size == 1 {
            if n < 16 {
                return Err("cut short".into());
            }
            size = be64(&header, 8);
        } else if size == 0 {
            size = end - offset;
        }
        if size < 8 || offset + size > end {
            return Err(format!("bad size for {} at {offset}", String::from_utf8_lossy(&kind)));
        }
        boxes.push((kind, offset, size));
        offset += size;
    }
    Ok(boxes)
}

pub fn write_st3d(path: &Path, mode: StereoMode) -> Result<(), String> {
    let fail = |e: std::io::Error| format!("{}: {e}", path.display());
    let mut file = OpenOptions::new().read(true).write(true).open(path).map_err(fail)?;
    let top = top_level(&mut file)?;
    let index = top.iter().position(|b| &b.0 == b"moov").ok_or("not an MP4: no moov box")?;
    let (_, offset, size) = top[index];
    file.seek(SeekFrom::Start(offset)).map_err(fail)?;
    let mut data = vec![0u8; size as usize];
    file.read_exact(&mut data).map_err(fail)?;
    let (mut moov, _) = Node::parse(&data, None)?;
    moov.add_st3d(mode);
    let mut new = moov.serialized();
    let delta = new.len() as i64 - size as i64;
    if index == top.len() - 1 || delta == 0 {
        file.seek(SeekFrom::Start(offset)).map_err(fail)?;
        file.write_all(&new).map_err(fail)?;
        file.set_len(offset + new.len() as u64).map_err(fail)?;
        return Ok(());
    }
    moov.shift_chunk_offsets(offset, delta);
    new = moov.serialized();
    let temporary = path.with_extension("st3d.partial");
    let mut out = File::create(&temporary).map_err(fail)?;
    copy(&mut file, 0, offset, &mut out)?;
    out.write_all(&new).map_err(fail)?;
    let end = file.seek(SeekFrom::End(0)).map_err(fail)?;
    copy(&mut file, offset + size, end - offset - size, &mut out)?;
    drop(out);
    drop(file);
    std::fs::rename(&temporary, path).map_err(fail)
}

fn copy(file: &mut File, from: u64, count: u64, out: &mut File) -> Result<(), String> {
    file.seek(SeekFrom::Start(from)).map_err(|e| e.to_string())?;
    std::io::copy(&mut Read::by_ref(file).take(count), out).map_err(|e| e.to_string())?;
    Ok(())
}

/// The stereo mode a file's video sample entry declares, if any.
#[cfg(test)]
pub fn read_st3d(path: &Path) -> Option<u8> {
    let mut file = File::open(path).ok()?;
    let top = top_level(&mut file).ok()?;
    let (_, offset, size) = *top.iter().find(|b| &b.0 == b"moov")?;
    file.seek(SeekFrom::Start(offset)).ok()?;
    let mut data = vec![0u8; size as usize];
    file.read_exact(&mut data).ok()?;
    let (moov, _) = Node::parse(&data, None).ok()?;
    moov.first(b"st3d").map(|n| n.payload[4])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leaf(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut out = ((payload.len() + 8) as u32).to_be_bytes().to_vec();
        out.extend(kind);
        out.extend(payload);
        out
    }

    /// A minimal file: ftyp, moov (one video trak, an stco pointing into mdat), mdat.
    fn file(moov_first: bool) -> (Vec<u8>, usize) {
        let mut hdlr = vec![0u8; 8];
        hdlr.extend(b"vide");
        hdlr.extend([0u8; 12]);
        let entry = leaf(b"hvc1", &[0u8; 78]);
        let mut stsd_payload = vec![0, 0, 0, 0, 0, 0, 0, 1];
        stsd_payload.extend(&entry);
        let stsd = leaf(b"stsd", &stsd_payload);
        let stco_payload = |offset: u32| {
            let mut p = vec![0, 0, 0, 0, 0, 0, 0, 1];
            p.extend(offset.to_be_bytes());
            p
        };
        let build = |offset: u32| {
            let stbl = leaf(b"stbl", &[stsd.clone(), leaf(b"stco", &stco_payload(offset))].concat());
            let minf = leaf(b"minf", &stbl);
            let mdia = leaf(b"mdia", &[leaf(b"hdlr", &hdlr), minf].concat());
            leaf(b"moov", &leaf(b"trak", &mdia))
        };
        let ftyp = leaf(b"ftyp", b"isom");
        let mdat = leaf(b"mdat", b"MEDIA");
        let moov_len = build(0).len();
        if moov_first {
            let media = ftyp.len() + moov_len + 8;
            ([ftyp, build(media as u32), mdat].concat(), media)
        } else {
            let media = ftyp.len() + 8;
            ([ftyp, mdat, build(media as u32)].concat(), media)
        }
    }

    #[test]
    fn adds_st3d_and_moves_chunk_offsets() {
        for moov_first in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("x.mp4");
            let (bytes, _) = file(moov_first);
            std::fs::write(&path, &bytes).unwrap();
            write_st3d(&path, StereoMode::LeftRight).unwrap();
            assert_eq!(read_st3d(&path), Some(2));
            // The chunk offset still points at the media.
            let data = std::fs::read(&path).unwrap();
            let stco = data.windows(4).position(|w| w == b"stco").unwrap();
            let offset = be32(&data, stco + 12) as usize;
            assert_eq!(&data[offset..offset + 5], b"MEDIA");
            write_st3d(&path, StereoMode::TopBottom).unwrap();
            assert_eq!(read_st3d(&path), Some(1));
            assert_eq!(std::fs::read(&path).unwrap().windows(4).filter(|w| *w == b"st3d").count(), 1);
        }
    }
}
