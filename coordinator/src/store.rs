//! Results on disk: one folder per title, named by a hash of its key (never
//! the title's name), holding its `meta.json` and each finished piece's
//! records (`pieces/<n>.rec`, the record frames only, no track header).
//! A chunk is served as a track header made from the title's profile, then
//! its pieces' records in order.

use std::io::Write;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::queue::{Title, TitleMeta};
use crate::track::{self, Header, ParsedTrack};

#[derive(Clone)]
pub struct Store {
    root: PathBuf,
}

pub fn key_hash(key: &str) -> String {
    let digest = Sha256::digest(key.as_bytes());
    digest[..12].iter().map(|b| format!("{b:02x}")).collect()
}

fn write_atomically(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    {
        let mut file = std::fs::File::create(&tmp)?;
        file.write_all(bytes)?;
        file.sync_data()?;
    }
    std::fs::rename(tmp, path)
}

impl Store {
    pub fn open(root: impl Into<PathBuf>) -> std::io::Result<Store> {
        let root = root.into();
        std::fs::create_dir_all(root.join("titles"))?;
        Ok(Store { root })
    }

    fn title_dir(&self, key: &str) -> PathBuf {
        self.root.join("titles").join(key_hash(key))
    }

    fn piece_path(&self, key: &str, piece: usize) -> PathBuf {
        self.title_dir(key).join("pieces").join(format!("{piece}.rec"))
    }

    /// Every stored title and its finished pieces.
    pub fn load(&self) -> std::io::Result<Vec<(TitleMeta, Vec<usize>)>> {
        let mut out = Vec::new();
        for entry in std::fs::read_dir(self.root.join("titles"))? {
            let dir = entry?.path();
            let Ok(bytes) = std::fs::read(dir.join("meta.json")) else { continue };
            let Ok(meta) = serde_json::from_slice::<TitleMeta>(&bytes) else {
                tracing::warn!(dir = %dir.display(), "unreadable title metadata, skipped");
                continue;
            };
            let mut done = Vec::new();
            if let Ok(pieces) = std::fs::read_dir(dir.join("pieces")) {
                for piece in pieces.flatten() {
                    let name = piece.file_name();
                    if let Some(n) = name.to_str().and_then(|n| n.strip_suffix(".rec")).and_then(|n| n.parse().ok()) {
                        done.push(n);
                    }
                }
            }
            done.sort_unstable();
            out.push((meta, done));
        }
        out.sort_by_key(|(meta, _)| meta.ordinal);
        Ok(out)
    }

    pub fn save_meta(&self, title: &Title) -> std::io::Result<()> {
        let dir = self.title_dir(&title.key);
        std::fs::create_dir_all(dir.join("pieces"))?;
        write_atomically(&dir.join("meta.json"), &serde_json::to_vec_pretty(&title.meta())?)
    }

    /// Drops every result of a title (its file was replaced, or it was removed).
    pub fn clear_pieces(&self, key: &str) -> std::io::Result<()> {
        let pieces = self.title_dir(key).join("pieces");
        match std::fs::remove_dir_all(&pieces) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(e),
            _ => {}
        }
        std::fs::create_dir_all(pieces)
    }

    pub fn remove_title(&self, key: &str) -> std::io::Result<()> {
        match std::fs::remove_dir_all(self.title_dir(key)) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e),
            _ => Ok(()),
        }
    }

    /// Keeps a checked result's records.
    pub fn save_piece(&self, key: &str, piece: usize, data: &[u8], track: &ParsedTrack) -> std::io::Result<usize> {
        let mut records = Vec::with_capacity(data.len());
        // In time order, so a chunk's records are too (the format doesn't need it; readers like it).
        let mut order: Vec<&track::RecordInfo> = track.records.iter().collect();
        order.sort_by(|a, b| {
            (a.time.value as i128 * b.time.timescale as i128)
                .cmp(&(b.time.value as i128 * a.time.timescale as i128))
                .then(a.kind.cmp(&b.kind))
        });
        for record in order {
            records.extend_from_slice(&data[record.frame.clone()]);
        }
        let path = self.piece_path(key, piece);
        std::fs::create_dir_all(path.parent().expect("a piece has a folder"))?;
        write_atomically(&path, &records)?;
        Ok(records.len())
    }

    /// The header every chunk of a title starts with.
    pub fn chunk_header(title: &Title) -> Header {
        let (w, h) = title.source_size.unwrap_or_else(|| title.info.stereo.eye_size(title.info.width, title.info.height));
        Header {
            media_id: key_hash(&title.key),
            producer: format!("vergenced {}", env!("CARGO_PKG_VERSION")),
            models: title.profile.clone().unwrap_or_default(),
            source_width: w,
            source_height: h,
            video_end_frame: None,
        }
    }

    /// A chunk: the title's header, then the given pieces' records.
    pub fn chunk(&self, title: &Title, pieces: &[usize]) -> std::io::Result<Vec<u8>> {
        let mut out = track::write_header(&Self::chunk_header(title));
        for &piece in pieces {
            out.extend(std::fs::read(self.piece_path(&title.key, piece))?);
        }
        Ok(out)
    }

    /// Bytes of results kept for a title.
    pub fn bytes(&self, key: &str) -> u64 {
        std::fs::read_dir(self.title_dir(key).join("pieces"))
            .map(|d| d.flatten().filter_map(|e| e.metadata().ok()).map(|m| m.len()).sum())
            .unwrap_or(0)
    }
}
