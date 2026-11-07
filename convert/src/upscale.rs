//! `--upscale 2x`: Real-ESRGAN's compact network (SRVGGNetCompact,
//! xinntao/Real-ESRGAN, BSD-3-Clause) through ONNX Runtime, as the Mac's
//! `FrameUpscaler` runs it through Core ML.
//!
//! Vergence ships no weights and no converted model: the authors' own `.pth`
//! is read (`checkpoint`), and the ONNX graph is written here, layer for
//! layer what `torch.onnx.export` would make of it: 3×3 convolutions and
//! per-channel PReLUs, a pixel shuffle by 4 (DepthToSpace, CRD), the input
//! enlarged by nearest neighbour and added, then averaged over 2×2 to 2×.
//! The graph takes any batch, height and width.
//!
//! A pair of eyes then has its fine detail brought into agreement, as the
//! Mac's `StereoDetailFusion` does on the GPU: disparity both ways by SAD
//! block matching on the source-size luma (7×7, ±4 % of the width, sub-pixel,
//! a texture floor, a uniqueness test and a left-right check), and each eye's
//! detail band (the picture less a σ 1.5 Gaussian of it) moved halfway
//! towards the other eye's where both see the point.

use std::path::{Path, PathBuf};

use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::TensorRef;
use sha2::{Digest, Sha256};

use crate::checkpoint::Checkpoint;
use crate::model::Provider;

struct Layer {
    weight: Vec<f32>,
    bias: Vec<f32>,
    outputs: usize,
    inputs: usize,
    slopes: Option<Vec<f32>>,
}

pub struct Network {
    layers: Vec<Layer>,
    /// The network's own enlargement (4).
    pub scale: usize,
    /// The file's name without its extension.
    pub name: String,
}

impl Network {
    pub fn load(path: &Path) -> Result<Network, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let file = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let checkpoint = Checkpoint::read(&bytes).map_err(|e| format!("{file}: {e}"))?;
        let tensors = checkpoint.tensors();
        let mut indices: Vec<usize> = tensors
            .keys()
            .filter_map(|k| {
                let parts: Vec<&str> = k.split('.').collect();
                (parts.len() == 3 && parts[0] == "body" && parts[2] == "weight").then(|| parts[1].parse().ok()).flatten()
            })
            .collect();
        indices.sort_unstable();
        if indices.is_empty() || indices != (0..indices.len()).collect::<Vec<_>>() {
            return Err(format!("{file} isn't a Real-ESRGAN compact network (SRVGGNetCompact): realesr-general-x4v3 or \
                                realesr-animevideov3 are; the RRDB networks (RealESRGAN_x4plus) are too slow for video"));
        }
        let mut layers: Vec<Layer> = Vec::new();
        for i in indices {
            let w = &tensors[&format!("body.{i}.weight")];
            if w.shape.len() == 4 {
                let b = tensors.get(&format!("body.{i}.bias")).filter(|b| b.values.len() == w.shape[0]);
                let inputs = layers.last().map(|l| l.outputs).unwrap_or(3);
                if w.shape[2] != 3 || w.shape[3] != 3 || b.is_none() || w.shape[1] != inputs {
                    return Err(format!("{file}: body.{i} isn't a 3×3 convolution taking {inputs} channels"));
                }
                layers.push(Layer { weight: w.values.clone(), bias: b.unwrap().values.clone(), outputs: w.shape[0], inputs, slopes: None });
            } else if w.shape.len() == 1
                && let Some(last) = layers.last_mut()
                && last.slopes.is_none()
                && (w.values.len() == last.outputs || w.values.len() == 1)
            {
                last.slopes = Some(if w.values.len() == 1 { vec![w.values[0]; last.outputs] } else { w.values.clone() });
            } else {
                return Err(format!("{file}: body.{i} is neither a convolution nor a PReLU"));
            }
        }
        let last = layers.last().ok_or("no layers")?;
        let scale = ((last.outputs / 3) as f64).sqrt().round() as usize;
        if layers.len() < 2 || last.slopes.is_some() || layers[..layers.len() - 1].iter().any(|l| l.slopes.is_none())
            || scale < 2 || scale * scale * 3 != last.outputs
        {
            return Err(format!("{file}: the layers aren't convolution-PReLU pairs ending in a convolution to 3·s² channels"));
        }
        Ok(Network { layers, scale, name: path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default() })
    }

    pub fn convolutions(&self) -> usize {
        self.layers.len()
    }

    /// The ONNX model: input "image" [N, 3, H, W] float32 RGB 0…1, output
    /// "upscaled" [N, 3, 2H, 2W].
    pub fn onnx(&self) -> Vec<u8> {
        let mut graph = Proto::default();
        let mut input = "image".to_string();
        let tensor = |g: &mut Proto, name: &str, dims: &[i64], values: &[f32]| {
            g.message(5, |t| {
                for d in dims {
                    t.uint(1, *d as u64);
                }
                t.uint(2, 1); // FLOAT
                t.string(8, name);
                t.bytes(9, &values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
            });
        };
        let node = |g: &mut Proto, op: &str, inputs: &[&str], output: &str, attributes: &dyn Fn(&mut Proto)| {
            g.message(1, |n| {
                for i in inputs {
                    n.string(1, i);
                }
                n.string(2, output);
                n.string(3, output);
                n.string(4, op);
                attributes(n);
            });
        };
        let ints = |n: &mut Proto, name: &str, values: &[i64]| {
            n.message(5, |a| {
                a.string(1, name);
                for v in values {
                    a.uint(8, *v as u64);
                }
                a.uint(20, 7); // INTS
            });
        };
        let int = |n: &mut Proto, name: &str, value: i64| {
            n.message(5, |a| {
                a.string(1, name);
                a.uint(3, value as u64);
                a.uint(20, 2); // INT
            });
        };
        let string = |n: &mut Proto, name: &str, value: &str| {
            n.message(5, |a| {
                a.string(1, name);
                a.string(4, value);
                a.uint(20, 3); // STRING
            });
        };
        for (i, layer) in self.layers.iter().enumerate() {
            let (w, b, conv) = (format!("w{i}"), format!("b{i}"), format!("conv{i}"));
            tensor(&mut graph, &w, &[layer.outputs as i64, layer.inputs as i64, 3, 3], &layer.weight);
            tensor(&mut graph, &b, &[layer.outputs as i64], &layer.bias);
            node(&mut graph, "Conv", &[&input, &w, &b], &conv, &|n| {
                ints(n, "kernel_shape", &[3, 3]);
                ints(n, "pads", &[1, 1, 1, 1]);
            });
            input = conv;
            if let Some(slopes) = &layer.slopes {
                let (s, act) = (format!("s{i}"), format!("prelu{i}"));
                tensor(&mut graph, &s, &[layer.outputs as i64, 1, 1], slopes);
                node(&mut graph, "PRelu", &[&input, &s], &act, &|_| {});
                input = act;
            }
        }
        let scale = self.scale as i64;
        node(&mut graph, "DepthToSpace", &[&input], "shuffle", &|n| {
            int(n, "blocksize", scale);
            string(n, "mode", "CRD");
        });
        tensor(&mut graph, "scales", &[4], &[1.0, 1.0, scale as f32, scale as f32]);
        node(&mut graph, "Resize", &["image", "", "scales"], "base", &|n| {
            string(n, "mode", "nearest");
            string(n, "coordinate_transformation_mode", "asymmetric");
            string(n, "nearest_mode", "floor");
        });
        node(&mut graph, "Add", &["shuffle", "base"], "sum", &|_| {});
        let pool = scale / 2;
        node(&mut graph, "AveragePool", &["sum"], "upscaled", &|n| {
            ints(n, "kernel_shape", &[pool, pool]);
            ints(n, "strides", &[pool, pool]);
        });
        graph.string(2, "srvgg");
        for (field, name) in [(11, "image"), (12, "upscaled")] {
            graph.message(field, |v| {
                v.string(1, name);
                v.message(2, |t| {
                    t.message(1, |tensor| {
                        tensor.uint(1, 1); // FLOAT
                        tensor.message(2, |shape| {
                            for (k, dim) in ["n", "3", "h", "w"].iter().enumerate() {
                                shape.message(1, |d| {
                                    if k == 1 {
                                        d.uint(1, 3);
                                    } else {
                                        d.string(2, &format!("{dim}{}", if field == 12 && k > 1 { "2" } else { "" }));
                                    }
                                });
                            }
                        });
                    });
                });
            });
        }
        let mut model = Proto::default();
        model.uint(1, 7); // IR version
        model.string(2, "vergence-convert");
        model.message(8, |o| {
            o.string(1, "");
            o.uint(2, 13);
        });
        model.message(7, |g| g.0.extend_from_slice(&graph.0));
        model.0
    }
}

/// Protocol buffers, written: what an ONNX model needs.
#[derive(Default)]
struct Proto(Vec<u8>);

impl Proto {
    fn varint(&mut self, mut v: u64) {
        while v >= 0x80 {
            self.0.push((v as u8 & 0x7F) | 0x80);
            v >>= 7;
        }
        self.0.push(v as u8);
    }
    fn uint(&mut self, field: u32, v: u64) {
        self.varint((field as u64) << 3);
        self.varint(v);
    }
    fn bytes(&mut self, field: u32, data: &[u8]) {
        self.varint((field as u64) << 3 | 2);
        self.varint(data.len() as u64);
        self.0.extend_from_slice(data);
    }
    fn string(&mut self, field: u32, s: &str) {
        self.bytes(field, s.as_bytes());
    }
    fn message(&mut self, field: u32, build: impl FnOnce(&mut Proto)) {
        let mut inner = Proto::default();
        build(&mut inner);
        self.bytes(field, &inner.0);
    }
}

/// The weights `--download-model` fetches, as the authors publish them.
pub struct Weights {
    pub name: &'static str,
    pub url: &'static str,
    pub sha256: &'static str,
    pub licence: &'static str,
}

pub const GENERAL: Weights = Weights {
    name: "realesr-general-x4v3",
    url: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
    sha256: "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292",
    licence: "BSD-3-Clause",
};

/// Beside the depth models: `…/Vergence/UpscaleModels`.
pub fn models_dir() -> PathBuf {
    crate::model::models_dir().with_file_name("UpscaleModels")
}

pub fn installed(weights: &Weights) -> Option<PathBuf> {
    let path = models_dir().join(format!("{}.pth", weights.name));
    path.is_file().then_some(path)
}

pub fn download(weights: &Weights) -> Result<PathBuf, String> {
    let dir = models_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let target = dir.join(format!("{}.pth", weights.name));
    let partial = dir.join(format!("{}.pth.partial", weights.name));
    let status = crate::probe::Tools::command(Path::new(if cfg!(windows) { "curl.exe" } else { "curl" }))
        .args(["-fsSL", "--retry", "3", "-o"])
        .arg(&partial)
        .arg(weights.url)
        .status()
        .map_err(|e| format!("the download needs curl: {e}"))?;
    if !status.success() {
        let _ = std::fs::remove_file(&partial);
        return Err(format!("the upscaling model's download failed ({status})"));
    }
    let bytes = std::fs::read(&partial).map_err(|e| e.to_string())?;
    let digest: String = Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect();
    if digest != weights.sha256 {
        let _ = std::fs::remove_file(&partial);
        return Err(format!("the downloaded {}.pth isn't the published one (checksum)", weights.name));
    }
    std::fs::rename(&partial, &target).map_err(|e| e.to_string())?;
    Ok(target)
}

/// The network in an ONNX Runtime session.
pub struct Upscaler {
    session: Session,
    pub provider: Provider,
    _env: ort::environment::Environment,
}

impl Upscaler {
    pub fn load(network: &Network, provider: Provider) -> Result<Upscaler, String> {
        let env = crate::model::environment("vergence-convert")?;
        let bytes = network.onnx();
        let cache = std::env::temp_dir().join("vergence-convert-model-cache");
        let mut errors = Vec::new();
        // Core ML wants fixed shapes and TensorRT builds an engine per size:
        // the graph's sizes are free, so the CUDA, DirectML or CPU provider runs it.
        let candidates: Vec<Provider> = match provider {
            Provider::Auto | Provider::TensorRt | Provider::CoreMl => vec![Provider::Cuda, Provider::DirectMl, Provider::Cpu],
            other => vec![other],
        };
        for candidate in candidates {
            let built = (|| -> Result<Session, String> {
                let fail = |e: ort::Error<_>| e.to_string();
                let mut builder = Session::builder(&env)
                    .map_err(|e| e.to_string())?
                    .with_optimization_level(GraphOptimizationLevel::Level3)
                    .map_err(fail)?;
                if candidate == Provider::DirectMl {
                    builder = builder.with_memory_pattern(false).map_err(fail)?.with_parallel_execution(false).map_err(fail)?;
                }
                if let Some(ep) = crate::model::dispatch(candidate, &cache)? {
                    builder = builder.with_execution_providers([ep]).map_err(fail)?;
                }
                builder.commit_from_memory(&bytes).map_err(|e| e.to_string())
            })();
            match built {
                Ok(session) => return Ok(Upscaler { session, provider: candidate, _env: env }),
                Err(e) => errors.push(format!("{}: {e}", candidate.name())),
            }
        }
        Err(format!("no execution provider could load the upscaling network ({})", errors.join("; ")))
    }

    /// `input`: `batch` planar RGB pictures of `width`×`height`; returns them at twice the size.
    pub fn run(&mut self, input: &[f32], batch: usize, width: usize, height: usize) -> Result<Vec<f32>, String> {
        let e = |e: ort::Error| format!("the upscaling network: {e}");
        let outputs = self
            .session
            .run(ort::inputs!["image" => TensorRef::from_array_view(([batch, 3, height, width], input)).map_err(e)?])
            .map_err(e)?;
        let (_, out) = outputs["upscaled"].try_extract_tensor::<f32>().map_err(e)?;
        if out.len() != batch * 3 * height * width * 4 {
            return Err(format!("the upscaling network gave {} values, not {}", out.len(), batch * 3 * height * width * 4));
        }
        Ok(out.to_vec())
    }
}

/// Where an eye is in the source frame, in its pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Region {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

/// One eye at `w`×`h` out of a planar frame (`planes` in R, G, B order, each
/// `fw`×`fh`), sampled bilinearly at pixel centres as the Metal kernel does:
/// a copy when the sizes match.
pub fn sample_eye(planes: [&[f32]; 3], fw: usize, fh: usize, region: Region, w: usize, h: usize, out: &mut Vec<f32>) {
    out.clear();
    out.reserve(3 * w * h);
    for plane in planes {
        for y in 0..h {
            let sy = (region.y + (y as f64 + 0.5) / h as f64 * region.height - 0.5).clamp(0.0, fh as f64 - 1.0);
            let (y0, ty) = (sy.floor() as usize, (sy - sy.floor()) as f32);
            let y1 = (y0 + 1).min(fh - 1);
            for x in 0..w {
                let sx = (region.x + (x as f64 + 0.5) / w as f64 * region.width - 0.5).clamp(0.0, fw as f64 - 1.0);
                let (x0, tx) = (sx.floor() as usize, (sx - sx.floor()) as f32);
                let x1 = (x0 + 1).min(fw - 1);
                let top = plane[y0 * fw + x0] * (1.0 - tx) + plane[y0 * fw + x1] * tx;
                let bottom = plane[y1 * fw + x0] * (1.0 - tx) + plane[y1 * fw + x1] * tx;
                out.push((top * (1.0 - ty) + bottom * ty).clamp(0.0, 1.0));
            }
        }
    }
}

/// Runs `f(first_row, rows)` over `height` rows split between the cores.
fn rows<F: Fn(usize, &mut [f32]) + Sync>(out: &mut [f32], width: usize, f: F) {
    let height = out.len() / width.max(1);
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).clamp(1, 16);
    let chunk = height.div_ceil(threads).max(1);
    std::thread::scope(|s| {
        for (i, part) in out.chunks_mut(chunk * width).enumerate() {
            let f = &f;
            s.spawn(move || f(i * chunk, part));
        }
    });
}

fn luma(rgb: &[f32], n: usize) -> Vec<f32> {
    (0..n).map(|i| 0.2126 * rgb[i] + 0.7152 * rgb[n + i] + 0.0722 * rgb[2 * n + i]).collect()
}

/// Disparity of `own` against `other` (luma, `w`×`h`): own(x) matches
/// other(x + d). Returns d and 1 where one match is clearly best, 0 and 0
/// where the window is flat or the match ambiguous.
fn block_match(own: &[f32], other: &[f32], w: usize, h: usize, radius: i32) -> Vec<(f32, f32)> {
    let n = w * h;
    let count = (2 * radius + 1) as usize;
    // One cost image per disparity: |own(x) − other(x + d)| summed over 7×7.
    let mut costs = vec![0f32; count * n];
    std::thread::scope(|s| {
        for (k, cost) in costs.chunks_mut(n).enumerate() {
            s.spawn(move || {
                let d = k as i32 - radius;
                let mut diff = vec![0f32; n];
                for y in 0..h {
                    for x in 0..w {
                        let ox = (x as i32 + d).clamp(0, w as i32 - 1) as usize;
                        diff[y * w + x] = (own[y * w + x] - other[y * w + ox]).abs();
                    }
                }
                box7(&diff, w, h, cost);
            });
        }
    });
    let mut sums = vec![0f32; n];
    let mut squares = vec![0f32; n];
    box7(own, w, h, &mut sums);
    box7(&own.iter().map(|v| v * v).collect::<Vec<_>>(), w, h, &mut squares);
    (0..n)
        .map(|i| {
            let mean = sums[i] / 49.0;
            if (squares[i] / 49.0 - mean * mean).max(0.0).sqrt() < TEXTURE_FLOOR {
                return (0.0, 0.0);
            }
            let (mut best, mut at) = (f32::INFINITY, 0usize);
            for k in 0..count {
                if costs[k * n + i] < best {
                    best = costs[k * n + i];
                    at = k;
                }
            }
            let second = (0..count).filter(|k| k.abs_diff(at) > 1).map(|k| costs[k * n + i]).fold(f32::INFINITY, f32::min);
            if best >= second * 0.95 || best.is_nan() {
                return (0.0, 0.0);
            }
            let mut offset = 0.0;
            if at > 0 && at + 1 < count {
                let (before, after) = (costs[(at - 1) * n + i], costs[(at + 1) * n + i]);
                let curve = before - 2.0 * best + after;
                if curve > 0.0 {
                    offset = (0.5 * (before - after) / curve).clamp(-0.5, 0.5);
                }
            }
            (at as f32 - radius as f32 + offset, 1.0)
        })
        .collect()
}

const TEXTURE_FLOOR: f32 = 0.008;

/// Sums over a 7×7 window, edges clamped, as the Metal kernel reads them.
fn box7(input: &[f32], w: usize, h: usize, out: &mut [f32]) {
    let mut across = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut s = 0.0;
            for i in -3i32..=3 {
                s += input[y * w + (x as i32 + i).clamp(0, w as i32 - 1) as usize];
            }
            across[y * w + x] = s;
        }
    }
    for y in 0..h {
        for x in 0..w {
            let mut s = 0.0;
            for j in -3i32..=3 {
                s += across[(y as i32 + j).clamp(0, h as i32 - 1) as usize * w + x];
            }
            out[y * w + x] = s;
        }
    }
}

const BLUR: [f32; 5] = [0.2666, 0.2134, 0.1096, 0.0361, 0.0076];

/// A Gaussian (σ 1.5) of one planar RGB picture.
fn blur(rgb: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut across = vec![0f32; rgb.len()];
    let mut out = vec![0f32; rgb.len()];
    rows(&mut across, w, |first, part| {
        for (r, row) in part.chunks_mut(w).enumerate() {
            let src = &rgb[(first + r) * w..(first + r + 1) * w];
            for (x, out) in row.iter_mut().enumerate() {
                let mut s = src[x] * BLUR[0];
                for (k, weight) in BLUR.iter().enumerate().skip(1) {
                    s += (src[x.saturating_sub(k)] + src[(x + k).min(w - 1)]) * weight;
                }
                *out = s;
            }
        }
    });
    let across = &across;
    rows(&mut out, w, |first, part| {
        for (r, row) in part.chunks_mut(w).enumerate() {
            let line = first + r;
            let (plane, y) = (line / h, line % h);
            let at = |yy: usize| &across[(plane * h + yy) * w..(plane * h + yy + 1) * w];
            let centre = at(y);
            for (x, out) in row.iter_mut().enumerate() {
                let mut s = centre[x] * BLUR[0];
                for (k, weight) in BLUR.iter().enumerate().skip(1) {
                    s += (at(y.saturating_sub(k))[x] + at((y + k).min(h - 1))[x]) * weight;
                }
                *out = s;
            }
        }
    });
    out
}

/// Both eyes' upscaled detail brought into agreement. `low`: the two eyes
/// as the network took them (planar RGB, `w`×`h` each); `high`: as it gave
/// them (2w×2h each), changed in place.
pub fn fuse_eyes(low: [&[f32]; 2], high: [&mut Vec<f32>; 2], w: usize, h: usize) {
    let n = w * h;
    let radius = ((w as f64 * 0.04).round() as i32).clamp(4, 48);
    let lumas = [luma(low[0], n), luma(low[1], n)];
    let flows = [block_match(&lumas[0], &lumas[1], w, h, radius), block_match(&lumas[1], &lumas[0], w, h, radius)];
    // Kept where the other eye's match points back.
    let confidence: Vec<Vec<(f32, f32)>> = (0..2)
        .map(|e| {
            (0..n)
                .map(|i| {
                    let (f, valid) = flows[e][i];
                    let (x, y) = (i % w, i / w);
                    let target = (x as f32 + f).round();
                    if valid < 0.5 || target < 0.0 || target >= w as f32 {
                        return (0.0, 0.0);
                    }
                    let (g, back) = flows[1 - e][y * w + target as usize];
                    (f, if back > 0.5 && (f + g).abs() <= 1.0 { 1.0 } else { 0.0 })
                })
                .collect()
        })
        .collect();
    let (hw, hh) = (2 * w, 2 * h);
    let pictures = [high[0].clone(), high[1].clone()];
    let blurred = [blur(&pictures[0], hw, hh), blur(&pictures[1], hw, hh)];
    for (e, out) in high.into_iter().enumerate() {
        let (mine, mine_blurred) = (&pictures[e], &blurred[e]);
        let (theirs, theirs_blurred) = (&pictures[1 - e], &blurred[1 - e]);
        let conf = &confidence[e];
        let sample = |plane: &[f32], c: usize, x: f32, y: usize| -> f32 {
            let x0 = x.floor() as usize;
            let x1 = (x0 + 1).min(hw - 1);
            let t = x - x0 as f32;
            let row = &plane[(c * hh + y) * hw..(c * hh + y + 1) * hw];
            row[x0] * (1.0 - t) + row[x1] * t
        };
        rows(out, hw, |first, part| {
            for (r, row) in part.chunks_mut(hw).enumerate() {
                let line = first + r;
                let (c, y) = (line / hh, line % hh);
                // The low-resolution confidence, bilinear, at this row.
                let ly = ((y as f32 + 0.5) / 2.0 - 0.5).clamp(0.0, h as f32 - 1.0);
                let (y0, ty) = (ly.floor() as usize, ly - ly.floor());
                let y1 = (y0 + 1).min(h - 1);
                for (x, value) in row.iter_mut().enumerate() {
                    let lx = ((x as f32 + 0.5) / 2.0 - 0.5).clamp(0.0, w as f32 - 1.0);
                    let (x0, tx) = (lx.floor() as usize, lx - lx.floor());
                    let x1 = (x0 + 1).min(w - 1);
                    let lerp = |a: (f32, f32), b: (f32, f32), t: f32| (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t);
                    let top = lerp(conf[y0 * w + x0], conf[y0 * w + x1], tx);
                    let bottom = lerp(conf[y1 * w + x0], conf[y1 * w + x1], tx);
                    let (f, c_weight) = lerp(top, bottom, ty);
                    let i = (c * hh + y) * hw + x;
                    let there = x as f32 + 2.0 * f;
                    if c_weight > 0.01 && there >= 0.0 && there <= hw as f32 - 1.0 {
                        let detail = mine[i] - mine_blurred[i];
                        let other = sample(theirs, c, there, y) - sample(theirs_blurred, c, there, y);
                        *value = (mine[i] + 0.5 * c_weight * (other - detail)).clamp(0.0, 1.0);
                    }
                }
            }
        });
    }
}

/// Planar RGB to the warp's 4:2:0 YCbCr code values (limited range), as the
/// decoder gives them: `bits` 8 or 10 (little-endian 16-bit samples).
pub fn rgb_to_yuv420(rgb: &[f32], w: usize, h: usize, (kr, kb): (f32, f32), bits: u32, out: &mut Vec<u8>) {
    let n = w * h;
    let kg = 1.0 - kr - kb;
    let scale = if bits > 8 { 4.0 } else { 1.0 };
    let (r, g, b) = (&rgb[..n], &rgb[n..2 * n], &rgb[2 * n..3 * n]);
    let luma = |i: usize| kr * r[i] + kg * g[i] + kb * b[i];
    let mut y_plane = Vec::with_capacity(n);
    for i in 0..n {
        y_plane.push(((16.0 + 219.0 * luma(i)) * scale).round().clamp(0.0, 255.0 * scale));
    }
    let (cw, ch) = (w / 2, h / 2);
    let mut u_plane = Vec::with_capacity(cw * ch);
    let mut v_plane = Vec::with_capacity(cw * ch);
    for cy in 0..ch {
        for cx in 0..cw {
            let mut ys = 0.0;
            let (mut bs, mut rs) = (0.0, 0.0);
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let i = (2 * cy + dy) * w + 2 * cx + dx;
                ys += luma(i);
                bs += b[i];
                rs += r[i];
            }
            let (yv, bv, rv) = (ys / 4.0, bs / 4.0, rs / 4.0);
            let cb = (bv - yv) / (2.0 * (1.0 - kb));
            let cr = (rv - yv) / (2.0 * (1.0 - kr));
            u_plane.push(((128.0 + 224.0 * cb) * scale).round().clamp(0.0, 255.0 * scale));
            v_plane.push(((128.0 + 224.0 * cr) * scale).round().clamp(0.0, 255.0 * scale));
        }
    }
    out.clear();
    for plane in [y_plane, u_plane, v_plane] {
        for v in plane {
            if bits > 8 {
                out.extend_from_slice(&(v as u16).to_le_bytes());
            } else {
                out.push(v as u8);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> PathBuf {
        PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../Tests/VergenceAppleTests/Fixtures")).join(name)
    }

    /// The tiny network PyTorch ran in `scripts`' making of the fixture: the
    /// written ONNX graph must give PyTorch's picture.
    #[test]
    fn matches_pytorch() {
        let network = Network::load(&fixture("tiny-srvgg.pth")).unwrap();
        assert_eq!((network.scale, network.convolutions()), (4, 3));
        let input = std::fs::read(fixture("tiny-srvgg-input.rgb")).unwrap();
        let expected: Vec<f32> =
            std::fs::read(fixture("tiny-srvgg-output.f32")).unwrap().chunks(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        let (w, h) = (24, 16);
        let mut planar = vec![0f32; 3 * w * h];
        for i in 0..w * h {
            for c in 0..3 {
                planar[c * w * h + i] = input[i * 3 + c] as f32 / 255.0;
            }
        }
        let mut upscaler = Upscaler::load(&network, Provider::Cpu).unwrap();
        let out = upscaler.run(&planar, 1, w, h).unwrap();
        let n = 4 * w * h;
        let worst = (0..n * 3).map(|i| (out[(i % 3) * n + i / 3].clamp(0.0, 1.0) - expected[i]).abs()).fold(0.0, f32::max);
        assert!(worst < 1e-4, "{worst}");
    }

    /// The fused pair agrees better than the pair it came from, where a
    /// texture shifted between the eyes carries different detail in each.
    #[test]
    fn fusion_brings_the_eyes_closer() {
        let (w, h) = (64, 48);
        let shift = 3usize;
        let texture = |x: usize, y: usize| ((x.wrapping_mul(73_856_093) ^ y.wrapping_mul(19_349_663)) % 1000) as f32 / 1000.0 * 0.6 + 0.2;
        let mut low = [vec![0f32; 3 * w * h], vec![0f32; 3 * w * h]];
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    low[0][c * w * h + y * w + x] = texture(x + shift, y);
                    low[1][c * w * h + y * w + x] = texture(x, y);
                }
            }
        }
        // Each eye's "upscaled" picture: the texture plus detail of its own.
        let (hw, hh) = (2 * w, 2 * h);
        let mut high = [vec![0f32; 3 * hw * hh], vec![0f32; 3 * hw * hh]];
        for (e, eye) in high.iter_mut().enumerate() {
            for c in 0..3 {
                for y in 0..hh {
                    for x in 0..hw {
                        let base = texture(x / 2 + if e == 0 { shift } else { 0 }, y / 2);
                        let noise = if (x * 31 + y * 17 + e * 7) % 5 == 0 { 0.08 } else { -0.02 };
                        eye[c * hw * hh + y * hw + x] = base + noise;
                    }
                }
            }
        }
        let disagreement = |high: &[Vec<f32>; 2]| -> f32 {
            let mut total = 0.0;
            for y in 8..hh - 8 {
                for x in 16..hw - 16 {
                    total += (high[0][y * hw + x] - high[1][y * hw + x + 2 * shift]).abs();
                }
            }
            total
        };
        let before = disagreement(&high);
        let [a, b] = &mut high;
        fuse_eyes([&low[0], &low[1]], [a, b], w, h);
        let after = disagreement(&high);
        assert!(after < before * 0.8, "{before} -> {after}");
    }

    #[test]
    fn converts_to_limited_range_yuv() {
        let (w, h) = (2, 2);
        let mut white = vec![1.0f32; 12];
        let mut out = Vec::new();
        rgb_to_yuv420(&white, w, h, (0.2126, 0.0722), 8, &mut out);
        assert_eq!(out, vec![235, 235, 235, 235, 128, 128]);
        white.iter_mut().for_each(|v| *v = 0.0);
        rgb_to_yuv420(&white, w, h, (0.2126, 0.0722), 10, &mut out);
        assert_eq!(&out[..2], &64u16.to_le_bytes());
    }
}
