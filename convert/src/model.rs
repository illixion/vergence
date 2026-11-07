//! Depth Anything V2 Small through ONNX Runtime, fed as Convolution feeds the
//! Core ML build (`CoreMLDepthProvider`): the frame fitted inside 518×392 and
//! centred on black, ImageNet-normalised; its map divided by its maximum,
//! as the Core ML build's last layer does, and stored as half floats.

use std::path::{Path, PathBuf};

use ort::ep::{self, ExecutionProviderDispatch};
use ort::environment::Environment;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::TensorRef;
use sha2::{Digest, Sha256};

/// The size a model takes when it doesn't say (an export with free height
/// and width, such as the onnx-community one): Apple's Core ML build's.
pub const WIDTH: usize = 518;
pub const HEIGHT: usize = 392;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// A model `--download-model` fetches: an onnx-community export (free input
/// size, `pixel_values` → `predicted_depth`), pinned to a revision and checksum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Download {
    /// What `--variant` calls it.
    pub variant: &'static str,
    /// The file it is kept as (the name the licence rule reads).
    pub name: &'static str,
    pub display: &'static str,
    pub repo: &'static str,
    pub url: &'static str,
    pub sha256: &'static str,
    pub megabytes: u32,
    pub licence: &'static str,
}

/// Depth Anything V2 Small (Apache-2.0): the same weights, inputs and outputs
/// as `scripts/export-depth-onnx.py` makes (checked equal).
pub const SMALL: Download = Download {
    variant: "small",
    name: "DepthAnythingV2Small",
    display: "Depth Anything V2 Small",
    repo: "huggingface.co/onnx-community/depth-anything-v2-small",
    url: "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/\
          4472b7362082ad9968fee890ca0f1e5aca36b93d/onnx/model.onnx",
    sha256: "afb6a5c28f3b6bf1618c6e43f02073ef9dfdc70e937502d51603e57b0a1df10c",
    megabytes: 100,
    licence: "Apache-2.0",
};

/// The models this build can fetch: Small, and with the `noncommercial-models`
/// feature the larger CC-BY-NC-4.0 ones (a default build names none of them).
pub fn downloads() -> Vec<Download> {
    #[allow(unused_mut)]
    let mut all = vec![SMALL];
    #[cfg(feature = "noncommercial-models")]
    all.extend([
        Download {
            variant: "base",
            name: "DepthAnythingV2Base",
            display: "Depth Anything V2 Base",
            repo: "huggingface.co/onnx-community/depth-anything-v2-base",
            url: "https://huggingface.co/onnx-community/depth-anything-v2-base/resolve/\
                  dd4557d492cd7b563738ac8d9ccff9094620983c/onnx/model.onnx",
            sha256: "dab40f67d5ead5e0ed52acfae41a717a708b02bc57f6cb761c5de828f26a7289",
            megabytes: 390,
            licence: "CC-BY-NC-4.0",
        },
        Download {
            variant: "large",
            name: "DepthAnythingV2Large",
            display: "Depth Anything V2 Large",
            repo: "huggingface.co/onnx-community/depth-anything-v2-large",
            url: "https://huggingface.co/onnx-community/depth-anything-v2-large/resolve/\
                  1fa1591c7b080e98da9655827c3a33a3972a4a83/onnx/model.onnx",
            sha256: "a93aa89b5e92e30e0afbe0f7c3ec692b35cfca791ae9004a190fb0ca2010e905",
            megabytes: 1340,
            licence: "CC-BY-NC-4.0",
        },
    ]);
    all
}

/// A download by its `--variant` name.
pub fn download_named(variant: &str) -> Result<Download, String> {
    let all = downloads();
    all.iter().copied().find(|d| d.variant.eq_ignore_ascii_case(variant)).ok_or_else(|| {
        let names: Vec<&str> = all.iter().map(|d| d.variant).collect();
        let more = if crate::licence::NONCOMMERCIAL_MODELS {
            ""
        } else {
            " (the larger, non-commercial models need a build with --features noncommercial-models)"
        };
        format!("unknown --variant {variant}: this build has {}{more}", names.join(", "))
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Provider {
    Auto,
    Cuda,
    TensorRt,
    DirectMl,
    CoreMl,
    Cpu,
}

impl Provider {
    /// VERGENCE_DEPTH_EP: auto (the default), cuda, tensorrt, directml, coreml or cpu.
    pub fn from_env() -> Result<Provider, String> {
        let name = std::env::var("VERGENCE_DEPTH_EP").unwrap_or_default().to_ascii_lowercase();
        Ok(match name.as_str() {
            "" | "auto" => Provider::Auto,
            "cuda" => Provider::Cuda,
            "tensorrt" | "trt" => Provider::TensorRt,
            "directml" | "dml" => Provider::DirectMl,
            "coreml" => Provider::CoreMl,
            "cpu" => Provider::Cpu,
            other => return Err(format!("VERGENCE_DEPTH_EP={other}: use auto, cuda, tensorrt, directml, coreml or cpu")),
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Provider::Auto => "auto",
            Provider::Cuda => "CUDA",
            Provider::TensorRt => "TensorRT",
            Provider::DirectMl => "DirectML",
            Provider::CoreMl => "CoreML",
            Provider::Cpu => "CPU",
        }
    }

    fn candidates(self) -> Vec<Provider> {
        match self {
            Provider::Auto => vec![Provider::Cuda, Provider::DirectMl, Provider::CoreMl, Provider::Cpu],
            other => vec![other],
        }
    }
}

fn dispatch(provider: Provider, cache: &Path) -> Result<Option<ExecutionProviderDispatch>, String> {
    let _ = cache;
    let ep = match provider {
        Provider::Auto | Provider::Cpu => return Ok(None),
        #[cfg(feature = "cuda")]
        Provider::Cuda => ep::CUDA::default().build(),
        #[cfg(feature = "tensorrt")]
        Provider::TensorRt => ep::TensorRT::default()
            .with_engine_cache(true)
            .with_engine_cache_path(cache.to_string_lossy().into_owned())
            .build(),
        #[cfg(any(feature = "directml", windows))]
        Provider::DirectMl => ep::DirectML::default().build(),
        #[cfg(any(feature = "coreml", target_os = "macos"))]
        Provider::CoreMl => {
            let mut coreml = ep::CoreML::default()
                .with_model_format(ep::coreml::ModelFormat::MLProgram)
                .with_static_input_shapes(true)
                .with_compute_units(ep::coreml::ComputeUnits::All);
            coreml = coreml.with_model_cache_dir(cache.to_string_lossy().into_owned());
            coreml.build()
        }
        #[allow(unreachable_patterns)]
        other => return Err(format!("this build has no {} (cargo build --features {})", other.name(), other.name().to_lowercase())),
    };
    Ok(Some(ep.error_on_failure()))
}

pub struct DepthModel {
    session: Session,
    pub provider: Provider,
    pub name: String,
    /// The input's size: the model's own when fixed, else the size asked for.
    pub width: usize,
    pub height: usize,
    _env: Environment,
}

/// ONNX Runtime's environment, one per process (the worker loads its models
/// again after standing aside).
pub fn environment(name: &str) -> Result<Environment, String> {
    static ENVIRONMENT: std::sync::OnceLock<Environment> = std::sync::OnceLock::new();
    if let Some(env) = ENVIRONMENT.get() {
        return Ok(env.clone());
    }
    let env = ort::init().with_name(name).build().map_err(|e| e.to_string())?;
    Ok(ENVIRONMENT.get_or_init(|| env).clone())
}

impl DepthModel {
    /// `size`: (width, height) for a model whose input size is free (multiples
    /// of 14); 518×392 when None. A fixed-size model keeps its own.
    pub fn load(path: &Path, provider: Provider, size: Option<(usize, usize)>) -> Result<DepthModel, String> {
        let env = environment("vergence-convert")?;
        // What TensorRT and Core ML compile, kept between runs (only they make it).
        let cache = std::env::temp_dir().join("vergence-convert-model-cache");
        let mut errors = Vec::new();
        for candidate in provider.candidates() {
            let built = (|| -> Result<Session, String> {
                let fail = |e: ort::Error<_>| format!("{}: {e}", path.display());
                let mut builder = Session::builder(&env)
                    .map_err(|e| e.to_string())?
                    .with_optimization_level(GraphOptimizationLevel::Level3)
                    .map_err(fail)?;
                if candidate == Provider::DirectMl {
                    builder = builder.with_memory_pattern(false).map_err(fail)?.with_parallel_execution(false).map_err(fail)?;
                }
                if matches!(candidate, Provider::TensorRt | Provider::CoreMl) {
                    let _ = std::fs::create_dir_all(&cache);
                }
                if let Some(ep) = dispatch(candidate, &cache)? {
                    builder = builder.with_execution_providers([ep]).map_err(fail)?;
                }
                builder.commit_from_file(path).map_err(|e| format!("{}: {e}", path.display()))
            })();
            match built {
                Ok(session) => {
                    let declared: Vec<i64> = session
                        .inputs()
                        .first()
                        .and_then(|i| i.dtype().tensor_shape().map(|s| s.iter().copied().collect()))
                        .unwrap_or_default();
                    let (width, height) = match declared[..] {
                        [_, _, h, w] if h > 0 && w > 0 => (w as usize, h as usize),
                        _ => size.unwrap_or((WIDTH, HEIGHT)),
                    };
                    let mut model = DepthModel {
                        session,
                        provider: candidate,
                        name: path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(),
                        width,
                        height,
                        _env: env,
                    };
                    // The first run specialises the graph for the device; pay it here.
                    model.infer(&vec![0.0; 3 * width * height])?;
                    return Ok(model);
                }
                Err(e) => errors.push(format!("{}: {e}", candidate.name())),
            }
        }
        Err(format!("no execution provider could load the depth model ({})", errors.join("; ")))
    }

    /// `input`: planar normalised RGB at the model's size. Returns the map
    /// divided by its maximum, at that size.
    pub fn infer(&mut self, input: &[f32]) -> Result<Vec<f32>, String> {
        let (w, h) = (self.width, self.height);
        let e = |e: ort::Error| format!("the depth model: {e}");
        let outputs = self
            .session
            .run(ort::inputs!["pixel_values" => TensorRef::from_array_view(([1usize, 3, h, w], input)).map_err(e)?])
            .map_err(e)?;
        let (_, depth) = outputs["predicted_depth"].try_extract_tensor::<f32>().map_err(e)?;
        if depth.len() != w * h {
            return Err(format!("the depth model gave {} values, not {}", depth.len(), w * h));
        }
        let max = depth.iter().copied().filter(|v| v.is_finite()).fold(f32::MIN, f32::max);
        let scale = if max > 0.0 { 1.0 / max } else { 1.0 };
        Ok(depth.iter().map(|v| v * scale).collect())
    }
}

/// Where the model `--download-model` fetches lives.
pub fn models_dir() -> PathBuf {
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).map(PathBuf::from).unwrap_or_default();
    if cfg!(windows) {
        std::env::var_os("LOCALAPPDATA").map(PathBuf::from).unwrap_or(home).join("Vergence").join("DepthModels")
    } else if cfg!(target_os = "macos") {
        home.join("Library/Application Support/Vergence/DepthModels")
    } else {
        std::env::var_os("XDG_DATA_HOME").map(PathBuf::from).unwrap_or_else(|| home.join(".local/share")).join("vergence/DepthModels")
    }
}

pub fn installed(model: &Download) -> Option<PathBuf> {
    let path = models_dir().join(format!("{}.onnx", model.name));
    path.is_file().then_some(path)
}

/// Fetches the model with curl (on Windows 10 and later, macOS and Linux
/// alike) and checks its digest before keeping it.
pub fn download(model: &Download) -> Result<PathBuf, String> {
    let dir = models_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let target = dir.join(format!("{}.onnx", model.name));
    let partial = dir.join(format!("{}.onnx.partial", model.name));
    let url: String = model.url.split_whitespace().collect();
    let status = crate::probe::Tools::command(Path::new(if cfg!(windows) { "curl.exe" } else { "curl" }))
        .args(["-fsSL", "--retry", "3", "-o"])
        .arg(&partial)
        .arg(&url)
        .status()
        .map_err(|e| format!("the download needs curl: {e}"))?;
    if !status.success() {
        let _ = std::fs::remove_file(&partial);
        return Err(format!("the model download failed ({status})"));
    }
    let bytes = std::fs::read(&partial).map_err(|e| e.to_string())?;
    let digest: String = Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect();
    if digest != model.sha256 {
        let _ = std::fs::remove_file(&partial);
        return Err("the downloaded model isn't the expected file (checksum)".into());
    }
    std::fs::rename(&partial, &target).map_err(|e| e.to_string())?;
    Ok(target)
}

// MARK: - The model's input

/// The fitted size inside the model's (`mw`×`mh`), even, as `CoreMLDepthProvider.preparedInput` makes it.
pub fn fitted(width: usize, height: usize, (mw, mh): (usize, usize)) -> (usize, usize) {
    let scale = (mw as f64 / width as f64).min(mh as f64 / height as f64).min(1.0);
    let even = |v: usize| ((v as f64 * scale / 2.0).round() as usize * 2).max(2);
    (even(width).min(mw), even(height).min(mh))
}

/// Resamples one channel-interleaved image with a tent filter as wide as the
/// reduction (an area-like downscale; bilinear when enlarging).
pub fn resize(src: &[f32], w: usize, h: usize, channels: usize, tw: usize, th: usize) -> Vec<f32> {
    fn weights(from: usize, to: usize) -> Vec<(usize, Vec<f32>)> {
        let scale = from as f64 / to as f64;
        let support = scale.max(1.0);
        (0..to)
            .map(|i| {
                let center = (i as f64 + 0.5) * scale - 0.5;
                let first = (center - support).floor().max(0.0) as usize;
                let last = ((center + support).ceil() as usize).min(from - 1);
                let mut w: Vec<f32> =
                    (first..=last).map(|j| (1.0 - ((j as f64 - center).abs() / support)).max(0.0) as f32).collect();
                let sum: f32 = w.iter().sum();
                if sum > 0.0 {
                    w.iter_mut().for_each(|v| *v /= sum);
                }
                (first, w)
            })
            .collect()
    }
    let (wx, wy) = (weights(w, tw), weights(h, th));
    // Rows in bands on a few threads: at decode size this is as long as the inference.
    let bands = 4;
    let mut across = vec![0f32; tw * h * channels];
    let rows = h.div_ceil(bands).max(1);
    std::thread::scope(|scope| {
        for (band, chunk) in across.chunks_mut(rows * tw * channels).enumerate() {
            let wx = &wx;
            scope.spawn(move || {
                for (r, row) in chunk.chunks_mut(tw * channels).enumerate() {
                    let y = band * rows + r;
                    for (x, (first, ws)) in wx.iter().enumerate() {
                        for c in 0..channels {
                            let mut v = 0.0;
                            for (k, wt) in ws.iter().enumerate() {
                                v += wt * src[(y * w + first + k) * channels + c];
                            }
                            row[x * channels + c] = v;
                        }
                    }
                }
            });
        }
    });
    let mut out = vec![0f32; tw * th * channels];
    let rows = th.div_ceil(bands).max(1);
    std::thread::scope(|scope| {
        for (band, chunk) in out.chunks_mut(rows * tw * channels).enumerate() {
            let (wy, across) = (&wy, &across);
            scope.spawn(move || {
                for (r, row) in chunk.chunks_mut(tw * channels).enumerate() {
                    let (first, ws) = &wy[band * rows + r];
                    for x in 0..tw {
                        for c in 0..channels {
                            let mut v = 0.0;
                            for (k, wt) in ws.iter().enumerate() {
                                v += wt * across[((first + k) * tw + x) * channels + c];
                            }
                            row[x * channels + c] = v;
                        }
                    }
                }
            });
        }
    });
    out
}

/// The model's input from an RGB frame (0…1, interleaved): fitted, centred on
/// black, planar, ImageNet-normalised. `hdr`: the frame holds PQ (false) or
/// HLG (true) code values, tone-mapped to BT.709 once at the model's size.
pub fn input(rgb: &[f32], w: usize, h: usize, hdr: Option<bool>, (mw, mh): (usize, usize)) -> Vec<f32> {
    let (tw, th) = fitted(w, h, (mw, mh));
    let mut small = resize(rgb, w, h, 3, tw, th);
    if let Some(hlg) = hdr {
        tone_map(&mut small, hlg);
    }
    let (ox, oy) = ((mw - tw) / 2, (mh - th) / 2);
    let mut out = vec![0f32; 3 * mw * mh];
    for c in 0..3 {
        let black = (0.0 - MEAN[c]) / STD[c];
        let plane = &mut out[c * mw * mh..(c + 1) * mw * mh];
        plane.fill(black);
        for y in 0..th {
            for x in 0..tw {
                plane[(y + oy) * mw + x + ox] = (small[(y * tw + x) * 3 + c].clamp(0.0, 1.0) - MEAN[c]) / STD[c];
            }
        }
    }
    out
}

/// An HDR frame's code values (PQ or HLG, BT.2020) as the BT.709 SDR picture
/// the model sees: Convolution hands the model a BT.709 copy too. A plain
/// global curve (diffuse white at 203 nits, Reinhard toward a 1000-nit peak).
pub fn tone_map(rgb: &mut [f32], hlg: bool) {
    const M: [[f32; 3]; 3] = [[1.6605, -0.5876, -0.0728], [-0.1246, 1.1329, -0.0083], [-0.0182, -0.1006, 1.1187]];
    let (eotf, oetf) = tone_tables(hlg);
    let white = 1000.0f32 / 203.0;
    for px in rgb.as_chunks_mut::<3>().0 {
        let lin = [0, 1, 2].map(|c| lookup(eotf, px[c]));
        for (c, row) in M.iter().enumerate() {
            let x = (row[0] * lin[0] + row[1] * lin[1] + row[2] * lin[2]).max(0.0);
            let y = (x * (1.0 + x / (white * white)) / (1.0 + x)).min(1.0);
            px[c] = lookup(oetf, y);
        }
    }
}

const TABLE: usize = 4096;

/// `table` over 0…1, interpolated.
fn lookup(table: &[f32], v: f32) -> f32 {
    let p = v.clamp(0.0, 1.0) * (TABLE - 1) as f32;
    let i = (p as usize).min(TABLE - 2);
    let f = p - i as f32;
    table[i] + (table[i + 1] - table[i]) * f
}

/// Code value → linear light (diffuse white 1), and linear 0…1 → BT.709 code value.
fn tone_tables(hlg: bool) -> (&'static [f32], &'static [f32]) {
    use std::sync::OnceLock;
    static PQ: OnceLock<Vec<f32>> = OnceLock::new();
    static HLG: OnceLock<Vec<f32>> = OnceLock::new();
    static OETF: OnceLock<Vec<f32>> = OnceLock::new();
    let at = |i: usize| i as f32 / (TABLE - 1) as f32;
    let eotf = if hlg {
        HLG.get_or_init(|| {
            let (a, b, c) = (0.178_832_77_f32, 0.284_668_92_f32, 0.559_910_7_f32);
            (0..TABLE)
                .map(|i| {
                    let e = at(i);
                    let scene = if e <= 0.5 { e * e / 3.0 } else { (((e - c) / a).exp() + b) / 12.0 };
                    scene * 1000.0 / 203.0
                })
                .collect()
        })
    } else {
        PQ.get_or_init(|| {
            let (m1, m2, c1, c2, c3) = (0.159_301_76_f32, 78.84375f32, 0.8359375f32, 18.851_563_f32, 18.6875f32);
            (0..TABLE)
                .map(|i| {
                    let p = at(i).powf(1.0 / m2);
                    10000.0 * ((p - c1).max(0.0) / (c2 - c3 * p)).powf(1.0 / m1) / 203.0
                })
                .collect()
        })
    };
    let oetf = OETF.get_or_init(|| {
        (0..TABLE).map(at).map(|y| if y < 0.018 { 4.5 * y } else { 1.099 * y.powf(0.45) - 0.099 }).collect()
    });
    (eotf, oetf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fits_like_the_provider() {
        // A 1080p frame decoded at 1036×582 goes in at 518×292, letterboxed by 50 rows.
        let size = (WIDTH, HEIGHT);
        assert_eq!(fitted(1036, 582, size), (518, 292));
        assert_eq!(fitted(400, 400, size), (392, 392));
        assert_eq!(fitted(300, 200, size), (300, 200));
        assert_eq!(fitted(1036, 582, (770, 574)), (770, 432));
    }

    #[test]
    fn downscales_keeping_the_mean() {
        let src: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let out = resize(&src, 4, 4, 1, 2, 2);
        // A tent as wide as the reduction: smooth, and the mean is kept.
        assert!(out[0] < out[1] && out[1] < out[2] && out[2] < out[3], "{out:?}");
        assert!((out.iter().sum::<f32>() / 4.0 - 7.5).abs() < 1e-4, "{out:?}");
        let flat = resize(&[0.5; 3 * 64], 8, 8, 3, 3, 5);
        assert!(flat.iter().all(|v| (v - 0.5).abs() < 1e-6));
    }

    #[test]
    fn letterboxes_on_black() {
        let rgb = vec![1.0f32; 3 * 1036 * 582];
        let x = input(&rgb, 1036, 582, None, (WIDTH, HEIGHT));
        let white = (1.0 - MEAN[0]) / STD[0];
        let black = (0.0 - MEAN[0]) / STD[0];
        assert!((x[200 * WIDTH + 10] - white).abs() < 1e-5);
        assert!((x[10 * WIDTH + 10] - black).abs() < 1e-5);
    }

    #[test]
    fn tone_maps_diffuse_white_near_white() {
        // PQ 203 nits ≈ 0.58 code value; grey stays grey.
        let mut px = vec![0.5807f32; 3];
        tone_map(&mut px, false);
        assert!(px.iter().all(|v| (*v - px[0]).abs() < 1e-3) && px[0] > 0.6 && px[0] < 0.95, "{px:?}");
    }
}
