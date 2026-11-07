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

pub const WIDTH: usize = 518;
pub const HEIGHT: usize = 392;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// The Apache-2.0 ONNX export `--download-model` fetches: the same weights,
/// inputs and outputs as `scripts/export-depth-onnx.py` makes (checked equal),
/// pinned to a revision and checksum.
pub const DOWNLOAD_NAME: &str = "DepthAnythingV2Small";
const DOWNLOAD_URL: &str = "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/\
                            4472b7362082ad9968fee890ca0f1e5aca36b93d/onnx/model.onnx";
const DOWNLOAD_SHA256: &str = "afb6a5c28f3b6bf1618c6e43f02073ef9dfdc70e937502d51603e57b0a1df10c";

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
    _env: Environment,
}

impl DepthModel {
    pub fn load(path: &Path, provider: Provider) -> Result<DepthModel, String> {
        let env = ort::init().with_name("vergence-convert").build().map_err(|e| e.to_string())?;
        let cache = std::env::temp_dir().join("vergence-convert-model-cache");
        let _ = std::fs::create_dir_all(&cache);
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
                if let Some(ep) = dispatch(candidate, &cache)? {
                    builder = builder.with_execution_providers([ep]).map_err(fail)?;
                }
                builder.commit_from_file(path).map_err(|e| format!("{}: {e}", path.display()))
            })();
            match built {
                Ok(session) => {
                    let mut model = DepthModel {
                        session,
                        provider: candidate,
                        name: path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(),
                        _env: env,
                    };
                    // The first run specialises the graph for the device; pay it here.
                    model.infer(&vec![0.0; 3 * WIDTH * HEIGHT])?;
                    return Ok(model);
                }
                Err(e) => errors.push(format!("{}: {e}", candidate.name())),
            }
        }
        Err(format!("no execution provider could load the depth model ({})", errors.join("; ")))
    }

    /// `input`: planar normalised RGB at 518×392. Returns the map divided by
    /// its maximum, at 518×392.
    pub fn infer(&mut self, input: &[f32]) -> Result<Vec<f32>, String> {
        let e = |e: ort::Error| format!("the depth model: {e}");
        let outputs = self
            .session
            .run(ort::inputs!["pixel_values" => TensorRef::from_array_view(([1usize, 3, HEIGHT, WIDTH], input)).map_err(e)?])
            .map_err(e)?;
        let (_, depth) = outputs["predicted_depth"].try_extract_tensor::<f32>().map_err(e)?;
        if depth.len() != WIDTH * HEIGHT {
            return Err(format!("the depth model gave {} values, not {}", depth.len(), WIDTH * HEIGHT));
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

pub fn installed() -> Option<PathBuf> {
    let path = models_dir().join(format!("{DOWNLOAD_NAME}.onnx"));
    path.is_file().then_some(path)
}

/// Fetches the model with curl (on Windows 10 and later, macOS and Linux
/// alike) and checks its digest before keeping it.
pub fn download() -> Result<PathBuf, String> {
    let dir = models_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let target = dir.join(format!("{DOWNLOAD_NAME}.onnx"));
    let partial = dir.join(format!("{DOWNLOAD_NAME}.onnx.partial"));
    let status = crate::probe::Tools::command(Path::new(if cfg!(windows) { "curl.exe" } else { "curl" }))
        .args(["-fsSL", "--retry", "3", "-o"])
        .arg(&partial)
        .arg(DOWNLOAD_URL)
        .status()
        .map_err(|e| format!("the download needs curl: {e}"))?;
    if !status.success() {
        let _ = std::fs::remove_file(&partial);
        return Err(format!("the model download failed ({status})"));
    }
    let bytes = std::fs::read(&partial).map_err(|e| e.to_string())?;
    let digest: String = Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect();
    if digest != DOWNLOAD_SHA256 {
        let _ = std::fs::remove_file(&partial);
        return Err("the downloaded model isn't the expected file (checksum)".into());
    }
    std::fs::rename(&partial, &target).map_err(|e| e.to_string())?;
    Ok(target)
}

// MARK: - The model's input

/// The fitted size inside 518×392, even, as `CoreMLDepthProvider.preparedInput` makes it.
pub fn fitted(width: usize, height: usize) -> (usize, usize) {
    let scale = (WIDTH as f64 / width as f64).min(HEIGHT as f64 / height as f64).min(1.0);
    let even = |v: usize| ((v as f64 * scale / 2.0).round() as usize * 2).max(2);
    (even(width).min(WIDTH), even(height).min(HEIGHT))
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
    let mut across = vec![0f32; tw * h * channels];
    for y in 0..h {
        for (x, (first, ws)) in wx.iter().enumerate() {
            for c in 0..channels {
                let mut v = 0.0;
                for (k, wt) in ws.iter().enumerate() {
                    v += wt * src[(y * w + first + k) * channels + c];
                }
                across[(y * tw + x) * channels + c] = v;
            }
        }
    }
    let mut out = vec![0f32; tw * th * channels];
    for (y, (first, ws)) in wy.iter().enumerate() {
        for x in 0..tw {
            for c in 0..channels {
                let mut v = 0.0;
                for (k, wt) in ws.iter().enumerate() {
                    v += wt * across[((first + k) * tw + x) * channels + c];
                }
                out[(y * tw + x) * channels + c] = v;
            }
        }
    }
    out
}

/// The model's input from an RGB frame (0…1, interleaved): fitted, centred on
/// black, planar, ImageNet-normalised.
pub fn input(rgb: &[f32], w: usize, h: usize) -> Vec<f32> {
    let (tw, th) = fitted(w, h);
    let small = resize(rgb, w, h, 3, tw, th);
    let (ox, oy) = ((WIDTH - tw) / 2, (HEIGHT - th) / 2);
    let mut out = vec![0f32; 3 * WIDTH * HEIGHT];
    for c in 0..3 {
        let black = (0.0 - MEAN[c]) / STD[c];
        let plane = &mut out[c * WIDTH * HEIGHT..(c + 1) * WIDTH * HEIGHT];
        plane.fill(black);
        for y in 0..th {
            for x in 0..tw {
                plane[(y + oy) * WIDTH + x + ox] = (small[(y * tw + x) * 3 + c].clamp(0.0, 1.0) - MEAN[c]) / STD[c];
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
    let pq = |e: f32| -> f32 {
        let (m1, m2, c1, c2, c3) = (0.159_301_76_f32, 78.84375f32, 0.8359375f32, 18.851_563_f32, 18.6875f32);
        let p = e.clamp(0.0, 1.0).powf(1.0 / m2);
        10000.0 * ((p - c1).max(0.0) / (c2 - c3 * p)).powf(1.0 / m1)
    };
    let hlg_inverse = |e: f32| -> f32 {
        let (a, b, c) = (0.178_832_77_f32, 0.284_668_92_f32, 0.559_910_7_f32);
        let e = e.clamp(0.0, 1.0);
        let scene = if e <= 0.5 { e * e / 3.0 } else { (((e - c) / a).exp() + b) / 12.0 };
        scene * 1000.0
    };
    let white = 1000.0f32 / 203.0;
    for px in rgb.chunks_exact_mut(3) {
        let lin: Vec<f32> = px.iter().map(|&e| if hlg { hlg_inverse(e) } else { pq(e) } / 203.0).collect();
        for (c, row) in M.iter().enumerate() {
            let x = (row[0] * lin[0] + row[1] * lin[1] + row[2] * lin[2]).max(0.0);
            let y = (x * (1.0 + x / (white * white)) / (1.0 + x)).min(1.0);
            px[c] = if y < 0.018 { 4.5 * y } else { 1.099 * y.powf(0.45) - 0.099 };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fits_like_the_provider() {
        // A 1080p frame decoded at 1036×582 goes in at 518×292, letterboxed by 50 rows.
        assert_eq!(fitted(1036, 582), (518, 292));
        assert_eq!(fitted(400, 400), (392, 392));
        assert_eq!(fitted(300, 200), (300, 200));
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
        let x = input(&rgb, 1036, 582);
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
