//! Runs Smooth Motion's two networks with ONNX Runtime: RIFE 4.25 lite at
//! 480×288 for the left eye's motion between two real frames, NeuFlow v2 at
//! 512×288 for each real pair's right→left disparity. The ONNX side of
//! VergenceApple's `StereoFlowAnalyzer`, fed and read the same way:
//!
//! - RIFE: `img0`, `img1` 1×3×288×480 RGB 0…1, `timestep` ½; `flow` 1×4 (F½→0,
//!   F½→1) and `mask` 1×1.
//! - NeuFlow, one way: `img0` = the right eye, `img1` = the left, RGB 0…255;
//!   `flow` 1×2 (x, y), right→left.
//!
//! The exports come from `scripts/export-onnx-models.py`. Unlike the Core ML
//! models (float16 inputs, RIFE computed in float16) these are float32
//! throughout; how far apart the two land is what `scripts/check-onnx-worker.sh`
//! measures.
//!
//! The other workloads' models load beside them on the same provider: RIFE at
//! 960×576 for 2D titles (the same model at another size), Depth Anything V2
//! Small through `vergence_convert`'s depth pipeline, and GMNet
//! (`scripts/export-gmnet-onnx.py`): `image` 1×3×H×W and `thumbnail`
//! 1×3×256×256, RGB 0…1; `gain` 1×1×H×W, the linear multiplier.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use ort::ep::{self, ExecutionProviderDispatch};
use ort::environment::Environment;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::TensorRef;

use crate::preprocess::{DISPARITY_SIZE, FLOW_SIZE};

/// Which ONNX Runtime execution provider runs the models.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Provider {
    /// CUDA, then DirectML, then Core ML, then CPU: the first that loads.
    Auto,
    Cuda,
    TensorRt,
    DirectMl,
    CoreMl,
    Cpu,
}

impl Provider {
    pub fn parse(name: &str) -> Option<Provider> {
        Some(match name.to_ascii_lowercase().as_str() {
            "auto" | "" => Provider::Auto,
            "cuda" => Provider::Cuda,
            "tensorrt" | "trt" => Provider::TensorRt,
            "directml" | "dml" => Provider::DirectMl,
            "coreml" => Provider::CoreMl,
            "cpu" => Provider::Cpu,
            _ => return None,
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

    /// The order `Auto` tries. TensorRT is never picked on its own: its first
    /// run builds an engine for minutes, which should be asked for.
    fn candidates(self) -> Vec<Provider> {
        match self {
            Provider::Auto => vec![Provider::Cuda, Provider::DirectMl, Provider::CoreMl, Provider::Cpu],
            other => vec![other],
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct EngineOptions {
    /// `ane`, `gpu`, `cpu` or `all`, for Core ML.
    #[cfg_attr(not(any(feature = "coreml", target_os = "macos")), allow(dead_code))]
    pub compute_units: Option<String>,
    /// The GPU, for CUDA, TensorRT and DirectML.
    pub device_id: Option<i32>,
    /// Where TensorRT and Core ML keep what they compile, between runs.
    pub cache_dir: Option<PathBuf>,
}

fn dispatch(provider: Provider, options: &EngineOptions) -> Result<Option<ExecutionProviderDispatch>, String> {
    let device = options.device_id.unwrap_or(0);
    let cache = options.cache_dir.as_ref().map(|p| p.to_string_lossy().into_owned());
    let _ = (device, &cache);
    let not_built = |name: &str| Err(format!("this build has no {name} (cargo build --features {})", name.to_lowercase()));
    let ep = match provider {
        Provider::Auto | Provider::Cpu => return Ok(None),
        #[cfg(feature = "cuda")]
        Provider::Cuda => ep::CUDA::default().with_device_id(device).build(),
        #[cfg(feature = "tensorrt")]
        Provider::TensorRt => {
            let mut trt = ep::TensorRT::default().with_device_id(device);
            if let Some(cache) = cache {
                trt = trt.with_engine_cache(true).with_engine_cache_path(cache);
            }
            trt.build()
        }
        #[cfg(any(feature = "directml", windows))]
        Provider::DirectMl => ep::DirectML::default().with_device_id(device).build(),
        #[cfg(any(feature = "coreml", target_os = "macos"))]
        Provider::CoreMl => {
            let units = match options.compute_units.as_deref() {
                Some("ane") => ep::coreml::ComputeUnits::CPUAndNeuralEngine,
                Some("gpu") => ep::coreml::ComputeUnits::CPUAndGPU,
                Some("cpu") => ep::coreml::ComputeUnits::CPUOnly,
                _ => ep::coreml::ComputeUnits::All,
            };
            let mut coreml = ep::CoreML::default()
                .with_model_format(ep::coreml::ModelFormat::MLProgram)
                .with_static_input_shapes(true)
                .with_compute_units(units);
            if let Some(cache) = cache {
                coreml = coreml.with_model_cache_dir(cache);
            }
            coreml.build()
        }
        #[allow(unreachable_patterns)]
        other => return not_built(other.name()),
    };
    Ok(Some(ep.error_on_failure()))
}

fn session(env: &Environment, path: &Path, provider: Provider, options: &EngineOptions) -> Result<Session, String> {
    if provider != Provider::DirectMl {
        return session_with(env, path, provider, options, false);
    }
    // DirectML fuses the graph into its own, and its fusion rejects NeuFlow
    // ("The parameter is incorrect" in DmlGraphFusionHelper); without fusion
    // the same operators run one by one.
    session_with(env, path, provider, options, false).or_else(|fused| {
        session_with(env, path, provider, options, true).map_err(|unfused| format!("{fused}; without graph fusion: {unfused}"))
    })
}

fn session_with(env: &Environment, path: &Path, provider: Provider, options: &EngineOptions, no_dml_fusion: bool)
    -> Result<Session, String> {
    let fail = |e: ort::Error<_>| format!("{}: {e}", path.display());
    let mut builder = Session::builder(env)
        .map_err(|e| e.to_string())?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(fail)?;
    if provider == Provider::DirectMl {
        // DirectML wants neither memory patterns nor parallel execution.
        builder = builder.with_memory_pattern(false).map_err(fail)?.with_parallel_execution(false).map_err(fail)?;
        if no_dml_fusion {
            builder = builder.with_config_entry("ep.dml.disable_graph_fusion", "1").map_err(fail)?;
        }
    }
    if let Some(ep) = dispatch(provider, options)? {
        builder = builder.with_execution_providers([ep]).map_err(fail)?;
    }
    builder.commit_from_file(path).map_err(|e| format!("{}: {e}", path.display()))
}

/// ONNX Runtime's environment: one per process, kept for good, so the models
/// can be loaded again after standing aside.
fn environment() -> Result<Environment, String> {
    // Shared with the depth pipeline's, which makes its own sessions.
    vergence_convert::model::environment("vergence-worker")
}

/// The newest-named `prefix*size*.onnx` in `folder`, preferring one-way NeuFlow
/// exports (as `vergence analyze` picks among Core ML models).
pub fn find_model(folder: &Path, prefix: &str, size: &str) -> Result<PathBuf, String> {
    let mut matches: Vec<PathBuf> = std::fs::read_dir(folder)
        .map_err(|e| format!("{}: {e}", folder.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            p.extension().is_some_and(|e| e == "onnx") && name.starts_with(prefix) && name.contains(size)
        })
        .collect();
    matches.sort_by_key(|p| {
        let name = p.file_name().unwrap().to_string_lossy().into_owned();
        (!name.contains("1way"), name)
    });
    matches.into_iter().next().ok_or_else(|| {
        format!("no {prefix} model at {size} in {} (see scripts/export-onnx-models.py)", folder.display())
    })
}

fn stem(path: &Path) -> String {
    path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
}

fn expect_input(session: &Session, name: &str, shape: &[i64], model: &str) -> Result<(), String> {
    let outlet = session.inputs().iter().find(|i| i.name() == name).ok_or(format!("{model} has no input {name}"))?;
    let actual: Vec<i64> = outlet.dtype().tensor_shape().map(|s| s.iter().copied().collect()).unwrap_or_default();
    if actual != shape {
        return Err(format!("{model}: {name} is {actual:?}, wanted {shape:?}"));
    }
    Ok(())
}

pub struct FlowModel {
    session: Session,
    pub name: String,
    /// The model's working size, from its input.
    pub width: usize,
    pub height: usize,
}

pub struct DisparityModel {
    session: Session,
    pub name: String,
}

/// The 3D workload's pair.
pub struct StereoModels {
    pub flow: FlowModel,
    pub disparity: DisparityModel,
}

pub struct GainModel {
    session: Session,
    pub name: String,
}

/// Depth Anything through the converter's offline pipeline, and the GPU its
/// refinement runs on.
pub struct DepthEngine {
    pub model: Arc<Mutex<vergence_convert::model::DepthModel>>,
    pub gpu: vergence_convert::gpu::Gpu,
}

/// What to load: the workloads this worker takes.
#[derive(Clone, Debug, Default)]
pub struct Wanted {
    pub stereo: bool,
    /// Required (named in `workloads`), or only when its model is there.
    pub mono: Option<bool>,
    /// The model and the size to run it at when its input size is free.
    pub depth: Option<(PathBuf, (usize, usize))>,
    pub gain: Option<PathBuf>,
}

/// Every model this worker runs, on one execution provider. In the 3D pair,
/// `flow` and `disparity` may run at the same time from two threads.
pub struct Engine {
    pub stereo: Option<StereoModels>,
    pub mono: Option<FlowModel>,
    pub gain: Option<GainModel>,
    pub depth: Option<DepthEngine>,
    /// AI HDR's sticky run, if the last piece left one (`gain::Run`).
    pub gain_run: Option<crate::gain::Run>,
    /// The provider that loaded (never `Auto`).
    pub provider: Provider,
    _env: Environment,
}

impl Engine {
    pub fn load(models: &Path, wanted: &Wanted, provider: Provider, options: &EngineOptions) -> Result<Engine, String> {
        let stereo_paths = if wanted.stereo {
            Some((find_model(models, "RIFE", "480x288")?, find_model(models, "NeuFlow", "512x288")?))
        } else {
            None
        };
        // The family the 3D workload and the export use first, as `vergence worker` picks.
        let mono_path = match wanted.mono {
            None => None,
            Some(required) => match find_model(models, "RIFE425lite", "960x576").or_else(|_| find_model(models, "RIFE", "960x576")) {
                Ok(path) => Some(path),
                Err(e) if required => return Err(format!("smooth-motion-2d: {e}")),
                Err(_) => None,
            },
        };
        let env = environment()?;
        let mut errors = Vec::new();
        for candidate in provider.candidates() {
            match Engine::load_on(&env, candidate, options, stereo_paths.as_ref(), mono_path.as_deref(), wanted) {
                Ok(engine) => return Ok(engine),
                Err(e) => errors.push(format!("{}: {e}", candidate.name())),
            }
        }
        Err(format!("no execution provider could load the models ({})", errors.join("; ")))
    }

    fn load_on(env: &Environment, candidate: Provider, options: &EngineOptions, stereo: Option<&(PathBuf, PathBuf)>,
               mono: Option<&Path>, wanted: &Wanted) -> Result<Engine, String> {
        let flow_model = |path: &Path| -> Result<FlowModel, String> {
            let session = session(env, path, candidate, options)?;
            let (w, h) = input_size(&session, "img0").ok_or(format!("{}: img0 isn't 1×3×H×W", path.display()))?;
            let mut model = FlowModel { session, name: stem(path), width: w, height: h };
            // The first run specialises the model for the device; pay it here.
            let grey = vec![0.5f32; 3 * w * h];
            model.run(&grey, &grey)?;
            Ok(model)
        };
        let stereo = match stereo {
            Some((flow_path, disparity_path)) => {
                let flow = flow_model(flow_path)?;
                let (fw, fh) = FLOW_SIZE;
                let (dw, dh) = DISPARITY_SIZE;
                if (flow.width, flow.height) != (fw, fh) {
                    return Err(format!("RIFE: {}x{}, wanted {fw}x{fh}", flow.width, flow.height));
                }
                let disparity = session(env, disparity_path, candidate, options)?;
                expect_input(&disparity, "img0", &[1, 3, dh as i64, dw as i64], "NeuFlow")?;
                let mut disparity = DisparityModel { session: disparity, name: stem(disparity_path) };
                disparity.run(&vec![0.5f32; 3 * dw * dh], &vec![0.5f32; 3 * dw * dh])?;
                Some(StereoModels { flow, disparity })
            }
            None => None,
        };
        let mono = mono.map(flow_model).transpose()?;
        // GMNet and Depth Anything take free input sizes, for which TensorRT
        // builds an engine per size (minutes, gigabytes): CUDA runs them instead.
        let free_size = if candidate == Provider::TensorRt { Provider::Cuda } else { candidate };
        let gain = match &wanted.gain {
            Some(path) => {
                let mut model = GainModel { session: session(env, path, free_size, options)?, name: stem(path) };
                let (w, h) = GAIN_SIZES[0];
                model.run(&vec![0.5f32; 3 * w * h], w, h, &vec![0.5f32; 3 * 256 * 256])?;
                Some(model)
            }
            None => None,
        };
        let depth = match &wanted.depth {
            Some((path, size)) => {
                use vergence_convert::model::{DepthModel, Provider as DepthProvider};
                let provider = match free_size {
                    Provider::Auto => DepthProvider::Auto,
                    Provider::Cuda => DepthProvider::Cuda,
                    Provider::TensorRt => DepthProvider::TensorRt,
                    Provider::DirectMl => DepthProvider::DirectMl,
                    Provider::CoreMl => DepthProvider::CoreMl,
                    Provider::Cpu => DepthProvider::Cpu,
                };
                let model = DepthModel::load(path, provider, Some(*size))?;
                Some(DepthEngine { model: Arc::new(Mutex::new(model)), gpu: vergence_convert::gpu::Gpu::new()? })
            }
            None => None,
        };
        Ok(Engine { stereo, mono, gain, depth, gain_run: None, provider: candidate, _env: env.clone() })
    }
}

/// An input's (width, height) when it is 1×3×H×W with both fixed.
fn input_size(session: &Session, name: &str) -> Option<(usize, usize)> {
    let outlet = session.inputs().iter().find(|i| i.name() == name)?;
    let shape: Vec<i64> = outlet.dtype().tensor_shape().map(|s| s.iter().copied().collect())?;
    match shape[..] {
        [1, 3, h, w] if h > 0 && w > 0 => Some((w as usize, h as usize)),
        _ => None,
    }
}

impl FlowModel {
    /// The motion between two frames, at t = ½.
    /// `a`, `b`: planar RGB 0…1 at the model's size. Returns F½→0 (x, y), F½→1 (x, y)
    /// planar, and frame 0's blend weight, at the model's size.
    pub fn run(&mut self, a: &[f32], b: &[f32]) -> Result<(Vec<f32>, Vec<f32>), String> {
        let (w, h) = (self.width, self.height);
        let shape = [1usize, 3, h, w];
        let t = [0.5f32];
        let e = |e: ort::Error| format!("RIFE: {e}");
        let outputs = self
            .session
            .run(ort::inputs![
                "img0" => TensorRef::from_array_view((shape, a)).map_err(e)?,
                "img1" => TensorRef::from_array_view((shape, b)).map_err(e)?,
                "timestep" => TensorRef::from_array_view(([1usize, 1, 1, 1], &t[..])).map_err(e)?,
            ])
            .map_err(e)?;
        let (_, flow) = outputs["flow"].try_extract_tensor::<f32>().map_err(e)?;
        let (_, mask) = outputs["mask"].try_extract_tensor::<f32>().map_err(e)?;
        if flow.len() != 4 * w * h || mask.len() != w * h {
            return Err("RIFE's outputs have the wrong size".into());
        }
        Ok((flow.to_vec(), mask.to_vec()))
    }
}

impl DisparityModel {
    /// Right→left disparity of one real pair, x then y planar at 512×288.
    /// `left`, `right`: planar RGB 0…1 at 512×288.
    pub fn run(&mut self, left: &[f32], right: &[f32]) -> Result<Vec<f32>, String> {
        let (w, h) = DISPARITY_SIZE;
        let shape = [1usize, 3, h, w];
        // NeuFlow takes 0…255; img0 → img1 is right → left.
        let right: Vec<f32> = right.iter().map(|v| v * 255.0).collect();
        let left: Vec<f32> = left.iter().map(|v| v * 255.0).collect();
        let e = |e: ort::Error| format!("NeuFlow: {e}");
        let outputs = self
            .session
            .run(ort::inputs![
                "img0" => TensorRef::from_array_view((shape, &right[..])).map_err(e)?,
                "img1" => TensorRef::from_array_view((shape, &left[..])).map_err(e)?,
            ])
            .map_err(e)?;
        let (_, flow) = outputs["flow"].try_extract_tensor::<f32>().map_err(e)?;
        if flow.len() < 2 * w * h {
            return Err("NeuFlow's output has the wrong size".into());
        }
        Ok(flow[..2 * w * h].to_vec())
    }
}

/// GMNet's input sizes (w, h): the Core ML conversion's enumerated ones.
pub const GAIN_SIZES: [(usize, usize); 5] = [(1024, 768), (768, 1024), (1024, 576), (576, 1024), (896, 896)];

impl GainModel {
    /// The linear gain per pixel of `image` (planar RGB 0…1 at `w`×`h`, one of
    /// `GAIN_SIZES`) given the whole frame's 256×256 `thumbnail`. Both are
    /// rounded to bytes first, as the Core ML model's pixel buffers are.
    /// Returns the gain and its (width, height).
    pub fn run(&mut self, image: &[f32], w: usize, h: usize, thumbnail: &[f32]) -> Result<(Vec<f32>, usize, usize), String> {
        let bytes = |v: &[f32]| -> Vec<f32> { v.iter().map(|x| (x.clamp(0.0, 1.0) * 255.0).round() / 255.0).collect() };
        let (image, thumbnail) = (bytes(image), bytes(thumbnail));
        let e = |e: ort::Error| format!("GMNet: {e}");
        let outputs = self
            .session
            .run(ort::inputs![
                "image" => TensorRef::from_array_view(([1usize, 3, h, w], &image[..])).map_err(e)?,
                "thumbnail" => TensorRef::from_array_view(([1usize, 3, 256, 256], &thumbnail[..])).map_err(e)?,
            ])
            .map_err(e)?;
        let (shape, gain) = outputs["gain"].try_extract_tensor::<f32>().map_err(e)?;
        let dims: Vec<i64> = shape.iter().copied().collect();
        let [_, _, gh, gw] = dims[..] else { return Err("GMNet's gain isn't 1×1×H×W".into()) };
        Ok((gain.to_vec(), gw as usize, gh as usize))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_models_like_analyze_does() {
        let dir = tempfile::tempdir().unwrap();
        for name in ["NeuFlowV2_512x288_i4.onnx", "NeuFlowV2_512x288_1way_i4_fp32.onnx", "RIFE425lite_480x288.onnx",
                     "RIFE425lite_960x576.onnx", "RIFE425lite_480x288.mlpackage"] {
            std::fs::write(dir.path().join(name), b"").unwrap();
        }
        assert_eq!(stem(&find_model(dir.path(), "NeuFlow", "512x288").unwrap()), "NeuFlowV2_512x288_1way_i4_fp32");
        assert_eq!(stem(&find_model(dir.path(), "RIFE", "480x288").unwrap()), "RIFE425lite_480x288");
        assert!(find_model(dir.path(), "RIFE", "640x384").is_err());
    }

    #[test]
    fn keeps_one_environment() {
        assert!(environment().is_ok());
        assert!(environment().is_ok(), "a second load must reuse the environment");
    }

    #[test]
    fn parses_provider_names() {
        assert_eq!(Provider::parse("DirectML"), Some(Provider::DirectMl));
        assert_eq!(Provider::parse("trt"), Some(Provider::TensorRt));
        assert_eq!(Provider::parse("auto"), Some(Provider::Auto));
        assert_eq!(Provider::parse("vulkan"), None);
    }
}
