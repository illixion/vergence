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

use std::path::{Path, PathBuf};

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
}

pub struct DisparityModel {
    session: Session,
    pub name: String,
}

/// Both models on one execution provider. `flow` and `disparity` may run at
/// the same time from two threads (each needs only its own half).
pub struct Engine {
    pub flow: FlowModel,
    pub disparity: DisparityModel,
    /// The provider that loaded (never `Auto`).
    pub provider: Provider,
    _env: Environment,
}

impl Engine {
    pub fn load(models: &Path, provider: Provider, options: &EngineOptions) -> Result<Engine, String> {
        let flow_path = find_model(models, "RIFE", "480x288")?;
        let disparity_path = find_model(models, "NeuFlow", "512x288")?;
        let env = ort::init().with_name("vergence-worker").build().map_err(|e| e.to_string())?;
        let mut errors = Vec::new();
        for candidate in provider.candidates() {
            let loaded = session(&env, &flow_path, candidate, options)
                .and_then(|f| session(&env, &disparity_path, candidate, options).map(|d| (f, d)));
            match loaded {
                Ok((flow, disparity)) => {
                    let (fw, fh) = FLOW_SIZE;
                    let (dw, dh) = DISPARITY_SIZE;
                    expect_input(&flow, "img0", &[1, 3, fh as i64, fw as i64], "RIFE")?;
                    expect_input(&disparity, "img0", &[1, 3, dh as i64, dw as i64], "NeuFlow")?;
                    let mut engine = Engine {
                        flow: FlowModel { session: flow, name: stem(&flow_path) },
                        disparity: DisparityModel { session: disparity, name: stem(&disparity_path) },
                        provider: candidate,
                        _env: env,
                    };
                    // The first run specialises the models for the device; pay it here.
                    let grey = vec![0.5f32; 3 * fw * fh];
                    engine.flow.run(&grey, &grey)?;
                    let grey = vec![0.5f32; 3 * dw * dh];
                    engine.disparity.run(&grey, &grey)?;
                    return Ok(engine);
                }
                Err(e) => errors.push(format!("{}: {e}", candidate.name())),
            }
        }
        Err(format!("no execution provider could load the models ({})", errors.join("; ")))
    }
}

impl FlowModel {
    /// The left eye's motion between two real frames, at t = ½.
    /// `a`, `b`: planar RGB 0…1 at 480×288. Returns F½→0 (x, y), F½→1 (x, y)
    /// planar, and frame 0's blend weight, at 480×288.
    pub fn run(&mut self, a: &[f32], b: &[f32]) -> Result<(Vec<f32>, Vec<f32>), String> {
        let (w, h) = FLOW_SIZE;
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
    fn parses_provider_names() {
        assert_eq!(Provider::parse("DirectML"), Some(Provider::DirectMl));
        assert_eq!(Provider::parse("trt"), Some(Provider::TensorRt));
        assert_eq!(Provider::parse("auto"), Some(Provider::Auto));
        assert_eq!(Provider::parse("vulkan"), None);
    }
}
