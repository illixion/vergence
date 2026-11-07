//! The parts of `vergence-convert` other programs use: the offline depth
//! pipeline (decode, Depth Anything through ONNX Runtime, the refinement, the
//! stored maps and their analysis-track records) and what it stands on.
//! `vergence-worker` runs it per piece for the `ai-3d-depth` workload.

pub mod decode;
pub mod depth;
pub mod gpu;
pub mod licence;
pub mod maths;
pub mod model;
pub mod probe;
pub mod track;
