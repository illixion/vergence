//! Which depth models' weights restrict their use: the Rust side of the
//! command line's `DepthModelLicence.swift`, by the same rule.
//!
//! Vergence is MIT and ships no weights. Depth Anything V2 Small, which
//! everything is built and tuned around, is Apache-2.0. Depth Anything V2 Base
//! and Large and Video Depth Anything Base and Large are CC-BY-NC-4.0
//! (non-commercial use only), and Depth Pro's weights come under Apple's
//! research licence. `vergence-convert` and `vergence-worker` take such a model
//! only when built with the `noncommercial-models` feature (off by default),
//! and say its licence when they do; a worker's profile names it, so its
//! titles never mix with Small's.
//!
//! Recognised by the model's file name (exports keep the family and size in
//! it); a licence an export's metadata states counts too, but can't clear a
//! name, since an exporter may copy the Small model's metadata.

/// A licence that restricts a model's use.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Restriction {
    pub name: String,
    pub url: String,
    pub note: String,
}

fn cc_by_nc() -> Restriction {
    Restriction {
        name: "CC-BY-NC-4.0".into(),
        url: "https://creativecommons.org/licenses/by-nc/4.0/".into(),
        note: "non-commercial use only; credit Depth Anything V2 (Yang et al., 2024)".into(),
    }
}

/// Whether this build takes models under a restrictive licence.
pub const NONCOMMERCIAL_MODELS: bool = cfg!(feature = "noncommercial-models");

/// The restriction a model's weights carry, or None when none is known
/// (Depth Anything V2 Small, Apache-2.0). `metadata_licence`: what the model's
/// metadata says, if read.
pub fn restriction(model_name: &str, metadata_licence: Option<&str>) -> Option<Restriction> {
    let name = model_name.to_lowercase().replace(['-', '_'], "");
    if name.contains("depthpro") {
        return Some(Restriction {
            name: "Apple ML Research Model licence".into(),
            url: "https://huggingface.co/apple/DepthPro".into(),
            note: "research and non-commercial use only".into(),
        });
    }
    let family = name.contains("depthanything") || name.contains("dav2") || name.contains("vda");
    if family && ["base", "large", "giant", "vitb", "vitl", "vitg"].iter().any(|s| name.contains(s)) {
        return Some(cc_by_nc());
    }
    let licence = metadata_licence?;
    let lower = licence.to_lowercase();
    ((lower.contains("nc") && lower.contains("cc")) || lower.contains("non-commercial") || lower.contains("noncommercial")
        || lower.contains("research"))
        .then(|| Restriction { name: licence.into(), url: String::new(), note: "non-commercial or research use only".into() })
}

/// The restriction of the model at `path`, by its file name.
pub fn restriction_of(path: &std::path::Path) -> Option<Restriction> {
    restriction(&path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(), None)
}

/// Refuses a restricted model unless this build takes them; otherwise the
/// line to say about it (None for an unrestricted one).
pub fn check(path: &std::path::Path) -> Result<Option<String>, String> {
    let Some(r) = restriction_of(path) else { return Ok(None) };
    let name = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    if NONCOMMERCIAL_MODELS {
        let url = if r.url.is_empty() { String::new() } else { format!(" ({})", r.url) };
        Ok(Some(format!("{name}'s weights are {}{url}: {}. Use it, and what you make with it, for non-commercial work only.",
                        r.name, r.note)))
    } else {
        Err(format!("{name}'s weights are {} ({}); this build takes only Apache-2.0 models such as Depth Anything V2 \
                     Small. Build with --features noncommercial-models to use it for non-commercial work.",
                    r.name, r.note))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn small_is_free_and_the_larger_ones_are_not() {
        assert_eq!(restriction("DepthAnythingV2Small", None), None);
        assert_eq!(restriction("DepthAnythingV2Small770x574", None), None);
        assert_eq!(restriction("DepthAnythingV2BaseF16", Some("Apache 2")).unwrap().name, "CC-BY-NC-4.0");
        assert_eq!(restriction("DepthAnything_v2-Large", None).unwrap().name, "CC-BY-NC-4.0");
        assert_eq!(restriction("depth-anything-v2-large", None).unwrap().name, "CC-BY-NC-4.0");
        assert!(restriction("DepthProNormalizedInverseDepth", None).is_some());
        assert!(restriction("Custom", Some("CC-BY-NC-4.0")).is_some());
    }

    #[test]
    fn without_the_feature_a_larger_model_is_refused() {
        let base = check(Path::new("/m/DepthAnythingV2Base.onnx"));
        if NONCOMMERCIAL_MODELS {
            assert!(base.unwrap().unwrap().contains("CC-BY-NC-4.0"));
        } else {
            assert!(base.unwrap_err().contains("noncommercial-models"));
        }
        assert_eq!(check(Path::new("/m/DepthAnythingV2Small.onnx")), Ok(None));
    }
}
