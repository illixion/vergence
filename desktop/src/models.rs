//! The depth models a conversion can download. A default build offers Depth
//! Anything V2 Small (Apache-2.0) only and names no other. With the
//! `noncommercial-models` feature (off by default: builds for non-commercial
//! use, sideloaded, never a store's) it also offers Base and Large, whose
//! weights are CC-BY-NC-4.0, after the person has read and accepted that
//! licence; on Windows and Linux only, where conversions run `vergence-convert`
//! (`--variant`, the same feature there). The table matches
//! `vergence_convert::model::downloads`.

use serde::Serialize;

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Variant {
    /// What `vergence-convert --variant` calls it.
    pub variant: &'static str,
    pub display: &'static str,
    pub licence: &'static str,
    pub licence_url: &'static str,
    /// What the licence means for the person, shown before the download.
    pub note: &'static str,
    pub megabytes: u32,
    /// Must be accepted before it is used.
    pub restricted: bool,
}

pub const SMALL: Variant = Variant {
    variant: "small",
    display: "Depth Anything V2 Small",
    licence: "Apache-2.0",
    licence_url: "https://www.apache.org/licenses/LICENSE-2.0",
    note: "",
    megabytes: 100,
    restricted: false,
};

/// The models this build offers here.
pub fn variants() -> Vec<Variant> {
    #[allow(unused_mut)]
    let mut all = vec![SMALL];
    #[cfg(all(feature = "noncommercial-models", not(target_os = "macos")))]
    {
        const NOTE: &str = "Non-commercial use only: the weights, and what you make with them, may not be used \
                            commercially. Credit Depth Anything V2 (Yang et al., 2024).";
        all.extend([
            Variant {
                variant: "base",
                display: "Depth Anything V2 Base",
                licence: "CC-BY-NC-4.0",
                licence_url: "https://creativecommons.org/licenses/by-nc/4.0/",
                note: NOTE,
                megabytes: 390,
                restricted: true,
            },
            Variant {
                variant: "large",
                display: "Depth Anything V2 Large",
                licence: "CC-BY-NC-4.0",
                licence_url: "https://creativecommons.org/licenses/by-nc/4.0/",
                note: NOTE,
                megabytes: 1340,
                restricted: true,
            },
        ]);
    }
    all
}

/// The variant a conversion uses: the chosen one if this build offers it and
/// (when restricted) its licence was accepted, else Small.
pub fn effective(chosen: Option<&str>, accepted: &[String]) -> Variant {
    variants()
        .into_iter()
        .find(|v| Some(v.variant) == chosen && (!v.restricted || accepted.iter().any(|a| a == v.variant)))
        .unwrap_or(SMALL)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offers_small_and_only_the_larger_ones_the_build_allows() {
        let names: Vec<&str> = variants().iter().map(|v| v.variant).collect();
        assert_eq!(names[0], "small");
        let larger = cfg!(all(feature = "noncommercial-models", not(target_os = "macos")));
        assert_eq!(names.contains(&"base"), larger);
        // Not accepted, or not offered: Small.
        assert_eq!(effective(Some("base"), &[]), SMALL);
        assert_eq!(effective(Some("base"), &["base".into()]).variant, if larger { "base" } else { "small" });
        assert_eq!(effective(None, &[]), SMALL);
    }
}
