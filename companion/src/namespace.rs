//! Which companion a device pairs with: Bonjour service types, key-derivation
//! labels and the link scheme (`RAVECompanionNamespace`).

/// An app's namespace. Longwave's is [`Namespace::longwave`]; Vergence's
/// coordinator uses its own so the two never answer each other.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Namespace {
    /// Bonjour service stem: `_<stem>-mac._tcp`, `_<stem>-knock._tcp`, `_<stem>-pair._tcp`.
    pub service_stem: String,
    /// Prefix of every key-derivation label (`<prefix>-Knock-v1`, `<prefix>-Pair-v2`,
    /// `<prefix>-Device-Proof-v1`).
    pub label_prefix: String,
    /// Scheme of the address link (`<scheme>://pair?…`).
    pub url_scheme: String,
    /// The host's fixed direct-pairing port, dialled from an address link.
    pub default_pair_port: u16,
    /// The host app's name, for messages the user reads.
    pub host_app_name: String,
}

impl Namespace {
    pub fn new(
        service_stem: impl Into<String>,
        label_prefix: impl Into<String>,
        url_scheme: impl Into<String>,
        default_pair_port: u16,
        host_app_name: impl Into<String>,
    ) -> Self {
        Self {
            service_stem: service_stem.into(),
            label_prefix: label_prefix.into(),
            url_scheme: url_scheme.into(),
            default_pair_port,
            host_app_name: host_app_name.into(),
        }
    }

    /// Longwave's Companion (Mac and Windows).
    pub fn longwave() -> Self {
        Self::new(
            "longwave",
            "Longwave",
            "longwave",
            4859,
            "Longwave Companion",
        )
    }

    /// The host's announcement: a bare record, no socket behind it.
    pub fn host_service_type(&self) -> String {
        format!("_{}-mac._tcp", self.service_stem)
    }

    /// A device asking a host to open its ports, or to pair.
    pub fn knock_service_type(&self) -> String {
        format!("_{}-knock._tcp", self.service_stem)
    }

    /// The host's one-off listener for one pairing.
    pub fn pair_service_type(&self) -> String {
        format!("_{}-pair._tcp", self.service_stem)
    }

    pub(crate) fn knock_salt(&self) -> Vec<u8> {
        format!("{}-Knock-v1", self.label_prefix).into_bytes()
    }

    pub(crate) fn pair_info(&self) -> Vec<u8> {
        format!("{}-Pair-v2", self.label_prefix).into_bytes()
    }

    pub(crate) fn proof_salt(&self) -> Vec<u8> {
        format!("{}-Device-Proof-v1", self.label_prefix).into_bytes()
    }
}
