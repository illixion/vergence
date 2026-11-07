//! The address link (`RAVECompanionPairLink`): the hint a host hands a device
//! that can't see it over Bonjour.
//!
//! `<scheme>://pair?mac=<id>&name=<name>&port=4859&addr=192.168.1.5&addr=100.64.0.3`
//!
//! It carries no secret: anyone could write one, and it grants nothing. The
//! device tries each address and runs the normal six-digit pairing over the
//! first that answers.

use std::net::IpAddr;

use percent_encoding::{AsciiSet, NON_ALPHANUMERIC, percent_decode_str, utf8_percent_encode};

use crate::namespace::Namespace;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PairLink {
    pub namespace: Namespace,
    pub mac_id: String,
    pub name: String,
    pub port: u16,
    pub addresses: Vec<String>,
}

pub const ACTION: &str = "pair";

/// What `URLComponents.queryItems` leaves alone in a value: the unreserved
/// characters and `!$'()*+,;:@/?`. It encodes `&`, `=` and `#`.
const QUERY_VALUE: &AsciiSet = &NON_ALPHANUMERIC
    .remove(b'-')
    .remove(b'.')
    .remove(b'_')
    .remove(b'~')
    .remove(b'!')
    .remove(b'$')
    .remove(b'\'')
    .remove(b'(')
    .remove(b')')
    .remove(b'*')
    .remove(b'+')
    .remove(b',')
    .remove(b';')
    .remove(b':')
    .remove(b'@')
    .remove(b'/')
    .remove(b'?');

impl PairLink {
    /// `port` defaults to the namespace's direct-pairing port.
    pub fn new(
        namespace: &Namespace,
        mac_id: impl Into<String>,
        name: impl Into<String>,
        port: Option<u16>,
        addresses: Vec<String>,
    ) -> Self {
        Self {
            namespace: namespace.clone(),
            mac_id: mac_id.into(),
            name: name.into(),
            port: port.unwrap_or(namespace.default_pair_port),
            addresses,
        }
    }

    /// The link as Foundation's `URLComponents` writes it.
    pub fn url(&self) -> String {
        let mut items = vec![
            ("mac", self.mac_id.clone()),
            ("name", self.name.clone()),
            ("port", self.port.to_string()),
        ];
        items.extend(self.addresses.iter().map(|a| ("addr", a.clone())));
        let query = items
            .iter()
            .map(|(k, v)| format!("{k}={}", utf8_percent_encode(v, QUERY_VALUE)))
            .collect::<Vec<_>>()
            .join("&");
        format!("{}://{ACTION}?{query}", self.namespace.url_scheme)
    }

    /// Parses and vets a link: the right scheme and action, a host id, at
    /// least one IP address (at most 12 are kept), and, if a port is given, the
    /// namespace's own, so a link can't redirect pairing to another service.
    pub fn parse(namespace: &Namespace, url: &str) -> Option<Self> {
        let (scheme, rest) = url.split_once("://")?;
        if !scheme.eq_ignore_ascii_case(&namespace.url_scheme) {
            return None;
        }
        let rest = rest.split('#').next().unwrap_or_default();
        let (authority, query) = match rest.split_once('?') {
            Some((a, q)) => (a, q),
            None => (rest, ""),
        };
        let host = authority.split('/').next().unwrap_or_default();
        if !host.eq_ignore_ascii_case(ACTION) {
            return None;
        }
        let items: Vec<(String, Option<String>)> = query
            .split('&')
            .filter(|s| !s.is_empty())
            .map(|item| {
                let decode = |s: &str| percent_decode_str(s).decode_utf8_lossy().into_owned();
                match item.split_once('=') {
                    Some((k, v)) => (decode(k), Some(decode(v))),
                    None => (decode(item), None),
                }
            })
            .collect();
        let first = |name: &str| {
            items
                .iter()
                .find(|(k, _)| k == name)
                .map(|(_, v)| v.clone())
        };
        let mac = first("mac")??;
        if mac.is_empty() || mac.chars().count() > 64 {
            return None;
        }
        let addresses: Vec<String> = items
            .iter()
            .filter(|(k, _)| k == "addr")
            .filter_map(|(_, v)| v.clone())
            .filter(|a| a.chars().count() < 64 && is_ip(a))
            .take(12)
            .collect();
        if addresses.is_empty() {
            return None;
        }
        if let Some(raw_port) = first("port")
            && raw_port?.parse::<u16>().ok()? != namespace.default_pair_port
        {
            return None;
        }
        let name = first("name").flatten().unwrap_or_else(|| "Mac".into());
        Some(Self {
            namespace: namespace.clone(),
            mac_id: mac,
            name: name.chars().take(80).collect(),
            port: namespace.default_pair_port,
            addresses,
        })
    }
}

/// An IPv4 or IPv6 address, an IPv6 zone (`%en0`) allowed as Network.framework does.
fn is_ip(text: &str) -> bool {
    let bare = text.split('%').next().unwrap_or_default();
    bare.parse::<IpAddr>()
        .is_ok_and(|ip| !text.contains('%') || ip.is_ipv6())
}
