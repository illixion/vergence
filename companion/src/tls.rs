//! The paired channel: TLS 1.2 with an external PSK, on OpenSSL.
//!
//! Matches what the other peers configure:
//! - Swift (Network.framework): `sec_protocol_options_add_pre_shared_key` per
//!   credential, min and max version pinned to TLS 1.2, ciphersuite 0x00A8
//!   (`TLS_PSK_WITH_AES_128_GCM_SHA256`), Nagle off.
//! - C# (BouncyCastle `PskTlsServer`): TLS 1.2 only, the same suite, PSK chosen
//!   by the identity the client presents, no certificate status echoed.
//!
//! rustls has no external-PSK support, hence OpenSSL. After the handshake each
//! connection proves which device it is with an [`Identify`] over the
//! session's RFC 5705 exporter value (label from
//! [`crate::credential::exporter_label`], **empty** context, 32 bytes).

use std::io::{Read, Write};
use std::sync::{Arc, OnceLock};

use openssl::error::ErrorStack;
use openssl::ex_data::Index;
use openssl::ssl::{HandshakeError, Ssl, SslContext, SslMethod, SslStream, SslVersion};

use crate::credential::{Channel, Credential, Identify, PSK_CIPHER_NAME};
use crate::keyring::{Keyring, KeyringEntry};
use crate::namespace::Namespace;

#[derive(Debug, thiserror::Error)]
pub enum TlsError {
    #[error("OpenSSL: {0}")]
    Ssl(#[from] ErrorStack),
    /// An unknown or revoked device, a client that isn't TLS-PSK, or a dropped connection.
    #[error("TLS handshake failed: {0}")]
    Handshake(String),
}

impl<S> From<HandshakeError<S>> for TlsError {
    fn from(error: HandshakeError<S>) -> Self {
        Self::Handshake(match error {
            HandshakeError::SetupFailure(e) => e.to_string(),
            HandshakeError::Failure(mid) | HandshakeError::WouldBlock(mid) => {
                format!("{}", mid.error())
            }
        })
    }
}

/// Which credential the handshake used, kept on the `SSL` by the PSK callback.
struct Matched(Option<String>);

fn matched_index() -> Index<Ssl, Matched> {
    static INDEX: OnceLock<Index<Ssl, Matched>> = OnceLock::new();
    *INDEX.get_or_init(|| Ssl::new_ex_index().expect("OpenSSL ex_data index"))
}

fn base_context() -> Result<openssl::ssl::SslContextBuilder, ErrorStack> {
    let mut builder = SslContext::builder(SslMethod::tls())?;
    builder.set_min_proto_version(Some(SslVersion::TLS1_2))?;
    builder.set_max_proto_version(Some(SslVersion::TLS1_2))?;
    builder.set_cipher_list(PSK_CIPHER_NAME)?;
    Ok(builder)
}

/// A listener's TLS side: holds every enrolled device's key at once, picking
/// each connection's by the PSK identity it presents.
#[derive(Clone)]
pub struct PskServer {
    context: SslContext,
}

impl std::fmt::Debug for PskServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PskServer").finish_non_exhaustive()
    }
}

impl PskServer {
    /// `token_for` maps a device id (`None`: the shared token) to its current
    /// token, or `None` to refuse it. It is asked on every handshake, so a
    /// revoked device is refused at once.
    pub fn new(
        channel: Channel,
        token_for: impl Fn(Option<&str>) -> Option<String> + Send + Sync + 'static,
    ) -> Result<Self, ErrorStack> {
        let mut builder = base_context()?;
        let channel = Arc::new(channel);
        builder.set_psk_server_callback(move |ssl, identity, psk| {
            // 0 tells OpenSSL the identity is unknown; the handshake then fails.
            let Some(device) = identity.and_then(|id| channel.parse_identity(id)) else {
                return Ok(0);
            };
            let Some(token) = token_for(device.as_deref()) else {
                return Ok(0);
            };
            let key = channel.derive(&token);
            if psk.len() < key.len() {
                return Ok(0);
            }
            psk[..key.len()].copy_from_slice(&key);
            ssl.set_ex_data(matched_index(), Matched(device));
            Ok(key.len())
        });
        Ok(Self {
            context: builder.build(),
        })
    }

    /// Keys from a keyring, which the pairing side keeps current.
    pub fn with_keyring(channel: Channel, keyring: Keyring) -> Result<Self, ErrorStack> {
        Self::new(channel, move |device| keyring.token_for(device))
    }

    /// Runs the server handshake on an accepted stream.
    pub fn accept<S: Read + Write>(&self, stream: S) -> Result<PskSession<S>, TlsError> {
        let stream = Ssl::new(&self.context)?.accept(stream)?;
        let device_id = stream
            .ssl()
            .ex_data(matched_index())
            .map(|m| m.0.clone())
            .ok_or_else(|| TlsError::Handshake("no PSK identity was matched".into()))?;
        Ok(PskSession { stream, device_id })
    }
}

/// An accepted connection and the credential its handshake used.
#[derive(Debug)]
pub struct PskSession<S> {
    pub stream: SslStream<S>,
    /// The device whose PSK the handshake used (`None`: the shared token).
    pub device_id: Option<String>,
}

impl<S: Read + Write> PskSession<S> {
    /// The session's exporter value for `label`.
    pub fn binding(&self, exporter_label: &str) -> Result<[u8; 32], ErrorStack> {
        binding(&self.stream, exporter_label)
    }

    /// Checks a connection's identify message: it must name the credential the
    /// handshake used and prove that device's token over this session.
    pub fn verify(
        &self,
        identify: &Identify,
        exporter_label: &str,
        keyring: &Keyring,
    ) -> Option<KeyringEntry> {
        if identify.device_id != self.device_id {
            return None;
        }
        keyring.verify(identify, &self.binding(exporter_label).ok()?)
    }
}

/// A device's TLS side for one channel and its own credential.
#[derive(Clone)]
pub struct PskClient {
    context: SslContext,
}

impl std::fmt::Debug for PskClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PskClient").finish_non_exhaustive()
    }
}

impl PskClient {
    pub fn new(channel: &Channel, credential: &Credential) -> Result<Self, ErrorStack> {
        let mut builder = base_context()?;
        let identity = channel.identity(credential);
        let key = channel.derive(&credential.token);
        builder.set_psk_client_callback(move |_ssl, _hint, identity_out, psk_out| {
            // The identity goes out as a NUL-terminated C string.
            if identity_out.len() <= identity.len() || psk_out.len() < key.len() {
                return Ok(0);
            }
            identity_out[..identity.len()].copy_from_slice(&identity);
            identity_out[identity.len()] = 0;
            psk_out[..key.len()].copy_from_slice(&key);
            Ok(key.len())
        });
        Ok(Self {
            context: builder.build(),
        })
    }

    pub fn connect<S: Read + Write>(&self, stream: S) -> Result<SslStream<S>, TlsError> {
        Ok(Ssl::new(&self.context)?.connect(stream)?)
    }
}

/// RFC 5705 keying material with an empty context, 32 bytes: what Apple's
/// `sec_protocol_metadata_create_secret` exports and BouncyCastle's
/// `ExportKeyingMaterial(label, new byte[0], 32)` matches.
pub fn binding<S>(stream: &SslStream<S>, exporter_label: &str) -> Result<[u8; 32], ErrorStack> {
    let mut out = [0u8; 32];
    stream
        .ssl()
        .export_keying_material(&mut out, exporter_label, Some(&[]))?;
    Ok(out)
}

/// A device's identify message for this connection.
pub fn identify<S>(
    namespace: &Namespace,
    credential: &Credential,
    stream: &SslStream<S>,
    exporter_label: &str,
) -> Result<Identify, ErrorStack> {
    Ok(Identify::make(
        namespace,
        credential,
        &binding(stream, exporter_label)?,
    ))
}
