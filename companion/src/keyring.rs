//! The host's current set of enrolled devices, as its listeners read it
//! (`RAVECompanionKeyring`): a lock-protected snapshot, cheap to share.

use std::collections::BTreeSet;
use std::sync::{Arc, RwLock};

use crate::credential::{Credential, Identify, validates};
use crate::namespace::Namespace;
use crate::store::DeviceRecord;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeyringEntry {
    pub credential: Credential,
    /// Raw grant names, as the pairing carried them.
    pub grants: BTreeSet<String>,
}

impl From<&DeviceRecord> for KeyringEntry {
    fn from(record: &DeviceRecord) -> Self {
        Self {
            credential: Credential::new(record.token.clone(), Some(record.device_id.clone())),
            grants: record.grants.iter().cloned().collect(),
        }
    }
}

/// Clones share the same entries.
#[derive(Clone, Debug)]
pub struct Keyring {
    namespace: Namespace,
    entries: Arc<RwLock<Vec<KeyringEntry>>>,
}

impl Keyring {
    pub fn new(namespace: Namespace, entries: Vec<KeyringEntry>) -> Self {
        Self {
            namespace,
            entries: Arc::new(RwLock::new(entries)),
        }
    }

    pub fn from_records<'a>(
        namespace: Namespace,
        records: impl IntoIterator<Item = &'a DeviceRecord>,
    ) -> Self {
        Self::new(
            namespace,
            records.into_iter().map(KeyringEntry::from).collect(),
        )
    }

    pub fn namespace(&self) -> &Namespace {
        &self.namespace
    }

    pub fn replace(&self, entries: Vec<KeyringEntry>) {
        *self.entries.write().unwrap_or_else(|e| e.into_inner()) = entries;
    }

    pub fn replace_from_records<'a>(&self, records: impl IntoIterator<Item = &'a DeviceRecord>) {
        self.replace(records.into_iter().map(KeyringEntry::from).collect());
    }

    fn read(&self) -> std::sync::RwLockReadGuard<'_, Vec<KeyringEntry>> {
        self.entries.read().unwrap_or_else(|e| e.into_inner())
    }

    /// Every credential a listener should accept.
    pub fn credentials(&self) -> Vec<Credential> {
        self.read().iter().map(|e| e.credential.clone()).collect()
    }

    /// Enrolled devices holding a grant.
    pub fn entries_with(&self, grant: &str) -> Vec<KeyringEntry> {
        self.read()
            .iter()
            .filter(|e| e.grants.contains(grant))
            .cloned()
            .collect()
    }

    pub fn entry(&self, device_id: Option<&str>) -> Option<KeyringEntry> {
        self.read()
            .iter()
            .find(|e| e.credential.device_id.as_deref() == device_id)
            .cloned()
    }

    /// The token a PSK identity's device holds, for a TLS listener.
    pub fn token_for(&self, device_id: Option<&str>) -> Option<String> {
        self.entry(device_id).map(|e| e.credential.token)
    }

    /// Checks a connection's identify message. Returns the entry it proves.
    pub fn verify(&self, identify: &Identify, binding: &[u8]) -> Option<KeyringEntry> {
        let entry = self.entry(identify.device_id.as_deref())?;
        validates(
            &self.namespace,
            &identify.proof,
            &entry.credential.token,
            binding,
        )
        .then_some(entry)
    }

    /// Whether this device (`None`: the shared token) is still enrolled with the grant.
    pub fn allows(&self, device_id: Option<&str>, grant: &str) -> bool {
        self.entry(device_id)
            .is_some_and(|e| e.grants.contains(grant))
    }
}
