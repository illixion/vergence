//! A host's paired devices in the macOS login Keychain instead of a file.
//!
//! The same contents a [`JsonFileStore`](crate::JsonFileStore) writes (the host id and each
//! device's record, its pairing token included) as one generic-password item, so on a Mac a
//! pairing token is never in a plain file. Windows and Linux keep the 0600 file.
//!
//! The item is read once and then served from memory; every change reads it again, applies
//! itself and writes it through at once. [`KeychainStore::import_file`] moves an older
//! `devices.json` in and deletes it.

use std::io;
use std::path::Path;
use std::sync::Mutex;

use security_framework::passwords::{delete_generic_password, get_generic_password, set_generic_password};

use crate::store::{CredentialStore, DeviceRecord, FileContents, random_uuid};

/// `errSecItemNotFound`.
const NOT_FOUND: i32 = -25300;

pub struct KeychainStore {
    service: String,
    account: String,
    cache: Mutex<Option<FileContents>>,
}

impl KeychainStore {
    /// The item the Keychain shows as `service` (e.g. the app's bundle id) / `account`.
    pub fn new(service: impl Into<String>, account: impl Into<String>) -> Self {
        KeychainStore { service: service.into(), account: account.into(), cache: Mutex::new(None) }
    }

    fn read(&self) -> io::Result<FileContents> {
        match get_generic_password(&self.service, &self.account) {
            Ok(bytes) => serde_json::from_slice(&bytes).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e)),
            Err(e) if e.code() == NOT_FOUND => Ok(FileContents::default()),
            Err(e) => Err(io::Error::other(format!("the Keychain: {e}"))),
        }
    }

    fn write(&self, contents: &FileContents) -> io::Result<()> {
        let bytes = serde_json::to_vec(contents).map_err(io::Error::other)?;
        set_generic_password(&self.service, &self.account, &bytes).map_err(|e| io::Error::other(format!("the Keychain: {e}")))
    }

    fn with<T>(&self, f: impl FnOnce(&FileContents) -> T) -> io::Result<T> {
        let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        if cache.is_none() {
            *cache = Some(self.read()?);
        }
        Ok(f(cache.as_ref().expect("just filled")))
    }

    fn update<T>(&self, change: impl FnOnce(&mut FileContents) -> T) -> io::Result<T> {
        let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        // Changes are rare: start from what the Keychain holds now, not the cache, so another
        // store on the same item (or another copy of the app) is never overwritten with old contents.
        let mut contents = self.read()?;
        let result = change(&mut contents);
        let written = self.write(&contents);
        // Keep what the Keychain now holds: the new contents, or (on failure) read it again next time.
        *cache = written.is_ok().then_some(contents);
        written.map(|()| result)
    }

    /// Moves the devices of an older `devices.json` at `path` into this store (a device already
    /// here wins) and deletes the file. Returns how many it moved.
    pub fn import_file(&self, path: &Path) -> io::Result<usize> {
        let bytes = match std::fs::read(path) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(0),
            Err(e) => return Err(e),
        };
        let old: FileContents = serde_json::from_slice(&bytes).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let moved = self.update(|contents| {
            if contents.host_id.is_none() {
                contents.host_id = old.host_id.clone();
            }
            let mut moved = 0;
            for device in old.devices {
                if !contents.devices.iter().any(|d| d.device_id == device.device_id) {
                    contents.devices.push(device);
                    moved += 1;
                }
            }
            moved
        })?;
        std::fs::remove_file(path)?;
        Ok(moved)
    }

    /// Deletes the item (tests, and a reset).
    pub fn delete(&self) -> io::Result<()> {
        *self.cache.lock().unwrap_or_else(|e| e.into_inner()) = None;
        match delete_generic_password(&self.service, &self.account) {
            Ok(()) => Ok(()),
            Err(e) if e.code() == NOT_FOUND => Ok(()),
            Err(e) => Err(io::Error::other(format!("the Keychain: {e}"))),
        }
    }
}

impl CredentialStore for KeychainStore {
    fn host_id(&self) -> io::Result<String> {
        if let Some(id) = self.with(|c| c.host_id.clone())? {
            return Ok(id);
        }
        self.update(|c| c.host_id.get_or_insert_with(random_uuid).clone())
    }

    fn devices(&self) -> io::Result<Vec<DeviceRecord>> {
        self.with(|c| c.devices.clone())
    }

    fn put(&self, record: DeviceRecord) -> io::Result<()> {
        self.update(|c| {
            c.devices.retain(|d| d.device_id != record.device_id);
            c.devices.push(record);
        })
    }

    fn remove(&self, device_id: &str) -> io::Result<bool> {
        self.update(|c| {
            let before = c.devices.len();
            c.devices.retain(|d| d.device_id != device_id);
            c.devices.len() != before
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Writes a real item to the login Keychain (and removes it). Ignored by default: it can
    /// raise a Keychain prompt on a locked keychain or a CI runner without one.
    #[test]
    #[ignore]
    fn keeps_devices_in_the_keychain_and_imports_a_file() {
        let store = KeychainStore::new("com.illixion.vergence.test", format!("devices-{}", random_uuid()));
        let record = DeviceRecord {
            device_id: "D1".into(),
            name: "Vision Pro".into(),
            token: "t".repeat(43),
            grants: vec!["play".into()],
            ssh_public_key: None,
            paired_at: 1,
        };
        store.put(record.clone()).unwrap();
        let again = KeychainStore::new(store.service.clone(), store.account.clone());
        assert_eq!(again.devices().unwrap(), vec![record.clone()]);
        let host = again.host_id().unwrap();
        assert_eq!(store.host_id().unwrap(), host);

        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("devices.json");
        let older = crate::JsonFileStore::new(&file);
        older.put(DeviceRecord { device_id: "D2".into(), ..record.clone() }).unwrap();
        older.put(DeviceRecord { device_id: "D1".into(), name: "older".into(), ..record }).unwrap();
        assert_eq!(again.import_file(&file).unwrap(), 1);
        assert!(!file.exists());
        let names: Vec<_> = again.devices().unwrap().into_iter().map(|d| (d.device_id, d.name)).collect();
        assert_eq!(names, [("D1".to_string(), "Vision Pro".to_string()), ("D2".to_string(), "Vision Pro".to_string())]);
        again.delete().unwrap();
    }
}
