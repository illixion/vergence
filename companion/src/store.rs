//! Where a host keeps what it paired: its own stable id and, per device, the
//! token and the grants the user allowed.

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use rand_core::{OsRng, RngCore};
use serde::{Deserialize, Serialize};

/// One enrolled device.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceRecord {
    #[serde(rename = "deviceID")]
    pub device_id: String,
    /// The name the device gave when it paired.
    pub name: String,
    /// The device's own token. Every channel key and proof derives from it.
    pub token: String,
    /// Raw grant names as the pairing carried them (sorted).
    pub grants: Vec<String>,
    /// A public key line the device offered and the user allowed.
    #[serde(
        rename = "sshPublicKey",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub ssh_public_key: Option<String>,
    /// Seconds since 1970 when it paired.
    #[serde(rename = "pairedAt", default)]
    pub paired_at: u64,
}

impl std::fmt::Debug for DeviceRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceRecord")
            .field("device_id", &self.device_id)
            .field("name", &self.name)
            .field("grants", &self.grants)
            .field("paired_at", &self.paired_at)
            .finish_non_exhaustive()
    }
}

impl DeviceRecord {
    pub fn credential(&self) -> crate::credential::Credential {
        crate::credential::Credential::new(self.token.clone(), Some(self.device_id.clone()))
    }
}

pub(crate) fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// A host's paired devices. Implementations must be safe to share between a
/// pairing thread and the listeners.
pub trait CredentialStore: Send + Sync {
    /// The id devices remember this host by: stable, random, and not tied to
    /// any token. Made on first use.
    fn host_id(&self) -> io::Result<String>;
    fn devices(&self) -> io::Result<Vec<DeviceRecord>>;
    fn device(&self, device_id: &str) -> io::Result<Option<DeviceRecord>> {
        Ok(self
            .devices()?
            .into_iter()
            .find(|d| d.device_id == device_id))
    }
    /// Adds a device, replacing any earlier pairing of the same id.
    fn put(&self, record: DeviceRecord) -> io::Result<()>;
    /// Forgets a device. Returns whether it was there.
    fn remove(&self, device_id: &str) -> io::Result<bool>;
}

/// What a store holds: the host's id and its devices. A [`JsonFileStore`]'s file is this as JSON,
/// and so is the one Keychain item a `KeychainStore` keeps.
#[derive(Clone, Default, Serialize, Deserialize)]
pub(crate) struct FileContents {
    #[serde(rename = "hostID", default, skip_serializing_if = "Option::is_none")]
    pub(crate) host_id: Option<String>,
    #[serde(default)]
    pub(crate) devices: Vec<DeviceRecord>,
}

/// A JSON file readable by its owner only (0600 on Unix), rewritten whole on
/// each change through a temporary file and a rename.
pub struct JsonFileStore {
    path: PathBuf,
    lock: Mutex<()>,
}

impl JsonFileStore {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            lock: Mutex::new(()),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn load(&self) -> io::Result<FileContents> {
        match fs::read(&self.path) {
            Ok(bytes) => serde_json::from_slice(&bytes)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e)),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(FileContents::default()),
            Err(e) => Err(e),
        }
    }

    fn save(&self, contents: &FileContents) -> io::Result<()> {
        let dir = self
            .path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        fs::create_dir_all(dir)?;
        let name = self
            .path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("credentials.json");
        let temp = dir.join(format!(".{name}.{}.tmp", std::process::id()));
        let bytes = serde_json::to_vec_pretty(contents).map_err(io::Error::other)?;
        {
            let mut options = fs::OpenOptions::new();
            options.write(true).create(true).truncate(true);
            #[cfg(unix)]
            std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
            let mut file = options.open(&temp)?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                file.set_permissions(fs::Permissions::from_mode(0o600))?;
            }
            file.write_all(&bytes)?;
            file.sync_all()?;
        }
        fs::rename(&temp, &self.path)
    }

    fn update<T>(&self, change: impl FnOnce(&mut FileContents) -> T) -> io::Result<T> {
        let _guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        let mut contents = self.load()?;
        let result = change(&mut contents);
        self.save(&contents)?;
        Ok(result)
    }
}

impl CredentialStore for JsonFileStore {
    fn host_id(&self) -> io::Result<String> {
        {
            let _guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(id) = self.load()?.host_id {
                return Ok(id);
            }
        }
        self.update(|c| c.host_id.get_or_insert_with(random_uuid).clone())
    }

    fn devices(&self) -> io::Result<Vec<DeviceRecord>> {
        let _guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        Ok(self.load()?.devices)
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

/// An uppercase version-4 UUID, as Foundation's `UUID().uuidString`.
pub fn random_uuid() -> String {
    let mut b = [0u8; 16];
    OsRng.fill_bytes(&mut b);
    b[6] = (b[6] & 0x0f) | 0x40;
    b[8] = (b[8] & 0x3f) | 0x80;
    let hex: String = b.iter().map(|x| format!("{x:02X}")).collect();
    format!(
        "{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}
