//! What the page can ask for: a bounded set of operations. Media-server keys go in (to be kept
//! in a 0600 file) and never come back out; a coordinator token is shown once, when it is made;
//! files are picked through native dialogs in Rust, so the page never names a path to read.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use serde_json::{Value, json};
use tauri::{AppHandle, Emitter, Manager, State};
use tauri_plugin_autostart::ManagerExt;
use tauri_plugin_dialog::DialogExt;
use vergence_coordinator::auth::Grant;

use crate::convert::{ConvertOptions, ConvertQueue};
use crate::coordinator::CoordinatorHost;
use crate::pairing::Pairing;
use crate::settings::{Paths, Server, Settings, key_hint, set_secret};
use crate::updater::Updates;
use crate::worker::WorkerHost;
use crate::{sidecar, tray};

pub struct App {
    pub paths: Paths,
    pub settings: Arc<Mutex<Settings>>,
    pub coordinator: Arc<CoordinatorHost>,
    pub pairing: Arc<Pairing>,
    pub worker: Arc<WorkerHost>,
    pub convert: Arc<ConvertQueue>,
    pub updates: Updates,
}

type Shared<'a> = State<'a, Arc<App>>;
type Reply = Result<Value, String>;

impl App {
    pub fn new(paths: Paths) -> App {
        let settings = Arc::new(Mutex::new(Settings::load(&paths)));
        let coordinator = Arc::new(CoordinatorHost::new(paths.clone()));
        App {
            pairing: Arc::new(Pairing::new(paths.clone(), coordinator.clone())),
            worker: Arc::new(WorkerHost::new(paths.clone(), coordinator.clone())),
            convert: Arc::new(ConvertQueue::new(paths.convert_queue(), settings.clone())),
            settings,
            coordinator,
            paths,
            updates: Updates::default(),
        }
    }

    pub fn settings(&self) -> Settings {
        self.settings.lock().unwrap().clone()
    }

    /// Starts the coordinator for the current settings (and the worker, if it should run).
    pub async fn start_services(&self) {
        let settings = self.settings();
        if let Err(e) = self.coordinator.start(&settings).await {
            tracing::warn!(error = %e, "coordinator");
        }
        if settings.worker_at_start
            && let Err(e) = self.worker.start(&settings)
        {
            tracing::warn!(error = %e, "worker");
        }
    }

    fn settings_view(&self, app: &AppHandle) -> Value {
        let settings = self.settings();
        json!({
            "settings": settings,
            "jellyfinKey": key_hint(&self.paths, Server::Jellyfin),
            "stashKey": key_hint(&self.paths, Server::Stash),
            "cli": sidecar::find_tool(&settings, sidecar::ToolKind::Convert).map(|t| t.program.to_string_lossy().into_owned()),
            "ffmpeg": sidecar::find_ffmpeg().map(|p| p.to_string_lossy().into_owned()),
            "launchAtLogin": app.autolaunch().is_enabled().unwrap_or(false),
            "dataDir": self.paths.root,
        })
    }

    pub fn status(&self) -> Value {
        let settings = self.settings();
        json!({
            "coordinator": self.coordinator.status(),
            "worker": { "running": self.worker.is_running(), "available": cfg!(target_os = "macos") },
            "localUrl": settings.local_url(),
            "lan": settings.lan,
            "addresses": crate::pairing::lan_addresses(),
            "port": settings.port,
        })
    }
}

#[tauri::command]
pub fn app_info(state: Shared<'_>) -> Value {
    json!({
        "version": env!("CARGO_PKG_VERSION"),
        "platform": std::env::consts::OS,
        "convertAvailable": crate::convert::AVAILABLE,
        "workerAvailable": cfg!(target_os = "macos"),
        "dataDir": state.paths.root,
    })
}

#[tauri::command]
pub fn status(state: Shared<'_>) -> Value {
    state.status()
}

#[tauri::command]
pub fn get_settings(app: AppHandle, state: Shared<'_>) -> Value {
    state.settings_view(&app)
}

/// Saves the settings. A key that is `None` stays as it is; an empty one is removed.
#[tauri::command]
pub async fn save_settings(
    app: AppHandle,
    state: Shared<'_>,
    settings: Settings,
    jellyfin_key: Option<String>,
    stash_key: Option<String>,
) -> Reply {
    let settings = settings.normalized()?;
    for (server, key) in [(Server::Jellyfin, jellyfin_key), (Server::Stash, stash_key)] {
        if let Some(key) = key {
            set_secret(&state.paths, server, &key).map_err(|e| format!("Couldn't keep the {} key: {e}", server.name()))?;
        }
    }
    settings.save(&state.paths).map_err(|e| format!("Couldn't save the settings: {e}"))?;
    *state.settings.lock().unwrap() = settings.clone();
    // The servers and the port are read at start: restart what depends on them.
    let worker_was_running = state.worker.is_running();
    state.worker.stop_and_wait().await;
    let started = state.coordinator.start(&settings).await;
    if worker_was_running {
        state.worker.start(&settings)?;
    }
    tray::refresh(&app);
    let mut view = state.settings_view(&app);
    view["coordinatorError"] = json!(started.err());
    Ok(view)
}

/// Asks a server who it is, with the stored key: the sign-in check.
#[tauri::command]
pub async fn test_server(state: Shared<'_>, server: Server) -> Reply {
    let settings = state.settings();
    let url = settings.url(server).ok_or("Save the server's address first.")?.to_string();
    let key = std::fs::read_to_string(state.paths.secret(server)).ok().map(|k| k.trim().to_string());
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .user_agent(concat!("Vergence/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|e| e.to_string())?;
    let response = match server {
        Server::Jellyfin => {
            let mut request = client.get(format!("{url}/System/Info"));
            if let Some(key) = &key {
                request = request.header("Authorization", format!("MediaBrowser Token=\"{key}\""));
            }
            request.send().await
        }
        Server::Stash => {
            let mut request = client.post(format!("{url}/graphql")).json(&json!({ "query": "{ version { version } }" }));
            if let Some(key) = &key {
                request = request.header("ApiKey", key);
            }
            request.send().await
        }
    }
    .map_err(|e| format!("Couldn't reach {url}: {e}"))?;
    let status = response.status();
    if status == 401 || status == 403 {
        return Err(format!("{url} refused the key ({status})."));
    }
    if !status.is_success() {
        return Err(format!("{url} answered {status}."));
    }
    let body: Value = response.json().await.unwrap_or(Value::Null);
    let detail = match server {
        Server::Jellyfin => format!(
            "{} (Jellyfin {})",
            body["ServerName"].as_str().unwrap_or("Jellyfin"),
            body["Version"].as_str().unwrap_or("?")
        ),
        Server::Stash => {
            if body["errors"].is_array() {
                return Err(format!("{url} refused the query: {}", body["errors"][0]["message"].as_str().unwrap_or("error")));
            }
            format!("Stash {}", body["data"]["version"]["version"].as_str().unwrap_or("?"))
        }
    };
    Ok(json!({ "ok": true, "detail": detail }))
}

#[tauri::command]
pub fn queue_snapshot(state: Shared<'_>) -> Reply {
    state.coordinator.snapshot()
}

#[tauri::command]
pub async fn scan_now(state: Shared<'_>) -> Reply {
    state.coordinator.scan_now().await
}

#[tauri::command]
pub fn remove_title(state: Shared<'_>, key: String) -> Reply {
    Ok(json!(state.coordinator.remove_title(&key)?))
}

#[tauri::command]
pub fn tokens_list(state: Shared<'_>) -> Reply {
    let entries = state.coordinator.tokens()?;
    Ok(json!(entries.iter().map(|e| json!({ "name": e.name, "grants": e.grants })).collect::<Vec<_>>()))
}

/// Makes a `play` or `work` token, returned this once.
#[tauri::command]
pub fn token_create(state: Shared<'_>, name: String, grant: String) -> Reply {
    let grant = match grant.as_str() {
        "play" => Grant::Play,
        "work" => Grant::Work,
        _ => return Err("A token is for play or work.".into()),
    };
    let token = state.coordinator.add_token(&name, vec![grant])?;
    Ok(json!({ "name": name.trim(), "token": token }))
}

#[tauri::command]
pub fn token_remove(state: Shared<'_>, name: String) -> Reply {
    Ok(json!(state.coordinator.remove_token(&name)?))
}

#[tauri::command]
pub fn pairing_status(state: Shared<'_>) -> Value {
    state.pairing.status()
}

#[tauri::command]
pub fn pairing_start(state: Shared<'_>) -> Reply {
    state.pairing.start(crate::pairing::namespace().default_pair_port)
}

#[tauri::command]
pub fn pairing_stop(state: Shared<'_>) -> Value {
    state.pairing.stop();
    state.pairing.status()
}

#[tauri::command]
pub fn pairing_decide(state: Shared<'_>, id: u64, pair: bool) -> Reply {
    state.pairing.decide(id, pair)?;
    Ok(state.pairing.status())
}

#[tauri::command]
pub fn device_forget(state: Shared<'_>, id: String) -> Reply {
    state.pairing.forget(&id)?;
    Ok(state.pairing.status())
}

#[tauri::command]
pub fn worker_status(state: Shared<'_>) -> Value {
    state.worker.status(&state.settings())
}

/// Async: the sidecar is spawned on the runtime.
#[tauri::command]
pub async fn worker_start(app: AppHandle, state: Shared<'_>) -> Reply {
    state.worker.start(&state.settings())?;
    tray::refresh(&app);
    Ok(state.worker.status(&state.settings()))
}

#[tauri::command]
pub async fn worker_stop(app: AppHandle, state: Shared<'_>) -> Reply {
    state.worker.stop_and_wait().await;
    tray::refresh(&app);
    Ok(state.worker.status(&state.settings()))
}

/// Picks the models folder (it goes straight into the settings).
#[tauri::command]
pub async fn pick_models_folder(app: AppHandle, state: Shared<'_>) -> Reply {
    let Some(folder) = pick_folder(&app, "Choose the interpolation models folder").await else {
        return Ok(state.settings_view(&app));
    };
    let mut settings = state.settings();
    settings.models_folder = Some(folder.to_string_lossy().into_owned());
    settings.save(&state.paths).map_err(|e| e.to_string())?;
    *state.settings.lock().unwrap() = settings;
    Ok(state.settings_view(&app))
}

#[tauri::command]
pub fn convert_status(state: Shared<'_>) -> Value {
    state.convert.status()
}

/// The options picked and dropped files are converted with.
#[tauri::command]
pub fn convert_set_options(state: Shared<'_>, options: ConvertOptions) -> Reply {
    state.convert.set_defaults(options)?;
    Ok(state.convert.status())
}

#[tauri::command]
pub async fn convert_pick(app: AppHandle, state: Shared<'_>, folder: bool) -> Reply {
    let picked: Vec<PathBuf> = if folder {
        pick_folder(&app, "Choose a folder of videos").await.into_iter().collect()
    } else {
        let (tx, rx) = tokio::sync::oneshot::channel();
        app.dialog().file().set_title("Choose videos to convert").pick_files(move |files| {
            let _ = tx.send(files.unwrap_or_default().into_iter().filter_map(|f| f.into_path().ok()).collect::<Vec<_>>());
        });
        rx.await.unwrap_or_default()
    };
    if !picked.is_empty() {
        state.convert.add(&picked)?;
    }
    Ok(state.convert.status())
}

#[tauri::command]
pub async fn convert_pick_output(app: AppHandle, state: Shared<'_>, beside: bool) -> Reply {
    if beside {
        state.convert.set_output_dir(None)?;
    } else if let Some(folder) = pick_folder(&app, "Where the 3D copies go").await {
        state.convert.set_output_dir(Some(folder))?;
    }
    Ok(state.convert.status())
}

#[tauri::command]
pub fn convert_cancel(state: Shared<'_>, id: u64) -> Reply {
    state.convert.cancel(id)?;
    Ok(state.convert.status())
}

#[tauri::command]
pub fn convert_retry(state: Shared<'_>, id: u64) -> Reply {
    state.convert.retry(id)?;
    Ok(state.convert.status())
}

#[tauri::command]
pub fn convert_remove(state: Shared<'_>, id: u64) -> Reply {
    state.convert.remove(id)?;
    Ok(state.convert.status())
}

#[tauri::command]
pub fn convert_clear_finished(state: Shared<'_>) -> Reply {
    state.convert.clear_finished()?;
    Ok(state.convert.status())
}

/// Shows a finished copy in Finder (or Explorer).
#[tauri::command]
pub fn convert_reveal(state: Shared<'_>, id: u64) -> Reply {
    let output = state.convert.output_of(id).filter(|p| p.exists()).ok_or("The copy isn't there.")?;
    #[cfg(target_os = "macos")]
    let result = std::process::Command::new("/usr/bin/open").arg("-R").arg(&output).status();
    #[cfg(windows)]
    let result = std::process::Command::new("explorer").arg(format!("/select,{}", output.display())).status();
    #[cfg(not(any(target_os = "macos", windows)))]
    let result = std::process::Command::new("xdg-open").arg(output.parent().unwrap_or(&output)).status();
    result.map_err(|e| e.to_string())?;
    Ok(json!(true))
}

#[tauri::command]
pub fn set_launch_at_login(app: AppHandle, on: bool) -> Reply {
    let manager = app.autolaunch();
    if on { manager.enable() } else { manager.disable() }.map_err(|e| format!("Couldn't change the login item: {e}"))?;
    Ok(json!(manager.is_enabled().unwrap_or(on)))
}

#[tauri::command]
pub fn update_status(state: Shared<'_>) -> Value {
    json!(*state.updates.status.lock().unwrap())
}

#[tauri::command]
pub async fn check_update(state: Shared<'_>) -> Reply {
    let settings = state.settings();
    Ok(json!(state.updates.check(settings.check_updates, settings.private_updates).await?))
}

#[tauri::command]
pub async fn install_update(app: AppHandle, state: Shared<'_>) -> Reply {
    if state.updates.install().await? {
        app.request_restart();
    }
    Ok(json!(true))
}

#[tauri::command]
pub fn quit_app(app: AppHandle) {
    app.exit(0);
}

async fn pick_folder(app: &AppHandle, title: &str) -> Option<PathBuf> {
    let (tx, rx) = tokio::sync::oneshot::channel();
    app.dialog().file().set_title(title).pick_folder(move |folder| {
        let _ = tx.send(folder.and_then(|f| f.into_path().ok()));
    });
    rx.await.ok().flatten()
}

/// Files dropped on the window or opened with the app: into the conversion queue.
pub fn add_files(app: &AppHandle, paths: Vec<PathBuf>) {
    let state = app.state::<Arc<App>>();
    let result = state.convert.add(&paths);
    let _ = app.emit("convert-added", json!({ "added": result.as_ref().ok(), "error": result.err() }));
}

/// The depth models a conversion can download, the chosen one, and whose licence was accepted.
#[tauri::command]
pub fn depth_variants(state: Shared<'_>) -> Value {
    let settings = state.settings();
    let effective = crate::models::effective(settings.depth_variant.as_deref(), &settings.licences_accepted);
    json!({ "variants": crate::models::variants(), "chosen": effective.variant, "accepted": settings.licences_accepted })
}

/// Chooses the model conversions download. A restricted one needs `accept_licence`: the page
/// has shown its licence and the person agreed.
#[tauri::command]
pub fn choose_depth_variant(state: Shared<'_>, variant: String, accept_licence: bool) -> Reply {
    let Some(chosen) = crate::models::variants().into_iter().find(|v| v.variant == variant) else {
        return Err(format!("This build doesn't offer {variant}."));
    };
    let mut settings = state.settings();
    if chosen.restricted && !settings.licences_accepted.contains(&variant) {
        if !accept_licence {
            return Err(format!("{} is {}: accept its licence first.", chosen.display, chosen.licence));
        }
        settings.licences_accepted.push(variant.clone());
    }
    settings.depth_variant = Some(variant).filter(|v| v != crate::models::SMALL.variant);
    settings.save(&state.paths).map_err(|e| format!("Couldn't save the settings: {e}"))?;
    *state.settings.lock().unwrap() = settings;
    Ok(depth_variants(state))
}

