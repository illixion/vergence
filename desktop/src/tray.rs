//! The tray (menu-bar) icon that keeps the app running with its window closed, autostart with
//! `--hidden`, and showing and hiding the window. Adapted from Apparition's
//! `desktop/src/companion.rs` (MIT, same author).

use std::sync::{Arc, Mutex};
use std::time::Duration;

use tauri::menu::{Menu, MenuItem, PredefinedMenuItem};
use tauri::tray::TrayIconBuilder;
use tauri::{AppHandle, Emitter, Manager, Wry};

use crate::commands::App;

/// The argument the login item starts the app with: straight to the tray.
pub const HIDDEN: &str = "--hidden";

struct Items {
    coordinator: MenuItem<Wry>,
    worker: MenuItem<Wry>,
    convert: MenuItem<Wry>,
}

static ITEMS: Mutex<Option<Items>> = Mutex::new(None);

pub fn setup(app: &AppHandle) -> tauri::Result<()> {
    let coordinator = MenuItem::with_id(app, "coordinator", "Coordinator", false, None::<&str>)?;
    let worker = MenuItem::with_id(app, "worker", "Worker", false, None::<&str>)?;
    let convert = MenuItem::with_id(app, "convert", "Conversions", false, None::<&str>)?;
    let open = MenuItem::with_id(app, "open", "Open Vergence", true, None::<&str>)?;
    let pair = MenuItem::with_id(app, "pair", "Pair a Headset…", true, None::<&str>)?;
    let quit = MenuItem::with_id(app, "quit", "Quit Vergence", true, None::<&str>)?;
    let separator = || PredefinedMenuItem::separator(app);
    let menu = Menu::with_items(app, &[&coordinator, &worker, &convert, &separator()?, &open, &pair, &separator()?, &quit])?;
    *ITEMS.lock().unwrap() = Some(Items { coordinator, worker, convert });
    // A template image on macOS (drawn in the menu bar's colour); the colour icon elsewhere.
    #[cfg(target_os = "macos")]
    let icon = tauri::image::Image::from_bytes(include_bytes!("../icons/tray.png"))?;
    #[cfg(not(target_os = "macos"))]
    let icon = tauri::image::Image::from_bytes(include_bytes!("../icons/32x32.png"))?;
    TrayIconBuilder::with_id("main")
        .icon(icon)
        .icon_as_template(cfg!(target_os = "macos"))
        .tooltip("Vergence")
        .menu(&menu)
        .show_menu_on_left_click(true)
        .on_menu_event(|app, event| match event.id.as_ref() {
            "open" => show_window(app),
            "pair" => {
                show_window(app);
                let _ = app.emit("show-tab", "headsets");
            }
            "quit" => app.exit(0),
            _ => {}
        })
        .build(app)?;
    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        loop {
            refresh(&app);
            tokio::time::sleep(Duration::from_secs(5)).await;
        }
    });
    Ok(())
}

/// Brings the menu's three status lines up to date.
pub fn refresh(app: &AppHandle) {
    let Some(state) = app.try_state::<Arc<App>>() else { return };
    let status = state.coordinator.status();
    let coordinator = match (status["running"].as_bool(), status["listen"].as_str(), status["error"].as_str()) {
        (Some(true), Some(listen), _) => format!("Coordinator on {listen}"),
        (_, _, Some(_)) => "Coordinator couldn't start".to_string(),
        _ => "Coordinator off".to_string(),
    };
    let worker = if state.worker.is_running() { "Worker running" } else { "Worker stopped" };
    let jobs = state.convert.jobs();
    let waiting = jobs.iter().filter(|j| matches!(j.state, crate::convert::JobState::Queued | crate::convert::JobState::Running)).count();
    let convert = match jobs.iter().find(|j| j.state == crate::convert::JobState::Running) {
        Some(job) => format!(
            "Converting {} ({:.0}%){}",
            job.input.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
            job.progress * 100.0,
            if waiting > 1 { format!(", {} more", waiting - 1) } else { String::new() }
        ),
        None if waiting > 0 => format!("{waiting} conversions waiting"),
        None => "No conversions".to_string(),
    };
    if let Some(items) = &*ITEMS.lock().unwrap() {
        let _ = items.coordinator.set_text(coordinator);
        let _ = items.worker.set_text(worker);
        let _ = items.convert.set_text(convert);
    }
}

/// Shows the window (and, on macOS, the Dock icon).
pub fn show_window(app: &AppHandle) {
    #[cfg(target_os = "macos")]
    let _ = app.set_activation_policy(tauri::ActivationPolicy::Regular);
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.unminimize();
        let _ = window.show();
        let _ = window.set_focus();
    }
}

/// Hides the window; everything keeps running from the tray (and, on macOS, leaves the Dock).
pub fn hide_window(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.hide();
    }
    #[cfg(target_os = "macos")]
    let _ = app.set_activation_policy(tauri::ActivationPolicy::Accessory);
}
