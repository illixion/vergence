//! Vergence, the desktop app: the analysis coordinator (`vergence-coordinator`, the same code
//! `vergenced` runs headless) and a worker in one tray app, headset pairing
//! (`vergence-companion`), and AI 3D conversion of video files through `vergence convert`.
//! Design: Hypnos `docs/plans/analysis-workers.md`, "The Vergence desktop app".
//!
//! The shell (tray, autostart, single instance, the window closing to the tray, the bounded
//! command set, the verified updater) is adapted from Apparition's desktop app (MIT, same
//! author).
//!
//! Secrets: media-server keys arrive from the page once, go into 0600 files, and are never sent
//! back; coordinator tokens are shown once when made and kept only as hashes.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod announce;
mod commands;
mod convert;
mod coordinator;
mod models;
mod pairing;
mod settings;
mod sidecar;
mod tray;
mod updater;
mod worker;

use std::path::PathBuf;
use std::sync::Arc;

use tauri::{Emitter, Manager};

use crate::commands::App;
use crate::settings::Paths;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "vergence_app=info,vergence_desktop=info,vergence_coordinator=info".into()),
        )
        .with_writer(std::io::stderr)
        .init();
    tauri::Builder::default()
        // Opening the app again shows the running one's window; files it was opened with are converted.
        .plugin(tauri_plugin_single_instance::init(|app, args, cwd| {
            tray::show_window(app);
            let files: Vec<PathBuf> = args.iter().skip(1).filter(|a| !a.starts_with('-')).map(|a| PathBuf::from(&cwd).join(a)).filter(|p| p.exists()).collect();
            if !files.is_empty() {
                commands::add_files(app, files);
            }
        }))
        .plugin(tauri_plugin_autostart::init(tauri_plugin_autostart::MacosLauncher::LaunchAgent, Some(vec![tray::HIDDEN])))
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            commands::app_info,
            commands::status,
            commands::get_settings,
            commands::save_settings,
            commands::test_server,
            commands::queue_snapshot,
            commands::scan_now,
            commands::remove_title,
            commands::tokens_list,
            commands::token_create,
            commands::token_remove,
            commands::pairing_status,
            commands::pairing_start,
            commands::pairing_stop,
            commands::pairing_decide,
            commands::device_forget,
            commands::worker_status,
            commands::worker_start,
            commands::worker_stop,
            commands::pick_models_folder,
            commands::convert_status,
            commands::convert_set_options,
            commands::convert_pick,
            commands::convert_pick_output,
            commands::convert_cancel,
            commands::convert_retry,
            commands::convert_remove,
            commands::convert_clear_finished,
            commands::convert_reveal,
            commands::set_launch_at_login,
            commands::depth_variants,
            commands::choose_depth_variant,
            commands::update_status,
            commands::check_update,
            commands::install_update,
            commands::quit_app,
        ])
        .setup(|app| {
            // VERGENCE_APP_DATA moves everything (for a second copy, or a test run).
            let root = match std::env::var_os("VERGENCE_APP_DATA") {
                Some(dir) => PathBuf::from(dir),
                None => app.path().app_data_dir()?,
            };
            std::fs::create_dir_all(&root)?;
            let state = Arc::new(App::new(Paths::new(root)));
            app.manage(state.clone());
            let handle = app.handle().clone();
            {
                let handle = handle.clone();
                state.pairing.on_change(Arc::new(move || {
                    let state = handle.state::<Arc<App>>();
                    let status = state.pairing.status();
                    // A device asking to pair brings the window up with its code.
                    if !status["waiting"].is_null() {
                        tray::show_window(&handle);
                        let _ = handle.emit("show-tab", "headsets");
                    }
                    let _ = handle.emit("pairing", status);
                }));
            }
            {
                let handle = handle.clone();
                state.convert.on_change(Arc::new(move || {
                    let _ = handle.emit("convert", ());
                }));
            }
            tauri::async_runtime::spawn(state.convert.clone().run());
            {
                let state = state.clone();
                let handle = handle.clone();
                tauri::async_runtime::spawn(async move {
                    state.start_services().await;
                    tray::refresh(&handle);
                    let _ = handle.emit("status", state.status());
                });
            }
            {
                // Updates: checked every few hours, only when turned on (and there is a site).
                let state = state.clone();
                tauri::async_runtime::spawn(async move {
                    loop {
                        if state.settings().check_updates {
                            let _ = state.updates.check(true).await;
                        }
                        tokio::time::sleep(updater::CHECK_INTERVAL).await;
                    }
                });
            }
            tray::setup(&handle)?;
            // Development: queue files to convert at launch (VERGENCE_APP_CONVERT=a.mp4:b.mkv).
            #[cfg(debug_assertions)]
            if let Some(files) = std::env::var_os("VERGENCE_APP_CONVERT") {
                commands::add_files(&handle, std::env::split_paths(&files).collect());
            }
            if std::env::args().any(|arg| arg == tray::HIDDEN) {
                tray::hide_window(&handle);
            }
            Ok(())
        })
        .on_window_event(|window, event| match event {
            // Closing the window keeps everything running; Quit is in the tray menu.
            tauri::WindowEvent::CloseRequested { api, .. } => {
                api.prevent_close();
                tray::hide_window(window.app_handle());
            }
            tauri::WindowEvent::DragDrop(tauri::DragDropEvent::Drop { paths, .. }) => {
                let _ = window.app_handle().emit("show-tab", "convert");
                commands::add_files(window.app_handle(), paths.clone());
            }
            _ => {}
        })
        .build(tauri::generate_context!())
        .expect("couldn't start Vergence")
        .run(|app, event| match event {
            tauri::RunEvent::Exit => {
                // Children are stopped by their own PIDs; a conversion stays queued to carry on.
                if let Some(state) = app.try_state::<Arc<App>>() {
                    let state = state.inner().clone();
                    state.pairing.stop();
                    tauri::async_runtime::block_on(async move {
                        state.convert.shutdown().await;
                        state.worker.stop_and_wait().await;
                        state.coordinator.shutdown().await;
                    });
                }
            }
            // "Open With" from Finder.
            #[cfg(target_os = "macos")]
            tauri::RunEvent::Opened { urls } => {
                let files: Vec<PathBuf> = urls.iter().filter_map(|u| u.to_file_path().ok()).collect();
                if !files.is_empty() {
                    tray::show_window(app);
                    let _ = app.emit("show-tab", "convert");
                    commands::add_files(app, files);
                }
            }
            #[cfg(target_os = "macos")]
            tauri::RunEvent::Reopen { .. } => tray::show_window(app),
            _ => {}
        });
}
