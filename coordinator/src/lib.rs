//! The Vergence analysis coordinator: queues analysis work for Jellyfin and
//! Stash titles, hands 2 s pieces to workers, checks and keeps their results,
//! and serves them to players in 10 s chunks. It runs headless as
//! `vergenced`, or inside the Vergence desktop app. Design: Hypnos
//! `docs/plans/analysis-workers.md`.

pub mod api;
pub mod auth;
pub mod config;
pub mod paired;
pub mod queue;
pub mod source;
pub mod store;
pub mod track;
