// Library root — re-exports all game modules so integration tests can access
// public types without needing a `use` path through the binary.

pub mod audio;
pub mod behavior;
pub mod biology;
pub mod bodies;
pub mod building;
pub mod camera;
pub mod chemistry;
pub mod data;
pub mod diagnostics;
pub mod entities;
pub mod floating_origin;
pub mod game_state;
pub mod gpu;
pub mod hud;
pub mod interaction;
pub mod lighting;
pub mod map;
pub mod persistence;
pub mod physics;
pub mod planet;
pub mod procgen;
pub mod simulation;
pub mod social;
pub mod weather;
pub mod world;
