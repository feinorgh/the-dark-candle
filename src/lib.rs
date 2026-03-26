// Library root — re-exports all game modules so integration tests can access
// public types without needing a `use` path through the binary.

pub mod behavior;
pub mod biology;
pub mod camera;
pub mod chemistry;
pub mod data;
pub mod diagnostics;
pub mod entities;
pub mod game_state;
pub mod gpu;
pub mod lighting;
pub mod persistence;
pub mod physics;
pub mod procgen;
pub mod simulation;
pub mod social;
pub mod weather;
pub mod world;
