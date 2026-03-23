// Library root — re-exports all game modules so integration tests can access
// public types without needing a `use` path through the binary.

pub mod behavior;
pub mod biology;
pub mod camera;
pub mod chemistry;
pub mod data;
pub mod entities;
pub mod persistence;
pub mod physics;
pub mod procgen;
pub mod social;
pub mod world;
