// Sky / celestial catalogue subsystem.
//
// Generates a procedural star catalogue (stars, nebulae, galaxies, host galaxy)
// from the system seed and bakes it into an HDR cubemap that the GPU sky-dome
// shader samples each frame.  See `docs/atmosphere-simulation.md` and the
// SKY-004 plan for the design rationale.
//
// Module map:
//   catalogue.rs — data types
//   spectrum.rs  — blackbody / magnitude / flux helpers (next phase)
//   generate.rs  — procedural generators                (next phase)
//   cubemap.rs   — CPU cubemap baker                    (next phase)

pub mod catalogue;
pub mod cubemap;
pub mod generate;
pub mod plugin;
pub mod spectrum;

pub use plugin::{SkyCatalogue, SkyPlugin, StarCubemapHandle};
