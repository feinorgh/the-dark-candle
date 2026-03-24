// FLIP/PIC particle simulation module.
//
// Hybrid Fluid-Implicit-Particle / Particle-In-Cell method for:
// snowfall, rain, spray, evaporation droplets, ash, sand particles.
//
// Particles carry velocity without grid diffusion, enabling thin sheets,
// droplets, and natural accumulation behavior.

pub mod accumulation;
pub mod advect;
pub mod emission;
pub mod g2p;
pub mod grid_solve;
pub mod octree_bridge;
pub mod p2g;
pub mod plugin;
pub mod step;
pub mod types;

// Re-export once plugin.rs is implemented:
pub use plugin::FlipPicPlugin;
