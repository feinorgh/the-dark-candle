// Lattice Boltzmann Method (LBM) gas simulation.
//
// D3Q19 lattice with BGK collision and Smagorinsky sub-grid turbulence model.
// Simulates atmospheric gas dynamics: wind, pressure waves, convection, steam
// transport. Operates on the same chunk grid as the AMR liquid solver.

pub mod collision;
pub mod lattice;
pub mod macroscopic;
pub mod octree_bridge;
pub mod plugin;
pub mod step;
pub mod streaming;
pub mod sync;
pub mod types;

pub use plugin::LbmGasPlugin;
