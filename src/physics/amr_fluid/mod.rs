// AMR Navier-Stokes fluid simulation module.
//
// Replaces the cellular automata fluid model with a proper incompressible
// Navier-Stokes solver using operator splitting:
//   1. Advection (semi-Lagrangian)
//   2. Diffusion (implicit Jacobi)
//   3. Pressure projection (Poisson solve)
//
// Fluid state is stored in a separate `FluidGrid` per chunk, keeping the
// 16-byte `Voxel` struct untouched. Bidirectional sync moves materials
// between FluidGrid and Chunk each tick.

pub mod advection;
pub mod diffusion;
pub mod injection;
pub mod octree_bridge;
pub mod plugin;
pub mod pressure;
pub mod step;
pub mod surface;
pub mod sync;
pub mod types;

pub use plugin::AmrFluidPlugin;
