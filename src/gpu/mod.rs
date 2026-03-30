//! GPU-accelerated rendering via wgpu compute shaders.
//!
//! Provides a headless GPU rendering path that replicates the CPU software
//! renderer (DDA voxel raymarching, sky scattering, volumetric clouds, fog,
//! cloud shadows) using WGSL compute shaders. Expected speedup: 100-1000×
//! over the single-threaded CPU path.

mod context;
pub mod particles;
mod renderer;
pub mod terrain_projection;

pub use context::GpuContext;
pub use renderer::{GpuRenderParams, GpuRenderer};
pub use terrain_projection::{GpuProjectionRenderer, render_animation_gpu, render_projection_gpu};
