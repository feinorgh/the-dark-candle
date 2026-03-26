//! Extracts wind velocities from the LBM gas simulation and uploads them
//! to the GPU particle system's wind field buffer.
//!
//! The [`extract_wind_field`] system copies per-cell velocity and density
//! from the nearest LBM chunk into a flat `[f32; 4]` buffer.  The
//! [`upload_wind_to_gpu`] system then pushes that buffer to the GPU when
//! the data is dirty.  All values are SI: velocity in m/s, density in
//! kg/m³.

use bevy::prelude::*;

use crate::gpu::particles::{GpuParticle, ParticleCompute};
use crate::physics::lbm_gas::plugin::LbmState;
use crate::world::chunk::ChunkCoord;

// ─── Resources ─────────────────────────────────────────────────────────────

/// Coarsened wind field buffer for GPU upload.
///
/// Holds a flat array of `[vx, vy, vz, density]` per cell, indexed as
/// `z * grid_size² + y * grid_size + x`.  Velocities are in m/s and
/// density in kg/m³ (SI).
#[derive(Resource)]
pub struct WindFieldUploader {
    /// Flat wind field data: `[vx, vy, vz, density]` per cell.
    wind_data: Vec<[f32; 4]>,
    /// Cubic grid dimension (cells per axis).
    pub grid_size: usize,
    /// How many LBM cells map to one wind grid cell (downsampling factor).
    pub coarsen_factor: usize,
    /// Whether the field has been updated this tick and needs GPU upload.
    pub dirty: bool,
}

impl Default for WindFieldUploader {
    fn default() -> Self {
        let grid_size = 32;
        Self {
            wind_data: vec![[0.0; 4]; grid_size * grid_size * grid_size],
            grid_size,
            coarsen_factor: 1,
            dirty: false,
        }
    }
}

impl WindFieldUploader {
    /// Returns a reference to the flat wind data buffer.
    pub fn wind_data(&self) -> &[[f32; 4]] {
        &self.wind_data
    }
}

/// Bridge resource holding the optional GPU particle compute pipeline.
///
/// `compute` is `None` when no GPU is available (headless / CI builds).
#[derive(Resource, Default)]
pub struct GpuWeatherState {
    /// GPU compute pipeline — `None` if no GPU is present.
    pub compute: Option<ParticleCompute>,
    /// CPU-side particle mirror for read-back.
    pub particles: Vec<GpuParticle>,
    /// Number of currently live particles.
    pub active_count: usize,
}

// ─── Core logic (testable without Bevy) ────────────────────────────────────

/// Pure extraction logic: reads `lbm` and fills `uploader.wind_data`.
fn do_extract_wind(lbm: &LbmState, uploader: &mut WindFieldUploader) {
    let grid_size = uploader.grid_size;
    let volume = grid_size * grid_size * grid_size;

    if uploader.wind_data.len() != volume {
        uploader.wind_data.resize(volume, [0.0; 4]);
    }

    let origin = ChunkCoord::new(0, 0, 0);
    let grid = lbm
        .get(&origin)
        .or_else(|| lbm.iter().next().map(|(_, g)| g));

    let Some(grid) = grid else {
        // No LBM grids available — fill with still air.
        uploader.wind_data.fill([0.0; 4]);
        uploader.dirty = true;
        return;
    };

    let src_size = grid.size();
    for z in 0..grid_size {
        for y in 0..grid_size {
            for x in 0..grid_size {
                let idx = z * grid_size * grid_size + y * grid_size + x;
                if x < src_size && y < src_size && z < src_size {
                    let cell = grid.get(x, y, z);
                    let vel = cell.velocity();
                    let rho = cell.density();
                    uploader.wind_data[idx] = [vel[0], vel[1], vel[2], rho];
                } else {
                    uploader.wind_data[idx] = [0.0; 4];
                }
            }
        }
    }

    uploader.dirty = true;
}

// ─── Bevy systems ──────────────────────────────────────────────────────────

/// Extracts wind velocities from the LBM gas simulation into
/// [`WindFieldUploader`].
///
/// Runs in `FixedUpdate`.  Reads the LBM grid at the origin chunk (or the
/// first available chunk) and copies per-cell velocity and density into the
/// flat wind buffer.
pub fn extract_wind_field(lbm: Res<LbmState>, mut uploader: ResMut<WindFieldUploader>) {
    do_extract_wind(&lbm, &mut uploader);
}

/// Uploads the dirty wind field buffer to the GPU particle compute pipeline.
///
/// Runs in `Update` so it can overlap with rendering.  Skips gracefully
/// when no GPU is available (`GpuWeatherState.compute` is `None`).
pub fn upload_wind_to_gpu(
    mut uploader: ResMut<WindFieldUploader>,
    mut gpu_state: ResMut<GpuWeatherState>,
) {
    if !uploader.dirty {
        return;
    }

    if let Some(compute) = gpu_state.compute.as_mut() {
        compute.upload_wind_field(&uploader.wind_data);
    }

    uploader.dirty = false;
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_wind_field_is_zero() {
        let uploader = WindFieldUploader::default();
        assert!(
            uploader.wind_data.iter().all(|v| *v == [0.0; 4]),
            "fresh WindFieldUploader must contain zero wind",
        );
    }

    #[test]
    fn wind_data_size_matches_grid() {
        let uploader = WindFieldUploader::default();
        let expected = uploader.grid_size.pow(3);
        assert_eq!(uploader.wind_data.len(), expected);
    }

    #[test]
    fn extract_from_empty_lbm_fills_zeros() {
        let lbm = LbmState::default();
        let mut uploader = WindFieldUploader::default();

        // Poke a non-zero value to confirm it gets overwritten.
        uploader.wind_data[0] = [1.0, 2.0, 3.0, 4.0];

        do_extract_wind(&lbm, &mut uploader);

        assert!(
            uploader.wind_data.iter().all(|v| *v == [0.0; 4]),
            "empty LBM state must produce an all-zero wind field",
        );
    }

    #[test]
    fn dirty_flag_set_on_update() {
        let lbm = LbmState::default();
        let mut uploader = WindFieldUploader::default();
        assert!(!uploader.dirty, "dirty must start false");

        do_extract_wind(&lbm, &mut uploader);

        assert!(uploader.dirty, "dirty must be true after extraction");
    }
}
