//! Equirectangular elevation-map baking for GPU heightmap sampling.
//!
//! Produces a flat `Vec<f32>` that can be uploaded to the GPU as a storage
//! buffer.  Each element stores the surface elevation **offset from
//! `mean_radius`** in metres, computed via the same `PlanetaryTerrainSampler`
//! path used by the CPU terrain generator (IDW interpolation + detail noise +
//! ocean-biome clamp).
//!
//! ## UV conventions (matching WGSL `lat_lon()` and `sample_heightmap()`)
//!
//! - Column 0 → longitude = −π  (antimeridian west)
//! - Row    0 → latitude  = +π/2 (north pole)

use std::f64::consts::PI;

use bevy::math::DVec3;

use crate::world::planetary_sampler::PlanetaryTerrainSampler;

/// Width of the baked heightmap in pixels (longitude axis).
pub const HEIGHTMAP_WIDTH: u32 = 2048;

/// Height of the baked heightmap in pixels (latitude axis).
pub const HEIGHTMAP_HEIGHT: u32 = 1024;

/// Bake a `HEIGHTMAP_WIDTH × HEIGHTMAP_HEIGHT` equirectangular elevation map.
///
/// Returns a row-major `Vec<f32>` of length `HEIGHTMAP_WIDTH * HEIGHTMAP_HEIGHT`.
/// Each element is the surface elevation offset from `mean_radius` in metres.
///
/// The bake calls `PlanetaryTerrainSampler::surface_radius_at()` once per
/// pixel so that the GPU heightmap is bit-identical (at f32 precision) to the
/// CPU terrain path, including IDW tectonic elevation, per-biome detail noise,
/// and the ocean-biome ceiling clamp that prevents phantom walls.
pub fn bake_elevation_map(sampler: &PlanetaryTerrainSampler) -> Vec<f32> {
    let w = HEIGHTMAP_WIDTH as usize;
    let h = HEIGHTMAP_HEIGHT as usize;
    let mean_r = sampler.planet_config.mean_radius;

    let mut buf = vec![0.0f32; w * h];

    for row in 0..h {
        // Row 0 = north pole (lat = +π/2), row H-1 = south pole (lat = -π/2).
        let lat = (0.5 - (row as f64 + 0.5) / h as f64) * PI;
        let cos_lat = lat.cos();
        let sin_lat = lat.sin();

        for col in 0..w {
            // Col 0 = antimeridian (lon = -π), col W-1 just before lon = +π.
            let lon = ((col as f64 + 0.5) / w as f64 - 0.5) * 2.0 * PI;

            // Convert lat/lon to unit-sphere position.
            // Matches GPU: lat = asin(y), lon = atan2(x, z).
            let unit_pos = DVec3::new(lon.sin() * cos_lat, sin_lat, lon.cos() * cos_lat);

            let (surface_r, _cell) = sampler.surface_radius_at(unit_pos);
            buf[row * w + col] = (surface_r - mean_r) as f32;
        }
    }

    buf
}
