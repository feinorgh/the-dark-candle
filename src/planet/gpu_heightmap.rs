//! Equirectangular elevation-map baking for GPU heightmap sampling.
//!
//! Produces three flat `Vec<f32>` buffers that are uploaded to the GPU:
//!
//! 1. **elevation**: IDW tectonic elevation offset (no TerrainNoise) — safe to
//!    bilinearly interpolate.
//! 2. **roughness**: per-pixel noise roughness in \[0, 1\] — safe to bilinearly
//!    interpolate.
//! 3. **ocean_mask**: 1.0 for ocean/deep-ocean biome cells, 0.0 otherwise — baked
//!    with nearest-neighbour intent; the GPU samples it with a nearest-neighbour
//!    function to prevent biome bleeding at coastlines.
//!
//! The GPU shader adds matching `TerrainNoise` (FBM + ridged) at the **exact**
//! column position instead of relying on the baked value, eliminating the
//! aliasing that previously caused floating terrain (TerrainNoise finest octave
//! wavelength ≈ 20 m vs ~100 m pixel size → up to 350 m height error).
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

/// Bake the three GPU heightmap buffers from planetary terrain data.
///
/// Returns `(elevation, roughness, ocean_mask)` as parallel `Vec<f32>` arrays of
/// length `HEIGHTMAP_WIDTH × HEIGHTMAP_HEIGHT`.
///
/// - `elevation[i]`: IDW tectonic elevation offset from `mean_radius` in metres.
///   **No TerrainNoise** — the GPU adds matching noise at the exact column position.
/// - `roughness[i]`: biome/boundary/volcanic noise roughness in \[0, 1\].
/// - `ocean_mask[i]`: 1.0 if the nearest geodesic cell is an ocean/deep-ocean
///   biome, 0.0 otherwise.  The GPU samples this with nearest-neighbour sampling
///   (not bilinear) to prevent coastline bleed.
pub fn bake_elevation_roughness_ocean(
    sampler: &PlanetaryTerrainSampler,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let w = HEIGHTMAP_WIDTH as usize;
    let h = HEIGHTMAP_HEIGHT as usize;

    let mut elevation = vec![0.0f32; w * h];
    let mut roughness = vec![0.0f32; w * h];
    let mut ocean = vec![0.0f32; w * h];

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

            let (idw, r, is_ocean) = sampler.idw_roughness_ocean_at(unit_pos);
            let idx = row * w + col;
            elevation[idx] = idw as f32;
            roughness[idx] = r as f32;
            ocean[idx] = if is_ocean { 1.0 } else { 0.0 };
        }
    }

    (elevation, roughness, ocean)
}

#[cfg(test)]
mod determinism_tests {
    use super::*;
    use crate::world::noise::NoiseConfig;
    use crate::world::planet::PlanetConfig as WorldPlanetConfig;
    use crate::world::terrain::UnifiedTerrainGenerator;
    use std::sync::Arc;

    fn make_sampler(seed: u64) -> Arc<UnifiedTerrainGenerator> {
        let gen_cfg = crate::planet::PlanetConfig {
            seed,
            grid_level: 4,
            ..Default::default()
        };
        let mut pd = crate::planet::PlanetData::new(gen_cfg);
        crate::planet::tectonics::run_tectonics(&mut pd, |_| {});
        let pd = Arc::new(pd);
        let world = WorldPlanetConfig {
            mean_radius: 32_000.0,
            sea_level_radius: 32_000.0,
            height_scale: 4_000.0,
            seed: seed as u32,
            noise: Some(NoiseConfig::default()),
            ..Default::default()
        };
        Arc::new(UnifiedTerrainGenerator::new(pd, world))
    }

    #[test]
    fn bake_is_deterministic_for_fixed_seed() {
        let s1 = make_sampler(1234);
        let s2 = make_sampler(1234);
        let (e1, r1, o1) = bake_elevation_roughness_ocean(&s1.0);
        let (e2, r2, o2) = bake_elevation_roughness_ocean(&s2.0);
        assert_eq!(e1, e2);
        assert_eq!(r1, r2);
        assert_eq!(o1, o2);
    }

    #[test]
    fn bake_differs_for_different_seeds() {
        let s1 = make_sampler(1234);
        let s2 = make_sampler(5678);
        let (e1, _, _) = bake_elevation_roughness_ocean(&s1.0);
        let (e2, _, _) = bake_elevation_roughness_ocean(&s2.0);
        assert_ne!(
            e1, e2,
            "different seeds must produce different elevation maps"
        );
    }
}
