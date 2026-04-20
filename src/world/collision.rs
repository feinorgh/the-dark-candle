// Simple voxel-based ground collision.
//
// Provides height queries for terrain collision detection using the terrain
// generator. Used by the camera gravity system to keep the player on the surface.

#![allow(dead_code)]

use bevy::math::DVec3;

/// Find the surface height using the terrain generator directly (for spawn positioning).
/// This doesn't require chunks to be loaded.
pub fn terrain_spawn_height(
    world_x: f32,
    world_z: f32,
    generator: &super::terrain::TerrainGenerator,
) -> f32 {
    let height = generator.sample_height(world_x as f64, world_z as f64);
    height as f32 + 1.0 // +1 to stand on top
}

/// Find the radial surface height at a world position using the spherical
/// terrain generator directly. Converts the position to (lat, lon) and samples
/// `sample_surface_radius`.
///
/// Returns the radial distance from planet center to the terrain surface,
/// plus 1 m to stand on top. This is the V2-pipeline equivalent of
/// `ground_height_radial` — it works without loaded chunks.
pub fn ground_height_from_terrain_gen(
    world_pos: DVec3,
    terrain_gen: &super::terrain::UnifiedTerrainGenerator,
) -> f32 {
    let (lat, lon) = terrain_gen.planet_config().lat_lon(world_pos);
    let surface_r = terrain_gen.sample_surface_radius_at(lat, lon);
    surface_r as f32 + 1.0 // +1 to stand on top of the voxel
}

#[cfg(test)]
mod tests {
    use super::super::terrain::{TerrainConfig, TerrainGenerator};
    use super::*;

    #[test]
    fn terrain_spawn_height_returns_above_surface() {
        let generator = TerrainGenerator::new(TerrainConfig::default());
        let h = terrain_spawn_height(0.0, 0.0, &generator);
        let sea = TerrainConfig::default().sea_level as f32;
        let scale = TerrainConfig::default().height_scale as f32;
        // Should be somewhere near sea level ± scale
        assert!(
            h > sea - scale * 1.5 && h < sea + scale * 1.5 + 1.0,
            "Spawn height {} outside expected range",
            h
        );
    }

    #[test]
    fn terrain_spawn_height_is_deterministic() {
        let generator = TerrainGenerator::new(TerrainConfig::default());
        let h1 = terrain_spawn_height(50.0, 50.0, &generator);
        let h2 = terrain_spawn_height(50.0, 50.0, &generator);
        assert_eq!(h1, h2);
    }

    #[test]
    fn ground_height_from_terrain_gen_near_surface() {
        use super::super::planet::PlanetConfig;
        use super::super::terrain::{SphericalTerrainGenerator, UnifiedTerrainGenerator};

        let planet = PlanetConfig {
            mean_radius: 32000.0,
            noise: None,
            height_scale: 0.0,
            ..Default::default()
        };

        let tgen =
            UnifiedTerrainGenerator::Spherical(Box::new(SphericalTerrainGenerator::new(planet)));
        // Position on +X axis at the surface
        let pos = DVec3::new(32000.0, 0.0, 0.0);
        let h = ground_height_from_terrain_gen(pos, &tgen);
        // With no noise, surface is exactly at mean_radius. +1 for standing.
        assert!((h - 32001.0).abs() < 1.0, "Expected ~32001, got {h}");
    }

    #[test]
    fn ground_height_from_terrain_gen_is_deterministic() {
        use super::super::planet::PlanetConfig;
        use super::super::terrain::{SphericalTerrainGenerator, UnifiedTerrainGenerator};

        let planet = PlanetConfig {
            mean_radius: 32000.0,
            ..Default::default()
        };
        let tgen =
            UnifiedTerrainGenerator::Spherical(Box::new(SphericalTerrainGenerator::new(planet)));

        let pos = DVec3::new(20000.0, 15000.0, 10000.0);
        let h1 = ground_height_from_terrain_gen(pos, &tgen);
        let h2 = ground_height_from_terrain_gen(pos, &tgen);
        assert_eq!(h1, h2);
    }
}
