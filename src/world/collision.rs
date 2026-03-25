// Simple voxel-based ground collision.
//
// Provides a height query that finds the topmost solid voxel at a given
// world XZ position by looking up the chunk and scanning the voxel column.
// Used by the camera gravity system to keep the player on the terrain surface.

#![allow(dead_code)]

use super::chunk::{CHUNK_SIZE, ChunkCoord};
use super::chunk_manager::ChunkMap;
use bevy::prelude::*;

use super::chunk::Chunk;

/// Find the Y coordinate of the topmost solid voxel at a given world XZ position.
/// Returns `None` if no chunk is loaded at that position or the column is all air.
pub fn ground_height_at(
    world_x: f32,
    world_z: f32,
    chunk_map: &ChunkMap,
    chunks: &Query<&Chunk>,
) -> Option<f32> {
    // We need to scan multiple vertical chunks. Start from a reasonable height
    // and work downward.
    let cs = CHUNK_SIZE as i32;
    let vx = world_x.floor() as i32;
    let vz = world_z.floor() as i32;
    let local_x = vx.rem_euclid(cs) as usize;
    let local_z = vz.rem_euclid(cs) as usize;

    // Scan from top to bottom across vertical chunks
    for chunk_y in (-4..=8).rev() {
        let coord = ChunkCoord::new(vx.div_euclid(cs), chunk_y, vz.div_euclid(cs));

        let Some(entity) = chunk_map.get(&coord) else {
            continue;
        };

        let Ok(chunk) = chunks.get(entity) else {
            continue;
        };

        // Scan this chunk's column from top to bottom
        for local_y in (0..CHUNK_SIZE).rev() {
            let voxel = chunk.get(local_x, local_y, local_z);
            if voxel.is_solid() {
                let world_y = chunk_y * cs + local_y as i32;
                return Some(world_y as f32 + 1.0); // +1 to stand on top
            }
        }
    }

    None
}

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
}
