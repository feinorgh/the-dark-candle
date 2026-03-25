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

/// Find the surface height along the radial direction from planet center.
///
/// For spherical planets: casts a ray outward from the planet center through
/// the entity position, scanning loaded chunks for the outermost solid voxel
/// along that radial column. Returns the radial distance from the planet center
/// to the top of the outermost solid voxel, or `None` if no solid voxel is found.
pub fn ground_height_radial(
    world_pos: Vec3,
    chunk_map: &ChunkMap,
    chunks: &Query<&Chunk>,
) -> Option<f32> {
    let cs = CHUNK_SIZE as i32;
    let r = world_pos.length();
    if r < 1e-6 {
        return None;
    }
    let dir = world_pos / r;

    // Scan along the radial direction: check several voxels inward from the
    // entity position, looking for the first solid→air transition (the surface).
    // We scan inward from entity, up to 64 voxels deep.
    let scan_depth = 64;
    for step in 0..scan_depth {
        let sample_r = r - step as f32;
        if sample_r < 0.0 {
            break;
        }
        let sample_pos = dir * sample_r;
        let vx = sample_pos.x.floor() as i32;
        let vy = sample_pos.y.floor() as i32;
        let vz = sample_pos.z.floor() as i32;

        let coord = ChunkCoord::new(vx.div_euclid(cs), vy.div_euclid(cs), vz.div_euclid(cs));

        let Some(entity) = chunk_map.get(&coord) else {
            continue;
        };
        let Ok(chunk) = chunks.get(entity) else {
            continue;
        };

        let lx = vx.rem_euclid(cs) as usize;
        let ly = vy.rem_euclid(cs) as usize;
        let lz = vz.rem_euclid(cs) as usize;

        if chunk.get(lx, ly, lz).is_solid() {
            return Some(sample_r + 1.0); // +1 to stand on top
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
