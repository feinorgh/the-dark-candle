// Macro-level LOD octree over chunk space.
//
// Groups chunks into a hierarchical octree where distant regions are
// represented at coarser resolution. Each LOD level doubles the spatial
// extent: L0 = 1 chunk (32m), L1 = 2³ chunks (64m), L2 = 4³ chunks (128m).
//
// This module provides the data structure and distance-based LOD selection.
// Mesh simplification and rendering integration are handled separately.

#![allow(dead_code)]

use super::chunk::{ChunkCoord, CHUNK_SIZE};
use super::octree::OctreeNode;
use super::voxel::{MaterialId, Voxel};
use bevy::prelude::*;

/// Summary data for a coarsened LOD node.
///
/// When a group of chunks is too far away for full-resolution rendering,
/// this summary provides enough information for a simplified representation.
#[derive(Debug, Clone, PartialEq)]
pub struct LodSummary {
    /// Dominant material in this region (most common non-air material).
    pub dominant_material: MaterialId,
    /// Fraction of the volume that is solid (0.0 = all air, 1.0 = all solid).
    pub solidity: f32,
    /// Average color for simplified rendering (RGB, 0.0–1.0).
    pub average_color: [f32; 3],
    /// Whether this region contains any surface (solid/air boundary).
    pub has_surface: bool,
}

impl Default for LodSummary {
    fn default() -> Self {
        Self {
            dominant_material: MaterialId::AIR,
            solidity: 0.0,
            average_color: [0.0; 3],
            has_surface: false,
        }
    }
}

/// LOD level determines the spatial scale.
///
/// Level 0 = single chunk (32m edge), each increment doubles the edge length.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LodLevel(pub u8);

impl LodLevel {
    /// Edge length of a single node at this LOD level, in voxels.
    pub fn extent_voxels(&self) -> u32 {
        (CHUNK_SIZE as u32) << self.0
    }

    /// Edge length of a single node at this LOD level, in meters (1 voxel = 1m).
    pub fn extent_meters(&self) -> f32 {
        self.extent_voxels() as f32
    }

    /// Number of L0 chunks along one edge at this LOD level.
    pub fn chunks_per_edge(&self) -> u32 {
        1u32 << self.0
    }
}

/// Configuration for LOD distance thresholds.
#[derive(Resource, Debug, Clone)]
pub struct LodConfig {
    /// Maximum LOD levels (0 = full resolution only).
    pub max_level: u8,
    /// Distance thresholds (in meters) for each LOD transition.
    /// `thresholds[0]` = distance beyond which L0→L1, etc.
    /// If fewer thresholds than max_level, remaining levels use extrapolation.
    pub thresholds: Vec<f32>,
}

impl Default for LodConfig {
    fn default() -> Self {
        Self {
            max_level: 4,
            thresholds: vec![128.0, 256.0, 512.0, 1024.0],
        }
    }
}

impl LodConfig {
    /// Determine the LOD level for a chunk at a given distance from the camera.
    pub fn level_for_distance(&self, distance: f32) -> LodLevel {
        for (i, &threshold) in self.thresholds.iter().enumerate() {
            if distance < threshold {
                return LodLevel(i as u8);
            }
        }
        LodLevel(self.max_level)
    }
}

/// Compute the LOD level for a chunk based on its distance from the camera.
pub fn chunk_lod_level(chunk_coord: &ChunkCoord, camera_pos: Vec3, config: &LodConfig) -> LodLevel {
    let chunk_center = chunk_coord.world_center();
    let distance = (chunk_center - camera_pos).length();
    config.level_for_distance(distance)
}

/// Compute LOD summary data from a flat voxel array.
///
/// Scans the voxels to find dominant material, solidity ratio, average color,
/// and surface presence.
pub fn summarize_voxels(voxels: &[Voxel], size: usize) -> LodSummary {
    let total = voxels.len();
    if total == 0 {
        return LodSummary::default();
    }

    let mut solid_count = 0usize;
    let mut material_counts: std::collections::HashMap<MaterialId, usize> =
        std::collections::HashMap::new();
    let mut has_surface = false;

    let idx = |x: usize, y: usize, z: usize| z * size * size + y * size + x;

    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let v = &voxels[idx(x, y, z)];
                if v.is_solid() {
                    solid_count += 1;
                    *material_counts.entry(v.material).or_insert(0) += 1;

                    // Check if this solid voxel has an air neighbor (surface).
                    if !has_surface {
                        for &(dx, dy, dz) in &[
                            (-1i32, 0, 0),
                            (1, 0, 0),
                            (0, -1, 0),
                            (0, 1, 0),
                            (0, 0, -1),
                            (0, 0, 1),
                        ] {
                            let nx = x as i32 + dx;
                            let ny = y as i32 + dy;
                            let nz = z as i32 + dz;
                            if nx < 0
                                || nx >= size as i32
                                || ny < 0
                                || ny >= size as i32
                                || nz < 0
                                || nz >= size as i32
                            {
                                has_surface = true;
                                break;
                            }
                            if voxels[idx(nx as usize, ny as usize, nz as usize)].is_air() {
                                has_surface = true;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    let dominant_material = material_counts
        .into_iter()
        .max_by_key(|&(_, count)| count)
        .map(|(mat, _)| mat)
        .unwrap_or(MaterialId::AIR);

    let solidity = solid_count as f32 / total as f32;

    // Placeholder average color — real colors come from MaterialData registry.
    // For now, use a simple material-based heuristic.
    let average_color = material_base_color(dominant_material);

    LodSummary {
        dominant_material,
        solidity,
        average_color,
        has_surface,
    }
}

/// Simple material color lookup for LOD rendering.
/// In production, this should read from the MaterialData registry.
fn material_base_color(mat: MaterialId) -> [f32; 3] {
    match mat.0 {
        0 => [0.0, 0.0, 0.0],    // Air (invisible)
        1 => [0.5, 0.5, 0.5],    // Stone (gray)
        2 => [0.45, 0.32, 0.18], // Dirt (brown)
        3 => [0.2, 0.4, 0.8],    // Water (blue)
        5 => [0.55, 0.35, 0.15], // Wood (brown)
        6 => [0.85, 0.78, 0.55], // Sand (tan)
        7 => [0.2, 0.55, 0.15],  // Grass (green)
        8 => [0.8, 0.9, 1.0],    // Ice (light blue)
        10 => [0.9, 0.3, 0.0],   // Lava (orange)
        _ => [0.6, 0.6, 0.6],    // Unknown (gray)
    }
}

/// Summarize an octree region for LOD representation.
pub fn summarize_octree(octree: &OctreeNode<Voxel>, size: usize) -> LodSummary {
    let mut solid_count = 0usize;
    let mut total_count = 0usize;
    let mut material_counts: std::collections::HashMap<MaterialId, usize> =
        std::collections::HashMap::new();

    octree.for_each_leaf(0, 0, 0, size, &mut |_, _, _, leaf_size, voxel| {
        let cells = leaf_size * leaf_size * leaf_size;
        total_count += cells;
        if voxel.is_solid() {
            solid_count += cells;
            *material_counts.entry(voxel.material).or_insert(0) += cells;
        }
    });

    let dominant_material = material_counts
        .into_iter()
        .max_by_key(|&(_, count)| count)
        .map(|(mat, _)| mat)
        .unwrap_or(MaterialId::AIR);

    let solidity = if total_count > 0 {
        solid_count as f32 / total_count as f32
    } else {
        0.0
    };

    LodSummary {
        dominant_material,
        solidity,
        average_color: material_base_color(dominant_material),
        has_surface: solidity > 0.0 && solidity < 1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::chunk::Chunk;

    #[test]
    fn lod_level_extent() {
        assert_eq!(LodLevel(0).extent_voxels(), 32);
        assert_eq!(LodLevel(1).extent_voxels(), 64);
        assert_eq!(LodLevel(2).extent_voxels(), 128);
        assert_eq!(LodLevel(3).extent_voxels(), 256);
        assert_eq!(LodLevel(0).chunks_per_edge(), 1);
        assert_eq!(LodLevel(2).chunks_per_edge(), 4);
    }

    #[test]
    fn lod_level_extent_meters() {
        assert_eq!(LodLevel(0).extent_meters(), 32.0);
        assert_eq!(LodLevel(1).extent_meters(), 64.0);
    }

    #[test]
    fn lod_config_distance_thresholds() {
        let config = LodConfig::default();
        assert_eq!(config.level_for_distance(50.0), LodLevel(0));
        assert_eq!(config.level_for_distance(130.0), LodLevel(1));
        assert_eq!(config.level_for_distance(260.0), LodLevel(2));
        assert_eq!(config.level_for_distance(600.0), LodLevel(3));
        assert_eq!(config.level_for_distance(2000.0), LodLevel(4));
    }

    #[test]
    fn lod_config_at_exact_threshold() {
        let config = LodConfig {
            max_level: 2,
            thresholds: vec![100.0, 200.0],
        };
        // At exactly the threshold, moves to next level.
        assert_eq!(config.level_for_distance(100.0), LodLevel(1));
        assert_eq!(config.level_for_distance(99.9), LodLevel(0));
    }

    #[test]
    fn chunk_lod_nearby_is_level_0() {
        let config = LodConfig::default();
        let coord = ChunkCoord::new(0, 0, 0);
        let camera = Vec3::new(16.0, 16.0, 16.0); // Chunk center
        assert_eq!(chunk_lod_level(&coord, camera, &config), LodLevel(0));
    }

    #[test]
    fn chunk_lod_distant_is_higher_level() {
        let config = LodConfig::default();
        let coord = ChunkCoord::new(20, 0, 0); // ~640m away from origin
        let camera = Vec3::ZERO;
        let level = chunk_lod_level(&coord, camera, &config);
        assert!(level.0 > 0);
    }

    #[test]
    fn summarize_empty_voxels() {
        let voxels = vec![Voxel::default(); 8]; // 2×2×2 air
        let summary = summarize_voxels(&voxels, 2);
        assert_eq!(summary.dominant_material, MaterialId::AIR);
        assert_eq!(summary.solidity, 0.0);
        assert!(!summary.has_surface);
    }

    #[test]
    fn summarize_full_solid() {
        let voxels = vec![Voxel::new(MaterialId::STONE); 8];
        let summary = summarize_voxels(&voxels, 2);
        assert_eq!(summary.dominant_material, MaterialId::STONE);
        assert_eq!(summary.solidity, 1.0);
        assert!(summary.has_surface); // Edge voxels touch the boundary.
    }

    #[test]
    fn summarize_mixed_detects_surface() {
        let mut voxels = vec![Voxel::default(); 8]; // 2×2×2
        voxels[0] = Voxel::new(MaterialId::STONE);
        let summary = summarize_voxels(&voxels, 2);
        assert_eq!(summary.solidity, 1.0 / 8.0);
        assert!(summary.has_surface);
        assert_eq!(summary.dominant_material, MaterialId::STONE);
    }

    #[test]
    fn summarize_chunk_via_flat() {
        let mut chunk = Chunk::new_empty(ChunkCoord::new(0, 0, 0));
        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                chunk.fill_column(x, z, 16, MaterialId::STONE);
            }
        }
        let summary = summarize_voxels(chunk.voxels(), CHUNK_SIZE);
        assert!((summary.solidity - 0.5).abs() < 0.01);
        assert!(summary.has_surface);
        assert_eq!(summary.dominant_material, MaterialId::STONE);
    }

    #[test]
    fn summarize_octree_matches_flat() {
        let mut chunk = Chunk::new_empty(ChunkCoord::new(0, 0, 0));
        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                chunk.fill_column(x, z, 16, MaterialId::STONE);
            }
        }
        let flat_summary = summarize_voxels(chunk.voxels(), CHUNK_SIZE);
        let octree = chunk.to_octree();
        let octree_summary = summarize_octree(&octree, CHUNK_SIZE);

        assert_eq!(
            flat_summary.dominant_material,
            octree_summary.dominant_material
        );
        assert!((flat_summary.solidity - octree_summary.solidity).abs() < 0.001);
        assert_eq!(flat_summary.has_surface, octree_summary.has_surface);
    }
}
