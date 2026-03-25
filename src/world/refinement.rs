// Adaptive refinement for voxel octrees.
//
// Analyzes voxel data to determine where subdivision provides meaningful detail
// (surfaces, material boundaries, thermal/pressure gradients) and where uniform
// regions should remain collapsed. This drives the SVO's adaptive resolution:
// detail where it matters, compact storage elsewhere.

#![allow(dead_code)]

use bevy::prelude::*;
use bevy_common_assets::ron::RonAssetPlugin;
use serde::Deserialize;

use super::chunk::{CHUNK_SIZE, Chunk};
use super::octree::OctreeNode;
use super::voxel::Voxel;

/// Configuration for adaptive refinement thresholds.
///
/// Loaded from `assets/data/subdivision_config.ron`. Controls when the octree
/// subdivides a region for higher detail vs. keeping it as a uniform leaf.
#[derive(Deserialize, Asset, TypePath, Debug, Clone, Resource)]
pub struct SubdivisionConfig {
    /// Maximum subdivision depth below the base 1m voxel.
    /// Depth 4 → minimum cell size ~0.0625m (~6cm).
    pub max_depth: u8,

    /// Minimum temperature difference (K) across a cell to trigger subdivision.
    pub thermal_gradient_threshold: f32,

    /// Minimum pressure difference (Pa) across a cell to trigger subdivision.
    pub pressure_gradient_threshold: f32,

    /// Whether to always subdivide at solid/air boundaries (surface crossings).
    pub refine_surfaces: bool,

    /// Whether to subdivide at boundaries between different solid materials.
    pub refine_material_boundaries: bool,

    /// Whether to refine based on damage gradients.
    pub refine_damage_gradients: bool,

    /// Minimum damage difference across a cell to trigger subdivision.
    pub damage_gradient_threshold: f32,
}

impl Default for SubdivisionConfig {
    fn default() -> Self {
        Self {
            max_depth: 4,
            thermal_gradient_threshold: 50.0,
            pressure_gradient_threshold: 5000.0,
            refine_surfaces: true,
            refine_material_boundaries: true,
            refine_damage_gradients: false,
            damage_gradient_threshold: 0.2,
        }
    }
}

pub struct RefinementPlugin;

impl Plugin for RefinementPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(RonAssetPlugin::<SubdivisionConfig>::new(&[
            "subdivision_config.ron",
        ]));
    }
}

/// Reason a voxel cell should be subdivided.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefinementReason {
    /// Cell contains both solid and non-solid voxels (surface crossing).
    SurfaceCrossing,
    /// Cell contains multiple different solid materials.
    MaterialBoundary,
    /// Temperature varies significantly across the cell.
    ThermalGradient,
    /// Pressure varies significantly across the cell.
    PressureGradient,
    /// Damage varies significantly across the cell.
    DamageGradient,
}

/// Result of analyzing a chunk for refinement opportunities.
#[derive(Debug, Default)]
pub struct RefinementAnalysis {
    /// Positions (base resolution) that should be refined, with reasons.
    pub candidates: Vec<(usize, usize, usize, RefinementReason)>,
    /// Number of surface-crossing cells found.
    pub surface_crossings: usize,
    /// Number of material-boundary cells found.
    pub material_boundaries: usize,
    /// Number of thermal-gradient cells found.
    pub thermal_gradients: usize,
    /// Number of pressure-gradient cells found.
    pub pressure_gradients: usize,
}

/// 6-connected face neighbors (dx, dy, dz).
const NEIGHBORS: [(i32, i32, i32); 6] = [
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
];

/// Analyze a chunk's flat voxel data to find cells that would benefit from
/// octree subdivision.
///
/// Returns a `RefinementAnalysis` listing every cell that meets at least one
/// refinement criterion according to the given `config`.
pub fn analyze_chunk(chunk: &Chunk, config: &SubdivisionConfig) -> RefinementAnalysis {
    let voxels = chunk.voxels();
    analyze_voxels(voxels, CHUNK_SIZE, config)
}

/// Analyze raw voxel data for refinement opportunities.
pub fn analyze_voxels(
    voxels: &[Voxel],
    size: usize,
    config: &SubdivisionConfig,
) -> RefinementAnalysis {
    let mut analysis = RefinementAnalysis::default();

    let idx = |x: usize, y: usize, z: usize| -> usize { z * size * size + y * size + x };

    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let v = &voxels[idx(x, y, z)];

                for &(dx, dy, dz) in &NEIGHBORS {
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
                        continue;
                    }

                    let n = &voxels[idx(nx as usize, ny as usize, nz as usize)];

                    // Surface crossing: solid ↔ air boundary.
                    if config.refine_surfaces && v.is_solid() != n.is_solid() {
                        analysis
                            .candidates
                            .push((x, y, z, RefinementReason::SurfaceCrossing));
                        analysis.surface_crossings += 1;
                        break; // One reason per cell is enough to trigger refinement.
                    }

                    // Material boundary: different solid materials meeting.
                    if config.refine_material_boundaries
                        && v.is_solid()
                        && n.is_solid()
                        && v.material != n.material
                    {
                        analysis
                            .candidates
                            .push((x, y, z, RefinementReason::MaterialBoundary));
                        analysis.material_boundaries += 1;
                        break;
                    }

                    // Thermal gradient.
                    if (v.temperature - n.temperature).abs() > config.thermal_gradient_threshold {
                        analysis
                            .candidates
                            .push((x, y, z, RefinementReason::ThermalGradient));
                        analysis.thermal_gradients += 1;
                        break;
                    }

                    // Pressure gradient.
                    if (v.pressure - n.pressure).abs() > config.pressure_gradient_threshold {
                        analysis
                            .candidates
                            .push((x, y, z, RefinementReason::PressureGradient));
                        analysis.pressure_gradients += 1;
                        break;
                    }

                    // Damage gradient.
                    if config.refine_damage_gradients
                        && (v.damage - n.damage).abs() > config.damage_gradient_threshold
                    {
                        analysis
                            .candidates
                            .push((x, y, z, RefinementReason::DamageGradient));
                        break;
                    }
                }
            }
        }
    }

    analysis
}

/// Build a refined octree from flat chunk data, subdividing only at cells
/// identified by the analysis.
///
/// Cells in `analysis.candidates` are kept at leaf resolution (size 1),
/// preventing them from being collapsed into coarser regions. For surface
/// crossings, surrounding voxels within the 2×2×2 neighborhood are also
/// preserved at full resolution to keep interpolation accurate at boundaries.
///
/// This ensures the octree compresses aggressively in uniform interiors
/// while maintaining detail where the analysis detected features.
pub fn build_refined_octree(
    voxels: &[Voxel],
    size: usize,
    analysis: &RefinementAnalysis,
) -> OctreeNode<Voxel> {
    use super::voxel_access::flat_to_octree;

    let mut tree = flat_to_octree(voxels, size);

    // For each candidate, ensure the voxel and its neighbors are stored at
    // leaf-level resolution (target_size = 1). This prevents try_collapse from
    // merging features into a uniform region.
    //
    // Surface crossings and material boundaries pin a 2×2×2 neighborhood so
    // that the Surface Nets algorithm sees all relevant corners. Gradient
    // candidates only pin the cell itself.
    let idx = |x: usize, y: usize, z: usize| -> usize { z * size * size + y * size + x };

    for &(cx, cy, cz, reason) in &analysis.candidates {
        let radius: i32 = match reason {
            RefinementReason::SurfaceCrossing | RefinementReason::MaterialBoundary => 1,
            _ => 0,
        };

        for dz in -radius..=radius {
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let nx = cx as i32 + dx;
                    let ny = cy as i32 + dy;
                    let nz = cz as i32 + dz;

                    if nx < 0
                        || ny < 0
                        || nz < 0
                        || nx >= size as i32
                        || ny >= size as i32
                        || nz >= size as i32
                    {
                        continue;
                    }

                    let (ux, uy, uz) = (nx as usize, ny as usize, nz as usize);
                    let v = voxels[idx(ux, uy, uz)];
                    // Re-set the voxel at target_size=1 to force a leaf node.
                    tree.set(ux, uy, uz, size, 1, v);
                }
            }
        }
    }

    // Final collapse pass: regions that weren't pinned may still merge.
    tree.collapse_recursive();

    tree
}

/// Compute compression statistics for a chunk's octree representation.
#[derive(Debug, Clone)]
pub struct CompressionStats {
    /// Flat array size in bytes.
    pub flat_bytes: usize,
    /// Octree size in bytes (approximate).
    pub octree_bytes: usize,
    /// Number of octree leaf nodes.
    pub leaf_count: usize,
    /// Number of total octree nodes.
    pub node_count: usize,
    /// Maximum tree depth.
    pub max_depth: usize,
    /// Compression ratio (flat / octree). >1.0 means octree is smaller.
    pub compression_ratio: f32,
}

/// Compute compression statistics for converting a chunk to octree.
pub fn compression_stats(chunk: &Chunk) -> CompressionStats {
    let flat_bytes = std::mem::size_of::<Voxel>() * CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;
    let octree = chunk.to_octree();
    let octree_bytes = octree.memory_bytes();
    let leaf_count = octree.leaf_count();
    let node_count = octree.node_count();
    let max_depth = octree.max_depth();
    let compression_ratio = if octree_bytes > 0 {
        flat_bytes as f32 / octree_bytes as f32
    } else {
        f32::INFINITY
    };

    CompressionStats {
        flat_bytes,
        octree_bytes,
        leaf_count,
        node_count,
        max_depth,
        compression_ratio,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::chunk::ChunkCoord;
    use crate::world::voxel::MaterialId;

    fn default_config() -> SubdivisionConfig {
        SubdivisionConfig::default()
    }

    #[test]
    fn empty_chunk_no_refinement() {
        let chunk = Chunk::new_empty(ChunkCoord::new(0, 0, 0));
        let analysis = analyze_chunk(&chunk, &default_config());
        assert_eq!(analysis.candidates.len(), 0);
        assert_eq!(analysis.surface_crossings, 0);
    }

    #[test]
    fn filled_chunk_no_refinement() {
        let chunk = Chunk::new_filled(ChunkCoord::new(0, 0, 0), MaterialId::STONE);
        let analysis = analyze_chunk(&chunk, &default_config());
        assert_eq!(analysis.candidates.len(), 0);
    }

    #[test]
    fn surface_crossing_detected() {
        let mut chunk = Chunk::new_empty(ChunkCoord::new(0, 0, 0));
        // Fill bottom half with stone.
        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                chunk.fill_column(x, z, 16, MaterialId::STONE);
            }
        }
        let analysis = analyze_chunk(&chunk, &default_config());
        assert!(analysis.surface_crossings > 0);
        // Every cell at y=15 (stone) and y=16 (air) along the surface should be a candidate.
        assert!(
            analysis
                .candidates
                .iter()
                .any(|&(_, _, _, r)| r == RefinementReason::SurfaceCrossing)
        );
    }

    #[test]
    fn material_boundary_detected() {
        let mut chunk = Chunk::new_filled(ChunkCoord::new(0, 0, 0), MaterialId::STONE);
        // Place a dirt voxel in the middle of stone.
        chunk.set_material(16, 16, 16, MaterialId::DIRT);
        let analysis = analyze_chunk(&chunk, &default_config());
        assert!(analysis.material_boundaries > 0);
    }

    #[test]
    fn thermal_gradient_detected() {
        let mut chunk = Chunk::new_filled(ChunkCoord::new(0, 0, 0), MaterialId::STONE);
        // Create a hot spot.
        chunk.get_mut(16, 16, 16).temperature = 1000.0;
        let config = default_config();
        let analysis = analyze_chunk(&chunk, &config);
        assert!(analysis.thermal_gradients > 0);
    }

    #[test]
    fn pressure_gradient_detected() {
        let mut chunk = Chunk::new_filled(ChunkCoord::new(0, 0, 0), MaterialId::STONE);
        chunk.get_mut(16, 16, 16).pressure = 200_000.0;
        let config = default_config();
        let analysis = analyze_chunk(&chunk, &config);
        assert!(analysis.pressure_gradients > 0);
    }

    #[test]
    fn config_disables_surface_refinement() {
        let mut chunk = Chunk::new_empty(ChunkCoord::new(0, 0, 0));
        chunk.set_material(0, 0, 0, MaterialId::STONE);
        let mut config = default_config();
        config.refine_surfaces = false;
        config.refine_material_boundaries = false;
        let analysis = analyze_chunk(&chunk, &config);
        assert_eq!(analysis.surface_crossings, 0);
    }

    #[test]
    fn compression_stats_empty_chunk() {
        let chunk = Chunk::new_empty(ChunkCoord::new(0, 0, 0));
        let stats = compression_stats(&chunk);
        assert_eq!(stats.flat_bytes, 16 * 32 * 32 * 32);
        assert!(
            stats.compression_ratio > 1.0,
            "Empty chunk should compress well"
        );
        assert_eq!(stats.leaf_count, 1); // Single leaf.
        assert_eq!(stats.node_count, 1);
        assert_eq!(stats.max_depth, 0);
    }

    #[test]
    fn compression_stats_terrain_chunk() {
        let mut chunk = Chunk::new_empty(ChunkCoord::new(0, 0, 0));
        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                chunk.fill_column(x, z, 16, MaterialId::STONE);
            }
        }
        let stats = compression_stats(&chunk);
        // Half stone / half air should compress well (two large uniform regions).
        assert!(stats.compression_ratio > 1.0);
        assert!(stats.leaf_count < CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE);
    }

    #[test]
    fn build_refined_octree_preserves_data() {
        let mut chunk = Chunk::new_empty(ChunkCoord::new(0, 0, 0));
        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                chunk.fill_column(x, z, 16, MaterialId::STONE);
            }
        }
        let analysis = analyze_chunk(&chunk, &default_config());
        let tree = build_refined_octree(chunk.voxels(), CHUNK_SIZE, &analysis);

        // Verify data integrity.
        for z in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    let expected = chunk.get(x, y, z);
                    let actual = tree.get(x, y, z, CHUNK_SIZE);
                    assert_eq!(expected, actual, "Mismatch at ({x},{y},{z})");
                }
            }
        }
    }
}
