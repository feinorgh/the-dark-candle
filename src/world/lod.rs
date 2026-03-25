// Macro-level LOD octree over chunk space.
//
// Groups chunks into a hierarchical octree where distant regions are
// represented at coarser resolution. Each LOD level doubles the spatial
// extent: L0 = 1 chunk (32m), L1 = 2³ chunks (64m), L2 = 4³ chunks (128m).
//
// This module provides the data structure and distance-based LOD selection.
// Mesh simplification and rendering integration are handled separately.

#![allow(dead_code)]

use super::chunk::{CHUNK_SIZE, ChunkCoord};
use super::octree::OctreeNode;
use super::voxel::{MaterialId, Voxel};
use bevy::prelude::*;
use std::collections::HashMap;

/// Hardcoded RGBA fallback colors for materials.
/// Used when no `MaterialRegistry` is loaded.
fn material_color_fallback(mat: MaterialId) -> [f32; 4] {
    match mat.0 {
        0 => [0.0, 0.0, 0.0, 0.0],    // Air (invisible)
        1 => [0.5, 0.5, 0.5, 1.0],    // Stone (gray)
        2 => [0.45, 0.30, 0.15, 1.0], // Dirt (brown)
        3 => [0.2, 0.4, 0.8, 0.8],    // Water (blue, semi-transparent)
        4 => [0.2, 0.6, 0.1, 1.0],    // Grass (green)
        5 => [0.7, 0.55, 0.1, 1.0],   // Iron (yellowish)
        6 => [0.4, 0.25, 0.1, 1.0],   // Wood (dark brown)
        7 => [0.85, 0.8, 0.55, 1.0],  // Sand (tan)
        8 => [0.7, 0.85, 1.0, 0.9],   // Ice (pale blue)
        9 => [0.9, 0.9, 0.95, 0.3],   // Steam (faint white)
        10 => [1.0, 0.3, 0.0, 1.0],   // Lava (orange-red)
        11 => [0.3, 0.3, 0.3, 1.0],   // Ash (dark gray)
        _ => [0.8, 0.0, 0.8, 1.0],    // Unknown (magenta)
    }
}

/// Lookup table mapping `MaterialId` → RGBA color.
///
/// Built from the `MaterialRegistry` at startup. Falls back to hardcoded
/// defaults for materials not found in the registry.
#[derive(Resource, Debug, Clone)]
pub struct MaterialColorMap {
    colors: HashMap<u16, [f32; 4]>,
}

impl Default for MaterialColorMap {
    fn default() -> Self {
        Self::from_defaults()
    }
}

impl MaterialColorMap {
    /// Build a color map from hardcoded fallback colors.
    pub fn from_defaults() -> Self {
        let mut colors = HashMap::new();
        for id in 0..=11u16 {
            colors.insert(id, material_color_fallback(MaterialId(id)));
        }
        Self { colors }
    }

    /// Insert or overwrite the color for a material.
    pub fn insert(&mut self, id: MaterialId, color: [f32; 4]) {
        self.colors.insert(id.0, color);
    }

    /// Insert from an RGB array (alpha = 1.0 for solids, 0.8 for water, 0.3 for steam).
    pub fn insert_rgb(&mut self, id: MaterialId, rgb: [f32; 3], alpha: f32) {
        self.colors.insert(id.0, [rgb[0], rgb[1], rgb[2], alpha]);
    }

    /// Look up the RGBA color for a material, falling back to the hardcoded table.
    pub fn get(&self, id: MaterialId) -> [f32; 4] {
        self.colors
            .get(&id.0)
            .copied()
            .unwrap_or_else(|| material_color_fallback(id))
    }
}

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

/// Configuration for LOD distance thresholds, hysteresis, and screen-space error.
#[derive(Resource, Debug, Clone)]
pub struct LodConfig {
    /// Maximum LOD levels (0 = full resolution only).
    pub max_level: u8,
    /// Distance thresholds (in meters) for each LOD transition.
    /// `thresholds[0]` = distance beyond which L0→L1, etc.
    /// If fewer thresholds than max_level, remaining levels use extrapolation.
    pub thresholds: Vec<f32>,
    /// Hysteresis fraction (0.0–1.0). When transitioning back to a finer LOD,
    /// the distance must drop below `threshold * (1 - hysteresis)` to prevent
    /// thrashing at threshold boundaries.
    pub hysteresis: f32,
    /// Maximum screen-space error in pixels before a coarser LOD is acceptable.
    /// Used by `level_for_screen_error()`. Set to 0.0 to disable screen-space LOD.
    pub screen_error_threshold: f32,
}

impl Default for LodConfig {
    fn default() -> Self {
        Self {
            max_level: 4,
            thresholds: vec![128.0, 256.0, 512.0, 1024.0],
            hysteresis: 0.12,
            screen_error_threshold: 2.0,
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

    /// Determine the LOD level with hysteresis to prevent thrashing.
    ///
    /// When moving to a *coarser* level (farther away), the standard threshold
    /// applies. When returning to a *finer* level (closer), the distance must
    /// drop below `threshold * (1 - hysteresis)` before the transition occurs.
    pub fn level_for_distance_with_hysteresis(&self, distance: f32, current: LodLevel) -> LodLevel {
        let raw = self.level_for_distance(distance);

        if raw.0 >= current.0 {
            // Same or coarser — use standard thresholds (no hysteresis needed).
            return raw;
        }

        // Going finer — require crossing the hysteresis band.
        let mut level = current.0;
        while level > 0 {
            let idx = (level - 1) as usize;
            if idx < self.thresholds.len() {
                let retreat_threshold = self.thresholds[idx] * (1.0 - self.hysteresis);
                if distance < retreat_threshold {
                    level -= 1;
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        LodLevel(level)
    }

    /// Determine LOD level based on projected screen-space error.
    ///
    /// Computes how many pixels a single voxel at each LOD level would occupy
    /// at the given distance. Returns the coarsest level whose voxel projection
    /// still exceeds `screen_error_threshold` pixels.
    ///
    /// # Arguments
    /// * `distance` — Camera distance to chunk center (meters).
    /// * `fov_y` — Vertical field of view (radians).
    /// * `screen_height` — Viewport height in pixels.
    pub fn level_for_screen_error(
        &self,
        distance: f32,
        fov_y: f32,
        screen_height: f32,
    ) -> LodLevel {
        if distance <= 0.0 || self.screen_error_threshold <= 0.0 {
            return LodLevel(0);
        }

        let half_fov_tan = (fov_y * 0.5).tan();
        if half_fov_tan <= 0.0 {
            return LodLevel(0);
        }
        let pixel_scale = screen_height / (2.0 * half_fov_tan);

        // Find the coarsest LOD whose geometric error (in pixels) stays
        // below the threshold. Level 0 has zero error (full resolution).
        let mut level = 0u8;
        for l in 1..=self.max_level {
            let geo_error = (1u32 << l) as f32;
            let screen_err = (geo_error / distance) * pixel_scale;
            if screen_err <= self.screen_error_threshold {
                level = l;
            } else {
                break;
            }
        }
        LodLevel(level)
    }
}

/// Compute the LOD level for a chunk based on its distance from the camera.
pub fn chunk_lod_level(chunk_coord: &ChunkCoord, camera_pos: Vec3, config: &LodConfig) -> LodLevel {
    let chunk_center = chunk_coord.world_center();
    let distance = (chunk_center - camera_pos).length();
    config.level_for_distance(distance)
}

/// Compute the LOD level for a chunk using hysteresis-aware distance check.
pub fn chunk_lod_level_with_hysteresis(
    chunk_coord: &ChunkCoord,
    camera_pos: Vec3,
    current: LodLevel,
    config: &LodConfig,
) -> LodLevel {
    let chunk_center = chunk_coord.world_center();
    let distance = (chunk_center - camera_pos).length();
    config.level_for_distance_with_hysteresis(distance, current)
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
    let mut material_counts: HashMap<MaterialId, usize> = HashMap::new();
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

    let average_color = material_base_color(dominant_material, None);

    LodSummary {
        dominant_material,
        solidity,
        average_color,
        has_surface,
    }
}

/// Simple material color lookup for LOD rendering.
/// In production, this should read from the MaterialData registry.
fn material_base_color_fallback(mat: MaterialId) -> [f32; 3] {
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

/// Resolve a material's base color, preferring the registry when available.
pub fn material_base_color(mat: MaterialId, color_map: Option<&MaterialColorMap>) -> [f32; 3] {
    if let Some(map) = color_map {
        let rgba = map.get(mat);
        [rgba[0], rgba[1], rgba[2]]
    } else {
        material_base_color_fallback(mat)
    }
}

/// Summarize an octree region for LOD representation.
pub fn summarize_octree(octree: &OctreeNode<Voxel>, size: usize) -> LodSummary {
    let mut solid_count = 0usize;
    let mut total_count = 0usize;
    let mut material_counts: HashMap<MaterialId, usize> = HashMap::new();

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
        average_color: material_base_color(dominant_material, None),
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
            ..Default::default()
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

    // --- Hysteresis tests ---

    #[test]
    fn hysteresis_prevents_thrashing_at_boundary() {
        let config = LodConfig::default(); // threshold[0] = 128, hysteresis = 0.12
        let retreat = 128.0 * (1.0 - config.hysteresis); // ~112.6m

        // Moving away: at 130m we transition L0 → L1.
        let level = config.level_for_distance_with_hysteresis(130.0, LodLevel(0));
        assert_eq!(level, LodLevel(1));

        // Moving back: at 120m (above retreat threshold) we stay at L1.
        let level = config.level_for_distance_with_hysteresis(120.0, LodLevel(1));
        assert_eq!(level, LodLevel(1));

        // Moving back further: below retreat threshold → back to L0.
        let level = config.level_for_distance_with_hysteresis(retreat - 1.0, LodLevel(1));
        assert_eq!(level, LodLevel(0));
    }

    #[test]
    fn hysteresis_allows_coarser_transitions_immediately() {
        let config = LodConfig::default();

        // Going coarser never requires hysteresis.
        let level = config.level_for_distance_with_hysteresis(300.0, LodLevel(0));
        assert_eq!(level, LodLevel(2));
    }

    #[test]
    fn hysteresis_same_level_stays() {
        let config = LodConfig::default();
        let level = config.level_for_distance_with_hysteresis(50.0, LodLevel(0));
        assert_eq!(level, LodLevel(0));
    }

    // --- Screen-space error tests ---

    #[test]
    fn screen_error_close_distance_is_level_0() {
        let config = LodConfig::default();
        let fov_y = std::f32::consts::FRAC_PI_2; // 90° FOV
        let level = config.level_for_screen_error(10.0, fov_y, 1080.0);
        assert_eq!(level, LodLevel(0));
    }

    #[test]
    fn screen_error_far_distance_increases_level() {
        let config = LodConfig::default();
        let fov_y = std::f32::consts::FRAC_PI_2;
        let near_level = config.level_for_screen_error(50.0, fov_y, 1080.0);
        let far_level = config.level_for_screen_error(5000.0, fov_y, 1080.0);
        assert!(
            far_level.0 >= near_level.0,
            "Far distance should use same or coarser LOD"
        );
    }

    #[test]
    fn screen_error_higher_resolution_keeps_finer_lod() {
        let config = LodConfig::default();
        let fov_y = std::f32::consts::FRAC_PI_2;
        let level_720 = config.level_for_screen_error(500.0, fov_y, 720.0);
        let level_4k = config.level_for_screen_error(500.0, fov_y, 2160.0);
        assert!(
            level_4k.0 <= level_720.0,
            "Higher screen resolution should use finer or equal LOD"
        );
    }

    #[test]
    fn screen_error_zero_distance_is_level_0() {
        let config = LodConfig::default();
        let level = config.level_for_screen_error(0.0, 1.0, 1080.0);
        assert_eq!(level, LodLevel(0));
    }

    // --- MaterialColorMap tests ---

    #[test]
    fn color_map_defaults_match_fallback() {
        let map = MaterialColorMap::from_defaults();
        assert_eq!(
            map.get(MaterialId::STONE),
            material_color_fallback(MaterialId::STONE)
        );
        assert_eq!(
            map.get(MaterialId::WATER),
            material_color_fallback(MaterialId::WATER)
        );
    }

    #[test]
    fn color_map_custom_overrides_fallback() {
        let mut map = MaterialColorMap::from_defaults();
        let custom = [0.1, 0.2, 0.3, 1.0];
        map.insert(MaterialId::STONE, custom);
        assert_eq!(map.get(MaterialId::STONE), custom);
    }

    #[test]
    fn color_map_unknown_returns_magenta() {
        let map = MaterialColorMap::from_defaults();
        let unknown = map.get(MaterialId(999));
        assert_eq!(unknown, [0.8, 0.0, 0.8, 1.0]);
    }
}
