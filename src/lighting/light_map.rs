// Per-voxel sunlight levels for chunk-based lighting.
//
// Stores RGB transmittance per voxel as a separate component to keep the
// Voxel struct cache-friendly (20 bytes). Sunlight is propagated top-down
// through each column with Beer-Lambert attenuation for transparent media
// and full occlusion for opaque solids.

use bevy::prelude::*;

use crate::data::MaterialRegistry;
use crate::world::chunk::CHUNK_SIZE;
use crate::world::voxel::{MaterialId, Voxel};

const VOLUME: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

/// Per-voxel RGB sunlight levels for a chunk.
///
/// Each voxel stores `[u8; 3]` representing RGB transmittance from the sky
/// above (0 = fully shadowed, 255 = fully lit). The light at a voxel tells
/// how much direct sunlight reaches that position, accounting for all
/// opaque/transparent material above it.
///
/// Indexed as `z * CHUNK_SIZE² + y * CHUNK_SIZE + x` (ZYX standard).
#[derive(Component, Clone, Debug)]
pub struct ChunkLightMap {
    size: usize,
    sun: Vec<[u8; 3]>,
    /// Per-voxel shadow factor: 0.0 = full shadow, 1.0 = full sun.
    shadow: Vec<f32>,
}

impl Default for ChunkLightMap {
    fn default() -> Self {
        Self {
            size: CHUNK_SIZE,
            sun: vec![[255, 255, 255]; VOLUME],
            shadow: vec![1.0; VOLUME],
        }
    }
}

impl ChunkLightMap {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a light map for a grid of the given size (for tests with non-chunk grids).
    pub fn with_size(size: usize) -> Self {
        Self {
            size,
            sun: vec![[255, 255, 255]; size * size * size],
            shadow: vec![1.0; size * size * size],
        }
    }

    #[inline]
    fn idx(&self, x: usize, y: usize, z: usize) -> usize {
        z * self.size * self.size + y * self.size + x
    }

    /// Get normalized RGB sunlight level `[0.0..1.0]` at voxel coordinates.
    pub fn get(&self, x: usize, y: usize, z: usize) -> [f32; 3] {
        let [r, g, b] = self.sun[self.idx(x, y, z)];
        [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0]
    }

    /// Get light level with bounds checking. Returns full light for
    /// out-of-bounds (open sky assumption outside the chunk).
    pub fn get_clamped(&self, x: i32, y: i32, z: i32) -> [f32; 3] {
        if x >= 0
            && y >= 0
            && z >= 0
            && (x as usize) < self.size
            && (y as usize) < self.size
            && (z as usize) < self.size
        {
            self.get(x as usize, y as usize, z as usize)
        } else {
            [1.0, 1.0, 1.0]
        }
    }

    pub fn set(&mut self, x: usize, y: usize, z: usize, rgb: [u8; 3]) {
        let i = self.idx(x, y, z);
        self.sun[i] = rgb;
    }

    fn set_f32(&mut self, x: usize, y: usize, z: usize, rgb: [f32; 3]) {
        let i = self.idx(x, y, z);
        self.sun[i] = [
            (rgb[0] * 255.0).round().clamp(0.0, 255.0) as u8,
            (rgb[1] * 255.0).round().clamp(0.0, 255.0) as u8,
            (rgb[2] * 255.0).round().clamp(0.0, 255.0) as u8,
        ];
    }

    /// Get shadow factor at a voxel position.
    pub fn get_shadow(&self, x: usize, y: usize, z: usize) -> f32 {
        self.shadow[self.idx(x, y, z)]
    }

    /// Set shadow factor at a voxel position.
    pub fn set_shadow(&mut self, x: usize, y: usize, z: usize, factor: f32) {
        let i = self.idx(x, y, z);
        self.shadow[i] = factor.clamp(0.0, 1.0);
    }

    /// Get shadow factor with bounds checking. Returns 1.0 (fully lit) for out-of-bounds.
    pub fn shadow_clamped(&self, x: i32, y: i32, z: i32) -> f32 {
        if x >= 0
            && y >= 0
            && z >= 0
            && (x as usize) < self.size
            && (y as usize) < self.size
            && (z as usize) < self.size
        {
            self.get_shadow(x as usize, y as usize, z as usize)
        } else {
            1.0
        }
    }

    /// Expose the size for external systems.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Reset all shadow factors to 1.0 (fully lit).
    pub fn clear_shadows(&mut self) {
        self.shadow.fill(1.0);
    }
}

/// Propagate sunlight top-down through a flat voxel array.
///
/// For each (x, z) column, starts with full RGB light at the top face and
/// walks downward. Opaque voxels block all light below; transparent voxels
/// attenuate per-channel via Beer-Lambert (T = e^(-α·d), d = 1 m per voxel).
///
/// `absorption_fn` maps a `MaterialId` to per-channel absorption coefficients:
/// - `Some([α_R, α_G, α_B])` for transparent materials (α in m⁻¹)
/// - `None` for fully opaque materials
pub fn propagate_sunlight<F>(voxels: &[Voxel], size: usize, absorption_fn: F) -> ChunkLightMap
where
    F: Fn(MaterialId) -> Option<[f32; 3]>,
{
    let mut map = ChunkLightMap {
        size,
        sun: vec![[0, 0, 0]; size * size * size],
        shadow: vec![1.0; size * size * size],
    };

    for x in 0..size {
        for z in 0..size {
            let mut transmittance = [1.0_f32; 3];

            for y in (0..size).rev() {
                let idx = z * size * size + y * size + x;
                let voxel = &voxels[idx];

                // Store transmittance arriving at this voxel (light from above).
                map.set_f32(x, y, z, transmittance);

                if voxel.is_air() {
                    continue;
                }

                match absorption_fn(voxel.material) {
                    Some(alpha) => {
                        // Beer-Lambert: T *= exp(-α × d), d = 1 m per voxel.
                        for c in 0..3 {
                            transmittance[c] *= (-alpha[c]).exp();
                        }
                    }
                    None => {
                        transmittance = [0.0; 3];
                    }
                }
            }
        }
    }

    map
}

/// Propagate sunlight for a chunk using the material registry for absorption.
pub fn propagate_sunlight_from_registry(
    voxels: &[Voxel],
    size: usize,
    registry: &MaterialRegistry,
) -> ChunkLightMap {
    propagate_sunlight(voxels, size, |mat| {
        registry.get(mat).and_then(|d| d.light_absorption_rgb())
    })
}

/// Apply per-voxel light levels to mesh vertex colors.
///
/// For each vertex, looks up the nearest voxel's light level in the light map
/// and modulates the vertex color (RGB only, alpha preserved).
pub fn apply_light_map(positions: &[[f32; 3]], colors: &mut [[f32; 4]], light_map: &ChunkLightMap) {
    for (i, color) in colors.iter_mut().enumerate() {
        let [px, py, pz] = positions[i];
        let light = light_map.get_clamped(px as i32, py as i32, pz as i32);
        let shadow = light_map.shadow_clamped(px as i32, py as i32, pz as i32);
        color[0] *= light[0] * shadow;
        color[1] *= light[1] * shadow;
        color[2] *= light[2] * shadow;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::voxel::Voxel;

    fn air() -> Voxel {
        Voxel::new(MaterialId::AIR)
    }

    fn stone() -> Voxel {
        Voxel::new(MaterialId::STONE)
    }

    fn water() -> Voxel {
        Voxel::new(MaterialId::WATER)
    }

    fn absorption(mat: MaterialId) -> Option<[f32; 3]> {
        match mat {
            MaterialId::WATER => Some([0.45, 0.07, 0.02]),
            MaterialId::AIR => Some([0.0, 0.0, 0.0]),
            _ => None,
        }
    }

    /// ZYX index helper for tests (avoids `1 * size` identity-op clippy lint).
    fn zyx(x: usize, y: usize, z: usize, size: usize) -> usize {
        z * size * size + y * size + x
    }

    #[test]
    fn empty_chunk_is_fully_lit() {
        let size = 4;
        let voxels = vec![air(); size * size * size];
        let map = propagate_sunlight(&voxels, size, absorption);

        for x in 0..size {
            for y in 0..size {
                for z in 0..size {
                    let [r, g, b] = map.get(x, y, z);
                    assert!(
                        (r - 1.0).abs() < 0.01 && (g - 1.0).abs() < 0.01 && (b - 1.0).abs() < 0.01,
                        "Air-only chunk should be fully lit at ({x},{y},{z}): [{r},{g},{b}]"
                    );
                }
            }
        }
    }

    #[test]
    fn opaque_blocks_all_light_below() {
        let size = 4;
        let mut voxels = vec![air(); size * size * size];

        // Place stone at (1, 3, 1) — top of a column
        voxels[zyx(1, 3, 1, size)] = stone();

        let map = propagate_sunlight(&voxels, size, absorption);

        // Stone itself receives full light (it's at the top face)
        let light_at_stone = map.get(1, 3, 1);
        assert!(
            light_at_stone[0] > 0.99,
            "Stone at top should be lit: {light_at_stone:?}"
        );

        // Below the stone should be dark
        for y in 0..3 {
            let [r, g, b] = map.get(1, y, 1);
            assert!(
                r < 0.01 && g < 0.01 && b < 0.01,
                "Below stone at y={y} should be dark: [{r},{g},{b}]"
            );
        }
    }

    #[test]
    fn water_attenuates_per_channel() {
        let size = 4;
        let mut voxels = vec![air(); size * size * size];

        // Fill column (1, *, 1) with water from y=1..3
        for y in 1..4 {
            voxels[zyx(1, y, 1, size)] = water();
        }

        let map = propagate_sunlight(&voxels, size, absorption);

        // Top water (y=3) gets full light
        let top = map.get(1, 3, 1);
        assert!(top[0] > 0.99, "Top water should be fully lit: {top:?}");

        // Below 3 meters of water: T = exp(-α * 3)
        let bottom = map.get(1, 0, 1);
        let expected_r = (-0.45 * 3.0_f32).exp();
        let expected_g = (-0.07 * 3.0_f32).exp();
        let expected_b = (-0.02 * 3.0_f32).exp();

        assert!(
            (bottom[0] - expected_r).abs() < 0.03,
            "Red should be ~{expected_r:.3}, got {:.3}",
            bottom[0]
        );
        assert!(
            (bottom[1] - expected_g).abs() < 0.03,
            "Green should be ~{expected_g:.3}, got {:.3}",
            bottom[1]
        );
        assert!(
            (bottom[2] - expected_b).abs() < 0.03,
            "Blue should be ~{expected_b:.3}, got {:.3}",
            bottom[2]
        );

        // Red attenuates most, blue least
        assert!(
            bottom[2] > bottom[1] && bottom[1] > bottom[0],
            "Water should attenuate R > G > B: {bottom:?}"
        );
    }

    #[test]
    fn stone_above_water_blocks_everything() {
        let size = 4;
        let mut voxels = vec![air(); size * size * size];

        // Stone at y=3, water at y=2, air at y=1,0
        voxels[zyx(1, 3, 1, size)] = stone();
        voxels[zyx(1, 2, 1, size)] = water();

        let map = propagate_sunlight(&voxels, size, absorption);

        // Everything below stone should be dark
        for y in 0..3 {
            let [r, g, b] = map.get(1, y, 1);
            assert!(
                r < 0.01 && g < 0.01 && b < 0.01,
                "Below stone at y={y} should be dark: [{r},{g},{b}]"
            );
        }
    }

    #[test]
    fn adjacent_columns_are_independent() {
        let size = 4;
        let mut voxels = vec![air(); size * size * size];

        // Stone only in column (0, *, 0) at y=3
        voxels[zyx(0, 3, 0, size)] = stone();

        let map = propagate_sunlight(&voxels, size, absorption);

        // Column (1, *, 1) should be fully lit
        for y in 0..size {
            let [r, g, b] = map.get(1, y, 1);
            assert!(
                r > 0.99,
                "Unblocked column at y={y} should be lit: [{r},{g},{b}]"
            );
        }

        // Column (0, *, 0) below stone should be dark
        let [r, g, b] = map.get(0, 0, 0);
        assert!(r < 0.01, "Below stone should be dark: [{r},{g},{b}]");
    }

    #[test]
    fn get_clamped_returns_full_light_outside() {
        let map = ChunkLightMap::default();
        assert_eq!(map.get_clamped(-1, 0, 0), [1.0, 1.0, 1.0]);
        assert_eq!(map.get_clamped(0, -1, 0), [1.0, 1.0, 1.0]);
        assert_eq!(map.get_clamped(0, 0, CHUNK_SIZE as i32), [1.0, 1.0, 1.0]);
    }

    #[test]
    fn apply_light_modulates_colors() {
        let mut map = ChunkLightMap::new();
        // Half light at (0, 0, 0)
        map.set(0, 0, 0, [128, 255, 0]);

        let positions = [[0.0, 0.0, 0.0]];
        let mut colors = [[1.0_f32, 1.0, 1.0, 1.0]];

        apply_light_map(&positions, &mut colors, &map);

        // 128/255 ≈ 0.502
        assert!(
            (colors[0][0] - 128.0 / 255.0).abs() < 0.01,
            "Red should be halved: {}",
            colors[0][0]
        );
        assert!(
            (colors[0][1] - 1.0).abs() < 0.01,
            "Green should be full: {}",
            colors[0][1]
        );
        assert!(colors[0][2] < 0.01, "Blue should be zero: {}", colors[0][2]);
        // Alpha preserved
        assert!(
            (colors[0][3] - 1.0).abs() < 0.001,
            "Alpha should be preserved"
        );
    }

    #[test]
    fn default_light_map_is_fully_lit() {
        let map = ChunkLightMap::default();
        let [r, g, b] = map.get(0, 0, 0);
        assert!((r - 1.0).abs() < 0.004 && (g - 1.0).abs() < 0.004 && (b - 1.0).abs() < 0.004);
    }

    #[test]
    fn deep_water_strongly_attenuates_red() {
        let size = 8;
        let mut voxels = vec![air(); size * size * size];

        // Fill entire column (2, *, 2) with water
        for y in 0..size {
            voxels[zyx(2, y, 2, size)] = water();
        }

        let map = propagate_sunlight(&voxels, size, absorption);

        // At depth 7 (bottom, 7 meters of water above): T_R = exp(-0.45*7) ≈ 0.043
        let bottom = map.get(2, 0, 2);
        assert!(
            bottom[0] < 0.06,
            "Red through 7m water should be very dim: {}",
            bottom[0]
        );
        // Blue barely attenuated: T_B = exp(-0.02*7) ≈ 0.869
        assert!(
            bottom[2] > 0.8,
            "Blue through 7m water should be bright: {}",
            bottom[2]
        );
    }

    #[test]
    fn shadow_defaults_to_fully_lit() {
        let map = ChunkLightMap::new();
        assert!((map.get_shadow(0, 0, 0) - 1.0).abs() < 0.001);
    }

    #[test]
    fn shadow_set_and_get() {
        let mut map = ChunkLightMap::with_size(4);
        map.set_shadow(1, 2, 1, 0.3);
        assert!((map.get_shadow(1, 2, 1) - 0.3).abs() < 0.001);
    }

    #[test]
    fn shadow_clamped_returns_one_outside() {
        let map = ChunkLightMap::new();
        assert!((map.shadow_clamped(-1, 0, 0) - 1.0).abs() < 0.001);
        assert!((map.shadow_clamped(0, 0, CHUNK_SIZE as i32) - 1.0).abs() < 0.001);
    }

    #[test]
    fn shadow_clamps_value_range() {
        let mut map = ChunkLightMap::with_size(4);
        map.set_shadow(0, 0, 0, -0.5);
        assert!(map.get_shadow(0, 0, 0) >= 0.0);
        map.set_shadow(0, 0, 0, 1.5);
        assert!(map.get_shadow(0, 0, 0) <= 1.0);
    }

    #[test]
    fn apply_light_with_shadow_modulates_correctly() {
        let mut map = ChunkLightMap::with_size(4);
        map.set_shadow(0, 0, 0, 0.5);

        let positions = [[0.0, 0.0, 0.0]];
        let mut colors = [[1.0_f32, 1.0, 1.0, 1.0]];
        apply_light_map(&positions, &mut colors, &map);

        // Expected: sun(1.0) * shadow(0.5) = 0.5
        assert!((colors[0][0] - 0.5).abs() < 0.01);
        assert!(
            (colors[0][3] - 1.0).abs() < 0.001,
            "Alpha should be preserved"
        );
    }

    #[test]
    fn clear_shadows_resets_all() {
        let mut map = ChunkLightMap::with_size(4);
        map.set_shadow(1, 1, 1, 0.0);
        assert!(map.get_shadow(1, 1, 1) < 0.01);
        map.clear_shadows();
        assert!((map.get_shadow(1, 1, 1) - 1.0).abs() < 0.001);
    }
}
