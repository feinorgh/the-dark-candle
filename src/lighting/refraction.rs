//! Chunk-level refraction map — per-voxel Fresnel split factors and
//! refracted light directions for transparent materials (glass, water, ice).
//!
//! # Design
//! For each voxel in a chunk whose material is refractive (n > 1), we trace
//! a sunlight ray using [`dda_march_ray_refractive`] and record:
//! - The net Fresnel transmittance at the air→material boundary (how much
//!   light enters).
//! - The refracted direction inside the medium (useful for rendering caustics
//!   and displaced light patches on surfaces below the material).
//!
//! The system mirrors the pattern of [`ChunkLightMap`] in `light_map.rs`:
//! it runs on [`Changed<Chunk>`] and inserts a [`ChunkRefractionMap`]
//! component onto each updated chunk entity.
//!
//! # SI Units
//! - Refractive index: dimensionless.
//! - Direction: unit vectors in chunk-local grid space.
//! - Transmittance: dimensionless scalar in [0, 1].
//!
//! [`ChunkLightMap`]: super::light_map::ChunkLightMap
//! [`dda_march_ray_refractive`]: crate::world::raycast::dda_march_ray_refractive

use bevy::prelude::*;

use crate::data::MaterialRegistry;
use crate::world::chunk::{Chunk, CHUNK_SIZE};
use crate::world::raycast::dda_march_ray_refractive;
use crate::world::voxel::{MaterialId, Voxel};

const VOLUME: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

/// Per-voxel refraction data stored on a chunk entity.
///
/// For non-refractive voxels (air, stone, etc.) the entry is the identity:
/// transmittance = 1.0, direction = sunlight direction at creation.
#[derive(Component, Debug, Clone)]
pub struct ChunkRefractionMap {
    size: usize,
    /// Per-voxel Fresnel transmittance (fraction of light that enters the
    /// refractive medium from above). 1.0 for non-refractive voxels.
    pub transmittance: Vec<f32>,
    /// Per-voxel refracted light direction `[x, y, z]` (unit vector).
    /// Equals the incident sun direction for non-refractive voxels.
    pub refracted_dir: Vec<[f32; 3]>,
}

impl Default for ChunkRefractionMap {
    fn default() -> Self {
        Self {
            size: CHUNK_SIZE,
            transmittance: vec![1.0; VOLUME],
            refracted_dir: vec![[0.0, -1.0, 0.0]; VOLUME],
        }
    }
}

impl ChunkRefractionMap {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a refraction map for a grid of the given `size` (for testing).
    pub fn with_size(size: usize) -> Self {
        Self {
            size,
            transmittance: vec![1.0; size * size * size],
            refracted_dir: vec![[0.0, -1.0, 0.0]; size * size * size],
        }
    }

    #[inline]
    fn idx(&self, x: usize, y: usize, z: usize) -> usize {
        z * self.size * self.size + y * self.size + x
    }

    /// Fresnel transmittance at voxel `(x, y, z)`.
    pub fn get_transmittance(&self, x: usize, y: usize, z: usize) -> f32 {
        self.transmittance[self.idx(x, y, z)]
    }

    /// Refracted light direction at voxel `(x, y, z)`.
    pub fn get_refracted_dir(&self, x: usize, y: usize, z: usize) -> [f32; 3] {
        self.refracted_dir[self.idx(x, y, z)]
    }
}

// ---------------------------------------------------------------------------
// Pure computation
// ---------------------------------------------------------------------------

/// Build a [`ChunkRefractionMap`] for a chunk voxel grid.
///
/// For each voxel that has a refractive index (transparent material), shoots
/// a ray from the top of the grid downward (in the sun direction) and records
/// the Fresnel transmittance and the refracted direction at the first boundary.
///
/// `sun_dir` is the normalised direction *toward* the sun (so the incident ray
/// direction is `-sun_dir`).
pub fn propagate_refraction<F>(
    voxels: &[Voxel],
    size: usize,
    sun_dir: [f32; 3],
    get_n: F,
) -> ChunkRefractionMap
where
    F: Fn(MaterialId) -> Option<f32>,
{
    let incident_dir = [-sun_dir[0], -sun_dir[1], -sun_dir[2]]; // toward surface
    let mut map = ChunkRefractionMap::with_size(size);

    for z in 0..size {
        for x in 0..size {
            // Start from above the column, shoot downward.
            let origin = [x as f32 + 0.5, size as f32 - 0.01, z as f32 + 0.5];
            let result = dda_march_ray_refractive(
                voxels,
                size,
                origin,
                incident_dir,
                size as f32 * 2.0,
                4,
                &get_n,
            );

            // Walk each segment and write data into the voxels it passes through.
            for seg in &result.segments {
                // Determine which voxel column this segment traverses.
                let sx = seg.origin[0].floor() as i32;
                let sy = seg.origin[1].floor() as i32;
                let sz = seg.origin[2].floor() as i32;
                if sx < 0
                    || sy < 0
                    || sz < 0
                    || sx as usize >= size
                    || sy as usize >= size
                    || sz as usize >= size
                {
                    continue;
                }
                let (ux, uy, uz) = (sx as usize, sy as usize, sz as usize);
                let idx = map.idx(ux, uy, uz);
                map.transmittance[idx] = seg.transmittance;
                map.refracted_dir[idx] = seg.dir;
            }
        }
    }

    map
}

/// Build a [`ChunkRefractionMap`] using the material registry.
pub fn propagate_refraction_from_registry(
    voxels: &[Voxel],
    size: usize,
    sun_dir: [f32; 3],
    registry: &MaterialRegistry,
) -> ChunkRefractionMap {
    propagate_refraction(voxels, size, sun_dir, |mat| {
        registry.get(mat).and_then(|d| d.refractive_index)
    })
}

// ---------------------------------------------------------------------------
// Bevy system
// ---------------------------------------------------------------------------

pub(super) fn update_chunk_refraction_maps(
    mut commands: Commands,
    registry: Option<Res<MaterialRegistry>>,
    sun_dir: Option<Res<super::SunDirection>>,
    chunk_q: Query<(Entity, &Chunk), Changed<Chunk>>,
) {
    let Some(registry) = registry else { return };
    let sun_direction = sun_dir.map_or([0.0_f32, 1.0, 0.0], |sd| sd.0);

    for (entity, chunk) in &chunk_q {
        if !chunk.is_dirty() {
            continue;
        }
        let rm = propagate_refraction_from_registry(
            chunk.voxels(),
            CHUNK_SIZE,
            sun_direction,
            &registry,
        );
        commands.entity(entity).insert(rm);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::voxel::Voxel;

    fn make_grid(size: usize) -> Vec<Voxel> {
        vec![Voxel::default(); size * size * size]
    }

    fn n_fn(mat: MaterialId) -> Option<f32> {
        match mat {
            MaterialId::AIR => Some(1.0),
            MaterialId::GLASS => Some(1.52),
            MaterialId::WATER => Some(1.33),
            _ => None,
        }
    }

    #[test]
    fn default_map_has_unit_transmittance() {
        let map = ChunkRefractionMap::default();
        // All voxels start with transmittance = 1.
        for &t in &map.transmittance {
            assert!((t - 1.0).abs() < 1e-6, "expected 1.0, got {t}");
        }
    }

    #[test]
    fn empty_grid_produces_unit_transmittance() {
        let size = 4;
        let grid = make_grid(size);
        let sun = [0.0_f32, 1.0, 0.0];
        let map = propagate_refraction(&grid, size, sun, n_fn);
        // Pure air — transmittance should remain 1.0 everywhere.
        for &t in &map.transmittance {
            assert!(
                (t - 1.0).abs() < 1e-5,
                "air should not attenuate, got {t}"
            );
        }
    }

    #[test]
    fn glass_slab_reduces_transmittance() {
        // Fill Y=1..3 with glass, shoot sun from above (+Y direction).
        let size = 8;
        let mut grid = make_grid(size);
        for z in 0..size {
            for x in 0..size {
                for y in 2..5 {
                    let idx = z * size * size + y * size + x;
                    grid[idx].material = MaterialId::GLASS;
                    grid[idx].density = 1.0;
                }
            }
        }
        let sun = [0.0_f32, 1.0, 0.0];
        let map = propagate_refraction(&grid, size, sun, n_fn);

        // At least some voxels inside the glass should have transmittance < 1.
        let has_attenuation = map.transmittance.iter().any(|&t| t < 0.999);
        assert!(
            has_attenuation,
            "glass slab should reduce transmittance for some voxels"
        );
    }

    #[test]
    fn refracted_dir_is_unit_length() {
        let size = 8;
        let mut grid = make_grid(size);
        // Single glass voxel in the middle.
        let idx = 4 * size * size + 4 * size + 4;
        grid[idx].material = MaterialId::GLASS;
        grid[idx].density = 1.0;

        let sun = [0.0_f32, 1.0, 0.0];
        let map = propagate_refraction(&grid, size, sun, n_fn);

        for dir in &map.refracted_dir {
            let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
            assert!(
                (len - 1.0).abs() < 0.01,
                "refracted direction must be unit length, got {len}"
            );
        }
    }
}
