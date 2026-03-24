// Octree bridge for FLIP/PIC particle simulation.
//
// Follows the established LBM octree_bridge pattern: flatten the octree to a
// flat array, run the FLIP simulation, then rebuild the octree. This ensures
// bit-identical results to flat-array simulation while supporting octree-based
// chunk storage.

use crate::data::FluidConfig;
use crate::world::octree::OctreeNode;
use crate::world::voxel::Voxel;
use crate::world::voxel_access::{flat_to_octree, octree_to_flat};

use super::step;
use super::types::{AccumulationGrid, Particle};

/// Run a complete FLIP step on an octree-stored chunk.
///
/// 1. Flattens the octree to a flat voxel array.
/// 2. Runs the FLIP particle simulation step.
/// 3. Rebuilds the octree from the modified flat data.
///
/// Returns `(updated_octree, voxels_changed)`.
pub fn flip_step_octree(
    tree: &OctreeNode<Voxel>,
    size: usize,
    particles: &mut Vec<Particle>,
    accum: &mut AccumulationGrid,
    config: &FluidConfig,
    dt: f32,
    tick: u64,
) -> (OctreeNode<Voxel>, usize) {
    let mut flat = octree_to_flat(tree, size);
    let changed = step::flip_step(particles, &mut flat, accum, config, dt, tick);
    let new_tree = flat_to_octree(&flat, size);
    (new_tree, changed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::voxel::MaterialId;

    #[test]
    fn no_particles_no_changes() {
        let size = 4;
        let tree = OctreeNode::new_leaf(Voxel::default());
        let config = FluidConfig::default();
        let mut particles = Vec::new();
        let mut accum = AccumulationGrid::new(size);

        let (result, changed) =
            flip_step_octree(&tree, size, &mut particles, &mut accum, &config, 0.01, 0);

        assert_eq!(changed, 0);
        assert!(result.is_leaf());
        assert!(particles.is_empty());
    }

    #[test]
    fn octree_roundtrip_preserves_solids() {
        let size = 4;
        let mut voxels = vec![Voxel::default(); size * size * size];

        // Place stone at (0,0,0).
        voxels[0] = Voxel {
            material: MaterialId::STONE,
            temperature: 500.0,
            ..Default::default()
        };

        let tree = flat_to_octree(&voxels, size);
        let config = FluidConfig::default();
        let mut particles = Vec::new();
        let mut accum = AccumulationGrid::new(size);

        let (result, _) =
            flip_step_octree(&tree, size, &mut particles, &mut accum, &config, 0.01, 0);

        let result_flat = octree_to_flat(&result, size);
        assert_eq!(result_flat[0].material, MaterialId::STONE);
        assert!((result_flat[0].temperature - 500.0).abs() < 1e-3);
    }

    #[test]
    fn particles_advect_through_octree() {
        let size = 8;
        let voxels = vec![Voxel::default(); size * size * size];
        let tree = flat_to_octree(&voxels, size);
        let config = FluidConfig::default();
        let mut accum = AccumulationGrid::new(size);
        let mut particles = vec![Particle::new(
            [4.0, 6.0, 4.0],
            [0.0, -1.0, 0.0],
            0.001,
            MaterialId::WATER,
        )];

        let (result, _) =
            flip_step_octree(&tree, size, &mut particles, &mut accum, &config, 0.01, 0);

        let result_flat = octree_to_flat(&result, size);
        assert_eq!(result_flat.len(), size * size * size);
        // Particle should still exist (in-bounds, in air).
        assert!(!particles.is_empty());
    }
}
