// Octree bridge for AMR fluid simulation.
//
// Follows the established voxel_bridge.rs pattern: flatten the octree to a
// flat array, run the fluid simulation, then rebuild the octree. This
// ensures bit-identical results to flat-array simulation while supporting
// octree-based chunk storage.

use crate::data::{FluidConfig, MaterialRegistry};
use crate::world::octree::OctreeNode;
use crate::world::voxel::Voxel;
use crate::world::voxel_access::{flat_to_octree, octree_to_flat};

use super::step;
use super::types::FluidGrid;

/// Run a complete fluid step on an octree-stored chunk.
///
/// 1. Flattens the octree to a flat voxel array.
/// 2. Builds a FluidGrid from the flat data.
/// 3. Runs the fluid simulation step.
/// 4. Syncs fluid changes back to the flat voxels.
/// 5. Rebuilds the octree from the modified flat data.
///
/// Returns `(updated_octree, voxels_changed)`.
pub fn fluid_step_octree(
    tree: &OctreeNode<Voxel>,
    size: usize,
    registry: Option<&MaterialRegistry>,
    config: &FluidConfig,
    dt: f32,
) -> (OctreeNode<Voxel>, usize) {
    let mut flat = octree_to_flat(tree, size);

    let mut grid = FluidGrid::from_voxels(&flat, size, registry);

    if !grid.has_fluid() {
        return (tree.clone(), 0);
    }

    step::fluid_step(&mut grid, registry, config, dt);

    // Sync fluid grid back to flat voxels using a temporary chunk-like approach.
    // We work directly on the flat array instead of going through Chunk.
    let changed = sync_grid_to_flat(&grid, &mut flat, size);

    let new_tree = flat_to_octree(&flat, size);
    (new_tree, changed)
}

/// Sync FluidGrid state back to a flat voxel array (analogous to sync_to_chunk).
fn sync_grid_to_flat(grid: &FluidGrid, voxels: &mut [Voxel], size: usize) -> usize {
    use super::types::CellTag;

    let mut changed = 0;

    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let idx = z * size * size + y * size + x;
                let cell = grid.get(x, y, z);

                match cell.tag {
                    CellTag::Liquid | CellTag::Surface => {
                        if voxels[idx].material.is_air() {
                            voxels[idx].material = cell.material;
                            voxels[idx].pressure = cell.pressure + 101_325.0;
                            changed += 1;
                        } else {
                            voxels[idx].pressure = cell.pressure + 101_325.0;
                        }
                    }
                    CellTag::Air => {
                        if is_fluid_material(voxels[idx].material) {
                            voxels[idx] = Voxel::default();
                            changed += 1;
                        }
                    }
                    CellTag::Solid => {}
                }
            }
        }
    }

    changed
}

fn is_fluid_material(mat: crate::world::voxel::MaterialId) -> bool {
    matches!(mat.0, 3 | 10)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::voxel::{MaterialId, Voxel};

    #[test]
    fn octree_step_no_fluid_is_noop() {
        let size = 4;
        // All air
        let tree = OctreeNode::new_leaf(Voxel::default());
        let config = FluidConfig::default();

        let (result, changed) = fluid_step_octree(&tree, size, None, &config, 1.0 / 60.0);
        assert_eq!(changed, 0);
        // Should still be a leaf (all uniform air).
        assert!(result.is_leaf());
    }

    #[test]
    fn octree_step_with_water_runs_simulation() {
        let size = 4;
        let mut voxels = vec![Voxel::default(); size * size * size];

        // Fill bottom half with water.
        for z in 0..size {
            for y in 0..size / 2 {
                for x in 0..size {
                    let idx = z * size * size + y * size + x;
                    voxels[idx] = Voxel::new(MaterialId::WATER);
                }
            }
        }

        let tree = flat_to_octree(&voxels, size);
        let config = FluidConfig::default();

        let (result, _changed) = fluid_step_octree(&tree, size, None, &config, 1.0 / 60.0);

        // Verify the result is a valid octree — extract back to flat.
        let result_flat = octree_to_flat(&result, size);
        assert_eq!(result_flat.len(), size * size * size);
    }

    #[test]
    fn octree_roundtrip_preserves_non_fluid_data() {
        let size = 4;
        let mut voxels = vec![Voxel::default(); size * size * size];

        // Stone at (0,0,0)
        voxels[0] = Voxel {
            material: MaterialId::STONE,
            temperature: 500.0,
            ..Default::default()
        };

        let tree = flat_to_octree(&voxels, size);
        let config = FluidConfig::default();

        let (result, _) = fluid_step_octree(&tree, size, None, &config, 1.0 / 60.0);
        let result_flat = octree_to_flat(&result, size);

        assert_eq!(result_flat[0].material, MaterialId::STONE);
        assert!((result_flat[0].temperature - 500.0).abs() < 1e-3);
    }
}
