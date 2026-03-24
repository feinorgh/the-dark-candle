// Octree bridge for LBM gas simulation.
//
// Follows the established voxel_bridge.rs pattern: flatten the octree to a
// flat array, run the LBM simulation, then rebuild the octree. This ensures
// bit-identical results to flat-array simulation while supporting octree-based
// chunk storage.

use crate::data::FluidConfig;
use crate::world::octree::OctreeNode;
use crate::world::voxel::{MaterialId, Voxel};
use crate::world::voxel_access::{flat_to_octree, octree_to_flat};

use super::step;
use super::types::LbmGrid;

/// Run a complete LBM step on an octree-stored chunk.
///
/// 1. Flattens the octree to a flat voxel array.
/// 2. Builds an LbmGrid from the flat data.
/// 3. Runs the LBM simulation step(s).
/// 4. Syncs gas changes back to the flat voxels.
/// 5. Rebuilds the octree from the modified flat data.
///
/// Returns `(updated_octree, voxels_changed)`.
pub fn lbm_step_octree(
    tree: &OctreeNode<Voxel>,
    size: usize,
    config: &FluidConfig,
    gravity_lattice: [f32; 3],
    rho_ambient: f32,
) -> (OctreeNode<Voxel>, usize) {
    let mut flat = octree_to_flat(tree, size);

    let mut grid = LbmGrid::from_voxels(&flat, size, None);

    if !grid.has_gas() {
        return (tree.clone(), 0);
    }

    // Only run if there's active gas (not just ambient air)
    if !has_active_gas(&flat) {
        return (tree.clone(), 0);
    }

    let n_steps = config.lbm_steps_per_tick.max(1);
    step::lbm_step_n(&mut grid, config, gravity_lattice, rho_ambient, n_steps);

    let changed = sync_grid_to_flat(&grid, &mut flat, size);

    let new_tree = flat_to_octree(&flat, size);
    (new_tree, changed)
}

/// Check if the flat voxel array contains any active gas (steam, etc.).
fn has_active_gas(voxels: &[Voxel]) -> bool {
    voxels.iter().any(|v| v.material == MaterialId::STEAM)
}

/// Sync LbmGrid state back to a flat voxel array.
fn sync_grid_to_flat(grid: &LbmGrid, voxels: &mut [Voxel], size: usize) -> usize {
    let mut changed = 0;

    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let idx = z * size * size + y * size + x;
                let cell = grid.get(x, y, z);

                if !cell.is_gas() {
                    continue;
                }

                let rho = cell.density();
                let new_pressure = rho * 101_325.0;

                if (new_pressure - voxels[idx].pressure).abs() > 1.0 {
                    voxels[idx].pressure = new_pressure;
                    changed += 1;
                }

                // Steam transport: LBM cell carries steam into air voxel
                if cell.material == MaterialId::STEAM
                    && voxels[idx].material == MaterialId::AIR
                    && rho > 1.05
                {
                    voxels[idx].material = MaterialId::STEAM;
                    voxels[idx].pressure = new_pressure;
                    changed += 1;
                }
            }
        }
    }

    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn octree_step_no_active_gas_is_noop() {
        let size = 4;
        // All air — no active gas (steam)
        let tree = OctreeNode::new_leaf(Voxel::default());
        let config = FluidConfig::default();

        let (result, changed) = lbm_step_octree(&tree, size, &config, [0.0; 3], 1.0);
        assert_eq!(changed, 0);
        assert!(result.is_leaf());
    }

    #[test]
    fn octree_step_with_steam_runs_simulation() {
        let size = 4;
        let mut voxels = vec![Voxel::default(); size * size * size];

        // Place steam in center
        let idx = 2 * size * size + 2 * size + 2;
        voxels[idx] = Voxel::new(MaterialId::STEAM);

        let tree = flat_to_octree(&voxels, size);
        let config = FluidConfig::default();

        let (result, _changed) = lbm_step_octree(&tree, size, &config, [0.0; 3], 1.0);

        // Should produce a valid octree
        let result_flat = octree_to_flat(&result, size);
        assert_eq!(result_flat.len(), size * size * size);
    }

    #[test]
    fn octree_preserves_solid_voxels() {
        let size = 4;
        let mut voxels = vec![Voxel::default(); size * size * size];

        // Stone at (0,0,0)
        voxels[0] = Voxel {
            material: MaterialId::STONE,
            temperature: 500.0,
            ..Default::default()
        };
        // Steam to trigger simulation
        let idx = 2 * size * size + 2 * size + 2;
        voxels[idx] = Voxel::new(MaterialId::STEAM);

        let tree = flat_to_octree(&voxels, size);
        let config = FluidConfig::default();

        let (result, _) = lbm_step_octree(&tree, size, &config, [0.0; 3], 1.0);
        let result_flat = octree_to_flat(&result, size);

        assert_eq!(result_flat[0].material, MaterialId::STONE);
        assert!((result_flat[0].temperature - 500.0).abs() < 1e-3);
    }
}
