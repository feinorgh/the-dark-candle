// Bridge layer: octree-compatible wrappers for flat-array physics functions.
//
// Each physics system (fluids, pressure, heat, integrity) currently operates on
// `&[Voxel]` / `&mut [Voxel]` flat arrays. These bridge functions accept an
// `OctreeNode<Voxel>` (or anything convertible to flat), run the original
// algorithm on the flattened data, and optionally rebuild the octree from results.
//
// This is the Phase 1 migration strategy: the algorithms stay unchanged while
// the API surface supports octree callers. A full algorithmic migration to
// native octree traversal is deferred to later phases.

#![allow(dead_code)]

use crate::chemistry;
use crate::data::MaterialRegistry;
use crate::world::octree::OctreeNode;
use crate::world::voxel::Voxel;
use crate::world::voxel_access::{flat_to_octree, octree_to_flat};

/// Simulate one tick of cellular automata fluid flow on an octree.
///
/// Flattens the tree to a `size³` array, runs `simulate_fluids_with_tick`,
/// and rebuilds the octree. Returns `(updated_tree, voxels_moved)`.
pub fn simulate_fluids_octree(
    tree: &OctreeNode<Voxel>,
    size: usize,
    tick: u64,
) -> (OctreeNode<Voxel>, usize) {
    let mut flat = octree_to_flat(tree, size);
    let moved = super::fluids::simulate_fluids_with_tick(&mut flat, size, tick);
    let new_tree = flat_to_octree(&flat, size);
    (new_tree, moved)
}

/// Diffuse gas pressure through an octree volume.
///
/// Flattens, runs `diffuse_pressure_with_rate`, rebuilds.
/// Returns `(updated_tree, max_delta)`.
pub fn diffuse_pressure_octree(
    tree: &OctreeNode<Voxel>,
    size: usize,
    rate: f32,
) -> (OctreeNode<Voxel>, f32) {
    let mut flat = octree_to_flat(tree, size);
    let max_delta = super::pressure::diffuse_pressure_with_rate(&mut flat, size, rate);
    let new_tree = flat_to_octree(&flat, size);
    (new_tree, max_delta)
}

/// Find unsupported solid voxels in an octree volume.
///
/// Read-only: flattens and runs `find_unsupported`. Returns coordinates.
pub fn find_unsupported_octree(
    tree: &OctreeNode<Voxel>,
    size: usize,
) -> Vec<(usize, usize, usize)> {
    let flat = octree_to_flat(tree, size);
    super::integrity::find_unsupported(&flat, size)
}

/// Collapse unsupported voxels in an octree volume.
///
/// Flattens, runs `collapse_unsupported`, rebuilds.
/// Returns `(updated_tree, voxels_moved)`.
pub fn collapse_unsupported_octree(
    tree: &OctreeNode<Voxel>,
    size: usize,
) -> (OctreeNode<Voxel>, usize) {
    let mut flat = octree_to_flat(tree, size);
    let moved = super::integrity::collapse_unsupported(&mut flat, size);
    let new_tree = flat_to_octree(&flat, size);
    (new_tree, moved)
}

/// Diffuse heat through an octree volume using Fourier's law.
///
/// Read-only on the tree: flattens and runs `diffuse_chunk`.
/// Returns new temperature values indexed by flat `z*size²+y*size+x`.
pub fn diffuse_chunk_octree(
    tree: &OctreeNode<Voxel>,
    size: usize,
    dt: f32,
    registry: &MaterialRegistry,
) -> Vec<f32> {
    let flat = octree_to_flat(tree, size);
    chemistry::heat::diffuse_chunk(&flat, size, dt, registry)
}

/// Apply a temperature update buffer to an octree volume.
///
/// Takes the output of `diffuse_chunk_octree` and produces a new octree
/// with updated temperatures.
pub fn apply_temperatures_octree(
    tree: &OctreeNode<Voxel>,
    size: usize,
    new_temps: &[f32],
) -> OctreeNode<Voxel> {
    let mut flat = octree_to_flat(tree, size);
    assert_eq!(flat.len(), new_temps.len());
    for (voxel, &temp) in flat.iter_mut().zip(new_temps.iter()) {
        voxel.temperature = temp;
    }
    flat_to_octree(&flat, size)
}

/// Convenience: diffuse heat and apply in one step.
///
/// Returns the updated octree with new temperatures baked in.
pub fn diffuse_heat_octree(
    tree: &OctreeNode<Voxel>,
    size: usize,
    dt: f32,
    registry: &MaterialRegistry,
) -> OctreeNode<Voxel> {
    let new_temps = diffuse_chunk_octree(tree, size, dt, registry);
    apply_temperatures_octree(tree, size, &new_temps)
}

/// Compute pressure gradient at a position within an octree volume.
///
/// Flattens and delegates to `pressure_gradient`.
pub fn pressure_gradient_octree(
    tree: &OctreeNode<Voxel>,
    size: usize,
    x: usize,
    y: usize,
    z: usize,
) -> (f32, f32, f32) {
    let flat = octree_to_flat(tree, size);
    super::pressure::pressure_gradient(&flat, size, x, y, z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{MaterialData, Phase};
    use crate::world::voxel::MaterialId;

    fn air() -> Voxel {
        Voxel::default()
    }

    fn stone() -> Voxel {
        Voxel::new(MaterialId::STONE)
    }

    fn water() -> Voxel {
        Voxel::new(MaterialId::WATER)
    }

    fn test_registry() -> MaterialRegistry {
        let mut reg = MaterialRegistry::new();
        reg.insert(MaterialData {
            id: 0,
            name: "Air".into(),
            default_phase: Phase::Gas,
            density: 1.225,
            thermal_conductivity: 0.026,
            specific_heat_capacity: 1005.0,
            ..Default::default()
        });
        reg.insert(MaterialData {
            id: 1,
            name: "Stone".into(),
            default_phase: Phase::Solid,
            density: 2700.0,
            thermal_conductivity: 2.5,
            specific_heat_capacity: 790.0,
            ..Default::default()
        });
        reg.insert(MaterialData {
            id: 3,
            name: "Water".into(),
            default_phase: Phase::Liquid,
            density: 1000.0,
            thermal_conductivity: 0.606,
            specific_heat_capacity: 4186.0,
            ..Default::default()
        });
        reg
    }

    // Helper: build a size³ octree from a flat array
    fn make_octree(voxels: &[Voxel], size: usize) -> OctreeNode<Voxel> {
        flat_to_octree(voxels, size)
    }

    fn idx(x: usize, y: usize, z: usize, size: usize) -> usize {
        z * size * size + y * size + x
    }

    // --- Fluid bridge tests ---

    #[test]
    fn fluids_bridge_water_falls() {
        let size = 4;
        let mut flat = vec![air(); size * size * size];
        flat[idx(1, 3, 1, size)] = water();
        let tree = make_octree(&flat, size);

        let (result_tree, moved) = simulate_fluids_octree(&tree, size, 0);
        assert!(moved > 0, "Water should fall");

        let result_flat = octree_to_flat(&result_tree, size);
        assert!(
            result_flat[idx(1, 3, 1, size)].material.is_air(),
            "Original position should be empty"
        );
    }

    #[test]
    fn fluids_bridge_roundtrip_conserves_material() {
        let size = 4;
        let mut flat = vec![air(); size * size * size];
        // Stone floor
        for x in 0..size {
            for z in 0..size {
                flat[idx(x, 0, z, size)] = stone();
            }
        }
        flat[idx(1, 1, 1, size)] = water();
        flat[idx(2, 1, 1, size)] = water();
        let initial_water = flat
            .iter()
            .filter(|v| v.material == MaterialId::WATER)
            .count();

        let tree = make_octree(&flat, size);
        let (result_tree, _) = simulate_fluids_octree(&tree, size, 0);
        let result_flat = octree_to_flat(&result_tree, size);

        let final_water = result_flat
            .iter()
            .filter(|v| v.material == MaterialId::WATER)
            .count();
        assert_eq!(
            initial_water, final_water,
            "Water count should be conserved"
        );
    }

    // --- Pressure bridge tests ---

    #[test]
    fn pressure_bridge_diffuses() {
        let size = 4;
        let mut flat = vec![air(); size * size * size];
        for v in &mut flat {
            v.pressure = 101_325.0;
        }
        flat[idx(2, 2, 2, size)].pressure = 1_013_250.0;

        let tree = make_octree(&flat, size);
        let (result_tree, max_delta) = diffuse_pressure_octree(&tree, size, 0.25);

        assert!(max_delta > 0.0, "Pressure should diffuse");
        let result_flat = octree_to_flat(&result_tree, size);
        assert!(result_flat[idx(2, 2, 2, size)].pressure < 1_013_250.0);
    }

    #[test]
    fn pressure_gradient_bridge_matches_flat() {
        let size = 4;
        let mut flat = vec![air(); size * size * size];
        for v in &mut flat {
            v.pressure = 101_325.0;
        }
        flat[idx(0, 2, 2, size)].pressure = 506_625.0;

        let tree = make_octree(&flat, size);

        let (gx_flat, gy_flat, gz_flat) =
            crate::physics::pressure::pressure_gradient(&flat, size, 1, 2, 2);
        let (gx_tree, gy_tree, gz_tree) = pressure_gradient_octree(&tree, size, 1, 2, 2);

        assert_eq!(gx_flat, gx_tree);
        assert_eq!(gy_flat, gy_tree);
        assert_eq!(gz_flat, gz_tree);
    }

    // --- Integrity bridge tests ---

    #[test]
    fn integrity_bridge_finds_floating_block() {
        let size = 4;
        let mut flat = vec![air(); size * size * size];
        flat[idx(1, 3, 1, size)] = stone();

        let tree = make_octree(&flat, size);
        let unsupported = find_unsupported_octree(&tree, size);

        assert_eq!(unsupported.len(), 1);
        assert_eq!(unsupported[0], (1, 3, 1));
    }

    #[test]
    fn integrity_bridge_collapse_drops_block() {
        let size = 4;
        let mut flat = vec![air(); size * size * size];
        flat[idx(1, 3, 1, size)] = stone();

        let tree = make_octree(&flat, size);
        let (result_tree, moved) = collapse_unsupported_octree(&tree, size);

        assert_eq!(moved, 1);
        let result_flat = octree_to_flat(&result_tree, size);
        assert!(result_flat[idx(1, 3, 1, size)].material.is_air());
        assert_eq!(result_flat[idx(1, 2, 1, size)].material, MaterialId::STONE);
    }

    // --- Heat bridge tests ---

    #[test]
    fn heat_bridge_uniform_stable() {
        let reg = test_registry();
        let size = 4;
        let flat: Vec<Voxel> = (0..size * size * size)
            .map(|_| Voxel::new(MaterialId::STONE))
            .collect();
        let ambient = flat[0].temperature;

        let tree = make_octree(&flat, size);
        let new_temps = diffuse_chunk_octree(&tree, size, 1.0 / 60.0, &reg);

        for &t in &new_temps {
            assert!(
                (t - ambient).abs() < f32::EPSILON,
                "Uniform temperature should be stable"
            );
        }
    }

    #[test]
    fn heat_bridge_hot_spot_spreads() {
        let reg = test_registry();
        let size = 4;
        let mut flat: Vec<Voxel> = (0..size * size * size)
            .map(|_| Voxel::new(MaterialId::STONE))
            .collect();
        let ambient = flat[0].temperature;
        let center = idx(1, 1, 1, size);
        flat[center].temperature = 1000.0;

        let tree = make_octree(&flat, size);
        let result_tree = diffuse_heat_octree(&tree, size, 1.0, &reg);

        let result_flat = octree_to_flat(&result_tree, size);
        assert!(
            result_flat[center].temperature < 1000.0,
            "Hot spot should cool"
        );
        assert!(
            result_flat[center + 1].temperature > ambient,
            "Neighbor should warm"
        );
    }

    #[test]
    fn heat_bridge_matches_flat() {
        let reg = test_registry();
        let size = 4;
        let mut flat: Vec<Voxel> = (0..size * size * size)
            .map(|_| Voxel::new(MaterialId::STONE))
            .collect();
        flat[idx(2, 2, 2, size)].temperature = 800.0;

        let tree = make_octree(&flat, size);
        let temps_from_tree = diffuse_chunk_octree(&tree, size, 1.0, &reg);
        let temps_from_flat = crate::chemistry::heat::diffuse_chunk(&flat, size, 1.0, &reg);

        assert_eq!(temps_from_tree.len(), temps_from_flat.len());
        for (a, b) in temps_from_tree.iter().zip(temps_from_flat.iter()) {
            assert!(
                (a - b).abs() < f32::EPSILON,
                "Bridge results should match flat: {a} vs {b}"
            );
        }
    }
}
