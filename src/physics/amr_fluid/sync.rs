// Bidirectional synchronization between FluidGrid and Chunk.
//
// The FluidGrid tracks velocity and cell tags; the Chunk owns the voxel
// materials. After each fluid step, `sync_to_chunk` propagates material
// movements (liquid filling air cells, liquid leaving cells) back to the
// chunk. Before a fluid step, `sync_from_chunk` re-tags cells if external
// systems (chemistry, biology) have altered voxels.

use super::types::{CellTag, FluidGrid};
use crate::data::MaterialRegistry;
use crate::world::chunk::Chunk;
use crate::world::voxel::{MaterialId, Voxel};

/// Propagate fluid state from FluidGrid back to the Chunk.
///
/// For each cell:
/// - If the FluidGrid says LIQUID/SURFACE but the chunk voxel is air,
///   set the chunk voxel to the fluid material.
/// - If the FluidGrid says AIR but the chunk voxel is a fluid material,
///   set the chunk voxel to air.
/// - Solid cells and non-fluid materials are never touched.
/// - Pressure from the FluidGrid is written to the voxel's pressure field.
///
/// Returns the number of voxels changed.
pub fn sync_to_chunk(grid: &FluidGrid, chunk: &mut Chunk) -> usize {
    let size = grid.size();
    let mut changed = 0;

    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let cell = grid.get(x, y, z);
                let voxel = chunk.get(x, y, z);

                match cell.tag {
                    CellTag::Liquid | CellTag::Surface => {
                        // Fluid cell: ensure the chunk has the fluid material.
                        if voxel.material.is_air() {
                            chunk.set(
                                x,
                                y,
                                z,
                                Voxel {
                                    material: cell.material,
                                    pressure: cell.pressure + 101_325.0,
                                    ..*voxel
                                },
                            );
                            changed += 1;
                        } else {
                            // Update pressure even if material didn't change.
                            let v = chunk.get_mut(x, y, z);
                            v.pressure = cell.pressure + 101_325.0;
                        }
                    }
                    CellTag::Air => {
                        // Air cell: if the chunk has a fluid material here, clear it.
                        if is_fluid_material(voxel.material) {
                            chunk.set(x, y, z, Voxel::default());
                            changed += 1;
                        }
                    }
                    CellTag::Solid => {
                        // Solid: never touch.
                    }
                }
            }
        }
    }

    changed
}

/// Re-read chunk voxels into the FluidGrid, updating tags for cells that
/// were changed by external systems (e.g., chemistry melted ice → water,
/// or a reaction consumed a liquid).
///
/// Returns the number of cells re-tagged.
pub fn sync_from_chunk(
    chunk: &Chunk,
    grid: &mut FluidGrid,
    registry: Option<&MaterialRegistry>,
) -> usize {
    let size = grid.size();
    let mut retagged = 0;

    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let voxel = chunk.get(x, y, z);
                let cell = grid.get_mut(x, y, z);

                let expected_tag = classify_material(voxel.material, registry);

                if cell.tag != expected_tag {
                    // Material changed externally — re-tag and reset velocity.
                    cell.tag = expected_tag;
                    cell.material = voxel.material;
                    cell.velocity = [0.0; 3];
                    cell.pressure = 0.0;
                    retagged += 1;
                }
            }
        }
    }

    retagged
}

/// Classify a material as Air, Liquid (fluid), or Solid for cell tagging.
fn classify_material(mat: MaterialId, registry: Option<&MaterialRegistry>) -> CellTag {
    if mat.is_air() {
        return CellTag::Air;
    }

    let is_liquid = if let Some(reg) = registry {
        reg.get(mat)
            .map(|m| m.default_phase == crate::data::Phase::Liquid)
            .unwrap_or(false)
    } else {
        is_fluid_material(mat)
    };

    if is_liquid {
        CellTag::Liquid
    } else {
        CellTag::Solid
    }
}

/// Fallback fluid material detection for known IDs.
fn is_fluid_material(mat: MaterialId) -> bool {
    matches!(mat.0, 3 | 10) // water=3, lava=10
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::chunk::ChunkCoord;

    #[test]
    fn sync_to_chunk_fills_air_with_fluid() {
        let mut chunk = Chunk::new_empty(ChunkCoord { x: 0, y: 0, z: 0 });
        let mut grid = FluidGrid::new_empty(32);

        // Mark one cell as liquid water in the grid.
        let cell = grid.get_mut(5, 5, 5);
        cell.tag = CellTag::Liquid;
        cell.material = MaterialId::WATER;
        cell.pressure = 500.0;

        assert!(chunk.get(5, 5, 5).material.is_air());

        let changed = sync_to_chunk(&grid, &mut chunk);

        assert_eq!(changed, 1);
        assert_eq!(chunk.get(5, 5, 5).material, MaterialId::WATER);
        // Pressure = cell.pressure + atmospheric (101325)
        assert!((chunk.get(5, 5, 5).pressure - 101_825.0).abs() < 1.0);
    }

    #[test]
    fn sync_to_chunk_clears_fluid_from_air_cell() {
        let mut chunk = Chunk::new_empty(ChunkCoord { x: 0, y: 0, z: 0 });
        // Place water in the chunk.
        chunk.set_material(5, 5, 5, MaterialId::WATER);

        let grid = FluidGrid::new_empty(32); // All cells are Air.

        let changed = sync_to_chunk(&grid, &mut chunk);

        assert_eq!(changed, 1);
        assert!(chunk.get(5, 5, 5).material.is_air());
    }

    #[test]
    fn sync_to_chunk_leaves_solids_alone() {
        let mut chunk = Chunk::new_empty(ChunkCoord { x: 0, y: 0, z: 0 });
        chunk.set_material(5, 5, 5, MaterialId::STONE);

        let mut grid = FluidGrid::new_empty(32);
        grid.get_mut(5, 5, 5).tag = CellTag::Solid;
        grid.get_mut(5, 5, 5).material = MaterialId::STONE;

        let changed = sync_to_chunk(&grid, &mut chunk);
        assert_eq!(changed, 0);
        assert_eq!(chunk.get(5, 5, 5).material, MaterialId::STONE);
    }

    #[test]
    fn sync_from_chunk_detects_ice_melted_to_water() {
        let mut chunk = Chunk::new_empty(ChunkCoord { x: 0, y: 0, z: 0 });
        let mut grid = FluidGrid::new_empty(32);

        // Initially, the grid thinks (5,5,5) is solid ice.
        grid.get_mut(5, 5, 5).tag = CellTag::Solid;
        grid.get_mut(5, 5, 5).material = MaterialId::ICE;

        // Chemistry system melted ice → water in the chunk.
        chunk.set_material(5, 5, 5, MaterialId::WATER);

        let retagged = sync_from_chunk(&chunk, &mut grid, None);

        assert_eq!(retagged, 1);
        assert_eq!(grid.get(5, 5, 5).tag, CellTag::Liquid);
        assert_eq!(grid.get(5, 5, 5).material, MaterialId::WATER);
    }

    #[test]
    fn sync_from_chunk_no_change_when_consistent() {
        let mut chunk = Chunk::new_empty(ChunkCoord { x: 0, y: 0, z: 0 });
        chunk.set_material(5, 5, 5, MaterialId::WATER);

        let mut grid = FluidGrid::new_empty(32);
        grid.get_mut(5, 5, 5).tag = CellTag::Liquid;
        grid.get_mut(5, 5, 5).material = MaterialId::WATER;

        let retagged = sync_from_chunk(&chunk, &mut grid, None);
        assert_eq!(retagged, 0);
    }
}
