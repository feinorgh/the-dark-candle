// Bidirectional sync between LbmGrid and Chunk, plus phase transition coupling.
//
// After each LBM step, `sync_to_chunk` writes gas pressure back to voxels.
// Before an LBM step, `sync_from_chunk` re-classifies cells when external
// systems (chemistry state transitions) have changed voxel materials.

use super::types::{GasCellTag, LbmCell, LbmGrid};
use crate::data::MaterialRegistry;
use crate::world::chunk::Chunk;
use crate::world::voxel::{MaterialId, Voxel};

/// Write LBM gas state back to the Chunk.
///
/// For each gas cell, updates the voxel's pressure field based on the
/// local LBM density. Material changes (steam spreading into air cells)
/// are also propagated.
///
/// Returns the number of voxels changed.
pub fn sync_to_chunk(grid: &LbmGrid, chunk: &mut Chunk) -> usize {
    let size = grid.size();
    let mut changed = 0;

    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let cell = grid.get(x, y, z);
                if !cell.is_gas() {
                    continue;
                }

                let voxel = *chunk.get(x, y, z);
                let rho = cell.density();

                // Convert lattice density to physical pressure
                let new_pressure = rho * 101_325.0;

                // Update pressure if it changed significantly
                if (new_pressure - voxel.pressure).abs() > 1.0 {
                    chunk.set(
                        x,
                        y,
                        z,
                        Voxel {
                            pressure: new_pressure,
                            ..voxel
                        },
                    );
                    changed += 1;
                }

                // If the LBM cell carries steam but the voxel is air,
                // convert the voxel to steam (steam transport)
                if cell.material == MaterialId::STEAM
                    && voxel.material == MaterialId::AIR
                    && rho > 1.05
                {
                    chunk.set(
                        x,
                        y,
                        z,
                        Voxel {
                            material: MaterialId::STEAM,
                            pressure: new_pressure,
                            ..voxel
                        },
                    );
                    changed += 1;
                }
            }
        }
    }

    changed
}

/// Re-classify LBM cells from chunk voxel state.
///
/// Called when external systems (chemistry, state transitions) have changed
/// voxel materials. For example, when water boils → steam, the corresponding
/// cell must transition from Liquid wall to Gas.
pub fn sync_from_chunk(
    chunk: &Chunk,
    grid: &mut LbmGrid,
    registry: Option<&MaterialRegistry>,
) -> usize {
    let size = grid.size();
    let mut changed = 0;

    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let voxel = chunk.get(x, y, z);
                let cell = grid.get(x, y, z);
                let mat = voxel.material;

                let should_be_gas = is_gas_material(mat, registry);
                let should_be_liquid = is_liquid_material(mat, registry);

                match cell.tag {
                    GasCellTag::Gas => {
                        if !should_be_gas {
                            // Gas cell became solid or liquid (condensation, freezing)
                            let new_cell = if should_be_liquid {
                                LbmCell::new_liquid(mat)
                            } else {
                                LbmCell::new_solid(mat)
                            };
                            *grid.get_mut(x, y, z) = new_cell;
                            changed += 1;
                        } else if cell.material != mat {
                            // Still gas but material changed (e.g., air → steam)
                            grid.get_mut(x, y, z).material = mat;
                            changed += 1;
                        }
                    }
                    GasCellTag::Solid | GasCellTag::Liquid => {
                        if should_be_gas {
                            // Wall cell became gas (evaporation: water → steam)
                            let rho = pressure_to_lattice_density(voxel.pressure);
                            *grid.get_mut(x, y, z) = LbmCell::new_gas(mat, rho);
                            changed += 1;
                        } else if should_be_liquid && cell.tag == GasCellTag::Solid {
                            *grid.get_mut(x, y, z) = LbmCell::new_liquid(mat);
                            changed += 1;
                        } else if !should_be_liquid
                            && !should_be_gas
                            && cell.tag == GasCellTag::Liquid
                        {
                            *grid.get_mut(x, y, z) = LbmCell::new_solid(mat);
                            changed += 1;
                        }
                    }
                }
            }
        }
    }

    changed
}

/// Inject steam mass into an LBM cell (called when water boils → steam).
///
/// Sets the cell to gas equilibrium at steam density.
pub fn inject_steam(grid: &mut LbmGrid, x: usize, y: usize, z: usize, rho: f32) {
    *grid.get_mut(x, y, z) = LbmCell::new_gas(MaterialId::STEAM, rho);
}

/// Remove gas mass from an LBM cell (called when steam condenses → water).
///
/// Sets the cell to a liquid wall.
pub fn remove_gas(grid: &mut LbmGrid, x: usize, y: usize, z: usize) {
    *grid.get_mut(x, y, z) = LbmCell::new_liquid(MaterialId::WATER);
}

fn is_gas_material(mat: MaterialId, registry: Option<&MaterialRegistry>) -> bool {
    if let Some(reg) = registry {
        if let Some(data) = reg.get(mat) {
            return data.default_phase == crate::data::Phase::Gas;
        }
    }
    mat == MaterialId::AIR || mat == MaterialId::STEAM
}

fn is_liquid_material(mat: MaterialId, registry: Option<&MaterialRegistry>) -> bool {
    if let Some(reg) = registry {
        if let Some(data) = reg.get(mat) {
            return data.default_phase == crate::data::Phase::Liquid;
        }
    }
    mat == MaterialId::WATER || mat == MaterialId::LAVA
}

fn pressure_to_lattice_density(pressure_pa: f32) -> f32 {
    const ATMOSPHERIC: f32 = 101_325.0;
    (pressure_pa / ATMOSPHERIC).max(0.01)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::chunk::{Chunk, ChunkCoord};

    #[test]
    fn sync_to_chunk_updates_pressure() {
        let _size = 4;
        let coord = ChunkCoord { x: 0, y: 0, z: 0 };
        let mut chunk = Chunk::new_filled(coord, MaterialId::AIR);
        let mut grid = LbmGrid::from_chunk(&chunk, None);

        // Increase density at one cell (higher pressure)
        *grid.get_mut(1, 1, 1) = LbmCell::new_gas(MaterialId::AIR, 1.5);

        let changed = sync_to_chunk(&grid, &mut chunk);
        assert!(changed > 0, "Should have changed at least one voxel");

        // Check pressure was updated
        let p = chunk.get(1, 1, 1).pressure;
        let expected = 1.5 * 101_325.0;
        assert!(
            (p - expected).abs() < 100.0,
            "Pressure = {p}, expected ~{expected}"
        );
    }

    #[test]
    fn sync_from_chunk_detects_evaporation() {
        let _size = 4;
        let coord = ChunkCoord { x: 0, y: 0, z: 0 };
        let mut chunk = Chunk::new_filled(coord, MaterialId::AIR);

        // Place a water voxel that will be treated as liquid wall
        chunk.set(1, 1, 1, Voxel::new(MaterialId::WATER));
        let mut grid = LbmGrid::from_chunk(&chunk, None);
        assert_eq!(grid.get(1, 1, 1).tag, GasCellTag::Liquid);

        // Now simulate evaporation: change the chunk voxel to steam
        chunk.set(1, 1, 1, Voxel::new(MaterialId::STEAM));

        let changed = sync_from_chunk(&chunk, &mut grid, None);
        assert!(changed > 0, "Should detect material change");
        assert_eq!(grid.get(1, 1, 1).tag, GasCellTag::Gas);
        assert_eq!(grid.get(1, 1, 1).material, MaterialId::STEAM);
    }

    #[test]
    fn sync_from_chunk_detects_condensation() {
        let _size = 4;
        let coord = ChunkCoord { x: 0, y: 0, z: 0 };
        let mut chunk = Chunk::new_filled(coord, MaterialId::AIR);

        // Start with steam (gas)
        chunk.set(2, 2, 2, Voxel::new(MaterialId::STEAM));
        let mut grid = LbmGrid::from_chunk(&chunk, None);
        assert_eq!(grid.get(2, 2, 2).tag, GasCellTag::Gas);

        // Condensation: steam → water
        chunk.set(2, 2, 2, Voxel::new(MaterialId::WATER));

        let changed = sync_from_chunk(&chunk, &mut grid, None);
        assert!(changed > 0);
        assert_eq!(grid.get(2, 2, 2).tag, GasCellTag::Liquid);
    }

    #[test]
    fn inject_steam_sets_gas_cell() {
        let mut grid = LbmGrid::new_empty(4);
        *grid.get_mut(1, 1, 1) = LbmCell::new_solid(MaterialId::STONE);

        inject_steam(&mut grid, 1, 1, 1, 0.6);

        let cell = grid.get(1, 1, 1);
        assert_eq!(cell.tag, GasCellTag::Gas);
        assert_eq!(cell.material, MaterialId::STEAM);
        assert!((cell.density() - 0.6).abs() < 1e-4);
    }

    #[test]
    fn remove_gas_sets_liquid_wall() {
        let mut grid = LbmGrid::new_empty(4);
        assert_eq!(grid.get(1, 1, 1).tag, GasCellTag::Gas);

        remove_gas(&mut grid, 1, 1, 1);

        let cell = grid.get(1, 1, 1);
        assert_eq!(cell.tag, GasCellTag::Liquid);
        assert_eq!(cell.material, MaterialId::WATER);
    }

    #[test]
    fn sync_roundtrip_preserves_air() {
        let coord = ChunkCoord { x: 0, y: 0, z: 0 };
        let mut chunk = Chunk::new_filled(coord, MaterialId::AIR);
        let grid = LbmGrid::from_chunk(&chunk, None);

        // All air → sync should change nothing
        let changed = sync_to_chunk(&grid, &mut chunk);
        assert_eq!(changed, 0, "All-air sync should change nothing");
    }
}
