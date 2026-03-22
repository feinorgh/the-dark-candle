// Plant growth: voxels that spread based on light, water, nutrients.
//
// Grass and other vegetation materials can spread to neighboring dirt voxels
// if conditions are met: sufficient light (air above), nearby water, and
// nutrients in the soil. This creates a simple vegetation simulation that
// interacts with the voxel world.

#![allow(dead_code)]

use crate::world::voxel::{MaterialId, Voxel};

/// Check if a voxel position has sky access (air above it all the way up).
fn has_light(voxels: &[Voxel], size: usize, x: usize, y: usize, z: usize) -> bool {
    for check_y in (y + 1)..size {
        let idx = z * size * size + check_y * size + x;
        if !voxels[idx].material.is_air() {
            return false;
        }
    }
    true
}

/// Check if any face-adjacent neighbor is water.
fn near_water(voxels: &[Voxel], size: usize, x: usize, y: usize, z: usize) -> bool {
    let neighbors: [(i32, i32, i32); 6] = [
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    ];

    for &(dx, dy, dz) in &neighbors {
        let nx = x as i32 + dx;
        let ny = y as i32 + dy;
        let nz = z as i32 + dz;

        if nx < 0 || ny < 0 || nz < 0 {
            continue;
        }
        let (nx, ny, nz) = (nx as usize, ny as usize, nz as usize);
        if nx >= size || ny >= size || nz >= size {
            continue;
        }

        let idx = nz * size * size + ny * size + nx;
        if voxels[idx].material == MaterialId::WATER {
            return true;
        }
    }
    false
}

/// Materials that can be overgrown by grass.
fn is_growable_surface(mat: MaterialId) -> bool {
    mat == MaterialId::DIRT
}

/// Materials that count as vegetation (can spread).
fn is_vegetation(mat: MaterialId) -> bool {
    mat == MaterialId(7) // Grass
}

/// Simulate one tick of plant growth on a chunk.
/// Grass spreads from existing grass to adjacent dirt voxels
/// if conditions (light, optional water proximity) are met.
/// Returns the number of new grass voxels created.
pub fn simulate_plant_growth(voxels: &mut [Voxel], size: usize, require_water: bool) -> usize {
    let total = size * size * size;
    assert_eq!(voxels.len(), total);

    let snapshot: Vec<Voxel> = voxels.to_vec();
    let mut grown = 0;

    let neighbors_xz: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let idx = z * size * size + y * size + x;
                if !is_vegetation(snapshot[idx].material) {
                    continue;
                }

                // Try to spread to horizontal neighbors
                for &(dx, dz) in &neighbors_xz {
                    let nx = x as i32 + dx;
                    let nz = z as i32 + dz;

                    if nx < 0 || nz < 0 {
                        continue;
                    }
                    let (nx, nz) = (nx as usize, nz as usize);
                    if nx >= size || nz >= size {
                        continue;
                    }

                    let nidx = nz * size * size + y * size + nx;

                    // Target must be growable surface
                    if !is_growable_surface(snapshot[nidx].material) {
                        continue;
                    }

                    // Must have light (air above)
                    if !has_light(&snapshot, size, nx, y, nz) {
                        continue;
                    }

                    // Optional water requirement
                    if require_water && !near_water(&snapshot, size, nx, y, nz) {
                        continue;
                    }

                    // Spread!
                    voxels[nidx].material = MaterialId(7); // Grass
                    grown += 1;
                }
            }
        }
    }

    grown
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_grid(size: usize) -> Vec<Voxel> {
        vec![Voxel::default(); size * size * size]
    }

    fn idx(x: usize, y: usize, z: usize, size: usize) -> usize {
        z * size * size + y * size + x
    }

    #[test]
    fn grass_spreads_to_adjacent_dirt() {
        let size = 8;
        let mut grid = make_grid(size);

        // Grass at (3, 2, 3)
        grid[idx(3, 2, 3, size)].material = MaterialId(7);
        // Dirt neighbors
        grid[idx(4, 2, 3, size)].material = MaterialId::DIRT;
        grid[idx(2, 2, 3, size)].material = MaterialId::DIRT;

        let grown = simulate_plant_growth(&mut grid, size, false);
        assert_eq!(grown, 2);
        assert_eq!(grid[idx(4, 2, 3, size)].material, MaterialId(7));
        assert_eq!(grid[idx(2, 2, 3, size)].material, MaterialId(7));
    }

    #[test]
    fn grass_does_not_spread_underground() {
        let size = 8;
        let mut grid = make_grid(size);

        // Grass at (3, 2, 3)
        grid[idx(3, 2, 3, size)].material = MaterialId(7);
        // Dirt neighbor with stone ceiling
        grid[idx(4, 2, 3, size)].material = MaterialId::DIRT;
        grid[idx(4, 3, 3, size)].material = MaterialId::STONE; // Blocks light

        let grown = simulate_plant_growth(&mut grid, size, false);
        // Should not spread to the blocked neighbor
        assert_eq!(grid[idx(4, 2, 3, size)].material, MaterialId::DIRT);
        assert_eq!(grown, 0);
    }

    #[test]
    fn grass_does_not_overwrite_stone() {
        let size = 8;
        let mut grid = make_grid(size);

        grid[idx(3, 2, 3, size)].material = MaterialId(7);
        grid[idx(4, 2, 3, size)].material = MaterialId::STONE;

        simulate_plant_growth(&mut grid, size, false);
        assert_eq!(grid[idx(4, 2, 3, size)].material, MaterialId::STONE);
    }

    #[test]
    fn water_requirement_prevents_growth() {
        let size = 8;
        let mut grid = make_grid(size);

        grid[idx(3, 2, 3, size)].material = MaterialId(7);
        grid[idx(4, 2, 3, size)].material = MaterialId::DIRT;
        // No water nearby

        let grown = simulate_plant_growth(&mut grid, size, true);
        assert_eq!(grown, 0, "Should not grow without water");
    }

    #[test]
    fn water_nearby_enables_growth() {
        let size = 8;
        let mut grid = make_grid(size);

        grid[idx(3, 2, 3, size)].material = MaterialId(7);
        grid[idx(4, 2, 3, size)].material = MaterialId::DIRT;
        grid[idx(4, 1, 3, size)].material = MaterialId::WATER; // Water below

        let grown = simulate_plant_growth(&mut grid, size, true);
        assert_eq!(grown, 1);
        assert_eq!(grid[idx(4, 2, 3, size)].material, MaterialId(7));
    }

    #[test]
    fn empty_grid_no_growth() {
        let size = 4;
        let mut grid = make_grid(size);
        let grown = simulate_plant_growth(&mut grid, size, false);
        assert_eq!(grown, 0);
    }

    #[test]
    fn growth_cascades_over_multiple_ticks() {
        let size = 8;
        let mut grid = make_grid(size);

        // Line of dirt with grass at one end
        grid[idx(1, 2, 3, size)].material = MaterialId(7);
        for x in 2..6 {
            grid[idx(x, 2, 3, size)].material = MaterialId::DIRT;
        }

        // Tick 1: spreads to x=2
        simulate_plant_growth(&mut grid, size, false);
        assert_eq!(grid[idx(2, 2, 3, size)].material, MaterialId(7));
        assert_eq!(grid[idx(3, 2, 3, size)].material, MaterialId::DIRT);

        // Tick 2: x=2 spreads to x=3
        simulate_plant_growth(&mut grid, size, false);
        assert_eq!(grid[idx(3, 2, 3, size)].material, MaterialId(7));
    }

    #[test]
    fn has_light_with_clear_sky() {
        let size = 8;
        let grid = make_grid(size);
        assert!(has_light(&grid, size, 4, 2, 4));
    }

    #[test]
    fn has_light_blocked_by_solid() {
        let size = 8;
        let mut grid = make_grid(size);
        grid[idx(4, 5, 4, size)].material = MaterialId::STONE;
        assert!(!has_light(&grid, size, 4, 2, 4));
    }
}
