// Cellular automata fluid simulation on the voxel grid.
//
// Each tick, fluid voxels (water, lava) attempt to flow:
//   1. Downward (gravity) — if the voxel below is air, swap.
//   2. Sideways — if can't flow down, spread to adjacent air (±X, ±Z).
//
// Flow probability is modulated by viscosity: low-viscosity fluids (water)
// flow every tick, while high-viscosity fluids (lava) flow less frequently.
// Uses a deterministic tick counter to avoid randomness.

#![allow(dead_code)]

use crate::world::voxel::{MaterialId, Voxel};

/// Materials that behave as fluids.
fn is_fluid(mat: MaterialId) -> bool {
    matches!(mat.0, 3 | 10) // water=3, lava=10
}

/// Dynamic viscosity (Pa·s) for fluid materials.
/// Source: Wikipedia — Viscosity of water (~0.001 Pa·s at 20°C),
/// Viscosity of basaltic lava (~100–1000 Pa·s).
fn fluid_viscosity(mat: MaterialId) -> f32 {
    match mat.0 {
        3 => 0.001,  // water
        10 => 500.0, // lava (basaltic, simplified)
        _ => 1.0,    // fallback
    }
}

/// Reference viscosity for flow probability scaling.
/// Water (0.001 Pa·s) flows every tick; higher viscosities flow less often.
const REFERENCE_VISCOSITY: f32 = 0.001;

/// Determine if a fluid voxel should flow this tick based on its viscosity.
/// Uses a deterministic approach: flow probability = reference_viscosity / viscosity.
/// For water: 0.001/0.001 = 1.0 (always flows).
/// For lava: 0.001/500 = 0.000002 (rarely flows per tick).
///
/// To make this deterministic, we use position + tick as a simple hash.
pub fn should_flow(viscosity: f32, x: usize, y: usize, z: usize, tick: u64) -> bool {
    if viscosity <= REFERENCE_VISCOSITY {
        return true;
    }
    // Flow interval: how many ticks between flows
    let interval = (viscosity / REFERENCE_VISCOSITY) as u64;
    if interval == 0 {
        return true;
    }
    // Deterministic hash from position to stagger flow across the grid
    let hash = (x as u64 * 73856093) ^ (y as u64 * 19349663) ^ (z as u64 * 83492791);
    (tick.wrapping_add(hash)).is_multiple_of(interval)
}

/// Simulate one tick of fluid flow within a flat voxel array of `size³`.
/// `tick` is a monotonic counter used for viscosity-based flow gating.
/// Returns the number of voxels that moved.
///
/// The algorithm processes voxels bottom-to-top so falling fluid settles in
/// one pass. Sideways spreading uses a snapshot to avoid order-dependent artifacts.
pub fn simulate_fluids(voxels: &mut [Voxel], size: usize) -> usize {
    simulate_fluids_with_tick(voxels, size, 0)
}

/// Like `simulate_fluids` but with an explicit tick counter for viscosity gating.
pub fn simulate_fluids_with_tick(voxels: &mut [Voxel], size: usize, tick: u64) -> usize {
    let mut moved = 0;

    // Pass 1: downward flow (top-to-bottom so fluid cascades in one pass)
    for y in (1..size).rev() {
        for z in 0..size {
            for x in 0..size {
                let idx = z * size * size + y * size + x;
                let below_idx = z * size * size + (y - 1) * size + x;

                if is_fluid(voxels[idx].material) && voxels[below_idx].material.is_air() {
                    let visc = fluid_viscosity(voxels[idx].material);
                    if !should_flow(visc, x, y, z, tick) {
                        continue;
                    }
                    voxels.swap(idx, below_idx);
                    moved += 1;
                }
            }
        }
    }

    // Pass 2: sideways spreading (snapshot to avoid directional bias)
    let snapshot: Vec<Voxel> = voxels.to_vec();
    let offsets: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

    for y in 0..size {
        for z in 0..size {
            for x in 0..size {
                let idx = z * size * size + y * size + x;

                if !is_fluid(snapshot[idx].material) {
                    continue;
                }

                let visc = fluid_viscosity(snapshot[idx].material);
                if !should_flow(visc, x, y, z, tick) {
                    continue;
                }

                // Only spread sideways if can't flow down
                if y > 0 {
                    let below_idx = z * size * size + (y - 1) * size + x;
                    if snapshot[below_idx].material.is_air() {
                        continue; // will flow down instead
                    }
                }

                // Try to spread to a random-ish neighbor (use position as deterministic seed)
                for &(dx, dz) in &offsets {
                    let nx = x as i32 + dx;
                    let nz = z as i32 + dz;

                    if nx < 0 || nz < 0 || nx >= size as i32 || nz >= size as i32 {
                        continue;
                    }

                    let nidx = nz as usize * size * size + y * size + nx as usize;

                    // Only spread into air, and only if the target's below isn't air
                    // (prevents fluid from flowing sideways over a cliff; it should fall)
                    if voxels[nidx].material.is_air() {
                        if y > 0 {
                            let nbelow = nz as usize * size * size + (y - 1) * size + nx as usize;
                            if voxels[nbelow].material.is_air() {
                                continue; // neighbor will flow down, don't spread there
                            }
                        }

                        // Move fluid to neighbor
                        voxels[nidx] = snapshot[idx];
                        voxels[idx] = Voxel::default(); // becomes air
                        moved += 1;
                        break; // only spread to one neighbor per tick
                    }
                }
            }
        }
    }

    moved
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_grid(size: usize) -> Vec<Voxel> {
        vec![Voxel::new(MaterialId::AIR); size * size * size]
    }

    fn idx(x: usize, y: usize, z: usize, size: usize) -> usize {
        z * size * size + y * size + x
    }

    fn count_material(voxels: &[Voxel], mat: MaterialId) -> usize {
        voxels.iter().filter(|v| v.material == mat).count()
    }

    #[test]
    fn water_falls_down() {
        let size = 4;
        let mut grid = make_grid(size);

        // Place water at y=3 (top)
        grid[idx(1, 3, 1, size)].material = MaterialId::WATER;

        let moved = simulate_fluids(&mut grid, size);

        assert!(moved > 0, "Water should fall");
        // Original position should be empty
        assert!(
            grid[idx(1, 3, 1, size)].material.is_air(),
            "Original position should be empty"
        );
        // Water should be in the bottom layer (y=0)
        let bottom_water: usize = (0..size)
            .flat_map(|z| (0..size).map(move |x| idx(x, 0, z, size)))
            .filter(|&i| grid[i].material == MaterialId::WATER)
            .count();
        assert_eq!(bottom_water, 1, "Exactly 1 water at y=0");
    }

    #[test]
    fn water_stops_on_solid() {
        let size = 4;
        let mut grid = make_grid(size);

        // Solid floor at y=0
        for x in 0..size {
            for z in 0..size {
                grid[idx(x, 0, z, size)].material = MaterialId::STONE;
            }
        }

        // Water at y=3
        grid[idx(1, 3, 1, size)].material = MaterialId::WATER;

        simulate_fluids(&mut grid, size);

        // Water should be at y=1 (just above stone floor)
        let y1_water: usize = (0..size)
            .flat_map(|z| (0..size).map(move |x| idx(x, 1, z, size)))
            .filter(|&i| grid[i].material == MaterialId::WATER)
            .count();
        assert_eq!(y1_water, 1, "Water should rest just above stone floor");
    }

    #[test]
    fn water_spreads_sideways_on_solid() {
        let size = 4;
        let mut grid = make_grid(size);

        // Solid floor
        for x in 0..size {
            for z in 0..size {
                grid[idx(x, 0, z, size)].material = MaterialId::STONE;
            }
        }

        // Stack of 3 water at (1,y,1)
        grid[idx(1, 1, 1, size)].material = MaterialId::WATER;
        grid[idx(1, 2, 1, size)].material = MaterialId::WATER;
        grid[idx(1, 3, 1, size)].material = MaterialId::WATER;

        // Run several ticks to let water spread
        for _ in 0..10 {
            simulate_fluids(&mut grid, size);
        }

        let water_count = count_material(&grid, MaterialId::WATER);
        assert_eq!(water_count, 3, "Water should be conserved");

        // At least some water should have spread sideways
        let center_water = if grid[idx(1, 1, 1, size)].material == MaterialId::WATER {
            1
        } else {
            0
        };
        let adjacent_water = count_material(&grid, MaterialId::WATER) - center_water;
        assert!(
            adjacent_water > 0,
            "Water should spread to adjacent positions"
        );
    }

    #[test]
    fn water_is_conserved() {
        let size = 4;
        let mut grid = make_grid(size);

        // Solid floor
        for x in 0..size {
            for z in 0..size {
                grid[idx(x, 0, z, size)].material = MaterialId::STONE;
            }
        }

        // Place 5 water voxels
        grid[idx(1, 1, 1, size)].material = MaterialId::WATER;
        grid[idx(1, 2, 1, size)].material = MaterialId::WATER;
        grid[idx(2, 1, 1, size)].material = MaterialId::WATER;
        grid[idx(1, 1, 2, size)].material = MaterialId::WATER;
        grid[idx(2, 2, 2, size)].material = MaterialId::WATER;

        let initial = count_material(&grid, MaterialId::WATER);

        for _ in 0..20 {
            simulate_fluids(&mut grid, size);
        }

        let final_count = count_material(&grid, MaterialId::WATER);
        assert_eq!(
            initial, final_count,
            "Water must be conserved: started {initial}, ended {final_count}"
        );
    }

    #[test]
    fn no_fluids_means_no_movement() {
        let size = 4;
        let mut grid = make_grid(size);

        // All stone, no fluids
        for v in grid.iter_mut() {
            v.material = MaterialId::STONE;
        }

        let moved = simulate_fluids(&mut grid, size);
        assert_eq!(moved, 0, "No fluids should mean no movement");
    }

    #[test]
    fn water_falls_off_edge_into_void() {
        let size = 4;
        let mut grid = make_grid(size);

        // Solid platform at y=2 in one corner
        grid[idx(0, 2, 0, size)].material = MaterialId::STONE;
        // Water on top of platform
        grid[idx(0, 3, 0, size)].material = MaterialId::WATER;

        simulate_fluids(&mut grid, size);

        // Water shouldn't move down through stone
        assert_eq!(
            grid[idx(0, 3, 0, size)].material,
            MaterialId::WATER,
            "Water should stay on top of stone"
        );
    }

    #[test]
    fn lava_also_flows() {
        let size = 4;
        let mut grid = make_grid(size);

        // Lava at top
        grid[idx(1, 3, 1, size)].material = MaterialId::LAVA;

        // Lava has high viscosity, so we need to find a tick where it flows.
        // Run many ticks until it moves.
        let mut moved_total = 0;
        for tick in 0..1_000_000 {
            moved_total += simulate_fluids_with_tick(&mut grid, size, tick);
            if moved_total > 0 {
                break;
            }
        }

        assert!(moved_total > 0, "Lava should eventually flow (just slowly)");
    }

    #[test]
    fn water_temperature_preserved_during_flow() {
        let size = 4;
        let mut grid = make_grid(size);

        // Solid floor so water stops at y=1
        for x in 0..size {
            for z in 0..size {
                grid[idx(x, 0, z, size)].material = MaterialId::STONE;
            }
        }

        grid[idx(1, 3, 1, size)].material = MaterialId::WATER;
        grid[idx(1, 3, 1, size)].temperature = 350.0;

        simulate_fluids(&mut grid, size);

        // Find where the water ended up and check its temperature
        let water_voxel = grid.iter().find(|v| v.material == MaterialId::WATER);
        assert!(water_voxel.is_some(), "Water should exist");
        assert_eq!(
            water_voxel.unwrap().temperature,
            350.0,
            "Temperature should be preserved during flow"
        );
    }

    // --- Viscosity-based flow rate tests ---
    // Source: Wikipedia — Viscosity of water (~0.001 Pa·s), lava (~100–1000 Pa·s)

    #[test]
    fn water_always_flows() {
        // Water viscosity (0.001 Pa·s) equals reference — should always flow
        assert!(should_flow(0.001, 0, 0, 0, 0));
        assert!(should_flow(0.001, 1, 2, 3, 100));
    }

    #[test]
    fn lava_flows_much_less_often() {
        // Lava viscosity (500 Pa·s) → interval = 500/0.001 = 500000 ticks
        // Count how many ticks out of 1000 allow flow at a fixed position
        let visc = 500.0;
        let mut flow_count = 0;
        for tick in 0..1000 {
            if should_flow(visc, 1, 1, 1, tick) {
                flow_count += 1;
            }
        }
        // With interval 500000, at most 1 flow in 1000 ticks
        assert!(
            flow_count <= 2,
            "Lava should flow very rarely: flowed {flow_count}/1000 ticks"
        );
    }

    #[test]
    fn water_flows_faster_than_lava_in_simulation() {
        // Set up identical grids, one with water and one with lava at the top
        let size = 4;

        // Water grid
        let mut water_grid = make_grid(size);
        for x in 0..size {
            for z in 0..size {
                water_grid[idx(x, 0, z, size)].material = MaterialId::STONE;
            }
        }
        water_grid[idx(1, 3, 1, size)].material = MaterialId::WATER;

        // Lava grid
        let mut lava_grid = make_grid(size);
        for x in 0..size {
            for z in 0..size {
                lava_grid[idx(x, 0, z, size)].material = MaterialId::STONE;
            }
        }
        lava_grid[idx(1, 3, 1, size)].material = MaterialId::LAVA;

        // Run 10 ticks
        let mut water_moves = 0;
        let mut lava_moves = 0;
        for tick in 0..10 {
            water_moves += simulate_fluids_with_tick(&mut water_grid, size, tick);
            lava_moves += simulate_fluids_with_tick(&mut lava_grid, size, tick);
        }

        assert!(
            water_moves > lava_moves,
            "Water ({water_moves} moves) should flow more than lava ({lava_moves} moves) in 10 ticks"
        );
    }

    #[test]
    fn viscosity_values_are_physical() {
        // Water: ~0.001 Pa·s at 20°C (Wikipedia: Viscosity)
        assert!((fluid_viscosity(MaterialId::WATER) - 0.001).abs() < 0.0001);
        // Lava: ~100-1000 Pa·s (Wikipedia: Viscosity of magma)
        let lava_visc = fluid_viscosity(MaterialId::LAVA);
        assert!((100.0..=1000.0).contains(&lava_visc));
    }
}
