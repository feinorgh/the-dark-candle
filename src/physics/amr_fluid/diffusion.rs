// Viscosity diffusion via implicit Jacobi iteration.
//
// Solves (I - ν·dt·∇²) u* = uⁿ implicitly to avoid the CFL restriction
// that explicit diffusion would impose at high viscosities (e.g., lava at
// 500 Pa·s). The Jacobi method iterates toward the solution — more
// iterations give better accuracy but cost more.

use super::types::{CellTag, FluidGrid};
use crate::data::MaterialRegistry;
use crate::world::voxel::MaterialId;

/// Diffuse velocity in-place using implicit Jacobi iteration.
///
/// `viscosity_fn` maps a MaterialId to its kinematic viscosity ν (m²/s).
/// Kinematic viscosity ν = μ / ρ where μ is dynamic viscosity (Pa·s)
/// and ρ is density (kg/m³).
///
/// `dt` is the timestep in seconds. `iterations` controls solver accuracy.
pub fn diffuse(
    grid: &mut FluidGrid,
    dt: f32,
    iterations: usize,
    viscosity_fn: &dyn Fn(MaterialId) -> f32,
) {
    let size = grid.size();

    // Snapshot the original velocity field (right-hand side of the implicit system).
    let rhs: Vec<[f32; 3]> = grid.cells().iter().map(|c| c.velocity).collect();

    // dx = 1.0 m (voxel size)
    // For implicit diffusion: (1 + 6·ν·dt/dx²) · u_center = u⁰ + ν·dt/dx² · Σ(u_neighbors)
    // Factor a = ν·dt / dx² = ν·dt (since dx=1)

    for _iter in 0..iterations {
        // Read current velocities into snapshot for Jacobi (don't read from
        // the array we're writing to within the same sweep).
        let current: Vec<[f32; 3]> = grid.cells().iter().map(|c| c.velocity).collect();

        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let idx = z * size * size + y * size + x;
                    let cell = &grid.cells()[idx];

                    if !cell.is_fluid() {
                        continue;
                    }

                    let nu = viscosity_fn(cell.material);
                    let a = nu * dt; // ν·dt / dx² with dx=1

                    // Sum neighbor velocities
                    let mut neighbor_sum = [0.0_f32; 3];
                    let mut neighbor_count = 0u32;

                    for (dx, dy, dz) in FACE_OFFSETS {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        let nz = z as i32 + dz;

                        if nx < 0
                            || ny < 0
                            || nz < 0
                            || nx >= size as i32
                            || ny >= size as i32
                            || nz >= size as i32
                        {
                            // Boundary: treat as zero velocity (wall).
                            neighbor_count += 1;
                            continue;
                        }

                        let nidx = nz as usize * size * size + ny as usize * size + nx as usize;
                        let ntag = grid.cells()[nidx].tag;

                        if ntag == CellTag::Solid {
                            // No-slip wall: contributes zero velocity.
                            neighbor_count += 1;
                            continue;
                        }

                        for i in 0..3 {
                            neighbor_sum[i] += current[nidx][i];
                        }
                        neighbor_count += 1;
                    }

                    // Jacobi update: u_new = (u⁰ + a·Σ(u_neighbors)) / (1 + neighbor_count·a)
                    let denom = 1.0 + neighbor_count as f32 * a;
                    let cell = grid.get_mut(x, y, z);
                    for i in 0..3 {
                        cell.velocity[i] = (rhs[idx][i] + a * neighbor_sum[i]) / denom;
                    }
                }
            }
        }
    }
}

/// Convenience: diffuse using MaterialRegistry for viscosity lookup.
///
/// Computes kinematic viscosity ν = μ / ρ from material properties.
/// Falls back to water viscosity (ν ≈ 1e-6 m²/s) if material is unknown.
pub fn diffuse_with_registry(
    grid: &mut FluidGrid,
    dt: f32,
    iterations: usize,
    registry: &MaterialRegistry,
) {
    let viscosity_fn = |mat: MaterialId| -> f32 {
        if let Some(data) = registry.get(mat) {
            let mu = data.viscosity.unwrap_or(1.0e-3); // Pa·s
            let rho = data.density.max(1.0); // kg/m³, avoid div-by-zero
            mu / rho // kinematic viscosity m²/s
        } else {
            1.0e-6 // water kinematic viscosity fallback
        }
    };
    diffuse(grid, dt, iterations, &viscosity_fn);
}

const FACE_OFFSETS: [(i32, i32, i32); 6] = [
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
];

#[cfg(test)]
mod tests {
    use super::*;

    fn water_grid(size: usize) -> FluidGrid {
        let mut grid = FluidGrid::new_empty(size);
        for cell in grid.cells_mut() {
            cell.tag = CellTag::Liquid;
            cell.material = MaterialId::WATER;
        }
        grid
    }

    /// Constant kinematic viscosity for tests (ν = 1e-3 m²/s, ~glycerin-like).
    fn test_viscosity(_mat: MaterialId) -> f32 {
        1.0e-3
    }

    #[test]
    fn zero_velocity_stays_zero() {
        let mut grid = water_grid(4);
        diffuse(&mut grid, 1.0 / 60.0, 20, &test_viscosity);

        for cell in grid.cells() {
            assert_eq!(cell.velocity, [0.0, 0.0, 0.0]);
        }
    }

    #[test]
    fn uniform_velocity_stays_uniform() {
        let mut grid = water_grid(4);
        for cell in grid.cells_mut() {
            cell.velocity = [2.0, -1.0, 0.5];
        }

        let original_v = [2.0_f32, -1.0, 0.5];
        diffuse(&mut grid, 1.0 / 60.0, 20, &test_viscosity);

        // Uniform field: all neighbors equal → diffusion is zero → no change.
        // Interior cells should be unchanged; boundary cells may differ slightly
        // because boundary = zero velocity (wall BC).
        let center = grid.get(2, 2, 2);
        for (i, &orig) in original_v.iter().enumerate() {
            assert!(
                (center.velocity[i] - orig).abs() < 0.1,
                "component {i}: expected ~{orig}, got {}",
                center.velocity[i]
            );
        }
    }

    #[test]
    fn high_viscosity_spreads_velocity_more() {
        // Center spike of velocity; after diffusion, the spike should flatten.
        let mut grid_low = water_grid(8);
        let mut grid_high = water_grid(8);

        let spike = [10.0_f32, 0.0, 0.0];
        grid_low.get_mut(4, 4, 4).velocity = spike;
        grid_high.get_mut(4, 4, 4).velocity = spike;

        // Low viscosity
        diffuse(&mut grid_low, 1.0 / 60.0, 20, &|_| 1.0e-6);
        // High viscosity
        diffuse(&mut grid_high, 1.0 / 60.0, 20, &|_| 1.0);

        let center_low = grid_low.get(4, 4, 4).velocity[0];
        let center_high = grid_high.get(4, 4, 4).velocity[0];

        // High viscosity should spread more, reducing the center peak further.
        assert!(
            center_high < center_low,
            "high viscosity center {} should be less than low {}",
            center_high,
            center_low
        );
    }

    #[test]
    fn solid_cells_untouched() {
        let mut grid = water_grid(4);
        grid.get_mut(0, 0, 0).tag = CellTag::Solid;
        grid.get_mut(0, 0, 0).material = MaterialId::STONE;
        grid.get_mut(1, 1, 1).velocity = [5.0, 0.0, 0.0];

        diffuse(&mut grid, 1.0 / 60.0, 10, &test_viscosity);
        assert_eq!(grid.get(0, 0, 0).velocity, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn air_cells_untouched() {
        let mut grid = FluidGrid::new_empty(4);
        // One liquid cell with velocity
        grid.get_mut(2, 2, 2).tag = CellTag::Liquid;
        grid.get_mut(2, 2, 2).material = MaterialId::WATER;
        grid.get_mut(2, 2, 2).velocity = [5.0, 0.0, 0.0];

        diffuse(&mut grid, 1.0 / 60.0, 10, &test_viscosity);
        assert_eq!(grid.get(0, 0, 0).velocity, [0.0, 0.0, 0.0]);
    }
}
