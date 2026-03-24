// Pressure projection: enforce incompressibility via the Poisson equation.
//
// After advection and diffusion produce an intermediate velocity field u*,
// we solve ∇²p = (ρ/dt)·∇·u* for the pressure correction, then subtract
// the pressure gradient from u* to get a divergence-free velocity field.
//
// Boundary conditions:
// - Solid cells: zero normal velocity (no-penetration).
// - Surface cells: p = 0 (atmospheric pressure reference).
// - Air cells: not included in the solve.
//
// Solver: Jacobi iteration (simple, parallelizable). Convergence is
// adequate for 32³ grids at 50 iterations; multigrid upgrade comes later.

use super::types::{CellTag, FluidGrid};

/// Project the velocity field to be divergence-free.
///
/// `dt` is the timestep in seconds. `density` is the fluid density in kg/m³
/// (uniform for now; per-cell density is a future extension).
/// `iterations` controls the pressure Poisson solve accuracy.
pub fn project(grid: &mut FluidGrid, dt: f32, density: f32, iterations: usize) {
    let size = grid.size();
    // dx = 1.0 m (voxel = 1 meter)

    // Step 1: Compute divergence of the velocity field.
    let mut divergence = vec![0.0_f32; size * size * size];
    compute_divergence(grid, &mut divergence);

    // Step 2: Solve ∇²p = (ρ / dt) · divergence via Jacobi iteration.
    let mut pressure = vec![0.0_f32; size * size * size];
    let rhs_scale = density / dt;

    for _iter in 0..iterations {
        let old_pressure = pressure.clone();

        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let idx = z * size * size + y * size + x;
                    let tag = grid.cells()[idx].tag;

                    match tag {
                        CellTag::Air | CellTag::Solid => {
                            pressure[idx] = 0.0;
                            continue;
                        }
                        CellTag::Surface => {
                            // Free surface: p = 0 (Dirichlet BC).
                            pressure[idx] = 0.0;
                            continue;
                        }
                        CellTag::Liquid => {}
                    }

                    // Sum neighbor pressures and count valid (non-solid) neighbors.
                    let mut p_sum = 0.0_f32;
                    let mut n_valid = 0u32;

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
                            // Boundary: treat as air (p=0).
                            n_valid += 1;
                            continue;
                        }

                        let nidx = nz as usize * size * size + ny as usize * size + nx as usize;
                        let ntag = grid.cells()[nidx].tag;

                        if ntag == CellTag::Solid {
                            // Solid: don't include in pressure stencil, but count
                            // as a valid face for the denominator (Neumann BC).
                            n_valid += 1;
                            p_sum += old_pressure[idx]; // mirror own pressure
                            continue;
                        }

                        n_valid += 1;
                        p_sum += old_pressure[nidx];
                    }

                    if n_valid > 0 {
                        // Jacobi: p_new = (Σ p_neighbors - rhs_scale · div · dx²) / n_valid
                        // With dx=1: p_new = (Σ p_neighbors - rhs_scale · div) / n_valid
                        pressure[idx] = (p_sum - rhs_scale * divergence[idx]) / n_valid as f32;
                    }
                }
            }
        }
    }

    // Step 3: Subtract pressure gradient from velocity.
    apply_pressure_gradient(grid, &pressure, dt, density);

    // Store computed pressure in cells for diagnostics.
    for (i, cell) in grid.cells_mut().iter_mut().enumerate() {
        cell.pressure = pressure[i];
    }
}

/// Compute the divergence of the velocity field.
///
/// div(u) = (u_right - u_left + v_top - v_bottom + w_front - w_back) / (2·dx)
/// Using central differences with dx = 1.
fn compute_divergence(grid: &FluidGrid, divergence: &mut [f32]) {
    let size = grid.size();

    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let idx = z * size * size + y * size + x;

                if !grid.cells()[idx].is_fluid() {
                    divergence[idx] = 0.0;
                    continue;
                }

                // Face velocities: use central differences.
                let vel = |cx: usize, cy: usize, cz: usize| -> [f32; 3] {
                    let c = grid.get(cx, cy, cz);
                    if c.tag == CellTag::Solid {
                        [0.0; 3]
                    } else {
                        c.velocity
                    }
                };

                let u_right = if x + 1 < size {
                    vel(x + 1, y, z)[0]
                } else {
                    0.0
                };
                let u_left = if x > 0 { vel(x - 1, y, z)[0] } else { 0.0 };
                let v_top = if y + 1 < size {
                    vel(x, y + 1, z)[1]
                } else {
                    0.0
                };
                let v_bottom = if y > 0 { vel(x, y - 1, z)[1] } else { 0.0 };
                let w_front = if z + 1 < size {
                    vel(x, y, z + 1)[2]
                } else {
                    0.0
                };
                let w_back = if z > 0 { vel(x, y, z - 1)[2] } else { 0.0 };

                // Central difference divergence (factor 0.5 for central diff / dx=1)
                divergence[idx] =
                    0.5 * ((u_right - u_left) + (v_top - v_bottom) + (w_front - w_back));
            }
        }
    }
}

/// Subtract the pressure gradient from the velocity field.
///
/// u -= (dt / ρ) · ∇p
fn apply_pressure_gradient(grid: &mut FluidGrid, pressure: &[f32], dt: f32, density: f32) {
    let size = grid.size();
    let scale = dt / density;

    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let idx = z * size * size + y * size + x;

                if !grid.cells()[idx].is_fluid() {
                    continue;
                }

                // Central difference pressure gradient.
                let p_center = pressure[idx];

                let p_right = if x + 1 < size {
                    pressure[z * size * size + y * size + (x + 1)]
                } else {
                    p_center
                };
                let p_left = if x > 0 {
                    pressure[z * size * size + y * size + (x - 1)]
                } else {
                    p_center
                };
                let p_top = if y + 1 < size {
                    pressure[z * size * size + (y + 1) * size + x]
                } else {
                    p_center
                };
                let p_bottom = if y > 0 {
                    pressure[z * size * size + (y - 1) * size + x]
                } else {
                    p_center
                };
                let p_front = if z + 1 < size {
                    pressure[(z + 1) * size * size + y * size + x]
                } else {
                    p_center
                };
                let p_back = if z > 0 {
                    pressure[(z - 1) * size * size + y * size + x]
                } else {
                    p_center
                };

                let grad_x = 0.5 * (p_right - p_left);
                let grad_y = 0.5 * (p_top - p_bottom);
                let grad_z = 0.5 * (p_front - p_back);

                let cell = grid.get_mut(x, y, z);
                cell.velocity[0] -= scale * grad_x;
                cell.velocity[1] -= scale * grad_y;
                cell.velocity[2] -= scale * grad_z;
            }
        }
    }
}

/// Measure residual divergence (should be near zero after projection).
pub fn max_divergence(grid: &FluidGrid) -> f32 {
    let size = grid.size();
    let mut div = vec![0.0_f32; size * size * size];
    compute_divergence(grid, &mut div);
    div.iter()
        .enumerate()
        .filter(|(i, _)| grid.cells()[*i].is_fluid())
        .map(|(_, d)| d.abs())
        .fold(0.0_f32, f32::max)
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
    use crate::world::voxel::MaterialId;

    fn water_grid(size: usize) -> FluidGrid {
        let mut grid = FluidGrid::new_empty(size);
        for cell in grid.cells_mut() {
            cell.tag = CellTag::Liquid;
            cell.material = MaterialId::WATER;
        }
        grid
    }

    #[test]
    fn zero_velocity_has_zero_divergence() {
        let grid = water_grid(4);
        assert!(max_divergence(&grid) < 1e-10);
    }

    #[test]
    fn projection_reduces_divergence() {
        let mut grid = water_grid(8);

        // Create a divergent field: source at center
        grid.get_mut(4, 4, 4).velocity = [5.0, 5.0, 5.0];

        let div_before = max_divergence(&grid);
        assert!(div_before > 0.1, "should start divergent: {div_before}");

        project(&mut grid, 1.0 / 60.0, 1000.0, 50);

        let div_after = max_divergence(&grid);
        assert!(
            div_after < div_before,
            "divergence should decrease: {div_before} → {div_after}"
        );
    }

    #[test]
    fn uniform_velocity_stays_divergence_free() {
        let mut grid = water_grid(8);
        for cell in grid.cells_mut() {
            cell.velocity = [1.0, 0.0, 0.0];
        }

        project(&mut grid, 1.0 / 60.0, 1000.0, 50);

        // Interior velocity should remain approximately uniform.
        // Boundary cells are affected by open-boundary conditions.
        let v = grid.get(4, 4, 4).velocity;
        assert!(
            (v[0] - 1.0).abs() < 0.5,
            "x-velocity should stay near 1.0: {v:?}"
        );
    }

    #[test]
    fn solid_walls_block_flow() {
        let mut grid = water_grid(4);
        // Place a solid wall at x=0
        for z in 0..4 {
            for y in 0..4 {
                let cell = grid.get_mut(0, y, z);
                cell.tag = CellTag::Solid;
                cell.material = MaterialId::STONE;
                cell.velocity = [0.0; 3];
            }
        }

        // Give fluid velocity toward the wall
        grid.get_mut(1, 2, 2).velocity = [-5.0, 0.0, 0.0];

        project(&mut grid, 1.0 / 60.0, 1000.0, 50);

        // Solid wall should still have zero velocity
        assert_eq!(grid.get(0, 2, 2).velocity, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn surface_cells_get_zero_pressure() {
        let mut grid = water_grid(4);
        // Mark corners as surface
        grid.get_mut(0, 0, 0).tag = CellTag::Surface;

        project(&mut grid, 1.0 / 60.0, 1000.0, 50);

        assert_eq!(grid.get(0, 0, 0).pressure, 0.0);
    }

    #[test]
    fn pressure_is_stored_in_cells() {
        let mut grid = water_grid(8);
        grid.get_mut(4, 4, 4).velocity = [5.0, 0.0, 0.0];

        project(&mut grid, 1.0 / 60.0, 1000.0, 50);

        // At least some interior cells should have nonzero pressure.
        let has_nonzero = grid.cells().iter().any(|c| c.pressure.abs() > 1e-10);
        assert!(has_nonzero, "some cells should have nonzero pressure");
    }
}
