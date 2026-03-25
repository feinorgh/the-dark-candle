// Semi-Lagrangian advection for the velocity field.
//
// Each cell's new velocity is found by backtracing along the current velocity
// to find where the "particle" came from, then trilinearly interpolating the
// old velocity at that source position.
//
// This method is unconditionally stable for any timestep (no CFL restriction
// on stability, though accuracy degrades for large dt). It introduces some
// numerical diffusion, which the viscosity diffusion step handles explicitly.

use super::types::FluidGrid;

/// Advect the velocity field using semi-Lagrangian backtracing.
///
/// Returns a new grid with updated velocities. Cell tags, materials, and
/// pressures are copied unchanged; only velocity is advected.
///
/// `dt` is the timestep in seconds.
pub fn advect(grid: &FluidGrid, dt: f32) -> FluidGrid {
    let size = grid.size();
    let mut result = grid.clone();

    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let cell = grid.get(x, y, z);

                if !cell.is_fluid() {
                    // Non-fluid cells keep zero velocity.
                    result.get_mut(x, y, z).velocity = [0.0; 3];
                    continue;
                }

                // Backtrace: find where this parcel came from.
                let fx = x as f32 - cell.velocity[0] * dt;
                let fy = y as f32 - cell.velocity[1] * dt;
                let fz = z as f32 - cell.velocity[2] * dt;

                // Clamp to grid (open boundary).
                let max = (size - 1) as f32;
                let fx = fx.clamp(0.0, max);
                let fy = fy.clamp(0.0, max);
                let fz = fz.clamp(0.0, max);

                // Trilinear interpolation of velocity at the backtrace point.
                result.get_mut(x, y, z).velocity = trilinear_velocity(grid, fx, fy, fz);
            }
        }
    }

    result
}

/// Maximum CFL number: max_speed * dt / dx. Should be ≤ 1 for accuracy.
pub fn cfl_number(grid: &FluidGrid, dt: f32) -> f32 {
    // dx = 1.0 (1 voxel = 1 meter)
    grid.max_speed() * dt
}

/// Trilinear interpolation of velocity at fractional coordinates.
fn trilinear_velocity(grid: &FluidGrid, fx: f32, fy: f32, fz: f32) -> [f32; 3] {
    let size = grid.size();
    let max = size - 1;

    let x0 = (fx.floor() as usize).min(max);
    let y0 = (fy.floor() as usize).min(max);
    let z0 = (fz.floor() as usize).min(max);
    let x1 = (x0 + 1).min(max);
    let y1 = (y0 + 1).min(max);
    let z1 = (z0 + 1).min(max);

    let tx = fx - x0 as f32;
    let ty = fy - y0 as f32;
    let tz = fz - z0 as f32;

    // Sample 8 corners, treating non-fluid cells as zero velocity.
    let sample = |x: usize, y: usize, z: usize| -> [f32; 3] {
        let c = grid.get(x, y, z);
        if c.is_fluid() { c.velocity } else { [0.0; 3] }
    };

    let c000 = sample(x0, y0, z0);
    let c100 = sample(x1, y0, z0);
    let c010 = sample(x0, y1, z0);
    let c110 = sample(x1, y1, z0);
    let c001 = sample(x0, y0, z1);
    let c101 = sample(x1, y0, z1);
    let c011 = sample(x0, y1, z1);
    let c111 = sample(x1, y1, z1);

    let mut result = [0.0_f32; 3];
    for i in 0..3 {
        let c00 = c000[i] * (1.0 - tx) + c100[i] * tx;
        let c10 = c010[i] * (1.0 - tx) + c110[i] * tx;
        let c01 = c001[i] * (1.0 - tx) + c101[i] * tx;
        let c11 = c011[i] * (1.0 - tx) + c111[i] * tx;

        let c0 = c00 * (1.0 - ty) + c10 * ty;
        let c1 = c01 * (1.0 - ty) + c11 * ty;

        result[i] = c0 * (1.0 - tz) + c1 * tz;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::amr_fluid::types::CellTag;
    use crate::world::voxel::MaterialId;

    fn water_grid(size: usize) -> FluidGrid {
        let mut grid = FluidGrid::new_empty(size);
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let cell = grid.get_mut(x, y, z);
                    cell.tag = CellTag::Liquid;
                    cell.material = MaterialId::WATER;
                }
            }
        }
        grid
    }

    #[test]
    fn uniform_flow_advects_cleanly() {
        let mut grid = water_grid(8);
        // Uniform velocity: +X at 1 m/s
        for cell in grid.cells_mut() {
            cell.velocity = [1.0, 0.0, 0.0];
        }

        let result = advect(&grid, 1.0);

        // After advection, uniform flow should remain uniform
        // (backtracing finds the same velocity everywhere).
        for z in 0..8 {
            for y in 0..8 {
                for x in 0..8 {
                    let v = result.get(x, y, z).velocity;
                    assert!(
                        (v[0] - 1.0).abs() < 1e-4,
                        "non-uniform at ({x},{y},{z}): {v:?}"
                    );
                    assert!(v[1].abs() < 1e-4);
                    assert!(v[2].abs() < 1e-4);
                }
            }
        }
    }

    #[test]
    fn zero_velocity_unchanged() {
        let grid = water_grid(4);
        let result = advect(&grid, 1.0 / 60.0);

        for cell in result.cells() {
            assert_eq!(cell.velocity, [0.0, 0.0, 0.0]);
        }
    }

    #[test]
    fn air_cells_stay_zero() {
        let mut grid = FluidGrid::new_empty(4);
        // One liquid cell with velocity, rest air
        let cell = grid.get_mut(2, 2, 2);
        cell.tag = CellTag::Liquid;
        cell.material = MaterialId::WATER;
        cell.velocity = [1.0, 0.0, 0.0];

        let result = advect(&grid, 1.0 / 60.0);
        assert_eq!(result.get(0, 0, 0).velocity, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn cfl_number_scales_with_speed_and_dt() {
        let mut grid = water_grid(4);
        grid.get_mut(0, 0, 0).velocity = [10.0, 0.0, 0.0];

        assert!((cfl_number(&grid, 0.1) - 1.0).abs() < 1e-6);
        assert!((cfl_number(&grid, 0.01) - 0.1).abs() < 1e-6);
    }

    #[test]
    fn trilinear_at_grid_point_returns_exact() {
        let mut grid = water_grid(4);
        grid.get_mut(1, 1, 1).velocity = [5.0, 3.0, -1.0];

        let v = trilinear_velocity(&grid, 1.0, 1.0, 1.0);
        assert!((v[0] - 5.0).abs() < 1e-6);
        assert!((v[1] - 3.0).abs() < 1e-6);
        assert!((v[2] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn trilinear_midpoint_averages() {
        let mut grid = water_grid(4);
        grid.get_mut(0, 0, 0).velocity = [0.0, 0.0, 0.0];
        grid.get_mut(1, 0, 0).velocity = [10.0, 0.0, 0.0];

        // Midpoint between (0,0,0) and (1,0,0)
        let v = trilinear_velocity(&grid, 0.5, 0.0, 0.0);
        assert!((v[0] - 5.0).abs() < 1e-4, "got {}", v[0]);
    }
}
