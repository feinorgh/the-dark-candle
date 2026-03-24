// Single-step fluid simulation pipeline.
//
// Chains the operator-splitting stages in order:
//   1. Surface tagging (classify LIQUID/AIR/SURFACE)
//   2. Advection (semi-Lagrangian)
//   3. Diffusion (implicit Jacobi with per-material viscosity)
//   4. Pressure projection (Poisson solve for divergence-free velocity)
//
// This is a pure function with no ECS dependency — takes a FluidGrid and
// configuration, returns the updated grid. The Bevy plugin calls this.

use super::advection;
use super::diffusion;
use super::pressure;
use super::surface;
use super::types::FluidGrid;
use crate::data::{FluidConfig, MaterialRegistry};
use crate::world::voxel::MaterialId;

/// Run one complete fluid simulation step.
///
/// `registry` provides per-material viscosity and density lookups.
/// If `None`, falls back to water properties.
/// `config` controls solver iterations and CFL limits.
/// `dt` is the timestep in seconds.
pub fn fluid_step(
    grid: &mut FluidGrid,
    registry: Option<&MaterialRegistry>,
    config: &FluidConfig,
    dt: f32,
) {
    // Skip if no fluid exists.
    if !grid.has_fluid() {
        return;
    }

    // 1. Reclassify cell surface tags based on neighbor adjacency.
    surface::update_tags(grid);

    // 2. Advect the velocity field.
    let advected = advection::advect(grid, dt);
    // Copy advected velocities back into grid (tags/materials unchanged).
    for (i, cell) in grid.cells_mut().iter_mut().enumerate() {
        cell.velocity = advected.cells()[i].velocity;
    }

    // 3. Diffuse velocity based on viscosity.
    if let Some(reg) = registry {
        diffusion::diffuse_with_registry(grid, dt, config.diffusion_iterations, reg);
    } else {
        // Fallback: water kinematic viscosity ≈ 1e-6 m²/s
        diffusion::diffuse(grid, dt, config.diffusion_iterations, &|_| 1.0e-6);
    }

    // 4. Project velocity to enforce incompressibility.
    let density = if let Some(reg) = registry {
        // Use density of the most common fluid material, or default.
        most_common_fluid_density(grid, reg).unwrap_or(config.density_default)
    } else {
        config.density_default
    };

    pressure::project(grid, dt, density, config.pressure_solver_iterations);
}

/// Run a fluid step with explicit uniform viscosity and density (for testing).
pub fn fluid_step_simple(
    grid: &mut FluidGrid,
    kinematic_viscosity: f32,
    density: f32,
    dt: f32,
    pressure_iterations: usize,
    diffusion_iterations: usize,
) {
    if !grid.has_fluid() {
        return;
    }

    surface::update_tags(grid);

    let advected = advection::advect(grid, dt);
    for (i, cell) in grid.cells_mut().iter_mut().enumerate() {
        cell.velocity = advected.cells()[i].velocity;
    }

    diffusion::diffuse(grid, dt, diffusion_iterations, &|_| kinematic_viscosity);
    pressure::project(grid, dt, density, pressure_iterations);
}

/// Find the density of the most common fluid material in the grid.
fn most_common_fluid_density(grid: &FluidGrid, registry: &MaterialRegistry) -> Option<f32> {
    let mut counts: std::collections::HashMap<MaterialId, usize> = std::collections::HashMap::new();
    for cell in grid.cells() {
        if cell.is_fluid() {
            *counts.entry(cell.material).or_insert(0) += 1;
        }
    }

    counts
        .into_iter()
        .max_by_key(|&(_, count)| count)
        .and_then(|(mat, _)| registry.get(mat))
        .map(|data| data.density)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::amr_fluid::types::CellTag;

    fn water_grid(size: usize) -> FluidGrid {
        let mut grid = FluidGrid::new_empty(size);
        for cell in grid.cells_mut() {
            cell.tag = CellTag::Liquid;
            cell.material = MaterialId::WATER;
        }
        grid
    }

    #[test]
    fn empty_grid_is_noop() {
        let mut grid = FluidGrid::new_empty(4);
        let config = FluidConfig::default();
        fluid_step(&mut grid, None, &config, 1.0 / 60.0);
        // All cells should remain air with zero velocity.
        for cell in grid.cells() {
            assert_eq!(cell.tag, CellTag::Air);
            assert_eq!(cell.velocity, [0.0; 3]);
        }
    }

    #[test]
    fn still_water_stays_still() {
        let mut grid = water_grid(8);
        let config = FluidConfig::default();

        fluid_step(&mut grid, None, &config, 1.0 / 60.0);

        // Zero initial velocity → zero after step (no external forces in fluid solver).
        let max_speed = grid.max_speed();
        assert!(
            max_speed < 1e-6,
            "still water should have near-zero velocity: {max_speed}"
        );
    }

    #[test]
    fn dam_break_spreads_water() {
        // Left half water, right half air in a 16³ grid.
        let size = 16;
        let mut grid = FluidGrid::new_empty(size);
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    if x < size / 2 {
                        let cell = grid.get_mut(x, y, z);
                        cell.tag = CellTag::Liquid;
                        cell.material = MaterialId::WATER;
                    }
                }
            }
        }

        let initial_volume = grid.fluid_volume();

        // Give the water column a rightward push (simulating pressure from the column).
        for z in 0..size {
            for y in 0..size {
                // Rightmost column of water gets velocity.
                let cell = grid.get_mut(size / 2 - 1, y, z);
                cell.velocity = [2.0, 0.0, 0.0];
            }
        }

        // Run several steps
        let dt = 1.0 / 60.0;
        for _ in 0..10 {
            fluid_step_simple(&mut grid, 1.0e-6, 1000.0, dt, 50, 10);
        }

        // Velocity should be bounded (no explosion).
        let max_speed = grid.max_speed();
        assert!(
            max_speed < 100.0,
            "velocity should stay bounded: {max_speed}"
        );

        // Fluid volume is conserved (cell tags don't change count in this solver —
        // material movement is handled by sync, which is tested separately).
        let final_volume = grid.fluid_volume();
        assert!(
            (final_volume - initial_volume).abs() < 1.0,
            "volume should be conserved: {initial_volume} → {final_volume}"
        );
    }

    #[test]
    fn pressure_projection_reduces_divergence() {
        let mut grid = water_grid(8);
        grid.get_mut(4, 4, 4).velocity = [10.0, 0.0, 0.0];

        let div_before = pressure::max_divergence(&grid);
        fluid_step_simple(&mut grid, 1.0e-6, 1000.0, 1.0 / 60.0, 50, 10);
        let div_after = pressure::max_divergence(&grid);

        assert!(
            div_after < div_before,
            "divergence should decrease: {div_before} → {div_after}"
        );
    }

    #[test]
    fn simple_api_matches_config_api() {
        let mut grid1 = water_grid(4);
        let mut grid2 = grid1.clone();

        grid1.get_mut(2, 2, 2).velocity = [1.0, 0.0, 0.0];
        grid2.get_mut(2, 2, 2).velocity = [1.0, 0.0, 0.0];

        let config = FluidConfig {
            pressure_solver_iterations: 30,
            diffusion_iterations: 10,
            density_default: 1000.0,
            cfl_max: 1.0,
        };

        fluid_step(&mut grid1, None, &config, 1.0 / 60.0);
        fluid_step_simple(&mut grid2, 1.0e-6, 1000.0, 1.0 / 60.0, 30, 10);

        // Both should produce similar results (not identical due to viscosity lookup
        // differences, but the structure should match).
        let speed1 = grid1.max_speed();
        let speed2 = grid2.max_speed();
        assert!(
            (speed1 - speed2).abs() < 1.0,
            "speeds should be similar: {speed1} vs {speed2}"
        );
    }
}
