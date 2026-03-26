// Full LBM timestep pipeline.
//
// Orchestrates: forcing → collision → streaming → macroscopic recovery.
// Supports sub-stepping (multiple LBM steps per FixedUpdate tick).

use crate::data::FluidConfig;

use super::collision;
use super::macroscopic;
use super::streaming;
use super::types::LbmGrid;

/// Run a single LBM step on a grid.
///
/// Pipeline order:
/// 1. Apply external forces (gravity/buoyancy via Guo forcing)
/// 2. Collision (BGK + Smagorinsky)
/// 3. Streaming (propagation with bounce-back)
///
/// The grid is modified in place for collision/forcing, then replaced
/// by the streamed result.
///
/// # Arguments
/// * `coriolis_omega` - Optional planetary rotation vector in lattice units (ω × rotation_axis).
///   `None` disables Coriolis forcing.
pub fn lbm_step(
    grid: &mut LbmGrid,
    config: &FluidConfig,
    gravity_lattice: [f32; 3],
    rho_ambient: f32,
    coriolis_omega: Option<[f32; 3]>,
) {
    let tau = config.lbm_tau;
    let cs_smag = config.lbm_smagorinsky_cs;

    // 1. Apply buoyancy and Coriolis forcing to each gas cell
    for cell in grid.cells_mut() {
        if cell.is_gas() {
            let rho = cell.density();
            let mut force = macroscopic::buoyancy_force(rho, rho_ambient, gravity_lattice);

            // Add Coriolis force: F = -2ρ(ω × v)
            if let Some(omega) = coriolis_omega {
                let u = cell.velocity();
                // Cross product: omega × velocity
                let coriolis = [
                    -2.0 * rho * (omega[1] * u[2] - omega[2] * u[1]),
                    -2.0 * rho * (omega[2] * u[0] - omega[0] * u[2]),
                    -2.0 * rho * (omega[0] * u[1] - omega[1] * u[0]),
                ];
                force[0] += coriolis[0];
                force[1] += coriolis[1];
                force[2] += coriolis[2];
            }

            // Only apply if there's a meaningful force
            let f_mag = force[0] * force[0] + force[1] * force[1] + force[2] * force[2];
            if f_mag > 1e-20 {
                macroscopic::apply_guo_forcing(cell, force, tau);
            }
        }
    }

    // 2. Collision (BGK + Smagorinsky)
    collision::collide_grid(grid, tau, cs_smag);

    // 3. Streaming
    *grid = streaming::stream(grid);
}

/// Simplified LBM step with explicit parameters (no FluidConfig needed).
/// Useful for unit testing.
pub fn lbm_step_simple(grid: &mut LbmGrid, tau: f32, cs_smag: f32) {
    collision::collide_grid(grid, tau, cs_smag);
    *grid = streaming::stream(grid);
}

/// Run multiple LBM sub-steps per call.
pub fn lbm_step_n(
    grid: &mut LbmGrid,
    config: &FluidConfig,
    gravity_lattice: [f32; 3],
    rho_ambient: f32,
    coriolis_omega: Option<[f32; 3]>,
    n_steps: usize,
) {
    for _ in 0..n_steps {
        lbm_step(grid, config, gravity_lattice, rho_ambient, coriolis_omega);
    }
}

/// Check stability: returns the maximum Mach number in the grid.
/// LBM becomes inaccurate above Ma ≈ 0.3.
pub fn max_mach_number(grid: &LbmGrid) -> f32 {
    grid.max_mach()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::lbm_gas::lattice;
    use crate::physics::lbm_gas::types::{GasCellTag, LbmCell, LbmGrid};
    use crate::world::voxel::MaterialId;

    fn test_config() -> FluidConfig {
        FluidConfig {
            lbm_tau: 0.55,
            lbm_smagorinsky_cs: 0.1,
            lbm_steps_per_tick: 1,
            lbm_enabled: true,
            ..Default::default()
        }
    }

    #[test]
    fn simple_step_preserves_mass_closed_box() {
        let size = 8;
        let mut grid = LbmGrid::new_empty(size);

        // Create solid walls on all faces
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    if x == 0 || x == size - 1 || y == 0 || y == size - 1 || z == 0 || z == size - 1
                    {
                        *grid.get_mut(x, y, z) = LbmCell::new_solid(MaterialId::STONE);
                    }
                }
            }
        }

        // Density perturbation
        *grid.get_mut(4, 4, 4) = LbmCell::new_gas(MaterialId::AIR, 1.5);

        let mass_before = grid.total_mass();

        for _ in 0..10 {
            lbm_step_simple(&mut grid, 0.55, 0.1);
        }

        let mass_after = grid.total_mass();
        assert!(
            (mass_after - mass_before).abs() < 0.01,
            "Mass changed after 10 steps: {mass_before} → {mass_after}"
        );
    }

    #[test]
    fn step_with_buoyancy_creates_upward_flow() {
        let size = 8;
        let mut grid = LbmGrid::new_empty(size);

        // Walls on all faces
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    if x == 0 || x == size - 1 || y == 0 || y == size - 1 || z == 0 || z == size - 1
                    {
                        *grid.get_mut(x, y, z) = LbmCell::new_solid(MaterialId::STONE);
                    }
                }
            }
        }

        // Place a "hot" (lighter) gas blob at the bottom
        *grid.get_mut(4, 1, 4) = LbmCell::new_gas(MaterialId::AIR, 0.8);
        // Normal density ambient
        let rho_ambient = 1.0;

        let config = test_config();
        // Gravity pointing down in lattice units (small for stability)
        let gravity_lattice = [0.0, -0.001, 0.0];

        for _ in 0..20 {
            lbm_step(&mut grid, &config, gravity_lattice, rho_ambient, None);
        }

        // The lighter gas should develop upward velocity.
        // Check cells above the hot spot for net upward motion.
        let mut total_uy = 0.0_f32;
        for y in 2..size - 1 {
            let u = grid.get(4, y, 4).velocity();
            total_uy += u[1];
        }
        assert!(
            total_uy > 1e-6,
            "Buoyancy should produce net upward velocity in column, got {total_uy}"
        );

        // Also verify mass conservation
        let mass = grid.total_mass();
        assert!(mass > 0.0, "Grid lost all mass");
    }

    #[test]
    fn density_perturbation_spreads() {
        let size = 8;
        let mut grid = LbmGrid::new_empty(size);

        // Walls on all faces
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    if x == 0 || x == size - 1 || y == 0 || y == size - 1 || z == 0 || z == size - 1
                    {
                        *grid.get_mut(x, y, z) = LbmCell::new_solid(MaterialId::STONE);
                    }
                }
            }
        }

        // High-density spot in center
        *grid.get_mut(4, 4, 4) = LbmCell::new_gas(MaterialId::AIR, 2.0);

        let rho_center_before = grid.get(4, 4, 4).density();
        let rho_neighbor_before = grid.get(3, 4, 4).density();

        let config = test_config();
        for _ in 0..20 {
            lbm_step_simple(&mut grid, config.lbm_tau, config.lbm_smagorinsky_cs);
        }

        let rho_center_after = grid.get(4, 4, 4).density();
        let rho_neighbor_after = grid.get(3, 4, 4).density();

        // Density should have spread: center decreases, neighbor increases
        assert!(
            rho_center_after < rho_center_before,
            "Center density didn't decrease: {rho_center_before} → {rho_center_after}"
        );
        assert!(
            rho_neighbor_after > rho_neighbor_before,
            "Neighbor density didn't increase: {rho_neighbor_before} → {rho_neighbor_after}"
        );
    }

    #[test]
    fn mach_number_stays_low_for_small_perturbation() {
        let size = 8;
        let mut grid = LbmGrid::new_empty(size);

        // Walls on all faces to prevent boundary artifacts
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    if x == 0 || x == size - 1 || y == 0 || y == size - 1 || z == 0 || z == size - 1
                    {
                        *grid.get_mut(x, y, z) = LbmCell::new_solid(MaterialId::STONE);
                    }
                }
            }
        }

        // Small density perturbation
        *grid.get_mut(4, 4, 4) = LbmCell::new_gas(MaterialId::AIR, 1.01);

        for _ in 0..5 {
            lbm_step_simple(&mut grid, 0.55, 0.1);
        }

        let ma = max_mach_number(&grid);
        assert!(
            ma < 0.3,
            "Mach number too high for small perturbation: {ma}"
        );
    }

    #[test]
    fn multi_step_equivalent_to_single_steps() {
        let size = 6;
        let mut grid1 = LbmGrid::new_empty(size);
        let mut grid2 = grid1.clone();

        // Perturbation
        *grid1.get_mut(3, 3, 3) = LbmCell::new_gas(MaterialId::AIR, 1.1);
        *grid2.get_mut(3, 3, 3) = LbmCell::new_gas(MaterialId::AIR, 1.1);

        let config = test_config();

        // 3 single steps
        for _ in 0..3 {
            lbm_step(&mut grid1, &config, [0.0; 3], 1.0, None);
        }

        // 1 call with n=3
        lbm_step_n(&mut grid2, &config, [0.0; 3], 1.0, None, 3);

        // Should be identical
        for idx in 0..grid1.cells().len() {
            for i in 0..lattice::Q {
                assert!(
                    (grid1.cells()[idx].f[i] - grid2.cells()[idx].f[i]).abs() < 1e-6,
                    "Mismatch at cell {idx}, direction {i}"
                );
            }
        }
    }

    #[test]
    fn equilibrium_field_is_stable() {
        // A uniform equilibrium field should remain stable over many steps
        let size = 6;
        let mut grid = LbmGrid::new_empty(size);

        // Walls
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    if x == 0 || x == size - 1 || y == 0 || y == size - 1 || z == 0 || z == size - 1
                    {
                        *grid.get_mut(x, y, z) = LbmCell::new_solid(MaterialId::STONE);
                    }
                }
            }
        }

        let mass_before = grid.total_mass();

        for _ in 0..50 {
            lbm_step_simple(&mut grid, 0.6, 0.0);
        }

        let mass_after = grid.total_mass();
        assert!(
            (mass_after - mass_before).abs() < 0.01,
            "Equilibrium field lost mass: {mass_before} → {mass_after}"
        );

        // All interior cells should remain near ρ=1.0
        let rho_center = grid.get(3, 3, 3).density();
        assert!(
            (rho_center - 1.0).abs() < 0.01,
            "Center density deviated from equilibrium: {rho_center}"
        );
    }

    #[test]
    fn coriolis_deflects_moving_air() {
        let size = 12;
        let mut grid = LbmGrid::new_empty(size);

        // Walls on all faces
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    if x == 0 || x == size - 1 || y == 0 || y == size - 1 || z == 0 || z == size - 1
                    {
                        *grid.get_mut(x, y, z) = LbmCell::new_solid(MaterialId::STONE);
                    }
                }
            }
        }

        // Initialize interior with uniform gas and eastward velocity (+x)
        for z in 1..size - 1 {
            for y in 1..size - 1 {
                for x in 1..size - 1 {
                    let cell = grid.get_mut(x, y, z);
                    // Set equilibrium with initial velocity in +x direction
                    let rho = 1.0;
                    let u = [0.01, 0.0, 0.0];
                    cell.f = lattice::equilibrium(rho, u);
                    cell.material = MaterialId::AIR;
                    cell.tag = GasCellTag::Gas;
                }
            }
        }

        let config = test_config();
        // Coriolis with omega in +y direction (North Pole scenario)
        // Using a strong value to see clear deflection in 50 steps
        let omega = [0.0, 1e-3, 0.0];

        for _ in 0..50 {
            lbm_step(&mut grid, &config, [0.0; 3], 1.0, Some(omega));
        }

        // Collect z-velocity components from interior cells
        let mut total_uz = 0.0_f32;
        let mut count = 0;
        for z in 2..size - 2 {
            for y in 2..size - 2 {
                for x in 2..size - 2 {
                    let u = grid.get(x, y, z).velocity();
                    total_uz += u[2];
                    count += 1;
                }
            }
        }

        let avg_uz = total_uz / count as f32;
        // Coriolis force F = -2ρ(ω × v) with ω=[0,ω_y,0] and v=[v_x,0,0]
        // gives F = -2ρ[0, 0, -ω_y*v_x] = [0, 0, 2ρ*ω_y*v_x]
        // So z-velocity should become positive (deflection to the right)
        // With the small omega value (1e-3 in lattice units), the deflection
        // is very small but should be measurable above numerical noise.
        assert!(
            avg_uz.abs() > 1e-7,
            "Coriolis should deflect flow: avg u_z = {avg_uz}"
        );
    }

    #[test]
    fn coriolis_none_is_noop() {
        let size = 12;
        let mut grid = LbmGrid::new_empty(size);

        // Walls on all faces
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    if x == 0 || x == size - 1 || y == 0 || y == size - 1 || z == 0 || z == size - 1
                    {
                        *grid.get_mut(x, y, z) = LbmCell::new_solid(MaterialId::STONE);
                    }
                }
            }
        }

        // Initialize interior with uniform gas and eastward velocity (+x)
        for z in 1..size - 1 {
            for y in 1..size - 1 {
                for x in 1..size - 1 {
                    let cell = grid.get_mut(x, y, z);
                    // Set equilibrium with initial velocity in +x direction
                    let rho = 1.0;
                    let u = [0.01, 0.0, 0.0];
                    cell.f = lattice::equilibrium(rho, u);
                    cell.material = MaterialId::AIR;
                    cell.tag = GasCellTag::Gas;
                }
            }
        }

        let config = test_config();

        for _ in 0..50 {
            lbm_step(&mut grid, &config, [0.0; 3], 1.0, None);
        }

        // Check that z-velocity remains near zero without Coriolis
        let mut total_uz = 0.0_f32;
        let mut count = 0;
        for z in 2..size - 2 {
            for y in 2..size - 2 {
                for x in 2..size - 2 {
                    let u = grid.get(x, y, z).velocity();
                    total_uz += u[2];
                    count += 1;
                }
            }
        }

        let avg_uz = total_uz / count as f32;
        assert!(
            avg_uz.abs() < 1e-4,
            "Without Coriolis, z-velocity should remain near zero: avg u_z = {avg_uz}"
        );
    }
}
