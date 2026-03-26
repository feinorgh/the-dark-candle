// End-to-end atmosphere physics integration tests.
//
// These tests exercise the full atmospheric pipeline:
//   LBM convection → moisture transport → cloud formation → precipitation
//
// Each test builds an isolated LBM grid scenario and verifies emergent behavior
// from first-principles physics. No scripted outcomes — just boundary conditions
// and conservation checks.

#[cfg(test)]
mod tests {
    use crate::data::FluidConfig;
    use crate::physics::atmosphere::{self, AtmosphereConfig};
    use crate::physics::lbm_gas::moisture;
    use crate::physics::lbm_gas::precipitation;
    use crate::physics::lbm_gas::step;
    use crate::physics::lbm_gas::types::{LbmCell, LbmGrid};
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

    fn make_box(size: usize) -> LbmGrid {
        let mut grid = LbmGrid::new_empty(size);
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
        grid
    }

    // --- Convection tests ---

    #[test]
    fn thermal_convection_hot_bottom_rises() {
        // Place lighter (warmer) air at the bottom of a box and verify
        // buoyancy drives it upward over multiple LBM steps.
        let size = 12;
        let mut grid = make_box(size);
        let config = test_config();
        let gravity = [0.0, -0.001, 0.0];
        let rho_ambient = 1.0;

        // Lighter air at the bottom (simulating heated surface layer)
        for x in 2..size - 2 {
            for z in 2..size - 2 {
                *grid.get_mut(x, 1, z) = LbmCell::new_gas(MaterialId::AIR, 0.85);
            }
        }

        // Snapshot density at center column before stepping
        let rho_upper_before = grid.get(6, 8, 6).density();

        // Run 30 LBM steps (not too many — catch the rising plume)
        step::lbm_step_n(&mut grid, &config, gravity, rho_ambient, None, 30);

        // The lighter air should have redistributed upward: density at upper
        // levels should be lower than ambient (lighter air has risen there)
        let rho_upper_after = grid.get(6, 8, 6).density();

        // Upper cell should have become lighter (lower density) as buoyant air arrived
        assert!(
            rho_upper_after < rho_upper_before - 1e-4,
            "Buoyant air should reduce upper density: before={rho_upper_before:.4}, after={rho_upper_after:.4}"
        );
    }

    #[test]
    fn convection_conserves_mass() {
        let size = 10;
        let mut grid = make_box(size);
        let config = test_config();
        let gravity = [0.0, -0.001, 0.0];
        let rho_ambient = 1.0;

        // Asymmetric initial density: hot spot near bottom
        *grid.get_mut(5, 2, 5) = LbmCell::new_gas(MaterialId::AIR, 0.7);
        *grid.get_mut(5, 7, 5) = LbmCell::new_gas(MaterialId::AIR, 1.3);

        let mass_before = grid.total_mass();

        step::lbm_step_n(&mut grid, &config, gravity, rho_ambient, None, 100);

        let mass_after = grid.total_mass();
        let rel_error = (mass_after - mass_before).abs() / mass_before;
        assert!(
            rel_error < 0.01,
            "Mass conservation violation: {mass_before} → {mass_after} (error {rel_error:.4})"
        );
    }

    #[test]
    fn density_perturbation_equilibrates() {
        // A localized density perturbation should spread out and reach
        // near-uniform density throughout the domain.
        let size = 10;
        let mut grid = make_box(size);
        let config = test_config();

        // Dense blob at center
        *grid.get_mut(5, 5, 5) = LbmCell::new_gas(MaterialId::AIR, 2.0);

        let initial_max = grid.get(5, 5, 5).density();

        // No gravity — pure pressure equilibration
        step::lbm_step_n(&mut grid, &config, [0.0; 3], 1.0, None, 200);

        // The peak density should have decreased significantly
        let final_max = grid.get(5, 5, 5).density();
        assert!(
            final_max < initial_max * 0.7,
            "Density perturbation should spread: {initial_max} → {final_max}"
        );
    }

    // --- Coriolis tests ---

    #[test]
    fn coriolis_deflects_eastward_wind() {
        // In the Northern Hemisphere, eastward wind should be deflected
        // southward (negative z in our coordinate system if z=north).
        // We use ω pointing along +y (up = north pole rotation axis).
        let size = 12;
        let mut grid = make_box(size);
        let config = test_config();

        // Set up initial east-west velocity (positive x)
        for x in 2..size - 2 {
            for y in 2..size - 2 {
                for z in 2..size - 2 {
                    let cell = grid.get_mut(x, y, z);
                    use crate::physics::lbm_gas::lattice;
                    cell.f = lattice::equilibrium(1.0, [0.02, 0.0, 0.0]);
                }
            }
        }

        // ω = rotation rate along y-axis (Northern Hemisphere, latitude ~45°)
        let omega = [0.0, 0.001, 0.0]; // Exaggerated for test visibility

        step::lbm_step_n(&mut grid, &config, [0.0; 3], 1.0, Some(omega), 30);

        // Measure z-velocity at center — should be deflected (nonzero)
        let center_vel = grid.get(6, 6, 6).velocity();
        let z_deflection = center_vel[2].abs();
        assert!(
            z_deflection > 1e-6,
            "Coriolis should deflect eastward wind laterally, got z_vel={:.2e}",
            center_vel[2]
        );
    }

    #[test]
    fn no_coriolis_when_disabled() {
        let size = 10;
        let mut grid = make_box(size);
        let config = test_config();

        // East-west wind
        for x in 2..size - 2 {
            for y in 2..size - 2 {
                for z in 2..size - 2 {
                    use crate::physics::lbm_gas::lattice;
                    grid.get_mut(x, y, z).f = lattice::equilibrium(1.0, [0.02, 0.0, 0.0]);
                }
            }
        }

        let _vel_before = grid.get(5, 5, 5).velocity();

        // Run without Coriolis
        step::lbm_step_n(&mut grid, &config, [0.0; 3], 1.0, None, 20);

        let vel_after = grid.get(5, 5, 5).velocity();
        // z-velocity should remain very small without Coriolis
        // Some numerical diffusion from wall interactions is expected
        assert!(
            vel_after[2].abs() < 0.005,
            "Without Coriolis, lateral deflection should be negligible: z_vel={}",
            vel_after[2]
        );
    }

    // --- Moisture / humidity tests ---

    #[test]
    fn evaporation_from_water_surface() {
        let size = 8;
        let mut grid = make_box(size);

        // Place a water surface at y=1
        for x in 1..size - 1 {
            for z in 1..size - 1 {
                *grid.get_mut(x, 1, z) = LbmCell::new_liquid(MaterialId::WATER);
            }
        }

        let n = size * size * size;
        let temps = vec![288.15_f32; n]; // ~15°C
        let pressures = vec![101_325.0_f32; n];
        let atm = AtmosphereConfig::default();

        let moisture_before = moisture::total_moisture(&grid);

        moisture::evaporate(&mut grid, &temps, &pressures, 1.0, &atm);

        let moisture_after = moisture::total_moisture(&grid);
        assert!(
            moisture_after > moisture_before,
            "Evaporation should increase total moisture: {moisture_before} → {moisture_after}"
        );

        // Cells directly above water should have gained the most moisture
        let above_water = grid.get(4, 2, 4).moisture;
        let far_from_water = grid.get(4, 5, 4).moisture;
        assert!(
            above_water > far_from_water,
            "Moisture should be highest near water surface: above={above_water}, far={far_from_water}"
        );
    }

    #[test]
    fn condensation_at_saturation() {
        let size = 8;
        let mut grid = make_box(size);

        // Supersaturate a cell: give it more moisture than saturation
        let temp = 280.0; // ~7°C
        let pres = 101_325.0;
        let q_sat = atmosphere::saturation_humidity(temp, pres);
        let q_excess = q_sat * 2.0; // Double saturation

        grid.get_mut(4, 4, 4).moisture = q_excess;

        let n = size * size * size;
        let temps = vec![temp; n];
        let pressures = vec![pres; n];

        let lwc_before = grid.get(4, 4, 4).cloud_lwc;
        assert_eq!(lwc_before, 0.0, "Should start with no cloud water");

        let temp_deltas = moisture::condense(&mut grid, &temps, &pressures, 1.0);

        let lwc_after = grid.get(4, 4, 4).cloud_lwc;
        assert!(
            lwc_after > 0.0,
            "Condensation should produce cloud LWC: {lwc_after}"
        );

        // Latent heat release should warm the air
        let idx = grid.index(4, 4, 4);
        assert!(
            temp_deltas[idx] > 0.0,
            "Condensation should release latent heat: ΔT={}",
            temp_deltas[idx]
        );

        // Moisture should decrease
        let q_after = grid.get(4, 4, 4).moisture;
        assert!(
            q_after < q_excess,
            "Moisture should decrease after condensation: {q_excess} → {q_after}"
        );
    }

    #[test]
    fn cloud_reevaporates_in_dry_warm_air() {
        let size = 8;
        let mut grid = make_box(size);

        // Place cloud LWC in a cell with zero moisture and warm temperature
        // (far below saturation at this temperature)
        grid.get_mut(4, 4, 4).cloud_lwc = 0.001;
        grid.get_mut(4, 4, 4).moisture = 0.0;

        let temp = 300.0; // 27°C — high saturation capacity
        let pres = 101_325.0;
        let n = size * size * size;
        let temps = vec![temp; n];
        let pressures = vec![pres; n];

        let temp_deltas = moisture::condense(&mut grid, &temps, &pressures, 1.0);

        // Cloud should re-evaporate (LWC decreases, moisture increases)
        let lwc_after = grid.get(4, 4, 4).cloud_lwc;
        assert!(
            lwc_after < 0.001,
            "Cloud should re-evaporate in dry air: {lwc_after}"
        );

        let q_after = grid.get(4, 4, 4).moisture;
        assert!(
            q_after > 0.0,
            "Re-evaporation should add moisture: {q_after}"
        );

        // Should absorb heat (cool the air)
        let idx = grid.index(4, 4, 4);
        assert!(
            temp_deltas[idx] < 0.0,
            "Re-evaporation should cool the air: ΔT={}",
            temp_deltas[idx]
        );
    }

    // --- Moisture + advection integration ---

    #[test]
    fn moisture_advected_by_wind() {
        // Set up a moist region on one side and wind blowing across.
        // After several LBM steps with moisture transport, the moisture
        // should spread in the wind direction.
        let size = 12;
        let mut grid = make_box(size);
        let config = test_config();

        // Uniform rightward wind
        for x in 1..size - 1 {
            for y in 1..size - 1 {
                for z in 1..size - 1 {
                    use crate::physics::lbm_gas::lattice;
                    grid.get_mut(x, y, z).f = lattice::equilibrium(1.0, [0.01, 0.0, 0.0]);
                }
            }
        }

        // Place moisture on the left side (x = 2..4)
        for x in 2..4 {
            for y in 2..size - 2 {
                for z in 2..size - 2 {
                    grid.get_mut(x, y, z).moisture = 0.01;
                }
            }
        }

        let right_moisture_before: f32 = {
            let mut sum = 0.0_f32;
            for x in 6..size - 2 {
                for y in 2..size - 2 {
                    for z in 2..size - 2 {
                        sum += grid.get(x, y, z).moisture;
                    }
                }
            }
            sum
        };
        assert_eq!(right_moisture_before, 0.0, "Right side should start dry");

        // Run LBM steps — moisture is advected as passive scalar in streaming
        for _ in 0..40 {
            step::lbm_step(&mut grid, &config, [0.0; 3], 1.0, None);
        }

        // Check moisture has been transported rightward
        let right_moisture_after: f32 = {
            let mut sum = 0.0_f32;
            for x in 6..size - 2 {
                for y in 2..size - 2 {
                    for z in 2..size - 2 {
                        sum += grid.get(x, y, z).moisture;
                    }
                }
            }
            sum
        };

        assert!(
            right_moisture_after > 1e-6,
            "Moisture should be advected to the right side by wind: {right_moisture_after}"
        );
    }

    // --- Full cycle: evaporation → cloud → precipitation ---

    #[test]
    fn full_moisture_cycle_evaporate_condense_precipitate() {
        let size = 10;
        let mut grid = make_box(size);
        let atm = AtmosphereConfig::default();

        // Water surface at y=1
        for x in 1..size - 1 {
            for z in 1..size - 1 {
                *grid.get_mut(x, 1, z) = LbmCell::new_liquid(MaterialId::WATER);
            }
        }

        let n = size * size * size;
        let warm_temp = 288.15_f32; // 15°C surface
        let cool_temp = 265.0_f32; // -8°C aloft (below freezing)

        // Temperature profile: warm at bottom, cold at top
        let temps: Vec<f32> = (0..n)
            .map(|idx| {
                let y = (idx / size) % size;
                let frac = y as f32 / (size - 1) as f32;
                warm_temp * (1.0 - frac) + cool_temp * frac
            })
            .collect();
        let pressures = vec![101_325.0_f32; n];

        // Step 1: Evaporate moisture from water surface
        for _ in 0..20 {
            moisture::evaporate(&mut grid, &temps, &pressures, 1.0, &atm);
        }
        let total_q = moisture::total_moisture(&grid);
        assert!(total_q > 0.0, "Should have evaporated some moisture");

        // Step 2: Directly set supersaturated moisture at high altitude
        // (simulating updraft carrying moisture to cold altitude)
        let cold_temp = temps[grid.index(5, 7, 5)];
        let q_sat_cold = atmosphere::saturation_humidity(cold_temp, 101_325.0);
        for x in 2..size - 2 {
            for z in 2..size - 2 {
                // Set moisture well above saturation at this temperature
                grid.get_mut(x, 7, z).moisture = q_sat_cold * 5.0;
            }
        }

        // Step 3: Condense at cold altitude
        let temp_deltas = moisture::condense(&mut grid, &temps, &pressures, 1.0);

        let total_lwc = moisture::total_cloud_lwc(&grid);
        assert!(
            total_lwc > 0.0,
            "Supersaturated moisture at cold altitude should condense into cloud: lwc={total_lwc}"
        );

        // Verify latent heat was released somewhere
        let heat_released: f32 = temp_deltas.iter().filter(|&&d| d > 0.0).sum();
        assert!(
            heat_released > 0.0,
            "Condensation should release latent heat: ΣΔT={heat_released}"
        );

        // Step 4: If cloud exceeds threshold, precipitate
        // Boost cloud LWC above coalescence threshold for a cell
        let boosted_cell = grid.index(5, 7, 5);
        grid.cells_mut()[boosted_cell].cloud_lwc = atm.cloud_coalescence_threshold * 5.0;

        let mut particles = Vec::new();
        let emitted = precipitation::precipitate(&mut grid, &temps, &atm, &mut particles, 1.0, 0);

        assert!(
            emitted > 0,
            "Should emit precipitation particles from thick cloud"
        );

        // Verify snow (since T at y=7 is below freezing)
        let y7_temp = temps[grid.index(5, 7, 5)];
        if y7_temp < 273.15 {
            assert_eq!(
                particles[0].material,
                MaterialId::ICE,
                "Below freezing should produce snow"
            );
        }
    }

    // --- Stability tests ---

    #[test]
    fn lbm_remains_stable_with_buoyancy_100_steps() {
        let size = 12;
        let mut grid = make_box(size);
        let config = test_config();
        let gravity = [0.0, -0.001, 0.0];
        let rho_ambient = 1.0;

        // Strong initial perturbation
        *grid.get_mut(6, 2, 6) = LbmCell::new_gas(MaterialId::AIR, 0.5);
        *grid.get_mut(6, 9, 6) = LbmCell::new_gas(MaterialId::AIR, 1.5);

        step::lbm_step_n(&mut grid, &config, gravity, rho_ambient, None, 100);

        // Check stability: no NaN or Inf, Mach number reasonable
        let mach = step::max_mach_number(&grid);
        assert!(
            mach.is_finite() && mach < 0.5,
            "Should remain stable after 100 steps: Ma={mach}"
        );

        for cell in grid.cells() {
            if cell.is_gas() {
                let rho = cell.density();
                assert!(
                    rho.is_finite() && rho > 0.0,
                    "Density should be finite positive"
                );
                let u = cell.velocity();
                for c in &u {
                    assert!(c.is_finite(), "Velocity should be finite");
                }
            }
        }
    }

    #[test]
    fn moisture_never_goes_negative() {
        let size = 8;
        let mut grid = make_box(size);

        // Start with some moisture
        for cell in grid.cells_mut() {
            if cell.is_gas() {
                cell.moisture = 0.001;
            }
        }

        let n = size * size * size;
        let temps = vec![300.0_f32; n]; // Warm → high saturation
        let pressures = vec![101_325.0_f32; n];

        // Repeated condensation should never make moisture negative
        for _ in 0..50 {
            moisture::condense(&mut grid, &temps, &pressures, 0.1);
        }

        for cell in grid.cells() {
            if cell.is_gas() {
                assert!(
                    cell.moisture >= 0.0,
                    "Moisture should never go negative: {}",
                    cell.moisture
                );
                assert!(
                    cell.cloud_lwc >= 0.0,
                    "Cloud LWC should never go negative: {}",
                    cell.cloud_lwc
                );
            }
        }
    }

    #[test]
    fn coriolis_deflects_but_does_not_amplify() {
        // Coriolis should redirect flow laterally but not significantly
        // increase total kinetic energy compared to the same run without Coriolis.
        let size = 10;
        let config = test_config();

        // Run WITH Coriolis
        let mut grid_with = make_box(size);
        for x in 1..size - 1 {
            for y in 1..size - 1 {
                for z in 1..size - 1 {
                    use crate::physics::lbm_gas::lattice;
                    grid_with.get_mut(x, y, z).f = lattice::equilibrium(1.0, [0.02, 0.0, 0.0]);
                }
            }
        }
        let omega = [0.0, 0.0005, 0.0];
        step::lbm_step_n(&mut grid_with, &config, [0.0; 3], 1.0, Some(omega), 20);
        let ke_with = kinetic_energy(&grid_with);

        // Run WITHOUT Coriolis (same initial conditions)
        let mut grid_without = make_box(size);
        for x in 1..size - 1 {
            for y in 1..size - 1 {
                for z in 1..size - 1 {
                    use crate::physics::lbm_gas::lattice;
                    grid_without.get_mut(x, y, z).f = lattice::equilibrium(1.0, [0.02, 0.0, 0.0]);
                }
            }
        }
        step::lbm_step_n(&mut grid_without, &config, [0.0; 3], 1.0, None, 20);
        let ke_without = kinetic_energy(&grid_without);

        // Coriolis shouldn't amplify KE significantly vs no-Coriolis baseline
        let ratio = ke_with / ke_without.max(1e-12);
        assert!(
            ratio < 1.5,
            "Coriolis should not amplify KE: with={ke_with:.4}, without={ke_without:.4}, ratio={ratio:.2}"
        );
    }

    #[test]
    fn precipitation_mass_conservation_full_cycle() {
        let size = 8;
        let mut grid = make_box(size);
        let atm = AtmosphereConfig::default();
        let n = size * size * size;
        let temps = vec![285.0_f32; n]; // Warm (rain, not snow)
        let pressures = vec![101_325.0_f32; n];

        // Set up: moisture → condense → precipitate
        for x in 2..size - 2 {
            for y in 3..6 {
                for z in 2..size - 2 {
                    // Supersaturate
                    let q_sat = atmosphere::saturation_humidity(285.0, 101_325.0);
                    grid.get_mut(x, y, z).moisture = q_sat * 3.0;
                }
            }
        }

        let total_q_before = moisture::total_moisture(&grid);

        // Condense
        moisture::condense(&mut grid, &temps, &pressures, 1.0);
        let total_lwc = moisture::total_cloud_lwc(&grid);
        let total_q_after = moisture::total_moisture(&grid);

        // Moisture lost should roughly equal cloud LWC gained (accounting for density conversion)
        // This is approximate due to mixing ratio → LWC density conversion
        assert!(total_q_after < total_q_before, "Moisture should decrease");
        assert!(total_lwc > 0.0, "Cloud LWC should increase");

        // Precipitate
        let lwc_before_precip = moisture::total_cloud_lwc(&grid);
        let mut particles = Vec::new();
        precipitation::precipitate(&mut grid, &temps, &atm, &mut particles, 1.0, 0);
        let lwc_after_precip = moisture::total_cloud_lwc(&grid);

        if !particles.is_empty() {
            let particle_mass: f32 = particles.iter().map(|p| p.mass).sum();
            let lwc_removed = lwc_before_precip - lwc_after_precip;

            // Mass removed from clouds should equal mass in particles
            assert!(
                (lwc_removed - particle_mass).abs() < 1e-4,
                "Precipitation mass conservation: removed={lwc_removed:.6}, particles={particle_mass:.6}"
            );
        }
    }

    // --- Helper ---

    fn kinetic_energy(grid: &LbmGrid) -> f32 {
        grid.cells()
            .iter()
            .filter(|c| c.is_gas())
            .map(|c| {
                let rho = c.density();
                let u = c.velocity();
                0.5 * rho * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2])
            })
            .sum()
    }
}
