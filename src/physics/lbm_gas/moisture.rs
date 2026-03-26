// Moisture transport: evaporation, condensation, and latent heat.
//
// Each LBM gas cell carries a `moisture` scalar (kg water vapor / kg dry air)
// and a `cloud_lwc` scalar (liquid water content, kg/m³). The streaming step
// advects both as passive scalars. This module handles source/sink terms:
//
//   - Evaporation: liquid/water voxels adjacent to unsaturated gas cells add
//     moisture at a rate driven by the Clausius-Clapeyron deficit.
//   - Condensation: when moisture exceeds saturation, excess condenses into
//     cloud_lwc and releases latent heat (warming the air).
//   - Re-evaporation: cloud_lwc in subsaturated air re-evaporates (cooling).
//
// All values in SI units. Temperatures in Kelvin, pressures in Pascals.

use super::types::LbmGrid;
use crate::physics::atmosphere::{self, AtmosphereConfig, LATENT_HEAT_VAPORIZATION, R_DRY_AIR};

/// Rate coefficient for surface evaporation (m/s).
/// Controls how fast moisture is added from water surfaces.
/// Derived from bulk aerodynamic formula: E = C_E × |u| × (q_s - q).
/// Using a default wind-independent baseline rate.
const EVAPORATION_RATE: f32 = 1.0e-4;

/// Apply evaporation from liquid surfaces adjacent to gas cells.
///
/// For each gas cell with a liquid neighbor (water, lava), add moisture
/// proportional to the saturation deficit: `Δq = rate × (q_sat - q) × dt`.
///
/// Also injects baseline humidity into completely dry interior cells.
pub fn evaporate(
    grid: &mut LbmGrid,
    temperatures: &[f32],
    pressures: &[f32],
    dt: f32,
    _config: &AtmosphereConfig,
) {
    let size = grid.size();
    let dirs: [(i32, i32, i32); 6] = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ];

    // Collect evaporation deltas to avoid borrow conflicts
    let mut deltas = vec![0.0f32; size * size * size];

    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let idx = grid.index(x, y, z);
                let cell = grid.get(x, y, z);
                if !cell.is_gas() {
                    continue;
                }

                let temp = temperatures[idx];
                let pres = pressures[idx];
                let q_sat = atmosphere::saturation_humidity(temp, pres);
                let q = cell.moisture;

                if q >= q_sat {
                    continue; // Already saturated
                }

                // Check neighbors for liquid surfaces
                let mut has_liquid_neighbor = false;
                for (dx, dy, dz) in &dirs {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    let nz = z as i32 + dz;
                    if nx < 0
                        || nx >= size as i32
                        || ny < 0
                        || ny >= size as i32
                        || nz < 0
                        || nz >= size as i32
                    {
                        continue;
                    }
                    let neighbor = grid.get(nx as usize, ny as usize, nz as usize);
                    if neighbor.tag == super::types::GasCellTag::Liquid {
                        has_liquid_neighbor = true;
                        break;
                    }
                }

                if has_liquid_neighbor {
                    let deficit = q_sat - q;
                    deltas[idx] = (EVAPORATION_RATE * deficit * dt).min(deficit);
                }
            }
        }
    }

    // Apply deltas
    for (idx, delta) in deltas.iter().enumerate() {
        if *delta > 0.0 {
            grid.cells_mut()[idx].moisture += delta;
        }
    }
}

/// Apply condensation and re-evaporation of cloud liquid water.
///
/// When moisture exceeds saturation: excess condenses into cloud_lwc,
/// releasing latent heat (ΔT = L_v × Δq / Cₚ_air).
///
/// When moisture is below saturation and cloud_lwc > 0: cloud water
/// re-evaporates, absorbing heat.
///
/// Returns temperature adjustments (Kelvin) to be applied to voxels.
pub fn condense(grid: &mut LbmGrid, temperatures: &[f32], pressures: &[f32], dt: f32) -> Vec<f32> {
    let n = grid.cells().len();
    let mut temp_deltas = vec![0.0f32; n];

    // Specific heat of dry air at constant pressure, J/(kg·K)
    const CP_AIR: f32 = 1005.0;

    for idx in 0..n {
        let cell = &grid.cells()[idx];
        if !cell.is_gas() {
            continue;
        }

        let temp = temperatures[idx];
        let pres = pressures[idx];
        let q_sat = atmosphere::saturation_humidity(temp, pres);
        let q = cell.moisture;

        if q > q_sat {
            // Supersaturated: condense excess into cloud LWC
            // Rate-limited to avoid overshooting in one step
            let excess = q - q_sat;
            let condense_rate = excess.min(excess * dt * 0.5 + 1e-6);
            let condensed = condense_rate.min(q); // Can't condense more than available

            grid.cells_mut()[idx].moisture -= condensed;
            // Convert mixing ratio to LWC: Δ_lwc = condensed × ρ_air
            // Use approximate air density from pressure: ρ = P / (R_d × T)
            let rho_air = pres / (R_DRY_AIR * temp);
            grid.cells_mut()[idx].cloud_lwc += condensed * rho_air;

            // Latent heat release: ΔT = L_v × condensed / Cₚ
            temp_deltas[idx] = LATENT_HEAT_VAPORIZATION * condensed / CP_AIR;
        } else if cell.cloud_lwc > 0.0 {
            // Subsaturated with cloud water: re-evaporate
            let deficit = q_sat - q;
            let rho_air = pres / (R_DRY_AIR * temp);
            // Convert available LWC back to mixing ratio space
            let available_q = cell.cloud_lwc / rho_air.max(0.01);
            let evaporated = deficit.min(available_q).min(available_q * dt * 0.5 + 1e-6);

            grid.cells_mut()[idx].moisture += evaporated;
            grid.cells_mut()[idx].cloud_lwc -= evaporated * rho_air;
            grid.cells_mut()[idx].cloud_lwc = grid.cells_mut()[idx].cloud_lwc.max(0.0);

            // Latent heat absorption: cools the air
            temp_deltas[idx] = -LATENT_HEAT_VAPORIZATION * evaporated / CP_AIR;
        }
    }

    temp_deltas
}

/// Total moisture (vapor) in the grid. Used for conservation checks.
pub fn total_moisture(grid: &LbmGrid) -> f32 {
    grid.cells()
        .iter()
        .filter(|c| c.is_gas())
        .map(|c| c.moisture)
        .sum()
}

/// Total cloud liquid water content in the grid.
pub fn total_cloud_lwc(grid: &LbmGrid) -> f32 {
    grid.cells()
        .iter()
        .filter(|c| c.is_gas())
        .map(|c| c.cloud_lwc)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::lbm_gas::types::LbmCell;
    use crate::world::voxel::MaterialId;

    fn default_config() -> AtmosphereConfig {
        AtmosphereConfig::default()
    }

    fn make_box_grid(size: usize) -> LbmGrid {
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

    #[test]
    fn evaporation_adds_moisture_near_liquid() {
        let size = 8;
        let mut grid = make_box_grid(size);

        // Place liquid water at y=1 (bottom interior)
        for z in 1..size - 1 {
            for x in 1..size - 1 {
                *grid.get_mut(x, 1, z) = LbmCell::new_liquid(MaterialId::WATER);
            }
        }

        let n = size * size * size;
        let temps = vec![288.15_f32; n]; // 15°C ambient
        let pres = vec![101325.0_f32; n];
        let config = default_config();

        let moisture_before = total_moisture(&grid);
        evaporate(&mut grid, &temps, &pres, 1.0, &config);
        let moisture_after = total_moisture(&grid);

        assert!(
            moisture_after > moisture_before,
            "Evaporation should add moisture: {moisture_before} → {moisture_after}"
        );

        // Gas cells at y=2 (just above water) should have moisture
        let cell = grid.get(4, 2, 4);
        assert!(
            cell.moisture > 0.0,
            "Cell above water should have moisture: {}",
            cell.moisture
        );
    }

    #[test]
    fn no_evaporation_without_liquid_neighbor() {
        let size = 6;
        let mut grid = make_box_grid(size);

        let n = size * size * size;
        let temps = vec![288.15_f32; n];
        let pres = vec![101325.0_f32; n];
        let config = default_config();

        evaporate(&mut grid, &temps, &pres, 1.0, &config);

        // No liquid neighbors → no moisture change
        let cell = grid.get(3, 3, 3);
        assert!(
            cell.moisture < 1e-10,
            "Should have no moisture without liquid: {}",
            cell.moisture
        );
    }

    #[test]
    fn condensation_removes_moisture_adds_cloud() {
        let size = 6;
        let mut grid = make_box_grid(size);

        // Set supersaturated moisture at center
        let temp = 280.0; // Cool temperature → low saturation
        let pres = 101325.0;
        let q_sat = atmosphere::saturation_humidity(temp, pres);
        let q_super = q_sat * 2.0; // 200% of saturation

        grid.get_mut(3, 3, 3).moisture = q_super;

        let n = size * size * size;
        let temps = vec![temp; n];
        let pressures = vec![pres; n];

        let temp_deltas = condense(&mut grid, &temps, &pressures, 1.0);

        let cell = grid.get(3, 3, 3);
        assert!(
            cell.moisture < q_super,
            "Condensation should reduce moisture: {} → {}",
            q_super,
            cell.moisture
        );
        assert!(
            cell.cloud_lwc > 0.0,
            "Condensation should produce cloud LWC: {}",
            cell.cloud_lwc
        );

        let idx = grid.index(3, 3, 3);
        assert!(
            temp_deltas[idx] > 0.0,
            "Condensation should release latent heat: {}",
            temp_deltas[idx]
        );
    }

    #[test]
    fn cloud_reevaporates_in_dry_air() {
        let size = 6;
        let mut grid = make_box_grid(size);

        // Place cloud LWC in a subsaturated cell
        let temp = 300.0; // Warm → high saturation capacity
        let pres = 101325.0;

        let cell = grid.get_mut(3, 3, 3);
        cell.moisture = 0.001; // Well below saturation at 300K
        cell.cloud_lwc = 0.5; // Cloud present

        let n = size * size * size;
        let temps = vec![temp; n];
        let pressures = vec![pres; n];

        let lwc_before = grid.get(3, 3, 3).cloud_lwc;
        let temp_deltas = condense(&mut grid, &temps, &pressures, 1.0);

        let cell = grid.get(3, 3, 3);
        assert!(
            cell.cloud_lwc < lwc_before,
            "Cloud should re-evaporate: {lwc_before} → {}",
            cell.cloud_lwc
        );
        assert!(
            cell.moisture > 0.001,
            "Re-evaporation should add moisture: {}",
            cell.moisture
        );

        let idx = grid.index(3, 3, 3);
        assert!(
            temp_deltas[idx] < 0.0,
            "Re-evaporation should absorb heat: {}",
            temp_deltas[idx]
        );
    }

    #[test]
    fn moisture_advection_conserves_total() {
        use crate::physics::lbm_gas::streaming;

        let size = 8;
        let mut grid = make_box_grid(size);

        // Set moisture in center with a velocity
        let cell = grid.get_mut(4, 4, 4);
        cell.moisture = 1.0;
        // Give it eastward velocity via asymmetric distributions
        cell.f[1] += 0.05; // +x boost

        let moisture_before = total_moisture(&grid);
        let streamed = streaming::stream(&grid);
        let moisture_after = total_moisture(&streamed);

        // Conservation within tolerance (closed box with walls)
        assert!(
            (moisture_after - moisture_before).abs() < 0.01,
            "Moisture not conserved: {moisture_before} → {moisture_after}"
        );
    }

    #[test]
    fn moisture_advects_with_flow() {
        use crate::physics::lbm_gas::streaming;

        let size = 8;
        let mut grid = make_box_grid(size);

        // Place moisture at (2,4,4) with eastward velocity
        let cell = grid.get_mut(2, 4, 4);
        cell.moisture = 1.0;
        // Boost +x distribution to create velocity
        cell.f[1] += 0.1;

        // Also set moisture=0 everywhere else explicitly
        for z in 1..size - 1 {
            for y in 1..size - 1 {
                for x in 1..size - 1 {
                    if x != 2 || y != 4 || z != 4 {
                        grid.get_mut(x, y, z).moisture = 0.0;
                    }
                }
            }
        }

        let streamed = streaming::stream(&grid);

        // Some moisture should have moved eastward (to x=3)
        let east_moisture = streamed.get(3, 4, 4).moisture;
        let origin_moisture = streamed.get(2, 4, 4).moisture;

        assert!(
            east_moisture > 0.0,
            "Moisture should advect eastward: east={east_moisture}"
        );
        assert!(
            origin_moisture < 1.0,
            "Moisture should leave origin: origin={origin_moisture}"
        );
    }
}
