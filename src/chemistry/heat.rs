// Heat transfer using Fourier's law on a discrete voxel grid.
//
// Each tick, heat flux between adjacent voxels is computed as:
//   Q = k_eff * (T_neighbor - T_self)
// where k_eff is the harmonic mean conductivity at the interface (A=1m², dx=1m).
//
// Temperature update per voxel:
//   ΔT = Σ(Q_neighbors) * dt / (ρ * Cₚ)      [V=1m³]
//
// Thermal properties (k, ρ, Cₚ) are read from MaterialData via MaterialRegistry.
// CFL stability: dt < dx² / (6 × α_max) where α = k/(ρ×Cₚ).
// For iron (α ≈ 2.27e-5 m²/s) with dx=1m: dt_max ≈ 7300s — stable at game timesteps.

#![allow(dead_code)]

use crate::data::MaterialRegistry;
use crate::world::voxel::{MaterialId, Voxel};

/// Legacy hardcoded conductivity lookup (normalized 0–1 scale).
/// Use `thermal_conductivity()` with a `MaterialRegistry` instead.
#[deprecated(note = "Use thermal_conductivity() with MaterialRegistry for SI values")]
pub fn conductivity(material: MaterialId) -> f32 {
    match material.0 {
        0 => 0.02,  // air
        1 => 0.8,   // stone
        2 => 0.3,   // dirt
        3 => 0.6,   // water
        4 => 1.0,   // iron
        5 => 0.15,  // wood
        6 => 0.25,  // sand
        7 => 0.1,   // grass
        8 => 0.5,   // ice
        9 => 0.03,  // steam
        10 => 0.9,  // lava
        11 => 0.15, // ash
        _ => 0.1,
    }
}

/// Look up thermal conductivity from MaterialData registry.
/// Returns the SI value in W/(m·K), or a small default for unknown materials.
pub fn thermal_conductivity(material: MaterialId, registry: &MaterialRegistry) -> f32 {
    registry
        .get(material)
        .map(|m| m.thermal_conductivity)
        .unwrap_or(0.1)
}

/// Compute new temperature for a voxel using Fourier's law.
///
/// Parameters:
///   - `current_temp`: this voxel's temperature (K)
///   - `self_k`: this voxel's thermal conductivity W/(m·K)
///   - `self_rho_cp`: this voxel's ρ×Cₚ (kg/m³ × J/(kg·K) = J/(m³·K))
///   - `neighbors`: slice of (temperature, conductivity) for adjacent voxels
///   - `dt`: timestep in seconds
pub fn diffuse_temperature(
    current_temp: f32,
    self_k: f32,
    self_rho_cp: f32,
    neighbors: &[(f32, f32)],
    dt: f32,
) -> f32 {
    if neighbors.is_empty() || self_rho_cp <= 0.0 {
        return current_temp;
    }

    let mut total_heat_flux = 0.0;
    for &(neighbor_temp, neighbor_k) in neighbors {
        // Harmonic mean conductivity at the interface
        let k_eff = if self_k + neighbor_k > 0.0 {
            2.0 * self_k * neighbor_k / (self_k + neighbor_k)
        } else {
            0.0
        };
        // Q = k_eff * (T_neighbor - T_self) [W, since A=1m², dx=1m]
        total_heat_flux += k_eff * (neighbor_temp - current_temp);
    }

    // ΔT = Q_total * dt / (ρ × Cₚ)  [since V=1m³]
    current_temp + total_heat_flux * dt / self_rho_cp
}

/// Apply one heat diffusion step to a flat 3D voxel array of size `size³`
/// using Fourier's law with material properties from the registry.
///
/// `dt` is the timestep in seconds (e.g. 1.0/60.0 for 60 Hz).
/// Returns the new temperature buffer.
pub fn diffuse_chunk(
    voxels: &[Voxel],
    size: usize,
    dt: f32,
    registry: &MaterialRegistry,
) -> Vec<f32> {
    let len = size * size * size;
    assert_eq!(voxels.len(), len);

    let mut new_temps = Vec::with_capacity(len);

    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let idx = z * size * size + y * size + x;
                let voxel = &voxels[idx];

                let mat = registry.get(voxel.material);
                let self_k = mat.map(|m| m.thermal_conductivity).unwrap_or(0.1);
                let density = mat.map(|m| m.density).unwrap_or(1.0);
                let cp = mat.map(|m| m.specific_heat_capacity).unwrap_or(1000.0);
                let self_rho_cp = density * cp;

                let mut neighbors = Vec::with_capacity(6);
                // ±X
                if x > 0 {
                    let n = &voxels[idx - 1];
                    neighbors.push((n.temperature, thermal_conductivity(n.material, registry)));
                }
                if x + 1 < size {
                    let n = &voxels[idx + 1];
                    neighbors.push((n.temperature, thermal_conductivity(n.material, registry)));
                }
                // ±Y
                if y > 0 {
                    let n = &voxels[idx - size];
                    neighbors.push((n.temperature, thermal_conductivity(n.material, registry)));
                }
                if y + 1 < size {
                    let n = &voxels[idx + size];
                    neighbors.push((n.temperature, thermal_conductivity(n.material, registry)));
                }
                // ±Z
                if z > 0 {
                    let n = &voxels[idx - size * size];
                    neighbors.push((n.temperature, thermal_conductivity(n.material, registry)));
                }
                if z + 1 < size {
                    let n = &voxels[idx + size * size];
                    neighbors.push((n.temperature, thermal_conductivity(n.material, registry)));
                }

                new_temps.push(diffuse_temperature(
                    voxel.temperature,
                    self_k,
                    self_rho_cp,
                    &neighbors,
                    dt,
                ));
            }
        }
    }

    new_temps
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{MaterialData, Phase};

    const AIR_ID: u16 = 0;
    const STONE_ID: u16 = 1;
    const WATER_ID: u16 = 3;
    const IRON_ID: u16 = 4;

    const DT: f32 = 1.0 / 60.0;

    fn test_registry() -> MaterialRegistry {
        let mut reg = MaterialRegistry::new();
        reg.insert(MaterialData {
            id: AIR_ID,
            name: "Air".into(),
            default_phase: Phase::Gas,
            density: 1.225,
            thermal_conductivity: 0.026,
            specific_heat_capacity: 1005.0,
            ..Default::default()
        });
        reg.insert(MaterialData {
            id: STONE_ID,
            name: "Stone".into(),
            default_phase: Phase::Solid,
            density: 2700.0,
            thermal_conductivity: 2.5,
            specific_heat_capacity: 790.0,
            ..Default::default()
        });
        reg.insert(MaterialData {
            id: WATER_ID,
            name: "Water".into(),
            default_phase: Phase::Liquid,
            density: 1000.0,
            thermal_conductivity: 0.606,
            specific_heat_capacity: 4186.0,
            ..Default::default()
        });
        reg.insert(MaterialData {
            id: IRON_ID,
            name: "Iron".into(),
            default_phase: Phase::Solid,
            density: 7874.0,
            thermal_conductivity: 80.2,
            specific_heat_capacity: 449.0,
            ..Default::default()
        });
        reg
    }

    #[test]
    fn uniform_temperature_stays_stable() {
        let _reg = test_registry();
        let k = 2.5; // stone
        let rho_cp = 2700.0 * 790.0;
        let neighbors = vec![(293.0, k), (293.0, k), (293.0, k)];
        let result = diffuse_temperature(293.0, k, rho_cp, &neighbors, DT);
        assert!(
            (result - 293.0).abs() < f32::EPSILON,
            "Uniform temp should be stable, got {result}"
        );
    }

    #[test]
    fn heat_flows_from_hot_to_cold() {
        let k = 80.2; // iron
        let rho_cp = 7874.0 * 449.0;
        let neighbors = vec![(200.0, k), (200.0, k)];
        let result = diffuse_temperature(400.0, k, rho_cp, &neighbors, DT);
        assert!(result < 400.0, "Hot voxel should cool down, got {result}");

        let cold_neighbors = vec![(500.0, k), (500.0, k)];
        let result2 = diffuse_temperature(200.0, k, rho_cp, &cold_neighbors, DT);
        assert!(result2 > 200.0, "Cold voxel should warm up, got {result2}");
    }

    #[test]
    fn no_neighbors_keeps_temperature() {
        let result = diffuse_temperature(350.0, 80.2, 7874.0 * 449.0, &[], DT);
        assert_eq!(result, 350.0);
    }

    #[test]
    fn zero_rho_cp_keeps_temperature() {
        let neighbors = vec![(500.0, 80.2)];
        let result = diffuse_temperature(350.0, 80.2, 0.0, &neighbors, DT);
        assert_eq!(result, 350.0, "Zero ρCₚ should not change temperature");
    }

    #[test]
    fn iron_conducts_faster_than_air() {
        // Iron hot spot: 3-voxel line, center hot
        let iron_k = 80.2_f32;
        let iron_rho_cp = 7874.0 * 449.0;
        let iron_neighbors = vec![(288.15, iron_k), (288.15, iron_k)];
        let iron_result = diffuse_temperature(1000.0, iron_k, iron_rho_cp, &iron_neighbors, DT);
        let iron_delta = (1000.0 - iron_result).abs();

        // Air hot spot: same geometry
        let air_k = 0.026_f32;
        let air_rho_cp = 1.225 * 1005.0;
        let air_neighbors = vec![(288.15, air_k), (288.15, air_k)];
        let air_result = diffuse_temperature(1000.0, air_k, air_rho_cp, &air_neighbors, DT);
        let air_delta = (1000.0 - air_result).abs();

        // Iron should transfer more heat per step (higher thermal diffusivity)
        assert!(
            iron_delta > air_delta,
            "Iron (ΔT={iron_delta:.4}) should conduct faster than air (ΔT={air_delta:.4})"
        );
    }

    #[test]
    fn energy_conservation_in_closed_system() {
        let reg = test_registry();
        let size = 4;
        let mut voxels: Vec<Voxel> = (0..size * size * size)
            .map(|_| Voxel::new(MaterialId::STONE))
            .collect();
        // Hot spot in center
        let center = size * size + size + 1;
        voxels[center].temperature = 1000.0;

        // Compute total thermal energy: Σ(ρ × Cₚ × T)
        let energy_before: f64 = voxels
            .iter()
            .map(|v| {
                let mat = reg.get(v.material);
                let rho = mat.map(|m| m.density).unwrap_or(1.0) as f64;
                let cp = mat.map(|m| m.specific_heat_capacity).unwrap_or(1000.0) as f64;
                rho * cp * v.temperature as f64
            })
            .sum();

        let new_temps = diffuse_chunk(&voxels, size, DT, &reg);

        let energy_after: f64 = voxels
            .iter()
            .zip(new_temps.iter())
            .map(|(v, &t)| {
                let mat = reg.get(v.material);
                let rho = mat.map(|m| m.density).unwrap_or(1.0) as f64;
                let cp = mat.map(|m| m.specific_heat_capacity).unwrap_or(1000.0) as f64;
                rho * cp * t as f64
            })
            .sum();

        let relative_error = ((energy_after - energy_before) / energy_before).abs();
        assert!(
            relative_error < 1e-5,
            "Energy should be conserved. Before: {energy_before:.2}, After: {energy_after:.2}, \
             relative error: {relative_error:.2e}"
        );
    }

    #[test]
    fn water_high_heat_capacity_resists_change() {
        // Water voxel with hot iron neighbors
        let water_k = 0.606;
        let water_rho_cp = 1000.0 * 4186.0;
        let iron_k = 80.2;
        let neighbors = vec![(1000.0, iron_k)];
        let water_result = diffuse_temperature(288.15, water_k, water_rho_cp, &neighbors, DT);
        let water_delta = (water_result - 288.15).abs();

        // Iron voxel with same hot neighbor setup
        let iron_rho_cp = 7874.0 * 449.0;
        let iron_result = diffuse_temperature(288.15, iron_k, iron_rho_cp, &neighbors, DT);
        let iron_delta = (iron_result - 288.15).abs();

        assert!(
            water_delta < iron_delta,
            "Water (ΔT={water_delta:.6}) should change less than iron (ΔT={iron_delta:.6}) \
             due to higher heat capacity"
        );
    }

    #[test]
    fn equilibrium_approaches_rho_cp_weighted_average() {
        // Two adjacent voxels: iron at 1000K, water at 288.15K
        // Equilibrium should be weighted by ρ×Cₚ.
        let iron_k = 80.2;
        let iron_rho_cp: f64 = 7874.0 * 449.0;
        let water_k = 0.606;
        let water_rho_cp: f64 = 1000.0 * 4186.0;

        let t_iron_init = 1000.0_f64;
        let t_water_init = 288.15_f64;

        let expected_eq = (iron_rho_cp * t_iron_init + water_rho_cp * t_water_init)
            / (iron_rho_cp + water_rho_cp);

        // Iterate many steps to approach equilibrium.
        // k_eff ≈ 1.2 W/(m·K), thermal capacities ~3.5M and ~4.2M J/(m³·K),
        // so the time constant τ ≈ 1.6M seconds. Use large dt and many steps.
        let mut t_iron = t_iron_init as f32;
        let mut t_water = t_water_init as f32;
        let dt = 1000.0_f32; // well under CFL limit of ~22000s

        for _ in 0..100_000 {
            let new_iron = diffuse_temperature(
                t_iron,
                iron_k,
                iron_rho_cp as f32,
                &[(t_water, water_k)],
                dt,
            );
            let new_water = diffuse_temperature(
                t_water,
                water_k,
                water_rho_cp as f32,
                &[(t_iron, iron_k)],
                dt,
            );
            t_iron = new_iron;
            t_water = new_water;
        }

        let tolerance = 1.0; // 1K tolerance due to f32 accumulation
        assert!(
            (t_iron as f64 - expected_eq).abs() < tolerance,
            "Iron should converge to {expected_eq:.2}K, got {t_iron:.2}K"
        );
        assert!(
            (t_water as f64 - expected_eq).abs() < tolerance,
            "Water should converge to {expected_eq:.2}K, got {t_water:.2}K"
        );
    }

    #[test]
    fn diffuse_chunk_uniform_is_stable() {
        let reg = test_registry();
        let size = 4;
        let voxels: Vec<Voxel> = (0..size * size * size)
            .map(|_| Voxel::new(MaterialId::STONE))
            .collect();
        let ambient = voxels[0].temperature;
        let new_temps = diffuse_chunk(&voxels, size, DT, &reg);
        for &t in &new_temps {
            assert!(
                (t - ambient).abs() < f32::EPSILON,
                "Uniform chunk should stay at {ambient} K, got {t}"
            );
        }
    }

    #[test]
    fn diffuse_chunk_hot_spot_spreads() {
        let reg = test_registry();
        let size = 4;
        let mut voxels: Vec<Voxel> = (0..size * size * size)
            .map(|_| Voxel::new(MaterialId::STONE))
            .collect();
        let ambient = voxels[0].temperature;
        let center = size * size + size + 1;
        voxels[center].temperature = 1000.0;

        // Use dt=1.0s so the temperature change is large enough for f32 precision
        let new_temps = diffuse_chunk(&voxels, size, 1.0, &reg);

        assert!(new_temps[center] < 1000.0, "Hot spot should cool down");

        let neighbor = center + 1;
        assert!(
            new_temps[neighbor] > ambient,
            "Neighbor of hot spot should warm up"
        );
    }

    #[allow(deprecated)]
    #[test]
    fn legacy_conductivity_lookup_returns_positive() {
        for id in 0..=7 {
            let c = conductivity(MaterialId(id));
            assert!(c > 0.0, "Material {id} should have positive conductivity");
        }
    }

    #[test]
    fn water_requires_more_energy_than_iron_per_kelvin() {
        // Energy to heat 1 m³ by 1 K = ρ × Cp × V × ΔT
        // Water: 1000 × 4186 × 1 × 1 = 4,186,000 J
        // Iron:  7874 × 449 × 1 × 1  = 3,535,426 J
        // Wikipedia: Specific heat capacity
        let water_energy = 1000.0_f32 * 4186.0 * 1.0;
        let iron_energy = 7874.0_f32 * 449.0 * 1.0;
        assert!(
            water_energy > iron_energy,
            "Water ({water_energy:.0} J) should need more energy than iron ({iron_energy:.0} J) to heat 1m³ by 1K"
        );
        // Ratio should be about 1.18
        let ratio = water_energy / iron_energy;
        assert!(
            (ratio - 1.184).abs() < 0.01,
            "Water/iron energy ratio: {ratio:.3}"
        );
    }

    #[test]
    fn thermal_conductivity_ordering() {
        // Wikipedia: Thermal conductivity values
        // Iron: 80.2, Water: 0.606, Wood: 0.15, Air: 0.026 W/(m·K)
        let k_iron = 80.2_f32;
        let k_water = 0.606;
        let k_wood = 0.15;
        let k_air = 0.026;
        assert!(
            k_iron > k_water && k_water > k_wood && k_wood > k_air,
            "Conductivity should be iron > water > wood > air"
        );
        // Iron conducts ~3000× better than air
        assert!(k_iron / k_air > 3000.0);
    }
}
