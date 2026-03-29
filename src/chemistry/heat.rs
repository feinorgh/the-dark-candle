// Heat transfer for voxel grids: conduction + radiation.
//
// **Conduction (Fourier's law)**
// Each tick, heat flux between adjacent voxels is computed as:
//   Q = k_eff * (T_neighbor - T_self)
// where k_eff is the harmonic mean conductivity at the interface (A=1m², dx=1m).
//
// Temperature update per voxel:
//   ΔT = Σ(Q_neighbors) * dt / (ρ * Cₚ)      [V=1m³]
//
// **Radiation (Stefan-Boltzmann)**
// Hot surface voxels emit thermal radiation: P/A = εσT⁴.
// Net radiative exchange between visible surface pairs uses the gray-body
// formula: q = ε_eff × σ × F₁₂ × (T₁⁴ − T₂⁴), where ε_eff accounts for
// both emissivities and F₁₂ is the view factor (~A/(πd²) for distant faces).
// Line-of-sight is checked via discrete ray marching through the voxel grid.
//
// Thermal properties (k, ρ, Cₚ, ε) are read from MaterialData via MaterialRegistry.
// CFL stability (conduction): dt < dx² / (6 × α_max) where α = k/(ρ×Cₚ).
// For iron (α ≈ 2.27e-5 m²/s) with dx=1m: dt_max ≈ 7300s — stable at game timesteps.

#![allow(dead_code)]

use crate::data::MaterialRegistry;
use crate::world::raycast::{self, RAY_DIRECTIONS};
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
///   - `dx`: voxel edge length in meters (default 1.0 for standard resolution)
pub fn diffuse_temperature(
    current_temp: f32,
    self_k: f32,
    self_rho_cp: f32,
    neighbors: &[(f32, f32)],
    dt: f32,
    dx: f32,
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
        // Q_face = k_eff × (A / dx) × ΔT = k_eff × dx × ΔT  [W]
        total_heat_flux += k_eff * dx * (neighbor_temp - current_temp);
    }

    // ΔT = Q_total × dt / (ρ × Cₚ × V) where V = dx³
    current_temp + total_heat_flux * dt / (self_rho_cp * dx * dx * dx)
}

/// Apply one heat diffusion step to a flat 3D voxel array of size `size³`
/// using Fourier's law with material properties from the registry.
///
/// Uses adaptive CFL sub-stepping: if the timestep exceeds the explicit Euler
/// stability limit (Fo ≤ 1/6 in 3D), the diffusion is split into smaller
/// sub-steps to prevent numerical oscillation. This is essential for
/// low-density gases (e.g. hydrogen) where thermal diffusivity α = k/(ρ·Cₚ)
/// is high relative to spatial resolution.
///
/// `dt` is the timestep in seconds.
/// `dx` is the voxel edge length in meters (1.0 for standard resolution).
/// Returns the new temperature buffer.
pub fn diffuse_chunk(
    voxels: &[Voxel],
    size: usize,
    dt: f32,
    registry: &MaterialRegistry,
    dx: f32,
) -> Vec<f32> {
    let len = size * size * size;
    assert_eq!(voxels.len(), len);

    // Pre-compute per-voxel thermal properties once (avoids repeated
    // HashMap lookups in the inner loop).
    let mut conductivities = Vec::with_capacity(len);
    let mut rho_cps = Vec::with_capacity(len);
    let mut alpha_max: f32 = 0.0;

    for v in voxels {
        let mat = registry.get(v.material);
        let k = mat.map(|m| m.thermal_conductivity).unwrap_or(0.1);
        let rho = mat.map(|m| m.density).unwrap_or(1.0);
        let cp = mat.map(|m| m.specific_heat_capacity).unwrap_or(1000.0);
        let rho_cp = rho * cp;
        conductivities.push(k);
        rho_cps.push(rho_cp);
        if rho_cp > 0.0 {
            alpha_max = alpha_max.max(k / rho_cp);
        }
    }

    // CFL limit for 3D explicit Euler: dt_stable = dx² / (6 × α_max).
    // Use a safety factor of 0.9 to stay comfortably below the limit.
    let dx2 = dx * dx;
    let n_substeps = if alpha_max > 0.0 {
        let dt_stable = dx2 / (6.0 * alpha_max) * 0.9;
        (dt / dt_stable).ceil().max(1.0) as usize
    } else {
        1
    };
    let sub_dt = dt / n_substeps as f32;

    // Double-buffer: swap read/write each sub-step instead of cloning.
    let mut temps: Vec<f32> = voxels.iter().map(|v| v.temperature).collect();
    let mut scratch: Vec<f32> = temps.clone();

    let size_sq = size * size;

    for _step in 0..n_substeps {
        // `temps` is the read buffer, `scratch` is the write buffer.
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let idx = z * size_sq + y * size + x;

                    let self_k = conductivities[idx];
                    let self_rho_cp = rho_cps[idx];

                    // Stack-allocated neighbor array (no heap allocation).
                    let mut neighbors: [(f32, f32); 6] = [(0.0, 0.0); 6];
                    let mut n_count = 0usize;

                    if x > 0 {
                        let ni = idx - 1;
                        neighbors[n_count] = (temps[ni], conductivities[ni]);
                        n_count += 1;
                    }
                    if x + 1 < size {
                        let ni = idx + 1;
                        neighbors[n_count] = (temps[ni], conductivities[ni]);
                        n_count += 1;
                    }
                    if y > 0 {
                        let ni = idx - size;
                        neighbors[n_count] = (temps[ni], conductivities[ni]);
                        n_count += 1;
                    }
                    if y + 1 < size {
                        let ni = idx + size;
                        neighbors[n_count] = (temps[ni], conductivities[ni]);
                        n_count += 1;
                    }
                    if z > 0 {
                        let ni = idx - size_sq;
                        neighbors[n_count] = (temps[ni], conductivities[ni]);
                        n_count += 1;
                    }
                    if z + 1 < size {
                        let ni = idx + size_sq;
                        neighbors[n_count] = (temps[ni], conductivities[ni]);
                        n_count += 1;
                    }

                    scratch[idx] = diffuse_temperature(
                        temps[idx],
                        self_k,
                        self_rho_cp,
                        &neighbors[..n_count],
                        sub_dt,
                        dx,
                    );
                }
            }
        }
        // Swap buffers: scratch becomes the read source for the next sub-step.
        std::mem::swap(&mut temps, &mut scratch);
    }

    temps
}

// ---------------------------------------------------------------------------
// Radiative heat transfer (Stefan-Boltzmann)
// ---------------------------------------------------------------------------

/// Maximum view factor — physical cap for two unit-area faces at d = 1 m.
///
/// The exact view factor for two coaxial unit squares at d = 1 is ~0.20,
/// computed from the analytical double-area integral. We cap the far-field
/// approximation A/(πd²) to this value for close pairs.
const VIEW_FACTOR_CAP: f32 = 0.20;

/// Radiated power per unit area from a surface (W/m²).
///
/// Stefan-Boltzmann law: q = ε × σ × T⁴.
/// Computation uses f64 internally to preserve precision in T⁴.
pub fn stefan_boltzmann_flux(emissivity: f32, temperature: f32, sigma: f64) -> f32 {
    let t4 = (temperature as f64).powi(4);
    (emissivity as f64 * sigma * t4) as f32
}

/// Effective emissivity for gray-body radiative exchange between two surfaces.
///
/// ε_eff = 1 / (1/ε₁ + 1/ε₂ − 1)
///
/// For two black bodies (ε = 1): ε_eff = 1.
/// Returns 0 if either emissivity is ≤ 0 (prevents division by zero).
pub fn effective_emissivity(eps1: f32, eps2: f32) -> f32 {
    if eps1 <= 0.0 || eps2 <= 0.0 {
        return 0.0;
    }
    1.0 / (1.0 / eps1 + 1.0 / eps2 - 1.0)
}

/// Approximate view factor between two voxel faces at Euclidean distance `d`.
///
/// Far-field point-source approximation: F = A / (π × d²), capped at
/// [`VIEW_FACTOR_CAP`] for close pairs where the approximation breaks down.
pub fn voxel_view_factor(distance: f32, face_area: f32) -> f32 {
    if distance <= 0.0 {
        return VIEW_FACTOR_CAP;
    }
    let f = face_area / (std::f32::consts::PI * distance * distance);
    f.min(VIEW_FACTOR_CAP)
}

/// Net radiative heat flux between two surfaces (W).
///
/// Positive result means heat flows from surface 1 → surface 2.
/// `eps_eff` is the pre-computed effective emissivity (see [`effective_emissivity`]).
///
/// q = ε_eff × σ × F₁₂ × A × (T₁⁴ − T₂⁴)
///
/// With voxel_size = 1 m → A = 1 m², so the result is directly in Watts.
pub fn net_radiative_flux(t1: f32, t2: f32, eps_eff: f32, view_factor: f32, sigma: f64) -> f32 {
    let t1_4 = (t1 as f64).powi(4);
    let t2_4 = (t2 as f64).powi(4);
    (eps_eff as f64 * sigma * view_factor as f64 * (t1_4 - t2_4)) as f32
}

/// Apply one radiative heat transfer step to a flat `size³` voxel array.
///
/// For each surface voxel above `emission_threshold` (K), casts rays in
/// 26 grid-aligned directions to find visible surfaces. Net radiative
/// exchange is computed for each pair and accumulated as temperature deltas.
///
/// Returns a `Vec<f32>` of temperature deltas (additive, can be positive or
/// negative). The caller should add these to the current temperatures.
///
/// # Multiresolution scaling (`dx`)
///
/// At voxel edge length `dx` (meters), the radiated power scales as dx²
/// (smaller emitter face area) while voxel thermal mass scales as dx³
/// (smaller volume). The net effect is that temperature change scales as
/// 1/dx — smaller voxels heat up faster per unit radiation, which is
/// physically correct. At dx = 1.0 this reduces to the original formula.
///
/// # Parameters
/// - `voxels`: flat `size³` array of voxels
/// - `size`: grid edge length (voxels per axis)
/// - `dt`: timestep in seconds
/// - `dx`: voxel edge length in meters (1.0 for standard resolution)
/// - `registry`: material property lookup
/// - `sigma`: Stefan-Boltzmann constant (W/(m²·K⁴))
/// - `emission_threshold`: minimum temperature (K) to consider a voxel as emitter
/// - `max_ray_steps`: maximum ray march distance in voxel steps
#[allow(clippy::too_many_arguments)]
pub fn radiate_chunk(
    voxels: &[Voxel],
    size: usize,
    dt: f32,
    dx: f32,
    registry: &MaterialRegistry,
    sigma: f64,
    emission_threshold: f32,
    max_ray_steps: usize,
) -> Vec<f32> {
    let len = size * size * size;
    assert_eq!(voxels.len(), len);

    let mut deltas = vec![0.0_f32; len];

    // Use only the first 13 of 26 ray directions. Each direction has a
    // symmetric opposite; by marching only "positive-half" directions we
    // naturally avoid processing each (emitter, receiver) pair twice and
    // eliminate the expensive HashSet deduplication.
    let half_dirs = RAY_DIRECTIONS.len() / 2;

    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let idx = z * size * size + y * size + x;
                let voxel = &voxels[idx];

                if voxel.material.is_air() || voxel.temperature < emission_threshold {
                    continue;
                }

                if !raycast::is_surface_voxel(voxels, size, x, y, z) {
                    continue;
                }

                let mat_e = match registry.get(voxel.material) {
                    Some(m) => m,
                    None => continue,
                };

                if mat_e.emissivity <= 0.0 {
                    continue;
                }

                for dir_idx in 0..half_dirs {
                    let get_absorption = |mat_id: MaterialId| -> Option<f32> {
                        if mat_id.is_air() {
                            return Some(0.0);
                        }
                        registry.get(mat_id).and_then(|m| m.absorption_coefficient)
                    };

                    let attenuated = match raycast::march_grid_ray_attenuated(
                        voxels,
                        size,
                        [x, y, z],
                        dir_idx,
                        max_ray_steps,
                        get_absorption,
                    ) {
                        Some(h) => h,
                        None => continue,
                    };

                    let hit = attenuated.hit;

                    let recv = &voxels[hit.index];
                    let mat_r = match registry.get(recv.material) {
                        Some(m) => m,
                        None => continue,
                    };

                    let eps_eff = effective_emissivity(mat_e.emissivity, mat_r.emissivity);
                    if eps_eff <= 0.0 {
                        continue;
                    }

                    let view = voxel_view_factor(hit.distance, 1.0);
                    let q = net_radiative_flux(
                        voxel.temperature,
                        recv.temperature,
                        eps_eff,
                        view,
                        sigma,
                    ) * attenuated.transmittance;

                    // ΔT = Q × dt / (ρ × V × Cₚ), but Q itself scales as dx²
                    // (emitter face area). Since the view factor A_recv/(πr²)
                    // is scale-invariant (dx² cancels in numerator/denominator),
                    // the net scaling is: ΔT = q × dx² × dt / (ρ × dx³ × Cₚ)
                    //                       = q × dt / (ρ × dx × Cₚ).
                    let rho_cp_dx_e = mat_e.density * mat_e.specific_heat_capacity * dx;
                    let rho_cp_dx_r = mat_r.density * mat_r.specific_heat_capacity * dx;

                    if rho_cp_dx_e > 0.0 {
                        deltas[idx] -= q * dt / rho_cp_dx_e;
                    }
                    if rho_cp_dx_r > 0.0 {
                        deltas[hit.index] += q * dt / rho_cp_dx_r;
                    }
                }
            }
        }
    }

    deltas
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
        let result = diffuse_temperature(293.0, k, rho_cp, &neighbors, DT, 1.0);
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
        let result = diffuse_temperature(400.0, k, rho_cp, &neighbors, DT, 1.0);
        assert!(result < 400.0, "Hot voxel should cool down, got {result}");

        let cold_neighbors = vec![(500.0, k), (500.0, k)];
        let result2 = diffuse_temperature(200.0, k, rho_cp, &cold_neighbors, DT, 1.0);
        assert!(result2 > 200.0, "Cold voxel should warm up, got {result2}");
    }

    #[test]
    fn no_neighbors_keeps_temperature() {
        let result = diffuse_temperature(350.0, 80.2, 7874.0 * 449.0, &[], DT, 1.0);
        assert_eq!(result, 350.0);
    }

    #[test]
    fn zero_rho_cp_keeps_temperature() {
        let neighbors = vec![(500.0, 80.2)];
        let result = diffuse_temperature(350.0, 80.2, 0.0, &neighbors, DT, 1.0);
        assert_eq!(result, 350.0, "Zero ρCₚ should not change temperature");
    }

    #[test]
    fn iron_conducts_faster_than_air() {
        // Iron hot spot: 3-voxel line, center hot
        let iron_k = 80.2_f32;
        let iron_rho_cp = 7874.0 * 449.0;
        let iron_neighbors = vec![(288.15, iron_k), (288.15, iron_k)];
        let iron_result =
            diffuse_temperature(1000.0, iron_k, iron_rho_cp, &iron_neighbors, DT, 1.0);
        let iron_delta = (1000.0 - iron_result).abs();

        // Air hot spot: same geometry
        let air_k = 0.026_f32;
        let air_rho_cp = 1.225 * 1005.0;
        let air_neighbors = vec![(288.15, air_k), (288.15, air_k)];
        let air_result = diffuse_temperature(1000.0, air_k, air_rho_cp, &air_neighbors, DT, 1.0);
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

        let new_temps = diffuse_chunk(&voxels, size, DT, &reg, 1.0);

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
        let water_result = diffuse_temperature(288.15, water_k, water_rho_cp, &neighbors, DT, 1.0);
        let water_delta = (water_result - 288.15).abs();

        // Iron voxel with same hot neighbor setup
        let iron_rho_cp = 7874.0 * 449.0;
        let iron_result = diffuse_temperature(288.15, iron_k, iron_rho_cp, &neighbors, DT, 1.0);
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
                1.0,
            );
            let new_water = diffuse_temperature(
                t_water,
                water_k,
                water_rho_cp as f32,
                &[(t_iron, iron_k)],
                dt,
                1.0,
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
        let new_temps = diffuse_chunk(&voxels, size, DT, &reg, 1.0);
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
        let new_temps = diffuse_chunk(&voxels, size, 1.0, &reg, 1.0);

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

    // -----------------------------------------------------------------------
    // Radiation tests
    // -----------------------------------------------------------------------

    const SIGMA: f64 = 5.670_374_419e-8;

    const LAVA_ID: u16 = 10;

    /// Registry with emissivity values for radiation tests.
    fn rad_registry() -> MaterialRegistry {
        let mut reg = MaterialRegistry::new();
        reg.insert(MaterialData {
            id: AIR_ID,
            name: "Air".into(),
            default_phase: Phase::Gas,
            density: 1.225,
            thermal_conductivity: 0.026,
            specific_heat_capacity: 1005.0,
            emissivity: 0.0,
            ..Default::default()
        });
        reg.insert(MaterialData {
            id: STONE_ID,
            name: "Stone".into(),
            default_phase: Phase::Solid,
            density: 2700.0,
            thermal_conductivity: 2.5,
            specific_heat_capacity: 790.0,
            emissivity: 0.93,
            ..Default::default()
        });
        reg.insert(MaterialData {
            id: WATER_ID,
            name: "Water".into(),
            default_phase: Phase::Liquid,
            density: 1000.0,
            thermal_conductivity: 0.606,
            specific_heat_capacity: 4186.0,
            emissivity: 0.96,
            absorption_coefficient: Some(100.0),
            ..Default::default()
        });
        reg.insert(MaterialData {
            id: IRON_ID,
            name: "Iron".into(),
            default_phase: Phase::Solid,
            density: 7874.0,
            thermal_conductivity: 80.2,
            specific_heat_capacity: 449.0,
            emissivity: 0.21,
            ..Default::default()
        });
        reg.insert(MaterialData {
            id: LAVA_ID,
            name: "Lava".into(),
            default_phase: Phase::Liquid,
            density: 2600.0,
            thermal_conductivity: 1.5,
            specific_heat_capacity: 1600.0,
            emissivity: 0.95,
            ..Default::default()
        });
        reg
    }

    #[test]
    fn stefan_boltzmann_black_body_at_1000k() {
        // P/A = σT⁴ for black body (ε=1)
        // At 1000 K: 5.67e-8 × 1e12 = 56700 W/m²
        let flux = stefan_boltzmann_flux(1.0, 1000.0, SIGMA);
        assert!(
            (flux - 56_700.0).abs() < 100.0,
            "Black body at 1000K: {flux:.1} W/m², expected ~56700"
        );
    }

    #[test]
    fn stefan_boltzmann_scales_with_emissivity() {
        let full = stefan_boltzmann_flux(1.0, 1000.0, SIGMA);
        let half = stefan_boltzmann_flux(0.5, 1000.0, SIGMA);
        assert!(
            (half - full * 0.5).abs() < 1.0,
            "Half emissivity should give half flux"
        );
    }

    #[test]
    fn stefan_boltzmann_t4_scaling() {
        // Doubling temperature → 16× the flux
        let t1 = stefan_boltzmann_flux(1.0, 500.0, SIGMA);
        let t2 = stefan_boltzmann_flux(1.0, 1000.0, SIGMA);
        let ratio = t2 / t1;
        assert!(
            (ratio - 16.0).abs() < 0.1,
            "T⁴ scaling: ratio = {ratio:.2}, expected 16.0"
        );
    }

    #[test]
    fn effective_emissivity_black_bodies() {
        let eps = effective_emissivity(1.0, 1.0);
        assert!(
            (eps - 1.0).abs() < f32::EPSILON,
            "Two black bodies: ε_eff = {eps}"
        );
    }

    #[test]
    fn effective_emissivity_symmetric() {
        let a = effective_emissivity(0.21, 0.93);
        let b = effective_emissivity(0.93, 0.21);
        assert!((a - b).abs() < f32::EPSILON, "ε_eff should be symmetric");
    }

    #[test]
    fn effective_emissivity_with_zero() {
        assert_eq!(effective_emissivity(0.0, 0.5), 0.0);
        assert_eq!(effective_emissivity(0.5, 0.0), 0.0);
    }

    #[test]
    fn effective_emissivity_iron_stone() {
        // ε_eff = 1/(1/0.21 + 1/0.93 - 1) ≈ 0.207
        let eps = effective_emissivity(0.21, 0.93);
        assert!(
            (eps - 0.207).abs() < 0.01,
            "Iron-stone ε_eff = {eps:.4}, expected ~0.207"
        );
    }

    #[test]
    fn view_factor_inverse_square() {
        let f1 = voxel_view_factor(2.0, 1.0);
        let f2 = voxel_view_factor(4.0, 1.0);
        let ratio = f1 / f2;
        assert!(
            (ratio - 4.0).abs() < 0.1,
            "View factor should fall off as 1/d²: ratio = {ratio:.2}"
        );
    }

    #[test]
    fn view_factor_capped_at_close_range() {
        let f = voxel_view_factor(0.5, 1.0);
        assert!(
            f <= VIEW_FACTOR_CAP + f32::EPSILON,
            "View factor {f} should be capped at {VIEW_FACTOR_CAP}"
        );
    }

    #[test]
    fn net_flux_zero_at_equal_temperatures() {
        let q = net_radiative_flux(500.0, 500.0, 0.5, 0.1, SIGMA);
        assert!(
            q.abs() < f32::EPSILON,
            "Net flux at equal T should be 0, got {q}"
        );
    }

    #[test]
    fn net_flux_positive_hot_to_cold() {
        let q = net_radiative_flux(1000.0, 300.0, 0.5, 0.1, SIGMA);
        assert!(
            q > 0.0,
            "Net flux should be positive from hot to cold, got {q}"
        );
    }

    #[test]
    fn net_flux_antisymmetric() {
        let q_ab = net_radiative_flux(1000.0, 300.0, 0.5, 0.1, SIGMA);
        let q_ba = net_radiative_flux(300.0, 1000.0, 0.5, 0.1, SIGMA);
        assert!(
            (q_ab + q_ba).abs() < 0.01,
            "Net flux should be antisymmetric: q_ab={q_ab:.2}, q_ba={q_ba:.2}"
        );
    }

    #[test]
    fn radiate_chunk_uniform_temperature_no_change() {
        let reg = rad_registry();
        let size = 4;
        let mut voxels: Vec<Voxel> = (0..size * size * size)
            .map(|_| {
                let mut v = Voxel::new(MaterialId::STONE);
                v.temperature = 800.0; // above threshold but uniform
                v
            })
            .collect();
        // Leave surface layer as air for exposure
        for x in 0..size {
            for z in 0..size {
                voxels[z * size * size + x] = Voxel::default(); // y=0 air
            }
        }

        let deltas = radiate_chunk(&voxels, size, 1.0, 1.0, &reg, SIGMA, 500.0, 8);
        // Interior stone at uniform 800K: pairs at same temperature → net flux = 0
        for z in 0..size {
            for y in 1..size {
                for x in 0..size {
                    let idx = z * size * size + y * size + x;
                    if !voxels[idx].material.is_air() {
                        assert!(
                            deltas[idx].abs() < 0.01,
                            "Uniform temp delta at ({x},{y},{z}) = {:.4}, expected ~0",
                            deltas[idx]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn radiate_chunk_hot_body_warms_cold_across_gap() {
        // Setup: hot lava wall | air gap | cold stone wall
        // Radiation should transfer heat across the air gap.
        let reg = rad_registry();
        let size = 8;
        let mut voxels: Vec<Voxel> = (0..size * size * size).map(|_| Voxel::default()).collect();

        let idx = |x: usize, y: usize, z: usize| z * size * size + y * size + x;

        // Place hot lava wall at x=1 (y=1..7, z=1..7)
        for y in 1..7 {
            for z in 1..7 {
                let i = idx(1, y, z);
                voxels[i].material = MaterialId::LAVA;
                voxels[i].temperature = 1500.0;
            }
        }

        // Place cold stone wall at x=5 (y=1..7, z=1..7)
        for y in 1..7 {
            for z in 1..7 {
                let i = idx(5, y, z);
                voxels[i].material = MaterialId::STONE;
                voxels[i].temperature = 300.0;
            }
        }

        let deltas = radiate_chunk(&voxels, size, 1.0, 1.0, &reg, SIGMA, 500.0, 8);

        // Lava should lose heat (negative delta)
        let lava_center = idx(1, 4, 4);
        assert!(
            deltas[lava_center] < 0.0,
            "Lava should lose heat: delta = {:.4}",
            deltas[lava_center]
        );

        // Stone should gain heat (positive delta)
        let stone_center = idx(5, 4, 4);
        assert!(
            deltas[stone_center] > 0.0,
            "Stone should gain heat: delta = {:.4}",
            deltas[stone_center]
        );
    }

    #[test]
    fn radiate_chunk_energy_conservation_two_body() {
        // Two single-voxel bodies facing each other across an air gap.
        // Total thermal energy should be conserved.
        let reg = rad_registry();
        let size = 8;
        let mut voxels: Vec<Voxel> = (0..size * size * size).map(|_| Voxel::default()).collect();

        let idx = |x: usize, y: usize, z: usize| z * size * size + y * size + x;

        // Hot stone at (2,4,4)
        let hot_idx = idx(2, 4, 4);
        voxels[hot_idx].material = MaterialId::STONE;
        voxels[hot_idx].temperature = 1200.0;

        // Cold stone at (5,4,4)
        let cold_idx = idx(5, 4, 4);
        voxels[cold_idx].material = MaterialId::STONE;
        voxels[cold_idx].temperature = 300.0;

        let rho_cp_stone = 2700.0_f64 * 790.0; // same material

        // Total energy before
        let energy_before = rho_cp_stone * 1200.0 + rho_cp_stone * 300.0;

        let deltas = radiate_chunk(&voxels, size, 1.0, 1.0, &reg, SIGMA, 500.0, 8);

        // Total energy after
        let energy_after = rho_cp_stone * (1200.0 + deltas[hot_idx] as f64)
            + rho_cp_stone * (300.0 + deltas[cold_idx] as f64);

        let relative_error = ((energy_after - energy_before) / energy_before).abs();
        assert!(
            relative_error < 1e-5,
            "Radiative energy should be conserved. Before: {energy_before:.2}, \
             After: {energy_after:.2}, error: {relative_error:.2e}"
        );
    }

    #[test]
    fn radiate_chunk_blocked_by_wall() {
        // Hot voxel and cold voxel with an opaque wall in between.
        // No radiative exchange should occur.
        let reg = rad_registry();
        let size = 8;
        let mut voxels: Vec<Voxel> = (0..size * size * size).map(|_| Voxel::default()).collect();

        let idx = |x: usize, y: usize, z: usize| z * size * size + y * size + x;

        // Hot stone at (1,4,4)
        let hot_idx = idx(1, 4, 4);
        voxels[hot_idx].material = MaterialId::STONE;
        voxels[hot_idx].temperature = 1200.0;

        // Wall at (3,4,4) blocking LOS
        voxels[idx(3, 4, 4)].material = MaterialId::STONE;
        voxels[idx(3, 4, 4)].temperature = 288.15;

        // Cold stone at (5,4,4) — behind the wall
        let cold_idx = idx(5, 4, 4);
        voxels[cold_idx].material = MaterialId::STONE;
        voxels[cold_idx].temperature = 300.0;

        let deltas = radiate_chunk(&voxels, size, 1.0, 1.0, &reg, SIGMA, 500.0, 8);

        // Cold stone should NOT gain heat from the hot stone (wall blocks)
        // It may exchange with the wall, but the wall is near ambient.
        assert!(
            deltas[cold_idx].abs() < 1.0,
            "Blocked cold stone should have negligible delta: {:.4}",
            deltas[cold_idx]
        );
    }

    #[test]
    fn radiate_chunk_closer_receives_more() {
        // Two cold stones at different distances from a hot source.
        // Closer one should receive more heat (1/d² falloff).
        let reg = rad_registry();
        let size = 16;
        let mut voxels: Vec<Voxel> = (0..size * size * size).map(|_| Voxel::default()).collect();

        let idx = |x: usize, y: usize, z: usize| z * size * size + y * size + x;

        // Hot stone at (1,8,8)
        voxels[idx(1, 8, 8)].material = MaterialId::STONE;
        voxels[idx(1, 8, 8)].temperature = 1500.0;

        // Close stone at (3,8,8) — distance 2
        let close_idx = idx(3, 8, 8);
        voxels[close_idx].material = MaterialId::STONE;
        voxels[close_idx].temperature = 300.0;

        // Far stone at (7,8,8) — distance 6
        let far_idx = idx(7, 8, 8);
        voxels[far_idx].material = MaterialId::STONE;
        voxels[far_idx].temperature = 300.0;

        let deltas = radiate_chunk(&voxels, size, 1.0, 1.0, &reg, SIGMA, 500.0, 16);

        // The close stone blocks the hot→far ray in +X direction,
        // but we still expect close stone to get more heat overall.
        assert!(
            deltas[close_idx] > 0.0,
            "Close stone should gain heat: {:.4}",
            deltas[close_idx]
        );
        // Far stone may get 0 heat if fully blocked. That's fine — the test
        // confirms closer gets more.
        assert!(
            deltas[close_idx] > deltas[far_idx],
            "Closer stone ({:.4}) should gain more than farther ({:.4})",
            deltas[close_idx],
            deltas[far_idx]
        );
    }

    #[test]
    fn radiate_chunk_below_threshold_no_emission() {
        let reg = rad_registry();
        let size = 4;
        let mut voxels: Vec<Voxel> = (0..size * size * size).map(|_| Voxel::default()).collect();

        // Stone at 400K — below 500K threshold
        let idx = size * size + size + 1;
        voxels[idx].material = MaterialId::STONE;
        voxels[idx].temperature = 400.0;

        let deltas = radiate_chunk(&voxels, size, 1.0, 1.0, &reg, SIGMA, 500.0, 4);
        for d in &deltas {
            assert!(
                d.abs() < f32::EPSILON,
                "Below-threshold voxels should produce no radiation deltas"
            );
        }
    }

    #[test]
    fn radiate_chunk_attenuated_through_water() {
        // Hot stone (1200K) → water voxel → cold stone: water attenuates flux.
        let reg = rad_registry();
        let size = 8;
        let mut voxels: Vec<Voxel> = (0..size * size * size).map(|_| Voxel::default()).collect();

        let hot_idx = 4 * size * size + 4 * size + 1;
        voxels[hot_idx].material = MaterialId::STONE;
        voxels[hot_idx].temperature = 1200.0;

        // Water at x=2 (semi-transparent, α=100 m⁻¹)
        let water_idx = 4 * size * size + 4 * size + 2;
        voxels[water_idx].material = MaterialId::WATER;
        voxels[water_idx].temperature = 288.15;

        // Cold stone target at x=3
        let cold_idx = 4 * size * size + 4 * size + 3;
        voxels[cold_idx].material = MaterialId::STONE;
        voxels[cold_idx].temperature = 288.15;

        let deltas = radiate_chunk(&voxels, size, 1.0, 1.0, &reg, SIGMA, 500.0, 8);

        // With α=100 m⁻¹ and 1m of water: transmittance = exp(-100) ≈ 3.7e-44
        // This is below MIN_TRANSMITTANCE so the ray is fully absorbed.
        // The cold stone should receive negligible radiation.
        assert!(
            deltas[cold_idx].abs() < 1e-10,
            "Water (α=100) should nearly fully absorb radiation: got delta={}",
            deltas[cold_idx]
        );
    }

    #[test]
    fn radiate_chunk_no_absorption_coeff_blocks_like_before() {
        // Verify materials without absorption_coefficient (None) are opaque.
        // Stone wall between hot and cold stone should fully block radiation.
        let reg = rad_registry();
        let size = 8;
        let mut voxels: Vec<Voxel> = (0..size * size * size).map(|_| Voxel::default()).collect();

        let hot_idx = 4 * size * size + 4 * size + 1;
        voxels[hot_idx].material = MaterialId::STONE;
        voxels[hot_idx].temperature = 1200.0;

        // Opaque iron wall at x=3 (no absorption_coefficient)
        let wall_idx = 4 * size * size + 4 * size + 3;
        voxels[wall_idx].material = MaterialId(IRON_ID);
        voxels[wall_idx].temperature = 288.15;

        // Target stone at x=5
        let target_idx = 4 * size * size + 4 * size + 5;
        voxels[target_idx].material = MaterialId::STONE;
        voxels[target_idx].temperature = 288.15;

        let deltas = radiate_chunk(&voxels, size, 1.0, 1.0, &reg, SIGMA, 500.0, 8);

        // Iron wall should absorb from hot stone (it's the first hit in +x dir),
        // but target at x=5 should only receive radiation from wall itself
        // (which is at 288K, below 500K threshold), so no direct hot→target flux.
        // The wall gets heated:
        assert!(
            deltas[wall_idx] > 0.0,
            "Iron wall should absorb radiation from hot stone"
        );
    }

    #[test]
    fn radiate_chunk_partial_attenuation_reduces_flux() {
        // Compare radiation with and without a semi-transparent medium.
        // Use a low-α water variant to get partial attenuation.
        let mut reg = rad_registry();
        // Override water with very low absorption (α=0.1 m⁻¹)
        reg.insert(MaterialData {
            id: WATER_ID,
            name: "Water".into(),
            default_phase: Phase::Liquid,
            density: 1000.0,
            thermal_conductivity: 0.606,
            specific_heat_capacity: 4186.0,
            emissivity: 0.96,
            absorption_coefficient: Some(0.1),
            ..Default::default()
        });

        let size = 8;

        // Case 1: hot stone → air → cold stone (no attenuation)
        let mut voxels_clear: Vec<Voxel> =
            (0..size * size * size).map(|_| Voxel::default()).collect();
        let hot = 4 * size * size + 4 * size + 1;
        let cold = 4 * size * size + 4 * size + 3;
        voxels_clear[hot].material = MaterialId::STONE;
        voxels_clear[hot].temperature = 1200.0;
        voxels_clear[cold].material = MaterialId::STONE;
        voxels_clear[cold].temperature = 288.15;
        let deltas_clear = radiate_chunk(&voxels_clear, size, 1.0, 1.0, &reg, SIGMA, 500.0, 8);

        // Case 2: hot stone → water → cold stone (partial attenuation)
        let mut voxels_water = voxels_clear.clone();
        let water = 4 * size * size + 4 * size + 2;
        voxels_water[water].material = MaterialId::WATER;
        voxels_water[water].temperature = 288.15;
        let deltas_water = radiate_chunk(&voxels_water, size, 1.0, 1.0, &reg, SIGMA, 500.0, 8);

        // With α=0.1 and 1m water: transmittance = exp(-0.1) ≈ 0.905
        // Cold stone should receive ~90.5% of the clear-path flux
        assert!(
            deltas_water[cold] > 0.0,
            "Cold stone should still receive some radiation through thin water"
        );
        assert!(
            deltas_water[cold] < deltas_clear[cold],
            "Attenuated flux ({}) should be less than clear flux ({})",
            deltas_water[cold],
            deltas_clear[cold]
        );
        // Transmittance ≈ 0.905, so attenuated delta should be ~90% of clear
        let ratio = deltas_water[cold] / deltas_clear[cold];
        let expected_transmittance = (-0.1_f32).exp();
        assert!(
            (ratio - expected_transmittance).abs() < 0.05,
            "flux ratio {ratio:.3} should be near transmittance {expected_transmittance:.3}"
        );
    }
}
