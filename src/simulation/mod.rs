// Shared simulation harness for multi-tick voxel chemistry/physics tests.
//
// Provides a deterministic tick loop that combines:
//   1. Chemical reactions (neighbor-pair matching) — fast processes first
//   2. Heat diffusion (Fourier's law, CFL sub-stepped for stability)
//   2b. Radiative heat transfer (Stefan-Boltzmann)
//   3. State transitions (melting/boiling/freezing)
//   4. Pressure diffusion (gas equalization)
//
// Used by `fire_propagation.rs` tests and by data-driven `.simulation.ron`
// scenario files discovered automatically from `tests/cases/simulation/`.

pub mod assertions;
pub mod geometry;
pub mod scenario;

use crate::chemistry::heat::{diffuse_chunk, radiate_chunk};
use crate::chemistry::reactions::{ReactionData, check_reaction};
use crate::chemistry::state_transitions::apply_transitions;
use crate::data::{MaterialRegistry, Phase};
use crate::physics::constants::STEFAN_BOLTZMANN;
use crate::world::voxel::{MaterialId, Voxel};

/// Face-adjacent neighbor offsets (6-connectivity).
const NEIGHBORS: [(i32, i32, i32); 6] = [
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
];

/// 3D index into a flat `size³` voxel array.
fn idx(x: usize, y: usize, z: usize, size: usize) -> usize {
    z * size * size + y * size + x
}

/// Statistics collected from a single simulation tick.
#[derive(Debug, Clone, Default)]
pub struct TickResult {
    /// Number of chemical reactions that fired this tick.
    pub reactions_fired: usize,
    /// Number of phase transitions that occurred this tick.
    pub transitions: usize,
    /// Maximum temperature across the grid after this tick.
    pub max_temp: f32,
    /// Maximum pressure across the grid after this tick.
    pub max_pressure: f32,
    /// Maximum pressure delta from diffusion (convergence indicator).
    pub max_pressure_delta: f32,
}

/// Cumulative statistics across the entire simulation run.
#[derive(Debug, Clone, Default)]
pub struct SimulationStats {
    /// Total reactions fired across all ticks.
    pub total_reactions: usize,
    /// Total phase transitions across all ticks.
    pub total_transitions: usize,
    /// Peak temperature observed at any tick.
    pub peak_temp: f32,
    /// Peak pressure observed at any tick.
    pub peak_pressure: f32,
}

impl SimulationStats {
    /// Update cumulative stats with results from one tick.
    pub fn accumulate(&mut self, tick: &TickResult) {
        self.total_reactions += tick.reactions_fired;
        self.total_transitions += tick.transitions;
        self.peak_temp = self.peak_temp.max(tick.max_temp);
        self.peak_pressure = self.peak_pressure.max(tick.max_pressure);
    }
}

/// Returns true if pressure can propagate through this material.
///
/// Unlike the hardcoded check in `physics::pressure`, this version consults
/// the `MaterialRegistry` so that any gas-phase material (Hydrogen, Oxygen,
/// Steam, Air, etc.) is treated as permeable.
fn is_permeable(material: crate::world::voxel::MaterialId, registry: &MaterialRegistry) -> bool {
    if material.is_air() {
        return true;
    }
    registry
        .get(material)
        .is_some_and(|m| m.default_phase == Phase::Gas)
}

/// Diffuse pressure across a `size³` voxel grid using the material registry
/// to determine permeability (any gas-phase material is permeable).
///
/// Returns the maximum pressure delta applied (useful for convergence checks).
fn diffuse_pressure_sim(
    voxels: &mut [Voxel],
    size: usize,
    rate: f32,
    registry: &MaterialRegistry,
) -> f32 {
    // Snapshot only the values we actually read (pressure + permeability),
    // not the entire Voxel array.
    let len = size * size * size;
    let mut pressures = Vec::with_capacity(len);
    let mut permeable = Vec::with_capacity(len);
    for v in voxels.iter() {
        pressures.push(v.pressure);
        permeable.push(is_permeable(v.material, registry));
    }

    let mut max_delta: f32 = 0.0;

    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let i = idx(x, y, z, size);
                if !permeable[i] {
                    continue;
                }

                let mut sum = 0.0_f32;
                let mut count = 0u32;

                for &(dx, dy, dz) in &NEIGHBORS {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    let nz = z as i32 + dz;

                    if nx < 0 || ny < 0 || nz < 0 {
                        continue;
                    }
                    let (nx, ny, nz) = (nx as usize, ny as usize, nz as usize);
                    if nx >= size || ny >= size || nz >= size {
                        continue;
                    }

                    let ni = idx(nx, ny, nz, size);
                    if permeable[ni] {
                        sum += pressures[ni];
                        count += 1;
                    }
                }

                if count > 0 {
                    let avg = sum / count as f32;
                    let delta = (avg - pressures[i]) * rate;
                    voxels[i].pressure = pressures[i] + delta;
                    max_delta = max_delta.max(delta.abs());
                }
            }
        }
    }

    max_delta
}

/// Run one simulation tick on a flat `size³` voxel array.
///
/// Steps per tick:
///  1. Chemical reactions (neighbor-pair `check_reaction`) — fast processes
///  2. Heat diffusion (Fourier's law via `diffuse_chunk`, CFL sub-stepped)
///     - 2b. Radiative heat transfer (Stefan-Boltzmann via `radiate_chunk`)
///  3. State transitions (`apply_transitions`)
///  4. Pressure diffusion (gas equalization)
///
/// Reactions run first because chemical processes (μs timescale) are
/// effectively instantaneous relative to thermal transport (s–min timescale).
/// This ensures ignition sources trigger reactions before diffusion smears
/// them across the grid.
///
/// `dt` is the simulation timestep in seconds.
pub fn simulate_tick(
    voxels: &mut [Voxel],
    size: usize,
    rules: &[ReactionData],
    registry: &MaterialRegistry,
    dt: f32,
) -> TickResult {
    // 1. Chemical reactions — check each voxel against ±X/±Y/±Z neighbors.
    //    Runs first so ignition hot-spots trigger before diffusion spreads them.
    //    Snapshot only the fields read by check_reaction (material + temperature)
    //    instead of the full Voxel array.
    let mut reactions_fired = 0;
    let snap_materials: Vec<MaterialId> = voxels.iter().map(|v| v.material).collect();
    let snap_temps: Vec<f32> = voxels.iter().map(|v| v.temperature).collect();

    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let i = idx(x, y, z, size);
                let mat_a = snap_materials[i];
                let temp_a = snap_temps[i];

                let mut reacted = false;
                for &(dx, dy, dz) in &NEIGHBORS {
                    if reacted {
                        break;
                    }

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
                        continue;
                    }

                    let ni = nz as usize * size * size + ny as usize * size + nx as usize;
                    let mat_b = snap_materials[ni];

                    for rule in rules {
                        if let Some(result) = check_reaction(rule, mat_a, mat_b, temp_a, registry) {
                            voxels[i].material = result.new_material_a;
                            voxels[i].temperature += result.heat_output;
                            if let Some(new_b) = result.new_material_b {
                                voxels[ni].material = new_b;
                            }
                            reactions_fired += 1;
                            reacted = true;
                            break;
                        }
                    }
                }
            }
        }
    }

    // 2. Heat diffusion (CFL sub-stepped for stability with low-density gases)
    let new_temps = diffuse_chunk(voxels, size, dt, registry);
    for (v, &t) in voxels.iter_mut().zip(new_temps.iter()) {
        v.temperature = t;
    }

    // 2b. Radiative heat transfer — hot surfaces exchange heat at distance
    let rad_deltas = radiate_chunk(voxels, size, dt, registry, STEFAN_BOLTZMANN, 500.0, 16);
    for (v, &delta) in voxels.iter_mut().zip(rad_deltas.iter()) {
        v.temperature += delta;
    }

    // 3. State transitions
    let transitions = apply_transitions(voxels, registry);

    // 4. Pressure diffusion
    let max_pressure_delta = diffuse_pressure_sim(voxels, size, 0.25, registry);

    // Collect grid-wide stats
    let mut max_temp: f32 = 0.0;
    let mut max_pressure: f32 = 0.0;
    for v in voxels.iter() {
        max_temp = max_temp.max(v.temperature);
        max_pressure = max_pressure.max(v.pressure);
    }

    TickResult {
        reactions_fired,
        transitions,
        max_temp,
        max_pressure,
        max_pressure_delta,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{MaterialData, Phase};
    use crate::world::voxel::{MaterialId, Voxel};

    fn minimal_registry() -> MaterialRegistry {
        let mut reg = MaterialRegistry::new();
        reg.insert(MaterialData {
            id: 0,
            name: "Air".into(),
            default_phase: Phase::Gas,
            density: 1.225,
            thermal_conductivity: 0.026,
            specific_heat_capacity: 1005.0,
            ..Default::default()
        });
        reg.insert(MaterialData {
            id: 5,
            name: "Wood".into(),
            default_phase: Phase::Solid,
            density: 600.0,
            thermal_conductivity: 0.15,
            specific_heat_capacity: 1700.0,
            ignition_point: Some(573.0),
            hardness: 0.3,
            color: [0.6, 0.4, 0.2],
            ..Default::default()
        });
        reg.insert(MaterialData {
            id: 11,
            name: "Ash".into(),
            default_phase: Phase::Solid,
            density: 500.0,
            thermal_conductivity: 0.15,
            specific_heat_capacity: 800.0,
            melting_point: Some(1273.0),
            hardness: 0.05,
            color: [0.3, 0.3, 0.3],
            ..Default::default()
        });
        reg
    }

    fn wood_combustion_rule() -> ReactionData {
        ReactionData {
            name: "Wood combustion".into(),
            input_a: "Wood".into(),
            input_b: Some("Air".into()),
            min_temperature: 573.0,
            max_temperature: 99999.0,
            output_a: "Ash".into(),
            output_b: None,
            heat_output: 3500.0,
        }
    }

    #[test]
    fn tick_with_no_rules_only_diffuses() {
        let size = 4;
        let mut voxels = vec![Voxel::new(MaterialId::AIR); size * size * size];
        voxels[0].temperature = 500.0;

        let registry = minimal_registry();
        let result = simulate_tick(&mut voxels, size, &[], &registry, 1.0);

        assert_eq!(result.reactions_fired, 0);
        assert_eq!(result.transitions, 0);
    }

    #[test]
    fn tick_fires_reaction_above_ignition() {
        let size = 4;
        let mut voxels = vec![Voxel::new(MaterialId::AIR); size * size * size];
        // Place one wood voxel at (0,0,0), neighbor air at (1,0,0)
        voxels[0].material = MaterialId::WOOD;
        voxels[0].temperature = 800.0;

        let registry = minimal_registry();
        let rules = vec![wood_combustion_rule()];
        let result = simulate_tick(&mut voxels, size, &rules, &registry, 1.0);

        assert!(result.reactions_fired > 0, "Should fire reaction");
        assert_eq!(voxels[0].material, MaterialId::ASH);
    }

    #[test]
    fn gas_phase_materials_are_permeable() {
        let mut reg = MaterialRegistry::new();
        reg.insert(MaterialData {
            id: 14,
            name: "Hydrogen".into(),
            default_phase: Phase::Gas,
            density: 0.0899,
            ..Default::default()
        });
        reg.insert(MaterialData {
            id: 1,
            name: "Stone".into(),
            default_phase: Phase::Solid,
            density: 2700.0,
            ..Default::default()
        });

        assert!(is_permeable(MaterialId::AIR, &reg));
        assert!(is_permeable(MaterialId::HYDROGEN, &reg));
        assert!(!is_permeable(MaterialId::STONE, &reg));
    }

    /// Regression test: oxyhydrogen chain reaction must propagate.
    ///
    /// Places H₂ and O₂ in an alternating pattern with a 900K hot spot.
    /// Verifies that the initial reaction fires AND heat propagates enough
    /// to ignite additional H₂/O₂ pairs over subsequent ticks.
    #[test]
    fn oxyhydrogen_chain_reaction_propagates() {
        let mut reg = MaterialRegistry::new();
        reg.insert(MaterialData {
            id: 0,
            name: "Air".into(),
            default_phase: Phase::Gas,
            density: 1.225,
            thermal_conductivity: 0.026,
            specific_heat_capacity: 1005.0,
            ..Default::default()
        });
        reg.insert(MaterialData {
            id: 14,
            name: "Hydrogen".into(),
            default_phase: Phase::Gas,
            density: 0.0899,
            thermal_conductivity: 0.1805,
            specific_heat_capacity: 14304.0,
            ..Default::default()
        });
        reg.insert(MaterialData {
            id: 13,
            name: "Oxygen".into(),
            default_phase: Phase::Gas,
            density: 1.429,
            thermal_conductivity: 0.02658,
            specific_heat_capacity: 918.0,
            ..Default::default()
        });
        reg.insert(MaterialData {
            id: 9,
            name: "Steam".into(),
            default_phase: Phase::Gas,
            density: 0.6,
            thermal_conductivity: 0.025,
            specific_heat_capacity: 2010.0,
            ..Default::default()
        });

        let rule = ReactionData {
            name: "Oxyhydrogen".into(),
            input_a: "Hydrogen".into(),
            input_b: Some("Oxygen".into()),
            min_temperature: 843.0,
            max_temperature: 99999.0,
            output_a: "Steam".into(),
            output_b: Some("Steam".into()),
            heat_output: 3500.0,
        };

        // 4³ grid: alternating H₂ and O₂ (checkerboard).
        // Pre-heat to 800 K so heat only needs to diffuse ~43 K to cross the
        // 843 K ignition threshold.  Use a large dt so heat traverses the O₂
        // buffer layer (H₂ voxels are always 2 steps apart in a checkerboard).
        let size = 4;
        let mut voxels = vec![Voxel::new(MaterialId::AIR); size * size * size];
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let i = z * size * size + y * size + x;
                    voxels[i].temperature = 800.0;
                    if (x + y + z) % 2 == 0 {
                        voxels[i].material = MaterialId::HYDROGEN;
                    } else {
                        voxels[i].material = MaterialId::OXYGEN;
                    }
                }
            }
        }

        // Hot spot at (2,2,2) — even sum so it's H₂
        let center = 2 * size * size + 2 * size + 2;
        assert_eq!(
            voxels[center].material,
            MaterialId::HYDROGEN,
            "hot spot should be H₂"
        );
        voxels[center].temperature = 900.0;

        let rules = vec![rule];
        let mut total_reactions = 0;
        for _tick in 0..100 {
            let result = simulate_tick(&mut voxels, size, &rules, &reg, 5000.0);
            total_reactions += result.reactions_fired;
        }

        let steam_count = voxels
            .iter()
            .filter(|v| v.material == MaterialId::STEAM)
            .count();

        assert!(
            total_reactions >= 5,
            "Chain reaction should propagate: got {total_reactions} reactions"
        );
        assert!(steam_count >= 5, "Should produce steam: got {steam_count}");
    }
}
