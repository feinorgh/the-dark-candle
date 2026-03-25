// Simulation scenario: data-driven multi-tick simulation test runner.
//
// A `SimulationScenario` is loaded from a `.simulation.ron` file and describes:
//   - Grid setup (size, ambient conditions)
//   - Material and reaction sources (from assets or inline)
//   - Geometry (regions that populate the voxel grid)
//   - Ignition (initial temperature perturbations)
//   - Simulation parameters (ticks, dt)
//   - Assertions (checked after the simulation completes)
//
// To add a new test: create a `.simulation.ron` file in `tests/cases/simulation/`.

use std::path::Path;

use serde::Deserialize;

use crate::chemistry::heat::thermal_conductivity;
use crate::chemistry::reactions::ReactionData;
use crate::data::{MaterialData, MaterialRegistry};
use crate::simulation::assertions::{Assertion, evaluate};
use crate::simulation::geometry::{Region, apply_regions};
use crate::simulation::{SimulationStats, simulate_tick};
use crate::world::voxel::Voxel;

/// Where to load materials from.
#[derive(Deserialize, Debug, Clone, PartialEq)]
pub enum MaterialSource {
    /// Load all `.material.ron` files from `assets/data/materials/`.
    FromAssets,
    /// Use inline material definitions (for self-contained test scenarios).
    Inline(Vec<MaterialData>),
}

/// Where to load reaction rules from.
#[derive(Deserialize, Debug, Clone, PartialEq)]
pub enum ReactionSource {
    /// Load all `.reaction.ron` files from `assets/data/reactions/`.
    FromAssets,
    /// Use inline reaction definitions.
    Inline(Vec<ReactionData>),
}

/// An initial temperature perturbation (spark, heat source, etc.).
#[derive(Deserialize, Debug, Clone, PartialEq)]
pub enum Ignition {
    /// Set a single voxel to a specific temperature.
    HotSpot {
        pos: (usize, usize, usize),
        temperature: f32,
    },
    /// Heat an entire region to a specific temperature.
    HeatRegion {
        min: (usize, usize, usize),
        max: (usize, usize, usize),
        temperature: f32,
    },
}

/// How the ambient temperature changes over the simulation.
#[derive(Deserialize, Debug, Clone, PartialEq, Default)]
pub enum AmbientSchedule {
    /// Constant ambient temperature throughout the simulation (default).
    #[default]
    Constant,
    /// Linearly ramp from the scenario's `ambient_temperature` to
    /// `end_temperature` over `ramp_seconds`, then hold at `end_temperature`.
    RampThenHold {
        end_temperature: f32,
        ramp_seconds: f32,
    },
}

impl AmbientSchedule {
    /// Compute the ambient temperature at `elapsed` seconds into the simulation.
    fn temperature_at(&self, base: f32, elapsed: f32) -> f32 {
        match self {
            AmbientSchedule::Constant => base,
            AmbientSchedule::RampThenHold {
                end_temperature,
                ramp_seconds,
            } => {
                if *ramp_seconds <= 0.0 || elapsed >= *ramp_seconds {
                    *end_temperature
                } else {
                    let t = elapsed / ramp_seconds;
                    base + t * (end_temperature - base)
                }
            }
        }
    }
}

/// When to stop the simulation tick loop.
#[derive(Deserialize, Debug, Clone, PartialEq, Default)]
pub enum StopCondition {
    /// Run for exactly `ticks` ticks (default).
    #[default]
    FixedTicks,
    /// Run until all assertions pass, up to `ticks` as the maximum cap.
    /// Reports the tick at which convergence was reached.
    UntilAllPass,
}

/// A complete simulation scenario loaded from `.simulation.ron`.
#[derive(Deserialize, Debug, Clone, PartialEq)]
pub struct SimulationScenario {
    /// Human-readable name for test output.
    pub name: String,
    /// Description of what this scenario tests.
    pub description: String,

    /// Grid edge length (grid is `size³` voxels).
    pub grid_size: usize,

    /// Ambient temperature in Kelvin (default 288.15 K).
    #[serde(default = "default_ambient_temperature")]
    pub ambient_temperature: f32,
    /// Ambient pressure in Pascals (default 101325 Pa).
    #[serde(default = "default_ambient_pressure")]
    pub ambient_pressure: f32,

    /// How ambient temperature changes over time (default: Constant).
    #[serde(default)]
    pub ambient_schedule: AmbientSchedule,

    /// Convective heat transfer coefficient (W/(m²·K)) for boundary air voxels.
    ///
    /// When set, boundary air voxels are clamped to the current ambient
    /// temperature each tick, modeling convective mixing that conduction alone
    /// cannot capture at 1 m voxel resolution.
    ///
    /// TODO: Replace with LOD/octree dynamic resolution model when native
    /// octree physics is implemented. Finer resolution at air–material
    /// interfaces will capture convective effects directly, eliminating the
    /// need for this boundary correction.
    #[serde(default)]
    pub boundary_htc: Option<f32>,

    /// Where to load material definitions.
    pub material_source: MaterialSource,
    /// Where to load reaction rules.
    pub reaction_source: ReactionSource,

    /// Geometry regions applied in order to build the initial voxel grid.
    pub regions: Vec<Region>,
    /// Temperature perturbations applied after geometry setup.
    #[serde(default)]
    pub ignition: Vec<Ignition>,

    /// Maximum number of simulation ticks to run.
    pub ticks: u32,
    /// Timestep in seconds per tick.
    pub dt: f32,

    /// When to stop the simulation (default: FixedTicks).
    #[serde(default)]
    pub stop_condition: StopCondition,

    /// Assertions evaluated after the simulation completes (or each tick for
    /// `StopCondition::UntilAllPass`).
    pub assertions: Vec<Assertion>,
}

fn default_ambient_temperature() -> f32 {
    288.15
}

fn default_ambient_pressure() -> f32 {
    101_325.0
}

/// 3D index into a flat `size³` array.
fn idx(x: usize, y: usize, z: usize, size: usize) -> usize {
    z * size * size + y * size + x
}

/// Build a `MaterialRegistry` from the given source.
fn build_registry(source: &MaterialSource) -> Result<MaterialRegistry, String> {
    match source {
        MaterialSource::FromAssets => load_materials_from_assets(),
        MaterialSource::Inline(materials) => {
            let mut reg = MaterialRegistry::new();
            for m in materials {
                reg.insert(m.clone());
            }
            Ok(reg)
        }
    }
}

/// Load all `.material.ron` files from `assets/data/materials/`.
fn load_materials_from_assets() -> Result<MaterialRegistry, String> {
    let dir = find_assets_dir()?.join("materials");
    let entries =
        std::fs::read_dir(&dir).map_err(|e| format!("cannot read {}: {e}", dir.display()))?;

    let mut reg = MaterialRegistry::new();
    let mut found = false;
    for entry in entries {
        let entry = entry.map_err(|e| format!("dir entry: {e}"))?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("ron") {
            continue;
        }
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        if !name.ends_with(".material.ron") {
            continue;
        }
        let text = std::fs::read_to_string(&path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let data: MaterialData =
            ron::from_str(&text).map_err(|e| format!("cannot parse {}: {e}", path.display()))?;
        reg.insert(data);
        found = true;
    }

    if !found {
        return Err(format!("no .material.ron files found in {}", dir.display()));
    }
    Ok(reg)
}

/// Load all `.reaction.ron` files from `assets/data/reactions/`.
fn load_reactions_from_assets() -> Result<Vec<ReactionData>, String> {
    let dir = find_assets_dir()?.join("reactions");
    let entries =
        std::fs::read_dir(&dir).map_err(|e| format!("cannot read {}: {e}", dir.display()))?;

    let mut rules = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| format!("dir entry: {e}"))?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("ron") {
            continue;
        }
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        if !name.ends_with(".reaction.ron") {
            continue;
        }
        let text = std::fs::read_to_string(&path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let data: ReactionData =
            ron::from_str(&text).map_err(|e| format!("cannot parse {}: {e}", path.display()))?;
        rules.push(data);
    }
    Ok(rules)
}

/// Build reaction list from the given source.
fn build_reactions(source: &ReactionSource) -> Result<Vec<ReactionData>, String> {
    match source {
        ReactionSource::FromAssets => load_reactions_from_assets(),
        ReactionSource::Inline(rules) => Ok(rules.clone()),
    }
}

/// Locate the `assets/data/` directory relative to the cargo manifest.
fn find_assets_dir() -> Result<std::path::PathBuf, String> {
    // In tests, CARGO_MANIFEST_DIR points to the project root.
    if let Ok(dir) = std::env::var("CARGO_MANIFEST_DIR") {
        let path = Path::new(&dir).join("assets").join("data");
        if path.is_dir() {
            return Ok(path);
        }
    }
    // Fallback: try relative to CWD.
    let path = Path::new("assets/data");
    if path.is_dir() {
        return Ok(path.to_path_buf());
    }
    Err("cannot find assets/data/ directory".into())
}

/// Face-adjacent neighbor offsets (6-connectivity).
const NEIGHBORS: [(i32, i32, i32); 6] = [
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
];

/// Apply convective boundary conditions.
///
/// 1. Clamp all air voxels to the current ambient temperature (models a
///    well-mixed atmospheric reservoir).
/// 2. For every non-air voxel adjacent to air, apply a correction flux that
///    boosts heat transfer from the conduction-only `k_eff` up to the target
///    `boundary_htc`. This models convective heat transfer at interfaces.
///
/// The correction formula per air-adjacent face:
///   `ΔT = (h − k_eff) × (T_air − T_voxel) × dt / (ρ × Cₚ)`
/// where `k_eff` is the harmonic mean conductivity already used by diffuse_chunk.
fn apply_boundary_conditions(
    voxels: &mut [Voxel],
    size: usize,
    ambient: f32,
    htc: f32,
    dt: f32,
    registry: &MaterialRegistry,
) {
    // Step 1: clamp air to ambient
    for v in voxels.iter_mut() {
        if v.material.is_air() {
            v.temperature = ambient;
        }
    }

    // Step 2: correction flux for non-air voxels adjacent to air
    // Use a snapshot so corrections don't cascade within a single tick.
    let snapshot: Vec<f32> = voxels.iter().map(|v| v.temperature).collect();

    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let i = idx(x, y, z, size);
                if voxels[i].material.is_air() {
                    continue;
                }

                let mat = registry.get(voxels[i].material);
                let self_k = mat.map(|m| m.thermal_conductivity).unwrap_or(0.1);
                let density = mat.map(|m| m.density).unwrap_or(1.0);
                let cp = mat.map(|m| m.specific_heat_capacity).unwrap_or(1000.0);
                let rho_cp = (density * cp).max(1.0);

                let mut correction = 0.0_f32;

                for &(dx, dy, dz) in &NEIGHBORS {
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
                        // Grid boundary — treat as ambient air
                        let k_eff = self_k.min(htc);
                        let extra = (htc - k_eff) * (ambient - snapshot[i]);
                        correction += extra * dt / rho_cp;
                        continue;
                    }

                    let ni = idx(nx as usize, ny as usize, nz as usize, size);
                    if !voxels[ni].material.is_air() {
                        continue;
                    }

                    // This neighbor is air (clamped to ambient).
                    // diffuse_chunk already applied k_eff conduction.
                    // Add correction: (h - k_eff) × ΔT × dt / ρCₚ
                    let air_k = thermal_conductivity(voxels[ni].material, registry);
                    let k_eff = if self_k + air_k > 0.0 {
                        2.0 * self_k * air_k / (self_k + air_k)
                    } else {
                        0.0
                    };
                    let extra = (htc - k_eff).max(0.0) * (ambient - snapshot[i]);
                    correction += extra * dt / rho_cp;
                }

                voxels[i].temperature += correction;
            }
        }
    }
}

/// Run a simulation scenario end-to-end: setup → simulate → assert.
///
/// Returns `Ok(())` if all assertions pass, or `Err(message)` with the first
/// failure and a summary of the simulation state.
pub fn run_scenario(scenario: &SimulationScenario) -> Result<(), String> {
    // 1. Build registry and reactions
    let registry = build_registry(&scenario.material_source)?;
    let rules = build_reactions(&scenario.reaction_source)?;

    // 2. Create voxel grid with ambient conditions
    let size = scenario.grid_size;
    let total = size * size * size;
    let mut voxels: Vec<Voxel> = (0..total)
        .map(|_| Voxel {
            temperature: scenario.ambient_temperature,
            pressure: scenario.ambient_pressure,
            ..Default::default()
        })
        .collect();

    // 3. Apply geometry regions
    apply_regions(&mut voxels, size, &scenario.regions, &registry)?;

    // 4. Apply ignition conditions
    for ig in &scenario.ignition {
        match ig {
            Ignition::HotSpot { pos, temperature } => {
                if pos.0 < size && pos.1 < size && pos.2 < size {
                    voxels[idx(pos.0, pos.1, pos.2, size)].temperature = *temperature;
                }
            }
            Ignition::HeatRegion {
                min,
                max,
                temperature,
            } => {
                for z in min.2..=max.2.min(size - 1) {
                    for y in min.1..=max.1.min(size - 1) {
                        for x in min.0..=max.0.min(size - 1) {
                            voxels[idx(x, y, z, size)].temperature = *temperature;
                        }
                    }
                }
            }
        }
    }

    // 5. Run simulation
    let mut stats = SimulationStats::default();
    let mut converged_at: Option<u32> = None;

    for tick_num in 0..scenario.ticks {
        // Compute time-varying ambient temperature
        let elapsed = tick_num as f32 * scenario.dt;
        let current_ambient = scenario
            .ambient_schedule
            .temperature_at(scenario.ambient_temperature, elapsed);

        // Apply convective boundary conditions
        if let Some(htc) = scenario.boundary_htc {
            apply_boundary_conditions(
                &mut voxels,
                size,
                current_ambient,
                htc,
                scenario.dt,
                &registry,
            );
        }

        let tick = simulate_tick(&mut voxels, size, &rules, &registry, scenario.dt);
        stats.accumulate(&tick);

        // For UntilAllPass: check assertions each tick
        if scenario.stop_condition == StopCondition::UntilAllPass {
            let all_pass = scenario
                .assertions
                .iter()
                .all(|a| evaluate(a, &voxels, size, &registry, &stats).is_ok());
            if all_pass {
                converged_at = Some(tick_num + 1);
                break;
            }
        }
    }

    if let Some(tick) = converged_at {
        let sim_days = tick as f64 * scenario.dt as f64 / 86400.0;
        eprintln!(
            "  ✓ {} converged at tick {} ({:.1} sim-days)",
            scenario.name, tick, sim_days
        );
    }

    // 6. Evaluate assertions against final state
    let mut failures = Vec::new();
    for assertion in &scenario.assertions {
        if let Err(msg) = evaluate(assertion, &voxels, size, &registry, &stats) {
            failures.push(msg);
        }
    }

    if failures.is_empty() {
        Ok(())
    } else {
        let actual_ticks = converged_at.unwrap_or(scenario.ticks);
        Err(format!(
            "{}: {} assertion(s) failed after {} ticks:\n  {}",
            scenario.name,
            failures.len(),
            actual_ticks,
            failures.join("\n  ")
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Phase;

    fn inline_registry_materials() -> Vec<MaterialData> {
        vec![
            MaterialData {
                id: 0,
                name: "Air".into(),
                default_phase: Phase::Gas,
                density: 1.225,
                thermal_conductivity: 0.026,
                specific_heat_capacity: 1005.0,
                transparent: true,
                color: [0.8, 0.9, 1.0],
                ..Default::default()
            },
            MaterialData {
                id: 1,
                name: "Stone".into(),
                default_phase: Phase::Solid,
                density: 2700.0,
                thermal_conductivity: 0.8,
                specific_heat_capacity: 840.0,
                hardness: 0.9,
                color: [0.5, 0.5, 0.5],
                ..Default::default()
            },
        ]
    }

    #[test]
    fn trivial_scenario_passes() {
        let scenario = SimulationScenario {
            name: "Trivial".into(),
            description: "Empty grid, no reactions".into(),
            grid_size: 4,
            ambient_temperature: 288.15,
            ambient_pressure: 101_325.0,
            ambient_schedule: AmbientSchedule::Constant,
            boundary_htc: None,
            material_source: MaterialSource::Inline(inline_registry_materials()),
            reaction_source: ReactionSource::Inline(vec![]),
            regions: vec![],
            ignition: vec![],
            ticks: 10,
            dt: 1.0,
            stop_condition: StopCondition::FixedTicks,
            assertions: vec![Assertion::NoReactions],
        };
        assert!(run_scenario(&scenario).is_ok());
    }

    #[test]
    fn scenario_with_failing_assertion() {
        let scenario = SimulationScenario {
            name: "Should fail".into(),
            description: "Expects stone but grid is empty".into(),
            grid_size: 4,
            ambient_temperature: 288.15,
            ambient_pressure: 101_325.0,
            ambient_schedule: AmbientSchedule::Constant,
            boundary_htc: None,
            material_source: MaterialSource::Inline(inline_registry_materials()),
            reaction_source: ReactionSource::Inline(vec![]),
            regions: vec![],
            ignition: vec![],
            ticks: 1,
            dt: 1.0,
            stop_condition: StopCondition::FixedTicks,
            assertions: vec![Assertion::MaterialCountGt {
                material: "Stone".into(),
                min_count: 1,
            }],
        };
        assert!(run_scenario(&scenario).is_err());
    }

    #[test]
    fn ambient_ramp_then_hold() {
        let schedule = AmbientSchedule::RampThenHold {
            end_temperature: 253.15,
            ramp_seconds: 100.0,
        };
        let base = 278.15;
        assert!((schedule.temperature_at(base, 0.0) - 278.15).abs() < 0.01);
        assert!((schedule.temperature_at(base, 50.0) - 265.65).abs() < 0.01);
        assert!((schedule.temperature_at(base, 100.0) - 253.15).abs() < 0.01);
        assert!((schedule.temperature_at(base, 200.0) - 253.15).abs() < 0.01);
    }

    #[test]
    fn until_all_pass_stops_early() {
        let scenario = SimulationScenario {
            name: "Early stop".into(),
            description: "Should stop before max ticks".into(),
            grid_size: 4,
            ambient_temperature: 288.15,
            ambient_pressure: 101_325.0,
            ambient_schedule: AmbientSchedule::Constant,
            boundary_htc: None,
            material_source: MaterialSource::Inline(inline_registry_materials()),
            reaction_source: ReactionSource::Inline(vec![]),
            regions: vec![],
            ignition: vec![],
            ticks: 1000,
            dt: 1.0,
            stop_condition: StopCondition::UntilAllPass,
            assertions: vec![Assertion::NoReactions],
        };
        // Should pass on the very first tick (no reactions in empty grid)
        assert!(run_scenario(&scenario).is_ok());
    }
}
