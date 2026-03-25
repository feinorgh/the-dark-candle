// Structured state dump for AI-agent-friendly diagnostics.
//
// Produces a serializable snapshot of voxel grid state that can be emitted
// as RON text for consumption by CLI-based AI agents. Works with both the
// headless simulation grid (`dump_grid_state`) and live Bevy ECS chunks
// (`dump_chunk`).
//
// The output is designed to be human-readable (pretty-printed RON) while
// remaining machine-parseable for automated analysis.

use std::collections::BTreeMap;

use serde::Serialize;

use crate::data::MaterialRegistry;
use crate::simulation::SimulationStats;
use crate::world::voxel::Voxel;

/// Top-level state dump combining grid summary with optional raw voxel data.
#[derive(Serialize, Debug, Clone)]
pub struct StateDump {
    /// Grid edge length (grid is `size³` voxels).
    pub grid_size: usize,
    /// Total voxel count in the grid.
    pub total_voxels: usize,
    /// Summary statistics for the grid.
    pub summary: GridSummary,
    /// Cumulative simulation stats (if from a simulation run).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub simulation_stats: Option<SimulationStatsSnapshot>,
    /// Full voxel data, keyed by `"x,y,z"`. Only present when requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voxels: Option<Vec<VoxelSnapshot>>,
}

/// Aggregated statistics across the entire grid.
#[derive(Serialize, Debug, Clone)]
pub struct GridSummary {
    /// Material distribution: material name → count.
    pub material_histogram: BTreeMap<String, usize>,
    /// Temperature statistics across all voxels.
    pub temperature: RangeStats,
    /// Pressure statistics across all voxels.
    pub pressure: RangeStats,
    /// Damage statistics across all non-air voxels.
    pub damage: RangeStats,
    /// Per-material breakdown with thermal/pressure details.
    pub per_material: BTreeMap<String, MaterialStats>,
}

/// Min/max/mean statistics for a scalar property.
#[derive(Serialize, Debug, Clone, Copy)]
pub struct RangeStats {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub count: usize,
}

impl Default for RangeStats {
    fn default() -> Self {
        Self {
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
            mean: 0.0,
            count: 0,
        }
    }
}

impl RangeStats {
    /// Accumulate a single value into the running stats.
    pub fn accumulate(&mut self, value: f32) {
        self.min = self.min.min(value);
        self.max = self.max.max(value);
        // Running sum stored temporarily in `mean`; finalized later.
        self.mean += value;
        self.count += 1;
    }

    /// Convert the running sum in `mean` to the actual average.
    pub fn finalize(&mut self) {
        if self.count > 0 {
            self.mean /= self.count as f32;
        } else {
            self.min = 0.0;
            self.max = 0.0;
        }
    }
}

/// Per-material aggregate statistics.
#[derive(Serialize, Debug, Clone)]
pub struct MaterialStats {
    pub count: usize,
    pub temperature: RangeStats,
    pub pressure: RangeStats,
}

/// Snapshot of cumulative simulation statistics.
#[derive(Serialize, Debug, Clone)]
pub struct SimulationStatsSnapshot {
    pub total_reactions: usize,
    pub total_transitions: usize,
    pub peak_temperature_k: f32,
    pub peak_pressure_pa: f32,
}

impl From<&SimulationStats> for SimulationStatsSnapshot {
    fn from(stats: &SimulationStats) -> Self {
        Self {
            total_reactions: stats.total_reactions,
            total_transitions: stats.total_transitions,
            peak_temperature_k: stats.peak_temp,
            peak_pressure_pa: stats.peak_pressure,
        }
    }
}

/// A single voxel's state for the full dump.
#[derive(Serialize, Debug, Clone)]
pub struct VoxelSnapshot {
    pub x: usize,
    pub y: usize,
    pub z: usize,
    pub material: String,
    pub temperature_k: f32,
    pub pressure_pa: f32,
    pub damage: f32,
    pub latent_heat_buffer_j_per_kg: f32,
}

/// Produce a structured state dump from a flat `size³` voxel array.
///
/// This works with the headless simulation grid (no Bevy ECS required).
///
/// # Arguments
/// * `voxels` — flat array of `size³` voxels
/// * `size` — grid edge length
/// * `registry` — material name lookup
/// * `stats` — optional cumulative simulation statistics
/// * `include_voxels` — if true, includes every non-air voxel in the output
pub fn dump_grid_state(
    voxels: &[Voxel],
    size: usize,
    registry: &MaterialRegistry,
    stats: Option<&SimulationStats>,
    include_voxels: bool,
) -> StateDump {
    let mut histogram: BTreeMap<String, usize> = BTreeMap::new();
    let mut temp_stats = RangeStats::default();
    let mut pressure_stats = RangeStats::default();
    let mut damage_stats = RangeStats::default();
    let mut per_material: BTreeMap<String, MaterialStats> = BTreeMap::new();

    for voxel in voxels {
        let name = registry
            .get(voxel.material)
            .map(|m| m.name.clone())
            .unwrap_or_else(|| format!("Unknown({})", voxel.material.0));

        *histogram.entry(name.clone()).or_default() += 1;
        temp_stats.accumulate(voxel.temperature);
        pressure_stats.accumulate(voxel.pressure);

        if !voxel.is_air() {
            damage_stats.accumulate(voxel.damage);
        }

        let mat_stats = per_material.entry(name).or_insert_with(|| MaterialStats {
            count: 0,
            temperature: RangeStats::default(),
            pressure: RangeStats::default(),
        });
        mat_stats.count += 1;
        mat_stats.temperature.accumulate(voxel.temperature);
        mat_stats.pressure.accumulate(voxel.pressure);
    }

    temp_stats.finalize();
    pressure_stats.finalize();
    damage_stats.finalize();
    for mat in per_material.values_mut() {
        mat.temperature.finalize();
        mat.pressure.finalize();
    }

    let voxel_snapshots = if include_voxels {
        let mut snaps = Vec::new();
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let i = z * size * size + y * size + x;
                    let v = &voxels[i];
                    if v.is_air() {
                        continue;
                    }
                    let name = registry
                        .get(v.material)
                        .map(|m| m.name.clone())
                        .unwrap_or_else(|| format!("Unknown({})", v.material.0));
                    snaps.push(VoxelSnapshot {
                        x,
                        y,
                        z,
                        material: name,
                        temperature_k: v.temperature,
                        pressure_pa: v.pressure,
                        damage: v.damage,
                        latent_heat_buffer_j_per_kg: v.latent_heat_buffer,
                    });
                }
            }
        }
        Some(snaps)
    } else {
        None
    };

    StateDump {
        grid_size: size,
        total_voxels: voxels.len(),
        summary: GridSummary {
            material_histogram: histogram,
            temperature: temp_stats,
            pressure: pressure_stats,
            damage: damage_stats,
            per_material,
        },
        simulation_stats: stats.map(SimulationStatsSnapshot::from),
        voxels: voxel_snapshots,
    }
}

/// Serialize a `StateDump` to pretty-printed RON text.
pub fn dump_to_ron(dump: &StateDump) -> Result<String, ron::Error> {
    let config = ron::ser::PrettyConfig::new()
        .depth_limit(8)
        .struct_names(true);
    ron::ser::to_string_pretty(dump, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{MaterialData, Phase};
    use crate::world::voxel::MaterialId;

    fn test_registry() -> MaterialRegistry {
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
            id: 1,
            name: "Stone".into(),
            default_phase: Phase::Solid,
            density: 2700.0,
            thermal_conductivity: 0.8,
            specific_heat_capacity: 840.0,
            ..Default::default()
        });
        reg.insert(MaterialData {
            id: 3,
            name: "Water".into(),
            default_phase: Phase::Liquid,
            density: 1000.0,
            thermal_conductivity: 0.6,
            specific_heat_capacity: 4186.0,
            ..Default::default()
        });
        reg
    }

    #[test]
    fn dump_empty_grid_all_air() {
        let size = 4;
        let voxels = vec![Voxel::new(MaterialId::AIR); size * size * size];
        let reg = test_registry();

        let dump = dump_grid_state(&voxels, size, &reg, None, false);

        assert_eq!(dump.grid_size, 4);
        assert_eq!(dump.total_voxels, 64);
        assert_eq!(dump.summary.material_histogram["Air"], 64);
        assert_eq!(dump.summary.material_histogram.len(), 1);
        assert!(dump.simulation_stats.is_none());
        assert!(dump.voxels.is_none());
    }

    #[test]
    fn dump_mixed_grid_histogram() {
        let size = 4;
        let mut voxels = vec![Voxel::new(MaterialId::AIR); size * size * size];
        voxels[0].material = MaterialId::STONE;
        voxels[1].material = MaterialId::STONE;
        voxels[2].material = MaterialId::WATER;
        let reg = test_registry();

        let dump = dump_grid_state(&voxels, size, &reg, None, false);

        assert_eq!(dump.summary.material_histogram["Air"], 61);
        assert_eq!(dump.summary.material_histogram["Stone"], 2);
        assert_eq!(dump.summary.material_histogram["Water"], 1);
    }

    #[test]
    fn dump_temperature_stats() {
        let size = 2;
        let mut voxels = vec![Voxel::new(MaterialId::AIR); size * size * size];
        voxels[0].temperature = 200.0;
        voxels[1].temperature = 400.0;
        let reg = test_registry();

        let dump = dump_grid_state(&voxels, size, &reg, None, false);

        assert_eq!(dump.summary.temperature.min, 200.0);
        assert_eq!(dump.summary.temperature.max, 400.0);
        assert_eq!(dump.summary.temperature.count, 8);
    }

    #[test]
    fn dump_with_simulation_stats() {
        let size = 2;
        let voxels = vec![Voxel::new(MaterialId::AIR); size * size * size];
        let reg = test_registry();
        let stats = SimulationStats {
            total_reactions: 42,
            total_transitions: 7,
            peak_temp: 1500.0,
            peak_pressure: 200_000.0,
        };

        let dump = dump_grid_state(&voxels, size, &reg, Some(&stats), false);
        let ss = dump.simulation_stats.unwrap();

        assert_eq!(ss.total_reactions, 42);
        assert_eq!(ss.total_transitions, 7);
        assert_eq!(ss.peak_temperature_k, 1500.0);
        assert_eq!(ss.peak_pressure_pa, 200_000.0);
    }

    #[test]
    fn dump_include_voxels_skips_air() {
        let size = 2;
        let mut voxels = vec![Voxel::new(MaterialId::AIR); size * size * size];
        voxels[0].material = MaterialId::STONE;
        voxels[0].temperature = 500.0;
        let reg = test_registry();

        let dump = dump_grid_state(&voxels, size, &reg, None, true);
        let snaps = dump.voxels.unwrap();

        assert_eq!(snaps.len(), 1);
        assert_eq!(snaps[0].material, "Stone");
        assert_eq!(snaps[0].x, 0);
        assert_eq!(snaps[0].y, 0);
        assert_eq!(snaps[0].z, 0);
        assert_eq!(snaps[0].temperature_k, 500.0);
    }

    #[test]
    fn dump_per_material_stats() {
        let size = 2;
        let mut voxels = vec![Voxel::new(MaterialId::AIR); size * size * size];
        voxels[0].material = MaterialId::STONE;
        voxels[0].temperature = 300.0;
        voxels[1].material = MaterialId::STONE;
        voxels[1].temperature = 500.0;
        let reg = test_registry();

        let dump = dump_grid_state(&voxels, size, &reg, None, false);
        let stone = &dump.summary.per_material["Stone"];

        assert_eq!(stone.count, 2);
        assert_eq!(stone.temperature.min, 300.0);
        assert_eq!(stone.temperature.max, 500.0);
        assert!((stone.temperature.mean - 400.0).abs() < 0.01);
    }

    #[test]
    fn dump_to_ron_roundtrip() {
        let size = 2;
        let voxels = vec![Voxel::new(MaterialId::AIR); size * size * size];
        let reg = test_registry();
        let dump = dump_grid_state(&voxels, size, &reg, None, false);

        let ron_text = dump_to_ron(&dump).expect("RON serialization failed");
        assert!(ron_text.contains("material_histogram"));
        assert!(ron_text.contains("Air"));
        assert!(ron_text.contains("temperature"));
    }

    #[test]
    fn dump_unknown_material_handled() {
        let size = 2;
        let mut voxels = vec![Voxel::new(MaterialId::AIR); size * size * size];
        voxels[0].material = MaterialId(999);
        let reg = test_registry();

        let dump = dump_grid_state(&voxels, size, &reg, None, false);
        assert!(dump.summary.material_histogram.contains_key("Unknown(999)"));
    }

    #[test]
    fn range_stats_empty_grid() {
        let mut stats = RangeStats::default();
        stats.finalize();
        assert_eq!(stats.min, 0.0);
        assert_eq!(stats.max, 0.0);
        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.count, 0);
    }
}
