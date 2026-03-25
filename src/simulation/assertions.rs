// Assertion types for verifying simulation outcomes.
//
// Each `Assertion` variant checks one property of the voxel grid after (or
// during) a simulation run. Assertions are evaluated against the final grid
// state and cumulative `SimulationStats`.
//
// To add a new assertion type: add a variant to the enum, implement its
// evaluation in `evaluate()`, and it becomes immediately usable in RON files.

use serde::Deserialize;

use crate::data::MaterialRegistry;
use crate::world::voxel::Voxel;

use super::SimulationStats;

/// A single assertion to check against the simulation outcome.
///
/// All variants are RON-deserializable. Material names are resolved via the
/// `MaterialRegistry` at evaluation time.
#[derive(Deserialize, Debug, Clone, PartialEq)]
pub enum Assertion {
    /// Exact material count (within tolerance).
    MaterialCountEq {
        material: String,
        count: usize,
        tolerance: usize,
    },

    /// At least `min_count` voxels of this material.
    MaterialCountGt { material: String, min_count: usize },

    /// At most `max_count` voxels of this material.
    MaterialCountLt { material: String, max_count: usize },

    /// No voxels of this material remain.
    MaterialAbsent { material: String },

    /// Average temperature in a region exceeds a threshold.
    RegionAvgTempGt {
        min: (usize, usize, usize),
        max: (usize, usize, usize),
        threshold: f32,
    },

    /// Average temperature in a region is below a threshold.
    RegionAvgTempLt {
        min: (usize, usize, usize),
        max: (usize, usize, usize),
        threshold: f32,
    },

    /// Temperature at a specific voxel exceeds a threshold.
    VoxelTempGt {
        pos: (usize, usize, usize),
        threshold: f32,
    },

    /// Average pressure in a region exceeds a threshold.
    RegionAvgPressureGt {
        min: (usize, usize, usize),
        max: (usize, usize, usize),
        threshold: f32,
    },

    /// Grid-wide maximum temperature exceeds a threshold.
    MaxTempGt { threshold: f32 },

    /// Total reactions fired across all ticks exceeds a minimum.
    TotalReactionsGt { min_count: usize },

    /// No reactions should have occurred (negative test).
    NoReactions,
}

/// 3D index into a flat `size³` array.
fn idx(x: usize, y: usize, z: usize, size: usize) -> usize {
    z * size * size + y * size + x
}

/// Count voxels of a given material by name.
fn count_material(voxels: &[Voxel], name: &str, registry: &MaterialRegistry) -> Option<usize> {
    let mat = registry.resolve_name(name)?;
    Some(voxels.iter().filter(|v| v.material == mat).count())
}

/// Compute the average of a float property across a box region.
fn region_avg(
    voxels: &[Voxel],
    size: usize,
    min: (usize, usize, usize),
    max: (usize, usize, usize),
    extract: impl Fn(&Voxel) -> f32,
) -> f32 {
    let mut sum = 0.0_f64;
    let mut count = 0u64;
    for z in min.2..=max.2.min(size - 1) {
        for y in min.1..=max.1.min(size - 1) {
            for x in min.0..=max.0.min(size - 1) {
                sum += extract(&voxels[idx(x, y, z, size)]) as f64;
                count += 1;
            }
        }
    }
    if count == 0 {
        return 0.0;
    }
    (sum / count as f64) as f32
}

/// Evaluate a single assertion against the grid state and simulation stats.
///
/// Returns `Ok(())` if the assertion holds, or `Err(message)` describing the
/// failure.
pub fn evaluate(
    assertion: &Assertion,
    voxels: &[Voxel],
    size: usize,
    registry: &MaterialRegistry,
    stats: &SimulationStats,
) -> Result<(), String> {
    match assertion {
        Assertion::MaterialCountEq {
            material,
            count,
            tolerance,
        } => {
            let actual = count_material(voxels, material, registry)
                .ok_or_else(|| format!("unknown material: {material:?}"))?;
            let diff = actual.abs_diff(*count);
            if diff > *tolerance {
                return Err(format!(
                    "MaterialCountEq({material:?}): expected {count} ±{tolerance}, got {actual}"
                ));
            }
        }

        Assertion::MaterialCountGt {
            material,
            min_count,
        } => {
            let actual = count_material(voxels, material, registry)
                .ok_or_else(|| format!("unknown material: {material:?}"))?;
            if actual < *min_count {
                return Err(format!(
                    "MaterialCountGt({material:?}): expected ≥{min_count}, got {actual}"
                ));
            }
        }

        Assertion::MaterialCountLt {
            material,
            max_count,
        } => {
            let actual = count_material(voxels, material, registry)
                .ok_or_else(|| format!("unknown material: {material:?}"))?;
            if actual > *max_count {
                return Err(format!(
                    "MaterialCountLt({material:?}): expected ≤{max_count}, got {actual}"
                ));
            }
        }

        Assertion::MaterialAbsent { material } => {
            let actual = count_material(voxels, material, registry)
                .ok_or_else(|| format!("unknown material: {material:?}"))?;
            if actual > 0 {
                return Err(format!(
                    "MaterialAbsent({material:?}): expected 0, got {actual}"
                ));
            }
        }

        Assertion::RegionAvgTempGt {
            min,
            max,
            threshold,
        } => {
            let avg = region_avg(voxels, size, *min, *max, |v| v.temperature);
            if avg <= *threshold {
                return Err(format!(
                    "RegionAvgTempGt: expected avg temp > {threshold} K, got {avg:.1} K"
                ));
            }
        }

        Assertion::RegionAvgTempLt {
            min,
            max,
            threshold,
        } => {
            let avg = region_avg(voxels, size, *min, *max, |v| v.temperature);
            if avg >= *threshold {
                return Err(format!(
                    "RegionAvgTempLt: expected avg temp < {threshold} K, got {avg:.1} K"
                ));
            }
        }

        Assertion::VoxelTempGt { pos, threshold } => {
            if pos.0 >= size || pos.1 >= size || pos.2 >= size {
                return Err(format!(
                    "VoxelTempGt: pos {pos:?} out of bounds (size={size})"
                ));
            }
            let temp = voxels[idx(pos.0, pos.1, pos.2, size)].temperature;
            if temp <= *threshold {
                return Err(format!(
                    "VoxelTempGt({pos:?}): expected > {threshold} K, got {temp:.1} K"
                ));
            }
        }

        Assertion::RegionAvgPressureGt {
            min,
            max,
            threshold,
        } => {
            let avg = region_avg(voxels, size, *min, *max, |v| v.pressure);
            if avg <= *threshold {
                return Err(format!(
                    "RegionAvgPressureGt: expected avg pressure > {threshold} Pa, got {avg:.1} Pa"
                ));
            }
        }

        Assertion::MaxTempGt { threshold } => {
            if stats.peak_temp <= *threshold {
                return Err(format!(
                    "MaxTempGt: expected peak temp > {threshold} K, got {:.1} K",
                    stats.peak_temp
                ));
            }
        }

        Assertion::TotalReactionsGt { min_count } => {
            if stats.total_reactions < *min_count {
                return Err(format!(
                    "TotalReactionsGt: expected ≥{min_count} reactions, got {}",
                    stats.total_reactions
                ));
            }
        }

        Assertion::NoReactions => {
            if stats.total_reactions > 0 {
                return Err(format!(
                    "NoReactions: expected 0 reactions, got {}",
                    stats.total_reactions
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{MaterialData, Phase};
    use crate::world::voxel::{MaterialId, Voxel};

    fn test_registry() -> MaterialRegistry {
        let mut reg = MaterialRegistry::new();
        reg.insert(MaterialData {
            id: 0,
            name: "Air".into(),
            default_phase: Phase::Gas,
            density: 1.225,
            ..Default::default()
        });
        reg.insert(MaterialData {
            id: 1,
            name: "Stone".into(),
            default_phase: Phase::Solid,
            density: 2700.0,
            ..Default::default()
        });
        reg
    }

    fn empty_stats() -> SimulationStats {
        SimulationStats::default()
    }

    #[test]
    fn material_count_eq_passes() {
        let size = 2;
        let mut voxels = vec![Voxel::new(MaterialId::AIR); size * size * size];
        voxels[0].material = MaterialId::STONE;
        voxels[1].material = MaterialId::STONE;

        let reg = test_registry();
        let a = Assertion::MaterialCountEq {
            material: "Stone".into(),
            count: 2,
            tolerance: 0,
        };
        assert!(evaluate(&a, &voxels, size, &reg, &empty_stats()).is_ok());
    }

    #[test]
    fn material_count_eq_fails() {
        let size = 2;
        let voxels = vec![Voxel::new(MaterialId::AIR); size * size * size];

        let reg = test_registry();
        let a = Assertion::MaterialCountEq {
            material: "Stone".into(),
            count: 2,
            tolerance: 0,
        };
        assert!(evaluate(&a, &voxels, size, &reg, &empty_stats()).is_err());
    }

    #[test]
    fn no_reactions_passes_with_zero() {
        let size = 2;
        let voxels = vec![Voxel::new(MaterialId::AIR); size * size * size];
        let reg = test_registry();
        let a = Assertion::NoReactions;
        assert!(evaluate(&a, &voxels, size, &reg, &empty_stats()).is_ok());
    }

    #[test]
    fn total_reactions_gt_fails_when_zero() {
        let size = 2;
        let voxels = vec![Voxel::new(MaterialId::AIR); size * size * size];
        let reg = test_registry();
        let a = Assertion::TotalReactionsGt { min_count: 1 };
        assert!(evaluate(&a, &voxels, size, &reg, &empty_stats()).is_err());
    }

    #[test]
    fn region_avg_temp_computes_correctly() {
        let size = 4;
        let mut voxels = vec![Voxel::new(MaterialId::AIR); size * size * size];
        // Set a 2x2x2 region to 500 K
        for z in 1..=2 {
            for y in 1..=2 {
                for x in 1..=2 {
                    voxels[idx(x, y, z, size)].temperature = 500.0;
                }
            }
        }

        let reg = test_registry();
        let a = Assertion::RegionAvgTempGt {
            min: (1, 1, 1),
            max: (2, 2, 2),
            threshold: 400.0,
        };
        assert!(evaluate(&a, &voxels, size, &reg, &empty_stats()).is_ok());
    }
}
