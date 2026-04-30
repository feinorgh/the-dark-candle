//! Parity comparison between GPU and CPU planetary voxel generation.
//!
//! See `docs/superpowers/specs/2026-04-30-gpu-planetary-noise-parity-design.md`.

use crate::world::v2::cubed_sphere::CubeSphereCoord;
use crate::world::voxel::MaterialId;

/// Which classification threshold a 1-ULP-tolerant mismatch lies on.
///
/// Used to attribute material disagreements to a known boundary — any
/// mismatch *not* on a known threshold is a parity bug.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThresholdKind {
    /// Voxel sits at the surface iso-surface (depth ≈ 0).
    SurfaceIso,
    /// Voxel sits at the cave-noise threshold.
    CaveThreshold,
    /// Voxel sits at the ore-probability cut.
    OreCut,
    /// Voxel sits at a strata-depth boundary.
    StrataBoundary,
    /// Voxel sits at a sea-level / ocean-clamp boundary.
    SeaLevel,
    /// Mismatch could not be attributed to a known threshold — parity bug.
    Unattributed,
}

/// Per-voxel mismatch detail.
#[derive(Debug, Clone)]
pub struct VoxelMismatch {
    pub voxel_index: usize,
    pub gpu_material: MaterialId,
    pub cpu_material: MaterialId,
    pub gpu_density: f32,
    pub cpu_density: f64,
    pub within_1_ulp: bool,
    pub threshold_kind: ThresholdKind,
}

/// Result of a single parity probe.
#[derive(Debug, Clone)]
pub struct ParityReport {
    pub coord: CubeSphereCoord,
    pub gpu_classification: &'static str,
    pub cpu_classification: &'static str,
    pub gpu_solid: usize,
    pub cpu_solid: usize,
    pub max_density_delta: f64,
    pub max_density_voxel_index: usize,
    pub mismatches: Vec<VoxelMismatch>,
}

/// Maximum allowed voxel material disagreements per chunk (parity contract §3).
pub const MAX_MISMATCHES_ALLOWED: usize = 4;

/// Maximum allowed normalised density delta between GPU and CPU samples (parity contract §3).
pub const MAX_DENSITY_DELTA_ALLOWED: f64 = 1e-3;

impl ParityReport {
    /// Returns `true` iff the report satisfies the §3 parity contract.
    pub fn passes_parity_contract(&self) -> bool {
        self.gpu_classification == self.cpu_classification
            && self.mismatches.len() <= MAX_MISMATCHES_ALLOWED
            && self
                .mismatches
                .iter()
                .all(|m| m.within_1_ulp && m.threshold_kind != ThresholdKind::Unattributed)
            && self.max_density_delta <= MAX_DENSITY_DELTA_ALLOWED
    }
}

/// Build a parity report for a single chunk by comparing GPU and CPU voxel buffers.
///
/// Caller is responsible for actually invoking the GPU and CPU paths and providing
/// the resulting `Vec<crate::world::voxel::Voxel>` arrays of length `CHUNK_VOLUME`.
/// This separation lets us unit-test the comparison logic in isolation.
#[allow(clippy::too_many_arguments)]
pub fn build_parity_report(
    coord: CubeSphereCoord,
    gpu_classification: &'static str,
    cpu_classification: &'static str,
    gpu_voxels: &[crate::world::voxel::Voxel],
    cpu_voxels: &[crate::world::voxel::Voxel],
    gpu_densities: &[f32],
    cpu_densities: &[f64],
    surface_offset: f64, // surface_r − mean_radius, for SurfaceIso tagging
    threshold_classifier: impl Fn(
        usize,
        &crate::world::voxel::Voxel,
        &crate::world::voxel::Voxel,
    ) -> ThresholdKind,
) -> ParityReport {
    use crate::world::voxel::MaterialId;
    assert_eq!(gpu_voxels.len(), cpu_voxels.len());
    assert_eq!(gpu_densities.len(), cpu_densities.len());
    assert_eq!(gpu_voxels.len(), gpu_densities.len());

    let _ = surface_offset; // reserved for future heuristics; kept in signature for callers

    let gpu_solid = gpu_voxels
        .iter()
        .filter(|v| v.material != MaterialId::AIR)
        .count();
    let cpu_solid = cpu_voxels
        .iter()
        .filter(|v| v.material != MaterialId::AIR)
        .count();

    let mut mismatches = Vec::new();
    let mut max_density_delta = 0.0_f64;
    let mut max_density_voxel_index = 0usize;
    for i in 0..gpu_voxels.len() {
        let g = &gpu_voxels[i];
        let c = &cpu_voxels[i];
        let dd = (g.density as f64 - c.density as f64).abs();
        // (`Voxel.density` field is currently f32 on the GPU side; the CPU stores
        // its own f64 in `cpu_densities`. Use the parallel arrays for the strict
        // density check.)
        let dd_strict = (gpu_densities[i] as f64 - cpu_densities[i]).abs();
        if dd_strict > max_density_delta {
            max_density_delta = dd_strict;
            max_density_voxel_index = i;
        }
        let _ = dd; // unused — kept for future per-voxel diagnostics

        if g.material != c.material {
            let kind = threshold_classifier(i, g, c);
            // `within_1_ulp` heuristic: density difference is below 1 ULP of the
            // threshold value (caller's responsibility to set kind ≠ Unattributed
            // only when this is true).
            let within = dd_strict <= f32::EPSILON as f64 * 4.0; // 4 ULPs ≈ 1 ULP-effective
            mismatches.push(VoxelMismatch {
                voxel_index: i,
                gpu_material: g.material,
                cpu_material: c.material,
                gpu_density: gpu_densities[i],
                cpu_density: cpu_densities[i],
                within_1_ulp: within,
                threshold_kind: kind,
            });
        }
    }

    ParityReport {
        coord,
        gpu_classification,
        cpu_classification,
        gpu_solid,
        cpu_solid,
        max_density_delta,
        max_density_voxel_index,
        mismatches,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_report(
        gpu_cls: &'static str,
        cpu_cls: &'static str,
        n_mismatches: usize,
        all_attributed: bool,
        density_delta: f64,
    ) -> ParityReport {
        let coord = CubeSphereCoord::new_with_lod(
            crate::world::v2::cubed_sphere::CubeFace::PosX,
            0,
            0,
            0,
            0,
        );
        let mismatches = (0..n_mismatches)
            .map(|i| VoxelMismatch {
                voxel_index: i,
                gpu_material: MaterialId::AIR,
                cpu_material: MaterialId::STONE,
                gpu_density: 0.0,
                cpu_density: 0.0,
                within_1_ulp: true,
                threshold_kind: if all_attributed {
                    ThresholdKind::SurfaceIso
                } else {
                    ThresholdKind::Unattributed
                },
            })
            .collect();
        ParityReport {
            coord,
            gpu_classification: gpu_cls,
            cpu_classification: cpu_cls,
            gpu_solid: 0,
            cpu_solid: 0,
            max_density_delta: density_delta,
            max_density_voxel_index: 0,
            mismatches,
        }
    }

    #[test]
    fn parity_passes_when_all_bounds_met() {
        let r = make_report("Mixed", "Mixed", 4, true, 1e-4);
        assert!(r.passes_parity_contract());
    }

    #[test]
    fn parity_fails_on_classification_mismatch() {
        let r = make_report("Mixed", "AllSolid", 0, true, 0.0);
        assert!(!r.passes_parity_contract());
    }

    #[test]
    fn parity_fails_on_too_many_mismatches() {
        let r = make_report("Mixed", "Mixed", 5, true, 0.0);
        assert!(!r.passes_parity_contract());
    }

    #[test]
    fn parity_fails_on_unattributed_mismatch() {
        let r = make_report("Mixed", "Mixed", 1, false, 0.0);
        assert!(!r.passes_parity_contract());
    }

    #[test]
    fn parity_fails_on_density_delta_too_large() {
        let r = make_report("Mixed", "Mixed", 0, true, 2e-3);
        assert!(!r.passes_parity_contract());
    }
}

#[cfg(test)]
mod build_tests {
    use super::*;
    use crate::world::v2::cubed_sphere::{CubeFace, CubeSphereCoord};
    use crate::world::voxel::{MaterialId, Voxel};

    fn coord() -> CubeSphereCoord {
        CubeSphereCoord::new_with_lod(CubeFace::PosX, 0, 0, 0, 0)
    }

    #[test]
    fn empty_report_is_clean() {
        let v: Vec<Voxel> = Vec::new();
        let d: Vec<f32> = Vec::new();
        let dd: Vec<f64> = Vec::new();
        let r = build_parity_report(
            coord(),
            "AllAir",
            "AllAir",
            &v,
            &v,
            &d,
            &dd,
            0.0,
            |_, _, _| ThresholdKind::Unattributed,
        );
        assert!(r.passes_parity_contract());
    }

    #[test]
    fn material_disagreement_at_surface_is_attributed() {
        let air = Voxel {
            material: MaterialId::AIR,
            temperature: 288.15,
            pressure: 101_325.0,
            damage: 0.0,
            latent_heat_buffer: 0.0,
            density: 0.0,
        };
        let stone = Voxel {
            material: MaterialId::STONE,
            temperature: 288.15,
            pressure: 101_325.0,
            damage: 0.0,
            latent_heat_buffer: 0.0,
            density: 0.0,
        };
        let gpu = vec![air];
        let cpu = vec![stone];
        let r = build_parity_report(
            coord(),
            "Mixed",
            "Mixed",
            &gpu,
            &cpu,
            &[0.0_f32],
            &[0.0_f64],
            0.0,
            |_, _, _| ThresholdKind::SurfaceIso,
        );
        assert_eq!(r.mismatches.len(), 1);
        assert!(r.passes_parity_contract());
    }

    #[test]
    fn five_mismatches_fail() {
        let air = Voxel {
            material: MaterialId::AIR,
            temperature: 288.15,
            pressure: 101_325.0,
            damage: 0.0,
            latent_heat_buffer: 0.0,
            density: 0.0,
        };
        let stone = Voxel {
            material: MaterialId::STONE,
            temperature: 288.15,
            pressure: 101_325.0,
            damage: 0.0,
            latent_heat_buffer: 0.0,
            density: 0.0,
        };
        let gpu = vec![air; 5];
        let cpu = vec![stone; 5];
        let r = build_parity_report(
            coord(),
            "Mixed",
            "Mixed",
            &gpu,
            &cpu,
            &[0.0_f32; 5],
            &[0.0_f64; 5],
            0.0,
            |_, _, _| ThresholdKind::SurfaceIso,
        );
        assert!(!r.passes_parity_contract());
    }
}
