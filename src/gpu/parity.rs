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

impl ParityReport {
    /// Returns `true` iff the report satisfies the §3 parity contract.
    pub fn passes_parity_contract(&self) -> bool {
        self.gpu_classification == self.cpu_classification
            && self.mismatches.len() <= 4
            && self
                .mismatches
                .iter()
                .all(|m| m.within_1_ulp && m.threshold_kind != ThresholdKind::Unattributed)
            && self.max_density_delta <= 1e-3
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
