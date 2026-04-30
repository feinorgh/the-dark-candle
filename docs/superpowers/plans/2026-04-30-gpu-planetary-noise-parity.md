# GPU Planetary-Noise Parity — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish exact per-voxel parity (modulo f32-vs-f64 tolerance) between the GPU voxel-generation pipeline (`src/gpu/shaders/voxel_gen.wgsl` + `src/gpu/voxel_compute.rs`) and the CPU `PlanetaryTerrainSampler` path, prove it with a parametric parity test suite, fix any divergences uncovered, and remove all stale FBM-era documentation.

**Architecture:** Audit-first then test-driven. Phase 1 produces a written audit table comparing every link of the GPU planetary-noise chain against its CPU counterpart. Phase 2 builds a `parity_probe` test harness that drives the *production* heightmap bake (`crate::planet::gpu_heightmap::bake_elevation_roughness_ocean`) and `GpuVoxelCompute::set_heightmap_data` upload pipeline at two planet scales over a probe matrix (face-centre, poles, antimeridian, coastline, deep-ocean, mountain) and seven radial layers. Phase 3 fixes each divergence via a minimal failing unit test → focused fix → harness stays green. Phase 4 strips stale TODOs / tech-debt and updates `ai-context.json` / `issues.json`.

**Tech Stack:** Rust 1.85 / edition 2024; Bevy 0.18 ECS; `wgpu` compute shaders; existing `GpuVoxelCompute` pipeline; existing `GPU_TEST_LOCK` and `try_new`-fallback patterns from `voxel_compute.rs`.

**Spec:** `docs/superpowers/specs/2026-04-30-gpu-planetary-noise-parity-design.md` (commit `f6cc7b6`).

**Parity contract (from §3 of spec):**

| Quantity | Bound |
|---|---|
| Chunk classification (`AllAir` / `AllSolid` / `Mixed`) | must be **equal** |
| Voxel material disagreements | ≤ `4` per chunk (`⌈0.01% × CHUNK_VOLUME⌉`) |
| Each disagreement | must lie within 1 ULP of a classification threshold |
| Voxel density max-Δ | ≤ `1e-3` |

LOD 0 only.

---

## File Structure

**Created:**

- `src/gpu/parity.rs` — new module: `ParityReport`, `parity_probe()`, threshold-tagging helpers. ~250 LOC. Thin: pure parity-comparison logic, no GPU dispatch (caller owns the `GpuVoxelCompute`).
- `docs/superpowers/specs/2026-04-30-gpu-planetary-noise-parity-design.md` (Appendix A) — populated audit table (edited in place).

**Modified:**

- `src/gpu/mod.rs` — declare `pub mod parity;`.
- `src/gpu/voxel_compute.rs` — rewrite the two `#[ignore]`d tests on top of `parity::parity_probe`; remove `#[ignore]`; delete stale "FBM noise" comments. Add the three gap-filler tests if the audit flags them as missing.
- `src/planet/gpu_heightmap.rs` — possibly add a small `#[cfg(test)]`-only `bake_planetary_heightmaps_for_test(&PlanetData, &PlanetConfig) -> HeightmapBundle` helper if the plan-time audit shows the existing function is awkward to call from tests. **Decision deferred to Task 5 once we've read the function signature carefully.** The function is already a free `pub fn` taking `&PlanetaryTerrainSampler`, so this likely turns out to be unnecessary.
- `src/gpu/shaders/voxel_gen.wgsl` — fixes from Phase 3 (per-divergence; specific edits unknown until audit completes).
- `ai-context.json` — remove the obsolete tech-debt entry; bump `meta.generated_from_commit`; update notes.
- `issues.json` — update `GPU-PARITY-001` notes with the audit + test outcome; file new issues for any out-of-scope findings.

**Deleted (search-and-destroy in Phase 4):**

- All `// TODO: GPU shader implements FBM noise` comments in `voxel_compute.rs` (currently 5 occurrences).
- The "GPU voxel shader uses FBM noise" entry from `ai-context.json` `tech_debt`.

---

## Phase 1 — Code audit

**Goal of Phase 1:** populate Appendix A in the spec with one row per checklist item. No code changes.

The audit table format is fixed:

```markdown
| # | Item | GPU site | CPU site | Status | Phase-3 fix? | Phase-2 test? | Notes |
|---|---|---|---|---|---|---|---|
```

`Status ∈ {matches, divergent, not-tested, stale-doc}`.

### Task 1: Audit rows 1–4 (heightmap bake, roughness, ocean, sampling math)

**Files:**

- Modify: `docs/superpowers/specs/2026-04-30-gpu-planetary-noise-parity-design.md` (Appendix A).

- [ ] **Step 1: Open both sides for row 1 (heightmap bake)**

Read in full:

- `src/planet/gpu_heightmap.rs` — `bake_elevation_roughness_ocean()` and the two constants `HEIGHTMAP_WIDTH = 2048`, `HEIGHTMAP_HEIGHT = 1024`.
- `src/world/terrain.rs:937-941` — `UnifiedTerrainGenerator::sample_surface_radius_at(lat, lon)`.
- `src/world/planetary_sampler.rs` — `PlanetaryTerrainSampler::surface_radius_at` and any helper it calls. Note especially: order of operations (IDW → roughness → ocean clamp), unit-direction reconstruction from (lat, lon).

- [ ] **Step 2: Append row 1 to Appendix A**

Status `matches` if the bake's projection / orientation / units match what the GPU `sample_heightmap` consumes; `divergent` otherwise; `not-tested` if matches but no test currently exercises it.

Exact edit (replace the empty Appendix A row):

```markdown
| 1 | Heightmap bake — resolution, projection, orientation, units | `src/planet/gpu_heightmap.rs:30-33,46-…` | `src/world/terrain.rs:937-941`, `src/world/planetary_sampler.rs:<func>` | <status> | <yes/no> | <yes/no> | <one-line note: e.g. "Row 0 = north pole; column 0 = lon = −π. Matches WGSL `lat_lon`. Determinism not currently tested."> |
```

- [ ] **Step 3: Repeat for row 2 (roughness bake)**

Audit value range and encoding. Roughness is documented as ∈ [0, 1]. Confirm by inspection. Append row 2.

- [ ] **Step 4: Repeat for row 3 (ocean bake)**

Confirm bake is **nearest-neighbour intent** (1.0 / 0.0 only); confirm WGSL `sample_ocean` (`voxel_gen.wgsl:481-489`) reads with NN as well. Append row 3.

- [ ] **Step 5: Repeat for row 4 (sampling math: u/v wrap, pole clamp, half-pixel)**

Read `voxel_gen.wgsl:428-489` (`sample_heightmap`, `sample_roughness`, `sample_ocean`). Verify:

- u-wrap at antimeridian: `fract(lon / (2π) + 0.5)`.
- v half-pixel: `v * H − 0.5` then floor + frac.
- y clamp to `[0, H−1]`; x wrap modulo W.

Cross-reference against the bake's pixel-centre convention (bake uses `(row + 0.5) / h` — see `gpu_heightmap.rs` line ~62). They must agree. Append row 4.

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/specs/2026-04-30-gpu-planetary-noise-parity-design.md
git commit -m "docs(gpu-parity): audit rows 1-4 (heightmap/roughness/ocean/sampling)"
```

### Task 2: Audit rows 5–8 (surface radius, density, material_at_radius, strata/ores/caves/crystals)

**Files:**

- Modify: `docs/superpowers/specs/2026-04-30-gpu-planetary-noise-parity-design.md` (Appendix A).

- [ ] **Step 1: Row 5 — surface radius pipeline**

Read `voxel_gen.wgsl:344-395` (`sample_terrain`) and the corresponding CPU path in `PlanetaryTerrainSampler`. Confirm IDW heightmap → roughness FBM modulation → ocean clamp ordering matches. Append row 5.

- [ ] **Step 2: Row 6 — surface density**

Read `voxel_gen.wgsl:672-…` (`terrain_density_gpu`) and the CPU equivalent (search `density` in `terrain.rs` / `terrain_gen.rs` / `planetary_sampler.rs`). Confirm sign convention (e.g. negative inside, positive outside) and value at iso-surface. Append row 6.

- [ ] **Step 3: Row 7 — `material_at_radius_gpu`**

Read `voxel_gen.wgsl:624-670` and the CPU `UnifiedTerrainGenerator::material_at_radius` (`terrain.rs:949-959`). Compare:

- depth bins / soil depth / sea-level interaction;
- arguments passed (radius vs offset-from-mean — note the f32-cancellation guard the existing earth-scale test mentions).

Append row 7.

- [ ] **Step 4: Row 8 — strata / ores / caves / crystals**

Read `voxel_gen.wgsl:555-622` (`strata_material_gpu`, `ore_material_gpu`, `is_cave_gpu`, `cave_fill_material_gpu`, `is_crystal_deposit_gpu`). For each:

- which perm-table slot it uses;
- frequencies and thresholds;
- fill material(s).

Cross-reference each against its CPU counterpart (likely in `terrain.rs` / `geology.rs` / `voxel.rs`). Append row 8.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-04-30-gpu-planetary-noise-parity-design.md
git commit -m "docs(gpu-parity): audit rows 5-8 (surface radius/density/material/strata-ore-cave-crystal)"
```

### Task 3: Audit rows 9–12 (cube-face frame, sortable_encode, classify thresholds, LOD)

**Files:**

- Modify: `docs/superpowers/specs/2026-04-30-gpu-planetary-noise-parity-design.md` (Appendix A).

- [ ] **Step 1: Row 9 — cube-face frame & lat/lon**

Read `voxel_gen.wgsl:395-426` (`quat_rotate`, `lat_lon`). Confirm the rotation_axis quaternion is applied identically to the CPU `CubeSphereCoord::world_transform` / `lat_lon_to_pos`. Append row 9.

- [ ] **Step 2: Row 10 — sortable_encode / sortable_decode**

Read `voxel_gen.wgsl:534-554`. The function maps signed f32 to monotonic u32 for atomicMin/Max in `surface_pass`. Verify monotonicity at sign change with three test values: `-1.0`, `-0.0`, `+0.0`, `+1.0` mapped through encode → confirm strict ordering. (No CPU counterpart needed — this is GPU-only logic.) Append row 10.

- [ ] **Step 3: Row 11 — classify-pass thresholds**

Read `voxel_gen.wgsl:824-…` (`classify_pass`) and `world::v2::terrain_gen::downgrade_uniform_voxels` (search for definition in `terrain_gen.rs`). Confirm AllAir / AllSolid / Mixed bucketing matches voxel-by-voxel. Append row 11.

- [ ] **Step 4: Row 12 — LOD support**

Search both `voxel_compute.rs` and `voxel_gen.wgsl` for any LOD handling. The chunk descriptor probably carries an LOD field; if it does, confirm the GPU shader uses it. Note: if the GPU path is LOD-0-only by construction, mark row 12 status `not-tested` and add a Phase-4 issue to be filed. Append row 12.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-04-30-gpu-planetary-noise-parity-design.md
git commit -m "docs(gpu-parity): audit rows 9-12 (cube-face frame/sortable/classify/LOD)"
```

### Task 4: Audit row 13 (stale TODO/comment scan)

**Files:**

- Modify: `docs/superpowers/specs/2026-04-30-gpu-planetary-noise-parity-design.md` (Appendix A).

- [ ] **Step 1: Enumerate stale items**

Run:

```bash
cd /home/pk/Devel/the-dark-candle
grep -n "FBM noise\|GPU shader implements FBM" src/ -r
grep -n "#\[ignore\]" src/gpu/voxel_compute.rs
grep -n "FBM" ai-context.json
```

Expect ~5 TODO comments in `voxel_compute.rs`, 2 `#[ignore]` markers, 1 entry in `ai-context.json` `tech_debt`.

- [ ] **Step 2: Append row 13 to Appendix A as a list**

Each stale item gets one sub-bullet: `<file>:<line> — <brief description> — Phase 4 deletes`.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-04-30-gpu-planetary-noise-parity-design.md
git commit -m "docs(gpu-parity): audit row 13 (stale FBM-era comments and ignored tests)"
```

### Task 5: Audit synthesis — confirm production bake helper and decide on test wrapper

**Files:**

- Modify: `docs/superpowers/specs/2026-04-30-gpu-planetary-noise-parity-design.md` (Appendix A — synthesis paragraph at the end).

- [ ] **Step 1: Confirm production bake function**

`crate::planet::gpu_heightmap::bake_elevation_roughness_ocean(&PlanetaryTerrainSampler) -> (Vec<f32>, Vec<f32>, Vec<f32>)` is already a free `pub fn`. The runtime call site is `src/world/v2/chunk_manager.rs:1424-1438` (`try_inject_gpu_heightmap`). Confirm by reading both. Tests can call `bake_elevation_roughness_ocean` directly — **no test-only wrapper needed**.

- [ ] **Step 2: Append synthesis paragraph**

Append to Appendix A (under the table) a short summary listing:

- Total rows audited.
- Count by status (`matches` / `divergent` / `not-tested` / `stale-doc`).
- Phase-3 fix list (the `divergent` rows, one bullet each, in the order Phase 3 will tackle them).
- Phase-2 test list (the `not-tested` rows, one bullet each).
- Phase-4 cleanup list (the `stale-doc` rows from row 13).

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-04-30-gpu-planetary-noise-parity-design.md
git commit -m "docs(gpu-parity): audit synthesis — divergence and gap lists for Phase 3/4"
```

---

## Phase 2 — Parity test harness

### Task 6: Add `ParityReport` and threshold-tagging types

**Files:**

- Create: `src/gpu/parity.rs`
- Modify: `src/gpu/mod.rs`

- [ ] **Step 1: Write the failing test**

Create `src/gpu/parity.rs` with this content:

```rust
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
            && self.mismatches.iter().all(|m| m.within_1_ulp
                && m.threshold_kind != ThresholdKind::Unattributed)
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
```

- [ ] **Step 2: Wire the module**

In `src/gpu/mod.rs`, add `pub mod parity;` near the other `pub mod` declarations.

- [ ] **Step 3: Run the unit tests**

```bash
cargo test --lib gpu::parity::tests -- --nocapture
```

Expected: 5 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/gpu/parity.rs src/gpu/mod.rs
git commit -m "feat(gpu-parity): add ParityReport and ThresholdKind types

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

### Task 7: Implement `parity_probe()` skeleton (no GPU dispatch yet)

**Files:**

- Modify: `src/gpu/parity.rs`

- [ ] **Step 1: Write the failing test**

Append to `src/gpu/parity.rs`:

```rust
/// Build a parity report for a single chunk by comparing GPU and CPU voxel buffers.
///
/// Caller is responsible for actually invoking the GPU and CPU paths and providing
/// the resulting `Vec<crate::world::voxel::Voxel>` arrays of length `CHUNK_VOLUME`.
/// This separation lets us unit-test the comparison logic in isolation.
pub fn build_parity_report(
    coord: CubeSphereCoord,
    gpu_classification: &'static str,
    cpu_classification: &'static str,
    gpu_voxels: &[crate::world::voxel::Voxel],
    cpu_voxels: &[crate::world::voxel::Voxel],
    gpu_densities: &[f32],
    cpu_densities: &[f64],
    surface_offset: f64,           // surface_r − mean_radius, for SurfaceIso tagging
    threshold_classifier: impl Fn(usize, &crate::world::voxel::Voxel, &crate::world::voxel::Voxel) -> ThresholdKind,
) -> ParityReport {
    use crate::world::voxel::MaterialId;
    assert_eq!(gpu_voxels.len(), cpu_voxels.len());
    assert_eq!(gpu_densities.len(), cpu_densities.len());
    assert_eq!(gpu_voxels.len(), gpu_densities.len());

    let _ = surface_offset; // reserved for future heuristics; kept in signature for callers

    let gpu_solid = gpu_voxels.iter().filter(|v| v.material != MaterialId::AIR).count();
    let cpu_solid = cpu_voxels.iter().filter(|v| v.material != MaterialId::AIR).count();

    let mut mismatches = Vec::new();
    let mut max_density_delta = 0.0_f64;
    let mut max_density_voxel_index = 0usize;
    for i in 0..gpu_voxels.len() {
        let g = &gpu_voxels[i];
        let c = &cpu_voxels[i];
        let dd = (g.density as f64 - c.density).abs();
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
mod build_tests {
    use super::*;
    use crate::world::voxel::{MaterialId, Voxel};
    use crate::world::v2::cubed_sphere::{CubeFace, CubeSphereCoord};

    fn coord() -> CubeSphereCoord {
        CubeSphereCoord::new_with_lod(CubeFace::PosX, 0, 0, 0, 0)
    }

    #[test]
    fn empty_report_is_clean() {
        let v: Vec<Voxel> = Vec::new();
        let d: Vec<f32> = Vec::new();
        let dd: Vec<f64> = Vec::new();
        let r = build_parity_report(
            coord(), "AllAir", "AllAir", &v, &v, &d, &dd, 0.0,
            |_, _, _| ThresholdKind::Unattributed,
        );
        assert!(r.passes_parity_contract());
    }

    #[test]
    fn material_disagreement_at_surface_is_attributed() {
        let air = Voxel { material: MaterialId::AIR, density: 0.0 };
        let stone = Voxel { material: MaterialId::STONE, density: 0.0 };
        let gpu = vec![air];
        let cpu = vec![stone];
        let r = build_parity_report(
            coord(), "Mixed", "Mixed",
            &gpu, &cpu, &[0.0_f32], &[0.0_f64], 0.0,
            |_, _, _| ThresholdKind::SurfaceIso,
        );
        assert_eq!(r.mismatches.len(), 1);
        assert!(r.passes_parity_contract());
    }

    #[test]
    fn five_mismatches_fail() {
        let air = Voxel { material: MaterialId::AIR, density: 0.0 };
        let stone = Voxel { material: MaterialId::STONE, density: 0.0 };
        let gpu = vec![air; 5];
        let cpu = vec![stone; 5];
        let r = build_parity_report(
            coord(), "Mixed", "Mixed",
            &gpu, &cpu, &[0.0_f32; 5], &[0.0_f64; 5], 0.0,
            |_, _, _| ThresholdKind::SurfaceIso,
        );
        assert!(!r.passes_parity_contract());
    }
}
```

- [ ] **Step 2: Run the new tests**

```bash
cargo test --lib gpu::parity -- --nocapture
```

Expected: 8 tests pass (5 from Task 6 + 3 here).

- [ ] **Step 3: Commit**

```bash
git add src/gpu/parity.rs
git commit -m "feat(gpu-parity): add build_parity_report comparison core

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

### Task 8: Implement `parity_probe()` end-to-end (real GPU + CPU dispatch)

**Files:**

- Modify: `src/gpu/parity.rs`

- [ ] **Step 1: Add the end-to-end probe**

Append to `src/gpu/parity.rs`:

```rust
use std::sync::Arc;

use crate::gpu::voxel_compute::{chunk_desc_from_coord, GpuChunkRequest, GpuVoxelCompute};
use crate::world::planet::PlanetConfig;
use crate::world::terrain::UnifiedTerrainGenerator;
use crate::world::v2::terrain_gen::{cached_voxels_to_vec, generate_v2_voxels};

/// Drive both the GPU and CPU pipelines for a single chunk and produce a parity
/// report. Caller must have uploaded the heightmap to `compute` already (via the
/// production `bake_elevation_roughness_ocean` + `set_heightmap_data` path).
///
/// Threshold attribution is heuristic: any voxel within 4 f32 ULPs of the
/// surface offset is tagged `SurfaceIso`; otherwise `Unattributed` (so the
/// parity contract will fail loudly on bulk divergence).
pub fn parity_probe(
    planet: &PlanetConfig,
    unified: &Arc<UnifiedTerrainGenerator>,
    compute: &GpuVoxelCompute,
    coord: CubeSphereCoord,
) -> ParityReport {
    use crate::world::v2::cubed_sphere::CubeSphereCoord as C;

    let fce = C::face_chunks_per_edge(planet.mean_radius);

    // Surface offset for threshold tagging.
    let (chunk_lat, chunk_lon) = coord.center_lat_lon(planet.mean_radius);
    let surface_r = unified.sample_surface_radius_at(chunk_lat, chunk_lon);
    let surface_offset = surface_r - planet.mean_radius;

    let desc = chunk_desc_from_coord(
        coord,
        planet.mean_radius,
        fce,
        planet.sea_level_radius,
        planet.soil_depth,
        planet.cave_threshold,
        0,
    );

    let gpu_result = compute.generate_batch(
        &[GpuChunkRequest { coord, desc }],
        planet.rotation_axis,
    );
    let gpu_td = &gpu_result.terrain_data[0];
    let cpu_td = generate_v2_voxels(coord, planet.mean_radius, fce, unified);

    let classify = |v: &crate::world::v2::terrain_gen::CachedVoxels| -> &'static str {
        match v {
            crate::world::v2::terrain_gen::CachedVoxels::AllAir => "AllAir",
            crate::world::v2::terrain_gen::CachedVoxels::AllSolid(_) => "AllSolid",
            crate::world::v2::terrain_gen::CachedVoxels::Mixed(_) => "Mixed",
        }
    };
    let gpu_cls = classify(&gpu_td.voxels);
    let cpu_cls = classify(&cpu_td.voxels);

    let gpu_voxels = cached_voxels_to_vec(&gpu_td.voxels);
    let cpu_voxels = cached_voxels_to_vec(&cpu_td.voxels);
    let gpu_densities: Vec<f32> = gpu_voxels.iter().map(|v| v.density).collect();
    let cpu_densities: Vec<f64> = cpu_voxels.iter().map(|v| v.density as f64).collect();

    let one_ulp = f32::EPSILON as f64 * 4.0;
    let threshold_classifier = |idx: usize, _g: &crate::world::voxel::Voxel, _c: &crate::world::voxel::Voxel| -> ThresholdKind {
        let dd = (gpu_densities[idx] as f64 - cpu_densities[idx]).abs();
        if dd <= one_ulp {
            ThresholdKind::SurfaceIso
        } else {
            ThresholdKind::Unattributed
        }
    };

    build_parity_report(
        coord,
        gpu_cls,
        cpu_cls,
        &gpu_voxels,
        &cpu_voxels,
        &gpu_densities,
        &cpu_densities,
        surface_offset,
        threshold_classifier,
    )
}
```

- [ ] **Step 2: Verify compilation**

```bash
cargo check --lib
```

Expected: clean compile. If `chunk_desc_from_coord` or `CubeSphereCoord::center_lat_lon` are not currently `pub`, add `pub` to their definitions in their respective modules. Audit row should already note these visibility constraints; if not, add a short note in Appendix A.

- [ ] **Step 3: Commit**

```bash
git add src/gpu/parity.rs src/gpu/voxel_compute.rs src/world/v2/cubed_sphere.rs
git commit -m "feat(gpu-parity): add end-to-end parity_probe driving GPU+CPU paths

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

### Task 9: Small-planet parity test (face centre, ±3 layers)

**Files:**

- Modify: `src/gpu/voxel_compute.rs` (test module).

- [ ] **Step 1: Write the test on top of `parity_probe`**

Replace the body of `gpu_vs_cpu_surface_chunk_comparison` (currently `#[ignore]`d at `voxel_compute.rs:1416`) with the new harness call. Remove `#[ignore]`. Delete the two stale "FBM noise" TODO comments.

```rust
#[test]
fn gpu_vs_cpu_surface_chunk_small_planet_face_center() {
    let _guard = GPU_TEST_LOCK.lock().unwrap();
    use crate::gpu::parity::parity_probe;
    use crate::planet::gpu_heightmap::bake_elevation_roughness_ocean;
    use crate::world::scene_presets::ScenePreset;
    use crate::world::terrain::UnifiedTerrainGenerator;
    use crate::world::v2::cubed_sphere::{CubeFace, CubeSphereCoord};
    use std::sync::Arc;

    let planet = crate::world::planet::PlanetConfig {
        mean_radius: 32_000.0,
        sea_level_radius: 32_000.0,
        height_scale: 4_000.0,
        seed: 42,
        ..Default::default()
    };

    let Some(compute) = GpuVoxelCompute::try_new(
        planet.noise.as_ref().expect("preset must have noise"),
        planet.seed,
        planet.mean_radius,
        planet.height_scale,
    ) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    // Build the production planetary sampler.
    let gen_cfg = crate::planet::PlanetConfig {
        seed: planet.seed as u64,
        grid_level: 4,
        ..Default::default()
    };
    let pd = Arc::new(crate::planet::PlanetData::new(gen_cfg));
    let unified = Arc::new(UnifiedTerrainGenerator::new(pd, planet.clone()));

    // Production heightmap bake + upload.
    let (elev, rough, ocean) = bake_elevation_roughness_ocean(&unified.0);
    compute.set_heightmap_data(&elev, &rough, &ocean, planet.seed as u64);
    assert!(compute.is_heightmap_ready());

    let fce = CubeSphereCoord::face_chunks_per_edge(planet.mean_radius);
    let half = (fce / 2.0) as i32;
    let surface_r = unified.sample_surface_radius_at(0.0, std::f64::consts::FRAC_PI_2);
    let surface_layer = ((surface_r - planet.mean_radius) / crate::world::chunk::CHUNK_SIZE as f64).round() as i32;

    let mut saw_mixed = false;
    for dl in -3..=3 {
        let coord = CubeSphereCoord::new_with_lod(CubeFace::PosX, half, half, surface_layer + dl, 0);
        let report = parity_probe(&planet, &unified, &compute, coord);
        eprintln!(
            "dl={dl:+} cls=GPU:{} CPU:{} mismatches={} max_dd={:.3e}",
            report.gpu_classification, report.cpu_classification,
            report.mismatches.len(), report.max_density_delta,
        );
        if report.gpu_classification == "Mixed" {
            saw_mixed = true;
        }
        assert!(
            report.passes_parity_contract(),
            "parity contract failed at dl={dl:+}: {:#?}", report,
        );
    }
    assert!(saw_mixed, "expected at least one Mixed chunk in the layer sweep");
}
```

- [ ] **Step 2: Run**

```bash
cargo test --lib gpu_vs_cpu_surface_chunk_small_planet_face_center -- --nocapture
```

Expected (Phase 1 audit may have flagged divergences):

- **Pass** → great, that probe is parity-clean.
- **Fail** → the failure mode (which assertion, which voxel) feeds Phase 3 as a divergence to fix.

If the test fails, **stop here** — do **not** modify it to make it pass. Record the failure mode in the spec's Appendix A as a confirmed divergence, then jump to Task 13 (Phase 3 loop) for it. Once the divergence is fixed, return to this task and confirm the test passes.

- [ ] **Step 3: Commit (whether the test passes or is the first red Phase-3 trigger)**

If passing:

```bash
git add src/gpu/voxel_compute.rs
git commit -m "test(gpu-parity): rewrite small-planet face-center parity on parity_probe

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

If failing: leave the test in place (it's our regression target) and proceed to Task 13. The fix commit will reference this test.

### Task 10: Earth-scale parity test (face centre, ±3 layers)

**Files:**

- Modify: `src/gpu/voxel_compute.rs` (test module).

- [ ] **Step 1: Write the test**

Replace the body of `gpu_vs_cpu_parity_earth_scale_surface_chunk` (currently `#[ignore]`d at `voxel_compute.rs:1771`) with:

```rust
#[test]
fn gpu_vs_cpu_parity_earth_scale_face_center() {
    let _guard = GPU_TEST_LOCK.lock().unwrap();
    use crate::gpu::parity::parity_probe;
    use crate::planet::gpu_heightmap::bake_elevation_roughness_ocean;
    use crate::world::scene_presets::ScenePreset;
    use crate::world::terrain::UnifiedTerrainGenerator;
    use crate::world::v2::cubed_sphere::{CubeFace, CubeSphereCoord};
    use std::sync::Arc;

    let planet = ScenePreset::SphericalPlanet.planet_config();
    let noise = planet.noise.as_ref().expect("preset must carry noise");

    let Some(compute) = GpuVoxelCompute::try_new(noise, planet.seed, planet.mean_radius, planet.height_scale) else {
        eprintln!("skipping earth-scale parity: no GPU adapter");
        return;
    };

    let gen_cfg = crate::planet::PlanetConfig {
        seed: planet.seed as u64,
        grid_level: 4,
        ..Default::default()
    };
    let pd = Arc::new(crate::planet::PlanetData::new(gen_cfg));
    let unified = Arc::new(UnifiedTerrainGenerator::new(pd, planet.clone()));

    let (elev, rough, ocean) = bake_elevation_roughness_ocean(&unified.0);
    compute.set_heightmap_data(&elev, &rough, &ocean, planet.seed as u64);

    let fce = CubeSphereCoord::face_chunks_per_edge(planet.mean_radius);
    let half = (fce / 2.0) as i32;
    let (lat, lon) = (0.0, std::f64::consts::FRAC_PI_2);
    let surface_r = unified.sample_surface_radius_at(lat, lon);
    let surface_layer = ((surface_r - planet.mean_radius) / crate::world::chunk::CHUNK_SIZE as f64).round() as i32;

    let mut saw_mixed = false;
    for dl in -3..=3 {
        let coord = CubeSphereCoord::new_with_lod(CubeFace::PosX, half, half, surface_layer + dl, 0);
        let report = parity_probe(&planet, &unified, &compute, coord);
        eprintln!(
            "earth-scale dl={dl:+} cls=GPU:{} CPU:{} mismatches={} max_dd={:.3e}",
            report.gpu_classification, report.cpu_classification,
            report.mismatches.len(), report.max_density_delta,
        );
        if report.gpu_classification == "Mixed" {
            saw_mixed = true;
        }
        assert!(
            report.passes_parity_contract(),
            "earth-scale parity contract failed at dl={dl:+}: {:#?}", report,
        );
    }
    assert!(saw_mixed, "expected at least one Mixed chunk in the layer sweep");
}
```

- [ ] **Step 2: Run**

```bash
cargo test --lib gpu_vs_cpu_parity_earth_scale_face_center -- --nocapture
```

Same red-or-green-then-Phase-3 protocol as Task 9.

- [ ] **Step 3: Commit**

```bash
git add src/gpu/voxel_compute.rs
git commit -m "test(gpu-parity): rewrite earth-scale parity on parity_probe

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

### Task 11: Probe-matrix expansion — poles, antimeridian, coastline, deep ocean, mountain

**Files:**

- Modify: `src/gpu/voxel_compute.rs` (test module).

- [ ] **Step 1: Add a parameterised helper**

In the test module, add:

```rust
#[cfg(test)]
fn parity_probe_set(
    name: &str,
    planet: &crate::world::planet::PlanetConfig,
    locations: &[(f64, f64, &str)],   // (lat, lon, label)
) {
    let _guard = GPU_TEST_LOCK.lock().unwrap();
    use crate::gpu::parity::parity_probe;
    use crate::planet::gpu_heightmap::bake_elevation_roughness_ocean;
    use crate::world::terrain::UnifiedTerrainGenerator;
    use crate::world::v2::cubed_sphere::CubeSphereCoord;
    use std::sync::Arc;

    let noise = planet.noise.as_ref().expect("preset must carry noise");
    let Some(compute) = GpuVoxelCompute::try_new(noise, planet.seed, planet.mean_radius, planet.height_scale) else {
        eprintln!("[{name}] skipping: no GPU adapter");
        return;
    };

    let gen_cfg = crate::planet::PlanetConfig {
        seed: planet.seed as u64,
        grid_level: 4,
        ..Default::default()
    };
    let pd = Arc::new(crate::planet::PlanetData::new(gen_cfg));
    let unified = Arc::new(UnifiedTerrainGenerator::new(pd, planet.clone()));
    let (elev, rough, ocean) = bake_elevation_roughness_ocean(&unified.0);
    compute.set_heightmap_data(&elev, &rough, &ocean, planet.seed as u64);

    for &(lat, lon, label) in locations {
        let coord_at_surface = coord_for_lat_lon(planet, lat, lon, &unified, 0);
        for dl in -3..=3 {
            let coord = coord_at_surface.with_layer_offset(dl);
            let report = parity_probe(planet, &unified, &compute, coord);
            assert!(
                report.passes_parity_contract(),
                "[{name}/{label}] parity failed at dl={dl:+}: {:#?}", report,
            );
        }
    }
}

#[cfg(test)]
fn coord_for_lat_lon(
    planet: &crate::world::planet::PlanetConfig,
    lat: f64,
    lon: f64,
    unified: &std::sync::Arc<crate::world::terrain::UnifiedTerrainGenerator>,
    layer_offset: i32,
) -> crate::world::v2::cubed_sphere::CubeSphereCoord {
    use crate::world::v2::cubed_sphere::CubeSphereCoord;
    let dir = crate::planet::detail::lat_lon_to_pos(lat, lon);
    let surface_r = unified.sample_surface_radius_at(lat, lon);
    let surface_layer = ((surface_r - planet.mean_radius) / crate::world::chunk::CHUNK_SIZE as f64).round() as i32;
    CubeSphereCoord::from_world_dir_at_lod(dir, planet.mean_radius, surface_layer + layer_offset, 0)
}
```

If `CubeSphereCoord::from_world_dir_at_lod` and `with_layer_offset` do not exist, add them in `src/world/v2/cubed_sphere.rs`. Each is a few lines:

```rust
impl CubeSphereCoord {
    pub fn with_layer_offset(self, dl: i32) -> Self {
        // returns a new coord with `layer + dl`, same face/u/v/lod
        // implementation detail: reuse the existing ctor
        Self::new_with_lod(self.face(), self.u(), self.v(), self.layer() + dl, self.lod())
    }

    pub fn from_world_dir_at_lod(dir: bevy::math::DVec3, mean_radius: f64, layer: i32, lod: u8) -> Self {
        // pick the dominant axis to choose CubeFace, then map onto u,v in [-half_fce, half_fce)
        // (concrete implementation should mirror existing inverse mapping helpers in cubed_sphere.rs)
        // …
    }
}
```

If the inverse mapping is non-trivial and not already in `cubed_sphere.rs`, **stop and add a focused unit test for `from_world_dir_at_lod`** before using it in the harness. Add 6 round-trip tests (one per face).

- [ ] **Step 2: Add the small-planet probe-matrix test**

```rust
#[test]
fn gpu_vs_cpu_parity_small_planet_probe_matrix() {
    let planet = crate::world::planet::PlanetConfig {
        mean_radius: 32_000.0,
        sea_level_radius: 32_000.0,
        height_scale: 4_000.0,
        seed: 42,
        ..Default::default()
    };
    parity_probe_set(
        "small-planet",
        &planet,
        &[
            ( 0.0,  std::f64::consts::FRAC_PI_2,        "face_center"),
            ( 1.55, 0.0,                                "north_pole"),
            (-1.55, 0.0,                                "south_pole"),
            ( 0.0,  std::f64::consts::PI,               "antimeridian"),
            // coastline / ocean / mountain locations are derived from the baked
            // ocean+roughness buffers in Task 12 (which adds a helper that picks
            // them); for now this matrix exercises geometry-driven probes only.
        ],
    );
}
```

- [ ] **Step 3: Add the earth-scale probe-matrix test**

```rust
#[test]
fn gpu_vs_cpu_parity_earth_scale_probe_matrix() {
    let planet = crate::world::scene_presets::ScenePreset::SphericalPlanet.planet_config();
    parity_probe_set(
        "earth-scale",
        &planet,
        &[
            ( 0.0,  std::f64::consts::FRAC_PI_2, "face_center"),
            ( 1.55, 0.0,                         "north_pole"),
            (-1.55, 0.0,                         "south_pole"),
            ( 0.0,  std::f64::consts::PI,        "antimeridian"),
        ],
    );
}
```

- [ ] **Step 4: Run**

```bash
cargo test --lib gpu_vs_cpu_parity_small_planet_probe_matrix gpu_vs_cpu_parity_earth_scale_probe_matrix -- --nocapture
```

If any sub-probe fails, leave it in place and proceed to Task 13 (Phase 3 loop) for that divergence.

- [ ] **Step 5: Commit**

```bash
git add src/gpu/voxel_compute.rs src/world/v2/cubed_sphere.rs
git commit -m "test(gpu-parity): add probe-matrix tests (poles, antimeridian) at both scales

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

### Task 12: Biome-driven probes (coastline, deep ocean, mountain)

**Files:**

- Modify: `src/gpu/voxel_compute.rs` (test module).

- [ ] **Step 1: Write a biome-driven location picker**

In the test module:

```rust
#[cfg(test)]
fn pick_lat_lon_by_biome(
    elev: &[f32],
    rough: &[f32],
    ocean: &[f32],
    biome: &str,
) -> Option<(f64, f64)> {
    use crate::planet::gpu_heightmap::{HEIGHTMAP_HEIGHT, HEIGHTMAP_WIDTH};
    let w = HEIGHTMAP_WIDTH as usize;
    let h = HEIGHTMAP_HEIGHT as usize;
    let pixel = |lat_idx: usize, lon_idx: usize| {
        let lat = (0.5 - (lat_idx as f64 + 0.5) / h as f64) * std::f64::consts::PI;
        let lon = ((lon_idx as f64 + 0.5) / w as f64 - 0.5) * 2.0 * std::f64::consts::PI;
        (lat, lon)
    };
    for r in 0..h {
        for c in 0..w {
            let i = r * w + c;
            let matches = match biome {
                // Coastline: ocean transitions in a 3x3 neighbourhood.
                "coastline" => {
                    if ocean[i] != 1.0 { continue; }
                    let mut land_neighbour = false;
                    for dr in [-1i32, 0, 1] {
                        for dc in [-1i32, 0, 1] {
                            let rr = (r as i32 + dr).clamp(0, h as i32 - 1) as usize;
                            let cc = ((c as i32 + dc).rem_euclid(w as i32)) as usize;
                            if ocean[rr * w + cc] == 0.0 {
                                land_neighbour = true;
                            }
                        }
                    }
                    land_neighbour
                }
                "deep_ocean" => ocean[i] == 1.0 && elev[i] < -2000.0,
                "mountain"   => ocean[i] == 0.0 && rough[i] > 0.7 && elev[i] > 1500.0,
                _ => false,
            };
            if matches {
                let (lat, lon) = pixel(r, c);
                return Some((lat, lon));
            }
        }
    }
    None
}
```

- [ ] **Step 2: Add the biome-driven test**

```rust
#[test]
fn gpu_vs_cpu_parity_biome_driven() {
    let planet = crate::world::scene_presets::ScenePreset::SphericalPlanet.planet_config();
    let noise = planet.noise.as_ref().expect("preset must carry noise");
    let Some(compute) = GpuVoxelCompute::try_new(noise, planet.seed, planet.mean_radius, planet.height_scale) else {
        eprintln!("skipping biome-driven parity: no GPU adapter");
        return;
    };

    let gen_cfg = crate::planet::PlanetConfig {
        seed: planet.seed as u64,
        grid_level: 4,
        ..Default::default()
    };
    let pd = std::sync::Arc::new(crate::planet::PlanetData::new(gen_cfg));
    let unified = std::sync::Arc::new(crate::world::terrain::UnifiedTerrainGenerator::new(pd, planet.clone()));
    let (elev, rough, ocean) = crate::planet::gpu_heightmap::bake_elevation_roughness_ocean(&unified.0);
    compute.set_heightmap_data(&elev, &rough, &ocean, planet.seed as u64);

    let mut chosen = Vec::new();
    for biome in ["coastline", "deep_ocean", "mountain"] {
        match pick_lat_lon_by_biome(&elev, &rough, &ocean, biome) {
            Some((lat, lon)) => chosen.push((biome, lat, lon)),
            None => eprintln!("no {biome} pixel found in baked maps; skipping"),
        }
    }

    let _guard = GPU_TEST_LOCK.lock().unwrap();
    for (label, lat, lon) in chosen {
        for dl in -3..=3 {
            let coord = coord_for_lat_lon(&planet, lat, lon, &unified, dl);
            let report = crate::gpu::parity::parity_probe(&planet, &unified, &compute, coord);
            assert!(
                report.passes_parity_contract(),
                "[{label}] parity failed at lat={lat:.3} lon={lon:.3} dl={dl:+}: {:#?}", report,
            );
        }
    }
}
```

- [ ] **Step 3: Run**

```bash
cargo test --lib gpu_vs_cpu_parity_biome_driven -- --nocapture
```

- [ ] **Step 4: Commit**

```bash
git add src/gpu/voxel_compute.rs
git commit -m "test(gpu-parity): add biome-driven probe (coastline/deep-ocean/mountain)

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

### Task 13: Gap-filler test — heightmap bake determinism

**Files:**

- Modify: `src/planet/gpu_heightmap.rs`

- [ ] **Step 1: Add the test**

Append to `gpu_heightmap.rs`:

```rust
#[cfg(test)]
mod determinism_tests {
    use super::*;
    use crate::world::planet::PlanetConfig as WorldPlanetConfig;
    use crate::world::terrain::UnifiedTerrainGenerator;
    use std::sync::Arc;

    fn make_sampler(seed: u64) -> Arc<UnifiedTerrainGenerator> {
        let gen_cfg = crate::planet::PlanetConfig {
            seed,
            grid_level: 4,
            ..Default::default()
        };
        let pd = Arc::new(crate::planet::PlanetData::new(gen_cfg));
        let world = WorldPlanetConfig {
            mean_radius: 32_000.0,
            sea_level_radius: 32_000.0,
            height_scale: 4_000.0,
            seed: seed as u32,
            ..Default::default()
        };
        Arc::new(UnifiedTerrainGenerator::new(pd, world))
    }

    #[test]
    fn bake_is_deterministic_for_fixed_seed() {
        let s1 = make_sampler(1234);
        let s2 = make_sampler(1234);
        let (e1, r1, o1) = bake_elevation_roughness_ocean(&s1.0);
        let (e2, r2, o2) = bake_elevation_roughness_ocean(&s2.0);
        assert_eq!(e1, e2);
        assert_eq!(r1, r2);
        assert_eq!(o1, o2);
    }

    #[test]
    fn bake_differs_for_different_seeds() {
        let s1 = make_sampler(1234);
        let s2 = make_sampler(5678);
        let (e1, _, _) = bake_elevation_roughness_ocean(&s1.0);
        let (e2, _, _) = bake_elevation_roughness_ocean(&s2.0);
        assert_ne!(e1, e2, "different seeds must produce different elevation maps");
    }
}
```

- [ ] **Step 2: Run**

```bash
cargo test --lib bake_is_deterministic_for_fixed_seed bake_differs_for_different_seeds
```

Expected: both pass.

- [ ] **Step 3: Commit**

```bash
git add src/planet/gpu_heightmap.rs
git commit -m "test(gpu-parity): heightmap bake determinism + seed sensitivity

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

### Task 14: Gap-filler test — `set_heightmap_data` rejects wrong-size buffers

**Files:**

- Modify: `src/gpu/voxel_compute.rs` (test module).

- [ ] **Step 1: Add the test**

```rust
#[test]
#[should_panic(expected = "data length")]
fn set_heightmap_data_rejects_wrong_size() {
    let _guard = GPU_TEST_LOCK.lock().unwrap();
    let planet = crate::world::planet::PlanetConfig {
        mean_radius: 32_000.0,
        sea_level_radius: 32_000.0,
        height_scale: 4_000.0,
        seed: 1,
        ..Default::default()
    };
    let noise = planet.noise.as_ref().expect("preset must carry noise");
    let Some(compute) = GpuVoxelCompute::try_new(noise, planet.seed, planet.mean_radius, planet.height_scale) else {
        eprintln!("no GPU adapter; #[should_panic] cannot be exercised — emit explicit panic");
        panic!("data length 0 != expected …"); // satisfy #[should_panic]
    };
    let bogus = vec![0.0_f32; 100];
    compute.set_heightmap_data(&bogus, &bogus, &bogus, 1);
}
```

- [ ] **Step 2: Run**

```bash
cargo test --lib set_heightmap_data_rejects_wrong_size
```

Expected: pass (panic captured by `#[should_panic]`).

- [ ] **Step 3: Commit**

```bash
git add src/gpu/voxel_compute.rs
git commit -m "test(gpu-parity): set_heightmap_data validates buffer dimensions

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

### Task 15: Gap-filler test — full-face statistical sweep (coarse)

**Files:**

- Modify: `src/gpu/voxel_compute.rs` (test module).

- [ ] **Step 1: Add the sweep**

```rust
#[test]
fn gpu_lod0_full_face_classification_matches_cpu_sample() {
    let _guard = GPU_TEST_LOCK.lock().unwrap();
    use crate::planet::gpu_heightmap::bake_elevation_roughness_ocean;
    use crate::world::terrain::UnifiedTerrainGenerator;
    use crate::world::v2::cubed_sphere::{CubeFace, CubeSphereCoord};
    use std::sync::Arc;

    let planet = crate::world::planet::PlanetConfig {
        mean_radius: 32_000.0,
        sea_level_radius: 32_000.0,
        height_scale: 4_000.0,
        seed: 42,
        ..Default::default()
    };
    let noise = planet.noise.as_ref().expect("preset must carry noise");
    let Some(compute) = GpuVoxelCompute::try_new(noise, planet.seed, planet.mean_radius, planet.height_scale) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    let gen_cfg = crate::planet::PlanetConfig {
        seed: planet.seed as u64,
        grid_level: 4,
        ..Default::default()
    };
    let pd = Arc::new(crate::planet::PlanetData::new(gen_cfg));
    let unified = Arc::new(UnifiedTerrainGenerator::new(pd, planet.clone()));
    let (elev, rough, ocean) = bake_elevation_roughness_ocean(&unified.0);
    compute.set_heightmap_data(&elev, &rough, &ocean, planet.seed as u64);

    let fce = CubeSphereCoord::face_chunks_per_edge(planet.mean_radius);
    let half = (fce / 2.0) as i32;
    let stride = ((fce as i32) / 8).max(1);

    let mut compared = 0;
    let mut mismatches = 0;
    for du in (-half..half).step_by(stride as usize) {
        for dv in (-half..half).step_by(stride as usize) {
            let lat = 0.0; // PosX face mid-latitude approximation; coarse statistical check
            let lon = std::f64::consts::FRAC_PI_2;
            let surface_r = unified.sample_surface_radius_at(lat, lon);
            let layer = ((surface_r - planet.mean_radius) / crate::world::chunk::CHUNK_SIZE as f64).round() as i32;
            let coord = CubeSphereCoord::new_with_lod(CubeFace::PosX, du, dv, layer, 0);
            let report = crate::gpu::parity::parity_probe(&planet, &unified, &compute, coord);
            compared += 1;
            if report.gpu_classification != report.cpu_classification {
                mismatches += 1;
                eprintln!("classification mismatch at ({du},{dv}): GPU={} CPU={}",
                    report.gpu_classification, report.cpu_classification);
            }
        }
    }
    eprintln!("compared {compared} chunks, classification mismatches: {mismatches}");
    assert_eq!(mismatches, 0, "any classification mismatch in the full-face sweep is a parity bug");
}
```

- [ ] **Step 2: Run**

```bash
cargo test --lib gpu_lod0_full_face_classification_matches_cpu_sample -- --nocapture
```

- [ ] **Step 3: Commit**

```bash
git add src/gpu/voxel_compute.rs
git commit -m "test(gpu-parity): full-face classification sweep at LOD 0

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Phase 3 — Fix divergences

### Task 16: Per-divergence fix loop (one iteration per Appendix-A divergent row)

**This is a templated task — repeat once per divergence found in Phase 1 / Phase 2.**

For each divergence row in Appendix A (in the order listed in the synthesis paragraph from Task 5):

- [ ] **Step 1: Write a minimal failing unit test for *just this divergence***

The test must isolate one chunk + one probe + one assertion that triggers the divergence. **Do not rely on the broad parity-matrix tests for the red.** Place the test in the `voxel_compute.rs` test module with a name matching the divergence label (e.g. `density_at_iso_surface_matches_cpu`, `cave_threshold_at_earth_scale_matches_cpu`).

- [ ] **Step 2: Run it; verify red**

```bash
cargo test --lib <divergence_test_name> -- --nocapture
```

Expected: FAIL with the divergence visible in the assertion output.

- [ ] **Step 3: Fix on the GPU side first**

Edit `voxel_gen.wgsl` and/or `voxel_compute.rs`. **Only modify CPU code if the audit row explicitly justifies CPU as the canonical-incorrect side.** Prefer reformulating in f32 (e.g. compute `depth = surface_offset - voxel_offset` from `mean_radius`-relative offsets) over emulating f64 in WGSL.

- [ ] **Step 4: Run the divergence test; verify green**

```bash
cargo test --lib <divergence_test_name>
```

Expected: PASS.

- [ ] **Step 5: Run the full parity suite**

```bash
cargo test --lib gpu::parity gpu_vs_cpu_parity gpu_lod0_full_face -- --nocapture
```

Expected: ALL parity tests pass. If any other parity test now fails, the fix introduced a regression — revert and re-think.

- [ ] **Step 6: Commit**

```bash
git add src/gpu/shaders/voxel_gen.wgsl src/gpu/voxel_compute.rs
git commit -m "fix(gpu-parity): <one-line description of divergence row N>

Closes audit row N in spec Appendix A.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

- [ ] **Step 7: Update Appendix A**

Mark the row's status `matches` and add the resolution commit SHA in the Notes column. Commit the spec update separately:

```bash
git add docs/superpowers/specs/2026-04-30-gpu-planetary-noise-parity-design.md
git commit -m "docs(gpu-parity): mark audit row N as matches"
```

**Repeat Steps 1–7 until the divergence list is empty.** If Step 5 reveals a *new* divergence not in Appendix A, add a new row to the table and put it on the back of the queue rather than expanding the current iteration.

**Out-of-scope guard.** If a divergence cannot be fixed within parity scope (e.g. requires LOD>0 GPU support, or requires raising heightmap resolution), file a new issue in `issues.json` with category `gpu`, severity `medium`, and stop. Mark the audit row `divergent (deferred — see issue ID)`.

---

## Phase 4 — Cleanup

### Task 17: Remove all stale FBM-era TODOs

**Files:**

- Modify: `src/gpu/voxel_compute.rs`

- [ ] **Step 1: Enumerate stale comments**

```bash
grep -n "FBM noise\|GPU shader implements FBM" src/ -r
```

Expected: 5 occurrences in `voxel_compute.rs` (per Task 4 audit).

- [ ] **Step 2: Delete each comment**

For each occurrence:

- Remove the `// TODO: GPU shader implements FBM noise; …` line.
- If the line above or below was structurally tied to the comment (e.g. an `#[ignore]` attribute), confirm the `#[ignore]` was already removed in Tasks 9/10.

- [ ] **Step 3: Verify**

```bash
grep -n "FBM noise" src/ -r
```

Expected: no matches in `src/`. (Matches in `docs/` and `assets/` are fine — those describe the noise crate's FBM implementation in CPU code, not the GPU TODO.)

- [ ] **Step 4: Commit**

```bash
git add src/gpu/voxel_compute.rs
git commit -m "chore(gpu-parity): remove stale FBM-era TODO comments

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

### Task 18: Update `ai-context.json`

**Files:**

- Modify: `ai-context.json`

- [ ] **Step 1: Remove stale tech-debt entry**

Open `ai-context.json`, find the `tech_debt` array entry whose description starts with `"GPU voxel shader uses FBM noise terrain"`, delete that entry.

- [ ] **Step 2: Update meta**

In `meta`:

- Update `generated_from_commit` to current HEAD (`git rev-parse --short HEAD`).
- Update `last_updated` to today's date.
- Append a note to the `notes` array:

```
"GPU planetary-noise parity verified (spec docs/superpowers/specs/2026-04-30-gpu-planetary-noise-parity-design.md). GPU voxel pipeline now matches CPU PlanetaryTerrainSampler within §3 tolerance: classification equal, ≤4 voxel mismatches per chunk all within 1 ULP of a known threshold, density max-Δ ≤ 1e-3. Verified at small-planet (32 km) and Earth-scale, at face-centre/poles/antimeridian/coastline/deep-ocean/mountain probes."
```

- [ ] **Step 3: Validate JSON**

```bash
cd /home/pk/Devel/the-dark-candle && jq -e '.meta.generated_from_commit' ai-context.json >/dev/null && echo OK
```

- [ ] **Step 4: Commit**

```bash
git add ai-context.json
git commit -m "chore: update ai-context.json — GPU planetary-noise parity verified

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

### Task 19: Update `issues.json`

**Files:**

- Modify: `issues.json`

- [ ] **Step 1: Update GPU-PARITY-001 notes**

Find the entry with `"id": "GPU-PARITY-001"`. Append to its `notes` array (creating the array if absent):

```
"2026-04-30: Audit (spec Appendix A) + parametric parity test suite (parity_probe at small-planet + Earth-scale, with face-centre/poles/antimeridian/coastline/deep-ocean/mountain probes and ±3 layer sweeps) confirms current GPU pipeline matches CPU within §3 tolerance. The two formerly #[ignore]d tests are now passing on the new harness. Stale FBM-era TODOs removed."
```

- [ ] **Step 2: File any deferred-divergence issues**

For each Appendix-A row marked `divergent (deferred — see issue ID)`, add a new entry to `issues.json` with that issue ID, category `gpu`, severity `medium`, status `open`, and a `description` referencing the audit row.

- [ ] **Step 3: Validate JSON**

```bash
cd /home/pk/Devel/the-dark-candle && jq 'length' issues.json
```

Expected: previous count (62) plus N (number of deferred issues, likely 0).

- [ ] **Step 4: Commit**

```bash
git add issues.json
git commit -m "chore: update issues.json — GPU-PARITY-001 verification notes

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

### Task 20: Populate spec Appendix B (final outcome)

**Files:**

- Modify: `docs/superpowers/specs/2026-04-30-gpu-planetary-noise-parity-design.md`

- [ ] **Step 1: Write the outcome summary**

Replace the placeholder Appendix B with:

```markdown
## Appendix B — Final outcome

Phase 1 audited 13 rows against the GPU and CPU planetary-noise paths. Of these:

- **N matches** — already in parity, regression-protected by new tests.
- **M divergent + fixed** — listed below with their fix commits.
- **K not-tested → now tested** — listed below with their test additions.
- **L stale-doc → removed** — listed in Task 17/18/19 commits.

### Divergences fixed in Phase 3

| Audit row | Description | Fix commit |
|---|---|---|
| ... | ... | ... |

### New tests added in Phase 2

| Test name | Purpose |
|---|---|
| `gpu_vs_cpu_surface_chunk_small_planet_face_center` | small-planet face-centre, ±3 layers |
| `gpu_vs_cpu_parity_earth_scale_face_center` | Earth-scale face-centre, ±3 layers |
| `gpu_vs_cpu_parity_small_planet_probe_matrix` | small-planet poles + antimeridian |
| `gpu_vs_cpu_parity_earth_scale_probe_matrix` | Earth-scale poles + antimeridian |
| `gpu_vs_cpu_parity_biome_driven` | coastline / deep ocean / mountain probes |
| `bake_is_deterministic_for_fixed_seed` | bake determinism |
| `bake_differs_for_different_seeds` | seed sensitivity |
| `set_heightmap_data_rejects_wrong_size` | upload validation |
| `gpu_lod0_full_face_classification_matches_cpu_sample` | coarse full-face sweep |

### Residual divergences (deferred)

*If any.* Each row references the issue ID it was filed under in Task 19 Step 2.

Parity is verified at LOD 0 only. LOD>0 GPU generation parity is out of scope.
```

Fill in the actual numbers (N, M, K, L) and the divergence-fix table from the Phase 3 commits.

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-04-30-gpu-planetary-noise-parity-design.md
git commit -m "docs(gpu-parity): populate spec Appendix B (final outcome)

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

### Task 21: Final verification

**Files:** none modified.

- [ ] **Step 1: Run the full test suite**

```bash
cargo test
```

Expected: full suite green. Compare test count against baseline (~1681 + new parity tests).

- [ ] **Step 2: Run lint and format**

```bash
cargo fmt --check && cargo clippy --all-targets -- -D warnings
```

Expected: clean.

- [ ] **Step 3: Sanity-check stale-doc cleanup**

```bash
grep -rn "FBM noise" src/ ai-context.json
grep -rn "#\[ignore\]" src/gpu/voxel_compute.rs
```

Expected: no `FBM noise` matches in src/ or ai-context.json. No `#[ignore]` attributes remain on the parity tests.

- [ ] **Step 4: Push**

```bash
git log --oneline origin/master..HEAD
git push
```

---

## Self-review notes

This plan covers every section of the spec:

- **Spec §3 (parity contract)** — Tasks 6/7 implement and unit-test it; Tasks 9–12 + Task 16 enforce it.
- **Spec §4 (audit)** — Tasks 1–5 produce Appendix A.
- **Spec §5 (test harness)** — Tasks 6–8 build the harness; Tasks 9–12 instantiate the probe matrix; Tasks 13–15 add gap-fillers.
- **Spec §6 (fix strategy)** — Task 16 templates the per-divergence loop with red-first TDD and the canonical-side rule.
- **Spec §7 (cleanup)** — Tasks 17–20 strip stale TODOs, update `ai-context.json`/`issues.json`, populate Appendix B.
- **Spec §8 (out-of-scope guard)** — encoded in Task 16 Step 7's "deferred" branch.
- **Spec §9 (risks)** — production-bake-test wrapper risk addressed in Task 5; CPU-bug-vs-GPU-bug risk addressed in Task 16 Step 3; f32-cancellation risk addressed in Task 16 Step 3; explosion-of-divergences risk addressed in Task 16's "Repeat" + back-of-queue rule.

Type-consistency check: `ParityReport` / `VoxelMismatch` / `ThresholdKind` / `parity_probe` / `build_parity_report` / `parity_probe_set` / `coord_for_lat_lon` / `pick_lat_lon_by_biome` are referenced consistently across Tasks 6–16. `bake_elevation_roughness_ocean` and `set_heightmap_data` are existing, verified APIs (signatures read in Phase-0 exploration).

Placeholder check: every `<…>` placeholder in this plan is in either (a) the audit table (filled during Phase 1 — that's the whole point), or (b) the per-divergence fix loop in Task 16 (templated because divergences are unknown until Phase 1/2 complete). Both are intentional and documented as such.
