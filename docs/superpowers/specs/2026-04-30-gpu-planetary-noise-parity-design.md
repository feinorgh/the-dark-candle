# GPU Planetary-Noise Parity — Design

**Date:** 2026-04-30
**Status:** Draft (awaiting user review)
**Related issues:** `GPU-PARITY-001` (issues.json — currently marked `resolved`, but stale tech-debt + ignored tests + stale TODOs remain).
**Related commits:** `e48337a` (Phase 3 GPU heightmap), `b58d5ca` (heightmap split into IDW/roughness/ocean), `fed1ca0` (skip GPU until heightmap ready), `aac0783` (Phase 3 doc update).

---

## 1. Problem

The GPU voxel-generation pipeline (`src/gpu/shaders/voxel_gen.wgsl` + `src/gpu/voxel_compute.rs`) was upgraded in commit `e48337a` to sample a baked equirectangular planetary heightmap (plus roughness and ocean buffers) instead of the original FBM terrain. That change made the GPU path *algorithmically* match the CPU `PlanetaryTerrainSampler`. However, several artefacts of the old FBM-only state remain:

- Two parity tests (`gpu_vs_cpu_surface_chunk_comparison`, `gpu_vs_cpu_parity_earth_scale_surface_chunk`) are still `#[ignore]`d with stale "GPU shader implements FBM noise" TODOs. They were never rewritten against the heightmap path.
- `ai-context.json` `tech_debt` still claims "GPU voxel shader uses FBM noise terrain — does not yet implement Planetary IDW".
- We have no positive evidence — at the test level — that GPU and CPU produce matching voxels under all the conditions the runtime actually uses (poles, antimeridian, coastlines, Earth-scale radii, varied roughness, deep underground).

**Goal:** establish *exact* per-voxel parity (modulo f32-vs-f64 rounding tolerance, see §3) between the GPU and CPU planetary terrain pipelines, prove it with a parametric parity test suite, fix any divergence that audit + tests uncover, and clean up the stale documentation that lingers from the FBM era.

## 2. Approach

Hybrid (audit-first then test-driven):

1. **Phase 1 — Code audit.** Side-by-side read of GPU and CPU paths. Produce an audit table: each link in the GPU planetary-noise chain → status (`matches` / `divergent` / `not-tested` / `stale-doc`).
2. **Phase 2 — Test harness.** Build a parametric parity test harness that drives the *production* pipeline (real `PlanetData`, real heightmap bake, real GPU upload). Replace the two existing `#[ignore]`d tests with this harness. Sweep a probe matrix at two planet scales.
3. **Phase 3 — Fix divergences.** Each `divergent` audit row → one minimal failing unit test → one focused fix → harness stays green.
4. **Phase 4 — Cleanup.** Remove stale TODOs / tech-debt / `#[ignore]` markers. Update `ai-context.json` and `issues.json`. Document final audit outcome in this spec.

## 3. Parity contract

**Tolerance level: boundary-tolerant.** GPU runs in f32, CPU runs in f64; rounding will sometimes flip a voxel that lies within 1 ULP of a classification threshold (cave threshold, surface iso-surface, ore probability cut). Such voxels are tolerable; bulk material divergence is not.

Per chunk, the parity contract is:

| Quantity | Bound |
|---|---|
| Chunk classification (`AllAir` / `AllSolid` / `Mixed`) | must be **equal** |
| Voxel material disagreements | ≤ `⌈0.01 % × CHUNK_VOLUME⌉ = 4` per chunk |
| Each disagreement | must be `within_1_ulp` of the relevant threshold; else hard fail |
| Voxel density max-Δ | ≤ `1e-3` (signed-distance units) |

These bounds apply at LOD 0. LOD > 0 parity is **out of scope** for this spec (see §8).

## 4. Phase 1 — Code audit

The audit is a single table committed alongside this spec (an appendix to this document, populated during Phase 1 work). For each row: GPU site (file:line), CPU site (file:line), status, notes, and "needs Phase-3 fix?" / "needs Phase-2 test?".

### Audit checklist

The audit must cover every item below. Items found `divergent` flow into Phase 3; items found `not-tested` flow into Phase 2.

1. **Heightmap bake** — CPU function producing the equirectangular elevation buffer: resolution (GPU has hard-coded `HEIGHTMAP_W = HEIGHTMAP_H = 1024` in `voxel_gen.wgsl`), projection orientation, value semantics (Δr from `mean_radius`, metres), determinism for fixed seed.
2. **Roughness bake** — same checks, plus value range/encoding.
3. **Ocean bake** — nearest-neighbour vs bilinear at coastlines; both sides agree. `voxel_gen.wgsl::sample_ocean` uses nearest-neighbour and a `u32` flag.
4. **Sampling math** (`sample_heightmap` / `sample_roughness` / `sample_ocean` in `voxel_gen.wgsl:428-489`) — u/v wrapping at antimeridian, pole clamping, texel-centre half-pixel offset, bilinear weights.
5. **Surface radius pipeline** — IDW heightmap → roughness FBM modulation → ocean clamp. Compare exactly to `PlanetaryTerrainSampler::sample_surface_radius_at` (CPU). Pay particular attention to the order of operations: ocean clamp *before* or *after* roughness modulation must match.
6. **Surface density** (used by surface-nets meshing) — sign convention, value at iso-surface. `terrain_density_gpu` (`voxel_gen.wgsl:672`) vs CPU equivalent.
7. **`material_at_radius_gpu`** (`voxel_gen.wgsl:624`) ↔ CPU material classifier — depth bins, soil depth, sea-level interaction.
8. **Strata, ores, caves, crystals** (`strata_material_gpu`, `ore_material_gpu`, `is_cave_gpu`, `cave_fill_material_gpu`, `is_crystal_deposit_gpu` in `voxel_gen.wgsl:555-622`) — per-function perm-table seeds, frequencies, thresholds, fill materials.
9. **Cube-face frame + lat/lon** — `quat_rotate(rotation_axis, …)` and `lat_lon` (`voxel_gen.wgsl:395-426`) vs CPU `CubeSphereCoord` axis basis. Confirm the rotation_axis quaternion is applied identically on both sides.
10. **`sortable_encode` / `sortable_decode`** (`voxel_gen.wgsl:534-554`) — monotonicity at sign change, used by atomic surface-offset reduction in `surface_pass`.
11. **Classify-pass thresholds** for `AllAir` / `AllSolid` / `Mixed` buckets (`voxel_gen.wgsl:824` `classify_pass`) — match CPU `CachedVoxels` downgrade rules in `world::v2::terrain_gen::downgrade_uniform_voxels`.
12. **LOD support** — does the GPU path actually generate anything other than LOD 0 today, and does the CPU path differ? If GPU is LOD-0-only by design, this spec restricts parity to LOD 0; see §8.
13. **Stale TODO / comment scan** — every "FBM noise" comment, every `#[ignore]`d test, every tech-debt entry — flagged with current truth. Phase 4 deletes them.

### Phase 1 output

Audit table appended to this document. No code changes in Phase 1.

## 5. Phase 2 — Parity test harness

### Module

New tests in `src/gpu/voxel_compute_parity_tests.rs` (or extend existing `#[cfg(test)]` block in `voxel_compute.rs`). All tests sit behind the existing `GPU_TEST_LOCK` and skip cleanly when no GPU adapter is present (existing CI pattern — see `voxel_compute.rs:1438-1442`).

### Harness API

```rust
fn parity_probe(
    planet: &PlanetConfig,                     // small or Earth-scale
    planet_data: &Arc<PlanetData>,
    unified: &Arc<UnifiedTerrainGenerator>,
    compute: &GpuVoxelCompute,                 // heightmap already uploaded
    coord: CubeSphereCoord,
) -> ParityReport;
```

`ParityReport` records:

- GPU and CPU classification.
- GPU and CPU solid-voxel count.
- Per-voxel material mismatch list, each tagged `{within_1_ulp: bool, threshold_kind: SurfaceIso | CaveThreshold | OreCut | StrataBoundary | …}`.
- Density max-Δ and the voxel index at which it occurs.

The test asserts the four bounds from §3 (classification equal, ≤4 mismatches, all within 1 ULP, density Δ ≤ 1e-3). Failure messages include the chunk coord, the failing voxel index, and the GPU/CPU values.

### Probe matrix (per scale)

**Locations:**

- PosX face centre (lat 0, lon π/2) — already exercised by the existing earth-scale test.
- North pole (lat ≈ +π/2) — new.
- South pole (lat ≈ −π/2) — new.
- Antimeridian seam (lon ≈ ±π, on a face boundary) — new.
- A coastline cell — sample the baked ocean buffer to find a chunk straddling 0/1 — new.
- A deep-ocean cell — new.
- A high-roughness mountain cell — sample the baked roughness buffer to find one — new.

**Layers:** for each location, sweep `surface_layer + Δ` for `Δ ∈ {-3, -2, -1, 0, +1, +2, +3}`. Guarantees at least one Mixed chunk per location, and exercises the surface, near-surface, and deep-interior classification paths.

### Scales

1. **Small planet** — 32 km radius, the parameter set already used by `gpu_vs_cpu_surface_chunk_comparison`. Faster compile/run; useful for debugging fixes.
2. **Earth-scale preset** — `ScenePreset::SphericalPlanet`. Catches f32 cancellation regressions like the one called out in the existing earth-scale test comment (`voxel_compute.rs:1755-1762`).

### Heightmap setup

Tests must call the **production** heightmap bake function — whatever the runtime uses to populate the buffers passed to `GpuVoxelCompute::set_heightmap`. The audit identifies which function this is. **Tests do not re-implement the bake**; otherwise we'd be validating against a parallel implementation, not against reality.

If the production bake is buried inside an async system or worker thread, Phase 2 includes a small refactor to expose a synchronous `bake_planetary_heightmaps(&PlanetData, &PlanetConfig) -> HeightmapBundle` helper that both the runtime and the tests call. That refactor is in-scope.

### Gap-filler tests (driven by audit findings)

- `heightmap_bake_is_deterministic_for_fixed_seed` (CPU-only — does not need a GPU adapter).
- `set_heightmap_validates_buffer_dimensions` — uploading a wrong-size buffer must error with a clear message, not silently corrupt sampling.
- `gpu_lod0_full_face_classification_matches_cpu_sample` — coarse statistical check sweeping ~64 chunks over a face. Catches "everything is AllAir" or "everything is AllSolid" regressions cheaply without per-voxel comparison.

### Two existing ignored tests

- `gpu_vs_cpu_surface_chunk_comparison` — rewritten on top of `parity_probe` (small-planet scale). `#[ignore]` removed.
- `gpu_vs_cpu_parity_earth_scale_surface_chunk` — rewritten on top of `parity_probe` (Earth-scale). `#[ignore]` removed.

Both stale "FBM noise" comments deleted.

## 6. Phase 3 — Fix strategy

For each `divergent` audit row:

1. **Red:** minimal failing unit test (single chunk, single probe) reproducing the divergence. Committed first.
2. **Green:** fix in WGSL or uploader; CPU only if the CPU path is demonstrably wrong (the audit row must justify which side is canonical).
3. **Confirm:** full parity harness stays green at both scales.
4. **Commit:** `fix(gpu-parity): <audit-row-label>` referencing the audit table row.

**Bounded f32 reformulations allowed.** When a mismatch comes from f32 cancellation, prefer reformulating the math (e.g. compute `depth_offset = surface_offset - voxel_offset` from `mean_radius`-relative offsets — exactly what the existing earth-scale test comment mandates) over emulating f64 in WGSL.

**Out-of-scope guard.** If the audit uncovers something larger than parity (e.g. CPU is itself wrong at the poles, or the heightmap projection has a half-pixel skew at sub-1 m precision), file a new issue in `issues.json` and **do not** expand scope here. Parity-as-defined-in-§3 ships first.

## 7. Phase 4 — Cleanup

- Remove the obsolete tech-debt entry from `ai-context.json` (the one claiming GPU uses FBM).
- Update `ai-context.json` notes / phases / `meta.generated_from_commit`.
- Update `issues.json`: confirm GPU-PARITY-001 "resolved" status now reflects measured parity; document the audit + test coverage state in its `notes`. File any new issues the audit uncovers.
- Strip every stale `// TODO: GPU shader implements FBM noise` from the codebase (search-and-destroy).
- Append the **final audit outcome** to this spec: mismatches found, mismatches accepted as f32-tolerance, mismatches fixed, residual divergences (if any) documented with their issue IDs.

## 8. Out of scope

- **LOD > 0 parity.** If the audit finds that the GPU path is currently LOD-0-only, parity in this spec is restricted to LOD 0, and a separate follow-up issue is filed for LOD-aware GPU generation.
- **Heightmap-resolution bumps** beyond the current 1024×1024. If the audit measures that 1024² is too coarse to satisfy the §3 tolerance at Earth scale near sharp features, the spec is amended in one of two ways: (a) widen the tolerance with explicit justification, or (b) raise the resolution as a follow-up issue. We do not silently change the resolution mid-spec.
- **GPU surface-nets meshing parity.** This spec covers voxel generation only. Surface-nets meshing on top of those voxels is a separate concern.

## 9. Risks

- **Production bake is awkward to call from tests** → solved by the synchronous `bake_planetary_heightmaps` helper described in §5.
- **Audit reveals a CPU bug, not a GPU bug.** Possible at the poles or antimeridian. Handled by the "side selection rule" in §6.
- **f32 cancellation at Earth scale produces > 4 mismatches per chunk in clean cells (not just within-1-ULP cells).** If so, the fix is the offset-from-`mean_radius` reformulation described in §6, not tolerance widening.
- **A failing parity test reveals dozens of unrelated divergences.** Mitigation: each fix gets its own commit + minimal test; if the list explodes, we stop, re-scope this spec, and list the residual items as follow-up issues rather than ballooning the present effort.

## 10. Deliverables

- This spec, with appendix A (audit table) populated during Phase 1 and appendix B (final outcome) populated during Phase 4.
- New parity test harness with the probe matrix above and the three gap-filler tests.
- Two existing tests rewritten and de-ignored.
- Per-divergence fix commits with their minimal unit tests.
- `ai-context.json` and `issues.json` updated.
- All stale FBM-era TODOs / comments / tech-debt entries removed.

## Appendix A — Audit table (populated in Phase 1)

*To be filled in during Phase 1 work.*

| # | Item | GPU site | CPU site | Status | Phase-3 fix? | Phase-2 test? | Notes |
|---|---|---|---|---|---|---|---|
| 1 | Heightmap bake — resolution, projection, orientation, units | `src/planet/gpu_heightmap.rs:30-33,56-78`; `src/gpu/shaders/voxel_gen.wgsl:419-420,428-454` | `src/world/terrain.rs:937-941`; `src/world/planetary_sampler.rs:188-218`; `src/planet/detail.rs:281-284` | matches | no | yes | Row 0 = north pole (+π/2); col 0 = lon −π. Pixel centres `(i+0.5)/res`. WGSL `u*W−0.5` matches. Determinism not tested. |
| 2 | Roughness bake — value range and encoding | `src/planet/gpu_heightmap.rs:70-73`; `src/gpu/shaders/voxel_gen.wgsl:458-476` | `src/world/planetary_sampler.rs:230-247`; `src/planet/detail.rs:102-129` | matches | no | yes | Roughness ∈ [0,1] enforced by `clamp(0.0,1.0)` at line 128. Bilinearly interpolated on both sides. |
| 3 | Ocean bake — nearest-neighbour intent | `src/planet/gpu_heightmap.rs:70-74`; `src/gpu/shaders/voxel_gen.wgsl:481-489` | `src/world/planetary_sampler.rs:242-245` | matches | no | yes | Baked as binary 1.0/0.0. WGSL samples with `i32(u*W)` truncation (NN). Prevents coastline bleed. |
| 4 | Sampling math — u/v wrap, pole clamp, half-pixel | `src/planet/gpu_heightmap.rs:56-68` | `src/gpu/shaders/voxel_gen.wgsl:428-489` | matches | no | yes | u-wrap: `fract(lon/(2π)+0.5)`; v: `0.5−lat/π`. Sub-pixel `u*W−0.5`, `floor+frac`. x wrap modulo W; y clamp [0,H−1]. |
| 5 | Surface radius pipeline — IDW→roughness FBM→ocean clamp order | `src/gpu/shaders/voxel_gen.wgsl:708-728` | `src/world/planetary_sampler.rs:188-218`; `src/planet/detail.rs:174-201` | matches | no | yes | IDW + (fbm/ridge mix × roughness × 2000m). Ocean clamp `min(-2.0)` if `is_ocean \|\| idw<0`. Order identical both sides. |
| 6 | Surface density — sign convention and iso-value | `src/gpu/shaders/voxel_gen.wgsl:672-674` | `src/world/terrain.rs:41-43` | matches | no | yes | `clamp(0.5 + depth*0.5, 0.0, 1.0)`. Iso=0.5 → depth=0 (surface). Positive=solid. Identical f32 cast. |
| 7 | `material_at_radius` — depth bins, soil, sea-level | `src/gpu/shaders/voxel_gen.wgsl:624-670` | `src/world/terrain.rs:754-849` | matches | no | yes | Functionally equivalent: GPU passes `r_offset` (from mean_radius) for f32-cancellation guard; depth `= surface_r − r` identical on both sides. |
| 8 | Strata/ores/caves/crystals — perm slots, freq, thresholds | `src/gpu/shaders/voxel_gen.wgsl:555-622` (PERM_STRATA=22, ORE_{COAL,COPPER,IRON,GOLD}=23-26, CAVE_{CAVERN,TUNNEL,TUBE_XZ,TUBE_XY}=27-30, CRYSTAL=31) | `src/world/terrain.rs:85-235` | matches | no | yes | Strata: freq 0.02, depths 20/60. Ores: coal 5-30m/-0.15, copper 15-50m/-0.20, iron 30-80m/-0.25, gold 50+m/-0.35. Caves: cavern 0.01/×0.5, tunnel 0.04/×1.2, tubes 0.025/×0.85 AND. Crystals: 40+m, 0.10/-0.30. |
| 9 | Cube-face frame & lat/lon | `src/gpu/shaders/voxel_gen.wgsl:395-414,698,700` | `src/world/v2/cubed_sphere.rs:230-305`; `src/planet/detail.rs:281-294` | matches | no | yes | GPU \`quat_rotate\` applies face-aligned quaternion to local→world. \`lat_lon\` uses Y-up: \`asin(dir.y)\`, \`atan2(dir.x,dir.z)\`. CPU \`world_transform_scaled\` builds identical rotation. Comment at wgsl:401-405 explicitly cites CPU convention match. |
| 10 | sortable_encode / sortable_decode | `src/gpu/shaders/voxel_gen.wgsl:534-551` | n/a (GPU-only logic) | matches | no | no | Neg: \`~bits\`; non-neg: \`bits \| 0x80000000\`. Verified monotonic: -1.0→0x407FFFFF < -0.0→0x7FFFFFFF < +0.0→0x80000000 < +1.0→0xBF800000. No CPU counterpart (atomicMin/Max surface-pass). |
| 11 | Classify-pass thresholds | `src/gpu/shaders/voxel_gen.wgsl:824-857` | `src/world/v2/terrain_gen.rs:189-219` | matches | no | yes | GPU predicts classification from bounding-box vs surface/sea/cave extents; CPU downgrades post-voxel-gen by scanning for uniformity. Both produce AllAir/AllSolid/Mixed. GPU AllSolid emits stone; CPU AllSolid excludes transparent/water. Semantically equivalent. |
| 12 | LOD support | `src/gpu/shaders/voxel_gen.wgsl:114,808-814`; `src/gpu/voxel_compute.rs:1009-1031` | `src/world/v2/cubed_sphere.rs:256,302`; `src/world/v2/terrain_gen.rs:102-108` | matches | no | yes | Both sides compute \`lod_scale = 1 << coord.lod\`. GPU \`ChunkDesc\` carries \`lod_scale\` field; shader scales radial extent + density gradient. CPU \`world_transform_scaled\` applies identical multiplier. Full LOD support on both sides; spec §3 restricts parity testing to LOD 0 only. |

## Appendix B — Final outcome (populated in Phase 4)

*To be filled in at the end of Phase 4.*
