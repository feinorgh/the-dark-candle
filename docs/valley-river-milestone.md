# Milestone: Valley River Scene ✅

A target demonstration scene that exercises terrain generation, fluid dynamics,
static prop scattering, and lighting together for the first time. The scene
shows a procedurally generated valley on the planetary surface with a river
flowing through it, rocky terrain with scattered boulders, stones, and pebbles,
lit by the day-night sun cycle with terrain self-shadowing.

See also: [ROADMAP.md](ROADMAP.md) (project-level phasing),
[terrain-generation.md](terrain-generation.md) (terrain detail upgrade),
[atmosphere-simulation.md](atmosphere-simulation.md) (weather & lighting).

---

## What the milestone proves

- Planetary terrain can produce recognizable landforms (not just noise hills)
- AMR fluid simulation produces visible, flowing water in-game
- The world feels inhabited and textured with scattered natural objects
- Lighting sells the scene with depth via shadows

## Gap 1 — Procedural Valley & Channel Carving ✅

**Problem:** Terrain is pure Perlin noise — no valleys, ridges, or river
channels. Noise alone cannot produce the directed, branching drainage patterns
that define real landscapes.

**Approach:** D8 flow-accumulation on a coarse heightmap grid, cached on the
terrain generator, with valley carving applied during `generate_chunk()`.

1. **Coarse heightmap sampling** — evaluate `sample_height()` on a 512×512
   grid at 8 m resolution covering a 4096×4096 m region centered at world
   origin. Cost: ~0.3 ms (262 K Perlin evaluations).
2. **Sink filling** — iterative priority-queue flooding (Planchon & Darboux
   2002) eliminates closed depressions where flow would pool artificially.
3. **D8 flow direction** — each cell flows to its steepest downhill neighbor
   (8 directions). Flat cells flow toward the nearest lower cell.
4. **Flow accumulation** — traverse cells in topological order (highest
   first). Each cell passes its accumulation (1 + upstream total) to its
   downstream neighbor. Cells with high accumulation become channels.
5. **Channel carving** — lower the heightmap along high-accumulation paths
   by depth proportional to `ln(flow_count)`. Width widens with
   `√(flow_count)`. Cross-section profile is configurable (V-shaped vs
   U-shaped via cosine/Gaussian falloff).
6. **Erosion texturing** — channel bed → SAND (id 6), valley walls →
   exposed STONE where slope exceeds threshold, valley floor → DIRT.

**Caching:** `FlowMap` stored in `OnceLock` on `TerrainGenerator`. Computed
once on first chunk generation. Fully deterministic from world seed.

**Configuration (`ErosionConfig`, embedded in `TerrainConfig`):**
- `flow_threshold`: 50 (min accumulation to carve)
- `depth_scale`: 3.0 (channel depth = `min(12 m, 3 × ln(flow))`)
- `width_scale`: 2.0 (valley width = `2 × √flow`)
- `valley_shape`: 0.3 (0 = V-shaped, 1 = U-shaped)
- `region_size`: 4096 m, `cell_size`: 8 m

**Integration point:** Inside `TerrainGenerator::generate_chunk()`, after
`sample_height()` returns the base height and before material assignment.
Channels carved below `sea_level` auto-fill with water via existing placement
logic.

**Location:** New `src/world/erosion.rs` module.

**Dependencies:** Terrain system (Phase 1).

**Scope:** Flat terrain mode only (Phase 1). Spherical mode support deferred
to a future pass (requires tangent-plane projection for flow routing).

## Gap 2 — AMR Fluid Visual Surface & Plugin Activation ✅

**Problem:** `AmrFluidPlugin` exists but is not registered in `PhysicsPlugin`.
The AMR simulation runs but its free surface is never extracted into a
renderable mesh. Water voxels are static blocks with no flow.

**Completed:**

1. **Activated `AmrFluidPlugin`** — registered in `PhysicsPlugin::build()`
   alongside `LbmGasPlugin` and `FlipPicPlugin`.
2. **River flow seeding** — `seed_river_flow` system reads FlowMap direction
   for each WATER voxel in newly added chunks, sets initial velocity on
   the corresponding `FluidGrid` cells. Uses `NeedsFluidSeeding` marker
   component for one-shot execution.
3. **Boundary injection** — `inject_river_sources` system runs every tick,
   re-injects fluid at upstream boundary faces where the FlowMap indicates
   high flow. Sustains river flow without finite-volume drain-away.
4. **`FluidState::coords()`** accessor for iterating active fluid chunks.
5. **`UnifiedTerrainGenerator::flow_map()`** accessor for erosion FlowMap.

**Location:** `src/physics/amr_fluid/injection.rs` (new), changes in
`plugin.rs`, `mod.rs`, `chunk_manager.rs`, `terrain.rs`.

**Dependencies:** AMR fluid (Phase 3), terrain carving (Gap 1), meshing
(Phase 1).

## Gap 3 — Static Prop System (`PropData` + Scattering) ✅

**Completed** in commit `5245945`. Implemented `PropData` struct with RON
loading, `PropRegistry`, biome `prop_spawns` tables, chunk decoration system
(`NeedsDecoration`/`ChunkProps`), and the `decorate_chunks()` scatter system
in `src/procgen/props.rs`. Six prop RON files created (boulder, rock, pebble,
cobble, log, stick). All four biomes updated with `prop_spawns` tables.

**Files:** `src/procgen/props.rs`, `src/data/mod.rs`, `assets/data/props/*.prop.ron`,
`assets/data/biomes/*.biome.ron`.

## Gap 4 — Terrain Shadow Casting ✅

**Completed** in commits `1e537de` + `4fef32f`. Implemented `SunDirection`
resource, `ShadowConfig` (angle threshold, cone samples, half-angle),
`compute_terrain_shadows()` with DDA ray-cast + golden-spiral cone sampling
for soft penumbra edges, and `update_terrain_shadows()` system with
`LastShadowAngles` caching. Added `shadow: Vec<f32>` field to `ChunkLightMap`
with `apply_light_map()` multiplying `sun_transmittance × shadow_factor`.
Performance fix: `update_chunk_light_maps` skips non-dirty chunks.

**Files:** `src/lighting/shadows.rs`, `src/lighting/light_map.rs`,
`src/lighting/mod.rs`.

## Gap 5 — Prop Spawn ECS Integration ✅

**Completed** as part of Gap 3 (commit `5245945`). The `decorate_chunks()`
system queries chunks with `NeedsDecoration` marker, calls
`plan_chunk_prop_spawns()`, spawns `Prop` entities with `Transform` and
collision shapes, and stores entity handles in `ChunkProps` for cleanup on
chunk despawn. Duplicate decoration prevented by component lifecycle.

## Implementation Order

```
Gap 3 (PropData) ──→ Gap 5 (Spawn ECS) ──────────────────── ✅ Done
Gap 4 (Terrain shadows) ─────────────────────────────────── ✅ Done
Gap 1 (Valley carving) ──→ Gap 2 (Water surface) ──→ Scene integration
         ✅ Done                   ✅ Done              ↑ remaining
```

- **Gap 3 + 5** completed: prop data, scattering, and ECS spawning.
- **Gap 4** completed: terrain shadow casting with soft penumbra.
- **Gap 1** completed: D8 flow accumulation + valley carving (commit `358f097`).
- **Gap 2** completed: AMR fluid plugin activation + river seeding + boundary
  injection.
- **Scene integration** is the final step — all gaps are now closed.

## Success Criteria

- A valley is visible on the planetary surface with a recognizable
  V/U-shaped cross section and sloped walls.
- Water flows through the valley channel with visible motion (not static
  blue blocks).
- At least 3 prop types (boulder, rock, pebble) are scattered on the
  terrain with density varying by slope and proximity to water.
- The valley walls cast shadows onto the valley floor that move with the
  sun cycle.
- The scene runs at interactive frame rates (≥30 FPS) with at least a
  4×4 chunk view distance.
