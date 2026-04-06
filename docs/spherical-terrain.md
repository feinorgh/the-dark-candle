# Phase 8 — Spherical Planetary Terrain ✅

Transform the flat-plane terrain into a spherical planetary model. The planet is
a sphere centered at world origin `(0, 0, 0)` with configurable radius. Surface
features come from noise sampled in spherical coordinates. Geological layers
(core → mantle → crust) are defined by radial bands. The rendering pipeline
uses the **V2 cubed-sphere system** (`src/world/v2/`) — six cube faces mapped
to a sphere with greedy-meshed 32³ chunks loaded in radial shells. The original
V1 Cartesian pipeline has been removed.

See also: [ROADMAP.md](ROADMAP.md) (project-level phasing),
[terrain-generation.md](terrain-generation.md) (detail & world-gen options).

---

## Design

- **Planet center at origin.** All radial math is trivial: `distance = length(pos)`.
- **Cubed-sphere chunk grid.** The V2 pipeline uses `CubeSphereCoord` (face,
  u, v, layer) to address chunks on the six cube faces mapped to the sphere.
  Each chunk is still 32³ voxels; only the coordinate addressing and chunk
  loader know about the sphere geometry. Greedy meshing and voxel simulation
  are coordinate-agnostic.
- **Spherical noise sampling.** Convert world position `(x, y, z)` → unit direction
  → `(lat, lon)` via `atan2`. Noise at `(lat, lon)` produces surface radius
  displacement. No pole distortion because noise is sampled on a sphere, not
  projected from a plane.
- **Shell-based chunk loading.** Only load chunks within a radial band around the
  surface (camera altitude ± view depth). Deep interior and outer space chunks are
  never allocated — same ~250 active chunks as the current flat system.
- **Real planetary gravity.** Apparent gravity = gravitational attraction toward
  planet center + centrifugal pseudo-force from rotation. The resultant vector
  defines local "down" — it points along the inward surface normal of the geoid
  (equipotential surface), just like on a real planet. On a non-rotating sphere
  this is purely radial; rotation deflects it toward the equatorial plane and
  bulges the geoid into an oblate spheroid. Slope forces, ground detection, and
  "which way is up" all derive from this single vector — no hardcoded direction.

## Scale (1 voxel = 1 m)

| Planet | Radius | Chunks across diameter | Feasibility |
|--------|--------|------------------------|-------------|
| Small moon | 10 km | ~625 | Immediate — entire surface tractable |
| Default | 32 km | ~2,000 | Immediate — shell loading keeps it light |
| Large | 100 km | ~6,250 | Needs aggressive LOD + streaming |
| Earth-scale | 6,371 km | ~398K | Future — requires multi-res octree leaves |

Default planet: **32 km radius** (~200 km² surface area, comparable to a large
island). Room for mountains, oceans, biomes, and plate tectonics without needing
Earth-scale infrastructure.

## Implementation Steps

1. **`PlanetConfig`** — new `src/world/planet.rs` + `assets/data/planet_config.ron`
   with center, mean radius, sea level radius, layer radii (inner core, outer core,
   mantle, crust thickness), height scale, rotation rate (rad/s), rotation axis,
   surface gravity (m/s²), planet mass (kg). Data-driven via `RonAssetPlugin`.

2. **Spherical terrain generator** — refactor `terrain.rs`. For each voxel:
   compute distance from center, derive `(lat, lon)`, sample surface radius from
   noise, assign material by radial depth band (core → mantle → crust → surface →
   soil → air/water). Cave carving via 3D noise within the crust band.

3. **Shell-aware chunk loading** — modify `chunk_manager.rs` `desired_chunks()`.
   Compute which chunks intersect the surface shell ± view distance. Skip chunks
   whose centers are deeper than `surface_radius - depth_limit` or higher than
   `surface_radius + sky_limit`.

4. **Planetary gravity model** — modify `gravity.rs` `apply_forces()`. Compute
   apparent gravity per entity as the sum of two forces:
   - **Gravitational acceleration**: `g⃗ = -(G·M / r²) · r̂` where `r̂` is the
     unit vector from center to entity. At the surface this equals the configured
     surface gravity (default 9.80665 m/s²); varies with altitude via inverse-square.
   - **Centrifugal pseudo-force**: `a⃗_c = ω² · d⊥` where `ω` is the planet's
     angular velocity and `d⊥` is the entity's perpendicular distance from the
     rotation axis. This deflects "down" toward the equatorial plane and naturally
     produces an oblate geoid.
   The resultant `g⃗_apparent = g⃗_grav + a⃗_centrifugal` defines local "down" for
   all physics — slope forces, ground detection, "which way is up", buoyancy
   direction, fluid settling. No hardcoded -Y anywhere.
   Update `collision.rs` to find ground along the local gravity vector instead of
   a Y-column scan. `PlanetConfig` gains `rotation_rate: f64` (rad/s; Earth ≈
   7.292e-5).

5. **Altitude-dependent systems** — `WorldConstants.sea_level_y` → `sea_level_radius`.
   Barometric formula uses `altitude = distance_from_center - sea_level_radius`.
   Biome height ranges reinterpreted as altitude above/below sea level radius.

6. **Test suite update** — validate spherical shape, shell loading, radial gravity,
   radial altitude calculations. Preserve existing test semantics where applicable.

7. **World-gen extension points** — architecture prep for future phases:
   `WorldGenPhase` enum, `TectonicPlateConfig`, `GeologicalLayer` struct, spherical
   Voronoi placeholder for plate boundaries. No full implementation yet — just the
   hooks that make the next phase possible.

## What stays unchanged

Chunk internals (`Chunk`, `ChunkCoord`, `Voxel`, `MaterialId`), octree (`OctreeNode`,
`VoxelAccess`), meshing, LOD, interpolation, refinement, all fluid simulations
(AMR, LBM, FLIP), chemistry, biology, behavior, social systems, material RON files.

## Future: World Generation Pipeline

Once spherical terrain is in place, subsequent phases can layer on:

- **Tectonic plates** — spherical Voronoi tessellation. Each plate gets a drift
  vector and rotation rate. Convergent boundaries → mountain ranges / subduction.
  Divergent boundaries → rift valleys / mid-ocean ridges. Transform boundaries →
  fault lines.
- **Geological strata** — material layers within the crust defined by depth,
  composition, and tectonic history. Sedimentary layers near the surface, igneous
  intrusions from volcanic activity, metamorphic zones at depth.
- **Erosion** — hydraulic and thermal erosion during world-gen. Rivers carve
  valleys, glaciers scour mountains, wind shapes deserts. Runs as a multi-pass
  simulation before the player enters.
- **Volcanic activity** — magma plumes from mantle, eruption probability based on
  crust thickness and plate stress. Lava flows use existing fluid simulation.
- **Ocean currents & climate** — large-scale fluid simulation determines temperature
  and moisture distribution. Drives biome placement and weather patterns.

---

## Planetary Terrain Connection ✅

The full geodesic world generation pipeline is now connected to the voxel game
via `--planet`. This replaces the future-work items above with a working system.

### How it works

1. **CLI pipeline** — `--planet [--planet-seed N] [--planet-level L]` runs
   `run_tectonics → run_biomes → run_geology` before Bevy starts, producing a
   `PlanetData` struct with per-cell elevation, biome, temperature, precipitation,
   rock type, and ore deposits on a geodesic icosahedral grid.

2. **`PlanetaryData` resource** — wraps `Arc<PlanetData>` + `Arc<CellIndex>` and
   is inserted into the ECS before plugins run.

3. **`rebuild_terrain_gen_if_planetary`** — a `PostStartup` system that swaps
   the `SharedTerrainGen` to `UnifiedTerrainGenerator::Planetary` if
   `PlanetaryData` is present.

4. **`PlanetaryTerrainSampler`** — samples one geodesic cell per column (lx, lz)
   via `CellIndex::nearest_cell`, then calls `sample_detailed_elevation` (IDW +
   fractal noise) for the surface radius. Sweeps ly to assign materials:
   - Surface layer: biome-mapped material (GRASS, SAND, ICE, etc.)
   - Sub-surface: rock-type-mapped material (STONE, LAVA, etc.)
   - Ore veins: from `ore_bitmask` at depth, ~3% sparsity
   - Water: ocean fill below sea level
   - Cave carving: reuses `SphericalTerrainGenerator` Perlin noise

5. **`ChunkBiomeData` component** — attached to every chunk entity in planetary
   mode. Carries `planet_biome`, `temperature_k`, `precipitation_mm`,
   `surface_rock`, `ore_bitmask`. Procgen systems (`spawn_creatures`,
   `spawn_items`) use temperature + precipitation to select `BiomeData` handles
   instead of the height heuristic.

6. **`CellIndex`** — 1°×1° lat/lon bin index with adaptive `lat_search` radius
   (scales with `sqrt(4π/n)` so it works correctly for all subdivision levels,
   including coarse grids used in tests).

### Usage

```bash
# Default (level 5, seed 42)
cargo run --release -- --planet

# Custom seed and resolution
cargo run --release -- --planet --planet-seed 1337 --planet-level 6

# Equivalent using --scene alias
cargo run --release -- --scene planet
```
