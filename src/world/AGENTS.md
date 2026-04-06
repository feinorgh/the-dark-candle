# World Module

Voxel world infrastructure: types, chunk storage, terrain generation (NoiseStack noise engine, biome integration, scene presets), greedy meshing, erosion (D8 valley carving + hydraulic), collision queries, and planetary terrain sampling.

The active rendering pipeline is the **V2 cubed-sphere system** (`src/world/v2/`) which uses `CubeSphereCoord` addressing and greedy meshing. The V1 Cartesian chunk pipeline has been removed.

## Spatial Mapping

**1 voxel = 1 meter.** This is the fundamental mapping that makes all SI units work without coordinate transforms. All physics values (density in kg/m³, force in N, energy in J) apply directly to voxel-sized volumes.

## Files

| File | Purpose |
|------|---------|
| `voxel.rs` | `MaterialId` (28 types), `Voxel` type |
| `chunk.rs` | `Chunk` (32³ flat array), `ChunkCoord` |
| `chunk_manager.rs` | Shared resource types: `ChunkMap`, `SharedTerrainGen`, `TerrainGeneratorRes`, `ChunkLoadRadius`, `PendingChunks` |
| `noise.rs` | `NoiseStack` — composable FBM, ridged fractal, domain warping, terrain selector, micro-detail, continent masks |
| `biome_map.rs` | `EnvironmentMap` — deterministic temperature/moisture, slope/altitude surface material selection |
| `terrain.rs` | Unified terrain generator (flat/spherical/planetary), geological strata, ore veins, multi-scale caves |
| `scene_presets.rs` | 8 scene presets with tuned `PlanetConfig` (alpine, archipelago, desert, plains, volcanic, tundra, valley, spherical) |
| `erosion.rs` | D8 valley carving + hydraulic erosion (droplet, grid, combined modes), repose slippage |
| `planet.rs` | `PlanetConfig`, `NoiseConfig`, `HydraulicErosionConfig`, terrain mode enum |
| `planetary_sampler.rs` | `PlanetaryTerrainSampler` — IDW interpolation from geodesic cells to voxel chunks |
| `meshing.rs` | Surface Nets mesh algorithm library (used internally by V2 greedy mesher) + per-material vertex coloring |
| `collision.rs` | Ground height queries for camera/entity gravity |
| `raycast.rs` | Discrete 3D grid ray march (26 directions), surface-voxel detection |

## Constants

- `CHUNK_SIZE = 32` — edge length in voxels (32³ = 32,768 per chunk)
- `CHUNK_VOLUME = 32,768`

### Voxel Defaults (SI Units)

| Field | Default | Unit | Notes |
|-------|---------|------|-------|
| `temperature` | 288.15 | K | Standard atmosphere (15 °C) |
| `pressure` | 101325.0 | Pa | Sea level atmospheric pressure |
| `damage` | 0.0 | — | 0.0 = intact, 1.0 = destroyed |

### MaterialId Registry (28 types)

| ID | Name | Category | Notes |
|----|------|----------|-------|
| 0 | Air | — | Default, transparent |
| 1 | Stone | Rock | Structural, high hardness |
| 2 | Dirt | Soil | Terrain surface layer |
| 3 | Water | Fluid | Swimmable |
| 4 | Iron | Metal | Dense, high melting point |
| 5 | Wood | Organic | Flammable |
| 6 | Sand | Granular | Desert/beach terrain surface |
| 7 | Grass | Organic | Spreads to dirt (biology) |
| 8 | Ice | Frozen | Frozen water |
| 9 | Steam | Gas | Boiled water |
| 10 | Lava | Molten | Molten stone |
| 11 | Ash | Combustion | Fire product |
| 12 | Glass | Mineral | Transparent, refractive (n=1.52) |
| 13 | Oxygen | Gas | Combustion oxidizer |
| 14 | Hydrogen | Gas | Combustion fuel |
| 15 | Organic Matter | Organic | Decomposition product |
| 16 | Twig | Organic | Tree branches |
| 17 | Dry Leaves | Organic | Tree foliage (flammable) |
| 18 | Bark | Organic | Tree outer layer |
| 19 | Charcoal | Combustion | Burned wood |
| 20 | Sandstone | Sedimentary | 2–20 m depth strata |
| 21 | Limestone | Sedimentary | 2–20 m depth strata |
| 22 | Granite | Igneous | 60 m+ depth strata |
| 23 | Basalt | Igneous | 60 m+ depth strata |
| 24 | Coal | Ore | 5–30 m depth range |
| 25 | Copper Ore | Ore | 15–50 m depth range |
| 26 | Gold Ore | Ore | 50 m+ depth range |
| 27 | Quartz Crystal | Mineral | Deep cave walls only |

When adding new materials: define a `MaterialId` constant in `voxel.rs`, create a `.material.ron` file in `assets/data/materials/`, and update this table.

## Dependencies

- **Imported by:** physics, chemistry, biology, behavior, camera
- **Imports from:** `crate::camera::FpsCamera` (for chunk loading around player)

## NoiseStack Engine (`noise.rs`)

Composable multi-layer noise system replacing the old 2-layer Perlin blend.

### Configuration

`NoiseConfig` (serde-deserializable) controls all noise parameters:
- `base_freq` — fundamental frequency (default 0.01)
- `octaves` — FBM octave count (3/6/8 for low/medium/high detail)
- `ridged_weight` — blend weight for ridged multi-fractal (mountain ridges)
- `warp_strength` — domain warping amplitude (terrain distortion)
- `micro_detail_freq` / `micro_detail_amp` — fine-scale surface variation
- `continent_freq` / `continent_threshold` / `shelf_blend_width` / `ocean_floor_depth` — continent/ocean masking

### Key Methods

- `NoiseStack::new(config, seed)` — build from config
- `sample(x, z)` — full terrain height (FBM + ridged + selector + warp + micro-detail + continent mask)
- `sample_land(x, z)` — raw land noise before continent blending
- `continent_value(x, z)` — continent mask (>threshold = land)
- `continent_blend(x, z)` — smoothstep shelf transition factor
- `ocean_floor_noise(x, z)` — low-amplitude underwater terrain

### Usage

```rust
let config = NoiseConfig { base_freq: 0.008, octaves: 6, ridged_weight: 0.4, ..default() };
let stack = NoiseStack::new(&config, 42);
let height = stack.sample(x as f64, z as f64); // returns 0.0..1.0 range
```

## Scene Presets (`scene_presets.rs`)

8 named terrain presets with tuned `PlanetConfig` values:

| Preset | Key Features |
|--------|-------------|
| `valley_river` | Deep valley carving, moderate height |
| `spherical_planet` | Spherical terrain mode |
| `alpine` | High ridged mountains, dense caves, heavy erosion |
| `archipelago` | Continent masking for island chains |
| `desert_canyon` | High warp, deep canyons, sparse caves |
| `rolling_plains` | Low height scale, gentle terrain |
| `volcanic` | Very high peaks, lava-filled caves |
| `tundra_fjords` | Ridged coastlines, moderate erosion |

CLI: `--scene <name>` selects a preset. Additional flags (`--seed`, `--terrain-detail`, `--height-scale`, `--caves`, `--erosion`, `--hydraulic-erosion`) override preset defaults.

## Geological Depth (`terrain.rs`)

Standalone functions for underground material selection:

- `strata_material(depth, seed, x, z)` — depth-based rock layers:
  - 0–20 m: sedimentary (sandstone/limestone, noise-selected)
  - 20–60 m: metamorphic (stone)
  - 60 m+: igneous (granite/basalt, noise-selected)
- `ore_material(depth, seed, x, y, z)` — 3D Perlin ore veins at depth-specific ranges (coal 5–30 m, copper 15–50 m, iron 30–80 m, gold 50 m+)
- `is_crystal_deposit(x, y, z, seed)` — quartz crystal deposits on deep cave walls
- `is_multi_scale_cave(x, y, z, seed, threshold)` — 3-layer OR-combined caves (caverns freq=0.01, tunnels freq=0.04, tubes freq=0.025)
- `cave_fill_material(y, sea_level, depth)` — AIR normally, WATER near sea level, LAVA at depth >80 m

## Biome-Terrain Integration (`biome_map.rs`)

Deterministic environment and surface material system (no ECS dependencies).

### EnvironmentMap

- `EnvironmentMap::new(seed)` — builds temperature and moisture noise fields
- `temperature(x, z)` — returns 0.0..1.0 (cold..hot)
- `moisture(x, z)` — returns 0.0..1.0 (dry..wet)

### Surface Material Selection

`surface_material(env_map, x, z, altitude, slope, sea_level)` applies rules in priority order:
1. Slope > 1.5 → Stone (cliffs)
2. Slope > 0.8 → Dirt (steep hillsides)
3. Altitude > 60 + cold → Ice (high peaks)
4. Altitude > 45 → Dirt/Stone (alpine zone)
5. Low altitude near sea level → Sand (coastal)
6. Arid + hot → Sand (desert)
7. Cold → Dirt (tundra)
8. Default → Grass

### Helpers

- `compute_slope(heightmap, x, z, size)` — central-difference gradient magnitude
- `adjusted_soil_depth(base, slope)` — valleys thicker, ridges thinner

## Hydraulic Erosion (`erosion.rs`)

Two erosion systems coexist:

### D8 Valley Carving (original)
- `fill_sinks()` → `compute_d8_directions()` → `compute_flow_accumulation()` → `carve_valley()`
- Depth ∝ ln(flow), width ∝ √flow

### Hydraulic Erosion (new)
- `HydraulicErosionConfig` — droplet_count, grid_iterations, max_lifetime, inertia, capacity, deposition, erosion, evaporation, min_slope
- `HydraulicMode` — Droplet / Grid / Combined
- `droplet_erode(heightmap, config, seed)` — Beyer-variant droplet simulation
- `grid_erode(heightmap, config)` — grid-based water/sediment flow
- `repose_slippage(heightmap, angle)` — 3-pass D8 angle-of-repose correction
- `hydraulic_erode(heightmap, config, seed)` — combined dispatcher (grid → repose → droplet)

## Ray-Cast Module (`raycast.rs`)

Reusable discrete 3D grid ray march. Currently used by `chemistry/heat.rs` for
radiative heat transfer visibility checks; designed to be extended for Phase 11 Optics.

### API

- `RayHit { index: usize, distance: f32 }` — hit result (flat array index + Euclidean distance)
- `RAY_DIRECTIONS: [[i32; 3]; 26]` — 6 cardinal + 12 edge + 8 corner direction vectors
- `STEP_DISTANCES: [f32; 26]` — precomputed step lengths (1.0, √2, √3)
- `march_grid_ray(start, dir_index, size, is_opaque) → Option<RayHit>` — march from `start` in direction `dir_index` (0..25) through a `size³` grid. Steps one voxel per iteration. Returns the first opaque hit, or `None` if the ray exits the grid.
- `is_surface_voxel(index, size, is_air_fn) → bool` — returns `true` if the voxel has at least one air-adjacent face (6 cardinal neighbors). Grid boundaries count as exposed.

### Design

- Integer-step marching (not DDA): each step moves exactly one voxel along the direction vector. Simple, cache-friendly, and correct for a uniform grid.
- `is_opaque` and `is_air_fn` are closures for decoupling from specific data structures.
- Max ray distance is controlled by the caller (e.g. `max_ray_steps` in `radiate_chunk`).

## Patterns

- `Chunk::get(x, y, z)` / `Chunk::get_mut(x, y, z)` — local coordinates only (0..CHUNK_SIZE)
- `ChunkCoord::from_voxel_pos()` — converts world voxel position to chunk coordinate
- `ChunkCoord::world_origin()` — minimum corner of chunk in world voxel space
- Dirty flag: `get_mut` automatically marks the chunk dirty for re-meshing

## Gotchas

- Voxel is 16 bytes (MaterialId:2 + temperature:f32 + pressure:f32 + damage:f32 + 2 padding). Keep it compact — no strings or heap allocations.
- `is_solid()` returns true for everything except air. Water is "not air" but also not structural (see physics/integrity).
- Surface Nets mesh algorithm is retained for internal use (density-field isosurface). The V2 pipeline uses greedy meshing for output. `meshing.rs` algorithms are called via `chunk_mesh_to_bevy_mesh`.
- Pressure is in **Pascals** (101325 Pa = 1 atm). Do NOT use atmospheres.
- Temperature is in **Kelvin** (288.15 K = 15 °C). Do NOT use Celsius or Fahrenheit.

## Meshing

`meshing.rs` provides the Surface Nets algorithm library and the
`chunk_mesh_to_bevy_mesh` function used by the V2 pipeline to convert
`V2ChunkData` into Bevy `Mesh` assets.  The old V1 dispatch systems
(`dispatch_mesh_tasks`, `collect_mesh_results`) have been removed.

### Functions

| Function | Input | Purpose |
|----------|-------|---------|
| `generate_mesh(chunk)` | `&Chunk` | Flat-array Surface Nets meshing at full 32³ resolution |
| `generate_mesh_lod(chunk, lod_step)` | `&Chunk, usize` | Reduced-resolution meshing (step=2 → 16³, step=4 → 8³) |
| `chunk_mesh_to_bevy_mesh(data)` | `&V2ChunkData` | **Primary V2 entry point** — produces Bevy `Mesh` from cubed-sphere chunk |

All Surface Nets paths delegate to `generate_mesh_generic()` — a
closure-parameterised implementation; the sampling function determines
where voxel data comes from.

## Octree (SVO) Subsystem

Sparse Voxel Octree for adaptive-resolution voxel storage. Used for terrain LOD, item voxel models, and creature damage maps.

### Files

| File | Purpose |
|------|---------|
| `octree.rs` | Generic `OctreeNode<T>` — Leaf/Branch enum with get/set/subdivide/collapse |
| `voxel_access.rs` | `VoxelAccess` trait + adapters: `FlatVoxelStorage`, `OctreeVoxelStorage`, flat↔octree conversion |

### OctreeNode<T>

- **Leaf(T):** Uniform region, single value for the whole subtree.
- **Branch(Box<[OctreeNode<T>; 8]>):** Subdivided into 8 children (2×2×2, Morton/Z-order indexed).
- **Child index:** `(z_bit << 2) | (y_bit << 1) | x_bit` — bit indicates high (1) or low (0) half.
- **Collapse:** If all 8 children are identical leaves, merge back to Leaf. Use `try_collapse()` after mutations or `collapse_recursive()` for full-tree optimization.
- **Depth:** Each level halves the cell size. At base resolution (CHUNK_SIZE=32), max 5 levels gives ~1m down to ~1/16m cells.
- `T` must be `Clone + PartialEq`. For voxels, use `Voxel` from `voxel.rs`.

### VoxelAccess Trait

Uniform read/write interface so physics/simulation systems work with both flat arrays and octrees:
- `get_voxel(x, y, z)` / `set_voxel(x, y, z, voxel)` — base resolution
- `get_voxel_at_depth(x, y, z, depth)` / `set_voxel_at_depth(...)` — sub-voxel resolution
- `depth_at(x, y, z)` — query current subdivision level
- `size()` — edge length at base resolution

### Adapters

| Adapter | Storage | Mutable | Use case |
|---------|---------|---------|----------|
| `FlatVoxelStorage` | `&[Voxel]` | No | Read-only backward compat for physics |
| `FlatVoxelStorageMut` | `&mut [Voxel]` | Yes | Mutable backward compat |
| `OctreeVoxelStorage` | `&OctreeNode<Voxel>` | No | Read-only octree access |
| `OctreeVoxelStorageMut` | `&mut OctreeNode<Voxel>` | Yes | Mutable octree access |

### Conversion Functions

- `flat_to_octree(voxels, size)` — Builds an SVO from a flat array, collapsing uniform regions.
- `octree_to_flat(root, size)` — Expands an SVO back to a flat array at base resolution.

### Gotchas

- `set()` takes a `target_size` parameter: use 1 for per-cell writes, larger values to set whole quadrants.
- `try_collapse()` is called automatically after `set()`, but only at the immediate parent. Use `collapse_recursive()` for deep optimization passes.
- `for_each_leaf()` / `for_each_leaf_mut()` iterate leaves with position and size — use for meshing, simulation sweeps.
- The flat↔octree roundtrip is lossless at base resolution. Sub-voxel data is lost when converting to flat.

## Adaptive Refinement

Analyzes voxel data to determine where octree subdivision provides meaningful detail. Controlled by `assets/data/subdivision_config.ron`.

### Files

| File | Purpose |
|------|---------|
| `refinement.rs` | `SubdivisionConfig`, `RefinementAnalysis`, `analyze_chunk()`, `build_refined_octree()`, `compression_stats()` |
| `lod.rs` | `LodSummary`, `LodLevel`, `LodConfig`, `chunk_lod_level()`, `summarize_voxels()`, `summarize_octree()` |
| `interpolation.rs` | Trilinear interpolation, LOD blend, upsample, nearest-neighbor material lookup |

### Refinement Criteria

| Reason | Description | Config Flag |
|--------|-------------|-------------|
| `SurfaceCrossing` | Solid ↔ air boundary | `refine_surfaces` |
| `MaterialBoundary` | Different solids meeting | `refine_material_boundaries` |
| `ThermalGradient` | Temperature delta > threshold | `thermal_gradient_threshold` (K) |
| `PressureGradient` | Pressure delta > threshold | `pressure_gradient_threshold` (Pa) |
| `DamageGradient` | Damage delta > threshold | `refine_damage_gradients` + `damage_gradient_threshold` |

### Usage

```rust
let config = SubdivisionConfig::default();
let analysis = analyze_chunk(&chunk, &config);
let octree = build_refined_octree(chunk.voxels(), CHUNK_SIZE, &analysis);
let stats = compression_stats(&chunk);
```

### Compression Stats

`compression_stats()` returns flat vs octree size, leaf/node counts, max depth, and compression ratio. Use for profiling and memory budgeting.

## LOD System

Macro-level octree for chunk-space level-of-detail.

### Types

| Type | Purpose |
|------|---------|
| `LodLevel(u8)` | LOD depth: L0 = 32m (single chunk), L1 = 64m, L2 = 128m, etc. |
| `LodSummary` | Aggregate data for a LOD region: dominant material, solidity ratio, average temperature/pressure |
| `LodConfig` | Camera-distance thresholds and LOD level mapping |

### Functions

| Function | Purpose |
|----------|---------|
| `chunk_lod_level(config, distance)` | Camera distance → LOD level |
| `summarize_voxels(voxels, size)` | Flat array → LodSummary |
| `summarize_octree(tree, size)` | OctreeNode → LodSummary |

### Integration

Use `chunk_lod_level()` to determine how to mesh/render each chunk:
- LOD 0: full 32³ mesh via `generate_mesh()`
- LOD 1+: reduced mesh via `generate_mesh_lod(chunk, 2^lod_level)`

## Interpolation

Smooth transitions between LOD levels and fractional-coordinate sampling.

### Functions

| Function | Purpose |
|----------|---------|
| `trilinear_sample(fx, fy, fz, size, sample_fn)` | Generic trilinear interpolation with custom sampler |
| `interpolate_temperature(voxels, size, fx, fy, fz)` | Temperature at fractional coords (flat) |
| `interpolate_pressure(voxels, size, fx, fy, fz)` | Pressure at fractional coords (flat) |
| `interpolate_solidity(voxels, size, fx, fy, fz)` | Solid/air field at fractional coords |
| `interpolate_temperature_octree(tree, size, fx, fy, fz)` | Temperature at fractional coords (octree) |
| `interpolate_pressure_octree(tree, size, fx, fy, fz)` | Pressure at fractional coords (octree) |
| `lod_blend_factor(distance, near, far)` | Hermite smoothstep blend factor (0→1) |
| `blend_scalar(near, far, factor)` | Linear blend of two values |
| `nearest_material(voxels, size, fx, fy, fz)` | Nearest-neighbor material ID (categorical, not interpolated) |
| `upsample_voxels(coarse, coarse_size, fine_size)` | Upsample grid with trilinear scalars + nearest-neighbor materials |

### Notes

- Materials are categorical — use `nearest_material()`, NOT trilinear interpolation.
- `lod_blend_factor()` uses Hermite smoothstep (3t² - 2t³) for visually smooth transitions.
- Trilinear sampling clamps out-of-bounds coordinates to grid edges.
