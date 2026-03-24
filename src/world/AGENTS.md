# World Module

Voxel world infrastructure: types, chunk storage, terrain generation, Surface Nets meshing, and collision queries.

## Spatial Mapping

**1 voxel = 1 meter.** This is the fundamental mapping that makes all SI units work without coordinate transforms. All physics values (density in kg/m³, force in N, energy in J) apply directly to voxel-sized volumes.

## Files

| File | Purpose |
|------|---------|
| `voxel.rs` | `MaterialId`, `Voxel` types |
| `chunk.rs` | `Chunk` (32³ flat array), `ChunkCoord` |
| `chunk_manager.rs` | `ChunkMap`, cylindrical chunk loading |
| `terrain.rs` | Layered Perlin noise terrain generation |
| `meshing.rs` | Surface Nets mesh extraction + per-material vertex coloring |
| `collision.rs` | Ground height queries for camera/entity gravity |

## Constants

- `CHUNK_SIZE = 32` — edge length in voxels (32³ = 32,768 per chunk)
- `CHUNK_VOLUME = 32,768`

### Voxel Defaults (SI Units)

| Field | Default | Unit | Notes |
|-------|---------|------|-------|
| `temperature` | 288.15 | K | Standard atmosphere (15 °C) |
| `pressure` | 101325.0 | Pa | Sea level atmospheric pressure |
| `damage` | 0.0 | — | 0.0 = intact, 1.0 = destroyed |

### MaterialId Registry

| ID | Name | Notes |
|----|------|-------|
| 0 | Air | Default, transparent |
| 1 | Stone | Structural, high hardness |
| 2 | Dirt | Terrain surface layer |
| 3 | Water | Fluid, swimmable |
| 4 | Iron | Dense, high melting point |
| 5 | Wood | Flammable |
| 6 | Sand | Granular |
| 7 | Grass | Spreads to dirt (biology) |
| 8 | Ice | Frozen water |
| 9 | Steam | Boiled water |
| 10 | Lava | Molten stone |
| 11 | Ash | Combustion product |
| 12 | Organic Matter | Corpse decomposition (biology) |

When adding new materials: define a `MaterialId` constant in `voxel.rs`, create a `.material.ron` file in `assets/data/materials/`, and update this table.

## Dependencies

- **Imported by:** physics, chemistry, biology, behavior, camera
- **Imports from:** `crate::camera::FpsCamera` (for chunk loading around player)

## Patterns

- `Chunk::get(x, y, z)` / `Chunk::get_mut(x, y, z)` — local coordinates only (0..CHUNK_SIZE)
- `ChunkCoord::from_voxel_pos()` — converts world voxel position to chunk coordinate
- `ChunkCoord::world_origin()` — minimum corner of chunk in world voxel space
- Dirty flag: `get_mut` automatically marks the chunk dirty for re-meshing

## Gotchas

- Voxel is 16 bytes (MaterialId:2 + temperature:f32 + pressure:f32 + damage:f32 + 2 padding). Keep it compact — no strings or heap allocations.
- `is_solid()` returns true for everything except air. Water is "not air" but also not structural (see physics/integrity).
- Surface Nets meshing expects the chunk plus a 1-voxel border from neighbors for seamless edges.
- Pressure is in **Pascals** (101325 Pa = 1 atm). Do NOT use atmospheres.
- Temperature is in **Kelvin** (288.15 K = 15 °C). Do NOT use Celsius or Fahrenheit.

## Meshing

Surface Nets mesh extraction from voxel data. Supports both flat chunks and octrees.

### Functions

| Function | Input | Purpose |
|----------|-------|---------|
| `generate_mesh(chunk)` | `&Chunk` | Standard flat-array meshing at full 32³ resolution |
| `generate_mesh_from_octree(tree, size)` | `&OctreeNode<Voxel>` | Mesh from octree, identical results to flat at base resolution |
| `generate_mesh_lod(chunk, lod_step)` | `&Chunk, usize` | Reduced-resolution meshing (step=2 → 16³, step=4 → 8³) |
| `generate_mesh_from_octree_lod(tree, size, lod_step)` | `&OctreeNode<Voxel>` | LOD meshing from octree |

All four delegate to `generate_mesh_generic()` — a closure-parameterized Surface Nets implementation. The sampling function determines where voxel data comes from.

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
