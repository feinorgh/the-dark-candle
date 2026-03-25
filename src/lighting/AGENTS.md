# Lighting Module

Sun cycle, atmospheric scattering, per-voxel sunlight propagation, and light map integration.

## Files

| File | Purpose |
|------|---------|
| `mod.rs` | `LightingPlugin`: sun cycle, day/night config, `TimeOfDay`, `SolarInsolation`, `update_chunk_light_maps` system |
| `light_map.rs` | `ChunkLightMap` component: per-voxel RGB sunlight level, column-based propagation, mesh integration |
| `sky.rs` | Rayleigh scattering sky model: wavelength-dependent atmospheric color from sun angle |

## Dependencies

- **Imports from:** `crate::data::{MaterialData, MaterialRegistry}`, `crate::world::chunk::Chunk`, `crate::world::voxel::MaterialId`, `crate::world::meshing::ChunkMesh`, `crate::physics::constants`
- **Imported by:** `crate::world::meshing` (light map applied to vertex colors), `crate::diagnostics::visualization` (sky color for software renderer)

## Key Types

- `TimeOfDay` — `Resource(f32)`: current time in seconds since midnight (0–86400)
- `DayNightConfig` — `Resource`: sunrise/sunset hours, day/night illuminance, ambient floor
- `SolarInsolation` — `Resource(f32)`: current sun intensity (W/m², used for glow rendering)
- `ChunkLightMap` — `Component`: per-voxel `Vec<[u8; 3]>` storing RGB sunlight transmittance, indexed ZYX
- `SkyConfig` (sky.rs) — atmosphere parameters: scale height, earth radius, β_R coefficients

## Data Files

- `assets/data/materials/*.material.ron` — optical properties per material: `refractive_index`, `reflectivity`, `absorption_rgb`, `transparent`
- `assets/data/universal_constants.ron` — `SPEED_OF_LIGHT = 299_792_458.0 m/s`

## Sun Cycle

`TimeOfDay` advances by `dt` each frame. Sun elevation is computed as a sinusoidal curve clamped to horizon. `DayNightConfig` controls sunrise/sunset hours and light intensity range.

```
sun_elevation = π × (time - sunrise) / (sunset - sunrise)
illuminance = lerp(night_lux, day_lux, sin(elevation).max(0))
```

The `DirectionalLight` rotation and ambient light are updated each frame to match.

## Rayleigh Scattering (sky.rs)

Physically-based atmospheric sky color using wavelength-dependent 1/λ⁴ scattering.

### Physics

Scattering coefficient:

```
β_R(λ) = 8π³(n²−1)² / (3 N λ⁴)
```

Pre-computed RGB coefficients for sea-level atmosphere:
- Red (680nm): β = 5.8 × 10⁻⁶ m⁻¹
- Green (550nm): β = 13.5 × 10⁻⁶ m⁻¹
- Blue (440nm): β = 33.1 × 10⁻⁶ m⁻¹

Scale height H_R = 8500 m. Earth radius = 6 371 000 m.

### Ray Integration

Each sky pixel traces a view ray through the atmosphere with 16 sample points.
At each sample, 8 light-direction samples integrate optical depth to the sun.
Scattering contribution is accumulated and tone-mapped (Reinhard + sRGB gamma).

### API

- `sky_color(view_dir: [f32; 3], sun_dir: [f32; 3]) -> [f32; 3]` — main entry, returns sRGB [0–1]
- `sky_color_from_angles(azimuth, elevation, sun_azimuth, sun_elevation) -> [f32; 3]` — angle-based variant
- `tonemap_to_srgb(hdr: [f32; 3]) -> [f32; 3]` — Reinhard + gamma, exposed for external use

## Per-Voxel Sunlight (light_map.rs)

Column-based top-down propagation with Beer-Lambert absorption for transparent materials.

### Algorithm

For each (x, z) column in the chunk:
1. Start at top with full sunlight `[255, 255, 255]`
2. March downward (y = size−1 → 0)
3. Air voxels: pass through unchanged
4. Opaque voxels: set to `[0, 0, 0]`, block all light below
5. Transparent voxels: attenuate per channel via Beer-Lambert: `I = I₀ × e^(−α × 1.0)`
6. Store RGB level per voxel

### ChunkLightMap

- Stored as separate `Component` (not in `Voxel` struct) to preserve cache efficiency (Voxel = 20 bytes)
- Indexed as `z * size² + y * size + x` (ZYX, matches chunk convention)
- `get_clamped(x, y, z)` returns `[1.0, 1.0, 1.0]` for out-of-bounds (open sky assumption)
- `apply_light_map(&light_map, &mut ChunkMesh)` post-processes mesh vertex colors

### API

- `ChunkLightMap::new(size) -> Self` — fully lit default
- `propagate_sunlight(&mut self, voxels, size, absorption_fn)` — core propagation
- `propagate_sunlight_from_registry(&mut self, voxels, size, registry)` — convenience wrapper using `MaterialRegistry`
- `apply_light_map(light_map, mesh)` — modulate vertex colors by light level
- `update_chunk_light_maps` — Bevy system, runs on `Changed<Chunk>`, attached to `LightingPlugin`

## Beer-Lambert Law

Light intensity decays exponentially through absorbing media:

```
I(d) = I₀ × e^(−α × d)
```

where α is the absorption coefficient (m⁻¹) per RGB channel and d is path length (1 voxel = 1 m).

`MaterialData::light_absorption_rgb()` provides the per-channel coefficients with fallback chain:
1. `absorption_rgb` field if present → `[r, g, b]`
2. `absorption_coefficient` scalar if present → `[α, α, α]`
3. `transparent == true` with neither → `[0, 0, 0]` (fully transparent)
4. Otherwise → `None` (opaque)

### Material Optical Values (SI)

| Material | n | α_R (m⁻¹) | α_G (m⁻¹) | α_B (m⁻¹) | Visual Effect |
|----------|---|------------|------------|------------|---------------|
| Water | 1.33 | 0.45 | 0.07 | 0.02 | Blue tint (absorbs red) |
| Ice | 1.31 | 0.35 | 0.06 | 0.02 | Slight blue tint |
| Glass | 1.52 | 0.05 | 0.03 | 0.04 | Nearly neutral, slight green |
| Iron | — | — | — | — | Opaque, reflectivity = 0.65 |
| Air | 1.0003 | 0 | 0 | 0 | Fully transparent |

## Patterns

- `ChunkLightMap` is a separate component, not embedded in `Voxel`, to keep voxel data compact for cache performance.
- Light maps are rebuilt whenever chunk voxels change (`Changed<Chunk>` query filter).
- The visualization renderer (`src/diagnostics/visualization.rs`) uses `sky_color()` and `dda_march_ray_attenuated()` from the shared raycast module — same physics as the in-game pipeline.

## Gotchas

- RON `[f32; 3]` arrays use tuple syntax: `(0.45, 0.07, 0.02)`, not `[0.45, 0.07, 0.02]`.
- Chunk voxel indexing is `z * size² + y * size + x` (ZYX). Visualization previously used XZY — this was fixed in commit `db0a3f8`. All modules are now consistent.
- `get_clamped()` returns fully lit for out-of-bounds coordinates. This means chunk edges assume open sky, which is correct for top-exposed chunks but may need inter-chunk propagation later.
- `propagate_sunlight` only handles top-down columns. Lateral light spread (e.g. light entering a cave mouth) requires a future flood-fill pass.
- Shadow ray origin is offset 0.5 in the to_light direction to avoid self-intersection.
