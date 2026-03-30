# Planet Module

Geodesic planetary generation pipeline: grid construction, tectonic simulation,
impact events, celestial mechanics, biome/climate classification, geological
layering, interactive 3D globe rendering, and 2D map projection export.

## Files

| File | Purpose |
|------|---------|
| `mod.rs` | Hub: submodule declarations, core enums (BiomeType, RockType, CrustType, BoundaryType), PlanetData, PlanetConfig |
| `grid.rs` | Icosahedral geodesic grid (subdivision, CellId, cell_position, cell_neighbors, cell_lat_lon, nearest_cell_from_pos) |
| `tectonics.rs` | Plate tectonic simulation (Voronoi seeding, boundary detection, convergent/divergent/transform, orogenesis, erosion) |
| `impacts.rs` | Astronomical impacts (crater stamping, ejecta, crust thinning, giant impacts) |
| `celestial.rs` | Celestial system (Star, Moon, Ring generation, Keplerian orbits, tidal forces) |
| `biomes.rs` | Climate classification (temperature from latitude/elevation, precipitation, 14 BiomeType variants) |
| `geology.rs` | Geological layering (RockType assignment by depth/context, ore deposit placement, 7 ore types) |
| `render.rs` | Interactive 3D globe (Bevy app, polygon fan mesh, orbital camera, 8 ColourMode variants, screenshot) |
| `projections.rs` | 2D map projections (equirectangular, Mollweide, orthographic), SpatialIndex, animation export |

## Key Types

- **`PlanetData`** — Central struct holding all generation output (grid, elevation,
  plate IDs, biomes, geology, celestial system). Passed by `&mut` through the
  pipeline, consumed by the globe viewer.
- **`PlanetConfig`** — Generation parameters (seed, grid_level, radius_m, mass_kg,
  tectonic_steps, etc.).
- **`IcosahedralGrid`** — The geodesic grid. Level n = 10×4^n + 2 cells. Exactly
  12 pentagons at all levels. Y-up coordinate convention.
- **`CellId(u32)`** — Opaque cell identifier. Use `.index()` for array indexing.
- **`ColourMode`** — Enum for visualization (Elevation, Biome, Plates, Temperature,
  GeologicalAge, CrustDepth, TidalAmplitude, Rock). Used by both render.rs and
  projections.rs.

## Pipeline Order

```rust
let mut data = PlanetData::new(config);  // grid + celestial
run_tectonics(&mut data, |_| {});         // plates + elevation
run_impacts(&mut data);                    // crater events
run_biomes(&mut data);                     // climate + biomes (before geology!)
run_geology(&mut data);                    // rock types + ores
```

`run_biomes` must precede `run_geology` because coal deposits require biome data.

## Grid Conventions

- **Y-up:** `pos.y = sin(latitude)`, XZ = equatorial plane
- **Neighbors:** `cell_neighbors()` returns `&[u32]` in CCW order
- **Pentagons:** 12 cells have 5 neighbors; all others have 6
- **Cell areas:** Nearly uniform. Slight variation at pentagons.

## CLI (worldgen binary)

```bash
cargo run --bin worldgen -- --seed 42 --level 4 --stats
cargo run --bin worldgen -- --seed 42 --level 4 --globe
cargo run --bin worldgen -- --seed 42 --level 4 --projection mollweide --colourmode biome --output world.png
cargo run --bin worldgen -- --seed 42 --level 4 --animate rotation.mp4 --width 512
```

## Bevy 0.18 Notes (render.rs)

- Spawn `Camera3d::default()` as component, no bundles
- Mesh entities: `Mesh3d(handle)` + `MeshMaterial3d(material)` as separate components
- Mouse input: `Res<AccumulatedMouseMotion>` / `Res<AccumulatedMouseScroll>`, NOT EventReader
- AmbientLight is a Component (spawn it), NOT a Resource
- `Query::single_mut()` returns `Result` — use `let Ok(..) = q.single_mut() else { return; };`
