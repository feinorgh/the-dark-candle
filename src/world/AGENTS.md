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
