# World Module

Voxel world infrastructure: types, chunk storage, terrain generation, Surface Nets meshing, and collision queries.

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

- Voxel is 12 bytes. Keep it compact — no strings or heap allocations.
- `is_solid()` returns true for everything except air. Water is "not air" but also not structural (see physics/integrity).
- Surface Nets meshing expects the chunk plus a 1-voxel border from neighbors for seamless edges.
