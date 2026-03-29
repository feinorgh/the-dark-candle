# Procgen Module

Procedural generation: creature instances, item instances, biome definitions, chunk-based spawning, and tree structures.

## Files

| File | Purpose |
|------|---------|
| `creatures.rs` | `Creature` component, `SimpleRng`, stat/color variation from templates |
| `items.rs` | `Item` component, material-based property scaling |
| `biomes.rs` | `BiomeData` RON struct, biome matching, weighted spawn selection |
| `spawning.rs` | Deterministic per-chunk spawn planning |
| `tree.rs` | L-system procedural tree generator with multiresolution voxel output |

## Dependencies

- **Imports from:** `crate::data::{BodySize, CreatureData, Diet, ItemCategory, ItemData, MaterialData, Phase}`
- **Internal:** `spawning.rs` uses `crate::procgen::biomes::*` and `crate::procgen::creatures::SimpleRng`
- **Imported by:** (standalone; game loop calls spawning functions)

## Data Files

- `assets/data/creatures/*.creature.ron` — species templates (wolf, deer, rabbit, cave_spider)
- `assets/data/items/*.item.ron` — item templates (sword, pickaxe, apple, wooden_shield)
- `assets/data/biomes/*.biome.ron` — biome definitions (forest, meadow, cave, tundra)
- `assets/data/trees/*.tree.ron` — tree species configs (oak)

## Key Design Decisions

### SimpleRng (xorshift64)

Deterministic PRNG that avoids external crate dependencies. Seeded per-creature or per-chunk for reproducible generation.

```rust
pub struct SimpleRng { state: u64 }
// next_f32() → [0.0, 1.0)
// next_signed() → [-1.0, 1.0)
```

### Stat Variation

`generate_creature()` applies `±stat_variation` as a fraction of base stats. A wolf with `base_health: 80, stat_variation: 0.15` produces health in [68, 92].

### Material-Based Item Properties

`generate_item()` scales weight by `material_density / reference_density` and durability by material hardness. An iron sword is heavier and more durable than a wood sword, emergently.

### Chunk Spawning

`plan_chunk_spawns()` is deterministic per `(chunk_x, chunk_z, world_seed)`:
1. Hash chunk coords into a local RNG seed.
2. Select biome by height/temperature/moisture.
3. Roll spawn table entries with weighted selection.
4. Place creatures at deterministic positions within the chunk.

## Gotchas

- `weighted_select()` and `select_biome()` must not have explicit lifetime annotations — clippy flags them as `needless_lifetimes`.
- `BiomeData` spawn tables use `weight: u32` for integer-based weighted selection (not floats).
- When adding new biomes: create the `.biome.ron` file AND update the integration test in `tests/validate_assets.rs`.

## Procedural Trees

`tree.rs` generates L-system trees with multiresolution voxel output.

### Key Types

- `TreeConfig` — loaded from `.tree.ron`, defines species params: trunk height/radius, branching angles, taper factor, branch probability, subdivision depth, material assignments
- `TreeTemplate` — pre-computed voxel layout stamped into chunks. Contains `Vec<(IVec3, MaterialId)>` positioned relative to trunk base
- `TreeRegistry` — Bevy resource holding loaded `TreeTemplate` instances, provides `stamp_tree_into_chunk()` for chunk integration
- `plant_trees` — ECS system that places trees on terrain surface based on biome rules

### Tree Structure

Trees are built from trunk base upward with recursive branching:
- **Trunk**: Bark shell + Wood core at base resolution
- **Primary branches**: fork from trunk, taper outward
- **Secondary/tertiary branches**: progressively thinner
- **Twigs + leaf clusters**: DryLeaves material at branch tips — the fire starters

### Materials Used

Trees use 5 material types: Wood (core), Bark (surface), Twig (small branches), DryLeaves (leaf clusters and branch tips), Charcoal (produced by pyrolysis during fire).

### Integration

`stamp_tree_into_chunk()` writes a `TreeTemplate` into a chunk's voxel array at a specified position, skipping out-of-bounds voxels. The `plant_trees` system runs during world generation and uses biome spawn rules to determine tree density and placement.
