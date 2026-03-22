# Procgen Module

Procedural generation: creature instances, item instances, biome definitions, and chunk-based spawning.

## Files

| File | Purpose |
|------|---------|
| `creatures.rs` | `Creature` component, `SimpleRng`, stat/color variation from templates |
| `items.rs` | `Item` component, material-based property scaling |
| `biomes.rs` | `BiomeData` RON struct, biome matching, weighted spawn selection |
| `spawning.rs` | Deterministic per-chunk spawn planning |

## Dependencies

- **Imports from:** `crate::data::{BodySize, CreatureData, Diet, ItemCategory, ItemData, MaterialData, Phase}`
- **Internal:** `spawning.rs` uses `crate::procgen::biomes::*` and `crate::procgen::creatures::SimpleRng`
- **Imported by:** (standalone; game loop calls spawning functions)

## Data Files

- `assets/data/creatures/*.creature.ron` — species templates (wolf, deer, rabbit, cave_spider)
- `assets/data/items/*.item.ron` — item templates (sword, pickaxe, apple, wooden_shield)
- `assets/data/biomes/*.biome.ron` — biome definitions (forest, meadow, cave, tundra)

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
