# Entities Module

Entity spawning from data assets plus the `Inventory` component for per-entity item storage.

## Files

| File | Purpose |
|------|---------|
| `mod.rs` | `Enemy` component, `load_entity_assets()`, `spawn_enemy_when_loaded()`, `pub mod inventory` |
| `inventory.rs` | `Inventory` component — item stacks keyed by name, weight (kg) and volume (m³) limits, `add_item`/`remove_item`/`has`/`count` API |

## Inventory API

```rust
let mut inv = Inventory::new(50.0 /*kg*/, 0.5 /*m³*/);
inv.add_item("stone", 10, 2.5 /*kg/unit*/, 0.001 /*m³/unit*/);
inv.count("stone");          // 10
inv.has("stone", 5);         // true
inv.remove_item("stone", 3, 2.5, 0.001);
```

All weights in **kg**, volumes in **m³**.

## Dependencies

- **Imports from:** `crate::data::{EnemyData, GameAssets}`
- **Imported by:** `crate::building::placement` (reads `Inventory` for placement validation)

## Notes

- `Enemy` predates the procgen creature system. New creature types should use `CreatureData` + `procgen::creatures::generate_creature()` instead.
- `Inventory` is designed to be added to any entity (player, chest, creature). Weight / volume limits of `0.0` mean unlimited.
