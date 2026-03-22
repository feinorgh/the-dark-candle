# Entities Module

Minimal entity spawning from data assets. Currently only contains the legacy `Enemy` component.

## Files

| File | Purpose |
|------|---------|
| `mod.rs` | `Enemy` component, `load_entity_assets()`, `spawn_enemy_when_loaded()` |

## Dependencies

- **Imports from:** `crate::data::{EnemyData, GameAssets}`
- **Imported by:** (standalone)

## Notes

This module predates the procgen creature system. New creature types should use `CreatureData` + `procgen::creatures::generate_creature()` instead of adding to this module. The `Enemy` component may be refactored or removed as the procgen pipeline matures.
