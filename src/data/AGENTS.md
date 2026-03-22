# Data Module

Central data structs loaded from RON files via `bevy_common_assets`. This is the foundational leaf module — it has no internal `crate::` dependencies.

## Files

| File | Purpose |
|------|---------|
| `mod.rs` | All data structs, enums, `DataPlugin` with RON loaders |

## Structs and Their RON Suffixes

| Struct | RON Suffix | Directory |
|--------|-----------|-----------|
| `EnemyData` | `.enemy.ron` | `assets/data/` |
| `MaterialData` | `.material.ron` | `assets/data/materials/` |
| `CreatureData` | `.creature.ron` | `assets/data/creatures/` |
| `ItemData` | `.item.ron` | `assets/data/items/` |

## Enums

- `Phase` — Solid, Liquid, Gas
- `Diet` — Herbivore, Carnivore, Omnivore, Scavenger
- `BodySize` — Tiny, Small, Medium, Large, Huge
- `ItemCategory` — Tool, Weapon, Armor, Food, Material, Container, Misc

## Dependencies

- **Imports from:** none (leaf module)
- **Imported by:** chemistry, biology, procgen, entities

## Adding New Data Types

1. Define the struct here with `#[derive(Deserialize, Asset, TypePath, Debug, Clone, PartialEq)]`.
2. Register the loader in `DataPlugin::build()`: `app.add_plugins(RonAssetPlugin::<YourStruct>::new(&["suffix.ron"]))`.
3. Create RON files in `assets/data/`.
4. Add deserialization tests in this module.
5. Add a glob-based validation test in `tests/validate_assets.rs`.

## RON Format Rules

- Fixed-size arrays `[f32; 3]` use tuple syntax: `(0.5, 0.5, 0.5)`.
- Optional fields use `#[serde(default)]` so they can be omitted from RON files.
- Default values use `#[serde(default = "function_name")]` with a standalone function.
