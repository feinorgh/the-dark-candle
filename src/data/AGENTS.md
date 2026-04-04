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

- `Phase` — Solid, Liquid, Gas (derives `Default`, default = `Solid`)
- `Diet` — Herbivore, Carnivore, Omnivore, Scavenger
- `BodySize` — Tiny, Small, Medium, Large, Huge
- `ItemCategory` — Tool, Weapon, Armor, Food, Material, Container, Misc

## ⚠️ MaterialData SI Unit Fields

All material properties use **real-world SI units**. When adding or editing materials, source values from Wikipedia or NIST. `MaterialData` derives `Default` — use `..Default::default()` in test struct literals.

| Field | Unit | Example (Iron) | Source |
|---|---|---|---|
| `density` | kg/m³ | 7874.0 | Wikipedia: Density of iron |
| `melting_point` | K | 1811.0 | Wikipedia: Iron § Properties |
| `boiling_point` | K | 3134.0 | Wikipedia: Iron § Properties |
| `ignition_point` | K | None (non-flammable) | — |
| `hardness` | Mohs (0–10) | 4.5 | Wikipedia: Mohs scale |
| `thermal_conductivity` | W/(m·K) | 80.2 | Wikipedia: List of thermal conductivities |
| `specific_heat_capacity` | J/(kg·K) | 449.0 | Wikipedia: Table of specific heat capacities |
| `latent_heat_fusion` | J/kg | 247000.0 | Wikipedia: Enthalpy of fusion |
| `latent_heat_vaporization` | J/kg | 6088000.0 | Wikipedia: Enthalpy of vaporization |
| `emissivity` | dimensionless (0–1) | 0.21 | Engineering tables |
| `viscosity` | Pa·s | None (solid) | Wikipedia: Viscosity |
| `friction_coefficient` | dimensionless (0–1) | 0.5 | Wikipedia: Friction |
| `restitution` | dimensionless (0–1) | 0.2 | — |
| `youngs_modulus` | Pa | 200e9 | Wikipedia: Young's modulus |
| `tensile_strength` | Pa | 400_000_000.0 | Wikipedia: Ultimate tensile strength |
| `compressive_strength` | Pa | 250_000_000.0 | Wikipedia: Compressive strength |
| `shear_strength` | Pa | 170_000_000.0 | Wikipedia: Shear strength |
| `flexural_strength` | Pa | 350_000_000.0 | Wikipedia: Flexural strength |
| `fracture_toughness` | Pa·√m | 50_000_000.0 | Wikipedia: Fracture toughness |
| `heat_of_combustion` | J/kg | None (non-flammable) | Wikipedia: Heat of combustion |
| `molar_mass` | kg/mol | None (used for gases) | — |

> **Note on RON notation:** RON does not parse scientific notation (`400e6`).  
> Always write full decimal literals: `400_000_000.0` not `400e6`.

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
