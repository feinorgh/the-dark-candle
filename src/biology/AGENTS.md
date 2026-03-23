# Biology Module

Metabolism, health, growth/aging, death/decomposition, and plant growth.

## Files

| File | Purpose |
|------|---------|
| `metabolism.rs` | Energy reserves, Kleiber's law basal rate, starvation |
| `health.rs` | Damage types, status effects, natural healing |
| `growth.rs` | Juvenile → Adult → Elder lifecycle |
| `decay.rs` | Corpse placement, decomposition, nutrient release |
| `plants.rs` | Grass spread to dirt with light/water checks |

## Constants

- `ORGANIC_MATTER = MaterialId(12)` — defined in `decay.rs`, used for corpse voxels

### Body Size & Metabolism (SI Units — Kleiber's Law)

Metabolic rates follow **Kleiber's law**: `P = 70 × m^0.75` (kcal/day), converted to Watts.
Source: Wikipedia — Kleiber's law.

| Size | Approx Mass (kg) | Basal Rate (W) | Max Energy (J, ~24h reserve) |
|------|------------------|----------------|------------------------------|
| Tiny | 0.5 | ~2.0 | ~172,800 |
| Small | 5.0 | ~11.3 | ~976,320 |
| Medium | 40.0 | ~54.0 | ~4,665,600 |
| Large | 200.0 | ~161.0 | ~13,910,400 |
| Huge | 1000.0 | ~603.0 | ~52,099,200 |

- `tick_metabolism(&mut Metabolism, dt: f32) -> f32` — takes `dt` in seconds
- `movement_energy_cost(mass_kg, distance_m) -> f32` — ~10 J/(kg·m) cost of transport
- `kleiber_basal_rate_watts(mass_kg) -> f32` — pure function for Kleiber's law

### Growth Stages (growth.rs)

| Stage | Lifespan Range | Effect |
|-------|---------------|--------|
| Juvenile | 0–20% | Size grows from 0.5 → 1.0 |
| Adult | 20–75% | Full stats (multiplier 1.0) |
| Elder | 75–100% | Stats decline up to -30% |

## Dependencies

- **Imports from:** `crate::data::BodySize`, `crate::world::voxel::{MaterialId, Voxel}`
- **Imported by:** (standalone; behavior module reads health/metabolism state)

## Patterns

- All tick functions are pure: `tick_metabolism(&mut Metabolism, dt) -> f32` (returns starvation damage).
- Health healing is gated by energy: `energy_fraction >= heal_threshold` to regenerate.
- Status effects are a `Vec<StatusEffect>` processed via `swap_remove` for O(1) expiry.
- Corpse placement uses `place_corpse()` which writes `ORGANIC_MATTER` voxels in a cluster, respecting bounds and only overwriting air.

## Gotchas

- At exactly 75% lifespan, the elder stage begins but `elder_frac = 0.0`, so stat_multiplier is still 1.0. Tests that assert `stat_multiplier < 1.0` need to advance past 75%.
- `tick_health()` processes status effects *before* natural healing — a creature can die from DoT even if healing conditions are met.
- Plant growth uses hardcoded `MaterialId(7)` for grass and `MaterialId(2)` for dirt. If these IDs change in `voxel.rs`, update `plants.rs` too.
- Energy is in Joules, not calories. All metabolism inputs/outputs use SI units.
