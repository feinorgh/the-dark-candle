# Biology Module

Metabolism, health, growth/aging, death/decomposition, and plant growth.

## Files

| File | Purpose |
|------|---------|
| `metabolism.rs` | Energy reserves, basal rate, starvation |
| `health.rs` | Damage types, status effects, natural healing |
| `growth.rs` | Juvenile → Adult → Elder lifecycle |
| `decay.rs` | Corpse placement, decomposition, nutrient release |
| `plants.rs` | Grass spread to dirt with light/water checks |

## Constants

- `ORGANIC_MATTER = MaterialId(12)` — defined in `decay.rs`, used for corpse voxels

### Body Size Scaling (metabolism.rs)

| Size | Max Energy | Basal Rate |
|------|-----------|------------|
| Tiny | 100 | 0.5 |
| Small | 250 | 1.0 |
| Medium | 500 | 2.0 |
| Large | 800 | 3.0 |
| Huge | 1500 | 5.0 |

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

- All tick functions are pure: `tick_metabolism(&mut Metabolism) -> f32` (returns starvation damage).
- Health healing is gated by energy: `energy_fraction >= heal_threshold` to regenerate.
- Status effects are a `Vec<StatusEffect>` processed via `swap_remove` for O(1) expiry.
- Corpse placement uses `place_corpse()` which writes `ORGANIC_MATTER` voxels in a cluster, respecting bounds and only overwriting air.

## Gotchas

- At exactly 75% lifespan, the elder stage begins but `elder_frac = 0.0`, so stat_multiplier is still 1.0. Tests that assert `stat_multiplier < 1.0` need to advance past 75%.
- `tick_health()` processes status effects *before* natural healing — a creature can die from DoT even if healing conditions are met.
- Plant growth uses hardcoded `MaterialId(7)` for grass and `MaterialId(2)` for dirt. If these IDs change in `voxel.rs`, update `plants.rs` too.
