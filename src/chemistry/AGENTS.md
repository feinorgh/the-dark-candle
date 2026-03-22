# Chemistry Module

Heat transfer, chemical reactions, material state transitions, and fire propagation.

## Files

| File | Purpose |
|------|---------|
| `heat.rs` | Temperature diffusion between neighboring voxels |
| `reactions.rs` | `ReactionData` RON struct, condition matching |
| `state_transitions.rs` | `MaterialRegistry`, phase changes (melt/boil/freeze/condense) |
| `fire_propagation.rs` | Integration: heat + reactions + transitions per tick |

## Dependencies

- **Imports from:** `crate::data::{MaterialData, Phase}`, `crate::world::voxel::{MaterialId, Voxel}`
- **Imported by:** (fire_propagation integrates heat + reactions + transitions)

## Key Types

- `MaterialRegistry` — `HashMap<u16, MaterialData>` for runtime material property lookups
- `ReactionData` — loaded from `.reaction.ron` files: reactant, catalyst, product, temperature range
- `TransitionResult` — what a voxel becomes when heated/cooled past a threshold

## Data Files

- `assets/data/materials/*.material.ron` — physical/chemical properties per material
- `assets/data/reactions/*.reaction.ron` — chemical reaction definitions

## Patterns

- `conductivity(material_id)` returns a thermal conductivity constant per material.
- `diffuse_chunk()` applies one tick of heat diffusion across a flat voxel array.
- `check_reaction()` tests if a voxel meets reaction conditions (material, temperature, neighbor).
- `check_transition()` tests if a voxel should change phase based on temperature.

## Fire Propagation Tuning

The fire chain reaction depends on heat equilibrium exceeding the ignition point:

```
equilibrium ≈ (heat_output + N × ambient) / (N + 1)
```

- For a 2D floor with ~5 neighbors: `heat_output = 3000` works; `1500` fails.
- `ignition_point` for wood is 573 K; ambient is 293 K.
- If equilibrium < ignition_point, fire dies out instead of spreading.

## Gotchas

- RON `[f32; 3]` arrays must use tuple syntax: `(0.5, 0.5, 0.5)`, not `[0.5, 0.5, 0.5]`.
- `melted_into`, `boiled_into`, `frozen_into`, `condensed_into` fields use `#[serde(default)]` — they can be omitted from RON files for materials that don't transition.
