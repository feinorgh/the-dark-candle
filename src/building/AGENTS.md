# Building Module

Freeform structural construction — data-driven parts, load-path stress analysis,
crafting recipes, player placement, and demolition with debris.

## Files

| File | Purpose |
|------|---------|
| `mod.rs` | `BuildingPlugin` — registers asset loaders, resources, and systems |
| `parts.rs` | `PartData` RON asset, `PartShape` enum, `PlacedPart` component, attachment face flags |
| `joints.rs` | `Joint` component (axial + shear stress vs material strength), `JointType`, `cleanup_broken_joints` system |
| `stress.rs` | `StressTick` resource, `PartLoad`/`GroundAnchor` components, load-path analysis, LBM wind loading, progressive collapse |
| `crafting.rs` | `RecipeData` RON asset, `CraftingQueue` component, temperature-gated tick-based crafting |
| `placement.rs` | `BuildMode` resource, 1 m grid snap, build-mode input (B = toggle, R = rotate 90°), auto joint creation |
| `demolition.rs` | `PendingDemolition` component, debris fragment spawning, `DroppedPart` placeholder |

## RON Asset Files

| Suffix | Directory | Struct |
|--------|-----------|--------|
| `.part.ron` | `assets/data/parts/` | `PartData` |
| `.recipe.ron` | `assets/data/recipes/` | `RecipeData` |

Existing parts: `block`, `slab`, `beam`, `column`, `wall`, `arch`, `stair`, `roof`.  
Existing recipes: `wood_to_planks`, `clay_to_brick`, `sand_to_glass`, `iron_ore_to_ingot`, `mix_concrete`.

## Key API

### Placement
```rust
// Snap world position to 1 m grid:
let grid_pos = snap_to_grid(world_pos); // Vec3 → nearest (n.5, n.5, n.5)

// Check if valid to place here:
validate_placement(grid_pos, "stone", &placed_query, player_inventory)
// returns PlacementResult::{Valid, Occupied, Unsupported, NoMaterial}

// Spawn part + auto-create joints to neighbours:
spawn_part_at(&mut commands, grid_pos, rotation_step, "stone", "block", &placed_query);
```

### Demolition
```rust
// Request removal of a part entity:
commands.entity(part).insert(PendingDemolition { drop_as_item: true });
// process_demolitions() runs in FixedUpdate and handles the rest.
```

### Crafting
```rust
commands.spawn(CraftingQueue {
    recipe_name: "clay_to_brick".to_string(),
    progress_ticks: 0,
    ..default()
});
// tick_crafting() advances progress each FixedUpdate tick.
```

## ⚠️ Structural Strength Values (SI)

Strength fields on `MaterialData` are all in **Pascals (Pa)**. Source from Wikipedia or materials tables.

| Field | Typical range |
|---|---|
| `tensile_strength` | 0.5 MPa (thatch) → 400 MPa (iron) |
| `compressive_strength` | 0.5 MPa (thatch) → 250 MPa (iron) |
| `shear_strength` | 0.4 MPa (thatch) → 170 MPa (iron) |
| `flexural_strength` | 1.5 MPa (thatch) → 350 MPa (iron) |
| `fracture_toughness` | 0.02 MPa·√m (thatch) → 50 MPa·√m (iron) |

All are `Option<f32>` — leave as `None` for non-structural materials (air, water, steam).  
**RON notation:** write `400_000_000.0` not `400e6` (RON does not parse scientific notation).

## Construction Material IDs

New construction materials use IDs 100–107 to avoid conflicts with voxel
material IDs (0–27 are reserved for terrain/chemistry materials):

| ID | Name |
|----|------|
| 100 | oak |
| 101 | pine |
| 102 | brick |
| 103 | concrete |
| 104 | wrought_iron |
| 105 | bronze |
| 106 | clay_dried |
| 107 | thatch |

## Dependencies

- **Imports from:** `crate::data::MaterialRegistry`, `crate::physics::lbm_gas::plugin::LbmState`, `crate::world::chunk::{ChunkCoord, CHUNK_SIZE}`, `crate::physics::constants::GRAVITY`, `crate::entities::inventory::Inventory`, `crate::game_state::GameState`
- **Imported by:** `src/main.rs` (registers `BuildingPlugin`)

## System Schedule

All structural systems run in `FixedUpdate` gated with `.run_if(in_state(GameState::Playing))`:

```
mark_ground_anchors → accumulate_self_weight → apply_wind_loading
  → propagate_stress_and_break → despawn_unsupported_parts
  → cleanup_broken_joints → process_demolitions
```

Stress analysis is budget-limited by `StressTick` — runs every `STRESS_TICK_INTERVAL` (10) ticks.

## Tests

26 unit tests across all submodules. Run with:
```bash
cargo test --lib -- building
```
