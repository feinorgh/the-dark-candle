# Behavior Module

Creature AI: motivational needs, utility-based action selection, 3D pathfinding, behavior execution, and perception.

## Files

| File | Purpose |
|------|---------|
| `needs.rs` | Five core drives (hunger, safety, rest, curiosity, social) |
| `utility.rs` | Score actions against needs, select best with tiebreaker |
| `pathfinding.rs` | A* on 3D voxel grid via `VoxelGrid` trait |
| `behaviors.rs` | Action executors producing `MovementIntent` |
| `perception.rs` | Sight (LOS), hearing (through walls), smell (blocked by solid) |

## Need Weights (default)

| Need | Growth Rate | AI Weight | Driven By |
|------|-----------|-----------|-----------|
| Hunger | 0.0 (external) | 1.0 | Metabolism energy fraction |
| Safety | 0.0 (external) | 1.5 | Perceived threats |
| Rest | 0.002/tick | 0.8 | Passive accumulation |
| Curiosity | 0.003/tick | 0.4 | Passive accumulation |
| Social | 0.001/tick | 0.5 | Passive accumulation |

## Action Score Multipliers (utility.rs)

| Action | Formula | Notes |
|--------|---------|-------|
| Idle | 0.05 (constant) | Fallback |
| Wander | curiosity × 0.6 | |
| Eat | hunger × 1.2 | Requires food nearby |
| Flee | safety × 2.0 | Highest priority when in danger |
| Sleep | rest × 0.8 | |
| Socialize | social × 0.5 | Requires ally nearby |
| Attack | hunger × 0.8 + 0.2 | Hostile creatures only |

## Dependencies

- **Imports from:** `crate::world::voxel::MaterialId` (pathfinding uses material checks)
- **Imported by:** (standalone; game loop will read `BehaviorOutput` for movement)

## Patterns

### VoxelGrid Trait (pathfinding.rs)

```rust
pub trait VoxelGrid {
    fn get_material(&self, x: i32, y: i32, z: i32) -> Option<MaterialId>;
}
```

Tests implement this on a simple `TestGrid` struct. Production code will implement it on `ChunkMap` + `Query<&Chunk>`.

### PerceptionGrid Trait (perception.rs)

```rust
pub trait PerceptionGrid {
    fn is_opaque(&self, x: i32, y: i32, z: i32) -> bool;
    fn is_solid(&self, x: i32, y: i32, z: i32) -> bool;
}
```

Same pattern — lightweight test grids, production wraps chunk lookups.

### Behavior Executors

Each `execute_*()` function returns a `BehaviorOutput` containing:
- `MovementIntent` (direction + speed multiplier)
- Action flags (`wants_to_eat`, `wants_to_attack`, `is_sleeping`)

## Gotchas

- Pathfinding `max_nodes` budget (default 2000) prevents unbounded search. If paths seem to fail, check if the budget is sufficient for the distance.
- `select_action()` uses a tiny RNG noise (0.02 scale) for tiebreaking — it won't override a clear winner but prevents deterministic loops between equal-scoring actions.
- Sight uses ray marching (not true Bresenham). At very long ranges, it may skip voxels. The default `sight_range = 32` keeps this accurate enough.
