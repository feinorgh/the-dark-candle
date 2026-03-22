# Physics Module

Gravity, AABB collision, fluid simulation, structural integrity, and gas pressure.

## Files

| File | Purpose |
|------|---------|
| `gravity.rs` | `PhysicsBody` component, `apply_gravity()` system |
| `collision.rs` | `Collider` AABB, `aabb_intersects_terrain()`, `resolve_collisions()` |
| `fluids.rs` | Cellular automata water/lava flow |
| `integrity.rs` | BFS flood-fill structural support from y=0 anchors |
| `pressure.rs` | Discrete gas pressure diffusion |

## Constants

- `GRAVITY = 20.0` (m/s²)
- `TERMINAL_VELOCITY = 50.0`
- `ATMOSPHERIC_PRESSURE = 1.0` (atmospheres)

## Dependencies

- **Imports from:** `crate::world::{chunk, chunk_manager, collision, voxel}`
- **Imported by:** (currently standalone; entities will use PhysicsBody)

## Patterns

### Snapshot-Based Simulation

Fluid and pressure systems use a two-pass pattern:
1. **Snapshot** the voxel grid (or read from an immutable copy).
2. **Mutate** in a second pass based on snapshot values.

This prevents read-write conflicts within a single tick and ensures deterministic behavior regardless of iteration order.

### Pure Functions

All simulation logic is in pure functions that take `&mut [Voxel]` slices or flat arrays. Bevy systems just call these functions on chunk data. Test with small flat arrays, not full chunks.

## Gotchas

- **Fluid flow direction matters:** downward flow must iterate top-to-bottom (`(1..size).rev()`) for gravity cascade in a single pass.
- **Water at y=0 boundary:** spreads sideways since it can't fall further. Tests should check the layer, not exact position.
- **Structural integrity:** fluids (water, lava) are NOT structural — only solid materials anchor to y=0.
- **`is_solid()` vs structural:** Water returns `is_solid() == true` (it's not air) but is not a structural support. Integrity checks must exclude fluids explicitly.
