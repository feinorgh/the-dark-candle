# Physics Module

Gravity, drag, buoyancy, friction, AABB collision, fluid simulation, structural integrity, and gas pressure.

## Files

| File | Purpose |
|------|---------|
| `constants.rs` | SI physical constants: `GRAVITY`, `AIR_DENSITY_SEA_LEVEL`, `ATMOSPHERIC_PRESSURE`, etc. |
| `gravity.rs` | `PhysicsBody`, `Mass`, `DragProfile` components; `apply_forces()` system (gravity + buoyancy + drag + friction) |
| `collision.rs` | `Collider` AABB, `aabb_intersects_terrain()`, `resolve_collisions()` |
| `fluids.rs` | Cellular automata water/lava flow |
| `integrity.rs` | BFS flood-fill structural support from y=0 anchors |
| `pressure.rs` | Discrete gas pressure diffusion |

## Force Model

`apply_forces()` runs on `FixedUpdate` and applies four constituent forces per tick:

| Force | Formula | Direction |
|-------|---------|-----------|
| Gravity | F_g = m × g × gravity_scale | downward |
| Buoyancy | F_b = ρ_medium × V_displaced × g | upward |
| Drag | F_d = ½ × ρ × v² × C_d × A | opposes velocity |
| Friction | F_f = μ × F_normal | opposes horizontal velocity (grounded only) |

**Emergent terminal velocity:** drag grows with v² while gravity is constant, so velocity naturally converges to v_t = √(2mg / (ρ·C_d·A)) without any hard cap. For an 80 kg entity with C_d=1.2 and A=0.7 m², this gives ~39 m/s in air.

**Backward compatibility:** entities with only `PhysicsBody` (no `Mass`/`DragProfile`) fall back to simple gravity acceleration + safety cap.

## Constants

All sourced from `constants.rs` (strict SI units):

- `GRAVITY = 9.80665` (m/s², NIST CODATA)
- `VELOCITY_SAFETY_CAP = 200.0` (m/s, absolute backstop)
- `AIR_DENSITY_SEA_LEVEL = 1.225` (kg/m³, ISO 2533)
- `ATMOSPHERIC_PRESSURE = 101325.0` (Pa, ISO 2533)

## Dependencies

- **Imports from:** `crate::world::{chunk, chunk_manager, collision, voxel}`, `crate::physics::constants`
- **Imported by:** gameplay systems that spawn entities with `PhysicsBody` + optional `Mass`/`DragProfile`

## Patterns

### Snapshot-Based Simulation

Fluid and pressure systems use a two-pass pattern:
1. **Snapshot** the voxel grid (or read from an immutable copy).
2. **Mutate** in a second pass based on snapshot values.

This prevents read-write conflicts within a single tick and ensures deterministic behavior regardless of iteration order.

### Pure Functions

All simulation logic is in pure functions that take `&mut [Voxel]` slices or flat arrays. Bevy systems just call these functions on chunk data. Test with small flat arrays, not full chunks.

Force helper functions (`gravitational_force`, `buoyancy_force`, `drag_force`, `friction_force`, `terminal_velocity`) are pure and unit-testable without ECS.

## Gotchas

- **Fluid flow direction matters:** downward flow must iterate top-to-bottom (`(1..size).rev()`) for gravity cascade in a single pass.
- **Water at y=0 boundary:** spreads sideways since it can't fall further. Tests should check the layer, not exact position.
- **Structural integrity:** fluids (water, lava) are NOT structural — only solid materials anchor to y=0.
- **`is_solid()` vs structural:** Water returns `is_solid() == true` (it's not air) but is not a structural support. Integrity checks must exclude fluids explicitly.
- **Medium density is currently constant:** `apply_forces()` uses `AIR_DENSITY_SEA_LEVEL` everywhere. Future work: sample the voxel at entity position for per-medium density (water = 1000 kg/m³).
