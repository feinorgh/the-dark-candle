# Camera Module

First-person camera controller: WASD movement, mouse look, gravity, jumping, and fly-mode toggle.

## Files

| File | Purpose |
|------|---------|
| `mod.rs` | `FpsCamera` component, movement/look/gravity systems, `CameraPlugin` |

## Constants (SI Units)

All physics constants reference `crate::physics::constants` for the single source of truth.

| Constant | Value | Unit | Source |
|----------|-------|------|--------|
| `EYE_HEIGHT` | 1.7 | m | Average adult standing eye height |
| `WALK_SPEED` | 5.0 | m/s | Gameplay-tuned (real human ~1.4 m/s) |
| `SPRINT_SPEED` | 8.0 | m/s | Gameplay-tuned (real human ~5–8 m/s) |
| `FLY_SPEED` | 20.0 | m/s | Debug/creative traversal speed |
| `JUMP_VELOCITY` | 4.95 | m/s | v₀ = √(2gh) for h = 1.25 m |
| Gravity | 9.80665 | m/s² | From `constants::GRAVITY` (NIST) |

## Controls

| Key | Action |
|-----|--------|
| WASD | Horizontal movement |
| Mouse | Look (yaw/pitch) |
| Space | Jump (when grounded) |
| Left Ctrl | Sprint |
| G | Toggle fly mode |
| Escape | Release cursor |
| Click | Grab cursor |

## Dependencies

- **Imports from:** `crate::physics::constants`, `crate::world::{chunk::Chunk, chunk_manager::ChunkMap, collision::ground_height_at}`
- **Imported by:** `crate::world` (for `FpsCamera` position in chunk loading)

## Bevy 0.18 Specifics

- Camera spawned as `Camera3d::default()` component (NOT `Camera3dBundle`).
- `CursorOptions` is a **separate ECS component** on the Window entity, NOT a field of `Window`.
- Mouse input uses `Res<AccumulatedMouseMotion>` resource, NOT `EventReader<MouseMotion>`.
- Pitch is clamped to ±89° to prevent gimbal lock.

## Atmosphere & Fog (on camera entity)

The camera entity includes atmospheric rendering components:

- `Atmosphere::earthlike(medium)` — GPU atmospheric scattering (Rayleigh+Mie+Ozone)
- `ScatteringMedium::default()` — standard Earth medium, added via `world.resource_mut::<Assets<ScatteringMedium>>()`
- `DistanceFog` — exponential squared falloff, 500 m visibility
- `Bloom::NATURAL` — satisfies HDR requirement for `Atmosphere`

Imports: `bevy::pbr::{Atmosphere, DistanceFog, FogFalloff, ScatteringMedium}` (NOT in prelude).

The fog color is dynamically updated by `update_fog()` in `src/lighting/mod.rs` — see lighting AGENTS.md.

## Map State Integration

When `GameState::Map` is entered (M key), `OnEnter(GameState::Map)` triggers `release_cursor` so the mouse is freed for map interaction. The map module (`src/map/`) handles all map UI; the camera module just handles cursor release.

## Gotchas

- The gravity system runs on `Update` (chained with camera_look and camera_move). Camera uses its own vertical_velocity for simplicity.
- `ground_height_at()` scans multiple vertical chunks (y = -4..8). If terrain is generated outside this range, the player will fall through.
- Fly mode disables gravity and enables vertical movement with Space/Shift. It does NOT disable collision.
- Player jump height (~1.25 m) is realistic but may feel different from games with higher gravity. Jump velocity is derived from `v₀ = √(2 × g × h)`.
- Walk speed is tuned for gameplay (5.0 m/s), not real human walking speed (1.4 m/s), because realistic walking feels too slow in a voxel game.
