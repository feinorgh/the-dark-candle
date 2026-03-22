# Camera Module

First-person camera controller: WASD movement, mouse look, gravity, jumping, and fly-mode toggle.

## Files

| File | Purpose |
|------|---------|
| `mod.rs` | `FpsCamera` component, movement/look/gravity systems, `CameraPlugin` |

## Constants

- `EYE_HEIGHT = 1.7` (meters above ground)
- `GRAVITY = 20.0` (m/s²)
- `TERMINAL_VELOCITY = 50.0`

## Controls

| Key | Action |
|-----|--------|
| WASD | Horizontal movement |
| Mouse | Look (yaw/pitch) |
| Space | Jump |
| G | Toggle fly mode |
| Escape | Release cursor |
| Click | Grab cursor |

## Dependencies

- **Imports from:** `crate::world::{chunk::Chunk, chunk_manager::ChunkMap, collision::ground_height_at}`
- **Imported by:** `crate::world` (for `FpsCamera` position in chunk loading)

## Bevy 0.18 Specifics

- Camera spawned as `Camera3d::default()` component (NOT `Camera3dBundle`).
- `CursorOptions` is a **separate ECS component** on the Window entity, NOT a field of `Window`.
- Mouse input uses `Res<AccumulatedMouseMotion>` resource, NOT `EventReader<MouseMotion>`.
- Pitch is clamped to ±89° to prevent gimbal lock.

## Gotchas

- The gravity system runs on `FixedUpdate`. Camera look/move run on `Update` for responsiveness.
- `ground_height_at()` scans multiple vertical chunks (y = -4..8). If terrain is generated outside this range, the player will fall through.
- Fly mode disables gravity and enables vertical movement with Space/Shift. It does NOT disable collision.
