use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::post_process::bloom::Bloom;
use bevy::prelude::*;
use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow};

use crate::biology::health::Health;
use crate::game_state::GameState;
use crate::hud::{FallTracker, Player};
use crate::physics::constants;
use crate::world::chunk::Chunk;
use crate::world::chunk_manager::{ChunkMap, TerrainGeneratorRes};
use crate::world::collision::ground_height_at;

pub struct CameraPlugin;

impl Plugin for CameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_camera)
            .add_systems(Update, cursor_grab.run_if(in_state(GameState::Playing)))
            .add_systems(
                Update,
                (camera_look, camera_move, camera_gravity)
                    .chain()
                    .after(cursor_grab)
                    .run_if(in_state(GameState::Playing)),
            )
            .add_systems(OnEnter(GameState::Playing), grab_cursor)
            .add_systems(OnEnter(GameState::Paused), release_cursor);
    }
}

/// Marker + settings for the first-person camera controller.
#[derive(Component, Debug)]
pub struct FpsCamera {
    pub speed: f32,
    pub sensitivity: f32,
    pub pitch: f32,
    pub yaw: f32,
    /// Vertical velocity for gravity (m/s).
    pub vertical_velocity: f32,
    /// Whether the camera is on the ground.
    pub grounded: bool,
    /// Whether gravity is enabled (toggle with G key).
    pub gravity_enabled: bool,
}

/// Player eye height above the ground surface (m).
/// Average adult eye height when standing: ~1.7 m.
const EYE_HEIGHT: f32 = 1.7;

/// Player walk speed (m/s). Average human: ~1.4 m/s.
const WALK_SPEED: f32 = 5.0;

/// Player sprint speed (m/s). Average human sprint: ~5–8 m/s.
const SPRINT_SPEED: f32 = 8.0;

/// Fly-mode speed (m/s). Fast traversal when gravity is disabled.
const FLY_SPEED: f32 = 20.0;

/// Jump initial velocity (m/s).
/// Derived from desired jump height of ~1.25 m: v₀ = sqrt(2×g×h).
/// Source: Wikipedia — Projectile motion.
const JUMP_VELOCITY: f32 = 4.95;

impl Default for FpsCamera {
    fn default() -> Self {
        Self {
            speed: WALK_SPEED,
            sensitivity: 0.002,
            pitch: 0.0,
            yaw: 0.0,
            vertical_velocity: 0.0,
            grounded: false,
            gravity_enabled: true,
        }
    }
}

fn spawn_camera(mut commands: Commands, terrain_gen: Option<Res<TerrainGeneratorRes>>) {
    // Compute spawn height from terrain generator if available, otherwise
    // fall back to a safe default above the typical surface.
    let spawn_x = 0.0_f32;
    let spawn_z = 0.0_f32;
    let surface_y = terrain_gen
        .map(|tg| tg.0.sample_height(spawn_x as f64, spawn_z as f64) as f32 + 1.0)
        .unwrap_or(100.0);
    let spawn_y = surface_y + EYE_HEIGHT;

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(spawn_x, spawn_y, spawn_z)
            .looking_at(Vec3::new(10.0, spawn_y - 1.0, 10.0), Vec3::Y),
        Bloom::NATURAL,
        FpsCamera::default(),
        Player,
        Health::new(100.0),
        FallTracker::default(),
    ));
}

/// Grab cursor on left-click (while Playing).
fn cursor_grab(
    mouse: Res<ButtonInput<MouseButton>>,
    mut cursor_q: Query<&mut CursorOptions, With<PrimaryWindow>>,
) {
    let Ok(mut cursor) = cursor_q.single_mut() else {
        return;
    };
    if mouse.just_pressed(MouseButton::Left) {
        cursor.grab_mode = CursorGrabMode::Locked;
        cursor.visible = false;
    }
}

/// Lock cursor and hide it when entering Playing state.
fn grab_cursor(mut cursor_q: Query<&mut CursorOptions, With<PrimaryWindow>>) {
    let Ok(mut cursor) = cursor_q.single_mut() else {
        return;
    };
    cursor.grab_mode = CursorGrabMode::Locked;
    cursor.visible = false;
}

/// Unlock cursor and show it when entering Paused state.
fn release_cursor(mut cursor_q: Query<&mut CursorOptions, With<PrimaryWindow>>) {
    let Ok(mut cursor) = cursor_q.single_mut() else {
        return;
    };
    cursor.grab_mode = CursorGrabMode::None;
    cursor.visible = true;
}

/// Rotate camera based on accumulated mouse movement (only when cursor is grabbed).
fn camera_look(
    cursor_q: Query<&CursorOptions, With<PrimaryWindow>>,
    accumulated: Res<AccumulatedMouseMotion>,
    mut cam_q: Query<(&mut FpsCamera, &mut Transform)>,
) {
    let Ok(cursor) = cursor_q.single() else {
        return;
    };
    if cursor.grab_mode == CursorGrabMode::None {
        return;
    }

    let delta = accumulated.delta;
    if delta == Vec2::ZERO {
        return;
    }

    let Ok((mut cam, mut transform)) = cam_q.single_mut() else {
        return;
    };

    cam.yaw -= delta.x * cam.sensitivity;
    cam.pitch -= delta.y * cam.sensitivity;
    cam.pitch = cam.pitch.clamp(-1.5, 1.5);

    transform.rotation =
        Quat::from_axis_angle(Vec3::Y, cam.yaw) * Quat::from_axis_angle(Vec3::X, cam.pitch);
}

/// WASD + Space/Shift movement (only when cursor is grabbed).
/// When gravity is enabled, movement is horizontal only (no fly).
/// Space jumps if grounded, Shift crouches (not yet implemented).
/// Press G to toggle gravity (fly mode).
fn camera_move(
    cursor_q: Query<&CursorOptions, With<PrimaryWindow>>,
    key: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut cam_q: Query<(&mut FpsCamera, &mut Transform)>,
) {
    let Ok(cursor) = cursor_q.single() else {
        return;
    };

    let Ok((mut cam, mut transform)) = cam_q.single_mut() else {
        return;
    };

    // Toggle gravity with G
    if key.just_pressed(KeyCode::KeyG) {
        cam.gravity_enabled = !cam.gravity_enabled;
        if !cam.gravity_enabled {
            cam.vertical_velocity = 0.0;
        }
    }

    if cursor.grab_mode == CursorGrabMode::None {
        return;
    }

    let forward = transform.forward().as_vec3();
    let right = transform.right().as_vec3();

    let mut direction = Vec3::ZERO;

    if cam.gravity_enabled {
        // Horizontal movement only (project forward onto XZ plane)
        let forward_xz = Vec3::new(forward.x, 0.0, forward.z).normalize_or_zero();
        let right_xz = Vec3::new(right.x, 0.0, right.z).normalize_or_zero();

        if key.pressed(KeyCode::KeyW) {
            direction += forward_xz;
        }
        if key.pressed(KeyCode::KeyS) {
            direction -= forward_xz;
        }
        if key.pressed(KeyCode::KeyD) {
            direction += right_xz;
        }
        if key.pressed(KeyCode::KeyA) {
            direction -= right_xz;
        }
        // Jump — v₀ = sqrt(2gh) for ~1.25 m jump height
        if key.just_pressed(KeyCode::Space) && cam.grounded {
            cam.vertical_velocity = JUMP_VELOCITY;
            cam.grounded = false;
        }
    } else {
        // Fly mode: full 3D movement
        if key.pressed(KeyCode::KeyW) {
            direction += forward;
        }
        if key.pressed(KeyCode::KeyS) {
            direction -= forward;
        }
        if key.pressed(KeyCode::KeyD) {
            direction += right;
        }
        if key.pressed(KeyCode::KeyA) {
            direction -= right;
        }
        if key.pressed(KeyCode::Space) {
            direction += Vec3::Y;
        }
        if key.pressed(KeyCode::ShiftLeft) {
            direction -= Vec3::Y;
        }
    }

    if direction != Vec3::ZERO {
        direction = direction.normalize();
    }

    // Determine effective movement speed
    let effective_speed = if !cam.gravity_enabled {
        FLY_SPEED
    } else if key.pressed(KeyCode::ControlLeft) {
        SPRINT_SPEED
    } else {
        cam.speed
    };

    transform.translation += direction * effective_speed * time.delta_secs();
}

/// Apply gravity and ground collision to the camera.
fn camera_gravity(
    time: Res<Time>,
    chunk_map: Res<ChunkMap>,
    chunks: Query<&Chunk>,
    mut cam_q: Query<(&mut FpsCamera, &mut Transform)>,
) {
    let Ok((mut cam, mut transform)) = cam_q.single_mut() else {
        return;
    };

    if !cam.gravity_enabled {
        return;
    }

    let dt = time.delta_secs();
    let pos = transform.translation;

    // Apply gravity (SI: 9.80665 m/s²)
    cam.vertical_velocity -= constants::GRAVITY * dt;
    // Safety cap at 200 m/s — real terminal velocity (~53 m/s for a human)
    // will be handled by the force-based drag model once applied to camera.
    cam.vertical_velocity = cam.vertical_velocity.max(-200.0);
    transform.translation.y += cam.vertical_velocity * dt;

    // Ground collision
    if let Some(ground_y) = ground_height_at(pos.x, pos.z, &chunk_map, &chunks) {
        let feet_y = transform.translation.y - EYE_HEIGHT;
        if feet_y <= ground_y {
            transform.translation.y = ground_y + EYE_HEIGHT;
            cam.vertical_velocity = 0.0;
            cam.grounded = true;
        } else {
            cam.grounded = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fps_camera_default_values() {
        let cam = FpsCamera::default();
        assert_eq!(cam.speed, WALK_SPEED);
        assert_eq!(cam.pitch, 0.0);
        assert_eq!(cam.yaw, 0.0);
        assert!(cam.sensitivity > 0.0);
        assert_eq!(cam.vertical_velocity, 0.0);
        assert!(!cam.grounded);
        assert!(cam.gravity_enabled);
    }

    #[test]
    fn pitch_clamp_limits() {
        let mut cam = FpsCamera::default();
        cam.pitch = 2.0;
        cam.pitch = cam.pitch.clamp(-1.5, 1.5);
        assert_eq!(cam.pitch, 1.5);

        cam.pitch = -2.0;
        cam.pitch = cam.pitch.clamp(-1.5, 1.5);
        assert_eq!(cam.pitch, -1.5);
    }

    #[test]
    fn eye_height_is_positive() {
        const { assert!(EYE_HEIGHT > 0.0) };
    }

    #[test]
    fn walk_speed_is_realistic() {
        // Average human walk speed: 1.4 m/s, our value set for gameplay feel
        const { assert!(WALK_SPEED >= 1.0 && WALK_SPEED <= 10.0) };
    }

    #[test]
    fn jump_velocity_matches_physics() {
        // v₀ = sqrt(2gh) for h = 1.25 m → v₀ ≈ 4.95 m/s
        // Source: Wikipedia — Projectile motion
        let expected = (2.0 * constants::GRAVITY * 1.25_f32).sqrt();
        assert!(
            (JUMP_VELOCITY - expected).abs() < 0.1,
            "Jump velocity {JUMP_VELOCITY} should be ~{expected} m/s for 1.25m jump"
        );
    }

    #[test]
    fn sprint_faster_than_walk() {
        const { assert!(SPRINT_SPEED > WALK_SPEED) };
    }
}
