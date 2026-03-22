use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::prelude::*;
use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow};

use crate::world::chunk::Chunk;
use crate::world::chunk_manager::ChunkMap;
use crate::world::collision::ground_height_at;

pub struct CameraPlugin;

impl Plugin for CameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_camera)
            .add_systems(Update, cursor_grab)
            .add_systems(
                Update,
                (camera_look, camera_move, camera_gravity)
                    .chain()
                    .after(cursor_grab),
            );
    }
}

/// Marker + settings for the first-person camera controller.
#[derive(Component, Debug)]
pub struct FpsCamera {
    pub speed: f32,
    pub sensitivity: f32,
    pub pitch: f32,
    pub yaw: f32,
    /// Vertical velocity for gravity.
    pub vertical_velocity: f32,
    /// Whether the camera is on the ground.
    pub grounded: bool,
    /// Whether gravity is enabled (toggle with G key).
    pub gravity_enabled: bool,
}

/// Player eye height above the ground surface.
const EYE_HEIGHT: f32 = 1.7;
/// Gravity acceleration in voxel units per second².
const GRAVITY: f32 = 20.0;
/// Terminal falling speed.
const TERMINAL_VELOCITY: f32 = 50.0;

impl Default for FpsCamera {
    fn default() -> Self {
        Self {
            speed: 10.0,
            sensitivity: 0.002,
            pitch: 0.0,
            yaw: 0.0,
            vertical_velocity: 0.0,
            grounded: false,
            gravity_enabled: true,
        }
    }
}

fn spawn_camera(mut commands: Commands) {
    // Start above sea level; gravity will pull us to the terrain surface.
    // The terrain generator has sea_level=64, height_scale=32, so terrain
    // can be up to ~96. Start at Y=100 to be safely above.
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 100.0, 0.0).looking_at(Vec3::new(10.0, 95.0, 10.0), Vec3::Y),
        FpsCamera::default(),
    ));
}

/// Grab cursor on left-click, release on Escape.
fn cursor_grab(
    mouse: Res<ButtonInput<MouseButton>>,
    key: Res<ButtonInput<KeyCode>>,
    mut cursor_q: Query<&mut CursorOptions, With<PrimaryWindow>>,
) {
    let Ok(mut cursor) = cursor_q.single_mut() else {
        return;
    };
    if mouse.just_pressed(MouseButton::Left) {
        cursor.grab_mode = CursorGrabMode::Locked;
        cursor.visible = false;
    }
    if key.just_pressed(KeyCode::Escape) {
        cursor.grab_mode = CursorGrabMode::None;
        cursor.visible = true;
    }
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
        // Jump
        if key.just_pressed(KeyCode::Space) && cam.grounded {
            cam.vertical_velocity = 8.0;
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

    transform.translation += direction * cam.speed * time.delta_secs();
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

    // Apply gravity
    cam.vertical_velocity -= GRAVITY * dt;
    cam.vertical_velocity = cam.vertical_velocity.max(-TERMINAL_VELOCITY);
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
        assert_eq!(cam.speed, 10.0);
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
    fn gravity_constants_are_positive() {
        const { assert!(GRAVITY > 0.0) };
        const { assert!(TERMINAL_VELOCITY > 0.0) };
    }
}
