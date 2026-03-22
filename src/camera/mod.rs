use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::prelude::*;
use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow};

pub struct CameraPlugin;

impl Plugin for CameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_camera)
            .add_systems(Update, cursor_grab)
            .add_systems(Update, (camera_look, camera_move).after(cursor_grab));
    }
}

/// Marker + settings for the first-person camera controller.
#[derive(Component, Debug)]
pub struct FpsCamera {
    pub speed: f32,
    pub sensitivity: f32,
    pub pitch: f32,
    pub yaw: f32,
}

impl Default for FpsCamera {
    fn default() -> Self {
        Self {
            speed: 10.0,
            sensitivity: 0.002,
            pitch: 0.0,
            yaw: 0.0,
        }
    }
}

fn spawn_camera(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 5.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
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
fn camera_move(
    cursor_q: Query<&CursorOptions, With<PrimaryWindow>>,
    key: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut cam_q: Query<(&FpsCamera, &mut Transform)>,
) {
    let Ok(cursor) = cursor_q.single() else {
        return;
    };
    if cursor.grab_mode == CursorGrabMode::None {
        return;
    }

    let Ok((cam, mut transform)) = cam_q.single_mut() else {
        return;
    };

    let mut direction = Vec3::ZERO;
    let forward = transform.forward().as_vec3();
    let right = transform.right().as_vec3();

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

    if direction != Vec3::ZERO {
        direction = direction.normalize();
    }

    transform.translation += direction * cam.speed * time.delta_secs();
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
}
