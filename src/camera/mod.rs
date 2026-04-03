use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll};
use bevy::pbr::{Atmosphere, DistanceFog, FogFalloff, ScatteringMedium};
use bevy::post_process::bloom::Bloom;
use bevy::prelude::*;
use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow};

use crate::biology::health::Health;
use crate::game_state::GameState;
use crate::hud::{FallTracker, Player};
use crate::physics::constants;
use crate::world::chunk::Chunk;
use crate::world::chunk_manager::{ChunkMap, TerrainGeneratorRes};
use crate::world::collision::{
    ground_height_at, ground_height_from_terrain_gen, ground_height_radial,
};
use crate::world::planet::PlanetConfig;
use crate::world::v2::chunk_manager::V2TerrainGen;

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
            .add_systems(OnEnter(GameState::Paused), release_cursor)
            .add_systems(OnEnter(GameState::Map), release_cursor);
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
    /// Fly-mode speed multiplier (scroll wheel adjusts, default 1.0).
    pub fly_speed_multiplier: f32,
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
            fly_speed_multiplier: 1.0,
        }
    }
}

/// Compute the local "up" direction (surface normal) for a position on a
/// spherical planet.  Returns the normalized radial direction from the planet
/// center, falling back to `Vec3::Y` near the origin.
fn local_up_from_position(pos: Vec3) -> Vec3 {
    let len = pos.length();
    if len > 1e-6 { pos / len } else { Vec3::Y }
}

fn spawn_camera(
    mut commands: Commands,
    terrain_gen: Option<Res<TerrainGeneratorRes>>,
    planet: Res<PlanetConfig>,
    mut media: ResMut<Assets<ScatteringMedium>>,
) {
    // Pick a spawn position on the terrain surface.
    //
    // Flat mode:  spawn at world origin (0, surface_height, 0).
    // Spherical mode: spawn on the planet surface at lat=45°, lon=0°. This
    //   avoids the north pole (Y-axis singularity) and the equator (where
    //   chunk density is highest during initial load). The actual position is
    //   a point at distance `surface_radius` from the planet center along the
    //   (lat, lon) direction vector.
    let (spawn_pos, look_target, up_hint) = if let Some(ref tg) = terrain_gen {
        if tg.0.is_spherical() {
            // Latitude 45°, longitude 0° — a mid-latitude spawn.
            let lat: f64 = std::f64::consts::FRAC_PI_4; // 45°
            let lon: f64 = 0.0;
            let surface_r = tg.0.sample_height(
                // sample_height for spherical uses the (x, z) to derive lat/lon internally,
                // but we already have lat/lon. To be consistent, we pass Cartesian coords
                // that reconstruct the desired lat/lon through the planet's lat_lon() method.
                // For rotation_axis = Y, a point at (cos(lat)*cos(lon), sin(lat), cos(lat)*sin(lon))
                // gives the correct lat/lon back.
                lat.cos() * lon.cos() * 1000.0, // world_x
                lat.cos() * lon.sin() * 1000.0, // world_z (passed as z)
            ) as f32;
            // Construct the spawn direction from lat/lon (Y-up planet, axis = Y).
            let dir = Vec3::new(
                lat.cos() as f32 * lon.cos() as f32,
                lat.sin() as f32,
                lat.cos() as f32 * lon.sin() as f32,
            )
            .normalize();
            let spawn = dir * (surface_r + EYE_HEIGHT);
            // Look tangent to the surface (slightly ahead along the equator direction).
            let look = spawn + Vec3::new(-dir.y, dir.x, 0.0).normalize() * 10.0;
            // Use local surface normal as the up hint so the initial frame is
            // consistent with the spherical camera_look rotation.
            (spawn, look, dir)
        } else {
            let spawn_x = 0.0_f32;
            let spawn_z = 0.0_f32;
            let surface_y = tg.0.sample_height(spawn_x as f64, spawn_z as f64) as f32 + 1.0;
            let spawn_y = surface_y + EYE_HEIGHT;
            let pos = Vec3::new(spawn_x, spawn_y, spawn_z);
            let look = Vec3::new(10.0, spawn_y - 1.0, 10.0);
            (pos, look, Vec3::Y)
        }
    } else {
        // No terrain generator yet — safe fallback.
        let pos = Vec3::new(0.0, 100.0 + EYE_HEIGHT, 0.0);
        let look = Vec3::new(10.0, 99.0, 10.0);
        (pos, look, Vec3::Y)
    };

    let medium = media.add(ScatteringMedium::default());

    // Configure atmosphere radii to match the actual planet.
    // ScatteringMedium::default() assumes a 60 km atmosphere height
    // (Rayleigh scale height 8 km = 8/60 normalized). We preserve that
    // thickness so the scattering coefficients remain physically correct.
    let atmosphere = if planet.is_spherical() {
        let r = planet.mean_radius as f32;
        Atmosphere {
            bottom_radius: r,
            top_radius: r + 60_000.0,
            ground_albedo: Vec3::splat(0.3),
            medium: medium.clone(),
        }
    } else {
        Atmosphere::earthlike(medium)
    };

    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(spawn_pos).looking_at(look_target, up_hint),
        Bloom::NATURAL,
        atmosphere,
        DistanceFog {
            color: Color::srgba(0.7, 0.78, 0.9, 1.0),
            directional_light_color: Color::srgba(1.0, 0.95, 0.85, 0.5),
            directional_light_exponent: 30.0,
            falloff: FogFalloff::from_visibility(500.0),
        },
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
///
/// In spherical mode the camera's "up" must match the local surface normal
/// (radial direction from planet center), not the global Y axis.  We compute a
/// base orientation via `Quat::from_rotation_arc(Y, local_up)` and apply
/// yaw/pitch on top.
fn camera_look(
    cursor_q: Query<&CursorOptions, With<PrimaryWindow>>,
    accumulated: Res<AccumulatedMouseMotion>,
    planet: Res<PlanetConfig>,
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

    let yaw_pitch =
        Quat::from_axis_angle(Vec3::Y, cam.yaw) * Quat::from_axis_angle(Vec3::X, cam.pitch);

    if planet.is_spherical() {
        let local_up = local_up_from_position(transform.translation);
        let base = Quat::from_rotation_arc(Vec3::Y, local_up);
        transform.rotation = base * yaw_pitch;
    } else {
        transform.rotation = yaw_pitch;
    }
}

/// WASD + Space/Shift movement (only when cursor is grabbed).
/// When gravity is enabled, movement is tangent to the local surface
/// (horizontal in flat mode, tangent to planet in spherical mode).
/// Space jumps if grounded, Shift crouches (not yet implemented).
/// Press G to toggle gravity (fly mode).
fn camera_move(
    cursor_q: Query<&CursorOptions, With<PrimaryWindow>>,
    key: Res<ButtonInput<KeyCode>>,
    scroll: Res<AccumulatedMouseScroll>,
    time: Res<Time>,
    planet: Res<PlanetConfig>,
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
        // Project movement onto the local tangent plane (perpendicular to gravity).
        let local_up = if planet.is_spherical() {
            local_up_from_position(transform.translation)
        } else {
            Vec3::Y
        };

        // Project forward/right onto the tangent plane and re-normalize.
        let forward_tangent = (forward - local_up * forward.dot(local_up)).normalize_or_zero();
        let right_tangent = (right - local_up * right.dot(local_up)).normalize_or_zero();

        if key.pressed(KeyCode::KeyW) {
            direction += forward_tangent;
        }
        if key.pressed(KeyCode::KeyS) {
            direction -= forward_tangent;
        }
        if key.pressed(KeyCode::KeyD) {
            direction += right_tangent;
        }
        if key.pressed(KeyCode::KeyA) {
            direction -= right_tangent;
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
        let up = if planet.is_spherical() {
            local_up_from_position(transform.translation)
        } else {
            Vec3::Y
        };
        if key.pressed(KeyCode::Space) {
            direction += up;
        }
        if key.pressed(KeyCode::ShiftLeft) {
            direction -= up;
        }
    }

    if direction != Vec3::ZERO {
        direction = direction.normalize();
    }

    // Fly-mode: scroll wheel adjusts speed multiplier (2× per notch)
    if !cam.gravity_enabled && scroll.delta.y != 0.0 {
        let old = cam.fly_speed_multiplier;
        cam.fly_speed_multiplier *= 2.0_f32.powf(scroll.delta.y);
        cam.fly_speed_multiplier = cam.fly_speed_multiplier.clamp(0.25, 512.0);
        if (cam.fly_speed_multiplier - old).abs() > f32::EPSILON {
            info!(
                "Fly speed: {:.0} m/s (×{:.2})",
                FLY_SPEED * cam.fly_speed_multiplier,
                cam.fly_speed_multiplier,
            );
        }
    }

    // Determine effective movement speed
    let effective_speed = if !cam.gravity_enabled {
        let base = FLY_SPEED * cam.fly_speed_multiplier;
        if key.pressed(KeyCode::ShiftLeft) {
            base * 5.0
        } else {
            base
        }
    } else if key.pressed(KeyCode::ControlLeft) {
        SPRINT_SPEED
    } else {
        cam.speed
    };

    transform.translation += direction * effective_speed * time.delta_secs();
}

/// Apply gravity and ground collision to the camera.
///
/// Flat mode:  gravity is -Y, ground check scans vertical voxel columns.
/// Spherical mode: gravity is radial (toward planet center), ground check
///   uses `ground_height_radial` which scans along the radial direction.
///   In V2 pipeline mode, falls back to terrain-gen sampling when V1 chunks
///   are unavailable.
fn camera_gravity(
    time: Res<Time>,
    chunk_map: Option<Res<ChunkMap>>,
    chunks: Query<&Chunk>,
    planet: Res<PlanetConfig>,
    v2_gen: Option<Res<V2TerrainGen>>,
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

    if planet.is_spherical() {
        // Radial gravity: pull toward planet center.
        let local_up = local_up_from_position(pos);

        // Apply gravity along the radial direction.
        cam.vertical_velocity -= constants::GRAVITY * dt;
        cam.vertical_velocity = cam.vertical_velocity.max(-200.0);
        // Move along the radial direction (positive = outward = up).
        transform.translation += local_up * cam.vertical_velocity * dt;

        // Radial ground collision — try V1 chunks first, fall back to terrain gen.
        let ground_r = chunk_map
            .as_ref()
            .and_then(|cm| ground_height_radial(pos, cm, &chunks))
            .or_else(|| {
                v2_gen
                    .as_ref()
                    .map(|tg| ground_height_from_terrain_gen(pos, &tg.0))
            });

        if let Some(ground_r) = ground_r {
            let feet_r = transform.translation.length() - EYE_HEIGHT;
            if feet_r <= ground_r {
                // Place player on the surface at the correct radial distance.
                let up = transform.translation.normalize_or(Vec3::Y);
                transform.translation = up * (ground_r + EYE_HEIGHT);
                cam.vertical_velocity = 0.0;
                cam.grounded = true;
            } else {
                cam.grounded = false;
            }
        } else {
            cam.grounded = false;
        }
    } else {
        // Flat mode: Y-axis gravity.
        cam.vertical_velocity -= constants::GRAVITY * dt;
        cam.vertical_velocity = cam.vertical_velocity.max(-200.0);
        transform.translation.y += cam.vertical_velocity * dt;

        // Flat ground collision (V1 chunks only — V2 is always spherical).
        let ground_y = chunk_map
            .as_ref()
            .and_then(|cm| ground_height_at(pos.x, pos.z, cm, &chunks));
        if let Some(ground_y) = ground_y {
            let feet_y = transform.translation.y - EYE_HEIGHT;
            if feet_y <= ground_y {
                transform.translation.y = ground_y + EYE_HEIGHT;
                cam.vertical_velocity = 0.0;
                cam.grounded = true;
            } else {
                cam.grounded = false;
            }
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
        assert_eq!(cam.fly_speed_multiplier, 1.0);
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

    #[test]
    fn fly_speed_multiplier_range() {
        // At min multiplier, speed should still be usable (≥5 m/s)
        let min_speed = FLY_SPEED * 0.25;
        assert!(min_speed >= 5.0, "min fly speed {min_speed} too low");
        // At max multiplier, speed should cover the planet quickly
        let max_speed = FLY_SPEED * 512.0;
        assert!(max_speed >= 10_000.0, "max fly speed {max_speed} too low for 32km planet");
    }

    #[test]
    fn fly_speed_multiplier_clamp() {
        let mut cam = FpsCamera::default();
        cam.fly_speed_multiplier = 1000.0;
        cam.fly_speed_multiplier = cam.fly_speed_multiplier.clamp(0.25, 512.0);
        assert_eq!(cam.fly_speed_multiplier, 512.0);

        cam.fly_speed_multiplier = 0.01;
        cam.fly_speed_multiplier = cam.fly_speed_multiplier.clamp(0.25, 512.0);
        assert_eq!(cam.fly_speed_multiplier, 0.25);
    }

    #[test]
    fn local_up_at_north_pole_is_y() {
        // At the north pole, position is along +Y → local up = +Y (same as flat mode).
        let up = local_up_from_position(Vec3::new(0.0, 6_371_000.0, 0.0));
        assert!((up - Vec3::Y).length() < 1e-5);
    }

    #[test]
    fn local_up_at_equator_is_radial() {
        // At the equator (lon=0), position is along +X → local up = +X.
        let up = local_up_from_position(Vec3::new(6_371_000.0, 0.0, 0.0));
        assert!((up - Vec3::X).length() < 1e-5);
    }

    #[test]
    fn local_up_at_45_lat() {
        // At 45° latitude (lon=0), local up should be (cos45, sin45, 0).
        let r = 6_371_000.0_f32;
        let lat = std::f32::consts::FRAC_PI_4;
        let pos = Vec3::new(r * lat.cos(), r * lat.sin(), 0.0);
        let up = local_up_from_position(pos);
        let expected = Vec3::new(lat.cos(), lat.sin(), 0.0);
        assert!(
            (up - expected).length() < 1e-5,
            "local_up at 45° lat should be ({}, {}, 0), got {up}",
            lat.cos(),
            lat.sin(),
        );
    }

    #[test]
    fn spherical_camera_rotation_aligns_up_with_surface_normal() {
        // At 45° latitude, the base rotation should align the camera's Y axis
        // with the local surface normal, not with global Y.
        let lat = std::f32::consts::FRAC_PI_4;
        let pos = Vec3::new(6_371_000.0 * lat.cos(), 6_371_000.0 * lat.sin(), 0.0);
        let local_up = local_up_from_position(pos);
        let base = Quat::from_rotation_arc(Vec3::Y, local_up);
        let yaw_pitch = Quat::from_axis_angle(Vec3::Y, 0.0) * Quat::from_axis_angle(Vec3::X, 0.0);
        let rotation = base * yaw_pitch;

        // Camera's local Y axis in world space should equal local_up.
        let cam_up = rotation * Vec3::Y;
        assert!(
            (cam_up - local_up).length() < 1e-5,
            "camera up {cam_up} should match surface normal {local_up}",
        );

        // Camera's forward (-Z) should be perpendicular to local_up (tangent to sphere).
        let cam_forward = rotation * Vec3::NEG_Z;
        assert!(
            cam_forward.dot(local_up).abs() < 1e-5,
            "camera forward {cam_forward} should be tangent to sphere",
        );
    }

    #[test]
    fn local_up_near_origin_falls_back_to_y() {
        let up = local_up_from_position(Vec3::new(1e-8, 1e-8, 1e-8));
        assert_eq!(up, Vec3::Y);
    }
}
