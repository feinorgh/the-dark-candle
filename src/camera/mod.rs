use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll};
use bevy::math::DVec3;
use bevy::pbr::{Atmosphere, DistanceFog, FogFalloff, ScatteringMedium};
use bevy::post_process::bloom::Bloom;
use bevy::prelude::*;
use bevy::ui::IsDefaultUiCamera;
use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow};

use crate::biology::health::Health;
use crate::floating_origin::{RenderOrigin, WorldPosition};
use crate::game_state::GameState;
use crate::hud::{FallTracker, Player};
use crate::physics::constants;
use crate::world::chunk_manager::TerrainGeneratorRes;
use crate::world::collision::ground_height_from_terrain_gen;
use crate::world::planet::PlanetConfig;
use crate::world::v2::chunk_manager::V2TerrainGen;

pub struct CameraPlugin;

impl Plugin for CameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_camera)
            .add_systems(Update, cursor_grab.run_if(in_state(GameState::Playing)))
            .add_systems(
                Update,
                (
                    camera_look,
                    camera_move,
                    camera_gravity,
                    sync_camera_transform,
                    toggle_flashlight,
                    adjust_flashlight_intensity,
                )
                    .chain()
                    .after(cursor_grab)
                    .run_if(in_state(GameState::Playing)),
            )
            .add_systems(OnEnter(GameState::Playing), (grab_cursor, snap_to_surface))
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
    /// Whether the player flashlight is on (toggle with F key).
    pub flashlight_enabled: bool,
    /// Current flashlight intensity in candela. Adjusted with Ctrl+`+`/`-`.
    pub flashlight_intensity: f32,
}

/// Marker component for the player-attached spotlight (flashlight).
#[derive(Component)]
pub struct Flashlight;

/// Luminous intensity of the flashlight in candela.
/// ~1 000 lm spread over a ≈20° half-angle cone ≈ 5 000 cd.
const FLASHLIGHT_INTENSITY: f32 = 5_000.0;
/// Warm-white flashlight colour.
const FLASHLIGHT_COLOR: Color = Color::srgb(1.0, 0.97, 0.90);
/// Effective range before intensity falls to zero (m).
const FLASHLIGHT_RANGE: f32 = 100.0;
/// Inner (full-intensity) cone half-angle (rad) — about 8°.
const FLASHLIGHT_INNER_ANGLE: f32 = 0.14;
/// Outer (fade-to-zero) cone half-angle (rad) — about 20°.
const FLASHLIGHT_OUTER_ANGLE: f32 = 0.35;

/// Player eye height above the ground surface (m).
/// Average adult eye height when standing: ~1.7 m.
pub const EYE_HEIGHT: f32 = 1.7;

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
            flashlight_enabled: false,
            flashlight_intensity: FLASHLIGHT_INTENSITY,
        }
    }
}

/// Compute the local "up" direction (surface normal) for an f64 world position
/// on a spherical planet.  Returns the normalized radial direction from the
/// planet center, falling back to `Vec3::Y` near the origin.
fn local_up_from_world_pos(pos: DVec3) -> Vec3 {
    let len = pos.length();
    if len > 1e-6 {
        let n = pos / len;
        Vec3::new(n.x as f32, n.y as f32, n.z as f32)
    } else {
        Vec3::Y
    }
}

/// Requested spawn location for the player camera (lat/lon in radians).
/// Consumed once by `spawn_camera` at startup.
#[derive(Resource, Debug, Clone)]
pub struct SpawnLocation {
    pub lat: f64,
    pub lon: f64,
}

/// Find a random land position on the planet (surface above sea level).
/// Returns `(lat, lon)` in radians, or `None` if no land found after `max_attempts`.
pub fn find_random_land(
    terrain: &crate::world::terrain::UnifiedTerrainGenerator,
    sea_level_radius: f64,
    max_attempts: u32,
    seed: u64,
) -> Option<(f64, f64)> {
    use std::f64::consts::{FRAC_PI_2, PI};

    // Simple LCG for deterministic pseudo-random sampling without pulling in rand.
    let mut rng_state = seed.wrapping_add(0xBEEF_CAFE);
    let mut next_f64 = || -> f64 {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (rng_state >> 11) as f64 / (1u64 << 53) as f64
    };

    for _ in 0..max_attempts {
        // Uniform distribution on sphere: lat = asin(2u-1), lon = 2π·v - π
        let lat = (2.0 * next_f64() - 1.0).clamp(-1.0, 1.0).asin();
        let lon = next_f64() * 2.0 * PI - PI;

        let lat = lat.clamp(-FRAC_PI_2, FRAC_PI_2);
        let surface_r = terrain.sample_surface_radius_at(lat, lon);
        if surface_r > sea_level_radius {
            return Some((lat, lon));
        }
    }
    None
}

/// Find a coastline position: a land point with at least one water neighbor.
/// Returns `(lat, lon)` in radians.
pub fn find_coastline(
    terrain: &crate::world::terrain::UnifiedTerrainGenerator,
    sea_level_radius: f64,
    max_attempts: u32,
    seed: u64,
) -> Option<(f64, f64)> {
    use std::f64::consts::{FRAC_PI_2, PI};

    let mut rng_state = seed.wrapping_add(0x0C0A_57A1);
    let mut next_f64 = || -> f64 {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (rng_state >> 11) as f64 / (1u64 << 53) as f64
    };

    // Angular offset for neighbor probing (~100 m at 32 km radius ≈ 0.003 rad).
    let probe = 0.003_f64;
    let offsets: [(f64, f64); 4] = [(probe, 0.0), (-probe, 0.0), (0.0, probe), (0.0, -probe)];

    for _ in 0..max_attempts {
        let lat = (2.0 * next_f64() - 1.0).clamp(-1.0, 1.0).asin();
        let lon = next_f64() * 2.0 * PI - PI;
        let lat = lat.clamp(-FRAC_PI_2, FRAC_PI_2);

        let surface_r = terrain.sample_surface_radius_at(lat, lon);
        if surface_r <= sea_level_radius {
            continue; // Not land.
        }

        // Check if any neighbor is water.
        for &(dlat, dlon) in &offsets {
            let nlat = (lat + dlat).clamp(-FRAC_PI_2, FRAC_PI_2);
            let nlon = lon + dlon;
            let nr = terrain.sample_surface_radius_at(nlat, nlon);
            if nr <= sea_level_radius {
                return Some((lat, lon));
            }
        }
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn spawn_camera(
    mut commands: Commands,
    terrain_gen: Option<Res<TerrainGeneratorRes>>,
    planet: Res<PlanetConfig>,
    spawn_loc: Option<Res<SpawnLocation>>,
    agent: Option<Res<crate::diagnostics::agent_capture::AgentCaptureConfig>>,
    mut media: ResMut<Assets<ScatteringMedium>>,
    mut render_origin: ResMut<RenderOrigin>,
    mut orbital: ResMut<crate::lighting::orbital::OrbitalState>,
) {
    // Pick a spawn position on the terrain surface.
    //
    // Flat mode:  spawn at world origin (0, surface_height, 0).
    // Spherical mode: use SpawnLocation if provided, otherwise default 45°N, 0°E.
    //
    // Positions are computed in f64 (WorldPosition) and the RenderOrigin is
    // set to the spawn point so that initial Transform.translation ≈ (0,0,0).
    let (spawn_world_pos, look_target, up_hint) = if let Some(ref tg) = terrain_gen {
        if tg.0.is_spherical() {
            let (lat, lon) = if let Some(ref loc) = spawn_loc {
                info!(
                    "Spawning at {:.1}°{}, {:.1}°{}",
                    loc.lat.to_degrees().abs(),
                    if loc.lat >= 0.0 { "N" } else { "S" },
                    loc.lon.to_degrees().abs(),
                    if loc.lon >= 0.0 { "E" } else { "W" },
                );
                (loc.lat, loc.lon)
            } else {
                // No explicit spawn location — find land near a coastline,
                // falling back to any land, then to a fixed default.
                let sea = planet.sea_level_radius;
                let seed = planet.seed as u64;
                if let Some(pos) = find_coastline(&tg.0, sea, 5000, seed) {
                    info!(
                        "Default spawn: coastline at {:.1}°N, {:.1}°E",
                        pos.0.to_degrees(),
                        pos.1.to_degrees()
                    );
                    pos
                } else if let Some(pos) = find_random_land(&tg.0, sea, 2000, seed) {
                    info!(
                        "Default spawn: random land at {:.1}°N, {:.1}°E",
                        pos.0.to_degrees(),
                        pos.1.to_degrees()
                    );
                    pos
                } else {
                    warn!("No land found, falling back to 45°N 0°E");
                    (std::f64::consts::FRAC_PI_4, 0.0)
                }
            };
            let surface_r = tg.0.sample_surface_radius_at(lat, lon);
            // Construct the spawn direction using the same convention the
            // voxel terrain uses (`lat_lon_to_pos`), otherwise the spawn
            // point ends up ~90° away from the land cell that was sampled.
            let dir_f64 = crate::planet::detail::lat_lon_to_pos(lat, lon).normalize();
            let spawn_r = surface_r.max(planet.sea_level_radius) + EYE_HEIGHT as f64;
            info!(
                "SPAWN_DBG lat={:.2}°N lon={:.2}°E surface_r={:.1} sea_level_r={:.1} \
                 spawn_r={:.1} dir={:.4?}",
                lat.to_degrees(),
                lon.to_degrees(),
                surface_r,
                planet.sea_level_radius,
                spawn_r,
                dir_f64,
            );
            let spawn = dir_f64 * spawn_r;
            // If the agent requests daylight, advance the orbital rotation so
            // solar noon falls at the spawn longitude.  Formula: for noon at
            // lon (radians), rotation_angle = π + lon.  This ensures the sun
            // is overhead at the spawn location regardless of its default value.
            if let Some(ref ag) = agent
                && ag.force_daylight
            {
                orbital.rotation_angle = std::f64::consts::PI + lon;
                info!(
                    "[AgentCapture] force_daylight: rotation_angle set to {:.3} rad \
                         (solar noon at lon={:.1}°)",
                    orbital.rotation_angle,
                    lon.to_degrees()
                );
            }
            // Look tangent to the surface (slightly ahead along the equator direction).
            let dir = Vec3::new(dir_f64.x as f32, dir_f64.y as f32, dir_f64.z as f32);
            let mut look_offset = Vec3::new(-dir.y, dir.x, 0.0).normalize() * 10.0;
            // Apply agent initial_yaw_deg / initial_pitch_deg if set.
            // Yaw rotates around the surface normal; pitch rotates around the
            // right axis (perpendicular to both surface normal and look direction).
            // IMPORTANT: also rotate the up_hint by the pitch so that
            // `looking_at` is not given a near-degenerate configuration when
            // pitching steeply (forward ≈ ±up_hint makes the cross-product
            // near-zero and produces a garbage camera frame).
            let mut up_hint = dir;
            if let Some(ref ag) = agent {
                if ag.initial_yaw_deg != 0.0 {
                    look_offset =
                        Quat::from_axis_angle(dir, ag.initial_yaw_deg.to_radians()) * look_offset;
                }
                if ag.initial_pitch_deg != 0.0 {
                    let right = dir.cross(look_offset.normalize()).normalize();
                    // Positive angle around (dir × look) pitches the nose DOWN;
                    // negate so that positive initial_pitch_deg means "look up"
                    // and negative means "look down" (the intuitive convention).
                    let pitch_quat =
                        Quat::from_axis_angle(right, -ag.initial_pitch_deg.to_radians());
                    look_offset = pitch_quat * look_offset;
                    up_hint = pitch_quat * up_hint;
                }
            }
            // Use local surface normal as the up hint so the initial frame is
            // consistent with the spherical camera_look rotation.
            (spawn, look_offset, up_hint)
        } else {
            let spawn_x = 0.0_f64;
            let spawn_z = 0.0_f64;
            let surface_y = tg.0.sample_height(spawn_x, spawn_z) + 1.0;
            let spawn_y = surface_y + EYE_HEIGHT as f64;
            let pos = DVec3::new(spawn_x, spawn_y, spawn_z);
            let look_offset = Vec3::new(10.0, -1.0, 10.0);
            (pos, look_offset, Vec3::Y)
        }
    } else {
        // No terrain generator yet — safe fallback.
        let pos = DVec3::new(0.0, 100.0 + EYE_HEIGHT as f64, 0.0);
        let look_offset = Vec3::new(10.0, -1.0, 10.0);
        (pos, look_offset, Vec3::Y)
    };

    // Set the render origin to the spawn position so that the camera
    // starts with Transform.translation ≈ (0,0,0).
    render_origin.0 = spawn_world_pos;
    let spawn_render = Vec3::ZERO; // WorldPosition - RenderOrigin at spawn

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

    // In spherical mode, the Bevy Atmosphere shader incorrectly computes
    // altitude (it offsets the camera by (0, bottom_radius, 0) assuming a
    // flat world). We skip it and use CPU sky color via ClearColor instead.
    if planet.is_spherical() {
        commands
            .spawn((
                Camera3d::default(),
                IsDefaultUiCamera,
                Transform::from_translation(spawn_render)
                    .looking_at(spawn_render + look_target, up_hint),
                Bloom::NATURAL,
                DistanceFog {
                    color: Color::srgba(0.7, 0.78, 0.9, 1.0),
                    directional_light_color: Color::srgba(1.0, 0.95, 0.85, 0.5),
                    directional_light_exponent: 30.0,
                    falloff: FogFalloff::from_visibility(500.0),
                },
                FpsCamera::default(),
                WorldPosition::from_dvec3(spawn_world_pos),
                Player,
                Health::new(100.0),
                FallTracker::default(),
            ))
            .with_children(|parent| {
                parent.spawn(flashlight_bundle());
            });
    } else {
        commands
            .spawn((
                Camera3d::default(),
                IsDefaultUiCamera,
                Transform::from_translation(spawn_render)
                    .looking_at(spawn_render + look_target, up_hint),
                Bloom::NATURAL,
                atmosphere,
                DistanceFog {
                    color: Color::srgba(0.7, 0.78, 0.9, 1.0),
                    directional_light_color: Color::srgba(1.0, 0.95, 0.85, 0.5),
                    directional_light_exponent: 30.0,
                    falloff: FogFalloff::from_visibility(500.0),
                },
                FpsCamera::default(),
                WorldPosition::from_dvec3(spawn_world_pos),
                Player,
                Health::new(100.0),
                FallTracker::default(),
            ))
            .with_children(|parent| {
                parent.spawn(flashlight_bundle());
            });
    }
}

/// Build the flashlight `SpotLight` child entity bundle.
///
/// Starts hidden; enabled/disabled by [`toggle_flashlight`] with the F key.
/// The light is positioned at the camera origin (eye position) and the default
/// transform points it along local −Z, which is the camera's forward direction
/// — no extra rotation is needed.
fn flashlight_bundle() -> impl Bundle {
    (
        SpotLight {
            intensity: FLASHLIGHT_INTENSITY,
            color: FLASHLIGHT_COLOR,
            range: FLASHLIGHT_RANGE,
            inner_angle: FLASHLIGHT_INNER_ANGLE,
            outer_angle: FLASHLIGHT_OUTER_ANGLE,
            shadows_enabled: true,
            ..default()
        },
        Transform::default(),
        Visibility::Hidden,
        Flashlight,
    )
}

/// Grab cursor on left-click (while Playing). Skipped in agent capture mode.
fn cursor_grab(
    agent: Option<Res<crate::diagnostics::agent_capture::AgentCaptureConfig>>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut cursor_q: Query<&mut CursorOptions, With<PrimaryWindow>>,
) {
    if agent.is_some() {
        return;
    }
    let Ok(mut cursor) = cursor_q.single_mut() else {
        return;
    };
    if mouse.just_pressed(MouseButton::Left) {
        cursor.grab_mode = CursorGrabMode::Locked;
        cursor.visible = false;
    }
}

/// Lock cursor and hide it when entering Playing state.
/// Skipped in agent capture mode (no mouse attached).
fn grab_cursor(
    agent: Option<Res<crate::diagnostics::agent_capture::AgentCaptureConfig>>,
    mut cursor_q: Query<&mut CursorOptions, With<PrimaryWindow>>,
) {
    if agent.is_some() {
        return;
    }
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

/// Place the camera exactly on the terrain surface when entering Playing.
///
/// Uses the analytical terrain generator (no chunks needed) to ensure the
/// player starts at the correct height, even if minor drift occurred during
/// the loading phase.  Resets vertical velocity so gravity starts clean.
fn snap_to_surface(
    v2_gen: Option<Res<V2TerrainGen>>,
    planet: Res<PlanetConfig>,
    origin: Res<RenderOrigin>,
    mut cam_q: Query<(&mut WorldPosition, &mut Transform, &mut FpsCamera)>,
) {
    let Ok((mut world_pos, mut transform, mut cam)) = cam_q.single_mut() else {
        return;
    };

    if !planet.is_spherical() {
        return;
    }

    let tgen = v2_gen.as_ref().map(|r| r.0.as_ref());
    if let Some(tg) = tgen {
        let terrain_r = ground_height_from_terrain_gen(world_pos.0, tg);
        // Never place the player below sea level (ocean floor).
        let ground_r = terrain_r.max(planet.sea_level_radius as f32 + 1.0);
        let up = local_up_from_world_pos(world_pos.0);
        // Set WorldPosition in f64
        world_pos.0 =
            DVec3::new(up.x as f64, up.y as f64, up.z as f64) * (ground_r + EYE_HEIGHT) as f64;
        transform.translation = world_pos.render_offset(&origin);
        cam.vertical_velocity = 0.0;
        cam.grounded = true;
    }
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
    mut cam_q: Query<(&mut FpsCamera, &WorldPosition, &mut Transform)>,
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

    let Ok((mut cam, world_pos, mut transform)) = cam_q.single_mut() else {
        return;
    };

    cam.yaw -= delta.x * cam.sensitivity;
    cam.pitch -= delta.y * cam.sensitivity;
    cam.pitch = cam.pitch.clamp(-1.5, 1.5);

    let yaw_pitch =
        Quat::from_axis_angle(Vec3::Y, cam.yaw) * Quat::from_axis_angle(Vec3::X, cam.pitch);

    if planet.is_spherical() {
        let local_up = local_up_from_world_pos(world_pos.0);
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
    mut cam_q: Query<(&mut FpsCamera, &mut WorldPosition, &Transform)>,
) {
    let Ok(cursor) = cursor_q.single() else {
        return;
    };

    let Ok((mut cam, mut world_pos, transform)) = cam_q.single_mut() else {
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
            local_up_from_world_pos(world_pos.0)
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
            local_up_from_world_pos(world_pos.0)
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

    // Accumulate movement into WorldPosition (f64 precision)
    let delta = direction * effective_speed * time.delta_secs();
    world_pos.0 += DVec3::new(delta.x as f64, delta.y as f64, delta.z as f64);
}

/// Apply gravity and ground collision to the camera.
///
/// Gravity is radial (toward planet center). Ground check uses the V2 terrain
/// generator to sample the surface radius without requiring loaded chunks.
fn camera_gravity(
    time: Res<Time>,
    planet: Res<PlanetConfig>,
    v2_gen: Option<Res<V2TerrainGen>>,
    mut cam_q: Query<(&mut FpsCamera, &mut WorldPosition, &Transform)>,
) {
    let Ok((mut cam, mut world_pos, _)) = cam_q.single_mut() else {
        return;
    };

    if !cam.gravity_enabled {
        return;
    }

    let dt = time.delta_secs();

    if planet.is_spherical() {
        // Radial gravity: pull toward planet center.
        let local_up = local_up_from_world_pos(world_pos.0);

        cam.vertical_velocity -= constants::GRAVITY * dt;
        cam.vertical_velocity = cam.vertical_velocity.max(-200.0);
        let vert_delta = local_up * cam.vertical_velocity * dt;
        world_pos.0 += DVec3::new(
            vert_delta.x as f64,
            vert_delta.y as f64,
            vert_delta.z as f64,
        );

        // Ground collision: use world-space position for terrain sampling
        let ground_r = v2_gen
            .as_ref()
            .map(|tg| ground_height_from_terrain_gen(world_pos.0, &tg.0))
            // Never let the player fall below sea level.
            .map(|r| r.max(planet.sea_level_radius as f32 + 1.0));

        if let Some(ground_r) = ground_r {
            let feet_r = world_pos.0.length() - EYE_HEIGHT as f64;
            if feet_r <= ground_r as f64 {
                let up_f64 = world_pos.0.normalize_or(DVec3::Y);
                world_pos.0 = up_f64 * (ground_r + EYE_HEIGHT) as f64;
                cam.vertical_velocity = 0.0;
                cam.grounded = true;
            } else {
                cam.grounded = false;
            }
        } else {
            cam.grounded = false;
        }
    } else {
        // Flat mode: Y-axis gravity with no terrain collision (flat mode is
        // no longer supported in V2; player floats until spherical mode is used).
        cam.vertical_velocity -= constants::GRAVITY * dt;
        cam.vertical_velocity = cam.vertical_velocity.max(-200.0);
        world_pos.0.y += cam.vertical_velocity as f64 * dt as f64;
        cam.grounded = false;
    }
}

/// Derive camera `Transform.translation` from `WorldPosition - RenderOrigin`.
///
/// Runs after camera_move and camera_gravity so the render-space position
/// reflects the latest f64 world position. Rotation is already set by
/// camera_look.
fn sync_camera_transform(
    origin: Res<RenderOrigin>,
    mut cam_q: Query<(&WorldPosition, &mut Transform), With<FpsCamera>>,
) {
    for (world_pos, mut transform) in &mut cam_q {
        transform.translation = world_pos.render_offset(&origin);
    }
}

/// Toggle the player flashlight on/off when F is pressed.
///
/// The flashlight is a [`SpotLight`] child of the camera entity, marked with
/// [`Flashlight`].  Toggling sets its [`Visibility`] so the light is only
/// active when the player wants it.  A log message confirms the current state.
fn toggle_flashlight(
    key: Res<ButtonInput<KeyCode>>,
    mut cam_q: Query<&mut FpsCamera>,
    mut light_q: Query<&mut Visibility, With<Flashlight>>,
) {
    if !key.just_pressed(KeyCode::KeyF) {
        return;
    }
    let Ok(mut cam) = cam_q.single_mut() else {
        return;
    };
    cam.flashlight_enabled = !cam.flashlight_enabled;
    for mut vis in &mut light_q {
        *vis = if cam.flashlight_enabled {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
    }
    info!(
        "Flashlight {}",
        if cam.flashlight_enabled { "on" } else { "off" }
    );
}

/// Adjust flashlight intensity with Ctrl + `+` / `-` (or Numpad equivalents).
///
/// Each press multiplies or divides intensity by √2 (~1.41×), giving smooth
/// ≈1.5 dB steps.  Intensity is clamped to `[500, 500_000]` candela.
/// The new value is logged so the user knows where they are.
fn adjust_flashlight_intensity(
    key: Res<ButtonInput<KeyCode>>,
    mut cam_q: Query<&mut FpsCamera>,
    mut light_q: Query<&mut SpotLight, With<Flashlight>>,
) {
    let ctrl = key.pressed(KeyCode::ControlLeft) || key.pressed(KeyCode::ControlRight);
    if !ctrl {
        return;
    }

    let brighter = key.just_pressed(KeyCode::NumpadAdd) || key.just_pressed(KeyCode::Equal);
    let dimmer = key.just_pressed(KeyCode::NumpadSubtract) || key.just_pressed(KeyCode::Minus);

    if !brighter && !dimmer {
        return;
    }

    let Ok(mut cam) = cam_q.single_mut() else {
        return;
    };

    let factor = if brighter {
        2.0f32.sqrt()
    } else {
        1.0 / 2.0f32.sqrt()
    };
    cam.flashlight_intensity = (cam.flashlight_intensity * factor).clamp(500.0, 500_000.0);

    for mut spot in &mut light_q {
        spot.intensity = cam.flashlight_intensity;
    }

    info!("Flashlight intensity: {:.0} cd", cam.flashlight_intensity);
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
        assert!(
            max_speed >= 10_000.0,
            "max fly speed {max_speed} too low for 32km planet"
        );
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
        let up = local_up_from_world_pos(DVec3::new(0.0, 6_371_000.0, 0.0));
        assert!((up - Vec3::Y).length() < 1e-5);
    }

    #[test]
    fn local_up_at_equator_is_radial() {
        // At the equator (lon=0), position is along +X → local up = +X.
        let up = local_up_from_world_pos(DVec3::new(6_371_000.0, 0.0, 0.0));
        assert!((up - Vec3::X).length() < 1e-5);
    }

    #[test]
    fn local_up_at_45_lat() {
        // At 45° latitude (lon=0), local up should be (cos45, sin45, 0).
        let lat = std::f64::consts::FRAC_PI_4;
        let r = 6_371_000.0_f64;
        let pos = DVec3::new(r * lat.cos(), r * lat.sin(), 0.0);
        let up = local_up_from_world_pos(pos);
        let expected = Vec3::new(lat.cos() as f32, lat.sin() as f32, 0.0);
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
        let pos = DVec3::new(
            6_371_000.0 * (lat.cos() as f64),
            6_371_000.0 * (lat.sin() as f64),
            0.0,
        );
        let local_up = local_up_from_world_pos(pos);
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
        let up = local_up_from_world_pos(DVec3::new(1e-8, 1e-8, 1e-8));
        assert_eq!(up, Vec3::Y);
    }

    #[test]
    fn find_random_land_returns_above_sea_level() {
        use crate::world::planet::PlanetConfig;
        use crate::world::terrain::UnifiedTerrainGenerator;
        use std::sync::Arc;

        let planet = PlanetConfig {
            mean_radius: 32000.0,
            sea_level_radius: 28000.0, // Sea level well below mean so flat terrain is "land".
            height_scale: 0.0,
            ..Default::default()
        };
        let gen_cfg = crate::planet::PlanetConfig {
            seed: planet.seed as u64,
            grid_level: 3,
            ..Default::default()
        };
        let mut data = crate::planet::PlanetData::new(gen_cfg);
        // Set non-ocean biome so planetary sampler treats cells as land.
        for b in &mut data.biome {
            *b = crate::planet::BiomeType::TemperateForest;
        }
        for e in &mut data.elevation {
            *e = 500.0; // 500 m above sea level
        }
        let pd = Arc::new(data);
        let tgen = UnifiedTerrainGenerator::new(pd, planet.clone());
        let result = find_random_land(&tgen, planet.sea_level_radius, 2000, 42);
        assert!(result.is_some(), "should find land on a default planet");
        let (lat, lon) = result.unwrap();
        let r = tgen.sample_surface_radius_at(lat, lon);
        assert!(
            r > planet.sea_level_radius,
            "land point should be above sea level: r={r}, sea={}",
            planet.sea_level_radius,
        );
    }

    #[test]
    fn find_random_land_is_deterministic() {
        use crate::world::planet::PlanetConfig;
        use crate::world::terrain::UnifiedTerrainGenerator;
        use std::sync::Arc;

        let planet = PlanetConfig {
            height_scale: 4000.0,
            noise: Some(crate::world::noise::NoiseConfig::default()),
            ..Default::default()
        };
        let gen_cfg = crate::planet::PlanetConfig {
            seed: planet.seed as u64,
            grid_level: 3,
            ..Default::default()
        };
        let pd = Arc::new(crate::planet::PlanetData::new(gen_cfg.clone()));
        let tgen = UnifiedTerrainGenerator::new(pd, planet.clone());
        let pd2 = Arc::new(crate::planet::PlanetData::new(gen_cfg));
        let tgen2 = UnifiedTerrainGenerator::new(pd2, planet.clone());
        let a = find_random_land(&tgen, planet.sea_level_radius, 2000, 42);
        let b = find_random_land(&tgen2, planet.sea_level_radius, 2000, 42);
        assert_eq!(a, b, "same seed should produce same spawn");
    }

    #[test]
    fn find_coastline_has_water_neighbor() {
        use crate::world::planet::PlanetConfig;
        use crate::world::terrain::UnifiedTerrainGenerator;
        use std::sync::Arc;

        let planet = PlanetConfig {
            height_scale: 4000.0,
            noise: Some(crate::world::noise::NoiseConfig::default()),
            ..Default::default()
        };
        let gen_cfg = crate::planet::PlanetConfig {
            seed: planet.seed as u64,
            grid_level: 3,
            ..Default::default()
        };
        let pd = Arc::new(crate::planet::PlanetData::new(gen_cfg));
        let tgen = UnifiedTerrainGenerator::new(pd, planet.clone());
        let result = find_coastline(&tgen, planet.sea_level_radius, 10000, 42);
        if let Some((lat, lon)) = result {
            // The point itself should be land.
            let r = tgen.sample_surface_radius_at(lat, lon);
            assert!(
                r > planet.sea_level_radius,
                "coastline point should be land"
            );

            // At least one neighbor should be water.
            let probe = 0.003_f64;
            let offsets = [(probe, 0.0), (-probe, 0.0), (0.0, probe), (0.0, -probe)];
            let has_water = offsets.iter().any(|(dlat, dlon)| {
                let nr = tgen.sample_surface_radius_at(lat + dlat, lon + dlon);
                nr <= planet.sea_level_radius
            });
            assert!(has_water, "coastline point should have a water neighbor");
        }
        // Note: coastline may not be found on all terrain configs — that's OK.
    }

    #[test]
    fn spawn_location_default() {
        let loc = SpawnLocation {
            lat: std::f64::consts::FRAC_PI_4,
            lon: 0.0,
        };
        assert!((loc.lat.to_degrees() - 45.0).abs() < 0.01);
        assert_eq!(loc.lon, 0.0);
    }
}
