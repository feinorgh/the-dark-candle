// Day-night cycle and dynamic lighting.
//
// Rotates the sun (DirectionalLight) through a 24-hour cycle, adjusting color
// temperature, intensity, and ambient brightness. Exposes a SolarInsolation
// resource for future solar heating integration.

pub mod aurora;
pub mod caustic_tile;
pub mod caustics;
pub mod clouds;
pub mod light_map;
pub mod mie_local;
pub mod optics;
pub mod orbital;
pub mod refraction;
pub mod scattering;
pub mod shadows;
pub mod sky;
pub mod sky_dome;
pub mod terrain_caustic_material;

use bevy::pbr::DistanceFog;
use bevy::pbr::MaterialPlugin;
use bevy::prelude::*;

use crate::world::planet::PlanetConfig;
use sky_dome::GameElapsedSeconds;

/// Marker component for the sun DirectionalLight entity.
#[derive(Component, Debug, Clone, Copy)]
pub struct Sun;

/// Current time of day in hours (0.0–24.0).
#[derive(Resource, Debug, Clone, Copy)]
pub struct TimeOfDay(pub f32);

impl Default for TimeOfDay {
    fn default() -> Self {
        // Start at 10:00 (pleasant morning light)
        Self(10.0)
    }
}

/// Configuration for the day-night cycle.
#[derive(Resource, Debug, Clone, Copy)]
pub struct DayNightConfig {
    /// Game-hours per real-second. Default ~72 → 20 real minutes = 1 game day.
    pub time_scale: f32,
    /// Peak sun illuminance at noon (lux).
    pub noon_illuminance: f32,
    /// Ambient brightness at noon.
    pub noon_ambient: f32,
    /// Ambient brightness at midnight (minimum floor, ~5% of noon).
    pub night_ambient: f32,
}

impl Default for DayNightConfig {
    fn default() -> Self {
        Self {
            time_scale: 72.0,
            noon_illuminance: 10_000.0,
            // Reduced from 200/10: the Atmosphere component now provides
            // sky-based IBL, so flat ambient is fill-only for shadowed areas.
            noon_ambient: 80.0,
            night_ambient: 5.0,
        }
    }
}

/// Computed surface insolation factor (0.0–1.0). Exposed for future solar
/// heating (Phase 9). Represents the cosine of the sun's zenith angle when
/// the sun is above the horizon, 0.0 otherwise.
#[derive(Resource, Debug, Clone, Copy, Default)]
pub struct SolarInsolation(pub f32);

/// Unit vector pointing toward the sun in world space.
/// Computed from TimeOfDay each frame. When sun is below horizon,
/// defaults to straight up (vertical shadows only).
#[derive(Resource, Debug, Clone, Copy)]
pub struct SunDirection(pub [f32; 3]);

impl Default for SunDirection {
    fn default() -> Self {
        Self([0.0, 1.0, 0.0])
    }
}

/// Unit vector pointing toward the star in planet body-frame coordinates.
///
/// Unlike `SunDirection` (which is in the observer's local tangent frame),
/// this is the same for the entire planet and produces physically correct
/// day/night hemispheres when used as the `DirectionalLight` direction.
#[derive(Resource, Debug, Clone, Copy)]
pub struct SunWorldDirection(pub bevy::math::DVec3);

impl Default for SunWorldDirection {
    fn default() -> Self {
        Self(bevy::math::DVec3::new(0.0, 1.0, 0.0))
    }
}

/// Configuration for terrain shadow casting.
#[derive(Resource, Debug, Clone, Copy)]
pub struct ShadowConfig {
    /// Minimum sun elevation change (degrees) before recomputing shadows.
    pub angle_threshold_degrees: f32,
    /// Number of jittered rays for soft shadow edges. 1 = hard shadows.
    pub shadow_samples: u32,
    /// Half-angle of the jitter cone in degrees.
    pub cone_half_angle_degrees: f32,
    /// Whether terrain shadows are enabled.
    pub enabled: bool,
    /// Maximum number of chunk shadow updates per frame. 0 = unlimited.
    /// Spreads shadow recomputation across multiple frames when the sun
    /// angle crosses the threshold, avoiding a single-frame spike.
    pub max_updates_per_frame: usize,
}

impl Default for ShadowConfig {
    fn default() -> Self {
        Self {
            angle_threshold_degrees: 2.0,
            shadow_samples: 3,
            cone_half_angle_degrees: 1.5,
            enabled: true,
            max_updates_per_frame: 8,
        }
    }
}

/// Tracks the last sun angles used for shadow computation.
#[derive(Resource, Debug, Clone, Copy, Default)]
pub struct LastShadowAngles {
    pub elevation: f32,
    pub azimuth: f32,
}

/// Entities awaiting shadow recomputation, filled when the sun angle
/// crosses the threshold and drained over multiple frames.
#[derive(Resource, Default)]
struct PendingShadowUpdates {
    entities: Vec<Entity>,
}

/// System: advance the orbital state, sync TimeOfDay, and accumulate game time.
fn advance_time(
    time: Res<Time>,
    mut orbital_state: ResMut<orbital::OrbitalState>,
    mut tod: ResMut<TimeOfDay>,
    mut elapsed: ResMut<GameElapsedSeconds>,
) {
    let real_dt = time.delta_secs_f64();
    let game_dt = real_dt * orbital_state.time_scale;

    let day_length = 86_400.0;
    let rotation_rate = std::f64::consts::TAU / day_length;
    orbital_state.rotation_angle =
        (orbital_state.rotation_angle + rotation_rate * game_dt) % std::f64::consts::TAU;

    let year_length = day_length * orbital_state.year_in_days;
    let orbital_rate = std::f64::consts::TAU / year_length;
    orbital_state.orbital_angle =
        (orbital_state.orbital_angle + orbital_rate * game_dt) % std::f64::consts::TAU;

    // Sync TimeOfDay from rotation angle for backward compatibility.
    tod.0 = orbital::time_of_day_from_rotation(orbital_state.rotation_angle);

    // Accumulate elapsed game-time for Keplerian moon orbit calculations.
    elapsed.0 += game_dt;
}

/// Sun elevation angle (radians) from time of day.
/// Color temperature of sunlight based on elevation.
/// Dawn/dusk: warm orange (~3500 K), noon: neutral white (~6500 K),
/// night: cool blue moonlight.
fn sun_color(elevation: f32) -> Color {
    if elevation <= 0.0 {
        // Night: cool blue moonlight
        Color::srgb(0.3, 0.35, 0.5)
    } else if elevation < 0.3 {
        // Dawn/dusk: warm orange, blending to neutral
        let t = elevation / 0.3;
        Color::srgb(1.0, 0.7 + 0.3 * t, 0.4 + 0.55 * t)
    } else {
        // Daytime: neutral warm white
        Color::srgb(1.0, 1.0, 0.95)
    }
}

/// System: update sun DirectionalLight from world-space sun direction.
///
/// The `DirectionalLight` direction is set to the planet-frame sun vector,
/// producing physically correct day/night hemispheres. Local elevation
/// (for illuminance scaling and color temperature) is derived from the
/// dot product of the sun direction with the observer's radial "up" vector.
#[allow(clippy::too_many_arguments)]
fn update_sun(
    orbital_state: Res<orbital::OrbitalState>,
    planet: Res<PlanetConfig>,
    config: Res<DayNightConfig>,
    mut insolation: ResMut<SolarInsolation>,
    mut sun_dir: ResMut<SunDirection>,
    mut sun_world: ResMut<SunWorldDirection>,
    mut sun_q: Query<(&mut DirectionalLight, &mut Transform), With<Sun>>,
    cam_q: Query<&crate::floating_origin::WorldPosition, With<crate::camera::FpsCamera>>,
) {
    let world_dir = orbital::compute_sun_direction_world(
        &orbital_state,
        planet.axial_tilt,
        planet.libration_amplitude,
        planet.libration_period,
    );
    sun_world.0 = world_dir;

    // Derive local elevation from the camera's position on the planet.
    // elevation = angle between sun direction and the observer's local horizon,
    // which equals asin(dot(sun_dir, observer_up)).
    let observer_up = cam_q
        .iter()
        .next()
        .map(|wp| wp.0.normalize_or(bevy::math::DVec3::Y))
        .unwrap_or(bevy::math::DVec3::Y);

    let sin_elevation = world_dir.dot(observer_up);
    let elevation = (sin_elevation.clamp(-1.0, 1.0) as f32).asin();

    // Insolation factor: sine of elevation when above horizon, else 0.
    insolation.0 = elevation.sin().max(0.0);

    // Update sun direction resource for terrain shadows (world-space).
    let wd = world_dir.as_vec3();
    if elevation > 0.0 {
        sun_dir.0 = [wd.x, wd.y, wd.z];
    } else {
        sun_dir.0 = [0.0, 1.0, 0.0];
    }

    for (mut light, mut transform) in &mut sun_q {
        // DirectionalLight direction is world-space — the terminator emerges
        // naturally from the dot product of light direction with surface normals.
        let sun_vec = wd;
        // Use Vec3::Z as fallback up-hint to avoid a degenerate looking_at when
        // sun_vec is exactly ±Y (axial tilt edge case).
        let up_hint = if sun_vec.abs().dot(Vec3::Y) > 0.999 {
            Vec3::Z
        } else {
            Vec3::Y
        };
        *transform = Transform::default().looking_at(-sun_vec, up_hint);

        if elevation > 0.0 {
            light.illuminance = config.noon_illuminance * elevation.sin();
            light.color = sun_color(elevation);
        } else {
            light.illuminance = config.noon_illuminance * 0.02;
            light.color = sun_color(elevation);
        }
    }
}

/// System: scale ambient light brightness with local sun elevation.
///
/// In spherical mode, applies a boost factor to compensate for the absence of
/// Atmosphere-driven image-based lighting.
///
/// Updates `GlobalAmbientLight`, which is the ambient that applies to the
/// player camera (the camera entity has no per-camera `AmbientLight` override).
fn update_ambient(
    planet: Res<PlanetConfig>,
    config: Res<DayNightConfig>,
    sun_world: Res<SunWorldDirection>,
    mut global_ambient: ResMut<GlobalAmbientLight>,
    cam_q: Query<&crate::floating_origin::WorldPosition, With<crate::camera::FpsCamera>>,
) {
    let observer_up = cam_q
        .iter()
        .next()
        .map(|wp| wp.0.normalize_or(bevy::math::DVec3::Y))
        .unwrap_or(bevy::math::DVec3::Y);

    let sin_elevation = sun_world.0.dot(observer_up);
    let elevation = (sin_elevation.clamp(-1.0, 1.0) as f32).asin();

    let factor = elevation.sin().max(0.0);
    let mut brightness =
        config.night_ambient + (config.noon_ambient - config.night_ambient) * factor;

    // Without the Atmosphere shader, there is no IBL contribution. Boost
    // ambient so spherical-mode scenes don't appear unnaturally dark.
    if planet.is_spherical() {
        brightness *= 2.5;
    }

    global_ambient.brightness = brightness;
    global_ambient.color = if elevation <= 0.0 {
        Color::srgb(0.3, 0.35, 0.5)
    } else if elevation < 0.3 {
        let t = elevation / 0.3;
        Color::srgb(0.6 + 0.4 * t, 0.65 + 0.35 * t, 0.7 + 0.3 * t)
    } else {
        Color::WHITE
    };
}

/// System: update distance fog color to match local time of day.
fn update_fog(
    sun_world: Res<SunWorldDirection>,
    mut fog_q: Query<&mut DistanceFog>,
    cam_q: Query<&crate::floating_origin::WorldPosition, With<crate::camera::FpsCamera>>,
) {
    let observer_up = cam_q
        .iter()
        .next()
        .map(|wp| wp.0.normalize_or(bevy::math::DVec3::Y))
        .unwrap_or(bevy::math::DVec3::Y);

    let sin_elevation = sun_world.0.dot(observer_up);
    let elevation = (sin_elevation.clamp(-1.0, 1.0) as f32).asin();

    let fog_color = if elevation <= 0.0 {
        // Night: dark blue-gray
        Color::srgb(0.05, 0.06, 0.1)
    } else if elevation < 0.3 {
        // Dawn/dusk: warm haze
        let t = elevation / 0.3;
        Color::srgb(0.4 + 0.3 * t, 0.35 + 0.43 * t, 0.3 + 0.6 * t)
    } else {
        // Day: light atmospheric haze
        Color::srgb(0.7, 0.78, 0.9)
    };

    for mut fog in &mut fog_q {
        fog.color = fog_color;
    }
}

/// Spawn the sun DirectionalLight and set the initial global ambient.
///
/// `AmbientLight` is NOT spawned on a separate entity: in Bevy 0.18,
/// `AmbientLight` carries `#[require(Camera)]`, so spawning it on a
/// non-camera entity auto-adds a phantom camera that conflicts with the
/// player camera (same render order / target → black screen).
///
/// `GlobalAmbientLight` is a `Resource` that applies to all cameras that
/// do not have a per-camera `AmbientLight` override — which is the player
/// camera.  It is kept up to date each frame by `update_ambient`.
fn spawn_lights(
    mut commands: Commands,
    config: Res<DayNightConfig>,
    orbital_state: Res<orbital::OrbitalState>,
    planet: Res<PlanetConfig>,
    mut global_ambient: ResMut<GlobalAmbientLight>,
) {
    let world_dir = orbital::compute_sun_direction_world(
        &orbital_state,
        planet.axial_tilt,
        planet.libration_amplitude,
        planet.libration_period,
    );

    // At startup, assume observer at north pole for initial elevation estimate.
    let sin_elevation = world_dir.y; // dot with Y-up
    let elevation = (sin_elevation.clamp(-1.0, 1.0) as f32).asin();

    let illuminance = if elevation > 0.0 {
        config.noon_illuminance * elevation.sin()
    } else {
        config.noon_illuminance * 0.02
    };

    let sun_vec = world_dir.as_vec3();
    // Use Vec3::Z as fallback up-hint to avoid a degenerate looking_at when
    // sun_vec is exactly ±Y (which would make Y an invalid up vector).
    let up_hint = if sun_vec.abs().dot(Vec3::Y) > 0.999 {
        Vec3::Z
    } else {
        Vec3::Y
    };
    let sun_transform = Transform::default().looking_at(-sun_vec, up_hint);

    commands.spawn((
        Sun,
        DirectionalLight {
            illuminance,
            shadows_enabled: true,
            color: sun_color(elevation),
            ..default()
        },
        sun_transform,
    ));

    // Set the initial global ambient to match the starting time of day.
    let factor = elevation.sin().max(0.0);
    let mut brightness =
        config.night_ambient + (config.noon_ambient - config.night_ambient) * factor;
    if planet.is_spherical() {
        brightness *= 2.5;
    }
    global_ambient.brightness = brightness;
    global_ambient.color = Color::WHITE;
}

/// System: recompute terrain shadows when the sun moves significantly.
///
/// When the sun angle crosses the configured threshold, all chunk entities
/// are queued into `PendingShadowUpdates`. Each frame, at most
/// `max_updates_per_frame` chunks are recomputed, spreading the cost.
#[allow(clippy::too_many_arguments)]
fn update_terrain_shadows(
    sun_dir: Res<SunDirection>,
    shadow_config: Res<ShadowConfig>,
    mut last_angles: ResMut<LastShadowAngles>,
    orbital_state: Res<orbital::OrbitalState>,
    registry: Option<Res<crate::data::MaterialRegistry>>,
    mut pending: ResMut<PendingShadowUpdates>,
    mut chunk_q: Query<(
        Entity,
        &mut crate::world::chunk::Chunk,
        &mut light_map::ChunkLightMap,
    )>,
) {
    if !shadow_config.enabled {
        return;
    }
    let Some(_registry) = registry else { return };

    let sun_world_dir = sun_dir.0;
    let elevation = sun_world_dir[1].asin(); // Y component ≈ sin(elevation) for terrain
    let azimuth = orbital_state.rotation_angle as f32;

    // Check angle threshold — enqueue all chunks when crossed.
    let elev_delta = (elevation - last_angles.elevation).abs().to_degrees();
    let azim_delta = (azimuth - last_angles.azimuth).abs().to_degrees();
    if elev_delta >= shadow_config.angle_threshold_degrees
        || azim_delta >= shadow_config.angle_threshold_degrees
    {
        last_angles.elevation = elevation;
        last_angles.azimuth = azimuth;

        pending.entities.clear();
        pending
            .entities
            .extend(chunk_q.iter().map(|(entity, _, _)| entity));
    }

    // Process a batch of pending shadow updates this frame.
    let limit = if shadow_config.max_updates_per_frame == 0 {
        pending.entities.len()
    } else {
        shadow_config
            .max_updates_per_frame
            .min(pending.entities.len())
    };
    let batch: Vec<Entity> = pending.entities.drain(..limit).collect();

    for entity in batch {
        if let Ok((_, mut chunk, mut light_map)) = chunk_q.get_mut(entity) {
            shadows::compute_terrain_shadows(
                chunk.voxels(),
                light_map.size(),
                sun_dir.0,
                shadow_config.shadow_samples,
                shadow_config.cone_half_angle_degrees,
                &mut light_map,
            );
            chunk.mark_dirty();
        }
    }
}

/// Plugin for the day-night cycle and dynamic lighting.
pub struct LightingPlugin;

impl Plugin for LightingPlugin {
    fn build(&self, app: &mut App) {
        // Black clear color — the sky dome material renders the sky over it.
        app.insert_resource(ClearColor(Color::BLACK));

        // Start at solar noon (rotation_angle = π) so the player always spawns
        // in full daylight.  Time scale 6× gives ~30 min of real-time before
        // dusk; press `]` to speed up or `[` to slow down, `\` resets to 72×.
        app.insert_resource(orbital::OrbitalState {
            rotation_angle: std::f64::consts::PI,
            time_scale: 6.0,
            ..Default::default()
        });

        // Register the sky dome material type.
        app.add_plugins(MaterialPlugin::<sky_dome::SkyMaterial>::default());

        // Register the aurora material type.
        app.add_plugins(aurora::aurora_material_plugin());
        app.init_resource::<aurora::AuroraMaterialHandle>();

        // Register the terrain caustic ExtendedMaterial.  Used by the v2
        // chunk manager, pairwise LOD-seam stitches, and corner caps for the
        // underwater caustic-light contribution.
        app.add_plugins(MaterialPlugin::<
            terrain_caustic_material::TerrainCausticMaterial,
        >::default());
        app.init_resource::<terrain_caustic_material::TerrainCausticHandles>();

        app.init_resource::<TimeOfDay>()
            .init_resource::<DayNightConfig>()
            .init_resource::<SolarInsolation>()
            .init_resource::<SunDirection>()
            .init_resource::<SunWorldDirection>()
            .init_resource::<ShadowConfig>()
            .init_resource::<LastShadowAngles>()
            .init_resource::<PendingShadowUpdates>()
            .init_resource::<GameElapsedSeconds>()
            .add_systems(Startup, (spawn_lights, sky_dome::spawn_sky_dome))
            .add_systems(
                Update,
                (
                    advance_time,
                    orbital::time_acceleration_input,
                    update_sun,
                    update_ambient,
                    update_fog.run_if(crate::diagnostics::debug_render::atmosphere_enabled),
                    sky_dome::update_sky_material
                        .run_if(crate::diagnostics::debug_render::atmosphere_enabled),
                    sky_dome::spawn_moon_billboards,
                    sky_dome::update_moon_positions,
                    update_chunk_light_maps,
                    refraction::update_chunk_refraction_maps,
                    update_terrain_shadows,
                )
                    .chain()
                    // Light-map and shadow systems access chunk entities with
                    // deferred commands, so they must run after chunk despawn
                    // commands have been flushed.
                    .after(crate::world::WorldSet::ChunkManagement),
            )
            .add_systems(
                PostUpdate,
                (
                    sky_dome::anchor_sky_dome_to_camera,
                    terrain_caustic_material::update_terrain_caustic_uniform,
                )
                    .after(bevy::transform::TransformSystems::Propagate),
            );
    }
}

/// System: recompute per-voxel sunlight for dirty chunks.
///
/// Runs after sun updates so the light direction is current. Only processes
/// chunks that are dirty (voxels changed). Requires a `MaterialRegistry` to
/// look up per-material absorption coefficients.
fn update_chunk_light_maps(
    mut commands: Commands,
    registry: Option<Res<crate::data::MaterialRegistry>>,
    sun_dir: Res<SunDirection>,
    shadow_config: Res<ShadowConfig>,
    chunk_q: Query<(Entity, &crate::world::chunk::Chunk), Changed<crate::world::chunk::Chunk>>,
) {
    let Some(registry) = registry else { return };
    for (entity, chunk) in &chunk_q {
        // Skip chunks that were only touched by update_terrain_shadows
        // (mark_dirty triggers Changed<Chunk> but voxels didn't actually change).
        if !chunk.is_dirty() {
            continue;
        }
        let mut lm = light_map::propagate_sunlight_from_registry(
            chunk.voxels(),
            crate::world::chunk::CHUNK_SIZE,
            &registry,
        );
        if shadow_config.enabled {
            shadows::compute_terrain_shadows(
                chunk.voxels(),
                crate::world::chunk::CHUNK_SIZE,
                sun_dir.0,
                shadow_config.shadow_samples,
                shadow_config.cone_half_angle_degrees,
                &mut lm,
            );
        }
        commands.entity(entity).insert(lm);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{FRAC_PI_2, TAU};

    /// Old sinusoidal sun elevation (kept for legacy tests).
    fn sun_elevation(hour: f32) -> f32 {
        use std::f32::consts::PI;
        let day_fraction = (hour - 6.0) / 12.0;
        (day_fraction * PI).sin() * FRAC_PI_2
    }

    /// Old sinusoidal sun azimuth (kept for legacy tests).
    fn sun_azimuth(hour: f32) -> f32 {
        hour / 24.0 * TAU
    }

    #[test]
    fn sun_elevation_at_noon_is_zenith() {
        let e = sun_elevation(12.0);
        assert!(
            (e - FRAC_PI_2).abs() < 0.01,
            "Noon elevation should be ~π/2, got {e}"
        );
    }

    #[test]
    fn sun_elevation_at_sunrise_is_zero() {
        let e = sun_elevation(6.0);
        assert!(e.abs() < 0.01, "Sunrise elevation should be ~0, got {e}");
    }

    #[test]
    fn sun_elevation_at_sunset_is_zero() {
        let e = sun_elevation(18.0);
        assert!(e.abs() < 0.01, "Sunset elevation should be ~0, got {e}");
    }

    #[test]
    fn sun_elevation_at_midnight_is_negative() {
        let e = sun_elevation(0.0);
        assert!(e < 0.0, "Midnight elevation should be negative, got {e}");
    }

    #[test]
    fn sun_azimuth_wraps_full_circle() {
        let a0 = sun_azimuth(0.0);
        let a24 = sun_azimuth(24.0);
        assert!(a0.abs() < 0.01, "Azimuth at 0h should be ~0");
        assert!((a24 - TAU).abs() < 0.01, "Azimuth at 24h should be ~2π");
    }

    #[test]
    fn solar_insolation_ranges() {
        // Above horizon
        let e_noon = sun_elevation(12.0);
        assert!(
            e_noon.sin().max(0.0) > 0.9,
            "Noon insolation should be ~1.0"
        );

        // Below horizon
        let e_midnight = sun_elevation(0.0);
        assert!(
            e_midnight.sin().max(0.0) < 0.01,
            "Midnight insolation should be ~0"
        );
    }

    #[test]
    fn day_night_config_defaults_are_sensible() {
        let cfg = DayNightConfig::default();
        assert!(cfg.time_scale > 0.0);
        assert!(cfg.noon_illuminance > 0.0);
        assert!(cfg.noon_ambient > cfg.night_ambient);
        assert!(cfg.night_ambient > 0.0, "Night should not be pitch black");
    }

    #[test]
    fn sun_color_night_is_cool() {
        let c = sun_color(-0.5);
        // Should be bluish
        let srgba = c.to_srgba();
        assert!(
            srgba.blue > srgba.red,
            "Night sun color should be cool/blue"
        );
    }

    #[test]
    fn sun_color_noon_is_warm_white() {
        let c = sun_color(FRAC_PI_2);
        let srgba = c.to_srgba();
        assert!(srgba.red > 0.9, "Noon should be near-white red channel");
        assert!(srgba.green > 0.9, "Noon should be near-white green channel");
    }

    #[test]
    fn sun_direction_at_noon_points_up() {
        let elevation = sun_elevation(12.0);
        let azimuth = sun_azimuth(12.0);
        let dir = [
            azimuth.cos() * elevation.cos(),
            elevation.sin(),
            azimuth.sin() * elevation.cos(),
        ];
        assert!(
            dir[1] > 0.99,
            "Noon sun should point nearly straight up, got {dir:?}"
        );
    }

    #[test]
    fn sun_direction_at_dawn_is_low() {
        let elevation = sun_elevation(7.0);
        let azimuth = sun_azimuth(7.0);
        let dir = [
            azimuth.cos() * elevation.cos(),
            elevation.sin(),
            azimuth.sin() * elevation.cos(),
        ];
        assert!(
            dir[1] > 0.0 && dir[1] < 0.5,
            "Dawn sun should be low, y={}",
            dir[1]
        );
    }

    #[test]
    fn shadow_config_defaults_are_sensible() {
        let cfg = ShadowConfig::default();
        assert!(cfg.angle_threshold_degrees > 0.0);
        assert!(cfg.shadow_samples >= 1);
        assert!(cfg.enabled);
    }

    // ── Spatial sun direction integration tests ──

    #[test]
    fn opposite_hemispheres_see_opposite_elevations() {
        // At noon (rotation=π, no tilt), sun points toward +X.
        // Observer on +X side (subsolar) should see positive elevation.
        // Observer on −X side (midnight) should see negative elevation.
        let state = orbital::OrbitalState {
            rotation_angle: std::f64::consts::PI,
            orbital_angle: 0.0,
            ..Default::default()
        };
        let sun_dir = orbital::compute_sun_direction_world(&state, 0.0, 0.0, 0.0);

        let day_up = bevy::math::DVec3::X; // observer on +X surface
        let night_up = bevy::math::DVec3::NEG_X; // observer on -X surface

        let day_elev = sun_dir.dot(day_up);
        let night_elev = sun_dir.dot(night_up);

        assert!(
            day_elev > 0.5,
            "Day-side should see sun high, got {day_elev}"
        );
        assert!(
            night_elev < -0.5,
            "Night-side should see sun below horizon, got {night_elev}"
        );
    }

    #[test]
    fn subsolar_point_sees_near_zenith() {
        // Sun at noon, no tilt → sun direction ≈ +X.
        // Observer at +X (equator, reference meridian) should see ~90° elevation.
        let state = orbital::OrbitalState {
            rotation_angle: std::f64::consts::PI,
            orbital_angle: 0.0,
            ..Default::default()
        };
        let sun_dir = orbital::compute_sun_direction_world(&state, 0.0, 0.0, 0.0);
        let observer_up = bevy::math::DVec3::X;
        let sin_elev = sun_dir.dot(observer_up);
        let elev_deg = sin_elev.asin().to_degrees();
        assert!(
            (elev_deg - 90.0).abs() < 2.0,
            "Subsolar observer should see ~90° elevation, got {elev_deg:.1}°"
        );
    }

    #[test]
    fn terminator_observer_sees_near_zero_elevation() {
        // Sun at noon → +X direction. Observer at +Z is 90° away (on the terminator).
        let state = orbital::OrbitalState {
            rotation_angle: std::f64::consts::PI,
            orbital_angle: 0.0,
            ..Default::default()
        };
        let sun_dir = orbital::compute_sun_direction_world(&state, 0.0, 0.0, 0.0);
        let observer_up = bevy::math::DVec3::Z; // 90° from subsolar point
        let sin_elev = sun_dir.dot(observer_up);
        let elev_deg = sin_elev.asin().to_degrees();
        assert!(
            elev_deg.abs() < 2.0,
            "Terminator observer should see ~0° elevation, got {elev_deg:.1}°"
        );
    }

    #[test]
    fn sun_world_direction_resource_defaults() {
        let swd = SunWorldDirection::default();
        assert!((swd.0.length() - 1.0).abs() < 1e-10);
    }
}
