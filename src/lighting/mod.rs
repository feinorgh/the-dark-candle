// Day-night cycle and dynamic lighting.
//
// Rotates the sun (DirectionalLight) through a 24-hour cycle, adjusting color
// temperature, intensity, and ambient brightness. Exposes a SolarInsolation
// resource for future solar heating integration.

pub mod clouds;
pub mod light_map;
pub mod scattering;
pub mod shadows;
pub mod sky;

use bevy::prelude::*;
use std::f32::consts::{FRAC_PI_2, PI, TAU};

/// Marker component for the sun DirectionalLight entity.
#[derive(Component, Debug, Clone, Copy)]
pub struct Sun;

/// Marker component for the ambient light entity.
#[derive(Component, Debug, Clone, Copy)]
pub struct SkyAmbient;

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
            noon_ambient: 200.0,
            night_ambient: 10.0,
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
}

impl Default for ShadowConfig {
    fn default() -> Self {
        Self {
            angle_threshold_degrees: 2.0,
            shadow_samples: 3,
            cone_half_angle_degrees: 1.5,
            enabled: true,
        }
    }
}

/// Tracks the last sun angles used for shadow computation.
#[derive(Resource, Debug, Clone, Copy, Default)]
pub struct LastShadowAngles {
    pub elevation: f32,
    pub azimuth: f32,
}

/// System: advance the clock each frame.
fn advance_time(mut tod: ResMut<TimeOfDay>, config: Res<DayNightConfig>, time: Res<Time>) {
    let dt_hours = time.delta_secs() * config.time_scale / 3600.0;
    tod.0 = (tod.0 + dt_hours) % 24.0;
}

/// Sun elevation angle (radians) from time of day.
/// Sunrise at 06:00, zenith at 12:00, sunset at 18:00.
/// Returns negative values when below the horizon.
fn sun_elevation(hour: f32) -> f32 {
    // Map hour to angle: 6h = 0 (horizon), 12h = π/2 (zenith), 18h = π (horizon)
    // Using sine: elevation = sin((hour - 6) / 12 * π) * π/2
    // This gives 0 at 6h, π/2 at 12h, 0 at 18h, -π/2 at 0h/24h
    let day_fraction = (hour - 6.0) / 12.0;
    (day_fraction * PI).sin() * FRAC_PI_2
}

/// Sun azimuth angle (radians) — rotates 360° over 24 hours.
fn sun_azimuth(hour: f32) -> f32 {
    hour / 24.0 * TAU
}

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

/// System: update sun DirectionalLight rotation, color, and intensity.
fn update_sun(
    tod: Res<TimeOfDay>,
    config: Res<DayNightConfig>,
    mut insolation: ResMut<SolarInsolation>,
    mut sun_dir: ResMut<SunDirection>,
    mut sun_q: Query<(&mut DirectionalLight, &mut Transform), With<Sun>>,
) {
    let elevation = sun_elevation(tod.0);
    let azimuth = sun_azimuth(tod.0);

    // Insolation factor: cosine of zenith when above horizon, else 0.
    insolation.0 = elevation.sin().max(0.0);

    // Compute sun direction vector for terrain shadow casting.
    if elevation > 0.0 {
        sun_dir.0 = [
            azimuth.cos() * elevation.cos(),
            elevation.sin(),
            azimuth.sin() * elevation.cos(),
        ];
    } else {
        // Night: straight up (vertical shadows only).
        sun_dir.0 = [0.0, 1.0, 0.0];
    }

    for (mut light, mut transform) in &mut sun_q {
        // Sun direction: rotate around Y (azimuth) then tilt by elevation.
        // The light shines along -Z of its transform, so we point it downward
        // at the elevation angle.
        let pitch = -(FRAC_PI_2 - elevation);
        *transform = Transform::from_rotation(Quat::from_euler(EulerRot::YXZ, azimuth, pitch, 0.0));

        // Intensity: proportional to elevation above horizon.
        if elevation > 0.0 {
            light.illuminance = config.noon_illuminance * elevation.sin();
            light.color = sun_color(elevation);
        } else {
            // Sun below horizon — negligible moonlight via directional.
            light.illuminance = config.noon_illuminance * 0.02;
            light.color = sun_color(elevation);
        }
    }
}

/// System: scale ambient light brightness with sun elevation.
fn update_ambient(
    tod: Res<TimeOfDay>,
    config: Res<DayNightConfig>,
    mut ambient_q: Query<&mut AmbientLight, With<SkyAmbient>>,
) {
    let elevation = sun_elevation(tod.0);
    let factor = elevation.sin().max(0.0);

    // Lerp between night_ambient and noon_ambient
    let brightness = config.night_ambient + (config.noon_ambient - config.night_ambient) * factor;

    for mut ambient in &mut ambient_q {
        ambient.brightness = brightness;

        // Shift ambient color to match time of day
        ambient.color = if elevation <= 0.0 {
            Color::srgb(0.3, 0.35, 0.5) // blue-ish night
        } else if elevation < 0.3 {
            let t = elevation / 0.3;
            Color::srgb(0.6 + 0.4 * t, 0.65 + 0.35 * t, 0.7 + 0.3 * t)
        } else {
            Color::WHITE
        };
    }
}

/// Spawn the sun and ambient light entities.
fn spawn_lights(mut commands: Commands, config: Res<DayNightConfig>, tod: Res<TimeOfDay>) {
    let elevation = sun_elevation(tod.0);
    let azimuth = sun_azimuth(tod.0);
    let pitch = -(FRAC_PI_2 - elevation);
    let illuminance = if elevation > 0.0 {
        config.noon_illuminance * elevation.sin()
    } else {
        config.noon_illuminance * 0.02
    };

    commands.spawn((
        Sun,
        DirectionalLight {
            illuminance,
            shadows_enabled: true,
            color: sun_color(elevation),
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::YXZ, azimuth, pitch, 0.0)),
    ));

    let factor = elevation.sin().max(0.0);
    let brightness = config.night_ambient + (config.noon_ambient - config.night_ambient) * factor;

    commands.spawn((
        SkyAmbient,
        AmbientLight {
            color: Color::WHITE,
            brightness,
            ..default()
        },
    ));
}

/// System: recompute terrain shadows when the sun moves significantly.
fn update_terrain_shadows(
    sun_dir: Res<SunDirection>,
    shadow_config: Res<ShadowConfig>,
    mut last_angles: ResMut<LastShadowAngles>,
    tod: Res<TimeOfDay>,
    registry: Option<Res<crate::data::MaterialRegistry>>,
    mut chunk_q: Query<(
        &mut crate::world::chunk::Chunk,
        &mut light_map::ChunkLightMap,
    )>,
) {
    if !shadow_config.enabled {
        return;
    }
    let Some(_registry) = registry else { return };

    let elevation = sun_elevation(tod.0);
    let azimuth = sun_azimuth(tod.0);

    // Check angle threshold.
    let elev_delta = (elevation - last_angles.elevation).abs().to_degrees();
    let azim_delta = (azimuth - last_angles.azimuth).abs().to_degrees();
    if elev_delta < shadow_config.angle_threshold_degrees
        && azim_delta < shadow_config.angle_threshold_degrees
    {
        return;
    }

    last_angles.elevation = elevation;
    last_angles.azimuth = azimuth;

    for (mut chunk, mut light_map) in &mut chunk_q {
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

/// Plugin for the day-night cycle and dynamic lighting.
pub struct LightingPlugin;

impl Plugin for LightingPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TimeOfDay>()
            .init_resource::<DayNightConfig>()
            .init_resource::<SolarInsolation>()
            .init_resource::<SunDirection>()
            .init_resource::<ShadowConfig>()
            .init_resource::<LastShadowAngles>()
            .add_systems(Startup, spawn_lights)
            .add_systems(
                Update,
                (
                    advance_time,
                    update_sun,
                    update_ambient,
                    update_chunk_light_maps,
                    update_terrain_shadows,
                )
                    .chain(),
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
}
