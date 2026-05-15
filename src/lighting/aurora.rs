//! Aurora rendering: dedicated outer-shell sphere mesh with a custom additive
//! material that integrates emission over the auroral oval.
//!
//! Architecture overview lives in
//! `docs/superpowers/specs/2026-05-14-aurora-design.md`.
//!
//! The Rust side of this module owns three roles:
//!   1. `AuroraMaterial` — Bevy `Material` with `AlphaMode::Add`, backed by
//!      `assets/shaders/aurora.wgsl`.
//!   2. Systems — `spawn_aurora_shell` (Startup), `update_aurora_material`
//!      (Update), `anchor_aurora_shell_to_planet` (PostUpdate after
//!      `TransformSystems::Propagate`).
//!   3. CPU oracle helpers (`magnetic_north_axis`, `aurora_band_mask`,
//!      `aurora_day_side_factor`) — mirror the WGSL logic and serve as the
//!      test fixtures for the algorithm.

use bevy::camera::visibility::NoFrustumCulling;
use bevy::math::{DVec3, Vec3};
use bevy::pbr::{Material, MaterialPlugin};
use bevy::prelude::*;
use bevy::render::render_resource::AsBindGroup;
use bevy::shader::ShaderRef;

use crate::floating_origin::RenderOrigin;

// ── AuroraMaterial ─────────────────────────────────────────────────────────────

/// Marker component for the aurora shell entity.
#[derive(Component, Debug, Clone, Copy)]
pub struct AuroraShell;

/// Handle to the single shared `AuroraMaterial`. Stored as a resource so the
/// update system can mutate the material in place each frame.
#[derive(Resource, Debug, Clone, Default)]
pub struct AuroraMaterialHandle(pub Handle<AuroraMaterial>);

/// Additive volumetric aurora material. Backed by `shaders/aurora.wgsl`.
#[derive(Asset, bevy::reflect::TypePath, AsBindGroup, Debug, Clone)]
pub struct AuroraMaterial {
    /// Planet center in render-space (= `-RenderOrigin.0` cast to f32). w unused.
    #[uniform(0)]
    pub planet_center_render: Vec4,
    /// Magnetic-north unit vector in render-space. xyz used, w unused.
    #[uniform(1)]
    pub magnetic_north_axis: Vec4,
    /// Sun direction (from surface toward sun) in render-space, unit. xyz used.
    #[uniform(2)]
    pub sun_world_direction: Vec4,
    /// x = r_inner (m), y = r_outer (m), z = strength, w = elapsed time (s).
    #[uniform(3)]
    pub params: Vec4,
    /// x = band centre lat (rad), y = half-width (rad), z = curtain freq,
    /// w = curtain animation speed (rad/s).
    #[uniform(4)]
    pub band: Vec4,
}

impl Material for AuroraMaterial {
    fn vertex_shader() -> ShaderRef {
        "shaders/aurora.wgsl".into()
    }
    fn fragment_shader() -> ShaderRef {
        "shaders/aurora.wgsl".into()
    }
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Add
    }
}

/// Plugin glue helper — call from `LightingPlugin::build` to register the
/// material asset type so `MeshMaterial3d<AuroraMaterial>` can be spawned.
pub fn aurora_material_plugin() -> MaterialPlugin<AuroraMaterial> {
    MaterialPlugin::<AuroraMaterial>::default()
}

/// Compute the magnetic-north unit vector in the same frame as
/// `planet_north_axis`, given a `(lat_offset_deg, lon_offset_deg)` tilt.
///
/// The tilt is built in an orthonormal basis derived from `planet_north_axis`:
///   - `up    = planet_north_axis.normalize()`
///   - `east  = up × X̂` (or `up × Ẑ` when `up` is near-X)
///   - `south = east × up`
///
/// `lat_offset_deg` is the tilt magnitude: at `lat = 0` the result equals
/// `up`; at `lat = 90°` it lies in the equatorial plane.
///
/// `lon_offset_deg` selects the tilt direction in the equatorial plane:
///   - `0°`  → `east`
///   - `90°` → `south`
///   - and rotates right-handed around `up` thereafter.
///
/// The convention of `east`/`south` directions is body-frame-dependent (it
/// follows the orthonormal-basis construction above); planets that want a
/// specific geographic alignment of their magnetic offset should choose
/// `lon_offset_deg` accordingly. `planet_north_axis` MUST be a unit vector.
pub fn magnetic_north_axis(
    planet_north_axis: DVec3,
    lat_offset_deg: f64,
    lon_offset_deg: f64,
) -> DVec3 {
    let up = planet_north_axis.normalize();
    let arb = if up.x.abs() < 0.9 { DVec3::X } else { DVec3::Z };
    let east = up.cross(arb).normalize();
    let south = east.cross(up).normalize();

    let lat = lat_offset_deg.to_radians();
    let lon = lon_offset_deg.to_radians();
    let tilt_dir = east * lon.cos() + south * lon.sin();
    (up * lat.cos() + tilt_dir * lat.sin()).normalize()
}

/// f32 convenience wrapper for the GPU uniform path.
pub fn magnetic_north_axis_f32(
    planet_north_axis: Vec3,
    lat_offset_deg: f32,
    lon_offset_deg: f32,
) -> Vec3 {
    magnetic_north_axis(
        planet_north_axis.as_dvec3(),
        lat_offset_deg as f64,
        lon_offset_deg as f64,
    )
    .as_vec3()
}

/// Smoothstep "band" mask in magnetic latitude space.
///
/// Returns `1.0` at `|lat_mag| == band_center_rad` and falls smoothly to `0.0`
/// outside `[band_center_rad - 2·half_width, band_center_rad + 2·half_width]`.
/// At the half-width edge the value is exactly `0.5`, mirroring a triangular
/// `smoothstep` peak: `1 - smoothstep(0, 2·half_width, |delta|)` where
/// `delta = |lat_mag| - band_center_rad`.
pub fn aurora_band_mask(lat_mag_rad: f64, band_center_rad: f64, half_width_rad: f64) -> f64 {
    let delta = (lat_mag_rad.abs() - band_center_rad).abs();
    let edge1 = 2.0 * half_width_rad;
    let t = (delta / edge1).clamp(0.0, 1.0);
    let smooth = t * t * (3.0 - 2.0 * t);
    1.0 - smooth
}

/// Day-side gate: `0.0` when the sun is overhead, `1.0` when the sun is below
/// the local horizon by a few degrees. Soft ramp around the horizon.
///
/// Linear ramp: `0` at `cos_zenith = 0.1` (sun ~5.7° above horizon),
/// `0.5` at `cos_zenith = 0` (sun on horizon), and `1` at `cos_zenith = -0.1`
/// (sun ~5.7° below horizon). Implements `f(c) = 0.5 - 5·c`, clamped.
/// This expression is mirrored exactly in `assets/shaders/aurora.wgsl`.
///
/// `up` and `sun` must be unit vectors in the same frame.
pub fn aurora_day_side_factor(up: DVec3, sun: DVec3) -> f64 {
    let cos_zenith_sun = up.dot(sun);
    (0.5 - 5.0 * cos_zenith_sun).clamp(0.0, 1.0)
}

// ── Systems ────────────────────────────────────────────────────────────────────

/// Update-schedule system: spawn one shell entity at planet center once
/// `PlanetConfig` is available. Idempotent — bails out if the shell already
/// exists. Runs in Update rather than Startup because `PlanetConfig` is
/// inserted by the world plugin asynchronously after RON load.
pub fn spawn_aurora_shell(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<AuroraMaterial>>,
    planet: Option<Res<crate::world::planet::PlanetConfig>>,
    existing: Query<&AuroraShell>,
) {
    if !existing.is_empty() {
        return;
    }
    let Some(planet) = planet else { return };

    let r_outer = (planet.sea_level_radius as f32) + 300_000.0;
    let r_inner = (planet.sea_level_radius as f32) + 95_000.0;

    let mesh = meshes.add(Sphere::new(1.0).mesh().uv(48, 24));

    let center_rad = (planet.aurora_band_center_deg as f32).to_radians();
    let half_rad = (planet.aurora_band_half_width_deg as f32).to_radians();

    let material = materials.add(AuroraMaterial {
        planet_center_render: Vec4::ZERO,
        magnetic_north_axis: Vec4::new(0.0, 1.0, 0.0, 0.0),
        sun_world_direction: Vec4::new(0.0, 1.0, 0.0, 0.0),
        params: Vec4::new(r_inner, r_outer, planet.aurora_strength as f32, 0.0),
        band: Vec4::new(center_rad, half_rad, 12.0, 0.05),
    });

    commands.insert_resource(AuroraMaterialHandle(material.clone()));

    commands.spawn((
        AuroraShell,
        Mesh3d(mesh),
        MeshMaterial3d(material),
        Transform::from_scale(Vec3::splat(r_outer)),
        NoFrustumCulling,
    ));
}

/// PostUpdate (after `TransformSystems::Propagate`): place the shell at the
/// planet centre in render-space. Planet body sits at world origin (0,0,0),
/// so its render-space position is `-RenderOrigin.0`.
pub fn anchor_aurora_shell_to_planet(
    render_origin: Res<RenderOrigin>,
    mut q: Query<&mut Transform, With<AuroraShell>>,
) {
    let Ok(mut tr) = q.single_mut() else { return };
    tr.translation = (-render_origin.0).as_vec3();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn magnetic_north_axis_default_matches_geographic() {
        let geo = DVec3::Y;
        let mag = magnetic_north_axis(geo, 0.0, 0.0);
        assert!((mag - geo).length() < 1e-9, "mag={mag:?}");
    }

    #[test]
    fn magnetic_north_axis_with_lat_tilt_rotates_toward_equator() {
        let geo = DVec3::Y;
        let mag = magnetic_north_axis(geo, 10.0, 0.0);
        let expected_y = 10.0_f64.to_radians().cos();
        assert!(
            (mag.y - expected_y).abs() < 1e-9,
            "mag.y={}, want {}",
            mag.y,
            expected_y
        );
        let horiz = (mag.x * mag.x + mag.z * mag.z).sqrt();
        let expected_horiz = 10.0_f64.to_radians().sin();
        assert!((horiz - expected_horiz).abs() < 1e-9);
    }

    #[test]
    fn magnetic_north_axis_with_lon_rotates_tilt_direction() {
        let geo = DVec3::Y;
        let a = magnetic_north_axis(geo, 5.0, 0.0);
        let b = magnetic_north_axis(geo, 5.0, 90.0);
        let horiz_a = DVec3::new(a.x, 0.0, a.z).normalize();
        let horiz_b = DVec3::new(b.x, 0.0, b.z).normalize();
        let dot = horiz_a.dot(horiz_b);
        assert!(dot.abs() < 1e-9, "dot={}", dot);
    }

    #[test]
    fn magnetic_north_axis_handles_x_aligned_planet_axis() {
        // |up.x| >= 0.9 triggers the arb = Z fallback. Result must still be
        // a unit vector with the requested lat tilt magnitude.
        let geo = DVec3::X;
        let mag = magnetic_north_axis(geo, 15.0, 0.0);
        assert!((mag.length() - 1.0).abs() < 1e-9);
        let expected_x = 15.0_f64.to_radians().cos();
        assert!(
            (mag.x - expected_x).abs() < 1e-9,
            "mag.x={}, want {}",
            mag.x,
            expected_x
        );
        // Tilt magnitude away from up: sqrt(1 - x²) = sin(15°).
        let off_axis = (mag.y * mag.y + mag.z * mag.z).sqrt();
        let expected_off = 15.0_f64.to_radians().sin();
        assert!((off_axis - expected_off).abs() < 1e-9);
    }

    #[test]
    fn aurora_band_mask_peaks_at_band_center_lat() {
        let center = 67.0_f64.to_radians();
        let half_w = 5.0_f64.to_radians();
        let peak = aurora_band_mask(center, center, half_w);
        let edge = aurora_band_mask(center - half_w, center, half_w);
        let outside = aurora_band_mask(center - 2.0 * half_w, center, half_w);

        assert!((peak - 1.0).abs() < 1e-6, "peak={peak}");
        assert!((edge - 0.5).abs() < 1e-3, "edge={edge}");
        assert!(outside <= 0.05, "outside={outside}");
    }

    #[test]
    fn aurora_band_mask_is_hemisphere_symmetric() {
        let center = 67.0_f64.to_radians();
        let half_w = 5.0_f64.to_radians();
        let north = aurora_band_mask(center, center, half_w);
        let south = aurora_band_mask(-center, center, half_w);
        assert!((north - south).abs() < 1e-12);
    }

    #[test]
    fn aurora_day_side_factor_is_zero_when_sun_overhead() {
        let up = DVec3::Y;
        let sun = DVec3::Y;
        let f = aurora_day_side_factor(up, sun);
        assert!(f < 1e-6, "f={f}");
    }

    #[test]
    fn aurora_day_side_factor_is_one_at_midnight() {
        let up = DVec3::Y;
        let sun = -DVec3::Y;
        let f = aurora_day_side_factor(up, sun);
        assert!((f - 1.0).abs() < 1e-6, "f={f}");
    }

    #[test]
    fn aurora_day_side_factor_is_half_at_horizon() {
        // Sun exactly on horizon ⇒ factor == 0.5.
        let up = DVec3::Y;
        let sun = DVec3::X;
        let f = aurora_day_side_factor(up, sun);
        assert!((f - 0.5).abs() < 1e-9, "f={f}");
    }

    #[test]
    fn aurora_day_side_factor_saturates_below_horizon() {
        // Sun 5.7° below horizon ⇒ factor == 1.0.
        let up = DVec3::Y;
        // cos_zenith = -0.1 ⇒ sun.y = -0.1, sun.x = sqrt(1-0.01)
        let sun = DVec3::new((1.0_f64 - 0.01).sqrt(), -0.1, 0.0);
        let f = aurora_day_side_factor(up, sun);
        assert!((f - 1.0).abs() < 1e-9, "f={f}");
    }

    #[test]
    fn aurora_day_side_factor_is_zero_above_horizon() {
        // Sun 5.7° above horizon ⇒ factor == 0.0.
        let up = DVec3::Y;
        let sun = DVec3::new((1.0_f64 - 0.01).sqrt(), 0.1, 0.0);
        let f = aurora_day_side_factor(up, sun);
        assert!(f < 1e-9, "f={f}");
    }
}
