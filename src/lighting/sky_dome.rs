// GPU sky dome: Rayleigh sky gradient, sun disk, and moon billboards.
//
// Architecture overview:
//   * `SkyMaterial`   – Bevy custom `Material` backed by `sky_dome.wgsl`.
//                       Uses `AlphaMode::Blend` so it bypasses the depth
//                       prepass and renders in the Transparent3d phase, after
//                       all opaque terrain/chunks have written depth.
//   * `SkyDome`       – marker component for the sky sphere entity.
//   * `MoonBillboard` – marker + per-moon metadata for each moon quad entity.
//   * `GameElapsedSeconds` – accumulated game-time for Keplerian moon orbits.
//
// Systems added by `LightingPlugin`:
//   Startup:    spawn_sky_dome
//   Update:     update_sky_material, spawn_moon_billboards, update_moon_positions
//   PostUpdate: anchor_sky_dome_to_camera

use bevy::camera::visibility::NoFrustumCulling;
use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::pbr::{Material, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, CompareFunction, RenderPipelineDescriptor, SpecializedMeshPipelineError,
};
use bevy::shader::ShaderRef;

use crate::floating_origin::WorldPosition;
use crate::lighting::SunWorldDirection;
use crate::planet::celestial::moon_position;
use crate::world::PlanetaryData;

// ── SkyMaterial ───────────────────────────────────────────────────────────────

/// Custom material for the atmospheric sky dome.
///
/// Uniforms uploaded each frame by `update_sky_material`:
///   `sun_direction`      – sun direction in the observer's local tangent frame.
///   `observer_up`        – planet-up unit vector in render-space world coords.
///   `body_to_celestial_*` – three rows of the body→celestial rotation matrix
///                          packed as Vec4 (w unused; mat3 alignment in WGSL
///                          is fragile, three Vec4s is the portable form).
///   `sky_params`         – x = night_brightness multiplier, y = star
///                          extinction strength, z/w reserved.
///
/// Texture bindings:
///   `star_cubemap` (binding 4) + `star_sampler` (binding 5) — HDR cubemap of
///   the procedural celestial catalogue, sampled along
///   `body_to_celestial · world_view`.
#[derive(Asset, bevy::reflect::TypePath, AsBindGroup, Debug, Clone)]
pub struct SkyMaterial {
    /// Sun direction in observer's local tangent frame (Y = zenith). xyz used.
    #[uniform(0)]
    pub sun_direction: Vec4,
    /// Observer's planet-up direction in world (render) space. xyz used.
    #[uniform(1)]
    pub observer_up: Vec4,
    /// Rows of the body→celestial rotation matrix (mat3 packed as 3×Vec4).
    #[uniform(2)]
    pub body_to_celestial_rows: [Vec4; 3],
    /// x = night brightness multiplier, y = horizon-extinction strength.
    #[uniform(3)]
    pub sky_params: Vec4,
    /// Cubemap texture of the procedural celestial catalogue.
    #[texture(4, dimension = "cube")]
    #[sampler(5)]
    pub star_cubemap: Handle<Image>,
}

impl Material for SkyMaterial {
    fn vertex_shader() -> ShaderRef {
        "shaders/sky_dome.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "shaders/sky_dome.wgsl".into()
    }

    /// AlphaMode::Blend bypasses the depth prepass entirely, so the sky dome
    /// sphere mesh never writes incorrect depth values during that phase.
    /// The sky renders in the Transparent3d phase with z = 0.0 (infinity).
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Blend
    }

    fn specialize(
        _pipeline: &bevy::pbr::MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // Render both faces so the sphere is visible from inside.
        descriptor.primitive.cull_mode = None;
        if let Some(depth) = descriptor.depth_stencil.as_mut() {
            // The sky writes no depth; terrain depth values must remain intact.
            depth.depth_write_enabled = false;
            // Reverse-Z: pass if incoming z (0.0 = sky-at-infinity) >= stored z.
            depth.depth_compare = CompareFunction::GreaterEqual;
        }
        Ok(())
    }
}

// ── Marker components ─────────────────────────────────────────────────────────

/// Marker for the sky dome sphere entity.
#[derive(Component, Debug)]
pub struct SkyDome;

/// Marker for a moon billboard quad entity.  Stores the moon index into
/// `PlanetaryData.0.celestial.moons`.
#[derive(Component, Debug)]
pub struct MoonBillboard {
    pub moon_index: usize,
}

// ── Resources ─────────────────────────────────────────────────────────────────

/// Accumulated game-time since session start in seconds.
///
/// Incremented in `advance_time` by `real_dt × time_scale` each frame.
/// Used to drive Keplerian moon orbital positions.
#[derive(Resource, Default, Debug, Clone, Copy)]
pub struct GameElapsedSeconds(pub f64);

// ── Constants ─────────────────────────────────────────────────────────────────

/// Visual distance (render space, metres) at which moon billboards are placed.
/// The actual distance is irrelevant for appearance; this value must be large
/// enough to be behind terrain but small enough for f32 precision.
const VISUAL_MOON_DIST: f32 = 50_000.0;

// ── Systems ───────────────────────────────────────────────────────────────────

/// Startup system: spawn the sky dome sphere.
///
/// The sphere uses a radius-1 UVSphere mesh scaled to 1e5 m via Transform.
/// `NoFrustumCulling` ensures it is never accidentally culled (it surrounds the
/// camera, but Bevy's frustum test uses the mesh AABB, not the camera position).
pub fn spawn_sky_dome(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<SkyMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    let mesh = meshes.add(Sphere::new(1.0).mesh().uv(32, 18));
    let placeholder = images.add(black_cube_placeholder());
    let mat = materials.add(SkyMaterial {
        sun_direction: Vec4::new(0.0, 1.0, 0.0, 0.0),
        observer_up: Vec4::new(0.0, 1.0, 0.0, 0.0),
        body_to_celestial_rows: [
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 1.0, 0.0),
        ],
        sky_params: Vec4::new(1.0, 0.18, 0.0, 0.0),
        star_cubemap: placeholder,
    });
    commands.spawn((
        SkyDome,
        Mesh3d(mesh),
        MeshMaterial3d(mat),
        Transform::from_scale(Vec3::splat(1e5)),
        NoFrustumCulling,
    ));
}

/// 1×1×6 black cubemap used until the SkyPlugin baker produces the real
/// catalogue cubemap.  Without a valid handle the AsBindGroup derive panics
/// at first frame.
fn black_cube_placeholder() -> Image {
    use bevy::asset::RenderAssetUsages;
    use bevy::render::render_resource::{
        Extent3d, TextureDimension, TextureFormat, TextureViewDescriptor, TextureViewDimension,
    };
    // 6 × 1px × RGBA16F = 6 × 1 × 8 = 48 bytes, all zero.
    let bytes = vec![0u8; 48];
    let mut image = Image::new(
        Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 6,
        },
        TextureDimension::D2,
        bytes,
        TextureFormat::Rgba16Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_view_descriptor = Some(TextureViewDescriptor {
        label: Some("sky_star_cubemap_placeholder_view"),
        dimension: Some(TextureViewDimension::Cube),
        ..default()
    });
    image
}

/// PostUpdate system: keep the sky dome centred on the camera.
///
/// The floating-origin `rebase_origin` system (also in PostUpdate) shifts all
/// non-camera entity Transforms when the camera moves beyond 512 m from the
/// current render origin.  Running after that rebase, this system copies the
/// camera's render-space translation to the sky dome so it always surrounds
/// the camera regardless of how many rebase cycles have accumulated.
pub fn anchor_sky_dome_to_camera(
    cam_q: Query<&Transform, With<Camera>>,
    mut dome_q: Query<&mut Transform, (With<SkyDome>, Without<Camera>)>,
) {
    let Ok(cam_tf) = cam_q.single() else {
        return;
    };
    for mut dome_tf in &mut dome_q {
        dome_tf.translation = cam_tf.translation;
    }
}

/// Update system: upload current sun direction, observer-up, body→celestial
/// rotation, and (lazily, once available) the baked star cubemap to the sky
/// material.
///
/// Sun direction is projected into the observer's local tangent frame
/// (Y = up), matching the coordinate system expected by the Rayleigh
/// scattering model in `sky_dome.wgsl`.
///
/// `body_to_celestial = R_y(-rotation_angle)` is the inverse of the planet's
/// hour-angle rotation, so a fixed celestial direction passed through the
/// matrix appears to sweep across the sky as rotation_angle advances —
/// reusing the *same* `OrbitalState.rotation_angle` as the sun calculation
/// makes sun and stars co-rotate by construction.
pub fn update_sky_material(
    sun_world: Res<SunWorldDirection>,
    orbital: Res<crate::lighting::orbital::OrbitalState>,
    star_cubemap: Option<Res<crate::sky::StarCubemapHandle>>,
    cam_q: Query<&WorldPosition, With<crate::camera::FpsCamera>>,
    dome_q: Query<&MeshMaterial3d<SkyMaterial>, With<SkyDome>>,
    mut materials: ResMut<Assets<SkyMaterial>>,
) {
    let Ok(handle) = dome_q.single() else {
        return;
    };
    let Some(mat) = materials.get_mut(handle.0.id()) else {
        return;
    };

    // Observer's planet-up: direction from planet centre to camera, normalised.
    let observer_up = cam_q
        .iter()
        .next()
        .map(|wp| wp.0.normalize_or(bevy::math::DVec3::Y))
        .unwrap_or(bevy::math::DVec3::Y);

    // Build the local tangent frame: up = planet-up, east and south from cross products.
    let up = observer_up;
    let arbitrary = if up.x.abs() < 0.9 {
        bevy::math::DVec3::X
    } else {
        bevy::math::DVec3::Z
    };
    let east = up.cross(arbitrary).normalize();
    let south = east.cross(up).normalize();

    // Project the world-space sun direction into the local frame.
    let wd = sun_world.0;
    let local_sun = Vec3::new(wd.dot(east) as f32, wd.dot(up) as f32, wd.dot(south) as f32);
    let up_f32 = Vec3::new(
        observer_up.x as f32,
        observer_up.y as f32,
        observer_up.z as f32,
    );

    mat.sun_direction = local_sun.extend(0.0);
    mat.observer_up = up_f32.extend(0.0);

    // Drive star/nebula brightness from solar altitude (SKY-007).
    // `local_sun.y` is sin(altitude) in the observer frame: 0 at horizon,
    // negative below, +1 at zenith.  Real-world astronomical twilight ends
    // (full night sky) when the sun is 18° below the horizon; stars become
    // visible at civil twilight (≈ −6°).  The fade between matches
    // observation: the sky still looks blue 5° below the horizon and stars
    // only saturate well after −18°.
    let night = night_brightness_from_solar_altitude(local_sun.y);
    mat.sky_params = Vec4::new(night, mat.sky_params.y, mat.sky_params.z, mat.sky_params.w);

    // Body→celestial rotation: R_y(-rotation_angle) in the body frame.
    // Stored row-major so each Vec4 in the shader is one row.
    let theta = -orbital.rotation_angle as f32;
    let (s, c) = theta.sin_cos();
    mat.body_to_celestial_rows = [
        Vec4::new(c, 0.0, s, 0.0),
        Vec4::new(0.0, 1.0, 0.0, 0.0),
        Vec4::new(-s, 0.0, c, 0.0),
    ];

    // Swap in the real cubemap as soon as the SkyPlugin baker has published
    // it.  Idempotent: comparing handle ids avoids needless GPU rebinding.
    if let Some(cube) = star_cubemap
        && mat.star_cubemap.id() != cube.0.id()
    {
        mat.star_cubemap = cube.0.clone();
    }
}

/// Update system: spawn moon billboard quads once `PlanetaryData` is available.
///
/// Uses a `Local<bool>` guard so the spawn happens exactly once.
pub fn spawn_moon_billboards(
    mut spawned: Local<bool>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    planetary_data: Option<Res<PlanetaryData>>,
) {
    if *spawned {
        return;
    }
    let Some(data) = planetary_data else {
        return;
    };
    let moons = &data.0.celestial.moons;
    if moons.is_empty() {
        *spawned = true;
        return;
    }

    for (i, moon) in moons.iter().enumerate() {
        let mesh = meshes.add(Rectangle::new(2.0, 2.0));
        let base = Color::srgb(
            moon.surface_color[0],
            moon.surface_color[1],
            moon.surface_color[2],
        );
        // Unlit emissive so moon brightness is albedo-driven, not affected by
        // the sun DirectionalLight shining on the billboard quad.
        let mat = materials.add(StandardMaterial {
            base_color: base,
            emissive: LinearRgba::from(base) * moon.albedo,
            unlit: true,
            alpha_mode: AlphaMode::Blend,
            double_sided: true,
            cull_mode: None,
            ..default()
        });
        commands.spawn((
            MoonBillboard { moon_index: i },
            Mesh3d(mesh),
            MeshMaterial3d(mat),
            Transform::default(),
            NoFrustumCulling,
        ));
    }
    *spawned = true;
}

/// Update system: reposition and orient moon billboard quads each frame.
///
/// Moon positions are computed from Keplerian orbital mechanics via
/// `moon_position(moon, t)`.  Billboards are placed at `VISUAL_MOON_DIST` from
/// the camera in the direction of the actual moon; their scale is set to match
/// the correct apparent angular size.
pub fn update_moon_positions(
    elapsed: Res<GameElapsedSeconds>,
    planetary_data: Option<Res<PlanetaryData>>,
    cam_q: Query<(&Transform, &WorldPosition), With<crate::camera::FpsCamera>>,
    mut moon_q: Query<
        (&MoonBillboard, &mut Transform, &mut Visibility),
        Without<crate::camera::FpsCamera>,
    >,
) {
    let Some(data) = planetary_data else {
        return;
    };
    let Ok((cam_render_tf, cam_world)) = cam_q.single() else {
        return;
    };

    let cam_up = cam_world.0.normalize_or(bevy::math::DVec3::Y);

    for (billboard, mut tf, mut vis) in &mut moon_q {
        let Some(moon) = data.0.celestial.moons.get(billboard.moon_index) else {
            continue;
        };

        // Moon position in planet-centred coordinates (Keplerian orbit).
        let moon_planet_pos = moon_position(moon, elapsed.0);

        // Direction from camera to moon (planet-scale f64 arithmetic).
        let to_moon = moon_planet_pos - cam_world.0;
        let distance_m = to_moon.length();
        let moon_dir_d = to_moon.normalize_or(bevy::math::DVec3::Y);
        let moon_dir = Vec3::new(
            moon_dir_d.x as f32,
            moon_dir_d.y as f32,
            moon_dir_d.z as f32,
        );

        // Place billboard at visual distance in the direction of the moon.
        let render_pos = cam_render_tf.translation + moon_dir * VISUAL_MOON_DIST;

        // Apparent angular radius (rad) → visual billboard radius (m at VISUAL_MOON_DIST).
        // apparent = moon.radius_m / distance_m; visual = apparent * VISUAL_MOON_DIST.
        let visual_radius = (moon.radius_m as f32 / distance_m as f32) * VISUAL_MOON_DIST;

        // Billboard orientation: make the quad face the camera.
        // Transform::looking_at makes local −Z point toward target; placing the
        // target at (render_pos + moon_dir) makes −Z point away from camera,
        // so +Z (the rectangle's front face normal) faces the camera.
        let up_hint = if moon_dir.abs().dot(Vec3::Y) > 0.99 {
            Vec3::Z
        } else {
            Vec3::Y
        };
        *tf = Transform::from_translation(render_pos)
            .looking_at(render_pos + moon_dir, up_hint)
            .with_scale(Vec3::splat(visual_radius));

        // Hide moon when it is below the horizon.
        let cam_up_f32 = Vec3::new(cam_up.x as f32, cam_up.y as f32, cam_up.z as f32);
        *vis = if moon_dir.dot(cam_up_f32) > -0.1 {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
    }
}

// ── Twilight fade helpers (SKY-007) ───────────────────────────────────────────

/// Civil twilight starts when the sun is 6° below the horizon: stars of
/// magnitude 0–1 begin to appear.  sin(-6°) ≈ -0.1045.
const SIN_CIVIL_TWILIGHT: f32 = -0.104_528_46;
/// Astronomical twilight ends when the sun is 18° below the horizon: the sky
/// is fully dark, all stars saturate.  sin(-18°) ≈ -0.3090.
const SIN_ASTRONOMICAL_TWILIGHT: f32 = -0.309_017;

/// Maps the sun's altitude (`sin_alt = local_sun.y`) to a star-brightness
/// multiplier in `[0, 1]`.
///
/// - Sun above the horizon (`sin_alt ≥ 0`): returns 0 — Rayleigh dominates,
///   stars are invisible anyway, and additive bleed during the day is the
///   main artefact this fade fixes.
/// - Sun between 0° and −6° (civil twilight): smoothly ramps from 0 to a
///   small visibility (first-magnitude stars).
/// - Sun between −6° and −18°: ramps to full visibility.
/// - Sun below −18°: 1.0 (full astronomical night).
///
/// The two-segment smoothstep matches the real-world progression where the
/// horizon glow extinguishes long before astronomical night begins, so the
/// brightest stars appear well before the Milky Way does.
pub fn night_brightness_from_solar_altitude(sin_alt: f32) -> f32 {
    if sin_alt >= 0.0 {
        return 0.0;
    }
    // Segment 1: 0° → −6°  produces 0 → 0.25 (first-magnitude stars only)
    let civil = smoothstep_neg(0.0, SIN_CIVIL_TWILIGHT, sin_alt);
    // Segment 2: −6° → −18°  produces 0 → 1 (full sky)
    let astronomical = smoothstep_neg(SIN_CIVIL_TWILIGHT, SIN_ASTRONOMICAL_TWILIGHT, sin_alt);
    (0.25 * civil + 0.75 * astronomical).clamp(0.0, 1.0)
}

/// Smoothstep where `edge0` is the value that maps to 0 and `edge1` maps to 1,
/// even if `edge1 < edge0` (as is the case for sin-altitude going negative).
fn smoothstep_neg(edge0: f32, edge1: f32, x: f32) -> f32 {
    if (edge1 - edge0).abs() < 1e-9 {
        return 0.0;
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn night_brightness_zero_when_sun_above_horizon() {
        assert_eq!(night_brightness_from_solar_altitude(1.0), 0.0);
        assert_eq!(night_brightness_from_solar_altitude(0.5), 0.0);
        assert_eq!(night_brightness_from_solar_altitude(0.01), 0.0);
        assert_eq!(night_brightness_from_solar_altitude(0.0), 0.0);
    }

    #[test]
    fn night_brightness_full_below_astronomical_twilight() {
        // Sun at −20° (well below astronomical twilight) → full stars.
        let sin20 = (-20.0_f32).to_radians().sin();
        assert!((night_brightness_from_solar_altitude(sin20) - 1.0).abs() < 1e-3);
        // Sun straight down.
        assert!((night_brightness_from_solar_altitude(-1.0) - 1.0).abs() < 1e-3);
    }

    #[test]
    fn night_brightness_smooth_through_civil_twilight() {
        let n_horizon = night_brightness_from_solar_altitude(0.0);
        let n_civil_mid = night_brightness_from_solar_altitude(SIN_CIVIL_TWILIGHT * 0.5);
        let n_civil = night_brightness_from_solar_altitude(SIN_CIVIL_TWILIGHT);
        let n_nautical = night_brightness_from_solar_altitude(
            (SIN_CIVIL_TWILIGHT + SIN_ASTRONOMICAL_TWILIGHT) * 0.5,
        );
        let n_astro = night_brightness_from_solar_altitude(SIN_ASTRONOMICAL_TWILIGHT);
        // Monotone non-decreasing as sun drops.
        assert!(n_horizon <= n_civil_mid);
        assert!(n_civil_mid <= n_civil);
        assert!(n_civil <= n_nautical);
        assert!(n_nautical <= n_astro);
        // At civil twilight only first-magnitude stars should be visible.
        assert!(n_civil < 0.30);
        assert!(n_civil > 0.20);
        // At astronomical twilight stars are essentially saturated.
        assert!(n_astro > 0.95);
    }

    #[test]
    fn night_brightness_within_unit_interval() {
        // Sweep sin_alt from -1.0 → +1.0; n should be in [0,1] and
        // non-increasing (more sun = fewer stars).
        let mut prev = 2.0; // > any valid n
        for i in 0..=200 {
            let sin_alt = -1.0 + (i as f32) * 0.01;
            let n = night_brightness_from_solar_altitude(sin_alt);
            assert!((0.0..=1.0).contains(&n), "n={n} sin_alt={sin_alt}");
            assert!(
                n <= prev + 1e-6,
                "non-monotone at sin_alt={sin_alt}: {prev} -> {n}"
            );
            prev = n;
        }
    }
}
