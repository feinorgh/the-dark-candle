// Sky dome: single-scatter Rayleigh atmospheric scattering + sun disk.
//
// Rendered as a large sphere mesh (radius 1e5 m) behind all solid geometry.
//
// Reverse-Z depth trick: the vertex stage outputs `clip.z = 0.0` so after the
// perspective divide NDC depth = 0.0 / w = 0.0, which is the far plane in
// Bevy's infinite reverse-Z projection.  This places the sky behind everything
// without a depth write (depth_write_enabled = false in the material pipeline).
//
// SkyMaterial is registered as AlphaMode::Blend so it bypasses the depth
// prepass entirely; it renders in the Transparent3d phase after all opaque
// geometry has filled the depth buffer.

#import bevy_pbr::mesh_functions::get_world_from_local
#import bevy_pbr::mesh_view_bindings::view
#import bevy_pbr::forward_io::Vertex

// Sun direction in the observer's local tangent frame (Y = zenith, X = east,
// Z = south).  Uploaded by update_sky_material() each frame.
@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> sky_sun_direction: vec4<f32>;

// Observer's "up" unit vector in render-space world coordinates.
// Used to build the local tangent frame in the fragment shader.
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var<uniform> sky_observer_up: vec4<f32>;

// Body→celestial rotation matrix, three rows packed as Vec4 (w unused).
// Maps a render-space (body-frame) direction into the catalogue's celestial
// inertial frame so stars rotate with the planet.
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var<uniform> sky_body_to_celestial: array<vec4<f32>, 3>;

// Sky tunables: x = night brightness multiplier, y = horizon-extinction
// strength, z/w reserved for future use (long-exposure gain, EM band, …).
@group(#{MATERIAL_BIND_GROUP}) @binding(3) var<uniform> sky_params: vec4<f32>;

// HDR cubemap containing the procedural celestial catalogue (stars, plus
// host galaxy, nebulae and remote galaxies in later phases).
@group(#{MATERIAL_BIND_GROUP}) @binding(4) var sky_star_cubemap: texture_cube<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(5) var sky_star_sampler: sampler;

// ── Vertex stage ──────────────────────────────────────────────────────────────

struct SkyOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
}

@vertex
fn vertex(in: Vertex) -> SkyOut {
    let world_from_local = get_world_from_local(in.instance_index);
    let world_pos = (world_from_local * vec4<f32>(in.position, 1.0)).xyz;
    let clip = view.clip_from_world * vec4<f32>(world_pos, 1.0);
    // z = 0.0 → NDC depth = 0.0 = far plane (reverse-Z infinity).
    return SkyOut(vec4<f32>(clip.xy, 0.0, clip.w), world_pos);
}

// ── Rayleigh scattering constants (SI) ───────────────────────────────────────

// Rayleigh scattering coefficients at sea level per RGB channel (m⁻¹).
const BETA_R: vec3<f32> = vec3<f32>(5.8e-6, 13.5e-6, 33.1e-6);
// Rayleigh density scale height (m).
const H_RAYLEIGH: f32 = 8500.0;
// Earth radius (m) used as atmosphere base for the scattering model.
const R_EARTH: f32 = 6371000.0;
// Atmosphere top radius (m).
const R_ATMOS: f32 = 6471000.0;
// Solar irradiance scale (dimensionless, tuned for Bevy HDR pipeline).
const SUN_INTENSITY: f32 = 20.0;
// View-ray integration samples.
const VIEW_SAMPLES: i32 = 16;
// Light-ray (sun) integration samples per view sample.
const LIGHT_SAMPLES: i32 = 8;
// Sun disk: cosine of the outer/inner edge of the limb transition.
const SUN_DISK_COS_OUTER: f32 = 0.9998;
const SUN_DISK_COS_INNER: f32 = 0.9999;
// Sun disk linear brightness multiplier relative to scattered sky.
const SUN_DISK_INTENSITY: f32 = 8.0;

// ── Scattering helpers ────────────────────────────────────────────────────────

/// Ray–sphere intersection distance.  Returns −1.0 if no forward intersection.
fn ray_sphere(origin: vec3<f32>, dir: vec3<f32>, radius: f32) -> f32 {
    let b = 2.0 * dot(origin, dir);
    let c = dot(origin, origin) - radius * radius;
    let disc = b * b - 4.0 * c;
    if disc < 0.0 { return -1.0; }
    let sq = sqrt(disc);
    let t0 = (-b - sq) * 0.5;
    let t1 = (-b + sq) * 0.5;
    if t1 < 0.0 { return -1.0; }
    if t0 < 0.0 { return t1; }
    return t0;
}

/// Rayleigh optical depth from `point` toward the sun through the atmosphere.
/// Returns a large value (1e10) if the path is blocked by the planet body.
fn optical_depth_to_sun(point: vec3<f32>, sun: vec3<f32>) -> vec3<f32> {
    let dist = ray_sphere(point, sun, R_ATMOS);
    if dist < 0.0 { return vec3<f32>(1e10); }
    let step_len = dist / f32(LIGHT_SAMPLES);
    var od = vec3<f32>(0.0);
    for (var j = 0; j < LIGHT_SAMPLES; j++) {
        let s = point + sun * ((f32(j) + 0.5) * step_len);
        let h = length(s) - R_EARTH;
        if h < 0.0 { return vec3<f32>(1e10); }
        od += BETA_R * exp(-h / H_RAYLEIGH) * step_len;
    }
    return od;
}

/// Single-scatter Rayleigh sky colour.
///
/// Both `view_dir` and `sun_dir` must be in the observer's local tangent frame:
///   Y = zenith (planet-up), X = east, Z = south.
/// The observer is modelled at the surface: origin = (0, R_EARTH, 0).
fn sky_color(view_dir: vec3<f32>, sun_dir: vec3<f32>) -> vec3<f32> {
    let origin = vec3<f32>(0.0, R_EARTH, 0.0);
    let atmos_dist = ray_sphere(origin, view_dir, R_ATMOS);
    if atmos_dist < 0.0 { return vec3<f32>(0.0); }

    let step_len = atmos_dist / f32(VIEW_SAMPLES);
    // Rayleigh phase: 3/(16π) × (1 + cos²θ).
    let cos_theta = dot(view_dir, sun_dir);
    let phase = (3.0 / (16.0 * 3.14159265)) * (1.0 + cos_theta * cos_theta);

    var total   = vec3<f32>(0.0);
    var od_view = vec3<f32>(0.0);

    for (var i = 0; i < VIEW_SAMPLES; i++) {
        let s       = origin + view_dir * ((f32(i) + 0.5) * step_len);
        let h       = length(s) - R_EARTH;
        let density = exp(-h / H_RAYLEIGH);
        let od_seg  = BETA_R * density * step_len;
        od_view    += od_seg;
        let od_sun  = optical_depth_to_sun(s, sun_dir);
        let atten   = exp(-(od_view + od_sun));
        total      += density * step_len * BETA_R * atten;
    }
    return total * phase * SUN_INTENSITY;
}

// ── Fragment stage ────────────────────────────────────────────────────────────

@fragment
fn fragment(in: SkyOut) -> @location(0) vec4<f32> {
    // World-space view direction: from camera to this sky fragment.
    let world_view = normalize(in.world_pos - view.world_position);

    // Build the observer's local tangent frame (Y = planet-up) from observer_up.
    let up  = normalize(sky_observer_up.xyz);
    let arb = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), abs(up.x) < 0.9);
    let east  = normalize(cross(up, arb));
    let south = cross(east, up); // east ⊥ up → already unit length

    // Project world-space view ray into local tangent frame for sky_color().
    let local_view = vec3<f32>(
        dot(world_view, east),
        dot(world_view, up),
        dot(world_view, south),
    );

    let sun = sky_sun_direction.xyz;
    var rgb = sky_color(local_view, sun);

    // Sun disk: smooth limb transition within ~0.01° of the sun centre.
    let cos_angle = dot(local_view, sun);
    let disk  = smoothstep(SUN_DISK_COS_OUTER, SUN_DISK_COS_INNER, cos_angle);
    // Warm white at the core, slightly orange at the limb.
    rgb += vec3<f32>(1.0, 0.9, 0.7) * (disk * SUN_DISK_INTENSITY);

    // Star cubemap: rotate the world view-direction into the catalogue's
    // celestial inertial frame, sample the baked HDR cubemap, and add it
    // additively to the Rayleigh result.  During daytime Rayleigh dominates
    // and stars are naturally drowned out; at night Rayleigh is dim and the
    // stars become visible.
    //
    // The rotation is uploaded row-major; `dot(row_i, v)` gives the i-th
    // component of `R · v`, which is what we want.
    let dir_celestial = vec3<f32>(
        dot(sky_body_to_celestial[0].xyz, world_view),
        dot(sky_body_to_celestial[1].xyz, world_view),
        dot(sky_body_to_celestial[2].xyz, world_view),
    );
    var stars = textureSample(sky_star_cubemap, sky_star_sampler, dir_celestial).rgb;

    // Horizon airmass extinction: stars near the horizon dim with view angle.
    // Approximate sec(z) = 1 / max(local_view.y, ε) and scale by the strength
    // factor.  Above the horizon (local_view.y ≤ 0) we pin extinction to a
    // small finite value so terrain-occluded stars don't go black instantly.
    let cos_z = max(local_view.y, 0.05);
    let airmass = 1.0 / cos_z;
    let extinction = exp(-sky_params.y * (airmass - 1.0));
    stars *= extinction;

    rgb += stars * sky_params.x;

    // Output linear HDR — Bevy's tonemapping pass handles the rest.
    return vec4<f32>(rgb, 1.0);
}
