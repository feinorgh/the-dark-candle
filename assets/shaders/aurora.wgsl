// Aurora — volumetric shell rendered as additive blend over the framebuffer.
//
// The mesh is a sphere of radius `aurora_params.y` (outer atmosphere top)
// centred on the planet. The fragment shader analytically intersects the view
// ray with the inner shell (`aurora_params.x`) and integrates emission over
// the segment with 12 samples.
//
// Emission per sample =
//     band_mask(|lat_mag|)               // smoothstep ring at the oval centre
//   * vertical_color(h_norm)             // green → red/purple ramp
//   * curtain_noise(lon_mag, h_norm, t)  // sharpened 2-D hash noise
//   * day_side_factor(up · sun)          // 0 day, 1 night
//   * aurora_strength
//
// Mirror of the CPU oracle in src/lighting/aurora.rs.

#import bevy_pbr::mesh_functions::get_world_from_local
#import bevy_pbr::mesh_view_bindings::view
#import bevy_pbr::forward_io::Vertex

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> planet_center_render: vec4<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var<uniform> magnetic_north_axis: vec4<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var<uniform> sun_world_direction: vec4<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(3) var<uniform> aurora_params: vec4<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(4) var<uniform> aurora_band: vec4<f32>;

const PI: f32 = 3.14159265358979;
const SAMPLES: i32 = 12;

struct AuroraOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
}

@vertex
fn vertex(in: Vertex) -> AuroraOut {
    let world_from_local = get_world_from_local(in.instance_index);
    let world_pos = (world_from_local * vec4<f32>(in.position, 1.0)).xyz;
    let clip = view.clip_from_world * vec4<f32>(world_pos, 1.0);
    return AuroraOut(clip, world_pos);
}

// Ray-sphere intersection: returns (t_near, t_far) or (-1, -1) if no hit.
fn ray_sphere_intersect(origin: vec3<f32>, dir: vec3<f32>, center: vec3<f32>, radius: f32) -> vec2<f32> {
    let oc = origin - center;
    let b = dot(oc, dir);
    let c = dot(oc, oc) - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return vec2<f32>(-1.0, -1.0);
    }
    let sq = sqrt(disc);
    return vec2<f32>(-b - sq, -b + sq);
}

// Hash → noise helpers (same family used in terrain_caustic.wgsl).
fn hash21(p: vec2<f32>) -> f32 {
    let q = vec3<f32>(p.xy, 0.1234);
    let s = fract(q * 0.1031);
    let r = s + dot(s, s.yzx + 33.33);
    return fract((r.x + r.y) * r.z);
}

fn value_noise2(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Magnetic-latitude-symmetric band mask. `delta = ||lat_mag| - center|`.
fn band_mask(lat_mag: f32, center: f32, half_width: f32) -> f32 {
    let delta = abs(abs(lat_mag) - center);
    let edge1 = 2.0 * half_width;
    let t = clamp(delta / edge1, 0.0, 1.0);
    let smooth = t * t * (3.0 - 2.0 * t);
    return 1.0 - smooth;
}

// Day-side gate: 0 when up·sun > 0.1, 1 when up·sun < -0.1.
// Mirrors aurora_day_side_factor in src/lighting/aurora.rs exactly.
fn day_side_factor(up: vec3<f32>, sun: vec3<f32>) -> f32 {
    let c = dot(up, sun);
    return clamp(0.5 - 5.0 * c, 0.0, 1.0);
}

// Vertical color ramp: bottom is green-cyan, top fades to red/purple.
fn vertical_color(h_norm: f32) -> vec3<f32> {
    let low = vec3<f32>(0.15, 1.0, 0.35);
    let high = vec3<f32>(0.55, 0.10, 0.65);
    let t = clamp(h_norm, 0.0, 1.0);
    // Sharpen the green band at the bottom, let the red bloom near the top.
    let bias = pow(t, 1.5);
    return mix(low, high, bias);
}

// Build an east/south basis for the magnetic frame so we can compute a
// longitude around the magnetic axis.
fn magnetic_east(up: vec3<f32>) -> vec3<f32> {
    let arb = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), abs(up.x) < 0.9);
    return normalize(cross(up, arb));
}

@fragment
fn fragment(in: AuroraOut) -> @location(0) vec4<f32> {
    let r_inner = aurora_params.x;
    let r_outer = aurora_params.y;
    let strength = aurora_params.z;
    let elapsed = aurora_params.w;

    if strength <= 0.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    let center = planet_center_render.xyz;
    let mag_axis = normalize(magnetic_north_axis.xyz);
    let sun = normalize(sun_world_direction.xyz);
    let band_center = aurora_band.x;
    let band_half = aurora_band.y;
    let curtain_freq = aurora_band.z;
    let curtain_speed = aurora_band.w;

    let cam = view.world_position;
    let dir = normalize(in.world_pos - cam);

    // Intersect outer + inner shells.
    let outer = ray_sphere_intersect(cam, dir, center, r_outer);
    if outer.y < 0.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    let inner = ray_sphere_intersect(cam, dir, center, r_inner);

    // Choose the integration segment. Three cases:
    //   * Camera outside outer shell: segment = [outer.x, inner.x] if inner hit,
    //     else [outer.x, outer.y].
    //   * Camera between shells: segment starts at 0, ends at inner.x (if hit)
    //     or outer.y.
    //   * Camera inside inner shell: segment = [inner.y, outer.y].
    let cam_dist = length(cam - center);
    var t_start: f32 = 0.0;
    var t_end: f32 = outer.y;
    if cam_dist > r_outer {
        t_start = max(outer.x, 0.0);
        if inner.x > t_start && inner.x < outer.y {
            t_end = inner.x;
        } else {
            t_end = outer.y;
        }
    } else if cam_dist > r_inner {
        if inner.x > 0.0 && inner.x < outer.y {
            t_end = inner.x;
        } else {
            t_end = outer.y;
        }
    } else {
        t_start = max(inner.y, 0.0);
        t_end = outer.y;
    }
    if t_end <= t_start {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    let dt = (t_end - t_start) / f32(SAMPLES);
    var rgb = vec3<f32>(0.0);

    for (var i = 0; i < SAMPLES; i = i + 1) {
        let t = t_start + (f32(i) + 0.5) * dt;
        let p = cam + dir * t;
        let to_p = p - center;
        let r = length(to_p);
        let up = to_p / r;

        // Magnetic latitude.
        let cos_lat_mag = clamp(dot(up, mag_axis), -1.0, 1.0);
        let lat_mag = asin(cos_lat_mag);

        let mask = band_mask(lat_mag, band_center, band_half);
        if mask < 1e-4 { continue; }

        let h_norm = clamp((r - r_inner) / max(r_outer - r_inner, 1.0), 0.0, 1.0);

        // Longitude around the magnetic axis.
        let east = magnetic_east(mag_axis);
        let north_proj = mag_axis - up * dot(mag_axis, up);
        let north_len = length(north_proj);
        let n_unit = select(east, north_proj / max(north_len, 1e-6), north_len > 1e-6);
        let e_local = normalize(cross(mag_axis, n_unit));
        let lon = atan2(dot(up, e_local), dot(up, n_unit));

        // Curtain noise: 2-D value noise in (curtain_freq · lon + speed·t,
        // h_norm · 3) then sharpened.
        let n = value_noise2(vec2<f32>(curtain_freq * lon + curtain_speed * elapsed, h_norm * 3.0));
        let curtain = pow(n, 2.0);

        let day = day_side_factor(up, sun);
        let color = vertical_color(h_norm);

        rgb = rgb + color * (mask * curtain * day);
    }

    // Normalise by sample count and scale by strength. The (r_outer - r_inner)
    // factor turns the per-sample sum into a quasi-integral in metres so the
    // visual brightness is approximately invariant to SAMPLES.
    let path_m = t_end - t_start;
    rgb = rgb * (path_m / f32(SAMPLES)) * strength * 1.0e-5;

    return vec4<f32>(rgb, 1.0);
}
