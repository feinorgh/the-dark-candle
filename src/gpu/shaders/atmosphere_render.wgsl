// atmosphere_render.wgsl — Uber compute shader for atmosphere visualization.
//
// Per-pixel: camera ray → sky scattering → cloud ray-march → DDA terrain
// raycast → Lambertian shading → cloud shadows → fog → composite → tonemap.
//
// Workgroup size: 16×16 = 256 threads.

// ─── Uniforms ───────────────────────────────────────────────────────────────

struct Camera {
    eye:        vec3<f32>,
    _pad0:      f32,
    forward:    vec3<f32>,
    _pad1:      f32,
    right:      vec3<f32>,
    _pad2:      f32,
    up:         vec3<f32>,
    _pad3:      f32,
    half_w:     f32,
    half_h:     f32,
    width:      u32,
    height:     u32,
}

struct Sun {
    direction:  vec3<f32>,  // unit vector TOWARD the sun
    _pad0:      f32,
    color:      vec3<f32>,
    intensity:  f32,
    elevation:  f32,        // radians above horizon
    ambient:    f32,
    _pad1:      vec2<f32>,
}

struct ScatterParams {
    rayleigh_coeff:         vec3<f32>,
    mie_coeff:              f32,
    mie_g:                  f32,
    atmosphere_radius:      f32,
    planet_radius:          f32,
    rayleigh_scale_height:  f32,
    mie_scale_height:       f32,
    sun_intensity:          f32,
    _pad0:                  vec2<f32>,
}

struct CloudParams {
    extinction_coeff:   f32,
    scattering_albedo:  f32,
    forward_scatter_g:  f32,
    ambient_factor:     f32,
    max_march_distance: f32,
    step_size:          f32,
    density_threshold:  f32,
    _pad0:              f32,
}

struct ShadowParams {
    sun_direction:      vec3<f32>,
    shadow_softness:    f32,
    min_shadow_factor:  f32,
    extinction_coeff:   f32,
    _pad0:              vec2<f32>,
}

struct FogUniforms {
    fog_density_base:   f32,
    fog_height_falloff: f32,
    humidity_scale:     f32,
    temperature_factor: f32,
    fog_color:          vec3<f32>,
    max_fog_distance:   f32,
}

struct GridInfo {
    grid_size:      u32,
    num_materials:  u32,
    enable_clouds:  u32,
    enable_fog:     u32,
    enable_shadows: u32,
    enable_stars:   u32,
    time_hash:      u32,
    _pad0:          u32,
}

// ─── Bind groups ────────────────────────────────────────────────────────────

@group(0) @binding(0) var<uniform> camera:          Camera;
@group(0) @binding(1) var<uniform> sun:             Sun;
@group(0) @binding(2) var<uniform> scatter_params:  ScatterParams;
@group(0) @binding(3) var<uniform> cloud_params:    CloudParams;
@group(0) @binding(4) var<uniform> shadow_params:   ShadowParams;
@group(0) @binding(5) var<uniform> fog_params:      FogUniforms;
@group(0) @binding(6) var<uniform> grid_info:       GridInfo;

// Voxel data: packed as [material_id: u32, temperature_bits: u32] per voxel.
@group(1) @binding(0) var<storage, read> voxels:    array<u32>;
// Material colors: [r, g, b, transparent_flag] per material, packed as vec4<f32>.
@group(1) @binding(1) var<storage, read> materials: array<vec4<f32>>;
// Cloud LWC field: flat f32 array, same indexing as voxels.
@group(1) @binding(2) var<storage, read> cloud_field: array<f32>;
// Shadow map: flat f32 array, size×size, precomputed on CPU.
@group(1) @binding(3) var<storage, read> shadow_map: array<f32>;
// Humidity field (for fog).
@group(1) @binding(4) var<storage, read> humidity_field: array<f32>;
// Temperature field (for fog).
@group(1) @binding(5) var<storage, read> temperature_field: array<f32>;

// Output image: RGBA u8 packed as u32 per pixel.
@group(2) @binding(0) var<storage, read_write> output: array<u32>;

// ─── Helpers ────────────────────────────────────────────────────────────────

fn voxel_index(x: u32, y: u32, z: u32, size: u32) -> u32 {
    return z * size * size + y * size + x;
}

fn get_material_id(x: u32, y: u32, z: u32, size: u32) -> u32 {
    let idx = voxel_index(x, y, z, size);
    return voxels[idx * 2u];
}

fn get_temperature(x: u32, y: u32, z: u32, size: u32) -> f32 {
    let idx = voxel_index(x, y, z, size);
    return bitcast<f32>(voxels[idx * 2u + 1u]);
}

fn is_air(mat_id: u32) -> bool {
    return mat_id == 0u;
}

fn pack_rgba(r: u32, g: u32, b: u32, a: u32) -> u32 {
    return r | (g << 8u) | (b << 16u) | (a << 24u);
}

// ─── Ray-sphere intersection ────────────────────────────────────────────────

fn ray_sphere(origin: vec3<f32>, dir: vec3<f32>, radius: f32) -> vec2<f32> {
    // Returns (t_near, t_far). If no intersection, t_near > t_far.
    let b = dot(origin, dir);
    let c = dot(origin, origin) - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return vec2<f32>(1e20, -1e20); // No hit.
    }
    let sq = sqrt(disc);
    return vec2<f32>(-b - sq, -b + sq);
}

// ─── Phase functions ────────────────────────────────────────────────────────

fn rayleigh_phase(cos_theta: f32) -> f32 {
    return 3.0 / (16.0 * 3.14159265) * (1.0 + cos_theta * cos_theta);
}

fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (1.0 - g2) / (4.0 * 3.14159265 * pow(denom, 1.5));
}

// ─── Optical depth integration ──────────────────────────────────────────────

fn optical_depth(
    origin: vec3<f32>,
    dir: vec3<f32>,
    path_len: f32,
    num_samples: u32,
) -> vec4<f32> {
    // Returns: xyz = Rayleigh depth per channel, w = Mie depth.
    var ray_depth = vec3<f32>(0.0);
    var mie_depth: f32 = 0.0;
    let step = path_len / f32(num_samples);

    for (var i = 0u; i < num_samples; i++) {
        let t = (f32(i) + 0.5) * step;
        let pos = origin + dir * t;
        let altitude = length(pos) - scatter_params.planet_radius;
        if altitude < 0.0 { break; }

        let ray_density = exp(-altitude / scatter_params.rayleigh_scale_height);
        let mie_density_val = exp(-altitude / scatter_params.mie_scale_height);

        ray_depth += scatter_params.rayleigh_coeff * ray_density * step;
        mie_depth += scatter_params.mie_coeff * mie_density_val * step;
    }

    return vec4<f32>(ray_depth, mie_depth);
}

// ─── Sky scattering ─────────────────────────────────────────────────────────

fn sky_color(ray_dir: vec3<f32>, sun_dir: vec3<f32>, camera_altitude: f32) -> vec3<f32> {
    let camera_pos = vec3<f32>(0.0, scatter_params.planet_radius + camera_altitude, 0.0);

    // Intersect ray with atmosphere.
    let atmos_hit = ray_sphere(camera_pos, ray_dir, scatter_params.atmosphere_radius);
    if atmos_hit.x > atmos_hit.y { return vec3<f32>(0.0); }

    var t_start = max(atmos_hit.x, 0.0);
    var t_end = max(atmos_hit.y, 0.0);
    if t_end <= t_start { return vec3<f32>(0.0); }

    // Check planet surface intersection.
    let surface_hit = ray_sphere(camera_pos, ray_dir, scatter_params.planet_radius);
    if surface_hit.y > t_start && surface_hit.y < t_end {
        t_end = surface_hit.y;
    }

    let path_length = t_end - t_start;
    if path_length <= 0.0 { return vec3<f32>(0.0); }

    let num_samples = 16u;
    let step_size = path_length / f32(num_samples);

    let cos_theta = dot(ray_dir, sun_dir);
    let phase_r = rayleigh_phase(cos_theta);
    let phase_m = henyey_greenstein(cos_theta, scatter_params.mie_g);

    var total_rayleigh = vec3<f32>(0.0);
    var total_mie = vec3<f32>(0.0);

    for (var i = 0u; i < num_samples; i++) {
        let t = t_start + (f32(i) + 0.5) * step_size;
        let sample_pos = camera_pos + ray_dir * t;

        let height_from_center = length(sample_pos);
        let altitude = height_from_center - scatter_params.planet_radius;
        if altitude < 0.0 { break; }

        let ray_density = exp(-altitude / scatter_params.rayleigh_scale_height);
        let mie_density_val = exp(-altitude / scatter_params.mie_scale_height);

        // Sun optical depth (secondary ray-march).
        let sun_hit = ray_sphere(sample_pos, sun_dir, scatter_params.atmosphere_radius);
        var sun_depth = vec4<f32>(0.0);
        if sun_hit.y > 0.0 {
            sun_depth = optical_depth(sample_pos, sun_dir, sun_hit.y, 8u);
        }

        // Check shadow (sun ray hits planet).
        let sun_surface = ray_sphere(sample_pos, sun_dir, scatter_params.planet_radius);
        if sun_surface.y > 0.0 { continue; } // In planet shadow.

        // View optical depth.
        let segment_len = (f32(i) + 0.5) * step_size;
        var view_depth = vec4<f32>(0.0);
        if segment_len > 0.0 {
            view_depth = optical_depth(camera_pos, ray_dir, segment_len, 8u);
        }

        // Beer-Lambert extinction.
        for (var ch = 0u; ch < 3u; ch++) {
            let total_tau = view_depth[ch] + sun_depth[ch] + view_depth.w + sun_depth.w;
            let extinction = exp(-total_tau);

            let ray_scatter = scatter_params.rayleigh_coeff[ch] * ray_density * phase_r;
            let mie_scatter = scatter_params.mie_coeff * mie_density_val * phase_m;

            total_rayleigh[ch] += ray_scatter * extinction * step_size;
            total_mie[ch] += mie_scatter * extinction * step_size;
        }
    }

    return (total_rayleigh + total_mie) * scatter_params.sun_intensity;
}

// ─── Cloud ray-march ────────────────────────────────────────────────────────

fn sample_cloud(pos: vec3<f32>) -> f32 {
    let size = f32(grid_info.grid_size);
    if pos.x < 0.0 || pos.y < 0.0 || pos.z < 0.0 ||
       pos.x >= size || pos.y >= size || pos.z >= size {
        return 0.0;
    }

    // Trilinear interpolation.
    let fx = clamp(pos.x, 0.0, size - 1.001);
    let fy = clamp(pos.y, 0.0, size - 1.001);
    let fz = clamp(pos.z, 0.0, size - 1.001);

    let ix = u32(floor(fx));
    let iy = u32(floor(fy));
    let iz = u32(floor(fz));

    let tx = fx - f32(ix);
    let ty = fy - f32(iy);
    let tz = fz - f32(iz);

    let gs = grid_info.grid_size;
    let ix1 = min(ix + 1u, gs - 1u);
    let iy1 = min(iy + 1u, gs - 1u);
    let iz1 = min(iz + 1u, gs - 1u);

    let c000 = cloud_field[iz  * gs * gs + iy  * gs + ix ];
    let c100 = cloud_field[iz  * gs * gs + iy  * gs + ix1];
    let c010 = cloud_field[iz  * gs * gs + iy1 * gs + ix ];
    let c110 = cloud_field[iz  * gs * gs + iy1 * gs + ix1];
    let c001 = cloud_field[iz1 * gs * gs + iy  * gs + ix ];
    let c101 = cloud_field[iz1 * gs * gs + iy  * gs + ix1];
    let c011 = cloud_field[iz1 * gs * gs + iy1 * gs + ix ];
    let c111 = cloud_field[iz1 * gs * gs + iy1 * gs + ix1];

    let c00 = mix(c000, c100, tx);
    let c01 = mix(c001, c101, tx);
    let c10 = mix(c010, c110, tx);
    let c11 = mix(c011, c111, tx);
    let c0  = mix(c00,  c10,  ty);
    let c1  = mix(c01,  c11,  ty);
    return mix(c0, c1, tz);
}

fn march_clouds(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    sun_dir: vec3<f32>,
    sun_color: vec3<f32>,
) -> vec4<f32> {
    // Returns: xyz = accumulated cloud color, w = remaining transmittance.
    if grid_info.enable_clouds == 0u { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }

    var transmittance: f32 = 1.0;
    var color = vec3<f32>(0.0);
    var distance: f32 = 0.0;

    let cos_theta = dot(ray_dir, sun_dir);
    let phase = henyey_greenstein(cos_theta, cloud_params.forward_scatter_g);

    let num_steps = u32(cloud_params.max_march_distance / cloud_params.step_size);

    for (var i = 0u; i < num_steps; i++) {
        let pos = ray_origin + ray_dir * distance;
        let density = sample_cloud(pos);

        if density > cloud_params.density_threshold {
            let extinction = cloud_params.extinction_coeff * density;
            let delta_tau = extinction * cloud_params.step_size;
            let seg_trans = exp(-delta_tau);

            let scatter_strength = cloud_params.scattering_albedo * phase;
            let light_contrib = scatter_strength + cloud_params.ambient_factor;

            let in_scatter = sun_color * light_contrib * (1.0 - seg_trans) * transmittance;
            color += in_scatter;
            transmittance *= seg_trans;

            if transmittance < 0.01 {
                transmittance = 0.0;
                break;
            }
        }

        distance += cloud_params.step_size;
    }

    return vec4<f32>(color, transmittance);
}

// ─── DDA terrain raycast ────────────────────────────────────────────────────

struct DdaResult {
    hit:        bool,
    pos:        vec3<u32>,
    face_axis:  u32,
    face_sign:  f32,
    t:          f32,
}

fn ray_aabb(origin: vec3<f32>, dir: vec3<f32>, size: f32) -> vec2<f32> {
    var t_min: f32 = -1e20;
    var t_max: f32 = 1e20;

    for (var i = 0u; i < 3u; i++) {
        if abs(dir[i]) > 1e-10 {
            let inv_d = 1.0 / dir[i];
            var t0 = (0.0 - origin[i]) * inv_d;
            var t1 = (size - origin[i]) * inv_d;
            if t0 > t1 { let tmp = t0; t0 = t1; t1 = tmp; }
            t_min = max(t_min, t0);
            t_max = min(t_max, t1);
        } else {
            if origin[i] < 0.0 || origin[i] > size {
                return vec2<f32>(1e20, -1e20); // Miss.
            }
        }
    }
    return vec2<f32>(t_min, t_max);
}

fn dda_march(origin: vec3<f32>, dir_raw: vec3<f32>, max_dist: f32) -> DdaResult {
    var result: DdaResult;
    result.hit = false;

    let len = length(dir_raw);
    if len < 1e-10 { return result; }
    let dir = dir_raw / len;
    let fs = f32(grid_info.grid_size);

    let aabb = ray_aabb(origin, dir, fs);
    if aabb.x >= aabb.y || aabb.y < 0.0 { return result; }

    let t_start = max(aabb.x, 0.0) + 0.001;
    let pos = origin + dir * t_start;

    let bound = i32(grid_info.grid_size);
    var ix = clamp(i32(floor(pos.x)), 0, bound - 1);
    var iy = clamp(i32(floor(pos.y)), 0, bound - 1);
    var iz = clamp(i32(floor(pos.z)), 0, bound - 1);

    var step_x: i32 = 1; if dir.x < 0.0 { step_x = -1; }
    var step_y: i32 = 1; if dir.y < 0.0 { step_y = -1; }
    var step_z: i32 = 1; if dir.z < 0.0 { step_z = -1; }

    var dt_x: f32 = 1e20; if abs(dir.x) > 1e-10 { dt_x = abs(1.0 / dir.x); }
    var dt_y: f32 = 1e20; if abs(dir.y) > 1e-10 { dt_y = abs(1.0 / dir.y); }
    var dt_z: f32 = 1e20; if abs(dir.z) > 1e-10 { dt_z = abs(1.0 / dir.z); }

    var t_max_x: f32;
    if dir.x >= 0.0 { t_max_x = (f32(ix + 1) - pos.x) * dt_x; }
    else             { t_max_x = (pos.x - f32(ix)) * dt_x; }

    var t_max_y: f32;
    if dir.y >= 0.0 { t_max_y = (f32(iy + 1) - pos.y) * dt_y; }
    else             { t_max_y = (pos.y - f32(iy)) * dt_y; }

    var t_max_z: f32;
    if dir.z >= 0.0 { t_max_z = (f32(iz + 1) - pos.z) * dt_z; }
    else             { t_max_z = (pos.z - f32(iz)) * dt_z; }

    var t_total = t_start;
    var last_axis = 1u;
    let max_steps = grid_info.grid_size * 3u;

    for (var step = 0u; step < max_steps; step++) {
        if ix < 0 || iy < 0 || iz < 0 { return result; }
        let ux = u32(ix);
        let uy = u32(iy);
        let uz = u32(iz);
        if ux >= grid_info.grid_size || uy >= grid_info.grid_size || uz >= grid_info.grid_size {
            return result;
        }

        let mat_id = get_material_id(ux, uy, uz, grid_info.grid_size);
        if !is_air(mat_id) {
            result.hit = true;
            result.pos = vec3<u32>(ux, uy, uz);
            result.face_axis = last_axis;
            if last_axis == 0u { result.face_sign = -f32(step_x); }
            else if last_axis == 1u { result.face_sign = -f32(step_y); }
            else { result.face_sign = -f32(step_z); }
            result.t = t_total;
            return result;
        }

        // Amanatides & Woo step.
        if t_max_x < t_max_y {
            if t_max_x < t_max_z {
                t_total = t_start + t_max_x;
                t_max_x += dt_x;
                ix += step_x;
                last_axis = 0u;
            } else {
                t_total = t_start + t_max_z;
                t_max_z += dt_z;
                iz += step_z;
                last_axis = 2u;
            }
        } else if t_max_y < t_max_z {
            t_total = t_start + t_max_y;
            t_max_y += dt_y;
            iy += step_y;
            last_axis = 1u;
        } else {
            t_total = t_start + t_max_z;
            t_max_z += dt_z;
            iz += step_z;
            last_axis = 2u;
        }

        if t_total - t_start > max_dist { return result; }
    }

    return result;
}

// Shadow ray: returns true if path to sun is blocked by solid voxel.
fn is_shadowed(hit_pos: vec3<f32>, to_light: vec3<f32>) -> bool {
    let shadow_origin = hit_pos + to_light * 0.5;
    let max_d = f32(grid_info.grid_size) * 3.0;
    let shadow_hit = dda_march(shadow_origin, to_light, max_d);
    return shadow_hit.hit;
}

// ─── Surface normal estimation ──────────────────────────────────────────────

fn estimate_normal(x: u32, y: u32, z: u32) -> vec3<f32> {
    let size = grid_info.grid_size;
    // Central difference on material density (air=0, solid=1).
    var gx: f32 = 0.0;
    var gy: f32 = 0.0;
    var gz: f32 = 0.0;

    if x > 0u && x < size - 1u {
        let left  = select(0.0, 1.0, !is_air(get_material_id(x - 1u, y, z, size)));
        let right = select(0.0, 1.0, !is_air(get_material_id(x + 1u, y, z, size)));
        gx = right - left;
    }
    if y > 0u && y < size - 1u {
        let below = select(0.0, 1.0, !is_air(get_material_id(x, y - 1u, z, size)));
        let above = select(0.0, 1.0, !is_air(get_material_id(x, y + 1u, z, size)));
        gy = above - below;
    }
    if z > 0u && z < size - 1u {
        let back  = select(0.0, 1.0, !is_air(get_material_id(x, y, z - 1u, size)));
        let front = select(0.0, 1.0, !is_air(get_material_id(x, y, z + 1u, size)));
        gz = front - back;
    }

    let grad = vec3<f32>(gx, gy, gz);
    let grad_len = length(grad);
    if grad_len > 0.001 { return grad / grad_len; }
    return vec3<f32>(0.0, 1.0, 0.0); // Fallback up.
}

// ─── Fog ────────────────────────────────────────────────────────────────────

fn compute_fog(ray_origin: vec3<f32>, ray_dir: vec3<f32>, distance: f32) -> vec4<f32> {
    // Returns: xyz = fog color contribution, w = transmittance.
    if grid_info.enable_fog == 0u { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }

    let dir = normalize(ray_dir);
    let integration_dist = min(distance, fog_params.max_fog_distance);
    let step_len: f32 = 1.0;
    let num_steps = u32(ceil(integration_dist / step_len));
    let size = grid_info.grid_size;

    var optical_depth_val: f32 = 0.0;

    for (var i = 0u; i < num_steps; i++) {
        let t = f32(i) * step_len;
        let pos = ray_origin + dir * t;

        if pos.x < 0.0 || pos.y < 0.0 || pos.z < 0.0 ||
           pos.x >= f32(size) || pos.y >= f32(size) || pos.z >= f32(size) {
            break;
        }

        let ix = u32(floor(pos.x));
        let iy = u32(floor(pos.y));
        let iz = u32(floor(pos.z));

        if ix < size && iy < size && iz < size {
            let idx = iz * size * size + iy * size + ix;
            let humidity = humidity_field[idx];
            let temperature = temperature_field[idx];

            // Simple fog density model.
            let height_factor = exp(-pos.y * fog_params.fog_height_falloff);
            let humidity_factor = 1.0 + humidity * fog_params.humidity_scale;
            // Cold air amplification (simplified dew point check).
            let cold_factor = max(0.0, (280.0 - temperature) * fog_params.temperature_factor);
            let density = fog_params.fog_density_base * height_factor * humidity_factor * (1.0 + cold_factor);

            optical_depth_val += density * step_len;
        }
    }

    let transmittance = exp(-optical_depth_val);
    let fog_contrib = fog_params.fog_color * (1.0 - transmittance);
    return vec4<f32>(fog_contrib, transmittance);
}

// ─── Star field ─────────────────────────────────────────────────────────────

fn star_brightness(px: u32, py: u32) -> f32 {
    if grid_info.enable_stars == 0u { return 0.0; }

    // Simple hash-based star field.
    var h = px * 374761393u + py * 668265263u + grid_info.time_hash;
    h = (h ^ (h >> 13u)) * 1274126177u;
    h = h ^ (h >> 16u);

    if h % 200u == 0u {
        return f32(h % 100u) / 100.0 * 0.15;
    }
    return 0.0;
}

// ─── Incandescence (hot voxels) ─────────────────────────────────────────────

fn incandescence(temperature: f32) -> vec3<f32> {
    if temperature < 800.0 { return vec3<f32>(0.0); }

    let t = (temperature - 800.0) / 1200.0; // 0 at 800K, 1 at 2000K
    let t4 = t * t * t * t; // T⁴ Stefan-Boltzmann scaling
    let r = clamp(t * 1.5, 0.0, 1.0);
    let g = clamp((t - 0.3) * 1.2, 0.0, 1.0);
    let b = clamp((t - 0.6) * 1.5, 0.0, 1.0);
    return vec3<f32>(r, g, b) * t4 * 5.0;
}

// ─── Main compute kernel ────────────────────────────────────────────────────

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;

    if px >= camera.width || py >= camera.height { return; }

    // Camera ray.
    let u = (2.0 * f32(px) / f32(camera.width) - 1.0) * camera.half_w;
    let v = (1.0 - 2.0 * f32(py) / f32(camera.height)) * camera.half_h;
    let dir = normalize(camera.forward + camera.right * u + camera.up * v);

    let sun_dir = sun.direction; // Unit vector toward sun.
    let to_light = sun_dir;      // For shadow rays (toward sun).
    let light_dir = -sun_dir;    // From light toward scene.

    // 1. Sky color (Rayleigh + Mie scattering).
    let sky = sky_color(dir, sun_dir, camera.eye.y);

    // 2. Cloud ray-march.
    let cloud_result = march_clouds(camera.eye, dir, sun_dir, sun.color);
    let cloud_color = cloud_result.xyz;
    let cloud_trans = cloud_result.w;

    // 3. Terrain DDA raycast.
    let max_march = f32(grid_info.grid_size) * 3.0;
    let terrain_hit = dda_march(camera.eye, dir, max_march);

    var final_color: vec3<f32>;

    if terrain_hit.hit {
        let hx = terrain_hit.pos.x;
        let hy = terrain_hit.pos.y;
        let hz = terrain_hit.pos.z;
        let mat_id = get_material_id(hx, hy, hz, grid_info.grid_size);
        let mat_color = materials[mat_id].xyz;
        let temperature = get_temperature(hx, hy, hz, grid_info.grid_size);

        // Face normal + gradient blending.
        var face_n = vec3<f32>(0.0);
        face_n[terrain_hit.face_axis] = terrain_hit.face_sign;
        let grad_n = estimate_normal(hx, hy, hz);
        let blended_n = normalize(face_n * 0.5 + grad_n * 0.5);

        // Lambertian shading.
        let n_dot_l = max(dot(blended_n, to_light), 0.0);

        // Shadow ray.
        let hit_center = vec3<f32>(f32(hx) + 0.5, f32(hy) + 0.5, f32(hz) + 0.5);
        var shadow: f32 = 1.0;
        if is_shadowed(hit_center, to_light) { shadow = 0.0; }

        // Cloud shadow (from precomputed shadow map).
        var cloud_shadow: f32 = 1.0;
        if grid_info.enable_shadows != 0u {
            let si = hz * grid_info.grid_size + hx;
            if si < arrayLength(&shadow_map) {
                cloud_shadow = shadow_map[si];
            }
        }

        let diffuse = n_dot_l * sun.intensity * shadow * cloud_shadow;
        var terrain_color = mat_color * (sun.ambient + diffuse);

        // Incandescence glow for hot voxels.
        terrain_color += incandescence(temperature);

        // Fog.
        let fog_result = compute_fog(camera.eye, dir, terrain_hit.t);
        let fog_contrib = fog_result.xyz;
        let fog_trans = fog_result.w;
        terrain_color = terrain_color * fog_trans + fog_contrib;

        // Depth fog (basic distance fade to sky).
        let fog_start = f32(grid_info.grid_size) * 0.5;
        let fog_end = f32(grid_info.grid_size) * 2.5;
        let depth_fog = clamp((terrain_hit.t - fog_start) / (fog_end - fog_start), 0.0, 1.0);

        // Composite: cloud over terrain, then fade to sky with depth fog.
        let terrain_with_sky = mix(terrain_color, sky, depth_fog);
        final_color = cloud_color + cloud_trans * terrain_with_sky;
    } else {
        // Sky only — composite clouds over sky.
        final_color = cloud_color + cloud_trans * sky;
    }

    // Star field (only visible at night / dark sky).
    let sky_brightness = max(max(final_color.x, final_color.y), final_color.z);
    if sky_brightness < 0.05 {
        let star = star_brightness(px, py);
        final_color += vec3<f32>(star);
    }

    // Reinhard tonemap.
    let mapped = final_color / (vec3<f32>(1.0) + final_color) * 255.0;
    let r = u32(clamp(mapped.x, 0.0, 255.0));
    let g = u32(clamp(mapped.y, 0.0, 255.0));
    let b = u32(clamp(mapped.z, 0.0, 255.0));

    let pixel_idx = py * camera.width + px;
    output[pixel_idx] = pack_rgba(r, g, b, 255u);
}
