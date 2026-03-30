// GPU-accelerated terrain projection renderer.
//
// Two compute passes:
//   1. elevation_pass: inverse projection → nearest cell → IDW interpolation → noise → store elevation + cell index
//   2. hillshade_pass: read elevation grid → gradient → hillshade → base colour → modulate → RGBA output

// ─── Uniforms ────────────────────────────────────────────────────────────────

struct Params {
    width: u32,
    height: u32,
    projection: u32,       // 0=equirect, 1=mollweide, 2=ortho
    colour_mode: u32,      // 0=elevation (computed from detailed elev), 1=cell (pre-uploaded)
    center_lon: f32,       // for orthographic
    cell_size_m: f32,
    z_factor: f32,
    radius_m: f32,
    noise_seed: u32,
    noise_seed_ridged: u32,
    num_cells: u32,
    lat_bins: u32,         // 180
    lon_bins: u32,         // 360
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    // Hillshade sun direction (NW cartographic).
    sun_x: f32,
    sun_y: f32,
    sun_z: f32,
    _pad3: f32,
}

@group(0) @binding(0) var<uniform> params: Params;

// ─── Data buffers (read-only) ────────────────────────────────────────────────

// Per-cell data, indexed by cell ID.
@group(1) @binding(0) var<storage, read> cell_positions: array<vec4<f32>>;   // xyz, w=0
@group(1) @binding(1) var<storage, read> cell_elevations: array<f32>;
@group(1) @binding(2) var<storage, read> cell_colors: array<vec4<f32>>;      // pre-computed RGBA
@group(1) @binding(3) var<storage, read> cell_roughness: array<f32>;         // pre-computed [0,1]

// Flattened neighbor adjacency: offset+count per cell, then flat neighbor IDs.
@group(1) @binding(4) var<storage, read> neighbor_offsets: array<vec2<u32>>; // [offset, count]
@group(1) @binding(5) var<storage, read> neighbor_ids: array<u32>;

// Spatial index: 180×360 bins of (offset, count) into spatial_cell_ids.
@group(1) @binding(6) var<storage, read> spatial_bins: array<vec2<u32>>;     // [offset, count]
@group(1) @binding(7) var<storage, read> spatial_cell_ids: array<u32>;

// ─── Intermediate + output buffers ───────────────────────────────────────────

@group(2) @binding(0) var<storage, read_write> elevations: array<f32>;       // width×height
@group(2) @binding(1) var<storage, read_write> cell_indices: array<u32>;     // width×height
@group(2) @binding(2) var<storage, read_write> output: array<u32>;           // RGBA packed

// ─── Constants ───────────────────────────────────────────────────────────────

const PI: f32 = 3.14159265358979323846;
const TAU: f32 = 6.28318530717958647692;
const HALF_PI: f32 = 1.57079632679489661923;
const SQRT_2: f32 = 1.41421356237309504880;
const NAN_MARKER: f32 = -1e30;

// ─── 3D Perlin noise (hash-based, no permutation table) ─────────────────────

// Integer hash combining position + seed.
fn ihash(x: i32, y: i32, z: i32, seed: u32) -> u32 {
    var n = u32(x) * 73856093u ^ u32(y) * 19349663u ^ u32(z) * 83492791u ^ seed;
    n = n ^ (n >> 13u);
    n = n * 1274126177u;
    n = n ^ (n >> 16u);
    return n;
}

// Gradient from hash: one of 12 directions (Perlin's classic gradient set).
fn grad3(hash: u32, x: f32, y: f32, z: f32) -> f32 {
    let h = hash & 15u;
    let u = select(y, x, h < 8u);
    let v = select(select(x, z, h == 12u || h == 14u), y, h < 4u);
    return select(-u, u, (h & 1u) == 0u) + select(-v, v, (h & 2u) == 0u);
}

fn fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn perlin3d(p: vec3<f32>, seed: u32) -> f32 {
    let pi = vec3<i32>(vec3<f32>(floor(p.x), floor(p.y), floor(p.z)));
    let pf = p - vec3<f32>(f32(pi.x), f32(pi.y), f32(pi.z));

    let u = fade(pf.x);
    let v = fade(pf.y);
    let w = fade(pf.z);

    let aaa = grad3(ihash(pi.x,     pi.y,     pi.z,     seed), pf.x,       pf.y,       pf.z);
    let baa = grad3(ihash(pi.x + 1, pi.y,     pi.z,     seed), pf.x - 1.0, pf.y,       pf.z);
    let aba = grad3(ihash(pi.x,     pi.y + 1, pi.z,     seed), pf.x,       pf.y - 1.0, pf.z);
    let bba = grad3(ihash(pi.x + 1, pi.y + 1, pi.z,     seed), pf.x - 1.0, pf.y - 1.0, pf.z);
    let aab = grad3(ihash(pi.x,     pi.y,     pi.z + 1, seed), pf.x,       pf.y,       pf.z - 1.0);
    let bab = grad3(ihash(pi.x + 1, pi.y,     pi.z + 1, seed), pf.x - 1.0, pf.y,       pf.z - 1.0);
    let abb = grad3(ihash(pi.x,     pi.y + 1, pi.z + 1, seed), pf.x,       pf.y - 1.0, pf.z - 1.0);
    let bbb = grad3(ihash(pi.x + 1, pi.y + 1, pi.z + 1, seed), pf.x - 1.0, pf.y - 1.0, pf.z - 1.0);

    let x1 = mix(aaa, baa, u);
    let x2 = mix(aba, bba, u);
    let x3 = mix(aab, bab, u);
    let x4 = mix(abb, bbb, u);

    let y1 = mix(x1, x2, v);
    let y2 = mix(x3, x4, v);

    return mix(y1, y2, w);
}

// FBM: 6 octaves, frequency 50, lacunarity 2.0, persistence 0.5.
fn fbm(p: vec3<f32>, seed: u32) -> f32 {
    var sum = 0.0;
    var amp = 1.0;
    var freq = 50.0;
    var total_amp = 0.0;
    for (var i = 0u; i < 6u; i++) {
        sum += perlin3d(p * freq, seed + i * 31u) * amp;
        total_amp += amp;
        freq *= 2.0;
        amp *= 0.5;
    }
    return sum / total_amp;
}

// Ridged multi-fractal: 4 octaves, frequency 35, lacunarity 2.2.
fn ridged(p: vec3<f32>, seed: u32) -> f32 {
    var sum = 0.0;
    var amp = 1.0;
    var freq = 35.0;
    var weight = 1.0;
    for (var i = 0u; i < 4u; i++) {
        var signal = perlin3d(p * freq, seed + i * 31u);
        signal = 1.0 - abs(signal);
        signal *= signal;
        signal *= weight;
        weight = clamp(signal * 2.0, 0.0, 1.0);
        sum += signal * amp;
        freq *= 2.2;
        amp *= 0.5;
    }
    return sum;
}

// Combined noise sample: matches CPU detail.rs TerrainNoise::sample().
fn sample_noise(pos: vec3<f32>, roughness: f32) -> f32 {
    let fbm_val = fbm(pos, params.noise_seed);
    let ridge_val = ridged(pos, params.noise_seed_ridged);
    let mix_factor = roughness * 0.6;
    let combined = fbm_val * (1.0 - mix_factor) + ridge_val * mix_factor;
    return combined * roughness * 2000.0;
}

// ─── Projection inverse ──────────────────────────────────────────────────────

// Returns (lat, lon, valid). If !valid, pixel is outside projection boundary.
fn projection_inverse(x: u32, y: u32) -> vec3<f32> {
    let w = f32(params.width);
    let h = f32(params.height);
    let fx = f32(x);
    let fy = f32(y);

    if params.projection == 0u {
        // Equirectangular.
        let lon = -PI + TAU * fx / w;
        let lat = HALF_PI - PI * fy / h;
        return vec3<f32>(lat, lon, 1.0);
    } else if params.projection == 1u {
        // Mollweide.
        let px = (2.0 * fx / w - 1.0) * 2.0 * SQRT_2;
        let py = (1.0 - 2.0 * fy / h) * SQRT_2;
        let test = (px / (2.0 * SQRT_2)) * (px / (2.0 * SQRT_2)) + (py / SQRT_2) * (py / SQRT_2);
        if test > 1.0 {
            return vec3<f32>(0.0, 0.0, 0.0);
        }
        let sin_theta = clamp(py / SQRT_2, -1.0, 1.0);
        let theta = asin(sin_theta);
        let lat_arg = clamp((2.0 * theta + sin(2.0 * theta)) / PI, -1.0, 1.0);
        let lat = asin(lat_arg);
        let cos_theta = cos(theta);
        var lon: f32;
        if abs(cos_theta) < 1e-10 {
            lon = 0.0;
        } else {
            lon = PI * px / (2.0 * SQRT_2 * cos_theta);
        }
        if abs(lon) > PI + 0.001 {
            return vec3<f32>(0.0, 0.0, 0.0);
        }
        return vec3<f32>(lat, clamp(lon, -PI, PI), 1.0);
    } else {
        // Orthographic.
        let px = 2.0 * fx / w - 1.0;
        let py = 1.0 - 2.0 * fy / h;
        let rr = px * px + py * py;
        if rr > 1.0 {
            return vec3<f32>(0.0, 0.0, 0.0);
        }
        let lat = asin(clamp(py, -1.0, 1.0));
        let z = sqrt(max(1.0 - rr, 0.0));
        var lon = params.center_lon + atan2(px, z);
        // Wrap to [-π, π].
        lon = ((lon + PI) % TAU + TAU) % TAU - PI;
        return vec3<f32>(lat, lon, 1.0);
    }
}

// ─── Coordinate conversion ───────────────────────────────────────────────────

fn lat_lon_to_pos(lat: f32, lon: f32) -> vec3<f32> {
    let cos_lat = cos(lat);
    return vec3<f32>(cos_lat * sin(lon), sin(lat), cos_lat * cos(lon));
}

// ─── Spatial index lookup ────────────────────────────────────────────────────

fn nearest_cell(lat: f32, lon: f32, pos: vec3<f32>) -> u32 {
    let lat_bins = params.lat_bins;
    let lon_bins = params.lon_bins;

    let lb = u32(clamp((lat + HALF_PI) / PI * f32(lat_bins), 0.0, f32(lat_bins - 1u)));
    let lob = u32(clamp((lon + PI) / TAU * f32(lon_bins), 0.0, f32(lon_bins - 1u)));

    let cos_lat = max(abs(cos(lat)), 0.01);
    let extra_lon = clamp(i32(2.0 / cos_lat), 2, i32(lon_bins) / 2);

    var best_id = 0u;
    var best_dot = -1e30;

    for (var dlat = -2; dlat <= 2; dlat++) {
        let lat_idx = u32(clamp(i32(lb) + dlat, 0, i32(lat_bins) - 1));
        for (var dlon = -extra_lon; dlon <= extra_lon; dlon++) {
            let lon_idx = u32((i32(lob) + dlon + i32(lon_bins) * 2) % i32(lon_bins));
            let bin_idx = lat_idx * lon_bins + lon_idx;
            let bin = spatial_bins[bin_idx];
            let offset = bin.x;
            let count = bin.y;

            for (var k = 0u; k < count; k++) {
                let cell_id = spatial_cell_ids[offset + k];
                let cpos = cell_positions[cell_id].xyz;
                let d = dot(pos, cpos);
                if d > best_dot {
                    best_dot = d;
                    best_id = cell_id;
                }
            }
        }
    }

    return best_id;
}

// ─── IDW interpolation ──────────────────────────────────────────────────────

fn interpolate_elevation(pos: vec3<f32>, nearest: u32) -> f32 {
    var total_w = 0.0;
    var weighted_e = 0.0;

    // Nearest cell itself.
    let cpos0 = cell_positions[nearest].xyz;
    let d0 = max(1.0 - dot(pos, cpos0), 1e-14);
    let w0 = 1.0 / (d0 * d0);
    total_w += w0;
    weighted_e += w0 * cell_elevations[nearest];

    // Its neighbors.
    let nb = neighbor_offsets[nearest];
    let nb_offset = nb.x;
    let nb_count = nb.y;

    for (var i = 0u; i < nb_count; i++) {
        let nid = neighbor_ids[nb_offset + i];
        let npos = cell_positions[nid].xyz;
        let d = max(1.0 - dot(pos, npos), 1e-14);
        let w = 1.0 / (d * d);
        total_w += w;
        weighted_e += w * cell_elevations[nid];
    }

    return weighted_e / total_w;
}

// ─── Elevation colour ────────────────────────────────────────────────────────

fn lerp_rgb(a: vec3<f32>, b: vec3<f32>, t: f32) -> vec3<f32> {
    return a + (b - a) * t;
}

fn elevation_color(elev: f32) -> vec4<f32> {
    var rgb: vec3<f32>;
    if elev < -4000.0 {
        rgb = vec3<f32>(0.05, 0.05, 0.3);
    } else if elev < 0.0 {
        rgb = lerp_rgb(vec3<f32>(0.05, 0.05, 0.3), vec3<f32>(0.2, 0.4, 0.7), (elev + 4000.0) / 4000.0);
    } else if elev < 500.0 {
        rgb = lerp_rgb(vec3<f32>(0.15, 0.5, 0.15), vec3<f32>(0.3, 0.6, 0.2), elev / 500.0);
    } else if elev < 2000.0 {
        rgb = lerp_rgb(vec3<f32>(0.3, 0.6, 0.2), vec3<f32>(0.6, 0.4, 0.2), (elev - 500.0) / 1500.0);
    } else if elev < 5000.0 {
        rgb = lerp_rgb(vec3<f32>(0.6, 0.4, 0.2), vec3<f32>(0.7, 0.7, 0.7), (elev - 2000.0) / 3000.0);
    } else {
        rgb = lerp_rgb(vec3<f32>(0.7, 0.7, 0.7), vec3<f32>(1.0, 1.0, 1.0), min((elev - 5000.0) / 4000.0, 1.0));
    }
    return vec4<f32>(rgb, 1.0);
}

// ─── Pass 1: elevation computation ───────────────────────────────────────────

@compute @workgroup_size(16, 16)
fn elevation_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;

    if x >= params.width || y >= params.height {
        return;
    }

    let idx = y * params.width + x;

    let inv = projection_inverse(x, y);
    if inv.z < 0.5 {
        // Outside projection boundary.
        elevations[idx] = NAN_MARKER;
        cell_indices[idx] = 0u;
        return;
    }

    let lat = inv.x;
    let lon = inv.y;
    let pos = lat_lon_to_pos(lat, lon);

    let cell = nearest_cell(lat, lon, pos);
    let interp = interpolate_elevation(pos, cell);
    let roughness = cell_roughness[cell];
    let noise_offset = sample_noise(pos, roughness);

    elevations[idx] = interp + noise_offset;
    cell_indices[idx] = cell;
}

// ─── Pass 2: hillshade + colour ──────────────────────────────────────────────

@compute @workgroup_size(16, 16)
fn hillshade_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;

    if x >= params.width || y >= params.height {
        return;
    }

    let idx = y * params.width + x;
    let center = elevations[idx];

    if center < NAN_MARKER + 1.0 {
        // Background pixel: dark blue-black.
        output[idx] = pack_rgba(10u, 10u, 20u, 255u);
        return;
    }

    // Central differences for gradient.
    let left = select(center, get_elev(x - 1u, y), x > 0u);
    let right = select(center, get_elev(x + 1u, y), x + 1u < params.width);
    let up = select(center, get_elev(x, y - 1u), y > 0u);
    let down = select(center, get_elev(x, y + 1u), y + 1u < params.height);

    let dzdx = (right - left) / (2.0 * params.cell_size_m) * params.z_factor;
    let dzdy = (up - down) / (2.0 * params.cell_size_m) * params.z_factor;

    let len = sqrt(dzdx * dzdx + dzdy * dzdy + 1.0);
    let nx = -dzdx / len;
    let ny = dzdy / len;
    let nz = 1.0 / len;

    let shade = clamp(nx * params.sun_x + ny * params.sun_y + nz * params.sun_z, 0.0, 1.0);
    let illumination = shade * 0.55 + 0.45;

    // Base colour.
    var base: vec4<f32>;
    if params.colour_mode == 0u {
        base = elevation_color(center);
    } else {
        let cell = cell_indices[idx];
        base = cell_colors[cell];
    }

    let r = u32(clamp(base.x * illumination, 0.0, 1.0) * 255.0);
    let g = u32(clamp(base.y * illumination, 0.0, 1.0) * 255.0);
    let b = u32(clamp(base.z * illumination, 0.0, 1.0) * 255.0);

    output[idx] = pack_rgba(r, g, b, 255u);
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn get_elev(x: u32, y: u32) -> f32 {
    let v = elevations[y * params.width + x];
    if v < NAN_MARKER + 1.0 {
        // NaN marker — treat as center.
        return elevations[y * params.width + x]; // will be overridden by select in caller
    }
    return v;
}

fn pack_rgba(r: u32, g: u32, b: u32, a: u32) -> u32 {
    return (a << 24u) | (b << 16u) | (g << 8u) | r;
}
