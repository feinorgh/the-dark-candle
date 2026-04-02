// GPU-accelerated terrain noise evaluation.
//
// Evaluates the full NoiseStack::sample() pipeline for batched columns.
// Each thread processes one (lon, lat) pair, producing a surface_radius.
//
// Hash-based Perlin noise (no permutation table) — same approach as
// terrain_projection.wgsl.  Produces visually similar but not bit-identical
// terrain compared to the CPU permutation-table Perlin.

// ─── Uniforms ────────────────────────────────────────────────────────────────

struct NoiseParams {
    // FBM
    fbm_octaves: u32,
    fbm_persistence: f32,
    fbm_lacunarity: f32,
    fbm_base_freq: f32,
    // Ridged
    ridged_octaves: u32,
    ridged_gain: f32,
    ridged_base_freq: f32,
    // Selector
    selector_freq: f32,
    selector_lo: f32,
    selector_hi: f32,
    // Warp
    warp_strength: f32,
    warp_freq: f32,
    // Micro-detail
    micro_freq: f32,
    micro_amplitude: f32,
    // Continent mask
    continent_enabled: u32,
    continent_freq: f32,
    continent_threshold: f32,
    shelf_blend_width: f32,
    ocean_floor_depth: f32,
    ocean_floor_amplitude: f32,
    // Pipeline params
    seed: u32,
    column_count: u32,
    mean_radius: f32,
    height_scale: f32,
}

@group(0) @binding(0) var<uniform> params: NoiseParams;

// ─── Buffers ─────────────────────────────────────────────────────────────────

// Input: (lon, lat) pairs per column.
@group(1) @binding(0) var<storage, read> columns: array<vec2<f32>>;

// Output: surface_radius per column.
@group(2) @binding(0) var<storage, read_write> heights: array<f32>;

// ─── Perlin noise (hash-based, matching terrain_projection.wgsl) ─────────────

fn ihash(x: i32, y: i32, z: i32, seed: u32) -> u32 {
    var n = u32(x) * 73856093u ^ u32(y) * 19349663u ^ u32(z) * 83492791u ^ seed;
    n = n ^ (n >> 13u);
    n = n * 1274126177u;
    n = n ^ (n >> 16u);
    return n;
}

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

// 2D Perlin via 3D with y=0.
fn perlin2d(x: f32, z: f32, seed: u32) -> f32 {
    return perlin3d(vec3<f32>(x, 0.0, z), seed);
}

// ─── Noise pipeline (mirrors NoiseStack::sample()) ───────────────────────────

fn fbm(x: f32, z: f32) -> f32 {
    var value = 0.0;
    var amplitude = 1.0;
    var frequency = params.fbm_base_freq;
    var normalization = 0.0;

    let base_seed = params.seed;
    for (var i = 0u; i < params.fbm_octaves; i++) {
        value += amplitude * perlin2d(x * frequency, z * frequency, base_seed + i);
        normalization += amplitude;
        amplitude *= params.fbm_persistence;
        frequency *= params.fbm_lacunarity;
    }

    if normalization > 0.0 {
        return value / normalization;
    }
    return 0.0;
}

fn ridged(x: f32, z: f32) -> f32 {
    var value = 0.0;
    var weight = 1.0;
    var frequency = params.ridged_base_freq;
    var amplitude = 1.0;
    let persistence = 0.5;
    var normalization = 0.0;

    let base_seed = params.seed + 50u;
    for (var i = 0u; i < params.ridged_octaves; i++) {
        let signal_raw = perlin2d(x * frequency, z * frequency, base_seed + i);
        var signal = 1.0 - abs(signal_raw);
        signal *= signal; // sharpen ridges
        signal *= weight;
        weight = clamp(signal * params.ridged_gain, 0.0, 1.0);

        value += signal * amplitude;
        normalization += amplitude;
        amplitude *= persistence;
        frequency *= params.fbm_lacunarity; // reuse lacunarity (matches CPU)
    }

    if normalization > 0.0 {
        return value / normalization;
    }
    return 0.0;
}

fn selector_value(x: f32, z: f32) -> f32 {
    return perlin2d(x * params.selector_freq, z * params.selector_freq, params.seed + 100u);
}

fn warp_offsets(x: f32, z: f32) -> vec2<f32> {
    let wx = perlin2d(x * params.warp_freq, z * params.warp_freq, params.seed + 200u) * params.warp_strength;
    let wz = perlin2d(x * params.warp_freq, z * params.warp_freq, params.seed + 201u) * params.warp_strength;
    return vec2<f32>(wx, wz);
}

fn micro_detail(x: f32, z: f32) -> f32 {
    return perlin2d(x * params.micro_freq, z * params.micro_freq, params.seed + 300u) * params.micro_amplitude;
}

fn continent_value(x: f32, z: f32) -> f32 {
    return perlin2d(x * params.continent_freq, z * params.continent_freq, params.seed + 350u);
}

fn ocean_floor_noise(x: f32, z: f32) -> f32 {
    return perlin2d(x * 0.01, z * 0.01, params.seed + 360u) * params.ocean_floor_amplitude;
}

fn smoothstep_custom(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

fn continent_blend(x: f32, z: f32) -> f32 {
    let cv = continent_value(x, z);
    let threshold = params.continent_threshold;
    let shelf = params.shelf_blend_width;
    let ocean_edge = threshold - shelf;

    if cv >= threshold {
        return 1.0;
    } else if cv <= ocean_edge {
        return 0.0;
    }
    return smoothstep_custom(ocean_edge, threshold, cv);
}

// Full terrain height at (x, z) — mirrors NoiseStack::sample().
fn sample_land(x: f32, z: f32) -> f32 {
    // 1. Domain warp
    let warp = warp_offsets(x, z);
    let sx = x + warp.x;
    let sz = z + warp.y;

    // 2. Selector
    let sel = selector_value(sx, sz);

    // 3. FBM and/or ridged
    if sel < params.selector_lo {
        return fbm(sx, sz) * 0.6;
    } else if sel > params.selector_hi {
        return ridged(sx, sz) * 1.5 - 0.5;
    }
    // Blended transition
    let t = smoothstep_custom(params.selector_lo, params.selector_hi, sel);
    let fbm_val = fbm(sx, sz) * 0.6;
    let ridged_val = ridged(sx, sz) * 1.5 - 0.5;
    return mix(fbm_val, ridged_val, t);
}

fn sample_terrain(x: f32, z: f32) -> f32 {
    let land = sample_land(x, z);

    if params.continent_enabled == 0u {
        return land;
    }

    let blend = continent_blend(x, z);

    if blend >= 1.0 {
        return land;
    }
    // Ocean floor
    let ocean = -params.ocean_floor_depth + ocean_floor_noise(x, z);
    if blend <= 0.0 {
        return ocean;
    }
    return mix(ocean, land, blend);
}

// ─── Entry point ─────────────────────────────────────────────────────────────

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.column_count {
        return;
    }

    let col = columns[idx];
    let lon = col.x;
    let lat = col.y;

    // NoiseStack::sample(lon, lat)
    let combined = sample_terrain(lon, lat);

    // surface_radius_at(lat, lon, combined) = mean_radius + combined * height_scale
    heights[idx] = params.mean_radius + combined * params.height_scale;
}
