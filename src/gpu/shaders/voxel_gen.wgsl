// GPU-accelerated voxel generation for V2 cubed-sphere chunks.
//
// Two-pass pipeline:
//   Pass 1 (surface_pass): 1024 threads/chunk compute surface_radius per column
//                          + parallel reduction for min/max → chunk classification.
//   Pass 2 (voxel_pass):   32768 threads/chunk fill material + density for mixed chunks.
//
// Reuses hash-based Perlin3D from noise_eval.wgsl (identical implementation).

// ─── Constants ───────────────────────────────────────────────────────────────

const CHUNK_SIZE: u32 = 32u;
const CHUNK_AREA: u32 = 1024u;  // 32 × 32
const CHUNK_VOLUME: u32 = 32768u;  // 32 × 32 × 32

// Material IDs (must match MaterialId constants in Rust).
const MAT_AIR: u32 = 0u;
const MAT_STONE: u32 = 1u;
const MAT_DIRT: u32 = 2u;
const MAT_WATER: u32 = 3u;
const MAT_IRON: u32 = 4u;
const MAT_SAND: u32 = 6u;
const MAT_GRASS: u32 = 7u;
const MAT_LAVA: u32 = 10u;
const MAT_SANDSTONE: u32 = 20u;
const MAT_LIMESTONE: u32 = 21u;
const MAT_GRANITE: u32 = 22u;
const MAT_BASALT: u32 = 23u;
const MAT_COAL: u32 = 24u;
const MAT_COPPER_ORE: u32 = 25u;
const MAT_GOLD_ORE: u32 = 26u;
const MAT_QUARTZ_CRYSTAL: u32 = 27u;

// Chunk classification flags.
const CLASS_MIXED: u32 = 0u;
const CLASS_ALL_AIR: u32 = 1u;
const CLASS_ALL_SOLID: u32 = 2u;

// ─── Uniforms ────────────────────────────────────────────────────────────────

// Noise parameters — same layout as noise_eval.wgsl NoiseParams (96 bytes).
struct NoiseParams {
    fbm_octaves: u32,
    fbm_persistence: f32,
    fbm_lacunarity: f32,
    fbm_base_freq: f32,
    ridged_octaves: u32,
    ridged_gain: f32,
    ridged_base_freq: f32,
    selector_freq: f32,
    selector_lo: f32,
    selector_hi: f32,
    warp_strength: f32,
    warp_freq: f32,
    micro_freq: f32,
    micro_amplitude: f32,
    continent_enabled: u32,
    continent_freq: f32,
    continent_threshold: f32,
    shelf_blend_width: f32,
    ocean_floor_depth: f32,
    ocean_floor_amplitude: f32,
    seed: u32,
    _pad0: u32,
    mean_radius: f32,
    height_scale: f32,
}

// Per-chunk descriptor uploaded for each chunk in the batch.
struct ChunkDesc {
    center: vec4<f32>,         // xyz = world center, w = unused
    rotation: vec4<f32>,       // quaternion (x, y, z, w)
    tangent_scale: vec4<f32>,  // xyz = scale per axis, w = unused
    base_r: f32,
    top_r: f32,
    sea_level: f32,
    lod_scale: f32,
    soil_depth: f32,
    cave_threshold: f32,
    half_diag: f32,
    chunk_index: u32,          // index into output buffers
}

// Global dispatch parameters.
struct DispatchParams {
    chunk_count: u32,
    rotation_axis: vec4<f32>,  // planet rotation axis (xyz, w=unused)
}

// ─── Bind groups ─────────────────────────────────────────────────────────────

@group(0) @binding(0) var<uniform> noise_params: NoiseParams;
@group(0) @binding(1) var<uniform> dispatch_params: DispatchParams;

@group(1) @binding(0) var<storage, read> chunks: array<ChunkDesc>;

// Pass 1 outputs:
@group(2) @binding(0) var<storage, read_write> surface_radii: array<f32>;
// Per-chunk classification + min/max surface:
// [chunk_index * 4 + 0] = classification (CLASS_*)
// [chunk_index * 4 + 1] = min_surface (as u32 bits)
// [chunk_index * 4 + 2] = max_surface (as u32 bits)
// [chunk_index * 4 + 3] = solid_material (for AllSolid)
@group(2) @binding(1) var<storage, read_write> chunk_info: array<atomic<u32>>;

// Pass 2 outputs:
@group(3) @binding(0) var<storage, read_write> voxel_materials: array<u32>;
@group(3) @binding(1) var<storage, read_write> voxel_densities: array<f32>;

// ─── Perlin noise (hash-based, identical to noise_eval.wgsl) ─────────────────

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

fn perlin2d(x: f32, z: f32, seed: u32) -> f32 {
    return perlin3d(vec3<f32>(x, 0.0, z), seed);
}

// ─── Noise pipeline (mirrors NoiseStack::sample()) ───────────────────────────

fn fbm(x: f32, z: f32) -> f32 {
    var value = 0.0;
    var amplitude = 1.0;
    var frequency = noise_params.fbm_base_freq;
    var normalization = 0.0;

    let base_seed = noise_params.seed;
    for (var i = 0u; i < noise_params.fbm_octaves; i++) {
        value += amplitude * perlin2d(x * frequency, z * frequency, base_seed + i);
        normalization += amplitude;
        amplitude *= noise_params.fbm_persistence;
        frequency *= noise_params.fbm_lacunarity;
    }

    if normalization > 0.0 {
        return value / normalization;
    }
    return 0.0;
}

fn ridged(x: f32, z: f32) -> f32 {
    var value = 0.0;
    var weight = 1.0;
    var frequency = noise_params.ridged_base_freq;
    var amplitude = 1.0;
    let persistence = 0.5;
    var normalization = 0.0;

    let base_seed = noise_params.seed + 50u;
    for (var i = 0u; i < noise_params.ridged_octaves; i++) {
        let signal_raw = perlin2d(x * frequency, z * frequency, base_seed + i);
        var signal = 1.0 - abs(signal_raw);
        signal *= signal;
        signal *= weight;
        weight = clamp(signal * noise_params.ridged_gain, 0.0, 1.0);

        value += signal * amplitude;
        normalization += amplitude;
        amplitude *= persistence;
        frequency *= noise_params.fbm_lacunarity;
    }

    if normalization > 0.0 {
        return value / normalization;
    }
    return 0.0;
}

fn smoothstep_custom(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

fn sample_terrain(x: f32, z: f32) -> f32 {
    // Domain warp
    let wx = perlin2d(x * noise_params.warp_freq, z * noise_params.warp_freq, noise_params.seed + 200u) * noise_params.warp_strength;
    let wz = perlin2d(x * noise_params.warp_freq, z * noise_params.warp_freq, noise_params.seed + 201u) * noise_params.warp_strength;
    let sx = x + wx;
    let sz = z + wz;

    // Selector
    let sel = perlin2d(sx * noise_params.selector_freq, sz * noise_params.selector_freq, noise_params.seed + 100u);

    // FBM and ridged blend
    var land: f32;
    if sel < noise_params.selector_lo {
        land = fbm(sx, sz) * 0.6;
    } else if sel > noise_params.selector_hi {
        land = ridged(sx, sz) * 1.5 - 0.5;
    } else {
        let t = smoothstep_custom(noise_params.selector_lo, noise_params.selector_hi, sel);
        let fbm_val = fbm(sx, sz) * 0.6;
        let ridged_val = ridged(sx, sz) * 1.5 - 0.5;
        land = mix(fbm_val, ridged_val, t);
    }

    // Micro-detail
    land += perlin2d(sx * noise_params.micro_freq, sz * noise_params.micro_freq, noise_params.seed + 300u) * noise_params.micro_amplitude;

    if noise_params.continent_enabled == 0u {
        return land;
    }

    // Continent blend
    let cv = perlin2d(sx * noise_params.continent_freq, sz * noise_params.continent_freq, noise_params.seed + 350u);
    let threshold = noise_params.continent_threshold;
    let shelf = noise_params.shelf_blend_width;
    let ocean_edge = threshold - shelf;

    if cv >= threshold {
        return land;
    }

    let ocean = -noise_params.ocean_floor_depth + perlin2d(sx * 0.01, sz * 0.01, noise_params.seed + 360u) * noise_params.ocean_floor_amplitude;

    if cv <= ocean_edge {
        return ocean;
    }

    let blend = smoothstep_custom(ocean_edge, threshold, cv);
    return mix(ocean, land, blend);
}

// ─── Coordinate helpers ──────────────────────────────────────────────────────

// Rotate a vector by a quaternion.
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

// Compute (lat, lon) from a world position, given the planet rotation axis.
fn lat_lon(pos: vec3<f32>, axis: vec3<f32>) -> vec2<f32> {
    let len = length(pos);
    if len < 1e-6 {
        return vec2<f32>(0.0, 0.0);
    }
    let dir = pos / len;

    // Latitude = asin(dot(dir, axis))
    let sin_lat = clamp(dot(dir, axis), -1.0, 1.0);
    let lat = asin(sin_lat);

    // Project onto equatorial plane
    let equatorial = dir - axis * sin_lat;
    let eq_len = length(equatorial);
    if eq_len < 1e-6 {
        return vec2<f32>(lat, 0.0);
    }
    let eq_norm = equatorial / eq_len;

    // Build equatorial basis
    let ref_x = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), abs(axis.x) < 0.9);
    let east = normalize(cross(axis, ref_x));
    let north_eq = normalize(cross(east, axis));

    let lon = atan2(dot(eq_norm, north_eq), dot(eq_norm, east));
    return vec2<f32>(lat, lon);
}

// ─── Material assignment (mirrors material_at_radius) ────────────────────────

fn strata_material_gpu(depth: f32, wx: f32, wy: f32, wz: f32) -> u32 {
    let n = perlin3d(vec3<f32>(wx * 0.02, wy * 0.02, wz * 0.02), noise_params.seed + 400u);

    if depth < 20.0 {
        return select(MAT_LIMESTONE, MAT_SANDSTONE, n > 0.0);
    } else if depth < 60.0 {
        return MAT_STONE;
    } else {
        return select(MAT_BASALT, MAT_GRANITE, n > 0.0);
    }
}

fn ore_material_gpu(depth: f32, wx: f32, wy: f32, wz: f32) -> u32 {
    // Coal: 5–30m
    if depth >= 5.0 && depth <= 30.0 {
        if perlin3d(vec3<f32>(wx * 0.08, wy * 0.08, wz * 0.08), noise_params.seed + 500u) < -0.15 {
            return MAT_COAL;
        }
    }
    // Copper: 15–50m
    if depth >= 15.0 && depth <= 50.0 {
        if perlin3d(vec3<f32>(wx * 0.06, wy * 0.06, wz * 0.06), noise_params.seed + 501u) < -0.20 {
            return MAT_COPPER_ORE;
        }
    }
    // Iron: 30–80m
    if depth >= 30.0 && depth <= 80.0 {
        if perlin3d(vec3<f32>(wx * 0.05, wy * 0.05, wz * 0.05), noise_params.seed + 502u) < -0.25 {
            return MAT_IRON;
        }
    }
    // Gold: 50m+
    if depth >= 50.0 {
        if perlin3d(vec3<f32>(wx * 0.04, wy * 0.04, wz * 0.04), noise_params.seed + 503u) < -0.35 {
            return MAT_GOLD_ORE;
        }
    }
    return 0xFFFFu; // Sentinel: no ore
}

fn is_cave_gpu(cave_threshold: f32, wx: f32, wy: f32, wz: f32) -> bool {
    // Caverns (cathedral-sized)
    let cavern = perlin3d(vec3<f32>(wx * 0.01, wy * 0.01, wz * 0.01), noise_params.seed + 600u);
    if cavern < cave_threshold * 0.5 {
        return true;
    }
    // Tunnels (narrow)
    let tunnel = perlin3d(vec3<f32>(wx * 0.04, wy * 0.04, wz * 0.04), noise_params.seed + 601u);
    if tunnel < cave_threshold * 1.2 {
        return true;
    }
    // Tube networks (worm-like)
    let t_xz = perlin3d(vec3<f32>(wx * 0.025, 0.0, wz * 0.025), noise_params.seed + 602u);
    let t_xy = perlin3d(vec3<f32>(wx * 0.025, wy * 0.025, 0.0), noise_params.seed + 603u);
    if t_xz < cave_threshold * 0.85 && t_xy < cave_threshold * 0.85 {
        return true;
    }
    return false;
}

fn cave_fill_material_gpu(depth: f32, sea_level_depth: f32) -> u32 {
    if depth > 80.0 {
        return MAT_LAVA;
    } else if depth > sea_level_depth + 5.0 {
        return MAT_WATER;
    }
    return MAT_AIR;
}

fn is_crystal_deposit_gpu(depth: f32, wx: f32, wy: f32, wz: f32) -> bool {
    if depth < 40.0 {
        return false;
    }
    return perlin3d(vec3<f32>(wx * 0.10, wy * 0.10, wz * 0.10), noise_params.seed + 604u) < -0.30;
}

fn material_at_radius_gpu(
    r: f32,
    surface_r: f32,
    sea_level: f32,
    soil_depth: f32,
    cave_threshold: f32,
    wx: f32, wy: f32, wz: f32,
) -> u32 {
    if r > surface_r {
        if r < sea_level {
            return MAT_WATER;
        }
        return MAT_AIR;
    }

    let depth = surface_r - r;

    if depth < 1.0 {
        return select(MAT_DIRT, MAT_GRASS, surface_r >= sea_level);
    }

    if depth < soil_depth {
        return MAT_DIRT;
    }

    // Geological strata
    let strata_depth = depth - soil_depth;
    let ore = ore_material_gpu(strata_depth, wx, wy, wz);
    var base_mat: u32;
    if ore != 0xFFFFu {
        base_mat = ore;
    } else {
        base_mat = strata_material_gpu(strata_depth, wx, wy, wz);
    }

    // Cave carving
    let cave_max_depth = 200.0;
    if depth > 2.0 && depth < cave_max_depth {
        if is_cave_gpu(cave_threshold, wx, wy, wz) {
            let sea_level_depth = max(sea_level - r, 0.0);
            return cave_fill_material_gpu(depth, sea_level_depth);
        }
        // Crystal deposits on cave-adjacent walls
        if is_crystal_deposit_gpu(depth, wx, wy, wz) {
            return MAT_QUARTZ_CRYSTAL;
        }
    }

    return base_mat;
}

fn terrain_density_gpu(depth: f32) -> f32 {
    return clamp(0.5 + depth * 0.5, 0.0, 1.0);
}

// ─── Pass 1: Surface radius computation ──────────────────────────────────────
//
// One thread per (chunk, column).
// global_invocation_id.x = chunk_index * CHUNK_AREA + column_index
// column_index = lz * 32 + lx

@compute @workgroup_size(256)
fn surface_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let chunk_idx = idx / CHUNK_AREA;
    let col_idx = idx % CHUNK_AREA;

    if chunk_idx >= dispatch_params.chunk_count {
        return;
    }

    let chunk = chunks[chunk_idx];
    let lz = col_idx / CHUNK_SIZE;
    let lx = col_idx % CHUNK_SIZE;
    let half = f32(CHUNK_SIZE) / 2.0;

    // Compute local position at column center (y=0)
    let local = vec3<f32>(
        (f32(lx) + 0.5 - half) * chunk.tangent_scale.x,
        0.0,
        (f32(lz) + 0.5 - half) * chunk.tangent_scale.z,
    );
    let world = chunk.center.xyz + quat_rotate(chunk.rotation, local);

    // Convert to lat/lon
    let ll = lat_lon(world, dispatch_params.rotation_axis.xyz);

    // Sample terrain noise → surface radius
    let combined = sample_terrain(ll.y, ll.x); // sample_terrain(lon, lat)
    let sr = noise_params.mean_radius + combined * noise_params.height_scale;

    // Store surface radius
    let out_idx = chunk.chunk_index * CHUNK_AREA + col_idx;
    surface_radii[out_idx] = sr;

    // Atomic min/max for classification.
    // We use bitwise atomics on float bits. Since surface radii are positive
    // and we only need min/max, we use atomicMin/Max on the raw u32 bits.
    // IEEE 754 positive floats preserve ordering under integer comparison.
    let sr_bits = bitcast<u32>(sr);
    let info_base = chunk.chunk_index * 4u;
    atomicMin(&chunk_info[info_base + 1u], sr_bits);
    atomicMax(&chunk_info[info_base + 2u], sr_bits);
}

// ─── Pass 2: Voxel fill ─────────────────────────────────────────────────────
//
// One thread per voxel.
// global_invocation_id.x = chunk_index * CHUNK_VOLUME + voxel_index
// voxel_index = lz * CS * CS + ly * CS + lx

@compute @workgroup_size(256)
fn voxel_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let chunk_idx = idx / CHUNK_VOLUME;
    let voxel_idx = idx % CHUNK_VOLUME;

    if chunk_idx >= dispatch_params.chunk_count {
        return;
    }

    let chunk = chunks[chunk_idx];

    // Check classification — skip if AllAir or AllSolid.
    let info_base = chunk.chunk_index * 4u;
    let classification = atomicLoad(&chunk_info[info_base]);
    if classification != CLASS_MIXED {
        return;
    }

    let lz = voxel_idx / (CHUNK_SIZE * CHUNK_SIZE);
    let ly = (voxel_idx / CHUNK_SIZE) % CHUNK_SIZE;
    let lx = voxel_idx % CHUNK_SIZE;
    let half = f32(CHUNK_SIZE) / 2.0;

    // Compute world position
    let local = vec3<f32>(
        (f32(lx) + 0.5 - half) * chunk.tangent_scale.x,
        (f32(ly) + 0.5 - half) * chunk.tangent_scale.y,
        (f32(lz) + 0.5 - half) * chunk.tangent_scale.z,
    );
    let world = chunk.center.xyz + quat_rotate(chunk.rotation, local);
    let r = length(world);

    // Read cached surface radius for this column
    let col_idx = lz * CHUNK_SIZE + lx;
    let sr_idx = chunk.chunk_index * CHUNK_AREA + col_idx;
    let surface_r = surface_radii[sr_idx];

    // Material assignment
    let material = material_at_radius_gpu(
        r, surface_r,
        chunk.sea_level,
        chunk.soil_depth,
        chunk.cave_threshold,
        world.x, world.y, world.z,
    );

    // Density
    var density: f32;
    if material == MAT_WATER {
        density = terrain_density_gpu(chunk.sea_level - r);
    } else {
        density = terrain_density_gpu(surface_r - r);
    }

    // Write output
    let out_idx = chunk.chunk_index * CHUNK_VOLUME + voxel_idx;
    voxel_materials[out_idx] = material;
    voxel_densities[out_idx] = density;
}

// ─── Classification pass (runs after surface_pass) ───────────────────────────
//
// One thread per chunk — reads min/max surface and determines classification.

@compute @workgroup_size(64)
fn classify_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let chunk_idx = gid.x;
    if chunk_idx >= dispatch_params.chunk_count {
        return;
    }

    let chunk = chunks[chunk_idx];
    let info_base = chunk.chunk_index * 4u;

    let min_bits = atomicLoad(&chunk_info[info_base + 1u]);
    let max_bits = atomicLoad(&chunk_info[info_base + 2u]);
    let min_surface = bitcast<f32>(min_bits);
    let max_surface = bitcast<f32>(max_bits);

    let sea = chunk.sea_level;
    let half_diag = chunk.half_diag;
    let base_r = chunk.base_r;
    let top_r = chunk.top_r;

    // Entirely above terrain AND above sea → all air
    if base_r - half_diag > max_surface && base_r - half_diag > sea {
        atomicStore(&chunk_info[info_base], CLASS_ALL_AIR);
        return;
    }

    // Entirely below terrain (and below cave floor)
    let cave_floor = min_surface - 200.0;
    if top_r + half_diag < min_surface && top_r + half_diag < cave_floor {
        atomicStore(&chunk_info[info_base], CLASS_ALL_SOLID);
        atomicStore(&chunk_info[info_base + 3u], MAT_STONE);
        return;
    }

    atomicStore(&chunk_info[info_base], CLASS_MIXED);
}
