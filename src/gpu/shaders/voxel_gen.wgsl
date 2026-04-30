// GPU-accelerated voxel generation for V2 cubed-sphere chunks.
//
// Three-pass pipeline:
//   Pass 1 (surface_pass):  1024 threads/chunk compute surface_radius per column
//                           + parallel reduction for min/max → chunk classification.
//   Pass 2 (classify_pass): 1 thread/chunk classifies as AllAir/AllSolid/Mixed.
//   Pass 3 (voxel_pass):    32768 threads/chunk fill material + density for mixed chunks.
//
// Uses permutation-table-based Perlin noise matching the CPU `noise` crate exactly.

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

// Permutation table slot indices (must match Rust constants).
const PERM_FBM_START: u32 = 0u;
const PERM_RIDGED_START: u32 = 8u;
const PERM_SELECTOR: u32 = 16u;
const PERM_WARP_X: u32 = 17u;
const PERM_WARP_Z: u32 = 18u;
const PERM_MICRO: u32 = 19u;
const PERM_CONTINENT: u32 = 20u;
const PERM_OCEAN_FLOOR: u32 = 21u;
const PERM_STRATA: u32 = 22u;
const PERM_ORE_COAL: u32 = 23u;
const PERM_ORE_COPPER: u32 = 24u;
const PERM_ORE_IRON: u32 = 25u;
const PERM_ORE_GOLD: u32 = 26u;
const PERM_CAVE_CAVERN: u32 = 27u;
const PERM_CAVE_TUNNEL: u32 = 28u;
const PERM_CAVE_TUBE_XZ: u32 = 29u;
const PERM_CAVE_TUBE_XY: u32 = 30u;
const PERM_CRYSTAL: u32 = 31u;
const PERM_TABLE_SIZE: u32 = 256u;

// Permutation table slots for TerrainNoise (planetary IDW noise added at exact column position).
// Must match PERM_TERRAIN_FBM_START / PERM_TERRAIN_RIDGED_START in voxel_compute.rs.
const PERM_TERRAIN_FBM_START: u32 = 32u;    // 6 octaves: slots 32-37
const PERM_TERRAIN_RIDGED_START: u32 = 38u; // 4 octaves: slots 38-41

// Perlin scale factors matching the noise crate.
const PERLIN_2D_SCALE: f32 = 1.4142135; // 2.0 / sqrt(2)
const PERLIN_3D_SCALE: f32 = 1.1547005; // 2.0 / sqrt(3)

// ─── Uniforms ────────────────────────────────────────────────────────────────

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
    /// 1 = sample the pre-baked planetary heightmap; 0 = use FBM Perlin noise.
    use_heightmap: u32,
    mean_radius: f32,
    height_scale: f32,
}

struct ChunkDesc {
    center: vec4<f32>,
    rotation: vec4<f32>,
    tangent_scale: vec4<f32>,
    // Radial bounds and sea level are stored as OFFSETS from `mean_radius`
    // rather than absolute radii. At Earth scale (r ≈ 6.37e6 m), f32 only
    // has ~0.5 m precision on absolute values, so any comparison like
    // `depth = surface_r - r` lost essentially all meaningful resolution
    // and randomized air/solid classification at the surface. Offsets are
    // bounded by `height_scale` (≈ 8.8 km for Earth) and so fit cleanly
    // into f32 with sub-mm precision.
    base_r_offset: f32,
    top_r_offset: f32,
    sea_level_offset: f32,
    lod_scale: f32,
    soil_depth: f32,
    cave_threshold: f32,
    half_diag: f32,
    chunk_index: u32,
    // `chunk_r_offset = layer * CHUNK_SIZE * lod_scale` — the chunk center's
    // radial distance from `mean_radius`. Exact when computed on the host
    // in f64 and cast to f32 (small value, no precision loss).
    chunk_r_offset: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

struct DispatchParams {
    chunk_count: u32,
    rotation_axis: vec4<f32>,
}

// ─── Bind groups ─────────────────────────────────────────────────────────────

@group(0) @binding(0) var<uniform> noise_params: NoiseParams;
@group(0) @binding(1) var<uniform> dispatch_params: DispatchParams;

@group(1) @binding(0) var<storage, read> chunks: array<ChunkDesc>;

@group(2) @binding(0) var<storage, read_write> surface_radii: array<f32>;
@group(2) @binding(1) var<storage, read_write> chunk_info: array<atomic<u32>>;

@group(3) @binding(0) var<storage, read_write> voxel_materials: array<u32>;
@group(3) @binding(1) var<storage, read_write> voxel_densities: array<f32>;

// Permutation tables: NUM_PERM_TABLES × 256 u32 values.
@group(1) @binding(1) var<storage, read> perm_tables: array<u32>;

// Bind group 1, binding 2: pre-baked equirectangular elevation map (2048×1024 f32).
// Each element is the IDW tectonic elevation offset from mean_radius in metres.
// TerrainNoise (FBM + ridged) is NOT baked here — it is computed at the exact
// column position below to avoid aliasing from 100 m/pixel bilinear interpolation.
// Sampled only when noise_params.use_heightmap == 1.
@group(1) @binding(2) var<storage, read> heightmap_data: array<f32>;

// Bind group 1, binding 3: biome noise roughness in [0, 1] (2048×1024 f32).
// Bilinearly interpolated — roughness varies slowly compared to TerrainNoise.
@group(1) @binding(3) var<storage, read> roughness_data: array<f32>;

// Bind group 1, binding 4: ocean biome flag (2048×1024 f32, 1.0 = ocean, 0.0 = land).
// Sampled with nearest-neighbour logic to prevent coastline biome bleed.
@group(1) @binding(4) var<storage, read> ocean_data: array<f32>;

// ─── Permutation-table-based Perlin noise ────────────────────────────────────
//
// Replicates the `noise` crate's Perlin implementation exactly, including:
// - PermutationTable hash chain
// - noise_floor() quirk (0.0 → -1, not 0)
// - True 2D with 4 gradients and scale 2/√2
// - 3D with 16 Ken Perlin gradients and scale 2/√3
// - Quintic interpolation

/// Read from permutation table `table_idx` at position `pos`.
fn perm(table_idx: u32, pos: u32) -> u32 {
    return perm_tables[table_idx * PERM_TABLE_SIZE + (pos & 255u)];
}

/// noise crate's floor_to_isize: NOT standard floor.
/// 0.0 → -1, -1.0 → -2, 0.5 → 0, -0.5 → -1, 1.0 → 1
fn noise_floor(x: f32) -> i32 {
    return select(i32(x), i32(x) - 1, x <= 0.0);
}

fn fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

/// 2D gradient: 4 directions {(1,1),(-1,1),(1,-1),(-1,-1)}
fn grad2d(hash: u32, x: f32, y: f32) -> f32 {
    let h = hash & 3u;
    let gx = select(-1.0, 1.0, (h & 1u) == 0u);
    let gy = select(-1.0, 1.0, (h & 2u) == 0u);
    return gx * x + gy * y;
}

/// 3D gradient matching noise crate's exact table (NOT the classic Perlin
/// shortcut which differs at cases 13 and 14).
fn grad3d(hash: u32, x: f32, y: f32, z: f32) -> f32 {
    let h = hash & 15u;
    // Cases 12,13 alias to 0,1 (x±y); 14,15 alias to 9,11 (-y±z).
    switch h {
        case 0u, 12u  { return  x + y; }
        case 1u, 13u  { return -x + y; }
        case 2u       { return  x - y; }
        case 3u       { return -x - y; }
        case 4u       { return  x + z; }
        case 5u       { return -x + z; }
        case 6u       { return  x - z; }
        case 7u       { return -x - z; }
        case 8u       { return  y + z; }
        case 9u, 14u  { return -y + z; }
        case 10u      { return  y - z; }
        case 11u, 15u { return -y - z; }
        default       { return 0.0; }
    }
}

/// 2D Perlin noise using permutation table at `table_idx`.
fn perlin2d(x: f32, z: f32, table_idx: u32) -> f32 {
    let ix = noise_floor(x);
    let iz = noise_floor(z);
    let fx = x - f32(ix);
    let fz = z - f32(iz);

    let ux = u32(ix) & 255u;
    let uz = u32(iz) & 255u;

    // Hash chain: perm[perm[ix] ^ iz]
    let h00 = perm(table_idx, perm(table_idx, ux) ^ uz);
    let h10 = perm(table_idx, perm(table_idx, ux + 1u) ^ uz);
    let h01 = perm(table_idx, perm(table_idx, ux) ^ (uz + 1u));
    let h11 = perm(table_idx, perm(table_idx, ux + 1u) ^ (uz + 1u));

    let g00 = grad2d(h00, fx, fz);
    let g10 = grad2d(h10, fx - 1.0, fz);
    let g01 = grad2d(h01, fx, fz - 1.0);
    let g11 = grad2d(h11, fx - 1.0, fz - 1.0);

    let cu = fade(fx);
    let cv = fade(fz);

    // Interpolate: first along z, then along x (matching noise crate order)
    let result = mix(mix(g00, g01, cv), mix(g10, g11, cv), cu) * PERLIN_2D_SCALE;
    return clamp(result, -1.0, 1.0);
}

/// 3D Perlin noise using permutation table at `table_idx`.
fn perlin3d(p: vec3<f32>, table_idx: u32) -> f32 {
    let ix = noise_floor(p.x);
    let iy = noise_floor(p.y);
    let iz = noise_floor(p.z);
    let fx = p.x - f32(ix);
    let fy = p.y - f32(iy);
    let fz = p.z - f32(iz);

    let ux = u32(ix) & 255u;
    let uy = u32(iy) & 255u;
    let uz = u32(iz) & 255u;

    // Hash chain: perm[perm[perm[ix] ^ iy] ^ iz]
    let h000 = perm(table_idx, perm(table_idx, perm(table_idx, ux) ^ uy) ^ uz);
    let h100 = perm(table_idx, perm(table_idx, perm(table_idx, ux + 1u) ^ uy) ^ uz);
    let h010 = perm(table_idx, perm(table_idx, perm(table_idx, ux) ^ (uy + 1u)) ^ uz);
    let h110 = perm(table_idx, perm(table_idx, perm(table_idx, ux + 1u) ^ (uy + 1u)) ^ uz);
    let h001 = perm(table_idx, perm(table_idx, perm(table_idx, ux) ^ uy) ^ (uz + 1u));
    let h101 = perm(table_idx, perm(table_idx, perm(table_idx, ux + 1u) ^ uy) ^ (uz + 1u));
    let h011 = perm(table_idx, perm(table_idx, perm(table_idx, ux) ^ (uy + 1u)) ^ (uz + 1u));
    let h111 = perm(table_idx, perm(table_idx, perm(table_idx, ux + 1u) ^ (uy + 1u)) ^ (uz + 1u));

    let g000 = grad3d(h000, fx, fy, fz);
    let g100 = grad3d(h100, fx - 1.0, fy, fz);
    let g010 = grad3d(h010, fx, fy - 1.0, fz);
    let g110 = grad3d(h110, fx - 1.0, fy - 1.0, fz);
    let g001 = grad3d(h001, fx, fy, fz - 1.0);
    let g101 = grad3d(h101, fx - 1.0, fy, fz - 1.0);
    let g011 = grad3d(h011, fx, fy - 1.0, fz - 1.0);
    let g111 = grad3d(h111, fx - 1.0, fy - 1.0, fz - 1.0);

    let cu = fade(fx);
    let cv = fade(fy);
    let cw = fade(fz);

    // Interpolate: z, then y, then x (matching noise crate order)
    let x0 = mix(mix(g000, g001, cw), mix(g010, g011, cw), cv);
    let x1 = mix(mix(g100, g101, cw), mix(g110, g111, cw), cv);
    let result = mix(x0, x1, cu) * PERLIN_3D_SCALE;
    return clamp(result, -1.0, 1.0);
}

// ─── Noise pipeline (mirrors NoiseStack::sample()) ───────────────────────────

fn fbm(x: f32, z: f32) -> f32 {
    var value = 0.0;
    var amplitude = 1.0;
    var frequency = noise_params.fbm_base_freq;
    var normalization = 0.0;

    for (var i = 0u; i < noise_params.fbm_octaves; i++) {
        value += amplitude * perlin2d(x * frequency, z * frequency, PERM_FBM_START + i);
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

    for (var i = 0u; i < noise_params.ridged_octaves; i++) {
        let signal_raw = perlin2d(x * frequency, z * frequency, PERM_RIDGED_START + i);
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

/// Sample terrain height excluding micro-detail (matches CPU sample()).
fn sample_terrain(x: f32, z: f32) -> f32 {
    // Domain warp
    let wx = perlin2d(x * noise_params.warp_freq, z * noise_params.warp_freq, PERM_WARP_X) * noise_params.warp_strength;
    let wz = perlin2d(x * noise_params.warp_freq, z * noise_params.warp_freq, PERM_WARP_Z) * noise_params.warp_strength;
    let sx = x + wx;
    let sz = z + wz;

    // Selector
    let sel = perlin2d(sx * noise_params.selector_freq, sz * noise_params.selector_freq, PERM_SELECTOR);

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

    // NO micro-detail here — matches CPU sample() which excludes it.

    if noise_params.continent_enabled == 0u {
        return land;
    }

    // Continent blend uses ORIGINAL (un-warped) coordinates, matching CPU.
    let cv = perlin2d(x * noise_params.continent_freq, z * noise_params.continent_freq, PERM_CONTINENT);
    let threshold = noise_params.continent_threshold;
    let shelf = noise_params.shelf_blend_width;
    let ocean_edge = threshold - shelf;

    if cv >= threshold {
        return land;
    }

    let ocean = -noise_params.ocean_floor_depth + perlin2d(x * 0.01, z * 0.01, PERM_OCEAN_FLOOR) * noise_params.ocean_floor_amplitude;

    if cv <= ocean_edge {
        return ocean;
    }

    let blend = smoothstep_custom(ocean_edge, threshold, cv);
    return mix(ocean, land, blend);
}

// ─── Coordinate helpers ──────────────────────────────────────────────────────

fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

fn lat_lon(pos: vec3<f32>, axis: vec3<f32>) -> vec2<f32> {
    // Must match CPU `planet::detail::pos_to_lat_lon`: for a unit direction,
    // lat = asin(y), lon = atan2(x, z) (Y-up). Earlier code used a different
    // east/north basis that did NOT invert `lat_lon_to_pos`, and so the GPU
    // classified chunks (AllAir/AllSolid/Mixed) at the wrong surface column,
    // leaving holes in the terrain.
    let len = length(pos);
    if len < 1e-6 {
        return vec2<f32>(0.0, 0.0);
    }
    let dir = pos / len;
    let lat = asin(clamp(dir.y, -1.0, 1.0));
    let lon = atan2(dir.x, dir.z);
    return vec2<f32>(lat, lon);
}

// ─── Planetary heightmap sampling ────────────────────────────────────────────

// Dimensions must match constants in src/planet/gpu_heightmap.rs.
const HEIGHTMAP_W: u32 = 2048u;
const HEIGHTMAP_H: u32 = 1024u;
const PI: f32 = 3.14159265358979323846;

/// Sample the baked equirectangular elevation map at `(lat, lon)` with
/// bilinear interpolation.
///
/// `lat` ∈ [−π/2, π/2],  `lon` ∈ [−π, π].
/// Returns the elevation offset from `mean_radius` in metres.
fn sample_heightmap(lat: f32, lon: f32) -> f32 {
    // Map lon → u ∈ [0, 1], wrapping at the antimeridian.
    let u = fract(lon / (2.0 * PI) + 0.5);
    // Map lat → v ∈ [0 (north), 1 (south)].
    let v = 0.5 - lat / PI;

    // Sub-pixel position (texel centres at +0.5).
    let px = u * f32(HEIGHTMAP_W) - 0.5;
    let py = v * f32(HEIGHTMAP_H) - 0.5;

    let ix = i32(floor(px));
    let iy = i32(floor(py));
    let tx = px - floor(px);
    let ty = py - floor(py);

    // Wrap x, clamp y.
    let x0 = u32((ix + i32(HEIGHTMAP_W)) % i32(HEIGHTMAP_W));
    let x1 = u32((ix + 1 + i32(HEIGHTMAP_W)) % i32(HEIGHTMAP_W));
    let y0 = u32(clamp(iy,     0, i32(HEIGHTMAP_H) - 1));
    let y1 = u32(clamp(iy + 1, 0, i32(HEIGHTMAP_H) - 1));

    let v00 = heightmap_data[y0 * HEIGHTMAP_W + x0];
    let v10 = heightmap_data[y0 * HEIGHTMAP_W + x1];
    let v01 = heightmap_data[y1 * HEIGHTMAP_W + x0];
    let v11 = heightmap_data[y1 * HEIGHTMAP_W + x1];

    return mix(mix(v00, v10, tx), mix(v01, v11, tx), ty);
}

/// Sample the roughness buffer at `(lat, lon)` with bilinear interpolation.
fn sample_roughness(lat: f32, lon: f32) -> f32 {
    let u = fract(lon / (2.0 * PI) + 0.5);
    let v = 0.5 - lat / PI;
    let px = u * f32(HEIGHTMAP_W) - 0.5;
    let py = v * f32(HEIGHTMAP_H) - 0.5;
    let ix = i32(floor(px));
    let iy = i32(floor(py));
    let tx = px - floor(px);
    let ty = py - floor(py);
    let x0 = u32((ix + i32(HEIGHTMAP_W)) % i32(HEIGHTMAP_W));
    let x1 = u32((ix + 1 + i32(HEIGHTMAP_W)) % i32(HEIGHTMAP_W));
    let y0 = u32(clamp(iy,     0, i32(HEIGHTMAP_H) - 1));
    let y1 = u32(clamp(iy + 1, 0, i32(HEIGHTMAP_H) - 1));
    let v00 = roughness_data[y0 * HEIGHTMAP_W + x0];
    let v10 = roughness_data[y0 * HEIGHTMAP_W + x1];
    let v01 = roughness_data[y1 * HEIGHTMAP_W + x0];
    let v11 = roughness_data[y1 * HEIGHTMAP_W + x1];
    return mix(mix(v00, v10, tx), mix(v01, v11, tx), ty);
}

/// Sample the ocean flag at `(lat, lon)` with nearest-neighbour lookup.
/// Returns 1u for ocean/deep-ocean biome, 0u for land.
/// Nearest-neighbour prevents biome bleed at coastlines.
fn sample_ocean(lat: f32, lon: f32) -> u32 {
    let u = fract(lon / (2.0 * PI) + 0.5);
    let v = 0.5 - lat / PI;
    let ix = i32(u * f32(HEIGHTMAP_W));
    let iy = i32(v * f32(HEIGHTMAP_H));
    let x = u32((ix + i32(HEIGHTMAP_W)) % i32(HEIGHTMAP_W));
    let y = u32(clamp(iy, 0, i32(HEIGHTMAP_H) - 1));
    return u32(ocean_data[y * HEIGHTMAP_W + x]);
}

/// FBM matching CPU `TerrainNoise::fbm` (Fbm<Perlin>, frequency=50, octaves=6,
/// lacunarity=2.0, persistence=0.5). Uses perm tables PERM_TERRAIN_FBM_START+i.
fn terrain_fbm3d(p: vec3<f32>) -> f32 {
    var point = p * 50.0f;
    var result = 0.0f;
    var att = 0.5f; // starts at persistence, not 1.0
    for (var i = 0u; i < 6u; i++) {
        result += perlin3d(point, PERM_TERRAIN_FBM_START + i) * att;
        att *= 0.5f;
        point *= 2.0f;
    }
    // scale_factor = 1.0 / (sum of 0.5^x for x in 1..=6) = 1.0 / 0.984375
    return result * 1.0158730f;
}

/// RidgedMulti matching CPU `TerrainNoise::ridged` (RidgedMulti<Perlin>,
/// frequency=35, octaves=4, lacunarity=2.2, attenuation=2.0, persistence=1.0).
/// Uses perm tables PERM_TERRAIN_RIDGED_START+i.
fn terrain_ridged3d(p: vec3<f32>) -> f32 {
    var point = p * 35.0f;
    var result = 0.0f;
    var weight = 1.0f;
    // persistence=1.0 → amplitude stays 1.0 every octave
    for (var i = 0u; i < 4u; i++) {
        var signal = perlin3d(point, PERM_TERRAIN_RIDGED_START + i);
        signal = abs(signal);
        signal = 1.0f - signal;
        signal *= signal;
        signal *= weight;
        weight = clamp(signal * 0.5f, 0.0f, 1.0f); // / attenuation(2.0) = * 0.5
        // signal *= amplitude (1.0^i = 1.0, no change needed)
        result += signal;
        point *= 2.2f;
    }
    // scale_factor = 2.0 / 1.6416015625 = 1.218322427
    // output = result * scale_factor - 1.0  (shift to [-1, 1])
    return result * 1.2183224f - 1.0f;
}

// Order-preserving float → u32 mapping for signed floats so that
// `atomicMin` / `atomicMax` on the encoded value match float ordering.
// Needed because surface offsets can be negative (e.g. ocean floor below
// `mean_radius`), unlike the previous absolute-radius encoding.
fn sortable_encode(f: f32) -> u32 {
    let bits = bitcast<u32>(f);
    if (bits & 0x80000000u) != 0u {
        // Negative: flip all bits.
        return ~bits;
    }
    // Non-negative: flip the sign bit so +0 maps above all negatives.
    return bits | 0x80000000u;
}

fn sortable_decode(u: u32) -> f32 {
    if (u & 0x80000000u) != 0u {
        // Was non-negative: clear the sign bit.
        return bitcast<f32>(u & 0x7FFFFFFFu);
    }
    // Was negative: undo the full flip.
    return bitcast<f32>(~u);
}

// ─── Material assignment (mirrors material_at_radius) ────────────────────────

fn strata_material_gpu(depth: f32, wx: f32, wy: f32, wz: f32) -> u32 {
    let n = perlin3d(vec3<f32>(wx * 0.02, wy * 0.02, wz * 0.02), PERM_STRATA);

    if depth < 20.0 {
        return select(MAT_LIMESTONE, MAT_SANDSTONE, n > 0.0);
    } else if depth < 60.0 {
        return MAT_STONE;
    } else {
        return select(MAT_BASALT, MAT_GRANITE, n > 0.0);
    }
}

fn ore_material_gpu(depth: f32, wx: f32, wy: f32, wz: f32) -> u32 {
    if depth >= 5.0 && depth <= 30.0 {
        if perlin3d(vec3<f32>(wx * 0.08, wy * 0.08, wz * 0.08), PERM_ORE_COAL) < -0.15 {
            return MAT_COAL;
        }
    }
    if depth >= 15.0 && depth <= 50.0 {
        if perlin3d(vec3<f32>(wx * 0.06, wy * 0.06, wz * 0.06), PERM_ORE_COPPER) < -0.20 {
            return MAT_COPPER_ORE;
        }
    }
    if depth >= 30.0 && depth <= 80.0 {
        if perlin3d(vec3<f32>(wx * 0.05, wy * 0.05, wz * 0.05), PERM_ORE_IRON) < -0.25 {
            return MAT_IRON;
        }
    }
    if depth >= 50.0 {
        if perlin3d(vec3<f32>(wx * 0.04, wy * 0.04, wz * 0.04), PERM_ORE_GOLD) < -0.35 {
            return MAT_GOLD_ORE;
        }
    }
    return 0xFFFFu;
}

fn is_cave_gpu(cave_threshold: f32, wx: f32, wy: f32, wz: f32) -> bool {
    let cavern = perlin3d(vec3<f32>(wx * 0.01, wy * 0.01, wz * 0.01), PERM_CAVE_CAVERN);
    if cavern < cave_threshold * 0.5 {
        return true;
    }
    let tunnel = perlin3d(vec3<f32>(wx * 0.04, wy * 0.04, wz * 0.04), PERM_CAVE_TUNNEL);
    if tunnel < cave_threshold * 1.2 {
        return true;
    }
    // Tube networks use 2D Perlin (XZ plane and XY plane respectively),
    // matching the CPU's `perlin.get([x, z])` / `perlin.get([x, y])` calls.
    // Previously these incorrectly used perlin3d with a zero third coordinate,
    // which produces completely different values (different hash chain depth and
    // different gradient table) — fix: use perlin2d.
    let t_xz = perlin2d(wx * 0.025, wz * 0.025, PERM_CAVE_TUBE_XZ);
    let t_xy = perlin2d(wx * 0.025, wy * 0.025, PERM_CAVE_TUBE_XY);
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
    return perlin3d(vec3<f32>(wx * 0.10, wy * 0.10, wz * 0.10), PERM_CRYSTAL) < -0.30;
}

fn material_at_radius_gpu(
    r_offset: f32,
    surface_r_offset: f32,
    sea_level_offset: f32,
    soil_depth: f32,
    cave_threshold: f32,
    wx: f32, wy: f32, wz: f32,
) -> u32 {
    if r_offset > surface_r_offset {
        if r_offset < sea_level_offset {
            return MAT_WATER;
        }
        return MAT_AIR;
    }

    let depth = surface_r_offset - r_offset;

    if depth < 1.0 {
        return select(MAT_DIRT, MAT_GRASS, surface_r_offset >= sea_level_offset);
    }

    if depth < soil_depth {
        return MAT_DIRT;
    }

    let strata_depth = depth - soil_depth;
    let ore = ore_material_gpu(strata_depth, wx, wy, wz);
    var base_mat: u32;
    if ore != 0xFFFFu {
        base_mat = ore;
    } else {
        base_mat = strata_material_gpu(strata_depth, wx, wy, wz);
    }

    let cave_max_depth = 200.0;
    if depth > 2.0 && depth < cave_max_depth {
        if is_cave_gpu(cave_threshold, wx, wy, wz) {
            let sea_level_depth = max(sea_level_offset - r_offset, 0.0);
            return cave_fill_material_gpu(depth, sea_level_depth);
        }
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

    let local = vec3<f32>(
        (f32(lx) + 0.5 - half) * chunk.tangent_scale.x,
        0.0,
        (f32(lz) + 0.5 - half) * chunk.tangent_scale.z,
    );
    let world = chunk.center.xyz + quat_rotate(chunk.rotation, local);

    let ll = lat_lon(world, dispatch_params.rotation_axis.xyz);

    // Compute surface radius offset from mean_radius in metres.
    // When the planetary heightmap is available, sample it directly — this
    // reproduces the CPU PlanetaryTerrainSampler path (IDW tectonic elevation +
    // detail noise + ocean-biome clamp) at f32 precision without running FBM.
    // Fall back to the FBM Perlin path when no heightmap has been uploaded yet.
    var sr_offset: f32;
    if noise_params.use_heightmap == 1u {
        // ll.x = lat, ll.y = lon (see lat_lon() return convention).
        let idw_elevation = sample_heightmap(ll.x, ll.y);
        let roughness     = sample_roughness(ll.x, ll.y);
        let is_ocean      = sample_ocean(ll.x, ll.y);

        // Compute matching TerrainNoise at the exact column position (unit sphere).
        // This avoids the ~100 m/pixel aliasing from baking noise into the heightmap.
        let unit = normalize(world);
        let fbm_val   = terrain_fbm3d(unit);
        let ridge_val = terrain_ridged3d(unit);
        let mix_r     = roughness * 0.6f;
        let combined  = fbm_val * (1.0f - mix_r) + ridge_val * mix_r;
        let noise_m   = combined * roughness * 2000.0f;

        // Ocean-biome clamp: matches CPU sample_detailed_elevation() clamp logic.
        // If IDW elevation < 0 or nearest cell is ocean biome, elevation ≤ -2 m.
        var elevation = idw_elevation + noise_m;
        if is_ocean == 1u || idw_elevation < 0.0f {
            elevation = min(elevation, -2.0f);
        }
        sr_offset = elevation;
    } else {
        // sample_terrain excludes micro-detail, matching CPU sample_surface_radius()
        let combined = sample_terrain(ll.y, ll.x);
        // Store surface offset from mean_radius, not absolute radius. Fits in f32
        // with sub-mm precision (|offset| ≤ height_scale ≈ 8.8 km for Earth) and
        // avoids catastrophic cancellation when compared against voxel r_offset
        // in `voxel_pass`.
        sr_offset = combined * noise_params.height_scale;
    }

    let out_idx = chunk.chunk_index * CHUNK_AREA + col_idx;
    surface_radii[out_idx] = sr_offset;

    let enc = sortable_encode(sr_offset);
    let info_base = chunk.chunk_index * 4u;
    atomicMin(&chunk_info[info_base + 1u], enc);
    atomicMax(&chunk_info[info_base + 2u], enc);
}

// ─── Pass 2: Voxel fill ─────────────────────────────────────────────────────

@compute @workgroup_size(256)
fn voxel_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let chunk_idx = idx / CHUNK_VOLUME;
    let voxel_idx = idx % CHUNK_VOLUME;

    if chunk_idx >= dispatch_params.chunk_count {
        return;
    }

    let chunk = chunks[chunk_idx];

    let info_base = chunk.chunk_index * 4u;
    let classification = atomicLoad(&chunk_info[info_base]);
    if classification != CLASS_MIXED {
        return;
    }

    let lz = voxel_idx / (CHUNK_SIZE * CHUNK_SIZE);
    let ly = (voxel_idx / CHUNK_SIZE) % CHUNK_SIZE;
    let lx = voxel_idx % CHUNK_SIZE;
    let half = f32(CHUNK_SIZE) / 2.0;

    let local = vec3<f32>(
        (f32(lx) + 0.5 - half) * chunk.tangent_scale.x,
        (f32(ly) + 0.5 - half) * chunk.tangent_scale.y,
        (f32(lz) + 0.5 - half) * chunk.tangent_scale.z,
    );
    // World position is only needed for the 3-D noise coordinates used by
    // strata / ore / cave / crystal sampling. We intentionally do NOT derive
    // the voxel's radius from `length(world)` — at Earth scale that loses
    // ~0.5 m of precision which is catastrophic for the surface test.
    let world = chunk.center.xyz + quat_rotate(chunk.rotation, local);

    // Radial offset of the voxel relative to `mean_radius`, computed in the
    // chunk's local tangent frame. Since `rotation` is orthonormal and maps
    // local +Y onto the radial `up` direction, the dot product of
    // `quat_rotate(rotation, local)` with that radial direction is just
    // `local.y`. A small quadratic correction comes from the curvature of
    // the sphere over the chunk's tangent footprint.
    let center_r = noise_params.mean_radius + chunk.chunk_r_offset;
    let tangent_sq = local.x * local.x + local.z * local.z;
    let r_offset = chunk.chunk_r_offset + local.y + tangent_sq / (2.0 * center_r);

    let col_idx = lz * CHUNK_SIZE + lx;
    let sr_idx = chunk.chunk_index * CHUNK_AREA + col_idx;
    let surface_r_offset = surface_radii[sr_idx];

    let material = material_at_radius_gpu(
        r_offset, surface_r_offset,
        chunk.sea_level_offset,
        chunk.soil_depth,
        chunk.cave_threshold,
        world.x, world.y, world.z,
    );

    var density: f32;
    // Scale depth by 1/lod_scale so the gradient spans ±1 voxel at every
    // LOD level, giving surface nets enough sub-voxel information to produce
    // smooth diagonal surfaces instead of Minecraft-style blocks.
    if material == MAT_WATER {
        density = terrain_density_gpu((chunk.sea_level_offset - r_offset) / chunk.lod_scale);
    } else {
        density = terrain_density_gpu((surface_r_offset - r_offset) / chunk.lod_scale);
    }

    let out_idx = chunk.chunk_index * CHUNK_VOLUME + voxel_idx;
    voxel_materials[out_idx] = material;
    voxel_densities[out_idx] = density;
}

// ─── Classification pass ─────────────────────────────────────────────────────

@compute @workgroup_size(64)
fn classify_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let chunk_idx = gid.x;
    if chunk_idx >= dispatch_params.chunk_count {
        return;
    }

    let chunk = chunks[chunk_idx];
    let info_base = chunk.chunk_index * 4u;

    let min_enc = atomicLoad(&chunk_info[info_base + 1u]);
    let max_enc = atomicLoad(&chunk_info[info_base + 2u]);
    let min_surface_offset = sortable_decode(min_enc);
    let max_surface_offset = sortable_decode(max_enc);

    let sea_offset = chunk.sea_level_offset;
    let half_diag = chunk.half_diag;
    let base_offset = chunk.base_r_offset;
    let top_offset = chunk.top_r_offset;

    if base_offset - half_diag > max_surface_offset && base_offset - half_diag > sea_offset {
        atomicStore(&chunk_info[info_base], CLASS_ALL_AIR);
        return;
    }

    let cave_floor_offset = min_surface_offset - 200.0;
    if top_offset + half_diag < min_surface_offset && top_offset + half_diag < cave_floor_offset {
        atomicStore(&chunk_info[info_base], CLASS_ALL_SOLID);
        atomicStore(&chunk_info[info_base + 3u], MAT_STONE);
        return;
    }

    atomicStore(&chunk_info[info_base], CLASS_MIXED);
}
