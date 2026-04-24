//! GPU compute pipeline for batched voxel generation.
//!
//! Offloads `generate_voxels_core()` to the GPU via `voxel_gen.wgsl`.
//! Three-pass pipeline per batch:
//!   1. `surface_pass`: 1024 threads/chunk compute surface radius per column
//!   2. `classify_pass`: 1 thread/chunk determines AllAir/AllSolid/Mixed
//!   3. `voxel_pass`: 32768 threads/chunk fill material + density (mixed only)
//!
//! Falls back gracefully when no GPU is available.

use std::sync::Arc;

use wgpu;

use crate::gpu::context::GpuContext;
use crate::world::chunk::{CHUNK_SIZE, CHUNK_VOLUME};
use crate::world::noise::NoiseConfig;
use crate::world::v2::cubed_sphere::CubeSphereCoord;
use crate::world::v2::terrain_gen::{CachedVoxels, V2TerrainData};
use crate::world::voxel::{MaterialId, Voxel};

// ─── Constants ─────────────────────────────────────────────────────────────

/// Maximum chunks per GPU dispatch.
pub const MAX_CHUNKS_PER_BATCH: usize = 32;

const CHUNK_AREA: usize = CHUNK_SIZE * CHUNK_SIZE; // 1024
const WORKGROUP_SIZE: u32 = 256;

// Classification flags (must match WGSL constants).
const CLASS_MIXED: u32 = 0;
const CLASS_ALL_AIR: u32 = 1;
const CLASS_ALL_SOLID: u32 = 2;

// Permutation table layout: fixed-index slots for each noise function.
// Each table is 256 u32 values. Total: NUM_PERM_TABLES × 256 = 8192 u32s.
const MAX_OCTAVES: usize = 8;
const PERM_TABLE_SIZE: usize = 256;

// Table index constants (must match WGSL constants).
const PERM_FBM_START: usize = 0; // 0..8
const PERM_RIDGED_START: usize = 8; // 8..16
const PERM_SELECTOR: usize = 16;
const PERM_WARP_X: usize = 17;
const PERM_WARP_Z: usize = 18;
const PERM_MICRO: usize = 19;
const PERM_CONTINENT: usize = 20;
const PERM_OCEAN_FLOOR: usize = 21;
const PERM_STRATA: usize = 22;
const PERM_ORE_COAL: usize = 23;
const PERM_ORE_COPPER: usize = 24;
const PERM_ORE_IRON: usize = 25;
const PERM_ORE_GOLD: usize = 26;
const PERM_CAVE_CAVERN: usize = 27;
const PERM_CAVE_TUNNEL: usize = 28;
const PERM_CAVE_TUBE_XZ: usize = 29;
const PERM_CAVE_TUBE_XY: usize = 30;
const PERM_CRYSTAL: usize = 31;
const NUM_PERM_TABLES: usize = 32;

// ─── Permutation table generation (matches noise crate exactly) ────────────

/// Replicate `noise::PermutationTable::new(seed)` exactly.
///
/// Uses the same XorShift RNG seeding and Fisher-Yates shuffle as the `noise`
/// crate (rand_xorshift 0.3 + rand 0.8).
fn generate_perm_table(seed: u32) -> [u8; 256] {
    // Step 1: Build XorShiftRng seed bytes (from noise crate's PermutationTable::new).
    let mut seed_bytes = [0u8; 16];
    seed_bytes[0] = 1;
    for i in 1..4 {
        seed_bytes[i * 4] = seed as u8;
        seed_bytes[i * 4 + 1] = (seed >> 8) as u8;
        seed_bytes[i * 4 + 2] = (seed >> 16) as u8;
        seed_bytes[i * 4 + 3] = (seed >> 24) as u8;
    }

    // Step 2: Parse as 4 little-endian u32 values → XorShiftRng state.
    let mut state = [
        u32::from_le_bytes([seed_bytes[0], seed_bytes[1], seed_bytes[2], seed_bytes[3]]),
        u32::from_le_bytes([seed_bytes[4], seed_bytes[5], seed_bytes[6], seed_bytes[7]]),
        u32::from_le_bytes([seed_bytes[8], seed_bytes[9], seed_bytes[10], seed_bytes[11]]),
        u32::from_le_bytes([
            seed_bytes[12],
            seed_bytes[13],
            seed_bytes[14],
            seed_bytes[15],
        ]),
    ];
    // All-zero guard (matching rand_xorshift 0.3):
    if state.iter().all(|&x| x == 0) {
        state = [0xBAD_5EED, 0xBAD_5EED, 0xBAD_5EED, 0xBAD_5EED];
    }

    // Step 3: Initialize identity permutation [0, 1, 2, ..., 255].
    let mut values = [0u8; 256];
    for (i, v) in values.iter_mut().enumerate() {
        *v = i as u8;
    }

    // Step 4: Fisher-Yates shuffle using gen_index (matches rand 0.8's SliceRandom::shuffle).
    for i in (1..256usize).rev() {
        let j = xorshift_gen_index(&mut state, (i + 1) as u32) as usize;
        values.swap(i, j);
    }

    values
}

/// XorShift128 next_u32 (matches rand_xorshift 0.3).
fn xorshift_next_u32(state: &mut [u32; 4]) -> u32 {
    let x = state[0];
    let t = x ^ (x << 11);
    state[0] = state[1];
    state[1] = state[2];
    state[2] = state[3];
    let w = state[3];
    state[3] = w ^ (w >> 19) ^ (t ^ (t >> 8));
    state[3]
}

/// Uniform random index in `[0, ubound)` matching rand 0.8's
/// `UniformInt<u32>::sample_single_inclusive` with widening multiply.
fn xorshift_gen_index(state: &mut [u32; 4], ubound: u32) -> u32 {
    let range = ubound;
    if range == 0 {
        return xorshift_next_u32(state);
    }

    let zone = (range << range.leading_zeros()).wrapping_sub(1);

    loop {
        let v = xorshift_next_u32(state);
        let wide = (v as u64) * (range as u64);
        let hi = (wide >> 32) as u32;
        let lo = wide as u32;
        if lo <= zone {
            return hi;
        }
    }
}

/// Generate all permutation tables needed for the noise pipeline.
///
/// Returns a flat `Vec<u32>` of `NUM_PERM_TABLES × 256` values, where each
/// `[u8; 256]` table is promoted to `u32` for GPU buffer alignment.
fn generate_all_perm_tables(seed: u32, config: &NoiseConfig) -> Vec<u32> {
    let mut tables = vec![0u32; NUM_PERM_TABLES * PERM_TABLE_SIZE];

    let mut store = |slot: usize, table_seed: u32| {
        let table = generate_perm_table(table_seed);
        let base = slot * PERM_TABLE_SIZE;
        for (i, &v) in table.iter().enumerate() {
            tables[base + i] = v as u32;
        }
    };

    // FBM octaves: seed + 0, seed + 1, ..., seed + N-1
    for i in 0..MAX_OCTAVES {
        if i < config.fbm_octaves as usize {
            store(PERM_FBM_START + i, seed.wrapping_add(i as u32));
        }
    }

    // Ridged octaves: seed + 50, seed + 51, ..., seed + 50 + M-1
    for i in 0..MAX_OCTAVES {
        if i < config.ridged_octaves as usize {
            store(PERM_RIDGED_START + i, seed.wrapping_add(50 + i as u32));
        }
    }

    // Single-instance noise functions
    store(PERM_SELECTOR, seed.wrapping_add(100));
    store(PERM_WARP_X, seed.wrapping_add(200));
    store(PERM_WARP_Z, seed.wrapping_add(201));
    store(PERM_MICRO, seed.wrapping_add(300));
    store(PERM_CONTINENT, seed.wrapping_add(350));
    store(PERM_OCEAN_FLOOR, seed.wrapping_add(360));

    // 3D material noise (matches CachedGeologyPerlin seeds in terrain.rs)
    store(PERM_STRATA, seed.wrapping_add(400));
    store(PERM_ORE_COAL, seed.wrapping_add(500));
    store(PERM_ORE_COPPER, seed.wrapping_add(501));
    store(PERM_ORE_IRON, seed.wrapping_add(502));
    store(PERM_ORE_GOLD, seed.wrapping_add(503));
    store(PERM_CAVE_CAVERN, seed.wrapping_add(600));
    store(PERM_CAVE_TUNNEL, seed.wrapping_add(601));
    store(PERM_CAVE_TUBE_XZ, seed.wrapping_add(602));
    store(PERM_CAVE_TUBE_XY, seed.wrapping_add(603));
    store(PERM_CRYSTAL, seed.wrapping_add(604));

    tables
}

// ─── GPU data structures ───────────────────────────────────────────────────

/// Noise parameters — matches WGSL `NoiseParams` struct (96 bytes, 16-byte aligned).
/// Identical to `noise_compute::NoiseParams` but kept separate to avoid coupling.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct NoiseParamsGpu {
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

impl NoiseParamsGpu {
    fn from_config(config: &NoiseConfig, seed: u32, mean_radius: f64, height_scale: f64) -> Self {
        Self {
            fbm_octaves: config.fbm_octaves,
            fbm_persistence: config.fbm_persistence as f32,
            fbm_lacunarity: config.fbm_lacunarity as f32,
            fbm_base_freq: config.fbm_base_freq as f32,
            ridged_octaves: config.ridged_octaves,
            ridged_gain: config.ridged_gain as f32,
            ridged_base_freq: config.ridged_base_freq as f32,
            selector_freq: config.selector_freq as f32,
            selector_lo: config.selector_thresholds.0 as f32,
            selector_hi: config.selector_thresholds.1 as f32,
            warp_strength: config.warp_strength as f32,
            warp_freq: config.warp_freq as f32,
            micro_freq: config.micro_freq as f32,
            micro_amplitude: config.micro_amplitude as f32,
            continent_enabled: config.continent_enabled as u32,
            continent_freq: config.continent_freq as f32,
            continent_threshold: config.continent_threshold as f32,
            shelf_blend_width: config.shelf_blend_width as f32,
            ocean_floor_depth: config.ocean_floor_depth as f32,
            ocean_floor_amplitude: config.ocean_floor_amplitude as f32,
            seed,
            _pad0: 0,
            mean_radius: mean_radius as f32,
            height_scale: height_scale as f32,
        }
    }
}

/// Per-chunk descriptor — matches WGSL `ChunkDesc`.
///
/// Radial values are stored as OFFSETS from `mean_radius` (not absolute
/// radii) so that the shader can do the surface/voxel comparison in f32
/// without catastrophic cancellation. At Earth scale (`r ≈ 6.37e6`), f32
/// only has ~0.5 m precision on the absolute radius; subtracting two such
/// values to recover a ±meter depth was randomizing the surface material
/// classification and producing severely fragmented terrain.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ChunkDesc {
    pub center: [f32; 4],
    pub rotation: [f32; 4],
    pub tangent_scale: [f32; 4],
    pub base_r_offset: f32,
    pub top_r_offset: f32,
    pub sea_level_offset: f32,
    pub lod_scale: f32,
    pub soil_depth: f32,
    pub cave_threshold: f32,
    pub half_diag: f32,
    pub chunk_index: u32,
    pub chunk_r_offset: f32,
    pub _pad0: f32,
    pub _pad1: f32,
    pub _pad2: f32,
}

/// Global dispatch parameters — matches WGSL `DispatchParams`.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct DispatchParamsGpu {
    chunk_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    rotation_axis: [f32; 4],
}

/// Request to generate voxels for a chunk on the GPU.
pub struct GpuChunkRequest {
    pub coord: CubeSphereCoord,
    pub desc: ChunkDesc,
}

/// Result of a completed GPU voxel batch.
pub struct GpuVoxelBatchResult {
    pub terrain_data: Vec<V2TerrainData>,
}

// ─── Pipeline ──────────────────────────────────────────────────────────────

/// GPU compute pipeline for batched voxel generation.
pub struct GpuVoxelCompute {
    ctx: Arc<GpuContext>,
    surface_pipeline: wgpu::ComputePipeline,
    classify_pipeline: wgpu::ComputePipeline,
    voxel_pipeline: wgpu::ComputePipeline,
    // Bind group layouts.
    bgl_uniforms: wgpu::BindGroupLayout,
    bgl_chunks: wgpu::BindGroupLayout,
    bgl_surface: wgpu::BindGroupLayout,
    bgl_voxels: wgpu::BindGroupLayout,
    // Pre-allocated buffers.
    noise_params_buffer: wgpu::Buffer,
    dispatch_params_buffer: wgpu::Buffer,
    chunks_buffer: wgpu::Buffer,
    surface_buffer: wgpu::Buffer,
    chunk_info_buffer: wgpu::Buffer,
    materials_buffer: wgpu::Buffer,
    densities_buffer: wgpu::Buffer,
    perm_tables_buffer: wgpu::Buffer,
    // Staging buffers for CPU readback.
    materials_staging: wgpu::Buffer,
    densities_staging: wgpu::Buffer,
    chunk_info_staging: wgpu::Buffer,
    // Cached noise params.
    noise_params: NoiseParamsGpu,
}

impl GpuVoxelCompute {
    /// Create a new voxel compute pipeline, or `None` if no GPU is available.
    pub fn try_new(
        config: &NoiseConfig,
        seed: u32,
        mean_radius: f64,
        height_scale: f64,
    ) -> Option<Self> {
        let ctx = Arc::new(GpuContext::try_new()?);
        Some(Self::new_with_context(
            ctx,
            config,
            seed,
            mean_radius,
            height_scale,
        ))
    }

    /// Create with a pre-existing GPU context.
    pub fn new_with_context(
        ctx: Arc<GpuContext>,
        config: &NoiseConfig,
        seed: u32,
        mean_radius: f64,
        height_scale: f64,
    ) -> Self {
        let shader_source = include_str!("shaders/voxel_gen.wgsl");
        let shader_module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("voxel_gen"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        // Bind group 0: uniforms (noise_params + dispatch_params)
        let bgl_uniforms = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("voxel_uniforms_layout"),
                entries: &[bgl_uniform(0), bgl_uniform(1)],
            });

        // Bind group 1: chunk descriptors + permutation tables (read-only storage)
        let bgl_chunks = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("voxel_chunks_layout"),
                entries: &[bgl_storage_ro(0), bgl_storage_ro(1)],
            });

        // Bind group 2: surface pass outputs (surface_radii + chunk_info)
        let bgl_surface = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("voxel_surface_layout"),
                entries: &[bgl_storage_rw(0), bgl_storage_rw(1)],
            });

        // Bind group 3: voxel pass outputs (materials + densities)
        let bgl_voxels = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("voxel_output_layout"),
                entries: &[bgl_storage_rw(0), bgl_storage_rw(1)],
            });

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("voxel_pipeline_layout"),
                bind_group_layouts: &[&bgl_uniforms, &bgl_chunks, &bgl_surface, &bgl_voxels],
                push_constant_ranges: &[],
            });

        let surface_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("surface_pass"),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: Some("surface_pass"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        let classify_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("classify_pass"),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: Some("classify_pass"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        let voxel_pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("voxel_pass"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("voxel_pass"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Pre-allocate buffers at maximum capacity.
        let max_chunks = MAX_CHUNKS_PER_BATCH;

        let noise_params = NoiseParamsGpu::from_config(config, seed, mean_radius, height_scale);
        let noise_params_buffer =
            ctx.create_uniform_buffer("voxel_noise_params", bytemuck::bytes_of(&noise_params));

        let dispatch_params = DispatchParamsGpu {
            chunk_count: 0,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            rotation_axis: [0.0, 1.0, 0.0, 0.0],
        };
        let dispatch_params_buffer = ctx.create_uniform_buffer(
            "voxel_dispatch_params",
            bytemuck::bytes_of(&dispatch_params),
        );

        let chunk_desc_size = std::mem::size_of::<ChunkDesc>();
        let chunks_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel_chunks"),
            size: (max_chunks * chunk_desc_size) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Surface radii: max_chunks * 1024 floats
        let surface_buf_size = (max_chunks * CHUNK_AREA * 4) as u64;
        let surface_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel_surface_radii"),
            size: surface_buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Chunk info: max_chunks * 4 u32s
        let chunk_info_size = (max_chunks * 4 * 4) as u64;
        let chunk_info_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel_chunk_info"),
            size: chunk_info_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Voxel materials: max_chunks * 32768 u32s
        let materials_buf_size = (max_chunks * CHUNK_VOLUME * 4) as u64;
        let materials_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel_materials"),
            size: materials_buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Voxel densities: max_chunks * 32768 f32s
        let densities_buf_size = (max_chunks * CHUNK_VOLUME * 4) as u64;
        let densities_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel_densities"),
            size: densities_buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Staging buffers for readback.
        let materials_staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel_materials_staging"),
            size: materials_buf_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let densities_staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel_densities_staging"),
            size: densities_buf_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let chunk_info_staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel_chunk_info_staging"),
            size: chunk_info_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Permutation tables: 32 tables × 256 u32 = 32 KB
        let perm_data = generate_all_perm_tables(seed, config);
        let perm_tables_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel_perm_tables"),
            size: (NUM_PERM_TABLES * PERM_TABLE_SIZE * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue
            .write_buffer(&perm_tables_buffer, 0, bytemuck::cast_slice(&perm_data));

        Self {
            ctx,
            surface_pipeline,
            classify_pipeline,
            voxel_pipeline,
            bgl_uniforms,
            bgl_chunks,
            bgl_surface,
            bgl_voxels,
            noise_params_buffer,
            dispatch_params_buffer,
            chunks_buffer,
            surface_buffer,
            chunk_info_buffer,
            materials_buffer,
            densities_buffer,
            perm_tables_buffer,
            materials_staging,
            densities_staging,
            chunk_info_staging,
            noise_params,
        }
    }

    /// Run the full voxel generation pipeline for a batch of chunks.
    ///
    /// This is a blocking call — intended to be run on a worker thread, not
    /// the main ECS thread.
    pub fn generate_batch(
        &self,
        requests: &[GpuChunkRequest],
        rotation_axis: [f64; 3],
    ) -> GpuVoxelBatchResult {
        assert!(
            requests.len() <= MAX_CHUNKS_PER_BATCH,
            "batch size {} exceeds MAX_CHUNKS_PER_BATCH {MAX_CHUNKS_PER_BATCH}",
            requests.len(),
        );

        if requests.is_empty() {
            return GpuVoxelBatchResult {
                terrain_data: Vec::new(),
            };
        }

        let chunk_count = requests.len() as u32;

        // Upload dispatch params.
        let dispatch_params = DispatchParamsGpu {
            chunk_count,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            rotation_axis: [
                rotation_axis[0] as f32,
                rotation_axis[1] as f32,
                rotation_axis[2] as f32,
                0.0,
            ],
        };
        self.ctx.queue.write_buffer(
            &self.dispatch_params_buffer,
            0,
            bytemuck::bytes_of(&dispatch_params),
        );

        // Upload noise params (in case they changed).
        self.ctx.queue.write_buffer(
            &self.noise_params_buffer,
            0,
            bytemuck::bytes_of(&self.noise_params),
        );

        // Upload chunk descriptors.
        let descs: Vec<ChunkDesc> = requests.iter().map(|r| r.desc).collect();
        self.ctx
            .queue
            .write_buffer(&self.chunks_buffer, 0, bytemuck::cast_slice(&descs));

        // Initialize chunk_info with sentinel values for the sortable-u32
        // atomic min/max reduction over signed surface offsets. See
        // `sortable_encode` / `sortable_decode` in voxel_gen.wgsl.
        //   min_surface_offset → sortable_encode(f32::MAX)  = 0xFF7FFFFF
        //   max_surface_offset → sortable_encode(-f32::MAX) = 0x00800000
        let min_init = sortable_encode_u32(f32::MAX);
        let max_init = sortable_encode_u32(-f32::MAX);
        let mut info_init = vec![0u32; requests.len() * 4];
        for i in 0..requests.len() {
            info_init[i * 4] = CLASS_MIXED; // classification (default)
            info_init[i * 4 + 1] = min_init;
            info_init[i * 4 + 2] = max_init;
            info_init[i * 4 + 3] = 0; // solid material
        }
        self.ctx
            .queue
            .write_buffer(&self.chunk_info_buffer, 0, bytemuck::cast_slice(&info_init));

        // Build bind groups.
        let bg_uniforms = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("voxel_uniforms_bg"),
                layout: &self.bgl_uniforms,
                entries: &[
                    bg_entry(0, &self.noise_params_buffer),
                    bg_entry(1, &self.dispatch_params_buffer),
                ],
            });

        let bg_chunks = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("voxel_chunks_bg"),
                layout: &self.bgl_chunks,
                entries: &[
                    bg_entry(0, &self.chunks_buffer),
                    bg_entry(1, &self.perm_tables_buffer),
                ],
            });

        let bg_surface = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("voxel_surface_bg"),
                layout: &self.bgl_surface,
                entries: &[
                    bg_entry(0, &self.surface_buffer),
                    bg_entry(1, &self.chunk_info_buffer),
                ],
            });

        let bg_voxels = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("voxel_voxels_bg"),
                layout: &self.bgl_voxels,
                entries: &[
                    bg_entry(0, &self.materials_buffer),
                    bg_entry(1, &self.densities_buffer),
                ],
            });

        // Encode passes.
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("voxel_gen"),
            });

        // Pass 1: Surface radius computation.
        {
            let total_threads = chunk_count * CHUNK_AREA as u32;
            let workgroups = total_threads.div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("surface_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.surface_pipeline);
            pass.set_bind_group(0, &bg_uniforms, &[]);
            pass.set_bind_group(1, &bg_chunks, &[]);
            pass.set_bind_group(2, &bg_surface, &[]);
            pass.set_bind_group(3, &bg_voxels, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Pass 2: Classification.
        {
            let workgroups = chunk_count.div_ceil(64);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("classify_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.classify_pipeline);
            pass.set_bind_group(0, &bg_uniforms, &[]);
            pass.set_bind_group(1, &bg_chunks, &[]);
            pass.set_bind_group(2, &bg_surface, &[]);
            pass.set_bind_group(3, &bg_voxels, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Pass 3: Voxel fill (only runs for mixed chunks — shader checks classification).
        {
            let total_threads = chunk_count * CHUNK_VOLUME as u32;
            let workgroups = total_threads.div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("voxel_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.voxel_pipeline);
            pass.set_bind_group(0, &bg_uniforms, &[]);
            pass.set_bind_group(1, &bg_chunks, &[]);
            pass.set_bind_group(2, &bg_surface, &[]);
            pass.set_bind_group(3, &bg_voxels, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Copy outputs to staging buffers.
        let info_copy_size = (requests.len() * 4 * 4) as u64;
        encoder.copy_buffer_to_buffer(
            &self.chunk_info_buffer,
            0,
            &self.chunk_info_staging,
            0,
            info_copy_size,
        );

        let mat_copy_size = (requests.len() * CHUNK_VOLUME * 4) as u64;
        encoder.copy_buffer_to_buffer(
            &self.materials_buffer,
            0,
            &self.materials_staging,
            0,
            mat_copy_size,
        );

        let density_copy_size = (requests.len() * CHUNK_VOLUME * 4) as u64;
        encoder.copy_buffer_to_buffer(
            &self.densities_buffer,
            0,
            &self.densities_staging,
            0,
            density_copy_size,
        );

        // Submit and wait.
        self.ctx.submit_and_wait(encoder);

        // Read back results.
        let info_raw = self
            .ctx
            .read_buffer(&self.chunk_info_staging, info_copy_size);
        let info: &[u32] = bytemuck::cast_slice(&info_raw);

        let mat_raw = self.ctx.read_buffer(&self.materials_staging, mat_copy_size);
        let materials: &[u32] = bytemuck::cast_slice(&mat_raw);

        let density_raw = self
            .ctx
            .read_buffer(&self.densities_staging, density_copy_size);
        let densities: &[f32] = bytemuck::cast_slice(&density_raw);

        // Unpack results into V2TerrainData.
        let mut terrain_data = Vec::with_capacity(requests.len());
        for (i, req) in requests.iter().enumerate() {
            let info_base = i * 4;
            let classification = info[info_base];
            let solid_mat = info[info_base + 3];

            let cached = match classification {
                CLASS_ALL_AIR => CachedVoxels::AllAir,
                CLASS_ALL_SOLID => CachedVoxels::AllSolid(MaterialId(solid_mat as u16)),
                _ => {
                    // Mixed — unpack voxel data.
                    let voxel_base = i * CHUNK_VOLUME;
                    let mut voxels = vec![Voxel::default(); CHUNK_VOLUME];
                    for j in 0..CHUNK_VOLUME {
                        voxels[j].material = MaterialId(materials[voxel_base + j] as u16);
                        voxels[j].density = densities[voxel_base + j];
                    }
                    CachedVoxels::Mixed(Arc::new(voxels))
                }
            };

            terrain_data.push(V2TerrainData {
                coord: req.coord,
                voxels: cached,
            });
        }

        GpuVoxelBatchResult { terrain_data }
    }
}

// ─── Helper: build ChunkDesc from CubeSphereCoord + planet config ──────────

/// Order-preserving encoding of a signed f32 into a u32 so that the GPU's
/// atomic min/max works correctly across negative surface offsets.
///
/// Mirrors `sortable_encode` in `voxel_gen.wgsl`. Used on the host to seed
/// the atomic reduction buffers; must match the WGSL version bit-for-bit.
#[inline]
fn sortable_encode_u32(f: f32) -> u32 {
    let bits = f.to_bits();
    if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000
    }
}

/// Build a `ChunkDesc` from a `CubeSphereCoord` and planet parameters.
pub fn chunk_desc_from_coord(
    coord: CubeSphereCoord,
    mean_radius: f64,
    face_chunks_per_edge: f64,
    sea_level: f64,
    soil_depth: f64,
    cave_threshold: f64,
    chunk_index: u32,
) -> ChunkDesc {
    let cs = CHUNK_SIZE;
    let (center, rotation, tangent_scale) =
        coord.world_transform_scaled(mean_radius, face_chunks_per_edge);

    let lod_scale = (1u64 << coord.lod) as f64;
    // Chunk center's radial offset from `mean_radius`. Exact in f64 and
    // bounded by `|layer| * CS * lod_scale`, comfortably within f32 range.
    let chunk_r_offset = coord.layer as f64 * cs as f64 * lod_scale;
    let half_cs_scaled = cs as f64 * lod_scale / 2.0;
    let base_r_offset = chunk_r_offset - half_cs_scaled;
    let top_r_offset = chunk_r_offset + half_cs_scaled;

    let half_diag = ((tangent_scale.x as f64).powi(2)
        + (tangent_scale.y as f64).powi(2)
        + (tangent_scale.z as f64).powi(2))
    .sqrt()
        * cs as f64
        / 2.0;

    ChunkDesc {
        center: [center.x, center.y, center.z, 0.0],
        rotation: [rotation.x, rotation.y, rotation.z, rotation.w],
        tangent_scale: [tangent_scale.x, tangent_scale.y, tangent_scale.z, 0.0],
        base_r_offset: base_r_offset as f32,
        top_r_offset: top_r_offset as f32,
        sea_level_offset: (sea_level - mean_radius) as f32,
        lod_scale: lod_scale as f32,
        soil_depth: soil_depth as f32,
        cave_threshold: cave_threshold as f32,
        half_diag: half_diag as f32,
        chunk_index,
        chunk_r_offset: chunk_r_offset as f32,
        _pad0: 0.0,
        _pad1: 0.0,
        _pad2: 0.0,
    }
}

// ─── Bind group helpers ────────────────────────────────────────────────────

fn bgl_uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_storage_ro(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_storage_rw(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bg_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // GPU tests crash with SIGSEGV when multiple tests concurrently create
    // wgpu devices. Serialize through crate-level lock.
    use crate::gpu::GPU_TEST_LOCK;

    #[test]
    fn noise_params_gpu_size() {
        assert_eq!(std::mem::size_of::<NoiseParamsGpu>(), 96);
    }

    #[test]
    fn chunk_desc_size() {
        // ChunkDesc must be a multiple of 16 bytes for GPU alignment.
        let size = std::mem::size_of::<ChunkDesc>();
        assert_eq!(size % 16, 0, "ChunkDesc size {size} not 16-byte aligned");
    }

    #[test]
    fn dispatch_params_size() {
        let size = std::mem::size_of::<DispatchParamsGpu>();
        assert_eq!(size, 32); // 1 u32 + 3 pad + vec4
    }

    #[test]
    fn empty_batch_returns_empty() {
        let _guard = GPU_TEST_LOCK.lock().unwrap();
        let config = NoiseConfig::default();
        let Some(compute) = GpuVoxelCompute::try_new(&config, 42, 6_371_000.0, 8_800.0) else {
            eprintln!("skipping GPU test: no adapter");
            return;
        };
        let result = compute.generate_batch(&[], [0.0, 1.0, 0.0]);
        assert!(result.terrain_data.is_empty());
    }

    #[test]
    fn single_chunk_generates_terrain() {
        let _guard = GPU_TEST_LOCK.lock().unwrap();
        let config = NoiseConfig::default();
        let mean_radius = 6_371_000.0;
        let height_scale = 8_800.0;
        let Some(compute) = GpuVoxelCompute::try_new(&config, 42, mean_radius, height_scale) else {
            eprintln!("skipping GPU test: no adapter");
            return;
        };

        // Surface chunk at face 0, origin position.
        let coord = CubeSphereCoord::new_with_lod(
            crate::world::v2::cubed_sphere::CubeFace::PosX,
            0,
            0,
            0, // surface layer
            0, // LOD 0
        );
        let fce = CubeSphereCoord::face_chunks_per_edge(mean_radius);
        let desc =
            chunk_desc_from_coord(coord, mean_radius, fce, mean_radius - 100.0, 4.0, -0.3, 0);

        let request = GpuChunkRequest { coord, desc };
        let result = compute.generate_batch(&[request], [0.0, 1.0, 0.0]);

        assert_eq!(result.terrain_data.len(), 1);
        // The chunk should produce some result (not panic).
        let td = &result.terrain_data[0];
        assert_eq!(td.coord, coord);
    }

    #[test]
    fn all_air_chunk_classification() {
        let _guard = GPU_TEST_LOCK.lock().unwrap();
        let config = NoiseConfig::default();
        let mean_radius = 6_371_000.0;
        let height_scale = 8_800.0;
        let Some(compute) = GpuVoxelCompute::try_new(&config, 42, mean_radius, height_scale) else {
            eprintln!("skipping GPU test: no adapter");
            return;
        };

        // Chunk well above the surface (layer = 100).
        let coord = CubeSphereCoord::new_with_lod(
            crate::world::v2::cubed_sphere::CubeFace::PosX,
            0,
            0,
            100, // way above surface
            0,
        );
        let fce = CubeSphereCoord::face_chunks_per_edge(mean_radius);
        let desc =
            chunk_desc_from_coord(coord, mean_radius, fce, mean_radius - 100.0, 4.0, -0.3, 0);

        let request = GpuChunkRequest { coord, desc };
        let result = compute.generate_batch(&[request], [0.0, 1.0, 0.0]);

        assert_eq!(result.terrain_data.len(), 1);
        assert!(
            matches!(result.terrain_data[0].voxels, CachedVoxels::AllAir),
            "chunk at layer 100 should be AllAir"
        );
    }

    #[test]
    fn multi_chunk_batch() {
        let _guard = GPU_TEST_LOCK.lock().unwrap();
        let config = NoiseConfig::default();
        let mean_radius = 6_371_000.0;
        let height_scale = 8_800.0;
        let Some(compute) = GpuVoxelCompute::try_new(&config, 42, mean_radius, height_scale) else {
            eprintln!("skipping GPU test: no adapter");
            return;
        };

        let fce = CubeSphereCoord::face_chunks_per_edge(mean_radius);
        let mut requests = Vec::new();
        for i in 0..8 {
            let coord = CubeSphereCoord::new_with_lod(
                crate::world::v2::cubed_sphere::CubeFace::PosX,
                i,
                0,
                0,
                0,
            );
            let desc = chunk_desc_from_coord(
                coord,
                mean_radius,
                fce,
                mean_radius - 100.0,
                4.0,
                -0.3,
                i as u32,
            );
            requests.push(GpuChunkRequest { coord, desc });
        }

        let result = compute.generate_batch(&requests, [0.0, 1.0, 0.0]);
        assert_eq!(result.terrain_data.len(), 8);

        // Each chunk should have the correct coord.
        for (i, td) in result.terrain_data.iter().enumerate() {
            assert_eq!(td.coord.u, i as i32);
        }
    }

    /// Diagnostic test: compare GPU and CPU terrain for a surface-crossing chunk.
    ///
    /// Uses the actual planet_config.ron parameters (32 km planet, height_scale=4000).
    /// Finds the actual surface layer by sampling CPU noise, then compares GPU and CPU
    /// terrain generation for that chunk.
    #[test]
    fn gpu_vs_cpu_surface_chunk_comparison() {
        let _guard = GPU_TEST_LOCK.lock().unwrap();
        use crate::world::terrain::{SphericalTerrainGenerator, UnifiedTerrainGenerator};
        use crate::world::v2::terrain_gen::generate_v2_voxels;

        let config = NoiseConfig {
            fbm_base_freq: 1.0,
            ridged_base_freq: 1.5,
            selector_freq: 0.5,
            warp_strength: 0.1,
            warp_freq: 0.8,
            micro_freq: 15.0,
            micro_amplitude: 0.02,
            continent_enabled: false,
            ..Default::default()
        };
        let mean_radius = 32_000.0;
        let height_scale = 4_000.0;
        let sea_level = 32_000.0;
        let seed = 42u32;

        // GPU path
        let Some(compute) = GpuVoxelCompute::try_new(&config, seed, mean_radius, height_scale)
        else {
            eprintln!("skipping GPU test: no adapter");
            return;
        };

        // CPU path
        let planet = crate::world::planet::PlanetConfig {
            mean_radius,
            sea_level_radius: sea_level,
            height_scale,
            seed,
            noise: Some(config.clone()),
            ..Default::default()
        };
        let tgen = SphericalTerrainGenerator::new(planet.clone());
        let unified = std::sync::Arc::new(UnifiedTerrainGenerator::Spherical(Box::new(tgen)));

        let fce = CubeSphereCoord::face_chunks_per_edge(mean_radius);
        let half_fce = (fce / 2.0) as i32;

        // First, find the actual surface height at the center of PosX face
        // by sampling CPU noise at (lat=0, lon=0).
        let surface_r = unified.sample_surface_radius_at(0.0, 0.0);
        let surface_layer = ((surface_r - mean_radius) / CHUNK_SIZE as f64) as i32;
        eprintln!(
            "Surface radius at (0,0): {surface_r:.1}, mean_radius: {mean_radius}, \
             surface_layer: {surface_layer}"
        );

        // Test the surface-crossing layer
        let coord = CubeSphereCoord::new_with_lod(
            crate::world::v2::cubed_sphere::CubeFace::PosX,
            half_fce,
            half_fce,
            surface_layer,
            0, // LOD 0
        );

        // GPU generation
        let desc = chunk_desc_from_coord(
            coord,
            mean_radius,
            fce,
            sea_level,
            planet.soil_depth,
            planet.cave_threshold,
            0,
        );
        eprintln!(
            "ChunkDesc: base_r_offset={}, top_r_offset={}, sea_level_offset={}, half_diag={}",
            desc.base_r_offset, desc.top_r_offset, desc.sea_level_offset, desc.half_diag
        );

        let request = GpuChunkRequest { coord, desc };
        let gpu_result = compute.generate_batch(&[request], planet.rotation_axis);
        let gpu_td = &gpu_result.terrain_data[0];

        // CPU generation
        let cpu_td = generate_v2_voxels(coord, mean_radius, fce, &unified);

        // Analyze classifications
        let label = |v: &CachedVoxels| match v {
            CachedVoxels::AllAir => "AllAir".to_string(),
            CachedVoxels::AllSolid(m) => format!("AllSolid({})", m.0),
            CachedVoxels::Mixed(_) => "Mixed".to_string(),
        };
        eprintln!("GPU classification: {}", label(&gpu_td.voxels));
        eprintln!("CPU classification: {}", label(&cpu_td.voxels));

        // Count materials for both
        let gpu_voxels = crate::world::v2::terrain_gen::cached_voxels_to_vec(&gpu_td.voxels);
        let cpu_voxels = crate::world::v2::terrain_gen::cached_voxels_to_vec(&cpu_td.voxels);

        let count_materials = |voxels: &[Voxel]| -> Vec<(u16, usize)> {
            let mut counts: std::collections::HashMap<u16, usize> =
                std::collections::HashMap::new();
            for v in voxels {
                *counts.entry(v.material.0).or_default() += 1;
            }
            let mut sorted: Vec<_> = counts.into_iter().collect();
            sorted.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
            sorted
        };

        eprintln!("GPU materials (top 5):");
        for (mat, count) in count_materials(&gpu_voxels).iter().take(5) {
            eprintln!(
                "  mat {mat}: {count} voxels ({:.1}%)",
                *count as f64 / CHUNK_VOLUME as f64 * 100.0
            );
        }
        eprintln!("CPU materials (top 5):");
        for (mat, count) in count_materials(&cpu_voxels).iter().take(5) {
            eprintln!(
                "  mat {mat}: {count} voxels ({:.1}%)",
                *count as f64 / CHUNK_VOLUME as f64 * 100.0
            );
        }

        // The critical check: GPU and CPU must AGREE on classification.
        assert_eq!(
            label(&gpu_td.voxels),
            label(&cpu_td.voxels),
            "GPU and CPU classifications must match for layer {surface_layer}"
        );

        // Compare material distributions.
        let gpu_air = gpu_voxels
            .iter()
            .filter(|v| v.material == MaterialId::AIR)
            .count();
        let gpu_solid = gpu_voxels.len() - gpu_air;
        let cpu_air = cpu_voxels
            .iter()
            .filter(|v| v.material == MaterialId::AIR)
            .count();
        let cpu_solid = cpu_voxels.len() - cpu_air;
        eprintln!(
            "GPU: {gpu_air} air, {gpu_solid} solid ({:.1}% solid)",
            gpu_solid as f64 / gpu_voxels.len() as f64 * 100.0
        );
        eprintln!(
            "CPU: {cpu_air} air, {cpu_solid} solid ({:.1}% solid)",
            cpu_solid as f64 / cpu_voxels.len() as f64 * 100.0
        );

        // Air/solid ratio should be close (allow f32 vs f64 drift).
        let gpu_pct = gpu_solid as f64 / gpu_voxels.len() as f64;
        let cpu_pct = cpu_solid as f64 / cpu_voxels.len() as f64;
        let pct_diff = (gpu_pct - cpu_pct).abs();
        eprintln!("Solid% diff: {pct_diff:.4}");
        assert!(
            pct_diff < 0.10,
            "GPU and CPU solid% differ too much: GPU {gpu_pct:.3} vs CPU {cpu_pct:.3}"
        );
    }

    /// Verify that our perm table generation matches the `noise` crate's Perlin.
    ///
    /// We can't access the internal perm table directly, so we verify by
    /// computing a Perlin noise reference value and comparing our CPU-side
    /// implementation using the generated table.
    #[test]
    fn perm_table_matches_noise_crate() {
        use noise::{NoiseFn, Perlin};

        let seed = 42u32;
        let table = generate_perm_table(seed);

        let perlin = Perlin::new(seed);

        // Compare 2D Perlin output at several test points.
        // We implement the same algorithm as the noise crate using our table.
        let test_points_2d: &[(f64, f64)] = &[
            (0.5, 0.5),
            (1.0, 2.0),
            (-0.3, 0.7),
            (100.123, -50.456),
            (0.001, 0.001),
            (-1.0, -1.0),
            (255.5, 255.5),
        ];

        for &(x, z) in test_points_2d {
            let expected = perlin.get([x, z]);
            let actual = cpu_perlin2d(x, z, &table);
            let diff = (expected - actual).abs();
            assert!(
                diff < 1e-6,
                "2D Perlin mismatch at ({x}, {z}): expected {expected}, got {actual}, diff {diff}"
            );
        }

        // Compare 3D Perlin output.
        let test_points_3d: &[(f64, f64, f64)] = &[
            (0.5, 0.5, 0.5),
            (1.0, 2.0, 3.0),
            (-0.3, 0.7, -1.5),
            (100.123, -50.456, 25.789),
        ];

        for &(x, y, z) in test_points_3d {
            let expected = perlin.get([x, y, z]);
            let actual = cpu_perlin3d(x, y, z, &table);
            let diff = (expected - actual).abs();
            assert!(
                diff < 1e-6,
                "3D Perlin mismatch at ({x}, {y}, {z}): expected {expected}, got {actual}, diff {diff}"
            );
        }
    }

    // ── CPU reference Perlin using our generated perm table ──────────────

    fn noise_floor_cpu(x: f64) -> isize {
        if x <= 0.0 { x as isize - 1 } else { x as isize }
    }

    fn fade_cpu(t: f64) -> f64 {
        t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    }

    fn grad2d_cpu(hash: u8, x: f64, y: f64) -> f64 {
        let h = hash & 3;
        let gx = if h & 1 == 0 { 1.0 } else { -1.0 };
        let gy = if h & 2 == 0 { 1.0 } else { -1.0 };
        gx * x + gy * y
    }

    fn grad3d_cpu(hash: u8, x: f64, y: f64, z: f64) -> f64 {
        // Must match the noise crate's exact gradient table (not the classic
        // Perlin shortcut, which differs at cases 13 and 14).
        match hash & 15 {
            0 | 12 => x + y,
            1 | 13 => -x + y,
            2 => x - y,
            3 => -x - y,
            4 => x + z,
            5 => -x + z,
            6 => x - z,
            7 => -x - z,
            8 => y + z,
            9 | 14 => -y + z,
            10 => y - z,
            11 | 15 => -y - z,
            _ => unreachable!(),
        }
    }

    fn cpu_perlin2d(x: f64, z: f64, table: &[u8; 256]) -> f64 {
        let scale = 2.0_f64 / std::f64::consts::SQRT_2;
        let ix = noise_floor_cpu(x);
        let iz = noise_floor_cpu(z);
        let fx = x - ix as f64;
        let fz = z - iz as f64;

        let ux = (ix & 0xff) as usize;
        let uz = (iz & 0xff) as usize;

        let h00 = table[table[ux] as usize ^ uz];
        let h10 = table[table[(ux + 1) & 0xff] as usize ^ uz];
        let h01 = table[table[ux] as usize ^ ((uz + 1) & 0xff)];
        let h11 = table[table[(ux + 1) & 0xff] as usize ^ ((uz + 1) & 0xff)];

        let g00 = grad2d_cpu(h00, fx, fz);
        let g10 = grad2d_cpu(h10, fx - 1.0, fz);
        let g01 = grad2d_cpu(h01, fx, fz - 1.0);
        let g11 = grad2d_cpu(h11, fx - 1.0, fz - 1.0);

        let cu = fade_cpu(fx);
        let cv = fade_cpu(fz);

        fn lerp(a: f64, b: f64, t: f64) -> f64 {
            a + (b - a) * t
        }

        let result = lerp(lerp(g00, g01, cv), lerp(g10, g11, cv), cu) * scale;
        result.clamp(-1.0, 1.0)
    }

    fn cpu_perlin3d(x: f64, y: f64, z: f64, table: &[u8; 256]) -> f64 {
        let scale = 2.0_f64 / 3.0_f64.sqrt();
        let ix = noise_floor_cpu(x);
        let iy = noise_floor_cpu(y);
        let iz = noise_floor_cpu(z);
        let fx = x - ix as f64;
        let fy = y - iy as f64;
        let fz = z - iz as f64;

        let ux = (ix & 0xff) as usize;
        let uy = (iy & 0xff) as usize;
        let uz = (iz & 0xff) as usize;

        let hash = |dx: usize, dy: usize, dz: usize| -> u8 {
            table[table[table[(ux + dx) & 0xff] as usize ^ ((uy + dy) & 0xff)] as usize
                ^ ((uz + dz) & 0xff)]
        };

        let h000 = hash(0, 0, 0);
        let h100 = hash(1, 0, 0);
        let h010 = hash(0, 1, 0);
        let h110 = hash(1, 1, 0);
        let h001 = hash(0, 0, 1);
        let h101 = hash(1, 0, 1);
        let h011 = hash(0, 1, 1);
        let h111 = hash(1, 1, 1);

        let g000 = grad3d_cpu(h000, fx, fy, fz);
        let g100 = grad3d_cpu(h100, fx - 1.0, fy, fz);
        let g010 = grad3d_cpu(h010, fx, fy - 1.0, fz);
        let g110 = grad3d_cpu(h110, fx - 1.0, fy - 1.0, fz);
        let g001 = grad3d_cpu(h001, fx, fy, fz - 1.0);
        let g101 = grad3d_cpu(h101, fx - 1.0, fy, fz - 1.0);
        let g011 = grad3d_cpu(h011, fx, fy - 1.0, fz - 1.0);
        let g111 = grad3d_cpu(h111, fx - 1.0, fy - 1.0, fz - 1.0);

        let cu = fade_cpu(fx);
        let cv = fade_cpu(fy);
        let cw = fade_cpu(fz);

        fn lerp(a: f64, b: f64, t: f64) -> f64 {
            a + (b - a) * t
        }

        let x0 = lerp(lerp(g000, g001, cw), lerp(g010, g011, cw), cv);
        let x1 = lerp(lerp(g100, g101, cw), lerp(g110, g111, cw), cv);
        let result = lerp(x0, x1, cu) * scale;
        result.clamp(-1.0, 1.0)
    }

    /// Earth-scale GPU/CPU parity test — regression guard for the f32
    /// catastrophic cancellation bug in `voxel_pass`.
    ///
    /// At `mean_radius ≈ 6.37e6`, absolute f32 radii only resolve to ~0.5 m,
    /// so the old code's `depth = surface_r - r` randomized the surface
    /// material classification and produced "severely fragmented" voxel
    /// terrain while the small-planet (`mean_radius = 32_000`) parity test
    /// happily passed. This test exercises the production preset parameters
    /// to ensure offsets-from-mean-radius arithmetic is used throughout.
    ///
    /// We sweep several radial layers around the surface so that at least
    /// one chunk straddles the boundary and actually exercises the
    /// `depth = surface_r - r` comparison.
    #[test]
    fn gpu_vs_cpu_parity_earth_scale_surface_chunk() {
        let _guard = GPU_TEST_LOCK.lock().unwrap();
        use crate::world::scene_presets::ScenePreset;
        use crate::world::terrain::{SphericalTerrainGenerator, UnifiedTerrainGenerator};
        use crate::world::v2::terrain_gen::generate_v2_voxels;

        let planet = ScenePreset::SphericalPlanet.planet_config();
        let config = planet
            .noise
            .as_ref()
            .expect("spherical_planet preset must carry noise config")
            .clone();
        let mean_radius = planet.mean_radius;
        let height_scale = planet.height_scale;
        let sea_level = planet.sea_level_radius;

        let Some(compute) =
            GpuVoxelCompute::try_new(&config, planet.seed, mean_radius, height_scale)
        else {
            eprintln!("skipping earth-scale parity test: no GPU adapter");
            return;
        };

        let tgen = SphericalTerrainGenerator::new(planet.clone());
        let unified = std::sync::Arc::new(UnifiedTerrainGenerator::Spherical(Box::new(tgen)));

        let fce = CubeSphereCoord::face_chunks_per_edge(mean_radius);
        let half_fce = (fce / 2.0) as i32;

        // PosX face-center direction is (1, 0, 0) → lat=0, lon=π/2.
        // Probe the actual surface at the chunk's lat/lon so the surface
        // layer is correct.
        let (chunk_lat, chunk_lon) = (0.0_f64, std::f64::consts::FRAC_PI_2);
        let surface_r = unified.sample_surface_radius_at(chunk_lat, chunk_lon);
        let surface_layer = ((surface_r - mean_radius) / CHUNK_SIZE as f64).round() as i32;
        eprintln!(
            "earth-scale: surface_r={surface_r:.1}, mean_radius={mean_radius}, \
             surface_layer={surface_layer}"
        );

        // Sweep layers around the surface. The chunk straddling the surface
        // must classify Mixed and must have matching voxel counts.
        let layer_range = -3..=3;
        let mut saw_mixed = false;
        for dl in layer_range {
            let layer = surface_layer + dl;
            let coord = CubeSphereCoord::new_with_lod(
                crate::world::v2::cubed_sphere::CubeFace::PosX,
                half_fce,
                half_fce,
                layer,
                0,
            );

            let desc = chunk_desc_from_coord(
                coord,
                mean_radius,
                fce,
                sea_level,
                planet.soil_depth,
                planet.cave_threshold,
                0,
            );
            let request = GpuChunkRequest { coord, desc };
            let gpu_result = compute.generate_batch(&[request], planet.rotation_axis);
            let gpu_td = &gpu_result.terrain_data[0];
            let cpu_td = generate_v2_voxels(coord, mean_radius, fce, &unified);

            let classification = |v: &CachedVoxels| -> &'static str {
                match v {
                    CachedVoxels::AllAir => "AllAir",
                    CachedVoxels::AllSolid(_) => "AllSolid",
                    CachedVoxels::Mixed(_) => "Mixed",
                }
            };
            let gc = classification(&gpu_td.voxels);
            let cc = classification(&cpu_td.voxels);

            let gpu_voxels = crate::world::v2::terrain_gen::cached_voxels_to_vec(&gpu_td.voxels);
            let cpu_voxels = crate::world::v2::terrain_gen::cached_voxels_to_vec(&cpu_td.voxels);
            let gpu_solid = gpu_voxels
                .iter()
                .filter(|v| v.material != MaterialId::AIR)
                .count();
            let cpu_solid = cpu_voxels
                .iter()
                .filter(|v| v.material != MaterialId::AIR)
                .count();
            let denom = CHUNK_VOLUME as f64;
            let gf = gpu_solid as f64 / denom;
            let cf = cpu_solid as f64 / denom;

            eprintln!(
                "layer={layer} (dl={dl:+}): GPU={gc} ({gpu_solid}), \
                 CPU={cc} ({cpu_solid}), diff={:.4}",
                (gf - cf).abs()
            );

            assert_eq!(
                gc, cc,
                "classification mismatch at layer {layer} (dl={dl:+})"
            );
            if gc == "Mixed" {
                saw_mixed = true;
                // At Earth scale the regression bug produced ~50% random
                // solid fraction in Mixed chunks instead of matching CPU.
                assert!(
                    (gf - cf).abs() < 0.05,
                    "solid fraction diverges at layer {layer}: \
                     GPU={gf:.4}, CPU={cf:.4} (likely f32 cancellation bug)"
                );
            }
        }

        assert!(
            saw_mixed,
            "expected at least one Mixed chunk while sweeping layers around the surface"
        );
    }
}
