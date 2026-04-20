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
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ChunkDesc {
    pub center: [f32; 4],
    pub rotation: [f32; 4],
    pub tangent_scale: [f32; 4],
    pub base_r: f32,
    pub top_r: f32,
    pub sea_level: f32,
    pub lod_scale: f32,
    pub soil_depth: f32,
    pub cave_threshold: f32,
    pub half_diag: f32,
    pub chunk_index: u32,
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

        // Bind group 1: chunk descriptors (read-only storage)
        let bgl_chunks = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("voxel_chunks_layout"),
                entries: &[bgl_storage_ro(0)],
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

        // Initialize chunk_info with sentinel values for atomic min/max.
        // min_surface → f32::MAX bits, max_surface → 0 (f32 positive zero)
        let mut info_init = vec![0u32; requests.len() * 4];
        for i in 0..requests.len() {
            info_init[i * 4] = CLASS_MIXED; // classification (default)
            info_init[i * 4 + 1] = f32::to_bits(f32::MAX); // min_surface
            info_init[i * 4 + 2] = 0; // max_surface (f32 +0.0)
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
                entries: &[bg_entry(0, &self.chunks_buffer)],
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
    let base_r = mean_radius + coord.layer as f64 * cs as f64 * lod_scale;
    let top_r = base_r + cs as f64 * lod_scale;

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
        base_r: base_r as f32,
        top_r: top_r as f32,
        sea_level: sea_level as f32,
        lod_scale: lod_scale as f32,
        soil_depth: soil_depth as f32,
        cave_threshold: cave_threshold as f32,
        half_diag: half_diag as f32,
        chunk_index,
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
}
