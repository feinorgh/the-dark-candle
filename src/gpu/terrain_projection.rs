//! GPU-accelerated terrain projection renderer.
//!
//! Mirrors the CPU path in `planet::projections` but runs the per-pixel work
//! (inverse projection, IDW interpolation, FBM/ridged noise, hillshading) on
//! the GPU via a WGSL compute shader. Expected speedup: 50–200× for large
//! images (2048+ px).
//!
//! Uses two compute passes from the same shader module:
//! - `elevation_pass`: inverse projection → nearest cell → IDW + noise → elevation grid
//! - `hillshade_pass`: 3×3 stencil gradient → hillshade → colour modulation → RGBA

use image::RgbImage;
use std::f64::consts::{FRAC_PI_2, PI, TAU};
use wgpu;

use crate::gpu::context::GpuContext;
use crate::planet::PlanetData;
use crate::planet::detail::terrain_roughness;
use crate::planet::grid::CellId;
use crate::planet::render::{ColourMode, cell_color};

// GPU struct layouts — must match WGSL exactly, 16-byte aligned.

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuParams {
    width: u32,
    height: u32,
    projection: u32,
    colour_mode: u32,
    center_lon: f32,
    cell_size_m: f32,
    z_factor: f32,
    radius_m: f32,
    noise_seed: u32,
    noise_seed_ridged: u32,
    num_cells: u32,
    lat_bins: u32,
    lon_bins: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    sun_x: f32,
    sun_y: f32,
    sun_z: f32,
    _pad3: f32,
}

/// GPU-accelerated terrain projection renderer.
///
/// Uploads planetary data once, then renders frames by updating only the
/// uniform buffer (projection, centre longitude, colour mode).
pub struct GpuProjectionRenderer {
    ctx: GpuContext,
    elevation_pipeline: wgpu::ComputePipeline,
    hillshade_pipeline: wgpu::ComputePipeline,
    bind_group_layout_0: wgpu::BindGroupLayout,
    bind_group_layout_1: wgpu::BindGroupLayout,
    bind_group_layout_2: wgpu::BindGroupLayout,
    // Persistent data buffers.
    cell_positions_buf: wgpu::Buffer,
    cell_elevations_buf: wgpu::Buffer,
    cell_colors_buf: wgpu::Buffer,
    cell_roughness_buf: wgpu::Buffer,
    neighbor_offsets_buf: wgpu::Buffer,
    neighbor_ids_buf: wgpu::Buffer,
    spatial_bins_buf: wgpu::Buffer,
    spatial_cell_ids_buf: wgpu::Buffer,
    // Intermediate.
    elevations_buf: wgpu::Buffer,
    cell_indices_buf: wgpu::Buffer,
    // Output + staging.
    output_buf: wgpu::Buffer,
    staging_buf: wgpu::Buffer,
    // Dimensions and metadata.
    num_cells: u32,
}

impl GpuProjectionRenderer {
    /// Create a renderer and upload all planetary data to the GPU.
    pub fn new(data: &PlanetData, mode: &ColourMode, width: u32, height: u32) -> Self {
        let ctx = GpuContext::new();
        let num_cells = data.grid.cell_count() as u32;

        // ── Shader + pipelines ───────────────────────────────────────────
        let shader_source = include_str!("shaders/terrain_projection.wgsl");
        let shader_module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("terrain_projection"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        let bind_group_layout_0 =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("tp_uniforms"),
                    entries: &[bgl_uniform(0)],
                });

        let bind_group_layout_1 =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("tp_data"),
                    entries: &[
                        bgl_storage_ro(0), // cell_positions
                        bgl_storage_ro(1), // cell_elevations
                        bgl_storage_ro(2), // cell_colors
                        bgl_storage_ro(3), // cell_roughness
                        bgl_storage_ro(4), // neighbor_offsets
                        bgl_storage_ro(5), // neighbor_ids
                        bgl_storage_ro(6), // spatial_bins
                        bgl_storage_ro(7), // spatial_cell_ids
                    ],
                });

        let bind_group_layout_2 =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("tp_intermediate"),
                    entries: &[
                        bgl_storage_rw(0), // elevations
                        bgl_storage_rw(1), // cell_indices
                        bgl_storage_rw(2), // output
                    ],
                });

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("tp_layout"),
                bind_group_layouts: &[
                    &bind_group_layout_0,
                    &bind_group_layout_1,
                    &bind_group_layout_2,
                ],
                push_constant_ranges: &[],
            });

        let elevation_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("tp_elevation"),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: Some("elevation_pass"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        let hillshade_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("tp_hillshade"),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: Some("hillshade_pass"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        // ── Upload persistent data ───────────────────────────────────────

        // Cell positions: vec4<f32> per cell (xyz, w=0 padding).
        let cell_positions_buf = {
            let mut buf = Vec::with_capacity(num_cells as usize * 16);
            for i in 0..num_cells {
                let pos = data.grid.cell_position(CellId(i));
                buf.extend_from_slice(bytemuck::bytes_of(&[
                    pos.x as f32,
                    pos.y as f32,
                    pos.z as f32,
                    0.0f32,
                ]));
            }
            ctx.create_storage_buffer("tp_cell_positions", &buf)
        };

        // Cell elevations: f32 per cell.
        let cell_elevations_buf = {
            let data_f32: Vec<f32> = data.elevation.iter().map(|&e| e as f32).collect();
            ctx.create_storage_buffer("tp_cell_elevations", bytemuck::cast_slice(&data_f32))
        };

        // Cell colors: vec4<f32> per cell (pre-computed for current mode).
        let cell_colors_buf = {
            let mut buf = Vec::with_capacity(num_cells as usize * 16);
            for i in 0..num_cells as usize {
                let c = cell_color(data, i, mode);
                buf.extend_from_slice(bytemuck::bytes_of(&c));
            }
            ctx.create_storage_buffer("tp_cell_colors", &buf)
        };

        // Cell roughness: f32 per cell (pre-computed).
        let cell_roughness_buf = {
            let roughness: Vec<f32> = (0..num_cells as usize)
                .map(|i| {
                    terrain_roughness(
                        data.biome[i],
                        data.elevation[i],
                        data.volcanic_activity[i],
                        data.boundary_type[i],
                    ) as f32
                })
                .collect();
            ctx.create_storage_buffer("tp_cell_roughness", bytemuck::cast_slice(&roughness))
        };

        // Neighbor adjacency: flatten to offset+count + flat IDs.
        let (neighbor_offsets_buf, neighbor_ids_buf) = {
            let mut offsets = Vec::with_capacity(num_cells as usize * 2);
            let mut ids = Vec::new();
            for i in 0..num_cells {
                let neighbors = data.grid.cell_neighbors(CellId(i));
                offsets.push(ids.len() as u32);
                offsets.push(neighbors.len() as u32);
                ids.extend_from_slice(neighbors);
            }
            (
                ctx.create_storage_buffer("tp_nb_offsets", bytemuck::cast_slice(&offsets)),
                ctx.create_storage_buffer("tp_nb_ids", bytemuck::cast_slice(&ids)),
            )
        };

        // Spatial index: 180×360 bins → (offset, count) + flat cell IDs.
        let (spatial_bins_buf, spatial_cell_ids_buf) = {
            let lat_bins = 180usize;
            let lon_bins = 360usize;
            let mut bin_lists: Vec<Vec<u32>> = vec![Vec::new(); lat_bins * lon_bins];

            for id in data.grid.cell_ids() {
                let (lat, lon) = data.grid.cell_lat_lon(id);
                let lb = ((lat + FRAC_PI_2) / PI * lat_bins as f64)
                    .clamp(0.0, (lat_bins - 1) as f64) as usize;
                let lob =
                    ((lon + PI) / TAU * lon_bins as f64).clamp(0.0, (lon_bins - 1) as f64) as usize;
                bin_lists[lb * lon_bins + lob].push(id.0);
            }

            let mut bins = Vec::with_capacity(lat_bins * lon_bins * 2);
            let mut flat_ids = Vec::new();
            for bin in &bin_lists {
                bins.push(flat_ids.len() as u32);
                bins.push(bin.len() as u32);
                flat_ids.extend_from_slice(bin);
            }

            (
                ctx.create_storage_buffer("tp_spatial_bins", bytemuck::cast_slice(&bins)),
                ctx.create_storage_buffer(
                    "tp_spatial_cell_ids",
                    if flat_ids.is_empty() {
                        bytemuck::cast_slice(&[0u32])
                    } else {
                        bytemuck::cast_slice(&flat_ids)
                    },
                ),
            )
        };

        // ── Intermediate + output buffers ────────────────────────────────
        let pixel_count = (width * height) as u64;
        let elevations_buf = ctx.create_output_buffer("tp_elevations", pixel_count * 4);
        let cell_indices_buf = ctx.create_output_buffer("tp_cell_indices", pixel_count * 4);
        let output_buf = ctx.create_output_buffer("tp_output", pixel_count * 4);
        let staging_buf = ctx.create_staging_buffer("tp_staging", pixel_count * 4);

        Self {
            ctx,
            elevation_pipeline,
            hillshade_pipeline,
            bind_group_layout_0,
            bind_group_layout_1,
            bind_group_layout_2,
            cell_positions_buf,
            cell_elevations_buf,
            cell_colors_buf,
            cell_roughness_buf,
            neighbor_offsets_buf,
            neighbor_ids_buf,
            spatial_bins_buf,
            spatial_cell_ids_buf,
            elevations_buf,
            cell_indices_buf,
            output_buf,
            staging_buf,
            num_cells,
        }
    }

    /// Update the per-cell colour buffer (e.g. when switching colour modes).
    pub fn update_colors(&mut self, data: &PlanetData, mode: &ColourMode) {
        let mut buf = Vec::with_capacity(self.num_cells as usize * 16);
        for i in 0..self.num_cells as usize {
            let c = cell_color(data, i, mode);
            buf.extend_from_slice(bytemuck::bytes_of(&c));
        }
        self.cell_colors_buf = self.ctx.create_storage_buffer("tp_cell_colors", &buf);
    }

    /// Render a projection frame and return the result as an RGB image.
    ///
    /// `projection`: 0=equirectangular, 1=mollweide, 2=orthographic.
    /// `center_lon`: center longitude in radians (for orthographic).
    #[allow(clippy::too_many_arguments)]
    pub fn render_frame(
        &self,
        width: u32,
        height: u32,
        projection: u32,
        center_lon: f32,
        colour_mode_is_elevation: bool,
        seed: u64,
        radius_m: f64,
    ) -> RgbImage {
        let cell_size_m = (PI * radius_m / height as f64) as f32;

        // Hillshade sun direction: NW (315°), 45° altitude.
        let azimuth: f32 = 315.0_f32.to_radians();
        let altitude: f32 = 45.0_f32.to_radians();

        let noise_seed = seed.wrapping_add(7919) as u32;

        let params = GpuParams {
            width,
            height,
            projection,
            colour_mode: if colour_mode_is_elevation { 0 } else { 1 },
            center_lon,
            cell_size_m,
            z_factor: 15.0,
            radius_m: radius_m as f32,
            noise_seed,
            noise_seed_ridged: noise_seed.wrapping_add(137),
            num_cells: self.num_cells,
            lat_bins: 180,
            lon_bins: 360,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            sun_x: azimuth.sin() * altitude.cos(),
            sun_y: azimuth.cos() * altitude.cos(),
            sun_z: altitude.sin(),
            _pad3: 0.0,
        };

        let params_buf = self
            .ctx
            .create_uniform_buffer("tp_params", bytemuck::bytes_of(&params));

        // ── Bind groups ──────────────────────────────────────────────────
        let bind_group_0 = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tp_uniforms"),
                layout: &self.bind_group_layout_0,
                entries: &[bg_entry(0, &params_buf)],
            });

        let bind_group_1 = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tp_data"),
                layout: &self.bind_group_layout_1,
                entries: &[
                    bg_entry(0, &self.cell_positions_buf),
                    bg_entry(1, &self.cell_elevations_buf),
                    bg_entry(2, &self.cell_colors_buf),
                    bg_entry(3, &self.cell_roughness_buf),
                    bg_entry(4, &self.neighbor_offsets_buf),
                    bg_entry(5, &self.neighbor_ids_buf),
                    bg_entry(6, &self.spatial_bins_buf),
                    bg_entry(7, &self.spatial_cell_ids_buf),
                ],
            });

        let bind_group_2 = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tp_intermediate"),
                layout: &self.bind_group_layout_2,
                entries: &[
                    bg_entry(0, &self.elevations_buf),
                    bg_entry(1, &self.cell_indices_buf),
                    bg_entry(2, &self.output_buf),
                ],
            });

        // ── Dispatch ─────────────────────────────────────────────────────
        let wg_x = width.div_ceil(16);
        let wg_y = height.div_ceil(16);

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tp_render"),
            });

        // Pass 1: elevation.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tp_elevation"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.elevation_pipeline);
            pass.set_bind_group(0, &bind_group_0, &[]);
            pass.set_bind_group(1, &bind_group_1, &[]);
            pass.set_bind_group(2, &bind_group_2, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // Pass 2: hillshade + colour.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tp_hillshade"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.hillshade_pipeline);
            pass.set_bind_group(0, &bind_group_0, &[]);
            pass.set_bind_group(1, &bind_group_1, &[]);
            pass.set_bind_group(2, &bind_group_2, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // Copy output → staging.
        let output_size = (width * height * 4) as u64;
        encoder.copy_buffer_to_buffer(&self.output_buf, 0, &self.staging_buf, 0, output_size);

        self.ctx.submit_and_wait(encoder);

        // Read back.
        let raw = self.ctx.read_buffer(&self.staging_buf, output_size);

        // Unpack RGBA u32 → RGB image.
        let mut img = RgbImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize * 4;
                let r = raw[idx];
                let g = raw[idx + 1];
                let b = raw[idx + 2];
                img.put_pixel(x, y, image::Rgb([r, g, b]));
            }
        }

        img
    }
}

/// Render a projection using the GPU. Convenience wrapper matching the
/// signature of `planet::projections::render_projection`.
pub fn render_projection_gpu(
    data: &PlanetData,
    projection: &crate::planet::projections::Projection,
    mode: &ColourMode,
    width: u32,
) -> RgbImage {
    let height = projection.natural_height(width);

    let proj_id = match projection {
        crate::planet::projections::Projection::Equirectangular => 0u32,
        crate::planet::projections::Projection::Mollweide => 1,
        crate::planet::projections::Projection::Orthographic { .. } => 2,
    };

    let center_lon = match projection {
        crate::planet::projections::Projection::Orthographic { center_lon_deg } => {
            center_lon_deg.to_radians() as f32
        }
        _ => 0.0,
    };

    let is_elevation = matches!(mode, ColourMode::Elevation);
    let renderer = GpuProjectionRenderer::new(data, mode, width, height);

    renderer.render_frame(
        width,
        height,
        proj_id,
        center_lon,
        is_elevation,
        data.config.seed,
        data.config.radius_m,
    )
}

/// Render a rotating orthographic animation on the GPU.
pub fn render_animation_gpu(
    data: &PlanetData,
    mode: &ColourMode,
    width: u32,
    frames: u32,
    output_path: &str,
) -> Result<(), String> {
    use crate::diagnostics::video::FrameEncoder;

    let height = width; // orthographic is 1:1
    let is_elevation = matches!(mode, ColourMode::Elevation);
    let renderer = GpuProjectionRenderer::new(data, mode, width, height);
    let mut encoder = FrameEncoder::new(output_path, width, height, 30)?;

    for f in 0..frames {
        let center_lon = (std::f64::consts::TAU * f as f64 / frames as f64) as f32;
        let img = renderer.render_frame(
            width,
            height,
            2, // orthographic
            center_lon,
            is_elevation,
            data.config.seed,
            data.config.radius_m,
        );
        encoder.push_frame(&img)?;
        if f % 30 == 0 {
            println!("  [GPU] Animation frame {f}/{frames}");
        }
    }

    encoder.finish()
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

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
