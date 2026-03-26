//! GPU atmosphere renderer — uploads scene data and dispatches the compute shader.

use image::RgbImage;
use wgpu;

use crate::data::MaterialRegistry;
use crate::gpu::context::GpuContext;
use crate::lighting::clouds::CloudRenderParams;
use crate::lighting::scattering::ScatteringParams;
use crate::lighting::shadows::{CloudShadowParams, FogParams};

/// All parameters needed to render a single frame.
pub struct GpuRenderParams {
    /// Camera eye position in world space.
    pub eye: [f32; 3],
    /// Camera look-at target in world space.
    pub target: [f32; 3],
    /// Vertical field of view in degrees.
    pub fov_degrees: f32,
    /// Unit vector TOWARD the sun.
    pub sun_direction: [f32; 3],
    /// Sun light color (linear RGB).
    pub sun_color: [f32; 3],
    /// Sun intensity multiplier.
    pub sun_intensity: f32,
    /// Sun elevation in radians.
    pub sun_elevation: f32,
    /// Ambient light floor.
    pub ambient: f32,
    /// Atmospheric scattering parameters.
    pub scatter_params: ScatteringParams,
    /// Cloud rendering parameters.
    pub cloud_params: CloudRenderParams,
    /// Cloud shadow parameters.
    pub shadow_params: CloudShadowParams,
    /// Fog parameters.
    pub fog_params: FogParams,
    /// Whether to render clouds.
    pub enable_clouds: bool,
    /// Whether to render fog.
    pub enable_fog: bool,
    /// Whether to apply cloud shadows.
    pub enable_shadows: bool,
    /// Whether to render stars.
    pub enable_stars: bool,
    /// Hash seed for star field variation.
    pub time_hash: u32,
}

impl Default for GpuRenderParams {
    fn default() -> Self {
        Self {
            eye: [16.0, 20.0, 16.0],
            target: [16.0, 0.0, 16.0],
            fov_degrees: 60.0,
            sun_direction: [0.0, 1.0, 0.0],
            sun_color: [1.0, 1.0, 1.0],
            sun_intensity: 1.0,
            sun_elevation: 1.0,
            ambient: 0.15,
            scatter_params: ScatteringParams::default(),
            cloud_params: CloudRenderParams::default(),
            shadow_params: CloudShadowParams::default(),
            fog_params: FogParams::default(),
            enable_clouds: false,
            enable_fog: false,
            enable_shadows: false,
            enable_stars: false,
            time_hash: 0,
        }
    }
}

/// GPU-accelerated atmosphere renderer.
///
/// Uploads voxel/cloud/fog data once, then renders frames by updating only
/// uniforms (camera, sun position) per frame.
pub struct GpuRenderer {
    ctx: GpuContext,
    width: u32,
    height: u32,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout_0: wgpu::BindGroupLayout,
    bind_group_layout_1: wgpu::BindGroupLayout,
    bind_group_layout_2: wgpu::BindGroupLayout,
    // Persistent data buffers.
    voxel_buffer: wgpu::Buffer,
    material_buffer: wgpu::Buffer,
    cloud_buffer: wgpu::Buffer,
    shadow_buffer: wgpu::Buffer,
    humidity_buffer: wgpu::Buffer,
    temperature_buffer: wgpu::Buffer,
    // Output.
    output_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    // Grid metadata.
    grid_size: u32,
}

// GPU struct layouts (must match WGSL exactly, 16-byte aligned).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuCamera {
    eye: [f32; 3],
    _pad0: f32,
    forward: [f32; 3],
    _pad1: f32,
    right: [f32; 3],
    _pad2: f32,
    up: [f32; 3],
    _pad3: f32,
    half_w: f32,
    half_h: f32,
    width: u32,
    height: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuSun {
    direction: [f32; 3],
    _pad0: f32,
    color: [f32; 3],
    intensity: f32,
    elevation: f32,
    ambient: f32,
    _pad1: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuScatterParams {
    rayleigh_coeff: [f32; 3],
    mie_coeff: f32,
    mie_g: f32,
    atmosphere_radius: f32,
    planet_radius: f32,
    rayleigh_scale_height: f32,
    mie_scale_height: f32,
    sun_intensity: f32,
    _pad0: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuCloudParams {
    extinction_coeff: f32,
    scattering_albedo: f32,
    forward_scatter_g: f32,
    ambient_factor: f32,
    max_march_distance: f32,
    step_size: f32,
    density_threshold: f32,
    _pad0: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuShadowParams {
    sun_direction: [f32; 3],
    shadow_softness: f32,
    min_shadow_factor: f32,
    extinction_coeff: f32,
    _pad0: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuFogParams {
    fog_density_base: f32,
    fog_height_falloff: f32,
    humidity_scale: f32,
    temperature_factor: f32,
    fog_color: [f32; 3],
    max_fog_distance: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuGridInfo {
    grid_size: u32,
    num_materials: u32,
    enable_clouds: u32,
    enable_fog: u32,
    enable_shadows: u32,
    enable_stars: u32,
    time_hash: u32,
    _pad0: u32,
}

impl GpuRenderer {
    /// Create a new GPU renderer for the given output dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        let ctx = GpuContext::new();

        let shader_source = include_str!("shaders/atmosphere_render.wgsl");
        let shader_module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("atmosphere_render"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        // Bind group layout 0: uniforms.
        let bind_group_layout_0 =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("uniforms_layout"),
                    entries: &[
                        bgl_uniform(0), // Camera
                        bgl_uniform(1), // Sun
                        bgl_uniform(2), // ScatterParams
                        bgl_uniform(3), // CloudParams
                        bgl_uniform(4), // ShadowParams
                        bgl_uniform(5), // FogParams
                        bgl_uniform(6), // GridInfo
                    ],
                });

        // Bind group layout 1: storage buffers (read-only data).
        let bind_group_layout_1 =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("data_layout"),
                    entries: &[
                        bgl_storage_ro(0), // voxels
                        bgl_storage_ro(1), // materials
                        bgl_storage_ro(2), // cloud_field
                        bgl_storage_ro(3), // shadow_map
                        bgl_storage_ro(4), // humidity
                        bgl_storage_ro(5), // temperature
                    ],
                });

        // Bind group layout 2: output buffer (read-write).
        let bind_group_layout_2 =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("output_layout"),
                    entries: &[bgl_storage_rw(0)],
                });

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("atmosphere_render_layout"),
                bind_group_layouts: &[
                    &bind_group_layout_0,
                    &bind_group_layout_1,
                    &bind_group_layout_2,
                ],
                push_constant_ranges: &[],
            });

        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("atmosphere_render_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let output_size = (width * height * 4) as u64; // RGBA u8 packed as u32.
        let output_buffer = ctx.create_output_buffer("output", output_size);
        let staging_buffer = ctx.create_staging_buffer("staging", output_size);

        // Create dummy data buffers (will be replaced by upload calls).
        let dummy_4 = [0u8; 4];
        let voxel_buffer = ctx.create_storage_buffer("voxels", &dummy_4);
        let material_buffer = ctx.create_storage_buffer("materials", &dummy_4);
        let cloud_buffer = ctx.create_storage_buffer("cloud_field", &dummy_4);
        let shadow_buffer = ctx.create_storage_buffer("shadow_map", &dummy_4);
        let humidity_buffer = ctx.create_storage_buffer("humidity", &dummy_4);
        let temperature_buffer = ctx.create_storage_buffer("temperature", &dummy_4);

        Self {
            ctx,
            width,
            height,
            pipeline,
            bind_group_layout_0,
            bind_group_layout_1,
            bind_group_layout_2,
            voxel_buffer,
            material_buffer,
            cloud_buffer,
            shadow_buffer,
            humidity_buffer,
            temperature_buffer,
            output_buffer,
            staging_buffer,
            grid_size: 0,
        }
    }

    /// Upload voxel grid data. Packs each voxel as [material_id: u32, temperature: u32].
    pub fn upload_voxels(&mut self, voxels: &[crate::world::voxel::Voxel], size: usize) {
        self.grid_size = size as u32;

        let mut data = Vec::with_capacity(voxels.len() * 8);
        for v in voxels {
            data.extend_from_slice(&(v.material.0 as u32).to_le_bytes());
            data.extend_from_slice(&v.temperature.to_le_bytes());
        }

        self.voxel_buffer = self.ctx.create_storage_buffer("voxels", &data);
    }

    /// Upload material color table. `max_material_id` is the highest MaterialId in use.
    pub fn upload_materials(&mut self, registry: &MaterialRegistry, max_material_id: u16) {
        let count = max_material_id as usize + 1;

        // Pack as vec4<f32> per material: [r, g, b, transparent_flag].
        let mut data = vec![0.0f32; count * 4];
        for id in 0..count {
            let mat_id = crate::world::voxel::MaterialId(id as u16);
            if let Some(mat) = registry.get(mat_id) {
                let base = id * 4;
                data[base] = mat.color[0];
                data[base + 1] = mat.color[1];
                data[base + 2] = mat.color[2];
                data[base + 3] = if mat.transparent { 1.0 } else { 0.0 };
            }
        }

        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        self.material_buffer = self.ctx.create_storage_buffer("materials", &bytes);
    }

    /// Upload cloud LWC field (flat f32 array, size³ elements).
    pub fn upload_cloud_field(&mut self, cloud_data: &[f32]) {
        let bytes: Vec<u8> = cloud_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        self.cloud_buffer = self.ctx.create_storage_buffer("cloud_field", &bytes);
    }

    /// Upload precomputed cloud shadow map (flat f32 array, size×size).
    pub fn upload_shadow_map(&mut self, shadow_data: &[f32]) {
        let bytes: Vec<u8> = shadow_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        self.shadow_buffer = self.ctx.create_storage_buffer("shadow_map", &bytes);
    }

    /// Upload humidity field for fog computation.
    pub fn upload_humidity(&mut self, humidity: &[f32]) {
        let bytes: Vec<u8> = humidity.iter().flat_map(|f| f.to_le_bytes()).collect();
        self.humidity_buffer = self.ctx.create_storage_buffer("humidity", &bytes);
    }

    /// Upload temperature field for fog computation.
    pub fn upload_temperature(&mut self, temperature: &[f32]) {
        let bytes: Vec<u8> = temperature.iter().flat_map(|f| f.to_le_bytes()).collect();
        self.temperature_buffer = self.ctx.create_storage_buffer("temperature", &bytes);
    }

    /// Render a single frame and return the result as an RGB image.
    pub fn render_frame(&self, params: &GpuRenderParams) -> RgbImage {
        // Build camera basis.
        let eye = params.eye;
        let target = params.target;
        let fwd = normalize_arr(sub_arr(target, eye));
        let world_up = [0.0, 1.0, 0.0];
        let right = normalize_arr(cross_arr(fwd, world_up));
        let up = normalize_arr(cross_arr(right, fwd));

        let fov_rad = params.fov_degrees.to_radians();
        let half_h = (fov_rad / 2.0).tan();
        let aspect = self.width as f32 / self.height as f32;
        let half_w = half_h * aspect;

        let camera = GpuCamera {
            eye,
            _pad0: 0.0,
            forward: fwd,
            _pad1: 0.0,
            right,
            _pad2: 0.0,
            up,
            _pad3: 0.0,
            half_w,
            half_h,
            width: self.width,
            height: self.height,
        };

        let sun_data = GpuSun {
            direction: params.sun_direction,
            _pad0: 0.0,
            color: params.sun_color,
            intensity: params.sun_intensity,
            elevation: params.sun_elevation,
            ambient: params.ambient,
            _pad1: [0.0; 2],
        };

        let sp = &params.scatter_params;
        let scatter = GpuScatterParams {
            rayleigh_coeff: sp.rayleigh_coeff,
            mie_coeff: sp.mie_coeff,
            mie_g: sp.mie_g,
            atmosphere_radius: sp.atmosphere_radius,
            planet_radius: sp.planet_radius,
            rayleigh_scale_height: sp.rayleigh_scale_height,
            mie_scale_height: sp.mie_scale_height,
            sun_intensity: sp.sun_intensity,
            _pad0: [0.0; 2],
        };

        let cp = &params.cloud_params;
        let clouds = GpuCloudParams {
            extinction_coeff: cp.extinction_coeff,
            scattering_albedo: cp.scattering_albedo,
            forward_scatter_g: cp.forward_scatter_g,
            ambient_factor: cp.ambient_factor,
            max_march_distance: cp.max_march_distance,
            step_size: cp.step_size,
            density_threshold: cp.density_threshold,
            _pad0: 0.0,
        };

        let shp = &params.shadow_params;
        let shadows = GpuShadowParams {
            sun_direction: shp.sun_direction,
            shadow_softness: shp.shadow_softness,
            min_shadow_factor: shp.min_shadow_factor,
            extinction_coeff: shp.extinction_coeff,
            _pad0: [0.0; 2],
        };

        let fp = &params.fog_params;
        let fog = GpuFogParams {
            fog_density_base: fp.fog_density_base,
            fog_height_falloff: fp.fog_height_falloff,
            humidity_scale: fp.humidity_scale,
            temperature_factor: fp.temperature_factor,
            fog_color: fp.fog_color,
            max_fog_distance: fp.max_fog_distance,
        };

        let grid = GpuGridInfo {
            grid_size: self.grid_size,
            num_materials: 0,
            enable_clouds: u32::from(params.enable_clouds),
            enable_fog: u32::from(params.enable_fog),
            enable_shadows: u32::from(params.enable_shadows),
            enable_stars: u32::from(params.enable_stars),
            time_hash: params.time_hash,
            _pad0: 0,
        };

        // Create uniform buffers.
        let camera_buf = self
            .ctx
            .create_uniform_buffer("camera", bytemuck::bytes_of(&camera));
        let sun_buf = self
            .ctx
            .create_uniform_buffer("sun", bytemuck::bytes_of(&sun_data));
        let scatter_buf = self
            .ctx
            .create_uniform_buffer("scatter", bytemuck::bytes_of(&scatter));
        let cloud_buf = self
            .ctx
            .create_uniform_buffer("clouds", bytemuck::bytes_of(&clouds));
        let shadow_buf = self
            .ctx
            .create_uniform_buffer("shadows", bytemuck::bytes_of(&shadows));
        let fog_buf = self
            .ctx
            .create_uniform_buffer("fog", bytemuck::bytes_of(&fog));
        let grid_buf = self
            .ctx
            .create_uniform_buffer("grid", bytemuck::bytes_of(&grid));

        // Bind group 0: uniforms.
        let bind_group_0 = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("uniforms"),
                layout: &self.bind_group_layout_0,
                entries: &[
                    bg_entry(0, &camera_buf),
                    bg_entry(1, &sun_buf),
                    bg_entry(2, &scatter_buf),
                    bg_entry(3, &cloud_buf),
                    bg_entry(4, &shadow_buf),
                    bg_entry(5, &fog_buf),
                    bg_entry(6, &grid_buf),
                ],
            });

        // Bind group 1: data.
        let bind_group_1 = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("data"),
                layout: &self.bind_group_layout_1,
                entries: &[
                    bg_entry(0, &self.voxel_buffer),
                    bg_entry(1, &self.material_buffer),
                    bg_entry(2, &self.cloud_buffer),
                    bg_entry(3, &self.shadow_buffer),
                    bg_entry(4, &self.humidity_buffer),
                    bg_entry(5, &self.temperature_buffer),
                ],
            });

        // Bind group 2: output.
        let bind_group_2 = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("output"),
                layout: &self.bind_group_layout_2,
                entries: &[bg_entry(0, &self.output_buffer)],
            });

        // Dispatch compute.
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render_frame"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("atmosphere_render"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group_0, &[]);
            pass.set_bind_group(1, &bind_group_1, &[]);
            pass.set_bind_group(2, &bind_group_2, &[]);

            let wg_x = self.width.div_ceil(16);
            let wg_y = self.height.div_ceil(16);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // Copy output to staging buffer.
        let output_size = (self.width * self.height * 4) as u64;
        encoder.copy_buffer_to_buffer(&self.output_buffer, 0, &self.staging_buffer, 0, output_size);

        self.ctx.submit_and_wait(encoder);

        // Read back.
        let data = self.ctx.read_buffer(&self.staging_buffer, output_size);

        // Convert packed RGBA u32 to RGB image.
        let mut img = RgbImage::new(self.width, self.height);
        for y in 0..self.height {
            for x in 0..self.width {
                let idx = (y * self.width + x) as usize * 4;
                let r = data[idx];
                let g = data[idx + 1];
                let b = data[idx + 2];
                img.put_pixel(x, y, image::Rgb([r, g, b]));
            }
        }

        img
    }
}

// ─── Helper functions ──────────────────────────────────────────────────────

fn sub_arr(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross_arr(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize_arr(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-10 {
        return [0.0, 0.0, 0.0];
    }
    [v[0] / len, v[1] / len, v[2] / len]
}

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
