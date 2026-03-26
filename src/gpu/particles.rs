//! GPU compute pipeline for weather particle simulation.
//!
//! Uses wgpu directly (not Bevy's render pipeline) to dispatch a compute shader
//! that integrates particle physics: gravity, aerodynamic drag, and wind forces.
//! Supports rain, snow, sand, and hail particle types with real SI parameters.

use wgpu;

use crate::gpu::context::GpuContext;

// ─── Constants ─────────────────────────────────────────────────────────────

/// Maximum number of particles the GPU buffer can hold.
pub const MAX_PARTICLES: usize = 100_000;

/// Bytes per particle (must match WGSL struct layout).
const PARTICLE_SIZE: usize = 64;

/// Compute shader workgroup size (x dimension).
const WORKGROUP_SIZE: u32 = 256;

// ─── GPU data structures ───────────────────────────────────────────────────

/// Per-particle data transferred to/from the GPU.
///
/// Layout matches the WGSL `GpuParticle` struct exactly (64 bytes).  The
/// `_pad` field accounts for WGSL's 16-byte `vec3<f32>` alignment gaps
/// between `mass` and the trailing `_pad` member plus struct tail padding.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuParticle {
    /// World position in meters.
    pub position: [f32; 3],
    /// Remaining lifetime in seconds (0 = dead).
    pub life: f32,
    /// Velocity in m/s.
    pub velocity: [f32; 3],
    /// Particle kind: 0=rain, 1=snow, 2=sand, 3=hail.
    pub kind: u32,
    /// Mass in kg.
    pub mass: f32,
    /// Padding to 64 bytes (covers WGSL vec3 alignment gap + struct tail).
    pub _pad: [f32; 7],
}

/// Uniform parameters for the particle compute shader.
///
/// Layout matches the WGSL `ParticleParams` struct (32 bytes).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ParticleParams {
    /// Simulation timestep in seconds.
    pub dt: f32,
    /// Number of active particles.
    pub particle_count: u32,
    /// Gravitational acceleration in m/s² (standard: 9.80665).
    pub gravity: f32,
    /// Wind field grid dimension (cubic, cells per axis). 0 = no grid.
    pub grid_size: u32,
    /// Global wind velocity fallback in m/s (used when `grid_size == 0`).
    pub wind_global: [f32; 3],
    /// Grid cell size in meters (1.0 = 1 voxel = 1 meter).
    pub cell_size: f32,
}

/// Particle kind constants.
pub mod kind {
    pub const RAIN: u32 = 0;
    pub const SNOW: u32 = 1;
    pub const SAND: u32 = 2;
    pub const HAIL: u32 = 3;
}

// ─── Pipeline ──────────────────────────────────────────────────────────────

/// GPU compute pipeline for weather particle physics.
///
/// Manages wgpu buffers and dispatches the `particle_step.wgsl` compute shader
/// to integrate particle positions and velocities each simulation tick.
pub struct ParticleCompute {
    ctx: GpuContext,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout_0: wgpu::BindGroupLayout,
    bind_group_layout_1: wgpu::BindGroupLayout,
    bind_group_layout_2: wgpu::BindGroupLayout,
    particle_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    wind_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    particle_count: u32,
}

impl ParticleCompute {
    /// Create a new particle compute pipeline.
    ///
    /// `grid_size` sets the cubic dimension of the wind velocity field
    /// (e.g. 32 for a 32³ grid).  Pass 0 to use only the global wind vector.
    pub fn new(ctx: GpuContext, grid_size: u32) -> Self {
        let shader_source = include_str!("shaders/particle_step.wgsl");
        let shader_module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("particle_step"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        // Bind group 0: uniform params.
        let bind_group_layout_0 =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("particle_params_layout"),
                    entries: &[bgl_uniform(0)],
                });

        // Bind group 1: wind field (read-only storage).
        let bind_group_layout_1 =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("wind_field_layout"),
                    entries: &[bgl_storage_ro(0)],
                });

        // Bind group 2: particle buffer (read-write storage).
        let bind_group_layout_2 =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("particle_buffer_layout"),
                    entries: &[bgl_storage_rw(0)],
                });

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("particle_pipeline_layout"),
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
                label: Some("particle_step_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Pre-allocate particle buffer at maximum capacity.
        let particle_buf_size = (MAX_PARTICLES * PARTICLE_SIZE) as u64;
        let particle_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particles"),
            size: particle_buf_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_buffer = ctx.create_staging_buffer("particle_staging", particle_buf_size);

        // Uniform buffer (updated each step via write_buffer).
        let params_data = ParticleParams {
            dt: 0.0,
            particle_count: 0,
            gravity: 0.0,
            grid_size: 0,
            wind_global: [0.0; 3],
            cell_size: 1.0,
        };
        let params_buffer =
            ctx.create_uniform_buffer("particle_params", bytemuck::bytes_of(&params_data));

        // Wind field: grid_size³ vec4<f32> elements, or a single dummy element.
        let wind_elements = if grid_size > 0 {
            (grid_size as usize).pow(3)
        } else {
            1
        };
        let wind_data = vec![[0.0f32; 4]; wind_elements];
        let wind_bytes = bytemuck::cast_slice::<[f32; 4], u8>(&wind_data);
        let wind_buffer = ctx.create_storage_buffer("wind_field", wind_bytes);

        Self {
            ctx,
            pipeline,
            bind_group_layout_0,
            bind_group_layout_1,
            bind_group_layout_2,
            particle_buffer,
            params_buffer,
            wind_buffer,
            staging_buffer,
            particle_count: 0,
        }
    }

    /// Upload particle data to the GPU.
    ///
    /// # Panics
    ///
    /// Panics if `particles.len()` exceeds [`MAX_PARTICLES`].
    pub fn upload_particles(&mut self, particles: &[GpuParticle]) {
        assert!(
            particles.len() <= MAX_PARTICLES,
            "particle count {} exceeds MAX_PARTICLES {MAX_PARTICLES}",
            particles.len(),
        );
        self.particle_count = particles.len() as u32;
        if !particles.is_empty() {
            self.ctx
                .queue
                .write_buffer(&self.particle_buffer, 0, bytemuck::cast_slice(particles));
        }
    }

    /// Upload the wind velocity field.
    ///
    /// Each element is `[vx, vy, vz, density]` laid out as a flat 3D array
    /// indexed `z × size² + y × size + x`.
    pub fn upload_wind_field(&mut self, wind: &[[f32; 4]]) {
        let bytes = bytemuck::cast_slice::<[f32; 4], u8>(wind);
        self.wind_buffer = self.ctx.create_storage_buffer("wind_field", bytes);
    }

    /// Dispatch the compute shader for one simulation step.
    pub fn step(&self, params: &ParticleParams) {
        self.ctx
            .queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));

        let bind_group_0 = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("particle_params_bg"),
                layout: &self.bind_group_layout_0,
                entries: &[bg_entry(0, &self.params_buffer)],
            });

        let bind_group_1 = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("wind_field_bg"),
                layout: &self.bind_group_layout_1,
                entries: &[bg_entry(0, &self.wind_buffer)],
            });

        let bind_group_2 = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("particle_buffer_bg"),
                layout: &self.bind_group_layout_2,
                entries: &[bg_entry(0, &self.particle_buffer)],
            });

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("particle_step"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("particle_step"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group_0, &[]);
            pass.set_bind_group(1, &bind_group_1, &[]);
            pass.set_bind_group(2, &bind_group_2, &[]);

            let workgroups = self.particle_count.div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Copy particle data to staging for CPU readback.
        let copy_size = self.particle_count as u64 * PARTICLE_SIZE as u64;
        if copy_size > 0 {
            encoder.copy_buffer_to_buffer(
                &self.particle_buffer,
                0,
                &self.staging_buffer,
                0,
                copy_size,
            );
        }

        self.ctx.submit_and_wait(encoder);
    }

    /// Read particle data back from the GPU after the most recent [`step`](Self::step).
    pub fn read_particles(&self) -> Vec<GpuParticle> {
        let byte_count = self.particle_count as u64 * PARTICLE_SIZE as u64;
        if byte_count == 0 {
            return Vec::new();
        }
        let raw = self.ctx.read_buffer(&self.staging_buffer, byte_count);
        bytemuck::cast_slice(&raw).to_vec()
    }

    /// Number of particles currently uploaded.
    pub fn particle_count(&self) -> u32 {
        self.particle_count
    }
}

// ─── Bind group helpers (same pattern as renderer.rs) ──────────────────────

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

    /// Attempt to create a GPU context; returns `None` if no GPU is available.
    fn try_gpu_context() -> Option<GpuContext> {
        std::panic::catch_unwind(GpuContext::new).ok()
    }

    fn default_params(count: u32) -> ParticleParams {
        ParticleParams {
            dt: 1.0 / 60.0,
            particle_count: count,
            gravity: 9.80665,
            grid_size: 0,
            wind_global: [0.0; 3],
            cell_size: 1.0,
        }
    }

    fn make_particle(particle_kind: u32, mass: f32) -> GpuParticle {
        GpuParticle {
            position: [16.0, 100.0, 16.0],
            life: 10.0,
            velocity: [0.0; 3],
            kind: particle_kind,
            mass,
            _pad: [0.0; 7],
        }
    }

    #[test]
    fn particle_struct_size() {
        assert_eq!(std::mem::size_of::<GpuParticle>(), 64);
    }

    #[test]
    fn params_struct_size() {
        assert_eq!(std::mem::size_of::<ParticleParams>(), 32);
    }

    #[test]
    fn dead_particles_stay_dead() {
        let Some(ctx) = try_gpu_context() else {
            eprintln!("skipping GPU test: no adapter");
            return;
        };
        let mut compute = ParticleCompute::new(ctx, 0);

        let dead = GpuParticle {
            position: [5.0, 50.0, 5.0],
            life: 0.0,
            velocity: [1.0, 2.0, 3.0],
            kind: kind::RAIN,
            mass: 3.35e-5,
            _pad: [0.0; 7],
        };
        compute.upload_particles(&[dead]);

        let params = default_params(1);
        compute.step(&params);

        let result = compute.read_particles();
        assert_eq!(result.len(), 1);
        let p = &result[0];
        assert_eq!(p.position, [5.0, 50.0, 5.0], "dead particle must not move");
        assert_eq!(
            p.velocity,
            [1.0, 2.0, 3.0],
            "dead particle velocity unchanged"
        );
        assert!(p.life <= 0.0);
    }

    #[test]
    fn gravity_accelerates_downward() {
        let Some(ctx) = try_gpu_context() else {
            eprintln!("skipping GPU test: no adapter");
            return;
        };
        let mut compute = ParticleCompute::new(ctx, 0);

        let rain = make_particle(kind::RAIN, 3.35e-5);
        compute.upload_particles(&[rain]);

        let params = default_params(1);
        compute.step(&params);

        let result = compute.read_particles();
        assert_eq!(result.len(), 1);
        assert!(
            result[0].velocity[1] < 0.0,
            "expected negative y velocity after gravity, got {}",
            result[0].velocity[1],
        );
    }

    #[test]
    fn rain_falls_faster_than_snow() {
        let Some(ctx) = try_gpu_context() else {
            eprintln!("skipping GPU test: no adapter");
            return;
        };
        let mut compute = ParticleCompute::new(ctx, 0);

        let rain = make_particle(kind::RAIN, 3.35e-5);
        let snow = make_particle(kind::SNOW, 3.0e-6);
        compute.upload_particles(&[rain, snow]);

        let params = default_params(2);
        // Run enough steps for drag to differentiate the two kinds.
        for _ in 0..100 {
            compute.step(&params);
        }

        let result = compute.read_particles();
        assert_eq!(result.len(), 2);

        let rain_vy = result[0].velocity[1];
        let snow_vy = result[1].velocity[1];

        assert!(rain_vy < 0.0, "rain should be falling, vy={rain_vy}");
        assert!(snow_vy < 0.0, "snow should be falling, vy={snow_vy}");
        assert!(
            rain_vy.abs() > snow_vy.abs(),
            "rain |vy|={} should exceed snow |vy|={}",
            rain_vy.abs(),
            snow_vy.abs(),
        );
    }
}
