//! GPU compute pipeline for batched terrain noise evaluation.
//!
//! Offloads the `NoiseStack::sample()` pipeline to the GPU, processing
//! up to 8192 columns (8 chunks × 1024 columns) per dispatch.  Produces
//! surface-radius values identical in character (but not bit-identical) to
//! the CPU path, since the GPU uses hash-based f32 Perlin while the CPU
//! uses permutation-table f64 Perlin.
//!
//! Falls back gracefully when no GPU is available (`GpuContext::try_new()`
//! returns `None`).

use wgpu;

use crate::gpu::context::GpuContext;
use crate::world::noise::NoiseConfig;

// ─── Constants ─────────────────────────────────────────────────────────────

/// Maximum columns per GPU dispatch (32 chunks × 32 × 32).
pub const MAX_COLUMNS: usize = 32_768;

/// Bytes per column input (vec2<f32> = 8 bytes).
const COLUMN_SIZE: usize = 8;

/// Bytes per output value (f32 = 4 bytes).
const OUTPUT_SIZE: usize = 4;

/// Compute shader workgroup size (must match noise_eval.wgsl).
const WORKGROUP_SIZE: u32 = 256;

// ─── GPU data structures ───────────────────────────────────────────────────

/// Uniform parameters for the noise compute shader.
///
/// Layout matches the WGSL `NoiseParams` struct (96 bytes, 16-byte aligned).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NoiseParams {
    // FBM
    pub fbm_octaves: u32,
    pub fbm_persistence: f32,
    pub fbm_lacunarity: f32,
    pub fbm_base_freq: f32,
    // Ridged
    pub ridged_octaves: u32,
    pub ridged_gain: f32,
    pub ridged_base_freq: f32,
    // Selector
    pub selector_freq: f32,
    pub selector_lo: f32,
    pub selector_hi: f32,
    // Warp
    pub warp_strength: f32,
    pub warp_freq: f32,
    // Micro-detail
    pub micro_freq: f32,
    pub micro_amplitude: f32,
    // Continent mask
    pub continent_enabled: u32,
    pub continent_freq: f32,
    pub continent_threshold: f32,
    pub shelf_blend_width: f32,
    pub ocean_floor_depth: f32,
    pub ocean_floor_amplitude: f32,
    // Pipeline params
    pub seed: u32,
    pub column_count: u32,
    pub mean_radius: f32,
    pub height_scale: f32,
}

impl NoiseParams {
    /// Build from a `NoiseConfig` and planet parameters.
    pub fn from_config(
        config: &NoiseConfig,
        seed: u32,
        mean_radius: f64,
        height_scale: f64,
    ) -> Self {
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
            column_count: 0,
            mean_radius: mean_radius as f32,
            height_scale: height_scale as f32,
        }
    }
}

// ─── Pipeline ──────────────────────────────────────────────────────────────

/// GPU compute pipeline for batched terrain noise evaluation.
///
/// Evaluates `NoiseStack::sample()` on the GPU for arrays of (lon, lat) columns,
/// returning surface-radius values.  Persistent buffers are reused across frames.
pub struct GpuNoiseCompute {
    ctx: GpuContext,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout_0: wgpu::BindGroupLayout,
    bind_group_layout_1: wgpu::BindGroupLayout,
    bind_group_layout_2: wgpu::BindGroupLayout,
    column_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    /// Base noise parameters (seed + config, column_count updated per dispatch).
    base_params: NoiseParams,
}

impl GpuNoiseCompute {
    /// Create a new noise compute pipeline.
    ///
    /// `config` provides the noise parameters, `seed` the terrain seed,
    /// `mean_radius` and `height_scale` the planet geometry.
    pub fn new(
        ctx: GpuContext,
        config: &NoiseConfig,
        seed: u32,
        mean_radius: f64,
        height_scale: f64,
    ) -> Self {
        let shader_source = include_str!("shaders/noise_eval.wgsl");
        let shader_module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("noise_eval"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        // Bind group 0: uniform params.
        let bind_group_layout_0 =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("noise_params_layout"),
                    entries: &[bgl_uniform(0)],
                });

        // Bind group 1: column input (read-only storage).
        let bind_group_layout_1 =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("noise_columns_layout"),
                    entries: &[bgl_storage_ro(0)],
                });

        // Bind group 2: output heights (read-write storage).
        let bind_group_layout_2 =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("noise_output_layout"),
                    entries: &[bgl_storage_rw(0)],
                });

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("noise_pipeline_layout"),
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
                label: Some("noise_eval_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Pre-allocate buffers at maximum capacity.
        let column_buf_size = (MAX_COLUMNS * COLUMN_SIZE) as u64;
        let output_buf_size = (MAX_COLUMNS * OUTPUT_SIZE) as u64;

        let column_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("noise_columns"),
            size: column_buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let output_buffer = ctx.create_output_buffer("noise_output", output_buf_size);
        let staging_buffer = ctx.create_staging_buffer("noise_staging", output_buf_size);

        let base_params = NoiseParams::from_config(config, seed, mean_radius, height_scale);
        let params_buffer =
            ctx.create_uniform_buffer("noise_params", bytemuck::bytes_of(&base_params));

        Self {
            ctx,
            pipeline,
            bind_group_layout_0,
            bind_group_layout_1,
            bind_group_layout_2,
            column_buffer,
            params_buffer,
            output_buffer,
            staging_buffer,
            base_params,
        }
    }

    /// Evaluate terrain noise for a batch of columns.
    ///
    /// `columns` is a slice of `[lon, lat]` pairs (f32).
    /// Returns a `Vec<f32>` of surface-radius values, one per column.
    ///
    /// # Panics
    ///
    /// Panics if `columns.len()` exceeds [`MAX_COLUMNS`].
    pub fn evaluate_batch(&self, columns: &[[f32; 2]]) -> Vec<f32> {
        assert!(
            columns.len() <= MAX_COLUMNS,
            "column count {} exceeds MAX_COLUMNS {MAX_COLUMNS}",
            columns.len(),
        );

        if columns.is_empty() {
            return Vec::new();
        }

        let column_count = columns.len() as u32;

        // Update params with actual column count.
        let mut params = self.base_params;
        params.column_count = column_count;
        self.ctx
            .queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

        // Upload column data.
        self.ctx
            .queue
            .write_buffer(&self.column_buffer, 0, bytemuck::cast_slice(columns));

        // Build bind groups.
        let bind_group_0 = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("noise_params_bg"),
                layout: &self.bind_group_layout_0,
                entries: &[bg_entry(0, &self.params_buffer)],
            });

        let bind_group_1 = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("noise_columns_bg"),
                layout: &self.bind_group_layout_1,
                entries: &[bg_entry(0, &self.column_buffer)],
            });

        let bind_group_2 = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("noise_output_bg"),
                layout: &self.bind_group_layout_2,
                entries: &[bg_entry(0, &self.output_buffer)],
            });

        // Encode and dispatch.
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("noise_eval"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("noise_eval"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group_0, &[]);
            pass.set_bind_group(1, &bind_group_1, &[]);
            pass.set_bind_group(2, &bind_group_2, &[]);

            let workgroups = column_count.div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Copy output to staging for CPU readback.
        let copy_size = column_count as u64 * OUTPUT_SIZE as u64;
        encoder.copy_buffer_to_buffer(&self.output_buffer, 0, &self.staging_buffer, 0, copy_size);

        self.ctx.submit_and_wait(encoder);

        // Readback.
        let raw = self.ctx.read_buffer(&self.staging_buffer, copy_size);
        bytemuck::cast_slice(&raw).to_vec()
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

    fn try_gpu_context() -> Option<GpuContext> {
        GpuContext::try_new()
    }

    #[test]
    fn noise_params_size() {
        assert_eq!(std::mem::size_of::<NoiseParams>(), 96);
    }

    #[test]
    fn noise_params_from_config() {
        let config = NoiseConfig::default();
        let params = NoiseParams::from_config(&config, 42, 32000.0, 4000.0);
        assert_eq!(params.fbm_octaves, 6);
        assert_eq!(params.seed, 42);
        assert!((params.mean_radius - 32000.0).abs() < 0.1);
        assert!((params.height_scale - 4000.0).abs() < 0.1);
    }

    #[test]
    fn empty_batch_returns_empty() {
        let Some(ctx) = try_gpu_context() else {
            eprintln!("skipping GPU test: no adapter");
            return;
        };
        let config = NoiseConfig::default();
        let compute = GpuNoiseCompute::new(ctx, &config, 42, 32000.0, 4000.0);
        let result = compute.evaluate_batch(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn single_column_produces_valid_radius() {
        let Some(ctx) = try_gpu_context() else {
            eprintln!("skipping GPU test: no adapter");
            return;
        };
        let config = NoiseConfig::default();
        let compute = GpuNoiseCompute::new(ctx, &config, 42, 32000.0, 4000.0);

        // Column at equator (lon=0, lat=0).
        let columns = [[0.0_f32, 0.0_f32]];
        let result = compute.evaluate_batch(&columns);
        assert_eq!(result.len(), 1);

        // Surface radius should be near mean_radius (32000) ± height_scale (4000).
        let r = result[0];
        assert!(
            r > 28000.0 && r < 36000.0,
            "surface radius {r} should be within mean_radius ± height_scale"
        );
    }

    #[test]
    fn batch_is_deterministic() {
        let Some(ctx) = try_gpu_context() else {
            eprintln!("skipping GPU test: no adapter");
            return;
        };
        let config = NoiseConfig::default();
        let compute = GpuNoiseCompute::new(ctx, &config, 42, 32000.0, 4000.0);

        let columns: Vec<[f32; 2]> = (0..64)
            .map(|i| {
                let lon = (i as f32) * 0.1 - std::f32::consts::PI;
                let lat = (i as f32) * 0.05 - 1.57;
                [lon, lat]
            })
            .collect();

        let result1 = compute.evaluate_batch(&columns);
        let result2 = compute.evaluate_batch(&columns);

        assert_eq!(result1.len(), result2.len());
        for (a, b) in result1.iter().zip(result2.iter()) {
            assert_eq!(a, b, "GPU noise must be deterministic");
        }
    }

    #[test]
    fn batch_varies_with_position() {
        let Some(ctx) = try_gpu_context() else {
            eprintln!("skipping GPU test: no adapter");
            return;
        };
        let config = NoiseConfig::default();
        let compute = GpuNoiseCompute::new(ctx, &config, 42, 32000.0, 4000.0);

        // Two widely separated columns should produce different heights.
        let columns = [[0.0_f32, 0.0_f32], [1.5_f32, 0.8_f32]];
        let result = compute.evaluate_batch(&columns);
        assert_eq!(result.len(), 2);
        assert!(
            (result[0] - result[1]).abs() > 0.1,
            "different positions should yield different heights: {} vs {}",
            result[0],
            result[1],
        );
    }

    #[test]
    fn full_chunk_batch() {
        let Some(ctx) = try_gpu_context() else {
            eprintln!("skipping GPU test: no adapter");
            return;
        };
        let config = NoiseConfig::default();
        let compute = GpuNoiseCompute::new(ctx, &config, 42, 32000.0, 4000.0);

        // Simulate one full chunk: 32×32 = 1024 columns.
        let columns: Vec<[f32; 2]> = (0..1024)
            .map(|i| {
                let lx = (i % 32) as f32;
                let lz = (i / 32) as f32;
                let lon = lx * 0.001;
                let lat = lz * 0.001;
                [lon, lat]
            })
            .collect();

        let result = compute.evaluate_batch(&columns);
        assert_eq!(result.len(), 1024);

        // All results should be valid radii.
        for (i, &r) in result.iter().enumerate() {
            assert!(
                r.is_finite() && r > 20000.0 && r < 40000.0,
                "column {i}: radius {r} out of expected range"
            );
        }
    }

    /// Verify GPU and CPU noise produce terrain with similar statistical character.
    ///
    /// The GPU uses hash-based f32 Perlin while the CPU uses permutation-table
    /// f64 Perlin, so values won't match — but their range, mean, and variance
    /// should be comparable for the same configuration.
    #[test]
    fn gpu_cpu_parity_statistics() {
        use crate::world::noise::NoiseStack;

        let Some(ctx) = try_gpu_context() else {
            eprintln!("skipping GPU test: no adapter");
            return;
        };
        let config = NoiseConfig::default();
        let mean_radius = 32000.0_f64;
        let height_scale = 4000.0_f64;

        let gpu = GpuNoiseCompute::new(ctx, &config, 42, mean_radius, height_scale);
        let cpu_stack = NoiseStack::new(42, config);

        // Sample a grid of 256 columns spanning a wide angular range.
        let n = 256;
        let columns: Vec<[f32; 2]> = (0..n)
            .map(|i| {
                let lon = (i % 16) as f32 * 0.3 - 2.4;
                let lat = (i / 16) as f32 * 0.15 - 1.2;
                [lon, lat]
            })
            .collect();

        let gpu_results = gpu.evaluate_batch(&columns);
        assert_eq!(gpu_results.len(), n);

        // Compute CPU results.
        let cpu_results: Vec<f64> = columns
            .iter()
            .map(|col| {
                let combined = cpu_stack.sample(col[0] as f64, col[1] as f64);
                mean_radius + combined * height_scale
            })
            .collect();

        // Stats: mean and stddev.
        let gpu_mean = gpu_results.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        let cpu_mean = cpu_results.iter().sum::<f64>() / n as f64;

        let gpu_var = gpu_results
            .iter()
            .map(|&v| (v as f64 - gpu_mean).powi(2))
            .sum::<f64>()
            / n as f64;
        let cpu_var = cpu_results
            .iter()
            .map(|&v| (v - cpu_mean).powi(2))
            .sum::<f64>()
            / n as f64;

        let gpu_std = gpu_var.sqrt();
        let cpu_std = cpu_var.sqrt();

        // Both should be centered near mean_radius.
        assert!(
            (gpu_mean - mean_radius).abs() < height_scale,
            "GPU mean {gpu_mean} too far from mean_radius {mean_radius}",
        );
        assert!(
            (cpu_mean - mean_radius).abs() < height_scale,
            "CPU mean {cpu_mean} too far from mean_radius {mean_radius}",
        );

        // Both should have similar spread (within 5× of each other).
        assert!(
            gpu_std > 0.0 && cpu_std > 0.0,
            "both paths should produce terrain variation (GPU std={gpu_std}, CPU std={cpu_std})",
        );
        let ratio = if gpu_std > cpu_std {
            gpu_std / cpu_std
        } else {
            cpu_std / gpu_std
        };
        assert!(
            ratio < 5.0,
            "GPU and CPU terrain should have comparable variation (ratio={ratio:.2}, GPU std={gpu_std:.1}, CPU std={cpu_std:.1})",
        );

        eprintln!(
            "GPU/CPU parity: GPU mean={gpu_mean:.1} std={gpu_std:.1}, CPU mean={cpu_mean:.1} std={cpu_std:.1}, ratio={ratio:.2}"
        );
    }

    /// Benchmark: measure GPU batch evaluation throughput.
    #[test]
    fn benchmark_gpu_batch() {
        let Some(ctx) = try_gpu_context() else {
            eprintln!("skipping GPU benchmark: no adapter");
            return;
        };
        let config = NoiseConfig::default();
        let compute = GpuNoiseCompute::new(ctx, &config, 42, 32000.0, 4000.0);

        // 8 chunks = 8192 columns (maximum batch size).
        let columns: Vec<[f32; 2]> = (0..8192)
            .map(|i| {
                let lon = (i % 128) as f32 * 0.001 - 0.064;
                let lat = (i / 128) as f32 * 0.001 - 0.032;
                [lon, lat]
            })
            .collect();

        // Warm up: first dispatch compiles the shader.
        let _ = compute.evaluate_batch(&columns);

        // Timed runs.
        let iterations = 10;
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = compute.evaluate_batch(&columns);
        }
        let elapsed = start.elapsed();

        let per_batch_us = elapsed.as_micros() / iterations;
        let per_chunk_us = per_batch_us / 8;

        eprintln!(
            "GPU noise benchmark: {per_batch_us}µs per batch (8 chunks), {per_chunk_us}µs per chunk"
        );

        // Sanity: should complete within 50ms per batch (very loose bound).
        assert!(
            elapsed.as_millis() / iterations < 50,
            "GPU batch took too long: {}ms avg",
            elapsed.as_millis() / iterations,
        );
    }
}
