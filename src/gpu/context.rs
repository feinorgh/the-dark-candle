//! Headless wgpu device initialization and buffer management.

use wgpu::{self, util::DeviceExt};

/// Wraps a wgpu device + queue for headless GPU compute work.
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl Default for GpuContext {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuContext {
    /// Request a GPU device suitable for compute work (no surface needed).
    pub fn new() -> Self {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("No suitable GPU adapter found");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("dark-candle-gpu-renderer"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits {
                    max_storage_buffer_binding_size: 256 * 1024 * 1024, // 256 MB
                    max_buffer_size: 256 * 1024 * 1024,
                    max_compute_invocations_per_workgroup: 256,
                    max_compute_workgroup_size_x: 16,
                    max_compute_workgroup_size_y: 16,
                    max_compute_workgroup_size_z: 1,
                    ..Default::default()
                },
                ..Default::default()
            })
            .await
            .expect("Failed to create GPU device");

        Self { device, queue }
    }

    /// Create a storage buffer initialized with data.
    pub fn create_storage_buffer(&self, label: &str, data: &[u8]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: data,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
    }

    /// Create a uniform buffer initialized with data.
    pub fn create_uniform_buffer(&self, label: &str, data: &[u8]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: data,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
    }

    /// Create an output storage buffer (for compute shader writes + CPU readback).
    pub fn create_output_buffer(&self, label: &str, size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// Create a staging buffer for GPU → CPU readback.
    pub fn create_staging_buffer(&self, label: &str, size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Submit a command encoder and wait for completion.
    pub fn submit_and_wait(&self, encoder: wgpu::CommandEncoder) {
        let index = self.queue.submit(std::iter::once(encoder.finish()));
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: Some(index),
            timeout: None,
        });
    }

    /// Read data back from a buffer (blocks until complete).
    pub fn read_buffer(&self, buffer: &wgpu::Buffer, size: u64) -> Vec<u8> {
        let slice = buffer.slice(..size);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        rx.recv().unwrap().expect("Buffer map failed");

        let data = slice.get_mapped_range();
        let result = data.to_vec();
        drop(data);
        buffer.unmap();
        result
    }
}
