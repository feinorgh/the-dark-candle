//! GPU-accelerated atmosphere rendering tests.
//!
//! These are the GPU equivalents of the CPU tests in `atmosphere_visual.rs`.
//! Expected to run 100-1000× faster.

use the_dark_candle::data::{MaterialRegistry, load_material_registry};
use the_dark_candle::diagnostics::video::FrameEncoder;
use the_dark_candle::gpu::{GpuRenderParams, GpuRenderer};
use the_dark_candle::lighting::clouds::CloudRenderParams;
use the_dark_candle::lighting::scattering::ScatteringParams;
use the_dark_candle::lighting::shadows::{CloudShadowParams, FogParams};
use the_dark_candle::world::voxel::{MaterialId, Voxel};

const CHUNK_SIZE: usize = 32;
const CHUNK_VOLUME: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

fn idx(x: usize, y: usize, z: usize) -> usize {
    z * CHUNK_SIZE * CHUNK_SIZE + y * CHUNK_SIZE + x
}

fn atmosphere_registry() -> MaterialRegistry {
    load_material_registry().unwrap_or_else(|_| {
        let mut reg = MaterialRegistry::new();
        // Minimal materials for rendering.
        reg.insert(the_dark_candle::data::MaterialData {
            id: 0,
            name: "air".to_string(),
            color: [0.0, 0.0, 0.0],
            transparent: true,
            ..Default::default()
        });
        reg.insert(the_dark_candle::data::MaterialData {
            id: 1,
            name: "stone".to_string(),
            color: [0.5, 0.5, 0.45],
            ..Default::default()
        });
        reg.insert(the_dark_candle::data::MaterialData {
            id: 2,
            name: "grass".to_string(),
            color: [0.2, 0.6, 0.15],
            ..Default::default()
        });
        reg.insert(the_dark_candle::data::MaterialData {
            id: 3,
            name: "water".to_string(),
            color: [0.1, 0.3, 0.7],
            transparent: true,
            ..Default::default()
        });
        reg
    })
}

/// Build a simple valley terrain for testing.
fn build_valley_terrain() -> Vec<Voxel> {
    let mut voxels = vec![Voxel::default(); CHUNK_VOLUME];

    for z in 0..CHUNK_SIZE {
        for x in 0..CHUNK_SIZE {
            // V-shaped valley.
            let cx = (x as f32 - 16.0).abs();
            let cz = (z as f32 - 16.0).abs();
            let base_height = 4.0 + cx * 0.4 + cz * 0.2;
            let h = base_height as usize;

            for y in 0..h.min(CHUNK_SIZE) {
                let mat = if y == h - 1 {
                    MaterialId(2) // grass
                } else {
                    MaterialId(1) // stone
                };
                voxels[idx(x, y, z)] = Voxel::new(mat);
            }

            // Water in the valley floor.
            if h < 6 {
                for y in h..6 {
                    voxels[idx(x, y, z)] = Voxel::new(MaterialId(3));
                }
            }
        }
    }

    voxels
}

/// Build a cloud field with a layer at y=18..22.
fn build_cloud_field() -> Vec<f32> {
    let mut cloud = vec![0.0f32; CHUNK_VOLUME];
    for z in 4..28 {
        for x in 4..28 {
            for y in 18..22 {
                let cx = (x as f32 - 16.0) / 12.0;
                let cz = (z as f32 - 16.0) / 12.0;
                let r2 = cx * cx + cz * cz;
                if r2 < 1.0 {
                    cloud[idx(x, y, z)] = 0.5e-3 * (1.0 - r2);
                }
            }
        }
    }
    cloud
}

fn sun_direction(hour: f32) -> [f32; 3] {
    let angle = (hour - 6.0) / 12.0 * std::f32::consts::PI;
    let elev = angle.sin();
    let horiz = angle.cos();
    let len = (elev * elev + horiz * horiz).sqrt();
    [horiz / len, elev / len, 0.0]
}

fn sun_color_for_elevation(elevation: f32) -> [f32; 3] {
    if elevation < 0.0 {
        return [0.0, 0.0, 0.0];
    }
    let warmth = 1.0 - (elevation / 1.5).min(1.0);
    [1.0, 1.0 - warmth * 0.3, 1.0 - warmth * 0.6]
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[test]
fn gpu_sky_panorama_video() {
    let width = 512;
    let height = 384;
    let mut renderer = GpuRenderer::new(width, height);

    // No terrain — sky only. Upload minimal dummy data.
    let voxels = vec![Voxel::default(); CHUNK_VOLUME];
    let registry = atmosphere_registry();
    renderer.upload_voxels(&voxels, CHUNK_SIZE);
    renderer.upload_materials(&registry, 3);
    renderer.upload_cloud_field(&vec![0.0f32; CHUNK_VOLUME]);
    renderer.upload_shadow_map(&vec![1.0f32; CHUNK_SIZE * CHUNK_SIZE]);
    renderer.upload_humidity(&vec![0.0f32; CHUNK_VOLUME]);
    renderer.upload_temperature(&vec![288.0f32; CHUNK_VOLUME]);

    let output_path = "test_output/gpu_sky_panorama.mp4";
    std::fs::create_dir_all("test_output").ok();
    let mut encoder =
        FrameEncoder::new(output_path, width, height, 30).expect("Failed to create encoder");

    let fps = 30;
    let duration_s = 12;
    let total_frames = fps * duration_s;

    for frame in 0..total_frames {
        let hour = frame as f32 / total_frames as f32 * 24.0;
        let sun_dir = sun_direction(hour);
        let elevation = sun_dir[1].asin();

        let params = GpuRenderParams {
            eye: [16.0, 5.0, 16.0],
            target: [16.0, 5.0, 0.0],
            fov_degrees: 90.0,
            sun_direction: sun_dir,
            sun_color: sun_color_for_elevation(elevation),
            sun_intensity: 1.0,
            sun_elevation: elevation,
            ambient: 0.1,
            scatter_params: ScatteringParams::default(),
            enable_stars: true,
            time_hash: frame as u32,
            ..Default::default()
        };

        let frame_img = renderer.render_frame(&params);
        encoder.push_frame(&frame_img).ok();
    }

    encoder.finish().expect("Failed to finish encoding");
    let metadata = std::fs::metadata(output_path).expect("Output file not found");
    assert!(metadata.len() > 100, "Output file too small");
    println!(
        "GPU sky panorama: {} frames, {} bytes",
        total_frames,
        metadata.len()
    );
}

#[test]
fn gpu_volumetric_clouds_video() {
    let width = 512;
    let height = 384;
    let mut renderer = GpuRenderer::new(width, height);

    let voxels = vec![Voxel::default(); CHUNK_VOLUME];
    let registry = atmosphere_registry();
    let cloud_data = build_cloud_field();

    renderer.upload_voxels(&voxels, CHUNK_SIZE);
    renderer.upload_materials(&registry, 3);
    renderer.upload_cloud_field(&cloud_data);
    renderer.upload_shadow_map(&vec![1.0f32; CHUNK_SIZE * CHUNK_SIZE]);
    renderer.upload_humidity(&vec![0.0f32; CHUNK_VOLUME]);
    renderer.upload_temperature(&vec![288.0f32; CHUNK_VOLUME]);

    let output_path = "test_output/gpu_volumetric_clouds.mp4";
    std::fs::create_dir_all("test_output").ok();
    let mut encoder =
        FrameEncoder::new(output_path, width, height, 30).expect("Failed to create encoder");

    let total_frames = 300;
    for frame in 0..total_frames {
        let angle = frame as f32 / total_frames as f32 * std::f32::consts::TAU;
        let radius = 40.0;
        let eye = [
            16.0 + angle.cos() * radius,
            20.0,
            16.0 + angle.sin() * radius,
        ];

        let params = GpuRenderParams {
            eye,
            target: [16.0, 20.0, 16.0],
            fov_degrees: 60.0,
            sun_direction: [0.3, 0.8, 0.1],
            sun_color: [1.0, 0.95, 0.85],
            sun_intensity: 1.0,
            sun_elevation: 0.8,
            ambient: 0.15,
            scatter_params: ScatteringParams::default(),
            cloud_params: CloudRenderParams {
                max_march_distance: 80.0,
                step_size: 0.5,
                ..Default::default()
            },
            enable_clouds: true,
            ..Default::default()
        };

        let frame_img = renderer.render_frame(&params);
        encoder.push_frame(&frame_img).ok();
    }

    encoder.finish().expect("Failed to finish encoding");
    let metadata = std::fs::metadata(output_path).expect("Output file not found");
    assert!(metadata.len() > 100, "Output file too small");
    println!(
        "GPU volumetric clouds: {} frames, {} bytes",
        total_frames,
        metadata.len()
    );
}

#[test]
fn gpu_atmosphere_showcase_video() {
    let width = 640;
    let height = 480;
    let mut renderer = GpuRenderer::new(width, height);

    let voxels = build_valley_terrain();
    let registry = atmosphere_registry();
    let cloud_data = build_cloud_field();

    renderer.upload_voxels(&voxels, CHUNK_SIZE);
    renderer.upload_materials(&registry, 3);
    renderer.upload_cloud_field(&cloud_data);

    // Precompute shadow map.
    let shadow_map = the_dark_candle::lighting::shadows::compute_shadow_map(
        &cloud_data,
        CHUNK_SIZE,
        10,
        &CloudShadowParams {
            sun_direction: [0.3, 0.8, 0.1],
            ..Default::default()
        },
    );
    renderer.upload_shadow_map(&shadow_map);

    // Humidity + temperature fields for fog.
    let mut humidity = vec![0.005f32; CHUNK_VOLUME];
    let mut temperature = vec![288.0f32; CHUNK_VOLUME];
    for z in 0..CHUNK_SIZE {
        for x in 0..CHUNK_SIZE {
            for y in 0..8 {
                humidity[idx(x, y, z)] = 0.015;
                temperature[idx(x, y, z)] = 275.0;
            }
        }
    }
    renderer.upload_humidity(&humidity);
    renderer.upload_temperature(&temperature);

    let output_path = "test_output/gpu_atmosphere_showcase.mp4";
    std::fs::create_dir_all("test_output").ok();
    let mut encoder =
        FrameEncoder::new(output_path, width, height, 30).expect("Failed to create encoder");

    let total_frames = 900; // 30 seconds.
    for frame in 0..total_frames {
        let t = frame as f32 / total_frames as f32;
        let hour = t * 24.0;

        let sun_dir = sun_direction(hour);
        let elevation = sun_dir[1].asin();

        // Slow orbit.
        let cam_angle = t * std::f32::consts::TAU * 0.5;
        let eye = [
            16.0 + cam_angle.cos() * 25.0,
            15.0,
            16.0 + cam_angle.sin() * 25.0,
        ];

        let params = GpuRenderParams {
            eye,
            target: [16.0, 8.0, 16.0],
            fov_degrees: 70.0,
            sun_direction: sun_dir,
            sun_color: sun_color_for_elevation(elevation),
            sun_intensity: 1.0,
            sun_elevation: elevation,
            ambient: 0.1 + elevation.max(0.0) * 0.1,
            scatter_params: ScatteringParams::default(),
            cloud_params: CloudRenderParams {
                max_march_distance: 80.0,
                step_size: 0.5,
                ..Default::default()
            },
            shadow_params: CloudShadowParams {
                sun_direction: sun_dir,
                ..Default::default()
            },
            fog_params: FogParams {
                fog_density_base: 0.03,
                fog_height_falloff: 0.15,
                humidity_scale: 8.0,
                temperature_factor: 0.02,
                fog_color: sun_color_for_elevation(elevation.max(0.0)),
                max_fog_distance: 60.0,
            },
            enable_clouds: true,
            enable_fog: true,
            enable_shadows: true,
            enable_stars: true,
            time_hash: frame as u32,
        };

        let frame_img = renderer.render_frame(&params);
        encoder.push_frame(&frame_img).ok();
    }

    encoder.finish().expect("Failed to finish encoding");
    let metadata = std::fs::metadata(output_path).expect("Output file not found");
    assert!(metadata.len() > 100, "Output file too small");
    println!(
        "GPU atmosphere showcase: {} frames, {} bytes",
        total_frames,
        metadata.len()
    );
}
