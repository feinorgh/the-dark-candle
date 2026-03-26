//! GPU-accelerated valley river scene test.
//!
//! Generates a realistic valley terrain using the actual `TerrainGenerator`
//! and D8 flow-accumulation erosion pipeline, then renders an orbital
//! camera flyover with 24-hour day/night cycle, fog, and clouds.
//!
//! This test is `#[ignore]`-gated because it requires a GPU and ffmpeg.
//! Run with: `cargo test --test valley_river_gpu -- --ignored`

use the_dark_candle::data::{MaterialRegistry, load_material_registry};
use the_dark_candle::diagnostics::video::FrameEncoder;
use the_dark_candle::gpu::{GpuRenderParams, GpuRenderer};
use the_dark_candle::lighting::clouds::CloudRenderParams;
use the_dark_candle::lighting::scattering::ScatteringParams;
use the_dark_candle::lighting::shadows::{CloudShadowParams, FogParams};
use the_dark_candle::world::chunk::{Chunk, ChunkCoord};
use the_dark_candle::world::scene_presets::{ScenePreset, valley_river_erosion_config};
use the_dark_candle::world::terrain::{TerrainConfig, TerrainGenerator};
use the_dark_candle::world::voxel::Voxel;

const CHUNK_SIZE: usize = 32;
const CHUNK_VOLUME: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

fn idx(x: usize, y: usize, z: usize) -> usize {
    z * CHUNK_SIZE * CHUNK_SIZE + y * CHUNK_SIZE + x
}

/// Build a material registry with the materials used by terrain generation.
fn valley_registry() -> MaterialRegistry {
    load_material_registry().unwrap_or_else(|_| {
        let mut reg = MaterialRegistry::new();
        let materials = [
            (0, "air", [0.0, 0.0, 0.0], true),
            (1, "stone", [0.5, 0.5, 0.45], false),
            (2, "dirt", [0.45, 0.3, 0.15], false),
            (3, "water", [0.1, 0.3, 0.7], true),
            (4, "iron", [0.6, 0.55, 0.5], false),
            (5, "wood", [0.55, 0.35, 0.15], false),
            (6, "sand", [0.85, 0.78, 0.55], false),
            (7, "grass", [0.2, 0.6, 0.15], false),
        ];
        for (id, name, color, transparent) in materials {
            reg.insert(the_dark_candle::data::MaterialData {
                id,
                name: name.to_string(),
                color,
                transparent,
                ..Default::default()
            });
        }
        reg
    })
}

/// Build a valley terrain using the real TerrainGenerator + erosion.
///
/// Generates a single chunk at a position chosen to intersect the carved
/// river valley. Returns a flat voxel array suitable for GPU upload.
fn build_eroded_valley_terrain() -> Vec<Voxel> {
    let preset = ScenePreset::ValleyRiver;
    let planet = preset.planet_config();

    let config = TerrainConfig {
        seed: planet.seed,
        sea_level: planet.sea_level_radius as i32,
        height_scale: planet.height_scale,
        continent_freq: planet.continent_freq,
        detail_freq: planet.detail_freq,
        cave_freq: planet.cave_freq,
        cave_threshold: planet.cave_threshold,
        soil_depth: planet.soil_depth as i32,
        erosion: valley_river_erosion_config(),
    };

    let generator = TerrainGenerator::new(config);

    // Force flow map computation so erosion is active.
    let _ = generator.get_or_compute_flow_map();

    // Search for a chunk that contains interesting river content.
    // We scan a few chunk positions near the world center and pick the one
    // with the most water + sand voxels (indicating a carved river channel).
    let mut best_chunk = None;
    let mut best_score: u32 = 0;

    for cx in -4..=4 {
        for cz in -4..=4 {
            // Use cy=1 to be above sea level in flat mode (sea_level=40, chunk y=1 => voxels 32..63)
            let coord = ChunkCoord { x: cx, y: 1, z: cz };
            let mut chunk = Chunk::new_empty(coord);
            generator.generate_chunk(&mut chunk);

            let mut score: u32 = 0;
            for v in chunk.voxels() {
                match v.material.0 {
                    3 => score += 3, // water — high value
                    6 => score += 2, // sand — channel bed
                    1 => score += 1, // stone — exposed walls
                    _ => {}
                }
            }
            if score > best_score {
                best_score = score;
                best_chunk = Some(chunk);
            }
        }
    }

    let chunk = best_chunk.expect("Should find at least one generated chunk");
    chunk.voxels().to_vec()
}

/// Build a cloud field — a thin layer at y=22..26 with a soft circular profile.
fn build_cloud_field() -> Vec<f32> {
    let mut cloud = vec![0.0f32; CHUNK_VOLUME];
    for z in 2..30 {
        for x in 2..30 {
            for y in 22..26 {
                let cx = (x as f32 - 16.0) / 14.0;
                let cz = (z as f32 - 16.0) / 14.0;
                let r2 = cx * cx + cz * cz;
                if r2 < 1.0 {
                    cloud[idx(x, y, z)] = 0.4e-3 * (1.0 - r2);
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
    [horiz / len, elev / len, 0.1]
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
#[ignore] // Requires GPU + ffmpeg; run with --ignored
fn gpu_valley_river_video() {
    let width = 640;
    let height = 480;
    let mut renderer = GpuRenderer::new(width, height);

    let voxels = build_eroded_valley_terrain();
    let registry = valley_registry();
    let cloud_data = build_cloud_field();

    // Count materials for diagnostics.
    let mut water_count = 0u32;
    let mut sand_count = 0u32;
    let mut stone_count = 0u32;
    for v in &voxels {
        match v.material.0 {
            3 => water_count += 1,
            6 => sand_count += 1,
            1 => stone_count += 1,
            _ => {}
        }
    }
    println!(
        "Valley terrain: water={water_count}, sand={sand_count}, stone={stone_count}, total={}",
        voxels.len()
    );

    renderer.upload_voxels(&voxels, CHUNK_SIZE);
    renderer.upload_materials(&registry, 7);
    renderer.upload_cloud_field(&cloud_data);

    // Precompute cloud shadow map.
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

    // Humidity + temperature for valley fog effect.
    let mut humidity = vec![0.003f32; CHUNK_VOLUME];
    let mut temperature = vec![288.0f32; CHUNK_VOLUME];
    for z in 0..CHUNK_SIZE {
        for x in 0..CHUNK_SIZE {
            // Dense fog in valley floor (low y values).
            for y in 0..10 {
                let depth_factor = 1.0 - (y as f32 / 10.0);
                humidity[idx(x, y, z)] = 0.02 * depth_factor;
                temperature[idx(x, y, z)] = 275.0 + 5.0 * (y as f32 / 10.0);
            }
        }
    }
    renderer.upload_humidity(&humidity);
    renderer.upload_temperature(&temperature);

    let output_path = "test_output/gpu_valley_river.mp4";
    std::fs::create_dir_all("test_output").ok();
    let mut encoder =
        FrameEncoder::new(output_path, width, height, 30).expect("Failed to create encoder");

    let total_frames = 900; // 30 seconds at 30 FPS.
    for frame in 0..total_frames {
        let t = frame as f32 / total_frames as f32;
        let hour = t * 24.0;

        let sun_dir = sun_direction(hour);
        let elevation = sun_dir[1].asin();

        // Orbital camera: circles the terrain at varying height.
        let cam_angle = t * std::f32::consts::TAU * 0.75;
        let cam_radius = 28.0;
        let cam_height = 18.0 + (t * std::f32::consts::TAU).sin() * 4.0;
        let eye = [
            16.0 + cam_angle.cos() * cam_radius,
            cam_height,
            16.0 + cam_angle.sin() * cam_radius,
        ];

        let params = GpuRenderParams {
            eye,
            target: [16.0, 6.0, 16.0],
            fov_degrees: 65.0,
            sun_direction: sun_dir,
            sun_color: sun_color_for_elevation(elevation),
            sun_intensity: 1.0,
            sun_elevation: elevation,
            ambient: 0.08 + elevation.max(0.0) * 0.12,
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
                fog_density_base: 0.04,
                fog_height_falloff: 0.2,
                humidity_scale: 10.0,
                temperature_factor: 0.025,
                fog_color: sun_color_for_elevation(elevation.max(0.0)),
                max_fog_distance: 55.0,
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
        "GPU valley river scene: {total_frames} frames, {} bytes",
        metadata.len()
    );
}

#[test]
fn valley_terrain_has_river_features() {
    let voxels = build_eroded_valley_terrain();

    let mut water_count = 0u32;
    let mut sand_count = 0u32;
    for v in &voxels {
        match v.material.0 {
            3 => water_count += 1,
            6 => sand_count += 1,
            _ => {}
        }
    }

    // The eroded terrain should contain at least some water and sand,
    // proving the erosion pipeline is working.
    assert!(
        water_count > 0 || sand_count > 0,
        "Expected river features (water={water_count}, sand={sand_count})"
    );
}
