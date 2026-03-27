// Atmosphere visual tests — produce MP4 videos for manual evaluation.
//
// These tests exercise the Phase 9 atmosphere simulation systems:
//   - Rayleigh + Mie sky scattering (sky panorama across day cycle)
//   - Volumetric cloud rendering (Beer-Lambert, Henyey-Greenstein)
//   - Cloud shadows on terrain (Beer-Lambert shadow map)
//   - Height-based atmospheric fog with humidity/cold-pooling
//   - Integrative showcase: terrain + clouds + fog + shadows + sky + day/night
//
// Run:
//   cargo test --test atmosphere_visual -- --nocapture
//
// Output files:
//   test_output/sky_panorama.mp4        — 24-hour sky scattering cycle
//   test_output/volumetric_clouds.mp4   — orbiting camera through cloud field
//   test_output/cloud_shadows.mp4       — terrain with animated cloud shadows
//   test_output/valley_fog.mp4          — morning fog dissipating with sunrise
//   test_output/atmosphere_showcase.mp4 — full integrative demo

use image::{Rgb, RgbImage};

use the_dark_candle::data::{MaterialData, MaterialRegistry};
use the_dark_candle::diagnostics::video::FrameEncoder;
use the_dark_candle::diagnostics::visualization::{
    ColorMode, SceneLight, ViewMode, render_frame_lit,
};
use the_dark_candle::lighting::clouds::{
    CloudRenderParams, composite_cloud_over_sky, march_cloud_ray,
};
use the_dark_candle::lighting::scattering::{self, ScatteringParams};
use the_dark_candle::lighting::shadows::{
    CloudShadowParams, FogParams, apply_fog, compute_shadow_map, fog_transmittance,
};
use the_dark_candle::physics::lbm_gas::plugin::CloudField;
use the_dark_candle::world::chunk::ChunkCoord;
use the_dark_candle::world::voxel::{MaterialId, Voxel};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn idx(x: usize, y: usize, z: usize, size: usize) -> usize {
    z * size * size + y * size + x
}

/// Reinhard tone-map a single HDR channel to [0, 255].
fn tonemap(v: f32) -> u8 {
    ((v / (1.0 + v)) * 255.0).min(255.0) as u8
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-12 {
        return [0.0, 1.0, 0.0];
    }
    [v[0] / len, v[1] / len, v[2] / len]
}

fn sun_elevation(hour: f32) -> f32 {
    // Sunrise at 6h, zenith at 12h, sunset at 18h.
    let phase = (hour - 6.0) / 12.0 * std::f32::consts::PI;
    phase.sin() * std::f32::consts::FRAC_PI_2
}

fn sun_direction(hour: f32) -> [f32; 3] {
    let elevation = sun_elevation(hour);
    let azimuth = hour / 24.0 * std::f32::consts::TAU;
    let y = elevation.sin();
    let xz = elevation.cos();
    normalize([xz * azimuth.cos(), y, xz * azimuth.sin()])
}

fn sun_color_for_elevation(elevation: f32) -> [f32; 3] {
    let e = elevation.max(0.0);
    let t = (e / std::f32::consts::FRAC_PI_2).clamp(0.0, 1.0);
    // Dawn/dusk warm orange → noon neutral white
    let r = 1.0;
    let g = 0.55 + t * 0.45;
    let b = 0.3 + t * 0.7;
    [r, g, b]
}

/// Full material registry with optical properties for atmosphere tests.
fn atmosphere_registry() -> MaterialRegistry {
    let mut reg = MaterialRegistry::new();
    reg.insert(MaterialData {
        id: 0,
        name: "Air".into(),
        color: [0.8, 0.9, 1.0],
        transparent: true,
        ..Default::default()
    });
    reg.insert(MaterialData {
        id: 1,
        name: "Stone".into(),
        color: [0.5, 0.5, 0.5],
        ..Default::default()
    });
    reg.insert(MaterialData {
        id: 2,
        name: "Dirt".into(),
        color: [0.45, 0.32, 0.18],
        ..Default::default()
    });
    reg.insert(MaterialData {
        id: 3,
        name: "Water".into(),
        color: [0.2, 0.4, 0.8],
        transparent: true,
        absorption_rgb: Some([0.45, 0.07, 0.02]),
        albedo: 0.06,
        ..Default::default()
    });
    reg.insert(MaterialData {
        id: 7,
        name: "Grass".into(),
        color: [0.3, 0.6, 0.2],
        albedo: 0.25,
        ..Default::default()
    });
    reg.insert(MaterialData {
        id: 6,
        name: "Sand".into(),
        color: [0.85, 0.78, 0.55],
        albedo: 0.4,
        ..Default::default()
    });
    reg.insert(MaterialData {
        id: 8,
        name: "Ice".into(),
        color: [0.7, 0.85, 0.95],
        transparent: true,
        absorption_rgb: Some([0.35, 0.06, 0.02]),
        albedo: 0.80,
        ..Default::default()
    });
    reg
}

/// Build terrain with a lake in a valley for fog/cloud demos.
fn build_valley_terrain(size: usize) -> Vec<Voxel> {
    let mut voxels = vec![Voxel::default(); size * size * size];

    for x in 0..size {
        for z in 0..size {
            let fx = x as f32 / size as f32;
            let fz = z as f32 / size as f32;

            // Valley shape: high on edges, low in center
            let cx = fx - 0.5;
            let cz = fz - 0.5;
            let dist_from_center = (cx * cx + cz * cz).sqrt();

            // Bowl shape: low center, raised edges
            let h1 = dist_from_center * 12.0;
            // Ridges
            let h2 = ((fx * 5.0).sin() * (fz * 4.0).cos()) * 2.0;
            let h3 = ((fx * 11.0 + 1.3).sin() * (fz * 9.0 + 0.7).cos()) * 0.8;

            let base_height = (size as f32 * 0.2) + h1 + h2 + h3;
            let height = (base_height as usize).clamp(2, size - 4);

            for y in 0..=height {
                let i = idx(x, y, z, size);
                if y == height {
                    voxels[i].material = MaterialId(7); // Grass
                } else if y > height.saturating_sub(2) {
                    voxels[i].material = MaterialId(2); // Dirt
                } else {
                    voxels[i].material = MaterialId::STONE;
                }
                voxels[i].temperature = 288.15;
            }

            // Lake in the valley center
            let water_level = (size as f32 * 0.25) as usize;
            if height < water_level {
                for y in height + 1..=water_level {
                    let i = idx(x, y, z, size);
                    voxels[i].material = MaterialId::WATER;
                    voxels[i].temperature = 285.0;
                }
            }
        }
    }

    voxels
}

/// Generate a cloud LWC field with cumulus-like blobs at altitude.
fn build_cloud_field(size: usize, cloud_base_y: f32, time_offset: f32) -> Vec<f32> {
    let mut field = vec![0.0_f32; size * size * size];

    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let fx = x as f32 / size as f32;
                let fy = y as f32;
                let fz = z as f32 / size as f32;

                // Height envelope: gaussian around cloud_base_y
                let dy = fy - cloud_base_y;
                let height_factor = (-dy * dy / 18.0).exp();

                // Horizontal structure: multi-scale blobs
                let b1 = ((fx * 3.0 + time_offset * 0.1).sin()
                    * (fz * 2.5 + time_offset * 0.05).cos())
                .max(0.0);
                let b2 = ((fx * 7.0 + 1.7 + time_offset * 0.15).sin()
                    * (fz * 6.0 + 2.3 + time_offset * 0.08).cos())
                .max(0.0)
                    * 0.4;
                let b3 = ((fx * 13.0 + 0.3 + time_offset * 0.2).sin()
                    * (fz * 11.0 + 4.1 + time_offset * 0.12).cos())
                .max(0.0)
                    * 0.15;

                let density = (b1 + b2 + b3) * height_factor;

                // Threshold to create distinct cloud shapes
                let lwc = if density > 0.2 {
                    (density - 0.2) * 0.5e-3 // Scale to realistic LWC (kg/m³)
                } else {
                    0.0
                };

                let i = z * size * size + y * size + x;
                field[i] = lwc;
            }
        }
    }
    field
}

/// Create a CloudField resource with cloud data at chunk (0,0,0).
fn cloud_field_at_origin(data: Vec<f32>) -> CloudField {
    let mut cf = CloudField::default();
    cf.chunks.insert(ChunkCoord::new(0, 0, 0), data);
    cf
}

// ---------------------------------------------------------------------------
// Test 1: Sky Scattering Panorama — 24-hour cycle
// ---------------------------------------------------------------------------

/// Renders a panoramic sky view cycling through 24 hours,
/// showing Rayleigh blue sky, orange/red sunset/sunrise, and dark night.
/// Each frame is a hemisphere of sky directions rendered via ray-marching.
#[test]
#[ignore] // CPU-intensive video rendering; run with --ignored
fn sky_panorama_video() {
    let _ = std::fs::create_dir_all("test_output");
    let path = "test_output/sky_panorama.mp4";

    let width = 640_u32;
    let height = 360_u32;
    let fps = 30_u32;
    let duration_s = 12.0_f32; // 12 seconds for full 24-hour cycle
    let total_frames = (duration_s * fps as f32) as u32;

    let params = ScatteringParams::default();
    let mut encoder = FrameEncoder::new(path, width, height, fps).expect("encoder");

    for frame in 0..total_frames {
        let t = frame as f32 / total_frames as f32;
        let hour = t * 24.0;
        let sun_dir = sun_direction(hour);

        let mut img = RgbImage::new(width, height);

        for py in 0..height {
            for px in 0..width {
                // Map pixels to hemisphere directions
                let u = (px as f32 / width as f32) * 2.0 - 1.0; // -1 to 1 (azimuth)
                let v = py as f32 / height as f32; // 0 (horizon) to 1 (zenith at top)

                // Ray direction: azimuth around Y, elevation from horizon to zenith
                let azimuth = u * std::f32::consts::PI;
                let elevation = (1.0 - v) * std::f32::consts::FRAC_PI_2;
                let ray_dir = normalize([
                    azimuth.cos() * elevation.cos(),
                    elevation.sin(),
                    azimuth.sin() * elevation.cos(),
                ]);

                let sky = scattering::sky_color(ray_dir, sun_dir, 0.0, &params);

                // Sun disk
                let cos_angle: f32 = ray_dir.iter().zip(sun_dir.iter()).map(|(a, b)| a * b).sum();
                let sun_disk = if cos_angle > 0.9995 {
                    let sun_c = scattering::sun_disk_color(sun_dir, 0.0, &params);
                    [
                        sky[0] + sun_c[0] * 0.5,
                        sky[1] + sun_c[1] * 0.5,
                        sky[2] + sun_c[2] * 0.5,
                    ]
                } else if cos_angle > 0.995 {
                    // Corona glow
                    let glow = ((cos_angle - 0.995) / 0.0045).powi(2);
                    let sun_c = scattering::sun_disk_color(sun_dir, 0.0, &params);
                    [
                        sky[0] + sun_c[0] * glow * 0.2,
                        sky[1] + sun_c[1] * glow * 0.2,
                        sky[2] + sun_c[2] * glow * 0.2,
                    ]
                } else {
                    sky
                };

                img.put_pixel(
                    px,
                    py,
                    Rgb([
                        tonemap(sun_disk[0]),
                        tonemap(sun_disk[1]),
                        tonemap(sun_disk[2]),
                    ]),
                );
            }
        }

        // HUD: time-of-day label area at bottom
        let label_y = height - 20;
        let hour_frac = hour / 24.0;
        let marker_x = (hour_frac * width as f32) as u32;
        for dx in 0..3_u32 {
            for dy in 0..10_u32 {
                let px = (marker_x + dx).min(width - 1);
                let py = (label_y + dy).min(height - 1);
                img.put_pixel(px, py, Rgb([255, 255, 0]));
            }
        }

        encoder.push_frame(&img).expect("frame");
    }

    encoder.finish().expect("finalize");
    eprintln!("  ✓ Sky scattering panorama → {path}");
    assert!(std::path::Path::new(path).exists());
}

// ---------------------------------------------------------------------------
// Test 2: Volumetric Cloud Rendering — orbiting camera
// ---------------------------------------------------------------------------

/// Renders volumetric clouds from a 3D perspective camera orbiting
/// a cloud field. Shows Beer-Lambert extinction, Henyey-Greenstein
/// forward scattering, and cloud structure at various angles.
#[test]
#[ignore] // CPU-intensive video rendering; run with --ignored
fn volumetric_clouds_video() {
    let _ = std::fs::create_dir_all("test_output");
    let path = "test_output/volumetric_clouds.mp4";

    let size = 32_usize;
    let width = 512_u32;
    let height = 384_u32;
    let fps = 30_u32;
    let duration_s = 10.0_f32;
    let total_frames = (duration_s * fps as f32) as u32;

    let scattering_params = ScatteringParams::default();
    let cloud_params = CloudRenderParams {
        max_march_distance: size as f32 * 1.5,
        step_size: 0.5,
        ..Default::default()
    };

    let mut encoder = FrameEncoder::new(path, width, height, fps).expect("encoder");

    let center = size as f32 / 2.0;
    let cam_radius = size as f32 * 1.2;
    let cloud_base = size as f32 * 0.55;

    for frame in 0..total_frames {
        let t = frame as f32 / total_frames as f32;

        // Slowly evolving cloud field
        let cloud_data = build_cloud_field(size, cloud_base, t * 10.0);
        let cloud_field = cloud_field_at_origin(cloud_data);

        // Camera orbits at cloud altitude
        let angle = t * std::f32::consts::TAU;
        let cam_x = center + cam_radius * angle.cos();
        let cam_z = center + cam_radius * angle.sin();
        let cam_y = cloud_base + 2.0;

        // Sun slowly moves (afternoon → sunset)
        let hour = 10.0 + t * 8.0; // 10:00 → 18:00
        let sun_dir = sun_direction(hour);
        let elevation = sun_elevation(hour);
        let sun_col = sun_color_for_elevation(elevation);

        let mut img = RgbImage::new(width, height);

        // Build camera basis
        let eye = [cam_x, cam_y, cam_z];
        let target = [center, cloud_base, center];
        let fwd = normalize([target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]]);
        let right = normalize(cross(fwd, [0.0, 1.0, 0.0]));
        let up = cross(right, fwd);
        let fov_rad = 60.0_f32.to_radians();
        let aspect = width as f32 / height as f32;
        let half_h = (fov_rad / 2.0).tan();
        let half_w = half_h * aspect;

        for py in 0..height {
            for px in 0..width {
                let u = (px as f32 + 0.5) / width as f32 * 2.0 - 1.0;
                let v = 1.0 - (py as f32 + 0.5) / height as f32 * 2.0;

                let ray_dir = normalize([
                    fwd[0] + right[0] * u * half_w + up[0] * v * half_h,
                    fwd[1] + right[1] * u * half_w + up[1] * v * half_h,
                    fwd[2] + right[2] * u * half_w + up[2] * v * half_h,
                ]);

                // Sky background
                let sky = scattering::sky_color(ray_dir, sun_dir, cam_y, &scattering_params);

                // Cloud ray-march
                let cloud =
                    march_cloud_ray(eye, ray_dir, sun_dir, sun_col, &cloud_field, &cloud_params);

                // Composite
                let final_color = composite_cloud_over_sky(&cloud, sky);

                img.put_pixel(
                    px,
                    py,
                    Rgb([
                        tonemap(final_color[0]),
                        tonemap(final_color[1]),
                        tonemap(final_color[2]),
                    ]),
                );
            }
        }

        encoder.push_frame(&img).expect("frame");
    }

    encoder.finish().expect("finalize");
    eprintln!("  ✓ Volumetric cloud rendering → {path}");
    assert!(std::path::Path::new(path).exists());
}

// ---------------------------------------------------------------------------
// Test 3: Cloud Shadows on Terrain
// ---------------------------------------------------------------------------

/// Terrain viewed from above with animated cloud shadows sweeping across.
/// The cloud field drifts horizontally (simulating wind), and the shadow
/// pattern moves across grass/stone/water terrain.
#[test]
#[ignore] // CPU-intensive video rendering; run with --ignored
fn cloud_shadows_video() {
    let _ = std::fs::create_dir_all("test_output");
    let path = "test_output/cloud_shadows.mp4";

    let size = 32_usize;
    let img_w = 512_u32;
    let img_h = 384_u32;
    let fps = 30_u32;
    let duration_s = 10.0_f32;
    let total_frames = (duration_s * fps as f32) as u32;

    let registry = atmosphere_registry();
    let voxels = build_valley_terrain(size);
    let mut encoder = FrameEncoder::new(path, img_w, img_h, fps).expect("encoder");

    let center = size as f32 / 2.0;
    let cam_radius = size as f32 * 1.1;
    let cam_height = size as f32 * 0.85;

    // Find terrain surface heights for shadow map application
    let mut surface_y = vec![0_usize; size * size];
    for x in 0..size {
        for z in 0..size {
            for y in (0..size).rev() {
                if !voxels[idx(x, y, z, size)].material.is_air() {
                    surface_y[z * size + x] = y;
                    break;
                }
            }
        }
    }

    for frame in 0..total_frames {
        let t = frame as f32 / total_frames as f32;
        let hour = 8.0 + t * 6.0; // Morning to afternoon

        // Cloud field drifts with wind
        let cloud_data = build_cloud_field(size, size as f32 * 0.7, t * 20.0);

        // Sun direction
        let sun_dir = sun_direction(hour);
        let elevation = sun_elevation(hour);
        let sun_col = sun_color_for_elevation(elevation);
        let sun_factor = elevation.sin().max(0.0);

        // Compute cloud shadow map
        let shadow_params = CloudShadowParams {
            sun_direction: sun_dir,
            shadow_softness: 1.5,
            min_shadow_factor: 0.25,
            extinction_coeff: 80.0, // Stronger for visible shadows
        };

        // Average surface Y for shadow map
        let avg_surface = surface_y.iter().sum::<usize>() / surface_y.len();
        let shadow_map = compute_shadow_map(&cloud_data, size, avg_surface, &shadow_params);

        // Build modified voxels with shadow-darkened surface colors
        // (We modulate the scene light intensity per-pixel by sampling the shadow map)
        let angle = t * std::f32::consts::TAU * 0.3; // Slow orbit
        let cam_x = center + cam_radius * angle.cos();
        let cam_z = center + cam_radius * angle.sin();

        let view = ViewMode::Perspective {
            eye: (cam_x, cam_height, cam_z),
            target: (center, center * 0.3, center),
            fov_degrees: 55.0,
            width: img_w,
            height: img_h,
        };

        // Base light from sun
        let light = SceneLight {
            direction: (-sun_dir[0], -sun_dir[1], -sun_dir[2]),
            color: (sun_col[0], sun_col[1], sun_col[2]),
            intensity: sun_factor,
            ambient: 0.15 + sun_factor * 0.1,
        };

        // Render terrain with standard lighting
        let base_img = render_frame_lit(
            &voxels,
            size,
            &registry,
            &view,
            &ColorMode::Material,
            1,
            &light,
        );

        // Overlay cloud shadows by darkening pixels based on shadow map
        // We approximate by projecting each pixel's terrain position back to the shadow map
        let mut img = base_img;
        // For a more accurate result, we composite the shadow map over the rendered frame
        // by computing per-pixel shadow factor from the xz position
        for py in 0..img_h {
            for px in 0..img_w {
                let pixel = img.get_pixel(px, py);
                // Approximate terrain position from pixel coordinates
                // Use a simple mapping: center of frame = center of terrain
                let u = px as f32 / img_w as f32;
                let v = py as f32 / img_h as f32;

                // Map screen coordinates to terrain xz (approximate)
                let terrain_x = (u * size as f32) as usize;
                let terrain_z = (v * size as f32) as usize;
                let tx = terrain_x.min(size - 1);
                let tz = terrain_z.min(size - 1);

                let shadow = shadow_map[tz * size + tx];

                // Darken pixel by shadow factor
                let r = (pixel[0] as f32 * shadow).min(255.0) as u8;
                let g = (pixel[1] as f32 * shadow).min(255.0) as u8;
                let b = (pixel[2] as f32 * shadow).min(255.0) as u8;
                img.put_pixel(px, py, Rgb([r, g, b]));
            }
        }

        encoder.push_frame(&img).expect("frame");
    }

    encoder.finish().expect("finalize");
    eprintln!("  ✓ Cloud shadows on terrain → {path}");
    assert!(std::path::Path::new(path).exists());
}

// ---------------------------------------------------------------------------
// Test 4: Valley Fog — morning fog dissipating with sunrise
// ---------------------------------------------------------------------------

/// A valley scene with ground-level fog that dissipates as the sun rises.
/// Demonstrates height-dependent exponential fog, humidity amplification,
/// and the cold-pooling effect (fog is thickest in the cold valley bottom).
#[test]
#[ignore] // CPU-intensive video rendering; run with --ignored
fn valley_fog_video() {
    let _ = std::fs::create_dir_all("test_output");
    let path = "test_output/valley_fog.mp4";

    let size = 32_usize;
    let img_w = 640_u32;
    let img_h = 480_u32;
    let fps = 30_u32;
    let duration_s = 15.0_f32; // Pre-dawn → mid-morning
    let total_frames = (duration_s * fps as f32) as u32;

    let registry = atmosphere_registry();
    let voxels = build_valley_terrain(size);
    let scattering_params = ScatteringParams::default();
    let mut encoder = FrameEncoder::new(path, img_w, img_h, fps).expect("encoder");

    let center = size as f32 / 2.0;
    let cam_radius = size as f32 * 1.1;
    let cam_height = size as f32 * 0.6;

    // Humidity and temperature fields for fog computation
    // Valley bottom is humid and cold; ridges are drier and warmer
    let n = size * size * size;
    let base_humidity = 0.008_f32; // kg/kg

    for frame in 0..total_frames {
        let t = frame as f32 / total_frames as f32;
        let hour = 4.5 + t * 5.0; // 4:30 AM → 9:30 AM

        // Temperature rises with sunrise
        let sunrise_warmth = ((hour - 6.0).max(0.0) / 3.0).min(1.0); // 0 at 6AM, 1 at 9AM
        let base_temp = 275.0 + sunrise_warmth * 10.0; // 2°C → 12°C

        // Build per-voxel temperature and humidity fields
        let mut temps = vec![base_temp; n];
        let mut humidities = vec![base_humidity; n];
        let pressures = vec![101_325.0_f32; n];

        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let i = z * size * size + y * size + x;
                    let altitude = y as f32;

                    // Temperature inversion: cold valley bottom, warmer aloft
                    let inversion_strength = (1.0 - sunrise_warmth) * 5.0;
                    let inversion = if altitude < size as f32 * 0.3 {
                        -inversion_strength * (1.0 - altitude / (size as f32 * 0.3))
                    } else {
                        0.0
                    };
                    temps[i] = base_temp + inversion - altitude * 0.3; // Lapse rate

                    // Higher humidity near water/ground
                    let ground_factor = (-altitude * 0.15).exp();
                    humidities[i] = base_humidity * (1.0 + 2.0 * ground_factor);
                }
            }
        }

        // Sun
        let sun_dir = sun_direction(hour);
        let elevation = sun_elevation(hour);
        let sun_col = sun_color_for_elevation(elevation);
        let sun_factor = elevation.sin().max(0.0);

        // Fog parameters: denser when cold, reduce with sunrise warming
        let fog_params = FogParams {
            fog_density_base: 0.04 * (1.0 - sunrise_warmth * 0.7), // Thins with sunrise
            fog_height_falloff: 0.08,
            humidity_scale: 8.0,
            temperature_factor: 0.015,
            fog_color: [
                0.65 + sun_col[0] * 0.1,
                0.70 + sun_col[1] * 0.1,
                0.78 + sun_col[2] * 0.1,
            ],
            max_fog_distance: size as f32 * 2.0,
        };

        // Camera: static viewpoint into valley
        let angle = 0.8 + t * 0.3; // Slight drift
        let cam_x = center + cam_radius * angle.cos();
        let cam_z = center + cam_radius * angle.sin();

        let view = ViewMode::Perspective {
            eye: (cam_x, cam_height, cam_z),
            target: (center, center * 0.25, center),
            fov_degrees: 55.0,
            width: img_w,
            height: img_h,
        };

        let light = SceneLight {
            direction: (-sun_dir[0], -sun_dir[1], -sun_dir[2]),
            color: (sun_col[0], sun_col[1], sun_col[2]),
            intensity: sun_factor,
            ambient: 0.12 + sun_factor * 0.18,
        };

        // Render terrain
        let base_img = render_frame_lit(
            &voxels,
            size,
            &registry,
            &view,
            &ColorMode::Material,
            1,
            &light,
        );

        // Apply atmospheric fog per-pixel
        let fwd = normalize([center - cam_x, center * 0.25 - cam_height, center - cam_z]);
        let right_vec = normalize(cross(fwd, [0.0, 1.0, 0.0]));
        let up_vec = cross(right_vec, fwd);
        let fov_rad = 55.0_f32.to_radians();
        let aspect = img_w as f32 / img_h as f32;
        let half_h = (fov_rad / 2.0).tan();
        let half_w = half_h * aspect;

        let eye = [cam_x, cam_height, cam_z];

        let mut img = RgbImage::new(img_w, img_h);
        for py in 0..img_h {
            for px in 0..img_w {
                let u = (px as f32 + 0.5) / img_w as f32 * 2.0 - 1.0;
                let v = 1.0 - (py as f32 + 0.5) / img_h as f32 * 2.0;
                let ray_dir = normalize([
                    fwd[0] + right_vec[0] * u * half_w + up_vec[0] * v * half_h,
                    fwd[1] + right_vec[1] * u * half_w + up_vec[1] * v * half_h,
                    fwd[2] + right_vec[2] * u * half_w + up_vec[2] * v * half_h,
                ]);

                let base_pixel = base_img.get_pixel(px, py);
                let scene_color = [
                    base_pixel[0] as f32 / 255.0,
                    base_pixel[1] as f32 / 255.0,
                    base_pixel[2] as f32 / 255.0,
                ];

                // Estimate hit distance (rough: from brightness falloff)
                let hit_dist = if scene_color[0] + scene_color[1] + scene_color[2] > 0.01 {
                    // Terrain was hit — estimate distance from pixel darkness/fog
                    // Use the camera-to-center distance as rough estimate
                    let to_center = ((cam_x - center).powi(2) + (cam_z - center).powi(2)).sqrt();
                    to_center * (0.5 + v.abs() * 0.5)
                } else {
                    // Sky — full fog distance
                    fog_params.max_fog_distance
                };

                let (transmittance, fog_color) = fog_transmittance(
                    eye,
                    ray_dir,
                    hit_dist,
                    &humidities,
                    &temps,
                    &pressures,
                    size,
                    &fog_params,
                );

                // Sky color for fog blending
                let sky = if sun_factor > 0.0 {
                    scattering::sky_color(ray_dir, sun_dir, cam_height, &scattering_params)
                } else {
                    [0.02, 0.02, 0.05] // Pre-dawn dark blue
                };

                // Blend fog over scene
                let fogged = apply_fog(scene_color, transmittance, fog_color);

                // For sky pixels, blend with sky color instead
                let final_color = if scene_color[0] + scene_color[1] + scene_color[2] < 0.01 {
                    // Sky pixel — apply fog to sky
                    [
                        sky[0] * transmittance + fog_color[0] * (1.0 - transmittance),
                        sky[1] * transmittance + fog_color[1] * (1.0 - transmittance),
                        sky[2] * transmittance + fog_color[2] * (1.0 - transmittance),
                    ]
                } else {
                    fogged
                };

                img.put_pixel(
                    px,
                    py,
                    Rgb([
                        tonemap(final_color[0]),
                        tonemap(final_color[1]),
                        tonemap(final_color[2]),
                    ]),
                );
            }
        }

        encoder.push_frame(&img).expect("frame");
    }

    encoder.finish().expect("finalize");
    eprintln!("  ✓ Valley fog → {path}");
    assert!(std::path::Path::new(path).exists());
}

// ---------------------------------------------------------------------------
// Test 5: Integrative Showcase — all atmosphere systems combined
// ---------------------------------------------------------------------------

/// Full integrative demo: terrain with lake, volumetric clouds, cloud shadows,
/// atmospheric fog, Rayleigh/Mie sky scattering, and day-night lighting — all
/// interacting together in a single scene over a full day cycle.
///
/// This is the capstone visualization showing emergent behavior from the
/// interaction of multiple Phase 9 systems.
#[test]
#[ignore] // CPU-intensive video rendering; run with --ignored
fn atmosphere_showcase_video() {
    let _ = std::fs::create_dir_all("test_output");
    let path = "test_output/atmosphere_showcase.mp4";

    let size = 32_usize;
    let img_w = 800_u32;
    let img_h = 600_u32;
    let fps = 30_u32;
    let duration_s = 30.0_f32; // Full day cycle
    let total_frames = (duration_s * fps as f32) as u32;

    let registry = atmosphere_registry();
    let voxels = build_valley_terrain(size);
    let scattering_params = ScatteringParams::default();
    let cloud_params = CloudRenderParams {
        max_march_distance: size as f32 * 1.5,
        step_size: 0.8,
        ..Default::default()
    };
    let n = size * size * size;

    let mut encoder = FrameEncoder::new(path, img_w, img_h, fps).expect("encoder");

    let center = size as f32 / 2.0;
    let cam_radius = size as f32 * 1.15;
    let cam_height = size as f32 * 0.65;

    let start_hour = 4.0_f32; // Pre-dawn start

    for frame in 0..total_frames {
        let t = frame as f32 / total_frames as f32;
        let hour = (start_hour + t * 24.0) % 24.0;

        // --- Sun ---
        let sun_dir = sun_direction(hour);
        let elevation = sun_elevation(hour);
        let sun_col = sun_color_for_elevation(elevation);
        let sun_factor = elevation.sin().max(0.0);

        // --- Clouds (drifting with time) ---
        let cloud_base = size as f32 * 0.65;
        let cloud_data = build_cloud_field(size, cloud_base, t * 25.0);
        let cloud_field = cloud_field_at_origin(cloud_data.clone());

        // --- Cloud shadow map ---
        let shadow_params = CloudShadowParams {
            sun_direction: sun_dir,
            shadow_softness: 1.5,
            min_shadow_factor: 0.2,
            extinction_coeff: 60.0,
        };
        let shadow_map = compute_shadow_map(
            &cloud_data,
            size,
            (size as f32 * 0.3) as usize,
            &shadow_params,
        );

        // --- Temperature / humidity fields ---
        let warmth = sun_factor.sqrt(); // Warms with sun
        let base_temp = 278.0 + warmth * 12.0; // 5°C night → 17°C noon
        let base_humidity = 0.007 + (1.0 - warmth) * 0.003; // More humid at night

        let mut temps = vec![base_temp; n];
        let mut humidities = vec![base_humidity; n];
        let pressures = vec![101_325.0_f32; n];

        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let i = z * size * size + y * size + x;
                    let alt = y as f32;
                    // Lapse rate + nocturnal inversion
                    let inversion = if sun_factor < 0.1 && alt < size as f32 * 0.3 {
                        -3.0 * (1.0 - alt / (size as f32 * 0.3))
                    } else {
                        0.0
                    };
                    temps[i] = base_temp + inversion - alt * 0.25;
                    humidities[i] = base_humidity * (-alt * 0.1).exp();
                }
            }
        }

        // --- Fog parameters ---
        let fog_params = FogParams {
            fog_density_base: 0.025 * (1.0 - warmth * 0.8),
            fog_height_falloff: 0.1,
            humidity_scale: 6.0,
            temperature_factor: 0.012,
            fog_color: [
                0.6 + sun_col[0] * 0.15,
                0.65 + sun_col[1] * 0.15,
                0.75 + sun_col[2] * 0.15,
            ],
            max_fog_distance: size as f32 * 2.0,
        };

        // --- Camera: slow orbit ---
        let angle = t * std::f32::consts::TAU;
        let cam_x = center + cam_radius * angle.cos();
        let cam_z = center + cam_radius * angle.sin();
        let eye = [cam_x, cam_height, cam_z];

        let view = ViewMode::Perspective {
            eye: (cam_x, cam_height, cam_z),
            target: (center, center * 0.3, center),
            fov_degrees: 55.0,
            width: img_w,
            height: img_h,
        };

        let light = SceneLight {
            direction: (-sun_dir[0], -sun_dir[1], -sun_dir[2]),
            color: (sun_col[0], sun_col[1], sun_col[2]),
            intensity: sun_factor,
            ambient: 0.10 + sun_factor * 0.15,
        };

        // --- Render terrain ---
        let base_img = render_frame_lit(
            &voxels,
            size,
            &registry,
            &view,
            &ColorMode::Material,
            1,
            &light,
        );

        // --- Composite: terrain + shadows + fog + clouds + sky ---
        let fwd = normalize([center - cam_x, center * 0.3 - cam_height, center - cam_z]);
        let right_vec = normalize(cross(fwd, [0.0, 1.0, 0.0]));
        let up_vec = cross(right_vec, fwd);
        let fov_rad = 55.0_f32.to_radians();
        let aspect = img_w as f32 / img_h as f32;
        let half_h = (fov_rad / 2.0).tan();
        let half_w = half_h * aspect;

        let mut img = RgbImage::new(img_w, img_h);

        for py in 0..img_h {
            for px in 0..img_w {
                let u = (px as f32 + 0.5) / img_w as f32 * 2.0 - 1.0;
                let v = 1.0 - (py as f32 + 0.5) / img_h as f32 * 2.0;
                let ray_dir = normalize([
                    fwd[0] + right_vec[0] * u * half_w + up_vec[0] * v * half_h,
                    fwd[1] + right_vec[1] * u * half_w + up_vec[1] * v * half_h,
                    fwd[2] + right_vec[2] * u * half_w + up_vec[2] * v * half_h,
                ]);

                let base_pixel = base_img.get_pixel(px, py);
                let is_sky = base_pixel[0] == 0
                    && base_pixel[1] == 0
                    && base_pixel[2] == 0
                    && light.ambient < 0.01
                    || (base_pixel[0] as u16 + base_pixel[1] as u16 + base_pixel[2] as u16) < 3;

                let mut scene_color = [
                    base_pixel[0] as f32 / 255.0,
                    base_pixel[1] as f32 / 255.0,
                    base_pixel[2] as f32 / 255.0,
                ];

                // Apply cloud shadow to terrain pixels
                if !is_sky {
                    // Map pixel to terrain xz (rough approximation)
                    let screen_u = px as f32 / img_w as f32;
                    let screen_v = py as f32 / img_h as f32;
                    let tx = (screen_u * size as f32) as usize;
                    let tz = (screen_v * size as f32) as usize;
                    let tx = tx.min(size - 1);
                    let tz = tz.min(size - 1);
                    let shadow = shadow_map[tz * size + tx];
                    scene_color[0] *= shadow;
                    scene_color[1] *= shadow;
                    scene_color[2] *= shadow;
                }

                // Fog
                let hit_dist = if !is_sky {
                    let to_center = ((cam_x - center).powi(2) + (cam_z - center).powi(2)).sqrt();
                    to_center * (0.6 + v.abs() * 0.4)
                } else {
                    fog_params.max_fog_distance
                };

                let (fog_t, fog_color) = fog_transmittance(
                    eye,
                    ray_dir,
                    hit_dist,
                    &humidities,
                    &temps,
                    &pressures,
                    size,
                    &fog_params,
                );

                // Sky color
                let sky = if sun_factor > 0.001 {
                    scattering::sky_color(ray_dir, sun_dir, cam_height, &scattering_params)
                } else {
                    // Night sky — very dark blue with stars effect
                    let star = pseudo_star(px, py);
                    [0.01 + star, 0.01 + star * 0.8, 0.04 + star * 0.5]
                };

                // Volumetric clouds
                let cloud =
                    march_cloud_ray(eye, ray_dir, sun_dir, sun_col, &cloud_field, &cloud_params);

                // Compositing order:
                // 1. Start with terrain (with shadow) or sky
                let base = if is_sky { sky } else { scene_color };

                // 2. Apply fog to terrain
                let fogged = apply_fog(base, fog_t, fog_color);

                // 3. Composite clouds over fogged scene
                let with_clouds = composite_cloud_over_sky(&cloud, fogged);

                // 4. Sun disk (only on clear sky patches)
                let cos_sun: f32 = ray_dir.iter().zip(sun_dir.iter()).map(|(a, b)| a * b).sum();
                let final_color = if cos_sun > 0.9995 && cloud.transmittance > 0.5 {
                    let sun_c = scattering::sun_disk_color(sun_dir, cam_height, &scattering_params);
                    [
                        with_clouds[0] + sun_c[0] * cloud.transmittance * 0.3,
                        with_clouds[1] + sun_c[1] * cloud.transmittance * 0.3,
                        with_clouds[2] + sun_c[2] * cloud.transmittance * 0.3,
                    ]
                } else {
                    with_clouds
                };

                img.put_pixel(
                    px,
                    py,
                    Rgb([
                        tonemap(final_color[0]),
                        tonemap(final_color[1]),
                        tonemap(final_color[2]),
                    ]),
                );
            }
        }

        encoder.push_frame(&img).expect("frame");
    }

    encoder.finish().expect("finalize");
    eprintln!("  ✓ Atmosphere showcase → {path}");
    assert!(std::path::Path::new(path).exists());
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Pseudo-random star field for night sky.
fn pseudo_star(px: u32, py: u32) -> f32 {
    let h = (px as u64)
        .wrapping_mul(374761393)
        .wrapping_add((py as u64).wrapping_mul(668265263));
    let h = h ^ (h >> 13);
    let h = h.wrapping_mul(1274126177);
    let h = h ^ (h >> 16);
    if h.is_multiple_of(200) {
        (h % 100) as f32 / 100.0 * 0.15 // Sparse dim stars
    } else {
        0.0
    }
}
