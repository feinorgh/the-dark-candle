// Visual rendering tests — produce MP4 videos for manual evaluation.
//
// These tests exercise the thermal glow (Phase 9c) and time-of-day (Phase 9d)
// systems, rendering video output to `test_output/` for visual inspection.
//
// Run:
//   cargo test --test visual_rendering -- --nocapture
//
// Output files:
//   test_output/incandescence_ramp.mp4   — temperature sweep 300K → 2500K
//   test_output/time_of_day.mp4          — 24-hour day-night cycle

use std::f32::consts::FRAC_PI_2;

use image::{Rgb, RgbImage};

use the_dark_candle::data::{MaterialData, MaterialRegistry};
use the_dark_candle::diagnostics::video::FrameEncoder;
use the_dark_candle::diagnostics::visualization::{ColorMode, ViewMode, render_frame};
use the_dark_candle::lighting::DayNightConfig;
use the_dark_candle::world::meshing::incandescence_color;
use the_dark_candle::world::voxel::{MaterialId, Voxel};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn test_registry() -> MaterialRegistry {
    let mut reg = MaterialRegistry::new();
    reg.insert(MaterialData {
        id: 0,
        name: "Air".into(),
        color: [0.0, 0.0, 0.0],
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
        id: 5,
        name: "Iron".into(),
        color: [0.7, 0.55, 0.1],
        ..Default::default()
    });
    reg
}

fn idx(x: usize, y: usize, z: usize, size: usize) -> usize {
    x + z * size + y * size * size
}

/// Reinhard tone-map a single HDR channel to [0, 255].
fn tonemap(v: f32) -> u8 {
    ((v / (1.0 + v)) * 255.0).min(255.0) as u8
}

// ---------------------------------------------------------------------------
// Test 1: Incandescence Ramp Video
// ---------------------------------------------------------------------------

/// Renders a video that sweeps temperature from 300 K to 2500 K on iron voxels,
/// showing the incandescence color ramp from cold grey → dark red → cherry →
/// orange → yellow-white → HDR bloom white.
///
/// Layout: 16×16 grid where each row represents a different temperature, each
/// column is the same material. Frames advance temperature over time.
#[test]
fn incandescence_ramp_video() {
    let _ = std::fs::create_dir_all("test_output");
    let path = "test_output/incandescence_ramp.mp4";

    // Image: 512 wide × 300 tall
    let img_w = 512;
    let img_h = 300;
    let fps = 30;
    let duration_s = 6.0_f32;
    let total_frames = (duration_s * fps as f32) as u32;

    let mut encoder = FrameEncoder::new(path, img_w, img_h, fps).expect("encoder");

    let iron_base = [0.7_f32, 0.55, 0.1, 1.0];
    let stone_base = [0.5_f32, 0.5, 0.5, 1.0];

    for frame in 0..total_frames {
        let t = frame as f32 / total_frames as f32;
        let mut img = RgbImage::new(img_w, img_h);

        // Top half: iron incandescence sweep
        // Bottom half: stone incandescence sweep
        // Each pixel column represents a temperature from 300 K to 2500 K.
        for px in 0..img_w {
            // Temperature varies across width AND shifts over time
            let base_temp = 300.0 + (px as f32 / img_w as f32) * 2200.0;
            // Animate: the whole gradient slides right over time
            let offset = t * 500.0;
            let temp = (base_temp + offset).min(2800.0);

            let iron_hdr = incandescence_color(iron_base, temp);
            let stone_hdr = incandescence_color(stone_base, temp);

            // Iron band (top half)
            let iron_rgb = Rgb([
                tonemap(iron_hdr[0]),
                tonemap(iron_hdr[1]),
                tonemap(iron_hdr[2]),
            ]);
            for py in 0..(img_h / 2 - 10) {
                img.put_pixel(px, py, iron_rgb);
            }

            // Separator band
            for py in (img_h / 2 - 10)..(img_h / 2 + 10) {
                img.put_pixel(px, py, Rgb([30, 30, 30]));
            }

            // Stone band (bottom half)
            let stone_rgb = Rgb([
                tonemap(stone_hdr[0]),
                tonemap(stone_hdr[1]),
                tonemap(stone_hdr[2]),
            ]);
            for py in (img_h / 2 + 10)..img_h {
                img.put_pixel(px, py, stone_rgb);
            }
        }

        // Temperature labels (drawn as colored tick marks every 200 K)
        for temp_k in (400..=2600).step_by(200) {
            let frac = (temp_k as f32 - 300.0) / 2200.0;
            let px = ((frac * img_w as f32) as u32).min(img_w - 1);
            // White tick mark at separator
            for py in (img_h / 2 - 12)..(img_h / 2 - 8) {
                if px < img_w {
                    img.put_pixel(px, py, Rgb([255, 255, 255]));
                }
            }
        }

        encoder.push_frame(&img).expect("frame");
    }

    encoder.finish().expect("finalize");
    eprintln!("  ✓ Incandescence ramp video → {path}");

    // Verify the file was created
    assert!(
        std::path::Path::new(path).exists(),
        "Video file should be created"
    );
}

// ---------------------------------------------------------------------------
// Test 2: Incandescence Voxel Grid Video (using simulation renderer)
// ---------------------------------------------------------------------------

/// Renders a voxel grid where iron blocks at various temperatures demonstrate
/// the incandescence color mode through the existing visualization pipeline.
#[test]
fn incandescence_voxel_grid_video() {
    let _ = std::fs::create_dir_all("test_output");
    let path = "test_output/incandescence_grid.mp4";

    let size = 16;
    let scale = 8_u32;
    let dim = size as u32 * scale;
    let fps = 30;
    let total_frames = 120; // 4 seconds

    let registry = test_registry();
    let mut encoder = FrameEncoder::new(path, dim, dim, fps).expect("encoder");

    for frame in 0..total_frames {
        let t = frame as f32 / total_frames as f32;

        // Build a voxel grid where each column has a different temperature
        let mut voxels = vec![Voxel::default(); size * size * size];

        // Fill y=0..7 with iron at varying temperatures
        for x in 0..size {
            for z in 0..size {
                // Temperature varies by column: 300 K → 2500 K
                let temp = 300.0 + (x as f32 / size as f32) * 2200.0;
                // Animate: heat slowly increases over time
                let animated_temp = (temp + t * 400.0).min(2800.0);

                for y in 0..8 {
                    let i = idx(x, y, z, size);
                    voxels[i].material = MaterialId(5); // iron
                    voxels[i].temperature = animated_temp;
                }
            }
        }

        let frame_img = render_frame(
            &voxels,
            size,
            &registry,
            &ViewMode::Slice {
                axis: the_dark_candle::diagnostics::visualization::SliceAxis::Y,
                depth: 4,
            },
            &ColorMode::Incandescence,
            scale,
        );
        encoder.push_frame(&frame_img).expect("frame");
    }

    encoder.finish().expect("finalize");
    eprintln!("  ✓ Incandescence voxel grid video → {path}");
    assert!(std::path::Path::new(path).exists());
}

// ---------------------------------------------------------------------------
// Test 3: Time-of-Day Cycle Video
// ---------------------------------------------------------------------------

/// Sun elevation angle from time of day (mirrors src/lighting/mod.rs).
fn sun_elevation(hour: f32) -> f32 {
    use std::f32::consts::PI;
    let day_fraction = (hour - 6.0) / 12.0;
    (day_fraction * PI).sin() * FRAC_PI_2
}

/// Sun color from elevation (mirrors src/lighting/mod.rs).
fn sun_color_rgb(elevation: f32) -> Rgb<u8> {
    let (r, g, b) = if elevation <= 0.0 {
        (0.3_f32, 0.35, 0.5) // night
    } else if elevation < 0.3 {
        let t = elevation / 0.3;
        (1.0, 0.7 + 0.3 * t, 0.4 + 0.55 * t)
    } else {
        (1.0, 1.0, 0.95) // noon
    };
    Rgb([(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8])
}

/// Ambient brightness factor from elevation (mirrors src/lighting/mod.rs).
fn ambient_factor(elevation: f32, config: &DayNightConfig) -> f32 {
    let factor = elevation.sin().max(0.0);
    config.night_ambient + (config.noon_ambient - config.night_ambient) * factor
}

/// Renders a video showing the full 24-hour day-night cycle.
///
/// Layout (512×400):
///   - Top band (100px): sky color gradient (ambient light color)
///   - Middle band (120px): sun disk position and color
///   - Bottom panel (180px): data chart — elevation, illuminance, color temp
///
/// The video runs 24 game-hours at accelerated pace (12 seconds real-time).
#[test]
fn time_of_day_cycle_video() {
    let _ = std::fs::create_dir_all("test_output");
    let path = "test_output/time_of_day.mp4";

    let img_w: u32 = 512;
    let img_h: u32 = 400;
    let fps: u32 = 30;
    let duration_s = 12.0_f32;
    let total_frames = (duration_s * fps as f32) as u32;

    let config = DayNightConfig::default();
    let mut encoder = FrameEncoder::new(path, img_w, img_h, fps).expect("encoder");

    // Pre-define regions
    let sky_h = 100_u32; // sky color band
    let sun_band_top = sky_h;
    let sun_band_h = 120_u32; // sun position area
    let chart_top = sun_band_top + sun_band_h;

    for frame in 0..total_frames {
        let t = frame as f32 / total_frames as f32;
        let hour = t * 24.0; // 0..24 over the video

        let elevation = sun_elevation(hour);
        let sun_color = sun_color_rgb(elevation);
        let brightness_factor = ambient_factor(elevation, &config);
        let brightness_norm = brightness_factor / config.noon_ambient;

        let mut img = RgbImage::new(img_w, img_h);

        // --- Sky band ---
        // Color represents ambient light, brightness scales with time
        let sky_color = if elevation <= 0.0 {
            let f = 0.05 + brightness_norm * 0.2;
            Rgb([(30.0 * f) as u8, (35.0 * f) as u8, (60.0 * f) as u8])
        } else if elevation < 0.3 {
            let t_blend = elevation / 0.3;
            Rgb([
                (120.0 + 135.0 * t_blend) as u8,
                (100.0 + 130.0 * t_blend) as u8,
                (80.0 + 160.0 * t_blend) as u8,
            ])
        } else {
            // Daytime sky blue
            let intensity = brightness_norm.min(1.0);
            Rgb([
                (135.0 * intensity) as u8,
                (206.0 * intensity) as u8,
                (235.0 * intensity) as u8,
            ])
        };

        for py in 0..sky_h {
            for px in 0..img_w {
                img.put_pixel(px, py, sky_color);
            }
        }

        // --- Sun position band ---
        // Dark background with sun disk at position based on azimuth
        let bg_night = Rgb([10, 10, 20]);
        for py in sun_band_top..(sun_band_top + sun_band_h) {
            for px in 0..img_w {
                img.put_pixel(px, py, bg_night);
            }
        }

        // Sun disk: horizontal position = time of day, vertical = elevation
        let sun_x = (t * img_w as f32) as i32;
        let sun_y = if elevation > 0.0 {
            let e_norm = elevation / FRAC_PI_2; // 0..1
            sun_band_top as i32 + sun_band_h as i32 - (e_norm * (sun_band_h as f32 - 20.0)) as i32
        } else {
            // Below horizon — draw at bottom of band, dimmed
            sun_band_top as i32 + sun_band_h as i32 - 10
        };

        let sun_radius = if elevation > 0.0 { 15_i32 } else { 8 };
        for dy in -sun_radius..=sun_radius {
            for dx in -sun_radius..=sun_radius {
                if dx * dx + dy * dy <= sun_radius * sun_radius {
                    let px = sun_x + dx;
                    let py = sun_y + dy;
                    if px >= 0 && (px as u32) < img_w && py >= 0 && (py as u32) < img_h {
                        img.put_pixel(px as u32, py as u32, sun_color);
                    }
                }
            }
        }

        // Horizon line
        let horizon_y = sun_band_top + sun_band_h - 10;
        if horizon_y < img_h {
            for px in 0..img_w {
                img.put_pixel(px, horizon_y, Rgb([80, 80, 80]));
            }
        }

        // --- Chart area: elevation + brightness curves ---
        // Black background
        let chart_h = img_h - chart_top;
        for py in chart_top..img_h {
            for px in 0..img_w {
                img.put_pixel(px, py, Rgb([15, 15, 15]));
            }
        }

        // Draw full-day curves as trailing pixels
        for prev_frame in 0..=frame {
            let pt = prev_frame as f32 / total_frames as f32;
            let ph = pt * 24.0;
            let pe = sun_elevation(ph);
            let px = (pt * img_w as f32) as u32;

            if px < img_w {
                // Elevation curve (green): maps [-π/2, π/2] → [chart_h, 0]
                let e_norm = (pe / FRAC_PI_2 + 1.0) / 2.0; // 0..1
                let py = chart_top + chart_h - (e_norm * chart_h as f32) as u32;
                if py < img_h {
                    img.put_pixel(px, py.min(img_h - 1), Rgb([0, 200, 0]));
                }

                // Brightness curve (yellow): 0..1 mapped to chart
                let b = ambient_factor(pe, &config) / config.noon_ambient;
                let by = chart_top + chart_h - (b.clamp(0.0, 1.0) * chart_h as f32) as u32;
                if by < img_h {
                    img.put_pixel(px, by.min(img_h - 1), Rgb([255, 220, 0]));
                }

                // Insolation curve (red): sin(elevation) when above horizon
                let ins = pe.sin().max(0.0);
                let iy = chart_top + chart_h - (ins * chart_h as f32) as u32;
                if iy < img_h {
                    img.put_pixel(px, iy.min(img_h - 1), Rgb([255, 60, 60]));
                }
            }
        }

        // Zero-line (where elevation = 0)
        let zero_y = chart_top + chart_h / 2;
        if zero_y < img_h {
            for px in 0..img_w {
                let existing = *img.get_pixel(px, zero_y);
                // Dim line unless already colored
                if existing == Rgb([15, 15, 15]) {
                    img.put_pixel(px, zero_y, Rgb([50, 50, 50]));
                }
            }
        }

        // Current time indicator (vertical white line)
        let time_x = (t * img_w as f32) as u32;
        if time_x < img_w {
            for py in chart_top..img_h {
                img.put_pixel(time_x, py, Rgb([200, 200, 200]));
            }
        }

        encoder.push_frame(&img).expect("frame");
    }

    encoder.finish().expect("finalize");
    eprintln!("  ✓ Time-of-day cycle video → {path}");
    assert!(std::path::Path::new(path).exists());
}

// ---------------------------------------------------------------------------
// Test 4: Combined Day-Night with Terrain
// ---------------------------------------------------------------------------

/// Renders a terrain-like voxel grid under day-night lighting changes,
/// showing how material colors shift with ambient light color temperature.
#[test]
fn daynight_terrain_video() {
    let _ = std::fs::create_dir_all("test_output");
    let path = "test_output/daynight_terrain.mp4";

    let size = 16;
    let scale = 8_u32;
    let dim = size as u32 * scale;
    let fps = 30;
    let duration_s = 12.0_f32;
    let total_frames = (duration_s * fps as f32) as u32;

    let registry = test_registry();
    let mut encoder = FrameEncoder::new(path, dim, dim, fps).expect("encoder");

    // Build a static terrain: stone hills
    let mut base_voxels = vec![Voxel::default(); size * size * size];
    for x in 0..size {
        for z in 0..size {
            // Simple hill: height varies with position
            let h = 4 + ((x as f32 * 0.5).sin() * 2.0 + (z as f32 * 0.7).cos() * 1.5) as usize;
            let h = h.min(size - 1);
            for y in 0..=h {
                let i = idx(x, y, z, size);
                base_voxels[i].material = MaterialId::STONE;
                base_voxels[i].temperature = 288.15;
            }
        }
    }

    for frame in 0..total_frames {
        let t = frame as f32 / total_frames as f32;
        let hour = t * 24.0;

        let elevation = sun_elevation(hour);
        let brightness_norm = {
            let config = DayNightConfig::default();
            ambient_factor(elevation, &config) / config.noon_ambient
        };

        // Modulate terrain colors by ambient brightness to simulate lighting
        let mut voxels = base_voxels.clone();
        for v in &mut voxels {
            if !v.material.is_air() {
                // Simulate ambient light effect: darker at night
                // We encode this in temperature for the visualization to pick up
                // via a creative use of Temperature color mode with adjusted range
                v.temperature = 288.15 + brightness_norm * 200.0;
            }
        }

        // Render with material colors (the brightness is baked into the
        // voxel visualization by tinting)
        let frame_img = render_frame(
            &voxels,
            size,
            &registry,
            &ViewMode::TopDown,
            &ColorMode::Temperature {
                min_k: 280.0,
                max_k: 500.0,
            },
            scale,
        );
        encoder.push_frame(&frame_img).expect("frame");
    }

    encoder.finish().expect("finalize");
    eprintln!("  ✓ Day-night terrain video → {path}");
    assert!(std::path::Path::new(path).exists());
}
