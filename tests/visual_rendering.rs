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
use the_dark_candle::diagnostics::visualization::{
    ColorMode, SceneLight, ViewMode, render_frame, render_frame_lit,
};
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
/// Smoothly blends from night blue through warm dawn/dusk to neutral noon.
fn sun_color_rgb(elevation: f32) -> Rgb<u8> {
    // Night tint
    let night = (0.3_f32, 0.35_f32, 0.5_f32);
    // Warm dawn/dusk
    let warm = (1.0_f32, 0.7_f32, 0.45_f32);
    // Noon white
    let noon = (1.0_f32, 1.0_f32, 0.95_f32);

    let (r, g, b) = if elevation <= -0.15 {
        night
    } else if elevation < 0.15 {
        // Smooth blend over ±0.15 rad (~17°) around horizon
        let t = (elevation + 0.15) / 0.3;
        (
            night.0 + (warm.0 - night.0) * t,
            night.1 + (warm.1 - night.1) * t,
            night.2 + (warm.2 - night.2) * t,
        )
    } else if elevation < 0.5 {
        // Warm dawn → neutral noon
        let t = (elevation - 0.15) / 0.35;
        (
            warm.0 + (noon.0 - warm.0) * t,
            warm.1 + (noon.1 - warm.1) * t,
            warm.2 + (noon.2 - warm.2) * t,
        )
    } else {
        noon
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
// Test 4: Combined Day-Night with 3D Terrain (Perspective Raymarcher)
// ---------------------------------------------------------------------------

/// Full material registry for terrain rendering.
fn terrain_registry() -> MaterialRegistry {
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
        ..Default::default()
    });
    reg.insert(MaterialData {
        id: 5,
        name: "Iron".into(),
        color: [0.7, 0.55, 0.1],
        ..Default::default()
    });
    reg.insert(MaterialData {
        id: 6,
        name: "Sand".into(),
        color: [0.85, 0.78, 0.55],
        ..Default::default()
    });
    reg.insert(MaterialData {
        id: 7,
        name: "Grass".into(),
        color: [0.3, 0.6, 0.2],
        ..Default::default()
    });
    reg
}

/// Build a 32³ terrain grid with Perlin-like hills, multi-material layers.
fn build_terrain(size: usize) -> Vec<Voxel> {
    let mut voxels = vec![Voxel::default(); size * size * size];

    for x in 0..size {
        for z in 0..size {
            // Multi-octave procedural height (no external crate needed).
            let fx = x as f32 / size as f32;
            let fz = z as f32 / size as f32;

            // Large-scale hills
            let h1 = ((fx * 2.5).sin() * (fz * 3.0).cos()) * 4.0;
            // Medium detail
            let h2 = ((fx * 7.0 + 1.3).sin() * (fz * 6.0 + 0.7).cos()) * 1.5;
            // Fine detail
            let h3 = ((fx * 13.0 + 2.7).sin() * (fz * 11.0 + 3.1).cos()) * 0.6;

            let base_height = (size as f32 * 0.35) + h1 + h2 + h3;
            let height = (base_height as usize).clamp(2, size - 2);

            for y in 0..=height {
                let i = idx(x, y, z, size);
                if y == height {
                    // Top layer: grass
                    voxels[i].material = MaterialId(7);
                } else if y > height.saturating_sub(3) {
                    // Soil layer: dirt
                    voxels[i].material = MaterialId(2);
                } else {
                    // Bedrock: stone
                    voxels[i].material = MaterialId::STONE;
                }
                voxels[i].temperature = 288.15;
            }
        }
    }

    voxels
}

/// Renders terrain from a 3D perspective camera under cycling day-night
/// lighting. Features: Perlin-like hills with grass/dirt/stone layers,
/// Lambertian shading, shadow casting, depth fog, and a sky background.
#[test]
fn daynight_terrain_video() {
    let _ = std::fs::create_dir_all("test_output");
    let path = "test_output/daynight_terrain.mp4";

    let size = 32;
    let img_w = 640;
    let img_h = 480;
    let fps = 30;
    let duration_s = 30.0_f32;
    let total_frames = (duration_s * fps as f32) as u32;

    let registry = terrain_registry();
    let voxels = build_terrain(size);
    let mut encoder = FrameEncoder::new(path, img_w, img_h, fps).expect("encoder");

    // Camera orbits the terrain, looking at the center.
    let center = size as f32 / 2.0;
    let cam_radius = size as f32 * 1.1;
    let cam_height = size as f32 * 0.75;

    // Start at pre-dawn (hour 4), run through a full 24-hour cycle.
    // 30 seconds gives ~7.5 s for sunrise→noon, much more gradual.
    let start_hour = 4.0_f32;

    for frame in 0..total_frames {
        let t = frame as f32 / total_frames as f32;
        let hour = (start_hour + t * 24.0) % 24.0;

        // Camera slowly orbits (one full revolution per video).
        let angle = t * std::f32::consts::TAU;
        let cam_x = center + cam_radius * angle.cos();
        let cam_z = center + cam_radius * angle.sin();

        let view = ViewMode::Perspective {
            eye: (cam_x, cam_height, cam_z),
            target: (center, center * 0.4, center),
            fov_degrees: 55.0,
            width: img_w,
            height: img_h,
        };

        // Sun direction follows time-of-day (elevation + azimuth).
        let elevation = sun_elevation(hour);
        let azimuth = hour / 24.0 * std::f32::consts::TAU;
        let sun_y = -elevation.sin();
        let sun_xz = elevation.cos();
        let sun_x = -sun_xz * azimuth.cos();
        let sun_z = -sun_xz * azimuth.sin();

        // Sun color from elevation
        let sun_rgb = sun_color_rgb(elevation);
        let sc = (
            sun_rgb.0[0] as f32 / 255.0,
            sun_rgb.0[1] as f32 / 255.0,
            sun_rgb.0[2] as f32 / 255.0,
        );

        // Ambient: generous floor so night isn't pitch black, scales up with
        // sun elevation for a visible but subtle day/night difference.
        let sun_factor = elevation.sin().max(0.0); // 0 at horizon, 1 at zenith
        let amb = 0.15 + sun_factor * 0.15; // night=0.15, noon=0.30

        // Sun intensity follows sin(elevation) for a smooth, physically
        // motivated curve — gradual brightening at dawn, peak at noon,
        // gradual dimming at dusk. No hard cutoffs.
        let sun_intensity = sun_factor; // 0.0 at horizon → 1.0 at zenith

        let light = SceneLight {
            direction: (sun_x, sun_y, sun_z),
            color: sc,
            intensity: sun_intensity,
            ambient: amb,
        };

        let frame_img = render_frame_lit(
            &voxels,
            size,
            &registry,
            &view,
            &ColorMode::Material,
            1,
            &light,
        );
        encoder.push_frame(&frame_img).expect("frame");
    }

    encoder.finish().expect("finalize");
    eprintln!("  ✓ Day-night 3D terrain video → {path}");
    assert!(std::path::Path::new(path).exists());
}

// ---------------------------------------------------------------------------
// Test 5: Optics — Colored Light Through Transparent Media
// ---------------------------------------------------------------------------

/// Registry with optical properties for transparent materials.
fn optics_registry() -> MaterialRegistry {
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
        color: [0.55, 0.55, 0.55],
        ..Default::default()
    });
    reg.insert(MaterialData {
        id: 3,
        name: "Water".into(),
        color: [0.2, 0.45, 0.85],
        transparent: true,
        absorption_rgb: Some([0.45, 0.07, 0.02]),
        ..Default::default()
    });
    reg.insert(MaterialData {
        id: 12,
        name: "Glass".into(),
        color: [0.85, 0.92, 0.88],
        transparent: true,
        absorption_rgb: Some([0.05, 0.03, 0.04]),
        ..Default::default()
    });
    reg.insert(MaterialData {
        id: 2,
        name: "Dirt".into(),
        color: [0.45, 0.32, 0.18],
        ..Default::default()
    });
    reg.insert(MaterialData {
        id: 7,
        name: "Grass".into(),
        color: [0.3, 0.6, 0.2],
        ..Default::default()
    });
    reg
}

/// Build a 32³ scene with transparent columns on a terrain floor:
/// - Water column (5 voxels tall)
/// - Glass column (5 voxels tall)
/// - Open-air reference area
/// - Stone floor with grass top layer
fn build_optics_scene(size: usize) -> Vec<Voxel> {
    let mut voxels = vec![Voxel::default(); size * size * size];

    // Flat stone floor with grass top at y=8
    for x in 0..size {
        for z in 0..size {
            for y in 0..8 {
                let i = idx(x, y, z, size);
                voxels[i].material = MaterialId::STONE;
            }
            let i = idx(x, 8, z, size);
            voxels[i].material = MaterialId(7); // Grass
        }
    }

    // Water column: x=6..12, z=6..12, y=9..14 (5 voxels tall)
    for x in 6..12 {
        for z in 6..12 {
            for y in 9..14 {
                let i = idx(x, y, z, size);
                voxels[i].material = MaterialId::WATER;
            }
        }
    }

    // Glass column: x=18..24, z=6..12, y=9..14 (5 voxels tall)
    for x in 18..24 {
        for z in 6..12 {
            for y in 9..14 {
                let i = idx(x, y, z, size);
                voxels[i].material = MaterialId(12); // Glass
            }
        }
    }

    voxels
}

/// Renders a video showing light passing through transparent media:
/// - Water column produces blue-tinted shadows (red absorbed)
/// - Glass column produces nearly neutral shadows
/// - Camera orbits the scene under noon sun with Rayleigh sky
#[test]
fn optics_colored_shadows_video() {
    let _ = std::fs::create_dir_all("test_output");
    let path = "test_output/optics_colored_shadows.mp4";

    let size = 32;
    let img_w = 640;
    let img_h = 480;
    let fps = 30;
    let duration_s = 12.0_f32;
    let total_frames = (duration_s * fps as f32) as u32;

    let registry = optics_registry();
    let voxels = build_optics_scene(size);
    let mut encoder = FrameEncoder::new(path, img_w, img_h, fps).expect("encoder");

    let center = size as f32 / 2.0;
    let cam_radius = size as f32 * 1.3;
    let cam_height = size as f32 * 0.7;

    for frame in 0..total_frames {
        let t = frame as f32 / total_frames as f32;

        // Camera slowly orbits the scene.
        let angle = t * std::f32::consts::TAU;
        let cam_x = center + cam_radius * angle.cos();
        let cam_z = center + cam_radius * angle.sin();

        let view = ViewMode::Perspective {
            eye: (cam_x, cam_height, cam_z),
            target: (center, center * 0.35, center),
            fov_degrees: 55.0,
            width: img_w,
            height: img_h,
        };

        // High sun (noon-ish) — good for showing shadows below columns.
        // Sun sweeps slightly across the sky for shadow movement.
        let sun_elevation = 1.0 + 0.2 * (t * std::f32::consts::PI).sin();
        let sun_azimuth = 0.5 + t * 0.8;
        let sun_y = -sun_elevation.sin();
        let sun_xz = sun_elevation.cos();
        let sun_x = -sun_xz * sun_azimuth.cos();
        let sun_z = -sun_xz * sun_azimuth.sin();

        let light = SceneLight {
            direction: (sun_x, sun_y, sun_z),
            color: (1.0, 1.0, 0.95),
            intensity: 1.0,
            ambient: 0.2,
        };

        let frame_img = render_frame_lit(
            &voxels,
            size,
            &registry,
            &view,
            &ColorMode::Material,
            1,
            &light,
        );
        encoder.push_frame(&frame_img).expect("frame");
    }

    encoder.finish().expect("finalize");
    eprintln!("  ✓ Optics colored shadows video → {path}");
    assert!(std::path::Path::new(path).exists());
}
