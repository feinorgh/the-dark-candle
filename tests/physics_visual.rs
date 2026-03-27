// Physics visualization tests — produce MP4 videos of simulation phenomena.
//
// These tests exercise the simulation pipeline (chemical reactions, heat
// diffusion, radiation, state transitions) combined with the software
// raymarching renderer, producing video output for visual inspection.
//
// Run:
//   cargo test --test physics_visual -- --nocapture
//
// Output files:
//   test_output/fire_propagation.mp4   — wood/grass fire spreading
//   test_output/lava_phase.mp4         — stone melting into glowing lava
//   test_output/water_boiling.mp4      — water heated to steam
//   test_output/oxyhydrogen_flame.mp4  — H₂/O₂ detonation chain reaction

use the_dark_candle::chemistry::reactions::ReactionData;
use the_dark_candle::chemistry::runtime::load_reaction_rules;
use the_dark_candle::data::{MaterialRegistry, load_material_registry};
use the_dark_candle::diagnostics::video::FrameEncoder;
use the_dark_candle::diagnostics::visualization::{
    ColorMode, SceneLight, ViewMode, render_frame_lit,
};
use the_dark_candle::simulation::simulate_tick;
use the_dark_candle::world::voxel::{MaterialId, Voxel};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn idx(x: usize, y: usize, z: usize, size: usize) -> usize {
    z * size * size + y * size + x
}

/// Standard noon-ish sun from upper right.
fn noon_light() -> SceneLight {
    SceneLight {
        direction: (-0.5, -0.8, -0.3),
        color: (1.0, 0.98, 0.92),
        intensity: 0.9,
        ambient: 0.2,
    }
}

/// Standard camera looking at center of a grid, positioned above and away.
fn orbit_camera(size: usize, angle: f32, elevation: f32, img_w: u32, img_h: u32) -> ViewMode {
    let center = size as f32 / 2.0;
    let radius = size as f32 * 1.2;
    let cam_x = center + radius * angle.cos() * elevation.cos();
    let cam_y = center * 0.6 + radius * elevation.sin();
    let cam_z = center + radius * angle.sin() * elevation.cos();
    ViewMode::Perspective {
        eye: (cam_x, cam_y, cam_z),
        target: (center, center * 0.4, center),
        fov_degrees: 55.0,
        width: img_w,
        height: img_h,
    }
}

// ---------------------------------------------------------------------------
// Test 1: Fire Propagation
// ---------------------------------------------------------------------------

/// Builds a 32³ scene with a wooden structure on a grass floor, surrounded
/// by air. Wood pillars and beams form a small cabin. The wood is pre-heated
/// with a vertical gradient — lower layers above ignition temperature (573K)
/// react to ash immediately, creating incandescent glow from the bottom up.
fn build_fire_scene(size: usize, registry: &MaterialRegistry) -> Vec<Voxel> {
    let mut voxels = vec![Voxel::default(); size * size * size];
    let ambient = 288.15; // 15°C

    // Set all voxels to ambient temp (air)
    for v in voxels.iter_mut() {
        v.temperature = ambient;
    }

    // Grass floor at y=0..1
    for x in 0..size {
        for z in 0..size {
            let i = idx(x, 0, z, size);
            voxels[i].material = MaterialId::GRASS; // Grass
        }
    }

    // Stone base layer at y=1 under the cabin
    for x in 8..24 {
        for z in 8..24 {
            let i = idx(x, 1, z, size);
            voxels[i].material = MaterialId::STONE;
        }
    }

    // Wood pillars at corners: 4 pillars from y=2 to y=8
    let pillars = [(9, 9), (9, 22), (22, 9), (22, 22)];
    for &(px, pz) in &pillars {
        for y in 2..9 {
            let i = idx(px, y, pz, size);
            voxels[i].material = MaterialId::WOOD; // Wood
        }
    }

    // Wood beams connecting pillars at y=8 (roof frame)
    for x in 9..23 {
        let i1 = idx(x, 8, 9, size);
        let i2 = idx(x, 8, 22, size);
        voxels[i1].material = MaterialId::WOOD;
        voxels[i2].material = MaterialId::WOOD;
    }
    for z in 9..23 {
        let i1 = idx(9, 8, z, size);
        let i2 = idx(22, 8, z, size);
        voxels[i1].material = MaterialId::WOOD;
        voxels[i2].material = MaterialId::WOOD;
    }

    // Wood walls (partial — leave gaps for windows)
    for y in 2..8 {
        // Front wall (z=9): wood except window at x=14..18
        for x in 9..23 {
            if !(14..18).contains(&x) || !(4..=6).contains(&y) {
                let i = idx(x, y, 9, size);
                voxels[i].material = MaterialId::WOOD;
            }
        }
        // Back wall (z=22): solid wood
        for x in 9..23 {
            let i = idx(x, y, 22, size);
            voxels[i].material = MaterialId::WOOD;
        }
        // Side walls (x=9, x=22): wood except window
        for z in 9..23 {
            if !(14..18).contains(&z) || !(4..=6).contains(&y) {
                let i = idx(9, y, z, size);
                voxels[i].material = MaterialId::WOOD;
            }
            let i = idx(22, y, z, size);
            voxels[i].material = MaterialId::WOOD;
        }
    }

    // Pre-heat the cabin wood with a vertical gradient simulating a fire
    // already in progress. Bottom layers are above wood ignition (573K) so
    // they react immediately to ash; upper layers stay as warm wood.
    for x in 0..size {
        for z in 0..size {
            for y in 0..size {
                let i = idx(x, y, z, size);
                if voxels[i].material == MaterialId::WOOD {
                    // Gradient: y=2 → 700K, y=8 → 400K
                    let frac = ((y as f32) - 2.0) / 6.0;
                    voxels[i].temperature = 700.0 - frac * 300.0;
                }
            }
        }
    }

    // Extra-hot ignition point at the back-right wall corner to seed the
    // brightest glow. This is ON the wood wall (x=22), not inside the cabin.
    for y in 2..5 {
        for z in 18..23 {
            let i = idx(22, y, z, size);
            if voxels[i].material == MaterialId::WOOD {
                voxels[i].temperature = 800.0;
            }
        }
    }

    // Verify air is transparent in registry
    if let Some(air) = registry.get(MaterialId::AIR) {
        assert!(air.transparent, "Air must be transparent for rendering");
    }

    voxels
}

/// Fire spreading through a wooden cabin. Ignition at one corner, flames
/// propagate through wood walls and beams. Rendered with incandescence
/// color mode to show thermal glow.
#[test]
fn fire_propagation_video() {
    let _ = std::fs::create_dir_all("test_output");
    let path = "test_output/fire_propagation.mp4";

    let size: usize = 32;
    let img_w: u32 = 640;
    let img_h: u32 = 480;
    let fps: u32 = 30;
    let duration_s = 20.0_f32;
    let total_frames = (duration_s * fps as f32) as u32;
    let ticks_per_frame = 4;
    let dt = 0.5;

    let registry = load_material_registry().expect("material registry");
    let rules = load_reaction_rules().expect("reaction rules");
    let mut voxels = build_fire_scene(size, &registry);
    let mut encoder = FrameEncoder::new(path, img_w, img_h, fps).expect("encoder");

    for frame in 0..total_frames {
        let t = frame as f32 / total_frames as f32;

        // Run physics
        for _ in 0..ticks_per_frame {
            simulate_tick(&mut voxels, size, &rules.0, &registry, dt);
        }

        // Camera slowly orbits, rising slightly over time
        let angle = std::f32::consts::FRAC_PI_4 + t * std::f32::consts::TAU * 0.6;
        let elev = 0.35 + t * 0.1;
        let view = orbit_camera(size, angle, elev, img_w, img_h);
        let light = noon_light();

        let frame_img = render_frame_lit(
            &voxels,
            size,
            &registry,
            &view,
            &ColorMode::Incandescence,
            1,
            &light,
        );
        encoder.push_frame(&frame_img).expect("frame");
    }

    encoder.finish().expect("finalize");
    eprintln!("  ✓ Fire propagation video → {path}");
    assert!(std::path::Path::new(path).exists());
}

// ---------------------------------------------------------------------------
// Test 2: Lava Phase Transition
// ---------------------------------------------------------------------------

/// Builds a 32³ scene with a stone mountain heated from below. The bottom
/// layers are pre-heated close to the melting point (1673K), with a central
/// magma chamber already at melting temperature.
fn build_lava_scene(size: usize) -> Vec<Voxel> {
    let mut voxels = vec![Voxel::default(); size * size * size];

    // Set ambient temperature
    for v in voxels.iter_mut() {
        v.temperature = 288.15;
    }

    // Stone mountain: conical shape, peak at center
    let center = size as f32 / 2.0;
    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 - center;
                let dz = z as f32 - center;
                let dist = (dx * dx + dz * dz).sqrt();
                // Height of cone at this distance from center
                let max_y = (size as f32 * 0.7 - dist * 0.9).max(2.0);
                if (y as f32) < max_y {
                    let i = idx(x, y, z, size);
                    voxels[i].material = MaterialId::STONE;

                    // Temperature gradient: hotter deeper down
                    let depth = max_y - y as f32;
                    let depth_fraction = depth / max_y;
                    // Bottom: 1500K, top: 400K
                    voxels[i].temperature = 400.0 + depth_fraction * 1100.0;
                }
            }
        }
    }

    // Central magma chamber: sphere at (center, 4, center), radius 4
    // Temperature above melting point to trigger stone→lava
    let chamber_y = 4.0_f32;
    let chamber_r = 4.0_f32;
    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 - center;
                let dy = y as f32 - chamber_y;
                let dz = z as f32 - center;
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                if dist < chamber_r {
                    let i = idx(x, y, z, size);
                    // Above melting point — will transition to lava
                    voxels[i].temperature = 1800.0;
                }
            }
        }
    }

    voxels
}

/// Stone mountain with internal magma chamber. Heat diffuses upward,
/// melting stone into glowing lava which slowly expands. Rendered with
/// incandescence to show the molten glow through the rock.
#[test]
fn lava_phase_transition_video() {
    let _ = std::fs::create_dir_all("test_output");
    let path = "test_output/lava_phase.mp4";

    let size: usize = 32;
    let img_w: u32 = 640;
    let img_h: u32 = 480;
    let fps: u32 = 30;
    let duration_s = 20.0_f32;
    let total_frames = (duration_s * fps as f32) as u32;
    let ticks_per_frame = 3;
    let dt = 1.0;

    let registry = load_material_registry().expect("material registry");
    // No combustion reactions needed — this is purely state transitions
    let rules: Vec<ReactionData> = Vec::new();
    let mut voxels = build_lava_scene(size);
    let mut encoder = FrameEncoder::new(path, img_w, img_h, fps).expect("encoder");

    // Dim warm light to emphasize incandescent glow
    let light = SceneLight {
        direction: (-0.4, -0.7, -0.5),
        color: (0.9, 0.85, 0.75),
        intensity: 0.4,
        ambient: 0.1,
    };

    for frame in 0..total_frames {
        let t = frame as f32 / total_frames as f32;

        for _ in 0..ticks_per_frame {
            simulate_tick(&mut voxels, size, &rules, &registry, dt);
        }

        // Camera orbits with slight elevation change
        let angle = t * std::f32::consts::TAU * 0.5;
        let elev = 0.3 + 0.1 * (t * std::f32::consts::TAU).sin();
        let view = orbit_camera(size, angle, elev, img_w, img_h);

        let frame_img = render_frame_lit(
            &voxels,
            size,
            &registry,
            &view,
            &ColorMode::Incandescence,
            1,
            &light,
        );
        encoder.push_frame(&frame_img).expect("frame");
    }

    encoder.finish().expect("finalize");
    eprintln!("  ✓ Lava phase transition video → {path}");
    assert!(std::path::Path::new(path).exists());
}

// ---------------------------------------------------------------------------
// Test 3: Water Boiling
// ---------------------------------------------------------------------------

/// Builds a 32³ scene with a stone cauldron filled with water, heated from
/// below by a hot stone plate.
fn build_boiling_scene(size: usize) -> Vec<Voxel> {
    let mut voxels = vec![Voxel::default(); size * size * size];

    for v in voxels.iter_mut() {
        v.temperature = 288.15;
    }

    // Stone floor at y=0..3
    for x in 0..size {
        for z in 0..size {
            for y in 0..3 {
                let i = idx(x, y, z, size);
                voxels[i].material = MaterialId::STONE;
            }
        }
    }

    // Stone cauldron walls: ring from y=3 to y=12
    let center = size as f32 / 2.0;
    let outer_r = 10.0_f32;
    let inner_r = 8.0_f32;
    for y in 3..13 {
        for x in 0..size {
            for z in 0..size {
                let dx = x as f32 - center;
                let dz = z as f32 - center;
                let dist = (dx * dx + dz * dz).sqrt();
                let i = idx(x, y, z, size);
                if dist < outer_r && dist >= inner_r {
                    // Wall
                    voxels[i].material = MaterialId::STONE;
                } else if dist < inner_r && y == 3 {
                    // Floor of cauldron
                    voxels[i].material = MaterialId::STONE;
                }
            }
        }
    }

    // Water fills cauldron interior from y=4 to y=10
    for y in 4..11 {
        for x in 0..size {
            for z in 0..size {
                let dx = x as f32 - center;
                let dz = z as f32 - center;
                let dist = (dx * dx + dz * dz).sqrt();
                if dist < inner_r {
                    let i = idx(x, y, z, size);
                    voxels[i].material = MaterialId::WATER;
                    // Water starts warm (80°C = 353K) — close to boiling
                    voxels[i].temperature = 353.0;
                }
            }
        }
    }

    // Hot plate below cauldron floor: y=2 heated to 500K
    for x in 0..size {
        for z in 0..size {
            let dx = x as f32 - center;
            let dz = z as f32 - center;
            let dist = (dx * dx + dz * dz).sqrt();
            if dist < inner_r {
                let i = idx(x, 2, z, size);
                voxels[i].temperature = 500.0;
            }
        }
    }

    voxels
}

/// Water heated in a stone cauldron until it boils. Steam rises from
/// the surface as temperature crosses 373K. Temperature heatmap shows
/// the thermal gradient.
#[test]
fn water_boiling_video() {
    let _ = std::fs::create_dir_all("test_output");
    let path = "test_output/water_boiling.mp4";

    let size: usize = 32;
    let img_w: u32 = 640;
    let img_h: u32 = 480;
    let fps: u32 = 30;
    let duration_s = 20.0_f32;
    let total_frames = (duration_s * fps as f32) as u32;
    let ticks_per_frame = 5;
    let dt = 0.5;

    let registry = load_material_registry().expect("material registry");
    let rules: Vec<ReactionData> = Vec::new();
    let mut voxels = build_boiling_scene(size);
    let mut encoder = FrameEncoder::new(path, img_w, img_h, fps).expect("encoder");

    for frame in 0..total_frames {
        let t = frame as f32 / total_frames as f32;

        // Maintain heat source (re-heat bottom plate each frame to simulate
        // continuous fire under the cauldron)
        let center = size as f32 / 2.0;
        for x in 0..size {
            for z in 0..size {
                let dx = x as f32 - center;
                let dz = z as f32 - center;
                let dist = (dx * dx + dz * dz).sqrt();
                if dist < 8.0 {
                    let i = idx(x, 2, z, size);
                    voxels[i].temperature = 500.0_f32.max(voxels[i].temperature);
                }
            }
        }

        for _ in 0..ticks_per_frame {
            simulate_tick(&mut voxels, size, &rules, &registry, dt);
        }

        // Camera orbits slowly
        let angle = std::f32::consts::FRAC_PI_4 + t * std::f32::consts::PI * 0.5;
        let elev = 0.35;
        let view = orbit_camera(size, angle, elev, img_w, img_h);
        let light = noon_light();

        // Use temperature color mode to show thermal gradient
        let color = ColorMode::Temperature {
            min_k: 288.0,
            max_k: 500.0,
        };

        let frame_img = render_frame_lit(&voxels, size, &registry, &view, &color, 1, &light);
        encoder.push_frame(&frame_img).expect("frame");
    }

    encoder.finish().expect("finalize");
    eprintln!("  ✓ Water boiling video → {path}");
    assert!(std::path::Path::new(path).exists());
}

// ---------------------------------------------------------------------------
// Test 4: Oxyhydrogen Flame
// ---------------------------------------------------------------------------

/// Builds a 32³ scene with a checkerboard of hydrogen and oxygen gas,
/// contained in a stone chamber. The entire gas volume is pre-heated above
/// auto-ignition temperature (843K), causing a uniform detonation on the
/// first simulation tick.
fn build_oxyhydrogen_scene(size: usize) -> Vec<Voxel> {
    let mut voxels = vec![Voxel::default(); size * size * size];

    for v in voxels.iter_mut() {
        v.temperature = 288.15;
    }

    // Stone containment walls: 2-voxel thick shell
    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let i = idx(x, y, z, size);
                if x < 2 || x >= size - 2 || y < 2 || z < 2 || z >= size - 2 {
                    voxels[i].material = MaterialId::STONE;
                }
            }
        }
    }

    // Checkerboard H₂/O₂ fill interior (y=2..size, x=2..size-2, z=2..size-2)
    // Leave top open (no stone ceiling) so steam can escape.
    // Pre-heat the entire gas volume ABOVE auto-ignition (843K) to model a
    // detonation: all H₂ reacts on the very first tick, producing a massive
    // thermal flash that fills the chamber with superheated steam.
    for z in 2..size - 2 {
        for y in 2..size {
            for x in 2..size - 2 {
                let i = idx(x, y, z, size);
                if (x + y + z) % 2 == 0 {
                    voxels[i].material = MaterialId::HYDROGEN;
                } else {
                    voxels[i].material = MaterialId::OXYGEN;
                }
                voxels[i].temperature = 850.0;
            }
        }
    }

    // No separate ignition hot spot needed — the entire gas is above
    // auto-ignition temperature. The detonation fires uniformly.

    voxels
}

/// Oxyhydrogen detonation: H₂ and O₂ in a checkerboard pattern with a
/// central ignition point. The chain reaction produces a ~3073K white
/// flame that rapidly propagates outward. Rendered with incandescence
/// to show the extreme thermal glow.
#[test]
fn oxyhydrogen_flame_video() {
    let _ = std::fs::create_dir_all("test_output");
    let path = "test_output/oxyhydrogen_flame.mp4";

    let size: usize = 32;
    let img_w: u32 = 640;
    let img_h: u32 = 480;
    let fps: u32 = 30;
    let duration_s = 15.0_f32;
    let total_frames = (duration_s * fps as f32) as u32;
    let ticks_per_frame = 2;
    let dt = 0.25;

    let registry = load_material_registry().expect("material registry");
    let rules = load_reaction_rules().expect("reaction rules");
    let mut voxels = build_oxyhydrogen_scene(size);
    let mut encoder = FrameEncoder::new(path, img_w, img_h, fps).expect("encoder");

    // Very dim ambient — let the flame be the light source
    let light = SceneLight {
        direction: (-0.3, -0.9, -0.2),
        color: (0.6, 0.6, 0.7),
        intensity: 0.2,
        ambient: 0.05,
    };

    for frame in 0..total_frames {
        let t = frame as f32 / total_frames as f32;

        for _ in 0..ticks_per_frame {
            simulate_tick(&mut voxels, size, &rules.0, &registry, dt);
        }

        // Camera orbits faster to capture the rapid reaction, with higher
        // elevation to look down into the open-top chamber.
        let angle = t * std::f32::consts::TAU * 0.8;
        let elev = 0.55 + 0.1 * (t * std::f32::consts::TAU * 2.0).sin();
        let view = orbit_camera(size, angle, elev, img_w, img_h);

        let frame_img = render_frame_lit(
            &voxels,
            size,
            &registry,
            &view,
            &ColorMode::Incandescence,
            1,
            &light,
        );
        encoder.push_frame(&frame_img).expect("frame");
    }

    encoder.finish().expect("finalize");
    eprintln!("  ✓ Oxyhydrogen flame video → {path}");
    assert!(std::path::Path::new(path).exists());
}
