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
use the_dark_candle::procgen::tree::{TreeConfig, generate_tree};
use the_dark_candle::simulation::{simulate_tick, simulate_tick_dx};
use the_dark_candle::world::voxel::{MaterialId, Voxel};
use the_dark_candle::world::voxel_access::octree_to_flat;

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

// ---------------------------------------------------------------------------
// 5. Burning forest — fire spreading from a central blaze to surrounding trees
// ---------------------------------------------------------------------------

/// Simple deterministic RNG for tree placement (avoids pulling in rand crate).
struct SimpleRng(u64);

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn next_f32(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Build a forest scene: 8 trees in a ring around a central fire with ground
/// litter trails connecting the bonfire to each tree for conduction-based spread.
fn build_forest_fire_scene(config: &TreeConfig, scene_size: usize, depth: u32) -> Vec<Voxel> {
    let scale = 1usize << depth;
    let hi_size = scene_size * scale;
    let ambient = config.ambient_temperature;

    let air = Voxel {
        material: MaterialId::AIR,
        temperature: ambient,
        pressure: 101_325.0,
        damage: 0.0,
        latent_heat_buffer: 0.0,
    };
    let mut scene = vec![air; hi_size * hi_size * hi_size];

    // Ground layer at y=0: dirt.
    let dirt = Voxel {
        material: MaterialId::DIRT,
        temperature: ambient,
        ..air
    };
    for z in 0..hi_size {
        for x in 0..hi_size {
            scene[z * hi_size * hi_size + x] = dirt;
        }
    }

    // Generate one tree template at depth=0 (base resolution).
    let tree_grid = (config.trunk_height * 2.0).ceil() as usize + 4;
    let tree_grid = tree_grid.next_power_of_two().max(8);
    let tree_config_base = TreeConfig {
        octree_depth: 0,
        ..*config
    };

    // Place 8 trees close to center for fire to spread via ground litter.
    let center = hi_size as f32 / 2.0;
    let ring_radius = hi_size as f32 * 0.22;
    let mut rng = SimpleRng::new(12345);

    let tree_positions: Vec<(i32, i32)> = (0..8)
        .map(|i| {
            let angle = (i as f32 / 8.0) * std::f32::consts::TAU + (rng.next_f32() - 0.5) * 0.25;
            let r = ring_radius * (0.9 + rng.next_f32() * 0.2);
            let x = (center + r * angle.cos()) as i32;
            let z = (center + r * angle.sin()) as i32;
            (x, z)
        })
        .collect();

    // Stamp each tree into the scene.
    for (i, &(tx, tz)) in tree_positions.iter().enumerate() {
        let mut tree_cfg = tree_config_base.clone();
        let seed = 1000 + i as u64 * 7;
        let mut trng = SimpleRng::new(seed);
        tree_cfg.trunk_height *= 0.8 + trng.next_f32() * 0.4;
        tree_cfg.trunk_radius *= 0.85 + trng.next_f32() * 0.3;

        let tree_octree = generate_tree(&tree_cfg, tree_grid);
        let tree_flat = octree_to_flat(&tree_octree, tree_grid);

        for tz_t in 0..tree_grid {
            for ty_t in 0..tree_grid {
                for tx_t in 0..tree_grid {
                    let ti = tz_t * tree_grid * tree_grid + ty_t * tree_grid + tx_t;
                    if tree_flat[ti].material.is_air() {
                        continue;
                    }
                    let half = tree_grid as i32 / 2;
                    for dz in 0..scale {
                        for dy in 0..scale {
                            for ddx in 0..scale {
                                let sx = tx + (tx_t as i32 - half) * scale as i32 + ddx as i32;
                                let sy = scale as i32 + ty_t as i32 * scale as i32 + dy as i32;
                                let sz = tz + (tz_t as i32 - half) * scale as i32 + dz as i32;
                                if sx >= 0
                                    && sx < hi_size as i32
                                    && sy >= 0
                                    && sy < hi_size as i32
                                    && sz >= 0
                                    && sz < hi_size as i32
                                {
                                    let si = sz as usize * hi_size * hi_size
                                        + sy as usize * hi_size
                                        + sx as usize;
                                    if scene[si].material.is_air() {
                                        scene[si] = tree_flat[ti];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Lay dry-leaf ground litter trails from center to each tree.
    // This provides a conductive fuel path for fire to spread.
    let cx = hi_size as i32 / 2;
    let cz = hi_size as i32 / 2;
    let ground_y = scale; // just above dirt layer
    let leaf = Voxel {
        material: MaterialId::DRY_LEAVES,
        temperature: ambient,
        ..air
    };
    for &(tx, tz) in &tree_positions {
        // Bresenham-like line from center to tree with width ~3 voxels.
        let steps = ((tx - cx).abs().max((tz - cz).abs()) + 1) as usize;
        for s in 0..=steps {
            let frac = s as f32 / steps.max(1) as f32;
            let lx = cx as f32 + (tx - cx) as f32 * frac;
            let lz = cz as f32 + (tz - cz) as f32 * frac;
            // Trail width: 3 voxels (±1).
            for dz in -1i32..=1 {
                for ddx in -1i32..=1 {
                    let gx = (lx as i32 + ddx) as usize;
                    let gz = (lz as i32 + dz) as usize;
                    if gx < hi_size && gz < hi_size && ground_y < hi_size {
                        let si = gz * hi_size * hi_size + ground_y * hi_size + gx;
                        if scene[si].material.is_air() {
                            scene[si] = leaf;
                        }
                    }
                }
            }
        }
    }

    // Also scatter some leaf litter around the central area (random fill).
    let scatter_radius = ring_radius as i32 + 2;
    let mut srng = SimpleRng::new(99999);
    for dz in -scatter_radius..=scatter_radius {
        for ddx in -scatter_radius..=scatter_radius {
            let dist_sq = ddx * ddx + dz * dz;
            if dist_sq > scatter_radius * scatter_radius {
                continue;
            }
            // ~60% chance of leaf litter at each ground position.
            if srng.next_f32() > 0.60 {
                continue;
            }
            let gx = (cx + ddx) as usize;
            let gz = (cz + dz) as usize;
            if gx < hi_size && gz < hi_size && ground_y < hi_size {
                let si = gz * hi_size * hi_size + ground_y * hi_size + gx;
                if scene[si].material.is_air() {
                    scene[si] = leaf;
                }
            }
        }
    }

    // Central bonfire: twig and dry-leaf pile, pre-heated to 1200K.
    let fire_radius = (hi_size as f32 * 0.08).ceil() as i32;
    let fire_height = (fire_radius * 3).min(hi_size as i32 / 4);

    for dz in -fire_radius..=fire_radius {
        for dy in 0..fire_height {
            for dx in -fire_radius..=fire_radius {
                let dist_sq = dx * dx + dz * dz;
                if dist_sq > fire_radius * fire_radius {
                    continue;
                }
                let gx = (cx + dx) as usize;
                let gy = ground_y + dy as usize;
                let gz = (cz + dz) as usize;
                if gx >= hi_size || gy >= hi_size || gz >= hi_size {
                    continue;
                }
                let si = gz * hi_size * hi_size + gy * hi_size + gx;
                let mat = if dist_sq < (fire_radius / 2) * (fire_radius / 2) {
                    MaterialId::DRY_LEAVES
                } else {
                    MaterialId::TWIG
                };
                scene[si] = Voxel {
                    material: mat,
                    temperature: 1200.0,
                    ..air
                };
            }
        }
    }

    scene
}

/// Count combustible voxels above ignition temperature (~453K is the lowest ignition).
fn count_actively_burning(voxels: &[Voxel]) -> usize {
    voxels
        .iter()
        .filter(|v| {
            let m = v.material;
            let combustible = m == MaterialId::WOOD
                || m == MaterialId::TWIG
                || m == MaterialId::DRY_LEAVES
                || m == MaterialId::BARK
                || m == MaterialId::CHARCOAL;
            combustible && v.temperature > 453.0
        })
        .count()
}

/// Count total combustible voxels (fuel remaining).
fn count_fuel(voxels: &[Voxel]) -> usize {
    voxels
        .iter()
        .filter(|v| {
            let m = v.material;
            m == MaterialId::WOOD
                || m == MaterialId::TWIG
                || m == MaterialId::DRY_LEAVES
                || m == MaterialId::BARK
                || m == MaterialId::CHARCOAL
        })
        .count()
}

/// Simplified radiative heat transfer: burning voxels heat combustible neighbors
/// within a small radius. Uses a buffer to cap total radiation per target voxel,
/// preventing the positive-feedback cascade where hundreds of burning voxels
/// heat the same target simultaneously.
fn apply_radiation_heating(voxels: &mut [Voxel], size: usize, dt: f32, dx: f32) {
    let h_rad = 5.0_f32; // W/(m²·K)
    let radius: i32 = 3;
    let cap_per_tick = 5.0_f32; // max temperature gain per voxel per tick from radiation

    let mut hot: Vec<(usize, usize, usize, f32)> = Vec::new();
    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let idx = z * size * size + y * size + x;
                let v = &voxels[idx];
                // Any voxel above 500K radiates — hot ash, hot air, burning fuel.
                if v.temperature > 500.0 && !v.material.is_air() {
                    hot.push((x, y, z, v.temperature));
                }
                // Hot air radiates too but at a higher threshold (less dense).
                if v.material.is_air() && v.temperature > 600.0 {
                    hot.push((x, y, z, v.temperature));
                }
            }
        }
    }

    // Accumulate into buffer, capped per target.
    let total = size * size * size;
    let mut heat_buf = vec![0.0_f32; total];

    for &(hx, hy, hz, t_hot) in &hot {
        for dz in -radius..=radius {
            for dy in -radius..=radius {
                for ddx in -radius..=radius {
                    if ddx == 0 && dy == 0 && dz == 0 {
                        continue;
                    }
                    let nx = hx as i32 + ddx;
                    let ny = hy as i32 + dy;
                    let nz = hz as i32 + dz;
                    if nx < 0
                        || nx >= size as i32
                        || ny < 0
                        || ny >= size as i32
                        || nz < 0
                        || nz >= size as i32
                    {
                        continue;
                    }
                    let ni = nz as usize * size * size + ny as usize * size + nx as usize;
                    if heat_buf[ni] >= cap_per_tick {
                        continue; // already at cap
                    }
                    let v = &voxels[ni];
                    let m = v.material;
                    let combustible = m == MaterialId::WOOD
                        || m == MaterialId::TWIG
                        || m == MaterialId::DRY_LEAVES
                        || m == MaterialId::BARK
                        || m == MaterialId::CHARCOAL;
                    if !combustible || v.temperature >= t_hot {
                        continue;
                    }

                    let r_sq = (ddx * ddx + dy * dy + dz * dz) as f32;
                    let r_m = r_sq.sqrt() * dx;
                    let area = dx * dx;
                    let q = h_rad * (t_hot - v.temperature) * area / (r_m * r_m).max(dx * dx);
                    let thermal_mass = 80.0 * dx.powi(3) * 1500.0;
                    let delta_t = q * dt / thermal_mass;
                    heat_buf[ni] = (heat_buf[ni] + delta_t).min(cap_per_tick);
                }
            }
        }
    }

    // Apply accumulated radiation.
    for (i, &delt) in heat_buf.iter().enumerate() {
        if delt > 0.0 {
            voxels[i].temperature += delt;
        }
    }
}

#[test]
fn burning_tree_video() {
    let _ = std::fs::create_dir_all("test_output");
    let path = "test_output/burning_tree.mp4";

    // Scene: 32 base-res → depth 1 → 64³ hi-res grid (262K voxels, dx=0.5m).
    // 8 trees in a ring, central bonfire, ground litter connecting them.
    let config = TreeConfig {
        trunk_radius: 0.5,
        trunk_height: 6.0,
        branch_depth: 3,
        length_ratio: 0.55,
        radius_ratio: 0.45,
        branch_angle_deg: 35.0,
        fork_count: 3,
        octree_depth: 1,
        ambient_temperature: 288.15,
    };
    let scene_size: usize = 32;
    let depth: u32 = 1;
    let scale = 1usize << depth;
    let hi_size = scene_size * scale; // 64
    let dx = 1.0 / scale as f32; // 0.5 m per voxel

    eprintln!(
        "  Building forest scene ({hi_size}³ = {} voxels)...",
        hi_size * hi_size * hi_size
    );
    let mut voxels = build_forest_fire_scene(&config, scene_size, depth);

    let initial_fuel = count_fuel(&voxels);
    eprintln!("  Initial fuel voxels: {initial_fuel}");

    let img_w: u32 = 480;
    let img_h: u32 = 360;
    let fps: u32 = 30;
    // Each tick = 1s real time; 3 ticks/frame → 3s real time per frame.
    // At 30fps: 1 second of video = 90 seconds of simulation.
    let dt = 1.0;
    let ticks_per_frame = 3;
    let max_frames = fps * 120; // hard cap at 2 minutes of video
    let cooldown_frames = fps * 5; // 5 seconds after fire dies out

    let registry = load_material_registry().expect("material registry");
    let rules = load_reaction_rules().expect("reaction rules");
    let mut encoder = FrameEncoder::new(path, img_w, img_h, fps).expect("encoder");

    // Dim ambient — fire is the primary light source.
    let light = SceneLight {
        direction: (-0.4, -0.8, -0.3),
        color: (0.8, 0.75, 0.65),
        intensity: 0.3,
        ambient: 0.08,
    };

    let mut frames_since_fuel_changed = 0u32;
    let mut prev_fuel = initial_fuel;
    let mut total_frames = 0u32;
    let mut peak_burning = 0usize;

    for frame in 0..max_frames {
        total_frames = frame + 1;

        for _ in 0..ticks_per_frame {
            simulate_tick_dx(&mut voxels, hi_size, &rules.0, &registry, dt, dx);
            apply_radiation_heating(&mut voxels, hi_size, dt, dx);
        }

        let burning = count_actively_burning(&voxels);
        let fuel = count_fuel(&voxels);
        peak_burning = peak_burning.max(burning);

        // Stop condition: fuel hasn't decreased for cooldown_frames.
        // This handles the case where smoldering wood cores stay warm but
        // can't combust (no adjacent air, below charcoal ignition).
        if frame > fps * 3 {
            if fuel == prev_fuel {
                frames_since_fuel_changed += 1;
                if frames_since_fuel_changed >= cooldown_frames {
                    eprintln!(
                        "  Fire stalled at frame {frame} ({:.1}s video, fuel={fuel}). Stopping.",
                        frame as f32 / fps as f32
                    );
                    break;
                }
            } else {
                frames_since_fuel_changed = 0;
                prev_fuel = fuel;
            }
        }

        // Log progress every 5 seconds of video.
        if frame % (fps * 5) == 0 {
            let sim_time = (frame as f64 + 1.0) * ticks_per_frame as f64 * dt as f64;
            eprintln!(
                "  Frame {frame} ({:.0}s sim, {:.1}s video): burning={burning}, fuel={fuel}/{initial_fuel}, peak={peak_burning}",
                sim_time,
                frame as f32 / fps as f32,
            );
        }

        // Camera orbits slowly, elevated to see the whole forest.
        let t = frame as f32 / max_frames as f32;
        let angle = t * std::f32::consts::TAU * 0.6;
        let elev = 0.45 + 0.1 * (t * std::f32::consts::TAU).sin();
        let center = hi_size as f32 / 2.0;
        let radius = hi_size as f32 * 0.85;
        let cam_x = center + radius * angle.cos() * elev.cos();
        let cam_y = hi_size as f32 * 0.35 + radius * elev.sin();
        let cam_z = center + radius * angle.sin() * elev.cos();
        let view = ViewMode::Perspective {
            eye: (cam_x, cam_y, cam_z),
            target: (center, hi_size as f32 * 0.25, center),
            fov_degrees: 55.0,
            width: img_w,
            height: img_h,
        };

        let frame_img = render_frame_lit(
            &voxels,
            hi_size,
            &registry,
            &view,
            &ColorMode::Incandescence,
            1,
            &light,
        );
        encoder.push_frame(&frame_img).expect("frame");
    }

    let final_fuel = count_fuel(&voxels);
    let consumed = initial_fuel.saturating_sub(final_fuel);
    let pct = if initial_fuel > 0 {
        consumed as f32 / initial_fuel as f32 * 100.0
    } else {
        0.0
    };

    encoder.finish().expect("finalize");
    eprintln!(
        "  ✓ Burning tree video → {path} ({total_frames} frames, {:.1}s)",
        total_frames as f32 / fps as f32
    );
    eprintln!(
        "    Fuel consumed: {consumed}/{initial_fuel} ({pct:.1}%), peak burning: {peak_burning}"
    );
    assert!(std::path::Path::new(path).exists());
}
