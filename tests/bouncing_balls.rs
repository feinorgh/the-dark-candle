// Three bouncing balls — ECS-driven rigid body simulation with video.
//
// Spawns three solid spheres (steel, rubber, glass) and six container
// walls as entities in a headless Bevy app.  The game's full physics
// pipeline handles:
//   - Gravitational acceleration and aerodynamic drag  (GravityPlugin)
//   - Broad-phase spatial grid → narrow-phase shape tests (RigidBodyPlugin)
//   - Sequential impulse solver with restitution and friction
//   - Angular dynamics (spin from off-center contacts)
//
// Walls are axis-aligned slabs with effectively infinite mass and zero
// gravity, so they behave as immovable boundaries while still
// participating in the impulse solver naturally.
//
// Visualization: dual-panel video — side view (XY) and top-down view
// (XZ) — showing the fully 3D bouncing motion.
//
// Run:
//   cargo test --test bouncing_balls -- --nocapture

use std::f32::consts::PI;
use std::time::Duration;

use bevy::prelude::*;
use bevy::time::TimeUpdateStrategy;
use image::{Rgb, RgbImage};

use the_dark_candle::diagnostics::video::FrameEncoder;
use the_dark_candle::physics::PhysicsPlugin;
use the_dark_candle::physics::constants::GRAVITY;
use the_dark_candle::physics::gravity::{DragProfile, Mass, PhysicsBody};
use the_dark_candle::physics::rigid_body::{AngularVelocity, MomentOfInertia, Torque};
use the_dark_candle::physics::shapes::{CollisionShape, PhysicsMaterial};
use the_dark_candle::world::planet::PlanetConfig;

// ---------------------------------------------------------------------------
// Container geometry (meters)
// ---------------------------------------------------------------------------

const CX_MIN: f32 = 0.0;
const CX_MAX: f32 = 10.0;
const CY_MIN: f32 = 0.0;
const CY_MAX: f32 = 8.0;
const CZ_MIN: f32 = -5.0;
const CZ_MAX: f32 = 5.0;

/// Half-thickness of each wall slab (m).
const WALL_HALF: f32 = 0.5;
/// Effectively infinite mass so walls don't budge.
const WALL_MASS: f32 = 1e10;
const WALL_RESTITUTION: f32 = 0.9;
const WALL_FRICTION: f32 = 0.5;

// ---------------------------------------------------------------------------
// Rendering constants
// ---------------------------------------------------------------------------

const PX_PER_M: f32 = 50.0;
const PANEL_GAP: u32 = 6;
const BORDER: u32 = 3;
const VIDEO_FPS: u32 = 60;

// ---------------------------------------------------------------------------
// Ball specification
// ---------------------------------------------------------------------------

struct BallSpec {
    pos: Vec3,
    vel: Vec3,
    radius: f32,
    density: f32,
    restitution: f32,
    friction: f32,
    color: Rgb<u8>,
}

impl BallSpec {
    fn volume(&self) -> f32 {
        (4.0 / 3.0) * PI * self.radius.powi(3)
    }

    fn mass(&self) -> f32 {
        self.density * self.volume()
    }
}

// ---------------------------------------------------------------------------
// Marker component to distinguish balls from walls in queries
// ---------------------------------------------------------------------------

#[derive(Component)]
struct BallMarker {
    color: Rgb<u8>,
    radius: f32,
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// Pixel dimensions for the side-view panel (XY).
fn side_panel_px() -> (u32, u32) {
    let w = ((CX_MAX - CX_MIN) * PX_PER_M) as u32;
    let h = ((CY_MAX - CY_MIN) * PX_PER_M) as u32;
    (w, h)
}

/// Pixel dimensions for the top-down panel (XZ).
fn top_panel_px() -> (u32, u32) {
    let w = ((CX_MAX - CX_MIN) * PX_PER_M) as u32;
    let h = ((CZ_MAX - CZ_MIN) * PX_PER_M) as u32;
    (w, h)
}

/// Full image size (both panels + gap + border).
fn image_size() -> (u32, u32) {
    let (sw, sh) = side_panel_px();
    let (tw, th) = top_panel_px();
    let total_w = sw + tw + PANEL_GAP + 2 * BORDER;
    let total_h = sh.max(th) + 2 * BORDER;
    (total_w, total_h)
}

/// Draw a filled circle with radial shading for a 3D sphere look.
fn draw_circle(img: &mut RgbImage, cx: f32, cy: f32, r_px: f32, color: Rgb<u8>) {
    let (w, h) = (img.width() as f32, img.height() as f32);
    let x0 = (cx - r_px).floor().max(0.0) as u32;
    let x1 = (cx + r_px).ceil().min(w - 1.0) as u32;
    let y0 = (cy - r_px).floor().max(0.0) as u32;
    let y1 = (cy + r_px).ceil().min(h - 1.0) as u32;

    for py in y0..=y1 {
        for px in x0..=x1 {
            let dx = px as f32 - cx;
            let dy = py as f32 - cy;
            let dist_sq = dx * dx + dy * dy;
            if dist_sq <= r_px * r_px {
                let t = (dist_sq.sqrt() / r_px).clamp(0.0, 1.0);
                let shade = 1.0 - 0.45 * t;
                let r = (color.0[0] as f32 * shade) as u8;
                let g = (color.0[1] as f32 * shade) as u8;
                let b = (color.0[2] as f32 * shade) as u8;
                img.put_pixel(px, py, Rgb([r, g, b]));
            }
        }
    }
}

/// Draw a thin rectangle border.
fn draw_rect_border(img: &mut RgbImage, x0: u32, y0: u32, w: u32, h: u32, color: Rgb<u8>) {
    for px in x0..(x0 + w) {
        img.put_pixel(px, y0, color);
        img.put_pixel(px, y0 + h - 1, color);
    }
    for py in y0..(y0 + h) {
        img.put_pixel(x0, py, color);
        img.put_pixel(x0 + w - 1, py, color);
    }
}

struct BallState {
    pos: Vec3,
    radius: f32,
    color: Rgb<u8>,
}

/// Render a dual-panel frame: side view (XY) on the left, top-down (XZ) on
/// the right.
fn render_frame(balls: &[BallState]) -> RgbImage {
    let (img_w, img_h) = image_size();
    let mut img = RgbImage::from_pixel(img_w, img_h, Rgb([30, 30, 35]));

    let (sw, sh) = side_panel_px();
    let (tw, th) = top_panel_px();
    let wall_color = Rgb([140, 140, 150]);
    let bg = Rgb([12, 12, 22]);

    // --- Side-view panel (left) ---
    let sx0 = BORDER;
    let sy0 = BORDER;
    for py in sy0..(sy0 + sh) {
        for px in sx0..(sx0 + sw) {
            img.put_pixel(px, py, bg);
        }
    }
    draw_rect_border(&mut img, sx0, sy0, sw, sh, wall_color);

    // --- Top-down panel (right) ---
    let tx0 = BORDER + sw + PANEL_GAP;
    let ty0 = BORDER;
    for py in ty0..(ty0 + th) {
        for px in tx0..(tx0 + tw) {
            img.put_pixel(px, py, bg);
        }
    }
    draw_rect_border(&mut img, tx0, ty0, tw, th, wall_color);

    // --- Draw balls in both views ---
    for ball in balls {
        let r_px = ball.radius * PX_PER_M;

        // Side view: X → right, Y → up (screen Y inverted)
        let side_cx = sx0 as f32 + (ball.pos.x - CX_MIN) * PX_PER_M;
        let side_cy = sy0 as f32 + (CY_MAX - ball.pos.y) * PX_PER_M;
        draw_circle(&mut img, side_cx, side_cy, r_px, ball.color);

        // Top-down view: X → right, Z → down
        let top_cx = tx0 as f32 + (ball.pos.x - CX_MIN) * PX_PER_M;
        let top_cy = ty0 as f32 + (ball.pos.z - CZ_MIN) * PX_PER_M;
        draw_circle(&mut img, top_cx, top_cy, r_px, ball.color);
    }

    img
}

// ---------------------------------------------------------------------------
// Entity spawning helpers
// ---------------------------------------------------------------------------

fn spawn_wall(world: &mut World, pos: Vec3, half_extents: Vec3) {
    world.spawn((
        Transform::from_translation(pos),
        PhysicsBody::weightless(),
        CollisionShape::Aabb { half_extents },
        Mass(WALL_MASS),
        PhysicsMaterial::new(WALL_FRICTION, WALL_RESTITUTION),
    ));
}

fn spawn_ball(world: &mut World, spec: &BallSpec) -> Entity {
    let mass = spec.mass();
    let shape = CollisionShape::sphere(spec.radius);
    let inertia = MomentOfInertia(shape.moment_of_inertia(mass));

    world
        .spawn((
            Transform::from_translation(spec.pos),
            PhysicsBody {
                velocity: spec.vel,
                ..Default::default()
            },
            shape,
            Mass(mass),
            PhysicsMaterial::new(spec.friction, spec.restitution),
            DragProfile {
                coefficient: 0.47, // smooth sphere
                area: PI * spec.radius * spec.radius,
                volume: spec.volume(),
            },
            AngularVelocity::default(),
            inertia,
            Torque::default(),
            BallMarker {
                color: spec.color,
                radius: spec.radius,
            },
        ))
        .id()
}

// ---------------------------------------------------------------------------
// Test
// ---------------------------------------------------------------------------

#[test]
fn three_bouncing_balls() {
    // --- Build headless Bevy app with the full physics engine ---
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_plugins(AssetPlugin::default())
        .add_plugins(PhysicsPlugin)
        .insert_resource(PlanetConfig::default())
        .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f64(
            1.0 / 60.0,
        )));

    // --- Spawn container walls (6 slabs forming a sealed box) ---
    //
    // Each wall extends beyond the container edges by WALL_HALF so that all
    // 12 edges and 8 corners are covered — no gaps a ball could slip through.
    // The resulting wall-wall overlap is safe because the solver skips
    // contacts between two effectively-static bodies (mass > 1e8).
    let cx = (CX_MIN + CX_MAX) / 2.0;
    let cy = (CY_MIN + CY_MAX) / 2.0;
    let cz = (CZ_MIN + CZ_MAX) / 2.0;
    let hx = (CX_MAX - CX_MIN) / 2.0 + WALL_HALF; // extends past side walls
    let hy = (CY_MAX - CY_MIN) / 2.0 + WALL_HALF; // extends past floor/ceiling
    let hz = (CZ_MAX - CZ_MIN) / 2.0 + WALL_HALF; // extends past front/back

    {
        let world = app.world_mut();
        // Floor / ceiling (full X+Z coverage including corners)
        spawn_wall(
            world,
            Vec3::new(cx, CY_MIN - WALL_HALF, cz),
            Vec3::new(hx, WALL_HALF, hz),
        );
        spawn_wall(
            world,
            Vec3::new(cx, CY_MAX + WALL_HALF, cz),
            Vec3::new(hx, WALL_HALF, hz),
        );
        // Left / right (full Y+Z coverage including corners)
        spawn_wall(
            world,
            Vec3::new(CX_MIN - WALL_HALF, cy, cz),
            Vec3::new(WALL_HALF, hy, hz),
        );
        spawn_wall(
            world,
            Vec3::new(CX_MAX + WALL_HALF, cy, cz),
            Vec3::new(WALL_HALF, hy, hz),
        );
        // Front / back (full X+Y coverage including corners)
        spawn_wall(
            world,
            Vec3::new(cx, cy, CZ_MIN - WALL_HALF),
            Vec3::new(hx, hy, WALL_HALF),
        );
        spawn_wall(
            world,
            Vec3::new(cx, cy, CZ_MAX + WALL_HALF),
            Vec3::new(hx, hy, WALL_HALF),
        );
    }

    // --- Three balls with real SI properties and 3D initial velocities ---
    let ball_specs = [
        BallSpec {
            pos: Vec3::new(2.0, 6.5, -2.0),
            vel: Vec3::new(4.0, 1.0, 3.0),
            radius: 0.4,
            density: 7874.0, // steel
            restitution: 0.6,
            friction: 0.3,
            color: Rgb([220, 60, 60]),
        },
        BallSpec {
            pos: Vec3::new(5.0, 7.0, 2.0),
            vel: Vec3::new(-2.5, -1.0, -2.0),
            radius: 0.3,
            density: 1100.0, // vulcanized rubber
            restitution: 0.85,
            friction: 0.8,
            color: Rgb([60, 200, 80]),
        },
        BallSpec {
            pos: Vec3::new(8.0, 5.0, -1.0),
            vel: Vec3::new(-1.5, 3.0, 1.5),
            radius: 0.35,
            density: 2500.0, // soda-lime glass
            restitution: 0.7,
            friction: 0.4,
            color: Rgb([60, 100, 220]),
        },
    ];

    let ball_entities: Vec<Entity> = {
        let world = app.world_mut();
        ball_specs.iter().map(|s| spawn_ball(world, s)).collect()
    };

    eprintln!("  Ball properties (SI):");
    for (i, s) in ball_specs.iter().enumerate() {
        eprintln!(
            "    [{i}] mass={:.1} kg, r={:.2} m, e={:.2}, v₀={:?}",
            s.mass(),
            s.radius,
            s.restitution,
            s.vel,
        );
    }

    // --- Simulation parameters ---
    let total_seconds = 10.0_f32;
    let ticks_per_second = 60_u32; // matches ManualDuration(1/60 s)
    let total_ticks = total_seconds as u32 * ticks_per_second;

    // Record initial mechanical energy for later comparison.
    let initial_ke: f32 = ball_specs
        .iter()
        .map(|s| 0.5 * s.mass() * s.vel.length_squared())
        .sum();
    let initial_pe: f32 = ball_specs
        .iter()
        .map(|s| s.mass() * GRAVITY * s.pos.y)
        .sum();
    let initial_energy = initial_ke + initial_pe;

    // --- Video encoder ---
    let _ = std::fs::create_dir_all("test_output");
    let video_path = "test_output/bouncing_balls.mp4";
    let (img_w, img_h) = image_size();
    let mut encoder =
        FrameEncoder::new(video_path, img_w, img_h, VIDEO_FPS).expect("video encoder");

    // --- Main simulation loop ---
    for _tick in 0..total_ticks {
        app.update();

        // Read ball positions from the ECS and render a video frame.
        let ball_states: Vec<BallState> = ball_entities
            .iter()
            .map(|&e| {
                let tf = app.world().get::<Transform>(e).unwrap();
                let marker = app.world().get::<BallMarker>(e).unwrap();
                BallState {
                    pos: tf.translation,
                    radius: marker.radius,
                    color: marker.color,
                }
            })
            .collect();

        let frame = render_frame(&ball_states);
        encoder.push_frame(&frame).expect("frame encode");
    }

    encoder.finish().expect("video finalize");

    // --- Read final state from the ECS ---
    struct FinalBall {
        pos: Vec3,
        vel: Vec3,
        radius: f32,
        mass: f32,
    }

    let final_balls: Vec<FinalBall> = ball_entities
        .iter()
        .zip(ball_specs.iter())
        .map(|(&e, s)| {
            let tf = app.world().get::<Transform>(e).unwrap();
            let body = app.world().get::<PhysicsBody>(e).unwrap();
            FinalBall {
                pos: tf.translation,
                vel: body.velocity,
                radius: s.radius,
                mass: s.mass(),
            }
        })
        .collect();

    let final_ke: f32 = final_balls
        .iter()
        .map(|b| 0.5 * b.mass * b.vel.length_squared())
        .sum();
    let final_pe: f32 = final_balls.iter().map(|b| b.mass * GRAVITY * b.pos.y).sum();
    let final_energy = final_ke + final_pe;

    // --- Assertions ---

    // 1. All balls remain inside the container (10 cm tolerance for solver slop)
    for (i, b) in final_balls.iter().enumerate() {
        let m = 0.1;
        assert!(
            b.pos.x - b.radius >= CX_MIN - m
                && b.pos.x + b.radius <= CX_MAX + m
                && b.pos.y - b.radius >= CY_MIN - m
                && b.pos.y + b.radius <= CY_MAX + m
                && b.pos.z - b.radius >= CZ_MIN - m
                && b.pos.z + b.radius <= CZ_MAX + m,
            "Ball {i} escaped: pos={:?}, r={}",
            b.pos,
            b.radius,
        );
    }

    // 2. No NaN or infinite values (simulation stability)
    for (i, b) in final_balls.iter().enumerate() {
        assert!(
            b.pos.is_finite() && b.vel.is_finite(),
            "Ball {i} has non-finite state: pos={:?}, vel={:?}",
            b.pos,
            b.vel,
        );
    }

    // 3. No ball–ball interpenetration at end
    for i in 0..final_balls.len() {
        for j in (i + 1)..final_balls.len() {
            let dist = (final_balls[i].pos - final_balls[j].pos).length();
            let min_dist = final_balls[i].radius + final_balls[j].radius;
            assert!(
                dist >= min_dist - 0.05,
                "Balls {i}–{j} interpenetrate: dist={dist:.4}, min={min_dist:.4}",
            );
        }
    }

    // 4. Balls are actively bouncing (velocities non-zero after 10 s)
    let total_speed: f32 = final_balls.iter().map(|b| b.vel.length()).sum();
    assert!(
        total_speed > 0.1,
        "Balls should still be moving after 10 s (total speed = {total_speed:.4})",
    );

    eprintln!(
        "  ✓ Three bouncing balls (ECS): {} ticks, {total_seconds} s",
        total_ticks,
    );
    let energy_delta_pct = (final_energy / initial_energy - 1.0) * 100.0;
    eprintln!("    Energy: {initial_energy:.0} J → {final_energy:.0} J ({energy_delta_pct:+.1}%)",);

    // With split-impulse, position correction no longer injects kinetic energy.
    // Energy should only decrease (restitution < 1, friction, drag).  Allow a
    // small tolerance for floating-point accumulation.
    assert!(
        final_energy <= initial_energy * 1.05,
        "Energy should not increase significantly: {initial_energy:.0} J → {final_energy:.0} J ({energy_delta_pct:+.1}%)",
    );
    eprintln!(
        "    Final positions: [{:.2}, {:.2}, {:.2}], [{:.2}, {:.2}, {:.2}], [{:.2}, {:.2}, {:.2}]",
        final_balls[0].pos.x,
        final_balls[0].pos.y,
        final_balls[0].pos.z,
        final_balls[1].pos.x,
        final_balls[1].pos.y,
        final_balls[1].pos.z,
        final_balls[2].pos.x,
        final_balls[2].pos.y,
        final_balls[2].pos.z,
    );
    eprintln!("    Video: {video_path}");
}
