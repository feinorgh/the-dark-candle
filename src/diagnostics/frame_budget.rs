// Frame budget diagnostics: tracks frame time via exponential moving average,
// computes headroom relative to a target FPS, and provides an optional F3-toggled
// HUD overlay showing live FPS / budget metrics and debug information.

use bevy::prelude::*;

use crate::camera::FpsCamera;
use crate::diagnostics::system_timings::{ChunkStats, SystemTimings};
use crate::lighting::TimeOfDay;
use crate::lighting::orbital::OrbitalState;
use crate::world::chunk::ChunkCoord;
use crate::world::chunk_manager::{ChunkLoadRadius, ChunkMap};

/// Tracks per-frame timing using an exponential moving average (EMA) and
/// reports how much of the per-frame budget is consumed.
#[derive(Resource, Debug, Clone)]
pub struct FrameBudget {
    /// Target frames per second.
    pub target_fps: f32,
    /// Exponential moving average of frame time in milliseconds.
    pub avg_frame_ms: f32,
    /// Smoothing factor for EMA (0..1). Higher = more responsive.
    pub smoothing: f32,
    /// How much headroom remains: `1.0 - (avg_frame_ms / budget_ms)`.
    /// Positive = under budget, negative = over budget.
    pub headroom: f32,
}

impl Default for FrameBudget {
    fn default() -> Self {
        Self {
            target_fps: 30.0,
            avg_frame_ms: 0.0,
            smoothing: 0.1,
            headroom: 1.0,
        }
    }
}

/// Updates [`FrameBudget`] every frame using the EMA of `Time::delta_secs`.
pub fn update_frame_budget(time: Res<Time>, mut budget: ResMut<FrameBudget>) {
    let dt_ms = time.delta_secs() * 1000.0;
    budget.avg_frame_ms = budget.smoothing * dt_ms + (1.0 - budget.smoothing) * budget.avg_frame_ms;
    let budget_ms = 1000.0 / budget.target_fps;
    budget.headroom = 1.0 - (budget.avg_frame_ms / budget_ms);
}

// ---------------------------------------------------------------------------
// Overlay
// ---------------------------------------------------------------------------

/// Marker component for the frame-budget HUD text entity.
#[derive(Component)]
pub struct FrameBudgetOverlay;

/// Toggles the frame-budget overlay on/off when F3 is pressed.
pub fn toggle_overlay(
    keyboard: Res<ButtonInput<KeyCode>>,
    overlay_q: Query<Entity, With<FrameBudgetOverlay>>,
    mut commands: Commands,
) {
    if !keyboard.just_pressed(KeyCode::F3) {
        return;
    }

    if let Some(entity) = overlay_q.iter().next() {
        commands.entity(entity).despawn();
    } else {
        commands.spawn((
            FrameBudgetOverlay,
            Text::new("FPS: -- (--ms)  Budget: --%\nPos: --\nGrounded: --\nChunks: --"),
            TextFont {
                font_size: 14.0,
                ..default()
            },
            TextColor(Color::WHITE),
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(10.0),
                top: Val::Px(10.0),
                ..default()
            },
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.6)),
        ));
    }
}

/// Refreshes the overlay text with debug HUD information.
#[allow(clippy::too_many_arguments)]
pub fn update_overlay_text(
    budget: Res<FrameBudget>,
    cam_q: Query<(&crate::floating_origin::WorldPosition, &FpsCamera)>,
    chunk_map: Option<Res<ChunkMap>>,
    load_radius: Option<Res<ChunkLoadRadius>>,
    orbital: Option<Res<OrbitalState>>,
    tod: Option<Res<TimeOfDay>>,
    timings: Option<Res<SystemTimings>>,
    chunk_stats: Option<Res<ChunkStats>>,
    planet: Res<crate::world::planet::PlanetConfig>,
    terrain_gen: Option<Res<crate::world::chunk_manager::SharedTerrainGen>>,
    mut query: Query<&mut Text, With<FrameBudgetOverlay>>,
) {
    for mut text in &mut query {
        let fps = if budget.avg_frame_ms > 0.0 {
            1000.0 / budget.avg_frame_ms
        } else {
            0.0
        };

        // Line 1: FPS and budget
        let mut lines = format!(
            "FPS: {:.0} ({:.1}ms)  Budget: {:.0}%",
            fps,
            budget.avg_frame_ms,
            budget.headroom * 100.0,
        );

        // Line 2: System timings
        if let Some(t) = &timings {
            lines.push_str(&format!(
                "\nSystems: world {:.1}ms  mesh {:.1}ms  phys {:.1}ms",
                t.world_ms, t.meshing_ms, t.physics_ms,
            ));
        }

        // Line 3-4: Player info
        if let Ok((world_pos, cam)) = cam_q.single() {
            let pos = world_pos.0;
            let cc = ChunkCoord::from_voxel_pos(pos.x as i32, pos.y as i32, pos.z as i32);
            lines.push_str(&format!(
                "\nPos: ({:.1}, {:.1}, {:.1})  Chunk: ({}, {}, {})",
                pos.x, pos.y, pos.z, cc.x, cc.y, cc.z,
            ));
            let grav = if cam.gravity_enabled { "on" } else { "off" };
            let ground = if cam.grounded { "yes" } else { "no" };
            lines.push_str(&format!(
                "\nGrounded: {}  Gravity: {}  Speed: {:.0} m/s",
                ground, grav, cam.speed,
            ));

            // Show altitude above surface when in flight mode (gravity off).
            if !cam.gravity_enabled {
                let cam_r = pos.length();
                let surface_r = if let Some(ref tgen) = terrain_gen {
                    let (lat, lon) = crate::planet::detail::pos_to_lat_lon(pos);
                    tgen.0.sample_surface_radius_at(lat, lon) as f32
                } else {
                    planet.mean_radius as f32
                };
                let altitude = cam_r as f32 - surface_r;
                if altitude >= 1000.0 {
                    lines.push_str(&format!("  Alt: {:.2} km", altitude / 1000.0));
                } else {
                    lines.push_str(&format!("  Alt: {:.1} m", altitude));
                }
            }
        }

        // Line 5: Chunk pipeline stats
        let chunks = chunk_map.as_ref().map(|m| m.len()).unwrap_or(0);
        let view = load_radius.as_ref().map(|r| r.horizontal).unwrap_or(0);
        let meshing = chunk_stats
            .as_ref()
            .map(|s| s.meshing_in_flight)
            .unwrap_or(0);
        let generating = chunk_stats.as_ref().map(|s| s.generating).unwrap_or(0);
        let time_scale = orbital.as_ref().map(|o| o.time_scale).unwrap_or(0.0);
        let hour = tod.as_ref().map(|t| t.0).unwrap_or(0.0);
        let hour_int = hour as u32;
        let minute = ((hour - hour_int as f32) * 60.0) as u32;
        lines.push_str(&format!(
            "\nChunks: {} (gen {} mesh {})  View: {}  Time: {:.0}x ({:02}:{:02})",
            chunks, generating, meshing, view, time_scale, hour_int, minute,
        ));

        **text = lines;
    }
}

/// Plugin that registers the frame-budget resource and its systems.
pub struct FrameBudgetOverlayPlugin;

impl Plugin for FrameBudgetOverlayPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FrameBudget>().add_systems(
            Update,
            (update_frame_budget, toggle_overlay, update_overlay_text),
        );
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_budget_defaults() {
        let fb = FrameBudget::default();
        assert_eq!(fb.target_fps, 30.0);
        assert_eq!(fb.avg_frame_ms, 0.0);
        assert_eq!(fb.smoothing, 0.1);
        assert_eq!(fb.headroom, 1.0);
    }

    /// Helper: run one EMA + headroom iteration in-place.
    fn step(budget: &mut FrameBudget, dt_ms: f32) {
        budget.avg_frame_ms =
            budget.smoothing * dt_ms + (1.0 - budget.smoothing) * budget.avg_frame_ms;
        let budget_ms = 1000.0 / budget.target_fps;
        budget.headroom = 1.0 - (budget.avg_frame_ms / budget_ms);
    }

    #[test]
    fn ema_converges() {
        let mut fb = FrameBudget {
            avg_frame_ms: 10.0,
            ..Default::default()
        };

        // Feed a constant 33.33 ms frame time (≈ 30 FPS) for many iterations.
        let target_dt = 1000.0 / 30.0;
        for _ in 0..200 {
            step(&mut fb, target_dt);
        }

        // EMA should converge close to the constant input.
        assert!(
            (fb.avg_frame_ms - target_dt).abs() < 0.01,
            "EMA should converge to {target_dt}, got {}",
            fb.avg_frame_ms,
        );
    }

    #[test]
    fn headroom_calculation() {
        let mut fb = FrameBudget {
            target_fps: 30.0,
            avg_frame_ms: 16.6667, // half of 33.33 ms budget
            smoothing: 1.0,        // instant snap so step sets avg directly
            headroom: 0.0,
        };

        let budget_ms = 1000.0 / fb.target_fps; // 33.333..
        let dt = fb.avg_frame_ms;
        step(&mut fb, dt);
        let expected = 1.0 - (fb.avg_frame_ms / budget_ms);
        assert!(
            (fb.headroom - expected).abs() < 1e-4,
            "expected headroom {expected}, got {}",
            fb.headroom,
        );
        assert!(
            fb.headroom > 0.0,
            "headroom should be positive when under budget"
        );
    }

    #[test]
    fn headroom_negative_when_over_budget() {
        let mut fb = FrameBudget {
            target_fps: 30.0,
            avg_frame_ms: 0.0,
            smoothing: 1.0,
            headroom: 0.0,
        };

        // Feed 50 ms frame time — well over the 33.33 ms budget.
        step(&mut fb, 50.0);
        assert!(
            fb.headroom < 0.0,
            "headroom should be negative when over budget, got {}",
            fb.headroom,
        );
    }
}
