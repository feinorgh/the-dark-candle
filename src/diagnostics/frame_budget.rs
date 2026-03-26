// Frame budget diagnostics: tracks frame time via exponential moving average,
// computes headroom relative to a target FPS, and provides an optional F3-toggled
// HUD overlay showing live FPS / budget metrics.

use bevy::prelude::*;

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
            Text::new("FPS: -- (--ms)  Budget: --%"),
            TextFont {
                font_size: 16.0,
                ..default()
            },
            TextColor(Color::WHITE),
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(10.0),
                top: Val::Px(10.0),
                ..default()
            },
        ));
    }
}

/// Refreshes the overlay text with the latest [`FrameBudget`] values.
pub fn update_overlay_text(
    budget: Res<FrameBudget>,
    mut query: Query<&mut Text, With<FrameBudgetOverlay>>,
) {
    for mut text in &mut query {
        let fps = if budget.avg_frame_ms > 0.0 {
            1000.0 / budget.avg_frame_ms
        } else {
            0.0
        };
        **text = format!(
            "FPS: {:.0} ({:.1}ms)  Budget: {:.0}%",
            fps,
            budget.avg_frame_ms,
            budget.headroom * 100.0,
        );
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
