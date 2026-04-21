//! Debug rendering toggles to aid diagnosis of terrain/atmosphere rendering.
//!
//! - **F7**: Toggle atmosphere rendering (fog, sky clear color, atmosphere shader).
//!   When off, the ClearColor becomes a neutral dark gray and DistanceFog /
//!   Atmosphere components are removed from the player camera so terrain is
//!   visible to the horizon without any atmospheric occlusion.
//! - **F8**: Toggle global wireframe rendering for all meshes.

use bevy::pbr::wireframe::WireframeConfig;
use bevy::pbr::{Atmosphere, DistanceFog, FogFalloff};
use bevy::prelude::*;

use crate::camera::FpsCamera;

/// Global state for debug render toggles.
#[derive(Resource, Debug, Clone)]
pub struct DebugRenderState {
    /// When false, atmosphere/fog/sky updates are disabled and fog is removed
    /// from the player camera.
    pub atmosphere_enabled: bool,
    /// When true, all meshes are rendered as wireframes.
    pub wireframe_enabled: bool,
}

impl Default for DebugRenderState {
    fn default() -> Self {
        Self {
            atmosphere_enabled: true,
            wireframe_enabled: false,
        }
    }
}

/// Run condition: only run the gated system when atmosphere rendering is enabled.
pub fn atmosphere_enabled(state: Res<DebugRenderState>) -> bool {
    state.atmosphere_enabled
}

/// System: handle F7 to toggle atmosphere rendering.
pub(super) fn toggle_atmosphere_system(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut state: ResMut<DebugRenderState>,
    mut clear_color: ResMut<ClearColor>,
    mut commands: Commands,
    cam_q: Query<Entity, With<FpsCamera>>,
) {
    if !keyboard.just_pressed(KeyCode::F7) {
        return;
    }

    state.atmosphere_enabled = !state.atmosphere_enabled;

    if state.atmosphere_enabled {
        info!("[DebugRender] Atmosphere ENABLED (F7)");
        // Re-insert DistanceFog with the same defaults used in the camera spawn.
        // update_fog / update_spherical_sky will take over from the next frame.
        for cam in &cam_q {
            commands.entity(cam).insert(DistanceFog {
                color: Color::srgba(0.7, 0.78, 0.9, 1.0),
                directional_light_color: Color::srgba(1.0, 0.95, 0.85, 0.5),
                directional_light_exponent: 30.0,
                falloff: FogFalloff::from_visibility(500.0),
            });
        }
    } else {
        info!("[DebugRender] Atmosphere DISABLED (F7) — fog/sky removed");
        // Remove fog and atmosphere components so terrain is fully visible.
        for cam in &cam_q {
            commands
                .entity(cam)
                .remove::<DistanceFog>()
                .remove::<Atmosphere>();
        }
        // Neutral dark gray so wireframes/meshes are readable without the
        // sky-color update system fighting us next frame.
        clear_color.0 = Color::srgb(0.12, 0.12, 0.14);
    }
}

/// System: handle F8 to toggle global wireframe rendering.
pub(super) fn toggle_wireframe_system(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut state: ResMut<DebugRenderState>,
    config: Option<ResMut<WireframeConfig>>,
) {
    if !keyboard.just_pressed(KeyCode::F8) {
        return;
    }

    let Some(mut config) = config else {
        warn!(
            "[DebugRender] Wireframe toggle requested but WireframeConfig not initialized (GPU may lack POLYGON_MODE_LINE support)"
        );
        return;
    };

    state.wireframe_enabled = !state.wireframe_enabled;
    config.global = state.wireframe_enabled;
    info!(
        "[DebugRender] Wireframe {} (F8)",
        if state.wireframe_enabled {
            "ENABLED"
        } else {
            "DISABLED"
        }
    );
}
