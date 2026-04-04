//! Building & Structural Construction plugin.
//!
//! Wires together:
//! - `PartData` and `RecipeData` RON asset loaders
//! - Structural stress analysis (load-path, joint breaking, collapse)
//! - Player placement (build mode, ghost, snapping)
//! - Demolition (part removal, debris spawning)
//! - Build-mode input (B to toggle, R to rotate)

use bevy::prelude::*;
use bevy_common_assets::ron::RonAssetPlugin;

pub mod crafting;
pub mod demolition;
pub mod joints;
pub mod parts;
pub mod placement;
pub mod stress;

use crate::game_state::GameState;

use crate::building::crafting::RecipeData;
use crate::building::demolition::process_demolitions;
use crate::building::joints::cleanup_broken_joints;
use crate::building::parts::PartData;
use crate::building::placement::{toggle_build_mode, rotate_selection, BuildMode};
use crate::building::stress::{
    accumulate_self_weight, apply_wind_loading, despawn_unsupported_parts,
    mark_ground_anchors, propagate_stress_and_break, StressTick,
};

pub struct BuildingPlugin;

impl Plugin for BuildingPlugin {
    fn build(&self, app: &mut App) {
        // --- Asset loaders ---
        app.add_plugins(RonAssetPlugin::<PartData>::new(&["part.ron"]))
            .add_plugins(RonAssetPlugin::<RecipeData>::new(&["recipe.ron"]));

        // --- Resources ---
        app.init_resource::<BuildMode>()
            .init_resource::<StressTick>();

        // --- Input systems (run during Playing) ---
        app.add_systems(
            Update,
            (toggle_build_mode, rotate_selection).run_if(in_state(GameState::Playing)),
        );

        // --- Structural analysis (FixedUpdate, playing only) ---
        app.add_systems(
            FixedUpdate,
            (
                mark_ground_anchors,
                accumulate_self_weight,
                apply_wind_loading,
                propagate_stress_and_break,
                despawn_unsupported_parts,
                cleanup_broken_joints,
                process_demolitions,
            )
                .chain()
                .run_if(in_state(GameState::Playing)),
        );
    }
}
