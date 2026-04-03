// Entity Bodies & Organic Physics — Phase 10.
//
// Physical embodiment for all living entities: data-driven articulated
// skeletons, soft/rigid tissue layers, procedural IK locomotion, physical
// perception upgrades, injury model, and plant body physics.
//
// All new code is additive — existing systems (rigid body, collision,
// biology, behavior) are extended, not replaced.

pub mod ik;
pub mod injury;
pub mod locomotion;
pub mod perception;
pub mod plant;
pub mod player;
pub mod skeleton;
pub mod tissue;

use bevy::prelude::*;
use bevy_common_assets::ron::RonAssetPlugin;

use locomotion::GaitData;
use plant::TreeSkeletonData;
use skeleton::SkeletonData;
use tissue::BodyData;

use player::{BipedGaitPath, HumanoidSkeletonPath};

/// System-set label for all body-physics systems.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct BodiesSet;

/// Plugin that registers all entity-body systems and asset loaders.
pub struct BodiesPlugin;

impl Plugin for BodiesPlugin {
    fn build(&self, app: &mut App) {
        // --- Asset loaders ---
        app.add_plugins(RonAssetPlugin::<SkeletonData>::new(&["skeleton.ron"]))
            .add_plugins(RonAssetPlugin::<BodyData>::new(&["body.ron"]))
            .add_plugins(RonAssetPlugin::<GaitData>::new(&["gait.ron"]))
            .add_plugins(RonAssetPlugin::<TreeSkeletonData>::new(&[
                "treeskeleton.ron",
            ]));

        // --- Resources ---
        app.init_resource::<HumanoidSkeletonPath>()
            .init_resource::<BipedGaitPath>();

        // --- Systems ---
        app.add_systems(
            FixedUpdate,
            (
                skeleton::init_skeletons,
                skeleton::apply_skeleton_fk,
                locomotion::update_locomotion,
                player::attach_player_body,
                player::player_gait_from_velocity,
            )
                .in_set(BodiesSet),
        );
        app.add_systems(
            FixedUpdate,
            (
                player::apply_head_bob,
                perception::update_sight,
                injury::tick_injury_healing,
                injury::spawn_severed_limbs,
                plant::apply_wind_to_plants,
                plant::check_tree_felling,
            )
                .in_set(BodiesSet),
        );
    }
}
