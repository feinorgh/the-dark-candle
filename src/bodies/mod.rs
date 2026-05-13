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
pub mod procedural_body;
pub mod procedural_body_anim;
pub mod skeleton;
pub mod tissue;

use bevy::prelude::*;
use bevy_common_assets::ron::RonAssetPlugin;

use locomotion::GaitData;
use plant::TreeSkeletonData;
use skeleton::SkeletonData;
use tissue::BodyData;

use player::{BipedGaitPath, HumanoidSkeletonPath, QuadrupedGaitPath};

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
            .init_resource::<BipedGaitPath>()
            .init_resource::<QuadrupedGaitPath>();

        // --- Systems ---
        app.add_systems(
            FixedUpdate,
            (
                skeleton::init_skeletons,
                skeleton::apply_skeleton_fk,
                player::attach_player_body,
                player::player_gait_from_velocity,
                // AI gait must run after BehaviorSet (writes velocity) and
                // before phase advance.
                locomotion::ai_gait_from_velocity
                    .after(crate::behavior::BehaviorSet)
                    .after(player::player_gait_from_velocity),
                locomotion::advance_gait_phase
                    .after(locomotion::ai_gait_from_velocity)
                    .after(player::player_gait_from_velocity),
                locomotion::apply_skeleton_gait_and_ik.after(locomotion::advance_gait_phase),
                procedural_body_anim::animate_procedural_body.after(locomotion::advance_gait_phase),
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
