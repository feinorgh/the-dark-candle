pub mod amr_fluid;
pub mod collision;
pub mod constants;
pub mod flip_pic;
pub mod fluids;
pub mod gravity;
pub mod integrity;
pub mod lbm_gas;
pub mod pressure;
pub mod voxel_bridge;

use bevy::prelude::*;
use bevy_common_assets::ron::RonAssetPlugin;

pub struct PhysicsPlugin;

impl Plugin for PhysicsPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(RonAssetPlugin::<constants::UniversalConstants>::new(&[
            "universal_constants.ron",
        ]))
        .add_plugins(RonAssetPlugin::<constants::WorldConstants>::new(&[
            "world_constants.ron",
        ]))
        .add_plugins((gravity::GravityPlugin, collision::CollisionPlugin));
    }
}
