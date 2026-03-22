pub mod collision;
pub mod fluids;
pub mod gravity;
pub mod integrity;

use bevy::prelude::*;

pub struct PhysicsPlugin;

impl Plugin for PhysicsPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((gravity::GravityPlugin, collision::CollisionPlugin));
    }
}
