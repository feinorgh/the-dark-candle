pub mod behaviors;
pub mod needs;
pub mod pathfinding;
pub mod perception;
pub mod utility;

use bevy::prelude::*;

pub struct BehaviorPlugin;

impl Plugin for BehaviorPlugin {
    fn build(&self, _app: &mut App) {
        // Behavior systems run on FixedUpdate after biology
    }
}
