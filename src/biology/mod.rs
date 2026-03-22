pub mod decay;
pub mod growth;
pub mod health;
pub mod metabolism;
pub mod plants;

use bevy::prelude::*;

pub struct BiologyPlugin;

impl Plugin for BiologyPlugin {
    fn build(&self, _app: &mut App) {
        // Biology systems run on FixedUpdate after physics
    }
}
