pub mod biomes;
pub mod creatures;
pub mod items;
pub mod spawning;

use bevy::prelude::*;

pub struct ProcgenPlugin;

impl Plugin for ProcgenPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(biomes::BiomePlugin);
    }
}
