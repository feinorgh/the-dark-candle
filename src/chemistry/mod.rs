pub mod heat;
pub mod reactions;

use bevy::prelude::*;

pub struct ChemistryPlugin;

impl Plugin for ChemistryPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(reactions::ReactionPlugin);
    }
}
