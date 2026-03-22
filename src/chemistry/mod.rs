pub mod fire_propagation;
pub mod heat;
pub mod reactions;
pub mod state_transitions;

use bevy::prelude::*;

pub struct ChemistryPlugin;

impl Plugin for ChemistryPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(reactions::ReactionPlugin);
    }
}
