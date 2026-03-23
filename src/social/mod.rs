pub mod factions;
pub mod group_behaviors;
pub mod relationships;
pub mod reputation;

use bevy::prelude::*;

pub struct SocialPlugin;

impl Plugin for SocialPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<factions::FactionRegistry>();
    }
}
