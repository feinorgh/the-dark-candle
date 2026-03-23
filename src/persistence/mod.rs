// Persistence plugin: save (F5) and load (F9) game state to RON files.
//
// Each persistent entity gets a `SaveId` component assigned at spawn.
// This stable u64 allows relationship maps (which use entity bits as keys)
// to be correctly remapped when entities are re-created after a load.

mod load;
mod save;
pub mod types;

pub use types::SaveId;

use bevy::prelude::*;

use crate::{
    entities::Enemy,
    procgen::{creatures::Creature, items::Item},
};

/// Monotonically increasing counter for assigning stable `SaveId` values.
#[derive(Resource, Default)]
pub struct SaveIdCounter(u64);

impl SaveIdCounter {
    pub fn next(&mut self) -> u64 {
        let id = self.0;
        self.0 += 1;
        id
    }
}

pub struct PersistencePlugin;

impl Plugin for PersistencePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SaveIdCounter>().add_systems(
            Update,
            (
                assign_enemy_save_ids,
                assign_creature_save_ids,
                assign_item_save_ids,
                save::save_game,
                load::load_game,
            ),
        );
    }
}

// ---------------------------------------------------------------------------
// SaveId assignment — runs every frame but only fires when new entities lack one
// ---------------------------------------------------------------------------

fn assign_enemy_save_ids(
    mut commands: Commands,
    mut counter: ResMut<SaveIdCounter>,
    query: Query<Entity, (With<Enemy>, Without<SaveId>)>,
) {
    for entity in &query {
        commands.entity(entity).insert(SaveId(counter.next()));
    }
}

fn assign_creature_save_ids(
    mut commands: Commands,
    mut counter: ResMut<SaveIdCounter>,
    query: Query<Entity, (With<Creature>, Without<SaveId>)>,
) {
    for entity in &query {
        commands.entity(entity).insert(SaveId(counter.next()));
    }
}

fn assign_item_save_ids(
    mut commands: Commands,
    mut counter: ResMut<SaveIdCounter>,
    query: Query<Entity, (With<Item>, Without<SaveId>)>,
) {
    for entity in &query {
        commands.entity(entity).insert(SaveId(counter.next()));
    }
}
