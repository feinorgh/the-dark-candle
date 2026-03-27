// Persistence plugin: save and load game state to RON files.
//
// Each persistent entity gets a `SaveId` component assigned at spawn.
// This stable u64 allows relationship maps (which use entity bits as keys)
// to be correctly remapped when entities are re-created after a load.
//
// Save slots: 1 autosave + 3 manual slots. F5/F9 for quick save/load.
// Pause menu integration via `SaveRequest` / `LoadRequest` resources.

mod load;
mod save;
pub mod types;

pub use load::{LoadRequest, list_save_slots};
pub use save::SaveRequest;
pub use types::{SaveId, SaveSlot};

use bevy::prelude::*;

use crate::{
    entities::Enemy,
    procgen::{creatures::Creature, items::Item},
};

/// Monotonically increasing counter for assigning stable `SaveId` values.
#[derive(Resource, Default)]
pub struct SaveIdCounter(u64);

impl SaveIdCounter {
    pub fn generate(&mut self) -> u64 {
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
        commands.entity(entity).insert(SaveId(counter.generate()));
    }
}

fn assign_creature_save_ids(
    mut commands: Commands,
    mut counter: ResMut<SaveIdCounter>,
    query: Query<Entity, (With<Creature>, Without<SaveId>)>,
) {
    for entity in &query {
        commands.entity(entity).insert(SaveId(counter.generate()));
    }
}

fn assign_item_save_ids(
    mut commands: Commands,
    mut counter: ResMut<SaveIdCounter>,
    query: Query<Entity, (With<Item>, Without<SaveId>)>,
) {
    for entity in &query {
        commands.entity(entity).insert(SaveId(counter.generate()));
    }
}
