// In-game map system: local discovery map + global planet map.
//
// Press M to toggle the map overlay. Tab switches between Local and Global
// views. Scroll wheel zooms in/out.

mod discovery;
mod global_map;
mod local_map;
mod ui;

use bevy::prelude::*;

use crate::game_state::GameState;

pub use discovery::DiscoveredColumns;

pub struct MapPlugin;

impl Plugin for MapPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<DiscoveredColumns>()
            .init_resource::<ui::MapViewState>()
            .add_systems(
                Update,
                toggle_map.run_if(in_state(GameState::Playing).or(in_state(GameState::Map))),
            )
            .add_systems(OnEnter(GameState::Map), ui::spawn_map_overlay)
            .add_systems(OnExit(GameState::Map), ui::despawn_map_overlay)
            .add_systems(
                Update,
                (
                    ui::tab_switch,
                    ui::scroll_zoom,
                    ui::map_pan,
                    local_map::update_local_map,
                    global_map::update_global_map,
                )
                    .chain()
                    .run_if(in_state(GameState::Map)),
            )
            // Discovery runs during gameplay (not gated to Map state).
            .add_systems(
                Update,
                discovery::track_discoveries.run_if(in_state(GameState::Playing)),
            );
    }
}

fn toggle_map(
    key: Res<ButtonInput<KeyCode>>,
    state: Res<State<GameState>>,
    mut next_state: ResMut<NextState<GameState>>,
) {
    if key.just_pressed(KeyCode::KeyM) {
        match state.get() {
            GameState::Playing => next_state.set(GameState::Map),
            GameState::Map => next_state.set(GameState::Playing),
            _ => {}
        }
    }
    // Also allow ESC to close the map.
    if *state.get() == GameState::Map && key.just_pressed(KeyCode::Escape) {
        next_state.set(GameState::Playing);
    }
}
