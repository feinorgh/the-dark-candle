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
            GameState::Playing => {
                debug!("Map toggle: Playing → Map");
                next_state.set(GameState::Map);
            }
            GameState::Map => {
                debug!("Map toggle: Map → Playing");
                next_state.set(GameState::Playing);
            }
            _ => {}
        }
    }
    // Also allow ESC to close the map.
    if *state.get() == GameState::Map && key.just_pressed(KeyCode::Escape) {
        debug!("Map toggle: ESC → Playing");
        next_state.set(GameState::Playing);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toggle_map_from_playing() {
        let mut key = ButtonInput::<KeyCode>::default();
        key.press(KeyCode::KeyM);

        let state = State::new(GameState::Playing);
        let mut next_state = NextState::<GameState>::default();

        if key.just_pressed(KeyCode::KeyM) {
            match state.get() {
                GameState::Playing => next_state.set(GameState::Map),
                GameState::Map => next_state.set(GameState::Playing),
                _ => {}
            }
        }

        assert!(matches!(next_state, NextState::Pending(GameState::Map)));
    }

    #[test]
    fn toggle_map_from_map() {
        let mut key = ButtonInput::<KeyCode>::default();
        key.press(KeyCode::KeyM);

        let state = State::new(GameState::Map);
        let mut next_state = NextState::<GameState>::default();

        if key.just_pressed(KeyCode::KeyM) {
            match state.get() {
                GameState::Playing => next_state.set(GameState::Map),
                GameState::Map => next_state.set(GameState::Playing),
                _ => {}
            }
        }

        assert!(matches!(next_state, NextState::Pending(GameState::Playing)));
    }

    #[test]
    fn esc_closes_map() {
        let mut key = ButtonInput::<KeyCode>::default();
        key.press(KeyCode::Escape);

        let state = State::new(GameState::Map);
        let mut next_state = NextState::<GameState>::default();

        if *state.get() == GameState::Map && key.just_pressed(KeyCode::Escape) {
            next_state.set(GameState::Playing);
        }

        assert!(matches!(next_state, NextState::Pending(GameState::Playing)));
    }

    #[test]
    fn m_key_ignored_in_other_states() {
        let mut key = ButtonInput::<KeyCode>::default();
        key.press(KeyCode::KeyM);

        let state = State::new(GameState::Paused);
        let mut next_state = NextState::<GameState>::default();

        if key.just_pressed(KeyCode::KeyM) {
            match state.get() {
                GameState::Playing => next_state.set(GameState::Map),
                GameState::Map => next_state.set(GameState::Playing),
                _ => {}
            }
        }

        assert!(matches!(next_state, NextState::Unchanged));
    }
}
