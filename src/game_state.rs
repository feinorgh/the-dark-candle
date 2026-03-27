// Game state machine: Loading → Playing ↔ Paused.
//
// Uses Bevy 0.18 States for state transitions.  FixedUpdate is paused via
// `Time<Virtual>` when not Playing (no need to add run_if to every system).
// Camera input is gated by the camera module checking GameState.

use bevy::app::AppExit;
use bevy::ecs::hierarchy::ChildSpawnerCommands;
use bevy::ecs::message::MessageWriter;
use bevy::prelude::*;

use crate::persistence::{LoadRequest, SaveRequest, SaveSlot, list_save_slots};
use crate::world::chunk_manager::PendingChunks;

/// Top-level game state.
#[derive(States, Debug, Clone, PartialEq, Eq, Hash, Default)]
pub enum GameState {
    /// Initial state: terrain is generating, loading screen shown.
    #[default]
    Loading,
    /// Gameplay active: cursor locked, world simulating.
    Playing,
    /// Overlay pause menu: world frozen, cursor unlocked.
    Paused,
}

pub struct GameStatePlugin;

impl Plugin for GameStatePlugin {
    fn build(&self, app: &mut App) {
        app.init_state::<GameState>()
            .add_systems(OnEnter(GameState::Loading), spawn_loading_screen)
            .add_systems(OnExit(GameState::Loading), despawn_loading_screen)
            .add_systems(
                Update,
                check_loading_complete.run_if(in_state(GameState::Loading)),
            )
            .add_systems(OnEnter(GameState::Playing), enter_playing)
            .add_systems(OnEnter(GameState::Paused), enter_paused)
            .add_systems(OnExit(GameState::Paused), despawn_pause_menu)
            .add_systems(
                Update,
                toggle_pause.run_if(in_state(GameState::Playing).or(in_state(GameState::Paused))),
            )
            .add_systems(
                Update,
                pause_menu_interaction.run_if(in_state(GameState::Paused)),
            );
    }
}

// ---------------------------------------------------------------------------
// Loading screen
// ---------------------------------------------------------------------------

#[derive(Component)]
struct LoadingScreen;

fn spawn_loading_screen(mut commands: Commands) {
    commands
        .spawn((
            LoadingScreen,
            Node {
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(20.0),
                ..default()
            },
            BackgroundColor(Color::srgb(0.05, 0.05, 0.08)),
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("The Dark Candle"),
                TextFont {
                    font_size: 48.0,
                    ..default()
                },
                TextColor(Color::srgb(0.9, 0.7, 0.3)),
            ));
            parent.spawn((
                LoadingText,
                Text::new("Generating world..."),
                TextFont {
                    font_size: 20.0,
                    ..default()
                },
                TextColor(Color::srgb(0.7, 0.7, 0.7)),
            ));
        });
}

#[derive(Component)]
struct LoadingText;

fn despawn_loading_screen(mut commands: Commands, q: Query<Entity, With<LoadingScreen>>) {
    for entity in &q {
        commands.entity(entity).despawn();
    }
}

/// Transition to Playing once there are no pending terrain generation tasks.
fn check_loading_complete(
    pending: Option<Res<PendingChunks>>,
    mut next_state: ResMut<NextState<GameState>>,
    mut frames_ready: Local<u32>,
) {
    let pending_count = pending.map(|p| p.len()).unwrap_or(0);
    if pending_count == 0 {
        // Wait a few frames after generation completes to let meshing catch up.
        *frames_ready += 1;
        if *frames_ready > 10 {
            next_state.set(GameState::Playing);
        }
    } else {
        *frames_ready = 0;
    }
}

// ---------------------------------------------------------------------------
// Playing / Paused transitions
// ---------------------------------------------------------------------------

fn enter_playing(mut time: ResMut<Time<Virtual>>) {
    time.unpause();
}

fn enter_paused(mut time: ResMut<Time<Virtual>>, mut commands: Commands) {
    time.pause();
    spawn_pause_menu(&mut commands);
}

fn toggle_pause(
    key: Res<ButtonInput<KeyCode>>,
    state: Res<State<GameState>>,
    mut next_state: ResMut<NextState<GameState>>,
) {
    if key.just_pressed(KeyCode::Escape) {
        match state.get() {
            GameState::Playing => next_state.set(GameState::Paused),
            GameState::Paused => next_state.set(GameState::Playing),
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Pause menu UI
// ---------------------------------------------------------------------------

#[derive(Component)]
struct PauseMenu;

/// Marker for the save/load slot sub-panel.
#[derive(Component)]
struct SlotPanel;

#[derive(Component)]
enum PauseButton {
    Resume,
    SaveGame,
    LoadGame,
    SaveSlot(SaveSlot),
    LoadSlot(SaveSlot),
    BackToMenu,
    Quit,
}

fn spawn_pause_menu(commands: &mut Commands) {
    commands
        .spawn((
            PauseMenu,
            Node {
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(16.0),
                ..default()
            },
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.6)),
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("Paused"),
                TextFont {
                    font_size: 40.0,
                    ..default()
                },
                TextColor(Color::WHITE),
            ));
            spawn_button(parent, "Resume", PauseButton::Resume);
            spawn_button(parent, "Save Game", PauseButton::SaveGame);
            spawn_button(parent, "Load Game", PauseButton::LoadGame);
            spawn_button(parent, "Quit", PauseButton::Quit);
        });
}

fn spawn_slot_panel(commands: &mut Commands, mode: SlotPanelMode) {
    let title = match mode {
        SlotPanelMode::Save => "Save to slot:",
        SlotPanelMode::Load => "Load from slot:",
    };

    let slots = list_save_slots();

    commands
        .spawn((
            SlotPanel,
            Node {
                position_type: PositionType::Absolute,
                right: Val::Px(20.0),
                top: Val::Percent(30.0),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(8.0),
                padding: UiRect::all(Val::Px(16.0)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.1, 0.1, 0.15, 0.9)),
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new(title.to_string()),
                TextFont {
                    font_size: 18.0,
                    ..default()
                },
                TextColor(Color::srgb(0.9, 0.7, 0.3)),
            ));

            for (slot, label, exists) in &slots {
                let display = if *exists {
                    format!("{label} ●")
                } else {
                    format!("{label} (empty)")
                };

                let marker = match mode {
                    SlotPanelMode::Save => PauseButton::SaveSlot(*slot),
                    SlotPanelMode::Load => {
                        if *exists {
                            PauseButton::LoadSlot(*slot)
                        } else {
                            continue; // skip empty slots in load mode
                        }
                    }
                };

                spawn_button(parent, &display, marker);
            }

            spawn_button(parent, "Back", PauseButton::BackToMenu);
        });
}

#[derive(Clone, Copy)]
enum SlotPanelMode {
    Save,
    Load,
}

fn spawn_button(parent: &mut ChildSpawnerCommands, label: &str, marker: PauseButton) {
    parent
        .spawn((
            marker,
            Button,
            Node {
                width: Val::Px(200.0),
                height: Val::Px(50.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(Color::srgb(0.2, 0.2, 0.25)),
        ))
        .with_child((
            Text::new(label.to_string()),
            TextFont {
                font_size: 22.0,
                ..default()
            },
            TextColor(Color::WHITE),
        ));
}

fn pause_menu_interaction(
    mut commands: Commands,
    mut interaction_q: Query<
        (&Interaction, &PauseButton, &mut BackgroundColor),
        Changed<Interaction>,
    >,
    mut next_state: ResMut<NextState<GameState>>,
    mut exit: MessageWriter<AppExit>,
    slot_panel: Query<Entity, With<SlotPanel>>,
) {
    for (interaction, button, mut bg) in &mut interaction_q {
        match interaction {
            Interaction::Pressed => {
                match button {
                    PauseButton::Resume => {
                        next_state.set(GameState::Playing);
                    }
                    PauseButton::SaveGame => {
                        // Despawn any existing slot panel, then show save slots
                        for e in &slot_panel {
                            commands.entity(e).despawn();
                        }
                        spawn_slot_panel(&mut commands, SlotPanelMode::Save);
                    }
                    PauseButton::LoadGame => {
                        for e in &slot_panel {
                            commands.entity(e).despawn();
                        }
                        spawn_slot_panel(&mut commands, SlotPanelMode::Load);
                    }
                    PauseButton::SaveSlot(slot) => {
                        commands.insert_resource(SaveRequest(*slot));
                        // Close slot panel
                        for e in &slot_panel {
                            commands.entity(e).despawn();
                        }
                    }
                    PauseButton::LoadSlot(slot) => {
                        commands.insert_resource(LoadRequest(*slot));
                        // Close panels and resume
                        for e in &slot_panel {
                            commands.entity(e).despawn();
                        }
                        next_state.set(GameState::Playing);
                    }
                    PauseButton::BackToMenu => {
                        for e in &slot_panel {
                            commands.entity(e).despawn();
                        }
                    }
                    PauseButton::Quit => {
                        // Auto-save before quitting
                        commands.insert_resource(SaveRequest(SaveSlot::Auto));
                        // Note: the save system runs this frame before exit
                        exit.write(AppExit::Success);
                    }
                }
            }
            Interaction::Hovered => {
                *bg = BackgroundColor(Color::srgb(0.3, 0.3, 0.38));
            }
            Interaction::None => {
                *bg = BackgroundColor(Color::srgb(0.2, 0.2, 0.25));
            }
        }
    }
}

// Clean up pause menu when leaving Paused state (handles both Resume and any
// other transition).  Registered as OnExit so we don't need the menu to clean
// itself up in the interaction handler.
fn despawn_pause_menu(
    mut commands: Commands,
    q: Query<Entity, With<PauseMenu>>,
    slot_q: Query<Entity, With<SlotPanel>>,
) {
    for entity in &q {
        commands.entity(entity).despawn();
    }
    for entity in &slot_q {
        commands.entity(entity).despawn();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_state_is_loading() {
        assert_eq!(GameState::default(), GameState::Loading);
    }
}
