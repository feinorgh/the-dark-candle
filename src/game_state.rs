// Game state machine: WorldCreation → Loading → Playing ↔ Paused.
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

/// Marker resource: when present, the WorldCreation state is skipped
/// (scene was selected via CLI).
#[derive(Resource)]
pub struct SkipWorldCreation;

/// Top-level game state.
#[derive(States, Debug, Clone, PartialEq, Eq, Hash, Default)]
pub enum GameState {
    /// World creation: user configures world parameters before generating.
    #[default]
    WorldCreation,
    /// Terrain is generating, loading screen shown.
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
            // World creation screen
            .add_systems(OnEnter(GameState::WorldCreation), spawn_world_creation_screen)
            .add_systems(OnExit(GameState::WorldCreation), despawn_world_creation_screen)
            .add_systems(
                Update,
                world_creation_interaction.run_if(in_state(GameState::WorldCreation)),
            )
            // Loading screen
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
// World creation screen
// ---------------------------------------------------------------------------

use crate::world::noise::NoiseConfig;
use crate::world::planet::PlanetConfig;
use crate::world::scene_presets::ScenePreset;

#[derive(Component)]
struct WorldCreationScreen;

#[derive(Component)]
enum WorldCreationButton {
    Preset(ScenePreset),
    Generate,
}

/// Currently selected preset for the world creation screen.
#[derive(Resource, Default)]
struct SelectedPreset(Option<ScenePreset>);

fn spawn_world_creation_screen(
    mut commands: Commands,
    skip: Option<Res<SkipWorldCreation>>,
    mut next_state: ResMut<NextState<GameState>>,
) {
    if skip.is_some() {
        // CLI already configured the world — jump straight to Loading.
        next_state.set(GameState::Loading);
        return;
    }

    commands.init_resource::<SelectedPreset>();

    commands
        .spawn((
            WorldCreationScreen,
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
            // Title
            parent.spawn((
                Text::new("Create Your World"),
                TextFont {
                    font_size: 42.0,
                    ..default()
                },
                TextColor(Color::srgb(0.9, 0.7, 0.3)),
            ));

            // Subtitle
            parent.spawn((
                Text::new("Choose a terrain preset:"),
                TextFont {
                    font_size: 18.0,
                    ..default()
                },
                TextColor(Color::srgb(0.7, 0.7, 0.7)),
            ));

            // Preset buttons in a row
            parent
                .spawn(Node {
                    flex_direction: FlexDirection::Row,
                    column_gap: Val::Px(10.0),
                    flex_wrap: FlexWrap::Wrap,
                    justify_content: JustifyContent::Center,
                    ..default()
                })
                .with_children(|row| {
                    let presets = [
                        (ScenePreset::ValleyRiver, "Valley River"),
                        (ScenePreset::Alpine, "Alpine"),
                        (ScenePreset::Archipelago, "Archipelago"),
                        (ScenePreset::DesertCanyon, "Desert Canyon"),
                        (ScenePreset::RollingPlains, "Rolling Plains"),
                        (ScenePreset::Volcanic, "Volcanic"),
                        (ScenePreset::TundraFjords, "Tundra Fjords"),
                    ];

                    for (preset, label) in presets {
                        spawn_preset_button(row, label, preset);
                    }
                });

            // Spacer
            parent.spawn(Node {
                height: Val::Px(20.0),
                ..default()
            });

            // Generate button
            spawn_world_button(parent, "Generate World", WorldCreationButton::Generate);
        });
}

fn spawn_preset_button(
    parent: &mut ChildSpawnerCommands,
    label: &str,
    preset: ScenePreset,
) {
    parent
        .spawn((
            WorldCreationButton::Preset(preset),
            Button,
            Node {
                width: Val::Px(140.0),
                height: Val::Px(40.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(Color::srgb(0.15, 0.15, 0.2)),
        ))
        .with_child((
            Text::new(label.to_string()),
            TextFont {
                font_size: 16.0,
                ..default()
            },
            TextColor(Color::WHITE),
        ));
}

fn spawn_world_button(
    parent: &mut ChildSpawnerCommands,
    label: &str,
    marker: WorldCreationButton,
) {
    parent
        .spawn((
            marker,
            Button,
            Node {
                width: Val::Px(220.0),
                height: Val::Px(55.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(Color::srgb(0.2, 0.5, 0.3)),
        ))
        .with_child((
            Text::new(label.to_string()),
            TextFont {
                font_size: 24.0,
                ..default()
            },
            TextColor(Color::WHITE),
        ));
}

fn world_creation_interaction(
    mut commands: Commands,
    mut next_state: ResMut<NextState<GameState>>,
    mut selected: ResMut<SelectedPreset>,
    mut interaction_query: Query<
        (&Interaction, &WorldCreationButton, &mut BackgroundColor),
        Changed<Interaction>,
    >,
) {
    for (interaction, button, mut bg) in &mut interaction_query {
        match *interaction {
            Interaction::Pressed => match button {
                WorldCreationButton::Preset(preset) => {
                    selected.0 = Some(preset.clone());
                }
                WorldCreationButton::Generate => {
                    let config = if let Some(ref preset) = selected.0 {
                        preset.planet_config()
                    } else {
                        PlanetConfig {
                            noise: Some(NoiseConfig::default()),
                            ..Default::default()
                        }
                    };
                    commands.insert_resource(config);
                    next_state.set(GameState::Loading);
                }
            },
            Interaction::Hovered => {
                let is_selected = matches!(button, WorldCreationButton::Preset(p) if selected.0.as_ref() == Some(p));
                if is_selected {
                    *bg = BackgroundColor(Color::srgb(0.3, 0.6, 0.4));
                } else {
                    *bg = BackgroundColor(Color::srgb(0.25, 0.25, 0.3));
                }
            }
            Interaction::None => {
                let is_selected = matches!(button, WorldCreationButton::Preset(p) if selected.0.as_ref() == Some(p));
                if is_selected {
                    *bg = BackgroundColor(Color::srgb(0.25, 0.5, 0.35));
                } else if matches!(button, WorldCreationButton::Generate) {
                    *bg = BackgroundColor(Color::srgb(0.2, 0.5, 0.3));
                } else {
                    *bg = BackgroundColor(Color::srgb(0.15, 0.15, 0.2));
                }
            }
        }
    }
}

fn despawn_world_creation_screen(
    mut commands: Commands,
    query: Query<Entity, With<WorldCreationScreen>>,
) {
    for entity in &query {
        commands.entity(entity).despawn();
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
    fn default_state_is_world_creation() {
        assert_eq!(GameState::default(), GameState::WorldCreation);
    }
}
