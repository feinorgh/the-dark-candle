// Map overlay UI: spawns the full-screen map panel with tab bar.

use bevy::ecs::hierarchy::ChildSpawnerCommands;
use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll};
use bevy::prelude::*;

/// Which map view is active.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MapView {
    #[default]
    Local,
    Global,
}

/// Persistent state for the map overlay across open/close cycles.
#[derive(Resource, Debug, Clone)]
pub struct MapViewState {
    pub view: MapView,
    /// Local map zoom: pixels per chunk column. 1, 4, or 8.
    pub local_zoom: u32,
    /// Global map zoom (1.0 = full view, higher = zoomed in).
    pub global_zoom: f32,
    /// Global map pan offset in normalised UV coordinates.
    pub global_pan: Vec2,
    /// Whether the mouse is being dragged for panning.
    pub dragging: bool,
}

impl Default for MapViewState {
    fn default() -> Self {
        Self {
            view: MapView::Local,
            local_zoom: 4,
            global_zoom: 1.0,
            global_pan: Vec2::ZERO,
            dragging: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Marker components
// ---------------------------------------------------------------------------

#[derive(Component)]
pub struct MapOverlay;

#[derive(Component)]
pub struct MapTabButton(pub MapView);

#[derive(Component)]
pub struct MapImageNode;

#[derive(Component)]
pub struct MapCoordText;

#[derive(Component)]
pub struct PlayerMarker;

// ---------------------------------------------------------------------------
// Spawn / despawn
// ---------------------------------------------------------------------------

pub fn spawn_map_overlay(mut commands: Commands, mut time: ResMut<Time<Virtual>>) {
    time.pause();

    commands
        .spawn((
            MapOverlay,
            Node {
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                flex_direction: FlexDirection::Column,
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.85)),
        ))
        .with_children(|parent| {
            // Tab bar
            parent
                .spawn(Node {
                    flex_direction: FlexDirection::Row,
                    justify_content: JustifyContent::Center,
                    column_gap: Val::Px(8.0),
                    margin: UiRect::vertical(Val::Px(12.0)),
                    ..default()
                })
                .with_children(|tabs| {
                    spawn_tab_button(tabs, "Local Map", MapView::Local);
                    spawn_tab_button(tabs, "Global Map", MapView::Global);
                });

            // Map image container (fills remaining space)
            parent
                .spawn((
                    Node {
                        width: Val::Percent(90.0),
                        height: Val::Percent(75.0),
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        ..default()
                    },
                    BackgroundColor(Color::srgb(0.05, 0.05, 0.08)),
                ))
                .with_children(|container| {
                    // The actual map image
                    container.spawn((
                        MapImageNode,
                        ImageNode::default(),
                        Node {
                            width: Val::Percent(100.0),
                            height: Val::Percent(100.0),
                            ..default()
                        },
                    ));

                    // Player marker (small bright dot)
                    container.spawn((
                        PlayerMarker,
                        Node {
                            position_type: PositionType::Absolute,
                            width: Val::Px(8.0),
                            height: Val::Px(8.0),
                            left: Val::Percent(50.0),
                            top: Val::Percent(50.0),
                            ..default()
                        },
                        BackgroundColor(Color::srgb(1.0, 0.3, 0.3)),
                    ));
                });

            // Coordinate text at the bottom
            parent.spawn((
                MapCoordText,
                Text::new("Position: (0, 0, 0)"),
                TextFont {
                    font_size: 14.0,
                    ..default()
                },
                TextColor(Color::srgba(0.7, 0.7, 0.7, 1.0)),
                Node {
                    margin: UiRect::top(Val::Px(8.0)),
                    ..default()
                },
            ));

            // Help text
            parent.spawn((
                Text::new("[M] Close  [Tab] Switch View  [Scroll] Zoom"),
                TextFont {
                    font_size: 12.0,
                    ..default()
                },
                TextColor(Color::srgba(0.5, 0.5, 0.5, 1.0)),
                Node {
                    margin: UiRect::top(Val::Px(4.0)),
                    ..default()
                },
            ));
        });
}

fn spawn_tab_button(parent: &mut ChildSpawnerCommands, label: &str, view: MapView) {
    parent
        .spawn((
            MapTabButton(view),
            Button,
            Node {
                padding: UiRect::axes(Val::Px(20.0), Val::Px(8.0)),
                ..default()
            },
            BackgroundColor(Color::srgb(0.2, 0.2, 0.25)),
        ))
        .with_child((
            Text::new(label),
            TextFont {
                font_size: 16.0,
                ..default()
            },
            TextColor(Color::WHITE),
        ));
}

pub fn despawn_map_overlay(
    mut commands: Commands,
    mut time: ResMut<Time<Virtual>>,
    q: Query<Entity, With<MapOverlay>>,
) {
    time.unpause();
    for e in &q {
        commands.entity(e).despawn();
    }
}

// ---------------------------------------------------------------------------
// Input: tab switching
// ---------------------------------------------------------------------------

pub fn tab_switch(
    key: Res<ButtonInput<KeyCode>>,
    mut state: ResMut<MapViewState>,
    mut tab_q: Query<(&MapTabButton, &mut BackgroundColor)>,
) {
    if key.just_pressed(KeyCode::Tab) {
        state.view = match state.view {
            MapView::Local => MapView::Global,
            MapView::Global => MapView::Local,
        };
    }

    // Update tab button highlights.
    for (tab, mut bg) in &mut tab_q {
        *bg = if tab.0 == state.view {
            BackgroundColor(Color::srgb(0.35, 0.35, 0.5))
        } else {
            BackgroundColor(Color::srgb(0.2, 0.2, 0.25))
        };
    }
}

// ---------------------------------------------------------------------------
// Input: scroll zoom
// ---------------------------------------------------------------------------

pub fn scroll_zoom(scroll: Res<AccumulatedMouseScroll>, mut state: ResMut<MapViewState>) {
    let dy = scroll.delta.y;
    if dy.abs() < 0.01 {
        return;
    }

    match state.view {
        MapView::Local => {
            const ZOOM_LEVELS: [u32; 4] = [1, 2, 4, 8];
            let current_idx = ZOOM_LEVELS
                .iter()
                .position(|&z| z == state.local_zoom)
                .unwrap_or(2);
            let new_idx = if dy > 0.0 {
                (current_idx + 1).min(ZOOM_LEVELS.len() - 1)
            } else {
                current_idx.saturating_sub(1)
            };
            state.local_zoom = ZOOM_LEVELS[new_idx];
        }
        MapView::Global => {
            if dy > 0.0 {
                state.global_zoom = (state.global_zoom * 1.25).min(8.0);
            } else {
                state.global_zoom = (state.global_zoom / 1.25).max(1.0);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Input: mouse drag pan (global map)
// ---------------------------------------------------------------------------

pub fn map_pan(
    mouse: Res<ButtonInput<MouseButton>>,
    motion: Res<AccumulatedMouseMotion>,
    mut state: ResMut<MapViewState>,
) {
    if state.view != MapView::Global || state.global_zoom <= 1.0 {
        state.dragging = false;
        return;
    }

    if mouse.just_pressed(MouseButton::Left) {
        state.dragging = true;
    }
    if mouse.just_released(MouseButton::Left) {
        state.dragging = false;
    }

    if state.dragging {
        let delta = motion.delta;
        if delta != Vec2::ZERO {
            let pan_speed = 0.001 / state.global_zoom;
            state.global_pan.x -= delta.x * pan_speed;
            state.global_pan.y -= delta.y * pan_speed;
            let max_pan = 0.5 - 0.5 / state.global_zoom;
            state.global_pan = state
                .global_pan
                .clamp(Vec2::splat(-max_pan), Vec2::splat(max_pan));
        }
    }
}
