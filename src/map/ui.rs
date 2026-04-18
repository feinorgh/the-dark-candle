// Map overlay UI: spawns the full-screen map panel with tab bar.

use bevy::ecs::hierarchy::ChildSpawnerCommands;
use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll};
use bevy::math::DVec3;
use bevy::prelude::*;
use bevy::window::PrimaryWindow;

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
pub struct MapContainer;

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

            // Map image container (fills remaining space, clips zoomed content)
            parent
                .spawn((
                    MapContainer,
                    Node {
                        width: Val::Percent(90.0),
                        height: Val::Percent(75.0),
                        overflow: Overflow::clip(),
                        ..default()
                    },
                    BackgroundColor(Color::srgb(0.05, 0.05, 0.08)),
                ))
                .with_children(|container| {
                    // The actual map image (absolutely positioned for zoom/pan)
                    container.spawn((
                        MapImageNode,
                        ImageNode::default(),
                        Node {
                            position_type: PositionType::Absolute,
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
                Text::new("[M] Close  [Tab] Switch View  [Scroll] Zoom  [Right-Click] Teleport"),
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
            state.global_pan.x += delta.x * pan_speed;
            state.global_pan.y += delta.y * pan_speed;
            let max_pan = 0.5 - 0.5 / state.global_zoom;
            state.global_pan = state
                .global_pan
                .clamp(Vec2::splat(-max_pan), Vec2::splat(max_pan));
        }
    }
}

// ---------------------------------------------------------------------------
// Pixel → lat/lon inverse equirectangular mapping
// ---------------------------------------------------------------------------

/// Convert a screen-space cursor position into (lat, lon) in radians,
/// accounting for the map container's zoom and pan state.
///
/// `cursor_pos` is in window coordinates (origin top-left).
/// `container_rect` is the computed rect of the map image node.
///
/// Returns `None` if the cursor is outside the map image bounds.
pub fn pixel_to_lat_lon(
    cursor_pos: Vec2,
    container_rect: &ComputedNode,
    container_global: &UiGlobalTransform,
    state: &MapViewState,
) -> Option<(f64, f64)> {
    use std::f64::consts::{PI, TAU};

    // Map image rect in window space.
    let size = container_rect.size();
    if size.x < 1.0 || size.y < 1.0 {
        return None;
    }
    // The node's global translation gives us the centre of the node in screen space.
    let center = container_global.translation;
    let half = size * 0.5;
    let min = center - half;

    // Normalise cursor position to [0,1] within the container.
    let rel = cursor_pos - min;
    let u_screen = rel.x / size.x;
    let v_screen = rel.y / size.y;

    if !(0.0..=1.0).contains(&u_screen) || !(0.0..=1.0).contains(&v_screen) {
        return None; // outside the map image
    }

    // Invert zoom + pan to get base UV.
    let zoom = state.global_zoom;
    let pan = state.global_pan;
    let u = (u_screen - 0.5 - pan.x * zoom) / zoom + 0.5;
    let v = (v_screen - 0.5 - pan.y * zoom) / zoom + 0.5;

    // Equirectangular: u ∈ [0,1] → lon ∈ [-π, π], v ∈ [0,1] → lat ∈ [π/2, -π/2]
    let lon = ((u as f64) - 0.5) * TAU;
    let lat = (0.5 - (v as f64)) * PI;

    Some((lat.clamp(-PI / 2.0, PI / 2.0), lon.clamp(-PI, PI)))
}

// ---------------------------------------------------------------------------
// Right-click teleport (global map)
// ---------------------------------------------------------------------------

use crate::camera::{EYE_HEIGHT, FpsCamera};
use crate::floating_origin::{RenderOrigin, WorldPosition};
use crate::game_state::GameState;
use crate::hud::Player;
use crate::world::chunk_manager::TerrainGeneratorRes;
use crate::world::planet::PlanetConfig;

#[allow(clippy::too_many_arguments)]
pub fn map_teleport(
    mouse: Res<ButtonInput<MouseButton>>,
    state: Res<MapViewState>,
    planet_config: Option<Res<PlanetConfig>>,
    terrain_gen: Option<Res<TerrainGeneratorRes>>,
    window_q: Query<&Window, With<PrimaryWindow>>,
    map_node_q: Query<(&ComputedNode, &UiGlobalTransform), With<MapContainer>>,
    mut camera_q: Query<(&mut WorldPosition, &mut FpsCamera), With<Player>>,
    mut render_origin: ResMut<RenderOrigin>,
    mut next_state: ResMut<NextState<GameState>>,
) {
    if state.view != MapView::Global {
        return;
    }
    if !mouse.just_pressed(MouseButton::Right) {
        return;
    }

    // Get cursor position from the primary window.
    let Ok(window) = window_q.single() else {
        return;
    };
    let Some(cursor_pos) = window.cursor_position() else {
        return;
    };

    // Get the map container's computed layout.
    let Ok((computed_node, global_tf)) = map_node_q.single() else {
        return;
    };

    let Some((lat, lon)) = pixel_to_lat_lon(cursor_pos, computed_node, global_tf, &state) else {
        return;
    };

    let Some(planet) = planet_config else {
        return;
    };
    let Some(tgen) = terrain_gen else {
        return;
    };

    // Sample terrain height at target.
    let surface_r = tgen.0.sample_surface_radius_at(lat, lon) as f32;
    let sea_r = planet.sea_level_radius as f32;

    // Build world position.
    let dir = DVec3::new(
        lat.cos() * lon.cos(),
        lat.sin(),
        lat.cos() * lon.sin(),
    )
    .normalize();
    let teleport_pos = dir * ((surface_r.max(sea_r) + EYE_HEIGHT) as f64);

    // Move the camera.
    if let Ok((mut wp, mut fps)) = camera_q.single_mut() {
        wp.0 = teleport_pos;
        render_origin.0 = teleport_pos;
        // Reset vertical velocity so we don't carry momentum.
        fps.vertical_velocity = 0.0;
        fps.grounded = false;

        let lat_deg = lat.to_degrees();
        let lon_deg = lon.to_degrees();
        let ns = if lat_deg >= 0.0 { "N" } else { "S" };
        let ew = if lon_deg >= 0.0 { "E" } else { "W" };
        info!(
            "Teleported to {:.1}°{ns} {:.1}°{ew} (r={:.0}m)",
            lat_deg.abs(),
            lon_deg.abs(),
            surface_r.max(sea_r),
        );
    }

    // Switch back to playing.
    next_state.set(GameState::Playing);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a `ComputedNode` with a given size.
    fn make_computed_node(width: f32, height: f32) -> ComputedNode {
        ComputedNode {
            size: Vec2::new(width, height),
            unrounded_size: Vec2::new(width, height),
            inverse_scale_factor: 1.0,
            ..Default::default()
        }
    }

    #[test]
    fn pixel_to_lat_lon_center_is_equator_prime() {
        // No zoom, no pan — the center of the map should be (0, 0).
        let state = MapViewState {
            view: MapView::Global,
            global_zoom: 1.0,
            global_pan: Vec2::ZERO,
            ..Default::default()
        };

        // Simulate a 1024×512 container centered at (512, 256) in screen space.
        let computed = make_computed_node(1024.0, 512.0);
        let global_tf = UiGlobalTransform::from_translation(Vec2::new(512.0, 256.0));

        // Click at the center of the container.
        let cursor = Vec2::new(512.0, 256.0);

        let result = pixel_to_lat_lon(cursor, &computed, &global_tf, &state);
        assert!(result.is_some());
        let (lat, lon) = result.unwrap();
        assert!(lat.abs() < 0.01, "center lat should be ~0, got {lat}");
        assert!(lon.abs() < 0.01, "center lon should be ~0, got {lon}");
    }

    #[test]
    fn pixel_to_lat_lon_top_left_is_north_west() {
        let state = MapViewState {
            view: MapView::Global,
            global_zoom: 1.0,
            global_pan: Vec2::ZERO,
            ..Default::default()
        };

        let computed = make_computed_node(1024.0, 512.0);
        let global_tf = UiGlobalTransform::from_translation(Vec2::new(512.0, 256.0));

        // Top-left corner: should be ~(90°N, -180°)
        let cursor = Vec2::new(0.0, 0.0);
        let (lat, lon) = pixel_to_lat_lon(cursor, &computed, &global_tf, &state).unwrap();
        assert!(lat > 1.5, "top-left lat should be ~π/2, got {lat}");
        assert!(lon < -3.0, "top-left lon should be ~-π, got {lon}");
    }

    #[test]
    fn pixel_to_lat_lon_outside_returns_none() {
        let state = MapViewState::default();
        let computed = make_computed_node(1024.0, 512.0);
        let global_tf = UiGlobalTransform::from_translation(Vec2::new(512.0, 256.0));

        // Far outside the map image bounds.
        let cursor = Vec2::new(2000.0, 2000.0);
        assert!(pixel_to_lat_lon(cursor, &computed, &global_tf, &state).is_none());
    }

    #[test]
    fn pixel_to_lat_lon_with_zoom_and_pan() {
        let state = MapViewState {
            view: MapView::Global,
            global_zoom: 2.0,
            global_pan: Vec2::new(0.1, -0.05),
            ..Default::default()
        };

        let computed = make_computed_node(1024.0, 512.0);
        let global_tf = UiGlobalTransform::from_translation(Vec2::new(512.0, 256.0));

        // Center click at 2× zoom with pan — should NOT be (0,0).
        let cursor = Vec2::new(512.0, 256.0);
        let (lat, lon) = pixel_to_lat_lon(cursor, &computed, &global_tf, &state).unwrap();
        // With pan (0.1, -0.05) at zoom 2, the center maps to a shifted lat/lon.
        assert!(
            lat.abs() > 0.05 || lon.abs() > 0.05,
            "should be offset from origin"
        );
    }
}
