// Block interaction: crosshair, world-space raycast, break/place, hotbar.
//
// The player can target blocks via a DDA raycast from the camera, break them
// with left-click, and place blocks from the hotbar with right-click.

use bevy::input::mouse::AccumulatedMouseScroll;
use bevy::prelude::*;

use crate::data::MaterialRegistry;
use crate::game_state::GameState;
use crate::hud::Player;
use crate::world::chunk::{CHUNK_SIZE, Chunk, ChunkCoord};
use crate::world::chunk_manager::ChunkMap;
use crate::world::voxel::MaterialId;

pub struct InteractionPlugin;

impl Plugin for InteractionPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Hotbar>()
            .init_resource::<BlockTarget>()
            .add_systems(OnEnter(GameState::Playing), spawn_crosshair)
            .add_systems(OnExit(GameState::Playing), despawn_crosshair)
            .add_systems(OnEnter(GameState::Playing), spawn_hotbar_ui)
            .add_systems(OnExit(GameState::Playing), despawn_hotbar_ui)
            .add_systems(
                Update,
                (update_block_target, break_block, place_block)
                    .chain()
                    .run_if(in_state(GameState::Playing)),
            )
            .add_systems(
                Update,
                (
                    hotbar_input,
                    update_hotbar_ui_system,
                    update_block_highlight,
                )
                    .run_if(in_state(GameState::Playing)),
            );
    }
}

// ---------------------------------------------------------------------------
// World-space DDA raycast
// ---------------------------------------------------------------------------

/// Result of a world-space voxel raycast.
#[derive(Debug, Clone, Copy)]
pub struct WorldHit {
    /// World-space voxel coordinates of the hit block.
    pub voxel_x: i32,
    pub voxel_y: i32,
    pub voxel_z: i32,
    /// Face normal of the entry face.
    pub normal: IVec3,
    /// Distance from origin to hit.
    pub distance: f32,
}

/// Maximum interaction reach in meters (voxels).
const REACH: f32 = 6.0;

/// Cast a ray through world-space voxels using DDA, checking each voxel
/// against loaded chunks. Returns the first non-air voxel hit.
fn world_raycast(
    origin: Vec3,
    dir: Vec3,
    max_dist: f32,
    chunk_map: &ChunkMap,
    chunks: &Query<&Chunk>,
) -> Option<WorldHit> {
    if dir.length_squared() < 1e-10 {
        return None;
    }
    let dir = dir.normalize();

    // Current voxel position (integer)
    let mut ix = origin.x.floor() as i32;
    let mut iy = origin.y.floor() as i32;
    let mut iz = origin.z.floor() as i32;

    let step_x: i32 = if dir.x >= 0.0 { 1 } else { -1 };
    let step_y: i32 = if dir.y >= 0.0 { 1 } else { -1 };
    let step_z: i32 = if dir.z >= 0.0 { 1 } else { -1 };

    let dt_x = if dir.x.abs() > 1e-10 {
        (1.0 / dir.x).abs()
    } else {
        f32::MAX
    };
    let dt_y = if dir.y.abs() > 1e-10 {
        (1.0 / dir.y).abs()
    } else {
        f32::MAX
    };
    let dt_z = if dir.z.abs() > 1e-10 {
        (1.0 / dir.z).abs()
    } else {
        f32::MAX
    };

    let mut t_max_x = if dir.x >= 0.0 {
        ((ix + 1) as f32 - origin.x) * dt_x
    } else {
        (origin.x - ix as f32) * dt_x
    };
    let mut t_max_y = if dir.y >= 0.0 {
        ((iy + 1) as f32 - origin.y) * dt_y
    } else {
        (origin.y - iy as f32) * dt_y
    };
    let mut t_max_z = if dir.z >= 0.0 {
        ((iz + 1) as f32 - origin.z) * dt_z
    } else {
        (origin.z - iz as f32) * dt_z
    };

    for _ in 0..256 {
        let t_min = t_max_x.min(t_max_y).min(t_max_z);
        if t_min > max_dist {
            return None;
        }

        // Step to next voxel
        let axis;
        if t_max_x <= t_max_y && t_max_x <= t_max_z {
            ix += step_x;
            t_max_x += dt_x;
            axis = 0;
        } else if t_max_y <= t_max_z {
            iy += step_y;
            t_max_y += dt_y;
            axis = 1;
        } else {
            iz += step_z;
            t_max_z += dt_z;
            axis = 2;
        }

        // Look up this voxel in loaded chunks
        let cc = ChunkCoord::from_voxel_pos(ix, iy, iz);
        let Some(entity) = chunk_map.get(&cc) else {
            continue;
        };
        let Ok(chunk) = chunks.get(entity) else {
            continue;
        };

        let origin_w = cc.world_origin();
        let lx = (ix - origin_w.x) as usize;
        let ly = (iy - origin_w.y) as usize;
        let lz = (iz - origin_w.z) as usize;

        if lx >= CHUNK_SIZE || ly >= CHUNK_SIZE || lz >= CHUNK_SIZE {
            continue;
        }

        let voxel = chunk.get(lx, ly, lz);
        if !voxel.is_air() {
            let normal = match axis {
                0 => IVec3::new(-step_x, 0, 0),
                1 => IVec3::new(0, -step_y, 0),
                _ => IVec3::new(0, 0, -step_z),
            };
            return Some(WorldHit {
                voxel_x: ix,
                voxel_y: iy,
                voxel_z: iz,
                normal,
                distance: t_min,
            });
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Block target (raycast result cached per frame)
// ---------------------------------------------------------------------------

/// Current block the player is looking at.
#[derive(Resource, Default)]
pub struct BlockTarget {
    pub hit: Option<WorldHit>,
}

fn update_block_target(
    cam_q: Query<&Transform, With<Player>>,
    chunk_map: Option<Res<ChunkMap>>,
    chunks: Query<&Chunk>,
    mut target: ResMut<BlockTarget>,
) {
    let Ok(transform) = cam_q.single() else {
        target.hit = None;
        return;
    };
    let Some(chunk_map) = chunk_map else {
        target.hit = None;
        return;
    };

    let origin = transform.translation;
    let dir = transform.forward().as_vec3();
    target.hit = world_raycast(origin, dir, REACH, &chunk_map, &chunks);
}

// ---------------------------------------------------------------------------
// Block break / place
// ---------------------------------------------------------------------------

/// How long (seconds) to hold left-click before breaking a block.
const BREAK_TIME: f32 = 0.3;

/// Tracks break progress for the currently targeted block.
#[derive(Resource, Default)]
struct BreakProgress {
    target: Option<(i32, i32, i32)>,
    elapsed: f32,
}

fn break_block(
    mouse: Res<ButtonInput<MouseButton>>,
    time: Res<Time>,
    target: Res<BlockTarget>,
    chunk_map: Option<Res<ChunkMap>>,
    mut chunks: Query<&mut Chunk>,
    mut progress: Local<BreakProgress>,
) {
    let Some(hit) = &target.hit else {
        progress.target = None;
        progress.elapsed = 0.0;
        return;
    };

    let Some(chunk_map) = chunk_map else {
        return;
    };

    let pos = (hit.voxel_x, hit.voxel_y, hit.voxel_z);

    if !mouse.pressed(MouseButton::Left) {
        progress.target = None;
        progress.elapsed = 0.0;
        return;
    }

    // Reset progress if target changed
    if progress.target != Some(pos) {
        progress.target = Some(pos);
        progress.elapsed = 0.0;
    }

    progress.elapsed += time.delta_secs();

    if progress.elapsed >= BREAK_TIME {
        // Break the block
        let cc = ChunkCoord::from_voxel_pos(hit.voxel_x, hit.voxel_y, hit.voxel_z);
        if let Some(entity) = chunk_map.get(&cc)
            && let Ok(mut chunk) = chunks.get_mut(entity)
        {
            let origin = cc.world_origin();
            let lx = (hit.voxel_x - origin.x) as usize;
            let ly = (hit.voxel_y - origin.y) as usize;
            let lz = (hit.voxel_z - origin.z) as usize;
            if lx < CHUNK_SIZE && ly < CHUNK_SIZE && lz < CHUNK_SIZE {
                chunk.set_material(lx, ly, lz, MaterialId::AIR);
            }
        }
        progress.target = None;
        progress.elapsed = 0.0;
    }
}

fn place_block(
    mouse: Res<ButtonInput<MouseButton>>,
    target: Res<BlockTarget>,
    hotbar: Res<Hotbar>,
    chunk_map: Option<Res<ChunkMap>>,
    mut chunks: Query<&mut Chunk>,
    cam_q: Query<&Transform, With<Player>>,
) {
    if !mouse.just_pressed(MouseButton::Right) {
        return;
    }

    let Some(hit) = &target.hit else {
        return;
    };
    let Some(chunk_map) = chunk_map else {
        return;
    };

    let material = hotbar.selected_material();

    // Place on the adjacent face
    let px = hit.voxel_x + hit.normal.x;
    let py = hit.voxel_y + hit.normal.y;
    let pz = hit.voxel_z + hit.normal.z;

    // Prevent placing inside the player (check player occupies 2 voxels vertically)
    if let Ok(transform) = cam_q.single() {
        let pos = transform.translation;
        let player_vx = pos.x.floor() as i32;
        let player_vy_feet = (pos.y - 1.7).floor() as i32; // feet
        let player_vy_head = pos.y.floor() as i32; // head
        let player_vz = pos.z.floor() as i32;
        if px == player_vx && pz == player_vz && (py == player_vy_feet || py == player_vy_head) {
            return;
        }
    }

    let cc = ChunkCoord::from_voxel_pos(px, py, pz);
    if let Some(entity) = chunk_map.get(&cc)
        && let Ok(mut chunk) = chunks.get_mut(entity)
    {
        let origin = cc.world_origin();
        let lx = (px - origin.x) as usize;
        let ly = (py - origin.y) as usize;
        let lz = (pz - origin.z) as usize;
        if lx < CHUNK_SIZE && ly < CHUNK_SIZE && lz < CHUNK_SIZE && chunk.get(lx, ly, lz).is_air() {
            chunk.set_material(lx, ly, lz, material);
        }
    }
}

// ---------------------------------------------------------------------------
// Hotbar
// ---------------------------------------------------------------------------

/// Number of hotbar slots.
const HOTBAR_SLOTS: usize = 9;

/// Default materials for the hotbar (IDs from material .ron files).
const DEFAULT_HOTBAR: [u16; HOTBAR_SLOTS] = [
    1,  // Stone
    2,  // Dirt
    7,  // Grass
    6,  // Sand
    5,  // Wood
    3,  // Water
    8,  // Ice
    10, // Lava
    12, // Glass
];

/// Player hotbar: 9 material slots with a selected index.
#[derive(Resource)]
pub struct Hotbar {
    pub slots: [MaterialId; HOTBAR_SLOTS],
    pub selected: usize,
}

impl Default for Hotbar {
    fn default() -> Self {
        let mut slots = [MaterialId::STONE; HOTBAR_SLOTS];
        for (i, &id) in DEFAULT_HOTBAR.iter().enumerate() {
            slots[i] = MaterialId(id);
        }
        Self { slots, selected: 0 }
    }
}

impl Hotbar {
    pub fn selected_material(&self) -> MaterialId {
        self.slots[self.selected]
    }
}

fn hotbar_input(
    key: Res<ButtonInput<KeyCode>>,
    scroll: Res<AccumulatedMouseScroll>,
    mut hotbar: ResMut<Hotbar>,
) {
    // Number keys 1-9
    let keys = [
        KeyCode::Digit1,
        KeyCode::Digit2,
        KeyCode::Digit3,
        KeyCode::Digit4,
        KeyCode::Digit5,
        KeyCode::Digit6,
        KeyCode::Digit7,
        KeyCode::Digit8,
        KeyCode::Digit9,
    ];
    for (i, &k) in keys.iter().enumerate() {
        if key.just_pressed(k) {
            hotbar.selected = i;
            return;
        }
    }

    // Scroll wheel
    if scroll.delta.y > 0.0 {
        hotbar.selected = (hotbar.selected + HOTBAR_SLOTS - 1) % HOTBAR_SLOTS;
    } else if scroll.delta.y < 0.0 {
        hotbar.selected = (hotbar.selected + 1) % HOTBAR_SLOTS;
    }
}

// ---------------------------------------------------------------------------
// Crosshair UI
// ---------------------------------------------------------------------------

#[derive(Component)]
struct Crosshair;

fn spawn_crosshair(mut commands: Commands) {
    commands.spawn((
        Crosshair,
        Text::new("+"),
        TextFont {
            font_size: 24.0,
            ..default()
        },
        TextColor(Color::srgba(1.0, 1.0, 1.0, 0.8)),
        Node {
            position_type: PositionType::Absolute,
            left: Val::Percent(50.0),
            top: Val::Percent(50.0),
            margin: UiRect {
                left: Val::Px(-6.0),
                top: Val::Px(-12.0),
                ..default()
            },
            ..default()
        },
    ));
}

fn despawn_crosshair(mut commands: Commands, q: Query<Entity, With<Crosshair>>) {
    for entity in &q {
        commands.entity(entity).despawn();
    }
}

// ---------------------------------------------------------------------------
// Block highlight
// ---------------------------------------------------------------------------

#[derive(Component)]
struct BlockHighlight;

fn update_block_highlight(
    mut commands: Commands,
    target: Res<BlockTarget>,
    existing: Query<Entity, With<BlockHighlight>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Remove old highlight
    for entity in &existing {
        commands.entity(entity).despawn();
    }

    let Some(hit) = &target.hit else {
        return;
    };

    // Spawn a slightly oversized wireframe-style cube at the hit position
    let pos = Vec3::new(
        hit.voxel_x as f32 + 0.5,
        hit.voxel_y as f32 + 0.5,
        hit.voxel_z as f32 + 0.5,
    );

    commands.spawn((
        BlockHighlight,
        Mesh3d(meshes.add(Cuboid::new(1.01, 1.01, 1.01))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgba(1.0, 1.0, 1.0, 0.15),
            alpha_mode: AlphaMode::Blend,
            unlit: true,
            ..default()
        })),
        Transform::from_translation(pos),
    ));
}

// ---------------------------------------------------------------------------
// Hotbar UI
// ---------------------------------------------------------------------------

#[derive(Component)]
struct HotbarRoot;

#[derive(Component)]
struct HotbarSlotUi(usize);

fn spawn_hotbar_ui(
    mut commands: Commands,
    hotbar: Res<Hotbar>,
    registry: Option<Res<MaterialRegistry>>,
) {
    commands
        .spawn((
            HotbarRoot,
            Node {
                position_type: PositionType::Absolute,
                bottom: Val::Px(16.0),
                left: Val::Percent(50.0),
                margin: UiRect {
                    left: Val::Px(-((HOTBAR_SLOTS as f32 * 48.0) / 2.0)),
                    ..default()
                },
                flex_direction: FlexDirection::Row,
                column_gap: Val::Px(4.0),
                ..default()
            },
        ))
        .with_children(|parent| {
            for i in 0..HOTBAR_SLOTS {
                let mat_id = hotbar.slots[i];
                let label = material_name(mat_id, registry.as_deref());
                let is_selected = i == hotbar.selected;

                parent
                    .spawn((
                        HotbarSlotUi(i),
                        Node {
                            width: Val::Px(44.0),
                            height: Val::Px(44.0),
                            justify_content: JustifyContent::Center,
                            align_items: AlignItems::Center,
                            ..default()
                        },
                        BackgroundColor(if is_selected {
                            Color::srgba(0.4, 0.4, 0.5, 0.8)
                        } else {
                            Color::srgba(0.15, 0.15, 0.2, 0.7)
                        }),
                    ))
                    .with_children(|slot| {
                        slot.spawn((
                            Text::new(label),
                            TextFont {
                                font_size: 10.0,
                                ..default()
                            },
                            TextColor(Color::WHITE),
                        ));
                    });
            }
        });
}

fn despawn_hotbar_ui(mut commands: Commands, q: Query<Entity, With<HotbarRoot>>) {
    for entity in &q {
        commands.entity(entity).despawn();
    }
}

fn update_hotbar_ui_system(
    hotbar: Res<Hotbar>,
    mut slot_q: Query<(&HotbarSlotUi, &mut BackgroundColor)>,
) {
    if !hotbar.is_changed() {
        return;
    }
    for (slot, mut bg) in &mut slot_q {
        *bg = if slot.0 == hotbar.selected {
            BackgroundColor(Color::srgba(0.4, 0.4, 0.5, 0.8))
        } else {
            BackgroundColor(Color::srgba(0.15, 0.15, 0.2, 0.7))
        };
    }
}

fn material_name(id: MaterialId, registry: Option<&MaterialRegistry>) -> String {
    if let Some(reg) = registry
        && let Some(data) = reg.get(id)
    {
        return data.name.chars().take(5).collect();
    }
    // Fallback names
    match id.0 {
        1 => "Stone".into(),
        2 => "Dirt".into(),
        3 => "Water".into(),
        5 => "Wood".into(),
        6 => "Sand".into(),
        7 => "Grass".into(),
        8 => "Ice".into(),
        10 => "Lava".into(),
        12 => "Glass".into(),
        _ => format!("#{}", id.0),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hotbar_default_has_9_slots() {
        let hb = Hotbar::default();
        assert_eq!(hb.slots.len(), HOTBAR_SLOTS);
        assert_eq!(hb.selected, 0);
        assert_eq!(hb.selected_material(), MaterialId::STONE);
    }

    #[test]
    fn hotbar_selection_wraps() {
        let mut hb = Hotbar::default();
        hb.selected = HOTBAR_SLOTS - 1;
        hb.selected = (hb.selected + 1) % HOTBAR_SLOTS;
        assert_eq!(hb.selected, 0);
    }

    #[test]
    fn hotbar_scroll_backward_wraps() {
        let mut hb = Hotbar::default();
        hb.selected = 0;
        hb.selected = (hb.selected + HOTBAR_SLOTS - 1) % HOTBAR_SLOTS;
        assert_eq!(hb.selected, HOTBAR_SLOTS - 1);
    }

    #[test]
    fn world_hit_normal_faces_outward() {
        let hit = WorldHit {
            voxel_x: 5,
            voxel_y: 10,
            voxel_z: 3,
            normal: IVec3::new(0, 1, 0),
            distance: 2.5,
        };
        // Adjacent placement position
        let px = hit.voxel_x + hit.normal.x;
        let py = hit.voxel_y + hit.normal.y;
        let pz = hit.voxel_z + hit.normal.z;
        assert_eq!((px, py, pz), (5, 11, 3));
    }

    #[test]
    fn reach_distance_realistic() {
        // Minecraft uses ~4.5 blocks. We use 6 for slightly more reach.
        const { assert!(REACH >= 4.0 && REACH <= 10.0) };
    }

    #[test]
    fn break_time_positive() {
        const { assert!(BREAK_TIME > 0.0) };
    }

    #[test]
    fn material_name_fallback() {
        assert_eq!(material_name(MaterialId::STONE, None), "Stone");
        assert_eq!(material_name(MaterialId::WATER, None), "Water");
        assert_eq!(material_name(MaterialId(99), None), "#99");
    }

    #[test]
    fn default_hotbar_materials_not_air() {
        let hb = Hotbar::default();
        for slot in &hb.slots {
            assert!(!slot.is_air(), "Hotbar should not contain air");
        }
    }
}
