// Audio foundation: ambient sounds, footsteps, block interaction, UI sounds.
//
// Sound files are loaded from `assets/audio/`. Missing files produce a Bevy
// asset warning but don't crash — the game is fully playable without audio.

use bevy::audio::{AudioPlayer, PlaybackSettings, Volume};
use bevy::prelude::*;

use crate::camera::FpsCamera;
use crate::game_state::GameState;
use crate::hud::Player;
use crate::interaction::BlockTarget;
use crate::world::chunk::{CHUNK_SIZE, Chunk, ChunkCoord};
use crate::world::voxel::MaterialId;

pub struct AudioPlugin;

impl Plugin for AudioPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GameAudio>()
            .init_resource::<FootstepTimer>()
            .add_systems(Startup, load_audio_assets)
            .add_systems(OnEnter(GameState::Playing), start_ambient)
            .add_systems(OnExit(GameState::Playing), stop_ambient)
            .add_systems(
                Update,
                (footstep_system, block_sound_system).run_if(in_state(GameState::Playing)),
            );
    }
}

// ---------------------------------------------------------------------------
// Audio asset handles
// ---------------------------------------------------------------------------

/// Holds all pre-loaded audio handles.
#[derive(Resource, Default)]
pub struct GameAudio {
    pub ambient_wind: Option<Handle<AudioSource>>,
    pub footstep_stone: Option<Handle<AudioSource>>,
    pub footstep_dirt: Option<Handle<AudioSource>>,
    pub footstep_wood: Option<Handle<AudioSource>>,
    pub footstep_sand: Option<Handle<AudioSource>>,
    pub footstep_water: Option<Handle<AudioSource>>,
    pub block_break: Option<Handle<AudioSource>>,
    pub block_place: Option<Handle<AudioSource>>,
    pub ui_click: Option<Handle<AudioSource>>,
}

fn load_audio_assets(asset_server: Res<AssetServer>, mut audio: ResMut<GameAudio>) {
    let load =
        |path: &str| -> Option<Handle<AudioSource>> { Some(asset_server.load(path.to_string())) };

    audio.ambient_wind = load("audio/ambient_wind.ogg");
    audio.footstep_stone = load("audio/footstep_stone.ogg");
    audio.footstep_dirt = load("audio/footstep_dirt.ogg");
    audio.footstep_wood = load("audio/footstep_wood.ogg");
    audio.footstep_sand = load("audio/footstep_sand.ogg");
    audio.footstep_water = load("audio/footstep_water.ogg");
    audio.block_break = load("audio/block_break.ogg");
    audio.block_place = load("audio/block_place.ogg");
    audio.ui_click = load("audio/ui_click.ogg");
}

// ---------------------------------------------------------------------------
// Ambient
// ---------------------------------------------------------------------------

#[derive(Component)]
struct AmbientSound;

fn start_ambient(mut commands: Commands, audio: Res<GameAudio>) {
    if let Some(handle) = &audio.ambient_wind {
        commands.spawn((
            AmbientSound,
            AudioPlayer(handle.clone()),
            PlaybackSettings {
                mode: bevy::audio::PlaybackMode::Loop,
                volume: Volume::Linear(0.3),
                ..default()
            },
        ));
    }
}

fn stop_ambient(mut commands: Commands, q: Query<Entity, With<AmbientSound>>) {
    for entity in &q {
        commands.entity(entity).despawn();
    }
}

// ---------------------------------------------------------------------------
// Footsteps
// ---------------------------------------------------------------------------

/// Time between footstep sounds when walking.
const FOOTSTEP_INTERVAL: f32 = 0.45;

/// Time between footstep sounds when sprinting.
const FOOTSTEP_SPRINT_INTERVAL: f32 = 0.3;

#[derive(Resource)]
struct FootstepTimer {
    elapsed: f32,
    was_moving: bool,
}

impl Default for FootstepTimer {
    fn default() -> Self {
        Self {
            elapsed: 0.0,
            was_moving: false,
        }
    }
}

fn footstep_system(
    mut commands: Commands,
    time: Res<Time>,
    audio: Res<GameAudio>,
    key: Res<ButtonInput<KeyCode>>,
    cam_q: Query<(&FpsCamera, &Transform), With<Player>>,
    chunks: Query<&Chunk>,
    mut timer: ResMut<FootstepTimer>,
) {
    let Ok((cam, transform)) = cam_q.single() else {
        return;
    };

    // Only play footsteps when grounded and moving
    let moving = cam.grounded
        && cam.gravity_enabled
        && (key.pressed(KeyCode::KeyW)
            || key.pressed(KeyCode::KeyA)
            || key.pressed(KeyCode::KeyS)
            || key.pressed(KeyCode::KeyD));

    if !moving {
        timer.was_moving = false;
        timer.elapsed = 0.0;
        return;
    }

    let interval = if key.pressed(KeyCode::ControlLeft) {
        FOOTSTEP_SPRINT_INTERVAL
    } else {
        FOOTSTEP_INTERVAL
    };

    // Play immediately on start of movement, then on interval
    if !timer.was_moving {
        timer.was_moving = true;
        timer.elapsed = interval; // trigger immediately
    }

    timer.elapsed += time.delta_secs();

    if timer.elapsed >= interval {
        timer.elapsed -= interval;

        // Determine ground material for sound selection
        let handle = ground_footstep_sound(transform.translation, &audio, &chunks);

        if let Some(h) = handle {
            commands.spawn((
                AudioPlayer(h),
                PlaybackSettings {
                    mode: bevy::audio::PlaybackMode::Despawn,
                    volume: Volume::Linear(0.5),
                    ..default()
                },
            ));
        }
    }
}

/// Pick the footstep sound based on the material under the player's feet.
fn ground_footstep_sound(
    pos: Vec3,
    audio: &GameAudio,
    chunks: &Query<&Chunk>,
) -> Option<Handle<AudioSource>> {
    let feet_y = (pos.y - 1.7 - 0.1).floor() as i32; // just below feet
    let vx = pos.x.floor() as i32;
    let vz = pos.z.floor() as i32;

    let cc = ChunkCoord::from_voxel_pos(vx, feet_y, vz);

    let chunk = chunks.iter().find(|c| c.coord == cc)?;

    let origin = cc.world_origin();
    let lx = (vx - origin.x) as usize;
    let ly = (feet_y - origin.y) as usize;
    let lz = (vz - origin.z) as usize;

    if lx >= CHUNK_SIZE || ly >= CHUNK_SIZE || lz >= CHUNK_SIZE {
        return audio.footstep_stone.clone();
    }

    let mat = chunk.get(lx, ly, lz).material;
    material_footstep_sound(mat, audio)
}

fn material_footstep_sound(mat: MaterialId, audio: &GameAudio) -> Option<Handle<AudioSource>> {
    match mat.0 {
        1 | 4 | 8 | 12 => audio.footstep_stone.clone(), // stone, iron, ice, glass
        2 | 7 => audio.footstep_dirt.clone(),           // dirt, grass
        5 => audio.footstep_wood.clone(),               // wood
        6 | 11 => audio.footstep_sand.clone(),          // sand, ash
        3 | 10 => audio.footstep_water.clone(),         // water, lava
        _ => audio.footstep_stone.clone(),
    }
}

// ---------------------------------------------------------------------------
// Block interaction sounds
// ---------------------------------------------------------------------------

fn block_sound_system(
    mut commands: Commands,
    audio: Res<GameAudio>,
    mouse: Res<ButtonInput<MouseButton>>,
    target: Res<BlockTarget>,
) {
    if target.hit.is_none() {
        return;
    }

    // Place sound on right-click
    if mouse.just_pressed(MouseButton::Right)
        && let Some(h) = &audio.block_place
    {
        commands.spawn((
            AudioPlayer(h.clone()),
            PlaybackSettings {
                mode: bevy::audio::PlaybackMode::Despawn,
                volume: Volume::Linear(0.6),
                ..default()
            },
        ));
    }

    // Break sound on left-click release
    if mouse.just_released(MouseButton::Left)
        && let Some(h) = &audio.block_break
    {
        commands.spawn((
            AudioPlayer(h.clone()),
            PlaybackSettings {
                mode: bevy::audio::PlaybackMode::Despawn,
                volume: Volume::Linear(0.6),
                ..default()
            },
        ));
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn footstep_interval_positive() {
        const { assert!(FOOTSTEP_INTERVAL > 0.0) };
        const { assert!(FOOTSTEP_SPRINT_INTERVAL > 0.0) };
    }

    #[test]
    fn sprint_faster_than_walk() {
        const { assert!(FOOTSTEP_SPRINT_INTERVAL < FOOTSTEP_INTERVAL) };
    }

    #[test]
    fn material_sound_mapping() {
        let audio = GameAudio::default();
        // Without loaded handles, all return None
        assert!(material_footstep_sound(MaterialId::STONE, &audio).is_none());
        assert!(material_footstep_sound(MaterialId::WATER, &audio).is_none());
        assert!(material_footstep_sound(MaterialId::DIRT, &audio).is_none());
    }

    #[test]
    fn game_audio_default_empty() {
        let audio = GameAudio::default();
        assert!(audio.ambient_wind.is_none());
        assert!(audio.block_break.is_none());
        assert!(audio.ui_click.is_none());
    }
}
