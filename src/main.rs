use bevy::prelude::*;
use bevy_common_assets::ron::RonAssetPlugin;
use serde::Deserialize;

// -----------------------------------------------------------------------------
// 1. DATA DEFINITIONS (This is what the AI will map the YAML/RON files to)
// -----------------------------------------------------------------------------
#[derive(Deserialize, Asset, TypePath, Debug, Clone)]
pub struct EnemyData {
    pub name: String,
    pub health: f32,
    pub speed: f32,
}

// We map our raw Data into ECS Components for runtime gameplay
#[derive(Component)]
pub struct Enemy {
    pub speed: f32,
}

// A global resource to hold the memory address (Handle) to our data file
#[derive(Resource)]
pub struct GameAssets {
    pub goblin_data: Handle<EnemyData>,
}

// -----------------------------------------------------------------------------
// 2. ENGINE INITIALIZATION
// -----------------------------------------------------------------------------
fn main() {
    App::new()
        // Initialize Bevy (Handles Wayland/X11 windowing, audio, inputs)
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Gentoo Bevy AI Project".into(),
                resolution: (800, 600).into(),
                ..default()
            }),
            ..default()
        }))
        // Register our Data-Driven RON plugin. 
        // Bevy will now listen for any file ending in .enemy.ron
        .add_plugins(RonAssetPlugin::<EnemyData>::new(&["enemy.ron"]))
        // Register Startup & Update Systems
        .add_systems(Startup, setup_scene)
        .add_systems(Update, spawn_enemy_when_loaded)
        .run();
}

// -----------------------------------------------------------------------------
// 3. SYSTEMS (Logic)
// -----------------------------------------------------------------------------
fn setup_scene(mut commands: Commands, asset_server: Res<AssetServer>) {
    // Setup a standard 2D Camera
    commands.spawn(Camera2d);

    // Tell the AssetServer to load our custom data file from the disk.
    // Paths are automatically relative to the `assets/` folder.
    let goblin_handle = asset_server.load("data/goblin.enemy.ron");

    // Store the handle globally so we can check on it later
    commands.insert_resource(GameAssets {
        goblin_data: goblin_handle,
    });
}

fn spawn_enemy_when_loaded(
    mut commands: Commands,
    game_assets: Res<GameAssets>,
    custom_assets: Res<Assets<EnemyData>>,
    mut has_spawned: Local<bool>, // Local state to ensure we only spawn it once
) {
    // If we already spawned the enemy, do nothing
    if *has_spawned {
        return;
    }

    // Check if the file has finished parsing from the disk into our Rust struct
    if let Some(goblin_data) = custom_assets.get(&game_assets.goblin_data) {
        println!("Data Loaded Successfully: {} has {} HP", goblin_data.name, goblin_data.health);

        // Spawn the entity, applying the parsed data to its components
        commands.spawn((
            Enemy { speed: goblin_data.speed, },
            Sprite {
                color: Color::srgb(0.8, 0.2, 0.2), // Enemy Red
                custom_size: Some(Vec2::new(50.0, 50.0)),
                ..default()
            },
            Transform::from_xyz(0.0, 0.0, 0.0),
        ));

        *has_spawned = true;
    }
}
