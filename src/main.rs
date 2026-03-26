use bevy::prelude::*;
use clap::Parser;
use the_dark_candle::{
    behavior::BehaviorPlugin,
    biology::BiologyPlugin,
    camera::CameraPlugin,
    chemistry::ChemistryPlugin,
    data::DataPlugin,
    diagnostics::DiagnosticsPlugin,
    entities::EntityPlugin,
    game_state::GameStatePlugin,
    lighting::LightingPlugin,
    persistence::PersistencePlugin,
    physics::PhysicsPlugin,
    procgen::ProcgenPlugin,
    social::SocialPlugin,
    weather::WeatherPlugin,
    world::{WorldPlugin, scene_presets::ScenePreset},
};

/// The Dark Candle — a data-driven voxel game with real-world SI physics.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Cli {
    /// Load a named scene preset (e.g. "valley_river").
    /// Available presets: valley_river
    #[arg(long)]
    scene: Option<String>,
}

fn main() {
    let cli = Cli::parse();

    let mut app = App::new();

    // If a scene preset is requested, insert its PlanetConfig before plugins run.
    if let Some(name) = &cli.scene {
        match ScenePreset::from_name(name) {
            Some(preset) => {
                info!("Loading scene preset: {name}");
                app.insert_resource(preset.planet_config());
            }
            None => {
                eprintln!(
                    "Unknown scene preset: '{name}'. Available: {}",
                    ScenePreset::available_names().join(", ")
                );
                std::process::exit(1);
            }
        }
    }

    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "The Dark Candle".into(),
            resolution: (800, 600).into(),
            ..default()
        }),
        ..default()
    }))
    .add_plugins((
        DataPlugin,
        GameStatePlugin,
        CameraPlugin,
        LightingPlugin,
        DiagnosticsPlugin,
        EntityPlugin,
        WorldPlugin,
        PhysicsPlugin,
        ChemistryPlugin,
        BiologyPlugin,
        ProcgenPlugin,
        BehaviorPlugin,
        SocialPlugin,
        PersistencePlugin,
        WeatherPlugin,
    ))
    .run();
}
