use bevy::ecs::error;
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
    /// Load a named scene preset (e.g. "valley_river", "spherical_planet").
    /// Available presets: valley_river, spherical_planet
    #[arg(long)]
    scene: Option<String>,

    /// Run the full planetary generation pipeline and play on a tectonic world.
    ///
    /// Equivalent to `--scene spherical_planet` with a generated PlanetaryData
    /// resource driving terrain instead of Perlin noise.
    ///
    /// Generation at the default level-5 grid takes < 1 second.
    #[arg(long)]
    planet: bool,

    /// Geodesic grid level used when `--planet` is set (default: 5 = 10242 cells).
    ///
    /// Higher levels give more detailed planetary features but take longer to
    /// generate: level 5 ~100ms, level 6 ~400ms, level 7 ~1.5s.
    #[arg(long, default_value_t = 5)]
    planet_level: u32,

    /// Random seed for planetary generation (used with `--planet`).
    #[arg(long, default_value_t = 42)]
    planet_seed: u64,
}

fn main() {
    let cli = Cli::parse();

    let mut app = App::new();

    // --planet shortcut: run the full planetary pipeline and insert PlanetaryData.
    let use_planet = cli.planet
        || cli.scene.as_deref().is_some_and(|s| {
            s.eq_ignore_ascii_case("spherical_planet")
                || s.eq_ignore_ascii_case("spherical-planet")
                || s.eq_ignore_ascii_case("planet")
                || s.eq_ignore_ascii_case("spherical")
        });

    if use_planet {
        use std::sync::Arc;
        use the_dark_candle::planet::{
            PlanetConfig as PlanetGenConfig, PlanetData, biomes::run_biomes, geology::run_geology,
            tectonics::run_tectonics,
        };
        use the_dark_candle::world::PlanetaryData;

        info!(
            "Generating planet: seed={}, level={}…",
            cli.planet_seed, cli.planet_level
        );
        let t0 = std::time::Instant::now();

        let gen_config = PlanetGenConfig {
            seed: cli.planet_seed,
            grid_level: cli.planet_level,
            ..Default::default()
        };
        let mut data = PlanetData::new(gen_config);
        run_tectonics(&mut data, |_| {});
        run_biomes(&mut data);
        run_geology(&mut data);

        info!(
            "Planet generated in {:.1}s ({} cells)",
            t0.elapsed().as_secs_f32(),
            data.grid.cell_count(),
        );

        // Insert both the PlanetaryData resource AND the spherical PlanetConfig.
        let preset = ScenePreset::SphericalPlanet;
        app.insert_resource(preset.planet_config());
        app.insert_resource(PlanetaryData(Arc::new(data)));
    } else if let Some(name) = &cli.scene {
        // Non-planet scene preset
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

    // Log stale-entity command errors instead of panicking.  This is a
    // safety net for rare frame-boundary races between chunk despawn and
    // systems that hold deferred commands on those entities.
    app.set_error_handler(error::error);

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
        the_dark_candle::hud::HudPlugin,
        the_dark_candle::interaction::InteractionPlugin,
        LightingPlugin,
        DiagnosticsPlugin,
        EntityPlugin,
        WorldPlugin,
        PhysicsPlugin,
        ChemistryPlugin,
    ))
    .add_plugins((
        BiologyPlugin,
        ProcgenPlugin,
        BehaviorPlugin,
        SocialPlugin,
        PersistencePlugin,
        WeatherPlugin,
        the_dark_candle::audio::AudioPlugin,
    ))
    .run();
}
