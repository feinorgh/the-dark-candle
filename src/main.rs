use bevy::ecs::error;
use bevy::prelude::*;
use clap::Parser;
use the_dark_candle::{
    behavior::BehaviorPlugin,
    biology::BiologyPlugin,
    bodies::BodiesPlugin,
    building::BuildingPlugin,
    camera::CameraPlugin,
    chemistry::ChemistryPlugin,
    data::DataPlugin,
    diagnostics::DiagnosticsPlugin,
    entities::EntityPlugin,
    game_state::{GameStatePlugin, SkipWorldCreation},
    lighting::LightingPlugin,
    map::MapPlugin,
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
    /// Load a named scene preset.
    /// Available: valley_river, spherical_planet, alpine, archipelago,
    /// desert_canyon, rolling_plains, volcanic, tundra_fjords
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

    /// World generation seed.  Overrides the preset's default seed.
    #[arg(long)]
    seed: Option<u32>,

    /// Terrain detail level: 1=fast (3 octaves), 2=balanced (6), 3=rich (8).
    #[arg(long, value_parser = clap::value_parser!(u8).range(1..=3))]
    terrain_detail: Option<u8>,

    /// Override the terrain height scale (amplitude in voxels/meters).
    #[arg(long)]
    height_scale: Option<f64>,

    /// Cave density: off, sparse, normal, dense.
    #[arg(long)]
    caves: Option<String>,

    /// Erosion intensity: off, light, moderate, heavy.
    #[arg(long)]
    erosion: Option<String>,

    /// Hydraulic erosion: off, light, moderate, heavy.
    #[arg(long)]
    hydraulic_erosion: Option<String>,

    /// Disable all terrain noise, producing a perfectly smooth surface.
    /// Useful for debugging rendering, camera, and chunk logic without
    /// waiting for terrain generation.
    #[arg(long)]
    no_noise: bool,

    /// Planet radius in meters.  Sets both mean_radius and sea_level_radius.
    /// Default for spherical_planet preset: 32000 (32 km).
    /// Smaller values (e.g. 1000) produce fewer chunks and faster startup.
    #[arg(long)]
    radius: Option<f64>,

    /// Player spawn location on spherical planets.
    ///
    /// Modes:
    ///   random-land  — random point on land (above sea level)
    ///   coastline    — random land point near water
    ///   LAT,LON      — explicit coordinates in degrees (e.g. "30.5,-45.2")
    ///
    /// Omit to spawn at the default 45°N, 0°E.
    #[arg(long)]
    spawn: Option<String>,

    // ── Agent capture flags ──────────────────────────────────────────────────
    /// Enable agent capture mode: suppress cursor grab, auto-exit after capture.
    /// Typically combined with `--capture`, `--settle`, and `--capture-out`.
    #[arg(long)]
    agent: bool,

    /// Frames to wait before capturing (lets chunks load and mesh).
    /// Default: 120 frames.
    #[arg(long, default_value_t = 120)]
    settle: u32,

    /// Capture mode: `screenshot` (single frame) or `video` (N frames → MP4).
    #[arg(long, value_enum, default_value = "screenshot")]
    capture: CaptureModeCli,

    /// Number of video frames to record (only used with `--capture video`).
    #[arg(long, default_value_t = 120)]
    capture_frames: u32,

    /// Capture one screenshot every N game frames (1 = every frame, 90 = ~3s
    /// intervals at 30 fps). Lets you spread captures over a long run.
    #[arg(long, default_value_t = 1)]
    capture_interval: u32,

    /// Output directory for captures. Created if it does not exist.
    #[arg(long, default_value = "agent_captures")]
    capture_out: String,

    /// Video FPS for ffmpeg encoding (only used with `--capture video`).
    #[arg(long, default_value_t = 30)]
    capture_fps: u32,
}

/// CLI enum for capture mode selection.
#[derive(clap::ValueEnum, Debug, Clone)]
enum CaptureModeCli {
    Screenshot,
    Video,
}

fn main() {
    let cli = Cli::parse();

    let mut app = App::new();

    // ── Agent capture mode ────────────────────────────────────────────────────
    if cli.agent {
        use the_dark_candle::diagnostics::agent_capture::{
            AgentCaptureConfig, AgentCapturePlugin, CaptureMode,
        };
        let mode = match cli.capture {
            CaptureModeCli::Screenshot => CaptureMode::Screenshot,
            CaptureModeCli::Video => CaptureMode::Video,
        };
        app.insert_resource(AgentCaptureConfig {
            settle_frames: cli.settle,
            mode,
            capture_frames: cli.capture_frames,
            capture_interval: cli.capture_interval,
            output_dir: std::path::PathBuf::from(&cli.capture_out),
            fps: cli.capture_fps,
        });
        app.add_plugins(AgentCapturePlugin);
    }

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
        use the_dark_candle::planet::persistence::{load_planet, save_planet};
        use the_dark_candle::planet::{
            PlanetConfig as PlanetGenConfig, PlanetData, biomes::run_biomes, geology::run_geology,
            tectonics::run_tectonics,
        };
        use the_dark_candle::world::PlanetaryData;

        let preset = ScenePreset::SphericalPlanet;
        let planet_config = preset.planet_config();

        // Try to load a previously saved planet for this seed/level combination.
        let data = if let Some(loaded) = load_planet(cli.planet_seed) {
            info!(
                "Loaded persisted planet: seed={}, cells={}",
                cli.planet_seed,
                loaded.grid.cell_count(),
            );
            loaded
        } else {
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

            if let Err(e) = save_planet(&data) {
                warn!("Failed to save planet to disk: {e}");
            }

            data
        };

        // Insert both the PlanetaryData resource AND the spherical PlanetConfig.
        app.insert_resource(planet_config);
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

    // Apply CLI overrides to the PlanetConfig (if one was inserted above).
    apply_cli_overrides(&cli, &mut app);

    // If a scene was selected via CLI, skip the world creation screen
    // by starting directly in the Loading state.
    if app
        .world()
        .contains_resource::<the_dark_candle::world::planet::PlanetConfig>()
    {
        app.insert_resource(SkipWorldCreation);
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
        the_dark_candle::floating_origin::FloatingOriginPlugin,
        DataPlugin,
        GameStatePlugin,
        CameraPlugin,
        MapPlugin,
        the_dark_candle::hud::HudPlugin,
        the_dark_candle::interaction::InteractionPlugin,
        LightingPlugin,
        DiagnosticsPlugin,
        bevy::pbr::wireframe::WireframePlugin::default(),
        EntityPlugin,
        WorldPlugin,
        PhysicsPlugin,
        ChemistryPlugin,
    ))
    .add_plugins((
        BiologyPlugin,
        BuildingPlugin,
        ProcgenPlugin,
        BehaviorPlugin,
        BodiesPlugin,
        SocialPlugin,
        PersistencePlugin,
        WeatherPlugin,
        the_dark_candle::audio::AudioPlugin,
        the_dark_candle::sky::SkyPlugin,
    ));

    // Resolve --spawn location now that the terrain generator exists.
    resolve_spawn_location(&cli, &mut app);

    app.run();
}

/// Apply CLI flag overrides to an already-inserted `PlanetConfig`.
fn apply_cli_overrides(cli: &Cli, app: &mut App) {
    use the_dark_candle::world::{
        erosion::ErosionConfig, noise::NoiseConfig, planet::PlanetConfig,
    };

    // If no PlanetConfig resource exists yet, nothing to override.
    if !app.world().contains_resource::<PlanetConfig>() {
        // Check if any override flags were given without a preset.
        let has_overrides = cli.seed.is_some()
            || cli.terrain_detail.is_some()
            || cli.height_scale.is_some()
            || cli.caves.is_some()
            || cli.erosion.is_some()
            || cli.hydraulic_erosion.is_some()
            || cli.no_noise
            || cli.radius.is_some();
        if has_overrides {
            // Insert a default config so overrides can apply.
            app.insert_resource(PlanetConfig::default());
        } else {
            return;
        }
    }

    let mut config = app.world().resource::<PlanetConfig>().clone();

    if let Some(seed) = cli.seed {
        config.seed = seed;
    }

    if let Some(hs) = cli.height_scale {
        config.height_scale = hs;
    }

    if let Some(r) = cli.radius {
        config.mean_radius = r;
        config.sea_level_radius = r;
    }

    if cli.no_noise {
        config.noise = None;
        config.height_scale = 0.0;
        config.erosion = None;
        config.hydraulic_erosion = None;
        info!("Noise disabled: terrain will be a smooth surface at mean_radius");
    }

    // Terrain detail: adjust FBM/ridged octaves.
    if let Some(detail) = cli.terrain_detail {
        let noise = config.noise.get_or_insert_with(NoiseConfig::default);
        match detail {
            1 => {
                noise.fbm_octaves = 3;
                noise.ridged_octaves = 3;
            }
            2 => {
                noise.fbm_octaves = 6;
                noise.ridged_octaves = 5;
            }
            3 => {
                noise.fbm_octaves = 8;
                noise.ridged_octaves = 7;
            }
            _ => {}
        }
    }

    // Cave density: adjust cave_threshold.
    if let Some(ref caves) = cli.caves {
        match caves.to_lowercase().as_str() {
            "off" | "none" => config.cave_threshold = -999.0,
            "sparse" => config.cave_threshold = -0.45,
            "normal" => config.cave_threshold = -0.3,
            "dense" => config.cave_threshold = -0.2,
            other => {
                eprintln!("Unknown cave density: '{other}'. Options: off, sparse, normal, dense");
                std::process::exit(1);
            }
        }
    }

    // Erosion intensity.
    if let Some(ref erosion) = cli.erosion {
        match erosion.to_lowercase().as_str() {
            "off" | "none" => config.erosion = None,
            "light" => {
                config.erosion = Some(ErosionConfig {
                    enabled: true,
                    flow_threshold: 60.0,
                    depth_scale: 2.0,
                    max_channel_depth: 8.0,
                    ..Default::default()
                });
            }
            "moderate" => {
                config.erosion = Some(ErosionConfig {
                    enabled: true,
                    ..Default::default()
                });
            }
            "heavy" => {
                config.erosion = Some(ErosionConfig {
                    enabled: true,
                    flow_threshold: 20.0,
                    depth_scale: 5.0,
                    max_channel_depth: 20.0,
                    width_scale: 3.0,
                    ..Default::default()
                });
            }
            other => {
                eprintln!(
                    "Unknown erosion intensity: '{other}'. Options: off, light, moderate, heavy"
                );
                std::process::exit(1);
            }
        }
    }

    if let Some(ref hydraulic) = cli.hydraulic_erosion {
        use the_dark_candle::world::erosion::{HydraulicErosionConfig, HydraulicMode};
        match hydraulic.to_lowercase().as_str() {
            "off" | "none" => config.hydraulic_erosion = None,
            "light" => {
                config.hydraulic_erosion = Some(HydraulicErosionConfig {
                    enabled: true,
                    mode: HydraulicMode::Droplet,
                    droplet_iterations: 10_000,
                    ..Default::default()
                });
            }
            "moderate" => {
                config.hydraulic_erosion = Some(HydraulicErosionConfig {
                    enabled: true,
                    mode: HydraulicMode::Combined,
                    droplet_iterations: 30_000,
                    grid_iterations: 15,
                    ..Default::default()
                });
            }
            "heavy" => {
                config.hydraulic_erosion = Some(HydraulicErosionConfig {
                    enabled: true,
                    mode: HydraulicMode::Combined,
                    droplet_iterations: 80_000,
                    grid_iterations: 30,
                    ..Default::default()
                });
            }
            other => {
                eprintln!(
                    "Unknown hydraulic erosion intensity: '{other}'. Options: off, light, moderate, heavy"
                );
                std::process::exit(1);
            }
        }
    }

    app.insert_resource(config);
}

/// Resolve `--spawn` CLI argument into a `SpawnLocation` resource.
///
/// Must be called after all plugins have been built so that
/// `TerrainGeneratorRes` is available for land/coastline sampling.
fn resolve_spawn_location(cli: &Cli, app: &mut App) {
    use the_dark_candle::camera::{SpawnLocation, find_coastline, find_random_land};
    use the_dark_candle::world::chunk_manager::TerrainGeneratorRes;
    use the_dark_candle::world::planet::PlanetConfig;

    let Some(ref spawn_arg) = cli.spawn else {
        return;
    };

    let Some(planet) = app.world().get_resource::<PlanetConfig>() else {
        eprintln!("--spawn requires a spherical world (use --scene or --planet)");
        std::process::exit(1);
    };
    let sea_level = planet.sea_level_radius;
    let seed = planet.seed;

    let Some(terrain) = app.world().get_resource::<TerrainGeneratorRes>() else {
        eprintln!("--spawn: terrain generator not available yet");
        std::process::exit(1);
    };

    let location = match spawn_arg.to_lowercase().as_str() {
        "random-land" | "land" | "random" => {
            info!("Resolving spawn: random land…");
            match find_random_land(&terrain.0, sea_level, 2000, seed as u64) {
                Some((lat, lon)) => SpawnLocation { lat, lon },
                None => {
                    warn!("Could not find land after 2000 attempts, using default spawn");
                    return;
                }
            }
        }
        "coastline" | "coast" | "beach" => {
            info!("Resolving spawn: coastline…");
            match find_coastline(&terrain.0, sea_level, 5000, seed as u64) {
                Some((lat, lon)) => SpawnLocation { lat, lon },
                None => {
                    warn!("Could not find coastline after 5000 attempts, trying random land");
                    match find_random_land(&terrain.0, sea_level, 2000, seed as u64) {
                        Some((lat, lon)) => SpawnLocation { lat, lon },
                        None => {
                            warn!("No land found either, using default spawn");
                            return;
                        }
                    }
                }
            }
        }
        coords => {
            // Parse "LAT,LON" in degrees.
            let parts: Vec<&str> = coords.split(',').collect();
            if parts.len() != 2 {
                eprintln!(
                    "Invalid --spawn value: '{coords}'. \
                     Use: random-land, coastline, or LAT,LON (e.g. 30.5,-45.2)"
                );
                std::process::exit(1);
            }
            let lat_deg: f64 = parts[0].trim().parse().unwrap_or_else(|_| {
                eprintln!("Invalid latitude: '{}'", parts[0]);
                std::process::exit(1);
            });
            let lon_deg: f64 = parts[1].trim().parse().unwrap_or_else(|_| {
                eprintln!("Invalid longitude: '{}'", parts[1]);
                std::process::exit(1);
            });
            if !(-90.0..=90.0).contains(&lat_deg) || !(-180.0..=180.0).contains(&lon_deg) {
                eprintln!(
                    "Coordinates out of range: lat must be [-90, 90], lon must be [-180, 180]"
                );
                std::process::exit(1);
            }
            SpawnLocation {
                lat: lat_deg.to_radians(),
                lon: lon_deg.to_radians(),
            }
        }
    };

    info!(
        "Spawn location: {:.1}°{} {:.1}°{}",
        location.lat.to_degrees().abs(),
        if location.lat >= 0.0 { "N" } else { "S" },
        location.lon.to_degrees().abs(),
        if location.lon >= 0.0 { "E" } else { "W" },
    );
    app.insert_resource(location);
}
