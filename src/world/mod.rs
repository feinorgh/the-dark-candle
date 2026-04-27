pub mod biome_map;
pub mod chunk;
pub mod chunk_manager;
pub mod collision;
pub mod erosion;
pub mod lod;
pub mod meshing;
pub mod noise;
pub mod octree;
pub mod planet;
pub mod planetary_sampler;
pub mod raycast;
pub mod scene_presets;
pub mod terrain;
pub mod v2;
pub mod voxel;
pub mod voxel_access;

use bevy::prelude::*;
use bevy_common_assets::ron::RonAssetPlugin;
use std::sync::Arc;

use planet::PlanetConfig;

/// Handle to the `planet_config.ron` asset so we can poll for load completion.
#[derive(Resource)]
struct PlanetConfigHandle(Handle<PlanetConfig>);

/// System set ordering for the world pipeline.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum WorldSet {
    /// Chunk loading/unloading.
    ChunkManagement,
    /// Mesh generation from voxel data.
    Meshing,
}

// ─── PlanetaryData resource ───────────────────────────────────────────────────

/// A fully-generated planet, ready for use as the terrain source.
///
/// Insert this resource **before** `WorldPlugin` builds to drive terrain
/// generation from the tectonic/biome simulation instead of Perlin noise.
///
/// ```rust,ignore
/// // In main() or a pre-plugin setup:
/// use the_dark_candle::planet::{PlanetConfig as PlanetGenConfig, PlanetData};
/// use the_dark_candle::planet::tectonics::run_tectonics;
/// use the_dark_candle::planet::biomes::run_biomes;
/// use the_dark_candle::planet::geology::run_geology;
///
/// let mut data = PlanetData::new(PlanetGenConfig::default());
/// run_tectonics(&mut data, |_| {});
/// run_biomes(&mut data);
/// run_geology(&mut data);
/// app.insert_resource(PlanetaryData(Arc::new(data)));
/// ```
#[derive(Resource, Clone)]
pub struct PlanetaryData(pub Arc<crate::planet::PlanetData>);

pub struct WorldPlugin;

/// Build a `UnifiedTerrainGenerator` from optional `PlanetaryData` and a `PlanetConfig`.
///
/// When `planetary` is `None` a minimal placeholder `PlanetData` (level-3 grid,
/// 642 cells) is constructed.  This is fast and allows the terrain system to
/// start immediately; `rebuild_terrain_gen_if_planetary` will upgrade to the
/// real dataset once `PlanetaryData` is available.
fn make_terrain_gen(
    planetary: Option<&PlanetaryData>,
    planet_config: &PlanetConfig,
) -> terrain::UnifiedTerrainGenerator {
    let planet_data = if let Some(pd) = planetary {
        pd.0.clone()
    } else {
        let gen_cfg = crate::planet::PlanetConfig {
            seed: planet_config.seed as u64,
            grid_level: 3,
            ..Default::default()
        };
        Arc::new(crate::planet::PlanetData::new(gen_cfg))
    };
    terrain::UnifiedTerrainGenerator::new(planet_data, planet_config.clone())
}

impl Plugin for WorldPlugin {
    fn build(&self, app: &mut App) {
        app.configure_sets(Update, WorldSet::Meshing.after(WorldSet::ChunkManagement))
            .add_plugins(RonAssetPlugin::<PlanetConfig>::new(&["planet_config.ron"]));

        // Only insert default PlanetConfig if one was not already provided
        // (e.g. by a scene preset inserted in main before adding plugins).
        if !app.world().contains_resource::<PlanetConfig>() {
            app.insert_resource(PlanetConfig::default());
        }

        // Kick off async load of planet_config.ron so the RON values can
        // replace the default PlanetConfig resource once the asset is ready.
        let handle = app
            .world_mut()
            .resource::<AssetServer>()
            .load::<PlanetConfig>("data/planet_config.ron");
        app.insert_resource(PlanetConfigHandle(handle));

        app.insert_resource(lod::LodConfig::default())
            .insert_resource(lod::MaterialColorMap::from_defaults());

        let planet = app
            .world()
            .get_resource::<PlanetConfig>()
            .cloned()
            .unwrap_or_default();
        let planetary = app.world().get_resource::<PlanetaryData>().cloned();
        let generator = make_terrain_gen(planetary.as_ref(), &planet);
        let shared_generator = make_terrain_gen(planetary.as_ref(), &planet);
        app.insert_resource(chunk_manager::TerrainGeneratorRes(generator))
            .insert_resource(chunk_manager::SharedTerrainGen(Arc::new(shared_generator)))
            .add_plugins(v2::chunk_manager::V2WorldPlugin);

        app.add_systems(Update, sync_color_map_from_registry)
            .add_systems(Update, apply_planet_config_from_asset)
            .add_systems(
                Update,
                rebuild_terrain_on_config_change.after(apply_planet_config_from_asset),
            )
            .add_systems(
                PostStartup,
                rebuild_terrain_gen_if_planetary.in_set(WorldSet::ChunkManagement),
            );
    }
}

/// Once the `planet_config.ron` asset finishes loading, replace the default
/// `PlanetConfig` resource with the loaded values and rebuild terrain generators.
///
/// Uses a `Local<bool>` guard so the work is done exactly once.
#[allow(clippy::too_many_arguments)]
fn apply_planet_config_from_asset(
    handle: Option<Res<PlanetConfigHandle>>,
    assets: Res<Assets<PlanetConfig>>,
    mut planet_config: ResMut<PlanetConfig>,
    mut shared_gen: ResMut<chunk_manager::SharedTerrainGen>,
    mut terrain_res: ResMut<chunk_manager::TerrainGeneratorRes>,
    v2_gen: Option<ResMut<v2::chunk_manager::V2TerrainGen>>,
    skip: Option<Res<crate::game_state::SkipWorldCreation>>,
    planetary: Option<Res<PlanetaryData>>,
    mut done: Local<bool>,
) {
    if *done {
        return;
    }
    let Some(handle) = handle else { return };
    let Some(loaded) = assets.get(&handle.0) else {
        return;
    };

    // When the CLI already configured a scene/preset (SkipWorldCreation),
    // the PlanetConfig was set explicitly.  Don't overwrite it with the
    // generic RON fallback — just mark done and keep the existing config.
    if skip.is_some() {
        info!(
            "Skipping planet_config.ron override — CLI/preset config active (seed={})",
            planet_config.seed,
        );
        *done = true;
        return;
    }

    info!(
        "Loaded planet config from RON: mode={:?}, radius={}, seed={}",
        loaded.mode, loaded.mean_radius, loaded.seed,
    );
    *planet_config = loaded.clone();

    // Rebuild both terrain generator resources so chunk generation and
    // camera/physics queries all use the new config.
    terrain_res.0 = make_terrain_gen(planetary.as_deref(), &planet_config);

    let shared_arc = Arc::new(make_terrain_gen(planetary.as_deref(), &planet_config));
    *shared_gen = chunk_manager::SharedTerrainGen(shared_arc.clone());

    // Rebuild the V2 terrain generator if the V2 pipeline is active.
    if let Some(mut v2) = v2_gen {
        v2.0 = shared_arc;
        info!("V2 pipeline: rebuilt terrain generator from updated PlanetConfig");
    }

    *done = true;
}

/// Rebuilds terrain generators whenever `PlanetConfig` is changed outside of
/// `apply_planet_config_from_asset` (e.g. by the world creation UI selecting a
/// preset, or by hot-reloading config).
fn rebuild_terrain_on_config_change(
    planet_config: Res<PlanetConfig>,
    mut shared_gen: ResMut<chunk_manager::SharedTerrainGen>,
    mut terrain_res: ResMut<chunk_manager::TerrainGeneratorRes>,
    v2_gen: Option<ResMut<v2::chunk_manager::V2TerrainGen>>,
    planetary: Option<Res<PlanetaryData>>,
) {
    if !planet_config.is_changed() || planet_config.is_added() {
        return;
    }

    info!(
        "PlanetConfig changed (seed={}) — rebuilding terrain generators",
        planet_config.seed,
    );

    terrain_res.0 = make_terrain_gen(planetary.as_deref(), &planet_config);

    let shared_arc = Arc::new(make_terrain_gen(planetary.as_deref(), &planet_config));
    *shared_gen = chunk_manager::SharedTerrainGen(shared_arc.clone());

    if let Some(mut v2) = v2_gen {
        v2.0 = shared_arc;
    }
}

/// Switch to a `PlanetaryTerrainSampler` backed by real tectonic/biome data.
///
/// Runs once in `PostStartup` after all plugins have been built, so that a
/// `PlanetaryData` resource inserted in `main()` is picked up automatically.
fn rebuild_terrain_gen_if_planetary(
    planetary: Option<Res<PlanetaryData>>,
    planet_config: Res<PlanetConfig>,
    mut shared_gen: ResMut<chunk_manager::SharedTerrainGen>,
    mut terrain_res: ResMut<chunk_manager::TerrainGeneratorRes>,
    v2_gen: Option<ResMut<v2::chunk_manager::V2TerrainGen>>,
) {
    let Some(planetary) = planetary else {
        return;
    };

    info!(
        "PlanetaryData detected — switching terrain generator to PlanetaryTerrainSampler \
         ({} cells, seed {})",
        planetary.0.grid.cell_count(),
        planetary.0.config.seed,
    );

    let unified = Arc::new(terrain::UnifiedTerrainGenerator::new(
        planetary.0.clone(),
        planet_config.clone(),
    ));
    // Build a second sampler for TerrainGeneratorRes so that surface queries
    // (spawn positioning, camera gravity, map) match the actual chunk terrain.
    terrain_res.0 =
        terrain::UnifiedTerrainGenerator::new(planetary.0.clone(), planet_config.clone());
    *shared_gen = chunk_manager::SharedTerrainGen(unified.clone());

    // Propagate the planetary generator to the V2 pipeline so V2 chunk
    // generation uses tectonic surface heights (fixes TERRAIN-007).
    if let Some(mut v2) = v2_gen {
        v2.0 = unified;
        info!("V2 pipeline: rebuilt terrain generator with PlanetaryTerrainSampler");
    }
}

/// Populates `MaterialColorMap` from `MaterialRegistry` once after startup.
fn sync_color_map_from_registry(
    registry: Option<Res<crate::data::MaterialRegistry>>,
    mut color_map: ResMut<lod::MaterialColorMap>,
    mut done: Local<bool>,
) {
    if *done {
        return;
    }
    if let Some(registry) = registry
        && !registry.is_empty()
    {
        color_map.populate_from_registry(&registry);
        info!(
            "MaterialColorMap populated from registry ({} materials)",
            registry.len()
        );
        *done = true;
    }
}
