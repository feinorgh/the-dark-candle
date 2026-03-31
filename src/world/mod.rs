pub mod chunk;
pub mod chunk_manager;
pub mod collision;
pub mod erosion;
pub mod interpolation;
pub mod lod;
pub mod meshing;
pub mod octree;
pub mod planet;
pub mod planetary_sampler;
pub mod raycast;
pub mod refinement;
pub mod scene_presets;
pub mod terrain;
pub mod voxel;
pub mod voxel_access;

use bevy::prelude::*;
use bevy_common_assets::ron::RonAssetPlugin;
use std::sync::Arc;

use chunk_manager::ChunkManagerPlugin;
use meshing::MeshingPlugin;
use planet::PlanetConfig;
use planetary_sampler::PlanetaryTerrainSampler;
use refinement::RefinementPlugin;

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

impl Plugin for WorldPlugin {
    fn build(&self, app: &mut App) {
        app.configure_sets(Update, WorldSet::Meshing.after(WorldSet::ChunkManagement))
            .add_plugins(RonAssetPlugin::<PlanetConfig>::new(&["planet_config.ron"]));

        // Only insert default PlanetConfig if one was not already provided
        // (e.g. by a scene preset inserted in main before adding plugins).
        if !app.world().contains_resource::<PlanetConfig>() {
            app.insert_resource(PlanetConfig::default());
        }

        app.insert_resource(lod::LodConfig::default())
            .insert_resource(lod::MaterialColorMap::from_defaults())
            .add_plugins(ChunkManagerPlugin)
            .add_plugins(MeshingPlugin)
            .add_plugins(RefinementPlugin)
            .add_systems(Update, sync_color_map_from_registry)
            .add_systems(
                PostStartup,
                rebuild_terrain_gen_if_planetary.in_set(WorldSet::ChunkManagement),
            );
    }
}

/// If `PlanetaryData` is present at startup, rebuild `SharedTerrainGen` with a
/// `PlanetaryTerrainSampler` that uses the real tectonic/biome data.
///
/// This runs once in `PostStartup` after all plugins have been built, so that
/// a `PlanetaryData` resource inserted in `main()` is picked up automatically.
fn rebuild_terrain_gen_if_planetary(
    planetary: Option<Res<PlanetaryData>>,
    planet_config: Res<PlanetConfig>,
    mut shared_gen: ResMut<chunk_manager::SharedTerrainGen>,
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

    let sampler = PlanetaryTerrainSampler::new(planetary.0.clone(), planet_config.clone());
    *shared_gen = chunk_manager::SharedTerrainGen(Arc::new(
        terrain::UnifiedTerrainGenerator::Planetary(sampler),
    ));
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
