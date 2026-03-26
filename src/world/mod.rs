pub mod chunk;
pub mod chunk_manager;
pub mod collision;
pub mod erosion;
pub mod interpolation;
pub mod lod;
pub mod meshing;
pub mod octree;
pub mod planet;
pub mod raycast;
pub mod refinement;
pub mod scene_presets;
pub mod terrain;
pub mod voxel;
pub mod voxel_access;

use bevy::prelude::*;
use bevy_common_assets::ron::RonAssetPlugin;

use chunk_manager::ChunkManagerPlugin;
use meshing::MeshingPlugin;
use planet::PlanetConfig;
use refinement::RefinementPlugin;

/// System set ordering for the world pipeline.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum WorldSet {
    /// Chunk loading/unloading.
    ChunkManagement,
    /// Mesh generation from voxel data.
    Meshing,
}

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
            .add_systems(Update, sync_color_map_from_registry);
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
