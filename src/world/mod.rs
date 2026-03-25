pub mod chunk;
pub mod chunk_manager;
pub mod collision;
pub mod interpolation;
pub mod lod;
pub mod meshing;
pub mod octree;
pub mod planet;
pub mod raycast;
pub mod refinement;
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
            .add_plugins(RonAssetPlugin::<PlanetConfig>::new(&["planet_config.ron"]))
            .insert_resource(PlanetConfig::default())
            .insert_resource(lod::LodConfig::default())
            .insert_resource(lod::MaterialColorMap::from_defaults())
            .add_plugins(ChunkManagerPlugin)
            .add_plugins(MeshingPlugin)
            .add_plugins(RefinementPlugin);
    }
}
