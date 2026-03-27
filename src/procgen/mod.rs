pub mod biomes;
pub mod creatures;
pub mod items;
pub mod props;
pub mod spawning;
pub mod tree;

use bevy::prelude::*;

pub struct ProcgenPlugin;

impl Plugin for ProcgenPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(biomes::BiomePlugin);

        match props::load_prop_registry() {
            Ok(registry) => {
                info!("Loaded PropRegistry with {} props", registry.len());
                app.insert_resource(registry);
            }
            Err(e) => {
                warn!("Failed to load PropRegistry: {e}");
                app.init_resource::<props::PropRegistry>();
            }
        }

        // Run after ChunkManagement so chunk despawn commands are flushed
        // before we try to decorate or access chunk entities.
        app.add_systems(
            Update,
            props::decorate_chunks.after(crate::world::WorldSet::ChunkManagement),
        );
    }
}
