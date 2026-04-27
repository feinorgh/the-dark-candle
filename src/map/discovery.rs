// Discovery tracking: records which chunk columns the player has visited.

use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::planet::BiomeType;
use crate::world::chunk::ChunkCoord;
use crate::world::planetary_sampler::ChunkBiomeData;

/// Metadata for a single discovered chunk column.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredColumn {
    pub biome: BiomeType,
    /// Approximate surface Y level (chunk Y coordinate).
    pub surface_y: i32,
}

/// Tracks which XZ chunk columns have been visited/generated.
///
/// Keyed by `[x, z]` chunk coordinates.
#[derive(Resource, Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiscoveredColumns {
    pub columns: HashMap<[i32; 2], DiscoveredColumn>,
}

/// System: record all chunk columns currently in the world.
pub fn track_discoveries(
    biome_q: Query<(&ChunkCoord, Option<&ChunkBiomeData>)>,
    mut discovered: ResMut<DiscoveredColumns>,
) {
    for (coord, biome_data) in &biome_q {
        let key = [coord.x, coord.z];
        // Only record if not already discovered, or update surface_y if higher.
        let biome = biome_data
            .map(|b| b.planet_biome)
            .unwrap_or(BiomeType::TemperateForest);

        discovered
            .columns
            .entry(key)
            .and_modify(|col| {
                if coord.y > col.surface_y {
                    col.surface_y = coord.y;
                    col.biome = biome;
                }
            })
            .or_insert(DiscoveredColumn {
                biome,
                surface_y: coord.y,
            });
    }
}
