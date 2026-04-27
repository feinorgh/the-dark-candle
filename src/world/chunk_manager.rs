// Shared chunk management resources.
//
// These types are used by the V2 cubed-sphere pipeline and by external systems
// (physics, audio, lighting, diagnostics) for voxel lookups. The V1
// ChunkManagerPlugin, ChunkMap, PendingChunks, and ChunkLoadRadius have been
// removed; only the terrain generator wrappers remain.

use bevy::prelude::*;
use std::sync::Arc;

use super::terrain::UnifiedTerrainGenerator;

/// Thread-safe handle to the terrain generator for async tasks.
#[derive(Resource, Clone)]
pub struct SharedTerrainGen(pub Arc<UnifiedTerrainGenerator>);

/// Wrapper resource holding the terrain generator.
#[derive(Resource)]
pub struct TerrainGeneratorRes(pub UnifiedTerrainGenerator);
