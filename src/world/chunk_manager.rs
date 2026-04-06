// Shared chunk management resources.
//
// These types are used by the V2 cubed-sphere pipeline and by external systems
// (physics, audio, lighting, diagnostics) for voxel lookups. The V1
// ChunkManagerPlugin and its dispatch systems have been removed.

use bevy::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::chunk::ChunkCoord;
use super::terrain::UnifiedTerrainGenerator;

/// How many chunks outward from the camera to load in each axis.
#[derive(Resource)]
pub struct ChunkLoadRadius {
    /// Horizontal (XZ) radius in chunk units.
    pub horizontal: i32,
    /// Vertical (Y) radius in chunk units.
    pub vertical: i32,
    /// For spherical mode: how many meters below the surface to load.
    pub shell_depth: f64,
    /// For spherical mode: how many meters above the surface to load.
    pub shell_height: f64,
}

impl Default for ChunkLoadRadius {
    fn default() -> Self {
        Self {
            horizontal: 4,
            vertical: 2,
            shell_depth: 128.0,  // 4 chunks deep
            shell_height: 128.0, // 4 chunks high
        }
    }
}

/// Minimum horizontal chunk radius. Never go below this.
#[allow(dead_code)]
const MIN_CHUNK_RADIUS: i32 = 2;

/// Maximum horizontal chunk radius. Never exceed this.
#[allow(dead_code)]
const MAX_CHUNK_RADIUS: i32 = 12;

/// Set of chunk coordinates whose terrain generation is in flight.
/// Prevents double-dispatch while an async task is still running.
#[derive(Resource, Default)]
pub struct PendingChunks {
    coords: HashSet<ChunkCoord>,
}

impl PendingChunks {
    pub fn contains(&self, coord: &ChunkCoord) -> bool {
        self.coords.contains(coord)
    }

    pub fn insert(&mut self, coord: ChunkCoord) {
        self.coords.insert(coord);
    }

    pub fn remove(&mut self, coord: &ChunkCoord) {
        self.coords.remove(coord);
    }

    pub fn len(&self) -> usize {
        self.coords.len()
    }

    pub fn is_empty(&self) -> bool {
        self.coords.is_empty()
    }
}

/// Thread-safe handle to the terrain generator for async tasks.
#[derive(Resource, Clone)]
pub struct SharedTerrainGen(pub Arc<UnifiedTerrainGenerator>);

/// Maps chunk coordinates to their ECS entity for O(1) lookup.
#[derive(Resource, Default)]
pub struct ChunkMap {
    map: HashMap<ChunkCoord, Entity>,
}

impl ChunkMap {
    pub fn get(&self, coord: &ChunkCoord) -> Option<Entity> {
        self.map.get(coord).copied()
    }

    pub fn contains(&self, coord: &ChunkCoord) -> bool {
        self.map.contains_key(coord)
    }

    pub fn insert(&mut self, coord: ChunkCoord, entity: Entity) {
        self.map.insert(coord, entity);
    }

    pub fn remove(&mut self, coord: &ChunkCoord) -> Option<Entity> {
        self.map.remove(coord)
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn clear(&mut self) {
        self.map.clear();
    }

    pub fn coords(&self) -> impl Iterator<Item = &ChunkCoord> {
        self.map.keys()
    }
}

/// Wrapper resource holding the terrain generator.
#[derive(Resource)]
pub struct TerrainGeneratorRes(pub UnifiedTerrainGenerator);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_map_insert_and_lookup() {
        let mut map = ChunkMap::default();
        assert!(map.is_empty());

        let coord = ChunkCoord::new(1, 2, 3);
        let entity = Entity::from_bits(42);
        map.insert(coord, entity);

        assert_eq!(map.len(), 1);
        assert!(map.contains(&coord));
        assert_eq!(map.get(&coord), Some(entity));
        assert!(!map.contains(&ChunkCoord::new(0, 0, 0)));
    }

    #[test]
    fn chunk_map_remove() {
        let mut map = ChunkMap::default();
        let coord = ChunkCoord::new(0, 0, 0);
        let entity = Entity::from_bits(1);
        map.insert(coord, entity);

        let removed = map.remove(&coord);
        assert_eq!(removed, Some(entity));
        assert!(map.is_empty());
        assert_eq!(map.remove(&coord), None);
    }

    #[test]
    fn chunk_load_radius_default() {
        let radius = ChunkLoadRadius::default();
        assert_eq!(radius.horizontal, 4);
        assert_eq!(radius.vertical, 2);
    }
}
