// Chunk manager: loads and unloads chunks around the camera.
//
// Each frame the manager compares the set of currently loaded chunks against
// the set that *should* be loaded (a sphere of chunks centered on the camera).
// New chunks are spawned and far-away chunks are despawned. Terrain generation
// fills newly created chunks via the TerrainGenerator.

#![allow(dead_code)]

use bevy::prelude::*;
use std::collections::{HashMap, HashSet};

use super::chunk::{Chunk, ChunkCoord, CHUNK_SIZE};
use super::terrain::{TerrainConfig, TerrainGenerator};
use crate::camera::FpsCamera;

/// How many chunks outward from the camera to load in each axis.
#[derive(Resource)]
pub struct ChunkLoadRadius {
    /// Horizontal (XZ) radius in chunk units.
    pub horizontal: i32,
    /// Vertical (Y) radius in chunk units.
    pub vertical: i32,
}

impl Default for ChunkLoadRadius {
    fn default() -> Self {
        Self {
            horizontal: 4,
            vertical: 2,
        }
    }
}

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

    pub fn coords(&self) -> impl Iterator<Item = &ChunkCoord> {
        self.map.keys()
    }
}

/// Compute the set of chunk coordinates that should be loaded around a world position.
pub fn desired_chunks(world_pos: Vec3, radius: &ChunkLoadRadius) -> HashSet<ChunkCoord> {
    let cs = CHUNK_SIZE as f32;
    let center_cx = (world_pos.x / cs).floor() as i32;
    let center_cy = (world_pos.y / cs).floor() as i32;
    let center_cz = (world_pos.z / cs).floor() as i32;

    let h = radius.horizontal;
    let v = radius.vertical;
    let h_sq = (h * h) as f32;

    let mut set = HashSet::new();
    for dy in -v..=v {
        for dz in -h..=h {
            for dx in -h..=h {
                // Use cylindrical distance (XZ plane) for more natural loading
                let dist_sq = (dx * dx + dz * dz) as f32;
                if dist_sq <= h_sq {
                    set.insert(ChunkCoord::new(
                        center_cx + dx,
                        center_cy + dy,
                        center_cz + dz,
                    ));
                }
            }
        }
    }
    set
}

/// System: spawns and despawns chunks to keep the loaded set matching the camera position.
pub fn update_chunks(
    mut commands: Commands,
    mut chunk_map: ResMut<ChunkMap>,
    radius: Res<ChunkLoadRadius>,
    terrain_gen: Res<TerrainGeneratorRes>,
    camera_q: Query<&Transform, With<FpsCamera>>,
) {
    let Ok(cam_transform) = camera_q.single() else {
        return;
    };

    let desired = desired_chunks(cam_transform.translation, &radius);

    // Despawn chunks no longer in range
    let loaded: Vec<ChunkCoord> = chunk_map.coords().copied().collect();
    for coord in loaded {
        if !desired.contains(&coord) {
            if let Some(entity) = chunk_map.remove(&coord) {
                commands.entity(entity).despawn();
            }
        }
    }

    // Spawn new chunks that are in range but not yet loaded
    for &coord in &desired {
        if !chunk_map.contains(&coord) {
            let mut chunk = Chunk::new_empty(coord);
            terrain_gen.0.generate_chunk(&mut chunk);
            let origin = coord.world_origin();
            let entity = commands
                .spawn((
                    chunk,
                    coord,
                    Transform::from_xyz(origin.x as f32, origin.y as f32, origin.z as f32),
                ))
                .id();
            chunk_map.insert(coord, entity);
        }
    }
}

/// Wrapper resource holding the terrain generator.
#[derive(Resource)]
pub struct TerrainGeneratorRes(pub TerrainGenerator);

/// Plugin that registers chunk management resources and systems.
pub struct ChunkManagerPlugin;

impl Plugin for ChunkManagerPlugin {
    fn build(&self, app: &mut App) {
        let config = TerrainConfig::default();
        let generator = TerrainGenerator::new(config);
        app.init_resource::<ChunkMap>()
            .init_resource::<ChunkLoadRadius>()
            .insert_resource(TerrainGeneratorRes(generator))
            .add_systems(
                Update,
                update_chunks.in_set(super::WorldSet::ChunkManagement),
            );
    }
}

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
    fn desired_chunks_at_origin() {
        let radius = ChunkLoadRadius {
            horizontal: 2,
            vertical: 1,
        };
        let chunks = desired_chunks(Vec3::ZERO, &radius);

        // Center chunk must be included
        assert!(chunks.contains(&ChunkCoord::new(0, 0, 0)));

        // Vertical neighbors
        assert!(chunks.contains(&ChunkCoord::new(0, 1, 0)));
        assert!(chunks.contains(&ChunkCoord::new(0, -1, 0)));

        // Horizontal neighbors within radius
        assert!(chunks.contains(&ChunkCoord::new(1, 0, 0)));
        assert!(chunks.contains(&ChunkCoord::new(-1, 0, 0)));
        assert!(chunks.contains(&ChunkCoord::new(0, 0, 1)));
        assert!(chunks.contains(&ChunkCoord::new(0, 0, -1)));
        assert!(chunks.contains(&ChunkCoord::new(2, 0, 0)));

        // Diagonal at distance sqrt(2*2 + 2*2) = 2.83 > 2, should be excluded
        assert!(!chunks.contains(&ChunkCoord::new(2, 0, 2)));
    }

    #[test]
    fn desired_chunks_count_is_reasonable() {
        let radius = ChunkLoadRadius {
            horizontal: 4,
            vertical: 2,
        };
        let chunks = desired_chunks(Vec3::ZERO, &radius);

        // With h=4, v=2: 5 vertical layers × ~π*4² horizontal ≈ 5*50 = ~250
        // Exact count depends on circular culling
        assert!(chunks.len() > 100, "Too few chunks: {}", chunks.len());
        assert!(chunks.len() < 500, "Too many chunks: {}", chunks.len());
    }

    #[test]
    fn desired_chunks_follows_camera() {
        let radius = ChunkLoadRadius {
            horizontal: 1,
            vertical: 0,
        };

        // Camera at origin → chunks around (0,0,0)
        let set_a = desired_chunks(Vec3::ZERO, &radius);
        assert!(set_a.contains(&ChunkCoord::new(0, 0, 0)));

        // Camera moved far away → chunks shift
        let far = Vec3::new(1000.0, 0.0, 0.0);
        let set_b = desired_chunks(far, &radius);
        let far_cx = (1000.0 / CHUNK_SIZE as f32).floor() as i32;
        assert!(set_b.contains(&ChunkCoord::new(far_cx, 0, 0)));
        assert!(!set_b.contains(&ChunkCoord::new(0, 0, 0)));
    }

    #[test]
    fn desired_chunks_negative_position() {
        let radius = ChunkLoadRadius {
            horizontal: 1,
            vertical: 0,
        };
        let pos = Vec3::new(-100.0, -100.0, -100.0);
        let chunks = desired_chunks(pos, &radius);

        let cx = (-100.0 / CHUNK_SIZE as f32).floor() as i32;
        let cy = (-100.0 / CHUNK_SIZE as f32).floor() as i32;
        let cz = (-100.0 / CHUNK_SIZE as f32).floor() as i32;
        assert!(chunks.contains(&ChunkCoord::new(cx, cy, cz)));
    }

    #[test]
    fn zero_radius_loads_only_center() {
        let radius = ChunkLoadRadius {
            horizontal: 0,
            vertical: 0,
        };
        let chunks = desired_chunks(Vec3::new(16.0, 16.0, 16.0), &radius);
        assert_eq!(chunks.len(), 1);
        assert!(chunks.contains(&ChunkCoord::new(0, 0, 0)));
    }

    #[test]
    fn chunk_load_radius_default() {
        let radius = ChunkLoadRadius::default();
        assert_eq!(radius.horizontal, 4);
        assert_eq!(radius.vertical, 2);
    }

    #[test]
    fn desired_chunks_respects_vertical_limit() {
        let radius = ChunkLoadRadius {
            horizontal: 2,
            vertical: 1,
        };
        let chunks = desired_chunks(Vec3::ZERO, &radius);

        // Y=2 should NOT be loaded (vertical radius is 1)
        assert!(!chunks.contains(&ChunkCoord::new(0, 2, 0)));
        assert!(!chunks.contains(&ChunkCoord::new(0, -2, 0)));

        // Y=1 SHOULD be loaded
        assert!(chunks.contains(&ChunkCoord::new(0, 1, 0)));
        assert!(chunks.contains(&ChunkCoord::new(0, -1, 0)));
    }

    #[test]
    fn desired_chunks_cylindrical_shape() {
        let radius = ChunkLoadRadius {
            horizontal: 3,
            vertical: 0,
        };
        let chunks = desired_chunks(Vec3::ZERO, &radius);

        // On-axis at radius: dx=3, dz=0 → dist²=9 ≤ 9 → included
        assert!(chunks.contains(&ChunkCoord::new(3, 0, 0)));

        // Corner: dx=3, dz=1 → dist²=10 > 9 → excluded
        assert!(!chunks.contains(&ChunkCoord::new(3, 0, 1)));

        // Diagonal: dx=2, dz=2 → dist²=8 ≤ 9 → included
        assert!(chunks.contains(&ChunkCoord::new(2, 0, 2)));
    }
}
