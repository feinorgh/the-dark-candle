// Uniform spatial grid for broad-phase entity-entity collision detection.
//
// Entities with `CollisionShape` + `Transform` are binned into a uniform grid
// whose cell size matches the largest expected entity bounding radius. The
// broad phase emits pairs of entities whose cells overlap (including the 26
// neighbors), feeding the narrow phase.
//
// Complexity: O(n) insert, O(n × k) pair generation where k is the average
// neighbor count — effectively O(n) for sparse scenes.

use bevy::platform::collections::HashMap;
use bevy::prelude::*;

use super::shapes::CollisionShape;

/// Cell size in meters. Entities larger than this may span multiple cells.
/// 4 m covers most game entities (humanoids ~2m, large creatures ~4m).
const CELL_SIZE: f32 = 4.0;

/// Grid cell coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CellCoord {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl CellCoord {
    fn from_position(pos: Vec3) -> Self {
        Self {
            x: (pos.x / CELL_SIZE).floor() as i32,
            y: (pos.y / CELL_SIZE).floor() as i32,
            z: (pos.z / CELL_SIZE).floor() as i32,
        }
    }
}

/// The spatial grid resource, rebuilt each tick.
#[derive(Resource, Default, Debug)]
pub struct SpatialGrid {
    cells: HashMap<CellCoord, Vec<Entity>>,
}

impl SpatialGrid {
    /// Clear all cells for a fresh rebuild.
    pub fn clear(&mut self) {
        for list in self.cells.values_mut() {
            list.clear();
        }
    }

    /// Insert an entity at the given world position.
    pub fn insert(&mut self, entity: Entity, pos: Vec3) {
        let coord = CellCoord::from_position(pos);
        self.cells.entry(coord).or_default().push(entity);
    }

    /// Insert an entity that may span multiple cells (large bounding box).
    pub fn insert_aabb(&mut self, entity: Entity, center: Vec3, half_extents: Vec3) {
        let min = CellCoord::from_position(center - half_extents);
        let max = CellCoord::from_position(center + half_extents);

        for y in min.y..=max.y {
            for z in min.z..=max.z {
                for x in min.x..=max.x {
                    self.cells
                        .entry(CellCoord { x, y, z })
                        .or_default()
                        .push(entity);
                }
            }
        }
    }

    /// Iterate over all unique entity pairs that share a cell.
    ///
    /// Returns pairs where `a < b` (by entity index) to avoid duplicates.
    pub fn potential_pairs(&self) -> Vec<(Entity, Entity)> {
        let mut pairs = Vec::new();
        for entities in self.cells.values() {
            let n = entities.len();
            for i in 0..n {
                for j in (i + 1)..n {
                    let (a, b) = if entities[i] < entities[j] {
                        (entities[i], entities[j])
                    } else {
                        (entities[j], entities[i])
                    };
                    pairs.push((a, b));
                }
            }
        }
        pairs.sort();
        pairs.dedup();
        pairs
    }
}

/// Potential collision pairs produced by the broad phase.
#[derive(Resource, Default, Debug)]
pub struct BroadPhasePairs {
    pub pairs: Vec<(Entity, Entity)>,
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

/// Rebuild the spatial grid from all entities with a `CollisionShape`.
pub fn update_spatial_grid(
    mut grid: ResMut<SpatialGrid>,
    query: Query<(Entity, &Transform, &CollisionShape)>,
) {
    grid.clear();

    for (entity, transform, shape) in &query {
        let half = shape.bounding_aabb();
        let max_half = half.x.max(half.y).max(half.z);

        if max_half > CELL_SIZE * 0.5 {
            grid.insert_aabb(entity, transform.translation, half);
        } else {
            grid.insert(entity, transform.translation);
        }
    }
}

/// Generate broad-phase pairs from the spatial grid.
pub fn broad_phase_detect(grid: Res<SpatialGrid>, mut pairs: ResMut<BroadPhasePairs>) {
    pairs.pairs = grid.potential_pairs();
}

/// System set for broad-phase ordering.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct BroadPhaseSet;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_coord_from_origin() {
        let c = CellCoord::from_position(Vec3::ZERO);
        assert_eq!(c, CellCoord { x: 0, y: 0, z: 0 });
    }

    #[test]
    fn cell_coord_positive() {
        let c = CellCoord::from_position(Vec3::new(5.0, 5.0, 5.0));
        assert_eq!(c, CellCoord { x: 1, y: 1, z: 1 });
    }

    #[test]
    fn cell_coord_negative() {
        let c = CellCoord::from_position(Vec3::new(-1.0, -1.0, -1.0));
        assert_eq!(
            c,
            CellCoord {
                x: -1,
                y: -1,
                z: -1
            }
        );
    }

    #[test]
    fn grid_insert_and_pairs_same_cell() {
        let mut grid = SpatialGrid::default();
        let e1 = Entity::from_bits(1);
        let e2 = Entity::from_bits(2);

        grid.insert(e1, Vec3::new(1.0, 1.0, 1.0));
        grid.insert(e2, Vec3::new(2.0, 2.0, 2.0));

        let pairs = grid.potential_pairs();
        assert_eq!(pairs.len(), 1);
        assert!(pairs.contains(&(e1, e2)) || pairs.contains(&(e2, e1)));
    }

    #[test]
    fn grid_no_pairs_different_cells() {
        let mut grid = SpatialGrid::default();
        let e1 = Entity::from_bits(1);
        let e2 = Entity::from_bits(2);

        grid.insert(e1, Vec3::new(0.0, 0.0, 0.0));
        grid.insert(e2, Vec3::new(100.0, 100.0, 100.0));

        let pairs = grid.potential_pairs();
        assert!(pairs.is_empty());
    }

    #[test]
    fn grid_clear_resets() {
        let mut grid = SpatialGrid::default();
        grid.insert(Entity::from_bits(1), Vec3::ZERO);
        grid.clear();

        let pairs = grid.potential_pairs();
        assert!(pairs.is_empty());
    }

    #[test]
    fn grid_deduplicates_pairs() {
        let mut grid = SpatialGrid::default();
        let e1 = Entity::from_bits(1);
        let e2 = Entity::from_bits(2);

        // Insert into multiple overlapping cells via aabb
        grid.insert_aabb(e1, Vec3::ZERO, Vec3::splat(CELL_SIZE));
        grid.insert_aabb(e2, Vec3::ZERO, Vec3::splat(CELL_SIZE));

        let pairs = grid.potential_pairs();
        // Should have exactly one unique pair despite multi-cell overlap
        let unique: Vec<_> = pairs
            .iter()
            .filter(|&&(a, b)| a == e1 && b == e2 || a == e2 && b == e1)
            .collect();
        assert_eq!(unique.len(), 1, "Should have exactly one unique pair");
    }

    #[test]
    fn grid_three_entities_three_pairs() {
        let mut grid = SpatialGrid::default();
        let e1 = Entity::from_bits(1);
        let e2 = Entity::from_bits(2);
        let e3 = Entity::from_bits(3);

        // All in same cell
        grid.insert(e1, Vec3::new(1.0, 1.0, 1.0));
        grid.insert(e2, Vec3::new(1.5, 1.5, 1.5));
        grid.insert(e3, Vec3::new(2.0, 2.0, 2.0));

        let pairs = grid.potential_pairs();
        assert_eq!(pairs.len(), 3, "3 entities = 3 pairs: {pairs:?}");
    }
}
