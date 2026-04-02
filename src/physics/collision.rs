// AABB vs voxel grid collision.
//
// Entities with a `Collider` component are checked against the voxel terrain
// each physics tick. Movement into solid voxels is rejected per-axis, allowing
// sliding along walls and walking up single-voxel steps.

#![allow(dead_code)]

use bevy::prelude::*;

use crate::world::chunk::{CHUNK_SIZE, Chunk, ChunkCoord};
use crate::world::chunk_manager::ChunkMap;

/// Axis-aligned bounding box collider, centered on the entity's Transform.
#[derive(Component, Debug, Clone)]
pub struct Collider {
    /// Half-extents of the bounding box (width/2, height/2, depth/2).
    pub half_extents: Vec3,
}

impl Collider {
    /// Create a collider from full width, height, and depth.
    pub fn new(width: f32, height: f32, depth: f32) -> Self {
        Self {
            half_extents: Vec3::new(width / 2.0, height / 2.0, depth / 2.0),
        }
    }

    /// Create a capsule-like collider (common for humanoids: width=depth, taller).
    pub fn capsule(radius: f32, height: f32) -> Self {
        Self::new(radius * 2.0, height, radius * 2.0)
    }

    /// Minimum corner of the AABB at a given position.
    pub fn min_at(&self, pos: Vec3) -> Vec3 {
        pos - self.half_extents
    }

    /// Maximum corner of the AABB at a given position.
    pub fn max_at(&self, pos: Vec3) -> Vec3 {
        pos + self.half_extents
    }
}

/// Check whether any solid voxel overlaps the given AABB.
pub fn aabb_intersects_terrain(
    aabb_min: Vec3,
    aabb_max: Vec3,
    chunk_map: &ChunkMap,
    chunks: &Query<&Chunk>,
) -> bool {
    let cs = CHUNK_SIZE as i32;

    // Iterate over all voxel positions the AABB covers
    let vx_min = aabb_min.x.floor() as i32;
    let vy_min = aabb_min.y.floor() as i32;
    let vz_min = aabb_min.z.floor() as i32;
    let vx_max = aabb_max.x.ceil() as i32;
    let vy_max = aabb_max.y.ceil() as i32;
    let vz_max = aabb_max.z.ceil() as i32;

    for vy in vy_min..vy_max {
        for vz in vz_min..vz_max {
            for vx in vx_min..vx_max {
                let coord =
                    ChunkCoord::new(vx.div_euclid(cs), vy.div_euclid(cs), vz.div_euclid(cs));

                let Some(entity) = chunk_map.get(&coord) else {
                    continue;
                };
                let Ok(chunk) = chunks.get(entity) else {
                    continue;
                };

                let lx = vx.rem_euclid(cs) as usize;
                let ly = vy.rem_euclid(cs) as usize;
                let lz = vz.rem_euclid(cs) as usize;

                if chunk.get(lx, ly, lz).is_solid() {
                    return true;
                }
            }
        }
    }

    false
}

/// Resolve collisions for entities with Collider + PhysicsBody.
/// Uses per-axis resolution: tries X, then Y, then Z independently.
/// This prevents entities from passing through terrain while allowing
/// sliding along surfaces.
pub fn resolve_collisions(
    chunk_map: Option<Res<ChunkMap>>,
    chunks: Query<&Chunk>,
    mut bodies: Query<(&Collider, &mut super::gravity::PhysicsBody, &mut Transform)>,
) {
    let Some(chunk_map) = chunk_map else {
        return;
    };
    for (collider, mut body, mut transform) in &mut bodies {
        let pos = transform.translation;

        // Check each axis independently: if moving along that axis causes
        // a collision, revert that axis and zero the velocity component.

        // X axis
        let test_pos = Vec3::new(pos.x, pos.y, pos.z);
        if aabb_intersects_terrain(
            collider.min_at(test_pos),
            collider.max_at(test_pos),
            &chunk_map,
            &chunks,
        ) {
            // Nudge out of collision along X
            let prev_x = pos.x - body.velocity.x * 0.016; // approximate
            transform.translation.x = prev_x;
            body.velocity.x = 0.0;
        }

        // Z axis
        let test_pos = transform.translation;
        if aabb_intersects_terrain(
            collider.min_at(test_pos),
            collider.max_at(test_pos),
            &chunk_map,
            &chunks,
        ) {
            let prev_z = test_pos.z - body.velocity.z * 0.016;
            transform.translation.z = prev_z;
            body.velocity.z = 0.0;
        }

        // Y axis (gravity already handles ground; this catches ceilings)
        let test_pos = transform.translation;
        if aabb_intersects_terrain(
            collider.min_at(test_pos),
            collider.max_at(test_pos),
            &chunk_map,
            &chunks,
        ) && body.velocity.y > 0.0
        {
            body.velocity.y = 0.0;
            // Let gravity system handle downward resolution
        }
    }
}

/// Check whether any solid voxel overlaps the given AABB using an octree.
///
/// This is the octree-aware counterpart to `aabb_intersects_terrain()`.
/// When chunks store octree data, this function can skip entire empty subtrees
/// for faster broad-phase rejection. Falls back to per-voxel checks at leaves.
pub fn aabb_intersects_octree(
    aabb_min: Vec3,
    aabb_max: Vec3,
    octree: &crate::world::octree::OctreeNode<crate::world::voxel::Voxel>,
    octree_origin: Vec3,
    octree_size: usize,
) -> bool {
    aabb_octree_recursive(
        aabb_min,
        aabb_max,
        octree,
        octree_origin.x,
        octree_origin.y,
        octree_origin.z,
        octree_size as f32,
    )
}

fn aabb_octree_recursive(
    aabb_min: Vec3,
    aabb_max: Vec3,
    node: &crate::world::octree::OctreeNode<crate::world::voxel::Voxel>,
    ox: f32,
    oy: f32,
    oz: f32,
    size: f32,
) -> bool {
    // Quick AABB-vs-AABB rejection: does the query AABB overlap this node's volume?
    let node_max_x = ox + size;
    let node_max_y = oy + size;
    let node_max_z = oz + size;

    if aabb_min.x >= node_max_x
        || aabb_max.x <= ox
        || aabb_min.y >= node_max_y
        || aabb_max.y <= oy
        || aabb_min.z >= node_max_z
        || aabb_max.z <= oz
    {
        return false; // No overlap with this node.
    }

    match node {
        crate::world::octree::OctreeNode::Leaf(voxel) => {
            // Uniform region: if solid, collision detected.
            voxel.is_solid()
        }
        crate::world::octree::OctreeNode::Branch(children) => {
            let half = size / 2.0;
            for (i, child) in children.iter().enumerate() {
                let cx = ox + (i & 1) as f32 * half;
                let cy = oy + ((i >> 1) & 1) as f32 * half;
                let cz = oz + ((i >> 2) & 1) as f32 * half;
                if aabb_octree_recursive(aabb_min, aabb_max, child, cx, cy, cz, half) {
                    return true;
                }
            }
            false
        }
    }
}

/// Plugin for AABB collision resolution.
pub struct CollisionPlugin;

impl Plugin for CollisionPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            FixedUpdate,
            resolve_collisions.after(super::gravity::GravitySet),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collider_new_computes_half_extents() {
        let c = Collider::new(2.0, 4.0, 2.0);
        assert_eq!(c.half_extents, Vec3::new(1.0, 2.0, 1.0));
    }

    #[test]
    fn collider_capsule_is_symmetric_xz() {
        let c = Collider::capsule(0.4, 1.8);
        assert_eq!(c.half_extents.x, c.half_extents.z);
        assert_eq!(c.half_extents.y, 0.9);
    }

    #[test]
    fn aabb_corners_are_correct() {
        let c = Collider::new(1.0, 2.0, 1.0);
        let pos = Vec3::new(10.0, 20.0, 30.0);
        assert_eq!(c.min_at(pos), Vec3::new(9.5, 19.0, 29.5));
        assert_eq!(c.max_at(pos), Vec3::new(10.5, 21.0, 30.5));
    }

    #[test]
    fn collider_zero_size_is_point() {
        let c = Collider::new(0.0, 0.0, 0.0);
        let pos = Vec3::new(5.0, 5.0, 5.0);
        assert_eq!(c.min_at(pos), pos);
        assert_eq!(c.max_at(pos), pos);
    }

    // --- Octree collision tests ---

    use crate::world::octree::OctreeNode;
    use crate::world::voxel::{MaterialId, Voxel};

    fn air() -> Voxel {
        Voxel::default()
    }

    fn stone() -> Voxel {
        Voxel::new(MaterialId::STONE)
    }

    #[test]
    fn octree_collision_all_air_no_hit() {
        let tree = OctreeNode::new_leaf(air());
        let hit = aabb_intersects_octree(
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(1.5, 1.5, 1.5),
            &tree,
            Vec3::ZERO,
            4,
        );
        assert!(!hit);
    }

    #[test]
    fn octree_collision_all_solid_hit() {
        let tree = OctreeNode::new_leaf(stone());
        let hit = aabb_intersects_octree(
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(1.5, 1.5, 1.5),
            &tree,
            Vec3::ZERO,
            4,
        );
        assert!(hit);
    }

    #[test]
    fn octree_collision_miss_outside_volume() {
        let tree = OctreeNode::new_leaf(stone());
        let hit = aabb_intersects_octree(
            Vec3::new(10.0, 10.0, 10.0),
            Vec3::new(11.0, 11.0, 11.0),
            &tree,
            Vec3::ZERO,
            4, // 4m extent
        );
        assert!(!hit);
    }

    #[test]
    fn octree_collision_partial_solid() {
        let mut tree = OctreeNode::new_leaf(air());
        // Set voxel (0,0,0) to stone in a 4×4×4 tree.
        tree.set(0, 0, 0, 4, 1, stone());

        // AABB overlapping the stone cell → hit.
        assert!(aabb_intersects_octree(
            Vec3::new(-0.5, -0.5, -0.5),
            Vec3::new(0.5, 0.5, 0.5),
            &tree,
            Vec3::ZERO,
            4,
        ));

        // AABB in the air region → no hit.
        assert!(!aabb_intersects_octree(
            Vec3::new(2.0, 2.0, 2.0),
            Vec3::new(3.0, 3.0, 3.0),
            &tree,
            Vec3::ZERO,
            4,
        ));
    }

    #[test]
    fn octree_collision_with_offset_origin() {
        let tree = OctreeNode::new_leaf(stone());
        // Tree at offset (100, 0, 0), size 4.
        let hit = aabb_intersects_octree(
            Vec3::new(101.0, 1.0, 1.0),
            Vec3::new(102.0, 2.0, 2.0),
            &tree,
            Vec3::new(100.0, 0.0, 0.0),
            4,
        );
        assert!(hit);

        // Outside the offset volume.
        let miss = aabb_intersects_octree(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 1.0),
            &tree,
            Vec3::new(100.0, 0.0, 0.0),
            4,
        );
        assert!(!miss);
    }
}
