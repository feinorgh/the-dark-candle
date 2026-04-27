//! Plant body system: tree skeletal data, wind response, and felling.
//!
//! # Architecture
//! - `TreeSkeletonData` is a RON-loaded asset describing a species' branch skeleton.
//! - `apply_wind_to_plants` samples LBM pressure gradients and applies lateral
//!   torques to canopy bones.
//! - `check_tree_felling` monitors trunk health and despawns dead trees, spawning
//!   log entities in their place.
//!
//! # SI units
//! - Bone length: metres.
//! - Bone mass: kilograms.
//! - Flexural strength: Pascals (material resistance to bending failure).
//! - Torques: N·m.
//! - LBM pressure: Pascals (deviation from reference pressure).

use bevy::prelude::*;
use serde::Deserialize;

use crate::physics::lbm_gas::plugin::LbmState;
use crate::world::chunk::{CHUNK_SIZE, ChunkCoord};

use super::injury::BodyHealth;
use super::locomotion::Skeleton;

// ---------------------------------------------------------------------------
// RON asset types
// ---------------------------------------------------------------------------

/// Data for a single bone in a tree skeleton.
#[derive(Deserialize, Debug, Clone)]
pub struct TreeBoneData {
    /// Unique bone name (e.g. `"trunk"`, `"branch_L1"`, `"twig_L2_3"`).
    pub name: String,
    /// Index of the parent bone, or `None` for the root (trunk base).
    pub parent: Option<usize>,
    /// Bone length in metres.
    pub length: f32,
    /// Bone mass in kilograms (used for inertia calculations).
    pub mass: f32,
    /// Flexural strength in Pascals — bending load the bone can withstand.
    pub flexural_strength_pa: f32,
}

/// Full tree skeleton definition for one species, loaded from
/// `assets/data/skeletons/{species}.skeleton.ron`.
#[derive(Deserialize, Asset, TypePath, Debug, Clone)]
pub struct TreeSkeletonData {
    /// Species identifier (e.g. `"oak"`, `"pine"`).
    pub species: String,
    /// Ordered list of bones; index 0 is always the trunk.
    pub bones: Vec<TreeBoneData>,
    /// Rest-pose world-space transforms for each bone.
    /// Length must match `bones`.
    pub rest_pose: Vec<super::skeleton::BoneTransform>,
    /// Depth of the root system in metres (how far roots anchor the tree).
    pub root_depth_m: f32,
    /// Indices into `bones` for canopy bones that should receive wind torque.
    pub canopy_bone_indices: Vec<usize>,
}

/// Component holding a handle to the entity's `TreeSkeletonData` asset.
#[derive(Component, Debug, Clone)]
pub struct TreeSkeletonHandle(pub Handle<TreeSkeletonData>);

// ---------------------------------------------------------------------------
// Wind response system
// ---------------------------------------------------------------------------

/// Apply wind-driven lateral torques to canopy bones via LBM pressure gradients.
///
/// For each canopy bone, the pressure difference across the canopy area is
/// estimated as `ΔP × area` where `area ≈ bone_length²` (m²).
/// The resulting force (Newtons) is applied as a torque about the bone's base.
///
/// If `LbmState` is not available (e.g. during startup or in headless tests)
/// this system is silently skipped.
pub fn apply_wind_to_plants(
    lbm_state: Option<Res<LbmState>>,
    tree_skeleton_assets: Res<Assets<TreeSkeletonData>>,
    mut query: Query<(&mut Skeleton, &TreeSkeletonHandle, &Transform)>,
) {
    let Some(lbm) = lbm_state else {
        return;
    };

    for (mut skeleton, handle, tree_transform) in &mut query {
        let Some(tree_data) = tree_skeleton_assets.get(&handle.0) else {
            continue;
        };

        let world_pos = tree_transform.translation;
        let vx = world_pos.x.floor() as i32;
        let vy = world_pos.y.floor() as i32;
        let vz = world_pos.z.floor() as i32;

        let chunk_coord = ChunkCoord::from_voxel_pos(vx, vy, vz);
        let Some(lbm_grid) = lbm.get(&chunk_coord) else {
            continue;
        };

        // Local coordinates of the tree base within its chunk.
        let origin = chunk_coord.world_origin();
        let lx = (vx - origin.x).rem_euclid(CHUNK_SIZE as i32) as usize;
        let ly = (vy - origin.y).rem_euclid(CHUNK_SIZE as i32) as usize;
        let lz = (vz - origin.z).rem_euclid(CHUNK_SIZE as i32) as usize;

        // Sample pressure and density at two horizontally adjacent cells to
        // estimate a pressure gradient (x-direction).
        let size = lbm_grid.size();
        let lx1 = (lx + 1).min(size - 1);
        let lx0 = if lx > 0 { lx - 1 } else { 0 };

        let rho_right = lbm_grid.get(lx1, ly, lz).density();
        let rho_left = lbm_grid.get(lx0, ly, lz).density();
        // Pressure deviation ≈ (ρ − ρ_ref) × cs² (lattice Boltzmann units).
        // cs² = 1/3 for D3Q19; we use the density difference as a proxy for ΔP.
        let dp_dx = (rho_right - rho_left) * (1.0 / 3.0); // [lattice pressure units]

        // Similarly for the z-direction.
        let lz1 = (lz + 1).min(size - 1);
        let lz0 = if lz > 0 { lz - 1 } else { 0 };
        let rho_front = lbm_grid.get(lx, ly, lz1).density();
        let rho_back = lbm_grid.get(lx, ly, lz0).density();
        let dp_dz = (rho_front - rho_back) * (1.0 / 3.0);

        // Apply lateral torques to each canopy bone.
        for &bone_idx in &tree_data.canopy_bone_indices {
            if bone_idx >= tree_data.bones.len() || bone_idx >= skeleton.torques.len() {
                continue;
            }
            let bone = &tree_data.bones[bone_idx];
            // Canopy area estimated as bone_length² (m²).
            let area_m2 = bone.length * bone.length;
            // Wind force in lattice pressure units × m² (proportional to Newtons).
            let force_x = dp_dx * area_m2;
            let force_z = dp_dz * area_m2;
            // Torque = r × F; lever arm ≈ bone_length / 2 for a uniform beam.
            let lever = bone.length * 0.5;
            // Torque vector: lateral force creates a torque in the horizontal plane.
            skeleton.torques[bone_idx] += Vec3::new(force_z * lever, 0.0, -force_x * lever);
        }
    }
}

// ---------------------------------------------------------------------------
// Felling system
// ---------------------------------------------------------------------------

/// Name of the region in `BodyHealth` that corresponds to the trunk (bone index 0).
const TRUNK_REGION: &str = "trunk";

/// When the trunk region's HP reaches zero, despawn the tree and spawn log
/// entities at the tree's position.
///
/// Two logs are spawned with slight offsets to simulate the tree falling.
pub fn check_tree_felling(
    mut commands: Commands,
    query: Query<(Entity, &BodyHealth, &Transform), With<TreeSkeletonHandle>>,
) {
    for (entity, body_health, transform) in &query {
        let trunk_dead = body_health
            .regions
            .get(TRUNK_REGION)
            .is_some_and(|r| r.hp <= 0.0);

        if !trunk_dead {
            continue;
        }

        // Despawn the tree.
        commands.entity(entity).despawn();

        // Spawn 2–3 log entities at the tree's position with slight offsets.
        let base = transform.translation;
        for i in 0_u8..3 {
            let offset = Vec3::new(i as f32 * 0.5, 0.0, i as f32 * 0.3);
            commands.spawn((
                Transform::from_translation(base + offset),
                LogMarker {
                    source_tree: entity,
                },
            ));
        }
    }
}

/// Marker component for spawned log entities.
#[derive(Component, Debug, Clone)]
pub struct LogMarker {
    /// The tree entity this log originated from.
    pub source_tree: Entity,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tree_bone_data_default_fields() {
        let bone = TreeBoneData {
            name: "trunk".into(),
            parent: None,
            length: 8.0,
            mass: 500.0,
            flexural_strength_pa: 40_000_000.0, // ~40 MPa for oak
        };
        assert_eq!(bone.name, "trunk");
        assert!(bone.parent.is_none());
        assert!((bone.length - 8.0).abs() < f32::EPSILON);
    }

    #[test]
    fn tree_skeleton_data_canopy_indices_in_range() {
        let data = TreeSkeletonData {
            species: "oak".into(),
            bones: vec![
                TreeBoneData {
                    name: "trunk".into(),
                    parent: None,
                    length: 8.0,
                    mass: 500.0,
                    flexural_strength_pa: 40_000_000.0,
                },
                TreeBoneData {
                    name: "branch_L1".into(),
                    parent: Some(0),
                    length: 3.0,
                    mass: 50.0,
                    flexural_strength_pa: 30_000_000.0,
                },
            ],
            rest_pose: vec![crate::bodies::skeleton::BoneTransform::default(); 2],
            root_depth_m: 2.0,
            canopy_bone_indices: vec![1],
        };
        for &idx in &data.canopy_bone_indices {
            assert!(idx < data.bones.len(), "canopy index {idx} out of range");
        }
    }

    #[test]
    fn log_marker_stores_source_entity() {
        let e = Entity::from_bits(42);
        let log = LogMarker { source_tree: e };
        assert_eq!(log.source_tree, e);
    }

    #[test]
    fn trunk_region_hp_zero_indicates_felling() {
        use super::super::injury::{InjuryTier, RegionHealth};
        use std::collections::HashMap;

        let mut regions = HashMap::new();
        regions.insert(
            TRUNK_REGION.to_string(),
            RegionHealth {
                hp: 0.0,
                max_hp: 500.0,
                tier: InjuryTier::Severed,
            },
        );
        let bh = BodyHealth { regions };
        let trunk_dead = bh.regions.get(TRUNK_REGION).is_some_and(|r| r.hp <= 0.0);
        assert!(trunk_dead);
    }
}
