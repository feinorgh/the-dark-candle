pub mod amr_fluid;
pub mod atmosphere;
pub mod broad_phase;
pub mod collision;
pub mod constants;
pub mod coupling;
pub mod flip_pic;
pub mod gravity;
pub mod integrity;
pub mod lbm_gas;
pub mod narrow_phase;
pub mod pressure;
pub mod rigid_body;
pub mod shapes;
pub mod solver;
pub mod voxel_bridge;

use bevy::prelude::*;
use bevy_common_assets::ron::RonAssetPlugin;

/// Configuration for physics level-of-detail: expensive fluid simulations
/// (LBM gas, AMR Navier-Stokes, FLIP/PIC particles) only run on chunks
/// within this radius (in chunk coordinates) of the camera.
#[derive(Resource, Debug, Clone)]
pub struct PhysicsLodConfig {
    /// Maximum Chebyshev distance (in chunks) from the camera chunk for
    /// fluid simulation. Chunks beyond this are skipped. 0 = disabled.
    pub active_radius_chunks: u32,
}

impl Default for PhysicsLodConfig {
    fn default() -> Self {
        Self {
            active_radius_chunks: 3,
        }
    }
}

impl PhysicsLodConfig {
    /// Returns true if the chunk at `coord` is within the active physics
    /// radius of `camera_chunk`.  Returns true for all chunks when the
    /// radius is 0 (disabled).
    pub fn is_active(
        &self,
        coord: &crate::world::chunk::ChunkCoord,
        camera_chunk: &crate::world::chunk::ChunkCoord,
    ) -> bool {
        if self.active_radius_chunks == 0 {
            return true;
        }
        let dx = (coord.x - camera_chunk.x).unsigned_abs();
        let dy = (coord.y - camera_chunk.y).unsigned_abs();
        let dz = (coord.z - camera_chunk.z).unsigned_abs();
        let dist = dx.max(dy).max(dz);
        dist <= self.active_radius_chunks
    }
}

pub struct PhysicsPlugin;

impl Plugin for PhysicsPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(RonAssetPlugin::<constants::UniversalConstants>::new(&[
            "universal_constants.ron",
        ]))
        .add_plugins(RonAssetPlugin::<constants::WorldConstants>::new(&[
            "world_constants.ron",
        ]))
        .add_plugins(RonAssetPlugin::<atmosphere::AtmosphereConfig>::new(&[
            "atmosphere_config.ron",
        ]))
        .init_resource::<PhysicsLodConfig>()
        .add_plugins((
            gravity::GravityPlugin,
            collision::CollisionPlugin,
            RigidBodyPlugin,
            lbm_gas::LbmGasPlugin,
            flip_pic::FlipPicPlugin,
            amr_fluid::AmrFluidPlugin,
            coupling::CouplingPlugin,
        ));
    }
}

/// Plugin for the rigid body physics pipeline: entity-entity collision
/// detection and response with angular dynamics.
///
/// System ordering within `FixedUpdate`:
///   GravitySet → RigidBodySet (angular integration)
///              → BroadPhaseSet → NarrowPhaseSet → SolverSet
///              → existing CollisionPlugin (terrain)
pub struct RigidBodyPlugin;

impl Plugin for RigidBodyPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<broad_phase::SpatialGrid>()
            .init_resource::<broad_phase::BroadPhasePairs>()
            .init_resource::<narrow_phase::Contacts>()
            .add_systems(
                FixedUpdate,
                (
                    rigid_body::angular_integration
                        .in_set(rigid_body::RigidBodySet)
                        .after(gravity::GravitySet),
                    broad_phase::update_spatial_grid
                        .in_set(broad_phase::BroadPhaseSet)
                        .after(gravity::GravitySet),
                    broad_phase::broad_phase_detect
                        .in_set(broad_phase::BroadPhaseSet)
                        .after(broad_phase::update_spatial_grid),
                    narrow_phase::narrow_phase_detect
                        .in_set(narrow_phase::NarrowPhaseSet)
                        .after(broad_phase::BroadPhaseSet),
                    solver::solve_contacts
                        .in_set(solver::SolverSet)
                        .after(narrow_phase::NarrowPhaseSet)
                        .after(rigid_body::RigidBodySet),
                ),
            );
    }
}

pub use atmosphere::AtmosphereConfig;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::chunk::ChunkCoord;

    #[test]
    fn physics_lod_disabled_allows_all() {
        let config = PhysicsLodConfig {
            active_radius_chunks: 0,
        };
        let cam = ChunkCoord::new(0, 0, 0);
        let far = ChunkCoord::new(100, 100, 100);
        assert!(config.is_active(&far, &cam));
    }

    #[test]
    fn physics_lod_within_radius() {
        let config = PhysicsLodConfig {
            active_radius_chunks: 3,
        };
        let cam = ChunkCoord::new(5, 5, 5);
        assert!(config.is_active(&ChunkCoord::new(5, 5, 5), &cam));
        assert!(config.is_active(&ChunkCoord::new(7, 5, 5), &cam));
        assert!(config.is_active(&ChunkCoord::new(8, 8, 8), &cam));
        assert!(!config.is_active(&ChunkCoord::new(9, 5, 5), &cam));
        assert!(!config.is_active(&ChunkCoord::new(5, 5, 9), &cam));
    }
}
