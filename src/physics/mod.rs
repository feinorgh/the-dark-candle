pub mod amr_fluid;
pub mod atmosphere;
pub mod broad_phase;
pub mod collision;
pub mod constants;
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
        .add_plugins((
            gravity::GravityPlugin,
            collision::CollisionPlugin,
            RigidBodyPlugin,
            lbm_gas::LbmGasPlugin,
            flip_pic::FlipPicPlugin,
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
