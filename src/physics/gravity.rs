// Entity gravity and ground detection.
//
// Any entity with a `PhysicsBody` component automatically receives gravity,
// terminal velocity, and ground collision against the voxel terrain.
// Runs on `FixedUpdate` for deterministic simulation.

#![allow(dead_code)]

use bevy::prelude::*;

use crate::world::chunk::Chunk;
use crate::world::chunk_manager::ChunkMap;
use crate::world::collision::ground_height_at;

/// Gravity acceleration in m/s² (voxel units).
pub const GRAVITY: f32 = 20.0;
/// Maximum falling speed.
pub const TERMINAL_VELOCITY: f32 = 50.0;

/// Marker + state for any entity affected by gravity.
#[derive(Component, Debug)]
pub struct PhysicsBody {
    /// Current velocity (full 3D; gravity affects Y component).
    pub velocity: Vec3,
    /// Whether the entity is resting on a solid surface.
    pub grounded: bool,
    /// Multiplier for gravity. 1.0 = normal, 0.0 = weightless.
    pub gravity_scale: f32,
    /// Distance from the entity's Transform origin to its feet.
    /// Ground collision places the origin at `ground_y + foot_offset`.
    pub foot_offset: f32,
}

impl Default for PhysicsBody {
    fn default() -> Self {
        Self {
            velocity: Vec3::ZERO,
            grounded: false,
            gravity_scale: 1.0,
            foot_offset: 0.0,
        }
    }
}

impl PhysicsBody {
    /// Create a physics body with a specific foot offset (e.g. half-height for centered origins).
    pub fn with_foot_offset(mut self, offset: f32) -> Self {
        self.foot_offset = offset;
        self
    }

    /// Create a weightless physics body (no gravity, still has velocity).
    pub fn weightless() -> Self {
        Self {
            gravity_scale: 0.0,
            ..Default::default()
        }
    }
}

/// Apply gravity to all entities with a `PhysicsBody`.
/// Also resolves ground collision against the voxel terrain.
pub fn apply_gravity(
    time: Res<Time>,
    chunk_map: Res<ChunkMap>,
    chunks: Query<&Chunk>,
    mut bodies: Query<(&mut PhysicsBody, &mut Transform)>,
) {
    let dt = time.delta_secs();
    if dt == 0.0 {
        return;
    }

    for (mut body, mut transform) in &mut bodies {
        if body.gravity_scale == 0.0 {
            // Apply velocity but no gravity (e.g. floating items)
            transform.translation += body.velocity * dt;
            continue;
        }

        // Apply gravity acceleration
        body.velocity.y -= GRAVITY * body.gravity_scale * dt;
        body.velocity.y = body.velocity.y.max(-TERMINAL_VELOCITY);

        // Integrate velocity
        transform.translation += body.velocity * dt;

        // Ground collision
        if let Some(ground_y) = ground_height_at(
            transform.translation.x,
            transform.translation.z,
            &chunk_map,
            &chunks,
        ) {
            let feet_y = transform.translation.y - body.foot_offset;
            if feet_y <= ground_y {
                transform.translation.y = ground_y + body.foot_offset;
                if body.velocity.y < 0.0 {
                    body.velocity.y = 0.0;
                }
                body.grounded = true;
            } else {
                body.grounded = false;
            }
        }
    }
}

/// System set for physics ordering.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct GravitySet;

pub struct GravityPlugin;

impl Plugin for GravityPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(FixedUpdate, apply_gravity.in_set(GravitySet));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_physics_body() {
        let body = PhysicsBody::default();
        assert_eq!(body.velocity, Vec3::ZERO);
        assert!(!body.grounded);
        assert_eq!(body.gravity_scale, 1.0);
        assert_eq!(body.foot_offset, 0.0);
    }

    #[test]
    fn with_foot_offset_builder() {
        let body = PhysicsBody::default().with_foot_offset(0.9);
        assert_eq!(body.foot_offset, 0.9);
        assert_eq!(body.gravity_scale, 1.0);
    }

    #[test]
    fn weightless_body_has_zero_gravity() {
        let body = PhysicsBody::weightless();
        assert_eq!(body.gravity_scale, 0.0);
        assert_eq!(body.velocity, Vec3::ZERO);
    }

    #[test]
    fn gravity_constants_are_sensible() {
        const { assert!(GRAVITY > 0.0) };
        const { assert!(TERMINAL_VELOCITY > 0.0) };
        const { assert!(TERMINAL_VELOCITY > GRAVITY) };
    }

    #[test]
    fn velocity_integration() {
        // Simulate gravity manually (no ECS needed)
        let mut body = PhysicsBody::default();
        let dt = 1.0 / 60.0;

        // One frame of gravity
        body.velocity.y -= GRAVITY * body.gravity_scale * dt;
        assert!(body.velocity.y < 0.0, "Should be falling");

        // Terminal velocity clamp
        body.velocity.y = -1000.0;
        body.velocity.y = body.velocity.y.max(-TERMINAL_VELOCITY);
        assert_eq!(body.velocity.y, -TERMINAL_VELOCITY);
    }

    #[test]
    fn grounded_stops_downward_velocity() {
        let mut body = PhysicsBody::default();
        body.velocity.y = -10.0;
        body.grounded = false;

        // Simulate ground contact
        if body.velocity.y < 0.0 {
            body.velocity.y = 0.0;
        }
        body.grounded = true;

        assert_eq!(body.velocity.y, 0.0);
        assert!(body.grounded);
    }

    #[test]
    fn horizontal_velocity_preserved_on_ground() {
        let mut body = PhysicsBody {
            velocity: Vec3::new(5.0, -10.0, 3.0),
            ..Default::default()
        };

        // Simulate ground collision (only zero Y)
        if body.velocity.y < 0.0 {
            body.velocity.y = 0.0;
        }
        body.grounded = true;

        assert_eq!(body.velocity.x, 5.0);
        assert_eq!(body.velocity.z, 3.0);
        assert_eq!(body.velocity.y, 0.0);
    }
}
