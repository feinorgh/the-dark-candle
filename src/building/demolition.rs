//! Demolition system — part removal and debris spawning.
//!
//! When the player removes a part, it either:
//! - Drops as an item if undamaged (> 50 % HP equivalent).
//! - Breaks into debris voxels if damaged or destroyed by physics.
//!
//! Debris entities are plain physics bodies that inherit the source part's
//! material and the velocity of the collapse.

use bevy::prelude::*;

use super::parts::PlacedPart;
use super::joints::Joint;

// ---------------------------------------------------------------------------
// Debris component
// ---------------------------------------------------------------------------

/// Marker placed on a debris fragment spawned when a part is destroyed.
#[derive(Component, Debug, Clone)]
pub struct DebrisFragment {
    /// Material name inherited from the source part.
    pub material_name: String,
    /// Linear velocity at spawn (m/s).
    pub initial_velocity: Vec3,
}

// ---------------------------------------------------------------------------
// Demolition request
// ---------------------------------------------------------------------------

/// Component added to a `PlacedPart` to request its removal next tick.
///
/// The demolition system reads this, spawns debris / drops item, removes all
/// joints connected to the part, then despawns the part entity.
#[derive(Component)]
pub struct PendingDemolition {
    /// If `true`, try to drop intact item. If `false`, always spawn debris.
    pub drop_as_item: bool,
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

/// Process pending demolitions: break joints, spawn debris, despawn part.
pub fn process_demolitions(
    mut commands: Commands,
    part_query: Query<(Entity, &Transform, &PlacedPart, &PendingDemolition)>,
    joint_query: Query<(Entity, &Joint)>,
) {
    for (part_entity, transform, part, pending) in &part_query {
        // Break all joints connected to this part.
        for (joint_entity, joint) in &joint_query {
            if joint.part_a == part_entity || joint.part_b == part_entity {
                commands.entity(joint_entity).despawn();
            }
        }

        if pending.drop_as_item {
            // Spawn a dropped item entity at the part's location.
            // TODO: wire into the item / inventory system when available.
            commands.spawn((
                Transform::from_translation(transform.translation),
                DroppedPart {
                    material_name: part.material_name.clone(),
                    part_name: part.part_name.clone(),
                },
            ));
        } else {
            // Spawn several debris fragments scattered around the part.
            spawn_debris(&mut commands, transform.translation, &part.material_name, 4);
        }

        commands.entity(part_entity).despawn();
    }
}

/// Spawn `count` debris fragments at `origin` with random scatter velocities.
pub fn spawn_debris(commands: &mut Commands, origin: Vec3, material_name: &str, count: usize) {
    // Deterministic scatter — equally-spaced azimuthal angles.
    for i in 0..count {
        let angle = i as f32 * std::f32::consts::TAU / count as f32;
        let speed = 2.0 + (i as f32 * 0.5); // 2–4 m/s
        let velocity = Vec3::new(angle.cos() * speed, 3.0, angle.sin() * speed);
        let offset = Vec3::new(angle.cos() * 0.3, 0.0, angle.sin() * 0.3);

        commands.spawn((
            Transform::from_translation(origin + offset),
            DebrisFragment {
                material_name: material_name.to_string(),
                initial_velocity: velocity,
            },
        ));
    }
}

// ---------------------------------------------------------------------------
// Dropped item entity (placeholder until inventory is wired)
// ---------------------------------------------------------------------------

/// Temporary component representing a part dropped on the ground as a
/// pick-up item. Full integration with the item/inventory system is TODO.
#[derive(Component, Debug, Clone)]
pub struct DroppedPart {
    pub material_name: String,
    pub part_name: String,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debris_fragment_has_correct_material() {
        let frag = DebrisFragment {
            material_name: "stone".to_string(),
            initial_velocity: Vec3::ZERO,
        };
        assert_eq!(frag.material_name, "stone");
    }

    #[test]
    fn dropped_part_stores_names() {
        let drop = DroppedPart {
            material_name: "wood".to_string(),
            part_name: "beam".to_string(),
        };
        assert_eq!(drop.material_name, "wood");
        assert_eq!(drop.part_name, "beam");
    }
}
