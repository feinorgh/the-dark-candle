// Basic behavior executors: translate chosen actions into concrete movement/state changes.
//
// Each behavior takes the creature's current state and the chosen action, and
// produces a movement intent (direction + speed) or state modification. The
// actual movement is applied by physics systems — behaviors only set intent.

#![allow(dead_code)]

/// Movement intent produced by behavior execution.
#[derive(Debug, Clone, PartialEq)]
pub struct MovementIntent {
    /// Desired direction (normalized or zero).
    pub direction: [f32; 3],
    /// Desired speed multiplier (0.0 = stop, 1.0 = normal, >1.0 = sprint).
    pub speed_multiplier: f32,
}

impl Default for MovementIntent {
    fn default() -> Self {
        Self {
            direction: [0.0, 0.0, 0.0],
            speed_multiplier: 0.0,
        }
    }
}

/// State changes produced by behavior execution.
#[derive(Debug, Clone, Default)]
pub struct BehaviorOutput {
    pub movement: MovementIntent,
    /// Whether the creature wants to eat (consume food at target).
    pub wants_to_eat: bool,
    /// Whether the creature wants to attack its target.
    pub wants_to_attack: bool,
    /// Whether the creature is sleeping this tick.
    pub is_sleeping: bool,
}

/// Execute idle: do nothing.
pub fn execute_idle() -> BehaviorOutput {
    BehaviorOutput::default()
}

/// Execute wander: pick a random direction based on RNG seed.
/// `rng_x`, `rng_z`: random values in [-1.0, 1.0] determining wander direction.
pub fn execute_wander(rng_x: f32, rng_z: f32) -> BehaviorOutput {
    let len = (rng_x * rng_x + rng_z * rng_z).sqrt();
    let (dx, dz) = if len > 0.001 {
        (rng_x / len, rng_z / len)
    } else {
        (1.0, 0.0) // default direction if RNG gives near-zero
    };

    BehaviorOutput {
        movement: MovementIntent {
            direction: [dx, 0.0, dz],
            speed_multiplier: 0.5, // wander is slower than running
        },
        ..Default::default()
    }
}

/// Execute eat: move toward food target, signal eating when close enough.
/// `pos`: creature's current position.
/// `target`: food position.
/// `eat_range`: distance within which eating can occur.
pub fn execute_eat(pos: [f32; 3], target: [i32; 3], eat_range: f32) -> BehaviorOutput {
    let dx = target[0] as f32 - pos[0];
    let dy = target[1] as f32 - pos[1];
    let dz = target[2] as f32 - pos[2];
    let dist = (dx * dx + dy * dy + dz * dz).sqrt();

    if dist <= eat_range {
        BehaviorOutput {
            movement: MovementIntent::default(),
            wants_to_eat: true,
            ..Default::default()
        }
    } else {
        let inv = 1.0 / dist;
        BehaviorOutput {
            movement: MovementIntent {
                direction: [dx * inv, dy * inv, dz * inv],
                speed_multiplier: 0.8,
            },
            ..Default::default()
        }
    }
}

/// Execute flee: move directly away from the threat.
/// `pos`: creature's current position.
/// `threat`: threat position.
pub fn execute_flee(pos: [f32; 3], threat: [i32; 3]) -> BehaviorOutput {
    let dx = pos[0] - threat[0] as f32;
    let dy = pos[1] - threat[1] as f32;
    let dz = pos[2] - threat[2] as f32;
    let dist = (dx * dx + dy * dy + dz * dz).sqrt();

    if dist < 0.001 {
        // On top of threat — pick arbitrary escape direction
        return BehaviorOutput {
            movement: MovementIntent {
                direction: [1.0, 0.0, 0.0],
                speed_multiplier: 1.5,
            },
            ..Default::default()
        };
    }

    let inv = 1.0 / dist;
    BehaviorOutput {
        movement: MovementIntent {
            direction: [dx * inv, dy * inv, dz * inv],
            speed_multiplier: 1.5, // sprint when fleeing
        },
        ..Default::default()
    }
}

/// Execute sleep: stop moving and rest.
pub fn execute_sleep() -> BehaviorOutput {
    BehaviorOutput {
        is_sleeping: true,
        ..Default::default()
    }
}

/// Execute socialize: move toward ally at moderate speed.
/// `pos`: creature's current position.
/// `ally`: ally position.
/// `social_range`: distance at which socializing is satisfied.
pub fn execute_socialize(pos: [f32; 3], ally: [i32; 3], social_range: f32) -> BehaviorOutput {
    let dx = ally[0] as f32 - pos[0];
    let dy = ally[1] as f32 - pos[1];
    let dz = ally[2] as f32 - pos[2];
    let dist = (dx * dx + dy * dy + dz * dz).sqrt();

    if dist <= social_range {
        // Close enough, just stand nearby
        BehaviorOutput::default()
    } else {
        let inv = 1.0 / dist;
        BehaviorOutput {
            movement: MovementIntent {
                direction: [dx * inv, dy * inv, dz * inv],
                speed_multiplier: 0.6,
            },
            ..Default::default()
        }
    }
}

/// Execute attack: move toward target and signal attack when in range.
/// `pos`: creature's current position.
/// `target`: target position.
/// `attack_range`: distance within which attack can land.
pub fn execute_attack(pos: [f32; 3], target: [i32; 3], attack_range: f32) -> BehaviorOutput {
    let dx = target[0] as f32 - pos[0];
    let dy = target[1] as f32 - pos[1];
    let dz = target[2] as f32 - pos[2];
    let dist = (dx * dx + dy * dy + dz * dz).sqrt();

    if dist <= attack_range {
        BehaviorOutput {
            movement: MovementIntent::default(),
            wants_to_attack: true,
            ..Default::default()
        }
    } else {
        let inv = 1.0 / dist;
        BehaviorOutput {
            movement: MovementIntent {
                direction: [dx * inv, dy * inv, dz * inv],
                speed_multiplier: 1.2, // charge toward target
            },
            ..Default::default()
        }
    }
}

/// Execute regroup: move toward a group rally point.
///
/// Mechanically similar to `execute_socialize` but with distinct semantics —
/// the target is a coordinated rally position selected by the social
/// group-planning system, not an ally to socialise with.
///
/// `pos`: creature's current position.
/// `rally`: rally point in voxel coordinates.
/// `stop_range`: distance at which the creature considers itself "regrouped".
pub fn execute_regroup(pos: [f32; 3], rally: [i32; 3], stop_range: f32) -> BehaviorOutput {
    let dx = rally[0] as f32 - pos[0];
    let dy = rally[1] as f32 - pos[1];
    let dz = rally[2] as f32 - pos[2];
    let dist = (dx * dx + dy * dy + dz * dz).sqrt();

    if dist <= stop_range {
        BehaviorOutput::default()
    } else {
        let inv = 1.0 / dist;
        BehaviorOutput {
            movement: MovementIntent {
                direction: [dx * inv, dy * inv, dz * inv],
                // Slightly faster than wander, slower than flee — coordinated
                // movement, not panic.
                speed_multiplier: 0.9,
            },
            ..Default::default()
        }
    }
}

/// Normalize a direction vector in-place.
fn normalize(d: &mut [f32; 3]) {
    let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    if len > 0.001 {
        d[0] /= len;
        d[1] /= len;
        d[2] /= len;
    }
}

/// Calculate distance between two 3D points.
pub fn distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idle_produces_no_movement() {
        let out = execute_idle();
        assert_eq!(out.movement.speed_multiplier, 0.0);
        assert!(!out.wants_to_eat);
        assert!(!out.wants_to_attack);
        assert!(!out.is_sleeping);
    }

    #[test]
    fn wander_produces_normalized_direction() {
        let out = execute_wander(0.7, -0.3);
        let d = out.movement.direction;
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        assert!((len - 1.0).abs() < 0.01);
        assert!(out.movement.speed_multiplier > 0.0);
        assert!(out.movement.speed_multiplier < 1.0); // slower than sprint
    }

    #[test]
    fn eat_when_close_to_food() {
        let out = execute_eat([5.0, 1.0, 5.0], [5, 1, 5], 2.0);
        assert!(out.wants_to_eat);
        assert_eq!(out.movement.speed_multiplier, 0.0);
    }

    #[test]
    fn move_toward_food_when_far() {
        let out = execute_eat([0.0, 1.0, 0.0], [10, 1, 0], 2.0);
        assert!(!out.wants_to_eat);
        assert!(out.movement.direction[0] > 0.5); // moving toward food (positive x)
        assert!(out.movement.speed_multiplier > 0.0);
    }

    #[test]
    fn flee_moves_away_from_threat() {
        let out = execute_flee([5.0, 1.0, 5.0], [3, 1, 5]);
        assert!(out.movement.direction[0] > 0.5); // away from threat (positive x)
        assert!(out.movement.speed_multiplier >= 1.5); // sprinting
    }

    #[test]
    fn flee_on_top_of_threat_escapes() {
        let out = execute_flee([3.0, 1.0, 5.0], [3, 1, 5]);
        // Should still have movement (not stuck)
        assert!(out.movement.speed_multiplier > 0.0);
    }

    #[test]
    fn sleep_stops_movement() {
        let out = execute_sleep();
        assert!(out.is_sleeping);
        assert_eq!(out.movement.speed_multiplier, 0.0);
    }

    #[test]
    fn socialize_stops_when_close() {
        let out = execute_socialize([5.0, 1.0, 5.0], [5, 1, 6], 3.0);
        assert_eq!(out.movement.speed_multiplier, 0.0);
    }

    #[test]
    fn socialize_moves_toward_distant_ally() {
        let out = execute_socialize([0.0, 1.0, 0.0], [10, 1, 0], 2.0);
        assert!(out.movement.direction[0] > 0.5);
        assert!(out.movement.speed_multiplier > 0.0);
    }

    #[test]
    fn attack_when_in_range() {
        let out = execute_attack([5.0, 1.0, 5.0], [5, 1, 5], 2.0);
        assert!(out.wants_to_attack);
        assert_eq!(out.movement.speed_multiplier, 0.0);
    }

    #[test]
    fn attack_charges_when_far() {
        let out = execute_attack([0.0, 1.0, 0.0], [10, 1, 0], 2.0);
        assert!(!out.wants_to_attack);
        assert!(out.movement.speed_multiplier >= 1.0);
    }

    #[test]
    fn distance_calculation() {
        assert!((distance([0.0, 0.0, 0.0], [3.0, 4.0, 0.0]) - 5.0).abs() < 0.001);
        assert_eq!(distance([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 0.0);
    }
}
