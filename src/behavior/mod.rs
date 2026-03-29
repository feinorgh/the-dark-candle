pub mod behaviors;
pub mod needs;
pub mod pathfinding;
pub mod perception;
pub mod utility;

use bevy::prelude::*;

use crate::biology::metabolism::Metabolism;
use crate::procgen::creatures::Creature;

/// System set for behavior systems running on `FixedUpdate`.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct BehaviorSet;

/// The creature's currently selected action (updated each tick by utility AI).
#[derive(Component, Debug, Clone)]
pub struct CurrentAction(pub utility::Action);

impl Default for CurrentAction {
    fn default() -> Self {
        Self(utility::Action::Idle)
    }
}

pub struct BehaviorPlugin;

impl Plugin for BehaviorPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<needs::NeedsConfig>().add_systems(
            FixedUpdate,
            (
                tick_needs_system,
                perceive_and_select_action.after(tick_needs_system),
                execute_action_system.after(perceive_and_select_action),
            )
                .in_set(BehaviorSet)
                .after(crate::biology::BiologySet),
        );
    }
}

/// Tick passive needs and sync hunger from metabolism.
fn tick_needs_system(
    mut query: Query<(&mut needs::Needs, &Metabolism), With<Creature>>,
    config: Res<needs::NeedsConfig>,
) {
    for (mut n, meta) in &mut query {
        needs::tick_needs(&mut n, &config);
        needs::update_hunger(&mut n, meta.energy_fraction());
    }
}

/// Build perception context and select best action via utility AI.
///
/// Uses simple distance checks between creatures. Full line-of-sight through
/// the voxel grid is deferred until a VoxelGrid adapter is implemented.
// TODO: implement VoxelGrid adapter for LOS-based perception (tick-stagger for perf).
fn perceive_and_select_action(
    mut actors: Query<(
        Entity,
        &Transform,
        &needs::Needs,
        &perception::Senses,
        &Creature,
        &mut CurrentAction,
    )>,
    time: Res<Time<Fixed>>,
) {
    // Fallback: use fixed-update count as cheap RNG for tiebreaking.
    let tick_rng = (time.elapsed_secs() * 1000.0).fract();

    // Pre-collect readonly data for the perception pass to avoid borrow conflicts.
    let others: Vec<(Entity, Vec3, bool, String)> = actors
        .iter()
        .map(|(e, t, _, _, c, _)| (e, t.translation, c.hostile, c.species.clone()))
        .collect();

    for (entity, transform, needs_comp, senses, creature, mut current) in &mut actors {
        let pos = transform.translation;
        let mut nearest_threat: Option<([i32; 3], f32)> = None;
        let mut nearest_ally: Option<([i32; 3], f32)> = None;
        let mut nearest_prey: Option<([i32; 3], f32)> = None;

        for &(other_entity, other_pos, other_hostile, ref other_species) in &others {
            if other_entity == entity {
                continue;
            }
            let dist = pos.distance(other_pos);
            if dist > senses.sight_range {
                continue;
            }

            let other_pos_i = [other_pos.x as i32, other_pos.y as i32, other_pos.z as i32];

            if other_hostile
                && !creature.hostile
                && (nearest_threat.is_none() || dist < nearest_threat.unwrap().1)
            {
                nearest_threat = Some((other_pos_i, dist));
            }

            if *other_species == creature.species
                && (nearest_ally.is_none() || dist < nearest_ally.unwrap().1)
            {
                nearest_ally = Some((other_pos_i, dist));
            }

            if creature.hostile
                && !other_hostile
                && (nearest_prey.is_none() || dist < nearest_prey.unwrap().1)
            {
                nearest_prey = Some((other_pos_i, dist));
            }
        }

        let threat_level = nearest_threat
            .map(|(_, d)| (1.0 - d / senses.sight_range).max(0.0))
            .unwrap_or(0.0);

        let ctx = utility::ActionContext {
            nearest_food: None, // TODO: food sources not yet implemented
            nearest_threat: nearest_threat.map(|(p, _)| p),
            nearest_ally: nearest_ally.map(|(p, _)| p),
            nearest_prey: nearest_prey.map(|(p, _)| p),
            is_hostile: creature.hostile,
        };

        let mut scoring_needs = needs_comp.clone();
        needs::update_safety(&mut scoring_needs, threat_level);

        let scored = utility::score_actions(&scoring_needs, &ctx);
        current.0 = utility::select_action(&scored, tick_rng);
    }
}

/// Execute the currently selected action, producing movement.
fn execute_action_system(
    mut query: Query<(&CurrentAction, &Transform, &Creature, &mut needs::Needs)>,
) {
    for (current, transform, creature, mut needs_comp) in &mut query {
        let pos = [
            transform.translation.x,
            transform.translation.y,
            transform.translation.z,
        ];

        let output = match &current.0 {
            utility::Action::Idle => behaviors::execute_idle(),
            utility::Action::Wander => {
                // Use position as deterministic wander seed.
                let rx = (pos[0] * 13.7 + pos[2] * 7.3).sin();
                let rz = (pos[2] * 11.3 + pos[0] * 5.7).cos();
                behaviors::execute_wander(rx, rz)
            }
            utility::Action::Eat { target } => behaviors::execute_eat(pos, *target, 1.5),
            utility::Action::Flee { from } => behaviors::execute_flee(pos, *from),
            utility::Action::Sleep => behaviors::execute_sleep(),
            utility::Action::Socialize { target } => {
                behaviors::execute_socialize(pos, *target, 3.0)
            }
            utility::Action::Attack { target } => behaviors::execute_attack(pos, *target, 2.0),
        };

        // Apply movement intent to transform via velocity-like displacement.
        // Full physics integration (rigid body forces) is deferred until
        // creatures are connected to the physics solver.
        // TODO: apply MovementIntent through the physics system instead of
        // direct transform manipulation once entity bodies are implemented.
        if output.movement.speed_multiplier > 0.0 {
            let speed = creature.speed * output.movement.speed_multiplier;
            let dir = output.movement.direction;
            let dt = 1.0 / 64.0; // FixedUpdate default rate
            let displacement = Vec3::new(
                dir[0] * speed * dt,
                dir[1] * speed * dt,
                dir[2] * speed * dt,
            );
            // Movement will be applied by a separate system once physics integration is ready.
            let _ = displacement;
        }

        // Satisfy rest if sleeping.
        if output.is_sleeping {
            needs::satisfy_need(&mut needs_comp.rest, 0.01);
        }
    }
}

/// `NeedsConfig` implements `Resource` for ECS injection.
impl Resource for needs::NeedsConfig {}
