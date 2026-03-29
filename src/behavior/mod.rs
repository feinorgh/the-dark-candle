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
    food_items: Query<(Entity, &Transform, &crate::procgen::items::Item)>,
    time: Res<Time<Fixed>>,
) {
    // Fallback: use fixed-update count as cheap RNG for tiebreaking.
    let tick_rng = (time.elapsed_secs() * 1000.0).fract();

    // Pre-collect readonly data for the perception pass to avoid borrow conflicts.
    let others: Vec<(Entity, Vec3, bool, String)> = actors
        .iter()
        .map(|(e, t, _, _, c, _)| (e, t.translation, c.hostile, c.species.clone()))
        .collect();

    // Pre-collect food item positions for perception.
    let food_sources: Vec<(Vec3, f32)> = food_items
        .iter()
        .filter(|(_, _, item)| item.category == crate::data::ItemCategory::Food)
        .map(|(_, t, item)| (t.translation, item.nutrition))
        .collect();

    for (entity, transform, needs_comp, senses, creature, mut current) in &mut actors {
        let pos = transform.translation;
        let mut nearest_threat: Option<([i32; 3], f32)> = None;
        let mut nearest_ally: Option<([i32; 3], f32)> = None;
        let mut nearest_prey: Option<([i32; 3], f32)> = None;
        let mut nearest_food: Option<([i32; 3], f32)> = None;

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

        // Herbivores and omnivores can eat food items; carnivores hunt prey instead.
        let can_eat_items = matches!(
            creature.diet,
            crate::data::Diet::Herbivore | crate::data::Diet::Omnivore
        );

        if can_eat_items {
            for &(food_pos, _nutrition) in &food_sources {
                let dist = pos.distance(food_pos);
                if dist > senses.sight_range {
                    continue;
                }
                if nearest_food.is_none() || dist < nearest_food.unwrap().1 {
                    let food_pos_i = [food_pos.x as i32, food_pos.y as i32, food_pos.z as i32];
                    nearest_food = Some((food_pos_i, dist));
                }
            }
        }

        let threat_level = nearest_threat
            .map(|(_, d)| (1.0 - d / senses.sight_range).max(0.0))
            .unwrap_or(0.0);

        let ctx = utility::ActionContext {
            nearest_food: nearest_food.map(|(p, _)| p),
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

/// Execute the currently selected action, producing movement and social events.
///
/// When a creature eats, the food entity is despawned and `Metabolism::feed()`
/// restores energy. Item removal from `ChunkItems` tracking happens lazily when
/// the chunk itself unloads.
fn execute_action_system(
    mut commands: Commands,
    mut query: Query<(
        Entity,
        &CurrentAction,
        &Transform,
        &Creature,
        &mut needs::Needs,
        &mut crate::physics::gravity::PhysicsBody,
        &mut crate::biology::metabolism::Metabolism,
    )>,
    food_query: Query<(Entity, &Transform, &crate::procgen::items::Item)>,
    time: Res<Time<Fixed>>,
    mut social_events: MessageWriter<crate::social::SocialActionMessage>,
) {
    // Tick counter mixed with entity bits gives per-creature-per-tick RNG.
    let tick = (time.elapsed_secs() * 64.0) as u64;

    for (entity, current, transform, creature, mut needs_comp, mut body, mut metabolism) in
        &mut query
    {
        let pos = [
            transform.translation.x,
            transform.translation.y,
            transform.translation.z,
        ];

        // Simple xorshift RNG seeded from entity + tick for wander variation.
        let seed = entity.to_bits() ^ tick.wrapping_mul(6364136223846793005);
        let rng_a = (seed & 0xFFFF) as f32 / 32768.0 - 1.0;
        let rng_b = ((seed >> 16) & 0xFFFF) as f32 / 32768.0 - 1.0;

        let output = match &current.0 {
            utility::Action::Idle => behaviors::execute_idle(),
            utility::Action::Wander => behaviors::execute_wander(rng_a, rng_b),
            utility::Action::Eat { target } => behaviors::execute_eat(pos, *target, 1.5),
            utility::Action::Flee { from } => behaviors::execute_flee(pos, *from),
            utility::Action::Sleep => behaviors::execute_sleep(),
            utility::Action::Socialize { target } => {
                behaviors::execute_socialize(pos, *target, 3.0)
            }
            utility::Action::Attack { target } => behaviors::execute_attack(pos, *target, 2.0),
        };

        // --- Food consumption ---
        if output.wants_to_eat {
            let creature_pos = transform.translation;
            let eat_range = 1.5;
            let mut best: Option<(Entity, f32, f32)> = None;

            for (food_entity, food_transform, item) in &food_query {
                if item.category != crate::data::ItemCategory::Food {
                    continue;
                }
                let dist = creature_pos.distance(food_transform.translation);
                if dist <= eat_range && (best.is_none() || dist < best.unwrap().1) {
                    best = Some((food_entity, dist, item.nutrition));
                }
            }

            if let Some((food_entity, _dist, nutrition)) = best {
                metabolism.feed(nutrition);
                needs::satisfy_need(&mut needs_comp.hunger, 0.3);
                commands.entity(food_entity).despawn();
            }
        }

        // Emit social action messages for observable behaviors.
        let actor = crate::social::relationships::CreatureId(entity.to_bits());
        if output.wants_to_attack {
            social_events.write(crate::social::SocialActionMessage(
                crate::social::reputation::SocialAction {
                    actor,
                    target: None,
                    kind: crate::social::reputation::ActionKind::Attack,
                },
            ));
        }
        if matches!(current.0, utility::Action::Flee { .. }) {
            social_events.write(crate::social::SocialActionMessage(
                crate::social::reputation::SocialAction {
                    actor,
                    target: None,
                    kind: crate::social::reputation::ActionKind::Flee,
                },
            ));
        }

        // Set horizontal velocity from movement intent; preserve vertical
        // velocity so gravity continues to work correctly.
        if output.movement.speed_multiplier > 0.0 {
            let speed = creature.speed * output.movement.speed_multiplier;
            let dir = output.movement.direction;
            body.velocity.x = dir[0] * speed;
            body.velocity.z = dir[2] * speed;
        } else {
            // No movement intent — apply ground friction (stop sliding).
            body.velocity.x = 0.0;
            body.velocity.z = 0.0;
        }

        // Satisfy rest if sleeping.
        if output.is_sleeping {
            needs::satisfy_need(&mut needs_comp.rest, 0.01);
        }
    }
}

/// `NeedsConfig` implements `Resource` for ECS injection.
impl Resource for needs::NeedsConfig {}
