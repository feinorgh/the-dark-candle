pub mod behaviors;
pub mod needs;
pub mod pathfinding;
pub mod perception;
pub mod utility;

use bevy::prelude::*;

use crate::biology::metabolism::Metabolism;
use crate::procgen::creatures::Creature;
use crate::world::chunk::{Chunk, ChunkCoord};

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

/// Cached A* path for a creature.  Recomputed when the action target changes
/// by more than `REPLAN_THRESHOLD` voxels or the path is fully consumed.
#[derive(Component, Debug, Clone, Default)]
pub struct CreaturePath {
    /// Remaining waypoints (next waypoint is index 0).
    pub waypoints: Vec<[i32; 3]>,
    /// The target position this path was computed toward.
    pub target: Option<[i32; 3]>,
}

/// How far a target must move before we recompute the path.
const REPLAN_THRESHOLD: f32 = 3.0;
/// Max creatures to re-path per tick (budget to avoid frame spikes).
const MAX_PATHS_PER_TICK: usize = 8;

pub struct BehaviorPlugin;

impl Plugin for BehaviorPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<needs::NeedsConfig>().add_systems(
            FixedUpdate,
            (
                tick_needs_system,
                perceive_and_select_action.after(tick_needs_system),
                compute_paths
                    .after(perceive_and_select_action)
                    .before(execute_action_system),
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
pub(crate) fn perceive_and_select_action(
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

/// Extract the target position from an [`Action`], if any.
fn action_target(action: &utility::Action) -> Option<[i32; 3]> {
    match action {
        utility::Action::Eat { target }
        | utility::Action::Socialize { target }
        | utility::Action::Attack { target }
        | utility::Action::RegroupAt { target } => Some(*target),
        utility::Action::Flee { from } => Some(*from),
        _ => None,
    }
}

fn target_distance(a: [i32; 3], b: [i32; 3]) -> f32 {
    let dx = (a[0] - b[0]) as f32;
    let dy = (a[1] - b[1]) as f32;
    let dz = (a[2] - b[2]) as f32;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Compute or update A* paths for creatures whose action targets have changed.
///
/// Builds a [`WorldVoxelGrid`](pathfinding::WorldVoxelGrid) once per tick from
/// loaded chunks, then runs [`find_path`](pathfinding::find_path) for up to
/// [`MAX_PATHS_PER_TICK`] creatures that need re-planning.
pub(crate) fn compute_paths(
    mut creatures: Query<(&Transform, &CurrentAction, &Creature, &mut CreaturePath)>,
    chunks: Query<(&Chunk, &ChunkCoord)>,
) {
    // Build world grid from loaded chunks.
    let mut grid = pathfinding::WorldVoxelGrid::new();
    for (chunk, coord) in &chunks {
        grid.insert(coord, chunk);
    }

    let mut budget = MAX_PATHS_PER_TICK;

    for (transform, action, creature, mut path) in &mut creatures {
        let Some(target) = action_target(&action.0) else {
            // No target — clear cached path.
            path.waypoints.clear();
            path.target = None;
            continue;
        };

        // Check if we already have a valid path to (roughly) this target.
        let needs_replan = match path.target {
            Some(prev) => {
                target_distance(prev, target) > REPLAN_THRESHOLD || path.waypoints.is_empty()
            }
            None => true,
        };

        if !needs_replan {
            continue;
        }

        if budget == 0 {
            continue;
        }
        budget -= 1;

        let pos = transform.translation;
        let start = [pos.x as i32, pos.y as i32, pos.z as i32];

        // For Flee, pathfind *away* — pick a point opposite to the threat.
        let goal = if matches!(action.0, utility::Action::Flee { .. }) {
            let dx = start[0] - target[0];
            let dz = start[2] - target[2];
            let flee_dist = 12;
            [
                start[0] + dx.signum() * flee_dist,
                start[1],
                start[2] + dz.signum() * flee_dist,
            ]
        } else {
            target
        };

        let config = pathfinding::PathConfig {
            max_jump: 1,
            max_drop: 3,
            can_swim: creature.diet == crate::data::Diet::Omnivore,
            max_nodes: 1000, // Reduced budget per creature for responsiveness.
            ..Default::default()
        };

        if let Some(found) = pathfinding::find_path(&grid, start, goal, &config) {
            path.waypoints = found.waypoints;
            path.target = Some(target);
        } else {
            // No path found — clear so we fall back to direct-line movement.
            path.waypoints.clear();
            path.target = Some(target);
        }
    }
}

/// Execute the currently selected action, producing movement and social events.
///
/// When a creature eats, the food entity is despawned and `Metabolism::feed()`
/// restores energy. Item removal from `ChunkItems` tracking happens lazily when
/// the chunk itself unloads.
#[allow(clippy::type_complexity)]
pub(crate) fn execute_action_system(
    mut commands: Commands,
    mut query: Query<(
        Entity,
        &CurrentAction,
        &Transform,
        &Creature,
        &mut needs::Needs,
        &mut crate::physics::gravity::PhysicsBody,
        &mut crate::biology::metabolism::Metabolism,
        &mut CreaturePath,
    )>,
    food_query: Query<(Entity, &Transform, &crate::procgen::items::Item)>,
    time: Res<Time<Fixed>>,
    mut social_events: MessageWriter<crate::social::SocialActionMessage>,
) {
    // Tick counter mixed with entity bits gives per-creature-per-tick RNG.
    let tick = (time.elapsed_secs() * 64.0) as u64;

    for (
        entity,
        current,
        transform,
        creature,
        mut needs_comp,
        mut body,
        mut metabolism,
        mut path,
    ) in &mut query
    {
        let pos = [
            transform.translation.x,
            transform.translation.y,
            transform.translation.z,
        ];

        // Advance waypoints: pop consumed waypoints that we're already near.
        let waypoint_reach = 1.2_f32;
        while let Some(&wp) = path.waypoints.first() {
            let dx = wp[0] as f32 - pos[0];
            let dy = wp[1] as f32 - pos[1];
            let dz = wp[2] as f32 - pos[2];
            if (dx * dx + dy * dy + dz * dz).sqrt() < waypoint_reach {
                path.waypoints.remove(0);
            } else {
                break;
            }
        }

        // Resolve the effective target: next waypoint if available,
        // otherwise the raw action target (direct-line fallback).
        let effective_target = |raw_target: [i32; 3]| -> [i32; 3] {
            path.waypoints.first().copied().unwrap_or(raw_target)
        };

        // Simple xorshift RNG seeded from entity + tick for wander variation.
        let seed = entity.to_bits() ^ tick.wrapping_mul(6364136223846793005);
        let rng_a = (seed & 0xFFFF) as f32 / 32768.0 - 1.0;
        let rng_b = ((seed >> 16) & 0xFFFF) as f32 / 32768.0 - 1.0;

        let output = match &current.0 {
            utility::Action::Idle => behaviors::execute_idle(),
            utility::Action::Wander => behaviors::execute_wander(rng_a, rng_b),
            utility::Action::Eat { target } => {
                behaviors::execute_eat(pos, effective_target(*target), 1.5)
            }
            utility::Action::Flee { from } => behaviors::execute_flee(pos, effective_target(*from)),
            utility::Action::Sleep => behaviors::execute_sleep(),
            utility::Action::Socialize { target } => {
                behaviors::execute_socialize(pos, effective_target(*target), 3.0)
            }
            utility::Action::Attack { target } => {
                behaviors::execute_attack(pos, effective_target(*target), 2.0)
            }
            utility::Action::RegroupAt { target } => {
                behaviors::execute_regroup(pos, effective_target(*target), 2.0)
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::biology::metabolism::Metabolism;
    use crate::data::{BodySize, Diet};
    use crate::physics::gravity::PhysicsBody;
    use crate::procgen::creatures::Creature;
    use crate::social::SocialActionMessage;

    fn make_test_creature(speed: f32) -> Creature {
        Creature {
            species: "test".into(),
            display_name: "Test".into(),
            health: 100.0,
            max_health: 100.0,
            speed,
            attack: 5.0,
            body_size: BodySize::Medium,
            diet: Diet::Herbivore,
            color: [0.5, 0.5, 0.5],
            hostile: false,
            lifespan: None,
            age: 0,
        }
    }

    /// Locks in the behavior → physics wiring: a creature with a `Wander`
    /// action must end the tick with a non-zero horizontal velocity on its
    /// `PhysicsBody`.  Earlier `ai-context.json` notes claimed
    /// `MovementIntent` was not integrated; this test exists to prove
    /// otherwise and prevent regression.
    #[test]
    fn wander_action_drives_physics_body_velocity() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.add_message::<SocialActionMessage>();
        app.init_resource::<needs::NeedsConfig>();
        app.add_systems(Update, execute_action_system);

        let entity = app
            .world_mut()
            .spawn((
                make_test_creature(4.0),
                needs::Needs::default(),
                Metabolism::for_body_size(BodySize::Medium),
                CurrentAction(utility::Action::Wander),
                CreaturePath::default(),
                PhysicsBody::default(),
                Transform::from_xyz(0.0, 64.0, 0.0),
            ))
            .id();

        app.update();

        let body = app
            .world()
            .entity(entity)
            .get::<PhysicsBody>()
            .expect("PhysicsBody must still be present after the tick");
        let horizontal = (body.velocity.x.powi(2) + body.velocity.z.powi(2)).sqrt();
        assert!(
            horizontal > 0.0,
            "Wander must produce non-zero horizontal velocity; got {:?}",
            body.velocity
        );
        // Wander uses speed_multiplier 0.5, so |v_xz| should not exceed
        // creature.speed.  Allows a small slack for rounding.
        assert!(
            horizontal <= 4.0 + 1e-3,
            "Wander velocity must respect creature.speed cap; got {horizontal}"
        );
    }

    /// Idle must produce *zero* horizontal velocity (the friction branch
    /// of `execute_action_system` zeroes x/z).  This guards the inverse
    /// case so a future refactor can't accidentally leak movement.
    #[test]
    fn idle_action_zeroes_horizontal_velocity() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.add_message::<SocialActionMessage>();
        app.init_resource::<needs::NeedsConfig>();
        app.add_systems(Update, execute_action_system);

        let body = PhysicsBody {
            velocity: Vec3::new(2.5, -1.0, -3.0), // pre-existing motion
            ..Default::default()
        };

        let entity = app
            .world_mut()
            .spawn((
                make_test_creature(4.0),
                needs::Needs::default(),
                Metabolism::for_body_size(BodySize::Medium),
                CurrentAction(utility::Action::Idle),
                CreaturePath::default(),
                body,
                Transform::from_xyz(0.0, 64.0, 0.0),
            ))
            .id();

        app.update();

        let body = app.world().entity(entity).get::<PhysicsBody>().unwrap();
        assert_eq!(body.velocity.x, 0.0, "Idle must zero horizontal x velocity");
        assert_eq!(body.velocity.z, 0.0, "Idle must zero horizontal z velocity");
        // Vertical (gravity-axis) velocity must be preserved so gravity
        // continues to integrate normally.
        assert_eq!(
            body.velocity.y, -1.0,
            "Idle must preserve vertical velocity for gravity"
        );
    }
}
