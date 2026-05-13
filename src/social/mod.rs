pub mod factions;
pub mod group_behaviors;
pub mod group_systems;
pub mod relationships;
pub mod reputation;

use bevy::prelude::*;

use crate::behavior::BehaviorSet;
use crate::procgen::creatures::Creature;

use relationships::{CreatureId, Relationships};
use reputation::SocialAction;

/// System set for social systems running on `FixedUpdate`.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct SocialSet;

/// Message emitted when a creature performs an observable social action.
#[derive(Message, Debug, Clone)]
pub struct SocialActionMessage(pub SocialAction);

/// Tick counter for periodic social maintenance (avoids float comparison).
#[derive(Resource, Default)]
struct SocialTickCounter(u32);

const DECAY_INTERVAL_TICKS: u32 = 64;

pub struct SocialPlugin;

impl Plugin for SocialPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<factions::FactionRegistry>()
            .init_resource::<SocialTickCounter>()
            .add_message::<SocialActionMessage>()
            .add_systems(
                FixedUpdate,
                (
                    relationship_decay_system,
                    witness_social_actions.after(relationship_decay_system),
                )
                    .in_set(SocialSet)
                    .after(BehaviorSet),
            )
            // Group-coordination override: must run after action selection
            // and before pathfinding so the rally target reaches `compute_paths`
            // the same tick. Lives in `BehaviorSet`, not `SocialSet`, because
            // it mutates `CurrentAction` (a behavior-layer component) and
            // `SocialSet` runs *after* `BehaviorSet`.
            .add_systems(
                FixedUpdate,
                group_systems::plan_group_rally_system
                    .after(crate::behavior::perceive_and_select_action)
                    .before(crate::behavior::compute_paths)
                    .in_set(BehaviorSet),
            );
    }
}

/// Periodically decay all creature relationships so memories fade over time.
fn relationship_decay_system(
    mut query: Query<&mut Relationships, With<Creature>>,
    mut counter: ResMut<SocialTickCounter>,
) {
    counter.0 += 1;
    if !counter.0.is_multiple_of(DECAY_INTERVAL_TICKS) {
        return;
    }
    for mut rels in &mut query {
        relationships::decay_relationships(&mut rels, 0.01);
        relationships::prune_forgotten(&mut rels);
    }
}

/// Process social action messages: nearby creatures witness the action and
/// update their relationships accordingly.
fn witness_social_actions(
    mut actions: MessageReader<SocialActionMessage>,
    mut creatures: Query<(Entity, &Transform, &mut Relationships), With<Creature>>,
    faction_registry: Res<factions::FactionRegistry>,
) {
    for action_msg in actions.read() {
        let action = &action_msg.0;

        // Find actor's position by looking up the entity.
        // CreatureId stores entity bits, so we can reconstruct it.
        let actor_pos = creatures
            .iter()
            .find(|(e, _, _)| CreatureId(e.to_bits()) == action.actor)
            .map(|(_, t, _)| t.translation);

        let Some(actor_pos) = actor_pos else {
            continue;
        };

        // Witness range: creatures within 30m can observe the action.
        const WITNESS_RANGE: f32 = 30.0;

        // Collect witness entity IDs first to avoid borrow conflicts.
        let witnesses: Vec<Entity> = creatures
            .iter()
            .filter(|(e, t, _)| {
                CreatureId(e.to_bits()) != action.actor
                    && t.translation.distance(actor_pos) <= WITNESS_RANGE
            })
            .map(|(e, _, _)| e)
            .collect();

        // Apply witness effects.
        for witness_entity in &witnesses {
            if let Ok((_, _, mut rels)) = creatures.get_mut(*witness_entity) {
                let observer_id = CreatureId(witness_entity.to_bits());
                reputation::witness_action(observer_id, &mut rels, action);
            }
        }

        // Faction propagation: members of the actor's faction who are out of
        // direct witness range still learn about the action (secondhand).
        let actor_id = action.actor;
        if let Some(faction_id) = faction_registry.creature_faction(actor_id)
            && let Some(faction) = faction_registry.factions.get(&faction_id)
        {
            let faction_members: Vec<Entity> = creatures
                .iter()
                .filter(|(e, t, _)| {
                    let cid = CreatureId(e.to_bits());
                    faction.is_member(cid)
                        && cid != action.actor
                        && t.translation.distance(actor_pos) > WITNESS_RANGE
                })
                .map(|(e, _, _)| e)
                .collect();

            for member_entity in &faction_members {
                if let Ok((_, _, mut rels)) = creatures.get_mut(*member_entity) {
                    // Halved effect for secondhand info (mirrors propagate_to_faction logic).
                    let observer_likes_target = action
                        .target
                        .and_then(|t| rels.get(t))
                        .map(|r| r.is_friendly())
                        .unwrap_or(false);

                    let effects = reputation::action_effects(action.kind, observer_likes_target);
                    rels.adjust_trust(action.actor, effects.trust_delta * 0.5);
                    rels.adjust_hostility(action.actor, effects.hostility_delta * 0.5);
                    rels.increase_familiarity(action.actor, 0.05);
                }
            }
        }
    }
}
