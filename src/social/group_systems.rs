// Group-coordination ECS systems.
//
// The pure planner functions in `super::group_behaviors` (cooperative hunt,
// territory defence, food sharing, rally) are domain logic with no Bevy
// dependency. This module wires them into the ECS by:
//
// 1. Querying creatures with the opt-in [`SocialGroupMember`] component.
// 2. Bucketing them by [`FactionId`] and computing per-group state.
// 3. Calling the relevant `plan_*` function.
// 4. Translating the returned [`GroupAction`](super::group_behaviors::GroupAction)
//    into per-creature [`Action`](crate::behavior::utility::Action) overrides.
//
// Currently only **rally** is wired (smallest viable slice — no perception,
// no pathfinding-against-arbitrary-targets, no inter-faction reasoning).
// The other three planners remain available for future wiring.
//
// Design constraints (from rubber-duck review of an earlier draft):
//
// * **Opt-in only**: no creature gets [`SocialGroupMember`] by default. There
//   is *no* "one auto-faction per species" — that would create world-scale
//   global factions and pull scattered packs toward arbitrary centroids. A
//   creature only joins a social group when something explicitly inserts the
//   component (e.g. spawn-side logic that reads a future "social: pack/herd"
//   tag in the creature RON).
// * **Don't override urgent actions**: the system skips creatures whose
//   current action is `Flee`, `Attack`, `Sleep`, or `Eat`, and skips creatures
//   below health/energy thresholds. Survival behaviour outranks coordination.
// * **Medoid rally point**: the rally target is the *member position closest
//   to the centroid*, not the raw centroid — the centroid can land midair, in
//   terrain, or off-surface; the medoid is by definition somewhere a member
//   currently stands.
// * **Order before `compute_paths`**: the system runs after
//   `perceive_and_select_action` and before `compute_paths`, so the override
//   lands in time for pathfinding to plan against the rally target.

use bevy::prelude::*;
use std::collections::HashMap;

use crate::behavior::CurrentAction;
use crate::behavior::utility::Action;
use crate::biology::health::Health;
use crate::biology::metabolism::Metabolism;

use super::factions::FactionId;
use super::group_behaviors::{CreatureState, GroupAction, plan_rally};
use super::relationships::CreatureId;

/// Opt-in marker placing a creature in a social/coordination group.
///
/// Creatures *without* this component are completely invisible to group
/// planning systems. There is no automatic species-wide assignment.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SocialGroupMember(pub FactionId);

/// Distance (metres) above which the average member-to-centroid spread is
/// considered "scattered" — beyond this, `plan_rally` is asked whether the
/// group should regroup.
const RALLY_SCATTER_THRESHOLD: f32 = 12.0;

/// Health fraction below which a creature is excused from group rally
/// (it should be focused on survival, not coordination).
const RALLY_MIN_HEALTH_FRACTION: f32 = 0.4;

/// Energy (metabolism) fraction below which a creature is excused from
/// group rally (low energy means foraging takes priority).
const RALLY_MIN_ENERGY_FRACTION: f32 = 0.3;

/// Returns `true` if the creature's current action is something more urgent
/// than coordination — survival, combat, or imminent need-satisfaction. The
/// rally system must not overwrite these.
fn is_urgent_action(action: &Action) -> bool {
    matches!(
        action,
        Action::Flee { .. } | Action::Attack { .. } | Action::Sleep | Action::Eat { .. }
    )
}

/// System: per faction, run [`plan_rally`] over members and override eligible
/// participants' [`CurrentAction`] with [`Action::RegroupAt`].
///
/// Designed to run after `perceive_and_select_action` and before
/// `compute_paths`, inside the [`BehaviorSet`](crate::behavior::BehaviorSet).
pub fn plan_group_rally_system(
    mut creatures: Query<(
        Entity,
        &Transform,
        &SocialGroupMember,
        &Health,
        &Metabolism,
        &mut CurrentAction,
    )>,
) {
    // Gather one entry per eligible creature, bucketed by faction.
    //
    // Entries hold (entity, current_action_is_urgent, CreatureState).
    // `is_urgent` exempts the creature from being *overridden*, but it still
    // counts toward the group's geometry / centroid (a fleeing wolf is still
    // part of the pack — just not available to follow a rally order).
    let mut groups: HashMap<FactionId, Vec<(Entity, bool, CreatureState)>> = HashMap::new();

    for (entity, transform, group_member, health, metabolism, action) in &creatures {
        let pos = transform.translation;
        let state = CreatureState {
            id: CreatureId(entity.to_bits()),
            position: [pos.x as i32, pos.y as i32, pos.z as i32],
            energy: metabolism.energy_fraction().clamp(0.0, 1.0),
            health: (health.current / health.max.max(f32::EPSILON)).clamp(0.0, 1.0),
            in_combat: matches!(action.0, Action::Attack { .. }),
        };

        let urgent = is_urgent_action(&action.0)
            || state.health < RALLY_MIN_HEALTH_FRACTION
            || state.energy < RALLY_MIN_ENERGY_FRACTION;

        groups
            .entry(group_member.0)
            .or_default()
            .push((entity, urgent, state));
    }

    for (_faction, members) in groups.iter() {
        if members.len() < 2 {
            // Rally is meaningless for a single creature.
            continue;
        }

        // Centroid of all members (urgent ones included — they're still part
        // of the group's geometry).
        let n = members.len() as f32;
        let mut cx = 0.0_f32;
        let mut cy = 0.0_f32;
        let mut cz = 0.0_f32;
        for (_, _, s) in members {
            cx += s.position[0] as f32;
            cy += s.position[1] as f32;
            cz += s.position[2] as f32;
        }
        let centroid = [cx / n, cy / n, cz / n];

        // Medoid: pick the member position closest to the centroid. This
        // guarantees the rally point sits where a creature currently stands —
        // never midair, inside a wall, or below the planet surface.
        let medoid_pos = members
            .iter()
            .map(|(_, _, s)| s.position)
            .min_by(|a, b| {
                let da = (a[0] as f32 - centroid[0]).powi(2)
                    + (a[1] as f32 - centroid[1]).powi(2)
                    + (a[2] as f32 - centroid[2]).powi(2);
                let db = (b[0] as f32 - centroid[0]).powi(2)
                    + (b[1] as f32 - centroid[1]).powi(2)
                    + (b[2] as f32 - centroid[2]).powi(2);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("members.len() >= 2 guarantees at least one entry");

        let states: Vec<CreatureState> = members.iter().map(|(_, _, s)| s.clone()).collect();
        let Some(GroupAction::Rally {
            position,
            participants,
        }) = plan_rally(medoid_pos, &states, RALLY_SCATTER_THRESHOLD)
        else {
            continue;
        };

        // Build a CreatureId → (Entity, is_urgent) lookup once for this group.
        let by_id: HashMap<CreatureId, (Entity, bool)> = members
            .iter()
            .map(|(e, urgent, s)| (s.id, (*e, *urgent)))
            .collect();

        for participant in &participants {
            let Some((entity, urgent)) = by_id.get(participant).copied() else {
                continue;
            };
            if urgent {
                // Survival/combat outranks coordination — leave the urgent
                // action in place this tick.
                continue;
            }
            if let Ok((_, _, _, _, _, mut current)) = creatures.get_mut(entity) {
                current.0 = Action::RegroupAt { target: position };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::BodySize;

    fn make_app() -> App {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.add_systems(Update, plan_group_rally_system);
        app
    }

    fn spawn_member(app: &mut App, x: f32, z: f32, faction: FactionId) -> Entity {
        app.world_mut()
            .spawn((
                Transform::from_xyz(x, 0.0, z),
                SocialGroupMember(faction),
                Health::new(100.0),
                Metabolism::for_body_size(BodySize::Medium),
                CurrentAction::default(),
            ))
            .id()
    }

    #[test]
    fn scattered_group_members_get_regroup_action() {
        let mut app = make_app();
        let f = FactionId(7);
        // Four members spread across ~30m diagonal — well above the 12m
        // scatter threshold, so rally must trigger.
        let e1 = spawn_member(&mut app, -15.0, -15.0, f);
        let e2 = spawn_member(&mut app, 15.0, 15.0, f);
        let e3 = spawn_member(&mut app, -15.0, 15.0, f);
        let e4 = spawn_member(&mut app, 15.0, -15.0, f);

        app.update();

        for e in [e1, e2, e3, e4] {
            let action = app.world().entity(e).get::<CurrentAction>().unwrap();
            assert!(
                matches!(action.0, Action::RegroupAt { .. }),
                "scattered member {e:?} should have been issued RegroupAt, got {:?}",
                action.0
            );
        }
    }

    #[test]
    fn tightly_grouped_members_are_not_rallied() {
        let mut app = make_app();
        let f = FactionId(1);
        // All within a 4m radius — well under the scatter threshold.
        let e1 = spawn_member(&mut app, 0.0, 0.0, f);
        let e2 = spawn_member(&mut app, 1.0, 1.0, f);
        let e3 = spawn_member(&mut app, -1.0, 1.0, f);

        app.update();

        for e in [e1, e2, e3] {
            let action = app.world().entity(e).get::<CurrentAction>().unwrap();
            assert!(
                !matches!(action.0, Action::RegroupAt { .. }),
                "tightly-grouped member {e:?} should NOT be rallied, got {:?}",
                action.0
            );
        }
    }

    #[test]
    fn urgent_actions_are_preserved() {
        let mut app = make_app();
        let f = FactionId(2);
        // Three scattered members — but one is fleeing and must keep that
        // action; the other two should still rally.
        let fleeing = app
            .world_mut()
            .spawn((
                Transform::from_xyz(-15.0, 0.0, -15.0),
                SocialGroupMember(f),
                Health::new(100.0),
                Metabolism::for_body_size(BodySize::Medium),
                CurrentAction(Action::Flee { from: [99, 0, 99] }),
            ))
            .id();
        let e2 = spawn_member(&mut app, 15.0, 15.0, f);
        let e3 = spawn_member(&mut app, 15.0, -15.0, f);

        app.update();

        let fleeing_action = app.world().entity(fleeing).get::<CurrentAction>().unwrap();
        assert!(
            matches!(fleeing_action.0, Action::Flee { .. }),
            "fleeing creature must keep Flee action, got {:?}",
            fleeing_action.0
        );

        for e in [e2, e3] {
            let action = app.world().entity(e).get::<CurrentAction>().unwrap();
            assert!(
                matches!(action.0, Action::RegroupAt { .. }),
                "non-urgent member {e:?} should be rallied, got {:?}",
                action.0
            );
        }
    }

    #[test]
    fn separate_factions_do_not_mix() {
        let mut app = make_app();
        let f_a = FactionId(10);
        let f_b = FactionId(20);
        // Two scattered A members → should rally to a point on the A side.
        let a1 = spawn_member(&mut app, -15.0, 0.0, f_a);
        let a2 = spawn_member(&mut app, 15.0, 0.0, f_a);
        // Two scattered B members on the opposite side.
        let b1 = spawn_member(&mut app, 0.0, -15.0, f_b);
        let b2 = spawn_member(&mut app, 0.0, 15.0, f_b);

        app.update();

        let a1_target = match app.world().entity(a1).get::<CurrentAction>().unwrap().0 {
            Action::RegroupAt { target } => target,
            ref other => panic!("a1 must be rallied, got {other:?}"),
        };
        let _ = app.world().entity(a2).get::<CurrentAction>().unwrap();
        let b1_target = match app.world().entity(b1).get::<CurrentAction>().unwrap().0 {
            Action::RegroupAt { target } => target,
            ref other => panic!("b1 must be rallied, got {other:?}"),
        };
        let _ = app.world().entity(b2).get::<CurrentAction>().unwrap();

        // A's medoid lies on the x-axis (z≈0); B's lies on the z-axis (x≈0).
        // The two rally points must be different.
        assert_ne!(
            a1_target, b1_target,
            "factions must rally to independent points"
        );
    }
}
