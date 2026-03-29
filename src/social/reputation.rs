// Reputation system: observed actions modify relationship values.
//
// When a creature performs an observable action (helping, attacking, stealing),
// nearby witnesses update their relationship with the actor. Information can
// propagate through social networks — faction members share reputation data.

use super::factions::FactionRegistry;
use super::relationships::{CreatureId, Relationships};

/// An observable social action.
#[derive(Debug, Clone)]
pub struct SocialAction {
    /// Who performed the action.
    pub actor: CreatureId,
    /// Who was affected (if any).
    pub target: Option<CreatureId>,
    /// What happened.
    pub kind: ActionKind,
}

/// Categories of observable social actions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionKind {
    /// Attacked another creature.
    Attack,
    /// Healed or helped another creature.
    Help,
    /// Shared food or resources.
    ShareFood,
    /// Stole from another creature.
    Steal,
    /// Defended another creature from a threat.
    Defend,
    /// Fled from combat (cowardice).
    Flee,
    /// Killed another creature.
    Kill,
}

/// How an action affects the observer's relationships.
#[derive(Debug, Clone)]
pub struct ReputationEffect {
    /// Trust change toward the actor.
    pub trust_delta: f32,
    /// Hostility change toward the actor.
    pub hostility_delta: f32,
}

pub fn action_effects(kind: ActionKind, observer_likes_target: bool) -> ReputationEffect {
    match kind {
        ActionKind::Attack => {
            if observer_likes_target {
                // Observer's friend was attacked → hostile toward actor
                ReputationEffect {
                    trust_delta: -0.3,
                    hostility_delta: 0.4,
                }
            } else {
                // Enemy was attacked → mild approval
                ReputationEffect {
                    trust_delta: 0.1,
                    hostility_delta: -0.1,
                }
            }
        }
        ActionKind::Kill => {
            if observer_likes_target {
                ReputationEffect {
                    trust_delta: -0.6,
                    hostility_delta: 0.7,
                }
            } else {
                ReputationEffect {
                    trust_delta: 0.15,
                    hostility_delta: -0.15,
                }
            }
        }
        ActionKind::Help | ActionKind::Defend => {
            if observer_likes_target {
                ReputationEffect {
                    trust_delta: 0.3,
                    hostility_delta: -0.2,
                }
            } else {
                ReputationEffect {
                    trust_delta: -0.1,
                    hostility_delta: 0.05,
                }
            }
        }
        ActionKind::ShareFood => ReputationEffect {
            trust_delta: 0.2,
            hostility_delta: -0.1,
        },
        ActionKind::Steal => ReputationEffect {
            trust_delta: -0.4,
            hostility_delta: 0.3,
        },
        ActionKind::Flee => ReputationEffect {
            trust_delta: -0.1,
            hostility_delta: 0.0,
        },
    }
}

/// Apply a witnessed action's reputation effects to an observer.
/// `observer_rels`: the observer's relationship map.
/// `action`: the observed action.
pub fn witness_action(
    observer: CreatureId,
    observer_rels: &mut Relationships,
    action: &SocialAction,
) {
    if observer == action.actor {
        return; // can't witness your own action
    }

    let observer_likes_target = action
        .target
        .and_then(|t| observer_rels.get(t))
        .map(|r| r.is_friendly())
        .unwrap_or(false);

    let effects = action_effects(action.kind, observer_likes_target);

    observer_rels.adjust_trust(action.actor, effects.trust_delta);
    observer_rels.adjust_hostility(action.actor, effects.hostility_delta);

    // Increase familiarity with actor (we saw them do something)
    observer_rels.increase_familiarity(action.actor, 0.1);
}

/// Propagate reputation within a faction: all members learn about the action.
/// Returns the number of creatures that received the information.
pub fn propagate_to_faction(
    actor: CreatureId,
    action: &SocialAction,
    faction_registry: &FactionRegistry,
    all_relationships: &mut [(CreatureId, &mut Relationships)],
) -> usize {
    // Find which faction the propagator belongs to
    let Some(faction_id) = faction_registry.creature_faction(actor) else {
        return 0;
    };
    let Some(faction) = faction_registry.factions.get(&faction_id) else {
        return 0;
    };

    let mut count = 0;
    for (creature_id, rels) in all_relationships.iter_mut() {
        if faction.is_member(*creature_id) && *creature_id != action.actor && *creature_id != actor
        {
            // Faction members get a weaker version of the effect (secondhand info)
            let observer_likes_target = action
                .target
                .and_then(|t| rels.get(t))
                .map(|r| r.is_friendly())
                .unwrap_or(false);

            let effects = action_effects(action.kind, observer_likes_target);

            // Halved effect for secondhand info
            rels.adjust_trust(action.actor, effects.trust_delta * 0.5);
            rels.adjust_hostility(action.actor, effects.hostility_delta * 0.5);
            rels.increase_familiarity(action.actor, 0.05);

            count += 1;
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::social::factions::{Faction, FactionId};

    #[test]
    fn witness_attack_on_friend_increases_hostility() {
        let observer = CreatureId(1);
        let attacker = CreatureId(2);
        let victim = CreatureId(3);

        let mut rels = Relationships::default();
        // Observer likes the victim
        rels.adjust_trust(victim, 0.5);
        rels.increase_familiarity(victim, 0.5);

        let action = SocialAction {
            actor: attacker,
            target: Some(victim),
            kind: ActionKind::Attack,
        };

        witness_action(observer, &mut rels, &action);

        let r = rels.get(attacker).unwrap();
        assert!(r.hostility > 0.0);
        assert!(r.trust < 0.0);
    }

    #[test]
    fn witness_attack_on_enemy_mild_approval() {
        let observer = CreatureId(1);
        let attacker = CreatureId(2);
        let victim = CreatureId(3);

        let mut rels = Relationships::default();
        // Observer dislikes the victim
        rels.adjust_hostility(victim, 0.8);
        rels.increase_familiarity(victim, 0.5);

        let action = SocialAction {
            actor: attacker,
            target: Some(victim),
            kind: ActionKind::Attack,
        };

        witness_action(observer, &mut rels, &action);

        let r = rels.get(attacker).unwrap();
        assert!(r.trust > 0.0);
    }

    #[test]
    fn witness_help_increases_trust() {
        let observer = CreatureId(1);
        let helper = CreatureId(2);
        let helped = CreatureId(3);

        let mut rels = Relationships::default();
        rels.adjust_trust(helped, 0.5);
        rels.increase_familiarity(helped, 0.5);

        let action = SocialAction {
            actor: helper,
            target: Some(helped),
            kind: ActionKind::Help,
        };

        witness_action(observer, &mut rels, &action);

        let r = rels.get(helper).unwrap();
        assert!(r.trust > 0.0);
        assert!(r.familiarity > 0.0);
    }

    #[test]
    fn witness_steal_decreases_trust() {
        let observer = CreatureId(1);
        let thief = CreatureId(2);

        let mut rels = Relationships::default();

        let action = SocialAction {
            actor: thief,
            target: None,
            kind: ActionKind::Steal,
        };

        witness_action(observer, &mut rels, &action);

        let r = rels.get(thief).unwrap();
        assert!(r.trust < 0.0);
        assert!(r.hostility > 0.0);
    }

    #[test]
    fn witness_share_food_builds_trust() {
        let observer = CreatureId(1);
        let sharer = CreatureId(2);

        let mut rels = Relationships::default();

        let action = SocialAction {
            actor: sharer,
            target: None,
            kind: ActionKind::ShareFood,
        };

        witness_action(observer, &mut rels, &action);

        let r = rels.get(sharer).unwrap();
        assert!(r.trust > 0.0);
    }

    #[test]
    fn cannot_witness_own_action() {
        let me = CreatureId(1);
        let mut rels = Relationships::default();

        let action = SocialAction {
            actor: me,
            target: None,
            kind: ActionKind::Attack,
        };

        witness_action(me, &mut rels, &action);
        assert!(rels.get(me).is_none());
    }

    #[test]
    fn kill_has_stronger_effect_than_attack() {
        let observer = CreatureId(1);
        let actor = CreatureId(2);
        let victim = CreatureId(3);

        let mut rels_attack = Relationships::default();
        rels_attack.adjust_trust(victim, 0.5);
        rels_attack.increase_familiarity(victim, 0.5);
        witness_action(
            observer,
            &mut rels_attack,
            &SocialAction {
                actor,
                target: Some(victim),
                kind: ActionKind::Attack,
            },
        );

        let mut rels_kill = Relationships::default();
        rels_kill.adjust_trust(victim, 0.5);
        rels_kill.increase_familiarity(victim, 0.5);
        witness_action(
            observer,
            &mut rels_kill,
            &SocialAction {
                actor,
                target: Some(victim),
                kind: ActionKind::Kill,
            },
        );

        let attack_hostility = rels_attack.get(actor).unwrap().hostility;
        let kill_hostility = rels_kill.get(actor).unwrap().hostility;
        assert!(kill_hostility > attack_hostility);
    }

    #[test]
    fn faction_propagation_reaches_members() {
        let mut registry = FactionRegistry::default();
        let mut faction = Faction::new(FactionId(1), "Pack");
        faction.add_member(CreatureId(10));
        faction.add_member(CreatureId(11));
        faction.add_member(CreatureId(12));
        registry.add_faction(faction);

        let action = SocialAction {
            actor: CreatureId(99),
            target: None,
            kind: ActionKind::Steal,
        };

        let mut rels_10 = Relationships::default();
        let mut rels_11 = Relationships::default();
        let mut rels_12 = Relationships::default();

        let mut all = vec![
            (CreatureId(10), &mut rels_10),
            (CreatureId(11), &mut rels_11),
            (CreatureId(12), &mut rels_12),
        ];

        let count = propagate_to_faction(CreatureId(10), &action, &registry, &mut all);
        assert_eq!(count, 2); // 11 and 12, not 99 (the actor)
    }

    #[test]
    fn faction_propagation_is_weaker() {
        let mut registry = FactionRegistry::default();
        let mut faction = Faction::new(FactionId(1), "Pack");
        faction.add_member(CreatureId(10));
        faction.add_member(CreatureId(11));
        registry.add_faction(faction);

        let action = SocialAction {
            actor: CreatureId(99),
            target: None,
            kind: ActionKind::Steal,
        };

        // Direct witness
        let mut direct_rels = Relationships::default();
        witness_action(CreatureId(10), &mut direct_rels, &action);

        // Faction propagation
        let mut indirect_rels = Relationships::default();
        let mut all = vec![(CreatureId(11), &mut indirect_rels)];
        propagate_to_faction(CreatureId(10), &action, &registry, &mut all);

        let direct = direct_rels.get(CreatureId(99)).unwrap();
        let indirect = indirect_rels.get(CreatureId(99)).unwrap();
        assert!(direct.hostility > indirect.hostility);
        assert!(direct.trust.abs() > indirect.trust.abs());
    }
}
