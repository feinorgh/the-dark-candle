// Utility AI: score available actions and select the best one.
//
// Each possible action is scored against the creature's current needs.
// The action with the highest weighted utility score is chosen, with
// a small random tiebreaker to prevent deterministic loops.

use super::needs::Needs;

/// Possible actions a creature can take.
#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    /// Do nothing — rest in place.
    Idle,
    /// Move randomly to satisfy curiosity.
    Wander,
    /// Move toward food and consume it.
    Eat { target: [i32; 3] },
    /// Run away from a threat.
    Flee { from: [i32; 3] },
    /// Find a safe spot and sleep to restore rest.
    Sleep,
    /// Move toward a creature of the same species.
    Socialize { target: [i32; 3] },
    /// Attack a target creature or destructible.
    Attack { target: [i32; 3] },
}

/// An action paired with its computed utility score.
#[derive(Debug, Clone)]
pub struct ScoredAction {
    pub action: Action,
    pub score: f32,
}

/// Context for scoring actions (what's available in the environment).
#[derive(Debug, Clone, Default)]
pub struct ActionContext {
    /// Nearest food source position, if any.
    pub nearest_food: Option<[i32; 3]>,
    /// Nearest threat position, if any.
    pub nearest_threat: Option<[i32; 3]>,
    /// Nearest same-species creature position, if any.
    pub nearest_ally: Option<[i32; 3]>,
    /// Nearest attackable target position, if any (for hostile creatures).
    pub nearest_prey: Option<[i32; 3]>,
    /// Whether the creature is hostile.
    pub is_hostile: bool,
}

/// Score all available actions given the creature's needs and context.
pub fn score_actions(needs: &Needs, ctx: &ActionContext) -> Vec<ScoredAction> {
    let mut scored = Vec::new();

    // Idle: always available, low base score — acts as fallback
    scored.push(ScoredAction {
        action: Action::Idle,
        score: 0.05,
    });

    // Wander: driven by curiosity
    scored.push(ScoredAction {
        action: Action::Wander,
        score: needs.curiosity * 0.6,
    });

    // Eat: driven by hunger, requires food nearby
    if let Some(target) = ctx.nearest_food {
        scored.push(ScoredAction {
            action: Action::Eat { target },
            score: needs.hunger * 1.2, // eating is urgent when hungry
        });
    }

    // Flee: driven by safety need, requires threat nearby
    if let Some(from) = ctx.nearest_threat {
        scored.push(ScoredAction {
            action: Action::Flee { from },
            score: needs.safety * 2.0, // fleeing gets extreme priority when danger is high
        });
    }

    // Sleep: driven by rest need
    scored.push(ScoredAction {
        action: Action::Sleep,
        score: needs.rest * 0.8,
    });

    // Socialize: driven by social need, requires ally nearby
    if let Some(target) = ctx.nearest_ally {
        scored.push(ScoredAction {
            action: Action::Socialize { target },
            score: needs.social * 0.5,
        });
    }

    // Attack: for hostile creatures when prey is available
    if ctx.is_hostile
        && let Some(target) = ctx.nearest_prey
    {
        // Hostile creatures attack when hungry or when prey is near
        let aggression = needs.hunger * 0.8 + 0.2;
        scored.push(ScoredAction {
            action: Action::Attack { target },
            score: aggression,
        });
    }

    scored
}

/// Select the best action. Uses a small random tiebreaker to prevent loops.
/// `rng_value`: a random f32 in [0.0, 1.0) for tiebreaking.
pub fn select_action(scored: &[ScoredAction], rng_value: f32) -> Action {
    if scored.is_empty() {
        return Action::Idle;
    }

    let noise_scale = 0.02;

    scored
        .iter()
        .max_by(|a, b| {
            let sa = a.score + rng_value * noise_scale;
            let sb = b.score + (1.0 - rng_value) * noise_scale;
            sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|s| s.action.clone())
        .unwrap_or(Action::Idle)
}

/// Select the best action without randomness (deterministic, for testing).
pub fn select_action_deterministic(scored: &[ScoredAction]) -> Action {
    scored
        .iter()
        .max_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|s| s.action.clone())
        .unwrap_or(Action::Idle)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hungry_needs() -> Needs {
        Needs {
            hunger: 0.9,
            safety: 0.0,
            rest: 0.1,
            curiosity: 0.2,
            social: 0.1,
        }
    }

    fn scared_needs() -> Needs {
        Needs {
            hunger: 0.3,
            safety: 0.9,
            rest: 0.1,
            curiosity: 0.0,
            social: 0.0,
        }
    }

    #[test]
    fn idle_is_always_available() {
        let needs = Needs::default();
        let ctx = ActionContext::default();
        let scored = score_actions(&needs, &ctx);
        assert!(scored.iter().any(|s| s.action == Action::Idle));
    }

    #[test]
    fn hungry_creature_eats_when_food_available() {
        let needs = hungry_needs();
        let ctx = ActionContext {
            nearest_food: Some([10, 5, 10]),
            ..Default::default()
        };
        let scored = score_actions(&needs, &ctx);
        let best = select_action_deterministic(&scored);
        assert!(matches!(best, Action::Eat { .. }));
    }

    #[test]
    fn scared_creature_flees() {
        let needs = scared_needs();
        let ctx = ActionContext {
            nearest_threat: Some([5, 5, 5]),
            nearest_food: Some([20, 5, 20]),
            ..Default::default()
        };
        let scored = score_actions(&needs, &ctx);
        let best = select_action_deterministic(&scored);
        assert!(matches!(best, Action::Flee { .. }));
    }

    #[test]
    fn flee_outscores_eat_at_high_danger() {
        let needs = Needs {
            hunger: 0.8,
            safety: 0.8,
            ..Default::default()
        };
        let ctx = ActionContext {
            nearest_food: Some([10, 5, 10]),
            nearest_threat: Some([3, 5, 3]),
            ..Default::default()
        };
        let scored = score_actions(&needs, &ctx);
        let best = select_action_deterministic(&scored);
        // Safety=0.8 * 2.0 = 1.6 > Hunger=0.8 * 1.2 = 0.96
        assert!(matches!(best, Action::Flee { .. }));
    }

    #[test]
    fn wander_when_curious_and_nothing_else() {
        let needs = Needs {
            hunger: 0.0,
            safety: 0.0,
            rest: 0.0,
            curiosity: 0.8,
            social: 0.0,
        };
        let ctx = ActionContext::default();
        let scored = score_actions(&needs, &ctx);
        let best = select_action_deterministic(&scored);
        assert_eq!(best, Action::Wander);
    }

    #[test]
    fn sleep_when_exhausted() {
        let needs = Needs {
            hunger: 0.0,
            safety: 0.0,
            rest: 0.9,
            curiosity: 0.0,
            social: 0.0,
        };
        let ctx = ActionContext::default();
        let scored = score_actions(&needs, &ctx);
        let best = select_action_deterministic(&scored);
        assert_eq!(best, Action::Sleep);
    }

    #[test]
    fn hostile_creature_attacks_prey() {
        let needs = Needs {
            hunger: 0.6,
            safety: 0.0,
            rest: 0.0,
            curiosity: 0.0,
            social: 0.0,
        };
        let ctx = ActionContext {
            nearest_prey: Some([8, 5, 8]),
            is_hostile: true,
            ..Default::default()
        };
        let scored = score_actions(&needs, &ctx);
        let best = select_action_deterministic(&scored);
        assert!(matches!(best, Action::Attack { .. }));
    }

    #[test]
    fn non_hostile_creature_does_not_attack() {
        let needs = Needs {
            hunger: 0.9,
            ..Default::default()
        };
        let ctx = ActionContext {
            nearest_prey: Some([8, 5, 8]),
            is_hostile: false,
            ..Default::default()
        };
        let scored = score_actions(&needs, &ctx);
        assert!(
            !scored
                .iter()
                .any(|s| matches!(s.action, Action::Attack { .. }))
        );
    }

    #[test]
    fn empty_scored_returns_idle() {
        let best = select_action(&[], 0.5);
        assert_eq!(best, Action::Idle);
    }

    #[test]
    fn rng_tiebreaker_does_not_change_clear_winner() {
        let scored = vec![
            ScoredAction {
                action: Action::Wander,
                score: 0.1,
            },
            ScoredAction {
                action: Action::Sleep,
                score: 0.9,
            },
        ];
        // Even with extreme rng values, sleep should always win
        for rng in [0.0, 0.5, 0.99] {
            let best = select_action(&scored, rng);
            assert_eq!(best, Action::Sleep);
        }
    }

    #[test]
    fn socialize_when_lonely_and_ally_nearby() {
        let needs = Needs {
            social: 0.9,
            ..Default::default()
        };
        let ctx = ActionContext {
            nearest_ally: Some([5, 5, 5]),
            ..Default::default()
        };
        let scored = score_actions(&needs, &ctx);
        let best = select_action_deterministic(&scored);
        assert!(matches!(best, Action::Socialize { .. }));
    }
}
