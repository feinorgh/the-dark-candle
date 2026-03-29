// Needs system: weighted motivational drives for creatures.
//
// Each creature tracks five core needs that increase over time and are
// satisfied by specific actions. The need values (0.0 = fully satisfied,
// 1.0 = critical) feed into the utility AI to select actions.

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

/// Core motivational drives for a creature.
#[derive(Serialize, Deserialize, Component, Debug, Clone)]
pub struct Needs {
    /// Hunger: driven by metabolism energy depletion (0 = sated, 1 = starving).
    pub hunger: f32,
    /// Safety: driven by nearby threats (0 = safe, 1 = extreme danger).
    pub safety: f32,
    /// Rest: increases over time with activity (0 = rested, 1 = exhausted).
    pub rest: f32,
    /// Curiosity: desire to explore new areas (0 = content, 1 = restless).
    pub curiosity: f32,
    /// Social: desire to be near others of same species (0 = content, 1 = lonely).
    pub social: f32,
}

impl Default for Needs {
    fn default() -> Self {
        Self {
            hunger: 0.0,
            safety: 0.0,
            rest: 0.0,
            curiosity: 0.3,
            social: 0.2,
        }
    }
}

/// Per-need configuration: how fast it grows and how important it is.
#[derive(Debug, Clone)]
pub struct NeedConfig {
    /// Rate at which this need increases per tick (before modifiers).
    pub growth_rate: f32,
    /// Weight used by utility AI to prioritize this need.
    pub weight: f32,
}

/// Full configuration for all creature needs.
#[derive(Debug, Clone)]
pub struct NeedsConfig {
    pub hunger: NeedConfig,
    pub safety: NeedConfig,
    pub rest: NeedConfig,
    pub curiosity: NeedConfig,
    pub social: NeedConfig,
}

impl Default for NeedsConfig {
    fn default() -> Self {
        Self {
            hunger: NeedConfig {
                growth_rate: 0.0, // driven externally by metabolism
                weight: 1.0,
            },
            safety: NeedConfig {
                growth_rate: 0.0, // driven externally by perception
                weight: 1.5,      // safety is highest priority
            },
            rest: NeedConfig {
                growth_rate: 0.002,
                weight: 0.8,
            },
            curiosity: NeedConfig {
                growth_rate: 0.003,
                weight: 0.4,
            },
            social: NeedConfig {
                growth_rate: 0.001,
                weight: 0.5,
            },
        }
    }
}

/// Tick passive needs (rest, curiosity, social grow over time).
/// Hunger and safety are set externally from metabolism/perception.
pub fn tick_needs(needs: &mut Needs, config: &NeedsConfig) {
    needs.rest = (needs.rest + config.rest.growth_rate).min(1.0);
    needs.curiosity = (needs.curiosity + config.curiosity.growth_rate).min(1.0);
    needs.social = (needs.social + config.social.growth_rate).min(1.0);
}

/// Map metabolism energy fraction → hunger need.
/// energy_fraction: 0.0 (starving) → 1.0 (full).
pub fn update_hunger(needs: &mut Needs, energy_fraction: f32) {
    needs.hunger = (1.0 - energy_fraction).clamp(0.0, 1.0);
}

/// Map perceived threat level → safety need.
/// threat_level: 0.0 (no threats) → 1.0+ (imminent danger).
pub fn update_safety(needs: &mut Needs, threat_level: f32) {
    needs.safety = threat_level.clamp(0.0, 1.0);
}

/// Satisfy a need by the given amount (reduces it toward 0).
pub fn satisfy_need(need: &mut f32, amount: f32) {
    *need = (*need - amount).max(0.0);
}

/// Returns the most urgent need (name, value, weighted_value).
pub fn most_urgent_need(needs: &Needs, config: &NeedsConfig) -> (&'static str, f32) {
    let candidates = [
        ("hunger", needs.hunger * config.hunger.weight),
        ("safety", needs.safety * config.safety.weight),
        ("rest", needs.rest * config.rest.weight),
        ("curiosity", needs.curiosity * config.curiosity.weight),
        ("social", needs.social * config.social.weight),
    ];

    candidates
        .into_iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(("hunger", 0.0))
}

/// Returns all needs as weighted scores (need_name, weighted_value).
pub fn weighted_needs(needs: &Needs, config: &NeedsConfig) -> Vec<(&'static str, f32)> {
    vec![
        ("hunger", needs.hunger * config.hunger.weight),
        ("safety", needs.safety * config.safety.weight),
        ("rest", needs.rest * config.rest.weight),
        ("curiosity", needs.curiosity * config.curiosity.weight),
        ("social", needs.social * config.social.weight),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_needs_are_low() {
        let n = Needs::default();
        assert_eq!(n.hunger, 0.0);
        assert_eq!(n.safety, 0.0);
        assert!(n.rest <= 0.1);
    }

    #[test]
    fn tick_increases_passive_needs() {
        let mut n = Needs::default();
        let config = NeedsConfig::default();
        let old_rest = n.rest;
        let old_curiosity = n.curiosity;
        tick_needs(&mut n, &config);
        assert!(n.rest > old_rest);
        assert!(n.curiosity > old_curiosity);
    }

    #[test]
    fn tick_does_not_change_hunger_or_safety() {
        let mut n = Needs {
            hunger: 0.5,
            safety: 0.3,
            ..Default::default()
        };
        let config = NeedsConfig::default();
        tick_needs(&mut n, &config);
        assert_eq!(n.hunger, 0.5);
        assert_eq!(n.safety, 0.3);
    }

    #[test]
    fn needs_cap_at_one() {
        let mut n = Needs {
            rest: 0.999,
            curiosity: 0.999,
            social: 0.999,
            ..Default::default()
        };
        let config = NeedsConfig::default();
        for _ in 0..100 {
            tick_needs(&mut n, &config);
        }
        assert!(n.rest <= 1.0);
        assert!(n.curiosity <= 1.0);
        assert!(n.social <= 1.0);
    }

    #[test]
    fn update_hunger_from_energy() {
        let mut n = Needs::default();
        update_hunger(&mut n, 1.0); // full energy
        assert_eq!(n.hunger, 0.0);
        update_hunger(&mut n, 0.0); // starving
        assert_eq!(n.hunger, 1.0);
        update_hunger(&mut n, 0.5);
        assert!((n.hunger - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn update_safety_clamps() {
        let mut n = Needs::default();
        update_safety(&mut n, 2.0);
        assert_eq!(n.safety, 1.0);
        update_safety(&mut n, -0.5);
        assert_eq!(n.safety, 0.0);
    }

    #[test]
    fn satisfy_need_reduces_value() {
        let mut need = 0.8;
        satisfy_need(&mut need, 0.3);
        assert!((need - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn satisfy_need_floors_at_zero() {
        let mut need = 0.2;
        satisfy_need(&mut need, 1.0);
        assert_eq!(need, 0.0);
    }

    #[test]
    fn most_urgent_when_hungry() {
        let n = Needs {
            hunger: 0.9,
            safety: 0.0,
            rest: 0.1,
            curiosity: 0.2,
            social: 0.1,
        };
        let config = NeedsConfig::default();
        let (name, _) = most_urgent_need(&n, &config);
        assert_eq!(name, "hunger");
    }

    #[test]
    fn safety_outweighs_hunger_due_to_weight() {
        let n = Needs {
            hunger: 0.6,
            safety: 0.5, // safety has weight 1.5 → 0.75 vs hunger 0.6
            rest: 0.0,
            curiosity: 0.0,
            social: 0.0,
        };
        let config = NeedsConfig::default();
        let (name, _) = most_urgent_need(&n, &config);
        assert_eq!(name, "safety");
    }

    #[test]
    fn weighted_needs_returns_all_five() {
        let n = Needs::default();
        let config = NeedsConfig::default();
        let w = weighted_needs(&n, &config);
        assert_eq!(w.len(), 5);
    }
}
