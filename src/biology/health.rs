// Health system: damage types, healing, disease.
//
// Creatures have health points that decrease from damage and recover slowly
// over time (if well-fed). Different damage types interact with the
// chemistry system — fire damage, cold damage, poison, etc.

#![allow(dead_code)]

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

/// Types of damage a creature can receive.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum DamageType {
    Physical,
    Fire,
    Cold,
    Poison,
    Starvation,
    Suffocation,
    Fall,
}

/// ECS component tracking creature health.
#[derive(Serialize, Deserialize, Component, Debug, Clone)]
pub struct Health {
    /// Current hit points.
    pub current: f32,
    /// Maximum hit points.
    pub max: f32,
    /// Natural healing rate per tick (when conditions are met).
    pub heal_rate: f32,
    /// Minimum energy fraction required for natural healing (0.0–1.0).
    pub heal_threshold: f32,
    /// Active status effects (damage-over-time).
    pub status_effects: Vec<StatusEffect>,
    /// Whether this creature is dead.
    pub dead: bool,
}

/// A timed status effect that deals damage or modifies stats.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StatusEffect {
    pub effect_type: DamageType,
    /// Damage per tick.
    pub damage_per_tick: f32,
    /// Remaining ticks.
    pub remaining_ticks: u32,
}

impl Health {
    pub fn new(max_health: f32) -> Self {
        Self {
            current: max_health,
            max: max_health,
            heal_rate: 0.1,
            heal_threshold: 0.3,
            status_effects: Vec::new(),
            dead: false,
        }
    }

    /// Apply damage, return actual damage dealt.
    pub fn take_damage(&mut self, amount: f32, _damage_type: DamageType) -> f32 {
        if self.dead {
            return 0.0;
        }
        let actual = amount.min(self.current);
        self.current -= actual;
        if self.current <= 0.0 {
            self.current = 0.0;
            self.dead = true;
        }
        actual
    }

    /// Heal by an amount (capped at max).
    pub fn heal(&mut self, amount: f32) {
        if !self.dead {
            self.current = (self.current + amount).min(self.max);
        }
    }

    /// Apply a status effect.
    pub fn apply_effect(&mut self, effect: StatusEffect) {
        self.status_effects.push(effect);
    }

    /// Returns health as a fraction (0.0–1.0).
    pub fn health_fraction(&self) -> f32 {
        if self.max > 0.0 {
            self.current / self.max
        } else {
            0.0
        }
    }
}

/// Tick health for a single creature: process status effects and natural healing.
/// `energy_fraction` is from the metabolism system (0.0–1.0).
/// Returns total damage dealt this tick from status effects.
pub fn tick_health(health: &mut Health, energy_fraction: f32) -> f32 {
    if health.dead {
        return 0.0;
    }

    let mut total_damage = 0.0;

    // Process status effects
    let mut i = 0;
    while i < health.status_effects.len() {
        let effect = &mut health.status_effects[i];
        total_damage += effect.damage_per_tick;
        health.current = (health.current - effect.damage_per_tick).max(0.0);

        effect.remaining_ticks = effect.remaining_ticks.saturating_sub(1);
        if effect.remaining_ticks == 0 {
            health.status_effects.swap_remove(i);
        } else {
            i += 1;
        }
    }

    // Natural healing (only if well-fed and not dead)
    if health.current > 0.0 && energy_fraction >= health.heal_threshold {
        health.current = (health.current + health.heal_rate).min(health.max);
    }

    if health.current <= 0.0 {
        health.dead = true;
    }

    total_damage
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_health_is_full() {
        let h = Health::new(100.0);
        assert_eq!(h.current, 100.0);
        assert_eq!(h.max, 100.0);
        assert!(!h.dead);
    }

    #[test]
    fn take_damage_reduces_health() {
        let mut h = Health::new(100.0);
        let dealt = h.take_damage(30.0, DamageType::Physical);
        assert_eq!(dealt, 30.0);
        assert_eq!(h.current, 70.0);
        assert!(!h.dead);
    }

    #[test]
    fn lethal_damage_kills() {
        let mut h = Health::new(50.0);
        h.take_damage(60.0, DamageType::Physical);
        assert_eq!(h.current, 0.0);
        assert!(h.dead);
    }

    #[test]
    fn dead_creature_takes_no_more_damage() {
        let mut h = Health::new(10.0);
        h.take_damage(10.0, DamageType::Physical);
        assert!(h.dead);
        let dealt = h.take_damage(50.0, DamageType::Fire);
        assert_eq!(dealt, 0.0);
    }

    #[test]
    fn heal_restores_health() {
        let mut h = Health::new(100.0);
        h.current = 50.0;
        h.heal(30.0);
        assert_eq!(h.current, 80.0);
    }

    #[test]
    fn heal_caps_at_max() {
        let mut h = Health::new(100.0);
        h.current = 90.0;
        h.heal(50.0);
        assert_eq!(h.current, 100.0);
    }

    #[test]
    fn dead_creature_cannot_heal() {
        let mut h = Health::new(100.0);
        h.dead = true;
        h.current = 0.0;
        h.heal(50.0);
        assert_eq!(h.current, 0.0);
    }

    #[test]
    fn status_effect_deals_damage_over_time() {
        let mut h = Health::new(100.0);
        h.apply_effect(StatusEffect {
            effect_type: DamageType::Poison,
            damage_per_tick: 5.0,
            remaining_ticks: 3,
        });

        let d1 = tick_health(&mut h, 1.0);
        assert_eq!(d1, 5.0);
        assert_eq!(h.status_effects.len(), 1);

        tick_health(&mut h, 1.0);
        tick_health(&mut h, 1.0);
        // Effect should have expired
        assert!(h.status_effects.is_empty());
    }

    #[test]
    fn natural_healing_requires_energy() {
        let mut h = Health::new(100.0);
        h.current = 50.0;
        h.heal_rate = 5.0;
        h.heal_threshold = 0.5;

        // Well-fed → heals
        tick_health(&mut h, 0.8);
        assert!(h.current > 50.0);

        // Starving → no healing
        let before = h.current;
        h.heal_rate = 5.0;
        tick_health(&mut h, 0.1);
        // Should not have healed (energy below threshold)
        assert!(h.current <= before);
    }

    #[test]
    fn health_fraction_correct() {
        let mut h = Health::new(200.0);
        assert_eq!(h.health_fraction(), 1.0);
        h.current = 50.0;
        assert_eq!(h.health_fraction(), 0.25);
    }

    #[test]
    fn multiple_status_effects_stack() {
        let mut h = Health::new(100.0);
        h.apply_effect(StatusEffect {
            effect_type: DamageType::Fire,
            damage_per_tick: 3.0,
            remaining_ticks: 2,
        });
        h.apply_effect(StatusEffect {
            effect_type: DamageType::Poison,
            damage_per_tick: 2.0,
            remaining_ticks: 2,
        });

        let total = tick_health(&mut h, 0.0);
        assert_eq!(total, 5.0);
    }
}
