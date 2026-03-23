// Metabolism system: energy from food, energy cost per action.
//
// Creatures have an energy reserve that depletes over time (basal metabolism)
// and is replenished by eating. When energy reaches zero, the creature begins
// to starve, losing health each tick. Different body sizes have different
// metabolic rates — larger creatures burn energy faster but can store more.

#![allow(dead_code)]

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::data::BodySize;

/// ECS component tracking a creature's metabolic state.
#[derive(Serialize, Deserialize, Component, Debug, Clone)]
pub struct Metabolism {
    /// Current energy reserve (calories).
    pub energy: f32,
    /// Maximum energy capacity.
    pub max_energy: f32,
    /// Basal metabolic rate: energy consumed per tick just to stay alive.
    pub basal_rate: f32,
    /// Whether the creature is currently starving (energy == 0).
    pub starving: bool,
    /// Health damage per tick while starving.
    pub starvation_damage: f32,
}

impl Metabolism {
    /// Create a metabolism appropriate for the given body size.
    pub fn for_body_size(size: BodySize) -> Self {
        let (max_energy, basal_rate) = match size {
            BodySize::Tiny => (100.0, 0.5),
            BodySize::Small => (250.0, 1.0),
            BodySize::Medium => (500.0, 2.0),
            BodySize::Large => (800.0, 3.0),
            BodySize::Huge => (1500.0, 5.0),
        };
        Self {
            energy: max_energy,
            max_energy,
            basal_rate,
            starving: false,
            starvation_damage: 1.0,
        }
    }

    /// Consume energy for an action. Returns true if the creature had enough.
    pub fn spend_energy(&mut self, amount: f32) -> bool {
        if self.energy >= amount {
            self.energy -= amount;
            true
        } else {
            self.energy = 0.0;
            false
        }
    }

    /// Gain energy from food (capped at max).
    pub fn feed(&mut self, calories: f32) {
        self.energy = (self.energy + calories).min(self.max_energy);
        if self.energy > 0.0 {
            self.starving = false;
        }
    }

    /// Returns the energy fraction (0.0 = empty, 1.0 = full).
    pub fn energy_fraction(&self) -> f32 {
        if self.max_energy > 0.0 {
            self.energy / self.max_energy
        } else {
            0.0
        }
    }
}

/// Tick metabolism for a single creature.
/// Returns the starvation damage dealt this tick (0 if not starving).
pub fn tick_metabolism(metabolism: &mut Metabolism) -> f32 {
    metabolism.energy = (metabolism.energy - metabolism.basal_rate).max(0.0);

    if metabolism.energy <= 0.0 {
        metabolism.starving = true;
        metabolism.starvation_damage
    } else {
        metabolism.starving = false;
        0.0
    }
}

/// Calculate energy cost for movement (speed-dependent).
pub fn movement_energy_cost(speed: f32, distance: f32) -> f32 {
    speed * distance * 0.1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metabolism_for_body_sizes() {
        let tiny = Metabolism::for_body_size(BodySize::Tiny);
        let huge = Metabolism::for_body_size(BodySize::Huge);
        assert!(huge.max_energy > tiny.max_energy);
        assert!(huge.basal_rate > tiny.basal_rate);
    }

    #[test]
    fn starts_full_and_not_starving() {
        let m = Metabolism::for_body_size(BodySize::Medium);
        assert_eq!(m.energy, m.max_energy);
        assert!(!m.starving);
    }

    #[test]
    fn tick_depletes_energy() {
        let mut m = Metabolism::for_body_size(BodySize::Medium);
        let initial = m.energy;
        tick_metabolism(&mut m);
        assert!(m.energy < initial);
        assert_eq!(m.energy, initial - m.basal_rate);
    }

    #[test]
    fn starvation_when_energy_depleted() {
        let mut m = Metabolism::for_body_size(BodySize::Medium);
        m.energy = 1.0;
        m.basal_rate = 5.0;

        let damage = tick_metabolism(&mut m);
        assert!(m.starving);
        assert_eq!(m.energy, 0.0);
        assert!(damage > 0.0);
    }

    #[test]
    fn feeding_restores_energy() {
        let mut m = Metabolism::for_body_size(BodySize::Medium);
        m.energy = 100.0;
        m.starving = true;

        m.feed(200.0);
        assert_eq!(m.energy, 300.0);
        assert!(!m.starving);
    }

    #[test]
    fn feeding_caps_at_max() {
        let mut m = Metabolism::for_body_size(BodySize::Small);
        m.feed(10000.0);
        assert_eq!(m.energy, m.max_energy);
    }

    #[test]
    fn spend_energy_succeeds_when_enough() {
        let mut m = Metabolism::for_body_size(BodySize::Medium);
        assert!(m.spend_energy(100.0));
        assert_eq!(m.energy, m.max_energy - 100.0);
    }

    #[test]
    fn spend_energy_fails_when_insufficient() {
        let mut m = Metabolism::for_body_size(BodySize::Medium);
        m.energy = 10.0;
        assert!(!m.spend_energy(50.0));
        assert_eq!(m.energy, 0.0);
    }

    #[test]
    fn energy_fraction_correct() {
        let mut m = Metabolism::for_body_size(BodySize::Medium);
        assert_eq!(m.energy_fraction(), 1.0);
        m.energy = m.max_energy * 0.5;
        assert!((m.energy_fraction() - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn movement_cost_scales_with_speed() {
        let slow = movement_energy_cost(2.0, 1.0);
        let fast = movement_energy_cost(10.0, 1.0);
        assert!(fast > slow);
    }
}
