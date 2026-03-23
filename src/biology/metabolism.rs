// Metabolism system: energy from food, energy cost per action.
//
// Creatures have an energy reserve that depletes over time (basal metabolism)
// and is replenished by eating. When energy reaches zero, the creature begins
// to starve, losing health each tick. Different body sizes have different
// metabolic rates — larger creatures burn energy faster but can store more.
//
// Metabolic rates follow Kleiber's law: P = 70 × m^0.75 (kcal/day),
// converted to Watts (J/s) for SI consistency.
// Source: Wikipedia — Kleiber's law.

#![allow(dead_code)]

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::data::BodySize;

/// Approximate mass (kg) for each body size category.
/// Assumes tissue density ≈ 1050 kg/m³ (close to water).
/// Source: Wikipedia — Body composition.
pub fn body_size_mass(size: BodySize) -> f32 {
    match size {
        BodySize::Tiny => 0.5,    // mouse, small bird
        BodySize::Small => 5.0,   // rabbit, cat
        BodySize::Medium => 40.0, // wolf, deer
        BodySize::Large => 200.0, // bear, horse
        BodySize::Huge => 1000.0, // elephant-scale
    }
}

/// Kleiber's law basal metabolic rate in Watts (J/s).
///
/// P = 70 × m^0.75 (kcal/day), converted:
///   1 kcal = 4184 J, 1 day = 86400 s
///   P_watts = 70 × m^0.75 × 4184 / 86400
///           ≈ 3.39 × m^0.75
///
/// Source: Wikipedia — Kleiber's law.
pub fn kleiber_basal_rate_watts(mass_kg: f32) -> f32 {
    3.39 * mass_kg.powf(0.75)
}

/// ECS component tracking a creature's metabolic state.
#[derive(Serialize, Deserialize, Component, Debug, Clone)]
pub struct Metabolism {
    /// Current energy reserve (Joules).
    pub energy: f32,
    /// Maximum energy capacity (Joules).
    pub max_energy: f32,
    /// Basal metabolic rate (Watts = J/s): energy consumed per second at rest.
    pub basal_rate: f32,
    /// Whether the creature is currently starving (energy == 0).
    pub starving: bool,
    /// Health damage per tick while starving.
    pub starvation_damage: f32,
}

impl Metabolism {
    /// Create a metabolism appropriate for the given body size using Kleiber's law.
    ///
    /// Max energy ≈ enough to survive ~24h without food at basal rate.
    pub fn for_body_size(size: BodySize) -> Self {
        let mass = body_size_mass(size);
        let basal_rate = kleiber_basal_rate_watts(mass);
        // Store enough energy for ~24 hours of basal metabolism
        let max_energy = basal_rate * 86400.0;
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

    /// Gain energy from food (capped at max). Amount in Joules.
    pub fn feed(&mut self, joules: f32) {
        self.energy = (self.energy + joules).min(self.max_energy);
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

/// Tick metabolism for a single creature over `dt` seconds.
/// Returns the starvation damage dealt this tick (0 if not starving).
pub fn tick_metabolism(metabolism: &mut Metabolism, dt: f32) -> f32 {
    metabolism.energy = (metabolism.energy - metabolism.basal_rate * dt).max(0.0);

    if metabolism.energy <= 0.0 {
        metabolism.starving = true;
        metabolism.starvation_damage
    } else {
        metabolism.starving = false;
        0.0
    }
}

/// Calculate energy cost for movement (J).
/// Based on cost of transport ≈ 10 J/(kg·m) for terrestrial locomotion.
/// Source: Wikipedia — Cost of transport.
pub fn movement_energy_cost(mass_kg: f32, distance_m: f32) -> f32 {
    10.0 * mass_kg * distance_m
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
        let dt = 1.0; // 1 second
        tick_metabolism(&mut m, dt);
        assert!(m.energy < initial);
        let expected = initial - m.basal_rate * dt;
        assert!((m.energy - expected).abs() < 0.01);
    }

    #[test]
    fn starvation_when_energy_depleted() {
        let mut m = Metabolism::for_body_size(BodySize::Medium);
        m.energy = 1.0;
        m.basal_rate = 100.0; // artificially high

        let damage = tick_metabolism(&mut m, 1.0);
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
        m.feed(1e12);
        assert_eq!(m.energy, m.max_energy);
    }

    #[test]
    fn spend_energy_succeeds_when_enough() {
        let mut m = Metabolism::for_body_size(BodySize::Medium);
        let initial = m.energy;
        assert!(m.spend_energy(100.0));
        assert!((m.energy - (initial - 100.0)).abs() < 0.01);
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
    fn movement_cost_scales_with_mass() {
        let light = movement_energy_cost(5.0, 1.0);
        let heavy = movement_energy_cost(50.0, 1.0);
        assert!(heavy > light);
    }

    // --- Real-world validation tests (Kleiber's law) ---
    // Source: Wikipedia — Kleiber's law: P = 70 × m^0.75 (kcal/day)

    #[test]
    fn kleiber_70kg_human_basal_rate() {
        // 70 kg human: P = 70 × 70^0.75 = 70 × 24.2 ≈ 1694 kcal/day
        // In Watts: 1694 × 4184 / 86400 ≈ 82 W
        let rate = kleiber_basal_rate_watts(70.0);
        assert!(
            (rate - 82.0).abs() < 5.0,
            "70 kg human basal rate should be ~82 W, got {rate} W"
        );
    }

    #[test]
    fn kleiber_5kg_small_animal() {
        // 5 kg animal: P = 70 × 5^0.75 = 70 × 3.34 ≈ 234 kcal/day
        // In Watts: 234 × 4184 / 86400 ≈ 11.3 W
        let rate = kleiber_basal_rate_watts(5.0);
        assert!(
            (rate - 11.3).abs() < 2.0,
            "5 kg animal basal rate should be ~11.3 W, got {rate} W"
        );
    }

    #[test]
    fn larger_animals_more_efficient_per_kg() {
        // Kleiber's law predicts metabolic rate per kg decreases with mass
        let small_rate_per_kg = kleiber_basal_rate_watts(5.0) / 5.0;
        let large_rate_per_kg = kleiber_basal_rate_watts(200.0) / 200.0;
        assert!(
            small_rate_per_kg > large_rate_per_kg,
            "Small animals should have higher metabolic rate per kg: {small_rate_per_kg} vs {large_rate_per_kg}"
        );
    }

    #[test]
    fn max_energy_lasts_24_hours() {
        let m = Metabolism::for_body_size(BodySize::Medium);
        let hours_of_reserve = m.max_energy / m.basal_rate / 3600.0;
        assert!(
            (hours_of_reserve - 24.0).abs() < 0.1,
            "Max energy should last ~24h, got {hours_of_reserve}h"
        );
    }
}
