// Growth and aging: size changes, stat progression, lifespan.
//
// Creatures age each tick. Young creatures grow (increasing stats),
// old creatures weaken. When age exceeds lifespan, the creature dies
// of old age. Growth stages affect body size and metabolic rate.

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

/// Growth stage of a creature's life cycle.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrowthStage {
    Juvenile,
    Adult,
    Elder,
}

/// ECS component tracking a creature's age and growth.
#[derive(Serialize, Deserialize, Component, Debug, Clone)]
pub struct Growth {
    /// Current age in simulation ticks.
    pub age: u32,
    /// Maximum lifespan in ticks (None = immortal).
    pub lifespan: Option<u32>,
    /// Current growth stage.
    pub stage: GrowthStage,
    /// Size multiplier (juveniles start at ~0.5, adults at 1.0, elders may shrink slightly).
    pub size_multiplier: f32,
    /// Stat multiplier affecting speed, damage, etc.
    pub stat_multiplier: f32,
}

impl Growth {
    pub fn new(lifespan: Option<u32>) -> Self {
        Self {
            age: 0,
            lifespan,
            stage: GrowthStage::Juvenile,
            size_multiplier: 0.5,
            stat_multiplier: 0.6,
        }
    }

    /// What fraction of lifespan has elapsed (0.0–1.0). Returns 0 for immortal.
    pub fn age_fraction(&self) -> f32 {
        match self.lifespan {
            Some(ls) if ls > 0 => self.age as f32 / ls as f32,
            _ => 0.0,
        }
    }
}

/// Tick growth for a single creature.
/// Returns true if the creature has died of old age.
pub fn tick_growth(growth: &mut Growth) -> bool {
    growth.age += 1;

    let fraction = growth.age_fraction();

    // Stage transitions based on life fraction
    if fraction < 0.2 {
        growth.stage = GrowthStage::Juvenile;
        // Grow from 0.5 to 1.0 during juvenile phase
        growth.size_multiplier = 0.5 + (fraction / 0.2) * 0.5;
        growth.stat_multiplier = 0.6 + (fraction / 0.2) * 0.4;
    } else if fraction < 0.75 {
        growth.stage = GrowthStage::Adult;
        growth.size_multiplier = 1.0;
        growth.stat_multiplier = 1.0;
    } else if fraction < 1.0 {
        growth.stage = GrowthStage::Elder;
        // Gradual decline
        let elder_frac = (fraction - 0.75) / 0.25;
        growth.size_multiplier = 1.0 - elder_frac * 0.1; // Shrink slightly
        growth.stat_multiplier = 1.0 - elder_frac * 0.3; // Stats decline
    }

    // Death from old age
    if let Some(lifespan) = growth.lifespan
        && growth.age >= lifespan
    {
        return true;
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_growth_is_juvenile() {
        let g = Growth::new(Some(1000));
        assert_eq!(g.stage, GrowthStage::Juvenile);
        assert_eq!(g.age, 0);
        assert_eq!(g.size_multiplier, 0.5);
    }

    #[test]
    fn juvenile_grows_over_time() {
        let mut g = Growth::new(Some(1000));
        // Tick to 10% of lifespan
        for _ in 0..100 {
            tick_growth(&mut g);
        }
        assert_eq!(g.stage, GrowthStage::Juvenile);
        assert!(g.size_multiplier > 0.5);
        assert!(g.size_multiplier < 1.0);
    }

    #[test]
    fn becomes_adult_at_20_percent() {
        let mut g = Growth::new(Some(1000));
        for _ in 0..200 {
            tick_growth(&mut g);
        }
        assert_eq!(g.stage, GrowthStage::Adult);
        assert_eq!(g.size_multiplier, 1.0);
        assert_eq!(g.stat_multiplier, 1.0);
    }

    #[test]
    fn becomes_elder_at_75_percent() {
        let mut g = Growth::new(Some(1000));
        for _ in 0..800 {
            tick_growth(&mut g);
        }
        assert_eq!(g.stage, GrowthStage::Elder);
        assert!(g.stat_multiplier < 1.0);
    }

    #[test]
    fn dies_at_lifespan() {
        let mut g = Growth::new(Some(100));
        for _ in 0..99 {
            assert!(!tick_growth(&mut g));
        }
        assert!(tick_growth(&mut g)); // tick 100 → dead
    }

    #[test]
    fn immortal_never_dies() {
        let mut g = Growth::new(None);
        for _ in 0..100_000 {
            assert!(!tick_growth(&mut g));
        }
        assert_eq!(g.age_fraction(), 0.0);
    }

    #[test]
    fn elder_stats_decline() {
        let mut g = Growth::new(Some(1000));
        for _ in 0..900 {
            tick_growth(&mut g);
        }
        assert_eq!(g.stage, GrowthStage::Elder);
        assert!(g.stat_multiplier < 1.0);
        assert!(g.size_multiplier < 1.0);
    }

    #[test]
    fn age_fraction_for_mortal() {
        let mut g = Growth::new(Some(200));
        g.age = 100;
        assert_eq!(g.age_fraction(), 0.5);
    }

    #[test]
    fn age_fraction_for_immortal() {
        let g = Growth::new(None);
        assert_eq!(g.age_fraction(), 0.0);
    }
}
