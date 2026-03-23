// Relationship tracking: entity-to-entity social bonds.
//
// Each creature maintains a map of relationships to other creatures,
// tracking trust, familiarity, and hostility as float components.
// These values drift over time and are modified by observed actions.

#![allow(dead_code)]

use std::collections::HashMap;

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

/// Unique identifier for a creature in the social system.
/// Maps to Bevy Entity bits but kept as u64 for pure-function testability.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CreatureId(pub u64);

/// How one creature feels about another.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Relationship {
    /// Trust: -1.0 (total distrust) to 1.0 (total trust).
    pub trust: f32,
    /// Familiarity: 0.0 (stranger) to 1.0 (well-known).
    pub familiarity: f32,
    /// Hostility: 0.0 (friendly) to 1.0 (hostile).
    pub hostility: f32,
}

impl Default for Relationship {
    fn default() -> Self {
        Self {
            trust: 0.0,
            familiarity: 0.0,
            hostility: 0.0,
        }
    }
}

impl Relationship {
    /// Overall disposition: positive = friendly, negative = hostile.
    pub fn disposition(&self) -> f32 {
        self.trust - self.hostility
    }

    /// Whether this relationship is broadly positive.
    pub fn is_friendly(&self) -> bool {
        self.disposition() > 0.1
    }

    /// Whether this relationship is broadly negative.
    pub fn is_hostile(&self) -> bool {
        self.disposition() < -0.1
    }
}

/// ECS component: all relationships this creature has with others.
#[derive(Component, Debug, Clone, Default)]
pub struct Relationships {
    pub map: HashMap<CreatureId, Relationship>,
}

impl Relationships {
    /// Get or create a relationship with another creature.
    pub fn get_or_default(&mut self, other: CreatureId) -> &mut Relationship {
        self.map.entry(other).or_default()
    }

    /// Get a relationship if it exists.
    pub fn get(&self, other: CreatureId) -> Option<&Relationship> {
        self.map.get(&other)
    }

    /// Modify trust toward a creature.
    pub fn adjust_trust(&mut self, other: CreatureId, delta: f32) {
        let rel = self.get_or_default(other);
        rel.trust = (rel.trust + delta).clamp(-1.0, 1.0);
    }

    /// Modify hostility toward a creature.
    pub fn adjust_hostility(&mut self, other: CreatureId, delta: f32) {
        let rel = self.get_or_default(other);
        rel.hostility = (rel.hostility + delta).clamp(0.0, 1.0);
    }

    /// Increase familiarity (from proximity/interaction).
    pub fn increase_familiarity(&mut self, other: CreatureId, amount: f32) {
        let rel = self.get_or_default(other);
        rel.familiarity = (rel.familiarity + amount).min(1.0);
    }
}

/// Decay all relationships slightly over time (memory fading).
/// Familiarity decays slowly, trust/hostility drift toward neutral.
pub fn decay_relationships(relationships: &mut Relationships, rate: f32) {
    for rel in relationships.map.values_mut() {
        rel.familiarity = (rel.familiarity - rate * 0.5).max(0.0);
        // Trust and hostility drift toward zero
        if rel.trust > 0.0 {
            rel.trust = (rel.trust - rate).max(0.0);
        } else {
            rel.trust = (rel.trust + rate).min(0.0);
        }
        rel.hostility = (rel.hostility - rate * 0.3).max(0.0);
    }
}

/// Remove relationships with zero familiarity (forgotten creatures).
pub fn prune_forgotten(relationships: &mut Relationships) {
    relationships
        .map
        .retain(|_, rel| rel.familiarity > f32::EPSILON);
}

/// Count friendly/hostile/neutral relationships.
pub fn count_relationships(relationships: &Relationships) -> (usize, usize, usize) {
    let mut friendly = 0;
    let mut hostile = 0;
    let mut neutral = 0;
    for rel in relationships.map.values() {
        if rel.is_friendly() {
            friendly += 1;
        } else if rel.is_hostile() {
            hostile += 1;
        } else {
            neutral += 1;
        }
    }
    (friendly, hostile, neutral)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_relationship_is_neutral() {
        let r = Relationship::default();
        assert_eq!(r.trust, 0.0);
        assert_eq!(r.familiarity, 0.0);
        assert_eq!(r.hostility, 0.0);
        assert!(!r.is_friendly());
        assert!(!r.is_hostile());
    }

    #[test]
    fn disposition_combines_trust_and_hostility() {
        let r = Relationship {
            trust: 0.5,
            hostility: 0.2,
            familiarity: 0.8,
        };
        assert!((r.disposition() - 0.3).abs() < f32::EPSILON);
        assert!(r.is_friendly());
    }

    #[test]
    fn hostile_relationship() {
        let r = Relationship {
            trust: -0.3,
            hostility: 0.8,
            familiarity: 0.5,
        };
        assert!(r.is_hostile());
        assert!(!r.is_friendly());
    }

    #[test]
    fn adjust_trust_clamps() {
        let mut rels = Relationships::default();
        let other = CreatureId(42);
        rels.adjust_trust(other, 2.0);
        assert_eq!(rels.get(other).unwrap().trust, 1.0);
        rels.adjust_trust(other, -5.0);
        assert_eq!(rels.get(other).unwrap().trust, -1.0);
    }

    #[test]
    fn adjust_hostility_clamps() {
        let mut rels = Relationships::default();
        let other = CreatureId(1);
        rels.adjust_hostility(other, 0.5);
        assert_eq!(rels.get(other).unwrap().hostility, 0.5);
        rels.adjust_hostility(other, -1.0);
        assert_eq!(rels.get(other).unwrap().hostility, 0.0);
    }

    #[test]
    fn familiarity_increases_and_caps() {
        let mut rels = Relationships::default();
        let other = CreatureId(10);
        rels.increase_familiarity(other, 0.3);
        assert_eq!(rels.get(other).unwrap().familiarity, 0.3);
        rels.increase_familiarity(other, 2.0);
        assert_eq!(rels.get(other).unwrap().familiarity, 1.0);
    }

    #[test]
    fn decay_reduces_values() {
        let mut rels = Relationships::default();
        let other = CreatureId(5);
        rels.adjust_trust(other, 0.8);
        rels.adjust_hostility(other, 0.6);
        rels.increase_familiarity(other, 0.9);

        decay_relationships(&mut rels, 0.1);
        let r = rels.get(other).unwrap();
        assert!(r.trust < 0.8);
        assert!(r.hostility < 0.6);
        assert!(r.familiarity < 0.9);
    }

    #[test]
    fn decay_negative_trust_toward_zero() {
        let mut rels = Relationships::default();
        let other = CreatureId(7);
        rels.adjust_trust(other, -0.5);
        rels.increase_familiarity(other, 1.0);

        decay_relationships(&mut rels, 0.1);
        let r = rels.get(other).unwrap();
        assert!(r.trust > -0.5); // drifted toward zero
    }

    #[test]
    fn prune_removes_forgotten() {
        let mut rels = Relationships::default();
        let a = CreatureId(1);
        let b = CreatureId(2);
        rels.increase_familiarity(a, 0.5);
        // b has default familiarity = 0.0
        rels.adjust_trust(b, 0.3);

        prune_forgotten(&mut rels);
        assert!(rels.get(a).is_some());
        assert!(rels.get(b).is_none()); // forgotten
    }

    #[test]
    fn count_relationships_categorizes() {
        let mut rels = Relationships::default();
        rels.adjust_trust(CreatureId(1), 0.5);
        rels.increase_familiarity(CreatureId(1), 0.5);
        rels.adjust_hostility(CreatureId(2), 0.8);
        rels.increase_familiarity(CreatureId(2), 0.5);
        rels.increase_familiarity(CreatureId(3), 0.5);

        let (friendly, hostile, neutral) = count_relationships(&rels);
        assert_eq!(friendly, 1);
        assert_eq!(hostile, 1);
        assert_eq!(neutral, 1);
    }

    #[test]
    fn get_or_default_creates_new() {
        let mut rels = Relationships::default();
        let other = CreatureId(99);
        assert!(rels.get(other).is_none());
        let rel = rels.get_or_default(other);
        assert_eq!(rel.trust, 0.0);
        assert!(rels.get(other).is_some());
    }
}
