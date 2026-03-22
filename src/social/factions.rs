// Faction system: groups with shared identity, territory, and goals.
//
// Factions are collections of creatures that share territory claims and
// collective attitudes. A creature's faction membership influences its
// default disposition toward others and enables group coordination.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

use super::relationships::CreatureId;

/// Unique faction identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FactionId(pub u32);

/// A faction: a group of creatures with shared identity.
#[derive(Debug, Clone)]
pub struct Faction {
    pub id: FactionId,
    pub name: String,
    /// Member creature IDs.
    pub members: HashSet<CreatureId>,
    /// Territory claim: list of chunk coordinates (x, z).
    pub territory: Vec<[i32; 2]>,
    /// Default disposition toward non-members (-1.0 hostile → 1.0 friendly).
    pub outsider_disposition: f32,
}

impl Faction {
    pub fn new(id: FactionId, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            members: HashSet::new(),
            territory: Vec::new(),
            outsider_disposition: 0.0,
        }
    }

    pub fn add_member(&mut self, creature: CreatureId) {
        self.members.insert(creature);
    }

    pub fn remove_member(&mut self, creature: CreatureId) {
        self.members.remove(&creature);
    }

    pub fn is_member(&self, creature: CreatureId) -> bool {
        self.members.contains(&creature)
    }

    pub fn member_count(&self) -> usize {
        self.members.len()
    }

    /// Claim a chunk as territory.
    pub fn claim_territory(&mut self, chunk_xz: [i32; 2]) {
        if !self.territory.contains(&chunk_xz) {
            self.territory.push(chunk_xz);
        }
    }

    /// Check if a chunk is in this faction's territory.
    pub fn owns_territory(&self, chunk_xz: [i32; 2]) -> bool {
        self.territory.contains(&chunk_xz)
    }
}

/// Inter-faction relationship.
#[derive(Debug, Clone)]
pub struct FactionRelation {
    /// -1.0 (war) to 1.0 (alliance).
    pub standing: f32,
}

impl Default for FactionRelation {
    fn default() -> Self {
        Self { standing: 0.0 }
    }
}

/// Registry of all factions and their inter-relationships.
#[derive(Debug, Clone, Default)]
pub struct FactionRegistry {
    pub factions: HashMap<FactionId, Faction>,
    /// Relations between faction pairs. Key is (min_id, max_id) for canonical ordering.
    pub relations: HashMap<(FactionId, FactionId), FactionRelation>,
    /// Creature → faction mapping for quick lookup.
    pub creature_faction: HashMap<CreatureId, FactionId>,
}

impl FactionRegistry {
    /// Register a new faction.
    pub fn add_faction(&mut self, faction: Faction) {
        let id = faction.id;
        for &member in &faction.members {
            self.creature_faction.insert(member, id);
        }
        self.factions.insert(id, faction);
    }

    /// Get a creature's faction.
    pub fn creature_faction(&self, creature: CreatureId) -> Option<FactionId> {
        self.creature_faction.get(&creature).copied()
    }

    /// Add a creature to a faction.
    pub fn join_faction(&mut self, creature: CreatureId, faction_id: FactionId) {
        // Remove from old faction
        if let Some(old_id) = self.creature_faction.get(&creature).copied() {
            if let Some(old) = self.factions.get_mut(&old_id) {
                old.remove_member(creature);
            }
        }
        if let Some(faction) = self.factions.get_mut(&faction_id) {
            faction.add_member(creature);
            self.creature_faction.insert(creature, faction_id);
        }
    }

    /// Remove a creature from its faction.
    pub fn leave_faction(&mut self, creature: CreatureId) {
        if let Some(faction_id) = self.creature_faction.remove(&creature) {
            if let Some(faction) = self.factions.get_mut(&faction_id) {
                faction.remove_member(creature);
            }
        }
    }

    /// Canonical key for a faction pair.
    fn pair_key(a: FactionId, b: FactionId) -> (FactionId, FactionId) {
        if a.0 <= b.0 {
            (a, b)
        } else {
            (b, a)
        }
    }

    /// Get standing between two factions.
    pub fn get_standing(&self, a: FactionId, b: FactionId) -> f32 {
        if a == b {
            return 1.0; // same faction = full alliance
        }
        self.relations
            .get(&Self::pair_key(a, b))
            .map(|r| r.standing)
            .unwrap_or(0.0)
    }

    /// Modify standing between two factions.
    pub fn adjust_standing(&mut self, a: FactionId, b: FactionId, delta: f32) {
        if a == b {
            return;
        }
        let key = Self::pair_key(a, b);
        let rel = self.relations.entry(key).or_default();
        rel.standing = (rel.standing + delta).clamp(-1.0, 1.0);
    }

    /// Are two factions allies (standing > 0.5)?
    pub fn are_allies(&self, a: FactionId, b: FactionId) -> bool {
        self.get_standing(a, b) > 0.5
    }

    /// Are two factions at war (standing < -0.5)?
    pub fn at_war(&self, a: FactionId, b: FactionId) -> bool {
        self.get_standing(a, b) < -0.5
    }

    /// Check if two creatures are in the same faction.
    pub fn same_faction(&self, a: CreatureId, b: CreatureId) -> bool {
        match (self.creature_faction(a), self.creature_faction(b)) {
            (Some(fa), Some(fb)) => fa == fb,
            _ => false,
        }
    }

    /// Get the default disposition a faction member should have toward a creature.
    pub fn default_disposition(&self, member: CreatureId, other: CreatureId) -> f32 {
        let Some(my_faction_id) = self.creature_faction(member) else {
            return 0.0;
        };

        // Same faction = friendly
        if let Some(other_faction_id) = self.creature_faction(other) {
            if my_faction_id == other_faction_id {
                return 0.8;
            }
            return self.get_standing(my_faction_id, other_faction_id);
        }

        // Other has no faction — use outsider disposition
        self.factions
            .get(&my_faction_id)
            .map(|f| f.outsider_disposition)
            .unwrap_or(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_registry() -> FactionRegistry {
        let mut reg = FactionRegistry::default();

        let mut wolves = Faction::new(FactionId(1), "Wolf Pack");
        wolves.outsider_disposition = -0.3;
        wolves.add_member(CreatureId(10));
        wolves.add_member(CreatureId(11));

        let mut deer = Faction::new(FactionId(2), "Deer Herd");
        deer.outsider_disposition = 0.0;
        deer.add_member(CreatureId(20));
        deer.add_member(CreatureId(21));

        reg.add_faction(wolves);
        reg.add_faction(deer);
        reg
    }

    #[test]
    fn faction_membership() {
        let reg = setup_registry();
        assert_eq!(reg.creature_faction(CreatureId(10)), Some(FactionId(1)));
        assert_eq!(reg.creature_faction(CreatureId(20)), Some(FactionId(2)));
        assert_eq!(reg.creature_faction(CreatureId(99)), None);
    }

    #[test]
    fn same_faction_check() {
        let reg = setup_registry();
        assert!(reg.same_faction(CreatureId(10), CreatureId(11)));
        assert!(!reg.same_faction(CreatureId(10), CreatureId(20)));
    }

    #[test]
    fn same_faction_standing_is_one() {
        let reg = setup_registry();
        assert_eq!(reg.get_standing(FactionId(1), FactionId(1)), 1.0);
    }

    #[test]
    fn default_standing_is_neutral() {
        let reg = setup_registry();
        assert_eq!(reg.get_standing(FactionId(1), FactionId(2)), 0.0);
    }

    #[test]
    fn adjust_standing_works() {
        let mut reg = setup_registry();
        reg.adjust_standing(FactionId(1), FactionId(2), -0.7);
        assert!(reg.at_war(FactionId(1), FactionId(2)));
        reg.adjust_standing(FactionId(1), FactionId(2), 1.5);
        assert!(reg.are_allies(FactionId(1), FactionId(2)));
    }

    #[test]
    fn standing_clamps() {
        let mut reg = setup_registry();
        reg.adjust_standing(FactionId(1), FactionId(2), -5.0);
        assert_eq!(reg.get_standing(FactionId(1), FactionId(2)), -1.0);
        reg.adjust_standing(FactionId(1), FactionId(2), 10.0);
        assert_eq!(reg.get_standing(FactionId(1), FactionId(2)), 1.0);
    }

    #[test]
    fn join_faction_moves_creature() {
        let mut reg = setup_registry();
        // Move wolf 10 to deer herd
        reg.join_faction(CreatureId(10), FactionId(2));
        assert_eq!(reg.creature_faction(CreatureId(10)), Some(FactionId(2)));
        assert!(!reg.factions[&FactionId(1)].is_member(CreatureId(10)));
        assert!(reg.factions[&FactionId(2)].is_member(CreatureId(10)));
    }

    #[test]
    fn leave_faction_removes_creature() {
        let mut reg = setup_registry();
        reg.leave_faction(CreatureId(10));
        assert_eq!(reg.creature_faction(CreatureId(10)), None);
        assert_eq!(reg.factions[&FactionId(1)].member_count(), 1);
    }

    #[test]
    fn territory_claims() {
        let mut reg = setup_registry();
        reg.factions
            .get_mut(&FactionId(1))
            .unwrap()
            .claim_territory([0, 0]);
        reg.factions
            .get_mut(&FactionId(1))
            .unwrap()
            .claim_territory([1, 0]);
        assert!(reg.factions[&FactionId(1)].owns_territory([0, 0]));
        assert!(!reg.factions[&FactionId(1)].owns_territory([5, 5]));
    }

    #[test]
    fn duplicate_territory_claim_ignored() {
        let mut reg = setup_registry();
        let f = reg.factions.get_mut(&FactionId(1)).unwrap();
        f.claim_territory([3, 3]);
        f.claim_territory([3, 3]);
        assert_eq!(f.territory.len(), 1);
    }

    #[test]
    fn default_disposition_same_faction() {
        let reg = setup_registry();
        let d = reg.default_disposition(CreatureId(10), CreatureId(11));
        assert_eq!(d, 0.8);
    }

    #[test]
    fn default_disposition_outsider() {
        let reg = setup_registry();
        let d = reg.default_disposition(CreatureId(10), CreatureId(99));
        assert_eq!(d, -0.3); // wolf pack outsider disposition
    }

    #[test]
    fn default_disposition_cross_faction() {
        let mut reg = setup_registry();
        reg.adjust_standing(FactionId(1), FactionId(2), -0.4);
        let d = reg.default_disposition(CreatureId(10), CreatureId(20));
        assert_eq!(d, -0.4);
    }
}
