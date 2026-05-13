// Procedural creature generation from templates + RNG variation.
//
// Given a CreatureData template, produces a unique Creature ECS component
// with stat offsets (health, speed, attack) and color variation driven
// by a seeded RNG. This keeps every spawned creature slightly different
// while staying within species bounds.

use std::collections::HashMap;

use crate::data::{BodySize, CreatureData, Diet};
use bevy::prelude::*;
use serde::{Deserialize, Serialize};

/// ECS component for a spawned creature instance.
#[derive(Serialize, Deserialize, Component, Debug, Clone)]
pub struct Creature {
    pub species: String,
    pub display_name: String,
    pub health: f32,
    pub max_health: f32,
    pub speed: f32,
    pub attack: f32,
    pub body_size: BodySize,
    pub diet: Diet,
    pub color: [f32; 3],
    pub hostile: bool,
    /// Remaining lifespan ticks (None = immortal).
    pub lifespan: Option<u32>,
    /// Age in simulation ticks.
    pub age: u32,
}

/// Simple deterministic RNG (xorshift64) for stat variation.
/// Keeps us independent of external crate dependencies.
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    /// Returns a value in [0.0, 1.0).
    pub fn next_f32(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state as f32) / (u64::MAX as f32)
    }

    /// Returns a value in [-1.0, 1.0).
    pub fn next_signed(&mut self) -> f32 {
        self.next_f32() * 2.0 - 1.0
    }
}

/// Generate a unique creature instance from a template + seed.
pub fn generate_creature(template: &CreatureData, seed: u64) -> Creature {
    let mut rng = SimpleRng::new(seed);
    let v = template.stat_variation;

    let health_offset = 1.0 + rng.next_signed() * v;
    let speed_offset = 1.0 + rng.next_signed() * v;
    let attack_offset = 1.0 + rng.next_signed() * v;

    let health = (template.base_health * health_offset).max(1.0);
    let speed = (template.base_speed * speed_offset).max(0.1);
    let attack = (template.base_attack * attack_offset).max(0.0);

    // Color variation: slight shift per channel
    let color = [
        (template.color[0] + rng.next_signed() * 0.05).clamp(0.0, 1.0),
        (template.color[1] + rng.next_signed() * 0.05).clamp(0.0, 1.0),
        (template.color[2] + rng.next_signed() * 0.05).clamp(0.0, 1.0),
    ];

    Creature {
        species: template.species.clone(),
        display_name: template.display_name.clone(),
        health,
        max_health: health,
        speed,
        attack,
        body_size: template.body_size,
        diet: template.diet,
        color,
        hostile: template.hostile,
        lifespan: template.lifespan,
        age: 0,
    }
}

/// Get collision half-extents from creature data hitbox.
pub fn creature_hitbox(template: &CreatureData) -> (f32, f32, f32) {
    (
        template.hitbox.0 * 0.5,
        template.hitbox.1 * 0.5,
        template.hitbox.2 * 0.5,
    )
}

/// Marker component: chunk needs creature spawning.
/// Added at chunk spawn time, consumed by `spawn_creatures`.
#[derive(Component)]
pub struct NeedsCreatureSpawning;

/// Tracks creature entities belonging to a chunk for cleanup on unload.
#[derive(Component, Default)]
pub struct ChunkCreatures {
    pub entities: Vec<Entity>,
}

/// Resource holding loaded CreatureData templates, indexed by species ID.
#[derive(Resource, Default)]
pub struct CreatureRegistry {
    creatures: HashMap<String, CreatureData>,
}

impl CreatureRegistry {
    pub fn get(&self, species: &str) -> Option<&CreatureData> {
        self.creatures.get(species)
    }

    pub fn insert(&mut self, data: CreatureData) {
        self.creatures.insert(data.species.clone(), data);
    }

    pub fn len(&self) -> usize {
        self.creatures.len()
    }

    pub fn is_empty(&self) -> bool {
        self.creatures.is_empty()
    }
}

/// Build a `CreatureRegistry` by reading all `.creature.ron` files from disk.
pub fn load_creature_registry() -> Result<CreatureRegistry, String> {
    let dir = crate::data::find_data_dir()?.join("creatures");
    if !dir.is_dir() {
        return Ok(CreatureRegistry::default());
    }
    let entries =
        std::fs::read_dir(&dir).map_err(|e| format!("cannot read {}: {e}", dir.display()))?;

    let mut registry = CreatureRegistry::default();
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        if !name.ends_with(".creature.ron") {
            continue;
        }
        let text = std::fs::read_to_string(&path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let data: CreatureData =
            ron::from_str(&text).map_err(|e| format!("cannot parse {}: {e}", path.display()))?;
        registry.insert(data);
    }

    Ok(registry)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wolf_template() -> CreatureData {
        CreatureData {
            species: "wolf".into(),
            display_name: "Grey Wolf".into(),
            base_health: 80.0,
            base_speed: 6.0,
            base_attack: 15.0,
            body_size: BodySize::Medium,
            diet: Diet::Carnivore,
            hitbox: (0.4, 0.5, 0.8),
            color: [0.5, 0.5, 0.55],
            stat_variation: 0.15,
            preferred_biomes: vec!["forest".into()],
            hostile: true,
            lifespan: Some(40000),
            body_plan: None,
        }
    }

    #[test]
    fn generate_creature_preserves_species() {
        let c = generate_creature(&wolf_template(), 42);
        assert_eq!(c.species, "wolf");
        assert_eq!(c.display_name, "Grey Wolf");
        assert!(c.hostile);
        assert_eq!(c.body_size, BodySize::Medium);
        assert_eq!(c.diet, Diet::Carnivore);
    }

    #[test]
    fn generate_creature_has_positive_stats() {
        for seed in 0..100 {
            let c = generate_creature(&wolf_template(), seed);
            assert!(c.health > 0.0, "seed {seed}: health={}", c.health);
            assert!(c.speed > 0.0, "seed {seed}: speed={}", c.speed);
            assert!(c.attack >= 0.0, "seed {seed}: attack={}", c.attack);
        }
    }

    #[test]
    fn different_seeds_produce_different_creatures() {
        let c1 = generate_creature(&wolf_template(), 1);
        let c2 = generate_creature(&wolf_template(), 2);
        // At least one stat should differ
        let same = (c1.health - c2.health).abs() < f32::EPSILON
            && (c1.speed - c2.speed).abs() < f32::EPSILON
            && (c1.attack - c2.attack).abs() < f32::EPSILON;
        assert!(!same, "Different seeds should produce different creatures");
    }

    #[test]
    fn same_seed_is_deterministic() {
        let c1 = generate_creature(&wolf_template(), 42);
        let c2 = generate_creature(&wolf_template(), 42);
        assert_eq!(c1.health, c2.health);
        assert_eq!(c1.speed, c2.speed);
        assert_eq!(c1.attack, c2.attack);
        assert_eq!(c1.color, c2.color);
    }

    #[test]
    fn stats_within_variation_bounds() {
        let t = wolf_template();
        for seed in 0..200 {
            let c = generate_creature(&t, seed);
            let v = t.stat_variation;
            assert!(
                c.health >= t.base_health * (1.0 - v) * 0.99,
                "seed {seed}: health {} below min {}",
                c.health,
                t.base_health * (1.0 - v)
            );
            assert!(
                c.health <= t.base_health * (1.0 + v) * 1.01,
                "seed {seed}: health {} above max {}",
                c.health,
                t.base_health * (1.0 + v)
            );
        }
    }

    #[test]
    fn max_health_equals_initial_health() {
        let c = generate_creature(&wolf_template(), 42);
        assert_eq!(c.health, c.max_health);
    }

    #[test]
    fn age_starts_at_zero() {
        let c = generate_creature(&wolf_template(), 42);
        assert_eq!(c.age, 0);
    }

    #[test]
    fn color_stays_in_valid_range() {
        for seed in 0..200 {
            let c = generate_creature(&wolf_template(), seed);
            for (i, &ch) in c.color.iter().enumerate() {
                assert!(
                    (0.0..=1.0).contains(&ch),
                    "seed {seed}: color[{i}]={ch} out of range"
                );
            }
        }
    }

    #[test]
    fn zero_variation_gives_exact_stats() {
        let mut t = wolf_template();
        t.stat_variation = 0.0;

        let c = generate_creature(&t, 99);
        assert_eq!(c.health, t.base_health);
        assert_eq!(c.speed, t.base_speed);
        assert_eq!(c.attack, t.base_attack);
    }

    #[test]
    fn creature_hitbox_halves_dimensions() {
        let t = wolf_template();
        let (hx, hy, hz) = creature_hitbox(&t);
        assert_eq!(hx, 0.2);
        assert_eq!(hy, 0.25);
        assert_eq!(hz, 0.4);
    }
}
