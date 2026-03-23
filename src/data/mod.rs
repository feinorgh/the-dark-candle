use bevy::prelude::*;
use bevy_common_assets::ron::RonAssetPlugin;
use serde::{Deserialize, Serialize};

pub struct DataPlugin;

impl Plugin for DataPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(RonAssetPlugin::<EnemyData>::new(&["enemy.ron"]))
            .add_plugins(RonAssetPlugin::<MaterialData>::new(&["material.ron"]))
            .add_plugins(RonAssetPlugin::<CreatureData>::new(&["creature.ron"]))
            .add_plugins(RonAssetPlugin::<ItemData>::new(&["item.ron"]));
    }
}

/// Raw data loaded from `.enemy.ron` files.
#[derive(Deserialize, Asset, TypePath, Debug, Clone, PartialEq)]
pub struct EnemyData {
    pub name: String,
    pub health: f32,
    pub speed: f32,
}

/// Material phase (solid, liquid, gas) at standard conditions.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Solid,
    Liquid,
    Gas,
}

/// Physical and chemical properties of a material, loaded from `.material.ron`.
/// The `id` field maps to `MaterialId` in the voxel system.
#[derive(Deserialize, Asset, TypePath, Debug, Clone, PartialEq)]
pub struct MaterialData {
    /// Numeric ID matching `MaterialId` in the voxel system (0 = air).
    pub id: u16,
    pub name: String,
    /// Phase at standard temperature/pressure.
    pub default_phase: Phase,
    /// Density in kg/m³.
    pub density: f32,
    /// Melting point in Kelvin (None for materials that don't melt, e.g. air).
    pub melting_point: Option<f32>,
    /// Boiling point in Kelvin.
    pub boiling_point: Option<f32>,
    /// Ignition temperature in Kelvin (None if non-flammable).
    pub ignition_point: Option<f32>,
    /// Structural hardness (0.0 = no resistance, 1.0 = maximum].
    pub hardness: f32,
    /// Base color for rendering (RGB, 0.0–1.0).
    pub color: [f32; 3],
    /// Whether light passes through this material.
    pub transparent: bool,
    /// MaterialId this becomes when heated above melting_point (solid → liquid).
    #[serde(default)]
    pub melted_into: Option<u16>,
    /// MaterialId this becomes when heated above boiling_point (liquid → gas).
    #[serde(default)]
    pub boiled_into: Option<u16>,
    /// MaterialId this becomes when cooled below melting_point (liquid → solid).
    #[serde(default)]
    pub frozen_into: Option<u16>,
    /// MaterialId this becomes when cooled below boiling_point (gas → liquid).
    #[serde(default)]
    pub condensed_into: Option<u16>,
}

/// Global resource holding handles to loaded data assets.
#[derive(Resource, Default)]
pub struct GameAssets {
    pub goblin_data: Handle<EnemyData>,
}

// --- Creature Data ---

/// Dietary classification for creatures.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Diet {
    Herbivore,
    Carnivore,
    Omnivore,
    Scavenger,
}

/// Size category affecting collision, visibility, and resource needs.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodySize {
    Tiny,
    Small,
    Medium,
    Large,
    Huge,
}

/// Base statistics for a creature species, loaded from `.creature.ron`.
#[derive(Deserialize, Asset, TypePath, Debug, Clone, PartialEq)]
pub struct CreatureData {
    /// Unique species identifier string (e.g., "wolf", "deer").
    pub species: String,
    /// Display name shown to the player.
    pub display_name: String,
    /// Base health points.
    pub base_health: f32,
    /// Base movement speed (voxels per second).
    pub base_speed: f32,
    /// Base attack damage (0 for passive creatures).
    #[serde(default)]
    pub base_attack: f32,
    /// Body size category.
    pub body_size: BodySize,
    /// Dietary classification.
    pub diet: Diet,
    /// Collision half-extents (x, y, z) for AABB.
    pub hitbox: (f32, f32, f32),
    /// Base color for rendering (RGB, 0.0–1.0).
    pub color: [f32; 3],
    /// How much variation is allowed in stats (0.0–1.0, fraction of base).
    #[serde(default = "default_variation")]
    pub stat_variation: f32,
    /// Preferred biome names (empty = spawns anywhere).
    #[serde(default)]
    pub preferred_biomes: Vec<String>,
    /// Whether the creature is hostile to the player by default.
    #[serde(default)]
    pub hostile: bool,
    /// Lifespan in simulation ticks (None = immortal).
    #[serde(default)]
    pub lifespan: Option<u32>,
}

fn default_variation() -> f32 {
    0.1
}

// --- Item Data ---

/// Category of item affecting how it can be used.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ItemCategory {
    Tool,
    Weapon,
    Armor,
    Food,
    Material,
    Container,
    Misc,
}

/// Template for procedural item generation, loaded from `.item.ron`.
#[derive(Deserialize, Asset, TypePath, Debug, Clone, PartialEq)]
pub struct ItemData {
    /// Unique item type identifier (e.g., "sword", "pickaxe").
    pub item_type: String,
    /// Display name template (may include "{material}" placeholder).
    pub display_name: String,
    /// Item category.
    pub category: ItemCategory,
    /// MaterialId this item is primarily made of (influences properties).
    pub primary_material: u16,
    /// Base weight in kg (modified by material density).
    pub base_weight: f32,
    /// Base durability (modified by material hardness).
    pub base_durability: f32,
    /// Base damage for weapons/tools (0 for non-weapons).
    #[serde(default)]
    pub base_damage: f32,
    /// Base armor value (0 for non-armor).
    #[serde(default)]
    pub base_armor: f32,
    /// Nutritional value if food (calories, 0 for non-food).
    #[serde(default)]
    pub nutrition: f32,
    /// Whether this item can be stacked in inventory.
    #[serde(default)]
    pub stackable: bool,
    /// Maximum stack size (only relevant if stackable).
    #[serde(default = "default_max_stack")]
    pub max_stack: u32,
}

fn default_max_stack() -> u32 {
    64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enemy_data_deserializes_from_ron() {
        let ron_str = r#"EnemyData(name: "Goblin", health: 50.0, speed: 2.5)"#;
        let data: EnemyData = ron::from_str(ron_str).expect("Failed to deserialize EnemyData");
        assert_eq!(data.name, "Goblin");
        assert_eq!(data.health, 50.0);
        assert_eq!(data.speed, 2.5);
    }

    #[test]
    fn enemy_data_deserializes_multiline_ron() {
        let ron_str = r#"
            EnemyData(
                name: "Skeleton",
                health: 100.0,
                speed: 1.5,
            )
        "#;
        let data: EnemyData = ron::from_str(ron_str).expect("Failed to deserialize EnemyData");
        assert_eq!(data.name, "Skeleton");
        assert_eq!(data.health, 100.0);
        assert_eq!(data.speed, 1.5);
    }

    #[test]
    fn enemy_data_rejects_missing_fields() {
        let ron_str = r#"EnemyData(name: "Orc", health: 80.0)"#;
        assert!(ron::from_str::<EnemyData>(ron_str).is_err());
    }

    #[test]
    fn enemy_data_clone_is_equal() {
        let data = EnemyData {
            name: "Troll".into(),
            health: 200.0,
            speed: 0.8,
        };
        assert_eq!(data, data.clone());
    }

    #[test]
    fn goblin_ron_file_is_valid() {
        let contents = include_str!("../../assets/data/goblin.enemy.ron");
        let data: EnemyData =
            ron::from_str(contents).expect("goblin.enemy.ron failed to deserialize");
        assert!(!data.name.is_empty());
        assert!(data.health > 0.0);
        assert!(data.speed > 0.0);
    }

    #[test]
    fn material_data_deserializes_from_ron() {
        let ron_str = r#"
            MaterialData(
                id: 1,
                name: "Stone",
                default_phase: Solid,
                density: 2700.0,
                melting_point: Some(1473.0),
                boiling_point: Some(2773.0),
                ignition_point: None,
                hardness: 0.9,
                color: (0.5, 0.5, 0.5),
                transparent: false,
            )
        "#;
        let data: MaterialData =
            ron::from_str(ron_str).expect("Failed to deserialize MaterialData");
        assert_eq!(data.name, "Stone");
        assert_eq!(data.id, 1);
        assert_eq!(data.default_phase, Phase::Solid);
        assert_eq!(data.melting_point, Some(1473.0));
        assert_eq!(data.ignition_point, None);
        assert!(!data.transparent);
    }

    #[test]
    fn material_data_air_has_no_melting_point() {
        let ron_str = r#"
            MaterialData(
                id: 0,
                name: "Air",
                default_phase: Gas,
                density: 1.225,
                melting_point: None,
                boiling_point: None,
                ignition_point: None,
                hardness: 0.0,
                color: (0.8, 0.9, 1.0),
                transparent: true,
            )
        "#;
        let data: MaterialData = ron::from_str(ron_str).expect("Failed to deserialize air");
        assert!(data.transparent);
        assert_eq!(data.hardness, 0.0);
        assert_eq!(data.melting_point, None);
    }

    #[test]
    fn material_data_flammable_has_ignition_point() {
        let ron_str = r#"
            MaterialData(
                id: 5,
                name: "Wood",
                default_phase: Solid,
                density: 600.0,
                melting_point: None,
                boiling_point: None,
                ignition_point: Some(573.0),
                hardness: 0.3,
                color: (0.6, 0.4, 0.2),
                transparent: false,
            )
        "#;
        let data: MaterialData = ron::from_str(ron_str).expect("Failed to deserialize wood");
        assert_eq!(data.ignition_point, Some(573.0));
    }

    #[test]
    fn all_material_ron_files_are_valid() {
        let pattern = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/data/materials/*.material.ron"
        );
        let files: Vec<_> = glob::glob(pattern)
            .expect("Failed to read glob pattern")
            .collect();

        assert!(
            !files.is_empty(),
            "No .material.ron files found in assets/data/materials/"
        );

        for entry in files {
            let path = entry.expect("Failed to read glob entry");
            let contents = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("Failed to read {}: {e}", path.display()));
            let data: MaterialData = ron::from_str(&contents)
                .unwrap_or_else(|e| panic!("Failed to parse {}: {e}", path.display()));
            assert!(!data.name.is_empty(), "{}: name is empty", path.display());
            assert!(data.density >= 0.0, "{}: negative density", path.display());
            assert!(
                data.hardness >= 0.0 && data.hardness <= 1.0,
                "{}: hardness out of range",
                path.display()
            );
        }
    }

    // --- CreatureData tests ---

    #[test]
    fn creature_data_deserializes_from_ron() {
        let ron_str = r#"
            CreatureData(
                species: "wolf",
                display_name: "Grey Wolf",
                base_health: 80.0,
                base_speed: 6.0,
                base_attack: 15.0,
                body_size: Medium,
                diet: Carnivore,
                hitbox: (0.4, 0.5, 0.8),
                color: (0.5, 0.5, 0.5),
                hostile: true,
            )
        "#;
        let data: CreatureData =
            ron::from_str(ron_str).expect("Failed to deserialize CreatureData");
        assert_eq!(data.species, "wolf");
        assert_eq!(data.base_health, 80.0);
        assert_eq!(data.diet, Diet::Carnivore);
        assert_eq!(data.body_size, BodySize::Medium);
        assert!(data.hostile);
        assert_eq!(data.stat_variation, 0.1); // default
        assert!(data.preferred_biomes.is_empty()); // default
        assert!(data.lifespan.is_none()); // default
    }

    #[test]
    fn creature_data_with_all_fields() {
        let ron_str = r#"
            CreatureData(
                species: "deer",
                display_name: "Forest Deer",
                base_health: 50.0,
                base_speed: 8.0,
                base_attack: 0.0,
                body_size: Large,
                diet: Herbivore,
                hitbox: (0.5, 0.7, 1.0),
                color: (0.6, 0.4, 0.2),
                stat_variation: 0.2,
                preferred_biomes: ["forest", "meadow"],
                hostile: false,
                lifespan: Some(50000),
            )
        "#;
        let data: CreatureData = ron::from_str(ron_str).expect("Failed to deserialize deer");
        assert_eq!(data.stat_variation, 0.2);
        assert_eq!(data.preferred_biomes, vec!["forest", "meadow"]);
        assert_eq!(data.lifespan, Some(50000));
        assert!(!data.hostile);
    }

    #[test]
    fn creature_data_rejects_missing_species() {
        let ron_str = r#"CreatureData(display_name: "?", base_health: 1.0, base_speed: 1.0, body_size: Tiny, diet: Omnivore, hitbox: (0.1, 0.1, 0.1), color: (1.0, 1.0, 1.0))"#;
        assert!(ron::from_str::<CreatureData>(ron_str).is_err());
    }

    #[test]
    fn all_creature_ron_files_are_valid() {
        let pattern = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/data/creatures/*.creature.ron"
        );
        let files: Vec<_> = glob::glob(pattern)
            .expect("Failed to read glob pattern")
            .collect();

        assert!(
            !files.is_empty(),
            "No .creature.ron files found in assets/data/creatures/"
        );

        for entry in files {
            let path = entry.expect("Failed to read glob entry");
            let contents = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("Failed to read {}: {e}", path.display()));
            let data: CreatureData = ron::from_str(&contents)
                .unwrap_or_else(|e| panic!("Failed to parse {}: {e}", path.display()));
            assert!(
                !data.species.is_empty(),
                "{}: species is empty",
                path.display()
            );
            assert!(
                data.base_health > 0.0,
                "{}: health must be positive",
                path.display()
            );
            assert!(
                data.base_speed > 0.0,
                "{}: speed must be positive",
                path.display()
            );
        }
    }

    // --- ItemData tests ---

    #[test]
    fn item_data_deserializes_from_ron() {
        let ron_str = r#"
            ItemData(
                item_type: "sword",
                display_name: "Iron Sword",
                category: Weapon,
                primary_material: 4,
                base_weight: 2.5,
                base_durability: 100.0,
                base_damage: 20.0,
            )
        "#;
        let data: ItemData = ron::from_str(ron_str).expect("Failed to deserialize ItemData");
        assert_eq!(data.item_type, "sword");
        assert_eq!(data.category, ItemCategory::Weapon);
        assert_eq!(data.primary_material, 4); // Iron
        assert_eq!(data.base_damage, 20.0);
        assert!(!data.stackable); // default
        assert_eq!(data.max_stack, 64); // default
    }

    #[test]
    fn item_data_food_with_nutrition() {
        let ron_str = r#"
            ItemData(
                item_type: "apple",
                display_name: "Apple",
                category: Food,
                primary_material: 0,
                base_weight: 0.2,
                base_durability: 10.0,
                nutrition: 150.0,
                stackable: true,
                max_stack: 16,
            )
        "#;
        let data: ItemData = ron::from_str(ron_str).expect("Failed to deserialize food");
        assert_eq!(data.category, ItemCategory::Food);
        assert_eq!(data.nutrition, 150.0);
        assert!(data.stackable);
        assert_eq!(data.max_stack, 16);
    }

    #[test]
    fn item_data_rejects_missing_fields() {
        let ron_str = r#"ItemData(item_type: "rock", display_name: "Rock")"#;
        assert!(ron::from_str::<ItemData>(ron_str).is_err());
    }

    #[test]
    fn all_item_ron_files_are_valid() {
        let pattern = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/data/items/*.item.ron");
        let files: Vec<_> = glob::glob(pattern)
            .expect("Failed to read glob pattern")
            .collect();

        assert!(
            !files.is_empty(),
            "No .item.ron files found in assets/data/items/"
        );

        for entry in files {
            let path = entry.expect("Failed to read glob entry");
            let contents = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("Failed to read {}: {e}", path.display()));
            let data: ItemData = ron::from_str(&contents)
                .unwrap_or_else(|e| panic!("Failed to parse {}: {e}", path.display()));
            assert!(
                !data.item_type.is_empty(),
                "{}: item_type is empty",
                path.display()
            );
            assert!(
                data.base_weight >= 0.0,
                "{}: negative weight",
                path.display()
            );
            assert!(
                data.base_durability >= 0.0,
                "{}: negative durability",
                path.display()
            );
        }
    }
}
