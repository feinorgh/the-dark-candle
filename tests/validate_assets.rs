// Integration tests verifying all RON asset files in assets/data/ are valid.
// These tests parse every RON file directly (without Bevy's asset pipeline)
// to catch data errors early, before they become runtime panics.

use serde::Deserialize;

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct EnemyData {
    name: String,
    health: f32,
    speed: f32,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
enum Phase {
    Solid,
    Liquid,
    Gas,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct MaterialData {
    id: u16,
    name: String,
    default_phase: Phase,
    density: f32,
    melting_point: Option<f32>,
    boiling_point: Option<f32>,
    ignition_point: Option<f32>,
    hardness: f32,
    color: [f32; 3],
    transparent: bool,
}

/// Validate that every .enemy.ron file in assets/data/ deserializes correctly.
#[test]
fn all_enemy_ron_files_are_valid() {
    let pattern = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/data/**/*.enemy.ron");
    let files: Vec<_> = glob::glob(pattern)
        .expect("Failed to read glob pattern")
        .collect();

    assert!(
        !files.is_empty(),
        "No .enemy.ron files found in assets/data/"
    );

    for entry in files {
        let path = entry.expect("Failed to read glob entry");
        let contents = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("Failed to read {}: {e}", path.display()));
        let _data: EnemyData = ron::from_str(&contents)
            .unwrap_or_else(|e| panic!("Failed to parse {}: {e}", path.display()));
    }
}

/// Validate that every .material.ron file in assets/data/materials/ deserializes correctly.
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
        let _data: MaterialData = ron::from_str(&contents)
            .unwrap_or_else(|e| panic!("Failed to parse {}: {e}", path.display()));
    }
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct ReactionData {
    name: String,
    input_a: u16,
    input_b: Option<u16>,
    min_temperature: f32,
    max_temperature: f32,
    output_a: u16,
    output_b: Option<u16>,
    heat_output: f32,
}

/// Validate that every .reaction.ron file in assets/data/reactions/ deserializes correctly.
#[test]
fn all_reaction_ron_files_are_valid() {
    let pattern = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/assets/data/reactions/*.reaction.ron"
    );
    let files: Vec<_> = glob::glob(pattern)
        .expect("Failed to read glob pattern")
        .collect();

    assert!(
        !files.is_empty(),
        "No .reaction.ron files found in assets/data/reactions/"
    );

    for entry in files {
        let path = entry.expect("Failed to read glob entry");
        let contents = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("Failed to read {}: {e}", path.display()));
        let data: ReactionData = ron::from_str(&contents)
            .unwrap_or_else(|e| panic!("Failed to parse {}: {e}", path.display()));
        assert!(!data.name.is_empty(), "{}: name is empty", path.display());
        assert!(
            data.min_temperature <= data.max_temperature,
            "{}: min_temp > max_temp",
            path.display()
        );
    }
}

// --- Creature RON validation ---

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
enum Diet {
    Herbivore,
    Carnivore,
    Omnivore,
    Scavenger,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
enum BodySize {
    Tiny,
    Small,
    Medium,
    Large,
    Huge,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct CreatureData {
    species: String,
    display_name: String,
    base_health: f32,
    base_speed: f32,
    #[serde(default)]
    base_attack: f32,
    body_size: BodySize,
    diet: Diet,
    hitbox: (f32, f32, f32),
    color: [f32; 3],
    #[serde(default)]
    stat_variation: f32,
    #[serde(default)]
    preferred_biomes: Vec<String>,
    #[serde(default)]
    hostile: bool,
    #[serde(default)]
    lifespan: Option<u32>,
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
    }
}

// --- Item RON validation ---

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
enum ItemCategory {
    Tool,
    Weapon,
    Armor,
    Food,
    Material,
    Container,
    Misc,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct ItemData {
    item_type: String,
    display_name: String,
    category: ItemCategory,
    primary_material: u16,
    base_weight: f32,
    base_durability: f32,
    #[serde(default)]
    base_damage: f32,
    #[serde(default)]
    base_armor: f32,
    #[serde(default)]
    nutrition: f32,
    #[serde(default)]
    stackable: bool,
    #[serde(default)]
    max_stack: u32,
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
    }
}
