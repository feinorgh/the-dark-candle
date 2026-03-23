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
    // SI thermal properties
    #[serde(default)]
    thermal_conductivity: f32,
    #[serde(default)]
    specific_heat_capacity: f32,
    #[serde(default)]
    latent_heat_fusion: Option<f32>,
    #[serde(default)]
    latent_heat_vaporization: Option<f32>,
    #[serde(default)]
    emissivity: f32,
    // SI mechanical properties
    #[serde(default)]
    viscosity: Option<f32>,
    #[serde(default)]
    friction_coefficient: f32,
    #[serde(default)]
    restitution: f32,
    #[serde(default)]
    youngs_modulus: Option<f32>,
    // Chemical / combustion properties
    #[serde(default)]
    heat_of_combustion: Option<f32>,
    #[serde(default)]
    molar_mass: Option<f32>,
    // Phase transition targets
    #[serde(default)]
    melted_into: Option<String>,
    #[serde(default)]
    boiled_into: Option<String>,
    #[serde(default)]
    frozen_into: Option<String>,
    #[serde(default)]
    condensed_into: Option<String>,
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

/// Validate that every .material.ron file has physically plausible SI values.
#[test]
fn all_materials_have_physically_valid_si_values() {
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
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let contents =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("Failed to read {name}: {e}"));
        let mat: MaterialData =
            ron::from_str(&contents).unwrap_or_else(|e| panic!("Failed to parse {name}: {e}"));

        // Density must be positive (kg/m³)
        assert!(
            mat.density > 0.0,
            "{name}: density must be > 0, got {}",
            mat.density
        );

        // Thermal conductivity must be non-negative (W/(m·K)); zero allowed for defaults
        assert!(
            mat.thermal_conductivity >= 0.0,
            "{name}: thermal_conductivity must be >= 0, got {}",
            mat.thermal_conductivity
        );

        // Specific heat capacity must be non-negative (J/(kg·K)); zero allowed for defaults
        assert!(
            mat.specific_heat_capacity >= 0.0,
            "{name}: specific_heat_capacity must be >= 0, got {}",
            mat.specific_heat_capacity
        );

        // Hardness 0–10 Mohs
        assert!(
            mat.hardness >= 0.0 && mat.hardness <= 10.0,
            "{name}: hardness must be 0–10 Mohs, got {}",
            mat.hardness
        );

        // Friction coefficient 0–1
        assert!(
            mat.friction_coefficient >= 0.0 && mat.friction_coefficient <= 1.0,
            "{name}: friction_coefficient must be 0–1, got {}",
            mat.friction_coefficient
        );

        // Restitution 0–1
        assert!(
            mat.restitution >= 0.0 && mat.restitution <= 1.0,
            "{name}: restitution must be 0–1, got {}",
            mat.restitution
        );

        // Emissivity 0–1
        assert!(
            mat.emissivity >= 0.0 && mat.emissivity <= 1.0,
            "{name}: emissivity must be 0–1, got {}",
            mat.emissivity
        );

        // Melting point must be below boiling point where both exist
        if let (Some(mp), Some(bp)) = (mat.melting_point, mat.boiling_point) {
            assert!(
                mp < bp,
                "{name}: melting_point ({mp} K) must be < boiling_point ({bp} K)"
            );
        }

        // Latent heats must be positive where defined
        if let Some(lf) = mat.latent_heat_fusion {
            assert!(lf > 0.0, "{name}: latent_heat_fusion must be > 0, got {lf}");
        }
        if let Some(lv) = mat.latent_heat_vaporization {
            assert!(
                lv > 0.0,
                "{name}: latent_heat_vaporization must be > 0, got {lv}"
            );
        }

        // Heat of combustion must be positive where defined
        if let Some(hc) = mat.heat_of_combustion {
            assert!(hc > 0.0, "{name}: heat_of_combustion must be > 0, got {hc}");
        }

        // Viscosity must be positive where defined
        if let Some(v) = mat.viscosity {
            assert!(v > 0.0, "{name}: viscosity must be > 0, got {v}");
        }

        // Molar mass must be positive where defined
        if let Some(mm) = mat.molar_mass {
            assert!(mm > 0.0, "{name}: molar_mass must be > 0, got {mm}");
        }

        // Young's modulus must be positive where defined
        if let Some(ym) = mat.youngs_modulus {
            assert!(ym > 0.0, "{name}: youngs_modulus must be > 0, got {ym}");
        }
    }
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct ReactionData {
    name: String,
    input_a: String,
    input_b: Option<String>,
    min_temperature: f32,
    max_temperature: f32,
    output_a: String,
    output_b: Option<String>,
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
    primary_material: String,
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

// --- Biome RON validation ---

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct SpawnEntry {
    id: String,
    weight: f32,
    max_per_chunk: u32,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct BiomeData {
    name: String,
    display_name: String,
    height_range: (f32, f32),
    temperature_range: (f32, f32),
    moisture_range: (f32, f32),
    surface_material: String,
    #[serde(default)]
    creature_spawns: Vec<SpawnEntry>,
    #[serde(default)]
    item_spawns: Vec<SpawnEntry>,
}

#[test]
fn all_biome_ron_files_are_valid() {
    let pattern = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/assets/data/biomes/*.biome.ron"
    );
    let files: Vec<_> = glob::glob(pattern)
        .expect("Failed to read glob pattern")
        .collect();

    assert!(
        !files.is_empty(),
        "No .biome.ron files found in assets/data/biomes/"
    );

    for entry in files {
        let path = entry.expect("Failed to read glob entry");
        let contents = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("Failed to read {}: {e}", path.display()));
        let data: BiomeData = ron::from_str(&contents)
            .unwrap_or_else(|e| panic!("Failed to parse {}: {e}", path.display()));
        assert!(!data.name.is_empty(), "{}: name is empty", path.display());
        assert!(
            data.height_range.0 <= data.height_range.1,
            "{}: invalid height_range",
            path.display()
        );
    }
}
