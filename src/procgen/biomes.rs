// Biome definitions and spawn tables, loaded from `.biome.ron`.
//
// Each biome controls which creatures and items spawn in its territory,
// at what density, and with what variation. Biome selection is driven by
// terrain parameters (height, temperature, moisture) computed from noise.

#![allow(dead_code)]

use bevy::prelude::*;
use bevy_common_assets::ron::RonAssetPlugin;
use serde::Deserialize;

pub struct BiomePlugin;

impl Plugin for BiomePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(RonAssetPlugin::<BiomeData>::new(&["biome.ron"]));
    }
}

/// A single entry in a biome's spawn table.
#[derive(Deserialize, Debug, Clone, PartialEq)]
pub struct SpawnEntry {
    /// Species or item_type identifier (matches CreatureData.species or ItemData.item_type).
    pub id: String,
    /// Relative spawn weight (higher = more common in this biome).
    pub weight: f32,
    /// Maximum number per chunk (density cap).
    pub max_per_chunk: u32,
}

/// Biome parameters and spawn tables, loaded from `.biome.ron`.
#[derive(Deserialize, Asset, TypePath, Debug, Clone, PartialEq)]
pub struct BiomeData {
    /// Unique biome identifier.
    pub name: String,
    /// Display name for the biome.
    pub display_name: String,
    /// Height range this biome occupies (min, max) in world units.
    pub height_range: (f32, f32),
    /// Temperature range (min, max) in Kelvin.
    pub temperature_range: (f32, f32),
    /// Moisture range (0.0 = arid, 1.0 = saturated).
    pub moisture_range: (f32, f32),
    /// Surface material name (what the top layer is made of).
    pub surface_material: String,
    /// Optional terrain modifiers for biome-specific terrain shaping.
    #[serde(default)]
    pub terrain: Option<BiomeTerrainModifiers>,
    /// Creature spawn table.
    #[serde(default)]
    pub creature_spawns: Vec<SpawnEntry>,
    /// Item spawn table (natural items like rocks, sticks, berries).
    #[serde(default)]
    pub item_spawns: Vec<SpawnEntry>,
    /// Prop spawn table (natural scenery — rocks, logs, pebbles).
    #[serde(default)]
    pub prop_spawns: Vec<SpawnEntry>,
    /// Tree spawn table (voxel trees stamped into chunks).
    #[serde(default)]
    pub tree_spawns: Vec<SpawnEntry>,
}

/// Terrain modifiers loaded from biome RON files.
///
/// These affect how terrain is generated within the biome region.
#[derive(Deserialize, Debug, Clone, PartialEq)]
pub struct BiomeTerrainModifiers {
    /// Height bias in voxels added to base terrain (negative = lower terrain).
    #[serde(default)]
    pub height_bias: f32,
    /// Roughness multiplier (0.0 = smooth, 1.0 = normal, 2.0 = very rough).
    #[serde(default = "default_roughness")]
    pub roughness: f32,
    /// Erosion rate multiplier (higher = more erosion in this biome).
    #[serde(default = "default_erosion_rate")]
    pub erosion_rate: f32,
    /// Override subsurface material (e.g. "sandstone" for deserts).
    #[serde(default)]
    pub subsurface: Option<String>,
}

fn default_roughness() -> f32 {
    1.0
}
fn default_erosion_rate() -> f32 {
    1.0
}

/// Select a spawn entry from a weighted table using a random value in [0.0, 1.0).
pub fn weighted_select(entries: &[SpawnEntry], rand_value: f32) -> Option<&SpawnEntry> {
    if entries.is_empty() {
        return None;
    }

    let total_weight: f32 = entries.iter().map(|e| e.weight).sum();
    if total_weight <= 0.0 {
        return None;
    }

    let mut threshold = rand_value * total_weight;
    for entry in entries {
        threshold -= entry.weight;
        if threshold <= 0.0 {
            return Some(entry);
        }
    }

    entries.last()
}

/// Check if a biome matches given terrain parameters.
pub fn biome_matches(biome: &BiomeData, height: f32, temperature: f32, moisture: f32) -> bool {
    height >= biome.height_range.0
        && height <= biome.height_range.1
        && temperature >= biome.temperature_range.0
        && temperature <= biome.temperature_range.1
        && moisture >= biome.moisture_range.0
        && moisture <= biome.moisture_range.1
}

/// Select the best matching biome from a list.
pub fn select_biome(
    biomes: &[BiomeData],
    height: f32,
    temperature: f32,
    moisture: f32,
) -> Option<&BiomeData> {
    biomes
        .iter()
        .find(|b| biome_matches(b, height, temperature, moisture))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn forest_biome() -> BiomeData {
        BiomeData {
            name: "forest".into(),
            display_name: "Dense Forest".into(),
            height_range: (50.0, 80.0),
            temperature_range: (260.0, 310.0),
            moisture_range: (0.4, 1.0),
            surface_material: "Grass".into(),
            terrain: None,
            creature_spawns: vec![
                SpawnEntry {
                    id: "wolf".into(),
                    weight: 2.0,
                    max_per_chunk: 3,
                },
                SpawnEntry {
                    id: "deer".into(),
                    weight: 5.0,
                    max_per_chunk: 5,
                },
                SpawnEntry {
                    id: "rabbit".into(),
                    weight: 8.0,
                    max_per_chunk: 10,
                },
            ],
            item_spawns: vec![SpawnEntry {
                id: "apple".into(),
                weight: 3.0,
                max_per_chunk: 5,
            }],
            prop_spawns: vec![],
            tree_spawns: vec![],
        }
    }

    fn cave_biome() -> BiomeData {
        BiomeData {
            name: "cave".into(),
            display_name: "Underground Cave".into(),
            height_range: (0.0, 40.0),
            temperature_range: (270.0, 290.0),
            moisture_range: (0.2, 0.8),
            surface_material: "Stone".into(),
            terrain: None,
            creature_spawns: vec![SpawnEntry {
                id: "cave_spider".into(),
                weight: 5.0,
                max_per_chunk: 4,
            }],
            item_spawns: vec![],
            prop_spawns: vec![],
            tree_spawns: vec![],
        }
    }

    #[test]
    fn biome_data_deserializes_from_ron() {
        let ron_str = r#"
            BiomeData(
                name: "meadow",
                display_name: "Open Meadow",
                height_range: (55.0, 75.0),
                temperature_range: (270.0, 310.0),
                moisture_range: (0.3, 0.7),
                surface_material: "Grass",
                creature_spawns: [
                    SpawnEntry(id: "rabbit", weight: 10.0, max_per_chunk: 15),
                ],
                item_spawns: [],
            )
        "#;
        let data: BiomeData = ron::from_str(ron_str).expect("Failed to deserialize BiomeData");
        assert_eq!(data.name, "meadow");
        assert_eq!(data.creature_spawns.len(), 1);
        assert_eq!(data.creature_spawns[0].id, "rabbit");
    }

    #[test]
    fn weighted_select_respects_weights() {
        let entries = vec![
            SpawnEntry {
                id: "rare".into(),
                weight: 1.0,
                max_per_chunk: 1,
            },
            SpawnEntry {
                id: "common".into(),
                weight: 9.0,
                max_per_chunk: 10,
            },
        ];

        // Low rand → should pick "rare" (first entry, weight 1/10)
        let selected = weighted_select(&entries, 0.05).unwrap();
        assert_eq!(selected.id, "rare");

        // High rand → should pick "common"
        let selected = weighted_select(&entries, 0.5).unwrap();
        assert_eq!(selected.id, "common");
    }

    #[test]
    fn weighted_select_empty_returns_none() {
        assert!(weighted_select(&[], 0.5).is_none());
    }

    #[test]
    fn biome_matches_in_range() {
        let b = forest_biome();
        assert!(biome_matches(&b, 65.0, 293.0, 0.6));
    }

    #[test]
    fn biome_rejects_out_of_range() {
        let b = forest_biome();
        // Too high
        assert!(!biome_matches(&b, 100.0, 293.0, 0.6));
        // Too cold
        assert!(!biome_matches(&b, 65.0, 200.0, 0.6));
        // Too dry
        assert!(!biome_matches(&b, 65.0, 293.0, 0.1));
    }

    #[test]
    fn select_biome_picks_matching() {
        let biomes = vec![forest_biome(), cave_biome()];

        let selected = select_biome(&biomes, 65.0, 293.0, 0.6);
        assert_eq!(selected.unwrap().name, "forest");

        let selected = select_biome(&biomes, 20.0, 280.0, 0.5);
        assert_eq!(selected.unwrap().name, "cave");
    }

    #[test]
    fn select_biome_returns_none_for_no_match() {
        let biomes = vec![forest_biome(), cave_biome()];
        // Extreme height — matches neither
        assert!(select_biome(&biomes, 200.0, 293.0, 0.6).is_none());
    }

    #[test]
    fn spawn_table_total_weight() {
        let b = forest_biome();
        let total: f32 = b.creature_spawns.iter().map(|e| e.weight).sum();
        assert_eq!(total, 15.0); // 2 + 5 + 8
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
}
