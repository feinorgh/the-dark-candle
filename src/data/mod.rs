use bevy::prelude::*;
use bevy_common_assets::ron::RonAssetPlugin;
use serde::Deserialize;

pub struct DataPlugin;

impl Plugin for DataPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(RonAssetPlugin::<EnemyData>::new(&["enemy.ron"]))
            .add_plugins(RonAssetPlugin::<MaterialData>::new(&["material.ron"]));
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
#[derive(Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
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
}
