use bevy::prelude::*;
use bevy_common_assets::ron::RonAssetPlugin;
use serde::Deserialize;

pub struct DataPlugin;

impl Plugin for DataPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(RonAssetPlugin::<EnemyData>::new(&["enemy.ron"]));
    }
}

/// Raw data loaded from `.enemy.ron` files.
#[derive(Deserialize, Asset, TypePath, Debug, Clone, PartialEq)]
pub struct EnemyData {
    pub name: String,
    pub health: f32,
    pub speed: f32,
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
}
