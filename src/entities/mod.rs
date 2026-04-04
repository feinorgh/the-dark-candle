use bevy::prelude::*;

use crate::data::{EnemyData, GameAssets};

pub mod inventory;

pub struct EntityPlugin;

impl Plugin for EntityPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, load_entity_assets)
            .add_systems(Update, spawn_enemy_when_loaded);
    }
}

#[derive(Component, Debug, PartialEq)]
pub struct Enemy {
    pub speed: f32,
}

fn load_entity_assets(mut commands: Commands, asset_server: Res<AssetServer>) {
    let goblin_handle = asset_server.load("data/goblin.enemy.ron");
    commands.insert_resource(GameAssets {
        goblin_data: goblin_handle,
    });
}

fn spawn_enemy_when_loaded(
    mut commands: Commands,
    game_assets: Res<GameAssets>,
    enemy_assets: Res<Assets<EnemyData>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut has_spawned: Local<bool>,
) {
    if *has_spawned {
        return;
    }

    if let Some(goblin_data) = enemy_assets.get(&game_assets.goblin_data) {
        info!(
            "Data Loaded Successfully: {} has {} HP",
            goblin_data.name, goblin_data.health
        );

        commands.spawn((
            Enemy {
                speed: goblin_data.speed,
            },
            Mesh3d(meshes.add(Cuboid::new(0.8, 1.2, 0.8))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgb(0.8, 0.2, 0.2),
                ..default()
            })),
            Transform::from_xyz(3.0, 0.6, 0.0),
        ));

        *has_spawned = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enemy_component_stores_speed() {
        let enemy = Enemy { speed: 3.5 };
        assert_eq!(enemy.speed, 3.5);
    }

    #[test]
    fn enemy_from_data() {
        let data = EnemyData {
            name: "Test".into(),
            health: 10.0,
            speed: 5.0,
        };
        let enemy = Enemy { speed: data.speed };
        assert_eq!(enemy.speed, 5.0);
    }
}
