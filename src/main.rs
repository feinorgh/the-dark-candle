mod behavior;
mod biology;
mod camera;
mod chemistry;
mod data;
mod entities;
mod persistence;
mod physics;
mod procgen;
mod social;
mod world;

use bevy::prelude::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "The Dark Candle".into(),
                resolution: (800, 600).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins((
            data::DataPlugin,
            camera::CameraPlugin,
            entities::EntityPlugin,
            world::WorldPlugin,
            physics::PhysicsPlugin,
            chemistry::ChemistryPlugin,
            biology::BiologyPlugin,
            procgen::ProcgenPlugin,
            behavior::BehaviorPlugin,
            social::SocialPlugin,
            persistence::PersistencePlugin,
        ))
        .run();
}
