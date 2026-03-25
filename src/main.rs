use bevy::prelude::*;
use the_dark_candle::{
    behavior::BehaviorPlugin, biology::BiologyPlugin, camera::CameraPlugin,
    chemistry::ChemistryPlugin, data::DataPlugin, diagnostics::DiagnosticsPlugin,
    entities::EntityPlugin, persistence::PersistencePlugin, physics::PhysicsPlugin,
    procgen::ProcgenPlugin, social::SocialPlugin, world::WorldPlugin,
};

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
            DataPlugin,
            CameraPlugin,
            DiagnosticsPlugin,
            EntityPlugin,
            WorldPlugin,
            PhysicsPlugin,
            ChemistryPlugin,
            BiologyPlugin,
            ProcgenPlugin,
            BehaviorPlugin,
            SocialPlugin,
            PersistencePlugin,
        ))
        .run();
}
