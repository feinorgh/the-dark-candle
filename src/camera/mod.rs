use bevy::prelude::*;

pub struct CameraPlugin;

impl Plugin for CameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_camera);
    }
}

fn spawn_camera(mut commands: Commands) {
    // 2D camera for now — will be replaced with 3D first-person in Phase 0.2
    commands.spawn(Camera2d);
}
