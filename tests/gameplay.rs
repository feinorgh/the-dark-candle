// Headless Bevy ECS integration tests for physics behaviour.
//
// Each test spins up a minimal Bevy `App` with only the physics plugin —
// no window, no renderer, no asset server. An empty `ChunkMap` is inserted so
// the gravity and collision systems have the resource they expect, but find no
// terrain, which lets entities fall freely.

use std::time::Duration;

use bevy::prelude::*;
use bevy::time::TimeUpdateStrategy;

use the_dark_candle::physics::gravity::{PhysicsBody, TERMINAL_VELOCITY};
use the_dark_candle::physics::PhysicsPlugin;
use the_dark_candle::world::chunk_manager::ChunkMap;

/// Build a minimal physics app with no display and deterministic time steps.
fn physics_app() -> App {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_plugins(PhysicsPlugin)
        .init_resource::<ChunkMap>()
        // Each call to app.update() advances virtual time by exactly 1/60 s.
        .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f64(
            1.0 / 60.0,
        )));
    app
}

#[test]
fn gravity_accelerates_unsupported_entity() {
    let mut app = physics_app();
    let entity = app
        .world_mut()
        .spawn((Transform::from_xyz(0.0, 100.0, 0.0), PhysicsBody::default()))
        .id();

    // ~0.5 s of simulated physics.
    for _ in 0..30 {
        app.update();
    }

    let transform = app.world().get::<Transform>(entity).unwrap();
    assert!(
        transform.translation.y < 100.0,
        "entity should have fallen below y=100, but is at y={}",
        transform.translation.y
    );
}

#[test]
fn weightless_entity_does_not_fall() {
    let mut app = physics_app();
    let entity = app
        .world_mut()
        .spawn((
            Transform::from_xyz(0.0, 100.0, 0.0),
            PhysicsBody::weightless(),
        ))
        .id();

    for _ in 0..30 {
        app.update();
    }

    let transform = app.world().get::<Transform>(entity).unwrap();
    // Zero velocity + zero gravity_scale → translation is unchanged.
    assert_eq!(
        transform.translation.y, 100.0,
        "weightless entity with no velocity should not move"
    );
}

#[test]
fn velocity_never_exceeds_terminal_velocity() {
    let mut app = physics_app();
    let entity = app
        .world_mut()
        .spawn((
            Transform::from_xyz(0.0, 10_000.0, 0.0),
            PhysicsBody::default(),
        ))
        .id();

    // ~10 s of free-fall — well past terminal velocity.
    for _ in 0..600 {
        app.update();
    }

    let body = app.world().get::<PhysicsBody>(entity).unwrap();
    assert!(
        body.velocity.y >= -TERMINAL_VELOCITY,
        "downward speed {:.2} m/s exceeds terminal velocity {:.2} m/s",
        -body.velocity.y,
        TERMINAL_VELOCITY
    );
}
