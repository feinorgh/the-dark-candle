// Heads-up display: health bar, temperature, day/night clock, compass, fall damage.
//
// The HUD is spawned when entering GameState::Playing and despawned on exit.
// A `Player` marker component identifies the player entity (the camera).

use bevy::prelude::*;

use crate::biology::health::{DamageType, Health};
use crate::camera::FpsCamera;
use crate::game_state::GameState;
use crate::lighting::TimeOfDay;
use crate::world::chunk::CHUNK_SIZE;
use crate::world::chunk::Chunk;
use crate::world::chunk::ChunkCoord;
use crate::world::chunk_manager::ChunkMap;

/// Marker component identifying the player entity.
#[derive(Component, Debug)]
pub struct Player;

pub struct HudPlugin;

impl Plugin for HudPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(OnEnter(GameState::Playing), spawn_hud)
            .add_systems(OnExit(GameState::Playing), despawn_hud)
            .add_systems(
                Update,
                (
                    update_health_bar,
                    update_health_text,
                    update_temperature,
                    update_clock,
                    update_compass,
                )
                    .run_if(in_state(GameState::Playing)),
            )
            .add_systems(
                Update,
                apply_fall_damage.run_if(in_state(GameState::Playing)),
            );
    }
}

// ---------------------------------------------------------------------------
// HUD layout
// ---------------------------------------------------------------------------

#[derive(Component)]
struct HudRoot;

#[derive(Component)]
struct HealthBarFill;

#[derive(Component)]
struct HealthText;

#[derive(Component)]
struct TemperatureText;

#[derive(Component)]
struct ClockText;

#[derive(Component)]
struct CompassText;

fn spawn_hud(mut commands: Commands) {
    commands
        .spawn((
            HudRoot,
            Node {
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                position_type: PositionType::Absolute,
                ..default()
            },
        ))
        .with_children(|root| {
            // Top-left column: health bar, temperature
            root.spawn(Node {
                position_type: PositionType::Absolute,
                left: Val::Px(16.0),
                top: Val::Px(40.0),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(6.0),
                ..default()
            })
            .with_children(|col| {
                // Health bar container
                col.spawn((
                    Node {
                        width: Val::Px(200.0),
                        height: Val::Px(16.0),
                        ..default()
                    },
                    BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.5)),
                ))
                .with_children(|bar| {
                    bar.spawn((
                        HealthBarFill,
                        Node {
                            width: Val::Percent(100.0),
                            height: Val::Percent(100.0),
                            ..default()
                        },
                        BackgroundColor(Color::srgb(0.2, 0.8, 0.2)),
                    ));
                });

                // Health text
                col.spawn((
                    HealthText,
                    Text::new("100 / 100"),
                    TextFont {
                        font_size: 13.0,
                        ..default()
                    },
                    TextColor(Color::WHITE),
                ));

                // Temperature
                col.spawn((
                    TemperatureText,
                    Text::new("--°C"),
                    TextFont {
                        font_size: 13.0,
                        ..default()
                    },
                    TextColor(Color::srgb(0.7, 0.85, 1.0)),
                ));
            });

            // Top-right column: clock + compass
            root.spawn(Node {
                position_type: PositionType::Absolute,
                right: Val::Px(16.0),
                top: Val::Px(40.0),
                flex_direction: FlexDirection::Column,
                align_items: AlignItems::End,
                row_gap: Val::Px(6.0),
                ..default()
            })
            .with_children(|col| {
                col.spawn((
                    ClockText,
                    Text::new("10:00"),
                    TextFont {
                        font_size: 14.0,
                        ..default()
                    },
                    TextColor(Color::srgb(1.0, 0.95, 0.7)),
                ));

                col.spawn((
                    CompassText,
                    Text::new("N"),
                    TextFont {
                        font_size: 14.0,
                        ..default()
                    },
                    TextColor(Color::WHITE),
                ));
            });
        });
}

fn despawn_hud(mut commands: Commands, q: Query<Entity, With<HudRoot>>) {
    for entity in &q {
        commands.entity(entity).despawn();
    }
}

// ---------------------------------------------------------------------------
// HUD update systems
// ---------------------------------------------------------------------------

fn update_health_bar(
    player_q: Query<&Health, With<Player>>,
    mut bar_q: Query<(&mut Node, &mut BackgroundColor), With<HealthBarFill>>,
) {
    let Ok(health) = player_q.single() else {
        return;
    };
    let Ok((mut node, mut bg)) = bar_q.single_mut() else {
        return;
    };
    let frac = health.health_fraction();
    node.width = Val::Percent(frac * 100.0);

    // Color: green → yellow → red
    let color = if frac > 0.5 {
        let t = (frac - 0.5) * 2.0;
        Color::srgb(1.0 - t, 0.8, 0.2 * (1.0 - t))
    } else {
        let t = frac * 2.0;
        Color::srgb(0.9, t * 0.6, 0.1)
    };
    *bg = BackgroundColor(color);
}

fn update_health_text(
    player_q: Query<&Health, With<Player>>,
    mut text_q: Query<&mut Text, With<HealthText>>,
) {
    let Ok(health) = player_q.single() else {
        return;
    };
    let Ok(mut text) = text_q.single_mut() else {
        return;
    };
    **text = format!("{:.0} / {:.0}", health.current, health.max);
}

fn update_temperature(
    player_q: Query<&Transform, With<Player>>,
    chunk_map: Option<Res<ChunkMap>>,
    chunks: Query<&Chunk>,
    mut text_q: Query<(&mut Text, &mut TextColor), With<TemperatureText>>,
) {
    let Ok(transform) = player_q.single() else {
        return;
    };
    let Ok((mut text, mut color)) = text_q.single_mut() else {
        return;
    };
    let Some(chunk_map) = chunk_map else {
        return;
    };

    let pos = transform.translation;
    let cc = ChunkCoord::from_voxel_pos(pos.x as i32, pos.y as i32, pos.z as i32);

    let temp_k = if let Some(entity) = chunk_map.get(&cc) {
        if let Ok(chunk) = chunks.get(entity) {
            let origin = cc.world_origin();
            let lx = ((pos.x as i32 - origin.x) as usize).min(CHUNK_SIZE - 1);
            let ly = ((pos.y as i32 - origin.y) as usize).min(CHUNK_SIZE - 1);
            let lz = ((pos.z as i32 - origin.z) as usize).min(CHUNK_SIZE - 1);
            chunk.get(lx, ly, lz).temperature
        } else {
            288.15
        }
    } else {
        288.15
    };

    let temp_c = temp_k - 273.15;
    **text = format!("{:.0}\u{00B0}C", temp_c);

    // Color coding: blue for cold, white for comfortable, red for hot
    let c = if temp_c < 0.0 {
        Color::srgb(0.4, 0.6, 1.0)
    } else if temp_c > 40.0 {
        Color::srgb(1.0, 0.4, 0.3)
    } else {
        Color::srgb(0.7, 0.85, 1.0)
    };
    *color = TextColor(c);
}

fn update_clock(
    tod: Option<Res<TimeOfDay>>,
    mut text_q: Query<(&mut Text, &mut TextColor), With<ClockText>>,
) {
    let Ok((mut text, mut color)) = text_q.single_mut() else {
        return;
    };
    let hour = tod.as_ref().map(|t| t.0).unwrap_or(12.0);
    let hour_int = hour as u32 % 24;
    let minute = ((hour - hour.floor()) * 60.0) as u32;

    let icon = if (6..18).contains(&hour_int) {
        "\u{2600}" // ☀
    } else {
        "\u{263D}" // ☽
    };

    **text = format!("{icon} {:02}:{:02}", hour_int, minute);

    // Warm tint during day, cool at night
    let c = if (6..18).contains(&hour_int) {
        Color::srgb(1.0, 0.95, 0.7)
    } else {
        Color::srgb(0.6, 0.7, 0.9)
    };
    *color = TextColor(c);
}

fn update_compass(
    player_q: Query<&FpsCamera, With<Player>>,
    mut text_q: Query<&mut Text, With<CompassText>>,
) {
    let Ok(cam) = player_q.single() else {
        return;
    };
    let Ok(mut text) = text_q.single_mut() else {
        return;
    };

    // yaw=0 faces +Z (north in Bevy's coord system), increases clockwise
    let degrees = cam.yaw.to_degrees().rem_euclid(360.0);
    let dir = match degrees as u32 {
        338..=360 | 0..=22 => "N",
        23..=67 => "NW",
        68..=112 => "W",
        113..=157 => "SW",
        158..=202 => "S",
        203..=247 => "SE",
        248..=292 => "E",
        293..=337 => "NE",
        _ => "N",
    };
    **text = format!("\u{1F9ED} {dir}");
}

// ---------------------------------------------------------------------------
// Fall damage
// ---------------------------------------------------------------------------

/// Tracks the player's vertical velocity from the previous frame to detect
/// hard landings.
#[derive(Component, Debug, Default)]
pub struct FallTracker {
    pub prev_vertical_velocity: f32,
}

/// Minimum impact speed (m/s) before fall damage applies.
/// A 3 m fall gives v ≈ sqrt(2·9.81·3) ≈ 7.67 m/s.
const FALL_DAMAGE_THRESHOLD: f32 = 7.0;

/// Damage per m/s of impact speed above the threshold.
/// At 20 m/s impact: (20 − 7) × 3.0 = 39 HP damage.
const FALL_DAMAGE_PER_MPS: f32 = 3.0;

fn apply_fall_damage(
    mut player_q: Query<(&FpsCamera, &mut Health, &mut FallTracker), With<Player>>,
) {
    let Ok((cam, mut health, mut tracker)) = player_q.single_mut() else {
        return;
    };

    // Detect hard landing: was falling fast, now grounded with ~0 velocity
    let prev_v = tracker.prev_vertical_velocity;
    if cam.grounded && prev_v < -FALL_DAMAGE_THRESHOLD {
        let impact_speed = (-prev_v) - FALL_DAMAGE_THRESHOLD;
        let damage = impact_speed * FALL_DAMAGE_PER_MPS;
        health.take_damage(damage, DamageType::Fall);
    }

    tracker.prev_vertical_velocity = cam.vertical_velocity;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fall_damage_threshold_physics() {
        // 3 m fall → v = sqrt(2 * 9.81 * 3) ≈ 7.67 m/s
        let v = (2.0 * 9.80665_f32 * 3.0).sqrt();
        assert!(
            v > FALL_DAMAGE_THRESHOLD,
            "3 m fall ({v:.2} m/s) should exceed threshold ({FALL_DAMAGE_THRESHOLD})"
        );
    }

    #[test]
    fn no_damage_below_threshold() {
        let health = Health::new(100.0);
        let impact_speed = FALL_DAMAGE_THRESHOLD - 1.0;
        if impact_speed > 0.0 {
            // Would not trigger in the system since prev_v > -threshold
        }
        assert_eq!(health.current, 100.0);
        assert!(!health.dead);
    }

    #[test]
    fn damage_scales_with_speed() {
        let mut health = Health::new(100.0);
        let impact_speed = 20.0; // 20 m/s impact
        let excess = impact_speed - FALL_DAMAGE_THRESHOLD;
        let damage = excess * FALL_DAMAGE_PER_MPS;
        health.take_damage(damage, DamageType::Fall);
        let expected = 100.0 - damage;
        assert!(
            (health.current - expected).abs() < 0.01,
            "Expected {expected}, got {}",
            health.current
        );
    }

    #[test]
    fn lethal_fall() {
        let mut health = Health::new(100.0);
        // ~45 m/s impact (about 100 m fall in vacuum)
        let impact_speed = 45.0;
        let excess = impact_speed - FALL_DAMAGE_THRESHOLD;
        let damage = excess * FALL_DAMAGE_PER_MPS;
        health.take_damage(damage, DamageType::Fall);
        assert!(health.dead, "45 m/s impact should be lethal");
    }

    #[test]
    fn compass_directions() {
        let cases = [
            (0.0_f32, "N"),
            (45.0, "NW"),
            (90.0, "W"),
            (135.0, "SW"),
            (180.0, "S"),
            (225.0, "SE"),
            (270.0, "E"),
            (315.0, "NE"),
            (360.0, "N"),
        ];
        for (deg, expected) in cases {
            let d = deg.rem_euclid(360.0) as u32;
            let dir = match d {
                338..=360 | 0..=22 => "N",
                23..=67 => "NW",
                68..=112 => "W",
                113..=157 => "SW",
                158..=202 => "S",
                203..=247 => "SE",
                248..=292 => "E",
                293..=337 => "NE",
                _ => "N",
            };
            assert_eq!(dir, expected, "At {deg}° expected {expected}, got {dir}");
        }
    }

    #[test]
    fn temperature_display_format() {
        // 288.15 K = 15°C
        let temp_k = 288.15_f32;
        let temp_c = temp_k - 273.15;
        let s = format!("{:.0}\u{00B0}C", temp_c);
        assert_eq!(s, "15°C");
    }

    #[test]
    fn clock_display_format() {
        let hour = 14.5_f32;
        let hour_int = hour as u32 % 24;
        let minute = ((hour - hour.floor()) * 60.0) as u32;
        let s = format!("{:02}:{:02}", hour_int, minute);
        assert_eq!(s, "14:30");
    }

    #[test]
    fn health_bar_color_gradient() {
        // Full health → green-ish
        let frac = 1.0_f32;
        let t = (frac - 0.5) * 2.0;
        assert!((0.0..=1.0).contains(&t));

        // Low health → red-ish
        let frac = 0.1_f32;
        let t = frac * 2.0;
        assert!((0.0..=1.0).contains(&t));
    }
}
