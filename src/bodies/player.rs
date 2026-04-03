// Player embodiment — attaches skeletal body to the player camera entity.
//
// The player entity (spawned by `CameraPlugin`) is a first-person camera.
// This module watches for new `Player` entities and attaches a humanoid
// `Skeleton` + gait state. Head-bob is applied each tick by offsetting the
// camera's translation along its local Y axis based on the gait phase.

use bevy::prelude::*;

use crate::hud::Player;

use super::locomotion::{GaitDataHandle, GaitMode, GaitState};
use super::skeleton::SkeletonHandle;

// ---------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------

/// Path to the humanoid skeleton RON asset.
#[derive(Resource)]
pub struct HumanoidSkeletonPath(pub String);

impl Default for HumanoidSkeletonPath {
    fn default() -> Self {
        Self("data/skeletons/humanoid.skeleton.ron".into())
    }
}

/// Path to the biped gait RON asset.
#[derive(Resource)]
pub struct BipedGaitPath(pub String);

impl Default for BipedGaitPath {
    fn default() -> Self {
        Self("data/gaits/biped.gait.ron".into())
    }
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

/// Attach skeletal body components to newly spawned `Player` entities.
///
/// Fires once when the camera is created (there is no `Skeleton` yet).
pub fn attach_player_body(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    skeleton_path: Res<HumanoidSkeletonPath>,
    gait_path: Res<BipedGaitPath>,
    query: Query<Entity, (With<Player>, Without<SkeletonHandle>)>,
) {
    for entity in &query {
        let skeleton_handle =
            asset_server.load::<super::skeleton::SkeletonData>(skeleton_path.0.clone());
        let gait_handle = asset_server.load::<super::locomotion::GaitData>(gait_path.0.clone());

        commands.entity(entity).insert((
            SkeletonHandle(skeleton_handle),
            GaitDataHandle(gait_handle),
            GaitState {
                mode: GaitMode::Idle,
                phase: 0.0,
                speed: 0.0,
            },
        ));

        debug!(
            "Attached humanoid skeleton body to player entity {:?}",
            entity
        );
    }
}

/// Update `GaitState` from the player's `FpsCamera` speed field.
///
/// The camera system already computes `cam.speed` (WALK_SPEED / SPRINT_SPEED).
/// We read that here and map it to the gait mode + speed.
pub fn player_gait_from_velocity(
    mut query: Query<(&mut GaitState, &crate::camera::FpsCamera), With<Player>>,
) {
    for (mut gait, cam) in &mut query {
        let speed = cam.speed;
        gait.speed = speed;
        gait.mode = if !cam.grounded && cam.gravity_enabled {
            // In the air — keep last ground gait for animation continuity.
            gait.mode.clone()
        } else if speed < 0.5 {
            GaitMode::Idle
        } else if speed < 6.0 {
            GaitMode::Walk
        } else {
            GaitMode::Run
        };
    }
}

/// Apply a subtle head-bob to the camera based on gait phase.
///
/// Offsets the camera Y position with a sine wave scaled by gait phase.
/// Amplitude is zero when idle so there is no jitter when standing still.
pub fn apply_head_bob(mut query: Query<(&GaitState, &mut Transform), With<Player>>) {
    for (gait, mut transform) in &mut query {
        let amplitude_m = match gait.mode {
            GaitMode::Walk => 0.04,
            GaitMode::Run => 0.08,
            _ => 0.0,
        };

        if amplitude_m > 1e-4 {
            // Two steps per full gait cycle → frequency = 2.
            let bob = (gait.phase * std::f32::consts::TAU * 2.0).sin() * amplitude_m;
            transform.translation.y += bob;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn mode_from_speed(speed: f32, grounded: bool, gravity: bool, prev: GaitMode) -> GaitMode {
        if !grounded && gravity {
            prev
        } else if speed < 0.5 {
            GaitMode::Idle
        } else if speed < 6.0 {
            GaitMode::Walk
        } else {
            GaitMode::Run
        }
    }

    #[test]
    fn gait_idle_at_zero_speed() {
        assert_eq!(
            mode_from_speed(0.0, true, true, GaitMode::Idle),
            GaitMode::Idle
        );
    }

    #[test]
    fn gait_walk_at_moderate_speed() {
        assert_eq!(
            mode_from_speed(5.0, true, true, GaitMode::Idle),
            GaitMode::Walk
        );
    }

    #[test]
    fn gait_run_at_sprint_speed() {
        assert_eq!(
            mode_from_speed(8.0, true, true, GaitMode::Idle),
            GaitMode::Run
        );
    }

    #[test]
    fn gait_unchanged_when_airborne() {
        assert_eq!(
            mode_from_speed(5.0, false, true, GaitMode::Walk),
            GaitMode::Walk
        );
    }

    #[test]
    fn head_bob_zero_when_idle() {
        let amplitude = match GaitMode::Idle {
            GaitMode::Walk => 0.04_f32,
            GaitMode::Run => 0.08,
            _ => 0.0,
        };
        assert_eq!(amplitude, 0.0);
    }
}
