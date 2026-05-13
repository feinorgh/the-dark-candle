//! Procedural-body animation: drives [`super::procedural_body::BodyPart`]
//! transforms each tick from the parent's [`super::locomotion::GaitState`].
//!
//! This is the visual counterpart to the gait-phase advance system. It takes
//! the already-advanced phase + speed + mode and computes per-part transform
//! offsets from rest pose using simple sine-wave kinematics.
//!
//! # Animation model
//!
//! - **Legs** swing fore/aft around their attachment point (rotation around
//!   the local X axis) with a per-leg phase offset implementing a trot-style
//!   diagonal pattern: front-left and back-right in phase; front-right and
//!   back-left antiphase. Hexapod uses a tripod gait. Bipeds alternate.
//! - **Torso** bobs up/down with twice the leg frequency and yaws slightly
//!   into the swing.
//! - **Head** counter-rotates a tiny amount to look stable.
//! - **Tail** sways gently.
//!
//! Amplitudes scale with `gait.speed` (clamped) so idle creatures animate
//! subtly and running creatures animate visibly without exploding.

use bevy::prelude::*;

use super::locomotion::{GaitMode, GaitState};
use super::procedural_body::{BodyAxisPos, BodyPart, BodyPartKind, BodySide, ProceduralBody};

/// Speed (m/s) at which leg swing reaches full amplitude.
const FULL_SWING_SPEED: f32 = 4.0;

/// Maximum leg pitch amplitude (radians) at full swing.
const LEG_SWING_RAD: f32 = 0.6;

/// Maximum torso vertical bob (metres) at full swing.
const TORSO_BOB_M: f32 = 0.04;

/// Maximum tail yaw sway (radians) at full swing.
const TAIL_SWAY_RAD: f32 = 0.35;

/// Idle breathing amplitude scale (fraction of full).
const IDLE_AMPLITUDE: f32 = 0.12;

/// Phase offset (cycles, 0–1) for each leg in a quadruped trot.
fn quadruped_leg_phase_offset(side: BodySide, fb: BodyAxisPos) -> f32 {
    match (side, fb) {
        (BodySide::Left, BodyAxisPos::Front) => 0.0,
        (BodySide::Right, BodyAxisPos::Back) => 0.0,
        (BodySide::Right, BodyAxisPos::Front) => 0.5,
        (BodySide::Left, BodyAxisPos::Back) => 0.5,
        // Mid legs (only used by hexapod) handled in caller.
        _ => 0.0,
    }
}

/// Phase offset (cycles, 0–1) for each leg in a hexapod tripod gait:
/// FL / MR / BL move together; FR / ML / BR move together (offset 0.5).
fn hexapod_leg_phase_offset(side: BodySide, fb: BodyAxisPos) -> f32 {
    match (side, fb) {
        (BodySide::Left, BodyAxisPos::Front) => 0.0,
        (BodySide::Right, BodyAxisPos::Mid) => 0.0,
        (BodySide::Left, BodyAxisPos::Back) => 0.0,
        (BodySide::Right, BodyAxisPos::Front) => 0.5,
        (BodySide::Left, BodyAxisPos::Mid) => 0.5,
        (BodySide::Right, BodyAxisPos::Back) => 0.5,
    }
}

/// Animation amplitude as a function of speed and mode.
///
/// Idle returns a small constant so creatures still breathe.
fn amplitude_for(state: &GaitState) -> f32 {
    if state.mode == GaitMode::Idle || state.speed < 0.01 {
        IDLE_AMPLITUDE
    } else {
        (state.speed / FULL_SWING_SPEED).clamp(IDLE_AMPLITUDE, 1.0)
    }
}

/// Compute the local transform offset for one body part given the parent's
/// gait state. Returned transform is **multiplied onto the rest pose**:
/// final translation = rest_translation + offset.translation;
/// final rotation = rest_rotation * offset.rotation.
pub fn body_part_offset(
    plan_kind: BodyPartKind,
    plan: super::procedural_body::BodyPlan,
    state: &GaitState,
) -> Transform {
    let amp = amplitude_for(state);
    let phase = state.phase * std::f32::consts::TAU; // 0..2π

    match plan_kind {
        BodyPartKind::Leg { side, fb } => {
            let leg_phase_offset = match plan {
                super::procedural_body::BodyPlan::Quadruped => {
                    quadruped_leg_phase_offset(side, fb) * std::f32::consts::TAU
                }
                super::procedural_body::BodyPlan::Hexapod => {
                    hexapod_leg_phase_offset(side, fb) * std::f32::consts::TAU
                }
                super::procedural_body::BodyPlan::Biped => {
                    if matches!(side, BodySide::Right) {
                        std::f32::consts::PI
                    } else {
                        0.0
                    }
                }
                super::procedural_body::BodyPlan::Serpent => 0.0,
            };
            let pitch = (phase + leg_phase_offset).sin() * LEG_SWING_RAD * amp;
            Transform {
                translation: Vec3::ZERO,
                rotation: Quat::from_rotation_x(pitch),
                scale: Vec3::ONE,
            }
        }
        BodyPartKind::Torso => {
            // Bob at twice leg frequency (each footfall lifts the body).
            let bob = (phase * 2.0).sin() * TORSO_BOB_M * amp;
            let yaw = phase.sin() * 0.04 * amp;
            Transform {
                translation: Vec3::new(0.0, bob, 0.0),
                rotation: Quat::from_rotation_y(yaw),
                scale: Vec3::ONE,
            }
        }
        BodyPartKind::Head => {
            // Counter-yaw a little so the head looks more stable than the body.
            let yaw = -phase.sin() * 0.03 * amp;
            Transform {
                translation: Vec3::ZERO,
                rotation: Quat::from_rotation_y(yaw),
                scale: Vec3::ONE,
            }
        }
        BodyPartKind::Tail => {
            let yaw = phase.sin() * TAIL_SWAY_RAD * amp;
            Transform {
                translation: Vec3::ZERO,
                rotation: Quat::from_rotation_y(yaw),
                scale: Vec3::ONE,
            }
        }
    }
}

/// System: animate every procedural body's child parts from the parent's
/// `GaitState`.
pub fn animate_procedural_body(
    parents: Query<(&GaitState, &ProceduralBody, &Children)>,
    mut parts: Query<(&BodyPart, &mut Transform)>,
) {
    for (gait, body, children) in &parents {
        for child in children.iter() {
            let Ok((part, mut tx)) = parts.get_mut(child) else {
                continue;
            };
            let offset = body_part_offset(part.kind, body.plan, gait);
            tx.translation = part.rest_translation + offset.translation;
            tx.rotation = part.rest_rotation * offset.rotation;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bodies::procedural_body::BodyPlan;

    fn idle_state() -> GaitState {
        GaitState {
            mode: GaitMode::Idle,
            phase: 0.25, // arbitrary mid-cycle
            speed: 0.0,
        }
    }

    fn moving_state() -> GaitState {
        GaitState {
            mode: GaitMode::Walk,
            phase: 0.25,
            speed: 2.0,
        }
    }

    #[test]
    fn idle_amplitude_is_small_but_nonzero() {
        let amp = amplitude_for(&idle_state());
        assert!(amp > 0.0);
        assert!(amp <= IDLE_AMPLITUDE + 1e-6);
    }

    #[test]
    fn moving_amplitude_scales_with_speed_and_clamps() {
        let mut s = moving_state();
        let a1 = amplitude_for(&s);
        s.speed = FULL_SWING_SPEED * 4.0;
        let a2 = amplitude_for(&s);
        assert!(a2 > a1);
        assert!(a2 <= 1.0 + 1e-6);
    }

    #[test]
    fn quadruped_diagonal_legs_are_in_phase_and_orthogonal_legs_antiphase() {
        let s = moving_state();
        let fl = body_part_offset(
            BodyPartKind::Leg {
                side: BodySide::Left,
                fb: BodyAxisPos::Front,
            },
            BodyPlan::Quadruped,
            &s,
        );
        let br = body_part_offset(
            BodyPartKind::Leg {
                side: BodySide::Right,
                fb: BodyAxisPos::Back,
            },
            BodyPlan::Quadruped,
            &s,
        );
        let fr = body_part_offset(
            BodyPartKind::Leg {
                side: BodySide::Right,
                fb: BodyAxisPos::Front,
            },
            BodyPlan::Quadruped,
            &s,
        );
        // FL and BR rotate the same way (diagonal pair).
        let fl_pitch = fl.rotation.to_euler(EulerRot::XYZ).0;
        let br_pitch = br.rotation.to_euler(EulerRot::XYZ).0;
        let fr_pitch = fr.rotation.to_euler(EulerRot::XYZ).0;
        assert!((fl_pitch - br_pitch).abs() < 1e-4);
        // FL and FR should be antiphase: opposite signs (or both zero at the
        // crossing). Since phase=0.25 → sin(π/2)=1, pair offset π → -1.
        assert!(fl_pitch * fr_pitch < 0.0);
    }

    #[test]
    fn nonzero_speed_produces_nonzero_leg_offset() {
        let s = moving_state();
        let off = body_part_offset(
            BodyPartKind::Leg {
                side: BodySide::Left,
                fb: BodyAxisPos::Front,
            },
            BodyPlan::Quadruped,
            &s,
        );
        let pitch = off.rotation.to_euler(EulerRot::XYZ).0;
        assert!(
            pitch.abs() > 0.05,
            "expected leg pitch movement, got {pitch}"
        );
    }

    #[test]
    fn animate_system_updates_child_transforms() {
        let mut app = App::new();
        app.add_systems(Update, animate_procedural_body);

        let parent = app
            .world_mut()
            .spawn((
                GaitState {
                    mode: GaitMode::Walk,
                    phase: 0.25,
                    speed: 2.0,
                },
                ProceduralBody {
                    plan: BodyPlan::Quadruped,
                },
                Transform::default(),
                Visibility::default(),
            ))
            .id();

        let rest_translation = Vec3::new(0.5, -0.4, 0.5);
        let rest_rotation = Quat::IDENTITY;
        let child = app
            .world_mut()
            .spawn((
                BodyPart {
                    kind: BodyPartKind::Leg {
                        side: BodySide::Left,
                        fb: BodyAxisPos::Front,
                    },
                    rest_translation,
                    rest_rotation,
                },
                Transform {
                    translation: rest_translation,
                    rotation: rest_rotation,
                    scale: Vec3::ONE,
                },
            ))
            .id();
        app.world_mut().entity_mut(parent).add_child(child);

        app.update();

        let tx = app.world().get::<Transform>(child).unwrap();
        // Translation should still equal rest (we only added a rotation
        // offset for legs).
        assert!((tx.translation - rest_translation).length() < 1e-5);
        // Rotation should differ from rest.
        assert!(tx.rotation.angle_between(rest_rotation) > 0.01);
    }

    /// End-to-end pipeline smoke test:
    /// `ai_gait_from_velocity` (PhysicsBody → GaitState.speed)
    ///   → `advance_gait_phase` (dt-driven phase update + mode selection)
    ///   → `animate_procedural_body` (GaitState → child Transform).
    ///
    /// Spawns a `Creature` with non-zero horizontal `PhysicsBody.velocity`
    /// and a body parented child via `spawn_procedural_body`, runs the
    /// chained schedule, and asserts at least one body part transform
    /// deviates from rest. Verifies the AI animation pipeline end-to-end
    /// without pulling in the full `BodiesPlugin` (which would load RON
    /// assets and the player system).
    #[test]
    fn ai_creature_pipeline_animates_body_part_from_velocity() {
        use crate::bodies::locomotion::{advance_gait_phase, ai_gait_from_velocity};
        use crate::bodies::procedural_body::spawn_procedural_body;
        use crate::data::{BodySize, Diet};
        use crate::physics::gravity::PhysicsBody;
        use crate::procgen::creatures::Creature;
        use bevy::ecs::system::RunSystemOnce;

        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        // Need Assets<Mesh> / Assets<StandardMaterial> registered for
        // spawn_procedural_body, and Assets<GaitData> for advance_gait_phase.
        app.init_resource::<Assets<Mesh>>();
        app.init_resource::<Assets<StandardMaterial>>();
        app.init_resource::<Assets<crate::bodies::locomotion::GaitData>>();

        app.add_systems(
            Update,
            (
                ai_gait_from_velocity,
                advance_gait_phase.after(ai_gait_from_velocity),
                animate_procedural_body.after(advance_gait_phase),
            ),
        );

        let creature = Creature {
            species: "test".into(),
            display_name: "Test".into(),
            health: 10.0,
            max_health: 10.0,
            speed: 2.0,
            attack: 0.0,
            body_size: BodySize::Medium,
            diet: Diet::Herbivore,
            color: [1.0, 1.0, 1.0],
            hostile: false,
            lifespan: None,
            age: 0,
        };
        let physics = PhysicsBody {
            velocity: Vec3::new(2.0, 0.0, 0.0),
            ..PhysicsBody::default()
        };

        let creature_entity = app
            .world_mut()
            .spawn((
                creature,
                physics,
                GaitState::default(),
                Transform::default(),
                Visibility::default(),
            ))
            .id();

        // Spawn body parts directly via a one-shot exclusive system.
        app.world_mut()
            .run_system_once(
                move |mut commands: Commands,
                      mut meshes: ResMut<Assets<Mesh>>,
                      mut materials: ResMut<Assets<StandardMaterial>>| {
                    spawn_procedural_body(
                        &mut commands,
                        creature_entity,
                        BodyPlan::Quadruped,
                        (1.0, 0.8, 1.5),
                        Color::WHITE,
                        &mut meshes,
                        &mut materials,
                    );
                },
            )
            .expect("run_system_once");

        // Tick the schedule a few times so phase advances off zero.
        for _ in 0..5 {
            app.update();
        }

        let world = app.world();
        let children = world.get::<Children>(creature_entity).expect("body parts");
        assert!(children.iter().count() > 0, "no body parts spawned");

        let mut found_movement = false;
        for child in children.iter() {
            let Some(part) = world.get::<BodyPart>(child) else {
                continue;
            };
            let Some(tx) = world.get::<Transform>(child) else {
                continue;
            };
            let rot_delta = tx.rotation.angle_between(part.rest_rotation);
            let trans_delta = (tx.translation - part.rest_translation).length();
            if rot_delta > 1e-3 || trans_delta > 1e-4 {
                found_movement = true;
                break;
            }
        }
        assert!(
            found_movement,
            "expected at least one BodyPart to deviate from rest after running the full \
             ai_gait_from_velocity → advance_gait_phase → animate_procedural_body pipeline"
        );

        let gait = world.get::<GaitState>(creature_entity).unwrap();
        assert!(
            gait.speed > 1.5,
            "ai_gait_from_velocity should have set GaitState.speed (got {})",
            gait.speed
        );
        assert!(
            !matches!(gait.mode, GaitMode::Idle),
            "creature moving at 2 m/s should not be Idle (mode = {:?})",
            gait.mode
        );
    }
}
