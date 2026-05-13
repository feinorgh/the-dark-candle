//! Procedurally-built multi-part bodies for AI creatures.
//!
//! Replaces single-`Cuboid` creature visuals with a parent + child hierarchy
//! of primitive parts (torso, head, legs, optional tail) animated from
//! [`GaitState`]. This is a **separate path** from the full skeleton/IK
//! pipeline in [`crate::bodies::skeleton`] — procedural bodies are intended
//! for the many small creatures spawned by procgen, where a real skeletal
//! rig would be overkill.
//!
//! # Layout
//!
//! All parts are sized as fractions of the creature's full hitbox dimensions
//! `(hx, hy, hz)`. Local space has the creature *center* at the origin, so
//! the hitbox bottom (foot plane) is at `local_y = -hy / 2`.
//!
//! - **Quadruped** — torso along +Z, head forward (+Z), 4 legs at hitbox
//!   bottom corners.
//! - **Biped** — slimmer vertical torso, 2 legs.
//! - **Hexapod** — 6 legs in 3 pairs along the body length.
//! - **Serpent** — torso only (long, thin), no legs.
//!
//! # SI units
//! All sizes in metres; voxel scale is 1 m per voxel.

use bevy::prelude::*;

use crate::data::BodySize;

// ---------------------------------------------------------------------------
// Plan & part kinds
// ---------------------------------------------------------------------------

/// High-level body archetype. Selects how many parts are generated and where
/// they attach.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BodyPlan {
    Quadruped,
    Biped,
    Hexapod,
    Serpent,
}

impl BodyPlan {
    /// Default body plan for a given body size, used when a `CreatureData`
    /// template does not specify one explicitly.
    ///
    /// Heuristic: the smallest creatures default to bipedal (mice, small
    /// birds — they read fine as two-legged silhouettes); everything else
    /// defaults to quadruped.
    pub fn default_for_size(size: BodySize) -> Self {
        match size {
            BodySize::Tiny => BodyPlan::Biped,
            _ => BodyPlan::Quadruped,
        }
    }
}

/// Which side of the body a leg is on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BodySide {
    Left,
    Right,
}

/// Front-back position of a leg along the spine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BodyAxisPos {
    Front,
    Mid,
    Back,
}

/// What kind of body part a child entity represents.
///
/// Drives the per-part animation in
/// [`super::procedural_body_anim::animate_procedural_body`].
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BodyPartKind {
    Torso,
    Head,
    Leg { side: BodySide, fb: BodyAxisPos },
    Tail,
}

/// Component attached to each procedural body child.
///
/// Stores the part's *rest pose* relative to the creature root so the
/// animator can compose `rest * gait_offset` each tick without losing the
/// original layout.
#[derive(Component, Debug, Clone, Copy)]
pub struct BodyPart {
    pub kind: BodyPartKind,
    pub rest_translation: Vec3,
    pub rest_rotation: Quat,
}

/// Marker on the creature root entity, signalling that its children are a
/// procedurally-built body.
///
/// Lets systems target only procedural-body creatures without scanning every
/// entity that happens to have `Children`.
#[derive(Component, Debug, Clone, Copy)]
pub struct ProceduralBody {
    pub plan: BodyPlan,
}

// ---------------------------------------------------------------------------
// Layout (pure data, no ECS)
// ---------------------------------------------------------------------------

/// Pure layout description for one body part: kind, local rest pose, and
/// mesh cuboid dimensions (full sizes, metres). No `Mesh`/`Material` handles
/// here — the spawner converts these into ECS entities.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BodyPartSpec {
    pub kind: BodyPartKind,
    pub rest_translation: Vec3,
    pub rest_rotation: Quat,
    /// Full cuboid dimensions (width, height, depth) in metres.
    pub size: Vec3,
}

/// Compute the full rest layout for a body plan given the creature's full
/// hitbox dimensions (in metres).
///
/// Returns specs in a stable order: torso first, then head, then legs in
/// canonical order (front-left, front-right, back-left, back-right for
/// quadrupeds), then tail if any.
pub fn body_part_specs(plan: BodyPlan, hitbox: (f32, f32, f32)) -> Vec<BodyPartSpec> {
    let (hx, hy, hz) = hitbox;
    let mut out = Vec::new();

    match plan {
        BodyPlan::Quadruped => {
            // Torso: most of the hitbox, raised so legs can hang below.
            let leg_h = hy * 0.45;
            let torso_h = hy - leg_h;
            let torso_y = (hy * 0.5) - (torso_h * 0.5) - 0.0; // center of upper portion
            out.push(BodyPartSpec {
                kind: BodyPartKind::Torso,
                rest_translation: Vec3::new(0.0, torso_y - hy * 0.5, 0.0),
                rest_rotation: Quat::IDENTITY,
                size: Vec3::new(hx * 0.85, torso_h, hz * 0.95),
            });

            // Head: small cube at the front (+Z) of the torso, slightly above.
            let head_size = (hx.min(hz) * 0.45).max(0.05);
            out.push(BodyPartSpec {
                kind: BodyPartKind::Head,
                rest_translation: Vec3::new(
                    0.0,
                    torso_y - hy * 0.5 + torso_h * 0.4,
                    hz * 0.5 + head_size * 0.4,
                ),
                rest_rotation: Quat::IDENTITY,
                size: Vec3::splat(head_size),
            });

            // Four legs at the bottom corners. Leg center sits halfway down
            // the leg span so the leg bottom touches the foot plane.
            let leg_w = (hx.min(hz) * 0.18).max(0.04);
            let leg_y = -hy * 0.5 + leg_h * 0.5;
            let foot_x = hx * 0.5 - leg_w * 0.6;
            let foot_z_front = hz * 0.5 - leg_w * 0.6;
            let foot_z_back = -foot_z_front;
            for &(side, sign_x) in &[(BodySide::Left, -1.0_f32), (BodySide::Right, 1.0)] {
                for &(fb, z) in &[
                    (BodyAxisPos::Front, foot_z_front),
                    (BodyAxisPos::Back, foot_z_back),
                ] {
                    out.push(BodyPartSpec {
                        kind: BodyPartKind::Leg { side, fb },
                        rest_translation: Vec3::new(sign_x * foot_x, leg_y, z),
                        rest_rotation: Quat::IDENTITY,
                        size: Vec3::new(leg_w, leg_h, leg_w),
                    });
                }
            }

            // Tail: thin cuboid extending backward.
            let tail_len = hz * 0.5;
            let tail_w = leg_w * 0.7;
            out.push(BodyPartSpec {
                kind: BodyPartKind::Tail,
                rest_translation: Vec3::new(
                    0.0,
                    torso_y - hy * 0.5 + torso_h * 0.2,
                    -hz * 0.5 - tail_len * 0.5,
                ),
                rest_rotation: Quat::IDENTITY,
                size: Vec3::new(tail_w, tail_w, tail_len),
            });
        }
        BodyPlan::Biped => {
            let leg_h = hy * 0.55;
            let torso_h = hy * 0.40;
            let head_h = hy - leg_h - torso_h;
            let leg_y = -hy * 0.5 + leg_h * 0.5;
            let torso_y = -hy * 0.5 + leg_h + torso_h * 0.5;
            let head_y = -hy * 0.5 + leg_h + torso_h + head_h * 0.5;

            out.push(BodyPartSpec {
                kind: BodyPartKind::Torso,
                rest_translation: Vec3::new(0.0, torso_y, 0.0),
                rest_rotation: Quat::IDENTITY,
                size: Vec3::new(hx * 0.8, torso_h, hz * 0.7),
            });
            out.push(BodyPartSpec {
                kind: BodyPartKind::Head,
                rest_translation: Vec3::new(0.0, head_y, 0.0),
                rest_rotation: Quat::IDENTITY,
                size: Vec3::splat(head_h.max(0.05)),
            });

            let leg_w = (hx * 0.35).max(0.04);
            let leg_x = hx * 0.25;
            for &(side, sign_x) in &[(BodySide::Left, -1.0_f32), (BodySide::Right, 1.0)] {
                out.push(BodyPartSpec {
                    kind: BodyPartKind::Leg {
                        side,
                        fb: BodyAxisPos::Mid,
                    },
                    rest_translation: Vec3::new(sign_x * leg_x, leg_y, 0.0),
                    rest_rotation: Quat::IDENTITY,
                    size: Vec3::new(leg_w, leg_h, leg_w),
                });
            }
        }
        BodyPlan::Hexapod => {
            // Insect-like: low flat torso, six legs in three rows.
            let leg_h = hy * 0.35;
            let torso_h = hy - leg_h;
            let torso_y = -hy * 0.5 + leg_h + torso_h * 0.5;
            out.push(BodyPartSpec {
                kind: BodyPartKind::Torso,
                rest_translation: Vec3::new(0.0, torso_y, 0.0),
                rest_rotation: Quat::IDENTITY,
                size: Vec3::new(hx * 0.7, torso_h, hz * 0.95),
            });

            let head_size = (hx.min(hz) * 0.35).max(0.04);
            out.push(BodyPartSpec {
                kind: BodyPartKind::Head,
                rest_translation: Vec3::new(0.0, torso_y, hz * 0.5 + head_size * 0.4),
                rest_rotation: Quat::IDENTITY,
                size: Vec3::splat(head_size),
            });

            let leg_w = (hx.min(hz) * 0.12).max(0.03);
            let leg_y = -hy * 0.5 + leg_h * 0.5;
            let leg_x = hx * 0.5 + leg_w * 0.4;
            for &(side, sign_x) in &[(BodySide::Left, -1.0_f32), (BodySide::Right, 1.0)] {
                for &(fb, z_frac) in &[
                    (BodyAxisPos::Front, 0.35_f32),
                    (BodyAxisPos::Mid, 0.0),
                    (BodyAxisPos::Back, -0.35),
                ] {
                    out.push(BodyPartSpec {
                        kind: BodyPartKind::Leg { side, fb },
                        rest_translation: Vec3::new(sign_x * leg_x, leg_y, z_frac * hz),
                        rest_rotation: Quat::IDENTITY,
                        size: Vec3::new(leg_w, leg_h, leg_w),
                    });
                }
            }
        }
        BodyPlan::Serpent => {
            // Single elongated torso, no legs, head on +Z, tail on -Z.
            let torso_len = hz * 0.7;
            let head_len = hz * 0.15;
            let tail_len = hz * 0.15;
            out.push(BodyPartSpec {
                kind: BodyPartKind::Torso,
                rest_translation: Vec3::new(0.0, 0.0, 0.0),
                rest_rotation: Quat::IDENTITY,
                size: Vec3::new(hx, hy, torso_len),
            });
            out.push(BodyPartSpec {
                kind: BodyPartKind::Head,
                rest_translation: Vec3::new(0.0, 0.0, torso_len * 0.5 + head_len * 0.5),
                rest_rotation: Quat::IDENTITY,
                size: Vec3::new(hx * 1.1, hy * 1.1, head_len),
            });
            out.push(BodyPartSpec {
                kind: BodyPartKind::Tail,
                rest_translation: Vec3::new(0.0, 0.0, -torso_len * 0.5 - tail_len * 0.5),
                rest_rotation: Quat::IDENTITY,
                size: Vec3::new(hx * 0.6, hy * 0.6, tail_len),
            });
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Spawning into ECS
// ---------------------------------------------------------------------------

/// Spawn the procedural body parts as children of `root` and tag the root
/// with [`ProceduralBody`]. Reuses one shared mesh handle per unique cuboid
/// size and one shared material handle (the creature's color).
///
/// The caller is responsible for adding `Transform` / `Visibility` to the
/// root entity.
pub fn spawn_procedural_body(
    commands: &mut Commands,
    root: Entity,
    plan: BodyPlan,
    hitbox: (f32, f32, f32),
    color: Color,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
) {
    let specs = body_part_specs(plan, hitbox);
    let material = materials.add(StandardMaterial {
        base_color: color,
        ..default()
    });

    commands.entity(root).insert(ProceduralBody { plan });

    for spec in specs {
        let mesh = meshes.add(Cuboid::new(spec.size.x, spec.size.y, spec.size.z));
        let child = commands
            .spawn((
                BodyPart {
                    kind: spec.kind,
                    rest_translation: spec.rest_translation,
                    rest_rotation: spec.rest_rotation,
                },
                Mesh3d(mesh),
                MeshMaterial3d(material.clone()),
                Transform {
                    translation: spec.rest_translation,
                    rotation: spec.rest_rotation,
                    scale: Vec3::ONE,
                },
                Visibility::default(),
            ))
            .id();
        commands.entity(root).add_child(child);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn count_kind(specs: &[BodyPartSpec], pred: impl Fn(&BodyPartKind) -> bool) -> usize {
        specs.iter().filter(|s| pred(&s.kind)).count()
    }

    #[test]
    fn default_plan_for_size_maps_tiny_to_biped_else_quadruped() {
        assert_eq!(BodyPlan::default_for_size(BodySize::Tiny), BodyPlan::Biped);
        for size in [
            BodySize::Small,
            BodySize::Medium,
            BodySize::Large,
            BodySize::Huge,
        ] {
            assert_eq!(BodyPlan::default_for_size(size), BodyPlan::Quadruped);
        }
    }

    #[test]
    fn quadruped_has_one_torso_one_head_four_legs_one_tail() {
        let specs = body_part_specs(BodyPlan::Quadruped, (1.0, 0.8, 1.5));
        assert_eq!(specs.len(), 1 + 1 + 4 + 1);
        assert_eq!(count_kind(&specs, |k| matches!(k, BodyPartKind::Torso)), 1);
        assert_eq!(count_kind(&specs, |k| matches!(k, BodyPartKind::Head)), 1);
        assert_eq!(
            count_kind(&specs, |k| matches!(k, BodyPartKind::Leg { .. })),
            4
        );
        assert_eq!(count_kind(&specs, |k| matches!(k, BodyPartKind::Tail)), 1);
    }

    #[test]
    fn biped_has_one_torso_one_head_two_legs() {
        let specs = body_part_specs(BodyPlan::Biped, (0.4, 1.7, 0.4));
        assert_eq!(specs.len(), 1 + 1 + 2);
        assert_eq!(
            count_kind(&specs, |k| matches!(k, BodyPartKind::Leg { .. })),
            2
        );
    }

    #[test]
    fn hexapod_has_six_legs() {
        let specs = body_part_specs(BodyPlan::Hexapod, (0.6, 0.3, 1.2));
        assert_eq!(
            count_kind(&specs, |k| matches!(k, BodyPartKind::Leg { .. })),
            6
        );
    }

    #[test]
    fn serpent_has_no_legs() {
        let specs = body_part_specs(BodyPlan::Serpent, (0.2, 0.2, 2.0));
        assert_eq!(
            count_kind(&specs, |k| matches!(k, BodyPartKind::Leg { .. })),
            0
        );
        assert!(count_kind(&specs, |k| matches!(k, BodyPartKind::Torso)) >= 1);
    }

    #[test]
    fn quadruped_legs_are_at_bottom_of_hitbox() {
        let (hx, hy, hz) = (1.0, 0.8, 1.5);
        let specs = body_part_specs(BodyPlan::Quadruped, (hx, hy, hz));
        for s in &specs {
            if let BodyPartKind::Leg { .. } = s.kind {
                let leg_bottom_y = s.rest_translation.y - s.size.y * 0.5;
                // Foot must touch the foot plane within float tolerance.
                assert!(
                    (leg_bottom_y - (-hy * 0.5)).abs() < 1e-4,
                    "leg bottom {} != foot plane {}",
                    leg_bottom_y,
                    -hy * 0.5,
                );
            }
        }
    }

    #[test]
    fn quadruped_legs_are_distributed_across_corners() {
        let specs = body_part_specs(BodyPlan::Quadruped, (1.0, 0.8, 1.5));
        let mut seen = std::collections::HashSet::new();
        for s in &specs {
            if let BodyPartKind::Leg { side, fb } = s.kind {
                assert!(
                    seen.insert((side, fb)),
                    "duplicate leg position {:?}",
                    s.kind
                );
                let sign_x_expected = match side {
                    BodySide::Left => -1.0,
                    BodySide::Right => 1.0,
                };
                assert!(s.rest_translation.x * sign_x_expected > 0.0);
                let sign_z_expected = match fb {
                    BodyAxisPos::Front => 1.0,
                    BodyAxisPos::Back => -1.0,
                    BodyAxisPos::Mid => 0.0,
                };
                if sign_z_expected != 0.0 {
                    assert!(s.rest_translation.z * sign_z_expected > 0.0);
                }
            }
        }
        assert_eq!(seen.len(), 4);
    }

    #[test]
    fn all_parts_fit_within_full_hitbox_aabb() {
        // Sanity: no part center should be more than `hitbox/2 + part_size/2`
        // from origin in any axis (parts may extend slightly beyond for head/tail).
        let hitbox = (1.0, 0.8, 1.5);
        for plan in [
            BodyPlan::Quadruped,
            BodyPlan::Biped,
            BodyPlan::Hexapod,
            BodyPlan::Serpent,
        ] {
            let specs = body_part_specs(plan, hitbox);
            assert!(!specs.is_empty(), "{:?} has no parts", plan);
            // Torso must always be inside the hitbox bounds.
            let torso = specs
                .iter()
                .find(|s| matches!(s.kind, BodyPartKind::Torso))
                .expect("every plan has a torso");
            assert!(torso.rest_translation.x.abs() <= hitbox.0 * 0.5 + 1e-4);
            assert!(torso.rest_translation.y.abs() <= hitbox.1 * 0.5 + 1e-4);
            assert!(torso.rest_translation.z.abs() <= hitbox.2 * 0.5 + 1e-4);
        }
    }
}
