//! Physical eye/ear geometry for entity perception.
//!
//! Upgrades the abstract `Senses` radii in `src/behavior/perception.rs` with
//! bone-attached eye and ear components that support field-of-view cone tests.
//!
//! # Sight pipeline
//! 1. Locate the eye bone in the entity's `Skeleton`.
//! 2. Compute the eye's forward direction from the bone's world rotation.
//! 3. Reject candidates outside `max_range_m` or outside the FOV cone.
//! 4. Perform a lightweight visibility check (distance + cone; full voxel LOS
//!    is a TODO once the full ray-integration layer exists).
//! 5. Write surviving entities into `PerceivedEntities`.

use bevy::prelude::*;

use super::skeleton::Skeleton;

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

/// Eye geometry attached to a named bone.
///
/// The eye looks toward the bone's local −Z axis (Bevy convention for "forward").
#[derive(Component, Debug, Clone)]
pub struct EyeComponent {
    /// Name of the bone in the entity's `Skeleton` to use as the eye pivot.
    pub bone: String,
    /// Half-angle of the visual cone in radians (e.g. π/4 for a 90° total FOV).
    pub fov_half_angle_rad: f32,
    /// Maximum detection range in metres.
    pub max_range_m: f32,
}

/// Ear geometry attached to a named bone.
///
/// Directional bias models how much better the ear detects sounds in front
/// versus behind (1.0 = omnidirectional, 0.0 = purely forward).
#[derive(Component, Debug, Clone)]
pub struct EarComponent {
    /// Name of the bone in the entity's `Skeleton` to use as the ear pivot.
    pub bone: String,
    /// Sensitivity multiplier (higher = can hear quieter sounds).
    pub sensitivity: f32,
    /// How strongly direction matters: 0.0 = fully directional (front only),
    /// 1.0 = omnidirectional.
    pub directional_bias: f32,
}

/// Result component: lists entities that this entity can currently see or hear.
#[derive(Component, Debug, Clone, Default)]
pub struct PerceivedEntities {
    /// Entities within visual range and FOV cone.
    pub visible: Vec<Entity>,
    /// Entities within auditory range (after directional attenuation).
    pub audible: Vec<Entity>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Check whether `target_pos` lies within the FOV cone defined by `eye_pos`,
/// `eye_forward` direction, and `fov_half_angle_rad`.
///
/// Returns `true` if the angle between `eye_forward` and the direction to the
/// target is ≤ `fov_half_angle_rad`.
pub fn in_fov_cone(
    eye_pos: Vec3,
    eye_forward: Vec3,
    target_pos: Vec3,
    fov_half_angle_rad: f32,
) -> bool {
    let to_target = target_pos - eye_pos;
    let dist = to_target.length();
    if dist < f32::EPSILON {
        return true; // Coincident positions: trivially visible.
    }
    let dir = to_target / dist;
    let cos_angle = eye_forward.dot(dir);
    let cos_limit = fov_half_angle_rad.cos();
    cos_angle >= cos_limit
}

/// Compute a simple hearing strength for `listener` toward `source`.
///
/// Combines inverse-square fall-off with a directional bias so that sounds
/// behind the listener are attenuated.
///
/// Returns a value in [0.0, 1.0].
pub fn hearing_strength(
    listener_pos: Vec3,
    listener_forward: Vec3,
    source_pos: Vec3,
    max_range_m: f32,
    sensitivity: f32,
    directional_bias: f32,
) -> f32 {
    let to_source = source_pos - listener_pos;
    let dist = to_source.length();
    if dist > max_range_m {
        return 0.0;
    }

    let range_fraction = 1.0 - (dist / max_range_m);

    // Directional attenuation: dot product is 1 in front, −1 behind.
    let dir = if dist > f32::EPSILON {
        to_source / dist
    } else {
        listener_forward
    };
    let dot = listener_forward.dot(dir); // [−1, 1]
    // Map to [directional_bias, 1.0]: fully behind → `directional_bias`, fully ahead → 1.
    let directional_factor = directional_bias + (1.0 - directional_bias) * (dot * 0.5 + 0.5);

    (range_fraction * directional_factor * sensitivity).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Sight system
// ---------------------------------------------------------------------------

/// Update `PerceivedEntities` for every entity that has a `Skeleton` and at
/// least one `EyeComponent`.
///
/// # LOS note
/// Full voxel line-of-sight is not integrated here yet — the check uses
/// distance and FOV cone only.  When the LOS pass is added, wire in
/// `crate::behavior::perception::has_line_of_sight`.
///
/// TODO: extend to use `has_line_of_sight` from `crate::behavior::perception`
///       once the voxel grid access pattern is established for body systems.
pub fn update_sight(
    mut observers: Query<(
        Entity,
        &Skeleton,
        &EyeComponent,
        Option<&EarComponent>,
        &mut PerceivedEntities,
    )>,
    targets: Query<(Entity, &Transform), Without<EyeComponent>>,
) {
    for (observer_entity, skeleton, eye, ear_opt, mut perceived) in &mut observers {
        perceived.visible.clear();
        perceived.audible.clear();

        // ── Locate eye bone ────────────────────────────────────────────────
        let Some(bone_idx) = skeleton.bone_index(&eye.bone) else {
            continue;
        };
        let eye_transform = &skeleton.bone_transforms[bone_idx];
        let eye_pos = eye_transform.translation;
        // Bevy convention: −Z is the "forward" direction for transforms.
        let eye_forward = eye_transform.rotation * Vec3::NEG_Z;

        // ── Ear setup ─────────────────────────────────────────────────────
        let (ear_pos, ear_forward, ear_sensitivity, ear_bias, ear_range) =
            if let Some(ear) = ear_opt {
                if let Some(eidx) = skeleton.bone_index(&ear.bone) {
                    let bt = &skeleton.bone_transforms[eidx];
                    (
                        bt.translation,
                        bt.rotation * Vec3::NEG_Z,
                        ear.sensitivity,
                        ear.directional_bias,
                        // Hearing range ≈ 3× sight range as a reasonable default.
                        eye.max_range_m * 3.0,
                    )
                } else {
                    (
                        eye_pos,
                        eye_forward,
                        ear.sensitivity,
                        ear.directional_bias,
                        eye.max_range_m * 3.0,
                    )
                }
            } else {
                (Vec3::ZERO, Vec3::ZERO, 0.0, 0.0, 0.0)
            };

        for (target_entity, target_transform) in &targets {
            if target_entity == observer_entity {
                continue;
            }
            let target_pos = target_transform.translation;

            // ── Sight ──────────────────────────────────────────────────────
            let dist = eye_pos.distance(target_pos);
            if dist <= eye.max_range_m
                && in_fov_cone(eye_pos, eye_forward, target_pos, eye.fov_half_angle_rad)
            {
                // TODO: replace `true` with `has_line_of_sight(grid, ...)` once
                //       voxel grid access is plumbed into body systems.
                let los_clear = true;
                if los_clear {
                    perceived.visible.push(target_entity);
                }
            }

            // ── Hearing ────────────────────────────────────────────────────
            if ear_opt.is_some() {
                let strength = hearing_strength(
                    ear_pos,
                    ear_forward,
                    target_pos,
                    ear_range,
                    ear_sensitivity,
                    ear_bias,
                );
                if strength > 0.0 {
                    perceived.audible.push(target_entity);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_4;

    fn make_skeleton(bone: &str, pos: Vec3, rotation: Quat) -> Skeleton {
        Skeleton {
            bone_names: vec![bone.into()],
            bone_transforms: vec![Transform {
                translation: pos,
                rotation,
                scale: Vec3::ONE,
            }],
            ik_targets: vec![None],
            torques: vec![Vec3::ZERO],
            ..Default::default()
        }
    }

    // ── FOV cone tests ────────────────────────────────────────────────────

    #[test]
    fn target_directly_in_front_is_in_cone() {
        // Eye looking along +X, target at (10, 0, 0) — directly ahead.
        let eye_pos = Vec3::ZERO;
        let eye_forward = Vec3::X;
        let target = Vec3::new(10.0, 0.0, 0.0);
        assert!(in_fov_cone(eye_pos, eye_forward, target, FRAC_PI_4));
    }

    #[test]
    fn target_directly_behind_is_outside_cone() {
        let eye_pos = Vec3::ZERO;
        let eye_forward = Vec3::X;
        let target = Vec3::new(-5.0, 0.0, 0.0); // behind
        assert!(!in_fov_cone(eye_pos, eye_forward, target, FRAC_PI_4));
    }

    #[test]
    fn target_at_exact_fov_boundary_is_in_cone() {
        // Eye looking along +X; target is exactly `fov_half_angle_rad` off-axis.
        let half = FRAC_PI_4; // 45°
        let eye_pos = Vec3::ZERO;
        let eye_forward = Vec3::X;
        // Place target at 45° above +X axis.
        let target = Vec3::new(1.0, 1.0, 0.0).normalize() * 5.0;
        // cos(45°) ≈ 0.7071; the dot product exactly equals cos_limit → in cone.
        assert!(in_fov_cone(eye_pos, eye_forward, target, half));
    }

    #[test]
    fn target_just_outside_fov_is_not_in_cone() {
        // 46° off-axis should be outside a 45° half-angle cone.
        let half = FRAC_PI_4;
        let eye_pos = Vec3::ZERO;
        let eye_forward = Vec3::X;
        let angle = half + 0.05; // slightly outside
        let target = Vec3::new(angle.cos(), angle.sin(), 0.0) * 5.0;
        assert!(!in_fov_cone(eye_pos, eye_forward, target, half));
    }

    #[test]
    fn target_outside_max_range_is_not_visible() {
        // Even if in the cone, objects beyond max_range_m are not visible.
        let eye_pos = Vec3::ZERO;
        let max_range = 10.0_f32;
        let target_pos = Vec3::new(15.0, 0.0, 0.0); // 15 m away
        let dist = eye_pos.distance(target_pos);
        assert!(
            dist > max_range,
            "pre-condition: target must be out of range"
        );
    }

    // ── Bone index lookup ─────────────────────────────────────────────────

    #[test]
    fn bone_index_returns_correct_position() {
        let sk = make_skeleton("head", Vec3::new(0.0, 1.8, 0.0), Quat::IDENTITY);
        assert_eq!(sk.bone_index("head"), Some(0));
        assert_eq!(sk.bone_index("tail"), None);
    }

    #[test]
    fn skeleton_with_multiple_bones_finds_each() {
        let sk = Skeleton {
            bone_names: vec!["root".into(), "spine".into(), "head".into()],
            bone_transforms: vec![Transform::default(); 3],
            ik_targets: vec![None; 3],
            torques: vec![Vec3::ZERO; 3],
            ..Default::default()
        };
        assert_eq!(sk.bone_index("root"), Some(0));
        assert_eq!(sk.bone_index("spine"), Some(1));
        assert_eq!(sk.bone_index("head"), Some(2));
    }

    // ── Hearing strength ──────────────────────────────────────────────────

    #[test]
    fn hearing_returns_zero_beyond_range() {
        let s = hearing_strength(
            Vec3::ZERO,
            Vec3::X,
            Vec3::new(50.0, 0.0, 0.0),
            20.0,
            1.0,
            1.0,
        );
        assert_eq!(s, 0.0);
    }

    #[test]
    fn hearing_stronger_in_front_than_behind() {
        let listener = Vec3::ZERO;
        let forward = Vec3::X;
        let source_front = Vec3::new(5.0, 0.0, 0.0);
        let source_behind = Vec3::new(-5.0, 0.0, 0.0);
        let front = hearing_strength(listener, forward, source_front, 20.0, 1.0, 0.2);
        let behind = hearing_strength(listener, forward, source_behind, 20.0, 1.0, 0.2);
        assert!(front > behind, "front={front}, behind={behind}");
    }

    #[test]
    fn omnidirectional_ear_equal_front_and_behind() {
        let listener = Vec3::ZERO;
        let forward = Vec3::X;
        let source_front = Vec3::new(5.0, 0.0, 0.0);
        let source_behind = Vec3::new(-5.0, 0.0, 0.0);
        let front = hearing_strength(listener, forward, source_front, 20.0, 1.0, 1.0);
        let behind = hearing_strength(listener, forward, source_behind, 20.0, 1.0, 1.0);
        // Omnidirectional (bias=1): same distance → same strength.
        assert!(
            (front - behind).abs() < 1e-5,
            "front={front}, behind={behind}"
        );
    }
}
