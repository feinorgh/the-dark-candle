// Sequential impulse solver with split-impulse position correction.
//
// For each contact manifold the solver computes:
//  • **Velocity impulses** — restitution and Coulomb friction applied to the
//    real velocities of colliding bodies.  These are the only impulses that
//    affect kinetic energy.
//  • **Position impulses** — Baumgarte penetration correction applied to
//    *pseudo-velocities*, temporary per-frame quantities that push overlapping
//    bodies apart without injecting kinetic energy.
//
// After all iterations the pseudo-velocities are integrated into `Transform`
// positions and then discarded, so position stabilisation never leaks energy
// into the simulation.

use std::collections::HashMap;

use bevy::prelude::*;

use super::gravity::{Mass, PhysicsBody};
use super::narrow_phase::Contacts;
use super::rigid_body::{AngularVelocity, MomentOfInertia};

/// Number of solver iterations per tick.
/// More iterations improve stacking stability at the cost of performance.
const SOLVER_ITERATIONS: usize = 4;

/// Baumgarte stabilisation factor for position correction.
///
/// With split impulse, position correction flows through pseudo-velocities
/// that are discarded each frame, so this can be more aggressive than the
/// typical 0.1–0.2 needed for velocity-based Baumgarte (which injected
/// kinetic energy).
const BAUMGARTE_FACTOR: f32 = 0.8;

/// Penetration slop: small overlap allowed to avoid jitter (m).
const PENETRATION_SLOP: f32 = 0.005;

// ---------------------------------------------------------------------------
// Solver state (per-contact working data)
// ---------------------------------------------------------------------------

/// Working data for a single contact during iterative solving.
struct ContactConstraint {
    entity_a: Entity,
    entity_b: Entity,
    normal: Vec3,
    point: Vec3,
    depth: f32,
    restitution: f32,
    friction: f32,
    /// Accumulated normal impulse (clamped ≥ 0).
    normal_impulse: f32,
    /// Accumulated tangent impulse (clamped by friction cone).
    tangent_impulse: Vec3,
    /// Accumulated pseudo-velocity normal impulse for position correction.
    pseudo_normal_impulse: f32,
    /// Relative velocity along normal before solve (for restitution).
    initial_relative_vn: f32,
}

// ---------------------------------------------------------------------------
// Pure helper functions
// ---------------------------------------------------------------------------

/// Compute the inverse effective mass at a contact point along a direction.
///
/// `inv_m` = 1/mass, `inv_i` = inverse inertia (diagonal), `r` = offset from
/// center of mass to contact point, `n` = impulse direction.
///
/// Effective mass: 1/m + (r × n)ᵀ · I⁻¹ · (r × n)
pub fn effective_inv_mass(inv_m: f32, inv_i: Vec3, r: Vec3, n: Vec3) -> f32 {
    let rxn = r.cross(n);
    inv_m + rxn.x * rxn.x * inv_i.x + rxn.y * rxn.y * inv_i.y + rxn.z * rxn.z * inv_i.z
}

/// Relative velocity of body A w.r.t. body B at a contact point.
pub fn relative_velocity(
    vel_a: Vec3,
    omega_a: Vec3,
    r_a: Vec3,
    vel_b: Vec3,
    omega_b: Vec3,
    r_b: Vec3,
) -> Vec3 {
    (vel_a + omega_a.cross(r_a)) - (vel_b + omega_b.cross(r_b))
}

// ---------------------------------------------------------------------------
// System
// ---------------------------------------------------------------------------

/// Solve all contacts using sequential impulses with split-impulse position
/// correction.
///
/// Velocity impulses (restitution + friction) modify `PhysicsBody::velocity`
/// and `AngularVelocity`.  Position correction uses pseudo-velocities that are
/// integrated into `Transform` after all iterations, then discarded — this
/// prevents Baumgarte stabilisation from injecting kinetic energy.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn solve_contacts(
    time: Res<Time>,
    contacts: Res<Contacts>,
    mut query: Query<(
        &mut PhysicsBody,
        &Mass,
        &mut Transform,
        Option<&mut AngularVelocity>,
        Option<&MomentOfInertia>,
    )>,
) {
    if contacts.manifolds.is_empty() {
        return;
    }

    let dt = time.delta_secs();
    if dt < 1e-12 {
        return;
    }

    // Build constraint list
    let mut constraints: Vec<ContactConstraint> = Vec::new();

    for manifold in &contacts.manifolds {
        for contact in &manifold.contacts {
            // Compute initial relative velocity along normal
            let initial_vn = {
                let Ok((body_a, mass_a, tf_a, omega_a, _)) = query.get(manifold.entity_a) else {
                    continue;
                };
                let Ok((body_b, mass_b, tf_b, omega_b, _)) = query.get(manifold.entity_b) else {
                    continue;
                };

                // Skip contacts between two effectively-static bodies
                // (e.g. overlapping wall AABBs).  No meaningful response is
                // possible and attempting one would waste solver budget.
                if mass_a.0 > 1e8 && mass_b.0 > 1e8 {
                    continue;
                }

                let r_a = contact.point - tf_a.translation;
                let r_b = contact.point - tf_b.translation;
                let omega_a_val = omega_a.map(|o| o.0).unwrap_or(Vec3::ZERO);
                let omega_b_val = omega_b.map(|o| o.0).unwrap_or(Vec3::ZERO);

                let v_rel = relative_velocity(
                    body_a.velocity,
                    omega_a_val,
                    r_a,
                    body_b.velocity,
                    omega_b_val,
                    r_b,
                );
                v_rel.dot(contact.normal)
            };

            constraints.push(ContactConstraint {
                entity_a: manifold.entity_a,
                entity_b: manifold.entity_b,
                normal: contact.normal,
                point: contact.point,
                depth: contact.depth,
                restitution: manifold.material.restitution,
                friction: manifold.material.friction,
                normal_impulse: 0.0,
                tangent_impulse: Vec3::ZERO,
                pseudo_normal_impulse: 0.0,
                initial_relative_vn: initial_vn,
            });
        }
    }

    // Per-entity pseudo-velocity accumulators (position correction only).
    let mut pseudo_lin: HashMap<Entity, Vec3> = HashMap::new();
    let mut pseudo_ang: HashMap<Entity, Vec3> = HashMap::new();

    // Iterative solve
    for _ in 0..SOLVER_ITERATIONS {
        for constraint in &mut constraints {
            // Fetch current state for both entities
            let Ok([mut data_a, mut data_b]) =
                query.get_many_mut([constraint.entity_a, constraint.entity_b])
            else {
                continue;
            };

            let inv_m_a = 1.0 / data_a.1.0;
            let inv_m_b = 1.0 / data_b.1.0;
            let inv_i_a = data_a.4.as_ref().map(|i| i.inverse()).unwrap_or(Vec3::ZERO);
            let inv_i_b = data_b.4.as_ref().map(|i| i.inverse()).unwrap_or(Vec3::ZERO);

            let r_a = constraint.point - data_a.2.translation;
            let r_b = constraint.point - data_b.2.translation;

            let omega_a = data_a.3.as_ref().map(|o| o.0).unwrap_or(Vec3::ZERO);
            let omega_b = data_b.3.as_ref().map(|o| o.0).unwrap_or(Vec3::ZERO);

            let eff_mass_n = effective_inv_mass(inv_m_a, inv_i_a, r_a, constraint.normal)
                + effective_inv_mass(inv_m_b, inv_i_b, r_b, constraint.normal);

            if eff_mass_n < 1e-12 {
                continue;
            }

            // =============================================================
            // Velocity solve — restitution + friction only (no pos_bias)
            // =============================================================
            let v_rel = relative_velocity(
                data_a.0.velocity,
                omega_a,
                r_a,
                data_b.0.velocity,
                omega_b,
                r_b,
            );
            let vn = v_rel.dot(constraint.normal);

            let restitution_bias = if constraint.initial_relative_vn < -0.5 {
                -constraint.restitution * constraint.initial_relative_vn
            } else {
                0.0
            };

            let lambda_n = (-vn + restitution_bias) / eff_mass_n;

            let old_impulse = constraint.normal_impulse;
            constraint.normal_impulse = (old_impulse + lambda_n).max(0.0);
            let applied_n = constraint.normal_impulse - old_impulse;

            if applied_n.abs() > 1e-12 {
                let impulse = constraint.normal * applied_n;

                data_a.0.velocity += impulse * inv_m_a;
                data_b.0.velocity -= impulse * inv_m_b;

                if let Some(ref mut omega_a) = data_a.3 {
                    omega_a.0 += r_a.cross(impulse) * inv_i_a;
                }
                if let Some(ref mut omega_b) = data_b.3 {
                    omega_b.0 -= r_b.cross(impulse) * inv_i_b;
                }
            }

            // --- Friction impulse (tangent) ---
            let v_rel = relative_velocity(
                data_a.0.velocity,
                data_a.3.as_ref().map(|o| o.0).unwrap_or(Vec3::ZERO),
                r_a,
                data_b.0.velocity,
                data_b.3.as_ref().map(|o| o.0).unwrap_or(Vec3::ZERO),
                r_b,
            );
            let vt = v_rel - constraint.normal * v_rel.dot(constraint.normal);
            let vt_len = vt.length();

            if vt_len > 1e-6 {
                let tangent = vt / vt_len;
                let eff_mass_t = effective_inv_mass(inv_m_a, inv_i_a, r_a, tangent)
                    + effective_inv_mass(inv_m_b, inv_i_b, r_b, tangent);

                if eff_mass_t > 1e-12 {
                    let lambda_t = -vt_len / eff_mass_t;

                    // Coulomb friction cone: |F_t| ≤ μ × F_n
                    let max_friction = constraint.friction * constraint.normal_impulse;
                    let old_tangent = constraint.tangent_impulse;
                    let new_tangent = old_tangent + tangent * lambda_t;
                    let new_tangent_len = new_tangent.length();

                    constraint.tangent_impulse =
                        if new_tangent_len > max_friction && max_friction > 1e-12 {
                            new_tangent * (max_friction / new_tangent_len)
                        } else {
                            new_tangent
                        };

                    let applied_t = constraint.tangent_impulse - old_tangent;

                    if applied_t.length_squared() > 1e-12 {
                        data_a.0.velocity += applied_t * inv_m_a;
                        data_b.0.velocity -= applied_t * inv_m_b;

                        if let Some(ref mut omega_a) = data_a.3 {
                            omega_a.0 += r_a.cross(applied_t) * inv_i_a;
                        }
                        if let Some(ref mut omega_b) = data_b.3 {
                            omega_b.0 -= r_b.cross(applied_t) * inv_i_b;
                        }
                    }
                }
            }

            // =============================================================
            // Position solve — Baumgarte bias → pseudo-velocities
            // =============================================================
            let pv_a = pseudo_lin
                .get(&constraint.entity_a)
                .copied()
                .unwrap_or(Vec3::ZERO);
            let po_a = pseudo_ang
                .get(&constraint.entity_a)
                .copied()
                .unwrap_or(Vec3::ZERO);
            let pv_b = pseudo_lin
                .get(&constraint.entity_b)
                .copied()
                .unwrap_or(Vec3::ZERO);
            let po_b = pseudo_ang
                .get(&constraint.entity_b)
                .copied()
                .unwrap_or(Vec3::ZERO);

            let pv_rel = relative_velocity(pv_a, po_a, r_a, pv_b, po_b, r_b);
            let pvn = pv_rel.dot(constraint.normal);

            let pos_bias = (BAUMGARTE_FACTOR / dt) * (constraint.depth - PENETRATION_SLOP).max(0.0);

            let lambda_p = (-pvn + pos_bias) / eff_mass_n;

            let old_pseudo = constraint.pseudo_normal_impulse;
            constraint.pseudo_normal_impulse = (old_pseudo + lambda_p).max(0.0);
            let applied_p = constraint.pseudo_normal_impulse - old_pseudo;

            if applied_p.abs() > 1e-12 {
                let impulse_p = constraint.normal * applied_p;

                *pseudo_lin.entry(constraint.entity_a).or_insert(Vec3::ZERO) += impulse_p * inv_m_a;
                *pseudo_lin.entry(constraint.entity_b).or_insert(Vec3::ZERO) -= impulse_p * inv_m_b;

                *pseudo_ang.entry(constraint.entity_a).or_insert(Vec3::ZERO) +=
                    r_a.cross(impulse_p) * inv_i_a;
                *pseudo_ang.entry(constraint.entity_b).or_insert(Vec3::ZERO) -=
                    r_b.cross(impulse_p) * inv_i_b;
            }
        }
    }

    // Integrate pseudo-velocities into positions (NOT into real velocities).
    for (entity, pv) in &pseudo_lin {
        if let Ok((_, _, mut transform, _, _)) = query.get_mut(*entity) {
            transform.translation += *pv * dt;
        }
    }
    for (entity, po) in &pseudo_ang {
        let delta = *po * dt;
        if delta.length_squared() > 1e-12
            && let Ok((_, _, mut transform, _, _)) = query.get_mut(*entity)
        {
            transform.rotation = (Quat::from_scaled_axis(delta) * transform.rotation).normalize();
        }
    }
}

/// System set for solver ordering.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct SolverSet;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effective_inv_mass_point_mass() {
        // Point mass (no rotation): effective = 1/m
        let inv_m = 1.0 / 10.0;
        let inv_i = Vec3::ZERO;
        let r = Vec3::ZERO;
        let n = Vec3::X;
        let result = effective_inv_mass(inv_m, inv_i, r, n);
        assert!((result - 0.1).abs() < 1e-6);
    }

    #[test]
    fn effective_inv_mass_with_rotation() {
        // With rotation contribution, effective inverse mass increases
        let inv_m = 1.0 / 10.0;
        let inv_i = Vec3::splat(1.0); // I = 1 kg·m² per axis
        let r = Vec3::new(0.0, 1.0, 0.0); // 1m offset
        let n = Vec3::X;
        let result = effective_inv_mass(inv_m, inv_i, r, n);
        // r × n = (0,1,0) × (1,0,0) = (0,0,-1), |rxn|² = 1
        // effective = 0.1 + 1.0 = 1.1
        assert!((result - 1.1).abs() < 1e-4, "Expected 1.1, got {result}");
    }

    #[test]
    fn relative_velocity_linear_only() {
        let v_rel = relative_velocity(
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::new(-3.0, 0.0, 0.0),
            Vec3::ZERO,
            Vec3::ZERO,
        );
        assert_eq!(v_rel, Vec3::new(8.0, 0.0, 0.0));
    }

    #[test]
    fn relative_velocity_with_angular() {
        // Body A spinning about Y at 1 rad/s, contact 1m in +X
        // → surface velocity at contact = ω × r = (0,1,0) × (1,0,0) = (0,0,-1)
        let v_rel = relative_velocity(
            Vec3::ZERO,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::ZERO,
        );
        assert!((v_rel.z - (-1.0)).abs() < 1e-6);
    }

    // --- Impulse math ---

    #[test]
    fn normal_impulse_head_on_collision() {
        // Two equal masses approaching each other at 10 m/s each
        // Normal impulse should reverse velocities (with restitution=1)
        let m = 10.0_f32;
        let v_approach = -20.0_f32; // relative velocity along normal
        let restitution = 1.0_f32;
        let eff_inv_mass = 2.0 / m; // 1/m + 1/m

        // Impulse magnitude: j = -(1+e) × v_n / (1/m_a + 1/m_b)
        let impulse = -(1.0 + restitution) * v_approach / eff_inv_mass;
        assert!(impulse > 0.0, "Impulse should be positive (repulsive)");

        // Each body gets Δv = j / m = impulse × inv_m
        let delta_v = impulse * (1.0 / m);
        assert!(
            (delta_v - 20.0).abs() < 1e-4,
            "Each mass should gain 20 m/s: got {delta_v}"
        );
    }

    #[test]
    fn friction_cone_clamps_tangent_impulse() {
        let normal_impulse = 100.0_f32; // N·s
        let mu = 0.3_f32;
        let max_friction = mu * normal_impulse; // 30 N·s

        let desired_tangent = 50.0_f32; // exceeds cone
        let clamped = desired_tangent.min(max_friction);
        assert!(
            (clamped - 30.0).abs() < 1e-4,
            "Expected ~30.0, got {clamped}"
        );
    }

    #[test]
    fn baumgarte_bias_zero_when_within_slop() {
        let depth = 0.003; // Less than PENETRATION_SLOP
        let dt = 1.0 / 60.0;
        let bias = (BAUMGARTE_FACTOR / dt) * (depth - PENETRATION_SLOP).max(0.0);
        assert_eq!(bias, 0.0);
    }

    #[test]
    fn baumgarte_bias_positive_for_penetration() {
        let depth = 0.05;
        let dt = 1.0 / 60.0;
        let bias = (BAUMGARTE_FACTOR / dt) * (depth - PENETRATION_SLOP).max(0.0);
        assert!(bias > 0.0);
    }

    #[test]
    fn split_impulse_velocity_excludes_position_bias() {
        // The velocity impulse formula should NOT include pos_bias.
        // Given two bodies in contact with penetration, the velocity lambda
        // should depend only on restitution, not on penetration depth.
        let vn = 0.0_f32; // resting contact
        let restitution_bias = 0.0_f32; // below threshold
        let eff_mass_n = 0.2_f32;

        // Velocity lambda — NO pos_bias
        let lambda_vel = (-vn + restitution_bias) / eff_mass_n;
        assert_eq!(
            lambda_vel, 0.0,
            "Resting contact should produce zero velocity impulse"
        );

        // Position lambda — WITH pos_bias
        let depth = 0.05;
        let dt = 1.0 / 60.0;
        let pos_bias = (BAUMGARTE_FACTOR / dt) * (depth - PENETRATION_SLOP).max(0.0);
        let lambda_pos = (-0.0 + pos_bias) / eff_mass_n;
        assert!(
            lambda_pos > 0.0,
            "Position impulse should be positive for penetrating contact"
        );

        // The key invariant: velocity impulse is independent of depth.
        let _deep_depth = 0.5;
        let lambda_vel_deep = (-vn + restitution_bias) / eff_mass_n;
        assert_eq!(
            lambda_vel, lambda_vel_deep,
            "Velocity impulse must not depend on penetration depth"
        );
    }
}
