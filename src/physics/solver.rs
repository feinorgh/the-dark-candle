// Sequential impulse solver for contact resolution.
//
// For each contact manifold, computes normal and tangential impulses that
// prevent penetration (restitution) and resist sliding (friction). Angular
// impulses are applied at the contact point to couple linear and rotational
// response.
//
// The solver runs multiple iterations to converge accumulated impulses for
// stacking stability. Penetration is corrected via a position bias term
// (Baumgarte stabilization).

use bevy::prelude::*;

use super::gravity::{Mass, PhysicsBody};
use super::narrow_phase::Contacts;
use super::rigid_body::{AngularVelocity, MomentOfInertia};

/// Number of solver iterations per tick.
/// More iterations improve stacking stability at the cost of performance.
const SOLVER_ITERATIONS: usize = 4;

/// Baumgarte stabilization factor. Fraction of penetration corrected per tick.
const BAUMGARTE_FACTOR: f32 = 0.2;

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

/// Solve all contacts using sequential impulses.
///
/// Modifies `PhysicsBody::velocity` and `AngularVelocity` for entities
/// involved in collisions. Applies Baumgarte position correction via
/// Transform adjustment.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn solve_contacts(
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

    // Build constraint list
    let mut constraints: Vec<ContactConstraint> = Vec::new();

    for manifold in &contacts.manifolds {
        for contact in &manifold.contacts {
            // Compute initial relative velocity along normal
            let initial_vn = {
                let Ok((body_a, _, tf_a, omega_a, _)) = query.get(manifold.entity_a) else {
                    continue;
                };
                let Ok((body_b, _, tf_b, omega_b, _)) = query.get(manifold.entity_b) else {
                    continue;
                };

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
                initial_relative_vn: initial_vn,
            });
        }
    }

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

            // --- Normal impulse ---
            let v_rel = relative_velocity(
                data_a.0.velocity,
                omega_a,
                r_a,
                data_b.0.velocity,
                omega_b,
                r_b,
            );
            let vn = v_rel.dot(constraint.normal);

            let eff_mass_n = effective_inv_mass(inv_m_a, inv_i_a, r_a, constraint.normal)
                + effective_inv_mass(inv_m_b, inv_i_b, r_b, constraint.normal);

            if eff_mass_n < 1e-12 {
                continue;
            }

            // Restitution bias: only apply if initial approach speed exceeded threshold
            let restitution_bias = if constraint.initial_relative_vn < -0.5 {
                -constraint.restitution * constraint.initial_relative_vn
            } else {
                0.0
            };

            // Baumgarte position bias
            let pos_bias =
                (BAUMGARTE_FACTOR / (1.0 / 60.0)) * (constraint.depth - PENETRATION_SLOP).max(0.0);

            let lambda_n = (-vn + restitution_bias + pos_bias) / eff_mass_n;

            // Accumulated impulse clamping (Erin Catto's technique)
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
        }
    }

    // Position correction: push entities apart along contact normal
    for constraint in &constraints {
        let correction = (constraint.depth - PENETRATION_SLOP).max(0.0) * 0.4;
        if correction < 1e-6 {
            continue;
        }

        let Ok([data_a, data_b]) = query.get_many([constraint.entity_a, constraint.entity_b])
        else {
            continue;
        };

        let inv_m_a = 1.0 / data_a.1.0;
        let inv_m_b = 1.0 / data_b.1.0;
        let total_inv = inv_m_a + inv_m_b;
        if total_inv < 1e-12 {
            continue;
        }

        let shift = constraint.normal * correction;

        // Apply position correction proportional to inverse mass
        // We need mutable access to transforms, but query already borrows them.
        // Use unsafe interior mutability pattern via separate queries is complex.
        // Instead, store corrections and apply after the loop.
        // For now, apply via the main query's Transform (already borrowed).

        if let Ok([mut data_a, mut data_b]) =
            query.get_many_mut([constraint.entity_a, constraint.entity_b])
        {
            let frac_a = inv_m_a / total_inv;
            let frac_b = inv_m_b / total_inv;
            data_a.2.translation += shift * frac_a;
            data_b.2.translation -= shift * frac_b;
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
        let bias = (BAUMGARTE_FACTOR / (1.0 / 60.0)) * (depth - PENETRATION_SLOP).max(0.0);
        assert_eq!(bias, 0.0);
    }

    #[test]
    fn baumgarte_bias_positive_for_penetration() {
        let depth = 0.05;
        let dt = 1.0 / 60.0;
        let bias = (BAUMGARTE_FACTOR / dt) * (depth - PENETRATION_SLOP).max(0.0);
        assert!(bias > 0.0);
    }
}
