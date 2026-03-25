// Angular dynamics for rigid body entities.
//
// Adds rotational state (`AngularVelocity`), resistance to angular
// acceleration (`MomentOfInertia`), and applied torques (`Torque`).
// The `angular_integration` system integrates these each `FixedUpdate`
// tick, coupling with the impulse solver at contact points.

use bevy::prelude::*;

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

/// Angular velocity in rad/s about each principal axis (world-space).
#[derive(Component, Debug, Clone, Default)]
pub struct AngularVelocity(pub Vec3);

/// Moment of inertia (kg·m²) about each principal axis.
///
/// Diagonal tensor stored as a `Vec3`. Computed from mass and shape via
/// `CollisionShape::moment_of_inertia()`.
#[derive(Component, Debug, Clone)]
pub struct MomentOfInertia(pub Vec3);

impl Default for MomentOfInertia {
    fn default() -> Self {
        Self(Vec3::ONE)
    }
}

impl MomentOfInertia {
    /// Inverse inertia tensor (1/I). Returns zero for locked axes.
    pub fn inverse(&self) -> Vec3 {
        Vec3::new(
            if self.0.x.abs() > 1e-12 {
                1.0 / self.0.x
            } else {
                0.0
            },
            if self.0.y.abs() > 1e-12 {
                1.0 / self.0.y
            } else {
                0.0
            },
            if self.0.z.abs() > 1e-12 {
                1.0 / self.0.z
            } else {
                0.0
            },
        )
    }
}

/// Accumulated torque applied this tick (N·m about each axis).
///
/// Zeroed at the end of each integration step. Systems that apply torques
/// should write to this component before `angular_integration` runs.
#[derive(Component, Debug, Clone, Default)]
pub struct Torque(pub Vec3);

/// Maximum angular speed (rad/s) to prevent numerical instability.
/// ~57 rad/s ≈ 9 revolutions/s, well above any physical scenario.
const ANGULAR_SPEED_CAP: f32 = 57.0;

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

/// Integrate angular dynamics: apply torque, update angular velocity, rotate entity.
///
/// Runs in `FixedUpdate` after force application. Uses semi-implicit Euler:
/// ω += (τ / I) × dt, then apply rotation to `Transform`.
pub fn angular_integration(
    time: Res<Time>,
    mut bodies: Query<(
        &mut AngularVelocity,
        &MomentOfInertia,
        &mut Torque,
        &mut Transform,
    )>,
) {
    let dt = time.delta_secs();
    if dt == 0.0 {
        return;
    }

    for (mut angular_vel, inertia, mut torque, mut transform) in &mut bodies {
        // Angular acceleration: α = τ / I (component-wise for diagonal tensor)
        let inv_i = inertia.inverse();
        let alpha = torque.0 * inv_i;

        // Update angular velocity
        angular_vel.0 += alpha * dt;

        // Safety cap
        let speed = angular_vel.0.length();
        if speed > ANGULAR_SPEED_CAP {
            angular_vel.0 *= ANGULAR_SPEED_CAP / speed;
        }

        // Apply rotation: small-angle quaternion approximation
        // For small dt, Δq ≈ Quat::from_scaled_axis(ω × dt)
        let delta_angle = angular_vel.0 * dt;
        if delta_angle.length_squared() > 1e-12 {
            let delta_rot = Quat::from_scaled_axis(delta_angle);
            transform.rotation = (delta_rot * transform.rotation).normalize();
        }

        // Clear torque accumulator for next tick
        torque.0 = Vec3::ZERO;
    }
}

/// System set for angular dynamics ordering.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct RigidBodySet;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_angular_velocity_is_zero() {
        let av = AngularVelocity::default();
        assert_eq!(av.0, Vec3::ZERO);
    }

    #[test]
    fn default_moment_of_inertia_is_one() {
        let moi = MomentOfInertia::default();
        assert_eq!(moi.0, Vec3::ONE);
    }

    #[test]
    fn moment_of_inertia_inverse() {
        let moi = MomentOfInertia(Vec3::new(2.0, 4.0, 8.0));
        let inv = moi.inverse();
        assert!((inv.x - 0.5).abs() < 1e-6);
        assert!((inv.y - 0.25).abs() < 1e-6);
        assert!((inv.z - 0.125).abs() < 1e-6);
    }

    #[test]
    fn moment_of_inertia_inverse_zero_axis_locked() {
        let moi = MomentOfInertia(Vec3::new(1.0, 0.0, 1.0));
        let inv = moi.inverse();
        assert_eq!(inv.y, 0.0);
        assert_eq!(inv.x, 1.0);
    }

    #[test]
    fn torque_default_is_zero() {
        let t = Torque::default();
        assert_eq!(t.0, Vec3::ZERO);
    }

    // --- Integration math (no ECS) ---

    #[test]
    fn angular_acceleration_from_torque() {
        // τ = 10 N·m, I = 5 kg·m² → α = 2 rad/s²
        let torque = Vec3::new(10.0, 0.0, 0.0);
        let inv_i = Vec3::new(1.0 / 5.0, 0.0, 0.0);
        let alpha = torque * inv_i;
        assert!((alpha.x - 2.0).abs() < 1e-6);
    }

    #[test]
    fn angular_velocity_after_1s_constant_torque() {
        // τ = 10 N·m, I = 5 kg·m², dt = 1/60, 60 steps = 1s → ω ≈ 2 rad/s
        let torque = 10.0_f32;
        let inv_i = 1.0 / 5.0_f32;
        let dt = 1.0 / 60.0;
        let mut omega = 0.0_f32;

        for _ in 0..60 {
            omega += torque * inv_i * dt;
        }

        assert!(
            (omega - 2.0).abs() < 0.1,
            "After 1s: ω = {omega}, expected ~2.0 rad/s"
        );
    }

    #[test]
    fn rotation_from_angular_velocity() {
        // ω = π rad/s about Y → after 1s should rotate 180°
        let omega = Vec3::new(0.0, std::f32::consts::PI, 0.0);
        let dt = 1.0 / 60.0;
        let mut rot = Quat::IDENTITY;

        for _ in 0..60 {
            let delta = Quat::from_scaled_axis(omega * dt);
            rot = (delta * rot).normalize();
        }

        // After 1s at π rad/s, should be ~180° rotation about Y
        let (_, angle) = rot.to_axis_angle();
        assert!(
            (angle - std::f32::consts::PI).abs() < 0.1,
            "Expected ~π radians rotation, got {angle}"
        );
    }

    #[test]
    fn angular_speed_cap_prevents_blowup() {
        let mut omega = Vec3::new(0.0, 100.0, 0.0);
        let speed = omega.length();
        if speed > ANGULAR_SPEED_CAP {
            omega *= ANGULAR_SPEED_CAP / speed;
        }
        assert!(
            omega.length() <= ANGULAR_SPEED_CAP + 1e-6,
            "Speed should be capped: {}",
            omega.length()
        );
    }

    #[test]
    fn zero_torque_preserves_angular_velocity() {
        let torque = Vec3::ZERO;
        let inv_i = Vec3::new(1.0, 1.0, 1.0);
        let dt = 1.0 / 60.0;
        let mut omega = Vec3::new(1.0, 2.0, 3.0);
        let original = omega;

        omega += torque * inv_i * dt;

        assert_eq!(omega, original);
    }
}
