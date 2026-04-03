// FABRIK (Forward And Backward Reaching Inverse Kinematics) solver.
//
// Solves multi-bone IK chains iteratively.  The algorithm alternates a
// forward pass (tip → root) with a backward pass (root → tip) until the
// tip converges to the target or the iteration limit is reached.
//
// Reference: Aristidou & Lasenby, "FABRIK: A fast, iterative solver for
// the Inverse Kinematics problem", Graphical Models 73(5), 2011.

use bevy::prelude::*;

use super::skeleton::{AngularLimits, Skeleton};

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

/// Tuning parameters for the FABRIK solver.
#[derive(Debug, Clone)]
pub struct FabrikParams {
    /// Maximum number of forward + backward pass iterations.
    pub max_iters: usize,
    /// Stop early when the tip is within this distance (metres) of the target.
    pub tolerance: f32,
}

impl Default for FabrikParams {
    fn default() -> Self {
        Self {
            max_iters: 10,
            tolerance: 0.01,
        }
    }
}

// ---------------------------------------------------------------------------
// IK target component
// ---------------------------------------------------------------------------

/// ECS component that drives IK on a named bone chain.
#[derive(Component, Debug, Clone)]
pub struct IkTarget {
    /// Name of the tip bone to solve for (e.g. `"left_hand"`).
    pub bone_name: String,
    /// World-space target position in metres.
    pub target: Vec3,
    /// Ordered chain of bone indices from root to tip.
    pub chain: Vec<usize>,
    /// If `false`, the solver is skipped this tick.
    pub active: bool,
}

// ---------------------------------------------------------------------------
// FABRIK solver
// ---------------------------------------------------------------------------

/// Solve a bone chain using the FABRIK algorithm.
///
/// # Arguments
/// - `positions`  World-space positions of each joint (N joints for N-1 bones).
/// - `lengths`    Length of each bone segment; `lengths[i]` connects `positions[i]`
///   to `positions[i+1]`. Must have `len() == positions.len() - 1`.
/// - `target`    Desired world-space position for the tip joint.
/// - `params`    Solver iteration limits.
/// - `limits`    Per-bone angular limits (may be shorter than `positions`; extras ignored).
///
/// Returns the solved joint positions.
pub fn solve_fabrik(
    positions: &[Vec3],
    lengths: &[f32],
    target: Vec3,
    params: &FabrikParams,
    limits: &[AngularLimits],
) -> Vec<Vec3> {
    let n = positions.len();
    if n == 0 || lengths.is_empty() {
        return positions.to_vec();
    }
    // Need at least one segment
    if lengths.len() < n.saturating_sub(1) {
        return positions.to_vec();
    }

    let total_reach: f32 = lengths.iter().copied().sum();
    let root = positions[0];
    let dist_to_target = (target - root).length();

    // --- Out-of-reach: stretch the chain toward the target ---
    if dist_to_target >= total_reach {
        let mut out = vec![Vec3::ZERO; n];
        out[0] = root;
        for i in 1..n {
            let dir = (target - out[i - 1]).normalize_or_zero();
            out[i] = out[i - 1] + dir * lengths[i - 1];
        }
        return out;
    }

    // --- Iterative FABRIK passes ---
    let mut p = positions.to_vec();

    for _ in 0..params.max_iters {
        // Forward pass: tip → root
        p[n - 1] = target;
        for i in (0..n - 1).rev() {
            let dir = (p[i] - p[i + 1]).normalize_or_zero();
            p[i] = p[i + 1] + dir * lengths[i];
        }

        // Backward pass: root → tip
        p[0] = root;
        for i in 0..n - 1 {
            let dir = (p[i + 1] - p[i]).normalize_or_zero();
            p[i + 1] = p[i] + dir * lengths[i];
        }

        // Enforce angular limits
        enforce_limits(&mut p, limits);

        // Convergence check
        if (p[n - 1] - target).length() < params.tolerance {
            break;
        }
    }

    p
}

/// Apply IK-solved joint positions back to skeleton bone transforms.
///
/// Updates `translation` for each bone in `chain` and sets `rotation` to
/// face the next joint in the chain using `Quat::from_rotation_arc(Y, dir)`.
pub fn apply_ik_to_skeleton(skeleton: &mut Skeleton, chain: &[usize], solved: &[Vec3]) {
    for (ci, &bone_i) in chain.iter().enumerate() {
        if bone_i >= skeleton.bone_transforms.len() || ci >= solved.len() {
            continue;
        }

        skeleton.bone_transforms[bone_i].translation = solved[ci];

        // Orient bone from current joint toward the next
        if ci + 1 < solved.len() {
            let dir = (solved[ci + 1] - solved[ci]).normalize_or_zero();
            if dir.length_squared() > 1e-6 {
                skeleton.bone_transforms[bone_i].rotation = Quat::from_rotation_arc(Vec3::Y, dir);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Clamp each bone's direction relative to its parent using per-joint limits.
///
/// For each bone i (index 1..n-1), the angle between the parent direction
/// and the bone direction is clamped to the cone defined by `limits[i-1]`.
/// This is a simplified spherical clamp suitable for real-time use.
fn enforce_limits(positions: &mut [Vec3], limits: &[AngularLimits]) {
    let n = positions.len();
    for i in 1..n {
        let limit_idx = i - 1;
        if limit_idx >= limits.len() {
            break;
        }

        let limit = &limits[limit_idx];

        // Parent direction (normalised)
        let parent_dir = if i >= 2 {
            (positions[i - 1] - positions[i - 2]).normalize_or_zero()
        } else {
            Vec3::Y // root has no parent; use world-up as reference
        };

        // Current bone direction
        let bone_dir = (positions[i] - positions[i - 1]).normalize_or_zero();

        // Compute the angle between parent_dir and bone_dir
        let cos_angle = parent_dir.dot(bone_dir).clamp(-1.0, 1.0);
        let angle = cos_angle.acos();

        // Use the symmetric max limit: the tightest of the three axes
        // (simplified — a proper implementation would decompose into Euler
        // swing/twist, but this is sufficient for real-time organic bodies).
        let max_angle = limit
            .max
            .iter()
            .copied()
            .map(f32::abs)
            .fold(f32::MAX, f32::min)
            .max(1e-4);

        if angle > max_angle {
            // Rotate bone_dir toward parent_dir until it just satisfies the limit
            let rotation_axis = parent_dir.cross(bone_dir).normalize_or_zero();
            if rotation_axis.length_squared() > 1e-6 {
                let clamped_rot = Quat::from_axis_angle(rotation_axis, max_angle);
                let clamped_dir = (clamped_rot * parent_dir).normalize_or_zero();
                let seg_len = (positions[i] - positions[i - 1]).length();
                positions[i] = positions[i - 1] + clamped_dir * seg_len;
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

    /// Build a straight chain of N joints along +Y, each 1 m apart.
    fn straight_chain(n: usize) -> (Vec<Vec3>, Vec<f32>) {
        let positions: Vec<Vec3> = (0..n).map(|i| Vec3::Y * i as f32).collect();
        let lengths = vec![1.0_f32; n - 1];
        (positions, lengths)
    }

    /// Solver should converge close to an in-reach target.
    #[test]
    fn converges_near_in_range_target() {
        let (positions, lengths) = straight_chain(4);
        let target = Vec3::new(1.5, 2.0, 0.5);
        let params = FabrikParams::default();
        let limits: Vec<AngularLimits> = vec![];

        let solved = solve_fabrik(&positions, &lengths, target, &params, &limits);

        let tip = solved[solved.len() - 1];
        let dist = (tip - target).length();
        assert!(
            dist < 0.05,
            "tip should be within 0.05 m of target, got dist={dist:.4}"
        );
    }

    /// For an out-of-reach target the chain should be fully extended.
    #[test]
    fn stretches_for_out_of_reach_target() {
        let (positions, lengths) = straight_chain(4);
        let total_reach: f32 = lengths.iter().sum(); // 3 m
        let far_target = Vec3::new(0.0, 100.0, 0.0); // way out of reach
        let params = FabrikParams::default();

        let solved = solve_fabrik(&positions, &lengths, far_target, &params, &[]);

        // Each segment should be at its full length
        for i in 0..solved.len() - 1 {
            let seg_len = (solved[i + 1] - solved[i]).length();
            assert!(
                (seg_len - lengths[i]).abs() < 1e-4,
                "segment {i} length {seg_len:.4} should equal {:.4}",
                lengths[i]
            );
        }

        // Chain extends toward target: total length == total_reach
        let chain_len: f32 = (0..solved.len() - 1)
            .map(|i| (solved[i + 1] - solved[i]).length())
            .sum();
        assert!(
            (chain_len - total_reach).abs() < 1e-3,
            "total chain length {chain_len:.4} should be {total_reach:.4}"
        );
    }

    /// Bone lengths must be preserved to within 2 % after solving.
    #[test]
    fn preserves_bone_lengths_within_2_percent() {
        let (positions, lengths) = straight_chain(5);
        let target = Vec3::new(1.0, 3.0, 1.0);
        let params = FabrikParams {
            max_iters: 30,
            tolerance: 0.001,
        };

        let solved = solve_fabrik(&positions, &lengths, target, &params, &[]);

        for (i, &expected_len) in lengths.iter().enumerate() {
            let actual_len = (solved[i + 1] - solved[i]).length();
            let error = (actual_len - expected_len).abs() / expected_len;
            assert!(
                error < 0.02,
                "segment {i}: length error {:.1}% exceeds 2%",
                error * 100.0
            );
        }
    }
}
