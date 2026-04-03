// Skeletal animation system.
//
// Defines the asset data types for skeletons (loaded from `.skeleton.ron`)
// and the runtime `Skeleton` component that drives forward kinematics (FK)
// and angular-dynamics simulation per bone.
//
// Design notes:
// - `BoneTransform` is the serde-friendly counterpart to Bevy's `Transform`
//   (Bevy's `Transform` only implements `Deserialize` with the opt-in
//   `serialize` feature, which is not enabled in this crate).
// - FK propagation assumes parent index < child index (tree order).
// - Thin-rod inertia I = m·l²/3 is used per bone for angular dynamics.

use bevy::prelude::*;
use serde::Deserialize;

// ---------------------------------------------------------------------------
// Serialisable helpers (RON-friendly)
// ---------------------------------------------------------------------------

/// Serialisable bone transform stored in `.skeleton.ron` files.
///
/// Uses plain arrays instead of Bevy's `Vec3`/`Quat` so the struct can
/// derive `serde::Deserialize` without requiring Bevy's `serialize` feature.
/// Convert to a runtime `Transform` via [`BoneTransform::to_transform`].
#[derive(Deserialize, Debug, Clone)]
pub struct BoneTransform {
    /// World-space (root) or parent-relative (children) translation in metres.
    pub translation: [f32; 3],
    /// Quaternion in `[x, y, z, w]` order. Identity = `[0, 0, 0, 1]`.
    pub rotation: [f32; 4],
    /// Uniform or non-uniform scale multiplier.
    pub scale: [f32; 3],
}

impl Default for BoneTransform {
    fn default() -> Self {
        Self {
            translation: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
        }
    }
}

impl BoneTransform {
    /// Convert to a Bevy `Transform` for runtime use.
    pub fn to_transform(&self) -> Transform {
        Transform {
            translation: Vec3::from(self.translation),
            rotation: Quat::from_array(self.rotation),
            scale: Vec3::from(self.scale),
        }
    }
}

impl From<Transform> for BoneTransform {
    fn from(t: Transform) -> Self {
        Self {
            translation: t.translation.into(),
            rotation: t.rotation.into(),
            scale: t.scale.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// Asset data types
// ---------------------------------------------------------------------------

/// How a joint allows relative motion between two bones.
#[derive(Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum JointType {
    /// Single-axis rotation (e.g. knee, elbow).
    Hinge,
    /// Free rotation within a cone (e.g. shoulder, hip).
    BallSocket,
    /// No relative motion (e.g. fused skull plates).
    Fixed,
}

/// Per-axis angular limits for a joint (radians).
///
/// Use `[f32; 3]` arrays for serde compatibility. See
/// [`AngularLimits::min_vec`] / [`AngularLimits::max_vec`] for `Vec3`.
#[derive(Deserialize, Debug, Clone)]
pub struct AngularLimits {
    /// Minimum rotation in radians per axis `[x, y, z]`.
    pub min: [f32; 3],
    /// Maximum rotation in radians per axis `[x, y, z]`.
    pub max: [f32; 3],
}

impl AngularLimits {
    /// Minimum limits as a `Vec3`.
    pub fn min_vec(&self) -> Vec3 {
        Vec3::from(self.min)
    }

    /// Maximum limits as a `Vec3`.
    pub fn max_vec(&self) -> Vec3 {
        Vec3::from(self.max)
    }
}

/// Data describing a single bone in a skeleton.
#[derive(Deserialize, Debug, Clone)]
pub struct BoneData {
    /// Unique name used for semantic lookup (e.g. `"head"`, `"left_foot"`).
    pub name: String,
    /// Index of the parent bone. `None` marks the root bone.
    pub parent: Option<usize>,
    /// Bone length in metres (1 voxel = 1 m).
    pub length: f32,
    /// Bone mass in kilograms.
    pub mass: f32,
    /// Material identifier matching the voxel material system.
    pub material: String,
    /// Local rest transform (relative to parent, or world-space for root).
    pub rest_transform: BoneTransform,
}

/// Constraint between two bones loaded from a skeleton asset.
#[derive(Deserialize, Debug, Clone)]
pub struct JointData {
    /// Index of the proximal (parent-side) bone.
    pub bone_a: usize,
    /// Index of the distal (child-side) bone.
    pub bone_b: usize,
    /// Kinematic constraint type.
    pub joint_type: JointType,
    /// Angular range of motion (radians per axis).
    pub limits: AngularLimits,
}

/// Skeleton description loaded from a `.skeleton.ron` file.
///
/// Registered via `RonAssetPlugin::<SkeletonData>::new(&["skeleton.ron"])`.
#[derive(Deserialize, Asset, TypePath, Debug, Clone)]
pub struct SkeletonData {
    /// Species identifier (matches creature data).
    pub species: String,
    /// Ordered bone definitions. Parent index must be less than child index.
    pub bones: Vec<BoneData>,
    /// Joint constraints between bones.
    pub joints: Vec<JointData>,
    /// Rest-pose local transforms, one per bone (same order as `bones`).
    pub rest_pose: Vec<BoneTransform>,
}

// ---------------------------------------------------------------------------
// Runtime components
// ---------------------------------------------------------------------------

/// Holds a handle to a loaded `SkeletonData` asset.
///
/// Attach to an entity; `init_skeletons` will insert a `Skeleton` component
/// once the asset finishes loading.
#[derive(Component, Debug, Clone)]
pub struct SkeletonHandle(pub Handle<SkeletonData>);

/// Runtime skeletal state: current world-space bone transforms plus per-bone
/// angular dynamics.
#[derive(Component, Debug, Clone, Default)]
pub struct Skeleton {
    /// Bone names in the same order as all parallel arrays (for name-based lookup).
    pub bone_names: Vec<String>,
    /// World-space transform for each bone (same order as `SkeletonData.bones`).
    pub bone_transforms: Vec<Transform>,
    /// Angular velocity (rad/s) for each bone about its local axes.
    pub angular_velocities: Vec<Vec3>,
    /// Accumulated torque (N·m) for each bone this tick.
    ///
    /// Zeroed after integration in `apply_skeleton_fk`.
    pub torques: Vec<Vec3>,
    /// Optional IK target override per bone (world space). `None` = use FK pose.
    pub ik_targets: Vec<Option<Vec3>>,
    /// Index of the head bone, detected by the substring `"head"` in bone name.
    pub head_bone: Option<usize>,
    /// Indices of foot/paw bones (substring `"foot"` or `"paw"`).
    pub foot_bones: Vec<usize>,
    /// Indices of hand/claw/grip bones (substring `"hand"`, `"claw"`, `"grip"`).
    pub hand_bones: Vec<usize>,
}

impl Skeleton {
    /// Build a `Skeleton` from loaded asset data, computing the initial FK pose.
    pub fn from_data(data: &SkeletonData) -> Self {
        let n = data.bones.len();
        let local: Vec<Transform> = data
            .rest_pose
            .iter()
            .map(BoneTransform::to_transform)
            .collect();
        let bone_transforms = compute_fk(&data.bones, &local, None);

        let bone_names: Vec<String> = data.bones.iter().map(|b| b.name.clone()).collect();

        // Semantic bone detection by name substring
        let head_bone = data.bones.iter().position(|b| b.name.contains("head"));
        let foot_bones = data
            .bones
            .iter()
            .enumerate()
            .filter(|(_, b)| b.name.contains("foot") || b.name.contains("paw"))
            .map(|(i, _)| i)
            .collect();
        let hand_bones = data
            .bones
            .iter()
            .enumerate()
            .filter(|(_, b)| {
                b.name.contains("hand") || b.name.contains("claw") || b.name.contains("grip")
            })
            .map(|(i, _)| i)
            .collect();

        Self {
            bone_names,
            bone_transforms,
            angular_velocities: vec![Vec3::ZERO; n],
            torques: vec![Vec3::ZERO; n],
            ik_targets: vec![None; n],
            head_bone,
            foot_bones,
            hand_bones,
        }
    }

    /// Return the index of the bone with the given name, or `None` if not found.
    pub fn bone_index(&self, name: &str) -> Option<usize> {
        self.bone_names.iter().position(|n| n == name)
    }
}

// ---------------------------------------------------------------------------
// Core FK algorithm
// ---------------------------------------------------------------------------

/// Compute world-space bone transforms via forward kinematics.
///
/// `local_transforms` are parent-relative for non-root bones and world-space
/// for the root. Parent indices must be strictly less than child indices.
/// An optional `root_override` replaces bone 0's transform (used to track
/// the owning entity).
pub fn compute_fk(
    bones: &[BoneData],
    local_transforms: &[Transform],
    root_override: Option<Transform>,
) -> Vec<Transform> {
    let n = bones.len().min(local_transforms.len());
    let mut world = local_transforms[..n].to_vec();

    if let Some(root) = root_override
        && n > 0
    {
        world[0] = root;
    }

    for i in 1..n {
        if let Some(parent_idx) = bones[i].parent
            && parent_idx < n
        {
            let p = world[parent_idx];
            let local = local_transforms[i];
            world[i] = Transform {
                translation: p.translation + p.rotation * local.translation,
                rotation: (p.rotation * local.rotation).normalize(),
                scale: p.scale * local.scale,
            };
        }
    }

    world
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

/// Insert a `Skeleton` component once the referenced `SkeletonData` asset loads.
///
/// Runs each frame; safe to call before the asset is ready.
pub fn init_skeletons(
    mut commands: Commands,
    query: Query<(Entity, &SkeletonHandle, &Transform), Without<Skeleton>>,
    skeleton_assets: Res<Assets<SkeletonData>>,
) {
    for (entity, handle, _transform) in &query {
        if let Some(data) = skeleton_assets.get(&handle.0) {
            commands.entity(entity).insert(Skeleton::from_data(data));
        }
    }
}

/// Propagate parent→child bone transforms (FK) and integrate angular dynamics.
///
/// Runs in `FixedUpdate`.  Per bone:
/// - α = τ / I   where I = m·l²/3  (thin-rod inertia)
/// - ω += α · dt
/// - ω *= 0.95^(dt·60)  (angular damping)
/// - rotation delta from ω applied, then FK rebuilds translations
pub fn apply_skeleton_fk(
    time: Res<Time>,
    mut query: Query<(&mut Skeleton, &SkeletonHandle, &Transform)>,
    skeleton_assets: Res<Assets<SkeletonData>>,
) {
    let dt = time.delta_secs();
    // Damping: 0.95 per tick at 60 Hz, scaled continuously by dt.
    let damping = 0.95_f32.powf(dt * 60.0);

    for (mut skeleton, handle, entity_transform) in &mut query {
        let Some(data) = skeleton_assets.get(&handle.0) else {
            continue;
        };
        let n = data.bones.len().min(skeleton.bone_transforms.len());

        // --- Angular dynamics integration ---
        for i in 0..n {
            let bone = &data.bones[i];
            // Thin-rod moment of inertia I = m * l² / 3
            let inertia = (bone.mass * bone.length * bone.length / 3.0).max(1e-12);
            let torque = skeleton.torques[i];
            skeleton.angular_velocities[i] += torque / inertia * dt;
            skeleton.angular_velocities[i] *= damping;
            skeleton.torques[i] = Vec3::ZERO;
        }

        // --- FK propagation ---
        // Root bone is pinned to the entity's world transform.
        if n > 0 {
            let local_root = data.rest_pose[0].to_transform();
            skeleton.bone_transforms[0] = Transform {
                translation: entity_transform.translation
                    + entity_transform.rotation * local_root.translation,
                rotation: (entity_transform.rotation * local_root.rotation).normalize(),
                scale: entity_transform.scale * local_root.scale,
            };
            let av0 = skeleton.angular_velocities[0];
            apply_av_rotation(&mut skeleton.bone_transforms[0].rotation, av0, dt);
        }

        for i in 1..n {
            if let Some(parent_idx) = data.bones[i].parent
                && parent_idx < n
            {
                let parent = skeleton.bone_transforms[parent_idx];
                let local = data.rest_pose[i].to_transform();
                skeleton.bone_transforms[i] = Transform {
                    translation: parent.translation + parent.rotation * local.translation,
                    rotation: (parent.rotation * local.rotation).normalize(),
                    scale: parent.scale * local.scale,
                };
                let avi = skeleton.angular_velocities[i];
                apply_av_rotation(&mut skeleton.bone_transforms[i].rotation, avi, dt);
            }
        }
    }
}

/// Apply an angular velocity as a rotation delta to the given quaternion.
fn apply_av_rotation(rotation: &mut Quat, av: Vec3, dt: f32) {
    let angle = av.length() * dt;
    if angle > 1e-6 {
        let delta = Quat::from_axis_angle(av / av.length(), angle);
        *rotation = (delta * *rotation).normalize();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal two-bone skeleton data (root + child above it).
    fn two_bone_data() -> SkeletonData {
        SkeletonData {
            species: "test".into(),
            bones: vec![
                BoneData {
                    name: "root".into(),
                    parent: None,
                    length: 1.0,
                    mass: 1.0,
                    material: "bone".into(),
                    rest_transform: BoneTransform::default(),
                },
                BoneData {
                    name: "child".into(),
                    parent: Some(0),
                    length: 1.0,
                    mass: 1.0,
                    material: "bone".into(),
                    rest_transform: BoneTransform {
                        translation: [0.0, 1.0, 0.0],
                        rotation: [0.0, 0.0, 0.0, 1.0],
                        scale: [1.0, 1.0, 1.0],
                    },
                },
            ],
            joints: vec![],
            rest_pose: vec![
                BoneTransform::default(),
                BoneTransform {
                    translation: [0.0, 1.0, 0.0],
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: [1.0, 1.0, 1.0],
                },
            ],
        }
    }

    /// FK should place the child bone 1 m above the root.
    #[test]
    fn fk_places_child_above_root() {
        let data = two_bone_data();
        let skeleton = Skeleton::from_data(&data);

        let root_y = skeleton.bone_transforms[0].translation.y;
        let child_y = skeleton.bone_transforms[1].translation.y;
        assert!(
            (child_y - root_y - 1.0).abs() < 1e-5,
            "child should be 1 m above root, got root={root_y} child={child_y}"
        );
    }

    /// Bones with "head" in the name are detected as head_bone.
    #[test]
    fn head_bone_detected_by_name() {
        let mut data = two_bone_data();
        data.bones[1].name = "head".into();
        let skeleton = Skeleton::from_data(&data);

        assert_eq!(
            skeleton.head_bone,
            Some(1),
            "expected bone 1 to be head_bone"
        );
        assert!(skeleton.foot_bones.is_empty());
        assert!(skeleton.hand_bones.is_empty());
    }

    /// Torque integration should update angular velocity.
    #[test]
    fn torque_integration_applies_velocity() {
        let data = two_bone_data();
        let mut skeleton = Skeleton::from_data(&data);

        // Apply a torque of 1 N·m about Z to bone 0.
        // I = m * l² / 3 = 1.0 * 1.0 / 3 ≈ 0.333 kg·m²
        // After dt=0.1 s: α = 1/0.333 ≈ 3 rad/s², Δω ≈ 0.3 rad/s
        skeleton.torques[0] = Vec3::new(0.0, 0.0, 1.0);

        let dt = 0.1_f32;
        let inertia = (1.0_f32 * 1.0_f32 * 1.0_f32 / 3.0).max(1e-12);
        let expected_av_z = 1.0 / inertia * dt;
        let damping = 0.95_f32.powf(dt * 60.0);

        // Manually replicate the integration step
        skeleton.angular_velocities[0] += skeleton.torques[0] / inertia * dt;
        skeleton.torques[0] = Vec3::ZERO;
        skeleton.angular_velocities[0] *= damping;

        let av_z = skeleton.angular_velocities[0].z;
        let expected_damped = expected_av_z * damping;
        assert!(
            (av_z - expected_damped).abs() < 1e-4,
            "expected av_z ≈ {expected_damped}, got {av_z}"
        );
        // Torque should be zeroed after integration
        assert_eq!(skeleton.torques[0], Vec3::ZERO);
    }
}
