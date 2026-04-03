// Biological tissue layering and compound collision shapes.
//
// Models the physical cross-section of body regions as a sequence of
// cylindrical shells (bone → muscle → fat → skin), producing a radius
// used for capsule-based compound collision detection.
//
// All values are SI: lengths in metres, masses in kg, densities in kg/m³.

use std::collections::HashMap;

use bevy::prelude::*;
use serde::Deserialize;

// ---------------------------------------------------------------------------
// Asset data types
// ---------------------------------------------------------------------------

/// A single concentric layer of biological tissue around a bone.
#[derive(Deserialize, Debug, Clone)]
pub struct TissueLayer {
    /// Tissue type identifier (e.g. `"muscle"`, `"fat"`, `"skin"`).
    pub tissue: String,
    /// Layer thickness in metres.
    pub thickness: f32,
    /// Tissue density in kg/m³.
    ///
    /// Reference values: muscle ≈ 1060, fat ≈ 920, skin ≈ 1100 kg/m³.
    pub density: f32,
}

/// A named region of the body, associated with a bone and its tissue cross-section.
#[derive(Deserialize, Debug, Clone)]
pub struct BodyRegion {
    /// Region name (e.g. `"head"`, `"torso"`, `"left_arm"`).
    pub name: String,
    /// Name of the bone this region wraps.
    pub bone: String,
    /// Concentric tissue layers from innermost (closest to bone) outward.
    pub layers: Vec<TissueLayer>,
}

impl BodyRegion {
    /// Total cross-sectional radius in metres.
    ///
    /// `bone_radius = clamp(bone_length × 0.05, 0.02 m, 0.15 m)`
    /// then each tissue layer adds its `thickness`.
    pub fn total_radius(&self, bone_length: f32) -> f32 {
        let bone_radius = (bone_length * 0.05).clamp(0.02, 0.15);
        bone_radius + self.layers.iter().map(|l| l.thickness).sum::<f32>()
    }

    /// Total tissue mass in kg, computed from cylindrical shell volumes.
    ///
    /// For each layer: V = π × (r_outer² − r_inner²) × bone_length
    /// then mass += V × density.
    pub fn tissue_mass(&self, bone_length: f32) -> f32 {
        let bone_radius = (bone_length * 0.05).clamp(0.02, 0.15);
        let mut inner = bone_radius;
        let mut total = 0.0_f32;

        for layer in &self.layers {
            let outer = inner + layer.thickness;
            let volume = std::f32::consts::PI * (outer * outer - inner * inner) * bone_length;
            total += volume * layer.density;
            inner = outer;
        }

        total
    }
}

/// Body composition data loaded from a `.body.ron` file.
///
/// Registered via `RonAssetPlugin::<BodyData>::new(&["body.ron"])`.
#[derive(Deserialize, Asset, TypePath, Debug, Clone)]
pub struct BodyData {
    /// Species identifier (matches skeleton and creature data).
    pub species: String,
    /// Per-region tissue descriptions.
    pub regions: Vec<BodyRegion>,
}

// ---------------------------------------------------------------------------
// Runtime collision shapes
// ---------------------------------------------------------------------------

/// A capsule collision shape for a single body region.
///
/// Used by `CompoundCollider` to build a multi-shape collider from body data.
#[derive(Debug, Clone)]
pub struct CapsuleShape {
    /// Name of the body region this capsule represents.
    pub region: String,
    /// Offset from the entity's world position (bone-relative).
    pub offset: Vec3,
    /// Capsule radius in metres.
    pub radius: f32,
    /// Half-height of the cylindrical section in metres.
    pub half_height: f32,
}

impl CapsuleShape {
    /// Compute the axis-aligned bounding box of this capsule at the given entity position.
    ///
    /// Returns `(min, max)` corner vectors in world space.
    pub fn aabb(&self, entity_pos: Vec3) -> (Vec3, Vec3) {
        let centre = entity_pos + self.offset;
        let extent = Vec3::new(self.radius, self.half_height + self.radius, self.radius);
        (centre - extent, centre + extent)
    }
}

/// Compound collider built from multiple capsule shapes, one per body region.
///
/// The `aabb_half_extents` field is the conservative half-extents of the
/// union AABB over all capsules, centred on the entity origin.
#[derive(Component, Debug, Clone)]
pub struct CompoundCollider {
    /// Per-region capsule shapes.
    pub capsules: Vec<CapsuleShape>,
    /// Half-extents of the enclosing AABB in metres.
    pub aabb_half_extents: Vec3,
}

impl CompoundCollider {
    /// Build a `CompoundCollider` from body data and a map of bone lengths.
    ///
    /// Each region produces one capsule. Bone lengths default to `1.0 m`
    /// if the bone name is not found in `bone_lengths`.
    pub fn from_body_data(body: &BodyData, bone_lengths: &HashMap<String, f32>) -> Self {
        let mut capsules = Vec::with_capacity(body.regions.len());
        let mut union_min = Vec3::splat(f32::MAX);
        let mut union_max = Vec3::splat(f32::MIN);

        for region in &body.regions {
            let bone_len = *bone_lengths.get(&region.bone).unwrap_or(&1.0);
            let radius = region.total_radius(bone_len);
            let half_height = bone_len * 0.5;
            let offset = region_offset(&region.name, bone_len);

            let capsule = CapsuleShape {
                region: region.name.clone(),
                offset,
                radius,
                half_height,
            };

            // Accumulate AABB at entity origin (pos = Vec3::ZERO)
            let (mn, mx) = capsule.aabb(Vec3::ZERO);
            union_min = union_min.min(mn);
            union_max = union_max.max(mx);

            capsules.push(capsule);
        }

        let aabb_half_extents = if capsules.is_empty() {
            Vec3::ZERO
        } else {
            // Half-extents from the union corners
            (union_max - union_min) * 0.5
        };

        Self {
            capsules,
            aabb_half_extents,
        }
    }

    /// Iterate over world-space AABBs for each capsule at the given entity position.
    pub fn aabbs<'a>(&'a self, entity_pos: Vec3) -> impl Iterator<Item = (Vec3, Vec3)> + 'a {
        self.capsules.iter().map(move |c| c.aabb(entity_pos))
    }
}

// ---------------------------------------------------------------------------
// Region offset heuristic
// ---------------------------------------------------------------------------

/// Estimate the body-region capsule offset from the entity origin.
///
/// Based on the region name: torso/chest regions centre at origin, limbs
/// are offset along Y by a fraction of their bone length.
pub fn region_offset(name: &str, bone_len: f32) -> Vec3 {
    if name.contains("head") {
        Vec3::Y * bone_len * 0.8
    } else if name.contains("torso") || name.contains("chest") || name.contains("trunk") {
        Vec3::ZERO
    } else if name.contains("pelvis") || name.contains("hip") {
        Vec3::NEG_Y * bone_len * 0.3
    } else if name.contains("upper_arm") || name.contains("upper_leg") || name.contains("thigh") {
        Vec3::Y * bone_len * 0.25
    } else if name.contains("lower_arm") || name.contains("forearm") || name.contains("shin") {
        Vec3::NEG_Y * bone_len * 0.25
    } else {
        Vec3::ZERO
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn muscle_fat_skin_region(name: &str, bone: &str) -> BodyRegion {
        BodyRegion {
            name: name.into(),
            bone: bone.into(),
            layers: vec![
                TissueLayer {
                    tissue: "muscle".into(),
                    thickness: 0.03,
                    density: 1060.0,
                },
                TissueLayer {
                    tissue: "fat".into(),
                    thickness: 0.01,
                    density: 920.0,
                },
                TissueLayer {
                    tissue: "skin".into(),
                    thickness: 0.002,
                    density: 1100.0,
                },
            ],
        }
    }

    /// total_radius = bone_radius + sum(thicknesses)
    #[test]
    fn total_radius_sums_correctly() {
        let region = muscle_fat_skin_region("torso", "spine");
        // bone_length = 0.5 m → bone_radius = clamp(0.025, 0.02, 0.15) = 0.025
        let bone_length = 0.5;
        let expected_radius = 0.025 + 0.03 + 0.01 + 0.002;
        let computed = region.total_radius(bone_length);
        assert!(
            (computed - expected_radius).abs() < 1e-5,
            "expected {expected_radius}, got {computed}"
        );
    }

    /// CompoundCollider should produce one capsule per region.
    #[test]
    fn compound_collider_correct_capsule_count() {
        let body = BodyData {
            species: "test".into(),
            regions: vec![
                muscle_fat_skin_region("head", "head_bone"),
                muscle_fat_skin_region("torso", "spine"),
                muscle_fat_skin_region("left_arm", "left_upper_arm"),
            ],
        };

        let bone_lengths: HashMap<String, f32> = [
            ("head_bone".to_string(), 0.22),
            ("spine".to_string(), 0.5),
            ("left_upper_arm".to_string(), 0.3),
        ]
        .into();

        let collider = CompoundCollider::from_body_data(&body, &bone_lengths);
        assert_eq!(
            collider.capsules.len(),
            3,
            "expected 3 capsules, got {}",
            collider.capsules.len()
        );
        // aabb_half_extents should be non-zero
        assert!(
            collider.aabb_half_extents.length() > 0.0,
            "aabb_half_extents should be non-zero"
        );
    }
}
