//! Joint system — force-transmitting connections between adjacent building parts.
//!
//! Each joint connects two parts at a shared face. The joint tracks accumulated
//! mechanical stress and breaks when stress exceeds the weakest material strength.
//!
//! # Stress model (SI)
//! ```text
//! σ_compression = F_axial   / contact_area   [Pa]
//! σ_tension     = F_pullout / contact_area   [Pa]
//! σ_shear       = F_lateral / contact_area   [Pa]
//! ```
//! A joint fails when the relevant stress exceeds the material strength.
//! Strength is the minimum of the two connected materials.

use bevy::prelude::*;

use crate::data::MaterialData;

// ---------------------------------------------------------------------------
// JointType
// ---------------------------------------------------------------------------

/// How two parts are connected at their shared face.
///
/// The joint type determines which stress modes the joint resists and at
/// what strength fraction (a friction joint has lower effective tensile strength
/// than a mortared/welded joint).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JointType {
    /// Bonded with mortar, adhesive, nails, or welding.
    /// Full tensile + compressive + shear strength.
    Rigid,
    /// Dry-stacked (stone blocks, loose planks).
    /// Full compressive strength; zero tensile strength; partial shear.
    Friction,
    /// Pinned/hinged connection (door hinge, gate pivot).
    /// Transmits compressive + shear; does not resist rotation.
    Hinge,
}

impl JointType {
    /// Multiplier on tensile strength for this joint type.
    pub fn tensile_factor(&self) -> f32 {
        match self {
            JointType::Rigid => 1.0,
            JointType::Friction => 0.0,
            JointType::Hinge => 0.5,
        }
    }

    /// Multiplier on shear strength for this joint type.
    pub fn shear_factor(&self) -> f32 {
        match self {
            JointType::Rigid => 1.0,
            JointType::Friction => 0.3,
            JointType::Hinge => 0.8,
        }
    }
}

// ---------------------------------------------------------------------------
// Joint component
// ---------------------------------------------------------------------------

/// ECS component representing a structural connection between two building parts.
///
/// Placed as a component on a dedicated joint entity that references
/// `part_a` and `part_b` by `Entity`.
#[derive(Component, Debug, Clone)]
pub struct Joint {
    /// First part entity.
    pub part_a: Entity,
    /// Second part entity.
    pub part_b: Entity,
    /// Contact area at the shared face in m².
    /// For a 1×1 m face this is 1.0 m².
    pub contact_area_m2: f32,
    /// How the parts are bonded.
    pub joint_type: JointType,
    /// Current cumulative stress ratio (0.0 = intact, ≥ 1.0 = broken).
    pub stress_ratio: f32,
    /// Accumulated fatigue damage (0.0 – 1.0).
    pub damage: f32,
    /// Whether this joint has broken and should be removed.
    pub broken: bool,
}

impl Joint {
    /// Create a new joint with default Rigid bonding and 1 m² contact area.
    pub fn new(part_a: Entity, part_b: Entity) -> Self {
        Self {
            part_a,
            part_b,
            contact_area_m2: 1.0,
            joint_type: JointType::Rigid,
            stress_ratio: 0.0,
            damage: 0.0,
            broken: false,
        }
    }

    /// Apply an axial force (tension positive, compression negative) in Newtons.
    ///
    /// Returns `true` if the joint broke.
    pub fn apply_axial(
        &mut self,
        force_n: f32,
        mat_a: &MaterialData,
        mat_b: &MaterialData,
    ) -> bool {
        if self.broken {
            return true;
        }
        let stress = force_n.abs() / self.contact_area_m2.max(1e-6);
        let strength = if force_n > 0.0 {
            // Tension — use minimum tensile strength × joint factor
            let t_a = mat_a.tensile_strength.unwrap_or(0.0);
            let t_b = mat_b.tensile_strength.unwrap_or(0.0);
            t_a.min(t_b) * self.joint_type.tensile_factor()
        } else {
            // Compression — use minimum compressive strength
            let c_a = mat_a.compressive_strength.unwrap_or(0.0);
            let c_b = mat_b.compressive_strength.unwrap_or(0.0);
            c_a.min(c_b)
        };
        if strength < 1.0 {
            self.broken = true;
            return true;
        }
        self.stress_ratio = (self.stress_ratio + stress / strength).min(2.0);
        if self.stress_ratio >= 1.0 {
            self.broken = true;
            return true;
        }
        false
    }

    /// Apply a lateral (shear) force in Newtons.
    ///
    /// Returns `true` if the joint broke.
    pub fn apply_shear(
        &mut self,
        force_n: f32,
        mat_a: &MaterialData,
        mat_b: &MaterialData,
    ) -> bool {
        if self.broken {
            return true;
        }
        let stress = force_n.abs() / self.contact_area_m2.max(1e-6);
        let s_a = mat_a.shear_strength.unwrap_or(0.0);
        let s_b = mat_b.shear_strength.unwrap_or(0.0);
        let strength = s_a.min(s_b) * self.joint_type.shear_factor();
        if strength < 1.0 {
            self.broken = true;
            return true;
        }
        self.stress_ratio = (self.stress_ratio + stress / strength).min(2.0);
        if self.stress_ratio >= 1.0 {
            self.broken = true;
            return true;
        }
        false
    }
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

/// Remove broken joints from the world.
///
/// Broken joints are first detected by the stress analysis system, then
/// cleaned up here so downstream systems don't see them.
pub fn cleanup_broken_joints(mut commands: Commands, query: Query<(Entity, &Joint)>) {
    for (entity, joint) in &query {
        if joint.broken {
            commands.entity(entity).despawn();
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_material(tensile: f32, compressive: f32, shear: f32) -> MaterialData {
        MaterialData {
            id: 0,
            name: "test".to_string(),
            default_phase: crate::data::Phase::Solid,
            density: 1000.0,
            melting_point: None,
            boiling_point: None,
            ignition_point: None,
            hardness: 5.0,
            color: [0.5; 3],
            transparent: false,
            thermal_conductivity: 1.0,
            specific_heat_capacity: 1000.0,
            latent_heat_fusion: None,
            latent_heat_vaporization: None,
            emissivity: 0.9,
            absorption_coefficient: None,
            viscosity: None,
            friction_coefficient: 0.5,
            restitution: 0.3,
            youngs_modulus: Some(70e9),
            tensile_strength: Some(tensile),
            compressive_strength: Some(compressive),
            shear_strength: Some(shear),
            flexural_strength: Some(tensile * 1.2),
            fracture_toughness: Some(1e6),
            heat_of_combustion: None,
            molar_mass: None,
            refractive_index: None,
            reflectivity: None,
            absorption_rgb: None,
            cauchy_b: None,
            albedo: 0.3,
            melted_into: None,
            boiled_into: None,
            frozen_into: None,
            condensed_into: None,
        }
    }

    #[test]
    fn joint_survives_small_axial_force() {
        let mat = dummy_material(40e6, 30e6, 8e6);
        let e = Entity::from_bits(1);
        let mut j = Joint::new(e, e);
        // 10 kN on 1 m² → 10 kPa stress vs 40 MPa tensile → well within limits
        let broke = j.apply_axial(10_000.0, &mat, &mat);
        assert!(!broke);
        assert!(!j.broken);
    }

    #[test]
    fn joint_breaks_under_excess_tension() {
        let mat = dummy_material(1000.0, 30e6, 8e6); // 1 kPa tensile (very weak)
        let e = Entity::from_bits(2);
        let mut j = Joint::new(e, e);
        // 5 kN on 1 m² → 5 kPa > 1 kPa limit → breaks
        let broke = j.apply_axial(5_000.0, &mat, &mat);
        assert!(broke);
        assert!(j.broken);
    }

    #[test]
    fn friction_joint_has_zero_tensile_strength() {
        let mat = dummy_material(40e6, 30e6, 8e6);
        let e = Entity::from_bits(3);
        let mut j = Joint {
            part_a: e,
            part_b: e,
            contact_area_m2: 1.0,
            joint_type: JointType::Friction,
            stress_ratio: 0.0,
            damage: 0.0,
            broken: false,
        };
        // Any tension should break a friction joint
        let broke = j.apply_axial(1.0, &mat, &mat);
        assert!(broke);
    }

    #[test]
    fn joint_type_shear_factors() {
        assert_eq!(JointType::Rigid.shear_factor(), 1.0);
        assert!(JointType::Friction.shear_factor() < 1.0);
    }
}
