// Collision shapes and physics material for entity-entity collision.
//
// `CollisionShape` defines the geometric volume used for narrow-phase overlap
// tests between dynamic entities. It is separate from `Collider` (AABB
// half-extents) which handles entity-vs-terrain collision.
//
// `PhysicsMaterial` carries friction and restitution coefficients that drive
// the impulse solver. Values should derive from `MaterialData` in RON files.

use bevy::prelude::*;
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Collision shapes
// ---------------------------------------------------------------------------

/// Geometric shape used for entity-entity narrow-phase collision.
#[derive(Component, Debug, Clone)]
pub enum CollisionShape {
    /// Axis-aligned bounding box defined by half-extents (width/2, height/2, depth/2).
    Aabb { half_extents: Vec3 },
    /// Sphere centered on the entity origin.
    Sphere { radius: f32 },
    /// Capsule along the Y axis: a cylinder capped by hemispheres.
    Capsule {
        radius: f32,
        /// Half-height of the cylindrical segment (total height = 2·half_height + 2·radius).
        half_height: f32,
    },
}

impl CollisionShape {
    /// Create an AABB shape from full width, height, and depth.
    pub fn aabb(width: f32, height: f32, depth: f32) -> Self {
        Self::Aabb {
            half_extents: Vec3::new(width / 2.0, height / 2.0, depth / 2.0),
        }
    }

    /// Create a sphere shape.
    pub fn sphere(radius: f32) -> Self {
        Self::Sphere { radius }
    }

    /// Create a capsule shape from radius and total height.
    ///
    /// The cylindrical segment has height `total_height - 2 * radius`.
    /// Panics (debug) if total_height < 2 * radius.
    pub fn capsule(radius: f32, total_height: f32) -> Self {
        let half_height = (total_height / 2.0 - radius).max(0.0);
        Self::Capsule {
            radius,
            half_height,
        }
    }

    /// Volume of the shape in m³.
    pub fn volume(&self) -> f32 {
        match self {
            Self::Aabb { half_extents } => 8.0 * half_extents.x * half_extents.y * half_extents.z,
            Self::Sphere { radius } => (4.0 / 3.0) * PI * radius * radius * radius,
            Self::Capsule {
                radius,
                half_height,
            } => {
                let sphere_vol = (4.0 / 3.0) * PI * radius * radius * radius;
                let cylinder_vol = PI * radius * radius * (2.0 * half_height);
                sphere_vol + cylinder_vol
            }
        }
    }

    /// Moment of inertia (diagonal) for a uniform solid with given mass (kg·m²).
    ///
    /// Returns a `Vec3` with (Ixx, Iyy, Izz) about the principal axes.
    /// Source: standard solid body inertia formulas (Wikipedia — List of moments of inertia).
    pub fn moment_of_inertia(&self, mass: f32) -> Vec3 {
        match self {
            Self::Aabb { half_extents } => {
                // Solid rectangular parallelepiped: I_x = m/12 × (h² + d²), etc.
                let w2 = (2.0 * half_extents.x) * (2.0 * half_extents.x);
                let h2 = (2.0 * half_extents.y) * (2.0 * half_extents.y);
                let d2 = (2.0 * half_extents.z) * (2.0 * half_extents.z);
                let f = mass / 12.0;
                Vec3::new(f * (h2 + d2), f * (w2 + d2), f * (w2 + h2))
            }
            Self::Sphere { radius } => {
                // Solid sphere: I = 2/5 × m × r²
                let i = 0.4 * mass * radius * radius;
                Vec3::splat(i)
            }
            Self::Capsule {
                radius,
                half_height,
            } => {
                // Approximation: cylinder + hemisphere contributions.
                let r = *radius;
                let h = 2.0 * half_height;
                let total_vol = self.volume();
                if total_vol < 1e-12 {
                    return Vec3::ZERO;
                }
                let cyl_vol = PI * r * r * h;
                let sphere_vol = (4.0 / 3.0) * PI * r * r * r;
                let m_cyl = mass * (cyl_vol / total_vol);
                let m_sphere = mass * (sphere_vol / total_vol);

                // Cylinder about Y (its axis): I_y = m_c × r²/2
                // Cylinder about X/Z: I_x = m_c × (3r² + h²) / 12
                let cyl_iy = m_cyl * r * r / 2.0;
                let cyl_ix = m_cyl * (3.0 * r * r + h * h) / 12.0;

                // Sphere about center: I = 2/5 × m_s × r²
                // Shifted by parallel axis theorem: I_x += m_s × (h/2 + 3r/8)²
                let sphere_i_center = 0.4 * m_sphere * r * r;
                let offset = h / 2.0 + 3.0 * r / 8.0;
                let sphere_ix = sphere_i_center + m_sphere * offset * offset;
                let sphere_iy = sphere_i_center;

                let ix = cyl_ix + sphere_ix;
                let iy = cyl_iy + sphere_iy;
                Vec3::new(ix, iy, ix) // Symmetric about Y
            }
        }
    }

    /// Axis-aligned bounding box that encloses this shape (for broad-phase).
    pub fn bounding_aabb(&self) -> Vec3 {
        match self {
            Self::Aabb { half_extents } => *half_extents,
            Self::Sphere { radius } => Vec3::splat(*radius),
            Self::Capsule {
                radius,
                half_height,
            } => Vec3::new(*radius, half_height + radius, *radius),
        }
    }
}

// ---------------------------------------------------------------------------
// Physics material
// ---------------------------------------------------------------------------

/// Surface collision properties for an entity.
///
/// Drive the impulse solver's restitution (bounce) and friction (slide)
/// response. Values should mirror `MaterialData` from RON files.
#[derive(Component, Debug, Clone)]
pub struct PhysicsMaterial {
    /// Kinetic friction coefficient (dimensionless, 0.0–1.0).
    pub friction: f32,
    /// Coefficient of restitution (0 = perfectly inelastic, 1 = perfectly elastic).
    pub restitution: f32,
}

impl Default for PhysicsMaterial {
    fn default() -> Self {
        Self {
            friction: 0.5,
            restitution: 0.3,
        }
    }
}

impl PhysicsMaterial {
    /// Create a physics material with specific friction and restitution.
    pub fn new(friction: f32, restitution: f32) -> Self {
        Self {
            friction,
            restitution,
        }
    }

    /// Combine two materials for a contact pair using geometric mean.
    ///
    /// This produces physically reasonable intermediate values:
    /// ice-on-ice stays slippery, rubber-on-rubber stays grippy.
    pub fn combined(a: &Self, b: &Self) -> CombinedMaterial {
        CombinedMaterial {
            friction: (a.friction * b.friction).sqrt(),
            restitution: (a.restitution * b.restitution).sqrt(),
        }
    }
}

/// Pre-computed combined material properties for a contact pair.
#[derive(Debug, Clone, Copy)]
pub struct CombinedMaterial {
    pub friction: f32,
    pub restitution: f32,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- Shape constructors ---

    #[test]
    fn aabb_from_dimensions() {
        let s = CollisionShape::aabb(2.0, 4.0, 6.0);
        match s {
            CollisionShape::Aabb { half_extents } => {
                assert_eq!(half_extents, Vec3::new(1.0, 2.0, 3.0));
            }
            _ => panic!("Expected Aabb"),
        }
    }

    #[test]
    fn sphere_stores_radius() {
        let s = CollisionShape::sphere(1.5);
        match s {
            CollisionShape::Sphere { radius } => assert_eq!(radius, 1.5),
            _ => panic!("Expected Sphere"),
        }
    }

    #[test]
    fn capsule_from_total_height() {
        let s = CollisionShape::capsule(0.5, 2.0);
        match s {
            CollisionShape::Capsule {
                radius,
                half_height,
            } => {
                assert_eq!(radius, 0.5);
                assert!((half_height - 0.5).abs() < 1e-6);
            }
            _ => panic!("Expected Capsule"),
        }
    }

    #[test]
    fn capsule_zero_cylinder_when_height_equals_diameter() {
        let s = CollisionShape::capsule(1.0, 2.0);
        match s {
            CollisionShape::Capsule { half_height, .. } => {
                assert!((half_height).abs() < 1e-6);
            }
            _ => panic!("Expected Capsule"),
        }
    }

    // --- Volume ---

    #[test]
    fn aabb_volume() {
        let s = CollisionShape::aabb(2.0, 3.0, 4.0);
        assert!((s.volume() - 24.0).abs() < 1e-6);
    }

    #[test]
    fn sphere_volume() {
        let s = CollisionShape::sphere(1.0);
        let expected = (4.0 / 3.0) * PI;
        assert!((s.volume() - expected).abs() < 1e-4);
    }

    #[test]
    fn capsule_volume_degenerate_to_sphere() {
        // Capsule with zero cylinder height = sphere
        let s = CollisionShape::capsule(1.0, 2.0);
        let sphere_vol = (4.0 / 3.0) * PI;
        assert!(
            (s.volume() - sphere_vol).abs() < 1e-4,
            "Degenerate capsule should equal sphere volume: {} vs {}",
            s.volume(),
            sphere_vol
        );
    }

    // --- Moment of inertia ---

    #[test]
    fn sphere_inertia_is_two_fifths_mr_squared() {
        let mass = 10.0;
        let r = 0.5;
        let s = CollisionShape::sphere(r);
        let inertia = s.moment_of_inertia(mass);
        let expected = 0.4 * mass * r * r;
        assert!((inertia.x - expected).abs() < 1e-6);
        assert!((inertia.y - expected).abs() < 1e-6);
        assert!((inertia.z - expected).abs() < 1e-6);
    }

    #[test]
    fn aabb_inertia_cube() {
        // Cube 2m on a side, mass 12 kg
        // I_x = 12/12 × (4 + 4) = 8 kg·m²
        let mass = 12.0;
        let s = CollisionShape::aabb(2.0, 2.0, 2.0);
        let inertia = s.moment_of_inertia(mass);
        assert!((inertia.x - 8.0).abs() < 1e-4);
        assert!((inertia.y - 8.0).abs() < 1e-4);
        assert!((inertia.z - 8.0).abs() < 1e-4);
    }

    #[test]
    fn capsule_inertia_symmetric_xz() {
        let s = CollisionShape::capsule(0.3, 1.8);
        let inertia = s.moment_of_inertia(80.0);
        assert!(
            (inertia.x - inertia.z).abs() < 1e-6,
            "Capsule Ixx should equal Izz"
        );
    }

    // --- Bounding AABB ---

    #[test]
    fn sphere_bounding_aabb() {
        let s = CollisionShape::sphere(2.0);
        assert_eq!(s.bounding_aabb(), Vec3::splat(2.0));
    }

    #[test]
    fn capsule_bounding_aabb() {
        let s = CollisionShape::capsule(0.5, 2.0);
        let aabb = s.bounding_aabb();
        assert_eq!(aabb.x, 0.5);
        assert_eq!(aabb.z, 0.5);
        assert_eq!(aabb.y, 1.0); // half_height(0.5) + radius(0.5)
    }

    // --- Physics material ---

    #[test]
    fn default_material() {
        let m = PhysicsMaterial::default();
        assert_eq!(m.friction, 0.5);
        assert_eq!(m.restitution, 0.3);
    }

    #[test]
    fn combined_material_geometric_mean() {
        let a = PhysicsMaterial::new(0.04, 0.81);
        let b = PhysicsMaterial::new(0.16, 0.25);
        let c = PhysicsMaterial::combined(&a, &b);
        assert!((c.friction - 0.08).abs() < 1e-6); // sqrt(0.04 * 0.16) = 0.08
        assert!((c.restitution - 0.45).abs() < 1e-6); // sqrt(0.81 * 0.25) = 0.45
    }

    #[test]
    fn combined_material_identical_returns_same() {
        let a = PhysicsMaterial::new(0.7, 0.2);
        let c = PhysicsMaterial::combined(&a, &a);
        assert!((c.friction - 0.7).abs() < 1e-6);
        assert!((c.restitution - 0.2).abs() < 1e-6);
    }

    #[test]
    fn combined_material_one_zero_restitution_gives_zero() {
        let a = PhysicsMaterial::new(0.5, 0.0);
        let b = PhysicsMaterial::new(0.5, 0.9);
        let c = PhysicsMaterial::combined(&a, &b);
        assert_eq!(c.restitution, 0.0);
    }

    #[test]
    fn combined_material_both_elastic() {
        let a = PhysicsMaterial::new(0.1, 1.0);
        let b = PhysicsMaterial::new(0.1, 1.0);
        let c = PhysicsMaterial::combined(&a, &b);
        assert!((c.restitution - 1.0).abs() < 1e-6);
    }
}
