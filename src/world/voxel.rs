// Voxel types and state definitions.
//
// Each voxel in the world grid stores a compact representation of what material
// occupies that space and its current physical state. These types are designed
// for cache-friendly storage in chunk arrays.

#![allow(dead_code)]

/// Identifies a material type. Index 0 is always air (empty).
/// Material properties are looked up from RON-loaded `MaterialData` via this ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct MaterialId(pub u16);

impl MaterialId {
    pub const AIR: Self = Self(0);
    pub const STONE: Self = Self(1);
    pub const DIRT: Self = Self(2);
    pub const WATER: Self = Self(3);

    pub fn is_air(self) -> bool {
        self.0 == 0
    }
}

/// Physical state of a single voxel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Voxel {
    pub material: MaterialId,
    /// Temperature in Kelvin (ambient ~293 K).
    pub temperature: f32,
    /// Ambient pressure in atmospheres (sea level = 1.0).
    pub pressure: f32,
    /// Structural damage [0.0 = intact, 1.0 = destroyed].
    pub damage: f32,
}

impl Default for Voxel {
    fn default() -> Self {
        Self {
            material: MaterialId::AIR,
            temperature: 293.0,
            pressure: 1.0,
            damage: 0.0,
        }
    }
}

impl Voxel {
    pub fn new(material: MaterialId) -> Self {
        Self {
            material,
            ..Default::default()
        }
    }

    pub fn is_air(&self) -> bool {
        self.material.is_air()
    }

    pub fn is_solid(&self) -> bool {
        !self.is_air()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_voxel_is_air() {
        let v = Voxel::default();
        assert!(v.is_air());
        assert!(!v.is_solid());
        assert_eq!(v.material, MaterialId::AIR);
    }

    #[test]
    fn voxel_new_sets_material_with_defaults() {
        let v = Voxel::new(MaterialId::STONE);
        assert_eq!(v.material, MaterialId::STONE);
        assert!(!v.is_air());
        assert!(v.is_solid());
        assert_eq!(v.temperature, 293.0);
        assert_eq!(v.pressure, 1.0);
        assert_eq!(v.damage, 0.0);
    }

    #[test]
    fn material_id_air_is_zero() {
        assert_eq!(MaterialId::AIR.0, 0);
        assert!(MaterialId::AIR.is_air());
        assert!(!MaterialId::STONE.is_air());
    }

    #[test]
    fn material_id_constants_are_distinct() {
        let ids = [
            MaterialId::AIR,
            MaterialId::STONE,
            MaterialId::DIRT,
            MaterialId::WATER,
        ];
        for (i, a) in ids.iter().enumerate() {
            for (j, b) in ids.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b);
                }
            }
        }
    }

    #[test]
    fn voxel_damage_range() {
        let mut v = Voxel::new(MaterialId::STONE);
        v.damage = 0.5;
        assert_eq!(v.damage, 0.5);
        assert!(v.is_solid());
    }

    #[test]
    fn voxel_temperature_and_pressure() {
        let mut v = Voxel::new(MaterialId::WATER);
        v.temperature = 373.15; // boiling point
        v.pressure = 2.0;
        assert_eq!(v.temperature, 373.15);
        assert_eq!(v.pressure, 2.0);
    }

    #[test]
    fn voxel_size_is_compact() {
        // Voxel should be ≤16 bytes for cache efficiency in chunk arrays
        assert!(
            std::mem::size_of::<Voxel>() <= 16,
            "Voxel is {} bytes, expected ≤16",
            std::mem::size_of::<Voxel>()
        );
    }
}
