// Voxel types and state definitions.
//
// Each voxel in the world grid stores a compact representation of what material
// occupies that space and its current physical state. These types are designed
// for cache-friendly storage in chunk arrays.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

/// Identifies a material type. Index 0 is always air (empty).
/// Material properties are looked up from RON-loaded `MaterialData` via this ID.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct MaterialId(pub u16);

impl MaterialId {
    pub const AIR: Self = Self(0);
    pub const STONE: Self = Self(1);
    pub const DIRT: Self = Self(2);
    pub const WATER: Self = Self(3);
    pub const ICE: Self = Self(8);
    pub const STEAM: Self = Self(9);
    pub const LAVA: Self = Self(10);
    pub const ASH: Self = Self(11);

    pub fn is_air(self) -> bool {
        self.0 == 0
    }
}

/// Physical state of a single voxel.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub struct Voxel {
    pub material: MaterialId,
    /// Temperature in Kelvin (ambient ~288.15 K = 15 °C).
    pub temperature: f32,
    /// Ambient pressure in Pascals (sea level = 101325 Pa).
    pub pressure: f32,
    /// Structural damage / mass fraction remaining [1.0 = intact, 0.0 = destroyed].
    pub damage: f32,
    /// Cumulative energy (J/kg) accumulated toward the next phase transition.
    ///
    /// When a voxel is at a phase boundary (e.g. water at 273.15 K), heat
    /// extracted each tick is stored here instead of changing temperature.
    /// The transition completes when this buffer reaches the material's latent
    /// heat. Materials without latent heat transition instantly (buffer stays 0).
    ///
    /// TODO: When native octree physics is implemented with LOD-based dynamic
    /// resolution, sub-voxel thermal gradients will make this buffer less
    /// critical — finer cells at interfaces will capture the phase front
    /// directly rather than averaging over 1 m³.
    pub latent_heat_buffer: f32,
}

impl Default for Voxel {
    fn default() -> Self {
        Self {
            material: MaterialId::AIR,
            temperature: 288.15,
            pressure: 101_325.0,
            damage: 0.0,
            latent_heat_buffer: 0.0,
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
        assert_eq!(v.temperature, 288.15);
        assert_eq!(v.pressure, 101_325.0);
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
        v.pressure = 200_000.0; // ~2 atm in Pascals
        assert_eq!(v.temperature, 373.15);
        assert_eq!(v.pressure, 200_000.0);
    }

    #[test]
    fn voxel_size_is_compact() {
        // Voxel should be ≤24 bytes for cache efficiency in chunk arrays.
        // Current layout: MaterialId(u16) + pad + 4×f32 = 20 bytes.
        assert!(
            std::mem::size_of::<Voxel>() <= 24,
            "Voxel is {} bytes, expected ≤24",
            std::mem::size_of::<Voxel>()
        );
    }
}
