// State transitions: phase changes driven by temperature vs material thresholds.
//
// Each tick, voxels are checked against their material's melting/boiling points.
// When a threshold is crossed, the voxel's material transforms:
//   - Solid heated above melting_point → melted_into (e.g. stone → lava)
//   - Liquid heated above boiling_point → boiled_into (e.g. water → steam)
//   - Liquid cooled below melting_point → frozen_into (e.g. water → ice)
//   - Gas cooled below boiling_point → condensed_into (e.g. steam → water)
//
// Transition targets are defined in the material RON files, making the system
// fully data-driven. The MaterialRegistry maps MaterialId → MaterialData.

#![allow(dead_code)]

use crate::data::MaterialRegistry;
use crate::world::voxel::{MaterialId, Voxel};

// MaterialRegistry is now defined in crate::data and re-exported from there.

/// Result of checking a single voxel for phase transitions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TransitionResult {
    /// No change needed.
    None,
    /// Material should change to the given ID.
    TransformTo(MaterialId),
}

/// Check whether a voxel should undergo a phase transition based on its temperature.
pub fn check_transition(voxel: &Voxel, registry: &MaterialRegistry) -> TransitionResult {
    let Some(data) = registry.get(voxel.material) else {
        return TransitionResult::None;
    };

    let temp = voxel.temperature;

    match data.default_phase {
        crate::data::Phase::Solid => {
            // Solid → liquid when above melting point
            if let (Some(mp), Some(target_name)) = (data.melting_point, &data.melted_into) {
                if temp > mp {
                    if let Some(target) = registry.resolve_name(target_name) {
                        return TransitionResult::TransformTo(target);
                    }
                }
            }
        }
        crate::data::Phase::Liquid => {
            // Liquid → gas when above boiling point
            if let (Some(bp), Some(target_name)) = (data.boiling_point, &data.boiled_into) {
                if temp > bp {
                    if let Some(target) = registry.resolve_name(target_name) {
                        return TransitionResult::TransformTo(target);
                    }
                }
            }
            // Liquid → solid when below melting point (freezing)
            if let (Some(mp), Some(target_name)) = (data.melting_point, &data.frozen_into) {
                if temp < mp {
                    if let Some(target) = registry.resolve_name(target_name) {
                        return TransitionResult::TransformTo(target);
                    }
                }
            }
        }
        crate::data::Phase::Gas => {
            // Gas → liquid when below boiling point (condensation)
            if let (Some(bp), Some(target_name)) = (data.boiling_point, &data.condensed_into) {
                if temp < bp {
                    if let Some(target) = registry.resolve_name(target_name) {
                        return TransitionResult::TransformTo(target);
                    }
                }
            }
        }
    }

    TransitionResult::None
}

/// Apply state transitions to a flat voxel array.
/// Returns the number of voxels that changed.
pub fn apply_transitions(voxels: &mut [Voxel], registry: &MaterialRegistry) -> usize {
    let mut changed = 0;
    for voxel in voxels.iter_mut() {
        if voxel.is_air() {
            continue;
        }
        match check_transition(voxel, registry) {
            TransitionResult::None => {}
            TransitionResult::TransformTo(new_mat) => {
                voxel.material = new_mat;
                changed += 1;
            }
        }
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{MaterialData, Phase};

    fn test_registry() -> MaterialRegistry {
        let mut reg = MaterialRegistry::new();
        reg.insert(MaterialData {
            id: 0,
            name: "Air".into(),
            default_phase: Phase::Gas,
            density: 1.225,
            melting_point: None,
            boiling_point: None,
            ignition_point: None,
            hardness: 0.0,
            color: [0.8, 0.9, 1.0],
            transparent: true,
            melted_into: None,
            boiled_into: None,
            frozen_into: None,
            condensed_into: None,
        });
        reg.insert(MaterialData {
            id: 1,
            name: "Stone".into(),
            default_phase: Phase::Solid,
            density: 2700.0,
            melting_point: Some(1473.0),
            boiling_point: Some(2773.0),
            ignition_point: None,
            hardness: 0.9,
            color: [0.5, 0.5, 0.5],
            transparent: false,
            melted_into: Some("Lava".into()),
            boiled_into: None,
            frozen_into: None,
            condensed_into: None,
        });
        reg.insert(MaterialData {
            id: 3,
            name: "Water".into(),
            default_phase: Phase::Liquid,
            density: 1000.0,
            melting_point: Some(273.15),
            boiling_point: Some(373.15),
            ignition_point: None,
            hardness: 0.0,
            color: [0.2, 0.4, 0.8],
            transparent: true,
            melted_into: None,
            boiled_into: Some("Steam".into()),
            frozen_into: Some("Ice".into()),
            condensed_into: None,
        });
        reg.insert(MaterialData {
            id: 8,
            name: "Ice".into(),
            default_phase: Phase::Solid,
            density: 917.0,
            melting_point: Some(273.15),
            boiling_point: None,
            ignition_point: None,
            hardness: 0.2,
            color: [0.7, 0.85, 1.0],
            transparent: true,
            melted_into: Some("Water".into()),
            boiled_into: None,
            frozen_into: None,
            condensed_into: None,
        });
        reg.insert(MaterialData {
            id: 9,
            name: "Steam".into(),
            default_phase: Phase::Gas,
            density: 0.6,
            melting_point: None,
            boiling_point: Some(373.15),
            ignition_point: None,
            hardness: 0.0,
            color: [0.9, 0.9, 0.95],
            transparent: true,
            melted_into: None,
            boiled_into: None,
            frozen_into: None,
            condensed_into: Some("Water".into()),
        });
        reg.insert(MaterialData {
            id: 10,
            name: "Lava".into(),
            default_phase: Phase::Liquid,
            density: 2700.0,
            melting_point: Some(1473.0),
            boiling_point: Some(2773.0),
            ignition_point: None,
            hardness: 0.0,
            color: [1.0, 0.3, 0.0],
            transparent: false,
            melted_into: None,
            boiled_into: Some("Air".into()),
            frozen_into: Some("Stone".into()),
            condensed_into: None,
        });
        reg
    }

    #[test]
    fn registry_insert_and_lookup() {
        let reg = test_registry();
        assert!(reg.len() >= 6);
        let stone = reg.get(MaterialId::STONE).unwrap();
        assert_eq!(stone.name, "Stone");
        assert!(reg.get(MaterialId(999)).is_none());
    }

    #[test]
    fn stone_melts_to_lava() {
        let reg = test_registry();
        let mut voxel = Voxel::new(MaterialId::STONE);
        voxel.temperature = 1500.0; // above melting point 1473
        assert_eq!(
            check_transition(&voxel, &reg),
            TransitionResult::TransformTo(MaterialId::LAVA)
        );
    }

    #[test]
    fn stone_stays_solid_below_melting() {
        let reg = test_registry();
        let mut voxel = Voxel::new(MaterialId::STONE);
        voxel.temperature = 1000.0; // below melting point
        assert_eq!(check_transition(&voxel, &reg), TransitionResult::None);
    }

    #[test]
    fn water_freezes_to_ice() {
        let reg = test_registry();
        let mut voxel = Voxel::new(MaterialId::WATER);
        voxel.temperature = 250.0; // below 273.15
        assert_eq!(
            check_transition(&voxel, &reg),
            TransitionResult::TransformTo(MaterialId::ICE)
        );
    }

    #[test]
    fn water_boils_to_steam() {
        let reg = test_registry();
        let mut voxel = Voxel::new(MaterialId::WATER);
        voxel.temperature = 400.0; // above 373.15
        assert_eq!(
            check_transition(&voxel, &reg),
            TransitionResult::TransformTo(MaterialId::STEAM)
        );
    }

    #[test]
    fn water_at_room_temp_stays_water() {
        let reg = test_registry();
        let voxel = Voxel::new(MaterialId::WATER);
        // Default 293K, between 273.15 and 373.15
        assert_eq!(check_transition(&voxel, &reg), TransitionResult::None);
    }

    #[test]
    fn ice_melts_to_water() {
        let reg = test_registry();
        let mut voxel = Voxel::new(MaterialId::ICE);
        voxel.temperature = 280.0; // above 273.15
        assert_eq!(
            check_transition(&voxel, &reg),
            TransitionResult::TransformTo(MaterialId::WATER)
        );
    }

    #[test]
    fn steam_condenses_to_water() {
        let reg = test_registry();
        let mut voxel = Voxel::new(MaterialId::STEAM);
        voxel.temperature = 350.0; // below 373.15
        assert_eq!(
            check_transition(&voxel, &reg),
            TransitionResult::TransformTo(MaterialId::WATER)
        );
    }

    #[test]
    fn lava_freezes_to_stone() {
        let reg = test_registry();
        let mut voxel = Voxel::new(MaterialId::LAVA);
        voxel.temperature = 1000.0; // below 1473 melting point
        assert_eq!(
            check_transition(&voxel, &reg),
            TransitionResult::TransformTo(MaterialId::STONE)
        );
    }

    #[test]
    fn air_has_no_transitions() {
        let reg = test_registry();
        let mut voxel = Voxel::new(MaterialId::AIR);
        voxel.temperature = 5000.0;
        // air is skipped by apply_transitions
        assert_eq!(check_transition(&voxel, &reg), TransitionResult::None);
    }

    #[test]
    fn apply_transitions_transforms_voxels() {
        let reg = test_registry();
        let mut voxels = vec![
            Voxel::new(MaterialId::STONE),
            Voxel::new(MaterialId::WATER),
            Voxel::new(MaterialId::AIR),
        ];
        voxels[0].temperature = 1500.0; // stone → lava
        voxels[1].temperature = 250.0; // water → ice

        let changed = apply_transitions(&mut voxels, &reg);
        assert_eq!(changed, 2);
        assert_eq!(voxels[0].material, MaterialId::LAVA);
        assert_eq!(voxels[1].material, MaterialId::ICE);
        assert_eq!(voxels[2].material, MaterialId::AIR); // unchanged
    }

    #[test]
    fn full_water_cycle() {
        let reg = test_registry();
        let mut voxel = Voxel::new(MaterialId::WATER);

        // Freeze
        voxel.temperature = 200.0;
        if let TransitionResult::TransformTo(m) = check_transition(&voxel, &reg) {
            voxel.material = m;
        }
        assert_eq!(voxel.material, MaterialId::ICE);

        // Melt
        voxel.temperature = 300.0;
        if let TransitionResult::TransformTo(m) = check_transition(&voxel, &reg) {
            voxel.material = m;
        }
        assert_eq!(voxel.material, MaterialId::WATER);

        // Boil
        voxel.temperature = 400.0;
        if let TransitionResult::TransformTo(m) = check_transition(&voxel, &reg) {
            voxel.material = m;
        }
        assert_eq!(voxel.material, MaterialId::STEAM);

        // Condense
        voxel.temperature = 350.0;
        if let TransitionResult::TransformTo(m) = check_transition(&voxel, &reg) {
            voxel.material = m;
        }
        assert_eq!(voxel.material, MaterialId::WATER);
    }
}
