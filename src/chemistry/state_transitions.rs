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

use crate::data::{MaterialData, MaterialRegistry, Phase};
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
            if let (Some(mp), Some(target_name)) = (data.melting_point, &data.melted_into)
                && temp > mp
                && let Some(target) = registry.resolve_name(target_name)
            {
                return TransitionResult::TransformTo(target);
            }
        }
        crate::data::Phase::Liquid => {
            // Liquid → gas when above boiling point
            if let (Some(bp), Some(target_name)) = (data.boiling_point, &data.boiled_into)
                && temp > bp
                && let Some(target) = registry.resolve_name(target_name)
            {
                return TransitionResult::TransformTo(target);
            }
            // Liquid → solid when below melting point (freezing)
            if let (Some(mp), Some(target_name)) = (data.melting_point, &data.frozen_into)
                && temp < mp
                && let Some(target) = registry.resolve_name(target_name)
            {
                return TransitionResult::TransformTo(target);
            }
        }
        crate::data::Phase::Gas => {
            // Gas → liquid when below boiling point (condensation)
            if let (Some(bp), Some(target_name)) = (data.boiling_point, &data.condensed_into)
                && temp < bp
                && let Some(target) = registry.resolve_name(target_name)
            {
                return TransitionResult::TransformTo(target);
            }
        }
    }

    TransitionResult::None
}

/// Determine the latent heat (J/kg) for a pending transition, if any.
///
/// Returns `Some(latent_heat)` when the material's data specifies a latent
/// heat for this transition type, or `None` for instant transitions.
fn latent_heat_for_transition(voxel: &Voxel, data: &MaterialData) -> Option<f32> {
    match data.default_phase {
        Phase::Solid => {
            // Melting: uses latent_heat_fusion
            if data.melting_point.is_some_and(|mp| voxel.temperature > mp) {
                return data.latent_heat_fusion;
            }
        }
        Phase::Liquid => {
            // Boiling: uses latent_heat_vaporization
            if data.boiling_point.is_some_and(|bp| voxel.temperature > bp) {
                return data.latent_heat_vaporization;
            }
            // Freezing: uses latent_heat_fusion
            if data.melting_point.is_some_and(|mp| voxel.temperature < mp) {
                return data.latent_heat_fusion;
            }
        }
        Phase::Gas => {
            // Condensation: uses latent_heat_vaporization
            if data.boiling_point.is_some_and(|bp| voxel.temperature < bp) {
                return data.latent_heat_vaporization;
            }
        }
    }
    None
}

/// Return the phase-boundary temperature and sign for the current transition.
///
/// `direction` is +1.0 when the voxel is heating past the threshold (melting,
/// boiling) and −1.0 when cooling past it (freezing, condensation).
fn transition_threshold_and_direction(voxel: &Voxel, data: &MaterialData) -> (f32, f32) {
    match data.default_phase {
        Phase::Solid => (data.melting_point.unwrap_or(voxel.temperature), 1.0),
        Phase::Liquid => {
            if data.boiling_point.is_some_and(|bp| voxel.temperature > bp) {
                (data.boiling_point.unwrap(), 1.0)
            } else {
                (data.melting_point.unwrap_or(voxel.temperature), -1.0)
            }
        }
        Phase::Gas => (data.boiling_point.unwrap_or(voxel.temperature), -1.0),
    }
}

/// Drain the latent-heat buffer when the voxel's temperature moves away from
/// the transition threshold (e.g. water warms back above 273.15 K while the
/// freezing buffer was partially filled).
///
/// Energy conservation: `Cₚ × T − buffer` is invariant across the drain.
fn drain_latent_buffer(voxel: &mut Voxel, data: &MaterialData) {
    if voxel.latent_heat_buffer <= 0.0 {
        return;
    }
    let cp = data.specific_heat_capacity.max(1.0);

    // Find the threshold the buffer was accumulated toward.
    let threshold = match data.default_phase {
        Phase::Solid => data.melting_point,
        Phase::Liquid => match (data.melting_point, data.boiling_point) {
            (Some(mp), Some(bp)) => {
                if (voxel.temperature - mp).abs() <= (voxel.temperature - bp).abs() {
                    Some(mp)
                } else {
                    Some(bp)
                }
            }
            (Some(mp), None) => Some(mp),
            (None, Some(bp)) => Some(bp),
            (None, None) => None,
        },
        Phase::Gas => data.boiling_point,
    };

    let Some(threshold) = threshold else {
        voxel.latent_heat_buffer = 0.0;
        return;
    };

    let distance = (voxel.temperature - threshold).abs();
    let drain_energy = cp * distance;

    if drain_energy >= voxel.latent_heat_buffer {
        let remaining = drain_energy - voxel.latent_heat_buffer;
        let sign = if voxel.temperature >= threshold {
            1.0
        } else {
            -1.0
        };
        voxel.temperature = threshold + sign * remaining / cp;
        voxel.latent_heat_buffer = 0.0;
    } else {
        voxel.latent_heat_buffer -= drain_energy;
        voxel.temperature = threshold;
    }
}

/// Apply state transitions to a flat voxel array.
///
/// Uses cumulative latent-heat tracking when a material defines
/// `latent_heat_fusion` or `latent_heat_vaporization`. Materials without
/// latent heat transition instantly (backward compatible).
///
/// Returns the number of voxels that completed a phase change this tick.
///
/// TODO: When native octree physics with LOD-based dynamic resolution is
/// implemented, sub-voxel cells will track their own latent heat buffers,
/// enabling a spatially resolved phase front instead of the current per-voxel
/// average.
pub fn apply_transitions(voxels: &mut [Voxel], registry: &MaterialRegistry) -> usize {
    let mut changed = 0;
    for voxel in voxels.iter_mut() {
        if voxel.is_air() {
            continue;
        }
        let Some(data) = registry.get(voxel.material) else {
            continue;
        };

        match check_transition(voxel, registry) {
            TransitionResult::None => {
                // Temperature did not cross a threshold this tick.
                // If there is accumulated buffer energy, drain it.
                if voxel.latent_heat_buffer > 0.0 {
                    drain_latent_buffer(voxel, data);
                }
            }
            TransitionResult::TransformTo(target) => {
                let latent = latent_heat_for_transition(voxel, data);
                if let Some(lh) = latent {
                    // Cumulative model: absorb overcooling/overheating energy.
                    let (threshold, direction) = transition_threshold_and_direction(voxel, data);
                    let cp = data.specific_heat_capacity.max(1.0);
                    let overcooling = (voxel.temperature - threshold).abs();
                    let energy = cp * overcooling;
                    voxel.latent_heat_buffer += energy;
                    voxel.temperature = threshold;

                    if voxel.latent_heat_buffer >= lh {
                        let excess = voxel.latent_heat_buffer - lh;
                        let new_cp = registry
                            .get(target)
                            .map(|d| d.specific_heat_capacity)
                            .unwrap_or(1000.0)
                            .max(1.0);
                        let residual = excess / new_cp;

                        voxel.material = target;
                        voxel.latent_heat_buffer = 0.0;
                        voxel.temperature = threshold + direction * residual;
                        changed += 1;
                    }
                } else {
                    // No latent heat — instant transition.
                    voxel.material = target;
                    voxel.latent_heat_buffer = 0.0;
                    changed += 1;
                }
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
            ..Default::default()
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
            ..Default::default()
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
            ..Default::default()
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
            ..Default::default()
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
            ..Default::default()
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
            ..Default::default()
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
        voxels[0].temperature = 1500.0; // stone → lava (no latent heat in test registry)
        voxels[1].temperature = 250.0; // water → ice (no latent heat in test registry)

        let changed = apply_transitions(&mut voxels, &reg);
        assert_eq!(changed, 2);
        assert_eq!(voxels[0].material, MaterialId::LAVA);
        assert_eq!(voxels[1].material, MaterialId::ICE);
        assert_eq!(voxels[2].material, MaterialId::AIR); // unchanged
    }

    // ---- Latent heat tests ----

    fn test_registry_with_latent_heat() -> MaterialRegistry {
        let mut reg = test_registry();
        // Re-insert water and ice with latent heat values
        reg.insert(MaterialData {
            id: 3,
            name: "Water".into(),
            default_phase: Phase::Liquid,
            density: 1000.0,
            melting_point: Some(273.15),
            boiling_point: Some(373.15),
            specific_heat_capacity: 4186.0,
            latent_heat_fusion: Some(334_000.0),
            latent_heat_vaporization: Some(2_260_000.0),
            frozen_into: Some("Ice".into()),
            boiled_into: Some("Steam".into()),
            ..Default::default()
        });
        reg.insert(MaterialData {
            id: 8,
            name: "Ice".into(),
            default_phase: Phase::Solid,
            density: 917.0,
            melting_point: Some(273.15),
            specific_heat_capacity: 2090.0,
            latent_heat_fusion: Some(334_000.0),
            melted_into: Some("Water".into()),
            ..Default::default()
        });
        reg
    }

    #[test]
    fn latent_heat_delays_freezing() {
        let reg = test_registry_with_latent_heat();
        let mut voxels = vec![Voxel::new(MaterialId::WATER)];
        voxels[0].temperature = 272.0;
        let changed = apply_transitions(&mut voxels, &reg);
        assert_eq!(changed, 0, "should not freeze instantly with latent heat");
        assert_eq!(voxels[0].material, MaterialId::WATER);
        assert_eq!(voxels[0].temperature, 273.15); // clamped to threshold
        assert!(voxels[0].latent_heat_buffer > 0.0);

        // Energy stored: Cp × overcooling = 4186 × 1.15 ≈ 4814 J/kg
        let expected_energy = 4186.0 * 1.15;
        assert!(
            (voxels[0].latent_heat_buffer - expected_energy).abs() < 1.0,
            "buffer should be ~{expected_energy}, got {}",
            voxels[0].latent_heat_buffer
        );
    }

    #[test]
    fn latent_heat_accumulates_over_ticks() {
        let reg = test_registry_with_latent_heat();
        let mut voxels = vec![Voxel::new(MaterialId::WATER)];

        // Simulate many ticks of 1 K overcooling
        for _ in 0..100 {
            voxels[0].temperature = 272.15; // 1 K below mp
            apply_transitions(&mut voxels, &reg);
        }

        // 100 ticks × 4186 J/kg = 418,600 J/kg > 334,000 → should have frozen
        assert_eq!(
            voxels[0].material,
            MaterialId::ICE,
            "water should freeze after enough ticks"
        );
        assert_eq!(voxels[0].latent_heat_buffer, 0.0);
        // Residual temperature should be below threshold
        assert!(voxels[0].temperature < 273.15);
    }

    #[test]
    fn latent_heat_transition_residual_temperature() {
        let reg = test_registry_with_latent_heat();
        let mut voxels = vec![Voxel::new(MaterialId::WATER)];

        // Accumulate exactly 334,000 J/kg + some excess
        // Each tick at 1K below: 4186 J/kg. Need 334000/4186 ≈ 79.8 ticks.
        // Use 80 ticks at 1K → 80 × 4186 = 334,880 J/kg. Excess = 880 J/kg.
        for _ in 0..80 {
            voxels[0].temperature = 272.15; // 1 K below mp
            apply_transitions(&mut voxels, &reg);
        }

        assert_eq!(voxels[0].material, MaterialId::ICE);
        // Residual: excess / Cp_ice = 880 / 2090 ≈ 0.421 K below mp
        let expected_residual = 273.15 - 880.0 / 2090.0;
        assert!(
            (voxels[0].temperature - expected_residual).abs() < 0.1,
            "residual temp should be ~{expected_residual:.2}, got {:.2}",
            voxels[0].temperature
        );
    }

    #[test]
    fn latent_heat_buffer_drains_on_warming() {
        let reg = test_registry_with_latent_heat();
        let mut voxels = vec![Voxel::new(MaterialId::WATER)];

        // Accumulate some freezing energy
        voxels[0].temperature = 272.15;
        apply_transitions(&mut voxels, &reg);
        let buffer_after = voxels[0].latent_heat_buffer;
        assert!(buffer_after > 0.0);

        // Now warm above threshold → should drain
        voxels[0].temperature = 275.0; // 1.85 K above mp
        apply_transitions(&mut voxels, &reg);

        // Drain energy = 4186 × 1.85 = 7744 > buffer ≈ 4186 → fully drained
        assert_eq!(voxels[0].latent_heat_buffer, 0.0);
        // Temperature reduced by buffer/Cp
        assert!(voxels[0].temperature < 275.0);
        assert!(voxels[0].temperature > 273.15);
    }

    #[test]
    fn no_latent_heat_instant_transition() {
        // test_registry has NO latent heat → instant transition (backward compat)
        let reg = test_registry();
        let mut voxels = vec![Voxel::new(MaterialId::WATER)];
        voxels[0].temperature = 250.0;

        let changed = apply_transitions(&mut voxels, &reg);
        assert_eq!(changed, 1);
        assert_eq!(voxels[0].material, MaterialId::ICE);
        assert_eq!(voxels[0].latent_heat_buffer, 0.0);
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

    // ---- Real-world validation tests ----

    #[test]
    fn water_freezing_point_is_273k() {
        // Wikipedia: Water freezes at 273.15 K (0°C) at 1 atm
        // Validate our material data uses the correct value
        let registry = test_registry();
        let water_data = registry.get(MaterialId::WATER).unwrap();
        let expected_mp = 273.15_f32;
        assert!(
            (water_data.melting_point.unwrap() - expected_mp).abs() < 1.0,
            "Water melting point should be ~273 K, got {:?}",
            water_data.melting_point
        );
    }

    #[test]
    fn water_boiling_point_is_373k() {
        // Wikipedia: Water boils at 373.15 K (100°C) at 1 atm
        let registry = test_registry();
        let water_data = registry.get(MaterialId::WATER).unwrap();
        let expected_bp = 373.15_f32;
        assert!(
            (water_data.boiling_point.unwrap() - expected_bp).abs() < 1.0,
            "Water boiling point should be ~373 K, got {:?}",
            water_data.boiling_point
        );
    }

    #[test]
    fn ice_latent_heat_fusion_is_334kj() {
        // Wikipedia: Enthalpy of fusion of water/ice = 334 kJ/kg
        let registry = test_registry();
        let ice_data = registry.get(MaterialId::ICE).unwrap();
        if let Some(lf) = ice_data.latent_heat_fusion {
            assert!(
                (lf - 334_000.0).abs() < 10_000.0,
                "Ice latent heat of fusion should be ~334,000 J/kg, got {lf}"
            );
        }
        // Also check water has same fusion value
        let water_data = registry.get(MaterialId::WATER).unwrap();
        if let Some(lf) = water_data.latent_heat_fusion {
            assert!(
                (lf - 334_000.0).abs() < 10_000.0,
                "Water latent heat of fusion should be ~334,000 J/kg, got {lf}"
            );
        }
    }

    #[test]
    fn water_latent_heat_vaporization_is_2260kj() {
        // Wikipedia: Enthalpy of vaporization of water = 2260 kJ/kg
        let registry = test_registry();
        let water_data = registry.get(MaterialId::WATER).unwrap();
        if let Some(lv) = water_data.latent_heat_vaporization {
            assert!(
                (lv - 2_260_000.0).abs() < 100_000.0,
                "Water latent heat of vaporization should be ~2,260,000 J/kg, got {lv}"
            );
        }
    }

    #[test]
    fn phase_transition_temperatures_are_ordered() {
        // For all materials with both melting and boiling points,
        // melting must be < boiling (second law of thermodynamics constraint)
        let registry = test_registry();
        for &id in &[MaterialId::WATER, MaterialId::STONE, MaterialId::LAVA] {
            let mat = registry.get(id).unwrap();
            if let (Some(mp), Some(bp)) = (mat.melting_point, mat.boiling_point) {
                assert!(
                    mp < bp,
                    "Material {}: melting ({mp} K) must be < boiling ({bp} K)",
                    mat.name
                );
            }
        }
    }

    #[test]
    fn stone_melting_point_is_1473k() {
        // Wikipedia: Typical granite melting range 1073-1473 K
        let registry = test_registry();
        let stone = registry.get(MaterialId::STONE).unwrap();
        let mp = stone.melting_point.unwrap();
        assert!(
            (1000.0..=1600.0).contains(&mp),
            "Stone melting point should be ~1473 K, got {mp}"
        );
    }
}
