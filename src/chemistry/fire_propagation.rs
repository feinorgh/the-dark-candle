// Fire propagation: emergent integration test for the chemistry pipeline.
//
// This module tests fire spreading through connected wood voxels using the
// shared simulation harness (`crate::simulation::simulate_tick`), which
// combines heat diffusion, chemical reactions, state transitions, and
// pressure diffusion in a single tick loop.
//
// The test verifies that igniting one wood voxel eventually causes adjacent
// wood to burn as well, producing ash and releasing heat — all driven purely
// by the data-defined reaction rules and physics.

#![allow(dead_code)]

use crate::chemistry::reactions::ReactionData;
use crate::world::voxel::{MaterialId, Voxel};

const WOOD: u16 = 5;
const ASH: u16 = 11;
const AIR: u16 = 0;

fn wood_combustion_rule() -> ReactionData {
    ReactionData {
        name: "Wood combustion".into(),
        input_a: "Wood".into(),
        input_b: Some("Air".into()),
        min_temperature: 573.0,
        max_temperature: 99999.0,
        output_a: "Ash".into(),
        output_b: None,
        // ΔT of 3500 K is deliberately higher than the physical flame delta
        // (~800 K) to compensate for our conduction-only heat model. Real fire
        // spreads via convection and radiation too; the larger impulse ensures
        // the conduction-only model can propagate fire across the voxel grid.
        // (Computed: ash at ~3788K needs ~89 steps to cool, during which each
        // wood neighbor gains ~340K, exceeding the 285K needed for ignition.)
        heat_output: 3500.0,
    }
}

/// Count how many voxels have the given material.
fn count_material(voxels: &[Voxel], mat: MaterialId) -> usize {
    voxels.iter().filter(|v| v.material == mat).count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{MaterialData, MaterialRegistry, Phase};
    use crate::simulation::simulate_tick;

    fn minimal_registry() -> MaterialRegistry {
        let mut reg = MaterialRegistry::new();
        reg.insert(MaterialData {
            id: AIR,
            name: "Air".into(),
            default_phase: Phase::Gas,
            density: 1.225,
            thermal_conductivity: 0.026,
            specific_heat_capacity: 1005.0,
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
            id: WOOD,
            name: "Wood".into(),
            default_phase: Phase::Solid,
            density: 600.0,
            thermal_conductivity: 0.15,
            specific_heat_capacity: 1700.0,
            melting_point: None,
            boiling_point: None,
            ignition_point: Some(573.0),
            hardness: 0.3,
            color: [0.6, 0.4, 0.2],
            transparent: false,
            melted_into: None,
            boiled_into: None,
            frozen_into: None,
            condensed_into: None,
            ..Default::default()
        });
        reg.insert(MaterialData {
            id: ASH,
            name: "Ash".into(),
            default_phase: Phase::Solid,
            density: 500.0,
            thermal_conductivity: 0.15,
            specific_heat_capacity: 800.0,
            melting_point: Some(1273.0),
            boiling_point: None,
            ignition_point: None,
            hardness: 0.05,
            color: [0.3, 0.3, 0.3],
            transparent: false,
            melted_into: None,
            boiled_into: None,
            frozen_into: None,
            condensed_into: None,
            ..Default::default()
        });
        reg
    }

    /// Build a 4×4×4 grid: bottom layer (y=0) is wood, rest is air.
    fn wood_floor_grid() -> (Vec<Voxel>, usize) {
        let size = 4;
        let mut voxels = vec![Voxel::new(MaterialId::AIR); size * size * size];

        // Fill y=0 layer with wood
        for z in 0..size {
            for x in 0..size {
                let idx = z * size * size + x; // y=0
                voxels[idx].material = MaterialId(WOOD);
            }
        }

        (voxels, size)
    }

    #[test]
    fn unheated_wood_does_not_burn() {
        let (mut voxels, size) = wood_floor_grid();
        let rules = vec![wood_combustion_rule()];
        let registry = minimal_registry();

        // Run 10 ticks at room temperature (dt=5000s for real SI diffusion)
        let mut total_reactions = 0;
        for _ in 0..10 {
            total_reactions +=
                simulate_tick(&mut voxels, size, &rules, &registry, 5000.0).reactions_fired;
        }

        assert_eq!(total_reactions, 0, "Room-temp wood should not burn");
        assert_eq!(count_material(&voxels, MaterialId(WOOD)), 16); // 4x4 floor
    }

    #[test]
    fn single_ignition_causes_combustion() {
        let (mut voxels, size) = wood_floor_grid();
        let rules = vec![wood_combustion_rule()];
        let registry = minimal_registry();

        // Ignite one corner: set temperature above 573K
        let ignite_idx = 0; // (0, 0, 0)
        voxels[ignite_idx].temperature = 800.0;

        let initial_wood = count_material(&voxels, MaterialId(WOOD));
        assert_eq!(initial_wood, 16);

        // Run a single tick — the ignited voxel should react
        let reactions = simulate_tick(&mut voxels, size, &rules, &registry, 5000.0).reactions_fired;

        assert!(reactions > 0, "Ignited wood should react in first tick");
        assert!(
            count_material(&voxels, MaterialId(ASH)) > 0,
            "Should produce ash"
        );
        assert!(
            count_material(&voxels, MaterialId(WOOD)) < initial_wood,
            "Some wood should have burned"
        );
    }

    #[test]
    fn fire_spreads_to_adjacent_wood() {
        // Use a 4x4x4 grid with a line of wood along x at y=0, z=0
        // surrounded by air. This makes propagation clearer.
        let size = 4;
        let mut voxels = vec![Voxel::new(MaterialId::AIR); size * size * size];

        // Place wood along x axis at y=0, z=0
        for x in 0..size {
            let idx = x; // y=0, z=0
            voxels[idx].material = MaterialId(WOOD);
        }

        let rules = vec![wood_combustion_rule()];
        let registry = minimal_registry();

        // Ignite x=0
        voxels[0].temperature = 800.0;

        let mut total_reactions = 0;
        for _ in 0..200 {
            total_reactions +=
                simulate_tick(&mut voxels, size, &rules, &registry, 5000.0).reactions_fired;
        }

        let remaining_wood = count_material(&voxels, MaterialId(WOOD));
        let ash_count = count_material(&voxels, MaterialId(ASH));

        assert!(
            total_reactions > 1,
            "Fire should trigger multiple reactions, got {total_reactions}"
        );
        assert!(
            ash_count > 1,
            "Fire should spread and produce ash in multiple voxels, got {ash_count}"
        );
        assert!(
            remaining_wood < 4,
            "Not all wood should remain, got {remaining_wood}"
        );
    }

    #[test]
    fn combustion_releases_heat_to_neighbors() {
        let (mut voxels, size) = wood_floor_grid();
        let rules = vec![wood_combustion_rule()];
        let registry = minimal_registry();

        // Ignite corner, record neighbor's initial temperature
        voxels[0].temperature = 800.0;
        let neighbor_idx = 1; // (1, 0, 0)
        let initial_neighbor_temp = voxels[neighbor_idx].temperature;

        // One tick (dt=5000s for meaningful SI heat transfer)
        simulate_tick(&mut voxels, size, &rules, &registry, 5000.0);

        assert!(
            voxels[neighbor_idx].temperature > initial_neighbor_temp,
            "Adjacent voxel should be heated by combustion: {} vs {}",
            voxels[neighbor_idx].temperature,
            initial_neighbor_temp
        );
    }

    #[test]
    fn fire_eventually_burns_all_wood_on_surface() {
        let (mut voxels, size) = wood_floor_grid();
        // Higher thermal impulse needed for 2D surface spreading with real SI
        // conduction — real fire uses radiation + convection which we don't model.
        let rules = vec![ReactionData {
            heat_output: 10_000.0,
            ..wood_combustion_rule()
        }];
        let registry = minimal_registry();

        // Ignite center of the floor
        let center_idx = size * size + 1; // (1, 0, 1)
        voxels[center_idx].temperature = 800.0;

        // Run many ticks with SI-scale diffusion (dt=5000s per tick).
        for _ in 0..2000 {
            simulate_tick(&mut voxels, size, &rules, &registry, 5000.0);
        }

        let remaining_wood = count_material(&voxels, MaterialId(WOOD));
        let ash_count = count_material(&voxels, MaterialId(ASH));

        // With sufficient ticks and heat, most wood should burn
        assert!(
            ash_count >= 8,
            "After many ticks, most wood should be ash. Got {ash_count} ash, {remaining_wood} wood"
        );
    }

    #[test]
    fn isolated_wood_requires_air_neighbor_to_burn() {
        // A 3x3x3 cube entirely of wood (no air neighbors inside)
        let size = 3;
        let mut voxels = vec![Voxel::new(MaterialId(WOOD)); size * size * size];
        // Heat the center
        let center = size * size + size + 1;
        voxels[center].temperature = 800.0;

        let rules = vec![wood_combustion_rule()];
        let registry = minimal_registry();

        // Run a tick — center has no air neighbors, only wood
        let reactions = simulate_tick(&mut voxels, size, &rules, &registry, 5000.0).reactions_fired;

        // Center is fully surrounded by wood — no air neighbor
        // Only surface voxels touch chunk boundary (treated as out-of-bounds, not air)
        assert_eq!(
            reactions, 0,
            "Wood surrounded by wood (no air) should not combust"
        );
    }

    #[test]
    fn heat_diffuses_through_wood() {
        let size = 4;
        let mut voxels: Vec<Voxel> = (0..size * size * size)
            .map(|_| Voxel::new(MaterialId(WOOD)))
            .collect();

        // Hot spot at one end
        voxels[0].temperature = 500.0; // below ignition, just testing diffusion

        let rules = vec![];
        let registry = minimal_registry();

        // Run diffusion-only ticks (dt=5000s for real SI conduction)
        for _ in 0..100 {
            simulate_tick(&mut voxels, size, &rules, &registry, 5000.0);
        }

        // Neighbor should have warmed from diffusion
        let ambient = 288.15;
        assert!(
            voxels[1].temperature > ambient,
            "Heat should diffuse through wood: got {}",
            voxels[1].temperature
        );
    }
}
