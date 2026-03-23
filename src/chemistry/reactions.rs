// Chemical reaction rules and processing.
//
// Reactions are defined as RON data: input materials + conditions → output materials + effects.
// The reaction processor checks adjacent voxel pairs each tick and applies matching rules.

#![allow(dead_code)]

use bevy::prelude::*;
use bevy_common_assets::ron::RonAssetPlugin;
use serde::Deserialize;

use crate::data::MaterialRegistry;
use crate::world::voxel::MaterialId;

/// A single reaction rule loaded from `.reaction.ron`.
#[derive(Deserialize, Asset, TypePath, Debug, Clone, PartialEq)]
pub struct ReactionData {
    pub name: String,
    /// Material name of the primary reactant.
    pub input_a: String,
    /// Material name of the adjacent catalyst/reactant (or None for self-reactions).
    pub input_b: Option<String>,
    /// Minimum temperature (K) for this reaction to occur.
    pub min_temperature: f32,
    /// Maximum temperature (K) — use a very high value for "no upper limit".
    pub max_temperature: f32,
    /// What material the primary voxel becomes (by name).
    pub output_a: String,
    /// What the adjacent voxel becomes (if input_b was specified, by name).
    pub output_b: Option<String>,
    /// Temperature change in Kelvin (ΔT) applied to the reacted voxel.
    ///
    /// Positive values are exothermic (combustion heats up the product voxel),
    /// negative values are endothermic (melting absorbs heat).
    ///
    /// This is a pragmatic temperature delta — the simulation loop adds it
    /// directly to the voxel's temperature. Values should approximate the
    /// expected product temperature relative to the reaction's trigger point.
    /// For example, wood combustion yields ~800 K delta, bringing a voxel
    /// from its ignition point (~573 K) up toward flame temperature (~1300 K).
    ///
    // TODO: Convert to J/kg (real SI energy) and use a sustained burn-rate
    // model: ΔT = heat_output × burn_rate × dt / (ρ_product × Cₚ_product).
    // This requires `MaterialRegistry` access at reaction-application time
    // and a per-voxel burn-progress tracker. See `MaterialData::heat_of_combustion`.
    pub heat_output: f32,
}

/// Result of checking a reaction rule against two voxels.
#[derive(Debug, Clone, PartialEq)]
pub struct ReactionResult {
    pub new_material_a: MaterialId,
    pub new_material_b: Option<MaterialId>,
    /// Temperature delta (K) to add to the reacted voxel. See [`ReactionData::heat_output`].
    pub heat_output: f32,
}

/// Check if a reaction rule matches the given conditions.
pub fn check_reaction(
    rule: &ReactionData,
    material_a: MaterialId,
    material_b: MaterialId,
    temperature: f32,
    registry: &MaterialRegistry,
) -> Option<ReactionResult> {
    let required_a = registry.resolve_name(&rule.input_a)?;
    if material_a != required_a {
        return None;
    }
    if let Some(ref req_b_name) = rule.input_b {
        let required_b = registry.resolve_name(req_b_name)?;
        if material_b != required_b {
            return None;
        }
    }
    if temperature < rule.min_temperature || temperature > rule.max_temperature {
        return None;
    }

    Some(ReactionResult {
        new_material_a: registry.resolve_name(&rule.output_a)?,
        new_material_b: rule
            .output_b
            .as_ref()
            .and_then(|name| registry.resolve_name(name)),
        heat_output: rule.heat_output,
    })
}

/// Plugin to register reaction RON loading.
pub struct ReactionPlugin;

impl Plugin for ReactionPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(RonAssetPlugin::<ReactionData>::new(&["reaction.ron"]));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{MaterialData, MaterialRegistry, Phase};

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
            boiled_into: None,
            frozen_into: None,
            condensed_into: None,
            ..Default::default()
        });
        reg.insert(MaterialData {
            id: 5,
            name: "Wood".into(),
            default_phase: Phase::Solid,
            density: 600.0,
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
            melted_into: None,
            boiled_into: None,
            frozen_into: None,
            condensed_into: None,
            ..Default::default()
        });
        reg.insert(MaterialData {
            id: 8,
            name: "Charcoal".into(),
            default_phase: Phase::Solid,
            density: 400.0,
            melting_point: None,
            boiling_point: None,
            ignition_point: None,
            hardness: 0.1,
            color: [0.2, 0.2, 0.2],
            transparent: false,
            melted_into: None,
            boiled_into: None,
            frozen_into: None,
            condensed_into: None,
            ..Default::default()
        });
        reg.insert(MaterialData {
            id: 9,
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
        reg
    }

    fn wood_burning_rule() -> ReactionData {
        ReactionData {
            name: "Wood combustion".into(),
            input_a: "Wood".into(),
            input_b: Some("Air".into()),
            min_temperature: 573.0,
            max_temperature: 99999.0,
            output_a: "Charcoal".into(),
            output_b: None,
            // ΔT ≈ flame temperature (1300 K) minus ignition point (573 K).
            heat_output: 800.0,
        }
    }

    fn ice_melting_rule() -> ReactionData {
        ReactionData {
            name: "Ice melting".into(),
            input_a: "Ice".into(),
            input_b: None,
            min_temperature: 273.15,
            max_temperature: 99999.0,
            output_a: "Water".into(),
            output_b: None,
            // Endothermic: latent heat absorption cools the melt product.
            // Approximate ΔT for the phase change (real latent heat TBD in Phase 5).
            heat_output: -10.0,
        }
    }

    #[test]
    fn reaction_matches_when_conditions_met() {
        let rule = wood_burning_rule();
        let reg = test_registry();
        let result = check_reaction(&rule, MaterialId(5), MaterialId(0), 600.0, &reg);
        assert!(result.is_some());
        let r = result.unwrap();
        assert_eq!(r.new_material_a, MaterialId(8));
        assert_eq!(r.heat_output, 800.0);
    }

    #[test]
    fn reaction_fails_wrong_material() {
        let rule = wood_burning_rule();
        let reg = test_registry();
        let result = check_reaction(&rule, MaterialId(1), MaterialId(0), 600.0, &reg);
        assert!(result.is_none(), "Stone shouldn't burn");
    }

    #[test]
    fn reaction_fails_wrong_neighbor() {
        let rule = wood_burning_rule();
        let reg = test_registry();
        let result = check_reaction(&rule, MaterialId(5), MaterialId(3), 600.0, &reg);
        assert!(result.is_none(), "Wood next to water shouldn't combust");
    }

    #[test]
    fn reaction_fails_below_min_temp() {
        let rule = wood_burning_rule();
        let reg = test_registry();
        let result = check_reaction(&rule, MaterialId(5), MaterialId(0), 293.0, &reg);
        assert!(result.is_none(), "Room temp shouldn't ignite wood");
    }

    #[test]
    fn self_reaction_ignores_neighbor_material() {
        let rule = ice_melting_rule();
        let reg = test_registry();
        // input_b is None, so neighbor material doesn't matter
        let result = check_reaction(&rule, MaterialId(9), MaterialId(1), 280.0, &reg);
        assert!(result.is_some());
        let r = result.unwrap();
        assert_eq!(r.new_material_a, MaterialId(3)); // becomes water
        assert!(r.heat_output < 0.0, "Melting is endothermic");
    }

    #[test]
    fn reaction_data_deserializes_from_ron() {
        let ron_str = r#"
            ReactionData(
                name: "Test Reaction",
                input_a: "Wood",
                input_b: Some("Air"),
                min_temperature: 573.0,
                max_temperature: 99999.0,
                output_a: "Charcoal",
                output_b: None,
                heat_output: 800.0,
            )
        "#;
        let data: ReactionData =
            ron::from_str(ron_str).expect("Failed to deserialize ReactionData");
        assert_eq!(data.name, "Test Reaction");
        assert_eq!(data.input_a, "Wood");
        assert_eq!(data.input_b, Some("Air".into()));
        assert_eq!(data.heat_output, 800.0);
    }
}
