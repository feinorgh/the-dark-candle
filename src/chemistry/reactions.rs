// Chemical reaction rules and processing.
//
// Reactions are defined as RON data: input materials + conditions → output materials + effects.
// The reaction processor checks adjacent voxel pairs each tick and applies matching rules.

#![allow(dead_code)]

use bevy::prelude::*;
use bevy_common_assets::ron::RonAssetPlugin;
use serde::Deserialize;

use crate::world::voxel::MaterialId;

/// A single reaction rule loaded from `.reaction.ron`.
#[derive(Deserialize, Asset, TypePath, Debug, Clone, PartialEq)]
pub struct ReactionData {
    pub name: String,
    /// Material ID of the primary reactant.
    pub input_a: u16,
    /// Material ID of the adjacent catalyst/reactant (or None for self-reactions).
    pub input_b: Option<u16>,
    /// Minimum temperature (K) for this reaction to occur.
    pub min_temperature: f32,
    /// Maximum temperature (K) — use a very high value for "no upper limit".
    pub max_temperature: f32,
    /// What material the primary voxel becomes.
    pub output_a: u16,
    /// What the adjacent voxel becomes (if input_b was specified).
    pub output_b: Option<u16>,
    /// Temperature change applied to the output voxel (positive = exothermic).
    pub heat_output: f32,
}

/// Result of checking a reaction rule against two voxels.
#[derive(Debug, Clone, PartialEq)]
pub struct ReactionResult {
    pub new_material_a: MaterialId,
    pub new_material_b: Option<MaterialId>,
    pub heat_output: f32,
}

/// Check if a reaction rule matches the given conditions.
pub fn check_reaction(
    rule: &ReactionData,
    material_a: MaterialId,
    material_b: MaterialId,
    temperature: f32,
) -> Option<ReactionResult> {
    if material_a.0 != rule.input_a {
        return None;
    }
    if let Some(req_b) = rule.input_b {
        if material_b.0 != req_b {
            return None;
        }
    }
    if temperature < rule.min_temperature || temperature > rule.max_temperature {
        return None;
    }

    Some(ReactionResult {
        new_material_a: MaterialId(rule.output_a),
        new_material_b: rule.output_b.map(MaterialId),
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

    fn wood_burning_rule() -> ReactionData {
        ReactionData {
            name: "Wood combustion".into(),
            input_a: 5,       // wood
            input_b: Some(0), // adjacent to air (oxygen)
            min_temperature: 573.0,
            max_temperature: 99999.0,
            output_a: 8,    // charcoal (hypothetical material ID)
            output_b: None, // air stays air
            heat_output: 200.0,
        }
    }

    fn ice_melting_rule() -> ReactionData {
        ReactionData {
            name: "Ice melting".into(),
            input_a: 9, // ice (hypothetical)
            input_b: None,
            min_temperature: 273.15,
            max_temperature: 99999.0,
            output_a: 3, // water
            output_b: None,
            heat_output: -50.0, // endothermic
        }
    }

    #[test]
    fn reaction_matches_when_conditions_met() {
        let rule = wood_burning_rule();
        let result = check_reaction(&rule, MaterialId(5), MaterialId(0), 600.0);
        assert!(result.is_some());
        let r = result.unwrap();
        assert_eq!(r.new_material_a, MaterialId(8));
        assert_eq!(r.heat_output, 200.0);
    }

    #[test]
    fn reaction_fails_wrong_material() {
        let rule = wood_burning_rule();
        let result = check_reaction(&rule, MaterialId(1), MaterialId(0), 600.0);
        assert!(result.is_none(), "Stone shouldn't burn");
    }

    #[test]
    fn reaction_fails_wrong_neighbor() {
        let rule = wood_burning_rule();
        let result = check_reaction(&rule, MaterialId(5), MaterialId(3), 600.0);
        assert!(result.is_none(), "Wood next to water shouldn't combust");
    }

    #[test]
    fn reaction_fails_below_min_temp() {
        let rule = wood_burning_rule();
        let result = check_reaction(&rule, MaterialId(5), MaterialId(0), 293.0);
        assert!(result.is_none(), "Room temp shouldn't ignite wood");
    }

    #[test]
    fn self_reaction_ignores_neighbor_material() {
        let rule = ice_melting_rule();
        // input_b is None, so neighbor material doesn't matter
        let result = check_reaction(&rule, MaterialId(9), MaterialId(1), 280.0);
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
                input_a: 5,
                input_b: Some(0),
                min_temperature: 573.0,
                max_temperature: 99999.0,
                output_a: 8,
                output_b: None,
                heat_output: 200.0,
            )
        "#;
        let data: ReactionData =
            ron::from_str(ron_str).expect("Failed to deserialize ReactionData");
        assert_eq!(data.name, "Test Reaction");
        assert_eq!(data.input_a, 5);
        assert_eq!(data.input_b, Some(0));
        assert_eq!(data.heat_output, 200.0);
    }
}
