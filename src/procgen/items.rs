// Procedural item generation: material + form → emergent properties.
//
// Given an ItemData template and an optional material override, produces
// a unique Item ECS component with properties influenced by the material's
// physical characteristics (density → weight, hardness → durability/damage).

#![allow(dead_code)]

use crate::data::{ItemCategory, ItemData, MaterialData};
use bevy::prelude::*;
use serde::{Deserialize, Serialize};

/// ECS component for a spawned item instance.
#[derive(Serialize, Deserialize, Component, Debug, Clone)]
pub struct Item {
    pub item_type: String,
    pub display_name: String,
    pub category: ItemCategory,
    pub material_id: u16,
    pub weight: f32,
    pub durability: f32,
    pub max_durability: f32,
    pub damage: f32,
    pub armor: f32,
    pub nutrition: f32,
    pub stackable: bool,
    pub max_stack: u32,
    pub stack_count: u32,
}

/// Material modifiers applied to item base stats.
struct MaterialModifiers {
    weight_mult: f32,
    durability_mult: f32,
    damage_mult: f32,
    armor_mult: f32,
}

/// Compute how a material modifies item properties.
fn material_modifiers(material: &MaterialData) -> MaterialModifiers {
    // Reference: iron density=7874, hardness=0.6
    let density_ref = 7874.0_f32;
    let hardness_ref = 0.6_f32;

    MaterialModifiers {
        weight_mult: material.density / density_ref,
        durability_mult: material.hardness / hardness_ref,
        damage_mult: (material.hardness / hardness_ref).sqrt(),
        armor_mult: (material.hardness / hardness_ref + material.density / density_ref) * 0.5,
    }
}

/// Generate an item instance from a template, optionally modified by material properties.
pub fn generate_item(template: &ItemData, material: Option<&MaterialData>) -> Item {
    let (weight, durability, damage, armor, display_name, material_id) = if let Some(mat) = material
    {
        let m = material_modifiers(mat);
        let name = template.display_name.replace("{material}", &mat.name);
        (
            template.base_weight * m.weight_mult,
            template.base_durability * m.durability_mult,
            template.base_damage * m.damage_mult,
            template.base_armor * m.armor_mult,
            name,
            mat.id,
        )
    } else {
        (
            template.base_weight,
            template.base_durability,
            template.base_damage,
            template.base_armor,
            template.display_name.clone(),
            template.primary_material,
        )
    };

    Item {
        item_type: template.item_type.clone(),
        display_name,
        category: template.category,
        material_id,
        weight: weight.max(0.01),
        durability: durability.max(0.0),
        max_durability: durability.max(0.0),
        damage: damage.max(0.0),
        armor: armor.max(0.0),
        nutrition: template.nutrition,
        stackable: template.stackable,
        max_stack: template.max_stack,
        stack_count: 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Phase;

    fn sword_template() -> ItemData {
        ItemData {
            item_type: "sword".into(),
            display_name: "{material} Sword".into(),
            category: ItemCategory::Weapon,
            primary_material: 4,
            base_weight: 2.5,
            base_durability: 150.0,
            base_damage: 20.0,
            base_armor: 0.0,
            nutrition: 0.0,
            stackable: false,
            max_stack: 1,
        }
    }

    fn iron_material() -> MaterialData {
        MaterialData {
            id: 4,
            name: "Iron".into(),
            default_phase: Phase::Solid,
            density: 7874.0,
            melting_point: Some(1811.0),
            boiling_point: Some(3134.0),
            ignition_point: None,
            hardness: 0.6,
            color: [0.7, 0.7, 0.72],
            transparent: false,
            melted_into: None,
            boiled_into: None,
            frozen_into: None,
            condensed_into: None,
        }
    }

    fn stone_material() -> MaterialData {
        MaterialData {
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
        }
    }

    fn wood_material() -> MaterialData {
        MaterialData {
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
        }
    }

    #[test]
    fn generate_item_without_material() {
        let item = generate_item(&sword_template(), None);
        assert_eq!(item.item_type, "sword");
        assert_eq!(item.display_name, "{material} Sword");
        assert_eq!(item.weight, 2.5);
        assert_eq!(item.durability, 150.0);
        assert_eq!(item.damage, 20.0);
    }

    #[test]
    fn iron_sword_uses_reference_material() {
        let item = generate_item(&sword_template(), Some(&iron_material()));
        assert_eq!(item.display_name, "Iron Sword");
        assert_eq!(item.material_id, 4);
        // Iron is the reference material, so multipliers ≈ 1.0
        assert!((item.weight - 2.5).abs() < 0.1);
        assert!((item.durability - 150.0).abs() < 1.0);
        assert!((item.damage - 20.0).abs() < 0.5);
    }

    #[test]
    fn stone_sword_is_heavier_more_durable() {
        let iron_item = generate_item(&sword_template(), Some(&iron_material()));
        let stone_item = generate_item(&sword_template(), Some(&stone_material()));

        // Stone is less dense than iron → lighter
        assert!(stone_item.weight < iron_item.weight);
        // Stone is harder → more durable
        assert!(stone_item.durability > iron_item.durability);
        assert_eq!(stone_item.display_name, "Stone Sword");
    }

    #[test]
    fn wood_sword_is_light_and_fragile() {
        let iron_item = generate_item(&sword_template(), Some(&iron_material()));
        let wood_item = generate_item(&sword_template(), Some(&wood_material()));

        assert!(
            wood_item.weight < iron_item.weight,
            "Wood should be lighter"
        );
        assert!(
            wood_item.durability < iron_item.durability,
            "Wood should be less durable"
        );
        assert!(
            wood_item.damage < iron_item.damage,
            "Wood should deal less damage"
        );
        assert_eq!(wood_item.display_name, "Wood Sword");
    }

    #[test]
    fn food_item_retains_nutrition() {
        let template = ItemData {
            item_type: "apple".into(),
            display_name: "Apple".into(),
            category: ItemCategory::Food,
            primary_material: 0,
            base_weight: 0.2,
            base_durability: 10.0,
            base_damage: 0.0,
            base_armor: 0.0,
            nutrition: 150.0,
            stackable: true,
            max_stack: 16,
        };
        let item = generate_item(&template, None);
        assert_eq!(item.nutrition, 150.0);
        assert!(item.stackable);
        assert_eq!(item.max_stack, 16);
        assert_eq!(item.stack_count, 1);
    }

    #[test]
    fn max_durability_equals_initial() {
        let item = generate_item(&sword_template(), Some(&iron_material()));
        assert_eq!(item.durability, item.max_durability);
    }

    #[test]
    fn weight_is_always_positive() {
        let mut t = sword_template();
        t.base_weight = 0.0;
        let item = generate_item(&t, Some(&iron_material()));
        assert!(item.weight > 0.0);
    }

    #[test]
    fn material_name_substitution() {
        let mut t = sword_template();
        t.display_name = "Mighty {material} Blade".into();
        let item = generate_item(&t, Some(&stone_material()));
        assert_eq!(item.display_name, "Mighty Stone Blade");
    }
}
