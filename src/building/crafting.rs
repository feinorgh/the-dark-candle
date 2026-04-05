//! Crafting recipe system — convert raw materials into building components.
//!
//! Recipes are defined in `assets/data/recipes/*.recipe.ron` and loaded via
//! `RonAssetPlugin`. Each recipe specifies:
//! - Input materials and quantities.
//! - Optional tool requirement.
//! - Processing conditions (heat in K, duration in seconds).
//! - Output material or part.
//!
//! # Examples
//! - Sand + heat (1700 K) for 30 s → glass
//! - Iron ore + heat (1811 K) for 60 s → iron
//! - Clay + heat (1200 K) for 120 s → brick
//! - Wood (axe tool) → planks (no heat needed)

use bevy::prelude::*;
use serde::Deserialize;

// ---------------------------------------------------------------------------
// RecipeData — RON asset
// ---------------------------------------------------------------------------

/// A single ingredient in a crafting recipe.
#[derive(Deserialize, Debug, Clone)]
pub struct RecipeIngredient {
    /// Name of the input material (matches `MaterialData::name`).
    pub material: String,
    /// Number of units required (each unit = 1 item stack).
    pub quantity: u32,
}

/// Optional tool class required to execute the recipe.
#[derive(Deserialize, Debug, Clone, PartialEq, Eq, Default)]
pub enum ToolClass {
    Axe,
    Pickaxe,
    Hammer,
    Furnace,
    Anvil,
    #[default]
    None,
}

/// Processing conditions required by this recipe.
#[derive(Deserialize, Debug, Clone, Default)]
pub struct ProcessingConditions {
    /// Minimum temperature required (K). 0.0 = ambient (no heat needed).
    #[serde(default)]
    pub min_temperature_k: f32,
    /// Duration the conditions must be sustained (seconds).
    #[serde(default)]
    pub duration_s: f32,
}

/// A crafting recipe loaded from a `.recipe.ron` file.
#[derive(Deserialize, Asset, TypePath, Debug, Clone)]
pub struct RecipeData {
    /// Unique identifier (e.g. `"wood_to_planks"`, `"sand_to_glass"`).
    pub id: String,
    /// Human-readable display name.
    pub name: String,
    /// Required input materials.
    pub inputs: Vec<RecipeIngredient>,
    /// Tool class required to perform the recipe.
    #[serde(default)]
    pub tool: ToolClass,
    /// Environmental conditions required.
    #[serde(default)]
    pub conditions: ProcessingConditions,
    /// Name of the output material (e.g. `"glass"`, `"iron"`).
    pub output_material: Option<String>,
    /// Name of the output part type (e.g. `"block"`, `"beam"`).
    pub output_part: Option<String>,
    /// Number of output units produced.
    #[serde(default = "default_output_quantity")]
    pub output_quantity: u32,
}

fn default_output_quantity() -> u32 {
    1
}

// ---------------------------------------------------------------------------
// CraftingQueue component
// ---------------------------------------------------------------------------

/// Component tracking an entity's active crafting progress.
///
/// Typically placed on a workbench or furnace entity.
#[derive(Component, Debug, Clone, Default)]
pub struct CraftingQueue {
    /// Currently active recipe handle.
    pub active_recipe: Option<Handle<RecipeData>>,
    /// Elapsed time in seconds since the recipe started.
    pub elapsed_s: f32,
    /// Current temperature at the crafting station (K).
    pub temperature_k: f32,
}

impl CraftingQueue {
    /// Start a new recipe.
    pub fn start(&mut self, recipe: Handle<RecipeData>) {
        self.active_recipe = Some(recipe);
        self.elapsed_s = 0.0;
    }

    /// Advance the recipe. Returns `true` when the recipe completes.
    pub fn tick(&mut self, dt: f32, recipe: &RecipeData) -> bool {
        if self.active_recipe.is_none() {
            return false;
        }
        // Temperature condition must be met.
        if self.temperature_k < recipe.conditions.min_temperature_k {
            return false; // Stalled — not hot enough.
        }
        self.elapsed_s += dt;
        self.elapsed_s >= recipe.conditions.duration_s
    }

    /// Clear the active recipe after completion.
    pub fn complete(&mut self) {
        self.active_recipe = None;
        self.elapsed_s = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crafting_queue_completes_at_duration() {
        let recipe = RecipeData {
            id: "test".to_string(),
            name: "Test".to_string(),
            inputs: vec![],
            tool: ToolClass::None,
            conditions: ProcessingConditions {
                min_temperature_k: 0.0,
                duration_s: 2.0,
            },
            output_material: Some("glass".to_string()),
            output_part: None,
            output_quantity: 1,
        };

        let mut queue = CraftingQueue {
            active_recipe: Some(Handle::default()),
            elapsed_s: 0.0,
            temperature_k: 300.0,
        };

        assert!(!queue.tick(1.0, &recipe)); // 1 s → not done
        assert!(queue.tick(1.5, &recipe)); // 2.5 s total → done
    }

    #[test]
    fn crafting_stalls_below_temperature() {
        let recipe = RecipeData {
            id: "furnace".to_string(),
            name: "Furnace Recipe".to_string(),
            inputs: vec![],
            tool: ToolClass::Furnace,
            conditions: ProcessingConditions {
                min_temperature_k: 1700.0,
                duration_s: 30.0,
            },
            output_material: Some("glass".to_string()),
            output_part: None,
            output_quantity: 1,
        };

        let mut queue = CraftingQueue {
            active_recipe: Some(Handle::default()),
            elapsed_s: 0.0,
            temperature_k: 300.0, // Too cold
        };

        // Should not progress at ambient temperature.
        let done = queue.tick(100.0, &recipe);
        assert!(!done);
        assert_eq!(queue.elapsed_s, 0.0);
    }

    #[test]
    fn default_output_quantity_is_one() {
        assert_eq!(default_output_quantity(), 1);
    }

    #[test]
    fn tool_class_default_is_none() {
        assert_eq!(ToolClass::default(), ToolClass::None);
    }
}
