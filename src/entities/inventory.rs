//! Inventory system — per-entity item storage with weight and volume limits.
//!
//! Each entity with an `Inventory` component can carry items. Items are
//! identified by their material name (for building materials) or item name
//! (for crafted objects). The inventory tracks stack counts and enforces
//! weight and volume limits derived from material properties.
//!
//! # Units
//! - Weight: kilograms (kg)
//! - Volume: cubic metres (m³) — each unit of material ≈ 0.01 m³ (1 dm³)

use std::collections::HashMap;

use bevy::prelude::*;

// ---------------------------------------------------------------------------
// Inventory component
// ---------------------------------------------------------------------------

/// Per-entity item storage.
///
/// Items are keyed by name (material name or item ID). Each entry is a stack
/// count. Maximum weight and volume are enforced on `add_item`.
#[derive(Component, Debug, Clone, Default)]
pub struct Inventory {
    /// Item stacks: name → quantity.
    pub stacks: HashMap<String, u32>,
    /// Maximum carry weight in kg. 0.0 = unlimited.
    pub max_weight_kg: f32,
    /// Current total weight in kg (tracked incrementally).
    pub current_weight_kg: f32,
    /// Maximum carry volume in m³. 0.0 = unlimited.
    pub max_volume_m3: f32,
    /// Current total volume in m³.
    pub current_volume_m3: f32,
}

impl Inventory {
    /// Create an inventory with the given weight and volume limits.
    pub fn new(max_weight_kg: f32, max_volume_m3: f32) -> Self {
        Self {
            max_weight_kg,
            max_volume_m3,
            ..Default::default()
        }
    }

    /// Return the number of units of `item` currently held.
    pub fn count(&self, item: &str) -> u32 {
        self.stacks.get(item).copied().unwrap_or(0)
    }

    /// Add `quantity` units of `item`, subject to weight/volume limits.
    ///
    /// Returns the number of units actually added (may be less than requested
    /// if limits would be exceeded).
    pub fn add_item(
        &mut self,
        item: &str,
        quantity: u32,
        unit_weight_kg: f32,
        unit_volume_m3: f32,
    ) -> u32 {
        if quantity == 0 {
            return 0;
        }
        // Calculate how many units fit within limits.
        let max_by_weight = if self.max_weight_kg > 0.0 && unit_weight_kg > 0.0 {
            let remaining = (self.max_weight_kg - self.current_weight_kg).max(0.0);
            (remaining / unit_weight_kg).floor() as u32
        } else {
            quantity
        };
        let max_by_volume = if self.max_volume_m3 > 0.0 && unit_volume_m3 > 0.0 {
            let remaining = (self.max_volume_m3 - self.current_volume_m3).max(0.0);
            (remaining / unit_volume_m3).floor() as u32
        } else {
            quantity
        };
        let actual = quantity.min(max_by_weight).min(max_by_volume);
        if actual == 0 {
            return 0;
        }
        *self.stacks.entry(item.to_string()).or_insert(0) += actual;
        self.current_weight_kg += unit_weight_kg * actual as f32;
        self.current_volume_m3 += unit_volume_m3 * actual as f32;
        actual
    }

    /// Remove `quantity` units of `item`. Returns units actually removed.
    pub fn remove_item(&mut self, item: &str, quantity: u32, unit_weight_kg: f32, unit_volume_m3: f32) -> u32 {
        let held = self.stacks.get_mut(item);
        let Some(held) = held else { return 0 };
        let actual = quantity.min(*held);
        *held -= actual;
        if *held == 0 {
            self.stacks.remove(item);
        }
        self.current_weight_kg = (self.current_weight_kg - unit_weight_kg * actual as f32).max(0.0);
        self.current_volume_m3 = (self.current_volume_m3 - unit_volume_m3 * actual as f32).max(0.0);
        actual
    }

    /// Returns `true` if the inventory has at least `quantity` of `item`.
    pub fn has(&self, item: &str, quantity: u32) -> bool {
        self.count(item) >= quantity
    }

    /// Returns `true` if all stacks are empty.
    pub fn is_empty(&self) -> bool {
        self.stacks.values().all(|&q| q == 0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_and_count_items() {
        let mut inv = Inventory::default();
        let added = inv.add_item("stone", 5, 2.5, 0.01);
        assert_eq!(added, 5);
        assert_eq!(inv.count("stone"), 5);
    }

    #[test]
    fn weight_limit_respected() {
        let mut inv = Inventory::new(10.0, 0.0); // 10 kg max
        let added = inv.add_item("iron", 10, 2.0, 0.0); // 2 kg each → fits 5
        assert_eq!(added, 5);
        assert_eq!(inv.count("iron"), 5);
        assert!((inv.current_weight_kg - 10.0).abs() < 1e-4);
    }

    #[test]
    fn volume_limit_respected() {
        let mut inv = Inventory::new(0.0, 0.1); // 0.1 m³ max
        let added = inv.add_item("wood", 20, 0.0, 0.01); // 0.01 m³ each → fits 10
        assert_eq!(added, 10);
    }

    #[test]
    fn remove_reduces_stack() {
        let mut inv = Inventory::default();
        inv.add_item("stone", 10, 0.0, 0.0);
        let removed = inv.remove_item("stone", 3, 0.0, 0.0);
        assert_eq!(removed, 3);
        assert_eq!(inv.count("stone"), 7);
    }

    #[test]
    fn remove_nonexistent_returns_zero() {
        let mut inv = Inventory::default();
        let removed = inv.remove_item("air", 5, 0.0, 0.0);
        assert_eq!(removed, 0);
    }

    #[test]
    fn has_checks_minimum_quantity() {
        let mut inv = Inventory::default();
        inv.add_item("brick", 3, 0.0, 0.0);
        assert!(inv.has("brick", 3));
        assert!(!inv.has("brick", 4));
    }

    #[test]
    fn empty_inventory_is_empty() {
        let inv = Inventory::default();
        assert!(inv.is_empty());
    }
}
