//! Player building placement — build mode, ghost preview, part spawning.
//!
//! # Build mode flow
//! 1. Player presses B → `BuildMode::Inactive` → `BuildMode::Active`.
//! 2. Player selects part type + material from build menu.
//! 3. A *ghost* entity follows the cursor showing the placement preview.
//! 4. Player left-clicks → validates support + inventory → spawns `PlacedPart`.
//! 5. Adjacent parts are scanned; joints created automatically.
//! 6. Player right-clicks → cancel / exit build mode.
//!
//! # Grid snapping
//! All parts snap to a 1 m grid (0.5 m offset so blocks sit on the ground
//! surface rather than straddling the voxel boundary).

use bevy::prelude::*;

use super::joints::Joint;
use super::parts::PlacedPart;
use crate::entities::inventory::Inventory;

// ---------------------------------------------------------------------------
// Build mode state
// ---------------------------------------------------------------------------

/// Resource tracking whether the player is in build mode.
#[derive(Resource, Debug, Clone, PartialEq, Eq, Default)]
pub enum BuildMode {
    #[default]
    Inactive,
    Active {
        /// The selected part type name.
        part_name: String,
        /// The selected material name.
        material_name: String,
        /// Current rotation step (0–3, each step = 90° around Y axis).
        rotation_step: u8,
    },
}

impl BuildMode {
    pub fn is_active(&self) -> bool {
        matches!(self, BuildMode::Active { .. })
    }
}

// ---------------------------------------------------------------------------
// Ghost entity marker
// ---------------------------------------------------------------------------

/// Marker component on the ghost preview entity.
#[derive(Component)]
pub struct GhostPart;

// ---------------------------------------------------------------------------
// Placement validation
// ---------------------------------------------------------------------------

/// Result of a placement validity check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlacementResult {
    /// Placement is valid.
    Valid,
    /// The target location is already occupied.
    Occupied,
    /// The part would be unsupported (floating with no adjacent part or terrain).
    Unsupported,
    /// The player lacks the required material in their inventory.
    NoMaterial,
}

/// Snap a world-space position to the 1 m building grid.
///
/// Returns the grid-aligned centre of the voxel the position falls in.
pub fn snap_to_grid(pos: Vec3) -> Vec3 {
    Vec3::new(pos.x.floor() + 0.5, pos.y.floor() + 0.5, pos.z.floor() + 0.5)
}

/// Check whether a part can be placed at `grid_pos`.
///
/// Rules:
/// 1. No other `PlacedPart` already occupies that grid cell.
/// 2. At least one adjacent cell has a `PlacedPart` **or** the part touches terrain (y ≤ 1).
/// 3. The player's `Inventory` contains at least 1 unit of `material_name`.
pub fn validate_placement(
    grid_pos: Vec3,
    material_name: &str,
    placed_parts: &Query<&Transform, With<PlacedPart>>,
    inventory: Option<&Inventory>,
) -> PlacementResult {
    // Rule 1 — no overlap.
    for other_transform in placed_parts {
        let other_pos = snap_to_grid(other_transform.translation);
        if (other_pos - grid_pos).length_squared() < 0.01 {
            return PlacementResult::Occupied;
        }
    }

    // Rule 2 — support check.
    let on_ground = grid_pos.y <= 1.5;
    let has_neighbour = placed_parts.iter().any(|t| {
        let other = snap_to_grid(t.translation);
        let diff = (other - grid_pos).abs();
        // Adjacent in exactly one axis at distance 1.
        (diff.x < 1.01 && diff.y < 0.01 && diff.z < 0.01)
            || (diff.x < 0.01 && diff.y < 1.01 && diff.z < 0.01)
            || (diff.x < 0.01 && diff.y < 0.01 && diff.z < 1.01)
    });

    if !on_ground && !has_neighbour {
        return PlacementResult::Unsupported;
    }

    // Rule 3 — inventory check.
    if let Some(inv) = inventory {
        if inv.count(material_name) == 0 {
            return PlacementResult::NoMaterial;
        }
    }

    PlacementResult::Valid
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

/// Toggle build mode on/off when the player presses B.
pub fn toggle_build_mode(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut build_mode: ResMut<BuildMode>,
) {
    if keyboard.just_pressed(KeyCode::KeyB) {
        *build_mode = match &*build_mode {
            BuildMode::Inactive => BuildMode::Active {
                part_name: "block".to_string(),
                material_name: "stone".to_string(),
                rotation_step: 0,
            },
            BuildMode::Active { .. } => BuildMode::Inactive,
        };
    }
}

/// Rotate the selected part 90° when R is pressed (in build mode).
pub fn rotate_selection(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut build_mode: ResMut<BuildMode>,
) {
    if !keyboard.just_pressed(KeyCode::KeyR) {
        return;
    }
    if let BuildMode::Active { rotation_step, .. } = &mut *build_mode {
        *rotation_step = (*rotation_step + 1) % 4;
    }
}

/// Spawn a placed part at `grid_pos` and create joints to all adjacent parts.
pub fn spawn_part_at(
    commands: &mut Commands,
    grid_pos: Vec3,
    rotation_step: u8,
    material_name: &str,
    part_name: &str,
    placed_query: &Query<(Entity, &Transform), With<PlacedPart>>,
) {
    let rotation = Quat::from_rotation_y(rotation_step as f32 * std::f32::consts::FRAC_PI_2);
    let new_entity = commands
        .spawn((
            Transform::from_translation(grid_pos).with_rotation(rotation),
            PlacedPart::new(material_name, part_name),
        ))
        .id();

    // Auto-create joints with all adjacent placed parts.
    for (adj_entity, adj_transform) in placed_query {
        let adj_pos = snap_to_grid(adj_transform.translation);
        let diff = (adj_pos - grid_pos).abs();
        let is_adjacent = (diff.x < 1.01 && diff.y < 0.01 && diff.z < 0.01)
            || (diff.x < 0.01 && diff.y < 1.01 && diff.z < 0.01)
            || (diff.x < 0.01 && diff.y < 0.01 && diff.z < 1.01);

        if is_adjacent {
            commands.spawn(Joint::new(new_entity, adj_entity));
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snap_to_grid_rounds_to_half() {
        let pos = Vec3::new(1.7, 0.2, -0.4);
        let snapped = snap_to_grid(pos);
        assert!((snapped.x - 1.5).abs() < 1e-5);
        assert!((snapped.y - 0.5).abs() < 1e-5);
        assert!((snapped.z - (-0.5)).abs() < 1e-5);
    }

    #[test]
    fn snap_negative_coords() {
        let pos = Vec3::new(-2.9, -0.1, -1.0);
        let snapped = snap_to_grid(pos);
        assert!((snapped.x - (-2.5)).abs() < 1e-5);
        assert!((snapped.y - (-0.5)).abs() < 1e-5);
        assert!((snapped.z - (-0.5)).abs() < 1e-5);
    }

    #[test]
    fn build_mode_default_inactive() {
        let mode = BuildMode::default();
        assert_eq!(mode, BuildMode::Inactive);
        assert!(!mode.is_active());
    }

    #[test]
    fn build_mode_active() {
        let mode = BuildMode::Active {
            part_name: "block".to_string(),
            material_name: "stone".to_string(),
            rotation_step: 0,
        };
        assert!(mode.is_active());
    }
}
