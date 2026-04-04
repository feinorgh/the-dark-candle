//! Building part definitions — data-driven shapes placed by the player.
//!
//! Each part type is loaded from a `.part.ron` file and describes:
//! - Its voxel occupancy at sub-voxel resolution (as a bitmask).
//! - Which faces accept attachment to other parts or terrain.
//! - The material slot (one material per part).
//!
//! # Units
//! - All dimensions in metres (1 voxel = 1 m).
//! - Subdivision depth 2 gives 0.25 m resolution inside a 1 m voxel.

use bevy::prelude::*;
use serde::Deserialize;

// ---------------------------------------------------------------------------
// Attachment face flags
// ---------------------------------------------------------------------------

/// Bit flags identifying which faces of a part's bounding box accept joints.
///
/// A value of `0b111111` means all six faces accept connections.
pub type AttachmentFaces = u8;

pub const FACE_PX: AttachmentFaces = 0b000001; // +X
pub const FACE_NX: AttachmentFaces = 0b000010; // −X
pub const FACE_PY: AttachmentFaces = 0b000100; // +Y (top)
pub const FACE_NY: AttachmentFaces = 0b001000; // −Y (bottom)
pub const FACE_PZ: AttachmentFaces = 0b010000; // +Z
pub const FACE_NZ: AttachmentFaces = 0b100000; // −Z

/// All six faces accept attachments (default for most solid parts).
pub const FACE_ALL: AttachmentFaces = 0b111111;

// ---------------------------------------------------------------------------
// PartShape
// ---------------------------------------------------------------------------

/// Geometric shape of a building part in voxel-grid space.
///
/// Each variant encodes the occupancy footprint as sub-voxel dimensions
/// (multiples of 0.25 m at subdivision depth 2).
#[derive(Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum PartShape {
    /// 1×1×1 m solid cube. 4×4×4 sub-voxels fully filled.
    Block,
    /// 1×0.5×1 m half-height slab. 4×2×4 sub-voxels.
    Slab,
    /// 0.25×N×0.25 m vertical column. Height N metres.
    Column { height_m: u8 },
    /// 0.25×0.25×N m horizontal beam. Length N metres.
    Beam { length_m: u8 },
    /// 1×N×0.25 m thin wall panel. Height N metres.
    Wall { height_m: u8 },
    /// 1×0.5×1 m angled roof panel at 45°.
    Roof,
    /// 1×1×1 m stair wedge.
    Stair,
    /// Semicircular arch spanning 1 m horizontally, 0.5 m thick.
    Arch,
}

// ---------------------------------------------------------------------------
// PartData — RON asset
// ---------------------------------------------------------------------------

/// Data asset for a building part type. Loaded from `assets/data/parts/*.part.ron`.
#[derive(Deserialize, Asset, TypePath, Debug, Clone)]
pub struct PartData {
    /// Human-readable identifier (e.g. `"block"`, `"beam"`, `"wall"`).
    pub name: String,
    /// Geometric shape of this part.
    pub shape: PartShape,
    /// Which faces of the bounding box accept joints (bit flags).
    pub attachment_faces: AttachmentFaces,
    /// Display description shown in the build UI.
    pub description: String,
}

// ---------------------------------------------------------------------------
// Part ECS component
// ---------------------------------------------------------------------------

/// Handle to the `PartData` asset for a placed building part.
#[derive(Component, Debug, Clone)]
pub struct PartHandle(pub Handle<PartData>);

/// ECS component marking a placed building part.
#[derive(Component, Debug, Clone)]
pub struct PlacedPart {
    /// The material this part is made from (index into `MaterialData` assets).
    pub material_name: String,
    /// Which part type this is.
    pub part_name: String,
    /// Whether the player can demolish this part.
    pub demolishable: bool,
}

impl PlacedPart {
    pub fn new(material_name: impl Into<String>, part_name: impl Into<String>) -> Self {
        Self {
            material_name: material_name.into(),
            part_name: part_name.into(),
            demolishable: true,
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
    fn face_flags_do_not_overlap() {
        let flags = [FACE_PX, FACE_NX, FACE_PY, FACE_NY, FACE_PZ, FACE_NZ];
        let combined = flags.iter().fold(0u8, |acc, &f| acc | f);
        assert_eq!(combined, FACE_ALL);
        // Each flag is distinct.
        for (i, &a) in flags.iter().enumerate() {
            for (j, &b) in flags.iter().enumerate() {
                if i != j {
                    assert_eq!(a & b, 0, "flags {i} and {j} overlap");
                }
            }
        }
    }

    #[test]
    fn part_shape_variants_are_distinct() {
        assert_ne!(PartShape::Block, PartShape::Slab);
        assert_ne!(PartShape::Roof, PartShape::Stair);
    }

    #[test]
    fn placed_part_is_demolishable_by_default() {
        let p = PlacedPart::new("wood", "block");
        assert!(p.demolishable);
        assert_eq!(p.material_name, "wood");
        assert_eq!(p.part_name, "block");
    }
}
