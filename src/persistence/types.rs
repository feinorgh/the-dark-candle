// Save-file data types.
//
// All types here are plain Rust structs that mirror ECS component data for
// serialisation. Vec3/Quat are stored as arrays to avoid needing bevy's
// `serialize` feature. Voxel arrays use run-length encoding to keep RON
// file sizes tractable (a solid-stone chunk compresses from ~2 MB to <1 KB).

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    behavior::needs::Needs,
    biology::{growth::Growth, health::Health, metabolism::Metabolism},
    procgen::{creatures::Creature, items::Item},
};

pub const SAVE_VERSION: u32 = 3;
pub const SAVE_DIR: &str = "saves";
/// Legacy single-file path kept for backward compatibility on load.
pub const LEGACY_SAVE_PATH: &str = "saves/save.ron";

/// Number of manual save slots (in addition to the autosave slot).
pub const MANUAL_SLOT_COUNT: usize = 3;

/// Identifies a save slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SaveSlot {
    Auto,
    Manual(u8), // 1..=MANUAL_SLOT_COUNT
}

impl SaveSlot {
    pub fn filename(&self) -> String {
        match self {
            SaveSlot::Auto => "autosave.ron".to_string(),
            SaveSlot::Manual(n) => format!("slot_{n}.ron"),
        }
    }

    pub fn path(&self) -> String {
        format!("{}/{}", SAVE_DIR, self.filename())
    }

    pub fn label(&self) -> String {
        match self {
            SaveSlot::Auto => "Autosave".to_string(),
            SaveSlot::Manual(n) => format!("Slot {n}"),
        }
    }

    /// All available slots in display order.
    pub fn all() -> Vec<SaveSlot> {
        let mut slots = vec![SaveSlot::Auto];
        for i in 1..=MANUAL_SLOT_COUNT as u8 {
            slots.push(SaveSlot::Manual(i));
        }
        slots
    }
}

// ---------------------------------------------------------------------------
// Stable entity identity
// ---------------------------------------------------------------------------

/// Stable cross-session entity identifier assigned at spawn. Used to remap
/// relationship references when entity bits change between sessions.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SaveId(pub u64);

// ---------------------------------------------------------------------------
// Root save document
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SaveGame {
    pub version: u32,
    pub terrain: TerrainConfigSave,
    pub chunks: Vec<ChunkSave>,
    pub creatures: Vec<CreatureSave>,
    pub items: Vec<ItemSave>,
    pub enemies: Vec<EnemySave>,
    pub factions: FactionRegistrySave,
    /// Player state (None for saves from v2 or earlier).
    #[serde(default)]
    pub player: Option<PlayerSave>,
}

// ---------------------------------------------------------------------------
// Player state
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PlayerSave {
    pub position: [f32; 3],
    pub pitch: f32,
    pub yaw: f32,
    pub health_current: f32,
    pub health_max: f32,
    pub gravity_enabled: bool,
    pub hotbar_slots: Vec<u16>,
    pub hotbar_selected: usize,
}

// ---------------------------------------------------------------------------
// Terrain
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TerrainConfigSave {
    pub seed: u32,
    pub sea_level: i32,
    pub height_scale: f64,
    pub continent_freq: f64,
    pub detail_freq: f64,
    pub cave_freq: f64,
    pub cave_threshold: f64,
    pub soil_depth: i32,
}

// ---------------------------------------------------------------------------
// Chunks (RLE-encoded voxels)
// ---------------------------------------------------------------------------

/// A single compressed voxel run.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct VoxelRun {
    pub count: u32,
    /// material id, temperature, pressure, damage
    pub material: u16,
    pub temperature: f32,
    pub pressure: f32,
    pub damage: f32,
}

/// Encode a flat voxel slice into runs of consecutive identical voxels.
pub fn encode_rle(voxels: &[crate::world::voxel::Voxel]) -> Vec<VoxelRun> {
    let mut runs: Vec<VoxelRun> = Vec::new();
    for v in voxels {
        match runs.last_mut() {
            Some(r)
                if r.material == v.material.0
                    && r.temperature == v.temperature
                    && r.pressure == v.pressure
                    && r.damage == v.damage =>
            {
                r.count += 1;
            }
            _ => runs.push(VoxelRun {
                count: 1,
                material: v.material.0,
                temperature: v.temperature,
                pressure: v.pressure,
                damage: v.damage,
            }),
        }
    }
    runs
}

/// Decode RLE runs back into a flat voxel Vec.
pub fn decode_rle(runs: &[VoxelRun]) -> Vec<crate::world::voxel::Voxel> {
    use crate::world::voxel::{MaterialId, Voxel};
    let mut out = Vec::new();
    for r in runs {
        let v = Voxel {
            material: MaterialId(r.material),
            temperature: r.temperature,
            pressure: r.pressure,
            damage: r.damage,
            latent_heat_buffer: 0.0,
        };
        for _ in 0..r.count {
            out.push(v);
        }
    }
    out
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ChunkSave {
    /// Chunk-space coordinate.
    pub coord: [i32; 3],
    /// RLE-compressed voxel array (total decoded length must equal CHUNK_VOLUME).
    pub runs: Vec<VoxelRun>,
}

// ---------------------------------------------------------------------------
// Physics mirror types (avoid requiring bevy/serialize feature)
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PhysicsBodySave {
    pub gravity_scale: f32,
    pub foot_offset: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ColliderSave {
    pub half_extents: [f32; 3],
}

// ---------------------------------------------------------------------------
// Relationships (serialised as Vec to avoid HashMap key issues in RON)
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RelationshipEntry {
    /// SaveId of the other creature (stable across sessions).
    pub other_id: u64,
    pub trust: f32,
    pub familiarity: f32,
    pub hostility: f32,
}

// ---------------------------------------------------------------------------
// Entity save records
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CreatureSave {
    pub id: u64,
    pub position: [f32; 3],
    pub rotation: [f32; 4],
    pub creature: Creature,
    pub health: Health,
    pub metabolism: Metabolism,
    pub growth: Growth,
    pub needs: Needs,
    pub relationships: Vec<RelationshipEntry>,
    pub physics: PhysicsBodySave,
    pub collider: ColliderSave,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ItemSave {
    pub id: u64,
    pub position: [f32; 3],
    pub rotation: [f32; 4],
    pub item: Item,
    pub physics: PhysicsBodySave,
    pub collider: ColliderSave,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EnemySave {
    pub id: u64,
    pub position: [f32; 3],
    pub rotation: [f32; 4],
    pub speed: f32,
}

// ---------------------------------------------------------------------------
// Factions (converted from HashMaps to flat Vecs for reliable RON output)
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FactionSave {
    pub id: u32,
    pub name: String,
    /// SaveIds of member creatures.
    pub member_ids: Vec<u64>,
    pub territory: Vec<[i32; 2]>,
    pub outsider_disposition: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FactionRelationEntry {
    pub faction_a: u32,
    pub faction_b: u32,
    pub standing: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FactionRegistrySave {
    pub factions: Vec<FactionSave>,
    pub relations: Vec<FactionRelationEntry>,
    /// (SaveId, FactionId) pairs.
    pub creature_factions: Vec<(u64, u32)>,
}
