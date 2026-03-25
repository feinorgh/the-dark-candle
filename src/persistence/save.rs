// Save system: serialize the current game state to `saves/save.ron`.
//
// Triggered by pressing F5. Queries all persistent ECS entities, converts
// them into serialisable save types, and writes a `SaveGame` document via
// the `ron` crate.

use std::collections::HashMap;

use bevy::prelude::*;

use crate::{
    behavior::needs::Needs,
    biology::{growth::Growth, health::Health, metabolism::Metabolism},
    entities::Enemy,
    physics::{collision::Collider, gravity::PhysicsBody},
    procgen::{creatures::Creature, items::Item},
    social::{factions::FactionRegistry, relationships::Relationships},
    world::{chunk::Chunk, chunk_manager::TerrainGeneratorRes},
};

use super::types::{
    ChunkSave, ColliderSave, CreatureSave, EnemySave, FactionRegistrySave, FactionRelationEntry,
    FactionSave, ItemSave, PhysicsBodySave, RelationshipEntry, SAVE_PATH, SAVE_VERSION, SaveGame,
    SaveId, TerrainConfigSave, encode_rle,
};

// ---------------------------------------------------------------------------
// Creature query helper type alias
// ---------------------------------------------------------------------------

#[allow(clippy::type_complexity)]
type CreatureQuery<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static SaveId,
        &'static Transform,
        &'static Creature,
        &'static Health,
        &'static Metabolism,
        &'static Growth,
        &'static Needs,
        &'static Relationships,
        &'static PhysicsBody,
        &'static Collider,
    ),
    With<Creature>,
>;

// ---------------------------------------------------------------------------
// Item query helper type alias
// ---------------------------------------------------------------------------

type ItemQuery<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static SaveId,
        &'static Transform,
        &'static Item,
        &'static PhysicsBody,
        &'static Collider,
    ),
    With<Item>,
>;

// ---------------------------------------------------------------------------
// Save system
// ---------------------------------------------------------------------------

pub fn save_game(
    keyboard: Res<ButtonInput<KeyCode>>,
    terrain_gen: Res<TerrainGeneratorRes>,
    faction_registry: Res<FactionRegistry>,
    chunk_query: Query<(&Chunk, &Transform)>,
    creature_query: CreatureQuery,
    item_query: ItemQuery,
    enemy_query: Query<(Entity, &SaveId, &Transform, &Enemy)>,
) {
    if !keyboard.just_pressed(KeyCode::F5) {
        return;
    }

    info!("Saving game to {}…", SAVE_PATH);

    // --- Terrain ---------------------------------------------------------
    let tc = terrain_gen.0.config();
    let terrain = TerrainConfigSave {
        seed: tc.seed,
        sea_level: tc.sea_level,
        height_scale: tc.height_scale,
        continent_freq: tc.continent_freq,
        detail_freq: tc.detail_freq,
        cave_freq: tc.cave_freq,
        cave_threshold: tc.cave_threshold,
        soil_depth: tc.soil_depth,
    };

    // --- Chunks ----------------------------------------------------------
    let chunks: Vec<ChunkSave> = chunk_query
        .iter()
        .map(|(chunk, _transform)| ChunkSave {
            coord: [chunk.coord.x, chunk.coord.y, chunk.coord.z],
            runs: encode_rle(chunk.voxels()),
        })
        .collect();

    // --- Build entity-bits → SaveId lookup (for relationship remapping) --
    let entity_to_save_id: HashMap<u64, u64> = creature_query
        .iter()
        .map(|(entity, save_id, ..)| (entity.to_bits(), save_id.0))
        .collect();

    // --- Creatures -------------------------------------------------------
    let creatures: Vec<CreatureSave> = creature_query
        .iter()
        .map(
            |(
                _entity,
                save_id,
                transform,
                creature,
                health,
                metabolism,
                growth,
                needs,
                relationships,
                physics,
                collider,
            )| {
                let t = transform.translation;
                let r = transform.rotation;

                let rel_entries: Vec<RelationshipEntry> = relationships
                    .map
                    .iter()
                    .filter_map(|(creature_id, rel)| {
                        let other_save_id = entity_to_save_id.get(&creature_id.0)?;
                        Some(RelationshipEntry {
                            other_id: *other_save_id,
                            trust: rel.trust,
                            familiarity: rel.familiarity,
                            hostility: rel.hostility,
                        })
                    })
                    .collect();

                CreatureSave {
                    id: save_id.0,
                    position: [t.x, t.y, t.z],
                    rotation: [r.x, r.y, r.z, r.w],
                    creature: creature.clone(),
                    health: health.clone(),
                    metabolism: metabolism.clone(),
                    growth: growth.clone(),
                    needs: needs.clone(),
                    relationships: rel_entries,
                    physics: PhysicsBodySave {
                        gravity_scale: physics.gravity_scale,
                        foot_offset: physics.foot_offset,
                    },
                    collider: ColliderSave {
                        half_extents: [
                            collider.half_extents.x,
                            collider.half_extents.y,
                            collider.half_extents.z,
                        ],
                    },
                }
            },
        )
        .collect();

    // --- Items -----------------------------------------------------------
    let items: Vec<ItemSave> = item_query
        .iter()
        .map(|(_entity, save_id, transform, item, physics, collider)| {
            let t = transform.translation;
            let r = transform.rotation;
            ItemSave {
                id: save_id.0,
                position: [t.x, t.y, t.z],
                rotation: [r.x, r.y, r.z, r.w],
                item: item.clone(),
                physics: PhysicsBodySave {
                    gravity_scale: physics.gravity_scale,
                    foot_offset: physics.foot_offset,
                },
                collider: ColliderSave {
                    half_extents: [
                        collider.half_extents.x,
                        collider.half_extents.y,
                        collider.half_extents.z,
                    ],
                },
            }
        })
        .collect();

    // --- Enemies ---------------------------------------------------------
    let enemies: Vec<EnemySave> = enemy_query
        .iter()
        .map(|(_entity, save_id, transform, enemy)| {
            let t = transform.translation;
            let r = transform.rotation;
            EnemySave {
                id: save_id.0,
                position: [t.x, t.y, t.z],
                rotation: [r.x, r.y, r.z, r.w],
                speed: enemy.speed,
            }
        })
        .collect();

    // --- Factions --------------------------------------------------------
    let factions_save: Vec<FactionSave> = faction_registry
        .factions
        .values()
        .map(|faction| {
            let member_ids: Vec<u64> = faction
                .members
                .iter()
                .filter_map(|cid| entity_to_save_id.get(&cid.0).copied())
                .collect();
            FactionSave {
                id: faction.id.0,
                name: faction.name.clone(),
                member_ids,
                territory: faction.territory.clone(),
                outsider_disposition: faction.outsider_disposition,
            }
        })
        .collect();

    let relations_save: Vec<FactionRelationEntry> = faction_registry
        .relations
        .iter()
        .map(|((fa, fb), rel)| FactionRelationEntry {
            faction_a: fa.0,
            faction_b: fb.0,
            standing: rel.standing,
        })
        .collect();

    let creature_factions: Vec<(u64, u32)> = faction_registry
        .creature_faction
        .iter()
        .filter_map(|(cid, fid)| {
            let save_id = entity_to_save_id.get(&cid.0)?;
            Some((*save_id, fid.0))
        })
        .collect();

    // --- Assemble & write ------------------------------------------------
    let save = SaveGame {
        version: SAVE_VERSION,
        terrain,
        chunks,
        creatures,
        items,
        enemies,
        factions: FactionRegistrySave {
            factions: factions_save,
            relations: relations_save,
            creature_factions,
        },
    };

    std::fs::create_dir_all("saves").ok();
    match ron::ser::to_string_pretty(&save, ron::ser::PrettyConfig::default()) {
        Ok(text) => match std::fs::write(SAVE_PATH, text) {
            Ok(()) => info!(
                "Game saved ({} chunks, {} creatures, {} items, {} enemies).",
                save.chunks.len(),
                save.creatures.len(),
                save.items.len(),
                save.enemies.len()
            ),
            Err(e) => error!("Failed to write save file: {e}"),
        },
        Err(e) => error!("Failed to serialise save: {e}"),
    }
}
