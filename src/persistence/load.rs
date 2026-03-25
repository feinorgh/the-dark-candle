// Load system: restore game state from `saves/save.ron`.
//
// Triggered by pressing F9. Despawns all persistent world entities, then
// reconstructs them from the save document. The terrain generator is
// re-seeded from the saved config so newly-loaded (out-of-range) chunks
// continue to match the saved world.

use std::collections::HashMap;

use bevy::prelude::*;

use crate::{
    entities::Enemy,
    physics::{collision::Collider, gravity::PhysicsBody},
    procgen::{creatures::Creature, items::Item},
    social::{
        factions::{Faction, FactionId, FactionRegistry, FactionRelation},
        relationships::{CreatureId, Relationship, Relationships},
    },
    world::{
        chunk::{Chunk, ChunkCoord},
        chunk_manager::{ChunkMap, TerrainGeneratorRes},
        terrain::{TerrainConfig, TerrainGenerator, UnifiedTerrainGenerator},
    },
};

use super::types::{SAVE_PATH, SAVE_VERSION, SaveId, decode_rle};

/// Migrate a save from an older version to the current format.
/// Returns true if migration was applied, false if already current.
fn migrate_save(save: &mut super::types::SaveGame) -> bool {
    if save.version >= SAVE_VERSION {
        return false;
    }

    // v1 → v2: pressure atm→Pa, temperature 293→288.15 default
    if save.version == 1 {
        info!("Migrating save v1 → v2: pressure atm→Pa, temperature to standard atmosphere");
        for chunk in &mut save.chunks {
            for run in &mut chunk.runs {
                // Convert pressure from atmospheres to Pascals
                run.pressure *= 101_325.0;
                // Adjust default temperature (293 K → 288.15 K for ambient voxels)
                if (run.temperature - 293.0).abs() < 0.01 {
                    run.temperature = 288.15;
                }
            }
        }
        save.version = 2;
    }

    true
}

// ---------------------------------------------------------------------------
// Load system
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub fn load_game(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut commands: Commands,
    mut chunk_map: ResMut<ChunkMap>,
    mut terrain_gen: ResMut<TerrainGeneratorRes>,
    mut faction_registry: ResMut<FactionRegistry>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    chunk_query: Query<Entity, With<Chunk>>,
    creature_query: Query<Entity, With<Creature>>,
    item_query: Query<Entity, With<Item>>,
    enemy_query: Query<Entity, With<Enemy>>,
) {
    if !keyboard.just_pressed(KeyCode::F9) {
        return;
    }

    let text = match std::fs::read_to_string(SAVE_PATH) {
        Ok(t) => t,
        Err(e) => {
            warn!("No save file found at {SAVE_PATH}: {e}");
            return;
        }
    };

    let mut save: super::types::SaveGame = match ron::from_str(&text) {
        Ok(s) => s,
        Err(e) => {
            error!("Failed to parse save file: {e}");
            return;
        }
    };

    if migrate_save(&mut save) {
        info!("Save file migrated to v{}", save.version);
    }

    info!(
        "Loading save (v{}) — {} chunks, {} creatures, {} items, {} enemies…",
        save.version,
        save.chunks.len(),
        save.creatures.len(),
        save.items.len(),
        save.enemies.len()
    );

    // --- Despawn all persistent entities ---------------------------------
    for e in &chunk_query {
        commands.entity(e).despawn();
    }
    for e in &creature_query {
        commands.entity(e).despawn();
    }
    for e in &item_query {
        commands.entity(e).despawn();
    }
    for e in &enemy_query {
        commands.entity(e).despawn();
    }
    chunk_map.clear();
    *faction_registry = FactionRegistry::default();

    // --- Restore terrain generator ---------------------------------------
    let tc = &save.terrain;
    terrain_gen.0 = UnifiedTerrainGenerator::Flat(TerrainGenerator::new(TerrainConfig {
        seed: tc.seed,
        sea_level: tc.sea_level,
        height_scale: tc.height_scale,
        continent_freq: tc.continent_freq,
        detail_freq: tc.detail_freq,
        cave_freq: tc.cave_freq,
        cave_threshold: tc.cave_threshold,
        soil_depth: tc.soil_depth,
    }));

    // --- Restore chunks --------------------------------------------------
    for cs in &save.chunks {
        let coord = ChunkCoord::new(cs.coord[0], cs.coord[1], cs.coord[2]);
        let voxels = decode_rle(&cs.runs);

        let mut chunk = Chunk::new_empty(coord);
        {
            let dst = chunk.voxels_mut();
            let len = dst.len().min(voxels.len());
            dst[..len].copy_from_slice(&voxels[..len]);
        }

        let origin = coord.world_origin();
        let entity = commands
            .spawn((
                chunk,
                coord,
                Transform::from_xyz(origin.x as f32, origin.y as f32, origin.z as f32),
            ))
            .id();
        chunk_map.insert(coord, entity);
    }

    // --- Restore creatures (phase 1: spawn, record SaveId → Entity) ------
    let mut save_id_to_entity: HashMap<u64, Entity> = HashMap::new();

    for cs in &save.creatures {
        let pos = Vec3::new(cs.position[0], cs.position[1], cs.position[2]);
        let rot = Quat::from_xyzw(
            cs.rotation[0],
            cs.rotation[1],
            cs.rotation[2],
            cs.rotation[3],
        );

        let entity = commands
            .spawn((
                SaveId(cs.id),
                Transform::from_translation(pos).with_rotation(rot),
                cs.creature.clone(),
                cs.health.clone(),
                cs.metabolism.clone(),
                cs.growth.clone(),
                cs.needs.clone(),
                PhysicsBody {
                    velocity: Vec3::ZERO,
                    grounded: false,
                    gravity_scale: cs.physics.gravity_scale,
                    foot_offset: cs.physics.foot_offset,
                },
                Collider::new(
                    cs.collider.half_extents[0] * 2.0,
                    cs.collider.half_extents[1] * 2.0,
                    cs.collider.half_extents[2] * 2.0,
                ),
                // Relationships are set in phase 2 after all entities are known.
                Relationships::default(),
            ))
            .id();
        save_id_to_entity.insert(cs.id, entity);
    }

    // --- Restore creatures (phase 2: remap relationships) ----------------
    let save_id_to_creature_id: HashMap<u64, CreatureId> = save_id_to_entity
        .iter()
        .map(|(&sid, &e)| (sid, CreatureId(e.to_bits())))
        .collect();

    for cs in &save.creatures {
        let Some(&entity) = save_id_to_entity.get(&cs.id) else {
            continue;
        };
        let mut rels = Relationships::default();
        for entry in &cs.relationships {
            if let Some(&cid) = save_id_to_creature_id.get(&entry.other_id) {
                rels.map.insert(
                    cid,
                    Relationship {
                        trust: entry.trust,
                        familiarity: entry.familiarity,
                        hostility: entry.hostility,
                    },
                );
            }
        }
        commands.entity(entity).insert(rels);
    }

    // --- Restore items ---------------------------------------------------
    for is in &save.items {
        let pos = Vec3::new(is.position[0], is.position[1], is.position[2]);
        let rot = Quat::from_xyzw(
            is.rotation[0],
            is.rotation[1],
            is.rotation[2],
            is.rotation[3],
        );
        commands.spawn((
            SaveId(is.id),
            Transform::from_translation(pos).with_rotation(rot),
            is.item.clone(),
            PhysicsBody {
                velocity: Vec3::ZERO,
                grounded: false,
                gravity_scale: is.physics.gravity_scale,
                foot_offset: is.physics.foot_offset,
            },
            Collider::new(
                is.collider.half_extents[0] * 2.0,
                is.collider.half_extents[1] * 2.0,
                is.collider.half_extents[2] * 2.0,
            ),
        ));
    }

    // --- Restore enemies -------------------------------------------------
    let enemy_mesh = meshes.add(Cuboid::new(0.8, 1.2, 0.8));
    let enemy_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.8, 0.2, 0.2),
        ..default()
    });

    for es in &save.enemies {
        let pos = Vec3::new(es.position[0], es.position[1], es.position[2]);
        let rot = Quat::from_xyzw(
            es.rotation[0],
            es.rotation[1],
            es.rotation[2],
            es.rotation[3],
        );
        commands.spawn((
            SaveId(es.id),
            Enemy { speed: es.speed },
            Mesh3d(enemy_mesh.clone()),
            MeshMaterial3d(enemy_material.clone()),
            Transform::from_translation(pos).with_rotation(rot),
        ));
    }

    // --- Restore faction registry ----------------------------------------
    for fs in &save.factions.factions {
        let mut faction = Faction::new(FactionId(fs.id), fs.name.clone());
        faction.territory = fs.territory.clone();
        faction.outsider_disposition = fs.outsider_disposition;
        for &member_save_id in &fs.member_ids {
            if let Some(&entity) = save_id_to_entity.get(&member_save_id) {
                faction.members.insert(CreatureId(entity.to_bits()));
            }
        }
        faction_registry.factions.insert(FactionId(fs.id), faction);
    }

    for rel_entry in &save.factions.relations {
        let fa = FactionId(rel_entry.faction_a);
        let fb = FactionId(rel_entry.faction_b);
        faction_registry.relations.insert(
            (fa, fb),
            FactionRelation {
                standing: rel_entry.standing,
            },
        );
    }

    for &(member_save_id, faction_id) in &save.factions.creature_factions {
        if let Some(&entity) = save_id_to_entity.get(&member_save_id) {
            faction_registry
                .creature_faction
                .insert(CreatureId(entity.to_bits()), FactionId(faction_id));
        }
    }

    info!("Load complete.");
}
