pub mod biomes;
pub mod creatures;
pub mod items;
pub mod props;
pub mod spawning;
pub mod tree;

use bevy::prelude::*;

use crate::world::chunk::{CHUNK_SIZE, Chunk, ChunkCoord};

pub struct ProcgenPlugin;

impl Plugin for ProcgenPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(biomes::BiomePlugin);

        match props::load_prop_registry() {
            Ok(registry) => {
                info!("Loaded PropRegistry with {} props", registry.len());
                app.insert_resource(registry);
            }
            Err(e) => {
                warn!("Failed to load PropRegistry: {e}");
                app.init_resource::<props::PropRegistry>();
            }
        }

        match tree::load_tree_registry() {
            Ok(registry) => {
                info!("Loaded TreeRegistry with {} trees", registry.len());
                app.insert_resource(registry);
            }
            Err(e) => {
                warn!("Failed to load TreeRegistry: {e}");
                app.init_resource::<tree::TreeRegistry>();
            }
        }

        match creatures::load_creature_registry() {
            Ok(registry) => {
                info!("Loaded CreatureRegistry with {} creatures", registry.len());
                app.insert_resource(registry);
            }
            Err(e) => {
                warn!("Failed to load CreatureRegistry: {e}");
                app.init_resource::<creatures::CreatureRegistry>();
            }
        }

        // Run after ChunkManagement so chunk despawn commands are flushed
        // before we try to decorate or access chunk entities.
        // plant_trees runs first (modifies chunk voxels), then decorate_chunks
        // (removes NeedsDecoration marker and spawns prop entities), then
        // spawn_creatures (spawns creature entities at surface positions).
        app.add_systems(
            Update,
            (tree::plant_trees, props::decorate_chunks, spawn_creatures)
                .chain()
                .after(crate::world::WorldSet::ChunkManagement),
        );
    }
}

/// System: spawns creature entities on newly loaded chunks.
///
/// Queries chunks with `NeedsCreatureSpawning`, determines the biome,
/// plans creature spawns via `plan_chunk_spawns()`, generates unique
/// instances via `generate_creature()`, and spawns full entities with
/// biology, behavior, and social components attached.
fn spawn_creatures(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    creature_registry: Res<creatures::CreatureRegistry>,
    biome_assets: Res<Assets<biomes::BiomeData>>,
    mut to_spawn: Query<
        (Entity, &Chunk, &ChunkCoord, &mut creatures::ChunkCreatures),
        With<creatures::NeedsCreatureSpawning>,
    >,
) {
    use creatures::generate_creature;
    use props::{is_valid_surface, surface_height};
    use spawning::plan_chunk_spawns;

    if creature_registry.is_empty() {
        // Still remove markers so we don't re-query every frame.
        for (entity, _, _, _) in &to_spawn {
            commands
                .entity(entity)
                .remove::<creatures::NeedsCreatureSpawning>();
        }
        return;
    }

    let biomes: Vec<&biomes::BiomeData> = biome_assets.iter().map(|(_, b)| b).collect();
    if biomes.is_empty() {
        return;
    }

    for (chunk_entity, chunk, coord, mut chunk_creatures) in &mut to_spawn {
        commands
            .entity(chunk_entity)
            .remove::<creatures::NeedsCreatureSpawning>();

        // Determine biome from chunk center height (same heuristic as tree/prop systems).
        let center_height = surface_height(chunk, CHUNK_SIZE / 2, CHUNK_SIZE / 2).unwrap_or(0);
        let world_y = coord.y as f32 * CHUNK_SIZE as f32 + center_height as f32;
        let biome = biomes
            .iter()
            .find(|b| world_y >= b.height_range.0 && world_y <= b.height_range.1)
            .or(biomes.first());
        let Some(biome) = biome else { continue };

        if biome.creature_spawns.is_empty() {
            continue;
        }

        let spawns = plan_chunk_spawns(biome, coord.x, coord.z, CHUNK_SIZE, 42);
        let origin = coord.world_origin();

        for (species_id, local_x, local_z, seed) in spawns {
            let Some(template) = creature_registry.get(&species_id) else {
                continue;
            };

            let ix = (local_x as usize).min(CHUNK_SIZE - 1);
            let iz = (local_z as usize).min(CHUNK_SIZE - 1);

            let Some(sy) = surface_height(chunk, ix, iz) else {
                continue;
            };

            if !is_valid_surface(chunk, ix, sy, iz) {
                continue;
            }

            let creature = generate_creature(template, seed);
            let half_y = template.hitbox.1 * 0.5;

            let world_pos = Vec3::new(
                origin.x as f32 + local_x,
                origin.y as f32 + (sy + 1) as f32 + half_y,
                origin.z as f32 + local_z,
            );

            let color = Color::srgb(creature.color[0], creature.color[1], creature.color[2]);
            let (hx, hy, hz) = template.hitbox;

            let creature_entity = commands
                .spawn((
                    creature,
                    crate::biology::health::Health::new(template.base_health),
                    crate::biology::growth::Growth::new(template.lifespan),
                    crate::biology::metabolism::Metabolism::for_body_size(template.body_size),
                    crate::behavior::needs::Needs::default(),
                    crate::behavior::perception::Senses::default(),
                    crate::behavior::CurrentAction::default(),
                    crate::social::relationships::Relationships::default(),
                    Mesh3d(meshes.add(Cuboid::new(hx, hy, hz))),
                    MeshMaterial3d(materials.add(StandardMaterial {
                        base_color: color,
                        ..default()
                    })),
                    Transform::from_translation(world_pos),
                ))
                .id();

            chunk_creatures.entities.push(creature_entity);
        }
    }
}
