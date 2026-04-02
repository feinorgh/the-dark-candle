// V2 chunk manager for cubed-sphere coordinates.
//
// Computes desired chunk set on the cubed sphere around the camera,
// dispatches async terrain generation + greedy meshing, collects results,
// and spawns entities with correct local-tangent-plane Transforms.

use std::collections::HashSet;
use std::sync::Arc;

use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, poll_once};

use crate::camera::FpsCamera;
use crate::world::chunk::CHUNK_SIZE;
use crate::world::lod::MaterialColorMap;
use crate::world::meshing::{ChunkMesh, ChunkMeshMarker, chunk_mesh_to_bevy_mesh};
use crate::world::planet::PlanetConfig;
use crate::world::terrain::SphericalTerrainGenerator;
use crate::world::v2::cubed_sphere::{CubeSphereCoord, world_pos_to_coord};
use crate::world::v2::greedy_mesh::{self, NeighborSlices};
use crate::world::v2::terrain_gen::generate_v2_chunk;

// ── Limits ────────────────────────────────────────────────────────────────

const MAX_DISPATCHES_PER_FRAME: usize = 16;
const MAX_PENDING: usize = 48;
const MAX_COLLECTS_PER_FRAME: usize = 16;

// ── Resources ─────────────────────────────────────────────────────────────

/// Thread-safe handle to the spherical terrain generator for V2 async tasks.
#[derive(Resource, Clone)]
pub struct V2TerrainGen(pub Arc<SphericalTerrainGenerator>);

/// Chunk load radius in chunk units for the V2 pipeline.
#[derive(Resource)]
pub struct V2LoadRadius {
    pub horizontal: i32,
    pub vertical: i32,
}

impl Default for V2LoadRadius {
    fn default() -> Self {
        Self {
            horizontal: 4,
            vertical: 2,
        }
    }
}

/// Tracks which CubeSphereCoords are currently loaded (entity spawned).
#[derive(Resource, Default)]
pub struct V2ChunkMap {
    loaded: HashSet<CubeSphereCoord>,
}

/// Tracks which CubeSphereCoords are pending (task dispatched, not yet collected).
#[derive(Resource, Default)]
pub struct V2PendingChunks {
    pending: HashSet<CubeSphereCoord>,
}

// ── Components ────────────────────────────────────────────────────────────

/// Marks a chunk entity as belonging to the V2 pipeline.
#[derive(Component)]
pub struct V2ChunkMarker;

/// Stores the cubed-sphere coordinate on a chunk entity.
#[derive(Component, Clone, Copy)]
pub struct V2ChunkCoord(pub CubeSphereCoord);

/// Async task that produces a completed V2 chunk (voxels + mesh).
#[derive(Component)]
pub struct V2ChunkTask(pub Task<V2ChunkResult>);

/// Result of a combined terrain generation + meshing task.
pub struct V2ChunkResult {
    pub coord: CubeSphereCoord,
    pub mesh: ChunkMesh,
}

// ── Systems ───────────────────────────────────────────────────────────────

/// Compute the set of desired chunks around the camera on the cubed sphere.
fn desired_chunks_v2(
    cam_pos: Vec3,
    planet: &PlanetConfig,
    radius: &V2LoadRadius,
) -> HashSet<CubeSphereCoord> {
    let fce = CubeSphereCoord::face_chunks_per_edge(planet.mean_radius);
    let cam_coord = world_pos_to_coord(cam_pos.as_dvec3(), planet.mean_radius, fce);

    let h = radius.horizontal;
    let v = radius.vertical;
    let max_uv = fce as i32;
    let mut set = HashSet::new();

    for du in -h..=h {
        for dv in -h..=h {
            for dl in -v..=v {
                let u = cam_coord.u + du;
                let vi = cam_coord.v + dv;
                let layer = cam_coord.layer + dl;
                let face = cam_coord.face;

                // Clamp u, v to valid face range. Cross-face wrapping is
                // handled by the neighbor system for meshing; for chunk loading
                // we simply skip out-of-range coords for simplicity.
                if u < 0 || u >= max_uv || vi < 0 || vi >= max_uv {
                    continue;
                }

                set.insert(CubeSphereCoord::new(face, u, vi, layer));
            }
        }
    }

    set
}

/// Main V2 chunk update system: compute desired set, dispatch generation,
/// despawn out-of-range chunks.
#[allow(clippy::too_many_arguments)]
pub fn v2_update_chunks(
    mut commands: Commands,
    mut chunk_map: ResMut<V2ChunkMap>,
    mut pending: ResMut<V2PendingChunks>,
    load_radius: Res<V2LoadRadius>,
    terrain_gen: Res<V2TerrainGen>,
    planet: Res<PlanetConfig>,
    color_map: Res<MaterialColorMap>,
    camera_q: Query<&Transform, With<FpsCamera>>,
    v2_chunks_q: Query<(Entity, &V2ChunkCoord), With<V2ChunkMarker>>,
) {
    let Ok(cam_transform) = camera_q.single() else {
        return;
    };

    let desired = desired_chunks_v2(cam_transform.translation, &planet, &load_radius);

    // Despawn chunks no longer desired
    for (entity, coord) in &v2_chunks_q {
        if !desired.contains(&coord.0) && !pending.pending.contains(&coord.0) {
            commands.entity(entity).despawn();
            chunk_map.loaded.remove(&coord.0);
        }
    }

    // Dispatch new chunks
    let pool = AsyncComputeTaskPool::get();
    let budget = MAX_PENDING.saturating_sub(pending.pending.len());
    let mut dispatched = 0usize;

    let mean_radius = planet.mean_radius;
    let fce = CubeSphereCoord::face_chunks_per_edge(mean_radius);

    for coord in &desired {
        if dispatched >= MAX_DISPATCHES_PER_FRAME.min(budget) {
            break;
        }
        if chunk_map.loaded.contains(coord) || pending.pending.contains(coord) {
            continue;
        }

        let coord = *coord;
        let tgen = terrain_gen.0.clone();
        let cmap = color_map.clone();

        let task = pool.spawn(async move {
            let data = generate_v2_chunk(coord, mean_radius, fce, &tgen);
            let mesh = greedy_mesh::greedy_mesh(&data.voxels, &NeighborSlices::empty(), &cmap);
            V2ChunkResult { coord, mesh }
        });

        commands.spawn(V2ChunkTask(task));
        pending.pending.insert(coord);
        dispatched += 1;
    }
}

/// Collect completed V2 chunk tasks and spawn renderable entities.
pub fn v2_collect_results(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut chunk_map: ResMut<V2ChunkMap>,
    mut pending: ResMut<V2PendingChunks>,
    planet: Res<PlanetConfig>,
    mut task_q: Query<(Entity, &mut V2ChunkTask)>,
) {
    let fce = CubeSphereCoord::face_chunks_per_edge(planet.mean_radius);
    let cs_half = Vec3::splat(CHUNK_SIZE as f32 / 2.0);

    let chunk_material = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        ..default()
    });

    let mut collected = 0usize;
    for (task_entity, mut task) in &mut task_q {
        if collected >= MAX_COLLECTS_PER_FRAME {
            break;
        }
        let Some(result) = block_on(poll_once(&mut task.0)) else {
            continue;
        };
        collected += 1;

        // Remove the task entity
        commands.entity(task_entity).despawn();
        pending.pending.remove(&result.coord);

        if result.mesh.is_empty() {
            // No visible geometry — record as loaded but don't spawn a mesh entity
            chunk_map.loaded.insert(result.coord);
            // Spawn a minimal entity so we track it for despawn
            let (center, rotation) = result.coord.world_transform(planet.mean_radius, fce);
            let adjusted = center - rotation * cs_half;
            commands.spawn((
                V2ChunkMarker,
                V2ChunkCoord(result.coord),
                Transform::from_translation(adjusted).with_rotation(rotation),
            ));
            continue;
        }

        // Build Bevy mesh and spawn entity
        let bevy_mesh = chunk_mesh_to_bevy_mesh(result.mesh);
        let mesh_handle = meshes.add(bevy_mesh);

        let (center, rotation) = result.coord.world_transform(planet.mean_radius, fce);
        // Offset translation: mesh vertices are in [0, CS], so local origin (0,0,0)
        // should map to the chunk's "base corner" in world space.
        let adjusted = center - rotation * cs_half;

        commands.spawn((
            V2ChunkMarker,
            V2ChunkCoord(result.coord),
            ChunkMeshMarker,
            Mesh3d(mesh_handle),
            MeshMaterial3d(chunk_material.clone()),
            Transform::from_translation(adjusted).with_rotation(rotation),
        ));

        chunk_map.loaded.insert(result.coord);
    }
}

// ── Plugin ────────────────────────────────────────────────────────────────

/// Plugin for the V2 cubed-sphere rendering pipeline.
pub struct V2WorldPlugin;

impl Plugin for V2WorldPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<V2LoadRadius>()
            .init_resource::<V2ChunkMap>()
            .init_resource::<V2PendingChunks>()
            .add_systems(Startup, v2_init_terrain_gen)
            .add_systems(
                Update,
                (
                    v2_update_chunks,
                    v2_collect_results,
                )
                    .chain()
                    .run_if(resource_exists::<V2TerrainGen>)
                    .run_if(not(in_state(crate::game_state::GameState::WorldCreation))),
            );
    }
}

/// Extract the spherical terrain generator from SharedTerrainGen and wrap it
/// as a V2TerrainGen resource.
fn v2_init_terrain_gen(
    mut commands: Commands,
    shared: Option<Res<crate::world::chunk_manager::SharedTerrainGen>>,
    planet: Res<PlanetConfig>,
) {
    if let Some(shared) = shared
        && shared.0.spherical().is_some()
    {
        let tgen = SphericalTerrainGenerator::new(planet.clone());
        commands.insert_resource(V2TerrainGen(Arc::new(tgen)));
        info!("V2 pipeline: initialized terrain generator (spherical mode)");
        return;
    }
    // Fallback: create from PlanetConfig directly
    let tgen = SphericalTerrainGenerator::new(planet.clone());
    commands.insert_resource(V2TerrainGen(Arc::new(tgen)));
    info!("V2 pipeline: initialized terrain generator from PlanetConfig");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::v2::cubed_sphere::CubeSphereCoord;

    #[test]
    fn desired_chunks_contains_camera_position() {
        let mut cfg = PlanetConfig::default();
        cfg.mean_radius = 32000.0;
        let radius = V2LoadRadius::default();
        let fce = CubeSphereCoord::face_chunks_per_edge(cfg.mean_radius);

        // Camera at the PosX face center, at the surface
        let cam_pos = Vec3::new(cfg.mean_radius as f32, 0.0, 0.0);
        let desired = desired_chunks_v2(cam_pos, &cfg, &radius);

        assert!(!desired.is_empty(), "Desired set should not be empty");

        // The camera's own coord should be in the set
        let cam_coord = world_pos_to_coord(cam_pos.as_dvec3(), cfg.mean_radius, fce);
        assert!(
            desired.contains(&cam_coord),
            "Camera's own chunk should be desired"
        );
    }

    #[test]
    fn desired_chunks_count_matches_radius() {
        let mut cfg = PlanetConfig::default();
        cfg.mean_radius = 32000.0;
        let radius = V2LoadRadius {
            horizontal: 2,
            vertical: 1,
        };

        let cam_pos = Vec3::new(cfg.mean_radius as f32, 0.0, 0.0);
        let desired = desired_chunks_v2(cam_pos, &cfg, &radius);

        // Expected: (2h+1)² × (2v+1) = 5² × 3 = 75 chunks max
        // May be less due to edge clamping
        assert!(
            desired.len() <= 75,
            "Should not exceed expected count, got {}",
            desired.len()
        );
        assert!(
            desired.len() > 10,
            "Should have a reasonable number of chunks, got {}",
            desired.len()
        );
    }
}
