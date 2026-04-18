// V2 chunk manager for cubed-sphere coordinates.
//
// Two-stage async pipeline:
// Stage 1 (terrain): generate voxels → store in voxel cache
// Stage 2 (meshing): build mesh from cached voxels + neighbor boundaries
//
// Boundary slices are extracted from the voxel cache for same-face neighbors.
// Cross-face boundaries fall back to terrain resampling.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use bevy::math::DVec3;
use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, poll_once};

use crate::camera::FpsCamera;
use crate::floating_origin::{RenderOrigin, RenderOriginShift, WorldPosition};
use crate::world::chunk::CHUNK_SIZE;
use crate::world::chunk_manager::SharedTerrainGen;
use crate::world::lod::MaterialColorMap;
use crate::world::meshing::{ChunkMesh, ChunkMeshMarker, chunk_mesh_to_bevy_mesh};
use crate::world::planet::PlanetConfig;
use crate::world::terrain::UnifiedTerrainGenerator;
use crate::world::v2::cubed_sphere::{CubeFace, CubeSphereCoord, world_pos_to_coord};
use crate::world::v2::greedy_mesh;
use crate::world::v2::greedy_mesh::NeighborSlices;
use crate::world::v2::terrain_gen::{
    CachedVoxels, V2TerrainData, cached_voxels_to_vec, extract_edge_slice,
    generate_single_boundary_slice, generate_v2_voxels,
};

// ── Limits ────────────────────────────────────────────────────────────────

const MAX_TERRAIN_DISPATCHES_PER_FRAME: usize = 32;
const MAX_TERRAIN_PENDING: usize = 128;
const MAX_TERRAIN_COLLECTS_PER_FRAME: usize = 32;

const MAX_MESH_DISPATCHES_PER_FRAME: usize = 16;
const MAX_MESH_PENDING: usize = 64;
const MAX_MESH_COLLECTS_PER_FRAME: usize = 16;

/// Maximum horizontal load radius in chunks (prevents runaway at extreme altitudes).
const MAX_HORIZONTAL_CHUNKS: i32 = 48;
/// Maximum vertical load extent in layers.
const MAX_VERTICAL_LAYERS: i32 = 64;

// ── Resources ─────────────────────────────────────────────────────────────

/// Thread-safe handle to the terrain generator for V2 async tasks.
///
/// Wraps `Arc<UnifiedTerrainGenerator>` so that both the plain spherical
/// generator and the planetary (tectonic/biome-aware) generator can be used
/// without changing the dispatch path.
#[derive(Resource, Clone)]
pub struct V2TerrainGen(pub Arc<UnifiedTerrainGenerator>);

/// Chunk load radius in chunk units for the V2 pipeline.
#[derive(Resource)]
pub struct V2LoadRadius {
    pub horizontal: i32,
    pub vertical: i32,
}

impl Default for V2LoadRadius {
    fn default() -> Self {
        Self {
            horizontal: 12,
            vertical: 2,
        }
    }
}

/// Tracks which CubeSphereCoords are currently loaded (entity spawned).
#[derive(Resource, Default)]
pub struct V2ChunkMap {
    loaded: HashSet<CubeSphereCoord>,
}

impl V2ChunkMap {
    pub fn loaded_count(&self) -> usize {
        self.loaded.len()
    }
}

/// Tracks which CubeSphereCoords have a terrain task dispatched.
#[derive(Resource, Default)]
pub struct V2PendingTerrain {
    pending: HashSet<CubeSphereCoord>,
}

impl V2PendingTerrain {
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }
}

/// Tracks which CubeSphereCoords have a mesh task dispatched.
#[derive(Resource, Default)]
pub struct V2PendingMeshes {
    pending: HashSet<CubeSphereCoord>,
}

impl V2PendingMeshes {
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }
}

/// Cached voxel data for chunks that have completed terrain generation.
///
/// Entries are stored as `Arc`-wrapped data for zero-copy sharing with mesh tasks.
/// Chunks are evicted when they leave the desired set + margin.
#[derive(Resource, Default)]
pub struct V2VoxelCache {
    entries: HashMap<CubeSphereCoord, CachedVoxels>,
    byte_size: usize,
}

impl V2VoxelCache {
    pub fn get(&self, coord: &CubeSphereCoord) -> Option<&CachedVoxels> {
        self.entries.get(coord)
    }

    pub fn contains(&self, coord: &CubeSphereCoord) -> bool {
        self.entries.contains_key(coord)
    }

    pub fn insert(&mut self, coord: CubeSphereCoord, voxels: CachedVoxels) {
        self.byte_size += voxels.byte_size();
        if let Some(old) = self.entries.insert(coord, voxels) {
            self.byte_size -= old.byte_size();
        }
    }

    pub fn remove(&mut self, coord: &CubeSphereCoord) {
        if let Some(old) = self.entries.remove(coord) {
            self.byte_size -= old.byte_size();
        }
    }

    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    pub fn byte_size(&self) -> usize {
        self.byte_size
    }
}

// ── Components ────────────────────────────────────────────────────────────

/// Marks a chunk entity as belonging to the V2 pipeline.
#[derive(Component)]
pub struct V2ChunkMarker;

/// Stores the cubed-sphere coordinate on a chunk entity.
#[derive(Component, Clone, Copy)]
pub struct V2ChunkCoord(pub CubeSphereCoord);

/// Async terrain generation task (stage 1).
#[derive(Component)]
pub struct V2TerrainTask(pub Task<V2TerrainData>);

/// Async mesh generation task (stage 2).
#[derive(Component)]
pub struct V2MeshTask(pub Task<V2MeshResult>);

/// Result of a mesh generation task.
pub struct V2MeshResult {
    pub coord: CubeSphereCoord,
    pub mesh: ChunkMesh,
}

// ── Systems ───────────────────────────────────────────────────────────────

/// Compute horizontal load radius in chunks from camera altitude using the
/// geometric horizon distance: `d = sqrt(2·R·h + h²)`.
fn horizon_load_radius(altitude_m: f64, mean_radius: f64, base_horizontal: i32) -> i32 {
    if altitude_m <= 0.0 {
        return base_horizontal;
    }
    let r = mean_radius;
    let h = altitude_m;
    let horizon_dist = (2.0 * r * h + h * h).sqrt();
    let cs = CHUNK_SIZE as f64;
    let radius_chunks = (horizon_dist / cs).ceil() as i32;
    radius_chunks.clamp(base_horizontal, MAX_HORIZONTAL_CHUNKS)
}

/// Compute the set of desired chunks around the camera on the cubed sphere.
///
/// Horizontal radius scales with altitude via the horizon distance formula.
/// Vertical range always extends from the surface layer (0) to the camera layer,
/// plus the base vertical padding in both directions.
fn desired_chunks_v2(
    cam_world_pos: DVec3,
    planet: &PlanetConfig,
    radius: &V2LoadRadius,
) -> HashSet<CubeSphereCoord> {
    let fce = CubeSphereCoord::face_chunks_per_edge(planet.mean_radius);
    let cam_coord = world_pos_to_coord(cam_world_pos, planet.mean_radius, fce);

    // Altitude above mean surface
    let altitude_m = cam_world_pos.length() - planet.mean_radius;

    let h = horizon_load_radius(altitude_m, planet.mean_radius, radius.horizontal);
    let max_uv = fce as i32;
    let mut set = HashSet::new();

    // Vertical range: always include surface (layer 0) and extend to camera.
    // The base vertical padding extends above/below this range.
    let cam_layer = cam_coord.layer;
    let layer_lo = (0.min(cam_layer) - radius.vertical).max(-MAX_VERTICAL_LAYERS);
    let layer_hi = (0.max(cam_layer) + radius.vertical).min(MAX_VERTICAL_LAYERS);

    for du in -h..=h {
        for dv in -h..=h {
            for layer in layer_lo..=layer_hi {
                let u = cam_coord.u + du;
                let vi = cam_coord.v + dv;
                let face = cam_coord.face;

                // Use cross-face wrapping for out-of-range u/v coords
                // so terrain loads seamlessly across cube faces.
                if u < 0 || u >= max_uv || vi < 0 || vi >= max_uv {
                    // Resolve via the neighbor wrapping system on the camera coord.
                    // Build the target coord on the camera's face and let the
                    // cross-face lookup translate it.
                    if let Some(wrapped) =
                        wrap_cross_face(face, u, vi, layer, max_uv)
                    {
                        set.insert(wrapped);
                    }
                    continue;
                }

                set.insert(CubeSphereCoord::new(face, u, vi, layer));
            }
        }
    }

    set
}

/// Resolve an out-of-range (u, v) on `face` to the correct cross-face coord.
/// Returns `None` if the coord is doubly out-of-range (corner wrap — skip).
fn wrap_cross_face(
    face: CubeFace,
    u: i32,
    v: i32,
    layer: i32,
    max_uv: i32,
) -> Option<CubeSphereCoord> {

    // Only handle single-axis overflow (edge); skip corners where both are OOB.
    let u_oob = u < 0 || u >= max_uv;
    let v_oob = v < 0 || v >= max_uv;
    if u_oob && v_oob {
        return None; // Corner wrap — too complex, skip
    }

    // Build a temporary coord on the face at clamped position, then use
    // the neighbor system to find the actual cross-face coord.
    if u_oob {
        let clamped_u = if u < 0 { 0 } else { max_uv - 1 };
        let base = CubeSphereCoord::new(face, clamped_u, v, layer);
        let delta = if u < 0 { -(clamped_u - u) } else { u - clamped_u };
        // Walk across the face boundary
        let neighbor = if u < 0 {
            base.neighbors(max_uv)[1] // -U neighbor
        } else {
            base.neighbors(max_uv)[0] // +U neighbor
        };
        // For deeper steps, we'd need recursive walking. For now, only load
        // the immediate cross-face layer (1 chunk deep across boundary).
        if delta.abs() <= 1 {
            return Some(CubeSphereCoord::new(
                neighbor.face, neighbor.u, neighbor.v, layer,
            ));
        }
        return None;
    }

    if v_oob {
        let clamped_v = if v < 0 { 0 } else { max_uv - 1 };
        let base = CubeSphereCoord::new(face, u, clamped_v, layer);
        let delta = if v < 0 { -(clamped_v - v) } else { v - clamped_v };
        let neighbor = if v < 0 {
            base.neighbors(max_uv)[3] // -V neighbor
        } else {
            base.neighbors(max_uv)[2] // +V neighbor
        };
        if delta.abs() <= 1 {
            return Some(CubeSphereCoord::new(
                neighbor.face, neighbor.u, neighbor.v, layer,
            ));
        }
        return None;
    }

    None
}

/// Main V2 chunk update system: compute desired set, dispatch terrain tasks,
/// despawn out-of-range chunks, evict stale cache entries.
#[allow(clippy::too_many_arguments)]
pub fn v2_update_chunks(
    mut commands: Commands,
    mut chunk_map: ResMut<V2ChunkMap>,
    mut pending_terrain: ResMut<V2PendingTerrain>,
    pending_meshes: Res<V2PendingMeshes>,
    mut cache: ResMut<V2VoxelCache>,
    load_radius: Res<V2LoadRadius>,
    terrain_gen: Res<V2TerrainGen>,
    planet: Res<PlanetConfig>,
    camera_q: Query<&WorldPosition, With<FpsCamera>>,
    v2_chunks_q: Query<(Entity, &V2ChunkCoord), With<V2ChunkMarker>>,
) {
    let Ok(cam_world_pos) = camera_q.single() else {
        return;
    };

    let desired = desired_chunks_v2(cam_world_pos.0, &planet, &load_radius);

    // Despawn chunk entities no longer desired and evict cache
    for (entity, coord) in &v2_chunks_q {
        if !desired.contains(&coord.0) {
            commands.entity(entity).despawn();
            chunk_map.loaded.remove(&coord.0);
            cache.remove(&coord.0);
        }
    }

    // Evict cache entries that are no longer in the desired set
    // (covers entries with no spawned entity yet)
    let stale: Vec<CubeSphereCoord> = cache
        .entries
        .keys()
        .filter(|c| !desired.contains(c))
        .copied()
        .collect();
    for c in stale {
        cache.remove(&c);
    }

    // Dispatch terrain generation tasks, closest to camera first.
    let pool = AsyncComputeTaskPool::get();
    let budget = MAX_TERRAIN_PENDING
        .saturating_sub(pending_terrain.pending.len())
        .min(MAX_TERRAIN_DISPATCHES_PER_FRAME);

    let mean_radius = planet.mean_radius;
    let fce = CubeSphereCoord::face_chunks_per_edge(mean_radius);
    let cam_coord = world_pos_to_coord(cam_world_pos.0, mean_radius, fce);

    let mut to_dispatch: Vec<CubeSphereCoord> = desired
        .iter()
        .filter(|c| {
            !chunk_map.loaded.contains(c)
                && !pending_terrain.pending.contains(c)
                && !pending_meshes.pending.contains(c)
                && !cache.contains(c)
        })
        .copied()
        .collect();

    // Sort: prioritize surface-layer chunks, then by distance to camera.
    to_dispatch.sort_unstable_by_key(|c| {
        let surface_bonus = if c.layer == 0 { 0 } else { 1_000_000 };
        let dist = if c.face == cam_coord.face {
            let du = c.u - cam_coord.u;
            let dv = c.v - cam_coord.v;
            let dl = c.layer - cam_coord.layer;
            du * du + dv * dv + dl * dl
        } else {
            i32::MAX / 2
        };
        surface_bonus + dist
    });

    for coord in to_dispatch.into_iter().take(budget) {
        let tgen = terrain_gen.0.clone();

        let task = pool.spawn(async move {
            generate_v2_voxels(coord, mean_radius, fce, &tgen)
        });

        commands.spawn(V2TerrainTask(task));
        pending_terrain.pending.insert(coord);
    }
}

/// Helper: compute the same-face neighbor coord for a given mesh direction.
///
/// Returns `None` if the neighbor crosses a face boundary (or is a radial neighbor).
/// The direction indices match greedy_mesh conventions:
///   0: +X → +U,  1: -X → -U,  2: +Y → +layer,  3: -Y → -layer,
///   4: +Z → -V,  5: -Z → +V
fn same_face_neighbor_for_dir(
    coord: CubeSphereCoord,
    dir: usize,
    max_uv: i32,
) -> Option<CubeSphereCoord> {
    let (du, dv, dl) = match dir {
        0 => (1, 0, 0),   // +X → +U
        1 => (-1, 0, 0),  // -X → -U
        2 => (0, 0, 1),   // +Y → +layer
        3 => (0, 0, -1),  // -Y → -layer
        4 => (0, -1, 0),  // +Z → -V
        5 => (0, 1, 0),   // -Z → +V
        _ => return None,
    };

    let new_u = coord.u + du;
    let new_v = coord.v + dv;
    let new_layer = coord.layer + dl;

    // Only same-face neighbors
    if new_u < 0 || new_u >= max_uv || new_v < 0 || new_v >= max_uv {
        return None;
    }

    Some(CubeSphereCoord::new(coord.face, new_u, new_v, new_layer))
}

/// Build neighbor slices from the voxel cache, falling back to terrain resampling.
///
/// For each of the 6 directions, tries to extract the boundary slice from the
/// cached voxels of the neighbor chunk. If the neighbor is not cached (or is on
/// a different face), falls back to terrain resampling.
fn build_neighbor_slices(
    coord: CubeSphereCoord,
    cache: &V2VoxelCache,
    mean_radius: f64,
    fce: f64,
    tgen: &UnifiedTerrainGenerator,
) -> NeighborSlices {
    let max_uv = fce as i32;
    let mut slices: [Option<Vec<crate::world::voxel::Voxel>>; 6] = [const { None }; 6];

    for dir in 0..6usize {
        if let Some(neighbor_coord) = same_face_neighbor_for_dir(coord, dir, max_uv) {
            if let Some(cached) = cache.get(&neighbor_coord) {
                // Extract the opposite edge from the neighbor
                let opposite_dir = dir ^ 1; // 0↔1, 2↔3, 4↔5
                slices[dir] = Some(extract_edge_slice(cached, opposite_dir));
                continue;
            }
        }
        // Fallback: resample terrain for this boundary
        slices[dir] = Some(generate_single_boundary_slice(
            coord, dir, mean_radius, fce, tgen,
        ));
    }

    NeighborSlices { slices }
}

/// Collect completed terrain tasks, cache voxels, and dispatch mesh tasks.
#[allow(clippy::too_many_arguments)]
pub fn v2_collect_terrain(
    mut commands: Commands,
    mut pending_terrain: ResMut<V2PendingTerrain>,
    mut pending_meshes: ResMut<V2PendingMeshes>,
    mut cache: ResMut<V2VoxelCache>,
    chunk_map: Res<V2ChunkMap>,
    terrain_gen: Res<V2TerrainGen>,
    planet: Res<PlanetConfig>,
    color_map: Res<MaterialColorMap>,
    mut task_q: Query<(Entity, &mut V2TerrainTask)>,
) {
    let mean_radius = planet.mean_radius;
    let fce = CubeSphereCoord::face_chunks_per_edge(mean_radius);

    // Stage 1: collect completed terrain tasks
    let mut newly_cached = Vec::new();
    let mut collected = 0usize;

    for (task_entity, mut task) in &mut task_q {
        if collected >= MAX_TERRAIN_COLLECTS_PER_FRAME {
            break;
        }
        let Some(result) = block_on(poll_once(&mut task.0)) else {
            continue;
        };
        collected += 1;

        commands.entity(task_entity).despawn();
        pending_terrain.pending.remove(&result.coord);

        // Store in cache
        cache.insert(result.coord, result.voxels);
        newly_cached.push(result.coord);
    }

    // Stage 2: dispatch mesh tasks for newly cached chunks
    // (also try chunks that were waiting for a neighbor to complete)
    let pool = AsyncComputeTaskPool::get();
    let mesh_budget = MAX_MESH_PENDING
        .saturating_sub(pending_meshes.pending.len())
        .min(MAX_MESH_DISPATCHES_PER_FRAME);

    // Candidates: any cached chunk that doesn't have a mesh yet and isn't pending
    let candidates: Vec<CubeSphereCoord> = cache
        .entries
        .keys()
        .filter(|c| {
            !chunk_map.loaded.contains(c) && !pending_meshes.pending.contains(c)
        })
        .copied()
        .collect();

    let mut dispatched = 0usize;
    for coord in candidates {
        if dispatched >= mesh_budget {
            break;
        }

        let tgen = terrain_gen.0.clone();
        let cmap = color_map.clone();
        let voxel_data = cache.get(&coord).unwrap().clone();

        // Build neighbor slices (from cache where possible, resample otherwise)
        let neighbor_slices = build_neighbor_slices(coord, &cache, mean_radius, fce, &tgen);

        let task = pool.spawn(async move {
            let voxels = cached_voxels_to_vec(&voxel_data);
            let mesh = greedy_mesh::greedy_mesh(&voxels, &neighbor_slices, &cmap);
            V2MeshResult { coord, mesh }
        });

        commands.spawn(V2MeshTask(task));
        pending_meshes.pending.insert(coord);
        dispatched += 1;
    }
}

/// Collect completed mesh tasks and spawn renderable entities.
#[allow(clippy::too_many_arguments)]
pub fn v2_collect_meshes(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut chunk_map: ResMut<V2ChunkMap>,
    mut pending_meshes: ResMut<V2PendingMeshes>,
    planet: Res<PlanetConfig>,
    origin: Res<RenderOrigin>,
    mut cached_mat: Local<Option<Handle<StandardMaterial>>>,
    mut task_q: Query<(Entity, &mut V2MeshTask)>,
) {
    let fce = CubeSphereCoord::face_chunks_per_edge(planet.mean_radius);
    let cs_half_f = CHUNK_SIZE as f32 / 2.0;

    let chunk_material = cached_mat
        .get_or_insert_with(|| {
            materials.add(StandardMaterial {
                base_color: Color::WHITE,
                ..default()
            })
        })
        .clone();

    let mut collected = 0usize;
    for (task_entity, mut task) in &mut task_q {
        if collected >= MAX_MESH_COLLECTS_PER_FRAME {
            break;
        }
        let Some(result) = block_on(poll_once(&mut task.0)) else {
            continue;
        };
        collected += 1;

        commands.entity(task_entity).despawn();
        pending_meshes.pending.remove(&result.coord);

        let (center_f64, rotation, tangent_scale) =
            result.coord.world_transform_scaled_f64(planet.mean_radius, fce);
        let cs_half_scaled = Vec3::new(
            cs_half_f * tangent_scale.x,
            cs_half_f,
            cs_half_f * tangent_scale.z,
        );
        let center_render = {
            let d = center_f64 - origin.0;
            Vec3::new(d.x as f32, d.y as f32, d.z as f32)
        };
        let adjusted = center_render - rotation * cs_half_scaled;

        if result.mesh.is_empty() {
            chunk_map.loaded.insert(result.coord);
            commands.spawn((
                V2ChunkMarker,
                V2ChunkCoord(result.coord),
                Transform::from_translation(adjusted)
                    .with_rotation(rotation)
                    .with_scale(tangent_scale),
            ));
            continue;
        }

        let bevy_mesh = chunk_mesh_to_bevy_mesh(result.mesh);
        let mesh_handle = meshes.add(bevy_mesh);

        commands.spawn((
            V2ChunkMarker,
            V2ChunkCoord(result.coord),
            ChunkMeshMarker,
            Mesh3d(mesh_handle),
            MeshMaterial3d(chunk_material.clone()),
            Transform::from_translation(adjusted)
                .with_rotation(rotation)
                .with_scale(tangent_scale),
        ));

        chunk_map.loaded.insert(result.coord);
    }
}

// ── Plugin ────────────────────────────────────────────────────────────────

/// When a `RenderOriginShift` occurs, update all V2 chunk Transforms.
fn v2_apply_origin_shift(
    shift: Option<Res<RenderOriginShift>>,
    mut chunk_q: Query<&mut Transform, With<V2ChunkMarker>>,
) {
    let Some(shift) = shift else { return };
    let delta = Vec3::new(shift.0.x as f32, shift.0.y as f32, shift.0.z as f32);
    for mut transform in &mut chunk_q {
        transform.translation -= delta;
    }
}

/// Plugin for the V2 cubed-sphere rendering pipeline.
pub struct V2WorldPlugin;

impl Plugin for V2WorldPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<V2LoadRadius>()
            .init_resource::<V2ChunkMap>()
            .init_resource::<V2PendingTerrain>()
            .init_resource::<V2PendingMeshes>()
            .init_resource::<V2VoxelCache>()
            .add_systems(Startup, v2_init_terrain_gen)
            .add_systems(
                Update,
                (v2_update_chunks, v2_collect_terrain, v2_collect_meshes, v2_diagnostics)
                    .chain()
                    .run_if(resource_exists::<V2TerrainGen>)
                    .run_if(not(in_state(crate::game_state::GameState::WorldCreation))),
            )
            .add_systems(
                PostUpdate,
                v2_apply_origin_shift.run_if(resource_exists::<RenderOriginShift>),
            );
    }
}

/// Periodic diagnostic logging for the V2 pipeline.
fn v2_diagnostics(
    chunk_map: Res<V2ChunkMap>,
    pending_terrain: Res<V2PendingTerrain>,
    pending_meshes: Res<V2PendingMeshes>,
    cache: Res<V2VoxelCache>,
    mesh_q: Query<Entity, (With<V2ChunkMarker>, With<Mesh3d>)>,
    mut timer: Local<f32>,
    time: Res<Time>,
) {
    *timer += time.delta_secs();
    if *timer < 5.0 {
        return;
    }
    *timer = 0.0;

    let mesh_count = mesh_q.iter().count();
    info!(
        "V2 pipeline: loaded={}, terrain_pending={}, mesh_pending={}, cached={} ({:.1} MB), meshed={}",
        chunk_map.loaded.len(),
        pending_terrain.pending.len(),
        pending_meshes.pending.len(),
        cache.entry_count(),
        cache.byte_size() as f64 / (1024.0 * 1024.0),
        mesh_count,
    );
}

/// Clone the unified terrain generator from SharedTerrainGen into V2TerrainGen.
///
/// V2 only supports spherical and planetary terrain modes.  Flat-mode
/// generators are skipped — the resource is not inserted, which prevents
/// `v2_update_chunks` from running (it has a `resource_exists::<V2TerrainGen>`
/// condition).  Flat mode lost its V1 rendering pipeline; until a V2 flat
/// mode is implemented no terrain will appear for those presets.
fn v2_init_terrain_gen(
    mut commands: Commands,
    shared: Option<Res<SharedTerrainGen>>,
    planet: Res<PlanetConfig>,
) {
    if let Some(shared) = shared {
        if matches!(
            shared.0.as_ref(),
            crate::world::terrain::UnifiedTerrainGenerator::Flat(_)
        ) {
            warn!(
                "V2 pipeline: flat terrain mode is not supported; \
                 V2 rendering disabled for this preset"
            );
            return;
        }
        commands.insert_resource(V2TerrainGen(shared.0.clone()));
        info!("V2 pipeline: initialized terrain generator from SharedTerrainGen");
        return;
    }
    // Fallback when SharedTerrainGen isn't ready yet: create from PlanetConfig.
    use crate::world::terrain::SphericalTerrainGenerator;
    let tgen = SphericalTerrainGenerator::new(planet.clone());
    let unified = Arc::new(crate::world::terrain::UnifiedTerrainGenerator::Spherical(
        Box::new(tgen),
    ));
    commands.insert_resource(V2TerrainGen(unified));
    info!("V2 pipeline: initialized terrain generator from PlanetConfig (fallback)");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::terrain::{SphericalTerrainGenerator, UnifiedTerrainGenerator};
    use crate::world::v2::cubed_sphere::CubeSphereCoord;

    #[test]
    fn desired_chunks_contains_camera_position() {
        let cfg = PlanetConfig {
            mean_radius: 32000.0,
            ..Default::default()
        };
        let radius = V2LoadRadius::default();
        let fce = CubeSphereCoord::face_chunks_per_edge(cfg.mean_radius);
        let cam_pos = DVec3::new(cfg.mean_radius, 0.0, 0.0);
        let desired = desired_chunks_v2(cam_pos, &cfg, &radius);

        assert!(!desired.is_empty(), "Desired set should not be empty");

        // The camera's own coord should be in the set
        let cam_coord = world_pos_to_coord(cam_pos, cfg.mean_radius, fce);
        assert!(
            desired.contains(&cam_coord),
            "Camera's own chunk should be desired"
        );
    }

    #[test]
    fn desired_chunks_count_matches_radius() {
        let cfg = PlanetConfig {
            mean_radius: 32000.0,
            ..Default::default()
        };
        let radius = V2LoadRadius {
            horizontal: 2,
            vertical: 1,
        };

        let cam_pos = DVec3::new(cfg.mean_radius, 0.0, 0.0);
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

    /// End-to-end lifecycle test: dispatch tasks, let them complete, collect
    /// results, and verify chunk entities are spawned with correct components.
    #[test]
    fn integration_v2_lifecycle() {
        use bevy::asset::AssetPlugin;
        use bevy::image::Image;
        use bevy::pbr::StandardMaterial;

        use crate::floating_origin::RenderOrigin;

        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.add_plugins(AssetPlugin::default());

        // Register asset types that DefaultPlugins normally provides.
        app.init_asset::<Mesh>();
        app.init_asset::<Image>();
        app.init_asset::<StandardMaterial>();

        // Small smooth planet for fast terrain gen.
        let planet = PlanetConfig {
            mean_radius: 200.0,
            sea_level_radius: 200.0,
            noise: None,
            height_scale: 0.0,
            cave_threshold: -999.0,
            ..Default::default()
        };

        let tgen = Arc::new(UnifiedTerrainGenerator::Spherical(Box::new(
            SphericalTerrainGenerator::new(planet.clone()),
        )));
        app.insert_resource(planet);
        app.insert_resource(V2TerrainGen(tgen));
        app.insert_resource(V2LoadRadius {
            horizontal: 2,
            vertical: 1,
        });
        app.init_resource::<V2ChunkMap>();
        app.init_resource::<V2PendingTerrain>();
        app.init_resource::<V2PendingMeshes>();
        app.init_resource::<V2VoxelCache>();
        app.insert_resource(MaterialColorMap::from_defaults());

        // RenderOrigin at the camera position so chunk Transforms are near zero.
        let cam_world = DVec3::new(200.0, 0.0, 0.0);
        app.insert_resource(RenderOrigin(cam_world));

        // Spawn camera at the surface of the +X face with WorldPosition.
        app.world_mut().spawn((
            FpsCamera::default(),
            WorldPosition::from_dvec3(cam_world),
            Transform::from_translation(Vec3::ZERO),
        ));

        // Register update systems (bypassing GameState gating for the test).
        app.add_systems(Update, (v2_update_chunks, v2_collect_terrain, v2_collect_meshes).chain());

        // Frame 1: dispatch tasks.
        app.update();

        let pending_count = app.world().resource::<V2PendingTerrain>().pending.len();
        assert!(
            pending_count > 0,
            "Should have dispatched pending terrain tasks, got 0"
        );

        // Give async tasks time to complete, then run several frames to collect.
        for _ in 0..20 {
            std::thread::sleep(std::time::Duration::from_millis(50));
            app.update();
        }

        let loaded_count = app.world().resource::<V2ChunkMap>().loaded.len();
        assert!(
            loaded_count > 0,
            "Should have collected at least one chunk, got 0 loaded"
        );

        // Verify chunk entities have the expected components.
        let mut marker_query = app
            .world_mut()
            .query_filtered::<(Entity, &V2ChunkCoord, &Transform), With<V2ChunkMarker>>();
        let entity_count = marker_query.iter(app.world()).count();
        assert!(
            entity_count > 0,
            "Should have spawned V2ChunkMarker entities, got 0"
        );

        // Check that at least some entities have meshes (non-empty chunks).
        let mut mesh_query = app
            .world_mut()
            .query_filtered::<Entity, (With<V2ChunkMarker>, With<Mesh3d>)>();
        let mesh_count = mesh_query.iter(app.world()).count();

        // On a 200m-radius smooth planet with h=2,v=1 from the +X face,
        // there should be some surface-layer chunks with visible geometry.
        assert!(
            mesh_count > 0,
            "Should have spawned V2 chunks with meshes, got 0 \
             (loaded={loaded_count}, entities={entity_count})"
        );

        // Verify transforms are near the camera (render-space, origin at camera).
        // With RenderOrigin at (200,0,0), chunks should have Translation near
        // their offset from that point.
        for (_entity, coord, transform) in marker_query.iter(app.world()) {
            let dist = transform.translation.length();
            // Chunks within load radius should be within a few chunk-sizes of origin
            let max_expected = (2 + 1) as f32 * CHUNK_SIZE as f32 * 3.0;
            assert!(
                dist < max_expected,
                "Chunk at {:?} has render distance {dist:.1}, \
                 expected < {max_expected:.1}",
                coord.0,
            );
        }
    }
}
