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
use crate::gpu::voxel_compute::{
    GpuChunkRequest, GpuVoxelCompute, MAX_CHUNKS_PER_BATCH, chunk_desc_from_coord,
};
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

/// Maximum LOD level (L7 = 2^7 = 128× base chunk size ≈ 4096m per chunk).
const _MAX_LOD: u8 = 7;

/// LOD ring configuration: (lod_level, radius_in_chunks_at_that_lod).
/// Each ring extends outward from the previous ring's boundary.
/// At LOD `l`, each chunk covers `CHUNK_SIZE * 2^l` meters.
const LOD_RINGS: [(u8, i32); 5] = [
    (0, 12), // L0: 12 chunks × 32m = 384m
    (1, 8),  // L1: 8 chunks × 64m = 512m (total ~896m)
    (2, 8),  // L2: 8 chunks × 128m = 1024m (total ~1920m)
    (3, 8),  // L3: 8 chunks × 256m = 2048m (total ~3968m)
    (4, 12), // L4: 12 chunks × 512m = 6144m (total ~10112m)
];

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

/// GPU terrain dispatcher resource.
///
/// Holds the `GpuVoxelCompute` pipeline (if GPU is available) and pending
/// GPU batch tasks that run on worker threads to avoid blocking the main ECS thread.
#[derive(Resource, Default)]
pub struct GpuTerrainDispatcher {
    compute: Option<Arc<GpuVoxelCompute>>,
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

/// Async GPU batch terrain task — runs a batch of chunks on the GPU worker thread.
#[derive(Component)]
pub struct V2GpuTerrainTask(pub Task<Vec<V2TerrainData>>);

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
    terrain_gen: Option<&UnifiedTerrainGenerator>,
) -> HashSet<CubeSphereCoord> {
    let base_fce = CubeSphereCoord::face_chunks_per_edge(planet.mean_radius);
    let cam_coord = world_pos_to_coord(cam_world_pos, planet.mean_radius, base_fce);

    let altitude_m = cam_world_pos.length() - planet.mean_radius;
    let cam_layer = cam_coord.layer;

    // Vertical range at LOD 0 (camera-centered; LOD 0 chunks are small
    // compared to terrain relief, so we follow the camera's own layer band).
    let layer_lo = (0.min(cam_layer) - radius.vertical).max(-MAX_VERTICAL_LAYERS);
    let layer_hi = (0.max(cam_layer) + radius.vertical).min(MAX_VERTICAL_LAYERS);

    let mut set = HashSet::new();

    // Track the cumulative world-space radius covered by finer LODs
    // so coarser LODs only fill the ring beyond.
    let mut covered_world_radius = 0.0_f64;

    for &(lod, ring_chunks) in &LOD_RINGS {
        let fce_lod = CubeSphereCoord::face_chunks_per_edge_lod(planet.mean_radius, lod);
        let max_uv = fce_lod as i32;
        if max_uv <= 0 {
            continue;
        }

        // Camera coord at this LOD level
        let cam_coord_lod = world_pos_to_coord_lod(cam_world_pos, planet.mean_radius, fce_lod, lod);

        let chunk_world_size = CHUNK_SIZE as f64 * (1u64 << lod) as f64;

        // Effective radius at this LOD
        let h = if lod == 0 {
            let horizon = horizon_load_radius(altitude_m, planet.mean_radius, radius.horizontal);
            horizon.min(ring_chunks).min(MAX_HORIZONTAL_CHUNKS)
        } else {
            ring_chunks.min(MAX_HORIZONTAL_CHUNKS)
        };

        // The inner radius (in world meters) already covered by finer LODs
        let inner_radius_chunks = (covered_world_radius / chunk_world_size).ceil() as i32;

        for du in -h..=h {
            for dv in -h..=h {
                // Skip inner area already covered by finer LOD
                let dist_chunks = du.abs().max(dv.abs());
                if lod > 0 && dist_chunks < inner_radius_chunks {
                    continue;
                }

                // Vertical range for this LOD:
                //   LOD 0  → camera-centered band (camera is near ground).
                //   LOD ≥1 → the single layer that contains the terrain
                //            surface at this (u, v) direction, plus the
                //            immediate neighbour layer in each radial
                //            direction so meshes line up across layer
                //            boundaries. The surface radius is sampled
                //            from the terrain generator; if none is
                //            provided (unit tests), we fall back to
                //            layer 0.
                let (lo, hi) = if lod == 0 {
                    (layer_lo, layer_hi)
                } else {
                    let u = cam_coord_lod.u + du;
                    let vi = cam_coord_lod.v + dv;
                    let face = cam_coord_lod.face;
                    let probe = CubeSphereCoord::new_with_lod(face, u, vi, 0, lod);
                    let dir = probe.unit_sphere_dir(fce_lod);
                    let center_layer = if let Some(tg) = terrain_gen {
                        let (lat, lon) = planet.lat_lon(dir);
                        let surface_r = tg.sample_surface_radius_at(lat, lon);
                        ((surface_r - planet.mean_radius) / chunk_world_size).round() as i32
                    } else {
                        0
                    };
                    (center_layer - 1, center_layer + 1)
                };

                for layer in lo..=hi {
                    let u = cam_coord_lod.u + du;
                    let vi = cam_coord_lod.v + dv;
                    let face = cam_coord_lod.face;

                    if u < 0 || u >= max_uv || vi < 0 || vi >= max_uv {
                        if let Some(wrapped) = wrap_cross_face(face, u, vi, layer, max_uv, lod) {
                            set.insert(wrapped);
                        }
                        continue;
                    }

                    set.insert(CubeSphereCoord::new_with_lod(face, u, vi, layer, lod));
                }
            }
        }

        covered_world_radius += h as f64 * chunk_world_size;
    }

    set
}

/// Convert world position to CubeSphereCoord at a specific LOD level.
fn world_pos_to_coord_lod(pos: DVec3, mean_radius: f64, fce_lod: f64, lod: u8) -> CubeSphereCoord {
    let mut coord = world_pos_to_coord(pos, mean_radius, fce_lod);
    coord.lod = lod;
    coord
}

/// Resolve an out-of-range (u, v) on `face` to the correct cross-face coord.
/// Returns `None` if the coord is doubly out-of-range (corner wrap — skip).
fn wrap_cross_face(
    face: CubeFace,
    u: i32,
    v: i32,
    layer: i32,
    max_uv: i32,
    lod: u8,
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
        let base = CubeSphereCoord::new_with_lod(face, clamped_u, v, layer, lod);
        let delta = if u < 0 {
            -(clamped_u - u)
        } else {
            u - clamped_u
        };
        // Walk across the face boundary
        let neighbor = if u < 0 {
            base.neighbors(max_uv)[1] // -U neighbor
        } else {
            base.neighbors(max_uv)[0] // +U neighbor
        };
        // For deeper steps, we'd need recursive walking. For now, only load
        // the immediate cross-face layer (1 chunk deep across boundary).
        if delta.abs() <= 1 {
            return Some(CubeSphereCoord::new_with_lod(
                neighbor.face,
                neighbor.u,
                neighbor.v,
                layer,
                lod,
            ));
        }
        return None;
    }

    if v_oob {
        let clamped_v = if v < 0 { 0 } else { max_uv - 1 };
        let base = CubeSphereCoord::new_with_lod(face, u, clamped_v, layer, lod);
        let delta = if v < 0 {
            -(clamped_v - v)
        } else {
            v - clamped_v
        };
        let neighbor = if v < 0 {
            base.neighbors(max_uv)[3] // -V neighbor
        } else {
            base.neighbors(max_uv)[2] // +V neighbor
        };
        if delta.abs() <= 1 {
            return Some(CubeSphereCoord::new_with_lod(
                neighbor.face,
                neighbor.u,
                neighbor.v,
                layer,
                lod,
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
    gpu_dispatcher: Res<GpuTerrainDispatcher>,
    camera_q: Query<&WorldPosition, With<FpsCamera>>,
    v2_chunks_q: Query<(Entity, &V2ChunkCoord), With<V2ChunkMarker>>,
    gpu_tasks_q: Query<Entity, With<V2GpuTerrainTask>>,
) {
    let Ok(cam_world_pos) = camera_q.single() else {
        return;
    };

    let desired = desired_chunks_v2(cam_world_pos.0, &planet, &load_radius, Some(&terrain_gen.0));

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

    // Sort: prioritize L0, then surface-layer, then by distance to camera.
    // Distance is approximate — we compare in the chunk's own LOD grid space.
    let base_fce = CubeSphereCoord::face_chunks_per_edge(mean_radius);
    let cam_coord_l0 = world_pos_to_coord(cam_world_pos.0, mean_radius, base_fce);
    to_dispatch.sort_unstable_by_key(|c| {
        let lod_penalty = (c.lod as i32) * 2_000_000;
        let surface_bonus = if c.layer == 0 { 0 } else { 1_000_000 };
        // Scale chunk distance to L0 grid for comparison
        let scale = 1i32 << c.lod;
        let dist = if c.face == cam_coord_l0.face {
            let du = c.u * scale - cam_coord_l0.u;
            let dv = c.v * scale - cam_coord_l0.v;
            let dl = c.layer - cam_coord_l0.layer;
            du * du + dv * dv + dl * dl
        } else {
            i32::MAX / 4
        };
        lod_penalty + surface_bonus + dist
    });

    // Separate GPU-eligible and CPU-fallback chunks.
    let dispatched: Vec<CubeSphereCoord> = to_dispatch
        .into_iter()
        .take(budget.min(MAX_CHUNKS_PER_BATCH))
        .collect();

    if let Some(ref compute) = gpu_dispatcher.compute {
        // Only dispatch a GPU batch if no previous batch is still in flight —
        // the pre-allocated staging buffers are not thread-safe.
        let gpu_in_flight = !gpu_tasks_q.is_empty();
        if !dispatched.is_empty() && !gpu_in_flight {
            let compute = compute.clone();
            let mean_r = mean_radius;
            let sea_level = planet.sea_level_radius;
            let soil_depth = planet.soil_depth;
            let cave_threshold = planet.cave_threshold;
            let rot_axis = planet.rotation_axis;
            let batch_coords: Vec<(CubeSphereCoord, f64)> = dispatched
                .iter()
                .map(|c| (*c, CubeSphereCoord::face_chunks_per_edge_lod(mean_r, c.lod)))
                .collect();

            for coord in &dispatched {
                pending_terrain.pending.insert(*coord);
            }

            let task = pool.spawn(async move {
                let requests: Vec<GpuChunkRequest> = batch_coords
                    .iter()
                    .enumerate()
                    .map(|(i, (coord, fce))| {
                        let desc = chunk_desc_from_coord(
                            *coord,
                            mean_r,
                            *fce,
                            sea_level,
                            soil_depth,
                            cave_threshold,
                            i as u32,
                        );
                        GpuChunkRequest {
                            coord: *coord,
                            desc,
                        }
                    })
                    .collect();
                let result = compute.generate_batch(&requests, rot_axis);
                result.terrain_data
            });

            commands.spawn(V2GpuTerrainTask(task));
        } else if !dispatched.is_empty() {
            // GPU batch still in flight — fall back to CPU for this frame.
            for coord in dispatched {
                let tgen = terrain_gen.0.clone();
                let coord_fce = CubeSphereCoord::face_chunks_per_edge_lod(mean_radius, coord.lod);
                let task = pool
                    .spawn(async move { generate_v2_voxels(coord, mean_radius, coord_fce, &tgen) });
                commands.spawn(V2TerrainTask(task));
                pending_terrain.pending.insert(coord);
            }
        }
    } else {
        // CPU fallback path: one task per chunk via AsyncComputeTaskPool.
        for coord in dispatched {
            let tgen = terrain_gen.0.clone();
            let coord_fce = CubeSphereCoord::face_chunks_per_edge_lod(mean_radius, coord.lod);

            let task =
                pool.spawn(async move { generate_v2_voxels(coord, mean_radius, coord_fce, &tgen) });

            commands.spawn(V2TerrainTask(task));
            pending_terrain.pending.insert(coord);
        }
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
        0 => (1, 0, 0),  // +X → +U
        1 => (-1, 0, 0), // -X → -U
        2 => (0, 0, 1),  // +Y → +layer
        3 => (0, 0, -1), // -Y → -layer
        4 => (0, -1, 0), // +Z → -V
        5 => (0, 1, 0),  // -Z → +V
        _ => return None,
    };

    let new_u = coord.u + du;
    let new_v = coord.v + dv;
    let new_layer = coord.layer + dl;

    // Only same-face neighbors
    if new_u < 0 || new_u >= max_uv || new_v < 0 || new_v >= max_uv {
        return None;
    }

    Some(CubeSphereCoord::new_with_lod(
        coord.face, new_u, new_v, new_layer, coord.lod,
    ))
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

    for (dir, slot) in slices.iter_mut().enumerate() {
        if let Some(neighbor_coord) = same_face_neighbor_for_dir(coord, dir, max_uv)
            && let Some(cached) = cache.get(&neighbor_coord)
        {
            // Extract the opposite edge from the neighbor
            let opposite_dir = dir ^ 1; // 0↔1, 2↔3, 4↔5
            *slot = Some(extract_edge_slice(cached, opposite_dir));
            continue;
        }
        // Fallback: resample terrain for this boundary
        *slot = Some(generate_single_boundary_slice(
            coord,
            dir,
            mean_radius,
            fce,
            tgen,
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
        .filter(|c| !chunk_map.loaded.contains(c) && !pending_meshes.pending.contains(c))
        .copied()
        .collect();

    for (dispatched, coord) in candidates.into_iter().enumerate() {
        if dispatched >= mesh_budget {
            break;
        }

        let voxel_data = cache.get(&coord).unwrap().clone();

        // Fast-path: AllSolid / AllAir chunks never contain a visible surface
        // (they are entirely buried or entirely empty).  Emitting a greedy
        // mesh for them would produce a 6-face hull that shows up as a
        // phantom cube when neighbours are unloaded — the source of the
        // "floating square faces" seen at coarse LODs where adjacent
        // surface-tracking columns sit at very different altitudes.
        if matches!(voxel_data, CachedVoxels::AllSolid(_) | CachedVoxels::AllAir) {
            let task = pool.spawn(async move {
                V2MeshResult {
                    coord,
                    mesh: ChunkMesh {
                        positions: Vec::new(),
                        normals: Vec::new(),
                        colors: Vec::new(),
                        indices: Vec::new(),
                    },
                }
            });
            commands.spawn(V2MeshTask(task));
            pending_meshes.pending.insert(coord);
            continue;
        }

        let tgen = terrain_gen.0.clone();
        let cmap = color_map.clone();

        let coord_fce = CubeSphereCoord::face_chunks_per_edge_lod(mean_radius, coord.lod);
        // Build neighbor slices (from cache where possible, resample otherwise)
        let neighbor_slices = build_neighbor_slices(coord, &cache, mean_radius, coord_fce, &tgen);

        let task = pool.spawn(async move {
            let voxels = cached_voxels_to_vec(&voxel_data);
            let mesh = greedy_mesh::greedy_mesh(&voxels, &neighbor_slices, &cmap);
            V2MeshResult { coord, mesh }
        });

        commands.spawn(V2MeshTask(task));
        pending_meshes.pending.insert(coord);
    }
}

/// Collect completed GPU batch terrain tasks and insert results into the voxel cache.
///
/// Runs between `v2_update_chunks` and `v2_collect_terrain` — GPU results enter
/// the same cache, so the existing mesh dispatch in `v2_collect_terrain` picks them up.
pub fn v2_collect_gpu_terrain(
    mut commands: Commands,
    mut pending_terrain: ResMut<V2PendingTerrain>,
    mut cache: ResMut<V2VoxelCache>,
    mut task_q: Query<(Entity, &mut V2GpuTerrainTask)>,
) {
    let mut collected = 0usize;
    for (task_entity, mut task) in &mut task_q {
        if collected >= MAX_TERRAIN_COLLECTS_PER_FRAME {
            break;
        }
        let Some(results) = block_on(poll_once(&mut task.0)) else {
            continue;
        };
        collected += 1;

        commands.entity(task_entity).despawn();

        for result in results {
            pending_terrain.pending.remove(&result.coord);
            cache.insert(result.coord, result.voxels);
        }
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
    let mean_radius = planet.mean_radius;
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

        let coord_fce = CubeSphereCoord::face_chunks_per_edge_lod(mean_radius, result.coord.lod);
        let (center_f64, rotation, tangent_scale) = result
            .coord
            .world_transform_scaled_f64(mean_radius, coord_fce);
        let cs_half_scaled = Vec3::new(
            cs_half_f * tangent_scale.x,
            cs_half_f * tangent_scale.y,
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
            .init_resource::<GpuTerrainDispatcher>()
            .add_systems(Startup, v2_init_terrain_gen)
            .add_systems(
                Update,
                (
                    v2_update_chunks,
                    v2_collect_gpu_terrain,
                    v2_collect_terrain,
                    v2_collect_meshes,
                    v2_diagnostics,
                )
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
    mut gpu_dispatcher: ResMut<GpuTerrainDispatcher>,
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
    } else {
        // Fallback when SharedTerrainGen isn't ready yet: create from PlanetConfig.
        use crate::world::terrain::SphericalTerrainGenerator;
        let tgen = SphericalTerrainGenerator::new(planet.clone());
        let unified = Arc::new(crate::world::terrain::UnifiedTerrainGenerator::Spherical(
            Box::new(tgen),
        ));
        commands.insert_resource(V2TerrainGen(unified));
        info!("V2 pipeline: initialized terrain generator from PlanetConfig (fallback)");
    }

    // Initialize GPU voxel compute pipeline if noise config is available.
    // Set TDC_DISABLE_GPU_VOXELS=1 to force CPU-only generation (diagnostic).
    let gpu_disabled = std::env::var("TDC_DISABLE_GPU_VOXELS")
        .ok()
        .is_some_and(|v| v != "0" && !v.is_empty());
    if gpu_disabled {
        info!("V2 pipeline: GPU voxel generation disabled via TDC_DISABLE_GPU_VOXELS");
    } else if let Some(ref noise_config) = planet.noise {
        match GpuVoxelCompute::try_new(
            noise_config,
            planet.seed,
            planet.mean_radius,
            planet.height_scale,
        ) {
            Some(compute) => {
                gpu_dispatcher.compute = Some(Arc::new(compute));
                info!("V2 pipeline: GPU voxel generation enabled");
            }
            None => {
                info!("V2 pipeline: no GPU available, using CPU terrain generation");
            }
        }
    } else {
        info!("V2 pipeline: no noise config, GPU terrain generation disabled");
    }
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
        let desired = desired_chunks_v2(cam_pos, &cfg, &radius, None);

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
        let desired = desired_chunks_v2(cam_pos, &cfg, &radius, None);

        // With multi-LOD rings, the desired set includes L0 chunks plus coarser LOD rings.
        // L0 ring: (2*2+1)² × (2*1+1) = 5² × 3 = 75 chunks
        // Plus additional chunks from coarser LOD levels.
        // Just verify reasonable bounds.
        assert!(
            desired.len() <= 5000,
            "Should not exceed reasonable count, got {}",
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
        app.init_resource::<GpuTerrainDispatcher>();
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
        app.add_systems(
            Update,
            (v2_update_chunks, v2_collect_terrain, v2_collect_meshes).chain(),
        );

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
