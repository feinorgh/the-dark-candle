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
use crate::world::v2::greedy_mesh::NeighborSlices;
use crate::world::v2::surface_nets;
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
/// Maximum vertical load extent in layers (currently unused — kept as a
/// reference for future clamping of runaway layer counts).  Do NOT use to
/// clamp the camera band: at high altitudes the camera can sit at layer
/// 200+, far outside any symmetric absolute cap.
#[allow(dead_code)]
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

/// Tracks which CubeSphereCoords currently have a spawned entity, mapped to
/// the entity itself so we can despawn for remesh-invalidation.
#[derive(Resource, Default)]
pub struct V2ChunkMap {
    loaded: HashMap<CubeSphereCoord, Entity>,
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

/// Newly-cached chunk coordinates queued for neighbour-mesh invalidation.
///
/// Populated by both `v2_collect_terrain` (CPU path) and `v2_collect_gpu_terrain`
/// (GPU path) when fresh terrain data arrives. Drained by `v2_collect_terrain`
/// before its mesh-dispatch stage: each newly-cached coord causes its 6
/// same-face neighbours to be despawned and removed from `chunk_map.loaded`,
/// so they re-mesh against the now-cached real boundary data instead of
/// keeping stale resampled-fallback geometry forever.
#[derive(Resource, Default)]
pub struct V2InvalidationQueue {
    pub coords: Vec<CubeSphereCoord>,
}

/// Per-frame stats for the v2 chunk pipeline (HUD diagnostics).
///
/// Updated by `v2_update_chunks` once per frame so the F3 overlay can show
/// what the v2 pipeline is doing right now without having to recompute the
/// (expensive) desired set itself.
#[derive(Resource, Default, Debug, Clone, Copy)]
pub struct V2PipelineStats {
    pub desired: usize,
    pub desired_lod0: usize,
    pub desired_lod_max: usize,
    pub pending_terrain: usize,
    pub pending_meshes: usize,
    pub cache_entries: usize,
    pub cache_all_air: usize,
    pub cache_all_solid: usize,
    pub cache_mixed: usize,
    pub loaded: usize,
    pub loaded_lod0: usize,
    pub loaded_lod_max: usize,
    pub meshed_lod0: usize,
    pub meshed_lod_max: usize,
    pub dispatched_this_frame: usize,
    pub gpu_in_flight: bool,
}

/// GPU terrain dispatcher resource.
///
/// Holds the `GpuVoxelCompute` pipeline (if GPU is available) and pending
/// GPU batch tasks that run on worker threads to avoid blocking the main ECS thread.
#[derive(Resource, Default)]
pub struct GpuTerrainDispatcher {
    compute: Option<Arc<GpuVoxelCompute>>,
}

/// Tracks whether the planetary heightmap has been injected into the GPU pipeline.
///
/// Once injected the system no longer runs, avoiding repeated async spawns.
#[derive(Resource, Default, PartialEq)]
pub struct GpuHeightmapInjected(bool);

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

    /// Returns (all_air, all_solid, mixed) counts for diagnostics.
    pub fn classification_counts(&self) -> (usize, usize, usize) {
        let mut air = 0;
        let mut solid = 0;
        let mut mixed = 0;
        for v in self.entries.values() {
            match v {
                CachedVoxels::AllAir => air += 1,
                CachedVoxels::AllSolid(_) => solid += 1,
                CachedVoxels::Mixed(_) => mixed += 1,
            }
        }
        (air, solid, mixed)
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

/// Insert a `(face, u, v, layer)` coord into `set`, handling cross-face
/// wrap when `(u, v)` lies outside the face.
#[inline]
fn insert_or_wrap(
    set: &mut HashSet<CubeSphereCoord>,
    face: CubeFace,
    u: i32,
    vi: i32,
    layer: i32,
    max_uv: i32,
    lod: u8,
) {
    if u < 0 || u >= max_uv || vi < 0 || vi >= max_uv {
        if let Some(wrapped) = wrap_cross_face(face, u, vi, layer, max_uv, lod) {
            set.insert(wrapped);
        }
        return;
    }
    set.insert(CubeSphereCoord::new_with_lod(face, u, vi, layer, lod));
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

    // Vertical range at LOD 0 centered on the *camera's* layer.
    //
    // The earlier formula `(0.min(cam_layer), 0.max(cam_layer))` baked in
    // the assumption that layer 0 (around mean_radius) is the surface, which
    // is only true for toy planets with tiny height_scale.  On an Earth-scale
    // preset the player can stand on a 5 km mountain (cam_layer ≈ 156) and
    // that formula tries to load ~160 layers per column — effectively
    // starving the real surface chunks behind an avalanche of AllAir/AllSolid
    // chunks and producing visible gaps in the near terrain.
    //
    // IMPORTANT: these are camera-RELATIVE bounds.  Do NOT clamp them with
    // `MAX_VERTICAL_LAYERS` (which is an absolute layer index around
    // mean_radius): at high altitude `cam_layer` can be 200+ and
    // `min(cam_layer + 2, 64)` would invert the interval and produce an
    // empty camera band, leaving a chunk-sized void around the player.
    // The explosion this cap was meant to prevent is handled by loading the
    // surface band and camera band as two *separate* intervals below,
    // instead of one contiguous range stretching from surface to camera.
    let layer_lo = cam_layer - radius.vertical;
    let layer_hi = cam_layer + radius.vertical;

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

                // Per-(u,v) surface tracking for ALL LODs: sample the
                // terrain surface radius at this column and load a small
                // band around the surface layer.  At LOD 0 we additionally
                // union with the camera-centered band so chunks directly
                // around the player (e.g. while flying or in caves) stay
                // loaded regardless of the ground below.
                //
                // Using `planet.lat_lon` here would be wrong: the voxel
                // terrain (`sample_surface_radius_at`) uses the
                // `lat_lon_to_pos` convention, which is NOT the inverse
                // of `planet.lat_lon`.  We convert the probe direction to
                // (lat, lon) with the matching inverse, `pos_to_lat_lon`.
                let u = cam_coord_lod.u + du;
                let vi = cam_coord_lod.v + dv;
                let face = cam_coord_lod.face;

                let (lo, hi) = if let Some(tg) = terrain_gen {
                    let probe = CubeSphereCoord::new_with_lod(face, u, vi, 0, lod);
                    let dir = probe.unit_sphere_dir(fce_lod);
                    let (lat, lon) = crate::planet::detail::pos_to_lat_lon(dir);
                    let surface_r = tg.sample_surface_radius_at(lat, lon);
                    let center_layer =
                        ((surface_r - planet.mean_radius) / chunk_world_size).round() as i32;
                    let surf_lo = center_layer - 1;
                    let surf_hi = center_layer + 1;
                    if lod == 0 {
                        // Load the surface band AND the camera band as two
                        // *separate* intervals.  Do NOT union them into one
                        // range (`min(surf_lo, layer_lo), max(surf_hi, layer_hi)`)
                        // — at high altitude that fills hundreds of layers
                        // per column with AllAir chunks between surface and
                        // camera, starving the generation budget and causing
                        // visible chunk-sized voids around the player.
                        for layer in surf_lo..=surf_hi {
                            insert_or_wrap(&mut set, face, u, vi, layer, max_uv, lod);
                        }
                        // Camera band (skip overlap with surface band).
                        for layer in layer_lo..=layer_hi {
                            if layer >= surf_lo && layer <= surf_hi {
                                continue;
                            }
                            insert_or_wrap(&mut set, face, u, vi, layer, max_uv, lod);
                        }
                        // Skip the generic loop below — already inserted.
                        continue;
                    } else {
                        (surf_lo, surf_hi)
                    }
                } else if lod == 0 {
                    (layer_lo, layer_hi)
                } else {
                    (-1, 1)
                };

                for layer in lo..=hi {
                    insert_or_wrap(&mut set, face, u, vi, layer, max_uv, lod);
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
    mut stats: ResMut<V2PipelineStats>,
    camera_q: Query<&WorldPosition, With<FpsCamera>>,
    v2_chunks_q: Query<(Entity, &V2ChunkCoord), With<V2ChunkMarker>>,
    gpu_tasks_q: Query<Entity, With<V2GpuTerrainTask>>,
) {
    let Ok(cam_world_pos) = camera_q.single() else {
        return;
    };

    let desired = desired_chunks_v2(cam_world_pos.0, &planet, &load_radius, Some(&terrain_gen.0));

    // Per-LOD desired breakdown for HUD diagnostics.
    let max_lod = LOD_RINGS.iter().map(|(l, _)| *l).max().unwrap_or(0);
    let desired_lod0 = desired.iter().filter(|c| c.lod == 0).count();
    let desired_lod_max = desired.iter().filter(|c| c.lod == max_lod).count();

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

    let to_dispatch: Vec<CubeSphereCoord> = desired
        .iter()
        .filter(|c| {
            !chunk_map.loaded.contains_key(c)
                && !pending_terrain.pending.contains(c)
                && !pending_meshes.pending.contains(c)
                && !cache.contains(c)
        })
        .copied()
        .collect();

    // Sort: prioritize L0, then proximity to the actual surface layer for each
    // chunk's column, then distance to camera. Generating surface chunks first
    // means the player sees terrain features appear immediately rather than
    // waiting for an avalanche of subsurface AllSolid / above-surface AllAir
    // chunks to finish.
    //
    // The previous heuristic used `c.layer == 0` (mean-radius layer) as a
    // "surface" proxy, which is wrong on Earth-scale planets: the actual
    // surface layer can be hundreds of layers above or below 0 depending on
    // terrain height, so the sort effectively ignored surface proximity and
    // generation order was dominated by raw camera distance.
    let base_fce = CubeSphereCoord::face_chunks_per_edge(mean_radius);
    let cam_coord_l0 = world_pos_to_coord(cam_world_pos.0, mean_radius, base_fce);
    let tgen_for_sort = terrain_gen.0.clone();
    // Compute sort keys ONCE per candidate (sample_surface_radius_at is
    // expensive — calling it inside `sort_unstable_by_key` re-runs it on
    // every comparison and tanks FPS during loading).
    let mut keyed: Vec<(i32, CubeSphereCoord)> = to_dispatch
        .into_iter()
        .map(|c| {
            let lod_penalty = (c.lod as i32) * 4_000_000;
            let fce_lod = CubeSphereCoord::face_chunks_per_edge_lod(mean_radius, c.lod);
            let chunk_world_size = CHUNK_SIZE as f64 * (1u64 << c.lod) as f64;
            let dir = c.unit_sphere_dir(fce_lod);
            let (lat, lon) = crate::planet::detail::pos_to_lat_lon(dir);
            let surface_r = tgen_for_sort.sample_surface_radius_at(lat, lon);
            let surface_layer = ((surface_r - mean_radius) / chunk_world_size).round() as i32;
            let layer_dist = (c.layer - surface_layer).abs();
            // Strong bias: surface-adjacent layers come first within each LOD.
            let surface_penalty = layer_dist * 50_000;
            let scale = 1i32 << c.lod;
            let dist = if c.face == cam_coord_l0.face {
                let du = c.u * scale - cam_coord_l0.u;
                let dv = c.v * scale - cam_coord_l0.v;
                du * du + dv * dv
            } else {
                i32::MAX / 8
            };
            (lod_penalty + surface_penalty + dist, c)
        })
        .collect();
    keyed.sort_unstable_by_key(|&(k, _)| k);
    let to_dispatch: Vec<CubeSphereCoord> = keyed.into_iter().map(|(_, c)| c).collect();

    // Separate GPU-eligible and CPU-fallback chunks.
    let dispatched: Vec<CubeSphereCoord> = to_dispatch
        .into_iter()
        .take(budget.min(MAX_CHUNKS_PER_BATCH))
        .collect();

    let dispatched_this_frame = dispatched.len();
    let gpu_in_flight_now = !gpu_tasks_q.is_empty();

    if let Some(ref compute) = gpu_dispatcher.compute {
        // Only dispatch a GPU batch if no previous batch is still in flight —
        // the pre-allocated staging buffers are not thread-safe.
        let gpu_in_flight = gpu_in_flight_now;
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

    // Update HUD stats — must happen *after* dispatch so pending counts
    // reflect this frame's newly-spawned tasks.
    let (cache_all_air, cache_all_solid, cache_mixed) = cache.classification_counts();
    let mut loaded_lod0 = 0;
    let mut loaded_lod_max = 0;
    let mut meshed_lod0 = 0;
    let mut meshed_lod_max = 0;
    for c in chunk_map.loaded.keys() {
        if c.lod == 0 {
            loaded_lod0 += 1;
        } else if c.lod == max_lod {
            loaded_lod_max += 1;
        }
    }
    for c in chunk_map.loaded.keys() {
        let is_meshed = !matches!(
            cache.get(c),
            Some(CachedVoxels::AllAir) | Some(CachedVoxels::AllSolid(_))
        );
        if !is_meshed {
            continue;
        }
        if c.lod == 0 {
            meshed_lod0 += 1;
        } else if c.lod == max_lod {
            meshed_lod_max += 1;
        }
    }

    *stats = V2PipelineStats {
        desired: desired.len(),
        desired_lod0,
        desired_lod_max,
        pending_terrain: pending_terrain.pending.len(),
        pending_meshes: pending_meshes.pending.len(),
        cache_entries: cache.entry_count(),
        cache_all_air,
        cache_all_solid,
        cache_mixed,
        loaded: chunk_map.loaded.len(),
        loaded_lod0,
        loaded_lod_max,
        meshed_lod0,
        meshed_lod_max,
        dispatched_this_frame,
        gpu_in_flight: gpu_in_flight_now,
    };
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
            // `slices[dir]` must contain the layer of the neighbor that is
            // directly adjacent to the current chunk's `dir` face. For +X
            // (dir=0) that is the +X-neighbor's x=0 column; for -X (dir=1)
            // it is the -X-neighbor's x=CS-1 column; etc.
            //
            // `extract_edge_slice` is already written in this convention
            // (dir=0 → x=0, dir=1 → x=CS-1, dir=2 → y=0, dir=3 → y=CS-1,
            // dir=4 → z=0, dir=5 → z=CS-1), so we pass `dir` directly
            // rather than inverting it.
            *slot = Some(extract_edge_slice(cached, dir));
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
    mut chunk_map: ResMut<V2ChunkMap>,
    mut invalidation_queue: ResMut<V2InvalidationQueue>,
    terrain_gen: Res<V2TerrainGen>,
    planet: Res<PlanetConfig>,
    color_map: Res<MaterialColorMap>,
    camera_q: Query<&WorldPosition, With<FpsCamera>>,
    mut task_q: Query<(Entity, &mut V2TerrainTask)>,
) {
    let mean_radius = planet.mean_radius;
    let cam_world_pos = camera_q.single().ok().copied();

    // Stage 1: collect completed terrain tasks
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

        // Store in cache and queue for neighbour invalidation.
        cache.insert(result.coord, result.voxels);
        invalidation_queue.coords.push(result.coord);
    }

    // Stage 1b: remesh-invalidate same-face neighbours of every chunk that
    // was newly cached this frame (from either CPU or GPU path).
    //
    // A neighbour that already meshed against a *resampled* boundary slice
    // (because this chunk wasn't cached yet) will have inaccurate geometry
    // on the seam facing this newly-arrived chunk. Despawning its entity
    // and removing it from `loaded` makes it a candidate for re-dispatch
    // in Stage 2 below, where it will mesh against the now-cached real
    // boundary data.
    if !invalidation_queue.coords.is_empty() {
        // Take ownership of the queue so we can iterate it and clear it.
        let drained: Vec<CubeSphereCoord> = invalidation_queue.coords.drain(..).collect();
        for coord in drained {
            let coord_fce = CubeSphereCoord::face_chunks_per_edge_lod(mean_radius, coord.lod);
            let max_uv = coord_fce as i32;
            for dir in 0..6 {
                if let Some(nc) = same_face_neighbor_for_dir(coord, dir, max_uv)
                    && let Some(ent) = chunk_map.loaded.remove(&nc)
                {
                    commands.entity(ent).despawn();
                }
            }
        }
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
        .filter(|c| !chunk_map.loaded.contains_key(c) && !pending_meshes.pending.contains(c))
        .copied()
        .collect();

    // Priority: Mixed chunks (the visually relevant ones) before AllAir /
    // AllSolid (which usually emit empty meshes). Then by camera distance so
    // near terrain meshes before far terrain. HashMap iteration is otherwise
    // unordered, which made the player wait on far / subsurface chunks while
    // near surface chunks sat idle in cache.
    //
    // Precompute keys once per candidate; doing this work inside
    // `sort_unstable_by_key` re-runs the cache lookup on every comparison
    // and contributes to FPS drops while ~5000 chunks are queued.
    let base_fce = CubeSphereCoord::face_chunks_per_edge(mean_radius);
    let cam_coord_for_mesh = cam_world_pos.map(|p| world_pos_to_coord(p.0, mean_radius, base_fce));
    let mut keyed_mesh: Vec<(i32, CubeSphereCoord)> = candidates
        .into_iter()
        .map(|c| {
            let class_penalty = match cache.get(&c) {
                Some(CachedVoxels::Mixed(_)) => 0,
                Some(CachedVoxels::AllSolid(_)) => 10_000_000,
                Some(CachedVoxels::AllAir) => 20_000_000,
                None => 30_000_000,
            };
            let lod_penalty = (c.lod as i32) * 2_000_000;
            let dist = if let Some(cam) = cam_coord_for_mesh
                && c.face == cam.face
            {
                let scale = 1i32 << c.lod;
                let du = c.u * scale - cam.u;
                let dv = c.v * scale - cam.v;
                let dl = c.layer - cam.layer;
                du * du + dv * dv + dl * dl
            } else {
                i32::MAX / 8
            };
            (class_penalty + lod_penalty + dist, c)
        })
        .collect();
    keyed_mesh.sort_unstable_by_key(|&(k, _)| k);
    let candidates: Vec<CubeSphereCoord> = keyed_mesh.into_iter().map(|(_, c)| c).collect();

    for (dispatched, coord) in candidates.into_iter().enumerate() {
        if dispatched >= mesh_budget {
            break;
        }

        let voxel_data = cache.get(&coord).unwrap().clone();

        // Fast-path: AllAir chunks can never emit any face. Skip the
        // mesher entirely and produce an empty mesh.
        //
        // AllSolid chunks no longer have a special gate. Surface Nets
        // with the unified density SDF naturally emits zero quads when
        // the chunk is surrounded by uniform same-density material
        // (no sign change anywhere), and emits the appropriate face
        // when a neighbour exposes a different material. The previous
        // "visible-boundary" gate caused two classes of bugs (provisional
        // boundary data + permanent empty-mesh lock-in) without saving
        // meaningful work.
        if matches!(voxel_data, CachedVoxels::AllAir) {
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
            let mesh = surface_nets::surface_nets_mesh(&voxels, &neighbor_slices, &cmap);
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
    mut invalidation_queue: ResMut<V2InvalidationQueue>,
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
            // Queue neighbour invalidation; v2_collect_terrain will drain
            // this queue and despawn stale-meshed neighbours before its
            // own dispatch stage.
            invalidation_queue.coords.push(result.coord);
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
            let ent = commands
                .spawn((
                    V2ChunkMarker,
                    V2ChunkCoord(result.coord),
                    Transform::from_translation(adjusted)
                        .with_rotation(rotation)
                        .with_scale(tangent_scale),
                ))
                .id();
            chunk_map.loaded.insert(result.coord, ent);
            continue;
        }

        let bevy_mesh = chunk_mesh_to_bevy_mesh(result.mesh);
        let mesh_handle = meshes.add(bevy_mesh);

        let ent = commands
            .spawn((
                V2ChunkMarker,
                V2ChunkCoord(result.coord),
                ChunkMeshMarker,
                Mesh3d(mesh_handle),
                MeshMaterial3d(chunk_material.clone()),
                Transform::from_translation(adjusted)
                    .with_rotation(rotation)
                    .with_scale(tangent_scale),
            ))
            .id();

        chunk_map.loaded.insert(result.coord, ent);
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
            .init_resource::<V2InvalidationQueue>()
            .init_resource::<V2VoxelCache>()
            .init_resource::<GpuTerrainDispatcher>()
            .init_resource::<GpuHeightmapInjected>()
            .init_resource::<V2PipelineStats>()
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
                Update,
                try_inject_gpu_heightmap.run_if(not(resource_equals(GpuHeightmapInjected(true)))),
            )
            .add_systems(
                PostUpdate,
                v2_apply_origin_shift.run_if(resource_exists::<RenderOriginShift>),
            );
    }
}

/// Periodic diagnostic logging for the V2 pipeline.
#[allow(clippy::too_many_arguments)]
fn v2_diagnostics(
    chunk_map: Res<V2ChunkMap>,
    pending_terrain: Res<V2PendingTerrain>,
    pending_meshes: Res<V2PendingMeshes>,
    cache: Res<V2VoxelCache>,
    stats: Res<V2PipelineStats>,
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
    // Mirror the F3 HUD line to stdout so it can be copy/pasted easily.
    info!(
        "V2 desired:{} (L0:{} Lmax:{}) pendT:{} pendM:{} cache:{} (air:{} solid:{} mix:{}) loaded:{} (L0:{} Lmax:{}) meshed:(L0:{} Lmax:{}) disp/f:{}{}",
        stats.desired,
        stats.desired_lod0,
        stats.desired_lod_max,
        stats.pending_terrain,
        stats.pending_meshes,
        stats.cache_entries,
        stats.cache_all_air,
        stats.cache_all_solid,
        stats.cache_mixed,
        stats.loaded,
        stats.loaded_lod0,
        stats.loaded_lod_max,
        stats.meshed_lod0,
        stats.meshed_lod_max,
        stats.dispatched_this_frame,
        if stats.gpu_in_flight { " gpu*" } else { "" },
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
    planetary: Option<Res<crate::world::PlanetaryData>>,
    mut gpu_dispatcher: ResMut<GpuTerrainDispatcher>,
) {
    if let Some(shared) = shared {
        commands.insert_resource(V2TerrainGen(shared.0.clone()));
        info!("V2 pipeline: initialized terrain generator from SharedTerrainGen");
    } else {
        // Fallback: create from PlanetaryData if available, else placeholder.
        let planet_data = if let Some(ref pd) = planetary {
            pd.0.clone()
        } else {
            let gen_cfg = crate::planet::PlanetConfig {
                seed: planet.seed as u64,
                grid_level: 3,
                ..Default::default()
            };
            Arc::new(crate::planet::PlanetData::new(gen_cfg))
        };
        let unified = Arc::new(crate::world::terrain::UnifiedTerrainGenerator::new(
            planet_data,
            planet.clone(),
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

/// One-shot system that bakes the planetary heightmap and uploads it to the GPU.
///
/// Runs every frame until both `V2TerrainGen` (which becomes available once
/// `PlanetaryData` is inserted) and a GPU pipeline are present.  The bake is
/// offloaded to `AsyncComputeTaskPool` so it does not block the render thread.
/// After the async task completes it writes directly into the `GpuVoxelCompute`
/// via `Arc::set_heightmap()` (which takes `&self`).
fn try_inject_gpu_heightmap(
    mut injected: ResMut<GpuHeightmapInjected>,
    terrain_gen: Option<Res<V2TerrainGen>>,
    gpu_dispatcher: Res<GpuTerrainDispatcher>,
) {
    if injected.0 {
        return;
    }
    let Some(tgen) = terrain_gen else { return };
    let Some(compute) = gpu_dispatcher.compute.clone() else {
        return;
    };

    injected.0 = true;

    // Clone the Arc so the async task owns its own reference.
    let gen_arc = tgen.0.clone();
    let pool = AsyncComputeTaskPool::get();
    pool.spawn(async move {
        use crate::planet::gpu_heightmap::bake_elevation_map;
        info!(
            "GPU heightmap: baking {}×{} elevation map from PlanetaryTerrainSampler…",
            crate::planet::gpu_heightmap::HEIGHTMAP_WIDTH,
            crate::planet::gpu_heightmap::HEIGHTMAP_HEIGHT,
        );
        let data = bake_elevation_map(&gen_arc.0);
        compute.set_heightmap(&data);
        info!(
            "GPU heightmap: uploaded {} floats — heightmap mode active",
            data.len()
        );
    })
    .detach();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::terrain::UnifiedTerrainGenerator;
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

        let tgen = {
            let gen_cfg = crate::planet::PlanetConfig {
                seed: planet.seed as u64,
                grid_level: 3,
                ..Default::default()
            };
            let pd = Arc::new(crate::planet::PlanetData::new(gen_cfg));
            Arc::new(UnifiedTerrainGenerator::new(pd, planet.clone()))
        };
        app.insert_resource(planet);
        app.insert_resource(V2TerrainGen(tgen));
        app.insert_resource(V2LoadRadius {
            horizontal: 2,
            vertical: 1,
        });
        app.init_resource::<V2ChunkMap>();
        app.init_resource::<V2PendingTerrain>();
        app.init_resource::<V2PendingMeshes>();
        app.init_resource::<V2InvalidationQueue>();
        app.init_resource::<V2VoxelCache>();
        app.init_resource::<GpuTerrainDispatcher>();
        app.init_resource::<V2PipelineStats>();
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

    /// Regression: the cached fast-path in `build_neighbor_slices` must
    /// return the layer of the neighbor chunk that is **adjacent** to the
    /// current chunk's `dir` face — i.e. the neighbor's x=0 column for
    /// dir=0 (+X neighbor), y=0 for dir=2 (+Y neighbor), etc.
    ///
    /// A previous bug XOR-inverted the direction (`opposite_dir = dir ^ 1`),
    /// causing the cache path to return the **far** edge of the neighbor
    /// (x=CS-1 for the +X neighbor) instead.  At chunk boundaries that meant
    /// greedy meshing saw the wrong material beyond the boundary, producing
    /// missing / spurious faces (visible in-game as transparent squares
    /// with terrain far below).
    #[test]
    fn build_neighbor_slices_uses_adjacent_neighbor_edge() {
        use crate::world::v2::cubed_sphere::CubeFace;
        use crate::world::v2::terrain_gen::CachedVoxels;

        // Earth-scale radius so chunks are Mixed and neighbors are on the
        // same face.
        let cfg = PlanetConfig {
            mean_radius: 6_371_000.0,
            sea_level_radius: 6_371_000.0,
            height_scale: 50.0,
            ..Default::default()
        };
        let tgen = {
            let gen_cfg = crate::planet::PlanetConfig {
                seed: cfg.seed as u64,
                grid_level: 3,
                ..Default::default()
            };
            let pd = Arc::new(crate::planet::PlanetData::new(gen_cfg));
            UnifiedTerrainGenerator::new(pd, cfg.clone())
        };
        let fce = CubeSphereCoord::face_chunks_per_edge(cfg.mean_radius);
        let max_uv = fce as i32;

        let center_uv = max_uv / 2;
        let coord = CubeSphereCoord::new_with_lod(CubeFace::PosX, center_uv, center_uv, 0, 0);

        // Populate cache with all 6 same-face neighbors (generated from the
        // real terrain so they contain a realistic mix of air/solid).
        let mut cache = V2VoxelCache::default();
        for dir in 0..6usize {
            let nc = same_face_neighbor_for_dir(coord, dir, max_uv)
                .expect("face-interior coord must have all 6 same-face neighbors");
            let data = generate_v2_voxels(nc, cfg.mean_radius, fce, &tgen);
            cache.insert(nc, data.voxels);
        }

        let slices = build_neighbor_slices(coord, &cache, cfg.mean_radius, fce, &tgen);

        // Track whether at least one direction has near ≠ far, so the test
        // is not vacuously passing.
        let mut at_least_one_distinguishes = false;

        for dir in 0..6usize {
            let nc = same_face_neighbor_for_dir(coord, dir, max_uv).unwrap();
            let cached = cache.get(&nc).unwrap();
            let expected = extract_edge_slice(cached, dir);
            let actual = slices.slices[dir]
                .as_ref()
                .expect("same-face neighbor must produce a cached slice");

            if let CachedVoxels::Mixed(_) = cached {
                let opposite = extract_edge_slice(cached, dir ^ 1);
                let differs = actual
                    .iter()
                    .zip(opposite.iter())
                    .any(|(a, b)| a.material != b.material);
                if differs {
                    at_least_one_distinguishes = true;
                }
            }

            for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
                assert_eq!(
                    a.material, e.material,
                    "dir={dir} voxel {i}: build_neighbor_slices returned a \
                     voxel that doesn't match the neighbor's adjacent edge",
                );
            }
        }

        assert!(
            at_least_one_distinguishes,
            "no direction had distinguishable near/far edges — test is \
             vacuous and cannot catch the dir-inversion regression",
        );
    }

    /// Regression test for the "square void around player at high altitude"
    /// bug (checkpoint 028).
    ///
    /// At Earth-scale (mean_radius = 6_371_000 m), a player floating 6.6 km
    /// above the surface has `cam_layer ≈ 218`.  Before the fix,
    /// `desired_chunks_v2` clamped the camera band with `MAX_VERTICAL_LAYERS
    /// = 64`, producing an inverted range `(216, 64)` that contributed no
    /// chunks, and unioned the surface band (≈ layer 10) with the clamped
    /// camera upper bound (64) — so no chunk within the player's own layer
    /// was ever loaded, leaving a rectangular void exactly where the player
    /// was standing.
    #[test]
    fn desired_chunks_includes_camera_at_high_altitude_earth_scale() {
        let cfg = PlanetConfig {
            mean_radius: 6_371_000.0,
            sea_level_radius: 6_371_000.0,
            height_scale: 0.0, // flat planet → deterministic surface_r == mean_radius
            noise: None,
            ..Default::default()
        };
        let radius = V2LoadRadius {
            horizontal: 4,
            vertical: 2,
        };

        // Place the camera ~6.6 km above the surface on the +X face.
        // This reproduces the altitude the user reported in the void bug.
        let altitude = 6600.0_f64;
        let cam_pos = DVec3::new(cfg.mean_radius + altitude, 0.0, 0.0);

        let tgen = {
            let gen_cfg = crate::planet::PlanetConfig {
                seed: cfg.seed as u64,
                grid_level: 3,
                ..Default::default()
            };
            let pd = Arc::new(crate::planet::PlanetData::new(gen_cfg));
            UnifiedTerrainGenerator::new(pd, cfg.clone())
        };
        let desired = desired_chunks_v2(cam_pos, &cfg, &radius, Some(&tgen));

        // The camera's own L0 chunk MUST be in the desired set.
        let base_fce = CubeSphereCoord::face_chunks_per_edge(cfg.mean_radius);
        let cam_coord = world_pos_to_coord(cam_pos, cfg.mean_radius, base_fce);
        assert!(
            cam_coord.layer > 64,
            "precondition: altitude {altitude} m should put cam_layer \
             ({}) above the old MAX_VERTICAL_LAYERS cap of 64 — otherwise \
             the test cannot catch the regression",
            cam_coord.layer
        );
        assert!(
            desired.contains(&cam_coord),
            "camera's own chunk (face={:?}, u={}, v={}, layer={}) must be \
             in the desired set — otherwise the player sits in a void",
            cam_coord.face,
            cam_coord.u,
            cam_coord.v,
            cam_coord.layer,
        );

        // And at least one chunk at the SURFACE layer of the camera's
        // column must also be present (so the ground below the player
        // still loads as well).
        let has_surface_chunk_under_camera = desired.iter().any(|c| {
            c.lod == 0
                && c.face == cam_coord.face
                && c.u == cam_coord.u
                && c.v == cam_coord.v
                && c.layer.abs() <= 2
        });
        assert!(
            has_surface_chunk_under_camera,
            "surface chunk in the camera's own column must also be loaded",
        );
    }
}
