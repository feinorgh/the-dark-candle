// Chunk manager: loads and unloads chunks around the camera.
//
// Each frame the manager compares the set of currently loaded chunks against
// the set that *should* be loaded. New chunks are dispatched for async terrain
// generation on the `AsyncComputeTaskPool`, and far-away chunks are despawned.
//
// Two loading modes:
//   **Flat** (legacy): cylindrical loading around camera in XZ plane.
//   **Spherical**: shell-based loading — only chunks near the planet surface
//     are loaded, skipping deep interior and outer space.

use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, poll_once};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::chunk::{CHUNK_SIZE, Chunk, ChunkCoord, ChunkOctree};
use super::planet::{PlanetConfig, TerrainMode};
use super::refinement::{SubdivisionConfig, analyze_chunk, build_refined_octree};
use super::terrain::UnifiedTerrainGenerator;
use crate::camera::FpsCamera;
use crate::chemistry::runtime::ChunkActivity;
use crate::gpu::noise_compute::GpuNoiseCompute;

/// How many chunks outward from the camera to load in each axis.
#[derive(Resource)]
pub struct ChunkLoadRadius {
    /// Horizontal (XZ) radius in chunk units.
    pub horizontal: i32,
    /// Vertical (Y) radius in chunk units.
    pub vertical: i32,
    /// For spherical mode: how many meters below the surface to load.
    pub shell_depth: f64,
    /// For spherical mode: how many meters above the surface to load.
    pub shell_height: f64,
}

impl Default for ChunkLoadRadius {
    fn default() -> Self {
        Self {
            horizontal: 4,
            vertical: 2,
            shell_depth: 128.0,  // 4 chunks deep
            shell_height: 128.0, // 4 chunks high
        }
    }
}

/// Minimum horizontal chunk radius. Never go below this.
const MIN_CHUNK_RADIUS: i32 = 2;

/// Maximum horizontal chunk radius. Never exceed this.
const MAX_CHUNK_RADIUS: i32 = 12;

/// Number of consecutive frames over/under budget before adjusting radius.
const ADAPTATION_HYSTERESIS: u32 = 60;

/// Headroom threshold below which we shrink (negative = over budget).
const SHRINK_THRESHOLD: f32 = -0.05;

/// Headroom threshold above which we grow.
const GROW_THRESHOLD: f32 = 0.30;

/// Maximum number of terrain generation tasks dispatched per frame.
const MAX_TERRAIN_DISPATCHES_PER_FRAME: usize = 8;

/// Tracks consecutive frames for hysteresis-based view distance adaptation.
#[derive(Resource, Debug, Default)]
pub struct ViewDistanceState {
    /// Consecutive frames where headroom < SHRINK_THRESHOLD.
    pub frames_over_budget: u32,
    /// Consecutive frames where headroom > GROW_THRESHOLD.
    pub frames_under_budget: u32,
}

/// Set of chunk coordinates whose terrain generation is in flight.
/// Prevents double-dispatch while an async task is still running.
#[derive(Resource, Default)]
pub struct PendingChunks {
    coords: HashSet<ChunkCoord>,
}

impl PendingChunks {
    pub fn contains(&self, coord: &ChunkCoord) -> bool {
        self.coords.contains(coord)
    }

    pub fn insert(&mut self, coord: ChunkCoord) {
        self.coords.insert(coord);
    }

    pub fn remove(&mut self, coord: &ChunkCoord) {
        self.coords.remove(coord);
    }

    pub fn len(&self) -> usize {
        self.coords.len()
    }

    pub fn is_empty(&self) -> bool {
        self.coords.is_empty()
    }
}

/// Output of an async terrain generation task.
struct TerrainGenResult {
    coord: ChunkCoord,
    chunk: Chunk,
    octree: ChunkOctree,
    biome_data: Option<crate::world::planetary_sampler::ChunkBiomeData>,
}

/// Component holding an in-flight async terrain generation task.
#[derive(Component)]
pub struct TerrainGenTask(Task<TerrainGenResult>);

/// Thread-safe handle to the terrain generator for async tasks.
#[derive(Resource, Clone)]
pub struct SharedTerrainGen(pub Arc<UnifiedTerrainGenerator>);

/// Thread-safe handle to the subdivision config for async tasks.
#[derive(Resource, Clone)]
pub struct SharedSubdivConfig(pub Arc<SubdivisionConfig>);

/// Optional GPU noise compute for terrain surface-radius evaluation.
///
/// When present, the chunk dispatch system batches column (lon, lat) pairs
/// and evaluates them on the GPU, freeing the CPU thread pool for meshing,
/// underground voxel fill, and physics.
#[derive(Resource)]
pub struct SharedGpuNoise(pub Option<GpuNoiseCompute>);

/// Maps chunk coordinates to their ECS entity for O(1) lookup.
#[derive(Resource, Default)]
pub struct ChunkMap {
    map: HashMap<ChunkCoord, Entity>,
}

impl ChunkMap {
    pub fn get(&self, coord: &ChunkCoord) -> Option<Entity> {
        self.map.get(coord).copied()
    }

    pub fn contains(&self, coord: &ChunkCoord) -> bool {
        self.map.contains_key(coord)
    }

    pub fn insert(&mut self, coord: ChunkCoord, entity: Entity) {
        self.map.insert(coord, entity);
    }

    pub fn remove(&mut self, coord: &ChunkCoord) -> Option<Entity> {
        self.map.remove(coord)
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn clear(&mut self) {
        self.map.clear();
    }

    pub fn coords(&self) -> impl Iterator<Item = &ChunkCoord> {
        self.map.keys()
    }
}

/// Compute the set of chunk coordinates that should be loaded around a world position.
/// This is the **flat-mode** (legacy) loader using cylindrical distance.
pub fn desired_chunks(world_pos: Vec3, radius: &ChunkLoadRadius) -> HashSet<ChunkCoord> {
    let cs = CHUNK_SIZE as f32;
    let center_cx = (world_pos.x / cs).floor() as i32;
    let center_cy = (world_pos.y / cs).floor() as i32;
    let center_cz = (world_pos.z / cs).floor() as i32;

    let h = radius.horizontal;
    let v = radius.vertical;
    let h_sq = (h * h) as f32;

    let mut set = HashSet::new();
    for dy in -v..=v {
        for dz in -h..=h {
            for dx in -h..=h {
                // Use cylindrical distance (XZ plane) for more natural loading
                let dist_sq = (dx * dx + dz * dz) as f32;
                if dist_sq <= h_sq {
                    set.insert(ChunkCoord::new(
                        center_cx + dx,
                        center_cy + dy,
                        center_cz + dz,
                    ));
                }
            }
        }
    }
    set
}

/// Compute the set of chunk coordinates for **spherical** shell-based loading.
///
/// Loads chunks in a spherical shell around the camera, constrained to
/// chunks whose centers fall within `[cam_r - depth, cam_r + height]`
/// of the planet center, where `cam_r` is the camera's radial distance.
/// This naturally tracks the local terrain surface regardless of where the
/// camera is relative to `mean_radius`.  Only chunks within `horizontal`
/// chunk-units of the camera are considered (Cartesian chunk-space distance).
pub fn desired_chunks_spherical(
    world_pos: Vec3,
    radius: &ChunkLoadRadius,
    planet: &PlanetConfig,
) -> HashSet<ChunkCoord> {
    let cs = CHUNK_SIZE as f32;
    let center_cx = (world_pos.x / cs).floor() as i32;
    let center_cy = (world_pos.y / cs).floor() as i32;
    let center_cz = (world_pos.z / cs).floor() as i32;

    let h = radius.horizontal;
    let h_sq = (h * h) as f32;

    // Center the shell on the camera's current radial distance so the loaded
    // region tracks the local terrain surface, not the mean radius.
    let cam_r = world_pos.length() as f64;
    // Use at least the configured depth/height, but widen if height_scale is
    // large to ensure we capture nearby terrain variation.
    let extra = planet.height_scale * 0.05; // 5% of height_scale as margin
    let depth = radius.shell_depth.max(extra);
    let height = radius.shell_height.max(extra);
    let shell_min = (cam_r - depth) as f32;
    let shell_max = (cam_r + height) as f32;

    let mut set = HashSet::new();
    for dz in -h..=h {
        for dy in -h..=h {
            for dx in -h..=h {
                let dist_sq = (dx * dx + dy * dy + dz * dz) as f32;
                if dist_sq > h_sq {
                    continue;
                }

                let cx = center_cx + dx;
                let cy = center_cy + dy;
                let cz = center_cz + dz;

                // Check if chunk center is within the surface shell
                let chunk_center = Vec3::new(
                    (cx as f32 + 0.5) * cs,
                    (cy as f32 + 0.5) * cs,
                    (cz as f32 + 0.5) * cs,
                );
                let r = chunk_center.length();
                if r >= shell_min && r <= shell_max {
                    set.insert(ChunkCoord::new(cx, cy, cz));
                }
            }
        }
    }
    set
}

/// System: despawns out-of-range chunks and dispatches async terrain generation
/// for newly needed chunks, prioritized by distance to the camera.
#[allow(clippy::too_many_arguments)]
pub fn update_chunks(
    mut commands: Commands,
    mut chunk_map: ResMut<ChunkMap>,
    mut pending: ResMut<PendingChunks>,
    radius: Res<ChunkLoadRadius>,
    shared_gen: Res<SharedTerrainGen>,
    shared_subdiv: Res<SharedSubdivConfig>,
    gpu_noise: Res<SharedGpuNoise>,
    planet: Res<PlanetConfig>,
    camera_q: Query<&Transform, With<FpsCamera>>,
    chunk_props_q: Query<&crate::procgen::props::ChunkProps>,
    chunk_creatures_q: Query<&crate::procgen::creatures::ChunkCreatures>,
    chunk_items_q: Query<&crate::procgen::items::ChunkItems>,
) {
    let Ok(cam_transform) = camera_q.single() else {
        return;
    };

    let desired = match planet.mode {
        TerrainMode::Flat => desired_chunks(cam_transform.translation, &radius),
        TerrainMode::Spherical => {
            desired_chunks_spherical(cam_transform.translation, &radius, &planet)
        }
    };

    // Despawn chunks no longer in range
    let loaded: Vec<ChunkCoord> = chunk_map.coords().copied().collect();
    for coord in loaded {
        if !desired.contains(&coord)
            && let Some(entity) = chunk_map.remove(&coord)
        {
            // Despawn prop entities tracked by this chunk
            if let Ok(chunk_props) = chunk_props_q.get(entity) {
                for &prop_entity in &chunk_props.entities {
                    commands.entity(prop_entity).despawn();
                }
            }
            // Despawn creature entities tracked by this chunk
            if let Ok(chunk_creatures) = chunk_creatures_q.get(entity) {
                for &creature_entity in &chunk_creatures.entities {
                    commands.entity(creature_entity).despawn();
                }
            }
            // Despawn item entities tracked by this chunk
            if let Ok(chunk_items) = chunk_items_q.get(entity) {
                for &item_entity in &chunk_items.entities {
                    commands.entity(item_entity).despawn();
                }
            }
            commands.entity(entity).despawn();
        }
    }

    // Cancel pending generation for chunks that left the desired set.
    let stale: Vec<ChunkCoord> = pending
        .coords
        .iter()
        .filter(|c| !desired.contains(c))
        .copied()
        .collect();
    for coord in stale {
        pending.remove(&coord);
    }

    // Collect coords that need generation, sorted closest-to-camera first.
    let cam_pos = cam_transform.translation;
    let mut to_generate: Vec<ChunkCoord> = desired
        .iter()
        .filter(|c| !chunk_map.contains(c) && !pending.contains(c))
        .copied()
        .collect();
    to_generate.sort_by(|a, b| {
        let da = a.world_center().distance_squared(cam_pos);
        let db = b.world_center().distance_squared(cam_pos);
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Dispatch async terrain generation for the closest chunks.
    let pool = AsyncComputeTaskPool::get();

    let batch: Vec<ChunkCoord> = to_generate
        .into_iter()
        .take(MAX_TERRAIN_DISPATCHES_PER_FRAME)
        .collect();

    // GPU batch: compute surface radii for all columns if GPU noise is available
    // and we're in spherical mode.
    let gpu_height_map: HashMap<ChunkCoord, Vec<f32>> = if let Some(ref gpu) = gpu_noise.0 {
        if matches!(planet.mode, TerrainMode::Spherical) {
            gpu_batch_surface_radii(gpu, &planet, &batch)
        } else {
            HashMap::new()
        }
    } else {
        HashMap::new()
    };

    for coord in batch {
        let terrain_gen = shared_gen.0.clone();
        let subdiv = shared_subdiv.0.clone();
        let heights = gpu_height_map.get(&coord).cloned();

        let task = pool.spawn(async move {
            let mut chunk = Chunk::new_empty(coord);
            let biome_data = if let Some(h) = heights {
                terrain_gen.generate_chunk_with_gpu_heights(&mut chunk, &h)
            } else {
                terrain_gen.generate_chunk(&mut chunk)
            };
            let analysis = analyze_chunk(&chunk, &subdiv);
            let octree = build_refined_octree(chunk.voxels(), CHUNK_SIZE, &analysis);
            TerrainGenResult {
                coord,
                chunk,
                octree: ChunkOctree(octree),
                biome_data,
            }
        });

        commands.spawn(TerrainGenTask(task));
        pending.insert(coord);
    }
}

/// Batch-compute surface radii on the GPU for a set of chunks.
///
/// For each chunk, computes (lon, lat) for every column, dispatches a single
/// GPU compute call, and splits the results back by chunk.
fn gpu_batch_surface_radii(
    gpu: &GpuNoiseCompute,
    planet: &PlanetConfig,
    chunks: &[ChunkCoord],
) -> HashMap<ChunkCoord, Vec<f32>> {
    if chunks.is_empty() {
        return HashMap::new();
    }

    let mut all_columns: Vec<[f32; 2]> = Vec::with_capacity(chunks.len() * CHUNK_SIZE * CHUNK_SIZE);
    let mut chunk_ranges: Vec<(ChunkCoord, usize, usize)> = Vec::with_capacity(chunks.len());

    for &coord in chunks {
        let origin = coord.world_origin();
        let mid_y = (origin.y + CHUNK_SIZE as i32 / 2) as f64;
        let start = all_columns.len();

        for lz in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                let wx = (origin.x + lx as i32) as f64;
                let wz = (origin.z + lz as i32) as f64;
                let pos = bevy::math::DVec3::new(wx, mid_y, wz);
                let (lat, lon) = planet.lat_lon(pos);
                all_columns.push([lon as f32, lat as f32]);
            }
        }

        chunk_ranges.push((coord, start, all_columns.len()));
    }

    let all_heights = gpu.evaluate_batch(&all_columns);

    chunk_ranges
        .into_iter()
        .map(|(coord, start, end)| (coord, all_heights[start..end].to_vec()))
        .collect()
}

/// System: collects completed async terrain generation tasks and spawns the
/// chunk entities with all required components.
pub fn collect_terrain_results(
    mut commands: Commands,
    mut chunk_map: ResMut<ChunkMap>,
    mut pending: ResMut<PendingChunks>,
    mut task_q: Query<(Entity, &mut TerrainGenTask)>,
    mut chunk_q: Query<&mut Chunk>,
) {
    /// The 6 face-neighbor offsets.
    const FACE_NEIGHBORS: [[i32; 3]; 6] = [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ];

    for (task_entity, mut gen_task) in task_q.iter_mut() {
        let Some(result) = block_on(poll_once(&mut gen_task.0)) else {
            continue;
        };

        // Task entity is just a carrier — despawn it.
        commands.entity(task_entity).despawn();
        pending.remove(&result.coord);

        // Skip if chunk was already loaded (e.g. race between despawn/respawn).
        if chunk_map.contains(&result.coord) {
            continue;
        }

        // Mark all 6 face-neighbor chunks dirty so they remesh with our
        // newly-available boundary data (fixes seams from incomplete neighbors).
        for [dx, dy, dz] in FACE_NEIGHBORS {
            let nc = ChunkCoord::new(
                result.coord.x + dx,
                result.coord.y + dy,
                result.coord.z + dz,
            );
            if let Some(neighbor_entity) = chunk_map.get(&nc)
                && let Ok(mut neighbor_chunk) = chunk_q.get_mut(neighbor_entity)
            {
                neighbor_chunk.mark_dirty();
            }
        }

        let origin = result.coord.world_origin();
        let mut entity_cmds = commands.spawn((
            result.chunk,
            result.coord,
            result.octree,
            ChunkActivity::default(),
            crate::procgen::props::NeedsDecoration,
            crate::procgen::props::ChunkProps::default(),
            crate::procgen::creatures::NeedsCreatureSpawning,
            crate::procgen::creatures::ChunkCreatures::default(),
            crate::procgen::items::NeedsItemSpawning,
            crate::procgen::items::ChunkItems::default(),
            crate::physics::amr_fluid::injection::NeedsFluidSeeding,
            Transform::from_xyz(origin.x as f32, origin.y as f32, origin.z as f32),
        ));
        if let Some(biome) = result.biome_data {
            entity_cmds.insert(biome);
        }
        let entity = entity_cmds.id();
        chunk_map.insert(result.coord, entity);
    }
}

/// Adjusts `ChunkLoadRadius` based on frame budget headroom.
///
/// Uses hysteresis to avoid thrashing: radius only changes after
/// `ADAPTATION_HYSTERESIS` consecutive frames over/under budget.
pub fn adapt_view_distance(
    budget: Option<Res<crate::diagnostics::frame_budget::FrameBudget>>,
    mut radius: ResMut<ChunkLoadRadius>,
    mut state: ResMut<ViewDistanceState>,
) {
    let Some(budget) = budget else { return };

    if budget.headroom < SHRINK_THRESHOLD {
        state.frames_over_budget += 1;
        state.frames_under_budget = 0;
    } else if budget.headroom > GROW_THRESHOLD {
        state.frames_under_budget += 1;
        state.frames_over_budget = 0;
    } else {
        // In the "OK" zone — reset both counters
        state.frames_over_budget = 0;
        state.frames_under_budget = 0;
    }

    if state.frames_over_budget >= ADAPTATION_HYSTERESIS && radius.horizontal > MIN_CHUNK_RADIUS {
        radius.horizontal -= 1;
        state.frames_over_budget = 0;
        info!(
            "View distance reduced to {} chunks (headroom: {:.0}%)",
            radius.horizontal,
            budget.headroom * 100.0
        );
    }

    if state.frames_under_budget >= ADAPTATION_HYSTERESIS && radius.horizontal < MAX_CHUNK_RADIUS {
        radius.horizontal += 1;
        state.frames_under_budget = 0;
        info!(
            "View distance increased to {} chunks (headroom: {:.0}%)",
            radius.horizontal,
            budget.headroom * 100.0
        );
    }
}

/// Wrapper resource holding the terrain generator.
#[derive(Resource)]
pub struct TerrainGeneratorRes(pub UnifiedTerrainGenerator);

/// Plugin that registers chunk management resources and systems.
pub struct ChunkManagerPlugin;

impl Plugin for ChunkManagerPlugin {
    fn build(&self, app: &mut App) {
        // Create a unified generator from the PlanetConfig resource.
        // The PlanetConfig is already inserted by WorldPlugin before this runs.
        let planet = app
            .world()
            .get_resource::<PlanetConfig>()
            .cloned()
            .unwrap_or_default();
        let generator = UnifiedTerrainGenerator::from_planet_config(&planet);
        let subdiv = app
            .world()
            .get_resource::<SubdivisionConfig>()
            .cloned()
            .unwrap_or_default();

        // Try to initialize GPU noise compute for spherical terrain.
        let gpu_noise = if matches!(planet.mode, TerrainMode::Spherical) {
            planet.noise.as_ref().and_then(|noise_cfg| {
                crate::gpu::GpuContext::try_new().map(|ctx| {
                    info!("GPU noise compute initialized — surface noise offloaded to GPU");
                    GpuNoiseCompute::new(
                        ctx,
                        noise_cfg,
                        planet.seed,
                        planet.mean_radius,
                        planet.height_scale,
                    )
                })
            })
        } else {
            None
        };

        app.init_resource::<ChunkMap>()
            .init_resource::<ChunkLoadRadius>()
            .init_resource::<ViewDistanceState>()
            .init_resource::<PendingChunks>()
            .insert_resource(TerrainGeneratorRes(generator))
            .insert_resource(SharedTerrainGen(Arc::new(
                UnifiedTerrainGenerator::from_planet_config(&planet),
            )))
            .insert_resource(SharedSubdivConfig(Arc::new(subdiv)))
            .insert_resource(SharedGpuNoise(gpu_noise))
            .add_systems(
                Update,
                (
                    adapt_view_distance,
                    (update_chunks, collect_terrain_results)
                        .chain()
                        .in_set(super::WorldSet::ChunkManagement),
                )
                    .chain(),
            );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_map_insert_and_lookup() {
        let mut map = ChunkMap::default();
        assert!(map.is_empty());

        let coord = ChunkCoord::new(1, 2, 3);
        let entity = Entity::from_bits(42);
        map.insert(coord, entity);

        assert_eq!(map.len(), 1);
        assert!(map.contains(&coord));
        assert_eq!(map.get(&coord), Some(entity));
        assert!(!map.contains(&ChunkCoord::new(0, 0, 0)));
    }

    #[test]
    fn chunk_map_remove() {
        let mut map = ChunkMap::default();
        let coord = ChunkCoord::new(0, 0, 0);
        let entity = Entity::from_bits(1);
        map.insert(coord, entity);

        let removed = map.remove(&coord);
        assert_eq!(removed, Some(entity));
        assert!(map.is_empty());
        assert_eq!(map.remove(&coord), None);
    }

    #[test]
    fn desired_chunks_at_origin() {
        let radius = ChunkLoadRadius {
            horizontal: 2,
            vertical: 1,
            ..Default::default()
        };
        let chunks = desired_chunks(Vec3::ZERO, &radius);

        // Center chunk must be included
        assert!(chunks.contains(&ChunkCoord::new(0, 0, 0)));

        // Vertical neighbors
        assert!(chunks.contains(&ChunkCoord::new(0, 1, 0)));
        assert!(chunks.contains(&ChunkCoord::new(0, -1, 0)));

        // Horizontal neighbors within radius
        assert!(chunks.contains(&ChunkCoord::new(1, 0, 0)));
        assert!(chunks.contains(&ChunkCoord::new(-1, 0, 0)));
        assert!(chunks.contains(&ChunkCoord::new(0, 0, 1)));
        assert!(chunks.contains(&ChunkCoord::new(0, 0, -1)));
        assert!(chunks.contains(&ChunkCoord::new(2, 0, 0)));

        // Diagonal at distance sqrt(2*2 + 2*2) = 2.83 > 2, should be excluded
        assert!(!chunks.contains(&ChunkCoord::new(2, 0, 2)));
    }

    #[test]
    fn desired_chunks_count_is_reasonable() {
        let radius = ChunkLoadRadius {
            horizontal: 4,
            vertical: 2,
            ..Default::default()
        };
        let chunks = desired_chunks(Vec3::ZERO, &radius);

        // With h=4, v=2: 5 vertical layers × ~π*4² horizontal ≈ 5*50 = ~250
        // Exact count depends on circular culling
        assert!(chunks.len() > 100, "Too few chunks: {}", chunks.len());
        assert!(chunks.len() < 500, "Too many chunks: {}", chunks.len());
    }

    #[test]
    fn desired_chunks_follows_camera() {
        let radius = ChunkLoadRadius {
            horizontal: 1,
            vertical: 0,
            ..Default::default()
        };

        // Camera at origin → chunks around (0,0,0)
        let set_a = desired_chunks(Vec3::ZERO, &radius);
        assert!(set_a.contains(&ChunkCoord::new(0, 0, 0)));

        // Camera moved far away → chunks shift
        let far = Vec3::new(1000.0, 0.0, 0.0);
        let set_b = desired_chunks(far, &radius);
        let far_cx = (1000.0 / CHUNK_SIZE as f32).floor() as i32;
        assert!(set_b.contains(&ChunkCoord::new(far_cx, 0, 0)));
        assert!(!set_b.contains(&ChunkCoord::new(0, 0, 0)));
    }

    #[test]
    fn desired_chunks_negative_position() {
        let radius = ChunkLoadRadius {
            horizontal: 1,
            vertical: 0,
            ..Default::default()
        };
        let pos = Vec3::new(-100.0, -100.0, -100.0);
        let chunks = desired_chunks(pos, &radius);

        let cx = (-100.0 / CHUNK_SIZE as f32).floor() as i32;
        let cy = (-100.0 / CHUNK_SIZE as f32).floor() as i32;
        let cz = (-100.0 / CHUNK_SIZE as f32).floor() as i32;
        assert!(chunks.contains(&ChunkCoord::new(cx, cy, cz)));
    }

    #[test]
    fn zero_radius_loads_only_center() {
        let radius = ChunkLoadRadius {
            horizontal: 0,
            vertical: 0,
            ..Default::default()
        };
        let chunks = desired_chunks(Vec3::new(16.0, 16.0, 16.0), &radius);
        assert_eq!(chunks.len(), 1);
        assert!(chunks.contains(&ChunkCoord::new(0, 0, 0)));
    }

    #[test]
    fn chunk_load_radius_default() {
        let radius = ChunkLoadRadius::default();
        assert_eq!(radius.horizontal, 4);
        assert_eq!(radius.vertical, 2);
    }

    #[test]
    fn desired_chunks_respects_vertical_limit() {
        let radius = ChunkLoadRadius {
            horizontal: 2,
            vertical: 1,
            ..Default::default()
        };
        let chunks = desired_chunks(Vec3::ZERO, &radius);

        // Y=2 should NOT be loaded (vertical radius is 1)
        assert!(!chunks.contains(&ChunkCoord::new(0, 2, 0)));
        assert!(!chunks.contains(&ChunkCoord::new(0, -2, 0)));

        // Y=1 SHOULD be loaded
        assert!(chunks.contains(&ChunkCoord::new(0, 1, 0)));
        assert!(chunks.contains(&ChunkCoord::new(0, -1, 0)));
    }

    #[test]
    fn desired_chunks_cylindrical_shape() {
        let radius = ChunkLoadRadius {
            horizontal: 3,
            vertical: 0,
            ..Default::default()
        };
        let chunks = desired_chunks(Vec3::ZERO, &radius);

        // On-axis at radius: dx=3, dz=0 → dist²=9 ≤ 9 → included
        assert!(chunks.contains(&ChunkCoord::new(3, 0, 0)));

        // Corner: dx=3, dz=1 → dist²=10 > 9 → excluded
        assert!(!chunks.contains(&ChunkCoord::new(3, 0, 1)));

        // Diagonal: dx=2, dz=2 → dist²=8 ≤ 9 → included
        assert!(chunks.contains(&ChunkCoord::new(2, 0, 2)));
    }

    // -----------------------------------------------------------------------
    // Spherical shell-based loading tests
    // -----------------------------------------------------------------------

    fn small_planet() -> PlanetConfig {
        PlanetConfig {
            mean_radius: 320.0, // 10 chunks radius
            sea_level_radius: 320.0,
            height_scale: 8.0,
            ..PlanetConfig::default()
        }
    }

    #[test]
    fn spherical_loads_surface_chunks() {
        let planet = small_planet();
        // Camera at surface along +X
        let cam = Vec3::new(planet.mean_radius as f32, 0.0, 0.0);
        let radius = ChunkLoadRadius {
            horizontal: 2,
            shell_depth: 64.0,
            shell_height: 64.0,
            ..Default::default()
        };
        let chunks = desired_chunks_spherical(cam, &radius, &planet);
        assert!(!chunks.is_empty(), "Should load chunks near surface");
    }

    #[test]
    fn spherical_excludes_deep_interior() {
        let planet = small_planet();
        // Camera at surface along +X
        let cam = Vec3::new(planet.mean_radius as f32, 0.0, 0.0);
        let radius = ChunkLoadRadius {
            horizontal: 4,
            shell_depth: 64.0,
            shell_height: 64.0,
            ..Default::default()
        };
        let chunks = desired_chunks_spherical(cam, &radius, &planet);

        // Origin chunk (0,0,0) is at planet center — way below shell
        assert!(
            !chunks.contains(&ChunkCoord::new(0, 0, 0)),
            "Planet core chunk should not be loaded"
        );
    }

    #[test]
    fn spherical_excludes_outer_space() {
        let planet = small_planet();
        let cam = Vec3::new(planet.mean_radius as f32, 0.0, 0.0);
        let radius = ChunkLoadRadius {
            horizontal: 4,
            shell_depth: 64.0,
            shell_height: 64.0,
            ..Default::default()
        };
        let chunks = desired_chunks_spherical(cam, &radius, &planet);

        // A chunk far above the surface (20 chunk units from surface = 640 m above)
        let far_out = ChunkCoord::new(20, 0, 0);
        assert!(
            !chunks.contains(&far_out),
            "Outer space chunk should not be loaded"
        );
    }

    #[test]
    fn spherical_chunk_count_is_reasonable() {
        let planet = small_planet();
        let cam = Vec3::new(planet.mean_radius as f32, 0.0, 0.0);
        let radius = ChunkLoadRadius {
            horizontal: 4,
            shell_depth: 128.0,
            shell_height: 128.0,
            ..Default::default()
        };
        let chunks = desired_chunks_spherical(cam, &radius, &planet);
        // Should be a reasonable number — not the full sphere
        assert!(chunks.len() > 10, "Too few chunks: {}", chunks.len());
        assert!(chunks.len() < 500, "Too many chunks: {}", chunks.len());
    }

    // -----------------------------------------------------------------------
    // Adaptive view distance tests
    // -----------------------------------------------------------------------

    #[test]
    fn adapt_shrinks_after_hysteresis() {
        let mut radius = ChunkLoadRadius {
            horizontal: 6,
            ..Default::default()
        };
        let mut state = ViewDistanceState {
            frames_over_budget: ADAPTATION_HYSTERESIS,
            frames_under_budget: 0,
        };

        if state.frames_over_budget >= ADAPTATION_HYSTERESIS && radius.horizontal > MIN_CHUNK_RADIUS
        {
            radius.horizontal -= 1;
            state.frames_over_budget = 0;
        }
        assert_eq!(radius.horizontal, 5);
        assert_eq!(state.frames_over_budget, 0);
    }

    #[test]
    fn adapt_grows_after_hysteresis() {
        let mut radius = ChunkLoadRadius {
            horizontal: 4,
            ..Default::default()
        };
        let mut state = ViewDistanceState {
            frames_over_budget: 0,
            frames_under_budget: ADAPTATION_HYSTERESIS,
        };

        if state.frames_under_budget >= ADAPTATION_HYSTERESIS
            && radius.horizontal < MAX_CHUNK_RADIUS
        {
            radius.horizontal += 1;
            state.frames_under_budget = 0;
        }
        assert_eq!(radius.horizontal, 5);
        assert_eq!(state.frames_under_budget, 0);
    }

    #[test]
    fn adapt_respects_min_radius() {
        let mut radius = ChunkLoadRadius {
            horizontal: MIN_CHUNK_RADIUS,
            ..Default::default()
        };
        let state = ViewDistanceState {
            frames_over_budget: ADAPTATION_HYSTERESIS,
            frames_under_budget: 0,
        };

        if state.frames_over_budget >= ADAPTATION_HYSTERESIS && radius.horizontal > MIN_CHUNK_RADIUS
        {
            radius.horizontal -= 1;
        }
        assert_eq!(
            radius.horizontal, MIN_CHUNK_RADIUS,
            "Should not go below minimum"
        );
    }

    #[test]
    fn adapt_respects_max_radius() {
        let mut radius = ChunkLoadRadius {
            horizontal: MAX_CHUNK_RADIUS,
            ..Default::default()
        };
        let state = ViewDistanceState {
            frames_over_budget: 0,
            frames_under_budget: ADAPTATION_HYSTERESIS,
        };

        if state.frames_under_budget >= ADAPTATION_HYSTERESIS
            && radius.horizontal < MAX_CHUNK_RADIUS
        {
            radius.horizontal += 1;
        }
        assert_eq!(
            radius.horizontal, MAX_CHUNK_RADIUS,
            "Should not exceed maximum"
        );
    }

    #[test]
    fn hysteresis_constants_are_sensible() {
        const { assert!(MIN_CHUNK_RADIUS >= 1) };
        const { assert!(MAX_CHUNK_RADIUS > MIN_CHUNK_RADIUS) };
        const { assert!(ADAPTATION_HYSTERESIS > 0) };
        const { assert!(SHRINK_THRESHOLD < GROW_THRESHOLD,) };
    }
}
