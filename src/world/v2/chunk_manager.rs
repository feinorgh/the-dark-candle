// V2 chunk manager for cubed-sphere coordinates.
//
// Computes desired chunk set on the cubed sphere around the camera,
// dispatches async terrain generation + greedy meshing, collects results,
// and spawns entities with correct local-tangent-plane Transforms.

use std::collections::HashSet;
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
use crate::world::v2::terrain_gen::generate_v2_chunk;

// ── Limits ────────────────────────────────────────────────────────────────

const MAX_DISPATCHES_PER_FRAME: usize = 32;
const MAX_PENDING: usize = 128;
const MAX_COLLECTS_PER_FRAME: usize = 32;

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

/// Tracks which CubeSphereCoords are pending (task dispatched, not yet collected).
#[derive(Resource, Default)]
pub struct V2PendingChunks {
    pending: HashSet<CubeSphereCoord>,
}

impl V2PendingChunks {
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }
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
    camera_q: Query<&WorldPosition, With<FpsCamera>>,
    v2_chunks_q: Query<(Entity, &V2ChunkCoord), With<V2ChunkMarker>>,
) {
    let Ok(cam_world_pos) = camera_q.single() else {
        return;
    };

    let desired = desired_chunks_v2(cam_world_pos.0, &planet, &load_radius);

    // Despawn chunks no longer desired
    for (entity, coord) in &v2_chunks_q {
        if !desired.contains(&coord.0) && !pending.pending.contains(&coord.0) {
            commands.entity(entity).despawn();
            chunk_map.loaded.remove(&coord.0);
        }
    }

    // Dispatch new chunks, closest to camera first.
    let pool = AsyncComputeTaskPool::get();
    let budget = MAX_PENDING
        .saturating_sub(pending.pending.len())
        .min(MAX_DISPATCHES_PER_FRAME);

    let mean_radius = planet.mean_radius;
    let fce = CubeSphereCoord::face_chunks_per_edge(mean_radius);
    let cam_coord = world_pos_to_coord(cam_world_pos.0, mean_radius, fce);

    let mut to_dispatch: Vec<CubeSphereCoord> = desired
        .iter()
        .filter(|c| !chunk_map.loaded.contains(c) && !pending.pending.contains(c))
        .copied()
        .collect();

    // Sort: prioritize surface-layer chunks, then by distance to camera.
    to_dispatch.sort_unstable_by_key(|c| {
        // Surface chunks (layer 0) get priority bonus
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
        let cmap = color_map.clone();

        let task = pool.spawn(async move {
            let data = generate_v2_chunk(coord, mean_radius, fce, &tgen);
            let mesh = greedy_mesh::greedy_mesh(&data.voxels, &data.neighbor_slices, &cmap);
            V2ChunkResult { coord, mesh }
        });

        commands.spawn(V2ChunkTask(task));
        pending.pending.insert(coord);
    }
}

/// Collect completed V2 chunk tasks and spawn renderable entities.
#[allow(clippy::too_many_arguments)]
pub fn v2_collect_results(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut chunk_map: ResMut<V2ChunkMap>,
    mut pending: ResMut<V2PendingChunks>,
    planet: Res<PlanetConfig>,
    origin: Res<RenderOrigin>,
    mut cached_mat: Local<Option<Handle<StandardMaterial>>>,
    mut task_q: Query<(Entity, &mut V2ChunkTask)>,
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
            let (center_f64, rotation, tangent_scale) =
                result.coord.world_transform_scaled_f64(planet.mean_radius, fce);
            let cs_half_scaled = Vec3::new(
                cs_half_f * tangent_scale.x,
                cs_half_f,
                cs_half_f * tangent_scale.z,
            );
            // Compute render-space position relative to RenderOrigin
            let center_render = {
                let d = center_f64 - origin.0;
                Vec3::new(d.x as f32, d.y as f32, d.z as f32)
            };
            let adjusted = center_render - rotation * cs_half_scaled;
            commands.spawn((
                V2ChunkMarker,
                V2ChunkCoord(result.coord),
                Transform::from_translation(adjusted)
                    .with_rotation(rotation)
                    .with_scale(tangent_scale),
            ));
            continue;
        }

        // Build Bevy mesh and spawn entity
        let bevy_mesh = chunk_mesh_to_bevy_mesh(result.mesh);
        let mesh_handle = meshes.add(bevy_mesh);

        let (center_f64, rotation, tangent_scale) =
            result.coord.world_transform_scaled_f64(planet.mean_radius, fce);
        // Offset translation: mesh vertices are in [0, CS], so local origin (0,0,0)
        // should map to the chunk's "base corner" in world space. With non-uniform
        // scale, the offset must account for the tangent-plane stretch.
        let cs_half_scaled = Vec3::new(
            cs_half_f * tangent_scale.x,
            cs_half_f,
            cs_half_f * tangent_scale.z,
        );
        // Compute render-space position relative to RenderOrigin
        let center_render = {
            let d = center_f64 - origin.0;
            Vec3::new(d.x as f32, d.y as f32, d.z as f32)
        };
        let adjusted = center_render - rotation * cs_half_scaled;

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
            .init_resource::<V2PendingChunks>()
            .add_systems(Startup, v2_init_terrain_gen)
            .add_systems(
                Update,
                (v2_update_chunks, v2_collect_results, v2_diagnostics)
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
    pending: Res<V2PendingChunks>,
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
        "V2 pipeline: loaded={}, pending={}, meshed={}",
        chunk_map.loaded.len(),
        pending.pending.len(),
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
        app.init_resource::<V2PendingChunks>();
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
        app.add_systems(Update, (v2_update_chunks, v2_collect_results).chain());

        // Frame 1: dispatch tasks.
        app.update();

        let pending_count = app.world().resource::<V2PendingChunks>().pending.len();
        assert!(
            pending_count > 0,
            "Should have dispatched pending chunks, got 0"
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
