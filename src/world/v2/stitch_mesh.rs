// Stitch mesh generation for RENDER-010 Phases 2+3 — same-face and cross-face LOD seams.
//
// When a fine chunk A (LOD l) and a coarse chunk B (LOD l+1) share a boundary,
// the two surface-nets meshes do not align: vertices on A's seam edge are denser
// and at slightly different world positions than those on B's seam edge.
// This file bridges that gap with explicit triangulation.
//
// Phase 2 (same-face): B is on the same cube face as A.
// Phase 3 (cross-face): B is on an adjacent cube face; the same triangulation
//   algorithm applies since boundary loops are already in world space.
//
// ## Algorithm
//
// 1. Both chunks' boundary loops are in world space (`ChunkBoundaryLoops`).
// 2. For direction `dir` from A, the relevant loops are:
//      fine  = A.loops[dir_idx]        (chain on A's face toward B)
//      coarse= B.loops[opposite_idx]   (chain on B's face toward A)
// 3. Both loops are projected onto the 1-D seam tangent axis.
// 4. The coarse loop is CLIPPED to the fine loop's tangential extent so that
//    adjacent fine chunks each only stitch their own half of B's boundary.
//    Clipped endpoints are linearly interpolated on the coarse chain.
// 5. The two chains are zipped into triangles (preserving chain order;
//    no sort of interior vertices).
// 6. Winding order is verified against the radial outward direction.
//
// ## Invalidation
//
// Each stitch entity caches both constituent entity IDs. The update system
// despawns a stitch when either entity ID changes (chunk remeshed/respawned).

use std::collections::HashMap;

use bevy::asset::RenderAssetUsages;
use bevy::math::DVec3;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;

use crate::floating_origin::RenderOrigin;
use crate::world::planet::PlanetConfig;
use crate::world::v2::boundary_loop::ChunkBoundaryLoops;
use crate::world::v2::chunk_manager::{V2ChunkCoord, V2ChunkMap};
use crate::world::v2::cubed_sphere::{ChunkDir, CubeSphereCoord};

// ── Components & resources ────────────────────────────────────────────────────

/// Marks a stitch-mesh entity as bridging the LOD seam between two specific
/// chunk entities.
///
/// `dir` is the direction **from** the fine chunk **toward** the coarse chunk.
#[derive(Component)]
pub struct StitchSeam {
    pub fine_coord: CubeSphereCoord,
    pub coarse_coord: CubeSphereCoord,
    /// Direction from fine chunk toward coarse chunk (index into ChunkDir::ALL).
    pub dir_idx: u8,
    /// Entity ID of the fine chunk at the time this stitch was generated.
    /// Used to detect remesh/respawn.
    pub fine_entity: Entity,
    /// Entity ID of the coarse chunk at the time this stitch was generated.
    pub coarse_entity: Entity,
}

/// Tracks all active stitch entities for deduplication and invalidation.
#[derive(Resource, Default)]
pub struct V2StitchMap {
    /// Key: (fine_coord, coarse_coord, dir_idx)
    /// Value: (stitch_entity, fine_entity, coarse_entity)
    stitches: HashMap<(CubeSphereCoord, CubeSphereCoord, u8), (Entity, Entity, Entity)>,
}

/// Marks a corner-stitch triangle entity that fills the small gap at the
/// junction of two pairwise LOD stitches.
///
/// At each corner of a fine chunk, the two orthogonal pairwise stitches leave
/// a small quadrilateral hole where: fine_corner_a, fine_corner_b,
/// coarse_A_endpoint, and coarse_B_endpoint do not quite meet.
#[derive(Component)]
pub struct CornerStitchSeam {
    pub fine_coord: CubeSphereCoord,
    pub coarse_a_coord: CubeSphereCoord,
    pub coarse_b_coord: CubeSphereCoord,
    pub corner_idx: u8,
    pub fine_entity: Entity,
    pub coarse_a_entity: Entity,
    pub coarse_b_entity: Entity,
}

type CornerStitchKey = (CubeSphereCoord, CubeSphereCoord, CubeSphereCoord, u8);
type CornerStitchVal = (Entity, Entity, Entity, Entity);

/// Tracks all active corner-stitch entities for deduplication and invalidation.
#[derive(Resource, Default)]
pub struct V2CornerStitchMap {
    /// Key: (fine_coord, coarse_a_coord, coarse_b_coord, corner_idx)
    /// Value: (corner_entity, fine_entity, coarse_a_entity, coarse_b_entity)
    stitches: HashMap<CornerStitchKey, CornerStitchVal>,
}

// ── Core triangulation ────────────────────────────────────────────────────────

/// Clip a loop of world-space vertices to the 1-D tangential interval `[t_min, t_max]`.
///
/// Vertices outside the interval are dropped; the chain endpoints at the clip
/// boundary are linearly interpolated from the adjacent pair so the clipped
/// chain exactly meets the interval boundary.
///
/// Returns an empty `Vec` when the entire chain falls outside the interval.
fn clip_loop_to_range(chain: &[DVec3], axis: DVec3, t_min: f64, t_max: f64) -> Vec<DVec3> {
    if chain.len() < 2 {
        return chain.to_vec();
    }

    let projs: Vec<f64> = chain.iter().map(|p| p.dot(axis)).collect();

    // Build clipped chain by walking the input and adding interpolated boundary
    // points where the chain enters and exits the [t_min, t_max] interval.
    let mut result: Vec<DVec3> = Vec::new();
    let n = chain.len();

    for i in 0..n {
        let ti = projs[i];
        let inside = ti >= t_min && ti <= t_max;

        if inside {
            // Add interpolated entry point from the previous vertex if the
            // previous was outside.
            if i > 0 {
                let t_prev = projs[i - 1];
                if t_prev < t_min {
                    let t_enter = t_min;
                    let s = (t_enter - t_prev) / (ti - t_prev);
                    result.push(chain[i - 1].lerp(chain[i], s));
                } else if t_prev > t_max {
                    let t_enter = t_max;
                    let s = (t_enter - t_prev) / (ti - t_prev);
                    result.push(chain[i - 1].lerp(chain[i], s));
                }
            }
            result.push(chain[i]);
        } else if i > 0 {
            // Vertex is outside; check if the segment crosses the boundary
            let t_prev = projs[i - 1];
            let prev_inside = t_prev >= t_min && t_prev <= t_max;
            if prev_inside {
                // Exiting the interval — add the interpolated exit point.
                let t_exit = if ti < t_min { t_min } else { t_max };
                let s = (t_exit - t_prev) / (ti - t_prev);
                result.push(chain[i - 1].lerp(chain[i], s));
            }
        }
    }

    result
}

/// Clip a coarse boundary chain to the tangential extent of a fine boundary chain.
///
/// Mirrors the clipping step in `v2_stitch_update` so that corner-stitch
/// endpoints exactly match those already used by the two adjacent pairwise stitches.
fn clip_coarse_to_fine_extent(
    fine_verts: &[DVec3],
    coarse_verts: &[DVec3],
    outward: DVec3,
) -> Vec<DVec3> {
    if coarse_verts.len() < 2 {
        return coarse_verts.to_vec();
    }
    let axis_raw = coarse_verts.last().unwrap() - coarse_verts.first().unwrap();
    let axis = if axis_raw.length_squared() > 1e-10 {
        axis_raw.normalize()
    } else {
        let arb = if outward.x.abs() < 0.9 {
            DVec3::X
        } else {
            DVec3::Y
        };
        outward.cross(arb).normalize()
    };
    let proj_f: Vec<f64> = fine_verts.iter().map(|p| p.dot(axis)).collect();
    let t_min = proj_f.iter().cloned().fold(f64::INFINITY, f64::min);
    let t_max = proj_f.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    clip_loop_to_range(coarse_verts, axis, t_min, t_max)
}

/// Returns the endpoint (first or last) of `chain` closest to `target`.
fn nearest_chain_endpoint(chain: &[DVec3], target: DVec3) -> DVec3 {
    let first = *chain.first().unwrap();
    let last = *chain.last().unwrap();
    if (first - target).length_squared() <= (last - target).length_squared() {
        first
    } else {
        last
    }
}

/// Triangulate a stitch ribbon between two world-space boundary loops.
///
/// `loop_f` — fine chunk boundary vertices (ordered polyline, world space).
/// `loop_c` — coarse chunk boundary vertices (ordered polyline, world space).
///             Must already be clipped to the fine loop's tangential extent.
/// `origin` — `RenderOrigin` (DVec3) subtracted to get render-space f32 coords.
/// `outward_hint` — approximate radial outward unit vector at the seam.
///
/// Returns `(positions, normals, indices)` for a Bevy `Mesh`.
pub fn stitch_triangulate(
    loop_f: &[DVec3],
    loop_c: &[DVec3],
    origin: DVec3,
    outward_hint: DVec3,
) -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<u32>) {
    if loop_f.len() < 2 || loop_c.len() < 2 {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    // ── Render-space conversion ───────────────────────────────────────────────
    let to_f32 = |p: DVec3| -> [f32; 3] {
        let d = p - origin;
        [d.x as f32, d.y as f32, d.z as f32]
    };

    // ── Projection axis: dominant direction of loop_c ─────────────────────────
    let axis_raw = *loop_c.last().unwrap() - *loop_c.first().unwrap();
    let axis = if axis_raw.length_squared() > 1e-10 {
        axis_raw.normalize()
    } else {
        // Degenerate coarse chain — fall back to cross of outward and arbitrary
        let arb = if outward_hint.x.abs() < 0.9 {
            DVec3::X
        } else {
            DVec3::Y
        };
        outward_hint.cross(arb).normalize()
    };

    // ── Ensure coarse loop goes in increasing-projection order ────────────────
    let proj_c_first = loop_c.first().unwrap().dot(axis);
    let proj_c_last = loop_c.last().unwrap().dot(axis);
    let (loop_c_ord, _proj_c_first_ord) = if proj_c_first <= proj_c_last {
        (loop_c.to_vec(), proj_c_first)
    } else {
        let mut v = loop_c.to_vec();
        v.reverse();
        (v, proj_c_last)
    };

    // ── Ensure fine loop goes in the same direction as coarse ─────────────────
    let proj_f_first = loop_f.first().unwrap().dot(axis);
    let proj_f_last = loop_f.last().unwrap().dot(axis);
    let loop_f_ord: Vec<DVec3> = if proj_f_first <= proj_f_last {
        loop_f.to_vec()
    } else {
        let mut v = loop_f.to_vec();
        v.reverse();
        v
    };

    let proj_c_ord: Vec<f64> = loop_c_ord.iter().map(|p| p.dot(axis)).collect();
    let proj_f_ord: Vec<f64> = loop_f_ord.iter().map(|p| p.dot(axis)).collect();

    // ── Vertex buffer: coarse first (indices 0..c_len), then fine ─────────────
    let c_len = loop_c_ord.len();
    let f_len = loop_f_ord.len();
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(c_len + f_len);
    for p in &loop_c_ord {
        positions.push(to_f32(*p));
    }
    for p in &loop_f_ord {
        positions.push(to_f32(*p));
    }

    // ── Two-pointer zipper merge ───────────────────────────────────────────────
    let mut indices: Vec<u32> = Vec::new();
    let mut ci = 0usize;
    let mut fi = 0usize;

    while ci + 1 < c_len || fi + 1 < f_len {
        let can_c = ci + 1 < c_len;
        let can_f = fi + 1 < f_len;

        let advance_c = match (can_c, can_f) {
            (true, false) => true,
            (false, true) => false,
            // Both can advance: pick side whose next vertex has smaller projection.
            // Tie → advance coarse (deterministic).
            (true, true) => proj_c_ord[ci + 1] <= proj_f_ord[fi + 1],
            (false, false) => break,
        };

        if advance_c {
            // Triangle: current_coarse → next_coarse → current_fine
            indices.push(ci as u32);
            indices.push((ci + 1) as u32);
            indices.push((c_len + fi) as u32);
            ci += 1;
        } else {
            // Triangle: current_coarse → current_fine → next_fine
            indices.push(ci as u32);
            indices.push((c_len + fi) as u32);
            indices.push((c_len + fi + 1) as u32);
            fi += 1;
        }
    }

    if indices.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    // ── Winding order: ensure normals face outward ────────────────────────────
    let outward_f32 = outward_hint.as_vec3().normalize();
    let first_normal = {
        let i0 = indices[0] as usize;
        let i1 = indices[1] as usize;
        let i2 = indices[2] as usize;
        let v0 = Vec3::from(positions[i0]);
        let v1 = Vec3::from(positions[i1]);
        let v2 = Vec3::from(positions[i2]);
        (v1 - v0).cross(v2 - v0)
    };
    if first_normal.dot(outward_f32) < 0.0 {
        for tri in indices.chunks_exact_mut(3) {
            tri.swap(1, 2);
        }
    }

    let normals = vec![outward_f32.to_array(); c_len + f_len];

    (positions, normals, indices)
}

/// Build a Bevy `Mesh` from stitch triangulation output.
pub fn build_stitch_mesh(
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    indices: Vec<u32>,
) -> Mesh {
    let n = positions.len();
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    // UV coordinates required by StandardMaterial
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0f32, 0.0f32]; n]);
    // Vertex colors: white so the chunk material tint applies uniformly
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, vec![[1.0f32, 1.0, 1.0, 1.0]; n]);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

// ── Update system ─────────────────────────────────────────────────────────────

/// System: maintain stitch meshes at all same-face LOD seams.
///
/// Runs after `v2_collect_meshes` every frame.  Work is incremental: only
/// missing seams are generated, only invalidated stitches are despawned.
///
/// Handles the lod+1 case only (same-face; cross-face stitching is Phase 3).
#[allow(clippy::too_many_arguments)]
pub fn v2_stitch_update(
    mut commands: Commands,
    mut stitch_map: ResMut<V2StitchMap>,
    planet: Res<PlanetConfig>,
    origin: Res<RenderOrigin>,
    chunk_map: Res<V2ChunkMap>,
    chunk_q: Query<&ChunkBoundaryLoops, With<V2ChunkCoord>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut cached_mat: Local<Option<Handle<StandardMaterial>>>,
    mut prev_stitch_count: Local<usize>,
) {
    let mean_radius = planet.mean_radius;
    let origin_pos = origin.0;

    let stitch_material = cached_mat
        .get_or_insert_with(|| {
            materials.add(StandardMaterial {
                base_color: Color::WHITE,
                double_sided: true,
                cull_mode: None,
                ..default()
            })
        })
        .clone();

    // ── Step 1: despawn stale stitches ────────────────────────────────────────
    // Proper retain with entity validation:
    let stale_keys: Vec<_> = stitch_map
        .stitches
        .iter()
        .filter_map(|(key, (_, fine_ent, coarse_ent))| {
            let (fine_coord, coarse_coord, _) = key;
            let current_fine = chunk_map.get(fine_coord);
            let current_coarse = chunk_map.get(coarse_coord);
            let still_valid =
                current_fine == Some(*fine_ent) && current_coarse == Some(*coarse_ent);
            if still_valid { None } else { Some(*key) }
        })
        .collect();

    for key in stale_keys {
        if let Some((stitch_ent, _, _)) = stitch_map.stitches.remove(&key) {
            commands.entity(stitch_ent).despawn();
        }
    }

    // ── Step 2: generate missing stitches ─────────────────────────────────────
    // Collect (coord, entity) pairs to avoid borrow conflicts in the inner loop.
    let candidates: Vec<(CubeSphereCoord, Entity)> =
        chunk_map.iter().map(|(&c, &e)| (c, e)).collect();

    for (fine_coord, fine_entity) in &candidates {
        let fine_lod = fine_coord.lod;
        let coarse_lod = match fine_lod.checked_add(1) {
            Some(l) => l,
            None => continue, // lod overflow guard (shouldn't happen)
        };

        for (dir_idx, &dir) in ChunkDir::ALL.iter().enumerate() {
            // Look up the single representative coarse neighbor — same face first,
            // cross-face fallback (Phase 3).
            let (coarse_coord, coarse_incoming_dir) =
                match fine_coord.same_face_neighbor_at_lod(dir, coarse_lod, mean_radius) {
                    Some(c) => (c, dir.opposite()),
                    None => {
                        match fine_coord.cross_face_neighbor_at_lod(dir, coarse_lod, mean_radius) {
                            Some((c, incoming)) => (c, incoming),
                            None => continue, // PosLayer/NegLayer — no cross-face
                        }
                    }
                };

            // Verify the coarse chunk is actually loaded at coarse_lod
            let Some(coarse_entity) = chunk_map.get(&coarse_coord) else {
                continue;
            };
            if coarse_coord.lod != coarse_lod {
                continue;
            }

            let real_key = (*fine_coord, coarse_coord, dir_idx as u8);

            // Already stitched and valid (checked in Step 1)
            if stitch_map.stitches.contains_key(&real_key) {
                continue;
            }

            // Get boundary loops for both chunks
            let Ok(fine_loops) = chunk_q.get(*fine_entity) else {
                continue;
            };
            let Ok(coarse_loops) = chunk_q.get(coarse_entity) else {
                continue;
            };

            let opposite_idx = coarse_incoming_dir.all_index();
            let fine_chains = &fine_loops.loops[dir_idx];
            let coarse_chains = &coarse_loops.loops[opposite_idx];

            if fine_chains.is_empty() || coarse_chains.is_empty() {
                continue;
            }

            // Phase 2: take the longest chain from each side
            let fine_chain = fine_chains.iter().max_by_key(|l| l.vertices.len()).unwrap();
            let coarse_chain = coarse_chains
                .iter()
                .max_by_key(|l| l.vertices.len())
                .unwrap();

            if fine_chain.vertices.len() < 2 || coarse_chain.vertices.len() < 2 {
                continue;
            }

            // Compute outward direction at the seam
            let seam_world: DVec3 = fine_chain
                .vertices
                .iter()
                .chain(coarse_chain.vertices.iter())
                .fold(DVec3::ZERO, |acc, &p| acc + p)
                / (fine_chain.vertices.len() + coarse_chain.vertices.len()) as f64;
            let outward_hint = seam_world.normalize();

            // Projection axis from coarse chain
            let axis_raw =
                coarse_chain.vertices.last().unwrap() - coarse_chain.vertices.first().unwrap();
            let axis = if axis_raw.length_squared() > 1e-10 {
                axis_raw.normalize()
            } else {
                let arb = if outward_hint.x.abs() < 0.9 {
                    DVec3::X
                } else {
                    DVec3::Y
                };
                outward_hint.cross(arb).normalize()
            };

            // Clip coarse loop to fine loop's tangential extent
            let proj_f: Vec<f64> = fine_chain.vertices.iter().map(|p| p.dot(axis)).collect();
            let t_min = proj_f.iter().cloned().fold(f64::INFINITY, f64::min);
            let t_max = proj_f.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            let clipped_coarse = clip_loop_to_range(&coarse_chain.vertices, axis, t_min, t_max);
            if clipped_coarse.len() < 2 {
                continue;
            }

            let (positions, normals, indices) = stitch_triangulate(
                &fine_chain.vertices,
                &clipped_coarse,
                origin_pos,
                outward_hint,
            );

            if indices.is_empty() {
                continue;
            }

            let mesh = build_stitch_mesh(positions, normals, indices);
            let mesh_handle = meshes.add(mesh);

            let stitch_entity = commands
                .spawn((
                    StitchSeam {
                        fine_coord: *fine_coord,
                        coarse_coord,
                        dir_idx: dir_idx as u8,
                        fine_entity: *fine_entity,
                        coarse_entity,
                    },
                    Mesh3d(mesh_handle),
                    MeshMaterial3d(stitch_material.clone()),
                    Transform::IDENTITY,
                ))
                .id();

            stitch_map
                .stitches
                .insert(real_key, (stitch_entity, *fine_entity, coarse_entity));
        }
    }

    let count = stitch_map.stitches.len();
    if count != *prev_stitch_count {
        info!(
            "[StitchMesh] active stitches: {} (was {})",
            count, *prev_stitch_count
        );
        *prev_stitch_count = count;
    }
}

// ── ChunkDir helper ───────────────────────────────────────────────────────────

trait ChunkDirExt {
    fn all_index(self) -> usize;
}

impl ChunkDirExt for ChunkDir {
    fn all_index(self) -> usize {
        ChunkDir::ALL.iter().position(|&d| d == self).unwrap()
    }
}

// ── Corner stitch system ──────────────────────────────────────────────────────

/// The four tangential corner direction-pairs of a chunk's boundary.
///
/// Each pair (dir_a, dir_b) names one of the four corners of the chunk in the
/// (U, V) plane. `PosLayer` and `NegLayer` are excluded because they have no
/// tangential neighbours.
const CORNER_PAIRS: [(ChunkDir, ChunkDir); 4] = [
    (ChunkDir::PosU, ChunkDir::PosV),
    (ChunkDir::PosU, ChunkDir::NegV),
    (ChunkDir::NegU, ChunkDir::PosV),
    (ChunkDir::NegU, ChunkDir::NegV),
];

/// System: maintain corner-cap triangles at all three-way LOD seam junctions.
///
/// Two adjacent pairwise stitches (dir_a and dir_b from the same fine chunk)
/// leave a small quadrilateral hole at the corner.  This system fills it with
/// two triangles sharing their edges with the endpoints of the two pairwise
/// stitches.  Handles both the case where dir_a and dir_b lead to the SAME
/// coarse chunk and the case where they lead to DIFFERENT coarse chunks.
///
/// Why both cases have a corner gap: the corner vertex of the fine mesh is
/// classified to one face only (boundary_loop.rs `classify_vertex`).  The
/// other chain ends one surface-nets vertex short of the corner, leaving a
/// ~1-voxel gap that this system fills.
///
/// The four corner vertices are derived by re-running the same coarse clip as
/// `v2_stitch_update` so that the cap exactly matches the already-emitted
/// stitch edges (no T-junctions).
#[allow(clippy::too_many_arguments)]
pub fn v2_corner_stitch_update(
    mut commands: Commands,
    mut corner_map: ResMut<V2CornerStitchMap>,
    planet: Res<PlanetConfig>,
    origin: Res<RenderOrigin>,
    chunk_map: Res<V2ChunkMap>,
    chunk_q: Query<&ChunkBoundaryLoops, With<V2ChunkCoord>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut cached_mat: Local<Option<Handle<StandardMaterial>>>,
    mut prev_count: Local<usize>,
) {
    let mean_radius = planet.mean_radius;
    let origin_pos = origin.0;

    let corner_material = cached_mat
        .get_or_insert_with(|| {
            materials.add(StandardMaterial {
                base_color: Color::WHITE,
                double_sided: true,
                cull_mode: None,
                ..default()
            })
        })
        .clone();

    // ── Step 1: despawn stale corner stitches ─────────────────────────────────
    let stale_keys: Vec<_> = corner_map
        .stitches
        .iter()
        .filter_map(|(key, (_, fine_ent, ca_ent, cb_ent))| {
            let (fine_coord, coarse_a, coarse_b, _) = key;
            let valid = chunk_map.get(fine_coord) == Some(*fine_ent)
                && chunk_map.get(coarse_a) == Some(*ca_ent)
                && chunk_map.get(coarse_b) == Some(*cb_ent);
            if valid { None } else { Some(*key) }
        })
        .collect();

    for key in stale_keys {
        if let Some((ent, _, _, _)) = corner_map.stitches.remove(&key) {
            commands.entity(ent).despawn();
        }
    }

    // ── Step 2: generate missing corner stitches ──────────────────────────────
    let candidates: Vec<(CubeSphereCoord, Entity)> =
        chunk_map.iter().map(|(&c, &e)| (c, e)).collect();

    for (fine_coord, fine_entity) in &candidates {
        let fine_lod = fine_coord.lod;
        let coarse_lod = match fine_lod.checked_add(1) {
            Some(l) => l,
            None => continue,
        };

        let Ok(fine_loops) = chunk_q.get(*fine_entity) else {
            continue;
        };

        for (corner_idx, &(dir_a, dir_b)) in CORNER_PAIRS.iter().enumerate() {
            // Resolve coarse neighbours
            let (coarse_a_coord, incoming_a) =
                match fine_coord.same_face_neighbor_at_lod(dir_a, coarse_lod, mean_radius) {
                    Some(c) => (c, dir_a.opposite()),
                    None => {
                        match fine_coord.cross_face_neighbor_at_lod(dir_a, coarse_lod, mean_radius)
                        {
                            Some((c, inc)) => (c, inc),
                            None => continue,
                        }
                    }
                };

            let (coarse_b_coord, incoming_b) =
                match fine_coord.same_face_neighbor_at_lod(dir_b, coarse_lod, mean_radius) {
                    Some(c) => (c, dir_b.opposite()),
                    None => {
                        match fine_coord.cross_face_neighbor_at_lod(dir_b, coarse_lod, mean_radius)
                        {
                            Some((c, inc)) => (c, inc),
                            None => continue,
                        }
                    }
                };

            let Some(coarse_a_entity) = chunk_map.get(&coarse_a_coord) else {
                continue;
            };
            let Some(coarse_b_entity) = chunk_map.get(&coarse_b_coord) else {
                continue;
            };
            if coarse_a_coord.lod != coarse_lod || coarse_b_coord.lod != coarse_lod {
                continue;
            }

            let real_key = (
                *fine_coord,
                coarse_a_coord,
                coarse_b_coord,
                corner_idx as u8,
            );
            if corner_map.stitches.contains_key(&real_key) {
                continue;
            }

            let Ok(coarse_a_loops) = chunk_q.get(coarse_a_entity) else {
                continue;
            };
            let Ok(coarse_b_loops) = chunk_q.get(coarse_b_entity) else {
                continue;
            };

            let dir_a_idx = dir_a.all_index();
            let dir_b_idx = dir_b.all_index();
            let incoming_a_idx = incoming_a.all_index();
            let incoming_b_idx = incoming_b.all_index();

            let Some(fine_chain_a) = fine_loops.loops[dir_a_idx]
                .iter()
                .max_by_key(|l| l.vertices.len())
            else {
                continue;
            };
            let Some(fine_chain_b) = fine_loops.loops[dir_b_idx]
                .iter()
                .max_by_key(|l| l.vertices.len())
            else {
                continue;
            };
            let Some(coarse_chain_a) = coarse_a_loops.loops[incoming_a_idx]
                .iter()
                .max_by_key(|l| l.vertices.len())
            else {
                continue;
            };
            let Some(coarse_chain_b) = coarse_b_loops.loops[incoming_b_idx]
                .iter()
                .max_by_key(|l| l.vertices.len())
            else {
                continue;
            };

            if fine_chain_a.vertices.len() < 2
                || fine_chain_b.vertices.len() < 2
                || coarse_chain_a.vertices.len() < 2
                || coarse_chain_b.vertices.len() < 2
            {
                continue;
            }

            // Outward hint from all four chains
            let total: DVec3 = fine_chain_a
                .vertices
                .iter()
                .chain(fine_chain_b.vertices.iter())
                .chain(coarse_chain_a.vertices.iter())
                .chain(coarse_chain_b.vertices.iter())
                .fold(DVec3::ZERO, |acc, &p| acc + p);
            let count_pts = (fine_chain_a.vertices.len()
                + fine_chain_b.vertices.len()
                + coarse_chain_a.vertices.len()
                + coarse_chain_b.vertices.len()) as f64;
            let outward = (total / count_pts).normalize();

            // Re-clip coarse chains exactly as the pairwise stitches do, so
            // our corner endpoints match the stitch ribbon endpoints.
            let clipped_a = clip_coarse_to_fine_extent(
                &fine_chain_a.vertices,
                &coarse_chain_a.vertices,
                outward,
            );
            let clipped_b = clip_coarse_to_fine_extent(
                &fine_chain_b.vertices,
                &coarse_chain_b.vertices,
                outward,
            );
            if clipped_a.len() < 2 || clipped_b.len() < 2 {
                continue;
            }

            // Identify which end of each chain is the "corner end".
            //
            // Trick: the centroid of the perpendicular chain lies closest to
            // the corner end of the parallel chain.  So:
            //   - fine_chain_a corner end  = endpoint of fine_chain_a nearest  centroid(fine_chain_b)
            //   - clipped_a corner end     = endpoint of clipped_a nearest      centroid(fine_chain_b)
            //   - fine_chain_b corner end  = endpoint of fine_chain_b nearest  centroid(fine_chain_a)
            //   - clipped_b corner end     = endpoint of clipped_b nearest      centroid(fine_chain_a)
            let centroid_b = fine_chain_b
                .vertices
                .iter()
                .fold(DVec3::ZERO, |a, &v| a + v)
                / fine_chain_b.vertices.len() as f64;
            let centroid_a = fine_chain_a
                .vertices
                .iter()
                .fold(DVec3::ZERO, |a, &v| a + v)
                / fine_chain_a.vertices.len() as f64;

            let fine_corner_a = nearest_chain_endpoint(&fine_chain_a.vertices, centroid_b);
            let coarse_a_pt = nearest_chain_endpoint(&clipped_a, centroid_b);
            let fine_corner_b = nearest_chain_endpoint(&fine_chain_b.vertices, centroid_a);
            let coarse_b_pt = nearest_chain_endpoint(&clipped_b, centroid_a);

            // Emit a quad as two triangles:
            //   T1: (fine_corner_a, coarse_a_pt, fine_corner_b)
            //   T2: (fine_corner_b, coarse_a_pt, coarse_b_pt)
            //
            // This fills the quadrilateral hole and shares edges with the two
            // adjacent pairwise stitch ribbons (no T-junctions).
            let to_f32 = |p: DVec3| -> [f32; 3] {
                let d = p - origin_pos;
                [d.x as f32, d.y as f32, d.z as f32]
            };

            let p0 = to_f32(fine_corner_a);
            let p1 = to_f32(coarse_a_pt);
            let p2 = to_f32(fine_corner_b);
            let p3 = to_f32(coarse_b_pt);

            let outward_f32 = outward.as_vec3();

            // Build triangles with winding verified against outward normal.
            let mut positions: Vec<[f32; 3]> = Vec::new();
            let mut indices: Vec<u32> = Vec::new();
            let mut base = 0u32;

            for &(a, b, c) in &[(p0, p1, p2), (p2, p1, p3)] {
                let va = Vec3::from(a);
                let vb = Vec3::from(b);
                let vc = Vec3::from(c);
                let area_cross = (vb - va).cross(vc - va);
                if area_cross.length_squared() < 1e-10 {
                    continue; // skip degenerate
                }
                let (b_out, c_out) = if area_cross.dot(outward_f32) < 0.0 {
                    (c, b)
                } else {
                    (b, c)
                };
                positions.push(a);
                positions.push(b_out);
                positions.push(c_out);
                indices.extend_from_slice(&[base, base + 1, base + 2]);
                base += 3;
            }

            if indices.is_empty() {
                continue;
            }

            let normals = vec![outward_f32.to_array(); positions.len()];
            let mesh = build_stitch_mesh(positions, normals, indices);
            let mesh_handle = meshes.add(mesh);

            let corner_entity = commands
                .spawn((
                    CornerStitchSeam {
                        fine_coord: *fine_coord,
                        coarse_a_coord,
                        coarse_b_coord,
                        corner_idx: corner_idx as u8,
                        fine_entity: *fine_entity,
                        coarse_a_entity,
                        coarse_b_entity,
                    },
                    Mesh3d(mesh_handle),
                    MeshMaterial3d(corner_material.clone()),
                    Transform::IDENTITY,
                ))
                .id();

            corner_map.stitches.insert(
                real_key,
                (
                    corner_entity,
                    *fine_entity,
                    coarse_a_entity,
                    coarse_b_entity,
                ),
            );
        }
    }

    let count = corner_map.stitches.len();
    if count != *prev_count {
        info!(
            "[CornerStitch] active corner stitches: {} (was {})",
            count, *prev_count
        );
        *prev_count = count;
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_line(start: DVec3, end: DVec3, n: usize) -> Vec<DVec3> {
        (0..n)
            .map(|i| start.lerp(end, i as f64 / (n - 1) as f64))
            .collect()
    }

    #[test]
    fn stitch_triangulate_basic_2_vs_4() {
        // Coarse: 2 vertices at x=0 and x=4
        // Fine:   4 vertices at x=0, 1, 2, 3
        // Both at y=0, z=0 (coarse slightly offset in z)
        let loop_c = vec![DVec3::new(0.0, 0.0, 0.1), DVec3::new(4.0, 0.0, 0.1)];
        let loop_f = vec![
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(2.0, 0.0, 0.0),
            DVec3::new(3.0, 0.0, 0.0),
        ];
        let origin = DVec3::ZERO;
        let outward = DVec3::Z;
        let (pos, nrm, idx) = stitch_triangulate(&loop_f, &loop_c, origin, outward);

        assert!(!idx.is_empty(), "should produce triangles");
        assert_eq!(idx.len() % 3, 0, "indices must be multiple of 3");
        assert_eq!(pos.len(), nrm.len(), "positions and normals must match");

        // No index out of bounds
        for &i in &idx {
            assert!((i as usize) < pos.len(), "index out of bounds: {i}");
        }
    }

    #[test]
    fn stitch_no_degenerate_triangles() {
        let loop_c = make_line(DVec3::new(0.0, 0.0, 0.05), DVec3::new(8.0, 0.0, 0.05), 3);
        let loop_f = make_line(DVec3::new(0.0, 0.0, 0.0), DVec3::new(8.0, 0.0, 0.0), 9);
        let (pos, _, idx) = stitch_triangulate(&loop_f, &loop_c, DVec3::ZERO, DVec3::Z);

        for tri in idx.chunks_exact(3) {
            let v0 = Vec3::from(pos[tri[0] as usize]);
            let v1 = Vec3::from(pos[tri[1] as usize]);
            let v2 = Vec3::from(pos[tri[2] as usize]);
            let area = (v1 - v0).cross(v2 - v0).length() * 0.5;
            assert!(area > 1e-8, "degenerate triangle detected");
        }
    }

    #[test]
    fn stitch_bounding_box_regression() {
        // All stitch vertices must lie within a bounding box expanded by a small
        // margin around the combined input loops — guard against "grey curtain".
        let loop_c = make_line(DVec3::new(0.0, 0.0, 0.1), DVec3::new(4.0, 0.0, 0.1), 3);
        let loop_f = make_line(DVec3::new(0.0, 0.0, 0.0), DVec3::new(4.0, 0.0, 0.0), 5);
        let all: Vec<DVec3> = loop_c.iter().chain(loop_f.iter()).cloned().collect();

        let min_w = all
            .iter()
            .fold(DVec3::splat(f64::INFINITY), |a, &p| a.min(p));
        let max_w = all
            .iter()
            .fold(DVec3::splat(f64::NEG_INFINITY), |a, &p| a.max(p));
        let margin = 2.0f64;

        let (pos, _, _) = stitch_triangulate(&loop_f, &loop_c, DVec3::ZERO, DVec3::Z);
        for p in &pos {
            let wp = DVec3::new(p[0] as f64, p[1] as f64, p[2] as f64);
            assert!(
                wp.cmpge(min_w - DVec3::splat(margin)).all()
                    && wp.cmple(max_w + DVec3::splat(margin)).all(),
                "stitch vertex {wp:?} outside seam bounding box [{min_w:?}, {max_w:?}]"
            );
        }
    }

    #[test]
    fn clip_loop_to_range_basic() {
        let chain: Vec<DVec3> = (0..=8).map(|i| DVec3::new(i as f64, 0.0, 0.0)).collect();
        let clipped = clip_loop_to_range(&chain, DVec3::X, 2.0, 6.0);
        assert!(clipped.len() >= 2);
        for p in &clipped {
            assert!(p.x >= 2.0 - 1e-9 && p.x <= 6.0 + 1e-9, "x={}", p.x);
        }
    }

    #[test]
    fn clip_loop_no_overlap_returns_empty() {
        let chain: Vec<DVec3> = vec![DVec3::new(10.0, 0.0, 0.0), DVec3::new(20.0, 0.0, 0.0)];
        let clipped = clip_loop_to_range(&chain, DVec3::X, 0.0, 5.0);
        assert!(clipped.is_empty());
    }

    #[test]
    fn stitch_winding_outward_normals() {
        let loop_c = vec![DVec3::new(0.0, 0.0, 1.0), DVec3::new(4.0, 0.0, 1.0)];
        let loop_f = vec![
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(2.0, 0.0, 0.0),
            DVec3::new(4.0, 0.0, 0.0),
        ];
        let outward = DVec3::Z;
        let (pos, _, idx) = stitch_triangulate(&loop_f, &loop_c, DVec3::ZERO, outward);
        let outward_f32 = Vec3::Z;
        for tri in idx.chunks_exact(3) {
            let v0 = Vec3::from(pos[tri[0] as usize]);
            let v1 = Vec3::from(pos[tri[1] as usize]);
            let v2 = Vec3::from(pos[tri[2] as usize]);
            let n = (v1 - v0).cross(v2 - v0);
            assert!(
                n.dot(outward_f32) > -1e-6,
                "triangle normal {n:?} should face outward"
            );
        }
    }

    #[test]
    fn nearest_chain_endpoint_returns_correct_end() {
        // Chain from (0,0,0) to (10,0,0); target near (10,0,0) → last
        let chain = vec![DVec3::new(0.0, 0.0, 0.0), DVec3::new(10.0, 0.0, 0.0)];
        let result = nearest_chain_endpoint(&chain, DVec3::new(9.5, 0.0, 0.0));
        assert!((result - DVec3::new(10.0, 0.0, 0.0)).length() < 1e-9);

        // Target near (0,0,0) → first
        let result2 = nearest_chain_endpoint(&chain, DVec3::new(0.1, 0.0, 0.0));
        assert!((result2 - DVec3::new(0.0, 0.0, 0.0)).length() < 1e-9);
    }

    #[test]
    fn clip_coarse_to_fine_extent_matches_chain_clip() {
        // fine chain: x in [1, 3]; coarse chain: x in [0, 5]
        let fine = vec![DVec3::new(1.0, 0.0, 0.1), DVec3::new(3.0, 0.0, 0.1)];
        let coarse: Vec<DVec3> = (0..=5).map(|i| DVec3::new(i as f64, 0.0, 0.2)).collect();
        let clipped = clip_coarse_to_fine_extent(&fine, &coarse, DVec3::Z);
        assert!(clipped.len() >= 2, "should produce at least 2 vertices");
        for p in &clipped {
            assert!(
                p.x >= 1.0 - 1e-9 && p.x <= 3.0 + 1e-9,
                "x={} outside fine extent",
                p.x
            );
        }
    }
}
