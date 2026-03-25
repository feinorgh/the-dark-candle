// Surface Nets meshing: converts voxel data into smooth triangle meshes.
//
// Surface Nets places one vertex per grid cell that contains a surface crossing
// (mix of solid and empty neighbors). The vertex is positioned at the centroid
// of the edge crossings within that cell. Quads are emitted for each edge shared
// by exactly four surface-crossing cells.
//
// This produces smoother meshes than Marching Cubes with simpler code and no
// lookup tables. The output is a Bevy Mesh with positions, normals, and colors.

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, Mesh, PrimitiveTopology};
use bevy::prelude::*;
use std::collections::HashMap;

use super::chunk::{CHUNK_SIZE, Chunk, ChunkOctree};
use super::lod::{LodConfig, LodLevel, MaterialColorMap, chunk_lod_level_with_hysteresis};
use super::octree::OctreeNode;
use super::voxel::{MaterialId, Voxel};

/// Output of the meshing pass for a single chunk.
pub struct ChunkMesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub colors: Vec<[f32; 4]>,
    pub indices: Vec<u32>,
}

impl ChunkMesh {
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }
}

/// Map a material ID to an RGBA color for rendering.
///
/// Hardcoded fallback used when no `MaterialColorMap` is available.
fn material_color_fallback(mat: MaterialId) -> [f32; 4] {
    match mat.0 {
        0 => [0.0, 0.0, 0.0, 0.0],    // air (shouldn't appear)
        1 => [0.5, 0.5, 0.5, 1.0],    // stone: grey
        2 => [0.45, 0.30, 0.15, 1.0], // dirt: brown
        3 => [0.2, 0.4, 0.8, 0.8],    // water: blue, semi-transparent
        4 => [0.2, 0.6, 0.1, 1.0],    // grass: green
        5 => [0.7, 0.55, 0.1, 1.0],   // iron: yellowish
        6 => [0.4, 0.25, 0.1, 1.0],   // wood: dark brown
        7 => [0.85, 0.8, 0.55, 1.0],  // sand: tan
        8 => [0.7, 0.85, 1.0, 0.9],   // ice: pale blue
        9 => [0.9, 0.9, 0.95, 0.3],   // steam: faint white
        10 => [1.0, 0.3, 0.0, 1.0],   // lava: orange-red
        11 => [0.3, 0.3, 0.3, 1.0],   // ash: dark grey
        _ => [0.8, 0.0, 0.8, 1.0],    // unknown: magenta
    }
}

/// Resolve a material's RGBA color, preferring the registry-backed color map.
fn material_color(mat: MaterialId, color_map: Option<&MaterialColorMap>) -> [f32; 4] {
    color_map
        .map(|m| m.get(mat))
        .unwrap_or_else(|| material_color_fallback(mat))
}

/// Sample the scalar field: 1.0 for solid, 0.0 for air.
/// Uses an extended grid (CHUNK_SIZE + 1) so we can form cells at the boundary.
/// Out-of-bounds samples return 0.0 (air) to create surfaces at chunk edges.
#[inline]
fn sample(chunk: &Chunk, x: i32, y: i32, z: i32) -> (f32, MaterialId) {
    if x >= 0
        && y >= 0
        && z >= 0
        && (x as usize) < CHUNK_SIZE
        && (y as usize) < CHUNK_SIZE
        && (z as usize) < CHUNK_SIZE
    {
        let v = chunk.get(x as usize, y as usize, z as usize);
        if v.is_air() {
            (0.0, MaterialId::AIR)
        } else {
            (1.0, v.material)
        }
    } else {
        (0.0, MaterialId::AIR)
    }
}

/// Generate a mesh from a chunk's voxel data using the Surface Nets algorithm.
pub fn generate_mesh(chunk: &Chunk) -> ChunkMesh {
    generate_mesh_with_colors(chunk, None)
}

/// Generate a mesh from a chunk, using the color map for material colors.
pub fn generate_mesh_with_colors(chunk: &Chunk, color_map: Option<&MaterialColorMap>) -> ChunkMesh {
    generate_mesh_generic(
        CHUNK_SIZE as i32,
        |x, y, z| sample(chunk, x, y, z),
        |mat| material_color(mat, color_map),
    )
}

/// Sample the scalar field from an octree at a given cell coordinate and size.
/// Returns (solidity, material). Out-of-bounds → air.
#[inline]
fn sample_octree(
    tree: &OctreeNode<Voxel>,
    size: usize,
    x: i32,
    y: i32,
    z: i32,
) -> (f32, MaterialId) {
    if x >= 0
        && y >= 0
        && z >= 0
        && (x as usize) < size
        && (y as usize) < size
        && (z as usize) < size
    {
        let v = tree.get(x as usize, y as usize, z as usize, size);
        if v.is_air() {
            (0.0, MaterialId::AIR)
        } else {
            (1.0, v.material)
        }
    } else {
        (0.0, MaterialId::AIR)
    }
}

/// Generate a mesh from an octree volume using Surface Nets.
///
/// This produces identical results to `generate_mesh()` when the octree
/// represents the same data as a flat chunk at base resolution. It also
/// works correctly with sub-voxel octree data (higher depth leaves are
/// read at base resolution via `OctreeNode::get()`).
///
/// `size` is the grid dimension (e.g. CHUNK_SIZE = 32).
pub fn generate_mesh_from_octree(tree: &OctreeNode<Voxel>, size: usize) -> ChunkMesh {
    generate_mesh_from_octree_with_colors(tree, size, None)
}

/// Generate a mesh from an octree volume, using the color map for materials.
pub fn generate_mesh_from_octree_with_colors(
    tree: &OctreeNode<Voxel>,
    size: usize,
    color_map: Option<&MaterialColorMap>,
) -> ChunkMesh {
    generate_mesh_generic(
        size as i32,
        |x, y, z| sample_octree(tree, size, x, y, z),
        |mat| material_color(mat, color_map),
    )
}

/// Generate a mesh at reduced resolution for LOD.
///
/// `lod_step` is the cell stride: 1 = full resolution (32³), 2 = half (16³),
/// 4 = quarter (8³), etc. The output mesh covers the same world-space volume
/// but with fewer vertices and triangles.
pub fn generate_mesh_lod(chunk: &Chunk, lod_step: usize) -> ChunkMesh {
    generate_mesh_lod_with_colors(chunk, lod_step, None)
}

/// Generate a mesh at reduced resolution for LOD, using the color map.
pub fn generate_mesh_lod_with_colors(
    chunk: &Chunk,
    lod_step: usize,
    color_map: Option<&MaterialColorMap>,
) -> ChunkMesh {
    assert!(lod_step > 0 && lod_step.is_power_of_two());
    let effective_size = CHUNK_SIZE / lod_step;
    let step = lod_step as i32;

    generate_mesh_generic(
        effective_size as i32,
        |x, y, z| {
            let fx = x * step;
            let fy = y * step;
            let fz = z * step;
            sample(chunk, fx, fy, fz)
        },
        |mat| material_color(mat, color_map),
    )
}

/// Generate a mesh from an octree at reduced resolution for LOD.
pub fn generate_mesh_from_octree_lod(
    tree: &OctreeNode<Voxel>,
    size: usize,
    lod_step: usize,
) -> ChunkMesh {
    generate_mesh_from_octree_lod_with_colors(tree, size, lod_step, None)
}

/// Generate a mesh from an octree at reduced resolution, using the color map.
pub fn generate_mesh_from_octree_lod_with_colors(
    tree: &OctreeNode<Voxel>,
    size: usize,
    lod_step: usize,
    color_map: Option<&MaterialColorMap>,
) -> ChunkMesh {
    assert!(lod_step > 0 && lod_step.is_power_of_two());
    let effective_size = size / lod_step;
    let step = lod_step as i32;

    generate_mesh_generic(
        effective_size as i32,
        |x, y, z| {
            let fx = x * step;
            let fy = y * step;
            let fz = z * step;
            sample_octree(tree, size, fx, fy, fz)
        },
        |mat| material_color(mat, color_map),
    )
}

/// Core Surface Nets implementation parameterized over sampling and color functions.
///
/// `grid_size` is the number of cells along each axis.
/// `sample_fn(x, y, z)` returns (scalar, material) for the given cell corner.
/// `color_fn(mat)` returns the RGBA color for a material ID.
fn generate_mesh_generic<F, C>(grid_size: i32, sample_fn: F, color_fn: C) -> ChunkMesh
where
    F: Fn(i32, i32, i32) -> (f32, MaterialId),
    C: Fn(MaterialId) -> [f32; 4],
{
    let corners: [(i32, i32, i32); 8] = [
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
    ];

    let edges: [(usize, usize); 12] = [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7), // X edges
        (0, 2),
        (1, 3),
        (4, 6),
        (5, 7), // Y edges
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7), // Z edges
    ];

    let mut vertex_map: HashMap<(i32, i32, i32), u32> = HashMap::new();
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut colors: Vec<[f32; 4]> = Vec::new();

    // Phase 1: Vertex placement
    for cz in 0..grid_size {
        for cy in 0..grid_size {
            for cx in 0..grid_size {
                let mut corner_values = [0.0f32; 8];
                let mut corner_mats = [MaterialId::AIR; 8];
                let mut solid_count = 0u32;

                for (i, &(dx, dy, dz)) in corners.iter().enumerate() {
                    let (val, mat) = sample_fn(cx + dx, cy + dy, cz + dz);
                    corner_values[i] = val;
                    corner_mats[i] = mat;
                    if val > 0.5 {
                        solid_count += 1;
                    }
                }

                if solid_count == 0 || solid_count == 8 {
                    continue;
                }

                let mut vertex_pos = Vec3::ZERO;
                let mut crossing_count = 0u32;

                for &(a, b) in &edges {
                    let va = corner_values[a];
                    let vb = corner_values[b];
                    if (va > 0.5) != (vb > 0.5) {
                        let t = (0.5 - va) / (vb - va);
                        let pa = Vec3::new(
                            corners[a].0 as f32,
                            corners[a].1 as f32,
                            corners[a].2 as f32,
                        );
                        let pb = Vec3::new(
                            corners[b].0 as f32,
                            corners[b].1 as f32,
                            corners[b].2 as f32,
                        );
                        vertex_pos += pa + t * (pb - pa);
                        crossing_count += 1;
                    }
                }

                if crossing_count == 0 {
                    continue;
                }

                vertex_pos /= crossing_count as f32;
                let world_pos = [
                    cx as f32 + vertex_pos.x,
                    cy as f32 + vertex_pos.y,
                    cz as f32 + vertex_pos.z,
                ];

                let dominant_mat = corner_mats
                    .iter()
                    .copied()
                    .find(|m| !m.is_air())
                    .unwrap_or(MaterialId::STONE);

                let idx = positions.len() as u32;
                vertex_map.insert((cx, cy, cz), idx);
                positions.push(world_pos);
                normals.push([0.0, 1.0, 0.0]);
                colors.push(color_fn(dominant_mat));
            }
        }
    }

    // Phase 2: Quad emission
    let mut indices: Vec<u32> = Vec::new();

    for (&(cx, cy, cz), &v0) in &vertex_map {
        // X-edge
        if let (Some(&v1), Some(&v2), Some(&v3)) = (
            vertex_map.get(&(cx, cy - 1, cz)),
            vertex_map.get(&(cx, cy, cz - 1)),
            vertex_map.get(&(cx, cy - 1, cz - 1)),
        ) {
            let (s0, _) = sample_fn(cx, cy, cz);
            let (s1, _) = sample_fn(cx + 1, cy, cz);
            if (s0 > 0.5) != (s1 > 0.5) {
                if s0 > 0.5 {
                    emit_quad(&mut indices, v0, v1, v3, v2);
                } else {
                    emit_quad(&mut indices, v0, v2, v3, v1);
                }
            }
        }

        // Y-edge
        if let (Some(&v1), Some(&v2), Some(&v3)) = (
            vertex_map.get(&(cx - 1, cy, cz)),
            vertex_map.get(&(cx, cy, cz - 1)),
            vertex_map.get(&(cx - 1, cy, cz - 1)),
        ) {
            let (s0, _) = sample_fn(cx, cy, cz);
            let (s1, _) = sample_fn(cx, cy + 1, cz);
            if (s0 > 0.5) != (s1 > 0.5) {
                if s0 > 0.5 {
                    emit_quad(&mut indices, v0, v2, v3, v1);
                } else {
                    emit_quad(&mut indices, v0, v1, v3, v2);
                }
            }
        }

        // Z-edge
        if let (Some(&v1), Some(&v2), Some(&v3)) = (
            vertex_map.get(&(cx - 1, cy, cz)),
            vertex_map.get(&(cx, cy - 1, cz)),
            vertex_map.get(&(cx - 1, cy - 1, cz)),
        ) {
            let (s0, _) = sample_fn(cx, cy, cz);
            let (s1, _) = sample_fn(cx, cy, cz + 1);
            if (s0 > 0.5) != (s1 > 0.5) {
                if s0 > 0.5 {
                    emit_quad(&mut indices, v0, v1, v3, v2);
                } else {
                    emit_quad(&mut indices, v0, v2, v3, v1);
                }
            }
        }
    }

    // Phase 3: Normals
    compute_normals(&positions, &indices, &mut normals);

    ChunkMesh {
        positions,
        normals,
        colors,
        indices,
    }
}

/// Emit a quad as two triangles.
#[inline]
fn emit_quad(indices: &mut Vec<u32>, a: u32, b: u32, c: u32, d: u32) {
    // Triangle 1: a, b, c
    indices.push(a);
    indices.push(b);
    indices.push(c);
    // Triangle 2: a, c, d
    indices.push(a);
    indices.push(c);
    indices.push(d);
}

/// Compute smooth vertex normals by averaging face normals of adjacent triangles.
fn compute_normals(positions: &[[f32; 3]], indices: &[u32], normals: &mut [[f32; 3]]) {
    // Reset all normals to zero
    for n in normals.iter_mut() {
        *n = [0.0, 0.0, 0.0];
    }

    // Accumulate face normals
    for tri in indices.chunks(3) {
        if tri.len() < 3 {
            continue;
        }
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let p0 = Vec3::from_array(positions[i0]);
        let p1 = Vec3::from_array(positions[i1]);
        let p2 = Vec3::from_array(positions[i2]);

        let face_normal = (p1 - p0).cross(p2 - p0);

        for &idx in &[i0, i1, i2] {
            normals[idx][0] += face_normal.x;
            normals[idx][1] += face_normal.y;
            normals[idx][2] += face_normal.z;
        }
    }

    // Normalize
    for n in normals.iter_mut() {
        let v = Vec3::from_array(*n);
        let len = v.length();
        if len > 1e-8 {
            *n = (v / len).to_array();
        } else {
            *n = [0.0, 1.0, 0.0]; // fallback up
        }
    }
}

/// Convert a ChunkMesh into a Bevy Mesh asset.
pub fn chunk_mesh_to_bevy_mesh(chunk_mesh: &ChunkMesh) -> Mesh {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, chunk_mesh.positions.clone());
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, chunk_mesh.normals.clone());
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, chunk_mesh.colors.clone());
    mesh.insert_indices(Indices::U32(chunk_mesh.indices.clone()));
    mesh
}

/// Marker component for entities that have a chunk mesh.
#[derive(Component)]
pub struct ChunkMeshMarker;

/// Tracks the current LOD level assigned to a chunk.
///
/// Used by the meshing system to determine mesh stride and detect LOD changes.
#[derive(Component, Debug, Clone, Copy)]
pub struct ChunkLod(pub LodLevel);

/// Active LOD transition state for smooth blending.
///
/// While present, the chunk fades between its old and new LOD meshes.
/// `factor` advances from 0.0 (old LOD) to 1.0 (new LOD fully visible).
#[derive(Component, Debug, Clone)]
pub struct LodTransition {
    /// LOD level before the transition.
    pub previous_level: LodLevel,
    /// Blend factor in [0.0, 1.0]. 1.0 = transition complete.
    pub factor: f32,
}

/// Duration of an LOD transition in seconds.
const LOD_TRANSITION_DURATION: f32 = 0.4;

/// Compute the mesh stride for a given LOD level.
/// L0 = 1 (full 32³), L1 = 2 (16³), L2 = 4 (8³), etc.
fn lod_step(level: LodLevel) -> usize {
    let step = 1usize << level.0;
    step.min(CHUNK_SIZE / 2) // Don't go below 2³
}

/// Generate a mesh for a chunk, preferring the octree path when available.
///
/// When a `ChunkOctree` is present, meshes from the sparse representation.
/// Falls back to flat-array meshing otherwise (e.g. for chunks modified
/// after octree construction whose octree hasn't been rebuilt yet).
fn generate_chunk_mesh(
    chunk: &Chunk,
    octree: Option<&ChunkOctree>,
    lod_step_size: usize,
    color_map: Option<&MaterialColorMap>,
) -> ChunkMesh {
    match octree {
        Some(oct) if lod_step_size <= 1 => {
            generate_mesh_from_octree_with_colors(&oct.0, CHUNK_SIZE, color_map)
        }
        Some(oct) => {
            generate_mesh_from_octree_lod_with_colors(&oct.0, CHUNK_SIZE, lod_step_size, color_map)
        }
        None if lod_step_size <= 1 => generate_mesh_with_colors(chunk, color_map),
        None => generate_mesh_lod_with_colors(chunk, lod_step_size, color_map),
    }
}

/// System: generates or updates meshes for dirty chunks, respecting LOD levels.
///
/// Queries the camera position to compute per-chunk LOD. When a `ChunkOctree`
/// component is present, meshes from the sparse octree for better cache locality
/// and compression-aware LOD. Falls back to flat-array meshing otherwise.
/// LOD transitions trigger remeshing with a smooth opacity fade via `LodTransition`.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn mesh_dirty_chunks(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    lod_config: Option<Res<LodConfig>>,
    color_map: Option<Res<MaterialColorMap>>,
    camera_q: Query<&Transform, With<Camera3d>>,
    mut chunk_q: Query<
        (
            Entity,
            &mut Chunk,
            &super::chunk::ChunkCoord,
            Option<&ChunkOctree>,
            Option<&Mesh3d>,
            Option<&ChunkLod>,
        ),
        Without<ChunkMeshMarker>,
    >,
    mut remesh_q: Query<
        (
            Entity,
            &mut Chunk,
            &super::chunk::ChunkCoord,
            Option<&ChunkOctree>,
            &Mesh3d,
            Option<&ChunkLod>,
        ),
        With<ChunkMeshMarker>,
    >,
) {
    let default_config = LodConfig::default();
    let config = lod_config.as_deref().unwrap_or(&default_config);
    let colors = color_map.as_deref();
    let camera_pos = camera_q
        .iter()
        .next()
        .map(|t| t.translation)
        .unwrap_or(Vec3::ZERO);

    // Default material for chunk meshes (vertex colored)
    let chunk_material = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        ..default()
    });

    // Initial mesh generation for new chunks without a mesh
    for (entity, mut chunk, coord, octree, existing_mesh, current_lod) in chunk_q.iter_mut() {
        let current_level = current_lod.map(|l| l.0).unwrap_or(LodLevel(0));
        let new_level = chunk_lod_level_with_hysteresis(coord, camera_pos, current_level, config);
        let lod_changed = current_lod.is_some() && new_level != current_level;

        if !chunk.is_dirty() && !lod_changed {
            continue;
        }

        let step = lod_step(new_level);
        let chunk_mesh = generate_chunk_mesh(&chunk, octree, step, colors);
        chunk.clear_dirty();

        if chunk_mesh.is_empty() {
            commands.entity(entity).insert(ChunkLod(new_level));
            continue;
        }

        let bevy_mesh = chunk_mesh_to_bevy_mesh(&chunk_mesh);
        let mesh_handle = meshes.add(bevy_mesh);

        let mut cmds = commands.entity(entity);
        cmds.insert(Mesh3d(mesh_handle))
            .insert(ChunkLod(new_level))
            .insert(ChunkMeshMarker);

        if existing_mesh.is_none() {
            cmds.insert(MeshMaterial3d(chunk_material.clone()));
        }

        if lod_changed {
            cmds.insert(LodTransition {
                previous_level: current_level,
                factor: 0.0,
            });
        }
    }

    // Remesh already-meshed chunks that got dirty or changed LOD
    for (entity, mut chunk, coord, octree, _mesh, current_lod) in remesh_q.iter_mut() {
        let current_level = current_lod.map(|l| l.0).unwrap_or(LodLevel(0));
        let new_level = chunk_lod_level_with_hysteresis(coord, camera_pos, current_level, config);
        let lod_changed = new_level != current_level;

        if !chunk.is_dirty() && !lod_changed {
            continue;
        }

        let step = lod_step(new_level);
        let chunk_mesh = generate_chunk_mesh(&chunk, octree, step, colors);
        chunk.clear_dirty();

        if chunk_mesh.is_empty() {
            commands
                .entity(entity)
                .remove::<Mesh3d>()
                .remove::<MeshMaterial3d<StandardMaterial>>()
                .remove::<ChunkMeshMarker>()
                .insert(ChunkLod(new_level));
        } else {
            let bevy_mesh = chunk_mesh_to_bevy_mesh(&chunk_mesh);
            let mesh_handle = meshes.add(bevy_mesh);
            let mut cmds = commands.entity(entity);
            cmds.insert(Mesh3d(mesh_handle)).insert(ChunkLod(new_level));

            if lod_changed {
                cmds.insert(LodTransition {
                    previous_level: current_level,
                    factor: 0.0,
                });
            }
        }
    }
}

/// System: advances LOD transitions and applies opacity fade.
///
/// Each frame, `factor` is advanced toward 1.0. During the transition,
/// the chunk's `StandardMaterial` alpha is scaled by the blend factor
/// using a Hermite smoothstep for visual smoothness. When the transition
/// completes, full opacity is restored and the component is removed.
pub fn tick_lod_transitions(
    mut commands: Commands,
    time: Res<Time>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut query: Query<(
        Entity,
        &mut LodTransition,
        Option<&MeshMaterial3d<StandardMaterial>>,
    )>,
) {
    let dt = time.delta_secs();

    for (entity, mut transition, mat_handle) in query.iter_mut() {
        transition.factor += dt / LOD_TRANSITION_DURATION;

        if transition.factor >= 1.0 {
            // Transition complete — restore full opacity and clean up.
            if let Some(mat_h) = mat_handle
                && let Some(mat) = materials.get_mut(&mat_h.0)
            {
                mat.base_color = Color::WHITE;
                mat.alpha_mode = AlphaMode::Opaque;
            }
            commands.entity(entity).remove::<LodTransition>();
        } else {
            // Hermite smoothstep for perceptually smooth fade.
            let t = transition.factor;
            let alpha = t * t * (3.0 - 2.0 * t);
            if let Some(mat_h) = mat_handle
                && let Some(mat) = materials.get_mut(&mat_h.0)
            {
                mat.base_color = Color::srgba(1.0, 1.0, 1.0, alpha);
                mat.alpha_mode = AlphaMode::Blend;
            }
        }
    }
}

/// Plugin that registers the chunk meshing system.
pub struct MeshingPlugin;

impl Plugin for MeshingPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (mesh_dirty_chunks, tick_lod_transitions).in_set(super::WorldSet::Meshing),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::super::chunk::ChunkCoord;
    use super::*;

    #[test]
    fn empty_chunk_produces_no_mesh() {
        let chunk = Chunk::new_empty(ChunkCoord::new(0, 0, 0));
        let mesh = generate_mesh(&chunk);
        assert!(mesh.is_empty());
        assert_eq!(mesh.vertex_count(), 0);
        assert_eq!(mesh.triangle_count(), 0);
    }

    #[test]
    fn fully_solid_chunk_produces_boundary_mesh() {
        // A fully solid chunk has surfaces at its boundaries (where solid meets
        // out-of-bounds air)
        let chunk = Chunk::new_filled(ChunkCoord::new(0, 0, 0), MaterialId::STONE);
        let mesh = generate_mesh(&chunk);
        // Should have vertices on the boundary faces
        assert!(
            mesh.vertex_count() > 0,
            "Fully solid chunk should have boundary vertices"
        );
        assert!(
            mesh.triangle_count() > 0,
            "Fully solid chunk should have boundary triangles"
        );
    }

    #[test]
    fn single_solid_voxel_produces_mesh() {
        let mut chunk = Chunk::new_empty(ChunkCoord::new(0, 0, 0));
        // Place a single stone block in the middle
        chunk.set_material(16, 16, 16, MaterialId::STONE);
        let mesh = generate_mesh(&chunk);

        assert!(
            mesh.vertex_count() > 0,
            "Single voxel should produce vertices"
        );
        assert!(
            mesh.triangle_count() > 0,
            "Single voxel should produce triangles"
        );
    }

    #[test]
    fn mesh_has_matching_attribute_counts() {
        let mut chunk = Chunk::new_empty(ChunkCoord::new(0, 0, 0));
        // Fill bottom half
        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                for y in 0..16 {
                    chunk.set_material(x, y, z, MaterialId::DIRT);
                }
            }
        }
        let mesh = generate_mesh(&chunk);

        assert_eq!(
            mesh.positions.len(),
            mesh.normals.len(),
            "Position and normal counts must match"
        );
        assert_eq!(
            mesh.positions.len(),
            mesh.colors.len(),
            "Position and color counts must match"
        );
        // All indices must reference valid vertices
        for &idx in &mesh.indices {
            assert!(
                (idx as usize) < mesh.positions.len(),
                "Index {} out of bounds (vertex count: {})",
                idx,
                mesh.positions.len()
            );
        }
        // Index count must be a multiple of 3 (triangles)
        assert_eq!(mesh.indices.len() % 3, 0);
    }

    #[test]
    fn normals_are_normalized() {
        let mut chunk = Chunk::new_empty(ChunkCoord::new(0, 0, 0));
        for x in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                chunk.set_material(x, 0, z, MaterialId::STONE);
            }
        }
        let mesh = generate_mesh(&chunk);

        for (i, n) in mesh.normals.iter().enumerate() {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!(
                (len - 1.0).abs() < 0.01,
                "Normal {} has length {}, expected ~1.0",
                i,
                len
            );
        }
    }

    #[test]
    fn material_color_returns_correct_colors() {
        let stone = material_color(MaterialId::STONE, None);
        assert_eq!(stone[3], 1.0); // opaque

        let water = material_color(MaterialId::WATER, None);
        assert!(water[3] < 1.0); // semi-transparent

        let air = material_color(MaterialId::AIR, None);
        assert_eq!(air[3], 0.0); // invisible

        // Unknown material should be magenta
        let unknown = material_color(MaterialId(999), None);
        assert_eq!(unknown, [0.8, 0.0, 0.8, 1.0]);
    }

    #[test]
    fn bevy_mesh_conversion_succeeds() {
        let mut chunk = Chunk::new_empty(ChunkCoord::new(0, 0, 0));
        chunk.set_material(5, 5, 5, MaterialId::STONE);
        let chunk_mesh = generate_mesh(&chunk);

        // This should not panic
        let _bevy_mesh = chunk_mesh_to_bevy_mesh(&chunk_mesh);
    }

    #[test]
    fn different_materials_get_different_colors() {
        let mut chunk = Chunk::new_empty(ChunkCoord::new(0, 0, 0));
        // Place two isolated blocks of different materials
        chunk.set_material(5, 5, 5, MaterialId::STONE);
        chunk.set_material(20, 20, 20, MaterialId::DIRT);

        let mesh = generate_mesh(&chunk);

        // Collect unique colors
        let unique_colors: std::collections::HashSet<[u32; 4]> = mesh
            .colors
            .iter()
            .map(|c| {
                [
                    c[0].to_bits(),
                    c[1].to_bits(),
                    c[2].to_bits(),
                    c[3].to_bits(),
                ]
            })
            .collect();

        assert!(
            unique_colors.len() >= 2,
            "Expected at least 2 distinct colors, got {}",
            unique_colors.len()
        );
    }

    #[test]
    fn half_filled_chunk_produces_surface() {
        let mut chunk = Chunk::new_empty(ChunkCoord::new(0, 0, 0));
        // Fill the bottom half with stone
        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                for y in 0..CHUNK_SIZE / 2 {
                    chunk.set_material(x, y, z, MaterialId::STONE);
                }
            }
        }

        let mesh = generate_mesh(&chunk);
        assert!(
            mesh.vertex_count() > 100,
            "Half-filled should have many vertices"
        );
        assert!(
            mesh.triangle_count() > 100,
            "Half-filled should have many triangles"
        );
    }

    // --- Octree meshing tests ---

    use super::super::octree::OctreeNode;
    use super::super::voxel::Voxel;
    use super::super::voxel_access::flat_to_octree;

    #[test]
    fn octree_mesh_empty_produces_nothing() {
        let tree = OctreeNode::new_leaf(Voxel::default());
        let mesh = generate_mesh_from_octree(&tree, CHUNK_SIZE);
        assert!(mesh.is_empty());
    }

    #[test]
    fn octree_mesh_matches_flat_single_voxel() {
        let mut chunk = Chunk::new_empty(ChunkCoord::new(0, 0, 0));
        chunk.set_material(16, 16, 16, MaterialId::STONE);

        let flat_mesh = generate_mesh(&chunk);
        let tree = chunk.to_octree();
        let octree_mesh = generate_mesh_from_octree(&tree, CHUNK_SIZE);

        assert_eq!(
            flat_mesh.vertex_count(),
            octree_mesh.vertex_count(),
            "Vertex counts should match: flat={}, octree={}",
            flat_mesh.vertex_count(),
            octree_mesh.vertex_count(),
        );
        assert_eq!(
            flat_mesh.triangle_count(),
            octree_mesh.triangle_count(),
            "Triangle counts should match"
        );
    }

    #[test]
    fn octree_mesh_matches_flat_half_filled() {
        let mut chunk = Chunk::new_empty(ChunkCoord::new(0, 0, 0));
        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                for y in 0..CHUNK_SIZE / 2 {
                    chunk.set_material(x, y, z, MaterialId::STONE);
                }
            }
        }

        let flat_mesh = generate_mesh(&chunk);
        let tree = chunk.to_octree();
        let octree_mesh = generate_mesh_from_octree(&tree, CHUNK_SIZE);

        assert_eq!(flat_mesh.vertex_count(), octree_mesh.vertex_count());
        assert_eq!(flat_mesh.triangle_count(), octree_mesh.triangle_count());
    }

    #[test]
    fn lod_mesh_has_fewer_vertices() {
        let mut chunk = Chunk::new_empty(ChunkCoord::new(0, 0, 0));
        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                for y in 0..CHUNK_SIZE / 2 {
                    chunk.set_material(x, y, z, MaterialId::STONE);
                }
            }
        }

        let full_mesh = generate_mesh(&chunk);
        let lod_mesh = generate_mesh_lod(&chunk, 2);

        assert!(
            lod_mesh.vertex_count() < full_mesh.vertex_count(),
            "LOD mesh should have fewer vertices: lod={}, full={}",
            lod_mesh.vertex_count(),
            full_mesh.vertex_count(),
        );
        assert!(lod_mesh.vertex_count() > 0, "LOD mesh should not be empty");
    }

    #[test]
    fn lod_mesh_from_octree_has_fewer_vertices() {
        let mut chunk = Chunk::new_empty(ChunkCoord::new(0, 0, 0));
        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                for y in 0..CHUNK_SIZE / 2 {
                    chunk.set_material(x, y, z, MaterialId::STONE);
                }
            }
        }

        let tree = chunk.to_octree();
        let full_mesh = generate_mesh_from_octree(&tree, CHUNK_SIZE);
        let lod_mesh = generate_mesh_from_octree_lod(&tree, CHUNK_SIZE, 4);

        assert!(
            lod_mesh.vertex_count() < full_mesh.vertex_count(),
            "LOD octree mesh should have fewer vertices"
        );
    }

    #[test]
    fn lod_mesh_attributes_consistent() {
        let mut chunk = Chunk::new_empty(ChunkCoord::new(0, 0, 0));
        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                chunk.set_material(x, 0, z, MaterialId::DIRT);
            }
        }

        let mesh = generate_mesh_lod(&chunk, 4);
        assert_eq!(mesh.positions.len(), mesh.normals.len());
        assert_eq!(mesh.positions.len(), mesh.colors.len());
        assert_eq!(mesh.indices.len() % 3, 0);
        for &idx in &mesh.indices {
            assert!((idx as usize) < mesh.positions.len());
        }
    }

    #[test]
    fn octree_mesh_fully_solid_has_boundary() {
        let tree = OctreeNode::new_leaf(Voxel::new(MaterialId::STONE));
        let mesh = generate_mesh_from_octree(&tree, CHUNK_SIZE);
        assert!(
            mesh.vertex_count() > 0,
            "Fully solid should have boundary vertices"
        );
    }

    #[test]
    fn small_octree_mesh_works() {
        // Test with a small 4×4×4 octree
        let size = 4;
        let mut flat = vec![Voxel::default(); size * size * size];
        flat[size * size + size + 1] = Voxel::new(MaterialId::STONE);
        let tree = flat_to_octree(&flat, size);

        let mesh = generate_mesh_from_octree(&tree, size);
        assert!(
            mesh.vertex_count() > 0,
            "Small octree mesh should have vertices"
        );
        assert!(
            mesh.triangle_count() > 0,
            "Small octree mesh should have triangles"
        );
    }
}
