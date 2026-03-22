// Surface Nets meshing: converts voxel data into smooth triangle meshes.
//
// Surface Nets places one vertex per grid cell that contains a surface crossing
// (mix of solid and empty neighbors). The vertex is positioned at the centroid
// of the edge crossings within that cell. Quads are emitted for each edge shared
// by exactly four surface-crossing cells.
//
// This produces smoother meshes than Marching Cubes with simpler code and no
// lookup tables. The output is a Bevy Mesh with positions, normals, and colors.

#![allow(dead_code)]

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, Mesh, PrimitiveTopology};
use bevy::prelude::*;
use std::collections::HashMap;

use super::chunk::{Chunk, CHUNK_SIZE};
use super::voxel::MaterialId;

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
fn material_color(mat: MaterialId) -> [f32; 4] {
    match mat.0 {
        0 => [0.0, 0.0, 0.0, 0.0],    // air (shouldn't appear)
        1 => [0.5, 0.5, 0.5, 1.0],    // stone: grey
        2 => [0.45, 0.30, 0.15, 1.0], // dirt: brown
        3 => [0.2, 0.4, 0.8, 0.8],    // water: blue, semi-transparent
        4 => [0.2, 0.6, 0.1, 1.0],    // grass: green
        5 => [0.7, 0.55, 0.1, 1.0],   // iron: yellowish
        6 => [0.4, 0.25, 0.1, 1.0],   // wood: dark brown
        7 => [0.85, 0.8, 0.55, 1.0],  // sand: tan
        _ => [0.8, 0.0, 0.8, 1.0],    // unknown: magenta
    }
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
    let size = CHUNK_SIZE as i32;

    // Phase 1: For each cell, determine if it contains a surface crossing.
    // A cell is the cube from (x,y,z) to (x+1,y+1,z+1).
    // We iterate cells from 0..size-1 (staying within chunk bounds for the +1 samples,
    // which may be out-of-bounds and treated as air).
    let mut vertex_map: HashMap<(i32, i32, i32), u32> = HashMap::new();
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut colors: Vec<[f32; 4]> = Vec::new();

    // The 8 corners of a cell, as offsets from (x, y, z)
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

    for cz in 0..size {
        for cy in 0..size {
            for cx in 0..size {
                // Sample all 8 corners of this cell
                let mut corner_values = [0.0f32; 8];
                let mut corner_mats = [MaterialId::AIR; 8];
                let mut solid_count = 0u32;

                for (i, &(dx, dy, dz)) in corners.iter().enumerate() {
                    let (val, mat) = sample(chunk, cx + dx, cy + dy, cz + dz);
                    corner_values[i] = val;
                    corner_mats[i] = mat;
                    if val > 0.5 {
                        solid_count += 1;
                    }
                }

                // Skip cells that are entirely solid or entirely empty
                if solid_count == 0 || solid_count == 8 {
                    continue;
                }

                // Place vertex at centroid of edge crossings
                let mut vertex_pos = Vec3::ZERO;
                let mut crossing_count = 0u32;

                // Check all 12 edges of the cell for sign changes
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

                for &(a, b) in &edges {
                    let va = corner_values[a];
                    let vb = corner_values[b];
                    // Sign change: one is solid, one is air
                    if (va > 0.5) != (vb > 0.5) {
                        // Interpolate position along the edge
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

                // Pick the dominant solid material for coloring
                let dominant_mat = corner_mats
                    .iter()
                    .copied()
                    .find(|m| !m.is_air())
                    .unwrap_or(MaterialId::STONE);

                let idx = positions.len() as u32;
                vertex_map.insert((cx, cy, cz), idx);
                positions.push(world_pos);
                normals.push([0.0, 1.0, 0.0]); // placeholder, computed later
                colors.push(material_color(dominant_mat));
            }
        }
    }

    // Phase 2: Emit quads for edges shared by 4 surface cells.
    // For each cell that has a vertex, check the 3 positive-direction edges (X, Y, Z).
    // Each edge is shared by 4 cells; if all 4 have vertices, emit a quad.
    let mut indices: Vec<u32> = Vec::new();

    for (&(cx, cy, cz), &v0) in &vertex_map {
        // X-edge: shared by cells (cx,cy,cz), (cx,cy-1,cz), (cx,cy,cz-1), (cx,cy-1,cz-1)
        if let (Some(&v1), Some(&v2), Some(&v3)) = (
            vertex_map.get(&(cx, cy - 1, cz)),
            vertex_map.get(&(cx, cy, cz - 1)),
            vertex_map.get(&(cx, cy - 1, cz - 1)),
        ) {
            let (s0, _) = sample(chunk, cx, cy, cz);
            let (s1, _) = sample(chunk, cx + 1, cy, cz);
            if (s0 > 0.5) != (s1 > 0.5) {
                if s0 > 0.5 {
                    emit_quad(&mut indices, v0, v1, v3, v2);
                } else {
                    emit_quad(&mut indices, v0, v2, v3, v1);
                }
            }
        }

        // Y-edge: shared by cells (cx,cy,cz), (cx-1,cy,cz), (cx,cy,cz-1), (cx-1,cy,cz-1)
        if let (Some(&v1), Some(&v2), Some(&v3)) = (
            vertex_map.get(&(cx - 1, cy, cz)),
            vertex_map.get(&(cx, cy, cz - 1)),
            vertex_map.get(&(cx - 1, cy, cz - 1)),
        ) {
            let (s0, _) = sample(chunk, cx, cy, cz);
            let (s1, _) = sample(chunk, cx, cy + 1, cz);
            if (s0 > 0.5) != (s1 > 0.5) {
                if s0 > 0.5 {
                    emit_quad(&mut indices, v0, v2, v3, v1);
                } else {
                    emit_quad(&mut indices, v0, v1, v3, v2);
                }
            }
        }

        // Z-edge: shared by cells (cx,cy,cz), (cx-1,cy,cz), (cx,cy-1,cz), (cx-1,cy-1,cz)
        if let (Some(&v1), Some(&v2), Some(&v3)) = (
            vertex_map.get(&(cx - 1, cy, cz)),
            vertex_map.get(&(cx, cy - 1, cz)),
            vertex_map.get(&(cx - 1, cy - 1, cz)),
        ) {
            let (s0, _) = sample(chunk, cx, cy, cz);
            let (s1, _) = sample(chunk, cx, cy, cz + 1);
            if (s0 > 0.5) != (s1 > 0.5) {
                if s0 > 0.5 {
                    emit_quad(&mut indices, v0, v1, v3, v2);
                } else {
                    emit_quad(&mut indices, v0, v2, v3, v1);
                }
            }
        }
    }

    // Phase 3: Compute normals from triangle faces.
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

/// System: generates or updates meshes for dirty chunks.
pub fn mesh_dirty_chunks(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut chunk_q: Query<(Entity, &mut Chunk, Option<&Mesh3d>), Without<ChunkMeshMarker>>,
    mut remesh_q: Query<(Entity, &mut Chunk, &Mesh3d), With<ChunkMeshMarker>>,
) {
    // Default material for chunk meshes (vertex colored)
    let chunk_material = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        // We use vertex colors, so set the base to white and let colors multiply
        ..default()
    });

    // Initial mesh generation for new chunks without a mesh
    for (entity, mut chunk, existing_mesh) in chunk_q.iter_mut() {
        if !chunk.is_dirty() {
            continue;
        }

        let chunk_mesh = generate_mesh(&chunk);
        chunk.clear_dirty();

        if chunk_mesh.is_empty() {
            continue;
        }

        let bevy_mesh = chunk_mesh_to_bevy_mesh(&chunk_mesh);
        let mesh_handle = meshes.add(bevy_mesh);

        if existing_mesh.is_some() {
            // Already has a mesh — just update the handle
            commands
                .entity(entity)
                .insert(Mesh3d(mesh_handle))
                .insert(ChunkMeshMarker);
        } else {
            commands
                .entity(entity)
                .insert(Mesh3d(mesh_handle))
                .insert(MeshMaterial3d(chunk_material.clone()))
                .insert(ChunkMeshMarker);
        }
    }

    // Remesh already-meshed chunks that got dirty again
    for (entity, mut chunk, _mesh) in remesh_q.iter_mut() {
        if !chunk.is_dirty() {
            continue;
        }

        let chunk_mesh = generate_mesh(&chunk);
        chunk.clear_dirty();

        if chunk_mesh.is_empty() {
            // Remove the mesh if chunk is now empty
            commands
                .entity(entity)
                .remove::<Mesh3d>()
                .remove::<MeshMaterial3d<StandardMaterial>>()
                .remove::<ChunkMeshMarker>();
        } else {
            let bevy_mesh = chunk_mesh_to_bevy_mesh(&chunk_mesh);
            let mesh_handle = meshes.add(bevy_mesh);
            commands.entity(entity).insert(Mesh3d(mesh_handle));
        }
    }
}

/// Plugin that registers the chunk meshing system.
pub struct MeshingPlugin;

impl Plugin for MeshingPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, mesh_dirty_chunks);
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
        let stone = material_color(MaterialId::STONE);
        assert_eq!(stone[3], 1.0); // opaque

        let water = material_color(MaterialId::WATER);
        assert!(water[3] < 1.0); // semi-transparent

        let air = material_color(MaterialId::AIR);
        assert_eq!(air[3], 0.0); // invisible

        // Unknown material should be magenta
        let unknown = material_color(MaterialId(999));
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
}
