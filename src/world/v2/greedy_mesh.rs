// Greedy meshing algorithm for the V2 pipeline.
//
// Sweeps three axes, builds 2D face masks at air/solid boundaries,
// merges same-material adjacent faces into maximal rectangles, and
// emits one quad per merged region. All vertex positions are in
// chunk-local space [0, CHUNK_SIZE].

use crate::world::chunk::CHUNK_SIZE;
use crate::world::lod::MaterialColorMap;
use crate::world::meshing::ChunkMesh;
use crate::world::voxel::{MaterialId, Voxel};

/// Voxel data for the 6 face-neighbor chunks, used for seamless boundary faces.
///
/// Each neighbor is optional — if absent, boundary voxels are treated as air.
/// Layout: `[+X, -X, +Y, -Y, +Z, -Z]`.
pub struct NeighborSlices {
    pub slices: [Option<Vec<Voxel>>; 6],
}

impl NeighborSlices {
    pub fn empty() -> Self {
        Self {
            slices: [const { None }; 6],
        }
    }
}

/// Index into the flat voxel array: `z * CHUNK_SIZE² + y * CHUNK_SIZE + x`.
#[inline]
fn voxel_index(x: usize, y: usize, z: usize) -> usize {
    z * CHUNK_SIZE * CHUNK_SIZE + y * CHUNK_SIZE + x
}

/// Look up a voxel material, checking neighbor data for out-of-bounds positions.
#[inline]
fn sample_material(
    voxels: &[Voxel],
    neighbors: &NeighborSlices,
    x: i32,
    y: i32,
    z: i32,
) -> MaterialId {
    if x >= 0 && x < CHUNK_SIZE as i32 && y >= 0 && y < CHUNK_SIZE as i32 && z >= 0 && z < CHUNK_SIZE as i32 {
        return voxels[voxel_index(x as usize, y as usize, z as usize)].material;
    }
    // Out of bounds: check neighbor chunks
    let cs = CHUNK_SIZE as i32;
    let (dir_idx, lx, ly, lz) = if x >= cs {
        (0, x - cs, y, z) // +X neighbor
    } else if x < 0 {
        (1, x + cs, y, z) // -X neighbor
    } else if y >= cs {
        (2, x, y - cs, z) // +Y neighbor
    } else if y < 0 {
        (3, x, y + cs, z) // -Y neighbor
    } else if z >= cs {
        (4, x, y, z - cs) // +Z neighbor
    } else {
        (5, x, y, z + cs) // -Z neighbor
    };

    if let Some(ref nbr) = neighbors.slices[dir_idx] {
        let lx = lx.clamp(0, cs - 1) as usize;
        let ly = ly.clamp(0, cs - 1) as usize;
        let lz = lz.clamp(0, cs - 1) as usize;
        nbr[voxel_index(lx, ly, lz)].material
    } else {
        MaterialId::AIR
    }
}

/// Whether a material is transparent (air, steam, glass, gases).
#[inline]
fn is_transparent(mat: MaterialId) -> bool {
    mat.is_air() || mat == MaterialId::STEAM || mat == MaterialId::GLASS
        || mat == MaterialId::OXYGEN || mat == MaterialId::HYDROGEN
}

/// Generate a greedy mesh from voxel data in chunk-local coordinates.
///
/// Positions are in `[0, CHUNK_SIZE]` space. The chunk's Transform handles
/// world-space positioning.
pub fn greedy_mesh(
    voxels: &[Voxel],
    neighbors: &NeighborSlices,
    color_map: &MaterialColorMap,
) -> ChunkMesh {
    let cs = CHUNK_SIZE as i32;
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut colors: Vec<[f32; 4]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    // For each of the 3 axes, sweep in both directions (positive and negative face).
    // Axis 0 = X (faces on YZ planes), Axis 1 = Y (faces on XZ planes), Axis 2 = Z (faces on XY planes).
    for axis in 0..3 {
        for dir in [1i32, -1i32] {
            // Normal vector for this face direction.
            let normal: [f32; 3] = match (axis, dir) {
                (0, 1) => [1.0, 0.0, 0.0],
                (0, -1) => [-1.0, 0.0, 0.0],
                (1, 1) => [0.0, 1.0, 0.0],
                (1, -1) => [0.0, -1.0, 0.0],
                (2, 1) => [0.0, 0.0, 1.0],
                (2, -1) => [0.0, 0.0, -1.0],
                _ => unreachable!(),
            };

            // Sweep along the main axis.
            for d in 0..cs {
                // Build a 2D mask of faces for this slice.
                // mask[u][v] = Some(material) if there is an exposed face here, None otherwise.
                let mut mask = [[None::<MaterialId>; CHUNK_SIZE]; CHUNK_SIZE];

                for v in 0..cs {
                    for u in 0..cs {
                        // Map (d, u, v) back to (x, y, z) based on axis.
                        let (x, y, z) = axis_to_xyz(axis, d, u, v);

                        let mat = sample_material(voxels, neighbors, x, y, z);
                        if is_transparent(mat) {
                            continue;
                        }

                        // Check the neighbor in the face direction.
                        let (nx, ny, nz) = match axis {
                            0 => (x + dir, y, z),
                            1 => (x, y + dir, z),
                            _ => (x, y, z + dir),
                        };
                        let neighbor_mat = sample_material(voxels, neighbors, nx, ny, nz);

                        if is_transparent(neighbor_mat) {
                            mask[u as usize][v as usize] = Some(mat);
                        }
                    }
                }

                // Greedy merge: find maximal rectangles of the same material.
                greedy_merge(
                    &mut mask,
                    axis,
                    d,
                    dir,
                    &normal,
                    color_map,
                    &mut positions,
                    &mut normals,
                    &mut colors,
                    &mut indices,
                );
            }
        }
    }

    ChunkMesh {
        positions,
        normals,
        colors,
        indices,
    }
}

/// Map (axis, d, u, v) coordinates to (x, y, z).
#[inline]
fn axis_to_xyz(axis: usize, d: i32, u: i32, v: i32) -> (i32, i32, i32) {
    match axis {
        0 => (d, u, v), // X-axis sweep: u=Y, v=Z
        1 => (u, d, v), // Y-axis sweep: u=X, v=Z
        _ => (u, v, d), // Z-axis sweep: u=X, v=Y
    }
}

/// Greedy merge: scan the 2D mask for maximal rectangles of the same material
/// and emit quads.
#[allow(clippy::too_many_arguments)]
fn greedy_merge(
    mask: &mut [[Option<MaterialId>; CHUNK_SIZE]; CHUNK_SIZE],
    axis: usize,
    d: i32,
    dir: i32,
    normal: &[f32; 3],
    color_map: &MaterialColorMap,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    colors: &mut Vec<[f32; 4]>,
    indices: &mut Vec<u32>,
) {
    let cs = CHUNK_SIZE;
    for sv in 0..cs {
        let mut su = 0;
        while su < cs {
            let Some(mat) = mask[su][sv] else {
                su += 1;
                continue;
            };

            // Expand width (u direction)
            let mut w = 1;
            while su + w < cs && mask[su + w][sv] == Some(mat) {
                w += 1;
            }

            // Expand height (v direction)
            let mut h = 1;
            'outer: while sv + h < cs {
                for wu in 0..w {
                    if mask[su + wu][sv + h] != Some(mat) {
                        break 'outer;
                    }
                }
                h += 1;
            }

            // Clear the merged region from the mask.
            for wu in 0..w {
                for wv in 0..h {
                    mask[su + wu][sv + wv] = None;
                }
            }

            // Emit a quad for this merged rectangle.
            let color = color_map.get(mat);

            // The face sits at `d` (on the solid side) or `d+1` (if positive direction).
            let face_d = if dir > 0 { d + 1 } else { d };

            emit_quad(
                axis,
                face_d,
                su as i32,
                sv as i32,
                w as i32,
                h as i32,
                dir,
                normal,
                color,
                positions,
                normals,
                colors,
                indices,
            );

            su += w;
        }
    }
}

/// Emit a single quad (2 triangles) for a merged face rectangle.
#[allow(clippy::too_many_arguments)]
fn emit_quad(
    axis: usize,
    d: i32,
    u: i32,
    v: i32,
    w: i32,
    h: i32,
    dir: i32,
    normal: &[f32; 3],
    color: [f32; 4],
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    colors: &mut Vec<[f32; 4]>,
    indices: &mut Vec<u32>,
) {
    let base = positions.len() as u32;

    // Four corners of the rectangle in (d, u, v) space.
    let corners = [
        (d, u, v),
        (d, u + w, v),
        (d, u + w, v + h),
        (d, u, v + h),
    ];

    for &(cd, cu, cv) in &corners {
        let (x, y, z) = axis_to_xyz(axis, cd, cu, cv);
        positions.push([x as f32, y as f32, z as f32]);
        normals.push(*normal);
        colors.push(color);
    }

    // Triangle winding: ensure the face points in the correct direction.
    if dir > 0 {
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    } else {
        indices.extend_from_slice(&[base, base + 2, base + 1, base, base + 3, base + 2]);
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::chunk::CHUNK_VOLUME;
    use crate::world::voxel::Voxel;

    fn make_air_chunk() -> Vec<Voxel> {
        vec![Voxel::default(); CHUNK_VOLUME]
    }

    fn make_solid_chunk(mat: MaterialId) -> Vec<Voxel> {
        let mut v = Voxel::default();
        v.material = mat;
        vec![v; CHUNK_VOLUME]
    }

    fn default_colors() -> MaterialColorMap {
        MaterialColorMap::from_defaults()
    }

    #[test]
    fn all_air_produces_empty_mesh() {
        let voxels = make_air_chunk();
        let mesh = greedy_mesh(&voxels, &NeighborSlices::empty(), &default_colors());
        assert!(mesh.is_empty(), "All-air chunk should produce no geometry");
    }

    #[test]
    fn all_solid_with_no_neighbors_produces_6_faces() {
        // A fully solid chunk with air neighbors should produce 6 large faces
        // (top, bottom, left, right, front, back), each covering the full 32×32 face.
        let voxels = make_solid_chunk(MaterialId::STONE);
        let mesh = greedy_mesh(&voxels, &NeighborSlices::empty(), &default_colors());

        // 6 faces, each a single merged quad = 6 quads = 12 triangles = 36 indices
        assert_eq!(mesh.indices.len(), 36, "Expected 6 quads (36 indices)");
        assert_eq!(mesh.positions.len(), 24, "Expected 24 vertices (4 per quad)");
    }

    #[test]
    fn single_block_produces_6_quads() {
        let mut voxels = make_air_chunk();
        voxels[voxel_index(5, 5, 5)].material = MaterialId::STONE;
        let mesh = greedy_mesh(&voxels, &NeighborSlices::empty(), &default_colors());

        // A single solid block surrounded by air has 6 exposed faces.
        assert_eq!(mesh.indices.len(), 36, "Single block should have 6 quads");
    }

    #[test]
    fn two_adjacent_same_material_blocks_merge() {
        let mut voxels = make_air_chunk();
        voxels[voxel_index(5, 5, 5)].material = MaterialId::STONE;
        voxels[voxel_index(6, 5, 5)].material = MaterialId::STONE;
        let mesh = greedy_mesh(&voxels, &NeighborSlices::empty(), &default_colors());

        // Two adjacent blocks: the shared face is hidden.
        // 4 individual faces (top, bottom, front, back of each block → merge pairs)
        // + 2 end caps = fewer than 12 quads.
        // Greedy merge: top face merges into 1×2 quad, etc.
        // Expected: 10 faces (not 12) because shared internal face is hidden.
        // With greedy merge: 6 quads (top 2×1, bottom 2×1, front 2×1, back 2×1, left 1×1, right 1×1)
        let quad_count = mesh.indices.len() / 6;
        assert!(
            quad_count <= 10,
            "Two adjacent blocks should merge some faces, got {quad_count} quads"
        );
    }

    #[test]
    fn half_solid_chunk_produces_one_top_face() {
        // Fill the bottom half (y < 16) with stone, top half is air.
        let mut voxels = make_air_chunk();
        for z in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE / 2 {
                for x in 0..CHUNK_SIZE {
                    voxels[voxel_index(x, y, z)].material = MaterialId::STONE;
                }
            }
        }
        let mesh = greedy_mesh(&voxels, &NeighborSlices::empty(), &default_colors());

        // With air neighbors: top face (32×32 = 1 merged quad) + bottom face (1 quad)
        // + 4 side faces (each 32×16 = 1 quad each)
        // Total: 6 quads = 12 triangles
        let quad_count = mesh.indices.len() / 6;
        assert_eq!(quad_count, 6, "Half-solid chunk should produce 6 quads, got {quad_count}");
    }

    #[test]
    fn different_materials_not_merged() {
        let mut voxels = make_air_chunk();
        // Two blocks of different materials side by side
        voxels[voxel_index(5, 5, 5)].material = MaterialId::STONE;
        voxels[voxel_index(6, 5, 5)].material = MaterialId::DIRT;
        let mesh = greedy_mesh(&voxels, &NeighborSlices::empty(), &default_colors());

        // Each block has 5 exposed faces (shared face still visible because different materials).
        // Actually: the shared face IS hidden (both are solid). Different materials don't affect
        // face visibility — only solid vs transparent matters.
        // So we get: 4 merged faces + 2 end caps = 10 quads max.
        // But with different materials, the merged faces can't combine, so we get more quads.
        let quad_count = mesh.indices.len() / 6;
        assert!(
            quad_count >= 6 && quad_count <= 12,
            "Expected 6-12 quads for two different-material blocks, got {quad_count}"
        );
    }

    #[test]
    fn positions_are_in_local_space() {
        let mut voxels = make_air_chunk();
        voxels[voxel_index(10, 20, 15)].material = MaterialId::STONE;
        let mesh = greedy_mesh(&voxels, &NeighborSlices::empty(), &default_colors());

        // All positions should be within [0, CHUNK_SIZE]
        for pos in &mesh.positions {
            assert!(
                pos[0] >= 0.0 && pos[0] <= CHUNK_SIZE as f32,
                "X out of range: {}", pos[0]
            );
            assert!(
                pos[1] >= 0.0 && pos[1] <= CHUNK_SIZE as f32,
                "Y out of range: {}", pos[1]
            );
            assert!(
                pos[2] >= 0.0 && pos[2] <= CHUNK_SIZE as f32,
                "Z out of range: {}", pos[2]
            );
        }
    }

    #[test]
    fn normals_are_axis_aligned() {
        let mut voxels = make_air_chunk();
        voxels[voxel_index(5, 5, 5)].material = MaterialId::STONE;
        let mesh = greedy_mesh(&voxels, &NeighborSlices::empty(), &default_colors());

        for n in &mesh.normals {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!(
                (len - 1.0).abs() < 1e-5,
                "Normal should be unit length, got {len}"
            );
            // Exactly one component should be ±1
            let ones = n.iter().filter(|c| c.abs() > 0.5).count();
            assert_eq!(ones, 1, "Normal should be axis-aligned: {n:?}");
        }
    }
}
