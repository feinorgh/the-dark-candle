// Naive Surface Nets meshing for the V2 pipeline.
//
// Replaces the binary greedy mesher when the per-voxel `density` field is
// meaningful (terrain chunks). Surface Nets places one vertex per cell that
// straddles the iso-surface, at the centroid of the iso-surface crossings on
// the cell's 12 edges, then emits a quad per edge that has a sign change.
//
// Two passes are run with different SDF interpretations so we can mesh both
// the solid surface (land + seabed) and the water-air surface separately
// while avoiding double-emission at solid-water boundaries:
//
// * Pass 1 — `SolidSdf`: SDF positive when material is solid (stone, etc.),
//   negative otherwise. Iso-surface = boundary of all solids.
// * Pass 2 — `WaterAirSdf`: SDF positive for ANY non-air material, negative
//   for air. Quads are only emitted when the "inside" endpoint is WATER, so
//   the result is the water-air surface only.
//
// Output positions are in chunk-local voxel units `[0, CHUNK_SIZE]`, matching
// the greedy mesher's convention. The chunk's `Transform` applies the
// per-LOD tangent scale.
//
// Known limitation: with single-layer neighbour slices, edges that lie
// exactly on a chunk's +X / +Y / +Z face (parallel to the boundary) cannot
// be evaluated because the surrounding cells need corners one layer beyond
// the slice. This produces a roughly 1-voxel crack along chunk seams. The
// crack is far less visually disruptive than the prior chunk-aligned
// terraces; eliminating it requires extending `NeighborSlices` to two layers
// on the positive faces (planned as a follow-up).

use crate::world::chunk::CHUNK_SIZE;
use crate::world::lod::MaterialColorMap;
use crate::world::meshing::ChunkMesh;
use crate::world::v2::greedy_mesh::{NeighborSlices, is_transparent};
use crate::world::voxel::{MaterialId, Voxel};

const CS: usize = CHUNK_SIZE;
const CSP1: usize = CS + 1; // corner grid extent per axis (one extra layer from neighbour)

#[inline]
fn corner_index(i: usize, j: usize, k: usize) -> usize {
    k * CSP1 * CSP1 + j * CSP1 + i
}

#[inline]
fn cell_index(i: usize, j: usize, k: usize) -> usize {
    k * CS * CS + j * CS + i
}

#[inline]
fn voxel_index(i: usize, j: usize, k: usize) -> usize {
    k * CS * CS + j * CS + i
}

#[inline]
fn slice_index(a: usize, b: usize) -> usize {
    a * CS + b
}

/// Sample the voxel at corner (i, j, k) of the (CS+1)^3 corner grid.
///
/// For interior corners (all of i, j, k < CS) this is a direct voxel lookup.
/// On the +X / +Y / +Z faces (one of the indices == CS) we read the
/// corresponding 1-layer neighbour slice. At corner-grid edges and the
/// far corner where two or three indices equal CS, the data isn't
/// available — we return AIR there, accepting a small crack at chunk
/// edges.
#[inline]
fn corner_voxel(voxels: &[Voxel], neighbors: &NeighborSlices, i: usize, j: usize, k: usize) -> Voxel {
    let hi = (i == CS) as u8 + (j == CS) as u8 + (k == CS) as u8;
    if hi == 0 {
        return voxels[voxel_index(i, j, k)];
    }
    if hi == 1 {
        // exactly one of i/j/k is CS — use the matching +face slice.
        if i == CS {
            if let Some(ref s) = neighbors.slices[0] {
                return s[slice_index(j, k)];
            }
        } else if j == CS {
            if let Some(ref s) = neighbors.slices[2] {
                return s[slice_index(i, k)];
            }
        } else if k == CS {
            if let Some(ref s) = neighbors.slices[4] {
                return s[slice_index(i, j)];
            }
        }
    }
    // hi >= 2 (corner grid edge / far corner) or missing neighbour slice.
    Voxel::default()
}

/// SDF interpretation for the solid pass.
#[inline]
fn sdf_solid(v: Voxel) -> f32 {
    if is_transparent(v.material) || v.material == MaterialId::WATER {
        -0.5
    } else {
        v.density - 0.5
    }
}

/// SDF interpretation for the water-air pass. Positive iff the voxel is
/// non-air (water or solid). Used together with a "must touch water"
/// filter so we only emit the water-air surface.
#[inline]
fn sdf_water_air(v: Voxel) -> f32 {
    if is_transparent(v.material) {
        -0.5
    } else {
        v.density - 0.5
    }
}

/// 12 edges of a unit cube as (corner_a, corner_b) index pairs into the 8
/// cube corners. Corner ordering: bit 0 = X, bit 1 = Y, bit 2 = Z.
const CUBE_EDGES: [(u8, u8); 12] = [
    (0b000, 0b001), (0b010, 0b011), (0b100, 0b101), (0b110, 0b111), // X-edges
    (0b000, 0b010), (0b001, 0b011), (0b100, 0b110), (0b101, 0b111), // Y-edges
    (0b000, 0b100), (0b001, 0b101), (0b010, 0b110), (0b011, 0b111), // Z-edges
];

/// Compute the surface-net vertex inside a cell from the 8 corner SDFs.
///
/// Returns the local offset within the cell (each component in [0, 1]).
/// `mask` is a bitfield of which corners had positive SDF (bit i for corner i).
fn cell_vertex_offset(corner_sdf: &[f32; 8], mask: u8) -> [f32; 3] {
    let mut sum = [0.0f32; 3];
    let mut count = 0u32;
    for &(a, b) in &CUBE_EDGES {
        let bit_a = (mask >> a) & 1;
        let bit_b = (mask >> b) & 1;
        if bit_a == bit_b {
            continue;
        }
        let sa = corner_sdf[a as usize];
        let sb = corner_sdf[b as usize];
        let denom = sa - sb;
        let t = if denom.abs() < 1e-6 { 0.5 } else { sa / denom };
        let t = t.clamp(0.0, 1.0);
        // corner positions within the unit cube
        let pa = [(a & 1) as f32, ((a >> 1) & 1) as f32, ((a >> 2) & 1) as f32];
        let pb = [(b & 1) as f32, ((b >> 1) & 1) as f32, ((b >> 2) & 1) as f32];
        sum[0] += pa[0] + t * (pb[0] - pa[0]);
        sum[1] += pa[1] + t * (pb[1] - pa[1]);
        sum[2] += pa[2] + t * (pb[2] - pa[2]);
        count += 1;
    }
    if count == 0 {
        return [0.5, 0.5, 0.5];
    }
    let inv = 1.0 / count as f32;
    [sum[0] * inv, sum[1] * inv, sum[2] * inv]
}

/// Pick the material that should colour faces emitted from this cell.
///
/// We prefer the corner whose SDF is the largest (deepest into solid),
/// excluding AIR — this gives stable solid colouring near the surface.
fn cell_material(corner_voxels: &[Voxel; 8], corner_sdf: &[f32; 8]) -> MaterialId {
    let mut best_idx = 0usize;
    let mut best_score = f32::MIN;
    for (i, v) in corner_voxels.iter().enumerate() {
        if is_transparent(v.material) {
            continue;
        }
        let score = corner_sdf[i];
        if score > best_score {
            best_score = score;
            best_idx = i;
        }
    }
    let m = corner_voxels[best_idx].material;
    if is_transparent(m) {
        // fall back: any non-air corner
        for v in corner_voxels.iter() {
            if !is_transparent(v.material) {
                return v.material;
            }
        }
    }
    m
}

struct CellGrid {
    /// `Some(idx)` = vertex index in the output buffers; `None` = no vertex
    vertex: Vec<Option<u32>>,
    materials: Vec<MaterialId>,
}

impl CellGrid {
    fn new() -> Self {
        Self {
            vertex: vec![None; CS * CS * CS],
            materials: vec![MaterialId::AIR; CS * CS * CS],
        }
    }
}

/// Run a single Surface Nets pass with the given SDF interpretation.
#[allow(clippy::too_many_arguments)]
fn run_pass<F>(
    voxels: &[Voxel],
    neighbors: &NeighborSlices,
    sdf_fn: F,
    color_map: &MaterialColorMap,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    colors: &mut Vec<[f32; 4]>,
    indices: &mut Vec<u32>,
    require_water: bool,
) where
    F: Fn(Voxel) -> f32 + Copy,
{
    // Step 1: build SDF + voxel grids over the (CS+1)^3 corner grid.
    let total_corners = CSP1 * CSP1 * CSP1;
    let mut corner_sdf = vec![0.0f32; total_corners];
    let mut corner_vox = vec![Voxel::default(); total_corners];
    for k in 0..CSP1 {
        for j in 0..CSP1 {
            for i in 0..CSP1 {
                let v = corner_voxel(voxels, neighbors, i, j, k);
                let idx = corner_index(i, j, k);
                corner_sdf[idx] = sdf_fn(v);
                corner_vox[idx] = v;
            }
        }
    }

    // Step 2: per-cell vertex placement.
    let mut grid = CellGrid::new();
    for k in 0..CS {
        for j in 0..CS {
            for i in 0..CS {
                let mut sdf8 = [0.0f32; 8];
                let mut vox8 = [Voxel::default(); 8];
                let mut mask: u8 = 0;
                for c in 0..8u8 {
                    let dx = (c & 1) as usize;
                    let dy = ((c >> 1) & 1) as usize;
                    let dz = ((c >> 2) & 1) as usize;
                    let ci = corner_index(i + dx, j + dy, k + dz);
                    sdf8[c as usize] = corner_sdf[ci];
                    vox8[c as usize] = corner_vox[ci];
                    if corner_sdf[ci] >= 0.0 {
                        mask |= 1 << c;
                    }
                }
                if mask == 0 || mask == 0xFF {
                    continue;
                }
                let off = cell_vertex_offset(&sdf8, mask);
                let pos = [
                    i as f32 + 0.5 + off[0],
                    j as f32 + 0.5 + off[1],
                    k as f32 + 0.5 + off[2],
                ];
                let mat = cell_material(&vox8, &sdf8);
                let cidx = cell_index(i, j, k);
                let v_idx = positions.len() as u32;
                positions.push(pos);
                normals.push([0.0, 0.0, 0.0]); // accumulated below
                let rgba = color_map.get(mat);
                colors.push(rgba);
                grid.vertex[cidx] = Some(v_idx);
                grid.materials[cidx] = mat;
            }
        }
    }

    // Step 3: emit one quad per axis-aligned edge that has a sign change.
    //
    // For an X-axis edge between corner (i, j, k) and (i+1, j, k): the four
    // surrounding cells share the same x-index i and have y,z in {j-1, j} ×
    // {k-1, k}. Quad winding is chosen so the front face points toward the
    // negative-SDF side.
    let emit_quad = |positions: &mut Vec<[f32; 3]>,
                     normals: &mut Vec<[f32; 3]>,
                     colors: &mut Vec<[f32; 4]>,
                     indices: &mut Vec<u32>,
                     grid: &CellGrid,
                     verts: [u32; 4],
                     reversed: bool| {
        // Build two triangles. For default winding (CCW from +normal side):
        //   tri1: 0,1,2 ; tri2: 0,2,3
        // If reversed, swap to: 0,2,1 ; 0,3,2.
        let p0 = positions[verts[0] as usize];
        let p1 = positions[verts[1] as usize];
        let p2 = positions[verts[2] as usize];
        let edge1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let edge2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        let mut nrm = [
            edge1[1] * edge2[2] - edge1[2] * edge2[1],
            edge1[2] * edge2[0] - edge1[0] * edge2[2],
            edge1[0] * edge2[1] - edge1[1] * edge2[0],
        ];
        let len = (nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2]).sqrt();
        if len > 0.0 {
            nrm[0] /= len;
            nrm[1] /= len;
            nrm[2] /= len;
        }
        if reversed {
            nrm = [-nrm[0], -nrm[1], -nrm[2]];
        }
        for &v in &verts {
            let n = &mut normals[v as usize];
            n[0] += nrm[0];
            n[1] += nrm[1];
            n[2] += nrm[2];
        }
        let _ = (colors, grid); // colours already set per-vertex
        if reversed {
            indices.extend_from_slice(&[verts[0], verts[2], verts[1], verts[0], verts[3], verts[2]]);
        } else {
            indices.extend_from_slice(&[verts[0], verts[1], verts[2], verts[0], verts[2], verts[3]]);
        }
    };

    // X-axis edges. Edge at corner (i, j, k) → (i+1, j, k). Surrounding cells
    // need j ≥ 1 and k ≥ 1 (and j ≤ CS-1, k ≤ CS-1) to all exist.
    for k in 1..CS {
        for j in 1..CS {
            for i in 0..CS {
                let ca = corner_index(i, j, k);
                let cb = corner_index(i + 1, j, k);
                let sa = corner_sdf[ca];
                let sb = corner_sdf[cb];
                let pos_a = sa >= 0.0;
                let pos_b = sb >= 0.0;
                if pos_a == pos_b {
                    continue;
                }
                // require_water filter: only emit if the positive endpoint is WATER
                if require_water {
                    let positive_vox = if pos_a { corner_vox[ca] } else { corner_vox[cb] };
                    if positive_vox.material != MaterialId::WATER {
                        continue;
                    }
                }
                let v00 = grid.vertex[cell_index(i, j - 1, k - 1)];
                let v10 = grid.vertex[cell_index(i, j, k - 1)];
                let v11 = grid.vertex[cell_index(i, j, k)];
                let v01 = grid.vertex[cell_index(i, j - 1, k)];
                let (Some(v00), Some(v10), Some(v11), Some(v01)) = (v00, v10, v11, v01) else {
                    continue;
                };
                // pos_a=true → solid is at i side, normal points +X
                // Default ordering [v00, v10, v11, v01] gives normal pointing +X if pos_a, else flip.
                emit_quad(positions, normals, colors, indices, &grid, [v00, v10, v11, v01], !pos_a);
            }
        }
    }

    // Y-axis edges. Edge at corner (i, j, k) → (i, j+1, k). Surrounding cells need i, k ≥ 1.
    for k in 1..CS {
        for j in 0..CS {
            for i in 1..CS {
                let ca = corner_index(i, j, k);
                let cb = corner_index(i, j + 1, k);
                let sa = corner_sdf[ca];
                let sb = corner_sdf[cb];
                let pos_a = sa >= 0.0;
                let pos_b = sb >= 0.0;
                if pos_a == pos_b {
                    continue;
                }
                if require_water {
                    let positive_vox = if pos_a { corner_vox[ca] } else { corner_vox[cb] };
                    if positive_vox.material != MaterialId::WATER {
                        continue;
                    }
                }
                let v00 = grid.vertex[cell_index(i - 1, j, k - 1)];
                let v10 = grid.vertex[cell_index(i, j, k - 1)];
                let v11 = grid.vertex[cell_index(i, j, k)];
                let v01 = grid.vertex[cell_index(i - 1, j, k)];
                let (Some(v00), Some(v10), Some(v11), Some(v01)) = (v00, v10, v11, v01) else {
                    continue;
                };
                // For +Y normal we need the four cells in order producing CCW from +Y.
                // Default ordering swept cw vs ccw: use reverse on pos_a (so +Y when !pos_a).
                emit_quad(positions, normals, colors, indices, &grid, [v00, v10, v11, v01], pos_a);
            }
        }
    }

    // Z-axis edges. Edge at (i, j, k) → (i, j, k+1). Surrounding cells need i, j ≥ 1.
    for k in 0..CS {
        for j in 1..CS {
            for i in 1..CS {
                let ca = corner_index(i, j, k);
                let cb = corner_index(i, j, k + 1);
                let sa = corner_sdf[ca];
                let sb = corner_sdf[cb];
                let pos_a = sa >= 0.0;
                let pos_b = sb >= 0.0;
                if pos_a == pos_b {
                    continue;
                }
                if require_water {
                    let positive_vox = if pos_a { corner_vox[ca] } else { corner_vox[cb] };
                    if positive_vox.material != MaterialId::WATER {
                        continue;
                    }
                }
                let v00 = grid.vertex[cell_index(i - 1, j - 1, k)];
                let v10 = grid.vertex[cell_index(i, j - 1, k)];
                let v11 = grid.vertex[cell_index(i, j, k)];
                let v01 = grid.vertex[cell_index(i - 1, j, k)];
                let (Some(v00), Some(v10), Some(v11), Some(v01)) = (v00, v10, v11, v01) else {
                    continue;
                };
                emit_quad(positions, normals, colors, indices, &grid, [v00, v10, v11, v01], !pos_a);
            }
        }
    }
}

/// Generate a Surface-Nets mesh from voxel data + neighbour slices.
pub fn surface_nets_mesh(
    voxels: &[Voxel],
    neighbors: &NeighborSlices,
    color_map: &MaterialColorMap,
) -> ChunkMesh {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut colors: Vec<[f32; 4]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    // Pass 1: solid surfaces (land + seabed).
    run_pass(
        voxels,
        neighbors,
        sdf_solid,
        color_map,
        &mut positions,
        &mut normals,
        &mut colors,
        &mut indices,
        false,
    );

    // Pass 2: water-air surface only.
    run_pass(
        voxels,
        neighbors,
        sdf_water_air,
        color_map,
        &mut positions,
        &mut normals,
        &mut colors,
        &mut indices,
        true,
    );

    // Normalize accumulated vertex normals.
    for n in normals.iter_mut() {
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if len > 1e-6 {
            n[0] /= len;
            n[1] /= len;
            n[2] /= len;
        } else {
            *n = [0.0, 1.0, 0.0];
        }
    }

    ChunkMesh { positions, normals, colors, indices }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::chunk::CHUNK_VOLUME;

    fn make_voxels<F: Fn(usize, usize, usize) -> (MaterialId, f32)>(f: F) -> Vec<Voxel> {
        let mut out = vec![Voxel::default(); CHUNK_VOLUME];
        for k in 0..CS {
            for j in 0..CS {
                for i in 0..CS {
                    let (m, d) = f(i, j, k);
                    out[voxel_index(i, j, k)].material = m;
                    out[voxel_index(i, j, k)].density = d;
                }
            }
        }
        out
    }

    #[test]
    fn empty_chunk_emits_no_geometry() {
        let voxels = vec![Voxel::default(); CHUNK_VOLUME];
        let neighbors = NeighborSlices::empty();
        let cmap = MaterialColorMap::default();
        let mesh = surface_nets_mesh(&voxels, &neighbors, &cmap);
        assert_eq!(mesh.positions.len(), 0);
        assert_eq!(mesh.indices.len(), 0);
    }

    #[test]
    fn full_solid_with_no_neighbors_emits_no_interior_geometry() {
        // All-stone chunk, no neighbor slices. Without neighbour data, the
        // +X/+Y/+Z corner row is AIR (sign change) so we DO emit quads on
        // those faces. Mostly a smoke-test that it doesn't panic and that
        // counts are sensible.
        let voxels = make_voxels(|_, _, _| (MaterialId::STONE, 1.0));
        let neighbors = NeighborSlices::empty();
        let cmap = MaterialColorMap::default();
        let mesh = surface_nets_mesh(&voxels, &neighbors, &cmap);
        assert_eq!(mesh.positions.len() * 1, mesh.normals.len());
        assert_eq!(mesh.positions.len(), mesh.colors.len());
        assert_eq!(mesh.indices.len() % 3, 0);
    }

    #[test]
    fn flat_surface_at_y_midplane_produces_horizontal_mesh() {
        // Lower half stone, upper half air, with a smooth density gradient
        // through the midplane. Surface Nets should emit a roughly flat
        // sheet near y = CS/2.
        let mid = CS as f32 / 2.0;
        let voxels = make_voxels(|_, j, _| {
            let depth = mid - (j as f32 + 0.5);
            let density = (0.5 + depth * 0.5).clamp(0.0, 1.0);
            if density >= 0.5 {
                (MaterialId::STONE, density)
            } else {
                (MaterialId::AIR, density)
            }
        });
        let neighbors = NeighborSlices::empty();
        let cmap = MaterialColorMap::default();
        let mesh = surface_nets_mesh(&voxels, &neighbors, &cmap);
        assert!(!mesh.positions.is_empty(), "should emit a surface");
        // All surface vertices should lie near y = mid.
        let mut y_sum = 0.0;
        let mut count = 0;
        for p in &mesh.positions {
            // Keep only cells safely inside the chunk (avoid +X/+Y/+Z fall-off
            // into the missing-neighbour AIR layer which biases averages).
            if p[0] > 2.0 && p[0] < (CS as f32 - 2.0)
                && p[2] > 2.0 && p[2] < (CS as f32 - 2.0)
                && p[1] > 0.5 && p[1] < (CS as f32 - 0.5)
            {
                y_sum += p[1];
                count += 1;
            }
        }
        assert!(count > 0, "no interior surface vertices found");
        let avg_y = y_sum / count as f32;
        assert!(
            (avg_y - mid).abs() < 1.0,
            "mean surface y ({avg_y}) should be near midplane ({mid})"
        );
    }
}
