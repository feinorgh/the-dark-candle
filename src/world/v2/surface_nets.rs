// Naive Surface Nets meshing for the V2 pipeline.
//
// Uses the per-voxel `density` field (computed by `terrain_density` in
// terrain_gen.rs) as the SDF, with the iso-surface at `density == 0.5`.
// By construction in terrain_gen, density already has the correct sign
// for every material:
//
//   * AIR  (r >  surface_r)        → density ∈ [0, 0.5)   (negative SDF)
//   * SOLID(r ≤  surface_r)        → density ∈ [0.5, 1.0] (positive SDF)
//   * WATER(surface_r < r ≤ sea_r) → density ∈ [0.5, 1.0] (positive SDF)
//
// So a single pass with `sdf = density - 0.5` produces:
//   * land surface at AIR↔SOLID boundary  (sub-voxel via gradient)
//   * water surface at AIR↔WATER boundary
//   * no faces at internal WATER↔SOLID seabed (both positive, no sign change)
//
// Cell colour comes from the corner with the highest density that's not
// transparent gas. Solids beat water at ties, so beaches read as stone.
//
// Output positions are in chunk-local voxel units `[0, CHUNK_SIZE]`,
// matching the greedy mesher's convention. The chunk's `Transform`
// applies the per-LOD tangent scale.
//
// Known limitation: with single-layer neighbour slices, edges that lie
// exactly on a chunk's +X / +Y / +Z face (parallel to the boundary)
// cannot be evaluated because the surrounding cells need corners one
// layer beyond the slice. This produces a roughly 1-voxel crack along
// chunk seams, far less visually disruptive than the prior chunk-aligned
// terraces. Eliminating it requires extending `NeighborSlices` to two
// layers on the positive faces (planned as a follow-up).

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
/// On the +X / +Y / +Z faces (exactly one index == CS) we read the
/// corresponding 1-layer neighbour slice.
///
/// At chunk-edge corners (exactly two indices == CS) we don't have the
/// diagonal-neighbour data, so we approximate by sampling the two
/// adjacent face slices and the chunk's own clamped voxel and picking
/// the highest-density value. Using max-density (rather than AIR
/// fallback) prevents fabricated sign changes along uniform-material
/// chunk edges, which would otherwise produce phantom geometry strips.
///
/// At the +X+Y+Z corner (all three == CS) we sample all three +face
/// slices' nearest cells plus the own corner voxel, again taking max
/// density.
fn corner_voxel(voxels: &[Voxel], neighbors: &NeighborSlices, i: usize, j: usize, k: usize) -> Voxel {
    let hi = (i == CS) as u8 + (j == CS) as u8 + (k == CS) as u8;
    if hi == 0 {
        return voxels[voxel_index(i, j, k)];
    }
    if hi == 1 {
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
        // Slice missing: best-effort fallback to the clamped own voxel.
        let ci = i.min(CS - 1);
        let cj = j.min(CS - 1);
        let ck = k.min(CS - 1);
        return voxels[voxel_index(ci, cj, ck)];
    }

    // hi >= 2: combine all available face slices that "touch" this corner
    // plus the chunk's own clamped voxel. Pick the one with the highest
    // density so that uniform-material chunk edges don't fabricate sign
    // changes.
    let mut best = voxels[voxel_index(i.min(CS - 1), j.min(CS - 1), k.min(CS - 1))];
    let mut best_density = best.density;

    let consider = |v: Voxel, best: &mut Voxel, best_density: &mut f32| {
        if v.density > *best_density {
            *best_density = v.density;
            *best = v;
        }
    };

    if i == CS {
        if let Some(ref s) = neighbors.slices[0] {
            let jj = j.min(CS - 1);
            let kk = k.min(CS - 1);
            consider(s[slice_index(jj, kk)], &mut best, &mut best_density);
        }
    }
    if j == CS {
        if let Some(ref s) = neighbors.slices[2] {
            let ii = i.min(CS - 1);
            let kk = k.min(CS - 1);
            consider(s[slice_index(ii, kk)], &mut best, &mut best_density);
        }
    }
    if k == CS {
        if let Some(ref s) = neighbors.slices[4] {
            let ii = i.min(CS - 1);
            let jj = j.min(CS - 1);
            consider(s[slice_index(ii, jj)], &mut best, &mut best_density);
        }
    }
    best
}

/// Unified SDF: `density - 0.5`. See module docs for why this works for
/// all materials produced by terrain_gen without needing material-aware
/// branching.
#[inline]
fn sdf_value(v: Voxel) -> f32 {
    v.density - 0.5
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

/// Pick the colour material for a cell vertex.
///
/// Among the corners with positive SDF (i.e. on the "filled" side of the
/// iso-surface), prefer non-water solids over WATER, breaking ties by
/// largest density. Falls back to any non-air corner if no positive
/// corner is solid.
fn cell_material(corner_voxels: &[Voxel; 8], corner_sdf: &[f32; 8]) -> MaterialId {
    let mut best_solid: Option<(usize, f32)> = None;
    let mut best_water: Option<(usize, f32)> = None;
    for (i, v) in corner_voxels.iter().enumerate() {
        if corner_sdf[i] < 0.0 {
            continue;
        }
        if is_transparent(v.material) {
            continue;
        }
        if v.material == MaterialId::WATER {
            if best_water.map_or(true, |(_, s)| corner_sdf[i] > s) {
                best_water = Some((i, corner_sdf[i]));
            }
        } else {
            if best_solid.map_or(true, |(_, s)| corner_sdf[i] > s) {
                best_solid = Some((i, corner_sdf[i]));
            }
        }
    }
    if let Some((i, _)) = best_solid {
        return corner_voxels[i].material;
    }
    if let Some((i, _)) = best_water {
        return corner_voxels[i].material;
    }
    // No positive non-air corner: cell straddles a sign change but all
    // positive corners were classified as gas (very rare). Fall back.
    for v in corner_voxels.iter() {
        if !is_transparent(v.material) {
            return v.material;
        }
    }
    MaterialId::AIR
}

struct CellGrid {
    /// `Some(idx)` = vertex index in the output buffers; `None` = no vertex
    vertex: Vec<Option<u32>>,
}

impl CellGrid {
    fn new() -> Self {
        Self {
            vertex: vec![None; CS * CS * CS],
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

    // Step 1: build SDF + voxel grids over the (CS+1)^3 corner grid.
    let total_corners = CSP1 * CSP1 * CSP1;
    let mut corner_sdf = vec![0.0f32; total_corners];
    let mut corner_vox = vec![Voxel::default(); total_corners];
    for k in 0..CSP1 {
        for j in 0..CSP1 {
            for i in 0..CSP1 {
                let v = corner_voxel(voxels, neighbors, i, j, k);
                let idx = corner_index(i, j, k);
                corner_sdf[idx] = sdf_value(v);
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
                colors.push(color_map.get(mat));
                grid.vertex[cidx] = Some(v_idx);
            }
        }
    }

    // Step 3: emit one quad per axis-aligned edge that has a sign change.
    //
    // For an X-axis edge between corner (i, j, k) and (i+1, j, k) the
    // four surrounding cells share x-index i and span (j-1..j) × (k-1..k).
    // Quad winding is chosen so the front face points toward the
    // negative-SDF side.
    let emit_quad = |positions: &mut Vec<[f32; 3]>,
                     normals: &mut Vec<[f32; 3]>,
                     indices: &mut Vec<u32>,
                     verts: [u32; 4],
                     reversed: bool| {
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
        if reversed {
            indices.extend_from_slice(&[verts[0], verts[2], verts[1], verts[0], verts[3], verts[2]]);
        } else {
            indices.extend_from_slice(&[verts[0], verts[1], verts[2], verts[0], verts[2], verts[3]]);
        }
    };

    // X-axis edges: edge at corner (i, j, k) → (i+1, j, k). Surrounding
    // cells need j ≥ 1 and k ≥ 1 (and j ≤ CS-1, k ≤ CS-1).
    for k in 1..CS {
        for j in 1..CS {
            for i in 0..CS {
                let ca = corner_index(i, j, k);
                let cb = corner_index(i + 1, j, k);
                let pos_a = corner_sdf[ca] >= 0.0;
                let pos_b = corner_sdf[cb] >= 0.0;
                if pos_a == pos_b {
                    continue;
                }
                let v00 = grid.vertex[cell_index(i, j - 1, k - 1)];
                let v10 = grid.vertex[cell_index(i, j, k - 1)];
                let v11 = grid.vertex[cell_index(i, j, k)];
                let v01 = grid.vertex[cell_index(i, j - 1, k)];
                let (Some(v00), Some(v10), Some(v11), Some(v01)) = (v00, v10, v11, v01) else {
                    continue;
                };
                // Default ordering [v00, v10, v11, v01] gives +X normal;
                // flip when solid sits on the +X side (pos_b).
                emit_quad(&mut positions, &mut normals, &mut indices, [v00, v10, v11, v01], !pos_a);
            }
        }
    }

    // Y-axis edges: edge at corner (i, j, k) → (i, j+1, k). Surrounding
    // cells need i ≥ 1 and k ≥ 1.
    for k in 1..CS {
        for j in 0..CS {
            for i in 1..CS {
                let ca = corner_index(i, j, k);
                let cb = corner_index(i, j + 1, k);
                let pos_a = corner_sdf[ca] >= 0.0;
                let pos_b = corner_sdf[cb] >= 0.0;
                if pos_a == pos_b {
                    continue;
                }
                let v00 = grid.vertex[cell_index(i - 1, j, k - 1)];
                let v10 = grid.vertex[cell_index(i, j, k - 1)];
                let v11 = grid.vertex[cell_index(i, j, k)];
                let v01 = grid.vertex[cell_index(i - 1, j, k)];
                let (Some(v00), Some(v10), Some(v11), Some(v01)) = (v00, v10, v11, v01) else {
                    continue;
                };
                // Default ordering produces -Y normal (CW in (i,k) plane
                // viewed from +Y); flip when solid is below (pos_a).
                emit_quad(&mut positions, &mut normals, &mut indices, [v00, v10, v11, v01], pos_a);
            }
        }
    }

    // Z-axis edges: edge at corner (i, j, k) → (i, j, k+1). Surrounding
    // cells need i ≥ 1 and j ≥ 1.
    for k in 0..CS {
        for j in 1..CS {
            for i in 1..CS {
                let ca = corner_index(i, j, k);
                let cb = corner_index(i, j, k + 1);
                let pos_a = corner_sdf[ca] >= 0.0;
                let pos_b = corner_sdf[cb] >= 0.0;
                if pos_a == pos_b {
                    continue;
                }
                let v00 = grid.vertex[cell_index(i - 1, j - 1, k)];
                let v10 = grid.vertex[cell_index(i, j - 1, k)];
                let v11 = grid.vertex[cell_index(i, j, k)];
                let v01 = grid.vertex[cell_index(i - 1, j, k)];
                let (Some(v00), Some(v10), Some(v11), Some(v01)) = (v00, v10, v11, v01) else {
                    continue;
                };
                emit_quad(&mut positions, &mut normals, &mut indices, [v00, v10, v11, v01], !pos_a);
            }
        }
    }

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
        // those faces. Smoke-test that buffers stay consistent.
        let voxels = make_voxels(|_, _, _| (MaterialId::STONE, 1.0));
        let neighbors = NeighborSlices::empty();
        let cmap = MaterialColorMap::default();
        let mesh = surface_nets_mesh(&voxels, &neighbors, &cmap);
        assert_eq!(mesh.positions.len(), mesh.normals.len());
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
            (avg_y - mid).abs() < 0.6,
            "mean surface y ({avg_y}) should be near midplane ({mid})"
        );
    }

    #[test]
    fn smooth_gradient_places_vertices_at_subvoxel_height() {
        // Stone bottom, air top, with a continuous density gradient
        // crossing 0.5 at exactly y = 12.7 (a non-integer height that
        // a binary mesher would round to 13).
        let target = 12.7f32;
        let voxels = make_voxels(|_, j, _| {
            let depth = target - (j as f32 + 0.5);
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
        let mut y_sum = 0.0;
        let mut count = 0;
        for p in &mesh.positions {
            if p[0] > 2.0 && p[0] < (CS as f32 - 2.0)
                && p[2] > 2.0 && p[2] < (CS as f32 - 2.0)
            {
                y_sum += p[1];
                count += 1;
            }
        }
        assert!(count > 0);
        let avg_y = y_sum / count as f32;
        assert!(
            (avg_y - target).abs() < 0.2,
            "Surface Nets should place verts within 0.2 voxel of the iso-surface; got avg_y={avg_y} target={target}"
        );
    }
}
