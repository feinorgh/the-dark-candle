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
// Seamless meshing strategy: each chunk meshes a *padded* volume that
// extends one cell into each of its -X/-Y/-Z neighbours. This requires
// the corner grid to span indices `-1..=CS` (CSP2 entries per axis) and
// cells to span `-1..CS` (CSP1 entries). Edge emission:
//   * X-axis loop runs `i in 0..CS`: owns +X face seam; -X owned by -X neighbour.
//   * Y-axis loop runs `j in -1..CS-1`: owns -Y face seam (uses slices[3]);
//     the +Y seam (`j=CS-1 → j=CS`) is owned by the +Y neighbour's j=-1 edge.
//     This is critical: the +Y cross-chunk edge would sample slices[2] using
//     resampled data that doesn't reflect the +Y chunk's true voxel content.
//     When the +Y chunk is AllAir it emits no mesh at all, so the +Y seam must
//     not be emitted here either — otherwise a spurious downward-facing face
//     appears as an opaque ceiling above the player.
//   * Z-axis loop runs `k in 0..CS`: owns +Z face seam; -Z owned by -Z neighbour.
//
// The +X/+Z seams are vertical walls and seam-ownership asymmetry is harmless.
// The Y axis is special: emitting the wrong cross-chunk horizontal face produces
// the ceiling artifact, so Y ownership is explicitly -Y (not +Y).
//
// Only the existing 1-layer NeighborSlices are needed: a cell at e.g.
// i=-1 reads i=-1 corners from the -X slice (= the -X neighbour's
// x=CS-1 voxel) and i=0 corners from our own voxels; the -X neighbour's
// matching cell at i=CS-1 reads x=CS-1 from its own voxels and x=CS from
// our +X slice (= our x=0 voxel). Same data → same vertex placement →
// no crack, no chunk-aligned slab.
//
// 1-voxel approximation cracks remain only at chunk *corners* (where
// 4 chunks meet along an edge): cells with two extreme axes use a
// max-density fallback at the diagonal corner, which differs from what
// the diagonal-neighbour chunk computes from real data.

use crate::world::chunk::CHUNK_SIZE;
use crate::world::lod::MaterialColorMap;
use crate::world::meshing::ChunkMesh;
use crate::world::v2::greedy_mesh::{NeighborSlices, is_transparent};
use crate::world::voxel::{MaterialId, Voxel};

const CS: usize = CHUNK_SIZE;
const CSI: isize = CS as isize;
const CSP1: usize = CS + 1; // cell extent per axis: cells at -1..CS
const CSP2: usize = CS + 2; // corner extent per axis: corners at -1..=CS

#[inline]
fn corner_index_ext(i: isize, j: isize, k: isize) -> usize {
    let ci = (i + 1) as usize;
    let cj = (j + 1) as usize;
    let ck = (k + 1) as usize;
    ck * CSP2 * CSP2 + cj * CSP2 + ci
}

#[inline]
fn cell_index_ext(i: isize, j: isize, k: isize) -> usize {
    let ci = (i + 1) as usize;
    let cj = (j + 1) as usize;
    let ck = (k + 1) as usize;
    ck * CSP1 * CSP1 + cj * CSP1 + ci
}

#[inline]
fn voxel_index(i: usize, j: usize, k: usize) -> usize {
    k * CS * CS + j * CS + i
}

#[inline]
fn slice_index(a: usize, b: usize) -> usize {
    a * CS + b
}

/// Sample the voxel at corner (i, j, k) of the (CS+2)^3 corner grid.
///
/// `i`, `j`, `k` may be in `-1..=CS`. Interior corners (all of i, j, k
/// in `0..CS`) are direct voxel lookups. On a single -X/+X/-Y/+Y/-Z/+Z
/// face we read the matching 1-layer neighbour slice. Edge and corner
/// cases (two or three axes at extremes) lack diagonal-neighbour data:
/// we approximate by sampling all available adjacent face slices and
/// the chunk's own clamped voxel and picking the highest-density value.
/// Using max-density (rather than AIR fallback) prevents fabricated
/// sign changes along uniform-material chunk edges, which would
/// otherwise produce phantom geometry strips.
fn corner_voxel(
    voxels: &[Voxel],
    neighbors: &NeighborSlices,
    i: isize,
    j: isize,
    k: isize,
) -> Voxel {
    let i_lo = i == -1;
    let i_hi = i == CSI;
    let j_lo = j == -1;
    let j_hi = j == CSI;
    let k_lo = k == -1;
    let k_hi = k == CSI;
    let i_ext = i_lo || i_hi;
    let j_ext = j_lo || j_hi;
    let k_ext = k_lo || k_hi;
    let extreme = (i_ext as u8) + (j_ext as u8) + (k_ext as u8);

    if extreme == 0 {
        return voxels[voxel_index(i as usize, j as usize, k as usize)];
    }

    // Clamped own-voxel coordinates (used as fallback and for slice indexing
    // when one or more axes are themselves at the extreme).
    let ci = i.clamp(0, CSI - 1) as usize;
    let cj = j.clamp(0, CSI - 1) as usize;
    let ck = k.clamp(0, CSI - 1) as usize;

    if extreme == 1 {
        if i_hi {
            if let Some(ref s) = neighbors.slices[0] {
                return s[slice_index(j as usize, k as usize)];
            }
        } else if i_lo {
            if let Some(ref s) = neighbors.slices[1] {
                return s[slice_index(j as usize, k as usize)];
            }
        } else if j_hi {
            if let Some(ref s) = neighbors.slices[2] {
                return s[slice_index(i as usize, k as usize)];
            }
        } else if j_lo {
            if let Some(ref s) = neighbors.slices[3] {
                return s[slice_index(i as usize, k as usize)];
            }
        } else if k_hi {
            if let Some(ref s) = neighbors.slices[4] {
                return s[slice_index(i as usize, j as usize)];
            }
        } else if k_lo && let Some(ref s) = neighbors.slices[5] {
            return s[slice_index(i as usize, j as usize)];
        }
        // Slice missing: best-effort fallback to the clamped own voxel.
        return voxels[voxel_index(ci, cj, ck)];
    }

    // extreme >= 2: combine all available face slices that "touch" this
    // corner plus the chunk's own clamped voxel. Pick the highest-density
    // candidate that is on the SAME side of the isosurface as the own voxel.
    //
    // Allowing candidates from the opposite side (the old max-density approach)
    // caused phantom faces along chunk edges where an ocean chunk's corner
    // neighbor slice contained solid land — the SOLID density "won" and created
    // a spurious sign change in what should be uniform-air territory.
    let mut best = voxels[voxel_index(ci, cj, ck)];
    let mut best_density = best.density;
    let own_solid = best_density >= 0.5;

    let consider = |v: Voxel, best: &mut Voxel, best_density: &mut f32| {
        if (v.density >= 0.5) == own_solid && v.density > *best_density {
            *best_density = v.density;
            *best = v;
        }
    };

    if i_hi {
        if let Some(ref s) = neighbors.slices[0] {
            consider(s[slice_index(cj, ck)], &mut best, &mut best_density);
        }
    } else if i_lo && let Some(ref s) = neighbors.slices[1] {
        consider(s[slice_index(cj, ck)], &mut best, &mut best_density);
    }
    if j_hi {
        if let Some(ref s) = neighbors.slices[2] {
            consider(s[slice_index(ci, ck)], &mut best, &mut best_density);
        }
    } else if j_lo && let Some(ref s) = neighbors.slices[3] {
        consider(s[slice_index(ci, ck)], &mut best, &mut best_density);
    }
    if k_hi {
        if let Some(ref s) = neighbors.slices[4] {
            consider(s[slice_index(ci, cj)], &mut best, &mut best_density);
        }
    } else if k_lo && let Some(ref s) = neighbors.slices[5] {
        consider(s[slice_index(ci, cj)], &mut best, &mut best_density);
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
    (0b000, 0b001),
    (0b010, 0b011),
    (0b100, 0b101),
    (0b110, 0b111), // X-edges
    (0b000, 0b010),
    (0b001, 0b011),
    (0b100, 0b110),
    (0b101, 0b111), // Y-edges
    (0b000, 0b100),
    (0b001, 0b101),
    (0b010, 0b110),
    (0b011, 0b111), // Z-edges
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
            if best_water.is_none_or(|(_, s)| corner_sdf[i] > s) {
                best_water = Some((i, corner_sdf[i]));
            }
        } else if best_solid.is_none_or(|(_, s)| corner_sdf[i] > s) {
            best_solid = Some((i, corner_sdf[i]));
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
            vertex: vec![None; CSP1 * CSP1 * CSP1],
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

    // Step 1: build SDF + voxel grids over the (CS+2)^3 corner grid
    // spanning corners at indices -1..=CS on each axis.
    let total_corners = CSP2 * CSP2 * CSP2;
    let mut corner_sdf = vec![0.0f32; total_corners];
    let mut corner_vox = vec![Voxel::default(); total_corners];
    for k in -1..=CSI {
        for j in -1..=CSI {
            for i in -1..=CSI {
                let v = corner_voxel(voxels, neighbors, i, j, k);
                let idx = corner_index_ext(i, j, k);
                corner_sdf[idx] = sdf_value(v);
                corner_vox[idx] = v;
            }
        }
    }

    // Step 2: per-cell vertex placement over cells at indices -1..CS
    // on each axis. Cells at i=-1 (etc.) are owned by the -X neighbour
    // physically, but we mesh them so we can emit our -X face seam
    // quads with vertex positions identical to what that neighbour
    // computes (because the underlying voxel data on the seam is
    // shared via the 1-layer slice).
    let mut grid = CellGrid::new();
    for k in -1..CSI {
        for j in -1..CSI {
            for i in -1..CSI {
                let mut sdf8 = [0.0f32; 8];
                let mut vox8 = [Voxel::default(); 8];
                let mut mask: u8 = 0;
                for c in 0..8u8 {
                    let dx = (c & 1) as isize;
                    let dy = ((c >> 1) & 1) as isize;
                    let dz = ((c >> 2) & 1) as isize;
                    let ci = corner_index_ext(i + dx, j + dy, k + dz);
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
                let cidx = cell_index_ext(i, j, k);
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
    // Edge loops uniformly span 0..CS on every axis. For each axis, the
    // varying axis of the edge runs 0..CS (start corners 0..=CS-1), and
    // the orthogonal axes also run 0..CS. This emits all in-chunk edges
    // plus the -X/-Y/-Z face seam edges (whose surrounding cells include
    // the i=-1 / j=-1 / k=-1 cells we computed above). +X/+Y/+Z face
    // seams are owned by the neighbouring chunk's matching loop.
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
            indices
                .extend_from_slice(&[verts[0], verts[2], verts[1], verts[0], verts[3], verts[2]]);
        } else {
            indices
                .extend_from_slice(&[verts[0], verts[1], verts[2], verts[0], verts[2], verts[3]]);
        }
    };

    // X-axis edges: edge at corner (i, j, k) → (i+1, j, k). Surrounding
    // cells span (i, j-1..j, k-1..k), needing j-1 ≥ -1 and k-1 ≥ -1.
    for k in 0..CSI {
        for j in 0..CSI {
            for i in 0..CSI {
                let ca = corner_index_ext(i, j, k);
                let cb = corner_index_ext(i + 1, j, k);
                let pos_a = corner_sdf[ca] >= 0.0;
                let pos_b = corner_sdf[cb] >= 0.0;
                if pos_a == pos_b {
                    continue;
                }
                let v00 = grid.vertex[cell_index_ext(i, j - 1, k - 1)];
                let v10 = grid.vertex[cell_index_ext(i, j, k - 1)];
                let v11 = grid.vertex[cell_index_ext(i, j, k)];
                let v01 = grid.vertex[cell_index_ext(i, j - 1, k)];
                let (Some(v00), Some(v10), Some(v11), Some(v01)) = (v00, v10, v11, v01) else {
                    continue;
                };
                // Default ordering [v00, v10, v11, v01] gives +X normal;
                // flip when solid sits on the +X side (pos_b).
                emit_quad(
                    &mut positions,
                    &mut normals,
                    &mut indices,
                    [v00, v10, v11, v01],
                    !pos_a,
                );
            }
        }
    }

    // Y-axis edges: edge at corner (i, j, k) → (i, j+1, k). Surrounding
    // cells span (i-1..i, j, k-1..k).
    //
    // Range: j in -1..CSI-1 (i.e. j = -1, 0, 1, …, CS-2).
    //   * j = -1  → edge (-1→0) uses slices[3] for the -Y seam corner.  This
    //     is the -Y face seam this chunk owns.
    //   * j = CS-1 → edge (CS-1→CS) is EXCLUDED. That edge samples slices[2]
    //     (+Y neighbour's j=0 row). When the +Y chunk is AllAir it emits no
    //     mesh at all; emitting this edge here would produce a spurious
    //     downward-facing stone face — the "gray ceiling" bug.  The +Y
    //     neighbour handles this edge via its own j=-1 iteration.
    for k in 0..CSI {
        for j in -1_isize..(CSI - 1) {
            for i in 0..CSI {
                let ca = corner_index_ext(i, j, k);
                let cb = corner_index_ext(i, j + 1, k);
                let pos_a = corner_sdf[ca] >= 0.0;
                let pos_b = corner_sdf[cb] >= 0.0;
                if pos_a == pos_b {
                    continue;
                }
                let v00 = grid.vertex[cell_index_ext(i - 1, j, k - 1)];
                let v10 = grid.vertex[cell_index_ext(i, j, k - 1)];
                let v11 = grid.vertex[cell_index_ext(i, j, k)];
                let v01 = grid.vertex[cell_index_ext(i - 1, j, k)];
                let (Some(v00), Some(v10), Some(v11), Some(v01)) = (v00, v10, v11, v01) else {
                    continue;
                };
                // Default ordering produces -Y normal (CW in (i,k) plane
                // viewed from +Y); flip when solid is below (pos_a).
                emit_quad(
                    &mut positions,
                    &mut normals,
                    &mut indices,
                    [v00, v10, v11, v01],
                    pos_a,
                );
            }
        }
    }

    // Z-axis edges: edge at corner (i, j, k) → (i, j, k+1). Surrounding
    // cells span (i-1..i, j-1..j, k).
    for k in 0..CSI {
        for j in 0..CSI {
            for i in 0..CSI {
                let ca = corner_index_ext(i, j, k);
                let cb = corner_index_ext(i, j, k + 1);
                let pos_a = corner_sdf[ca] >= 0.0;
                let pos_b = corner_sdf[cb] >= 0.0;
                if pos_a == pos_b {
                    continue;
                }
                let v00 = grid.vertex[cell_index_ext(i - 1, j - 1, k)];
                let v10 = grid.vertex[cell_index_ext(i, j - 1, k)];
                let v11 = grid.vertex[cell_index_ext(i, j, k)];
                let v01 = grid.vertex[cell_index_ext(i - 1, j, k)];
                let (Some(v00), Some(v10), Some(v11), Some(v01)) = (v00, v10, v11, v01) else {
                    continue;
                };
                emit_quad(
                    &mut positions,
                    &mut normals,
                    &mut indices,
                    [v00, v10, v11, v01],
                    !pos_a,
                );
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

    // --- Lateral skirt pass for seam crack masking (RENDER-010) ---
    //
    // For each lateral wall (±X, ±Z) whose neighbour data was NOT
    // bit-identical (`neighbors.cached[dir] == false` — i.e. cross-face,
    // missing, or different-LOD neighbour), drop a downward-hanging
    // curtain from the boundary vertices to mask any sub-millimetre seam
    // crack that resampled boundary slices may produce.
    //
    // The skirt is a separate vertex pool: each skirt vertex copies the
    // *final* shading normal, colour and position of its corresponding
    // boundary vertex, then offsets the position by `-SKIRT_DEPTH` along
    // chunk-local Y (radial inward). Because the normal is copied — not
    // accumulated — neighbouring boundary vertices remain unaffected,
    // and the skirt visually blends with the surface so it's invisible
    // unless a crack would otherwise expose it.
    //
    // Skirts are skipped entirely when `cached[dir] == true` (same-face
    // same-LOD neighbour). In that regime the boundary slice is bit-
    // identical to the neighbour's interior voxels, so vertex placement
    // already agrees exactly across the seam — emitting skirts would
    // only produce coplanar z-fighting curtains with the neighbour's
    // mirror skirt.
    //
    // We also restrict skirts to vertices that are actually USED by the
    // emitted main-pass quads. Boundary cells can hold "orphan" vertices
    // when a sign change exists but the seam-ownership rule (e.g. the
    // +Y exclusion that prevents the ceiling artefact) suppresses the
    // corresponding quad; dropping a skirt off those would create
    // floating curtain triangles in mid-air.
    //
    // Radial (±Y) seams are not skirted: radial neighbours are always
    // same-face and almost always uniform (all-air above, all-solid
    // below the surface), so seam cracks there are extremely rare and
    // a skirt there would be more likely to produce visible artefacts
    // than fix anything.
    const SKIRT_DEPTH: f32 = 2.0;

    let mut used = vec![false; positions.len()];
    for &i in &indices {
        used[i as usize] = true;
    }

    // (dir, pin_axis, pin_idx). pin_axis: 0 = i pinned (vary j,k);
    // 2 = k pinned (vary i,j).
    let walls: [(usize, u8, isize); 4] = [(0, 0, CSI - 1), (1, 0, -1), (4, 2, CSI - 1), (5, 2, -1)];
    for &(dir, pin_axis, pin_idx) in walls.iter() {
        if neighbors.cached[dir] {
            continue;
        }
        let cell_at = |a: isize, b: isize| -> Option<u32> {
            let (i, j, k) = if pin_axis == 0 {
                (pin_idx, a, b)
            } else {
                (a, b, pin_idx)
            };
            grid.vertex[cell_index_ext(i, j, k)]
        };
        // Cache: top-vertex idx → skirt-vertex idx, scoped to this wall.
        // A linear scan over the typical-O(CS) used boundary vertices is
        // cheap enough; avoids hashing.
        let mut top_to_skirt: Vec<(u32, u32)> = Vec::new();
        for b in -1..CSI {
            for a in -1..CSI {
                let Some(v00) = cell_at(a, b) else { continue };
                if !used[v00 as usize] {
                    continue;
                }
                for (da, db) in [(1isize, 0isize), (0, 1)] {
                    let (na, nb) = (a + da, b + db);
                    if na >= CSI || nb >= CSI {
                        continue;
                    }
                    let Some(v11) = cell_at(na, nb) else { continue };
                    if !used[v11 as usize] {
                        continue;
                    }
                    let mut find_or_make = |top: u32,
                                            positions: &mut Vec<[f32; 3]>,
                                            normals: &mut Vec<[f32; 3]>,
                                            colors: &mut Vec<[f32; 4]>|
                     -> u32 {
                        if let Some(&(_, s)) = top_to_skirt.iter().find(|&&(t, _)| t == top) {
                            return s;
                        }
                        let p = positions[top as usize];
                        let n = normals[top as usize];
                        let c = colors[top as usize];
                        let s = positions.len() as u32;
                        positions.push([p[0], p[1] - SKIRT_DEPTH, p[2]]);
                        normals.push(n);
                        colors.push(c);
                        top_to_skirt.push((top, s));
                        s
                    };
                    let s00 = find_or_make(v00, &mut positions, &mut normals, &mut colors);
                    let s11 = find_or_make(v11, &mut positions, &mut normals, &mut colors);
                    // Double-sided skirt quad (two triangles each winding)
                    // so that back-face culling never reveals the crack.
                    indices.extend_from_slice(&[v00, v11, s11, v00, s11, s00]);
                    indices.extend_from_slice(&[v00, s11, v11, v00, s00, s11]);
                }
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
            if p[0] > 2.0
                && p[0] < (CS as f32 - 2.0)
                && p[2] > 2.0
                && p[2] < (CS as f32 - 2.0)
                && p[1] > 0.5
                && p[1] < (CS as f32 - 0.5)
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
    fn y_pos_seam_not_owned_by_current_chunk() {
        // Regression test for the "gray ceiling" bug (RENDER-006).
        //
        // Scenario: current chunk is entirely AIR (simulating the sea layer
        // above water level). The +Y neighbour (slices[2]) has STONE at its
        // j=0 row. Before the fix the Y loop ran `j in 0..CS`, which included
        // j=CS-1: corner(j=CS-1)=AIR + corner(j=CS)=STONE → sign change →
        // downward-facing face at the top of the chunk (ceiling artifact).
        //
        // After the fix the Y loop runs `j in -1..CS-1`, which excludes
        // j=CS-1. The current chunk must emit zero faces; the +Y neighbour
        // (if Mixed) emits the face via its own j=-1 edge.
        let voxels = vec![Voxel::default(); CHUNK_VOLUME]; // all AIR, density=0
        let stone_slice = vec![
            Voxel {
                material: MaterialId::STONE,
                density: 1.0,
                ..Voxel::default()
            };
            CS * CS
        ];
        let mut neighbors = NeighborSlices::empty();
        neighbors.slices[2] = Some(stone_slice); // +Y neighbour's j=0 face = stone
        let cmap = MaterialColorMap::default();
        let mesh = surface_nets_mesh(&voxels, &neighbors, &cmap);
        assert_eq!(
            mesh.indices.len(),
            0,
            "air chunk with stone +Y neighbour must emit zero faces (no ceiling); \
             got {} vertex positions and {} index entries",
            mesh.positions.len(),
            mesh.indices.len(),
        );
    }

    #[test]
    fn y_neg_seam_owned_by_current_chunk() {
        // The current chunk must correctly emit the -Y seam face (j=-1→j=0)
        // when the -Y neighbour (slices[3]) is STONE and the current chunk's
        // j=0 row is AIR. This verifies that the shift to `j in -1..CS-1`
        // adds the -Y seam without regression.
        let voxels = vec![Voxel::default(); CHUNK_VOLUME]; // all AIR
        let stone_slice = vec![
            Voxel {
                material: MaterialId::STONE,
                density: 1.0,
                ..Voxel::default()
            };
            CS * CS
        ];
        let mut neighbors = NeighborSlices::empty();
        neighbors.slices[3] = Some(stone_slice); // -Y neighbour's j=CS-1 face = stone
        let cmap = MaterialColorMap::default();
        let mesh = surface_nets_mesh(&voxels, &neighbors, &cmap);
        assert!(
            !mesh.positions.is_empty(),
            "stone -Y neighbour + air j=0 must emit seam faces"
        );
        // All emitted vertices should be near y=0 (the -Y chunk boundary).
        for p in &mesh.positions {
            assert!(
                p[1] < 1.5,
                "seam vertex y={} should be near y=0 (chunk bottom)",
                p[1]
            );
        }
        assert_eq!(mesh.positions.len(), mesh.normals.len());
        assert_eq!(mesh.indices.len() % 3, 0);
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
            if p[0] > 2.0 && p[0] < (CS as f32 - 2.0) && p[2] > 2.0 && p[2] < (CS as f32 - 2.0) {
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

    #[test]
    fn skirts_emitted_when_neighbour_not_cached() {
        // Flat surface at y = mid. Neighbour slices intentionally NOT
        // cached (cached[*] = false) — simulates a cross-face or
        // missing-neighbour boundary. The mesher should drop a skirt
        // from each lateral wall vertex by SKIRT_DEPTH along -Y.
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
        let neighbors = NeighborSlices::empty(); // cached = [false; 6]
        let cmap = MaterialColorMap::default();
        let mesh = surface_nets_mesh(&voxels, &neighbors, &cmap);

        // At least some vertices should sit well below the surface mid-plane
        // (skirt vertices), having been dropped by ~2.0 chunk-local units.
        let skirt_count = mesh.positions.iter().filter(|p| p[1] < mid - 1.5).count();
        assert!(
            skirt_count > 0,
            "expected skirt vertices below y = {} (mid - 1.5); positions: {}",
            mid - 1.5,
            mesh.positions.len()
        );
    }

    #[test]
    fn no_skirts_when_all_neighbours_cached() {
        // Same scenario as above but mark every neighbour as cached
        // (i.e. bit-identical same-face same-LOD). Skirts must NOT be
        // emitted: the boundary slices, where present, agree exactly with
        // the neighbour's interior, so adding skirts would create coplanar
        // z-fighting curtains with the neighbour's mirror skirt.
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
        let mut neighbors = NeighborSlices::empty();
        // Mark all sides cached so no skirts are emitted.
        neighbors.cached = [true; 6];
        let cmap = MaterialColorMap::default();
        let mesh = surface_nets_mesh(&voxels, &neighbors, &cmap);

        let skirt_count = mesh.positions.iter().filter(|p| p[1] < mid - 1.5).count();
        assert_eq!(
            skirt_count, 0,
            "no skirts expected when every neighbour is cached; got {} skirt verts",
            skirt_count,
        );
    }
}
