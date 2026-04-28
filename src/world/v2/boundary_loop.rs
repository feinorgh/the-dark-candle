// Per-chunk boundary-loop data types for RENDER-010 (explicit stitch meshes).
//
// A "boundary loop" is the ordered set of open surface-nets edges that lie on
// one of a chunk's six faces.  The vertices are stored in world space so that
// the stitch-mesh generator (Phase 2) can work directly in a single canonical
// frame without having to reconcile the differing local-frame rotations of the
// two chunks being stitched.
//
// ## Axis mapping (verified against `same_face_neighbor_for_dir`)
//
// `world_transform_scaled` builds the chunk rotation as:
//   Local X  = face U tangent projected onto tangent plane  → `ChunkDir::PosU`/`NegU`
//   Local Y  = radial up (layer axis)                       → `ChunkDir::PosLayer`/`NegLayer`
//   Local Z  = right × up  (right-handed)                  → REVERSED V axis
//
// The sign reversal on Z is confirmed by `same_face_neighbor_for_dir`:
//   dir=4 (local +Z) → v−1 = `ChunkDir::NegV`
//   dir=5 (local −Z) → v+1 = `ChunkDir::PosV`
//
// Face planes in local voxel coordinates [0, CS]:
//   PosU      x = CS    NegU      x = 0
//   PosV      z = 0     NegV      z = CS   (PosV = local −Z, NegV = local +Z)
//   PosLayer  y = CS    NegLayer  y = 0
//
// Seam ownership (from surface_nets loop ranges):
//   Owned   : PosU (x > CS−ε), NegLayer (y < ε), NegV (z > CS−ε)
//   Unowned : NegU, PosLayer, PosV  (each owned by the adjacent chunk)
//
// Both owned and unowned boundary loops are extracted; Phase 2 uses whichever
// pair forms the stitch interface.

use std::collections::{HashMap, HashSet};

use bevy::math::DVec3;
use bevy::prelude::*;

use crate::world::chunk::CHUNK_SIZE;
use crate::world::meshing::ChunkMesh;

use super::cubed_sphere::{ChunkDir, CubeSphereCoord};

// ── Constants ─────────────────────────────────────────────────────────────────

/// Distance threshold (in voxel units) for classifying a vertex as lying on a
/// face plane.
///
/// Surface-nets vertex positions are `i + 0.5 + off` where `off ∈ [0,1]`.
/// Boundary-cell vertices (cell index `i = −1` or `i = CS`) reach at most
/// `±0.5` from the face plane.  Interior cells closest to the face (i = 0)
/// can also reach `0.5` in degenerate cases (`off = 0`), so there is no
/// clean separator.  We use `0.55` — slightly above `0.5` — to capture all
/// practical boundary edges.  The rare ambiguous vertex (interior cell at
/// exactly `off = 0`) would only appear on an open edge if the terrain is
/// pathologically degenerate, not for smooth terrain.
const FACE_EPS: f32 = 0.55;

// ── Data types ────────────────────────────────────────────────────────────────

/// A single ordered chain of world-space vertices on one chunk boundary face.
///
/// Each chain is a connected component of open surface-nets edges that lie on
/// the face identified by `face`.  A face may have multiple chains (e.g. a
/// solid island surrounded by air on a horizontal layer).
///
/// Vertices are double-precision world-space positions (1 metre per voxel).
/// `mesh_indices` maps each entry in `vertices` back to the source mesh vertex
/// index so Phase 2 can share normals and material data.
#[derive(Debug, Clone)]
pub struct BoundaryLoop {
    /// Which of the six chunk faces this chain lies on.
    pub face: ChunkDir,
    /// Ordered world-space vertex positions of the chain.
    pub vertices: Vec<DVec3>,
    /// Original mesh vertex indices, parallel to `vertices`.
    pub mesh_indices: Vec<u32>,
}

/// Per-face boundary loops extracted from a chunk's surface-nets mesh.
///
/// Stored as a [`Component`] on the chunk entity alongside its [`Mesh3d`].
///
/// Slot order matches [`ChunkDir::ALL`]:
/// `[PosU, NegU, PosV, NegV, PosLayer, NegLayer]`.
///
/// Each slot holds zero or more chains (connected components).  An empty `Vec`
/// means no open edges were found on that face (all-air chunk, or the face is
/// fully interior).
#[derive(Component, Debug, Default)]
pub struct ChunkBoundaryLoops {
    pub loops: [Vec<BoundaryLoop>; 6],
}

// ── Extraction ────────────────────────────────────────────────────────────────

/// Extract boundary loops from a surface-nets chunk mesh, transforming vertices
/// to world space.
///
/// Returns [`ChunkBoundaryLoops::default`] for an empty mesh.
pub fn extract_boundary_loops(
    mesh: &ChunkMesh,
    coord: CubeSphereCoord,
    mean_radius: f64,
) -> ChunkBoundaryLoops {
    if mesh.is_empty() {
        return ChunkBoundaryLoops::default();
    }

    let cs = CHUNK_SIZE as f32;

    // ── Step 1: collect open edges (appear in exactly one triangle) ───────────
    let mut edge_count: HashMap<(u32, u32), u32> = HashMap::new();
    for tri in mesh.indices.chunks_exact(3) {
        let (a, b, c) = (tri[0], tri[1], tri[2]);
        for &(p, q) in &[(a, b), (b, c), (c, a)] {
            let key = if p < q { (p, q) } else { (q, p) };
            *edge_count.entry(key).or_insert(0) += 1;
        }
    }
    let open_edges: Vec<(u32, u32)> = edge_count
        .into_iter()
        .filter(|(_, count)| *count == 1)
        .map(|(edge, _)| edge)
        .collect();

    if open_edges.is_empty() {
        return ChunkBoundaryLoops::default();
    }

    // ── Step 2: classify each vertex by its nearest face plane ───────────────
    // Returns an index into ChunkDir::ALL, or None for interior vertices.
    let vertex_face: Vec<Option<usize>> = mesh
        .positions
        .iter()
        .map(|&p| classify_vertex(p, cs))
        .collect();

    // Group open edges by face, discarding edges that cross face boundaries.
    let mut edges_by_face: [Vec<(u32, u32)>; 6] = Default::default();
    for (a, b) in open_edges {
        let fa = vertex_face[a as usize];
        let fb = vertex_face[b as usize];
        if let (Some(fa), Some(fb)) = (fa, fb)
            && fa == fb
        {
            edges_by_face[fa].push((a, b));
        }
    }

    // ── Step 3: world-space transform ─────────────────────────────────────────
    let fce = CubeSphereCoord::face_chunks_per_edge_lod(mean_radius, coord.lod);
    let (center, rotation, tangent_scale) = coord.world_transform_scaled_f64(mean_radius, fce);
    let cs_half = cs / 2.0;

    let to_world = |pos: [f32; 3]| -> DVec3 {
        // entity Transform = translation(center) · rotation · scale(tangent_scale)
        // world = center + rotation * ((pos − CS/2) * tangent_scale)
        let local = Vec3::new(
            (pos[0] - cs_half) * tangent_scale.x,
            (pos[1] - cs_half) * tangent_scale.y,
            (pos[2] - cs_half) * tangent_scale.z,
        );
        center + DVec3::from(rotation * local)
    };

    // ── Step 4: walk connected chains per face ─────────────────────────────────
    let mut result = ChunkBoundaryLoops::default();
    for (face_idx, face_edges) in edges_by_face.iter().enumerate() {
        if face_edges.is_empty() {
            continue;
        }
        let face = ChunkDir::ALL[face_idx];
        for chain_indices in walk_chains(face_edges) {
            if chain_indices.len() < 2 {
                continue;
            }
            let vertices: Vec<DVec3> = chain_indices
                .iter()
                .map(|&vi| to_world(mesh.positions[vi as usize]))
                .collect();
            result.loops[face_idx].push(BoundaryLoop {
                face,
                vertices,
                mesh_indices: chain_indices,
            });
        }
    }
    result
}

/// Classify a mesh vertex by which face plane it lies closest to.
///
/// Returns an index into [`ChunkDir::ALL`] = `[PosU, NegU, PosV, NegV, PosLayer, NegLayer]`,
/// or `None` if the vertex is not within [`FACE_EPS`] of any face plane.
///
/// When a vertex is equidistant from two faces (corner case), the face that
/// appears first in [`ChunkDir::ALL`] is returned — this is deterministic.
fn classify_vertex(pos: [f32; 3], cs: f32) -> Option<usize> {
    let [x, y, z] = pos;
    // Distance from each face plane (0 = on the plane).
    // Order must match ChunkDir::ALL: [PosU, NegU, PosV, NegV, PosLayer, NegLayer].
    let dists: [f32; 6] = [
        cs - x, // PosU      face plane x = CS
        x,      // NegU      face plane x = 0
        z,      // PosV      face plane z = 0   (local −Z = +V)
        cs - z, // NegV      face plane z = CS  (local +Z = −V)
        cs - y, // PosLayer  face plane y = CS
        y,      // NegLayer  face plane y = 0
    ];
    dists
        .iter()
        .enumerate()
        .filter(|&(_, &d)| d < FACE_EPS)
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
}

/// Walk open edges into ordered chains (connected components).
///
/// Each returned `Vec<u32>` is the vertex index sequence for one chain.
/// For open paths the chain starts at a degree-1 endpoint; for closed loops
/// it starts at the vertex with the smallest index (for determinism) and does
/// not repeat the first vertex at the end.
fn walk_chains(edges: &[(u32, u32)]) -> Vec<Vec<u32>> {
    if edges.is_empty() {
        return Vec::new();
    }

    let mut adj: HashMap<u32, Vec<u32>> = HashMap::new();
    for &(a, b) in edges {
        adj.entry(a).or_default().push(b);
        adj.entry(b).or_default().push(a);
    }

    let mut unvisited: HashSet<u32> = adj.keys().copied().collect();
    let mut chains: Vec<Vec<u32>> = Vec::new();

    while !unvisited.is_empty() {
        // Prefer degree-1 vertices (path endpoints) as chain start;
        // for closed loops (all degree-2) pick the smallest index for determinism.
        let start = *unvisited
            .iter()
            .min_by_key(|&&v| (adj[&v].len(), v))
            .unwrap();

        let mut chain: Vec<u32> = Vec::new();
        let mut cur = start;
        let mut prev = u32::MAX;

        loop {
            unvisited.remove(&cur);
            chain.push(cur);
            let next = adj[&cur]
                .iter()
                .copied()
                .find(|&n| n != prev && unvisited.contains(&n));
            match next {
                None => break,
                Some(n) => {
                    prev = cur;
                    cur = n;
                }
            }
        }

        if chain.len() >= 2 {
            chains.push(chain);
        }
    }

    chains
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::v2::cubed_sphere::CubeFace;

    fn make_mesh(positions: Vec<[f32; 3]>, indices: Vec<u32>) -> ChunkMesh {
        let n = positions.len();
        ChunkMesh {
            positions,
            normals: vec![[0.0, 1.0, 0.0]; n],
            colors: vec![[1.0, 1.0, 1.0, 1.0]; n],
            indices,
        }
    }

    fn test_coord() -> CubeSphereCoord {
        CubeSphereCoord::new_with_lod(CubeFace::PosZ, 0, 0, 4, 0)
    }

    #[test]
    fn empty_mesh_returns_no_loops() {
        let mesh = make_mesh(Vec::new(), Vec::new());
        let loops = extract_boundary_loops(&mesh, test_coord(), 6_400_000.0);
        for face_loops in &loops.loops {
            assert!(face_loops.is_empty());
        }
    }

    #[test]
    fn classify_vertex_maps_correct_faces() {
        let cs = CHUNK_SIZE as f32;
        let mid = cs / 2.0;
        // PosU: index 0 (x near CS)
        assert_eq!(classify_vertex([cs - 0.3, mid, mid], cs), Some(0));
        // NegU: index 1 (x near 0)
        assert_eq!(classify_vertex([0.3, mid, mid], cs), Some(1));
        // PosV: index 2 (z near 0, local −Z = +V)
        assert_eq!(classify_vertex([mid, mid, 0.3], cs), Some(2));
        // NegV: index 3 (z near CS, local +Z = −V)
        assert_eq!(classify_vertex([mid, mid, cs - 0.3], cs), Some(3));
        // PosLayer: index 4 (y near CS)
        assert_eq!(classify_vertex([mid, cs - 0.3, mid], cs), Some(4));
        // NegLayer: index 5 (y near 0)
        assert_eq!(classify_vertex([mid, 0.3, mid], cs), Some(5));
        // Interior vertex → None
        assert_eq!(classify_vertex([mid, mid, mid], cs), None);
        // Corner: PosU (x=CS-0.2, dist=0.2) wins over NegLayer (y=0.5, dist=0.5)
        assert_eq!(classify_vertex([cs - 0.2, 0.5, mid], cs), Some(0));
    }

    #[test]
    fn walk_chains_simple_path() {
        // 0 ── 1 ── 2 ── 3
        let edges = vec![(0u32, 1u32), (1, 2), (2, 3)];
        let chains = walk_chains(&edges);
        assert_eq!(chains.len(), 1);
        let chain = &chains[0];
        assert_eq!(chain.len(), 4);
        // Must start from a degree-1 endpoint (0 or 3)
        let first = chain[0];
        let last = *chain.last().unwrap();
        assert!(first == 0 || first == 3, "chain must start at endpoint");
        assert!(last == 0 || last == 3, "chain must end at endpoint");
        assert_ne!(first, last);
    }

    #[test]
    fn walk_chains_two_components() {
        // Component A: 0 ── 1 ── 2
        // Component B: 3 ── 4
        let edges = vec![(0u32, 1u32), (1, 2), (3, 4)];
        let chains = walk_chains(&edges);
        assert_eq!(chains.len(), 2);
        let mut lens: Vec<usize> = chains.iter().map(|c| c.len()).collect();
        lens.sort_unstable();
        assert_eq!(lens, vec![2, 3]);
    }

    #[test]
    fn walk_chains_closed_loop() {
        // 0 ── 1 ── 2 ── 0  (triangle boundary)
        let edges = vec![(0u32, 1u32), (1, 2), (2, 0)];
        let chains = walk_chains(&edges);
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].len(), 3); // closed loop: last vertex's neighbor is start
    }

    /// A synthetic mesh with four open boundary edges on the side faces
    /// (PosU, NegU, PosV, NegV) but none on the layer faces (PosLayer, NegLayer).
    /// Simulates a horizontal terrain cut.
    #[test]
    fn flat_plane_no_layer_loops() {
        let cs = CHUNK_SIZE as f32;
        let y = cs / 2.0;
        let eps = 0.3; // clearly inside FACE_EPS = 0.55
        let mid = cs / 4.0;
        let far = 3.0 * cs / 4.0;
        //                index  face
        let positions = vec![
            [cs - eps, y, mid], // 0 PosU
            [cs - eps, y, far], // 1 PosU
            [eps, y, mid],      // 2 NegU
            [eps, y, far],      // 3 NegU
            [mid, y, cs - eps], // 4 NegV
            [far, y, cs - eps], // 5 NegV
            [mid, y, eps],      // 6 PosV
            [far, y, eps],      // 7 PosV
        ];
        // Interior vertex to complete each "triangle" — keeps each open edge
        // with count=1 (appears in only one triangle).
        let mut all_positions = positions.clone();
        let interior = all_positions.len() as u32;
        all_positions.push([cs / 2.0, y, cs / 2.0]);
        // Each boundary edge appears exactly once → open edge.
        let indices = vec![
            0, 1, interior, // PosU open edge 0–1
            2, 3, interior, // NegU open edge 2–3
            4, 5, interior, // NegV open edge 4–5
            6, 7, interior, // PosV open edge 6–7
        ];
        let mesh = make_mesh(all_positions, indices);
        let loops = extract_boundary_loops(&mesh, test_coord(), 6_400_000.0);

        assert!(
            !loops.loops[0].is_empty(),
            "PosU (index 0) should have a loop"
        );
        assert!(
            !loops.loops[1].is_empty(),
            "NegU (index 1) should have a loop"
        );
        assert!(
            !loops.loops[2].is_empty(),
            "PosV (index 2) should have a loop"
        );
        assert!(
            !loops.loops[3].is_empty(),
            "NegV (index 3) should have a loop"
        );
        assert!(
            loops.loops[4].is_empty(),
            "PosLayer (index 4) should be empty"
        );
        assert!(
            loops.loops[5].is_empty(),
            "NegLayer (index 5) should be empty"
        );
    }

    /// Two disconnected solid islands on the same face must produce two separate
    /// BoundaryLoop entries in that face's slot.
    #[test]
    fn two_disconnected_chains_same_face() {
        let cs = CHUNK_SIZE as f32;
        let y = cs / 2.0;
        let eps = 0.3;
        // Island A: vertices 0,1 on PosU face
        // Island B: vertices 2,3 on PosU face — not connected to A
        let positions = vec![
            [cs - eps, y, 2.0],      // 0
            [cs - eps, y, 4.0],      // 1
            [cs - eps, y, cs - 4.0], // 2
            [cs - eps, y, cs - 2.0], // 3
        ];
        let mut all_positions = positions.clone();
        let interior = all_positions.len() as u32;
        all_positions.push([cs / 2.0, y, cs / 2.0]);
        let indices = vec![
            0, 1, interior, // open edge A
            2, 3, interior, // open edge B
        ];
        let mesh = make_mesh(all_positions, indices);
        let loops = extract_boundary_loops(&mesh, test_coord(), 6_400_000.0);

        assert_eq!(
            loops.loops[0].len(),
            2,
            "PosU should have two disconnected chains"
        );
    }
}
