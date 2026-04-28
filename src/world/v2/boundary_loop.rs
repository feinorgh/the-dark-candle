// Per-chunk boundary-loop data types for RENDER-010 (explicit stitch meshes).
//
// A "boundary loop" is the ordered set of open surface-nets edges that lie on
// one of a chunk's six faces.  The vertices are stored in world space so that
// the stitch-mesh generator can work directly in a single canonical frame.
//
// Phase 0 (this commit): types and plumbing only.  The component is never
// inserted; systems that need it simply skip entities that lack it.
// Phase 1 will populate the loops after each mesh task completes.

use bevy::math::DVec3;
use bevy::prelude::*;

use super::cubed_sphere::ChunkDir;

/// A single ordered polyline of world-space vertices on one chunk boundary face.
///
/// `vertices` are double-precision world-space positions (1 metre per unit).
/// They form the boundary of the surface-nets mesh on the face identified by
/// `face`.  The loop may be open (terrain intersects the face along one path)
/// or closed (an island of solid in an otherwise-air layer).
///
/// `indices` hold per-vertex metadata reserved for Phase 1 (e.g., the original
/// mesh vertex index so the stitch mesher can share normals).
#[derive(Debug, Clone)]
pub struct BoundaryLoop {
    /// Which of the six chunk faces this loop lies on.
    pub face: ChunkDir,
    /// Ordered world-space vertex positions of the loop.
    pub vertices: Vec<DVec3>,
    /// Per-vertex index metadata (reserved; empty in Phase 0–1).
    pub indices: Vec<u32>,
}

/// Per-face boundary loops extracted from a chunk's surface-nets mesh.
///
/// Stored as a [`Component`] on the chunk entity alongside its [`Mesh`].
/// The slot order matches [`ChunkDir::ALL`]:
/// `[PosU, NegU, PosV, NegV, PosLayer, NegLayer]`.
///
/// In Phase 0 this component is never inserted (all slots `None`).  Phase 1
/// populates it after each mesh task completes.
#[derive(Component, Debug, Default)]
pub struct ChunkBoundaryLoops {
    pub loops: [Option<BoundaryLoop>; 6],
}
