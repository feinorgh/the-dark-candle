// Shared mesh types used by the V2 cubed-sphere pipeline.
//
// The V1 surface-nets mesh generation has been removed; this module now
// only exposes the output types and the Bevy-mesh conversion helper that
// V2 mesh workers produce and consume.

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, Mesh, PrimitiveTopology};
use bevy::prelude::*;

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

/// Convert a [`ChunkMesh`] into a Bevy [`Mesh`] asset, consuming the mesh data.
pub fn chunk_mesh_to_bevy_mesh(chunk_mesh: ChunkMesh) -> Mesh {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, chunk_mesh.positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, chunk_mesh.normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, chunk_mesh.colors);
    mesh.insert_indices(Indices::U32(chunk_mesh.indices));
    mesh
}

/// Marker component for entities that have a chunk mesh.
#[derive(Component)]
pub struct ChunkMeshMarker;
