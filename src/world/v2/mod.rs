// V2 rendering pipeline: cubed-sphere chunks with greedy meshing.
//
// This module provides an alternative world rendering pipeline where chunks
// are addressed on a cubed-sphere grid and oriented with local Y pointing
// radially outward. Greedy meshing produces blocky Minecraft-style geometry
// with far fewer triangles than the Surface Nets pipeline in v1.

pub mod boundary_loop;
pub mod chunk_manager;
pub mod cubed_sphere;
pub mod debug;
pub mod greedy_mesh;
pub mod surface_nets;
pub mod terrain_gen;
