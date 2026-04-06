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
use bevy::tasks::Task;
use std::collections::HashMap;
use std::sync::Arc;

use super::chunk::{CHUNK_SIZE, Chunk, ChunkOctree};
use super::lod::{LodLevel, MaterialColorMap};
use super::octree::OctreeNode;
use super::voxel::{MaterialId, Voxel};
use crate::lighting::light_map::{ChunkLightMap, apply_light_map};

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

/// Resource toggling thermal-vision debug overlay. When enabled, vertex colors
/// use a blue→cyan→green→yellow→red heatmap based on temperature.
#[derive(Resource, Debug, Clone, Copy, Default)]
pub struct ThermalVisionMode(pub bool);

// --- Incandescence ---

/// Temperature (K) below which voxels have no glow.
const GLOW_THRESHOLD: f32 = 800.0;

/// Compute an incandescent color for a voxel given its base material color
/// and temperature. Below `GLOW_THRESHOLD` the base color is returned
/// unchanged. Above it, the color blends toward a physically-motivated
/// incandescence ramp and HDR values > 1.0 enable bloom.
pub fn incandescence_color(base: [f32; 4], temperature: f32) -> [f32; 4] {
    if temperature < GLOW_THRESHOLD {
        return base;
    }

    // Incandescence ramp: dark red → cherry red → orange → yellow-white
    let (r, g, b) = if temperature < 1200.0 {
        let t = (temperature - 800.0) / 400.0;
        (0.3 + 0.4 * t, 0.02 * t, 0.0)
    } else if temperature < 1500.0 {
        let t = (temperature - 1200.0) / 300.0;
        (0.7 + 0.3 * t, 0.02 + 0.28 * t, 0.0)
    } else if temperature < 1800.0 {
        let t = (temperature - 1500.0) / 300.0;
        (1.0, 0.3 + 0.4 * t, 0.1 * t)
    } else {
        (
            1.0,
            0.7 + 0.3 * ((temperature - 1800.0) / 500.0).min(1.0),
            0.1 + 0.9 * ((temperature - 1800.0) / 500.0).min(1.0),
        )
    };

    // Blend base toward incandescent color
    let blend = ((temperature - GLOW_THRESHOLD) / 400.0).min(1.0);
    let cr = base[0] * (1.0 - blend) + r * blend;
    let cg = base[1] * (1.0 - blend) + g * blend;
    let cb = base[2] * (1.0 - blend) + b * blend;

    // HDR emissive multiplier — scales with T⁴ (Stefan-Boltzmann) normalized
    // to a moderate bloom range. At 1000 K → ~1.5×, at 2000 K → ~24×.
    let t_norm = temperature / 1000.0;
    let emissive = t_norm * t_norm * t_norm * t_norm * 0.1;
    let scale = 1.0 + emissive;

    [cr * scale, cg * scale, cb * scale, base[3]]
}

/// Like [`incandescence_color`] but also applies material-based emission glow
/// for active emitters (LEDs, light panels).  The material emission is additive
/// on top of the temperature-based incandescence.
pub fn incandescence_color_with_emission(
    base: [f32; 4],
    temperature: f32,
    mat: &crate::data::MaterialData,
) -> [f32; 4] {
    let mut result = incandescence_color(base, temperature);

    if let Some(power) = mat.emission_power.filter(|&p| p > 0.0) {
        let color = mat.resolved_emission_color();
        let spectrum = mat.resolved_emission_spectrum();
        let intensity = (power * spectrum.visible / 10_000.0).min(2.0);
        result[0] += color[0] * intensity;
        result[1] += color[1] * intensity;
        result[2] += color[2] * intensity;
    }

    result
}

/// Map temperature to a debug heatmap color (blue→cyan→green→yellow→red).
fn thermal_heatmap_color(temperature: f32) -> [f32; 4] {
    let min_k = 250.0_f32;
    let max_k = 2000.0_f32;
    let t = ((temperature - min_k) / (max_k - min_k)).clamp(0.0, 1.0);

    let (r, g, b) = if t < 0.25 {
        let s = t / 0.25;
        (0.0, s, 1.0)
    } else if t < 0.5 {
        let s = (t - 0.25) / 0.25;
        (0.0, 1.0, 1.0 - s)
    } else if t < 0.75 {
        let s = (t - 0.5) / 0.25;
        (s, 1.0, 0.0)
    } else {
        let s = (t - 0.75) / 0.25;
        (1.0, 1.0 - s, 0.0)
    };

    [r, g, b, 1.0]
}

/// Map a material ID to an RGBA color for rendering.
///
/// Hardcoded fallback used when no `MaterialColorMap` is available.
fn material_color_fallback(mat: MaterialId) -> [f32; 4] {
    match mat.0 {
        0 => [0.0, 0.0, 0.0, 0.0],     // Air (invisible gas)
        1 => [0.5, 0.5, 0.5, 1.0],     // Stone (gray)
        2 => [0.45, 0.32, 0.18, 1.0],  // Dirt (brown)
        3 => [0.2, 0.4, 0.8, 0.8],     // Water (blue, semi-transparent)
        4 => [0.6, 0.6, 0.65, 1.0],    // Iron (silver/gray metallic)
        5 => [0.6, 0.4, 0.2, 1.0],     // Wood (brown)
        6 => [0.85, 0.78, 0.55, 1.0],  // Sand (tan)
        7 => [0.3, 0.6, 0.2, 1.0],     // Grass (green)
        8 => [0.7, 0.85, 0.95, 0.9],   // Ice (pale blue)
        9 => [0.9, 0.9, 0.95, 0.3],    // Steam (faint white)
        10 => [0.9, 0.3, 0.1, 1.0],    // Lava (orange-red)
        11 => [0.65, 0.65, 0.6, 1.0],  // Ash (gray)
        12 => [0.85, 0.9, 0.92, 0.4],  // Glass (transparent)
        13 => [0.0, 0.0, 0.0, 0.0],    // Oxygen (invisible gas)
        14 => [0.0, 0.0, 0.0, 0.0],    // Hydrogen (invisible gas)
        15 => [0.4, 0.35, 0.2, 1.0],   // Organic matter (dark brown-green)
        16 => [0.45, 0.30, 0.15, 1.0], // Twig (light brown)
        17 => [0.55, 0.42, 0.18, 1.0], // Dry leaves (orange-brown)
        18 => [0.35, 0.22, 0.12, 1.0], // Bark (dark brown)
        19 => [0.15, 0.12, 0.10, 1.0], // Charcoal (very dark)
        20 => [0.76, 0.65, 0.42, 1.0], // Sandstone (warm beige)
        21 => [0.82, 0.80, 0.72, 1.0], // Limestone (pale gray)
        22 => [0.66, 0.60, 0.58, 1.0], // Granite (pinkish gray)
        23 => [0.30, 0.30, 0.32, 1.0], // Basalt (very dark gray)
        24 => [0.12, 0.12, 0.12, 1.0], // Coal (near-black)
        25 => [0.45, 0.58, 0.40, 1.0], // Copper ore (greenish-brown)
        26 => [0.70, 0.60, 0.30, 1.0], // Gold ore (yellowish)
        27 => [0.90, 0.88, 0.95, 1.0], // Quartz crystal (pale violet)
        _ => [0.8, 0.0, 0.8, 1.0],     // Unknown (magenta)
    }
}

/// Resolve a material's RGBA color, preferring the registry-backed color map.
fn material_color(mat: MaterialId, color_map: Option<&MaterialColorMap>) -> [f32; 4] {
    color_map
        .map(|m| m.get(mat))
        .unwrap_or_else(|| material_color_fallback(mat))
}

/// Sample the scalar field using the voxel's continuous density.
/// Out-of-bounds samples return 0.0 (air) to create surfaces at chunk edges
/// when no neighbor data is available.
/// Returns (density, material, temperature_K).
#[inline]
fn sample(chunk: &Chunk, x: i32, y: i32, z: i32) -> (f32, MaterialId, f32) {
    if x >= 0
        && y >= 0
        && z >= 0
        && (x as usize) < CHUNK_SIZE
        && (y as usize) < CHUNK_SIZE
        && (z as usize) < CHUNK_SIZE
    {
        let v = chunk.get(x as usize, y as usize, z as usize);
        (v.density, v.material, v.temperature)
    } else {
        (0.0, MaterialId::AIR, 288.15)
    }
}

/// Voxel snapshots for the 6 face-neighbors of a chunk (±X, ±Y, ±Z).
///
/// When present each slice is a flat `CHUNK_SIZE³` voxel array (same layout as
/// `Chunk::flat_snapshot()`). `None` means the neighbor is not loaded; the
/// mesher will fall back to treating that boundary as air, which is correct for
/// chunks at the edge of the loaded region.
#[derive(Default, Clone)]
pub struct NeighborVoxels {
    pub px: Option<Arc<Vec<Voxel>>>, // +X neighbor
    pub nx: Option<Arc<Vec<Voxel>>>, // −X neighbor
    pub py: Option<Arc<Vec<Voxel>>>, // +Y neighbor
    pub ny: Option<Arc<Vec<Voxel>>>, // −Y neighbor
    pub pz: Option<Arc<Vec<Voxel>>>, // +Z neighbor
    pub nz: Option<Arc<Vec<Voxel>>>, // −Z neighbor
}

impl NeighborVoxels {
    /// Sample a voxel from a flat neighbor slice.
    #[inline]
    fn sample_slice(slice: &[Voxel], x: usize, y: usize, z: usize) -> (f32, MaterialId, f32) {
        let idx = x * CHUNK_SIZE * CHUNK_SIZE + y * CHUNK_SIZE + z;
        let v = &slice[idx];
        (v.density, v.material, v.temperature)
    }

    /// Sample the scalar field, crossing into neighbor chunks when needed.
    ///
    /// Coordinates may range from −1 to CHUNK_SIZE (inclusive on both ends),
    /// allowing the Surface Nets cell loop to read the full boundary layer.
    #[inline]
    pub fn sample(&self, chunk: &Chunk, x: i32, y: i32, z: i32) -> (f32, MaterialId, f32) {
        let cs = CHUNK_SIZE as i32;

        // All three coordinates in-bounds → read directly from chunk.
        if x >= 0 && y >= 0 && z >= 0 && x < cs && y < cs && z < cs {
            return sample(chunk, x, y, z);
        }

        // Clamp the in-chunk coordinates for the neighbor we're crossing into.
        let nx = x.rem_euclid(cs) as usize;
        let ny = y.rem_euclid(cs) as usize;
        let nz = z.rem_euclid(cs) as usize;

        let neighbor = if x < 0 {
            self.nx.as_deref()
        } else if x >= cs {
            self.px.as_deref()
        } else if y < 0 {
            self.ny.as_deref()
        } else if y >= cs {
            self.py.as_deref()
        } else if z < 0 {
            self.nz.as_deref()
        } else {
            self.pz.as_deref()
        };

        match neighbor {
            Some(slice) => Self::sample_slice(slice, nx, ny, nz),
            None => (0.0, MaterialId::AIR, 288.15),
        }
    }
}

/// Generate a mesh from a chunk's voxel data using the Surface Nets algorithm.
pub fn generate_mesh(chunk: &Chunk) -> ChunkMesh {
    generate_mesh_with_colors(chunk, &NeighborVoxels::default(), None, false, false)
}

/// Generate a mesh from a chunk, using the color map for material colors.
///
/// `neighbors` provides the 6 face-adjacent chunk snapshots so the mesher can
/// read across chunk boundaries and eliminate seam gaps.
pub fn generate_mesh_with_colors(
    chunk: &Chunk,
    neighbors: &NeighborVoxels,
    color_map: Option<&MaterialColorMap>,
    thermal_vision: bool,
    spherical: bool,
) -> ChunkMesh {
    generate_mesh_generic(
        CHUNK_SIZE as i32 + 1,
        1.0,
        spherical,
        |x, y, z| neighbors.sample(chunk, x, y, z),
        |mat, temp| {
            if thermal_vision {
                thermal_heatmap_color(temp)
            } else {
                incandescence_color(material_color(mat, color_map), temp)
            }
        },
    )
}

/// Sample the scalar field from an octree at a given cell coordinate and size.
/// Returns (density, material, temperature_K). Out-of-bounds → air.
#[inline]
fn sample_octree(
    tree: &OctreeNode<Voxel>,
    size: usize,
    x: i32,
    y: i32,
    z: i32,
) -> (f32, MaterialId, f32) {
    if x >= 0
        && y >= 0
        && z >= 0
        && (x as usize) < size
        && (y as usize) < size
        && (z as usize) < size
    {
        let v = tree.get(x as usize, y as usize, z as usize, size);
        (v.density, v.material, v.temperature)
    } else {
        (0.0, MaterialId::AIR, 288.15)
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
    generate_mesh_from_octree_with_colors(
        tree,
        size,
        &NeighborVoxels::default(),
        None,
        false,
        false,
    )
}

/// Generate a mesh from an octree volume, using the color map for materials.
///
/// `neighbors` provides the 6 face-adjacent chunk snapshots so boundary seams
/// are eliminated. The octree covers the same `size`³ space; boundary samples
/// that fall outside fall through to the neighbor slices.
pub fn generate_mesh_from_octree_with_colors(
    tree: &OctreeNode<Voxel>,
    size: usize,
    neighbors: &NeighborVoxels,
    color_map: Option<&MaterialColorMap>,
    thermal_vision: bool,
    spherical: bool,
) -> ChunkMesh {
    generate_mesh_generic(
        size as i32 + 1,
        1.0,
        spherical,
        |x, y, z| {
            let cs = size as i32;
            if x >= 0 && y >= 0 && z >= 0 && x < cs && y < cs && z < cs {
                sample_octree(tree, size, x, y, z)
            } else {
                // Delegate out-of-bounds to neighbor chunks.
                // Neighbor slices are always at base CHUNK_SIZE resolution.
                let nx = x.rem_euclid(cs) as usize;
                let ny = y.rem_euclid(cs) as usize;
                let nz = z.rem_euclid(cs) as usize;
                let neighbor = if x < 0 {
                    neighbors.nx.as_deref()
                } else if x >= cs {
                    neighbors.px.as_deref()
                } else if y < 0 {
                    neighbors.ny.as_deref()
                } else if y >= cs {
                    neighbors.py.as_deref()
                } else if z < 0 {
                    neighbors.nz.as_deref()
                } else {
                    neighbors.pz.as_deref()
                };
                match neighbor {
                    Some(slice) => NeighborVoxels::sample_slice(slice, nx, ny, nz),
                    None => (0.0, MaterialId::AIR, 288.15),
                }
            }
        },
        |mat, temp| {
            if thermal_vision {
                thermal_heatmap_color(temp)
            } else {
                incandescence_color(material_color(mat, color_map), temp)
            }
        },
    )
}

/// Generate a mesh at reduced resolution for LOD.
///
/// `lod_step` is the cell stride: 1 = full resolution (32³), 2 = half (16³),
/// 4 = quarter (8³), etc. The output mesh covers the same world-space volume
/// but with fewer vertices and triangles.
pub fn generate_mesh_lod(chunk: &Chunk, lod_step: usize) -> ChunkMesh {
    generate_mesh_lod_with_colors(chunk, lod_step, None, false, false)
}

/// Generate a mesh at reduced resolution for LOD, using the color map.
pub fn generate_mesh_lod_with_colors(
    chunk: &Chunk,
    lod_step: usize,
    color_map: Option<&MaterialColorMap>,
    thermal_vision: bool,
    spherical: bool,
) -> ChunkMesh {
    assert!(lod_step > 0 && lod_step.is_power_of_two());
    let effective_size = CHUNK_SIZE / lod_step;
    let step = lod_step as i32;

    generate_mesh_generic(
        effective_size as i32,
        lod_step as f32,
        spherical,
        |x, y, z| {
            let fx = x * step;
            let fy = y * step;
            let fz = z * step;
            sample(chunk, fx, fy, fz)
        },
        |mat, temp| {
            if thermal_vision {
                thermal_heatmap_color(temp)
            } else {
                incandescence_color(material_color(mat, color_map), temp)
            }
        },
    )
}

/// Generate a mesh from an octree at reduced resolution for LOD.
pub fn generate_mesh_from_octree_lod(
    tree: &OctreeNode<Voxel>,
    size: usize,
    lod_step: usize,
) -> ChunkMesh {
    generate_mesh_from_octree_lod_with_colors(tree, size, lod_step, None, false, false)
}

/// Generate a mesh from an octree at reduced resolution, using the color map.
pub fn generate_mesh_from_octree_lod_with_colors(
    tree: &OctreeNode<Voxel>,
    size: usize,
    lod_step: usize,
    color_map: Option<&MaterialColorMap>,
    thermal_vision: bool,
    spherical: bool,
) -> ChunkMesh {
    assert!(lod_step > 0 && lod_step.is_power_of_two());
    let effective_size = size / lod_step;
    let step = lod_step as i32;

    generate_mesh_generic(
        effective_size as i32,
        lod_step as f32,
        spherical,
        |x, y, z| {
            let fx = x * step;
            let fy = y * step;
            let fz = z * step;
            sample_octree(tree, size, fx, fy, fz)
        },
        |mat, temp| {
            if thermal_vision {
                thermal_heatmap_color(temp)
            } else {
                incandescence_color(material_color(mat, color_map), temp)
            }
        },
    )
}

/// Core Surface Nets implementation parameterized over sampling and color functions.
///
/// `grid_size` is the number of cells along each axis.
/// `scale` multiplies each vertex position so that LOD meshes (with reduced
/// grid_size) still span the full chunk extent.
/// `sample_fn(x, y, z)` returns (scalar, material, temperature) for the given cell corner.
/// `color_fn(mat, temperature)` returns the RGBA color for a material and temperature.
fn generate_mesh_generic<F, C>(
    grid_size: i32,
    scale: f32,
    spherical: bool,
    sample_fn: F,
    color_fn: C,
) -> ChunkMesh
where
    F: Fn(i32, i32, i32) -> (f32, MaterialId, f32),
    C: Fn(MaterialId, f32) -> [f32; 4],
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

    // Pre-allocate with estimated capacity: surface vertices scale as O(grid²).
    // Typical terrain chunks use ~2000 vertices for a 32³ grid.
    let estimated_vertices = (grid_size * grid_size) as usize;
    let estimated_indices = estimated_vertices * 3;

    let mut vertex_map: HashMap<(i32, i32, i32), u32> = HashMap::with_capacity(estimated_vertices);
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(estimated_vertices);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(estimated_vertices);
    let mut colors: Vec<[f32; 4]> = Vec::with_capacity(estimated_vertices);

    // Phase 1: Vertex placement
    for cz in 0..grid_size {
        for cy in 0..grid_size {
            for cx in 0..grid_size {
                let mut corner_values = [0.0f32; 8];
                let mut corner_mats = [MaterialId::AIR; 8];
                let mut corner_temps = [288.15_f32; 8];
                let mut solid_count = 0u32;

                for (i, &(dx, dy, dz)) in corners.iter().enumerate() {
                    let (val, mat, temp) = sample_fn(cx + dx, cy + dy, cz + dz);
                    corner_values[i] = val;
                    corner_mats[i] = mat;
                    corner_temps[i] = temp;
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
                    (cx as f32 + vertex_pos.x) * scale,
                    (cy as f32 + vertex_pos.y) * scale,
                    (cz as f32 + vertex_pos.z) * scale,
                ];

                // Pick the dominant non-air material and its temperature.
                let dominant_idx = corner_mats.iter().position(|m| !m.is_air()).unwrap_or(0);
                let dominant_mat = corner_mats[dominant_idx];
                let dominant_temp = corner_temps[dominant_idx];

                let idx = positions.len() as u32;
                vertex_map.insert((cx, cy, cz), idx);
                positions.push(world_pos);
                normals.push([0.0, 1.0, 0.0]);
                colors.push(color_fn(dominant_mat, dominant_temp));
            }
        }
    }

    // Phase 2: Quad emission
    let mut indices: Vec<u32> = Vec::with_capacity(estimated_indices);

    for (&(cx, cy, cz), &v0) in &vertex_map {
        // X-edge
        if let (Some(&v1), Some(&v2), Some(&v3)) = (
            vertex_map.get(&(cx, cy - 1, cz)),
            vertex_map.get(&(cx, cy, cz - 1)),
            vertex_map.get(&(cx, cy - 1, cz - 1)),
        ) {
            let (s0, _, _) = sample_fn(cx, cy, cz);
            let (s1, _, _) = sample_fn(cx + 1, cy, cz);
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
            let (s0, _, _) = sample_fn(cx, cy, cz);
            let (s1, _, _) = sample_fn(cx, cy + 1, cz);
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
            let (s0, _, _) = sample_fn(cx, cy, cz);
            let (s1, _, _) = sample_fn(cx, cy, cz + 1);
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
    compute_normals(&positions, &indices, &mut normals, spherical);

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
///
/// When `spherical` is true, degenerate normals (zero accumulated face area)
/// fall back to the radial direction `normalize(vertex_position)` instead of
/// global Y.  This is correct for terrain on a sphere centered at the origin.
fn compute_normals(
    positions: &[[f32; 3]],
    indices: &[u32],
    normals: &mut [[f32; 3]],
    spherical: bool,
) {
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
    for (i, n) in normals.iter_mut().enumerate() {
        let v = Vec3::from_array(*n);
        let len = v.length();
        if len > 1e-8 {
            *n = (v / len).to_array();
        } else if spherical {
            // On a sphere centered at the origin, the surface normal is the
            // radial direction from center through the vertex.
            let pos = Vec3::from_array(positions[i]);
            *n = pos.normalize_or(Vec3::Y).to_array();
        } else {
            *n = [0.0, 1.0, 0.0];
        }
    }
}

/// Convert a ChunkMesh into a Bevy Mesh asset, consuming the mesh data
/// to avoid cloning the position/normal/color/index buffers.
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

/// Result from an async mesh computation.
#[allow(dead_code)]
struct ChunkMeshResult {
    mesh: ChunkMesh,
    lod_level: u8,
}

/// Component holding a pending async mesh task along with dispatch-time
/// metadata needed when the result is applied.
#[derive(Component)]
#[allow(dead_code)]
pub struct MeshTask {
    task: Task<ChunkMeshResult>,
    previous_level: LodLevel,
    lod_changed: bool,
    had_mesh: bool,
}

/// Compute the mesh stride for a given LOD level.
/// L0 = 1 (full 32³), L1 = 2 (16³), L2 = 4 (8³), etc.
#[allow(dead_code)]
fn lod_step(level: LodLevel) -> usize {
    let step = 1usize << level.0;
    step.min(CHUNK_SIZE / 2) // Don't go below 2³
}

/// Generate a mesh for a chunk, preferring the octree path when available.
///
/// When a `ChunkOctree` is present, meshes from the sparse representation.
/// Falls back to flat-array meshing otherwise (e.g. for chunks modified
/// after octree construction whose octree hasn't been rebuilt yet).
///
/// `neighbors` carries the 6 face-adjacent chunk snapshots needed to eliminate
/// seam gaps at chunk boundaries.
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn generate_chunk_mesh(
    chunk: &Chunk,
    octree: Option<&ChunkOctree>,
    lod_step_size: usize,
    neighbors: &NeighborVoxels,
    color_map: Option<&MaterialColorMap>,
    thermal_vision: bool,
    light_map: Option<&ChunkLightMap>,
    spherical: bool,
) -> ChunkMesh {
    let mut mesh = match octree {
        Some(oct) if lod_step_size <= 1 => generate_mesh_from_octree_with_colors(
            &oct.0,
            CHUNK_SIZE,
            neighbors,
            color_map,
            thermal_vision,
            spherical,
        ),
        Some(oct) => generate_mesh_from_octree_lod_with_colors(
            &oct.0,
            CHUNK_SIZE,
            lod_step_size,
            color_map,
            thermal_vision,
            spherical,
        ),
        None if lod_step_size <= 1 => {
            generate_mesh_with_colors(chunk, neighbors, color_map, thermal_vision, spherical)
        }
        None => generate_mesh_lod_with_colors(
            chunk,
            lod_step_size,
            color_map,
            thermal_vision,
            spherical,
        ),
    };

    if let Some(lm) = light_map {
        apply_light_map(&mesh.positions, &mut mesh.colors, lm);
    }

    mesh
}

/// System: toggle thermal vision mode with the T key. Marks all chunks dirty
/// so they remesh with the new color mode.
pub fn toggle_thermal_vision(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut mode: ResMut<ThermalVisionMode>,
    mut chunks: Query<&mut Chunk>,
) {
    if !keyboard.just_pressed(KeyCode::KeyT) {
        return;
    }
    mode.0 = !mode.0;
    info!("Thermal vision: {}", if mode.0 { "ON" } else { "OFF" });
    // Mark all chunks dirty to force remesh with new color mode.
    for mut chunk in &mut chunks {
        chunk.mark_dirty();
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
        let _bevy_mesh = chunk_mesh_to_bevy_mesh(chunk_mesh);
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

    // --- Thermal glow tests ---

    #[test]
    fn incandescence_below_threshold_returns_base() {
        let base = [0.5, 0.5, 0.5, 1.0];
        let result = incandescence_color(base, 700.0);
        assert_eq!(result, base, "Below 800 K, base color should be unchanged");
    }

    #[test]
    fn incandescence_above_threshold_shifts_toward_red() {
        let base = [0.5, 0.5, 0.5, 1.0];
        let result = incandescence_color(base, 1000.0);
        // Red channel should be boosted toward incandescent warm tones
        assert!(
            result[0] > base[0],
            "At 1000 K, red channel should increase"
        );
        // Alpha stays the same
        assert!(
            (result[3] - base[3]).abs() < f32::EPSILON,
            "Alpha should be preserved"
        );
    }

    #[test]
    fn incandescence_high_temp_produces_hdr() {
        let base = [0.5, 0.5, 0.5, 1.0];
        let result = incandescence_color(base, 2000.0);
        // At 2000 K the emissive multiplier (T⁴ scaling) pushes channels > 1.0
        let max_channel = result[0].max(result[1]).max(result[2]);
        assert!(
            max_channel > 1.0,
            "At 2000 K, HDR emissive should push channels above 1.0, got {max_channel}"
        );
    }

    #[test]
    fn thermal_heatmap_cold_is_blue() {
        let color = thermal_heatmap_color(250.0);
        assert!(
            color[2] > color[0] && color[2] > color[1],
            "At 250 K, heatmap should be predominantly blue"
        );
    }

    #[test]
    fn thermal_heatmap_hot_is_red() {
        let color = thermal_heatmap_color(2000.0);
        assert!(
            color[0] > color[1] && color[0] > color[2],
            "At 2000 K, heatmap should be predominantly red"
        );
    }

    #[test]
    fn thermal_heatmap_alpha_is_opaque() {
        for temp in [250.0, 1000.0, 2000.0] {
            let color = thermal_heatmap_color(temp);
            assert!(
                (color[3] - 1.0).abs() < f32::EPSILON,
                "Heatmap alpha should always be 1.0"
            );
        }
    }

    #[test]
    fn mesh_result_has_correct_lod() {
        // Verify that generate_chunk_mesh at various LOD levels produces
        // meshes consistent with the direct LOD functions, and that the
        // lod_level field in ChunkMeshResult would carry through correctly.
        let mut chunk = Chunk::new_empty(ChunkCoord::new(0, 0, 0));
        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                for y in 0..CHUNK_SIZE / 2 {
                    chunk.set_material(x, y, z, MaterialId::STONE);
                }
            }
        }

        for level_val in [0u8, 1, 2] {
            let level = LodLevel(level_val);
            let step = lod_step(level);
            let mesh = generate_chunk_mesh(
                &chunk,
                None,
                step,
                &NeighborVoxels::default(),
                None,
                false,
                None,
                false,
            );
            let direct = if step <= 1 {
                generate_mesh(&chunk)
            } else {
                generate_mesh_lod(&chunk, step)
            };
            assert_eq!(
                mesh.vertex_count(),
                direct.vertex_count(),
                "LOD {} mesh vertex count should match direct generation",
                level_val,
            );

            // Simulate what ChunkMeshResult would store.
            let result = ChunkMeshResult {
                mesh,
                lod_level: level_val,
            };
            assert_eq!(result.lod_level, level_val);
        }
    }

    #[test]
    fn neighbor_voxels_eliminates_boundary_seam() {
        // A fully solid chunk meshed WITHOUT neighbors produces boundary faces
        // (false surfaces where solid meets out-of-bounds air). With a solid
        // neighbor on the +X face, the boundary there should merge seamlessly.
        let coord = ChunkCoord::new(0, 0, 0);
        let chunk = Chunk::new_filled(coord, MaterialId::STONE);

        // Mesh with no neighbors → boundary surfaces on all 6 faces.
        let mesh_no_neighbors =
            generate_mesh_with_colors(&chunk, &NeighborVoxels::default(), None, false, false);

        // Provide a solid +X neighbor snapshot.
        let neighbor_chunk = Chunk::new_filled(ChunkCoord::new(1, 0, 0), MaterialId::STONE);
        let neighbors = NeighborVoxels {
            px: Some(neighbor_chunk.flat_snapshot()),
            ..Default::default()
        };
        let mesh_with_px = generate_mesh_with_colors(&chunk, &neighbors, None, false, false);

        // With a solid +X neighbor, the +X boundary surface is eliminated (solid
        // meets solid → no surface). So fewer vertices/triangles overall.
        assert!(
            mesh_with_px.vertex_count() < mesh_no_neighbors.vertex_count(),
            "Solid neighbor on +X should eliminate boundary vertices: {} < {}",
            mesh_with_px.vertex_count(),
            mesh_no_neighbors.vertex_count(),
        );
    }

    #[test]
    fn neighbor_voxels_sample_crosses_boundary() {
        let chunk = Chunk::new_filled(ChunkCoord::new(0, 0, 0), MaterialId::STONE);

        // Without neighbor, out-of-bounds returns AIR (solidity 0).
        let empty_neighbors = NeighborVoxels::default();
        let (solidity, mat, _temp) = empty_neighbors.sample(&chunk, CHUNK_SIZE as i32, 0, 0);
        assert_eq!(solidity, 0.0, "No +X neighbor → AIR");
        assert_eq!(mat, MaterialId::AIR);

        // With a stone +X neighbor, out-of-bounds returns STONE (solidity 1).
        let neighbor = Chunk::new_filled(ChunkCoord::new(1, 0, 0), MaterialId::STONE);
        let full_neighbors = NeighborVoxels {
            px: Some(neighbor.flat_snapshot()),
            ..Default::default()
        };
        let (solidity, mat, _temp) = full_neighbors.sample(&chunk, CHUNK_SIZE as i32, 0, 0);
        assert_eq!(solidity, 1.0, "+X neighbor stone → solid");
        assert_eq!(mat, MaterialId::STONE);
    }

    #[test]
    fn smooth_density_shifts_vertex_positions() {
        use crate::world::terrain::terrain_density;

        // Create a chunk with a terrain surface at height y≈16.7.
        // Voxels at y≤16 are solid (density > 0.5), y≥17 are air (density < 0.5).
        let surface_height = 16.7;
        let coord = ChunkCoord::new(0, 0, 0);

        // Binary version: density = 0.0 or 1.0
        let mut binary_chunk = Chunk::new_empty(coord);
        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                for y in 0..CHUNK_SIZE {
                    if (y as f64) <= surface_height {
                        binary_chunk.set_material(x, y, z, MaterialId::STONE);
                        // set_material sets density=1.0 for solid, 0.0 for air
                    }
                }
            }
        }

        // Smooth version: density encodes fractional surface position
        let mut smooth_chunk = Chunk::new_empty(coord);
        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                for y in 0..CHUNK_SIZE {
                    let depth = surface_height - y as f64;
                    if (y as f64) <= surface_height {
                        smooth_chunk.set_material(x, y, z, MaterialId::STONE);
                    }
                    smooth_chunk.get_mut(x, y, z).density = terrain_density(depth);
                }
            }
        }

        let binary_mesh = generate_mesh(&binary_chunk);
        let smooth_mesh = generate_mesh(&smooth_chunk);

        // Both should produce valid meshes
        assert!(binary_mesh.vertex_count() > 0);
        assert!(smooth_mesh.vertex_count() > 0);

        // With smooth density, Surface Nets vertex positions should differ
        // from the binary mesh (shifted toward the actual surface at y=16.7).
        // The Y coordinate of surface vertices should average closer to 16.7
        // in the smooth mesh vs ~16.5 in the binary mesh.
        let binary_avg_y: f32 = binary_mesh.positions.iter().map(|p| p[1]).sum::<f32>()
            / binary_mesh.vertex_count() as f32;
        let smooth_avg_y: f32 = smooth_mesh.positions.iter().map(|p| p[1]).sum::<f32>()
            / smooth_mesh.vertex_count() as f32;

        let target_y = surface_height as f32;
        assert!(
            (smooth_avg_y - target_y).abs() < (binary_avg_y - target_y).abs(),
            "Smooth density avg Y ({smooth_avg_y}) should be closer to surface \
             ({target_y}) than binary avg Y ({binary_avg_y})"
        );
    }

    #[test]
    fn lod_mesh_spans_full_chunk_size() {
        // A chunk with a horizontal stone surface at y=16 so the mesh has
        // geometry spanning a large portion of the X and Z axes.
        let coord = ChunkCoord::new(0, 0, 0);
        let mut chunk = Chunk::new_empty(coord);
        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                for y in 0..=16 {
                    chunk.set_material(x, y, z, MaterialId::STONE);
                }
            }
        }

        for &stride in &[1, 2, 4] {
            let mesh = generate_mesh_lod(&chunk, stride);
            assert!(
                mesh.vertex_count() > 0,
                "stride {stride}: mesh should have vertices"
            );

            let max_x = mesh
                .positions
                .iter()
                .map(|p| p[0])
                .fold(f32::NEG_INFINITY, f32::max);
            let max_z = mesh
                .positions
                .iter()
                .map(|p| p[2])
                .fold(f32::NEG_INFINITY, f32::max);

            let chunk_f = CHUNK_SIZE as f32;
            let tolerance = stride as f32;
            assert!(
                max_x >= chunk_f - tolerance,
                "stride {stride}: max X ({max_x}) should be near {chunk_f} (tolerance {tolerance})"
            );
            assert!(
                max_z >= chunk_f - tolerance,
                "stride {stride}: max Z ({max_z}) should be near {chunk_f} (tolerance {tolerance})"
            );

            // Must NOT be stuck in the reduced cell-space range.
            let effective = (CHUNK_SIZE / stride) as f32;
            if stride > 1 {
                assert!(
                    max_x > effective + 1.0,
                    "stride {stride}: max X ({max_x}) should exceed effective_size ({effective})"
                );
                assert!(
                    max_z > effective + 1.0,
                    "stride {stride}: max Z ({max_z}) should exceed effective_size ({effective})"
                );
            }
        }
    }
}
