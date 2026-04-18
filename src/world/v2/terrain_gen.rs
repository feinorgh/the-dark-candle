// V2 terrain generation in local tangent space.
//
// For each chunk identified by CubeSphereCoord:
// 1. Compute the chunk's world-space frame (center + rotation)
// 2. For each voxel (lx, ly, lz), transform to world position
// 3. Convert world position to (lat, lon, radius)
// 4. Sample terrain noise to get surface radius
// 5. Determine material based on depth below/above surface

use std::sync::Arc;

use bevy::math::DVec3;
use bevy::prelude::*;

use crate::world::chunk::{CHUNK_SIZE, CHUNK_VOLUME};
use crate::world::terrain::{UnifiedTerrainGenerator, terrain_density};
use crate::world::v2::cubed_sphere::CubeSphereCoord;
use crate::world::v2::greedy_mesh::NeighborSlices;
use crate::world::voxel::{MaterialId, Voxel};

/// Result of V2 terrain generation for a single chunk (legacy single-stage path).
pub struct V2ChunkData {
    pub coord: CubeSphereCoord,
    pub voxels: Vec<Voxel>,
    /// Boundary slices from the 6 neighbor directions for seamless meshing.
    pub neighbor_slices: NeighborSlices,
}

// ── Two-stage pipeline types ──────────────────────────────────────────────

/// Cached voxel data with memory-efficient variants for trivial chunks.
#[derive(Clone)]
pub enum CachedVoxels {
    /// Chunk is entirely air (above surface and sea level).
    AllAir,
    /// Chunk is entirely a single solid material (deep underground).
    AllSolid(MaterialId),
    /// Chunk has mixed content — full voxel array stored behind an Arc.
    Mixed(Arc<Vec<Voxel>>),
}

impl CachedVoxels {
    /// Approximate heap allocation in bytes.
    pub fn byte_size(&self) -> usize {
        match self {
            CachedVoxels::AllAir | CachedVoxels::AllSolid(_) => 0,
            CachedVoxels::Mixed(v) => v.len() * std::mem::size_of::<Voxel>(),
        }
    }
}

/// Result of terrain-only generation (no boundary slices).
pub struct V2TerrainData {
    pub coord: CubeSphereCoord,
    pub voxels: CachedVoxels,
}

/// Internal classification of voxel generation result.
enum VoxelGenResult {
    AllAir,
    AllSolid(MaterialId),
    Mixed(Vec<Voxel>),
}

// ── Core voxel generation (shared by single-stage and two-stage paths) ────

/// Internal: generate voxels and classify the result.
#[allow(clippy::needless_range_loop)]
fn generate_voxels_core(
    coord: CubeSphereCoord,
    mean_radius: f64,
    face_chunks_per_edge: f64,
    tgen: &UnifiedTerrainGenerator,
) -> VoxelGenResult {
    let cs = CHUNK_SIZE;
    let half = cs as f32 / 2.0;
    let (center, rotation, tangent_scale) =
        coord.world_transform_scaled(mean_radius, face_chunks_per_edge);

    // Surface radius cache: sample once per (lx, lz) column.
    let mut surface_cache = [[0.0_f64; CHUNK_SIZE]; CHUNK_SIZE];
    for lz in 0..cs {
        for lx in 0..cs {
            let local = Vec3::new(
                (lx as f32 + 0.5 - half) * tangent_scale.x,
                0.0,
                (lz as f32 + 0.5 - half) * tangent_scale.z,
            );
            let world = center + rotation * local;
            let wpos = DVec3::new(world.x as f64, world.y as f64, world.z as f64);
            let (lat, lon) = tgen.planet_config().lat_lon(wpos);
            surface_cache[lz][lx] = tgen.sample_surface_radius_at(lat, lon);
        }
    }

    let lod_scale = (1u64 << coord.lod) as f64;
    let base_r = mean_radius + coord.layer as f64 * cs as f64 * lod_scale;
    let top_r = base_r + cs as f64 * lod_scale;
    let half_diag_tangent = ((tangent_scale.x as f64).powi(2)
        + (tangent_scale.y as f64).powi(2)
        + (tangent_scale.z as f64).powi(2))
    .sqrt()
        * cs as f64
        / 2.0;

    let mut min_surface = f64::MAX;
    let mut max_surface = f64::MIN;
    for row in &surface_cache {
        for &sr in row {
            min_surface = min_surface.min(sr);
            max_surface = max_surface.max(sr);
        }
    }
    let sea = tgen.planet_config().sea_level_radius;

    // Entirely above terrain AND above sea → all air
    if base_r - half_diag_tangent > max_surface && base_r - half_diag_tangent > sea {
        return VoxelGenResult::AllAir;
    }

    // Entirely below terrain → solid stone
    let cave_floor = min_surface - 200.0;
    if top_r + half_diag_tangent < min_surface && top_r + half_diag_tangent < cave_floor {
        return VoxelGenResult::AllSolid(MaterialId::STONE);
    }

    // Per-voxel fill
    let mut voxels = vec![Voxel::default(); CHUNK_VOLUME];
    for lz in 0..cs {
        for lx in 0..cs {
            let surface_r = surface_cache[lz][lx];
            for ly in 0..cs {
                let local = Vec3::new(
                    (lx as f32 + 0.5 - half) * tangent_scale.x,
                    (ly as f32 + 0.5 - half) * tangent_scale.y,
                    (lz as f32 + 0.5 - half) * tangent_scale.z,
                );
                let world = center + rotation * local;
                let wpos = DVec3::new(world.x as f64, world.y as f64, world.z as f64);
                let r = wpos.length();

                let material = tgen.material_at_radius(r, surface_r, wpos.x, wpos.y, wpos.z);

                let idx = lz * cs * cs + ly * cs + lx;
                voxels[idx].material = material;

                let density = if material == MaterialId::WATER {
                    terrain_density(sea - r)
                } else {
                    terrain_density(surface_r - r)
                };
                voxels[idx].density = density;
            }
        }
    }

    VoxelGenResult::Mixed(voxels)
}

// ── Single-stage path (legacy, used by tests) ────────────────────────────

/// Generate voxel data for a V2 chunk in local tangent space (single-stage).
///
/// Returns voxels and pre-computed boundary slices for seamless meshing.
pub fn generate_v2_chunk(
    coord: CubeSphereCoord,
    mean_radius: f64,
    face_chunks_per_edge: f64,
    tgen: &UnifiedTerrainGenerator,
) -> V2ChunkData {
    match generate_voxels_core(coord, mean_radius, face_chunks_per_edge, tgen) {
        VoxelGenResult::AllAir => V2ChunkData {
            coord,
            voxels: vec![Voxel::default(); CHUNK_VOLUME],
            neighbor_slices: NeighborSlices::empty(),
        },
        VoxelGenResult::AllSolid(mat) => {
            let mut voxels = vec![Voxel::default(); CHUNK_VOLUME];
            for v in &mut voxels {
                v.material = mat;
                v.density = 1.0;
            }
            V2ChunkData {
                coord,
                voxels,
                neighbor_slices: generate_boundary_slices(
                    coord,
                    mean_radius,
                    face_chunks_per_edge,
                    tgen,
                ),
            }
        }
        VoxelGenResult::Mixed(voxels) => V2ChunkData {
            coord,
            voxels,
            neighbor_slices: generate_boundary_slices(
                coord,
                mean_radius,
                face_chunks_per_edge,
                tgen,
            ),
        },
    }
}

// ── Two-stage path ───────────────────────────────────────────────────────

/// Generate voxels only (no boundary slices) for the two-stage pipeline.
///
/// Stage 1 of the pipeline: produces voxels that are cached for later
/// boundary extraction by neighboring chunks.
pub fn generate_v2_voxels(
    coord: CubeSphereCoord,
    mean_radius: f64,
    face_chunks_per_edge: f64,
    tgen: &UnifiedTerrainGenerator,
) -> V2TerrainData {
    let cached = match generate_voxels_core(coord, mean_radius, face_chunks_per_edge, tgen) {
        VoxelGenResult::AllAir => CachedVoxels::AllAir,
        VoxelGenResult::AllSolid(mat) => CachedVoxels::AllSolid(mat),
        VoxelGenResult::Mixed(voxels) => CachedVoxels::Mixed(Arc::new(voxels)),
    };
    V2TerrainData { coord, voxels: cached }
}

/// Convert `CachedVoxels` to a flat `Vec<Voxel>` for the greedy mesher.
pub fn cached_voxels_to_vec(cached: &CachedVoxels) -> Vec<Voxel> {
    match cached {
        CachedVoxels::AllAir => vec![Voxel::default(); CHUNK_VOLUME],
        CachedVoxels::AllSolid(mat) => {
            let mut v = vec![Voxel::default(); CHUNK_VOLUME];
            for vx in &mut v {
                vx.material = *mat;
                vx.density = 1.0;
            }
            v
        }
        CachedVoxels::Mixed(v) => v.as_ref().clone(),
    }
}

/// Extract a boundary slice from cached voxels for the greedy mesher.
///
/// `dir` is the mesh direction index: `[+X, -X, +Y, -Y, +Z, -Z]`.
/// Returns the appropriate face layer that the adjacent chunk needs:
/// - dir=0 (+X): x=0 layer (face closest to the requesting chunk's +X edge)
/// - dir=1 (-X): x=CS-1 layer
/// - dir=2 (+Y): y=0 layer
/// - dir=3 (-Y): y=CS-1 layer
/// - dir=4 (+Z): z=0 layer
/// - dir=5 (-Z): z=CS-1 layer
///
/// Slice layout matches `sample_material` expectations in `greedy_mesh.rs`.
pub fn extract_edge_slice(cached: &CachedVoxels, dir: usize) -> Vec<Voxel> {
    let cs = CHUNK_SIZE;
    let slice_size = cs * cs;

    match cached {
        CachedVoxels::AllAir => vec![Voxel::default(); slice_size],
        CachedVoxels::AllSolid(mat) => {
            let mut v = Voxel::default();
            v.material = *mat;
            v.density = 1.0;
            vec![v; slice_size]
        }
        CachedVoxels::Mixed(data) => {
            let mut slice = vec![Voxel::default(); slice_size];
            match dir {
                0 => {
                    // +X: x=0 layer, indexed as slice[y * CS + z]
                    for y in 0..cs {
                        for z in 0..cs {
                            slice[y * cs + z] = data[z * cs * cs + y * cs];
                        }
                    }
                }
                1 => {
                    // -X: x=CS-1 layer, indexed as slice[y * CS + z]
                    for y in 0..cs {
                        for z in 0..cs {
                            slice[y * cs + z] = data[z * cs * cs + y * cs + (cs - 1)];
                        }
                    }
                }
                2 => {
                    // +Y: y=0 layer, indexed as slice[x * CS + z]
                    for x in 0..cs {
                        for z in 0..cs {
                            slice[x * cs + z] = data[z * cs * cs + x];
                        }
                    }
                }
                3 => {
                    // -Y: y=CS-1 layer, indexed as slice[x * CS + z]
                    for x in 0..cs {
                        for z in 0..cs {
                            slice[x * cs + z] = data[z * cs * cs + (cs - 1) * cs + x];
                        }
                    }
                }
                4 => {
                    // +Z: z=0 layer, indexed as slice[x * CS + y]
                    for x in 0..cs {
                        for y in 0..cs {
                            slice[x * cs + y] = data[y * cs + x];
                        }
                    }
                }
                5 => {
                    // -Z: z=CS-1 layer, indexed as slice[x * CS + y]
                    for x in 0..cs {
                        for y in 0..cs {
                            slice[x * cs + y] = data[(cs - 1) * cs * cs + y * cs + x];
                        }
                    }
                }
                _ => {}
            }
            slice
        }
    }
}

/// Generate boundary slices for the 6 face directions.
///
/// For each direction, samples terrain one voxel past the chunk boundary in
/// the chunk's OWN local frame. This avoids any axis-mapping issues between
/// local [X, Y, Z] and cubed-sphere [U, V, layer] coordinates.
///
/// Layout matches greedy_mesh expectations: `[+X, -X, +Y, -Y, +Z, -Z]`.
fn generate_boundary_slices(
    coord: CubeSphereCoord,
    mean_radius: f64,
    face_chunks_per_edge: f64,
    tgen: &UnifiedTerrainGenerator,
) -> NeighborSlices {
    let mut slices: [Option<Vec<Voxel>>; 6] = [const { None }; 6];
    for dir in 0..6usize {
        slices[dir] = Some(generate_single_boundary_slice(
            coord,
            dir,
            mean_radius,
            face_chunks_per_edge,
            tgen,
        ));
    }
    NeighborSlices { slices }
}

/// Generate a single boundary slice for one direction via terrain resampling.
///
/// `dir` is the mesh direction index: `[+X, -X, +Y, -Y, +Z, -Z]`.
/// Samples terrain one voxel past the chunk boundary in the chunk's local
/// frame to determine what material is adjacent.
pub fn generate_single_boundary_slice(
    coord: CubeSphereCoord,
    dir: usize,
    mean_radius: f64,
    face_chunks_per_edge: f64,
    tgen: &UnifiedTerrainGenerator,
) -> Vec<Voxel> {
    let cs = CHUNK_SIZE;
    let half = cs as f32 / 2.0;
    let (center, rotation, tangent_scale) =
        coord.world_transform_scaled(mean_radius, face_chunks_per_edge);
    let sea = tgen.planet_config().sea_level_radius;
    let slice_size = cs * cs;

    let mut slice = vec![Voxel::default(); slice_size];

    for a in 0..cs {
        for b in 0..cs {
            // Build local position one voxel past the chunk boundary.
            // All axes (X, Y, Z) are scaled by tangent_scale.
            let local = match dir {
                0 => Vec3::new(
                    (cs as f32 + 0.5 - half) * tangent_scale.x,
                    (a as f32 + 0.5 - half) * tangent_scale.y,
                    (b as f32 + 0.5 - half) * tangent_scale.z,
                ), // +X
                1 => Vec3::new(
                    (-1.0 + 0.5 - half) * tangent_scale.x,
                    (a as f32 + 0.5 - half) * tangent_scale.y,
                    (b as f32 + 0.5 - half) * tangent_scale.z,
                ), // -X
                2 => Vec3::new(
                    (a as f32 + 0.5 - half) * tangent_scale.x,
                    (cs as f32 + 0.5 - half) * tangent_scale.y,
                    (b as f32 + 0.5 - half) * tangent_scale.z,
                ), // +Y
                3 => Vec3::new(
                    (a as f32 + 0.5 - half) * tangent_scale.x,
                    (-1.0 + 0.5 - half) * tangent_scale.y,
                    (b as f32 + 0.5 - half) * tangent_scale.z,
                ), // -Y
                4 => Vec3::new(
                    (a as f32 + 0.5 - half) * tangent_scale.x,
                    (b as f32 + 0.5 - half) * tangent_scale.y,
                    (cs as f32 + 0.5 - half) * tangent_scale.z,
                ), // +Z
                _ => Vec3::new(
                    (a as f32 + 0.5 - half) * tangent_scale.x,
                    (b as f32 + 0.5 - half) * tangent_scale.y,
                    (-1.0 + 0.5 - half) * tangent_scale.z,
                ), // -Z
            };

            let world = center + rotation * local;
            let wpos = DVec3::new(world.x as f64, world.y as f64, world.z as f64);
            let r = wpos.length();
            let (lat, lon) = tgen.planet_config().lat_lon(wpos);
            let surface_r = tgen.sample_surface_radius_at(lat, lon);

            let material =
                tgen.material_at_radius(r, surface_r, wpos.x, wpos.y, wpos.z);
            let density = if material == MaterialId::WATER {
                terrain_density(sea - r)
            } else {
                terrain_density(surface_r - r)
            };

            let idx = a * cs + b;
            slice[idx].material = material;
            slice[idx].density = density;
        }
    }

    slice
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::planet::PlanetConfig;
    use crate::world::terrain::SphericalTerrainGenerator;
    use crate::world::v2::cubed_sphere::CubeFace;

    fn small_planet_gen() -> (UnifiedTerrainGenerator, PlanetConfig) {
        let cfg = PlanetConfig {
            mean_radius: 200.0,
            sea_level_radius: 190.0,
            height_scale: 0.0,
            soil_depth: 2.0,
            cave_threshold: -999.0,
            ..Default::default()
        };
        let tgen = UnifiedTerrainGenerator::Spherical(Box::new(SphericalTerrainGenerator::new(
            cfg.clone(),
        )));
        (tgen, cfg)
    }

    #[test]
    fn surface_layer_has_ground_material() {
        let (tgen, cfg) = small_planet_gen();
        let fce = CubeSphereCoord::face_chunks_per_edge(cfg.mean_radius);

        // Layer 0 at the surface: should have both air and solid voxels
        let coord = CubeSphereCoord {
            face: CubeFace::PosX,
            u: fce as i32 / 2,
            v: fce as i32 / 2,
            layer: 0,
            lod: 0,
        };
        let data = generate_v2_chunk(coord, cfg.mean_radius, fce, &tgen);

        let solid_count = data.voxels.iter().filter(|v| !v.material.is_air()).count();
        assert!(
            solid_count > 0,
            "Surface chunk should contain solid voxels, got all air"
        );
        let air_count = data.voxels.iter().filter(|v| v.material.is_air()).count();
        assert!(
            air_count > 0,
            "Surface chunk should contain air voxels, got all solid"
        );
    }

    #[test]
    fn deep_underground_is_all_solid() {
        let (tgen, cfg) = small_planet_gen();
        let fce = CubeSphereCoord::face_chunks_per_edge(cfg.mean_radius);

        // Layer -4: below surface (200 - 4*32 = 72m from center, way underground)
        let coord = CubeSphereCoord {
            face: CubeFace::PosZ,
            u: fce as i32 / 2,
            v: fce as i32 / 2,
            layer: -4,
            lod: 0,
        };
        let data = generate_v2_chunk(coord, cfg.mean_radius, fce, &tgen);

        let solid_count = data
            .voxels
            .iter()
            .filter(|v| !v.material.is_air() && v.material != MaterialId::WATER)
            .count();
        assert!(
            solid_count > CHUNK_VOLUME / 2,
            "Deep underground should be mostly solid, got {solid_count}/{CHUNK_VOLUME}"
        );
    }

    #[test]
    fn high_above_surface_is_all_air() {
        let (tgen, cfg) = small_planet_gen();
        let fce = CubeSphereCoord::face_chunks_per_edge(cfg.mean_radius);

        // Layer +3: well above surface (200 + 3*32 = 296m from center, above 200m surface)
        let coord = CubeSphereCoord {
            face: CubeFace::NegY,
            u: fce as i32 / 2,
            v: fce as i32 / 2,
            layer: 3,
            lod: 0,
        };
        let data = generate_v2_chunk(coord, cfg.mean_radius, fce, &tgen);

        let air_count = data.voxels.iter().filter(|v| v.material.is_air()).count();
        assert_eq!(
            air_count, CHUNK_VOLUME,
            "High-altitude chunk should be all air"
        );
    }

    #[test]
    fn voxel_count_is_chunk_volume() {
        let (tgen, cfg) = small_planet_gen();
        let fce = CubeSphereCoord::face_chunks_per_edge(cfg.mean_radius);

        let coord = CubeSphereCoord {
            face: CubeFace::PosY,
            u: 0,
            v: 0,
            layer: 0,
            lod: 0,
        };
        let data = generate_v2_chunk(coord, cfg.mean_radius, fce, &tgen);
        assert_eq!(data.voxels.len(), CHUNK_VOLUME);
    }

    // ── Two-stage pipeline tests ──────────────────────────────────────────

    #[test]
    fn generate_v2_voxels_all_air() {
        let (tgen, cfg) = small_planet_gen();
        let fce = CubeSphereCoord::face_chunks_per_edge(cfg.mean_radius);
        // High above surface
        let coord = CubeSphereCoord::new(CubeFace::NegY, fce as i32 / 2, fce as i32 / 2, 3);
        let data = generate_v2_voxels(coord, cfg.mean_radius, fce, &tgen);
        assert!(matches!(data.voxels, CachedVoxels::AllAir));
        assert_eq!(data.voxels.byte_size(), 0);
    }

    #[test]
    fn generate_v2_voxels_mixed_surface() {
        let (tgen, cfg) = small_planet_gen();
        let fce = CubeSphereCoord::face_chunks_per_edge(cfg.mean_radius);
        let coord = CubeSphereCoord::new(CubeFace::PosX, fce as i32 / 2, fce as i32 / 2, 0);
        let data = generate_v2_voxels(coord, cfg.mean_radius, fce, &tgen);
        assert!(matches!(data.voxels, CachedVoxels::Mixed(_)));
        assert!(data.voxels.byte_size() > 0);
    }

    #[test]
    fn cached_voxels_to_vec_roundtrip() {
        let (tgen, cfg) = small_planet_gen();
        let fce = CubeSphereCoord::face_chunks_per_edge(cfg.mean_radius);
        let coord = CubeSphereCoord::new(CubeFace::PosX, fce as i32 / 2, fce as i32 / 2, 0);

        // Generate via both paths and compare
        let chunk_data = generate_v2_chunk(coord, cfg.mean_radius, fce, &tgen);
        let terrain_data = generate_v2_voxels(coord, cfg.mean_radius, fce, &tgen);
        let reconstituted = cached_voxels_to_vec(&terrain_data.voxels);

        assert_eq!(chunk_data.voxels.len(), reconstituted.len());
        for (i, (a, b)) in chunk_data.voxels.iter().zip(reconstituted.iter()).enumerate() {
            assert_eq!(
                a.material, b.material,
                "Voxel {i} material mismatch: legacy={:?}, two-stage={:?}",
                a.material, b.material,
            );
        }
    }

    #[test]
    fn extract_edge_slice_all_air_is_all_air() {
        let cached = CachedVoxels::AllAir;
        for dir in 0..6 {
            let slice = extract_edge_slice(&cached, dir);
            assert_eq!(slice.len(), CHUNK_SIZE * CHUNK_SIZE);
            assert!(slice.iter().all(|v| v.material.is_air()));
        }
    }

    #[test]
    fn extract_edge_slice_all_solid() {
        let cached = CachedVoxels::AllSolid(MaterialId::STONE);
        for dir in 0..6 {
            let slice = extract_edge_slice(&cached, dir);
            assert_eq!(slice.len(), CHUNK_SIZE * CHUNK_SIZE);
            assert!(slice.iter().all(|v| v.material == MaterialId::STONE));
        }
    }

    #[test]
    fn extract_edge_slice_mixed_correct_layer() {
        // Create a voxel array with a known pattern: material = (x + y + z) as MaterialId
        let mut voxels = vec![Voxel::default(); CHUNK_VOLUME];
        let cs = CHUNK_SIZE;
        for z in 0..cs {
            for y in 0..cs {
                for x in 0..cs {
                    let idx = z * cs * cs + y * cs + x;
                    voxels[idx].material = MaterialId((x + y + z) as u16);
                }
            }
        }
        let cached = CachedVoxels::Mixed(Arc::new(voxels.clone()));

        // +X (dir=0): x=0 layer, indexed as slice[y * CS + z]
        let slice = extract_edge_slice(&cached, 0);
        for y in 0..cs {
            for z in 0..cs {
                let expected = MaterialId((0 + y + z) as u16);
                assert_eq!(
                    slice[y * cs + z].material, expected,
                    "+X slice at (y={y}, z={z})"
                );
            }
        }

        // -X (dir=1): x=CS-1 layer, indexed as slice[y * CS + z]
        let slice = extract_edge_slice(&cached, 1);
        for y in 0..cs {
            for z in 0..cs {
                let expected = MaterialId(((cs - 1) + y + z) as u16);
                assert_eq!(
                    slice[y * cs + z].material, expected,
                    "-X slice at (y={y}, z={z})"
                );
            }
        }

        // +Y (dir=2): y=0 layer, indexed as slice[x * CS + z]
        let slice = extract_edge_slice(&cached, 2);
        for x in 0..cs {
            for z in 0..cs {
                let expected = MaterialId((x + 0 + z) as u16);
                assert_eq!(
                    slice[x * cs + z].material, expected,
                    "+Y slice at (x={x}, z={z})"
                );
            }
        }

        // -Y (dir=3): y=CS-1 layer, indexed as slice[x * CS + z]
        let slice = extract_edge_slice(&cached, 3);
        for x in 0..cs {
            for z in 0..cs {
                let expected = MaterialId((x + (cs - 1) + z) as u16);
                assert_eq!(
                    slice[x * cs + z].material, expected,
                    "-Y slice at (x={x}, z={z})"
                );
            }
        }

        // +Z (dir=4): z=0 layer, indexed as slice[x * CS + y]
        let slice = extract_edge_slice(&cached, 4);
        for x in 0..cs {
            for y in 0..cs {
                let expected = MaterialId((x + y + 0) as u16);
                assert_eq!(
                    slice[x * cs + y].material, expected,
                    "+Z slice at (x={x}, y={y})"
                );
            }
        }

        // -Z (dir=5): z=CS-1 layer, indexed as slice[x * CS + y]
        let slice = extract_edge_slice(&cached, 5);
        for x in 0..cs {
            for y in 0..cs {
                let expected = MaterialId((x + y + (cs - 1)) as u16);
                assert_eq!(
                    slice[x * cs + y].material, expected,
                    "-Z slice at (x={x}, y={y})"
                );
            }
        }
    }
}
