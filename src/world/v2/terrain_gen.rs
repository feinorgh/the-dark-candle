// V2 terrain generation in local tangent space.
//
// For each chunk identified by CubeSphereCoord:
// 1. Compute the chunk's world-space frame (center + rotation)
// 2. For each voxel (lx, ly, lz), transform to world position
// 3. Convert world position to (lat, lon, radius)
// 4. Sample terrain noise to get surface radius
// 5. Determine material based on depth below/above surface

use bevy::math::DVec3;
use bevy::prelude::*;

use crate::world::chunk::{CHUNK_SIZE, CHUNK_VOLUME};
use crate::world::terrain::{SphericalTerrainGenerator, terrain_density};
use crate::world::v2::cubed_sphere::CubeSphereCoord;
use crate::world::voxel::{MaterialId, Voxel};

/// Result of V2 terrain generation for a single chunk.
pub struct V2ChunkData {
    pub coord: CubeSphereCoord,
    pub voxels: Vec<Voxel>,
}

/// Generate voxel data for a V2 chunk in local tangent space.
///
/// The chunk's local frame has Y = radial up. Voxels at local (lx, ly, lz)
/// are mapped to world positions via the chunk's rotation and translation.
/// The surface noise is sampled once per (lx, lz) column and reused for
/// all ly values, matching the v1 column-caching optimization.
#[allow(clippy::needless_range_loop)]
pub fn generate_v2_chunk(
    coord: CubeSphereCoord,
    mean_radius: f64,
    face_chunks_per_edge: f64,
    tgen: &SphericalTerrainGenerator,
) -> V2ChunkData {
    let cs = CHUNK_SIZE;
    let half = cs as f32 / 2.0;
    let (center, rotation) = coord.world_transform(mean_radius, face_chunks_per_edge);

    // Surface radius cache: sample once per (lx, lz) column at the vertical
    // midpoint (ly = CHUNK_SIZE/2). The angular span of a 32m chunk at 32km
    // radius is ~0.001 rad — negligible error from reusing the column value.
    let mut surface_cache = [[0.0_f64; CHUNK_SIZE]; CHUNK_SIZE];
    for lz in 0..cs {
        for lx in 0..cs {
            let local = Vec3::new(lx as f32 + 0.5 - half, 0.0, lz as f32 + 0.5 - half);
            let world = center + rotation * local;
            let wpos = DVec3::new(world.x as f64, world.y as f64, world.z as f64);
            let (lat, lon) = tgen.planet().lat_lon(wpos);
            surface_cache[lz][lx] = tgen.sample_surface_radius(lat, lon);
        }
    }

    // Early exit: check if entire chunk is above or below all surfaces.
    let base_r = mean_radius + coord.layer as f64 * cs as f64;
    let top_r = base_r + cs as f64;
    let half_diag_tangent = (2.0_f64).sqrt() * cs as f64 / 2.0;

    let mut min_surface = f64::MAX;
    let mut max_surface = f64::MIN;
    for row in &surface_cache {
        for &sr in row {
            min_surface = min_surface.min(sr);
            max_surface = max_surface.max(sr);
        }
    }
    let sea = tgen.planet().sea_level_radius;

    let mut voxels = vec![Voxel::default(); CHUNK_VOLUME];

    // Entirely above terrain AND above sea → all air (default)
    if base_r - half_diag_tangent > max_surface && base_r - half_diag_tangent > sea {
        return V2ChunkData { coord, voxels };
    }

    // Entirely below terrain → solid stone fill
    let cave_floor = min_surface - 200.0;
    if top_r + half_diag_tangent < min_surface && top_r + half_diag_tangent < cave_floor {
        for v in &mut voxels {
            v.material = MaterialId::STONE;
            v.density = 1.0;
        }
        return V2ChunkData { coord, voxels };
    }

    // Per-voxel fill
    for lz in 0..cs {
        for lx in 0..cs {
            let surface_r = surface_cache[lz][lx];
            for ly in 0..cs {
                let local = Vec3::new(
                    lx as f32 + 0.5 - half,
                    ly as f32 + 0.5 - half,
                    lz as f32 + 0.5 - half,
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

    V2ChunkData { coord, voxels }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::planet::PlanetConfig;
    use crate::world::v2::cubed_sphere::CubeFace;

    fn small_planet_gen() -> (SphericalTerrainGenerator, PlanetConfig) {
        let cfg = PlanetConfig {
            mean_radius: 200.0,
            sea_level_radius: 190.0,
            height_scale: 0.0,
            soil_depth: 2.0,
            cave_threshold: -999.0,
            ..Default::default()
        };
        let tgen = SphericalTerrainGenerator::new(cfg.clone());
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
        };
        let data = generate_v2_chunk(coord, cfg.mean_radius, fce, &tgen);
        assert_eq!(data.voxels.len(), CHUNK_VOLUME);
    }
}
