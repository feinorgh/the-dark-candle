// Terrain generation: layered noise → voxel fill.
//
// Generates terrain by combining multiple noise octaves:
// 1. Continental heightmap (low-frequency Perlin) → base elevation
// 2. Mountain ridges (ridged multi-fractal) → peaks and valleys
// 3. Cave carving (3D Perlin) → underground tunnels
// 4. Ore placement (high-frequency 3D noise) → mineral deposits
//
// The generator is deterministic given a seed. Each chunk is filled independently
// based on its world-space coordinates, enabling parallel generation.

#![allow(dead_code)]

use noise::{NoiseFn, Perlin};
use serde::{Deserialize, Serialize};

use super::chunk::{CHUNK_SIZE, Chunk};
use super::voxel::MaterialId;

/// Configuration for terrain generation, stored as a Bevy resource.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TerrainConfig {
    pub seed: u32,
    /// Sea level in voxel Y coordinates.
    pub sea_level: i32,
    /// Controls how high terrain can rise above sea level.
    pub height_scale: f64,
    /// Base frequency for the continental heightmap.
    pub continent_freq: f64,
    /// Frequency for mountain/detail noise.
    pub detail_freq: f64,
    /// 3D cave noise frequency.
    pub cave_freq: f64,
    /// Threshold below which caves are carved (0.0–1.0).
    pub cave_threshold: f64,
    /// Depth of dirt/grass layer above stone.
    pub soil_depth: i32,
}

impl Default for TerrainConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            sea_level: 64,
            height_scale: 32.0,
            continent_freq: 0.005,
            detail_freq: 0.02,
            cave_freq: 0.03,
            cave_threshold: -0.3,
            soil_depth: 4,
        }
    }
}

/// Stateless terrain generator. Holds noise functions seeded from config.
pub struct TerrainGenerator {
    config: TerrainConfig,
    continent_noise: Perlin,
    detail_noise: Perlin,
    cave_noise: Perlin,
}

impl TerrainGenerator {
    pub fn new(config: TerrainConfig) -> Self {
        let continent_noise = Perlin::new(config.seed);
        let detail_noise = Perlin::new(config.seed.wrapping_add(1));
        let cave_noise = Perlin::new(config.seed.wrapping_add(2));
        Self {
            config,
            continent_noise,
            detail_noise,
            cave_noise,
        }
    }

    pub fn config(&self) -> &TerrainConfig {
        &self.config
    }

    /// Sample the terrain height at a world XZ position.
    /// Returns a float height in voxel units.
    pub fn sample_height(&self, world_x: f64, world_z: f64) -> f64 {
        let cx = world_x * self.config.continent_freq;
        let cz = world_z * self.config.continent_freq;
        let continent = self.continent_noise.get([cx, cz]);

        let dx = world_x * self.config.detail_freq;
        let dz = world_z * self.config.detail_freq;
        let detail = self.detail_noise.get([dx, dz]);

        // Combine: continent provides broad shape, detail adds mountains/hills
        let combined = continent * 0.7 + detail * 0.3;
        self.config.sea_level as f64 + combined * self.config.height_scale
    }

    /// Check if a world position should be carved as a cave.
    pub fn is_cave(&self, world_x: f64, world_y: f64, world_z: f64) -> bool {
        let nx = world_x * self.config.cave_freq;
        let ny = world_y * self.config.cave_freq;
        let nz = world_z * self.config.cave_freq;
        self.cave_noise.get([nx, ny, nz]) < self.config.cave_threshold
    }

    /// Fill a chunk with terrain based on its world position.
    pub fn generate_chunk(&self, chunk: &mut Chunk) {
        let origin = chunk.coord.world_origin();

        for lz in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                let world_x = (origin.x + lx as i32) as f64;
                let world_z = (origin.z + lz as i32) as f64;
                let height = self.sample_height(world_x, world_z);

                for ly in 0..CHUNK_SIZE {
                    let world_y = origin.y + ly as i32;
                    let wy_f64 = world_y as f64;

                    let material = if wy_f64 > height {
                        // Above terrain surface
                        if world_y < self.config.sea_level {
                            MaterialId::WATER
                        } else {
                            MaterialId::AIR
                        }
                    } else if wy_f64 > height - 1.0 {
                        // Top layer: grass (if above water)
                        if world_y >= self.config.sea_level {
                            MaterialId(4) // grass
                        } else {
                            MaterialId::DIRT
                        }
                    } else if wy_f64 > height - self.config.soil_depth as f64 {
                        // Soil layers
                        MaterialId::DIRT
                    } else {
                        // Deep underground: stone
                        MaterialId::STONE
                    };

                    // Cave carving (only underground)
                    if material != MaterialId::AIR
                        && material != MaterialId::WATER
                        && wy_f64 < height - 2.0
                        && self.is_cave(world_x, wy_f64, world_z)
                    {
                        chunk.set_material(lx, ly, lz, MaterialId::AIR);
                    } else {
                        chunk.set_material(lx, ly, lz, material);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::chunk::ChunkCoord;
    use super::*;

    fn default_generator() -> TerrainGenerator {
        TerrainGenerator::new(TerrainConfig::default())
    }

    #[test]
    fn terrain_config_defaults_are_sensible() {
        let cfg = TerrainConfig::default();
        assert_eq!(cfg.sea_level, 64);
        assert!(cfg.height_scale > 0.0);
        assert!(cfg.continent_freq > 0.0);
        assert!(cfg.cave_threshold < 0.0);
        assert!(cfg.soil_depth > 0);
    }

    #[test]
    fn sample_height_returns_near_sea_level() {
        let generator = default_generator();
        // At any point, height should be within sea_level ± height_scale
        let h = generator.sample_height(100.0, 200.0);
        let sea = generator.config.sea_level as f64;
        let scale = generator.config.height_scale;
        assert!(
            h > sea - scale * 1.5 && h < sea + scale * 1.5,
            "Height {} outside expected range [{}, {}]",
            h,
            sea - scale * 1.5,
            sea + scale * 1.5
        );
    }

    #[test]
    fn sample_height_is_deterministic() {
        let generator = default_generator();
        let h1 = generator.sample_height(42.0, 99.0);
        let h2 = generator.sample_height(42.0, 99.0);
        assert_eq!(h1, h2);
    }

    #[test]
    fn different_seeds_produce_different_terrain() {
        let gen_a = TerrainGenerator::new(TerrainConfig {
            seed: 1,
            ..Default::default()
        });
        let gen_b = TerrainGenerator::new(TerrainConfig {
            seed: 9999,
            ..Default::default()
        });
        let h_a = gen_a.sample_height(50.0, 50.0);
        let h_b = gen_b.sample_height(50.0, 50.0);
        assert!(
            (h_a - h_b).abs() > 0.001,
            "Different seeds should produce different heights"
        );
    }

    #[test]
    fn height_varies_across_space() {
        let generator = default_generator();
        let mut heights = Vec::new();
        for x in (0..500).step_by(50) {
            heights.push(generator.sample_height(x as f64, 0.0));
        }
        // Not all heights should be the same
        let min = heights.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = heights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            max - min > 1.0,
            "Height variation too small: min={min}, max={max}"
        );
    }

    #[test]
    fn generate_chunk_at_surface_has_mixed_materials() {
        let generator = default_generator();
        // Chunk at Y=2 → voxels 64..95, straddles sea level (64) and terrain surface
        let coord = ChunkCoord::new(0, 2, 0);
        let mut chunk = Chunk::new_empty(coord);
        generator.generate_chunk(&mut chunk);

        let mut air_count = 0usize;
        let mut solid_count = 0usize;
        for v in chunk.voxels() {
            if v.is_air() {
                air_count += 1;
            } else {
                solid_count += 1;
            }
        }
        // Surface chunk should have both air and solid voxels
        assert!(air_count > 0, "Surface chunk has no air");
        assert!(solid_count > 0, "Surface chunk has no solids");
    }

    #[test]
    fn generate_chunk_deep_underground_is_mostly_stone() {
        let generator = default_generator();
        // Y=-3 → voxels -96..-65, well below sea level
        let coord = ChunkCoord::new(0, -3, 0);
        let mut chunk = Chunk::new_empty(coord);
        generator.generate_chunk(&mut chunk);

        let stone_count = chunk
            .voxels()
            .iter()
            .filter(|v| v.material == MaterialId::STONE)
            .count();
        let total = chunk.voxels().len();
        let ratio = stone_count as f64 / total as f64;
        // Deep underground should be mostly stone (allowing some caves)
        assert!(
            ratio > 0.5,
            "Deep chunk is only {:.1}% stone, expected >50%",
            ratio * 100.0
        );
    }

    #[test]
    fn generate_chunk_high_up_is_all_air() {
        let generator = default_generator();
        // Y=10 → voxels 320..351, well above any terrain
        let coord = ChunkCoord::new(0, 10, 0);
        let mut chunk = Chunk::new_empty(coord);
        generator.generate_chunk(&mut chunk);

        assert!(
            chunk.is_empty(),
            "High-altitude chunk should be all air, but has {} solid voxels",
            chunk.solid_count()
        );
    }

    #[test]
    fn cave_carving_creates_air_underground() {
        let generator = TerrainGenerator::new(TerrainConfig {
            cave_threshold: 0.5, // Very aggressive cave carving
            ..Default::default()
        });
        // Underground chunk
        let coord = ChunkCoord::new(0, -2, 0);
        let mut chunk = Chunk::new_empty(coord);
        generator.generate_chunk(&mut chunk);

        let air_count = chunk.voxels().iter().filter(|v| v.is_air()).count();
        assert!(
            air_count > 0,
            "Aggressive cave threshold should carve some air underground"
        );
    }

    #[test]
    fn generate_chunk_marks_dirty() {
        let generator = default_generator();
        let mut chunk = Chunk::new_empty(ChunkCoord::new(0, 0, 0));
        chunk.clear_dirty();
        generator.generate_chunk(&mut chunk);
        assert!(chunk.is_dirty());
    }
}
