// Terrain generation: layered noise → voxel fill.
//
// Two generation modes:
//
// **Flat** (legacy): 2D Perlin heightmap over the XZ plane with Y-axis as up.
//   Sea level at a fixed Y coordinate. Material layers: air → water → grass →
//   dirt → stone. Cave carving via 3D Perlin below the surface.
//
// **Spherical** (Phase 8): planet centered at origin with configurable radius.
//   Surface height from 2D noise sampled in spherical coordinates (lat, lon).
//   Material layers defined by radial depth bands (core → mantle → crust →
//   soil → air/water). Cave carving via 3D noise within the crust band.
//
// Both generators are deterministic given a seed. Each chunk is filled
// independently based on its world-space coordinates, enabling parallel
// generation.

#![allow(dead_code)]

use noise::{NoiseFn, Perlin};
use serde::{Deserialize, Serialize};

use super::chunk::{CHUNK_SIZE, Chunk};
use super::planet::{PlanetConfig, TerrainMode};
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

// ---------------------------------------------------------------------------
// Spherical terrain generator
// ---------------------------------------------------------------------------

/// Terrain generator for a spherical planet.
///
/// For each voxel, computes radial distance from planet center, derives
/// `(lat, lon)` via the `PlanetConfig`, samples surface radius from noise,
/// and assigns material by radial depth band.
pub struct SphericalTerrainGenerator {
    planet: PlanetConfig,
    continent_noise: Perlin,
    detail_noise: Perlin,
    cave_noise: Perlin,
}

impl SphericalTerrainGenerator {
    pub fn new(planet: PlanetConfig) -> Self {
        let continent_noise = Perlin::new(planet.seed);
        let detail_noise = Perlin::new(planet.seed.wrapping_add(1));
        let cave_noise = Perlin::new(planet.seed.wrapping_add(2));
        Self {
            planet,
            continent_noise,
            detail_noise,
            cave_noise,
        }
    }

    pub fn planet(&self) -> &PlanetConfig {
        &self.planet
    }

    /// Sample the terrain surface radius at a given `(lat, lon)`.
    ///
    /// Returns the radial distance from planet center to the terrain surface
    /// at that angular position.
    pub fn sample_surface_radius(&self, lat: f64, lon: f64) -> f64 {
        // Use lat/lon as 2D noise coordinates (avoids pole distortion because
        // the noise function is sampled on the sphere, not projected from a plane).
        let cx = lon * self.planet.continent_freq;
        let cz = lat * self.planet.continent_freq;
        let continent = self.continent_noise.get([cx, cz]);

        let dx = lon * self.planet.detail_freq;
        let dz = lat * self.planet.detail_freq;
        let detail = self.detail_noise.get([dx, dz]);

        let combined = continent * 0.7 + detail * 0.3;
        self.planet.surface_radius_at(lat, lon, combined)
    }

    /// Check if a world position should be carved as a cave.
    ///
    /// Uses 3D Perlin noise in Cartesian coordinates (works fine within chunks
    /// regardless of spherical projection).
    pub fn is_cave(&self, world_x: f64, world_y: f64, world_z: f64) -> bool {
        let nx = world_x * self.planet.cave_freq;
        let ny = world_y * self.planet.cave_freq;
        let nz = world_z * self.planet.cave_freq;
        self.cave_noise.get([nx, ny, nz]) < self.planet.cave_threshold
    }

    /// Fill a chunk with spherical terrain based on its world position.
    pub fn generate_chunk(&self, chunk: &mut Chunk) {
        let origin = chunk.coord.world_origin();

        for lz in 0..CHUNK_SIZE {
            for ly in 0..CHUNK_SIZE {
                for lx in 0..CHUNK_SIZE {
                    let world_x = (origin.x + lx as i32) as f64;
                    let world_y = (origin.y + ly as i32) as f64;
                    let world_z = (origin.z + lz as i32) as f64;

                    let pos = bevy::math::DVec3::new(world_x, world_y, world_z);
                    let r = self.planet.distance_from_center(pos);
                    let (lat, lon) = self.planet.lat_lon(pos);
                    let surface_r = self.sample_surface_radius(lat, lon);

                    let material = self.material_at_radius(r, surface_r, world_x, world_y, world_z);

                    chunk.set_material(lx, ly, lz, material);
                }
            }
        }
    }

    /// Determine material at a given radius relative to the terrain surface.
    fn material_at_radius(
        &self,
        r: f64,
        surface_r: f64,
        world_x: f64,
        world_y: f64,
        world_z: f64,
    ) -> MaterialId {
        if r > surface_r {
            // Above terrain surface
            if r < self.planet.sea_level_radius {
                return MaterialId::WATER;
            }
            return MaterialId::AIR;
        }

        // At or below surface
        let depth_below_surface = surface_r - r;

        if depth_below_surface < 1.0 {
            // Top layer: grass (if above sea level)
            if surface_r >= self.planet.sea_level_radius {
                return MaterialId(4); // grass
            }
            return MaterialId::DIRT;
        }

        if depth_below_surface < self.planet.soil_depth {
            return MaterialId::DIRT;
        }

        // Deep: assign by geological layer
        let mat = self
            .planet
            .layer_at_radius(r)
            .map(|l| material_from_layer_name(&l.material))
            .unwrap_or(MaterialId::STONE);

        // Cave carving (only within crust, not too close to surface)
        if depth_below_surface > 2.0
            && r > self.planet.mean_radius - 4000.0 // Only in crust band
            && self.is_cave(world_x, world_y, world_z)
        {
            return MaterialId::AIR;
        }

        mat
    }
}

/// Map a geological layer material name to a `MaterialId`.
///
/// This is a simple lookup; a more robust version would use the MaterialRegistry.
fn material_from_layer_name(name: &str) -> MaterialId {
    match name {
        "Iron" => MaterialId(6), // iron material ID
        "Stone" => MaterialId::STONE,
        "Dirt" => MaterialId::DIRT,
        "Water" => MaterialId::WATER,
        _ => MaterialId::STONE, // fallback
    }
}

// ---------------------------------------------------------------------------
// Unified terrain generator
// ---------------------------------------------------------------------------

/// Unified terrain generator that dispatches to flat or spherical mode.
pub enum UnifiedTerrainGenerator {
    Flat(TerrainGenerator),
    Spherical(SphericalTerrainGenerator),
}

impl UnifiedTerrainGenerator {
    /// Create from a `PlanetConfig` (uses mode to decide).
    pub fn from_planet_config(planet: &PlanetConfig) -> Self {
        match planet.mode {
            TerrainMode::Flat => {
                let config = TerrainConfig {
                    seed: planet.seed,
                    sea_level: planet.sea_level_radius as i32,
                    height_scale: planet.height_scale,
                    continent_freq: planet.continent_freq,
                    detail_freq: planet.detail_freq,
                    cave_freq: planet.cave_freq,
                    cave_threshold: planet.cave_threshold,
                    soil_depth: planet.soil_depth as i32,
                };
                Self::Flat(TerrainGenerator::new(config))
            }
            TerrainMode::Spherical => {
                Self::Spherical(SphericalTerrainGenerator::new(planet.clone()))
            }
        }
    }

    /// Fill a chunk with terrain.
    pub fn generate_chunk(&self, chunk: &mut Chunk) {
        match self {
            Self::Flat(g) => g.generate_chunk(chunk),
            Self::Spherical(g) => g.generate_chunk(chunk),
        }
    }

    /// Access the flat terrain config, if in flat mode.
    pub fn config(&self) -> Option<&TerrainConfig> {
        match self {
            Self::Flat(g) => Some(g.config()),
            Self::Spherical(_) => None,
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
        let tgen_a = TerrainGenerator::new(TerrainConfig {
            seed: 1,
            ..Default::default()
        });
        let tgen_b = TerrainGenerator::new(TerrainConfig {
            seed: 9999,
            ..Default::default()
        });
        let h_a = tgen_a.sample_height(50.0, 50.0);
        let h_b = tgen_b.sample_height(50.0, 50.0);
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

    // -----------------------------------------------------------------------
    // Spherical terrain generator tests
    // -----------------------------------------------------------------------

    fn spherical_generator() -> SphericalTerrainGenerator {
        SphericalTerrainGenerator::new(PlanetConfig::default())
    }

    /// A small planet for fast chunk-level tests.
    fn small_planet_generator() -> SphericalTerrainGenerator {
        SphericalTerrainGenerator::new(PlanetConfig {
            mean_radius: 100.0,
            sea_level_radius: 100.0,
            height_scale: 8.0,
            soil_depth: 2.0,
            cave_threshold: -999.0, // Disable caves for deterministic tests
            layers: vec![
                super::super::planet::GeologicalLayer {
                    name: "core".into(),
                    inner_radius: 0.0,
                    outer_radius: 50.0,
                    material: "Iron".into(),
                },
                super::super::planet::GeologicalLayer {
                    name: "mantle".into(),
                    inner_radius: 50.0,
                    outer_radius: 90.0,
                    material: "Stone".into(),
                },
                super::super::planet::GeologicalLayer {
                    name: "crust".into(),
                    inner_radius: 90.0,
                    outer_radius: 100.0,
                    material: "Stone".into(),
                },
            ],
            ..PlanetConfig::default()
        })
    }

    #[test]
    fn spherical_surface_radius_is_deterministic() {
        let tgen = spherical_generator();
        let r1 = tgen.sample_surface_radius(0.5, 1.0);
        let r2 = tgen.sample_surface_radius(0.5, 1.0);
        assert_eq!(r1, r2);
    }

    #[test]
    fn spherical_surface_radius_varies() {
        let tgen = spherical_generator();
        let mut radii = Vec::new();
        for i in 0..10 {
            let lat = i as f64 * 0.3;
            radii.push(tgen.sample_surface_radius(lat, 0.0));
        }
        let min = radii.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = radii.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            max - min > 0.1,
            "Surface radius should vary: min={min}, max={max}"
        );
    }

    #[test]
    fn spherical_chunk_at_surface_has_mixed_materials() {
        let tgen = small_planet_generator();
        let r = tgen.planet().mean_radius;
        // Chunk at the surface along +X axis: center at (r, 0, 0)
        let cx = (r / CHUNK_SIZE as f64).floor() as i32;
        let coord = ChunkCoord::new(cx, 0, 0);
        let mut chunk = Chunk::new_empty(coord);
        tgen.generate_chunk(&mut chunk);

        let air_count = chunk.voxels().iter().filter(|v| v.is_air()).count();
        let solid_count = chunk.voxels().iter().filter(|v| !v.is_air()).count();
        assert!(air_count > 0, "Surface chunk has no air");
        assert!(solid_count > 0, "Surface chunk has no solids");
    }

    #[test]
    fn spherical_chunk_deep_inside_is_all_solid() {
        let tgen = small_planet_generator();
        // Chunk at origin (deep inside planet core)
        let coord = ChunkCoord::new(0, 0, 0);
        let mut chunk = Chunk::new_empty(coord);
        tgen.generate_chunk(&mut chunk);

        let air_count = chunk.voxels().iter().filter(|v| v.is_air()).count();
        assert_eq!(air_count, 0, "Core chunk should be 100% solid");
    }

    #[test]
    fn spherical_chunk_far_outside_is_all_air() {
        let tgen = small_planet_generator();
        // Chunk well outside the planet
        let coord = ChunkCoord::new(10, 10, 10); // (320, 320, 320) — far from r=100
        let mut chunk = Chunk::new_empty(coord);
        tgen.generate_chunk(&mut chunk);

        assert!(
            chunk.is_empty(),
            "Chunk far outside planet should be all air"
        );
    }

    #[test]
    fn spherical_different_seeds_produce_different_terrain() {
        let tgen_a = SphericalTerrainGenerator::new(PlanetConfig {
            seed: 1,
            ..PlanetConfig::default()
        });
        let tgen_b = SphericalTerrainGenerator::new(PlanetConfig {
            seed: 9999,
            ..PlanetConfig::default()
        });
        let r_a = tgen_a.sample_surface_radius(0.5, 0.5);
        let r_b = tgen_b.sample_surface_radius(0.5, 0.5);
        assert!(
            (r_a - r_b).abs() > 0.001,
            "Different seeds should produce different surfaces"
        );
    }

    #[test]
    fn spherical_material_layers_are_correct() {
        let tgen = small_planet_generator();
        // Core voxel (inside inner_core layer)
        let mat = tgen.material_at_radius(25.0, 110.0, 0.0, 25.0, 0.0);
        assert_eq!(mat, MaterialId(6), "Core should be Iron (MaterialId 6)");

        // Mantle voxel
        let mat = tgen.material_at_radius(60.0, 110.0, 0.0, 60.0, 0.0);
        assert_eq!(mat, MaterialId::STONE, "Mantle should be Stone");

        // Above surface
        let mat = tgen.material_at_radius(120.0, 110.0, 0.0, 120.0, 0.0);
        assert_eq!(mat, MaterialId::AIR, "Above surface should be Air");
    }

    #[test]
    fn spherical_soil_depth_layers() {
        let tgen = small_planet_generator();
        let surface_r = 108.0; // surface height
        // Just at surface (depth < 1.0)
        let mat = tgen.material_at_radius(surface_r - 0.5, surface_r, 0.0, surface_r - 0.5, 0.0);
        assert!(
            mat == MaterialId(4) || mat == MaterialId::DIRT,
            "Surface should be grass or dirt, got {:?}",
            mat
        );
        // Soil layer (depth 1.0 to soil_depth)
        let mat = tgen.material_at_radius(surface_r - 1.5, surface_r, 0.0, surface_r - 1.5, 0.0);
        assert_eq!(mat, MaterialId::DIRT, "Soil layer should be Dirt");
    }

    #[test]
    fn spherical_water_below_sea_level() {
        let tgen = SphericalTerrainGenerator::new(PlanetConfig {
            mean_radius: 100.0,
            sea_level_radius: 105.0, // Sea level ABOVE mean radius
            height_scale: 2.0,       // Small variation
            cave_threshold: -999.0,
            layers: vec![super::super::planet::GeologicalLayer {
                name: "crust".into(),
                inner_radius: 0.0,
                outer_radius: 100.0,
                material: "Stone".into(),
            }],
            ..PlanetConfig::default()
        });
        // Position above surface but below sea level
        let mat = tgen.material_at_radius(103.0, 101.0, 0.0, 103.0, 0.0);
        assert_eq!(mat, MaterialId::WATER, "Below sea level should be water");
    }

    // -----------------------------------------------------------------------
    // Unified generator tests
    // -----------------------------------------------------------------------

    #[test]
    fn unified_flat_mode_uses_flat_generator() {
        let planet = PlanetConfig {
            mode: TerrainMode::Flat,
            ..PlanetConfig::default()
        };
        let tgen = UnifiedTerrainGenerator::from_planet_config(&planet);
        assert!(matches!(tgen, UnifiedTerrainGenerator::Flat(_)));
    }

    #[test]
    fn unified_spherical_mode_uses_spherical_generator() {
        let planet = PlanetConfig {
            mode: TerrainMode::Spherical,
            ..PlanetConfig::default()
        };
        let tgen = UnifiedTerrainGenerator::from_planet_config(&planet);
        assert!(matches!(tgen, UnifiedTerrainGenerator::Spherical(_)));
    }
}
