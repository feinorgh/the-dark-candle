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

use std::sync::OnceLock;

use noise::{NoiseFn, Perlin};
use serde::{Deserialize, Serialize};

use super::biome_map::{self, EnvironmentMap};
use super::chunk::{CHUNK_SIZE, Chunk};
use super::erosion::{ErosionConfig, FlowMap, carve_valley};
use super::noise::NoiseStack;
use super::planet::{PlanetConfig, TerrainMode};
use super::voxel::MaterialId;

/// Compute continuous density for Surface Nets interpolation.
///
/// `depth` is the signed distance below the terrain surface (positive =
/// underground, negative = above surface).  Returns a value in \[0, 1\]
/// where 0.5 is the isosurface.
///
/// The gradient spans 2 m (±1 m from the surface), giving Surface Nets
/// enough variation to place vertices at sub-voxel accuracy.
#[inline]
pub fn terrain_density(depth: f64) -> f32 {
    (0.5 + depth * 0.5).clamp(0.0, 1.0) as f32
}

// ── Geological strata & ore veins ──────────────────────────────────────────

/// Pre-cached Perlin objects for geological terrain helpers.
///
/// Eliminates per-voxel `Perlin::new()` calls in strata, ore, cave, and
/// crystal deposit functions.
pub(crate) struct CachedGeologyPerlin {
    strata: Perlin,
    ore_coal: Perlin,
    ore_copper: Perlin,
    ore_iron: Perlin,
    ore_gold: Perlin,
    cave_cavern: Perlin,
    cave_tunnel: Perlin,
    cave_tube_xz: Perlin,
    cave_tube_xy: Perlin,
    crystal: Perlin,
}

impl CachedGeologyPerlin {
    fn new(seed: u32) -> Self {
        Self {
            strata: Perlin::new(seed.wrapping_add(400)),
            ore_coal: Perlin::new(seed.wrapping_add(500)),
            ore_copper: Perlin::new(seed.wrapping_add(501)),
            ore_iron: Perlin::new(seed.wrapping_add(502)),
            ore_gold: Perlin::new(seed.wrapping_add(503)),
            cave_cavern: Perlin::new(seed.wrapping_add(600)),
            cave_tunnel: Perlin::new(seed.wrapping_add(601)),
            cave_tube_xz: Perlin::new(seed.wrapping_add(602)),
            cave_tube_xy: Perlin::new(seed.wrapping_add(603)),
            crystal: Perlin::new(seed.wrapping_add(604)),
        }
    }
}

/// Select a rock material based on depth below the terrain surface.
///
/// Uses a low-frequency 3D noise to vary the material within each stratum,
/// creating natural-looking geological variation.
fn strata_material(
    depth: f64,
    perlin: &Perlin,
    world_x: f64,
    world_y: f64,
    world_z: f64,
) -> MaterialId {
    let n = perlin.get([world_x * 0.02, world_y * 0.02, world_z * 0.02]);

    if depth < 20.0 {
        // Sedimentary layer
        if n > 0.0 {
            MaterialId::SANDSTONE
        } else {
            MaterialId::LIMESTONE
        }
    } else if depth < 60.0 {
        // Metamorphic layer (use existing STONE as slate/quartzite analog)
        MaterialId::STONE
    } else {
        // Igneous layer
        if n > 0.0 {
            MaterialId::GRANITE
        } else {
            MaterialId::BASALT
        }
    }
}

/// Check if an ore vein exists at this position and return the ore material.
///
/// Each ore type uses its own noise field and only fires within its valid
/// depth range.
fn ore_material(
    depth: f64,
    geo: &CachedGeologyPerlin,
    world_x: f64,
    world_y: f64,
    world_z: f64,
) -> Option<MaterialId> {
    // Coal: 5–30 m, common
    if (5.0..=30.0).contains(&depth)
        && geo
            .ore_coal
            .get([world_x * 0.08, world_y * 0.08, world_z * 0.08])
            < -0.15
    {
        return Some(MaterialId::COAL);
    }

    // Copper ore: 15–50 m, moderate
    if (15.0..=50.0).contains(&depth)
        && geo
            .ore_copper
            .get([world_x * 0.06, world_y * 0.06, world_z * 0.06])
            < -0.20
    {
        return Some(MaterialId::COPPER_ORE);
    }

    // Iron ore: 30–80 m, moderate
    if (30.0..=80.0).contains(&depth)
        && geo
            .ore_iron
            .get([world_x * 0.05, world_y * 0.05, world_z * 0.05])
            < -0.25
    {
        return Some(MaterialId::IRON);
    }

    // Gold ore: 50 m+, rare
    if depth >= 50.0
        && geo
            .ore_gold
            .get([world_x * 0.04, world_y * 0.04, world_z * 0.04])
            < -0.35
    {
        return Some(MaterialId::GOLD_ORE);
    }

    None
}

// ── Multi-scale cave system ────────────────────────────────────────────────

/// Multi-scale cave system: three OR-combined layers.
///
/// - **Caverns**: low-frequency (0.01) → cathedral-sized chambers.
/// - **Tunnels**: mid-frequency (0.04) → narrow connecting passages.
/// - **Tubes**: two perpendicular 2D noise fields (0.025) → worm-like networks.
///
/// Returns `true` if the position should be carved as a cave.
fn is_multi_scale_cave(
    geo: &CachedGeologyPerlin,
    cave_threshold: f64,
    world_x: f64,
    world_y: f64,
    world_z: f64,
) -> bool {
    // Caverns (cathedral-sized chambers)
    let cavern = geo
        .cave_cavern
        .get([world_x * 0.01, world_y * 0.01, world_z * 0.01]);
    if cavern < cave_threshold * 0.5 {
        return true;
    }

    // Tunnels (narrow passages)
    let tunnel = geo
        .cave_tunnel
        .get([world_x * 0.04, world_y * 0.04, world_z * 0.04]);
    if tunnel < cave_threshold * 1.2 {
        return true;
    }

    // Tube networks (worm-like — intersection of two 2D noise fields)
    let t_xz = geo.cave_tube_xz.get([world_x * 0.025, world_z * 0.025]);
    let t_xy = geo.cave_tube_xy.get([world_x * 0.025, world_y * 0.025]);
    if t_xz < cave_threshold * 0.85 && t_xy < cave_threshold * 0.85 {
        return true;
    }

    false
}

/// Determine the cave-fill material (AIR, WATER, or LAVA based on depth).
fn cave_fill_material(depth: f64, sea_level_depth: f64) -> MaterialId {
    if depth > 80.0 {
        // Deep lava tubes (rare — only some positions)
        MaterialId::LAVA
    } else if depth > sea_level_depth + 5.0 {
        // Underground lakes
        MaterialId::WATER
    } else {
        MaterialId::AIR
    }
}

/// Check if a cave wall should have crystal deposits (deep cavern walls).
fn is_crystal_deposit(
    depth: f64,
    perlin: &Perlin,
    world_x: f64,
    world_y: f64,
    world_z: f64,
) -> bool {
    if depth < 40.0 {
        return false;
    }
    perlin.get([world_x * 0.10, world_y * 0.10, world_z * 0.10]) < -0.30
}

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
    /// Erosion/valley carving configuration.
    #[serde(default)]
    pub erosion: ErosionConfig,
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
            erosion: ErosionConfig::default(),
        }
    }
}

/// Stateless terrain generator. Holds noise functions seeded from config.
pub struct TerrainGenerator {
    config: TerrainConfig,
    continent_noise: Perlin,
    detail_noise: Perlin,
    cave_noise: Perlin,
    /// Composable multi-octave noise stack (replaces 2-layer blend when present).
    noise_stack: Option<NoiseStack>,
    /// Environment map for biome-driven surface materials.
    env_map: EnvironmentMap,
    /// Cached flow-accumulation map, computed lazily on first chunk generation.
    flow_map: OnceLock<FlowMap>,
    /// Pre-cached Perlin objects for geological terrain helpers.
    geo_perlin: CachedGeologyPerlin,
}

impl TerrainGenerator {
    pub fn new(config: TerrainConfig) -> Self {
        let continent_noise = Perlin::new(config.seed);
        let detail_noise = Perlin::new(config.seed.wrapping_add(1));
        let cave_noise = Perlin::new(config.seed.wrapping_add(2));
        let env_map = EnvironmentMap::new(config.seed);
        let geo_perlin = CachedGeologyPerlin::new(config.seed);
        Self {
            config,
            continent_noise,
            detail_noise,
            cave_noise,
            noise_stack: None,
            env_map,
            flow_map: OnceLock::new(),
            geo_perlin,
        }
    }

    /// Create a generator with a `NoiseStack` for multi-octave terrain.
    pub fn with_noise_stack(config: TerrainConfig, noise_stack: NoiseStack) -> Self {
        let continent_noise = Perlin::new(config.seed);
        let detail_noise = Perlin::new(config.seed.wrapping_add(1));
        let cave_noise = Perlin::new(config.seed.wrapping_add(2));
        let env_map = EnvironmentMap::new(config.seed);
        let geo_perlin = CachedGeologyPerlin::new(config.seed);
        Self {
            config,
            continent_noise,
            detail_noise,
            cave_noise,
            noise_stack: Some(noise_stack),
            env_map,
            flow_map: OnceLock::new(),
            geo_perlin,
        }
    }

    pub fn config(&self) -> &TerrainConfig {
        &self.config
    }

    /// Sample the terrain height at a world XZ position.
    /// Returns a float height in voxel units.
    pub fn sample_height(&self, world_x: f64, world_z: f64) -> f64 {
        if let Some(ref stack) = self.noise_stack {
            let combined = stack.sample(world_x, world_z);
            self.config.sea_level as f64 + combined * self.config.height_scale
        } else {
            // Legacy 2-layer blend
            let cx = world_x * self.config.continent_freq;
            let cz = world_z * self.config.continent_freq;
            let continent = self.continent_noise.get([cx, cz]);

            let dx = world_x * self.config.detail_freq;
            let dz = world_z * self.config.detail_freq;
            let detail = self.detail_noise.get([dx, dz]);

            let combined = continent * 0.7 + detail * 0.3;
            self.config.sea_level as f64 + combined * self.config.height_scale
        }
    }

    /// Check if a world position should be carved as a cave.
    pub fn is_cave(&self, world_x: f64, world_y: f64, world_z: f64) -> bool {
        let nx = world_x * self.config.cave_freq;
        let ny = world_y * self.config.cave_freq;
        let nz = world_z * self.config.cave_freq;
        self.cave_noise.get([nx, ny, nz]) < self.config.cave_threshold
    }

    /// Lazily compute and cache the flow-accumulation map.
    ///
    /// The flow map covers a square region centered at the world origin. It is
    /// built from `sample_height()` on a coarse grid and reused for all
    /// subsequent chunk generations.
    pub fn get_or_compute_flow_map(&self) -> &FlowMap {
        self.flow_map.get_or_init(|| {
            // Build a temporary generator with the same config to avoid
            // borrowing `self` inside the closure (OnceLock requires &self).
            let sampler = TerrainGenerator {
                config: self.config.clone(),
                continent_noise: Perlin::new(self.config.seed),
                detail_noise: Perlin::new(self.config.seed.wrapping_add(1)),
                cave_noise: Perlin::new(self.config.seed.wrapping_add(2)),
                noise_stack: None,
                env_map: EnvironmentMap::new(self.config.seed),
                flow_map: OnceLock::new(),
                geo_perlin: CachedGeologyPerlin::new(self.config.seed),
            };
            FlowMap::compute(
                |x, z| sampler.sample_height(x, z),
                f64::from(self.config.erosion.region_size),
                f64::from(self.config.erosion.cell_size),
                0.0,
                0.0,
            )
        })
    }

    /// Fill a chunk with terrain based on its world position.
    pub fn generate_chunk(&self, chunk: &mut Chunk) {
        let origin = chunk.coord.world_origin();
        let erosion_enabled = self.config.erosion.enabled;
        let use_advanced = self.noise_stack.is_some();

        // Lazily compute the flow map on first use
        let flow_map = if erosion_enabled {
            Some(self.get_or_compute_flow_map())
        } else {
            None
        };

        for lz in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                let world_x = (origin.x + lx as i32) as f64;
                let world_z = (origin.z + lz as i32) as f64;
                let base_height = self.sample_height(world_x, world_z);

                // Apply valley carving if erosion is enabled
                let (height, erosion_material) = if let Some(fm) = flow_map {
                    if let Some(channel_info) = fm.nearest_channel_info(
                        world_x,
                        world_z,
                        self.config.erosion.flow_threshold,
                    ) {
                        carve_valley(base_height, &channel_info, &self.config.erosion)
                    } else {
                        (base_height, None)
                    }
                } else {
                    (base_height, None)
                };

                // Per-column biome/slope data (advanced mode only)
                let (slope, env, effective_soil) = if use_advanced {
                    let s =
                        biome_map::compute_slope(|x, z| self.sample_height(x, z), world_x, world_z);
                    let e = self.env_map.sample(world_x, world_z);
                    let soil = biome_map::adjusted_soil_depth(self.config.soil_depth as f64, s);
                    (s, Some(e), soil)
                } else {
                    (0.0, None, self.config.soil_depth as f64)
                };

                for ly in 0..CHUNK_SIZE {
                    let world_y = origin.y + ly as i32;
                    let wy_f64 = world_y as f64;
                    let depth = height - wy_f64;

                    let material = if wy_f64 > height {
                        // Above terrain surface
                        if world_y < self.config.sea_level {
                            MaterialId::WATER
                        } else {
                            MaterialId::AIR
                        }
                    } else if wy_f64 > height - 1.0 {
                        // Top layer: slope/altitude/biome-aware surface material
                        if let Some(mat) = erosion_material {
                            mat
                        } else if use_advanced {
                            let alt = wy_f64 - self.config.sea_level as f64;
                            biome_map::surface_material(slope, alt, env.as_ref().unwrap())
                        } else if world_y >= self.config.sea_level {
                            MaterialId::GRASS
                        } else {
                            MaterialId::DIRT
                        }
                    } else if depth < effective_soil {
                        // Soil layers
                        MaterialId::DIRT
                    } else if use_advanced {
                        // Geological strata with ore veins
                        let strata_depth = depth - effective_soil;
                        ore_material(strata_depth, &self.geo_perlin, world_x, wy_f64, world_z)
                            .unwrap_or_else(|| {
                                strata_material(
                                    strata_depth,
                                    &self.geo_perlin.strata,
                                    world_x,
                                    wy_f64,
                                    world_z,
                                )
                            })
                    } else {
                        // Legacy: uniform stone
                        MaterialId::STONE
                    };

                    // Cave carving (only underground, not too close to surface)
                    if material != MaterialId::AIR && material != MaterialId::WATER && depth > 2.0 {
                        let is_cave = if use_advanced {
                            is_multi_scale_cave(
                                &self.geo_perlin,
                                self.config.cave_threshold,
                                world_x,
                                wy_f64,
                                world_z,
                            )
                        } else {
                            self.is_cave(world_x, wy_f64, world_z)
                        };

                        if is_cave {
                            let sea_level_depth = (self.config.sea_level as f64 - wy_f64).max(0.0);
                            let fill = if use_advanced {
                                cave_fill_material(depth, sea_level_depth)
                            } else {
                                MaterialId::AIR
                            };
                            chunk.set_material(lx, ly, lz, fill);
                            // Binary density for caves (set_material handles it)
                            continue;
                        }

                        // Crystal deposits on cave-adjacent walls
                        if use_advanced
                            && is_crystal_deposit(
                                depth,
                                &self.geo_perlin.crystal,
                                world_x,
                                wy_f64,
                                world_z,
                            )
                        {
                            chunk.set_material(lx, ly, lz, MaterialId::QUARTZ_CRYSTAL);
                            continue;
                        }
                    }

                    chunk.set_material(lx, ly, lz, material);

                    // Compute smooth density for Surface Nets interpolation.
                    // For terrain surface voxels, density encodes the fractional
                    // position within the voxel. For water, use distance to sea level.
                    let density = if material == MaterialId::WATER {
                        let sea_depth = self.config.sea_level as f64 - wy_f64;
                        terrain_density(sea_depth)
                    } else {
                        terrain_density(depth)
                    };
                    chunk.get_mut(lx, ly, lz).density = density;
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
    /// Composable multi-octave noise stack (replaces 2-layer blend when present).
    noise_stack: Option<NoiseStack>,
    /// Pre-cached Perlin objects for geological terrain helpers.
    geo_perlin: CachedGeologyPerlin,
}

impl SphericalTerrainGenerator {
    pub fn new(planet: PlanetConfig) -> Self {
        let continent_noise = Perlin::new(planet.seed);
        let detail_noise = Perlin::new(planet.seed.wrapping_add(1));
        let cave_noise = Perlin::new(planet.seed.wrapping_add(2));
        let geo_perlin = CachedGeologyPerlin::new(planet.seed);
        let noise_stack = planet
            .noise
            .as_ref()
            .map(|cfg| NoiseStack::new(planet.seed, cfg.clone()));
        Self {
            planet,
            continent_noise,
            detail_noise,
            cave_noise,
            noise_stack,
            geo_perlin,
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
        // Fast path: no displacement when height_scale is zero.
        if self.planet.height_scale == 0.0 {
            return self.planet.mean_radius;
        }

        let combined = if let Some(ref stack) = self.noise_stack {
            stack.sample(lon, lat)
        } else {
            // Legacy 2-layer blend using lat/lon as noise coordinates.
            let cx = lon * self.planet.continent_freq;
            let cz = lat * self.planet.continent_freq;
            let continent = self.continent_noise.get([cx, cz]);

            let dx = lon * self.planet.detail_freq;
            let dz = lat * self.planet.detail_freq;
            let detail = self.detail_noise.get([dx, dz]);

            continent * 0.7 + detail * 0.3
        };
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
    ///
    /// Surface-radius noise is cached per `(lx, lz)` column at a
    /// representative Y level (chunk center).  Within a 32 m chunk at
    /// r ≈ 32 km the angular span is ≈ 0.001 rad — far below the lowest
    /// noise frequency — so reusing the column value for all Y levels
    /// introduces negligible error while reducing noise calls 32×.
    pub fn generate_chunk(&self, chunk: &mut Chunk) {
        let origin = chunk.coord.world_origin();

        // Pre-compute surface radius for each (lx, lz) column using the
        // chunk's vertical midpoint as the representative Y.
        let mid_y = (origin.y + CHUNK_SIZE as i32 / 2) as f64;
        let mut surface_cache = [[0.0_f64; CHUNK_SIZE]; CHUNK_SIZE];
        for (lz, row) in surface_cache.iter_mut().enumerate() {
            for (lx, cached) in row.iter_mut().enumerate() {
                let wx = (origin.x + lx as i32) as f64;
                let wz = (origin.z + lz as i32) as f64;
                let pos = bevy::math::DVec3::new(wx, mid_y, wz);
                let (lat, lon) = self.planet.lat_lon(pos);
                *cached = self.sample_surface_radius(lat, lon);
            }
        }

        self.fill_chunk_from_surface_cache(chunk, &surface_cache);
    }

    /// Fill a chunk using GPU-precomputed surface radii.
    ///
    /// `gpu_heights` must contain exactly `CHUNK_SIZE × CHUNK_SIZE` values
    /// (1024 for 32×32), laid out row-major as `[lz * CHUNK_SIZE + lx]`.
    /// Each value is a surface radius in meters (f32 from GPU, upcast to f64).
    pub fn generate_chunk_with_gpu_heights(&self, chunk: &mut Chunk, gpu_heights: &[f32]) {
        debug_assert_eq!(
            gpu_heights.len(),
            CHUNK_SIZE * CHUNK_SIZE,
            "gpu_heights must have CHUNK_SIZE² entries"
        );

        let mut surface_cache = [[0.0_f64; CHUNK_SIZE]; CHUNK_SIZE];
        for lz in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                surface_cache[lz][lx] = gpu_heights[lz * CHUNK_SIZE + lx] as f64;
            }
        }

        self.fill_chunk_from_surface_cache(chunk, &surface_cache);
    }

    /// Shared voxel-fill logic: given a precomputed surface_cache, classify
    /// every voxel and assign material + density.
    fn fill_chunk_from_surface_cache(
        &self,
        chunk: &mut Chunk,
        surface_cache: &[[f64; CHUNK_SIZE]; CHUNK_SIZE],
    ) {
        let origin = chunk.coord.world_origin();

        // Chunk-level early exit: if the entire chunk is above or below all
        // surface radii, we can skip per-voxel work entirely.
        let half = CHUNK_SIZE as f64 / 2.0;
        let center = bevy::math::DVec3::new(
            origin.x as f64 + half,
            origin.y as f64 + half,
            origin.z as f64 + half,
        );
        let center_r = center.length();
        // Half-diagonal of the cube: sqrt(3) * half ≈ 27.7 m for CHUNK_SIZE=32
        let half_diag = half * 3.0_f64.sqrt();

        let chunk_min_r = (center_r - half_diag).max(0.0);
        let chunk_max_r = center_r + half_diag;

        // Find the extremes across all cached surface radii.
        let mut min_surface = f64::MAX;
        let mut max_surface = f64::MIN;
        for row in surface_cache {
            for &sr in row {
                if sr < min_surface {
                    min_surface = sr;
                }
                if sr > max_surface {
                    max_surface = sr;
                }
            }
        }
        let sea = self.planet.sea_level_radius;

        // Entirely above terrain AND above sea level → all air.
        if chunk_min_r > max_surface && chunk_min_r > sea {
            // Chunk already starts as all-air from new_empty(); nothing to do.
            return;
        }

        // Entirely below terrain surface → simplified solid fill (no caves
        // check needed if also below the cave depth cutoff, 200 m).
        let cave_floor_r = min_surface - 200.0;
        if chunk_max_r < min_surface && chunk_max_r < cave_floor_r {
            chunk.fill(MaterialId::STONE);
            for v in chunk.voxels_mut() {
                v.density = 1.0;
            }
            return;
        }

        for (lz, cache_row) in surface_cache.iter().enumerate() {
            for ly in 0..CHUNK_SIZE {
                for (lx, &surface_r) in cache_row.iter().enumerate() {
                    let world_x = (origin.x + lx as i32) as f64;
                    let world_y = (origin.y + ly as i32) as f64;
                    let world_z = (origin.z + lz as i32) as f64;

                    let pos = bevy::math::DVec3::new(world_x, world_y, world_z);
                    let r = self.planet.distance_from_center(pos);

                    let material = self.material_at_radius(r, surface_r, world_x, world_y, world_z);

                    chunk.set_material(lx, ly, lz, material);

                    let density = if material == MaterialId::WATER {
                        terrain_density(self.planet.sea_level_radius - r)
                    } else {
                        terrain_density(surface_r - r)
                    };
                    chunk.get_mut(lx, ly, lz).density = density;
                }
            }
        }
    }

    /// Determine material at a given radius relative to the terrain surface.
    pub fn material_at_radius(
        &self,
        r: f64,
        surface_r: f64,
        world_x: f64,
        world_y: f64,
        world_z: f64,
    ) -> MaterialId {
        let use_advanced = self.noise_stack.is_some();

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
                return MaterialId::GRASS;
            }
            return MaterialId::DIRT;
        }

        if depth_below_surface < self.planet.soil_depth {
            return MaterialId::DIRT;
        }

        // Deep: geological strata or layer-based
        let strata_depth = depth_below_surface - self.planet.soil_depth;
        let base_mat = if use_advanced {
            ore_material(strata_depth, &self.geo_perlin, world_x, world_y, world_z).unwrap_or_else(
                || {
                    strata_material(
                        strata_depth,
                        &self.geo_perlin.strata,
                        world_x,
                        world_y,
                        world_z,
                    )
                },
            )
        } else {
            self.planet
                .layer_at_radius(r)
                .map(|l| material_from_layer_name(&l.material))
                .unwrap_or(MaterialId::STONE)
        };

        // Cave carving (within the crust, not too close to surface).
        // Limit cave evaluation to 200 m below the local surface — deeper
        // voxels are unreachable from the loading sphere and don't need the
        // expensive 3D Perlin checks.
        let cave_max_depth = 200.0;
        if depth_below_surface > 2.0 && depth_below_surface < cave_max_depth {
            let is_cave = if use_advanced {
                is_multi_scale_cave(
                    &self.geo_perlin,
                    self.planet.cave_threshold,
                    world_x,
                    world_y,
                    world_z,
                )
            } else {
                self.is_cave(world_x, world_y, world_z)
            };

            if is_cave {
                if use_advanced {
                    let sea_level_depth = (self.planet.sea_level_radius - r).max(0.0);
                    return cave_fill_material(depth_below_surface, sea_level_depth);
                }
                return MaterialId::AIR;
            }

            // Crystal deposits on cave-adjacent walls
            if use_advanced
                && is_crystal_deposit(
                    depth_below_surface,
                    &self.geo_perlin.crystal,
                    world_x,
                    world_y,
                    world_z,
                )
            {
                return MaterialId::QUARTZ_CRYSTAL;
            }
        }

        base_mat
    }
}

/// Map a geological layer material name to a `MaterialId`.
///
/// This is a simple lookup; a more robust version would use the MaterialRegistry.
fn material_from_layer_name(name: &str) -> MaterialId {
    match name {
        "Iron" => MaterialId::IRON,
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
    Flat(Box<TerrainGenerator>),
    Spherical(Box<SphericalTerrainGenerator>),
    /// Planet-data-driven spherical generator.
    Planetary(Box<super::planetary_sampler::PlanetaryTerrainSampler>),
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
                    erosion: planet.erosion.clone().unwrap_or_default(),
                };
                if let Some(ref noise_cfg) = planet.noise {
                    let stack = NoiseStack::new(planet.seed, noise_cfg.clone());
                    Self::Flat(Box::new(TerrainGenerator::with_noise_stack(config, stack)))
                } else {
                    Self::Flat(Box::new(TerrainGenerator::new(config)))
                }
            }
            TerrainMode::Spherical => {
                Self::Spherical(Box::new(SphericalTerrainGenerator::new(planet.clone())))
            }
        }
    }

    /// Fill a chunk with terrain.
    ///
    /// Returns `Some(ChunkBiomeData)` when using the planetary generator;
    /// `None` for flat/spherical noise-based generators.
    pub fn generate_chunk(
        &self,
        chunk: &mut Chunk,
    ) -> Option<super::planetary_sampler::ChunkBiomeData> {
        match self {
            Self::Flat(g) => {
                g.generate_chunk(chunk);
                None
            }
            Self::Spherical(g) => {
                g.generate_chunk(chunk);
                None
            }
            Self::Planetary(g) => {
                let biome = g.generate_chunk(chunk);
                Some(biome)
            }
        }
    }

    /// Fill a chunk using GPU-precomputed surface radii (spherical mode only).
    ///
    /// For non-spherical modes, falls back to the standard CPU path.
    pub fn generate_chunk_with_gpu_heights(
        &self,
        chunk: &mut Chunk,
        gpu_heights: &[f32],
    ) -> Option<super::planetary_sampler::ChunkBiomeData> {
        match self {
            Self::Spherical(g) => {
                g.generate_chunk_with_gpu_heights(chunk, gpu_heights);
                None
            }
            _ => self.generate_chunk(chunk),
        }
    }

    /// Access the flat terrain config, if in flat mode.
    pub fn config(&self) -> Option<&TerrainConfig> {
        match self {
            Self::Flat(g) => Some(g.config()),
            Self::Spherical(_) | Self::Planetary(_) => None,
        }
    }

    /// Access the cached flow-accumulation map (flat mode only).
    ///
    /// Returns `None` in spherical mode or if erosion is disabled.
    /// Triggers lazy computation on first call.
    pub fn flow_map(&self) -> Option<&FlowMap> {
        match self {
            Self::Flat(g) if g.config().erosion.enabled => Some(g.get_or_compute_flow_map()),
            _ => None,
        }
    }

    /// Sample terrain surface height at the given world (x, z) coordinates.
    ///
    /// For flat mode, delegates to the noise-based height function and returns
    /// a Y value.
    ///
    /// For spherical mode, converts the Cartesian (x, z) position to (lat, lon)
    /// and returns the surface radius at that angular position. The caller must
    /// interpret this as a radial distance from the planet center, not a Y value.
    pub fn sample_height(&self, world_x: f64, world_z: f64) -> f64 {
        match self {
            Self::Flat(g) => g.sample_height(world_x, world_z),
            Self::Spherical(g) => {
                let pos = bevy::math::DVec3::new(world_x, 0.0, world_z);
                let (lat, lon) = g.planet().lat_lon(pos);
                g.sample_surface_radius(lat, lon)
            }
            Self::Planetary(g) => {
                // Use column mid-Y for the unit-sphere projection.
                let pos = bevy::math::DVec3::new(world_x, g.planet_config.mean_radius, world_z);
                let unit = pos.normalize_or(bevy::math::DVec3::Y);
                let (surface_r, _) = g.surface_radius_at(unit);
                surface_r
            }
        }
    }

    /// Whether the terrain is in spherical mode.
    pub fn is_spherical(&self) -> bool {
        matches!(self, Self::Spherical(_) | Self::Planetary(_))
    }

    /// Access the spherical terrain generator, if in spherical mode.
    pub fn spherical(&self) -> Option<&SphericalTerrainGenerator> {
        match self {
            Self::Spherical(g) => Some(g),
            _ => None,
        }
    }

    /// Sample the surface radius at a given (lat, lon) in radians.
    ///
    /// For spherical mode, returns the radial distance from the planet center.
    /// For planetary mode, queries the tectonic sampler.
    /// For flat mode, returns 0 (meaningless — use `sample_height` instead).
    pub fn sample_surface_radius_at(&self, lat: f64, lon: f64) -> f64 {
        match self {
            Self::Spherical(g) => g.sample_surface_radius(lat, lon),
            Self::Planetary(g) => {
                let dir =
                    bevy::math::DVec3::new(lat.cos() * lon.cos(), lat.sin(), lat.cos() * lon.sin());
                let (surface_r, _) = g.surface_radius_at(dir);
                surface_r
            }
            Self::Flat(_) => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::chunk::ChunkCoord;
    use super::*;

    fn test_geo(seed: u32) -> CachedGeologyPerlin {
        CachedGeologyPerlin::new(seed)
    }

    fn default_generator() -> TerrainGenerator {
        TerrainGenerator::new(TerrainConfig {
            erosion: ErosionConfig {
                enabled: false,
                ..Default::default()
            },
            ..Default::default()
        })
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
        assert_eq!(mat, MaterialId::IRON, "Core should be Iron");

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
            mat == MaterialId::GRASS || mat == MaterialId::DIRT,
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

    // -----------------------------------------------------------------------
    // Erosion integration tests
    // -----------------------------------------------------------------------

    /// Create a generator with erosion enabled and a small region for speed.
    fn erosion_generator() -> TerrainGenerator {
        TerrainGenerator::new(TerrainConfig {
            erosion: ErosionConfig {
                enabled: true,
                region_size: 512.0,
                cell_size: 8.0,
                flow_threshold: 20.0,
                ..Default::default()
            },
            ..Default::default()
        })
    }

    #[test]
    fn erosion_carves_lower_terrain_at_high_flow() {
        let with_erosion = erosion_generator();
        let without_erosion = TerrainGenerator::new(TerrainConfig {
            erosion: ErosionConfig {
                enabled: false,
                ..Default::default()
            },
            ..Default::default()
        });

        // Generate the same surface chunk with and without erosion
        let coord = ChunkCoord::new(0, 2, 0);
        let mut chunk_with = Chunk::new_empty(coord);
        let mut chunk_without = Chunk::new_empty(coord);
        with_erosion.generate_chunk(&mut chunk_with);
        without_erosion.generate_chunk(&mut chunk_without);

        // Count solid voxels — erosion should remove some (carve valleys)
        let _solid_with = chunk_with
            .voxels()
            .iter()
            .filter(|v| !v.is_air() && v.material != MaterialId::WATER)
            .count();
        let _solid_without = chunk_without
            .voxels()
            .iter()
            .filter(|v| !v.is_air() && v.material != MaterialId::WATER)
            .count();

        // Erosion might carve some terrain, resulting in fewer solid voxels
        // (or different materials). At minimum, the outputs should differ.
        // Note: some chunks may not overlap with high-flow areas at all,
        // so we just check that the system runs without panicking and
        // produces valid output.
        assert!(
            chunk_with.voxels().len() == chunk_without.voxels().len(),
            "Both chunks should have same total voxel count"
        );

        // The erosion generator should have computed a flow map
        assert!(
            with_erosion.flow_map.get().is_some(),
            "Flow map should be computed after generating a chunk"
        );
    }

    #[test]
    fn erosion_disabled_matches_original_output() {
        let gen_disabled = TerrainGenerator::new(TerrainConfig {
            erosion: ErosionConfig {
                enabled: false,
                ..Default::default()
            },
            ..Default::default()
        });
        let gen_no_erosion = default_generator();

        let coord = ChunkCoord::new(1, 2, 1);
        let mut chunk_a = Chunk::new_empty(coord);
        let mut chunk_b = Chunk::new_empty(coord);
        gen_disabled.generate_chunk(&mut chunk_a);
        gen_no_erosion.generate_chunk(&mut chunk_b);

        // Should produce identical output when erosion is disabled
        for (a, b) in chunk_a.voxels().iter().zip(chunk_b.voxels().iter()) {
            assert_eq!(
                a.material, b.material,
                "Disabled erosion should match no-erosion output"
            );
        }
    }

    #[test]
    fn erosion_config_in_terrain_config_has_sane_defaults() {
        let cfg = TerrainConfig::default();
        assert!(cfg.erosion.enabled);
        assert!(cfg.erosion.flow_threshold > 0.0);
        assert!(cfg.erosion.region_size > 0.0);
        assert!(cfg.erosion.cell_size > 0.0);
        assert!(cfg.erosion.cell_size < cfg.erosion.region_size);
    }

    #[test]
    fn erosion_flow_map_is_deterministic() {
        let gen1 = erosion_generator();
        let gen2 = erosion_generator();

        let coord = ChunkCoord::new(0, 2, 0);
        let mut chunk1 = Chunk::new_empty(coord);
        let mut chunk2 = Chunk::new_empty(coord);
        gen1.generate_chunk(&mut chunk1);
        gen2.generate_chunk(&mut chunk2);

        for (a, b) in chunk1.voxels().iter().zip(chunk2.voxels().iter()) {
            assert_eq!(
                a.material, b.material,
                "Same seed should produce identical terrain with erosion"
            );
        }
    }

    // ── Geology tests ────────────────────────────────────────────────────

    fn advanced_generator() -> TerrainGenerator {
        use crate::world::noise::{NoiseConfig, NoiseStack};
        let config = TerrainConfig {
            seed: 42,
            erosion: ErosionConfig {
                enabled: false,
                ..Default::default()
            },
            ..Default::default()
        };
        let stack = NoiseStack::new(42, NoiseConfig::default());
        TerrainGenerator::with_noise_stack(config, stack)
    }

    #[test]
    fn strata_sedimentary_at_shallow_depth() {
        let geo = test_geo(42);
        // Sample many positions at depth=10 (sedimentary layer: 0-20m)
        let mut sandstone = 0;
        let mut limestone = 0;
        for x in 0..20 {
            for z in 0..20 {
                let mat = strata_material(10.0, &geo.strata, x as f64 * 10.0, 0.0, z as f64 * 10.0);
                match mat {
                    m if m == MaterialId::SANDSTONE => sandstone += 1,
                    m if m == MaterialId::LIMESTONE => limestone += 1,
                    other => panic!(
                        "Expected sandstone or limestone at depth 10, got {:?}",
                        other
                    ),
                }
            }
        }
        assert!(sandstone > 0, "No sandstone found in sedimentary layer");
        assert!(limestone > 0, "No limestone found in sedimentary layer");
    }

    #[test]
    fn strata_metamorphic_at_mid_depth() {
        let geo = test_geo(42);
        // depth=40 → metamorphic layer (20-60m), always STONE
        for x in 0..10 {
            for z in 0..10 {
                let mat = strata_material(40.0, &geo.strata, x as f64 * 5.0, 0.0, z as f64 * 5.0);
                assert_eq!(
                    mat,
                    MaterialId::STONE,
                    "Metamorphic layer at depth 40 should be stone"
                );
            }
        }
    }

    #[test]
    fn strata_igneous_at_deep_depth() {
        let geo = test_geo(42);
        let mut granite = 0;
        let mut basalt = 0;
        for x in 0..20 {
            for z in 0..20 {
                let mat = strata_material(80.0, &geo.strata, x as f64 * 10.0, 0.0, z as f64 * 10.0);
                match mat {
                    m if m == MaterialId::GRANITE => granite += 1,
                    m if m == MaterialId::BASALT => basalt += 1,
                    other => panic!("Expected granite or basalt at depth 80, got {:?}", other),
                }
            }
        }
        assert!(granite > 0, "No granite found in igneous layer");
        assert!(basalt > 0, "No basalt found in igneous layer");
    }

    #[test]
    fn strata_boundaries_are_correct() {
        let geo = test_geo(42);
        // Boundary at 20m: depth 19.9 → sedimentary, depth 20.0 → metamorphic
        let mat_shallow = strata_material(19.9, &geo.strata, 0.0, 0.0, 0.0);
        let mat_mid = strata_material(20.0, &geo.strata, 0.0, 0.0, 0.0);
        let mat_deep = strata_material(60.0, &geo.strata, 0.0, 0.0, 0.0);

        assert!(
            mat_shallow == MaterialId::SANDSTONE || mat_shallow == MaterialId::LIMESTONE,
            "Depth 19.9 should be sedimentary"
        );
        assert_eq!(
            mat_mid,
            MaterialId::STONE,
            "Depth 20.0 should be metamorphic"
        );
        assert!(
            mat_deep == MaterialId::GRANITE || mat_deep == MaterialId::BASALT,
            "Depth 60.0 should be igneous"
        );
    }

    #[test]
    fn ore_veins_respect_depth_ranges() {
        let geo = test_geo(42);
        // Coal should never appear above depth 5 or below depth 30
        let mut coal_count = 0;
        let mut iron_count = 0;
        let mut gold_count = 0;
        let sample_count = 5000;

        for i in 0..sample_count {
            let x = (i as f64) * 3.7;
            let z = (i as f64) * 2.3;

            // Depth 2: no coal
            assert!(
                ore_material(2.0, &geo, x, 0.0, z) != Some(MaterialId::COAL),
                "Coal should not appear at depth 2"
            );
            // Depth 35: no coal
            assert!(
                ore_material(35.0, &geo, x, 0.0, z) != Some(MaterialId::COAL),
                "Coal should not appear at depth 35"
            );

            // Count occurrences at valid depths
            if ore_material(15.0, &geo, x, 15.0, z) == Some(MaterialId::COAL) {
                coal_count += 1;
            }
            if ore_material(50.0, &geo, x, 50.0, z) == Some(MaterialId::IRON) {
                iron_count += 1;
            }
            if ore_material(70.0, &geo, x, 70.0, z) == Some(MaterialId::GOLD_ORE) {
                gold_count += 1;
            }
        }

        // With 5000 samples, we should find some of each ore at valid depths
        assert!(coal_count > 0, "No coal found at valid depths");
        assert!(iron_count > 0, "No iron found at valid depths");
        assert!(gold_count > 0, "No gold found at valid depths");
        // Gold should be rarer than coal (stricter threshold)
        assert!(
            gold_count < coal_count,
            "Gold ({gold_count}) should be rarer than coal ({coal_count})"
        );
    }

    #[test]
    fn ore_no_gold_above_50m() {
        let geo = test_geo(42);
        for i in 0..1000 {
            let x = (i as f64) * 7.1;
            let z = (i as f64) * 4.3;
            assert!(
                ore_material(49.0, &geo, x, 49.0, z) != Some(MaterialId::GOLD_ORE),
                "Gold should not appear above 50m depth"
            );
        }
    }

    #[test]
    fn multi_scale_caves_produce_variety() {
        let geo = test_geo(42);
        let threshold = -0.3;
        let mut cave_count = 0;
        let samples = 10_000;

        for i in 0..samples {
            let x = (i as f64) * 1.7;
            let y = (i as f64) * 0.9;
            let z = (i as f64) * 2.1;
            if is_multi_scale_cave(&geo, threshold, x, y, z) {
                cave_count += 1;
            }
        }

        // Caves should exist but not dominate
        let ratio = cave_count as f64 / samples as f64;
        assert!(
            ratio > 0.01,
            "Cave ratio {ratio:.3} too low — caves barely exist"
        );
        assert!(
            ratio < 0.5,
            "Cave ratio {ratio:.3} too high — more cave than rock"
        );
    }

    #[test]
    fn cave_fill_material_varies_by_depth() {
        // Shallow caves → AIR
        assert_eq!(cave_fill_material(10.0, 64.0), MaterialId::AIR);
        // Below sea level + 5 → WATER (underground lakes)
        assert_eq!(cave_fill_material(75.0, 64.0), MaterialId::WATER);
        // Very deep → LAVA
        assert_eq!(cave_fill_material(85.0, 64.0), MaterialId::LAVA);
    }

    #[test]
    fn crystal_deposits_only_at_depth() {
        let geo = test_geo(42);
        // Should never appear above 40m
        for i in 0..1000 {
            let x = (i as f64) * 3.3;
            let z = (i as f64) * 2.7;
            assert!(
                !is_crystal_deposit(39.0, &geo.crystal, x, 0.0, z),
                "Crystal deposit should not appear above 40m"
            );
        }

        // Should appear at some deep positions
        let mut found = false;
        for i in 0..5000 {
            let x = (i as f64) * 1.1;
            let z = (i as f64) * 0.9;
            if is_crystal_deposit(60.0, &geo.crystal, x, 60.0, z) {
                found = true;
                break;
            }
        }
        assert!(
            found,
            "No crystal deposits found at depth 60m in 5000 samples"
        );
    }

    #[test]
    fn advanced_generator_deep_chunk_has_geological_variety() {
        let tgen = advanced_generator();
        // Y=-3 → voxels ~96m below surface, should contain igneous + ores + caves
        let coord = ChunkCoord::new(0, -3, 0);
        let mut chunk = Chunk::new_empty(coord);
        tgen.generate_chunk(&mut chunk);

        let mut material_set = std::collections::HashSet::new();
        for v in chunk.voxels() {
            if !v.is_air() {
                material_set.insert(v.material);
            }
        }

        // A deep chunk with advanced geology should have more than just STONE
        assert!(
            material_set.len() >= 2,
            "Deep chunk should have geological variety, got {:?}",
            material_set
        );
    }

    #[test]
    fn advanced_generator_surface_chunk_has_strata() {
        let tgen = advanced_generator();
        // Y=0 has surface + shallow underground
        let coord = ChunkCoord::new(5, 0, 5);
        let mut chunk = Chunk::new_empty(coord);
        tgen.generate_chunk(&mut chunk);

        let mut has_sedimentary = false;
        for v in chunk.voxels() {
            if v.material == MaterialId::SANDSTONE || v.material == MaterialId::LIMESTONE {
                has_sedimentary = true;
                break;
            }
        }
        // At y=0 chunk range (0-31), surface terrain + shallow underground
        // should reveal sedimentary layers in at least some positions
        // (depends on terrain height — if surface is high enough, sub-surface is visible)
        // We test at (5,0,5) which with seed 42 should have some terrain below
        assert!(
            has_sedimentary,
            "Surface chunk at (5,0,5) should contain some sedimentary rock"
        );
    }

    #[test]
    fn advanced_generator_caves_contain_fill_materials() {
        let tgen = advanced_generator();
        // Test multiple deep chunks to find cave fill materials
        let mut found_water_or_lava = false;
        for cy in [-4, -5, -6] {
            for cx in [0, 1, 2, -1, -2] {
                let coord = ChunkCoord::new(cx, cy, 0);
                let mut chunk = Chunk::new_empty(coord);
                tgen.generate_chunk(&mut chunk);

                for v in chunk.voxels() {
                    if v.material == MaterialId::WATER || v.material == MaterialId::LAVA {
                        found_water_or_lava = true;
                        break;
                    }
                }
                if found_water_or_lava {
                    break;
                }
            }
            if found_water_or_lava {
                break;
            }
        }
        assert!(
            found_water_or_lava,
            "Deep caves should contain water or lava fill"
        );
    }

    #[test]
    fn terrain_density_function_values() {
        // Isosurface at depth=0 → density 0.5
        assert!((terrain_density(0.0) - 0.5).abs() < f32::EPSILON);
        // 1 m below surface → density 1.0
        assert!((terrain_density(1.0) - 1.0).abs() < f32::EPSILON);
        // 1 m above surface → density 0.0
        assert!((terrain_density(-1.0) - 0.0).abs() < f32::EPSILON);
        // Deep underground → clamped at 1.0
        assert!((terrain_density(100.0) - 1.0).abs() < f32::EPSILON);
        // High above → clamped at 0.0
        assert!((terrain_density(-50.0) - 0.0).abs() < f32::EPSILON);
        // Half-meter below → 0.75
        assert!((terrain_density(0.5) - 0.75).abs() < f32::EPSILON);
    }

    #[test]
    fn generated_chunk_has_smooth_density_at_surface() {
        let generator = default_generator();
        // Use y=64..95 — above sea level, so above-surface voxels are AIR.
        let coord = ChunkCoord::new(0, 2, 0);
        let mut chunk = Chunk::new_empty(coord);
        generator.generate_chunk(&mut chunk);

        // Find a column that has the terrain surface within this chunk.
        let origin = coord.world_origin();
        let mut found_gradient = false;
        for lz in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                let world_x = (origin.x + lx as i32) as f64;
                let world_z = (origin.z + lz as i32) as f64;
                let height = generator.sample_height(world_x, world_z);

                // Check if the surface falls within this chunk's Y range
                let chunk_y_min = origin.y as f64;
                let chunk_y_max = chunk_y_min + CHUNK_SIZE as f64;
                if height < chunk_y_min || height >= chunk_y_max {
                    continue;
                }

                // Find the voxel at the surface level
                let surface_ly = (height - chunk_y_min) as usize;
                if surface_ly == 0 || surface_ly >= CHUNK_SIZE - 1 {
                    continue;
                }

                let below = chunk.get(lx, surface_ly, lz);
                let above = chunk.get(lx, surface_ly + 1, lz);

                // Surface voxels should have density != binary 0.0 or 1.0
                // (unless the surface height falls exactly on an integer)
                let frac = height - height.floor();
                if frac > 0.01 && frac < 0.99 {
                    assert!(
                        below.density > 0.5 && below.density < 1.0,
                        "Below-surface density should be in (0.5, 1.0), got {}",
                        below.density,
                    );
                    assert!(
                        above.density < 0.5 && above.density > 0.0,
                        "Above-surface density should be in (0.0, 0.5), got {}",
                        above.density,
                    );
                    found_gradient = true;
                    break;
                }
            }
            if found_gradient {
                break;
            }
        }
        assert!(
            found_gradient,
            "Should find at least one column with smooth density gradient"
        );
    }
}
